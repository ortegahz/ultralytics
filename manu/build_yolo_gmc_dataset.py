#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Generate YOLO26-format GMC Aligned Dataset: [I_t, |I_t - W(I_{t-1})|, |I_t - W(I_{t-2})|]

Directly mirrors an existing reference YOLO dataset (e.g. /mnt/data/siping/datasets/manu/uav):
1. Reads existing filenames directly from {ref_dataset}/images/{split}.
2. Fetches raw I_t, I_{t-1}, I_{t-2} from the raw sequence folder.
3. Computes Global Motion Compensation (GMC) affine matrix H via Sparse LK Optical Flow.
4. Warps I_{t-1} and I_{t-2} into I_t coordinate system, then computes absolute differences.
5. Saves 3-channel composite image [I_t, diff1_gmc, diff2_gmc] with 100% identical filenames.
6. Copies labels 1:1 from {ref_dataset}/labels/{split}.
7. Supports parallel multi-processing across sequence partitions for ultra-fast generation.

Usage on Server:
    # 1. Build validation set only (fast verification, ~2-5 mins with 16 workers):
    python manu/build_yolo_gmc_dataset.py \
        --ref-dataset /mnt/data/siping/datasets/manu/uav \
        --raw-root /mnt/data/siping/datasets/manu/anti-uav \
        --output /mnt/data/siping/datasets/manu/uav_gmc \
        --splits val \
        --workers 16

    # 2. Build full dataset (train + val):
    python manu/build_yolo_gmc_dataset.py \
        --ref-dataset /mnt/data/siping/datasets/manu/uav \
        --raw-root /mnt/data/siping/datasets/manu/anti-uav \
        --output /mnt/data/siping/datasets/manu/uav_gmc \
        --splits train,val \
        --workers 16
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import re
import shutil
import sys
import time

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build GMC-aligned dataset [I_t, |I_t - W(I_{t-1})|, |I_t - W(I_{t-2})|]"
    )
    parser.add_argument(
        "--ref-dataset",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav",
        help="Path to reference YOLO dataset",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/anti-uav",
        help="Root directory containing raw sequences",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc",
        help="Target output directory for the new GMC dataset",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="val",
        help="Comma-separated splits to process (e.g. 'val' or 'train,val')",
    )
    parser.add_argument("--lag1", type=int, default=1, help="First lag for GMC diff (default: 1)")
    parser.add_argument("--lag2", type=int, default=2, help="Second lag for GMC diff (default: 2)")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel worker processes")
    parser.add_argument("--gmc-downscale", type=int, default=2, help="Downscale factor for GMC estimation")
    return parser.parse_args()


def natural_key(path: Path | str):
    stem = Path(path).stem
    parts = re.split(r"(\d+)", stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def parse_seq_and_frame(im_name: str) -> tuple[str, int]:
    stem = Path(im_name).stem
    if "___" in stem:
        parts = stem.split("___")
        seq = parts[0]
        match = re.search(r"(\d+)$", parts[1])
        frame_idx = int(match.group(1)) if match else 0
        return seq, frame_idx

    if "__" in stem:
        parts = stem.split("__")
        seq = parts[0]
        match = re.search(r"(\d+)$", parts[1])
        frame_idx = int(match.group(1)) if match else 0
        return seq, frame_idx

    match = re.search(r"^(.*?)(?:[_-]+)?(\d+)$", stem)
    if match:
        seq = match.group(1).rstrip("_-")
        frame_idx = int(match.group(2))
        return seq, frame_idx

    raise ValueError(f"Cannot parse sequence name and frame number from image name: {im_name}")


def find_sequence_folder(raw_root: Path, seq_name: str, cache: dict[str, Path]) -> Path | None:
    if seq_name in cache:
        return cache[seq_name]

    # 1. Direct child
    cand = raw_root / seq_name
    if cand.is_dir():
        cache[seq_name] = cand
        return cand

    # 2. Inside Data/train, train, or val
    for sub in [
        raw_root / "Data" / "val" / seq_name,
        raw_root / "val" / seq_name,
        raw_root / "Data" / "train" / seq_name,
        raw_root / "train" / seq_name,
    ]:
        if sub.is_dir():
            cache[seq_name] = sub
            return sub

    # 3. Recursive search
    for p in raw_root.rglob(seq_name):
        if p.is_dir():
            cache[seq_name] = p
            return p

    return None


class FastGMCEstimator:
    """Lightweight Sparse LK Optical Flow GMC Estimator."""

    def __init__(self, downscale: int = 2):
        self.downscale = downscale
        self.feature_params = {
            "maxCorners": 600,
            "qualityLevel": 0.01,
            "minDistance": 4,
            "blockSize": 3,
        }

    def align_diff(self, curr_gray: np.ndarray, prev_gray: np.ndarray) -> np.ndarray:
        h, w = curr_gray.shape[:2]
        ds = self.downscale
        H = np.eye(2, 3, dtype=np.float32)

        if ds > 1:
            prev_small = cv2.resize(prev_gray, (w // ds, h // ds))
            curr_small = cv2.resize(curr_gray, (w // ds, h // ds))
        else:
            prev_small = prev_gray
            curr_small = curr_gray

        pts_prev = cv2.goodFeaturesToTrack(prev_small, mask=None, **self.feature_params)
        if pts_prev is not None and len(pts_prev) >= 6:
            pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_small, curr_small, pts_prev, None, winSize=(15, 15), maxLevel=2
            )
            good = (status.ravel() == 1)
            p0 = pts_prev[good]
            p1 = pts_curr[good]

            if len(p0) >= 6:
                M, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if M is not None:
                    H = M.astype(np.float32)
                    if ds > 1:
                        H[0, 2] *= ds
                        H[1, 2] *= ds

        # Warp and calculate difference
        warped_prev = cv2.warpAffine(
            prev_gray,
            H,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return cv2.absdiff(curr_gray, warped_prev)


def process_sequence_chunk(
    seq_name: str,
    img_names: list[str],
    ref_img_dir: str,
    ref_lbl_dir: str,
    raw_root_str: str,
    out_img_dir: str,
    out_lbl_dir: str,
    lag1: int,
    lag2: int,
    downscale: int,
) -> dict:
    """Worker function processing all images belonging to a single sequence."""
    raw_root = Path(raw_root_str)
    out_img_p = Path(out_img_dir)
    out_lbl_p = Path(out_lbl_dir)
    ref_img_p = Path(ref_img_dir)
    ref_lbl_p = Path(ref_lbl_dir)

    cache: dict[str, Path] = {}
    seq_dir = find_sequence_folder(raw_root, seq_name, cache)
    if seq_dir is None:
        return {"seq": seq_name, "success": 0, "fail": len(img_names), "status": "missing_seq"}

    frames = [f for f in seq_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES]
    frames.sort(key=natural_key)
    if not frames:
        return {"seq": seq_name, "success": 0, "fail": len(img_names), "status": "no_raw_frames"}

    idx_map = {}
    for list_i, f in enumerate(frames):
        match = re.search(r"(\d+)$", f.stem)
        if match:
            idx_map[int(match.group(1))] = list_i
        else:
            idx_map[list_i] = list_i

    estimator = FastGMCEstimator(downscale=downscale)
    success_cnt = 0
    fail_cnt = 0

    # Sort images by frame index to process chronologically
    sorted_im_names = sorted(img_names, key=natural_key)

    for im_name in sorted_im_names:
        _, frame_idx = parse_seq_and_frame(im_name)
        curr_list_idx = idx_map.get(frame_idx)
        if curr_list_idx is None:
            curr_list_idx = min(frame_idx, len(frames) - 1)

        idx1 = max(0, curr_list_idx - lag1)
        idx2 = max(0, curr_list_idx - lag2)

        p_curr = frames[curr_list_idx]
        p_prev1 = frames[idx1]
        p_prev2 = frames[idx2]

        im_curr = cv2.imread(str(p_curr), cv2.IMREAD_GRAYSCALE)
        im_p1 = cv2.imread(str(p_prev1), cv2.IMREAD_GRAYSCALE)
        im_p2 = cv2.imread(str(p_prev2), cv2.IMREAD_GRAYSCALE)

        if im_curr is None or im_p1 is None or im_p2 is None:
            fail_cnt += 1
            continue

        # Compute GMC-aligned differences
        diff1 = estimator.align_diff(im_curr, im_p1)
        diff2 = estimator.align_diff(im_curr, im_p2)

        # Merge 3 channels: [I_t, diff1_gmc, diff2_gmc]
        merged = np.stack([im_curr, diff1, diff2], axis=-1)

        # Write to destination image
        dst_img_file = out_img_p / im_name
        cv2.imwrite(str(dst_img_file), merged)

        # Copy corresponding label file (1:1 identical)
        src_lbl_file = ref_lbl_p / f"{Path(im_name).stem}.txt"
        dst_lbl_file = out_lbl_p / f"{Path(im_name).stem}.txt"
        if src_lbl_file.exists():
            shutil.copy(src_lbl_file, dst_lbl_file)
        else:
            dst_lbl_file.write_text("", encoding="utf-8")

        success_cnt += 1

    return {"seq": seq_name, "success": success_cnt, "fail": fail_cnt, "status": "ok"}


def process_split_parallel(
    ref_dir: Path,
    split_name: str,
    raw_root: Path,
    out_dir: Path,
    lag1: int,
    lag2: int,
    workers: int,
    downscale: int,
):
    print(f"\n==================== Processing [{split_name}] Split ====================")
    ref_img_dir = ref_dir / "images" / split_name
    ref_lbl_dir = ref_dir / "labels" / split_name

    if not ref_img_dir.is_dir():
        raise FileNotFoundError(f"Reference split images directory not found: {ref_img_dir}")

    out_img_dir = out_dir / "images" / split_name
    out_lbl_dir = out_dir / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    ref_images = [p.name for p in ref_img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    ref_images.sort(key=natural_key)
    total_imgs = len(ref_images)
    print(f"[INFO] Found {total_imgs} images in reference split [{split_name}].")

    # Group images by sequence name
    seq_groups: dict[str, list[str]] = {}
    for im_name in ref_images:
        seq, _ = parse_seq_and_frame(im_name)
        seq_groups.setdefault(seq, []).append(im_name)

    print(f"[INFO] Discovered {len(seq_groups)} distinct video sequences in [{split_name}].")
    print(f"[INFO] Launching parallel pool with {workers} worker processes...")

    t0 = time.time()
    total_success = 0
    total_fail = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for seq_name, img_names in seq_groups.items():
            f = executor.submit(
                process_sequence_chunk,
                seq_name,
                img_names,
                str(ref_img_dir),
                str(ref_lbl_dir),
                str(raw_root),
                str(out_img_dir),
                str(out_lbl_dir),
                lag1,
                lag2,
                downscale,
            )
            futures.append(f)

        for f in futures:
            res = f.result()
            total_success += res["success"]
            total_fail += res["fail"]
            if res["status"] != "ok":
                print(f"[WARN] Sequence '{res['seq']}': {res['status']} (fail={res['fail']})")
            else:
                print(f"  --> Completed Sequence [{res['seq']:<28}] : {res['success']:>5} images written.")

    elapsed = time.time() - t0
    fps = total_imgs / max(0.1, elapsed)
    print(f"\n[{split_name}] Done in {elapsed:.1f}s ({fps:.1f} imgs/s) | Success: {total_success}, Fail: {total_fail}")


def write_data_yaml(out_dir: Path):
    yaml_content = f"""# Ultralytics UAV Dataset: GMC Motion-Compensated Mode [I_t, |I_t - W(I_{{t-1}})|, |I_t - W(I_{{t-2}})|]
path: {out_dir.resolve()}
train: images/train
val: images/val

names:
  0: uav
"""
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"[INFO] Successfully created data configuration: {yaml_path.resolve()}")


def main():
    args = parse_args()
    ref_dir = Path(args.ref_dataset).resolve()
    raw_root = Path(args.raw_root).resolve()
    out_dir = Path(args.output).resolve()

    if not ref_dir.is_dir():
        # Fallback candidate check
        for cand in [Path("/mnt/data/siping/datasets/manu/uav"), Path("/home/manu/mnt/datasets/manu/uav")]:
            if cand.is_dir():
                ref_dir = cand
                break
    if not raw_root.is_dir():
        for cand in [Path("/mnt/data/siping/datasets/manu/anti-uav"), Path("/home/manu/mnt/datasets/manu/anti-uav")]:
            if cand.is_dir():
                raw_root = cand
                break

    print("=" * 80)
    print("   UAV Tiny Object Detection: Offline GMC Dataset Builder")
    print(f"   Reference Dataset : {ref_dir}")
    print(f"   Raw Root          : {raw_root}")
    print(f"   Target Output     : {out_dir}")
    print(f"   Lags (1, 2)       : ({args.lag1}, {args.lag2}) | Workers: {args.workers}")
    print("=" * 80)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for s in splits:
        process_split_parallel(
            ref_dir=ref_dir,
            split_name=s,
            raw_root=raw_root,
            out_dir=out_dir,
            lag1=args.lag1,
            lag2=args.lag2,
            workers=args.workers,
            downscale=args.gmc_downscale,
        )

    write_data_yaml(out_dir)
    print(f"\n[SUCCESS] Entire GMC dataset successfully generated at: {out_dir}\n")


if __name__ == "__main__":
    main()
