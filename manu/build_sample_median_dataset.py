#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Build Small-Scale Benchmark Dataset with Aligned Temporal Median Background:
Input Composition: [I_t, |I_t - W(I_{t-2})|, (I_t - B_t)^+]
where:
  - Channel 0: I_t (Raw infrared frame from official dataset)
  - Channel 1: |I_t - W(I_{t-2})| (2-lag GMC aligned difference)
  - Channel 2: (I_t - B_t)^+ (GMC-aligned sliding window temporal median background residual)

Zero-Drift Guarantee:
1. 100% mirrors filenames and labels from reference YOLO dataset (e.g. /mnt/data/siping/datasets/manu/uav).
2. Label coordinates and bounding boxes match with 0.0px spatial error.
3. Supports subset filtering (e.g. --sequences) to build a small verification set in ~15 seconds.

Usage on Server:
    python manu/build_sample_median_dataset.py \
        --ref-dataset /mnt/data/siping/datasets/manu/uav \
        --raw-root /mnt/data/siping/datasets/manu/anti-uav \
        --output /mnt/data/siping/datasets/manu/uav_median_check \
        --sequences 34_1,5_1,43_1,3700000000002_102004_3,DJI_0051_2,DJI_0175_2,wg2022_ir_011_split_03,wg2022_ir_020_split_03 \
        --window 21 \
        --workers 16
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import re
import shutil
import sys
import time

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build Sample Aligned Median Background YOLO Dataset")
    parser.add_argument(
        "--ref-dataset",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav",
        help="Path to official reference YOLO dataset",
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
        default="/mnt/data/siping/datasets/manu/uav_median_check",
        help="Output directory for sample median verification dataset",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default="34_1,5_1,43_1,3700000000002_102004_3,DJI_0051_2,DJI_0175_2,wg2022_ir_011_split_03,wg2022_ir_020_split_03",
        help="Comma-separated sequence names to include (empty = all sequences in split)",
    )
    parser.add_argument("--split", type=str, default="val", help="Dataset split to mirror (default: val)")
    parser.add_argument("--window", type=int, default=21, help="Temporal sliding window size (default: 21)")
    parser.add_argument("--stride-step", type=int, default=2, help="Temporal sampling stride step (default: 2)")
    parser.add_argument("--workers", type=int, default=16, help="Parallel worker processes")
    parser.add_argument("--downscale", type=int, default=2, help="Downscale factor for optical flow GMC estimation")
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
        return seq, int(match.group(1)) if match else 0
    if "__" in stem:
        parts = stem.split("__")
        seq = parts[0]
        match = re.search(r"(\d+)$", parts[1])
        return seq, int(match.group(1)) if match else 0

    match = re.search(r"^(.*?)(?:[_-]+)?(\d+)$", stem)
    if match:
        return match.group(1).rstrip("_-"), int(match.group(2))
    raise ValueError(f"Cannot parse sequence and frame from {im_name}")


def find_sequence_folder(raw_root: Path, seq_name: str, cache: dict[str, Path]) -> Path | None:
    if seq_name in cache:
        return cache[seq_name]
    cand = raw_root / seq_name
    if cand.is_dir():
        cache[seq_name] = cand
        return cand
    for sub in [
        raw_root / "Data" / "val" / seq_name,
        raw_root / "val" / seq_name,
        raw_root / "Data" / "train" / seq_name,
        raw_root / "train" / seq_name,
    ]:
        if sub.is_dir():
            cache[seq_name] = sub
            return sub
    for p in raw_root.rglob(seq_name):
        if p.is_dir():
            cache[seq_name] = p
            return p
    return None


class FastGMCEstimator:
    def __init__(self, downscale: int = 2):
        self.downscale = downscale
        self.feature_params = {
            "maxCorners": 600,
            "qualityLevel": 0.01,
            "minDistance": 4,
            "blockSize": 3,
        }

    def compute_affine(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        h, w = curr_gray.shape[:2]
        ds = self.downscale
        H = np.eye(2, 3, dtype=np.float32)

        if ds > 1:
            prev_small = cv2.resize(prev_gray, (w // ds, h // ds))
            curr_small = cv2.resize(curr_gray, (w // ds, h // ds))
        else:
            prev_small, curr_small = prev_gray, curr_gray

        pts_prev = cv2.goodFeaturesToTrack(prev_small, mask=None, **self.feature_params)
        if pts_prev is not None and len(pts_prev) >= 6:
            pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_small, curr_small, pts_prev, None, winSize=(15, 15), maxLevel=2
            )
            good = status.ravel() == 1
            p0 = pts_prev[good]
            p1 = pts_curr[good]
            if len(p0) >= 6:
                M, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if M is not None:
                    H = M.astype(np.float32)
                    if ds > 1:
                        H[0, 2] *= ds
                        H[1, 2] *= ds
        return H

    def warp(self, img: np.ndarray, H: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        return cv2.warpAffine(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def process_sequence_chunk(
    seq_name: str,
    img_names: list[str],
    ref_lbl_dir_str: str,
    raw_root_str: str,
    out_img_dir_str: str,
    out_lbl_dir_str: str,
    window: int,
    stride_step: int,
    downscale: int,
) -> dict:
    raw_root = Path(raw_root_str)
    out_img_p = Path(out_img_dir_str)
    out_lbl_p = Path(out_lbl_dir_str)
    ref_lbl_p = Path(ref_lbl_dir_str)

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
        idx_map[int(match.group(1)) if match else list_i] = list_i

    estimator = FastGMCEstimator(downscale=downscale)
    success_cnt = 0
    fail_cnt = 0

    sorted_im_names = sorted(img_names, key=natural_key)

    # Frame buffer to avoid redundant reads
    frame_cache: dict[int, np.ndarray] = {}

    for im_name in sorted_im_names:
        _, frame_idx = parse_seq_and_frame(im_name)
        curr_list_idx = idx_map.get(frame_idx, min(frame_idx, len(frames) - 1))

        # Read current frame
        if curr_list_idx not in frame_cache:
            im_curr = cv2.imread(str(frames[curr_list_idx]), cv2.IMREAD_GRAYSCALE)
            frame_cache[curr_list_idx] = im_curr
        else:
            im_curr = frame_cache[curr_list_idx]

        if im_curr is None:
            fail_cnt += 1
            continue

        # 1. Compute 2-lag aligned difference |I_t - W(I_{t-2})|
        idx_prev2 = max(0, curr_list_idx - 2)
        if idx_prev2 not in frame_cache:
            im_prev2 = cv2.imread(str(frames[idx_prev2]), cv2.IMREAD_GRAYSCALE)
            frame_cache[idx_prev2] = im_prev2
        else:
            im_prev2 = frame_cache[idx_prev2]

        H2 = estimator.compute_affine(im_prev2, im_curr)
        diff2 = cv2.absdiff(im_curr, estimator.warp(im_prev2, H2))

        # 2. Compute Long-term Temporal Median Background Residual (I_t - B_t)^+
        history_warped = []
        for step in range(1, window + 1):
            h_idx = max(0, curr_list_idx - step * stride_step)
            if h_idx not in frame_cache:
                im_h = cv2.imread(str(frames[h_idx]), cv2.IMREAD_GRAYSCALE)
                frame_cache[h_idx] = im_h
            else:
                im_h = frame_cache[h_idx]

            if im_h is not None:
                H_h = estimator.compute_affine(im_h, im_curr)
                history_warped.append(estimator.warp(im_h, H_h))

        if len(history_warped) >= 5:
            stack = np.stack(history_warped, axis=0)
            median_bg = np.median(stack, axis=0).astype(np.float32)
            res_median = np.clip(im_curr.astype(np.float32) - median_bg, 0, 255).astype(np.uint8)
        else:
            res_median = diff2

        # 3. Channel composition: [I_t, diff2_gmc, (I_t - B_t)^+]
        merged = np.stack([im_curr, diff2, res_median], axis=-1)

        # 4. Save image
        dst_img_file = out_img_p / im_name
        cv2.imwrite(str(dst_img_file), merged)

        # 5. Copy corresponding label 1:1
        src_lbl_file = ref_lbl_p / f"{Path(im_name).stem}.txt"
        dst_lbl_file = out_lbl_p / f"{Path(im_name).stem}.txt"
        if src_lbl_file.exists():
            shutil.copy(src_lbl_file, dst_lbl_file)
        else:
            dst_lbl_file.write_text("", encoding="utf-8")

        success_cnt += 1

        # Clean cache to preserve memory
        if (curr_list_idx - (window * stride_step + 10)) in frame_cache:
            del frame_cache[curr_list_idx - (window * stride_step + 10)]

    return {"seq": seq_name, "success": success_cnt, "fail": fail_cnt, "status": "ok"}


def main():
    args = parse_args()
    ref_dir = Path(args.ref_dataset).resolve()
    raw_root = Path(args.raw_root).resolve()
    out_dir = Path(args.output).resolve()

    if not ref_dir.is_dir():
        for cand in [Path("/mnt/data/siping/datasets/manu/uav"), Path("/home/manu/mnt/datasets/manu/uav")]:
            if cand.is_dir():
                ref_dir = cand
                break
    if not raw_root.is_dir():
        for cand in [Path("/mnt/data/siping/datasets/manu/anti-uav"), Path("/home/manu/mnt/datasets/manu/anti-uav")]:
            if cand.is_dir():
                raw_root = cand
                break

    print("=" * 88)
    print("   UAV Tiny Object Detection: Aligned Median Background Dataset Builder")
    print(f"   Reference Dataset : {ref_dir}")
    print(f"   Raw Video Root    : {raw_root}")
    print(f"   Output Dataset    : {out_dir}")
    print(f"   Median Window     : {args.window} frames (step: {args.stride_step})")
    print("=" * 88)

    ref_img_dir = ref_dir / "images" / args.split
    ref_lbl_dir = ref_dir / "labels" / args.split

    out_img_dir = out_dir / "images" / args.split
    out_lbl_dir = out_dir / "labels" / args.split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    target_seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]

    ref_images = [p.name for p in ref_img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    ref_images.sort(key=natural_key)

    seq_groups: dict[str, list[str]] = {}
    for im_name in ref_images:
        seq, _ = parse_seq_and_frame(im_name)
        if not target_seqs or any(ts == seq or ts in seq for ts in target_seqs):
            seq_groups.setdefault(seq, []).append(im_name)

    total_selected = sum(len(v) for v in seq_groups.values())
    print(f"[INFO] Selected {len(seq_groups)} sequences ({total_selected} images) for evaluation.")
    print(f"[INFO] Launching parallel pool with {args.workers} workers...")

    t0 = time.time()
    total_success = 0
    total_fail = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for seq_name, img_names in seq_groups.items():
            f = executor.submit(
                process_sequence_chunk,
                seq_name,
                img_names,
                str(ref_lbl_dir),
                str(raw_root),
                str(out_img_dir),
                str(out_lbl_dir),
                args.window,
                args.stride_step,
                args.downscale,
            )
            futures.append(f)

        for f in futures:
            res = f.result()
            total_success += res["success"]
            total_fail += res["fail"]
            status_tag = f"Fail={res['fail']}" if res["status"] != "ok" else "OK"
            print(f"  --> [{res['seq']:<28}] : {res['success']:>5} images generated ({status_tag})")

    elapsed = time.time() - t0
    fps = total_selected / max(0.1, elapsed)
    print(f"\n[DONE] Finished in {elapsed:.1f}s ({fps:.1f} imgs/s) | Success: {total_success}, Fail: {total_fail}")

    # Write data.yaml for official validation scripts
    yaml_content = f"""# UAV Aligned Temporal Median Residual Dataset
path: {out_dir.resolve()}
train: images/{args.split}
val: images/{args.split}

names:
  0: uav
"""
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"[SUCCESS] Dataset configuration created: {yaml_path.resolve()}\n")


if __name__ == "__main__":
    main()
