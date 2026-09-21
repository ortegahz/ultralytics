#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Build 3-Channel GMC Median Dataset from fpv_data (Infrared Modality) for Zero-Shot & Finetuning.

Input Pipeline Standard:
  Channel 0: I_t (Raw infrared grayscale)
  Channel 1: |I_t - W(I_{t-2})| (GMC Affine-aligned 2-step temporal frame difference)
  Channel 2: (I_t - B_t)^+ (GMC Affine-aligned 21-frame sliding window temporal median residual)

Output Directory Structure (Ultralytics Standard):
  <output_dir>/
    images/
      val/
        <seq_name>__<frame_name>.jpg
      train/
        ...
    labels/
      val/
        <seq_name>__<frame_name>.txt
      train/
        ...
    data.yaml

Features:
- Multi-processing parallelized by sequence chunk.
- Supports single-sequence probe mode (--seq <name> or --num-seqs 2) for rapid zero-shot testing.
- Supports full dataset conversion (--split val / --split train / --split all).
- Automatically cleans up classes.txt from labels and ensures 1:1 image-label mirroring.
- Caches optical flow points and affine estimation with downscale factor for speed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import re
import shutil
import sys
import time

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_key(path: Path | str):
    stem = Path(path).stem
    parts = re.split(r"(\d+)", stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert fpv_data IR sequences to 3-channel GMC+Median dataset")
    parser.add_argument(
        "--fpv-root",
        type=str,
        default="/mnt/data/siping/datasets/fpv_data",
        help="Root directory of fpv_data (containing val/, train/, test/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/data/siping/datasets/manu/fpv_ir_gmc_median",
        help="Output dataset root directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "train", "test", "all"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default="",
        help="Process only a specific sequence name (e.g. 20190926_183941_1_3) for quick testing",
    )
    parser.add_argument(
        "--num-seqs",
        type=int,
        default=0,
        help="Limit number of sequences to process (0 = all sequences in split)",
    )
    parser.add_argument("--window", type=int, default=21, help="Temporal median history window size (default: 21)")
    parser.add_argument("--stride-step", type=int, default=2, help="Temporal sampling stride step for median (default: 2)")
    parser.add_argument("--downscale", type=int, default=2, help="Downscale factor for fast GMC estimation (default: 2)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes (default: 8)")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames per sequence; uses the first contiguous segment",
    )
    parser.add_argument(
        "--seq-path",
        type=str,
        default="",
        help="Direct path to one sequence directory containing images/ and labels/",
    )
    parser.add_argument(
        "--all-segments",
        action="store_true",
        help="In direct-sequence mode, process every contiguous numeric segment",
    )
    return parser.parse_args()


class FastGMCEstimator:
    """Fast GMC Affine Motion Estimator using LK Optical Flow on Harris corners."""

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


def process_single_ir_sequence(
    seq_name: str,
    seq_dir_str: str,
    out_img_dir_str: str,
    out_lbl_dir_str: str,
    window: int,
    stride_step: int,
    downscale: int,
    max_frames: int = 0,
    image_paths: list[Path] | None = None,
    labeled_only: bool = False,
) -> dict:
    """
    Process one IR sequence directory:
    - Reads ir/images/*.jpg in natural order.
    - Computes GMC diff and sliding temporal median residual.
    - Writes <seq_name>__<im_name>.jpg and copies <seq_name>__<im_name>.txt.
    """
    seq_dir = Path(seq_dir_str)
    out_img_p = Path(out_img_dir_str)
    out_lbl_p = Path(out_lbl_dir_str)

    ir_img_dir = seq_dir / "ir" / "images"
    ir_lbl_dir = seq_dir / "ir" / "labels"
    if not ir_img_dir.exists():
        ir_img_dir = seq_dir / "images"
        ir_lbl_dir = seq_dir / "labels"

    if not ir_img_dir.exists():
        return {"seq": seq_name, "success": 0, "fail": 0, "status": "no_ir_dir"}

    img_paths = sorted(
        image_paths
        if image_paths is not None
        else [p for p in ir_img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
        key=natural_key,
    )
    if not img_paths:
        return {"seq": seq_name, "success": 0, "fail": 0, "status": "no_images"}

    if max_frames > 0:
        contiguous_paths = [img_paths[0]]
        expected_number = None
        first_number = re.search(r"(\d+)$", img_paths[0].stem)
        if first_number:
            expected_number = int(first_number.group(1)) + 1
            for path in img_paths[1:]:
                match = re.search(r"(\d+)$", path.stem)
                if not match or int(match.group(1)) != expected_number:
                    break
                contiguous_paths.append(path)
                expected_number += 1
                if len(contiguous_paths) >= max_frames:
                    break
        img_paths = contiguous_paths[:max_frames]

    estimator = FastGMCEstimator(downscale=downscale)
    success_cnt = 0
    fail_cnt = 0
    num_frames = len(img_paths)

    # Pre-cache affine transforms between consecutive frames for massive speedup
    affine_step1: dict[int, np.ndarray] = {}
    frame_cache: dict[int, np.ndarray] = {}

    def get_frame(idx: int) -> np.ndarray:
        if idx not in frame_cache:
            frame_cache[idx] = cv2.imread(str(img_paths[idx]), cv2.IMREAD_GRAYSCALE)
        return frame_cache[idx]

    def get_step1_affine(i: int) -> np.ndarray:
        """Compute or retrieve affine H from frame i to i+1."""
        if i not in affine_step1:
            im_a = get_frame(i)
            im_b = get_frame(i + 1)
            affine_step1[i] = estimator.compute_affine(im_a, im_b)
        return affine_step1[i]

    def compose_affine(H_list: list[np.ndarray]) -> np.ndarray:
        """Compose 2x3 affine matrices: H_total = H_n @ ... @ H_1."""
        if not H_list:
            return np.eye(2, 3, dtype=np.float32)
        M = np.eye(3, dtype=np.float32)
        for H in H_list:
            H3 = np.eye(3, dtype=np.float32)
            H3[:2, :] = H
            M = H3 @ M
        return M[:2, :]

    for curr_idx in range(num_frames):
        img_p = img_paths[curr_idx]
        stem = img_p.stem
        src_lbl_file = ir_lbl_dir / f"{stem}.txt"
        if labeled_only and not src_lbl_file.is_file():
            continue

        # Target filenames
        dst_im_name = f"{seq_name}__{stem}.jpg"
        dst_lbl_name = f"{seq_name}__{stem}.txt"
        dst_img_path = out_img_p / dst_im_name
        dst_lbl_path = out_lbl_p / dst_lbl_name

        # 1. Load Current Frame
        im_curr = get_frame(curr_idx)
        if im_curr is None:
            fail_cnt += 1
            continue

        # 2. Channel 1: 2-step GMC motion difference |I_t - W(I_{t-2})|
        idx_prev2 = max(0, curr_idx - 2)
        if curr_idx >= 2:
            im_prev2 = get_frame(idx_prev2)
            # Compose affine: idx_prev2 -> curr_idx-1 -> curr_idx
            H_prev2 = compose_affine([get_step1_affine(curr_idx - 2), get_step1_affine(curr_idx - 1)])
            diff2 = cv2.absdiff(im_curr, estimator.warp(im_prev2, H_prev2))
        elif curr_idx == 1:
            im_prev1 = get_frame(0)
            H1 = get_step1_affine(0)
            diff2 = cv2.absdiff(im_curr, estimator.warp(im_prev1, H1))
        else:
            diff2 = np.zeros_like(im_curr)

        # 3. Channel 2: 21-frame sliding window GMC temporal median residual (I_t - B_t)^+
        history_warped = []
        for step in range(1, window + 1):
            h_idx = curr_idx - step * stride_step
            if h_idx < 0:
                break
            im_h = get_frame(h_idx)
            if im_h is None:
                continue
            # Compose affine chain from h_idx to curr_idx
            chain = [get_step1_affine(k) for k in range(h_idx, curr_idx)]
            H_chain = compose_affine(chain)
            history_warped.append(estimator.warp(im_h, H_chain))

        if len(history_warped) >= 3:
            stack = np.stack(history_warped, axis=0)
            median_bg = np.median(stack, axis=0).astype(np.float32)
            res_median = np.clip(im_curr.astype(np.float32) - median_bg, 0, 255).astype(np.uint8)
        else:
            res_median = diff2

        # 4. Merge into 3-channel input [I_t, diff2, res_median]
        merged_3ch = np.stack([im_curr, diff2, res_median], axis=-1)

        # 5. Write image
        cv2.imwrite(str(dst_img_path), merged_3ch)

        # 6. Copy or create label file
        if src_lbl_file.exists():
            lines = []
            for line in src_lbl_file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    parts[0] = "0"
                    lines.append(" ".join(parts))
            dst_lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        else:
            dst_lbl_path.write_text("", encoding="utf-8")

        success_cnt += 1

        # Evict old frames and affines from cache to prevent RAM accumulation
        evict_idx = curr_idx - (window * stride_step + 4)
        if evict_idx in frame_cache:
            del frame_cache[evict_idx]
        if evict_idx in affine_step1:
            del affine_step1[evict_idx]

    return {
        "seq": seq_name,
        "success": success_cnt,
        "fail": fail_cnt,
        "status": "ok",
        "first_frame": img_paths[0].name,
        "last_frame": img_paths[-1].name,
    }


def split_contiguous_paths(paths: list[Path]) -> list[list[Path]]:
    segments = []
    current = []
    previous_number = None
    for path in sorted(paths, key=natural_key):
        match = re.search(r"(\d+)$", path.stem)
        number = int(match.group(1)) if match else None
        if current and (number is None or previous_number is None or number != previous_number + 1):
            segments.append(current)
            current = []
        current.append(path)
        previous_number = number
    if current:
        segments.append(current)
    return segments


def process_split(
    fpv_root: Path,
    output_root: Path,
    split_name: str,
    target_seq: str,
    num_seqs: int,
    window: int,
    stride_step: int,
    downscale: int,
    workers: int,
    max_frames: int = 0,
):
    split_dir = fpv_root / split_name
    if not split_dir.exists():
        print(f"[WARN] Split directory {split_dir} does not exist. Skipping.")
        return

    out_img_dir = output_root / "images" / split_name
    out_lbl_dir = output_root / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    all_seq_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()], key=natural_key)

    if target_seq:
        seq_dirs = [d for d in all_seq_dirs if d.name == target_seq]
    else:
        seq_dirs = all_seq_dirs

    if num_seqs > 0:
        seq_dirs = seq_dirs[:num_seqs]

    print("\n" + "=" * 90)
    print(f"PROCESSING SPLIT [{split_name.upper()}]: {len(seq_dirs)} sequence(s)")
    print(f"  Source Root : {split_dir}")
    print(f"  Output Imgs : {out_img_dir}")
    print(f"  Output Lbls : {out_lbl_dir}")
    print(f"  Workers     : {workers}")
    print(f"  Window/Step : {window} / {stride_step}")
    print("=" * 90)

    total_success = 0
    total_fail = 0

    if workers <= 1 or len(seq_dirs) == 1:
        for sd in tqdm(seq_dirs, desc=f"Split [{split_name}]"):
            res = process_single_ir_sequence(
                seq_name=sd.name,
                seq_dir_str=str(sd),
                out_img_dir_str=str(out_img_dir),
                out_lbl_dir_str=str(out_lbl_dir),
                window=window,
                stride_step=stride_step,
                downscale=downscale,
                max_frames=max_frames,
            )
            total_success += res["success"]
            total_fail += res["fail"]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_seq = {
                executor.submit(
                    process_single_ir_sequence,
                    sd.name,
                    str(sd),
                    str(out_img_dir),
                    str(out_lbl_dir),
                    window,
                    stride_step,
                    downscale,
                    max_frames,
                ): sd.name
                for sd in seq_dirs
            }

            pbar = tqdm(total=len(future_to_seq), desc=f"Split [{split_name}]")
            for future in as_completed(future_to_seq):
                seq_name = future_to_seq[future]
                try:
                    res = future.result()
                    total_success += res["success"]
                    total_fail += res["fail"]
                except Exception as e:
                    print(f"\n[ERROR] Sequence {seq_name} failed with error: {e}")
                finally:
                    pbar.update(1)
            pbar.close()

    print(f"\n[DONE] Split [{split_name}] Completed! Success: {total_success} frames, Failed: {total_fail} frames.")


def write_data_yaml(output_root: Path):
    yaml_path = output_root / "data.yaml"
    content = f"""# Ultralytics UAV Dataset: FPV Infrared GMC+Median Mode
path: {output_root.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: uav
"""
    yaml_path.write_text(content, encoding="utf-8")
    print(f"[INFO] Created dataset metadata -> {yaml_path}")


def main():
    args = parse_args()
    fpv_root = Path(args.fpv_root)
    output_root = Path(args.output_dir)
    t_start = time.time()

    if args.seq_path:
        seq_path = Path(args.seq_path).resolve()
        if not seq_path.exists():
            print(f"[ERROR] Sequence directory not found: {seq_path}")
            sys.exit(1)
        out_img_dir = output_root / "images" / "val"
        out_lbl_dir = output_root / "labels" / "val"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        img_dir = seq_path / "ir" / "images"
        lbl_dir = seq_path / "ir" / "labels"
        if not img_dir.exists():
            img_dir = seq_path / "images"
            lbl_dir = seq_path / "labels"
        all_paths = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        segments = split_contiguous_paths(all_paths) if args.all_segments else [sorted(all_paths, key=natural_key)]
        total_success = total_fail = 0
        for segment_index, segment_paths in enumerate(segments):
            if args.max_frames > 0:
                segment_paths = segment_paths[: args.max_frames]
            segment_name = f"{seq_path.name}___seg{segment_index:03d}"
            result = process_single_ir_sequence(
                seq_name=segment_name,
                seq_dir_str=str(seq_path),
                out_img_dir_str=str(out_img_dir),
                out_lbl_dir_str=str(out_lbl_dir),
                window=args.window,
                stride_step=args.stride_step,
                downscale=args.downscale,
                image_paths=segment_paths,
                labeled_only=True,
            )
            total_success += result["success"]
            total_fail += result["fail"]
            print(
                f"[DONE] {result['seq']}: {result['success']} labeled frames, {result['fail']} failed "
                f"({result.get('first_frame', '?')} -> {result.get('last_frame', '?')})"
            )
        print(f"[DONE] Total labeled frames: {total_success}, failed: {total_fail}, segments: {len(segments)}")
    else:
        if not fpv_root.exists():
            for cand in [
                Path("/mnt/data/siping/datasets/fpv_data"),
                Path("/home/manu/mnt/datasets/fpv_data"),
            ]:
                if cand.exists():
                    fpv_root = cand
                    break

        if not fpv_root.exists():
            print(f"[ERROR] fpv_data root not found: {args.fpv_root}")
            sys.exit(1)

        t_start = time.time()
        splits = ["val", "train"] if args.split == "all" else [args.split]

        for sp in splits:
            process_split(
                fpv_root=fpv_root,
                output_root=output_root,
                split_name=sp,
                target_seq=args.seq,
                num_seqs=args.num_seqs,
                window=args.window,
                stride_step=args.stride_step,
                downscale=args.downscale,
                workers=args.workers,
                max_frames=args.max_frames,
            )

    write_data_yaml(output_root)
    total_time = time.time() - t_start
    print(f"\n=========================================================================================")
    print(f"✅ ALL TASKS FINISHED in {total_time:.1f}s (~{total_time/60:.1f} mins)!")
    print(f"Output Dataset Directory: {output_root.resolve()}")
    print(f"=========================================================================================\n")


if __name__ == "__main__":
    main()
