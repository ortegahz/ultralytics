#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Build High-Fidelity Stride=2 Continuous UAV Training & Validation Dataset:
`uav_gmc_median_s2`

Problem Solved:
1. Existing uav_gmc_median/train was downsampled at Stride=5 (43,008 images from 199 sequences).
2. For temporal sequence attention at Stride=2 (K=8 frames), train and val require uniform Stride=2 sampling.
3. This script parses raw full-rate anti-uav/train sequences, selects every 2nd frame (000001, 000003, ...),
   computes 100% matched GMC 2-lag diff + 21-frame temporal median residual, and maps ground truth
   directly from IR_label.json with 0.0px coordinate drift.
4. Validation split (val) is already dense (stride=1) in uav_gmc_median, which can be linked or built.

Dataset Structure:
/mnt/data/siping/datasets/manu/uav_gmc_median_s2/
├── data.yaml
├── images
│   ├── train (Stride=2 sampling, ~58,000 images, 3-channel [I_t, Diff, Median])
│   └── val (Symlinked or copied from uav_gmc_median/images/val)
└── labels
    ├── train (0.0px drift YOLO txt)
    └── val (Symlinked from uav_gmc_median/labels/val)

Usage on Server:
    python manu/build_s2_median_dataset.py \
        --raw-train-root /mnt/data/siping/datasets/manu/anti-uav/train \
        --val-ref-dataset /mnt/data/siping/datasets/manu/uav_gmc_median \
        --output /mnt/data/siping/datasets/manu/uav_gmc_median_s2 \
        --stride 2 \
        --window 21 \
        --workers 16
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re
import shutil
import sys
import time

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build Stride=2 Continuous UAV Dataset")
    parser.add_argument(
        "--raw-train-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/anti-uav/train",
        help="Path to raw anti-uav/train directory containing full sequences",
    )
    parser.add_argument(
        "--val-ref-dataset",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median",
        help="Path to validated reference dataset (for reusing val split)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median_s2",
        help="Output destination path",
    )
    parser.add_argument("--stride", type=int, default=2, help="Sampling stride on raw sequences (default: 2)")
    parser.add_argument("--window", type=int, default=21, help="Temporal sliding window size for median (default: 21)")
    parser.add_argument("--workers", type=int, default=16, help="Parallel worker processes (default: 16)")
    parser.add_argument("--downscale", type=int, default=2, help="Downscale factor for GMC estimation (default: 2)")
    parser.add_argument("--max-frames-per-seq", type=int, default=1500, help="Max frames per sequence")
    return parser.parse_args()


def natural_key(path: Path | str):
    stem = Path(path).stem
    parts = re.split(r"(\d+)", stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


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


def process_sequence_stride2(
    seq_dir_str: str,
    out_img_dir_str: str,
    out_lbl_dir_str: str,
    sample_stride: int,
    window: int,
    downscale: int,
) -> dict:
    seq_dir = Path(seq_dir_str)
    seq_name = seq_dir.name
    out_img_p = Path(out_img_dir_str)
    out_lbl_p = Path(out_lbl_dir_str)

    # 1. Load IR_label.json
    label_json = seq_dir / "IR_label.json"
    if not label_json.exists():
        return {"seq": seq_name, "success": 0, "fail": 0, "status": "no_label_json"}

    with open(label_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    gt_rect_list = meta.get("gt_rect", [])
    exist_list = meta.get("exist", [1] * len(gt_rect_list))

    raw_frames = [f for f in seq_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES]
    raw_frames.sort(key=natural_key)
    if not raw_frames:
        return {"seq": seq_name, "success": 0, "fail": 0, "status": "no_frames"}

    # 2. Sample target indices with Stride=2 (indices: 0, 2, 4, 6, ...)
    target_indices = list(range(0, len(raw_frames), sample_stride))

    estimator = FastGMCEstimator(downscale=downscale)
    frame_cache: dict[int, np.ndarray] = {}
    success_cnt = 0
    fail_cnt = 0

    for curr_idx in target_indices:
        raw_f = raw_frames[curr_idx]
        frame_num = int(re.search(r"(\d+)$", raw_f.stem).group(1)) if re.search(r"(\d+)$", raw_f.stem) else curr_idx + 1
        im_out_name = f"{seq_name}__{frame_num:06d}.jpg"
        lbl_out_name = f"{seq_name}__{frame_num:06d}.txt"

        # Read current frame
        if curr_idx not in frame_cache:
            im_curr = cv2.imread(str(raw_f), cv2.IMREAD_GRAYSCALE)
            frame_cache[curr_idx] = im_curr
        else:
            im_curr = frame_cache[curr_idx]

        if im_curr is None:
            fail_cnt += 1
            continue

        h_img, w_img = im_curr.shape[:2]

        # 3. 2-lag GMC aligned short difference |I_t - W(I_{t-2})|
        # In Stride=2 stream, 2-lag difference in raw time corresponds to curr_idx - 2!
        prev2_idx = max(0, curr_idx - 2)
        if prev2_idx not in frame_cache:
            im_prev2 = cv2.imread(str(raw_frames[prev2_idx]), cv2.IMREAD_GRAYSCALE)
            frame_cache[prev2_idx] = im_prev2
        else:
            im_prev2 = frame_cache[prev2_idx]

        H2 = estimator.compute_affine(im_prev2, im_curr)
        diff2 = cv2.absdiff(im_curr, estimator.warp(im_prev2, H2))

        # 4. 21-frame sliding window GMC aligned median residual: (I_t - B_t)^+
        # In Stride=2 sequence, history steps correspond to curr_idx - step * 2
        history_warped = []
        for step in range(1, window + 1):
            h_idx = max(0, curr_idx - step * 2)
            if h_idx not in frame_cache:
                im_h = cv2.imread(str(raw_frames[h_idx]), cv2.IMREAD_GRAYSCALE)
                frame_cache[h_idx] = im_h
            else:
                im_h = frame_cache[h_idx]

            if im_h is not None:
                H_h = estimator.compute_affine(im_h, im_curr)
                history_warped.append(estimator.warp(im_h, H_h))

        if len(history_warped) >= 3:
            stack = np.stack(history_warped, axis=0)
            median_bg = np.median(stack, axis=0).astype(np.float32)
            res_median = np.clip(im_curr.astype(np.float32) - median_bg, 0, 255).astype(np.uint8)
        else:
            res_median = diff2

        # 5. Composite 3-Channel: [I_t, diff2, res_median]
        merged = np.stack([im_curr, diff2, res_median], axis=-1)
        dst_img_file = out_img_p / im_out_name
        cv2.imwrite(str(dst_img_file), merged)

        # 6. Generate 100% matched YOLO label from IR_label.json
        dst_lbl_file = out_lbl_p / lbl_out_name
        has_gt = False
        if curr_idx < len(gt_rect_list) and curr_idx < len(exist_list):
            exist_flag = exist_list[curr_idx]
            rect = gt_rect_list[curr_idx]
            if exist_flag == 1 and len(rect) == 4 and rect[2] > 0 and rect[3] > 0:
                cx = (rect[0] + rect[2] / 2.0) / w_img
                cy = (rect[1] + rect[3] / 2.0) / h_img
                bw = rect[2] / w_img
                bh = rect[3] / h_img
                # Clip to valid [0, 1] range
                cx = float(np.clip(cx, 0.0, 1.0))
                cy = float(np.clip(cy, 0.0, 1.0))
                bw = float(np.clip(bw, 1e-4, 1.0))
                bh = float(np.clip(bh, 1e-4, 1.0))
                lbl_line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
                dst_lbl_file.write_text(lbl_line, encoding="utf-8")
                has_gt = True

        if not has_gt:
            dst_lbl_file.write_text("", encoding="utf-8")

        success_cnt += 1

        # Memory eviction: evict frames older than window * 2 + 10
        evict_idx = curr_idx - (window * 2 + 10)
        if evict_idx in frame_cache:
            del frame_cache[evict_idx]

    return {"seq": seq_name, "success": success_cnt, "fail": fail_cnt, "status": "ok"}


def main():
    args = parse_args()
    print("=" * 90)
    print("🚀 UAV Tiny Object Dataset Generator: Stride=2 Continuous Time Series")
    print(f"Raw Train Root   : {args.raw_train_root}")
    print(f"Val Ref Dataset  : {args.val_ref_dataset}")
    print(f"Output Dataset   : {args.output}")
    print(f"Sampling Stride  : {args.stride}")
    print(f"Median Window    : {args.window}")
    print(f"Worker Processes : {args.workers}")
    print("=" * 90)

    out_p = Path(args.output)
    out_img_train = out_p / "images" / "train"
    out_lbl_train = out_p / "labels" / "train"
    out_img_val = out_p / "images" / "val"
    out_lbl_val = out_p / "labels" / "val"

    out_img_train.mkdir(parents=True, exist_ok=True)
    out_lbl_train.mkdir(parents=True, exist_ok=True)

    # 1. Handle Validation Split: link directly from validated uav_gmc_median (31,613 images)
    val_ref_p = Path(args.val_ref_dataset)
    val_img_ref = val_ref_p / "images" / "val"
    val_lbl_ref = val_ref_p / "labels" / "val"

    if val_img_ref.is_dir() and val_lbl_ref.is_dir():
        print(f"\n[INFO] Reusing validated dense validation split from: {val_ref_p}")
        if out_img_val.exists() or out_img_val.is_symlink():
            if out_img_val.is_symlink():
                out_img_val.unlink()
            else:
                shutil.rmtree(out_img_val)
        if out_lbl_val.exists() or out_lbl_val.is_symlink():
            if out_lbl_val.is_symlink():
                out_lbl_val.unlink()
            else:
                shutil.rmtree(out_lbl_val)

        out_img_val.parent.mkdir(parents=True, exist_ok=True)
        out_lbl_val.parent.mkdir(parents=True, exist_ok=True)
        try:
            out_img_val.symlink_to(val_img_ref.resolve(), target_is_directory=True)
            out_lbl_val.symlink_to(val_lbl_ref.resolve(), target_is_directory=True)
            print("[INFO] Successfully symlinked images/val and labels/val.")
        except Exception as e:
            print(f"[WARN] Symlink failed ({e}), copying directory structure instead...")
            shutil.copytree(val_img_ref, out_img_val)
            shutil.copytree(val_lbl_ref, out_lbl_val)
    else:
        print(f"[ERROR] Validation reference split not found at: {val_ref_p}")
        sys.exit(1)

    # 2. Build Stride=2 Train Split in parallel
    raw_train_p = Path(args.raw_train_root)
    subdirs = sorted([d for d in raw_train_p.iterdir() if d.is_dir()])
    print(f"\n[INFO] Found {len(subdirs)} raw video sequences in {raw_train_p}.")
    print(f"[INFO] Starting parallel Stride={args.stride} GMC+Median generation with {args.workers} workers...")

    t0 = time.time()
    total_generated = 0
    total_fails = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for d in subdirs:
            f = executor.submit(
                process_sequence_stride2,
                str(d),
                str(out_img_train),
                str(out_lbl_train),
                args.stride,
                args.window,
                args.downscale,
            )
            futures.append(f)

        for i, f in enumerate(futures):
            res = f.result()
            total_generated += res["success"]
            total_fails += res["fail"]
            if (i + 1) % 10 == 0 or (i + 1) == len(futures):
                elapsed = time.time() - t0
                fps = total_generated / max(1.0, elapsed)
                print(
                    f"  [{i+1:>3}/{len(futures)}] Processed {res['seq']:<30} | "
                    f"Generated: {res['success']:>4} | Total: {total_generated:>6} ({fps:.1f} imgs/s)"
                )

    total_time = time.time() - t0
    print(f"\n[SUCCESS] Train split built in {total_time:.1f}s ({total_generated} images generated, {total_fails} failed).")

    # 3. Write data.yaml
    yaml_content = f"""# Ultralytics UAV Dataset: Stride=2 Continuous Sequence Mode
path: {out_p.resolve()}
train: images/train
val: images/val

names:
  0: uav
"""
    yaml_path = out_p / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"[SUCCESS] Saved data configuration to: {yaml_path.resolve()}\n")


if __name__ == "__main__":
    main()
