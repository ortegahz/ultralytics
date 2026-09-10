#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Build Multi-Timeframe Heatmap (MTH) Dataset with Precise GMC Alignment:
Dataset Destination: /mnt/data/siping/datasets/manu/uav_gmc_mth

Key Design:
1. Reuses existing 3-channel images from /mnt/data/siping/datasets/manu/uav_gmc_median (symlinks, zero disk expansion).
2. Reads ground truth from anti-uav sequence IR_label.json for physical 1-frame continuity (t-1, t, t+1).
3. Computes affine transformation H_{t-1 -> t} and H_{t+1 -> t} using SparseOptFlow to align past and future coordinates.
4. Outputs:
   - images/{split}/*.jpg (symlinks to uav_gmc_median)
   - labels/{split}/*.txt (standard 5-column YOLO format: class x_curr y_curr w_curr h_curr, 100% compatible with YOLO loader)
   - mth_labels/{split}/*.txt (extended multi-timeframe format: class x_curr y_curr w_curr h_curr x_past y_past x_fut y_fut exist_past exist_fut)

Usage on Server:
    python manu/build_mth_dataset.py \
        --ref-median /mnt/data/siping/datasets/manu/uav_gmc_median \
        --raw-root /mnt/data/siping/datasets/manu/anti-uav \
        --output /mnt/data/siping/datasets/manu/uav_gmc_mth \
        --workers 16
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build Multi-Timeframe Heatmap Dataset")
    parser.add_argument(
        "--ref-median",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median",
        help="Path to existing uav_gmc_median dataset",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/anti-uav",
        help="Path to raw anti-uav sequences containing IR_label.json and frames",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_mth",
        help="Destination directory for uav_gmc_mth",
    )
    parser.add_argument("--workers", type=int, default=16, help="Process pool workers")
    parser.add_argument("--splits", type=str, default="train,val", help="Splits to process")
    parser.add_argument("--link-images", action="store_true", default=True, help="Symlink images to save disk")
    return parser.parse_args()


def natural_key(p: Path | str):
    stem = Path(p).stem
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", stem)]


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
    raise ValueError(f"Cannot parse sequence name: {im_name}")


class GMCAligner:
    def __init__(self, max_points: int = 400):
        self.feature_detector = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
        self.max_points = max_points

    def compute_affine(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        h, w = curr_gray.shape[:2]
        if prev_gray.shape[:2] != (h, w):
            return np.eye(2, 3, dtype=np.float32)

        sh, sw = h // 2, w // 2
        p_small = cv2.resize(prev_gray, (sw, sh), interpolation=cv2.INTER_AREA)
        c_small = cv2.resize(curr_gray, (sw, sh), interpolation=cv2.INTER_AREA)

        keypoints = self.feature_detector.detect(p_small, None)
        if len(keypoints) < 8:
            return np.eye(2, 3, dtype=np.float32)

        pts_p = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        if len(pts_p) > self.max_points:
            indices = np.random.choice(len(pts_p), self.max_points, replace=False)
            pts_p = pts_p[indices]

        pts_c, status, _ = cv2.calcOpticalFlowPyrLK(
            p_small, c_small, pts_p, None, winSize=(15, 15), maxLevel=2
        )
        if pts_c is None or status is None:
            return np.eye(2, 3, dtype=np.float32)

        good_p = pts_p[status.ravel() == 1]
        good_c = pts_c[status.ravel() == 1]
        if len(good_p) < 6:
            return np.eye(2, 3, dtype=np.float32)

        H, _ = cv2.estimateAffinePartial2D(good_p, good_c, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H is None:
            return np.eye(2, 3, dtype=np.float32)

        H_full = H.copy()
        H_full[0, 2] *= 2.0
        H_full[1, 2] *= 2.0
        return H_full.astype(np.float32)

    def warp_point(self, pt: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
        x, y = pt
        new_x = H[0, 0] * x + H[0, 1] * y + H[0, 2]
        new_y = H[1, 0] * x + H[1, 1] * y + H[1, 2]
        return float(new_x), float(new_y)


def process_sequence_mth(
    seq_name: str,
    im_names: list[str],
    ref_img_dir: Path,
    out_lbl_dir: Path,
    out_mth_dir: Path,
    raw_seq_dir: Path,
    img_h: int = 512,
    img_w: int = 640,
) -> dict:
    json_path = raw_seq_dir / "IR_label.json"
    if not json_path.exists():
        for cand in raw_seq_dir.rglob("IR_label.json"):
            json_path = cand
            break

    if not json_path.exists():
        return {"seq": seq_name, "success": 0, "status": "no_json"}

    with open(json_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    gt_rects = gt_data.get("gt_rect", [])
    gt_exists = gt_data.get("exist", [1] * len(gt_rects))
    num_total_frames = len(gt_rects)

    aligner = GMCAligner()
    frame_cache = {}
    success_cnt = 0

    im_names_sorted = sorted(im_names, key=natural_key)

    for im_name in im_names_sorted:
        _, frame_idx = parse_seq_and_frame(im_name)

        curr_idx = max(0, min(frame_idx - 1, num_total_frames - 1))
        past_idx = max(0, curr_idx - 1)
        fut_idx = min(num_total_frames - 1, curr_idx + 1)

        rect_curr = gt_rects[curr_idx]
        exist_curr = gt_exists[curr_idx]
        rect_past = gt_rects[past_idx]
        exist_past = gt_exists[past_idx]
        rect_fut = gt_rects[fut_idx]
        exist_fut = gt_exists[fut_idx]

        stem = Path(im_name).stem
        out_std_file = out_lbl_dir / f"{stem}.txt"
        out_mth_file = out_mth_dir / f"{stem}.txt"

        if exist_curr == 0 or len(rect_curr) < 4:
            out_std_file.write_text("", encoding="utf-8")
            out_mth_file.write_text("", encoding="utf-8")
            success_cnt += 1
            continue

        # Centers and sizes
        cx_curr = rect_curr[0] + rect_curr[2] / 2.0
        cy_curr = rect_curr[1] + rect_curr[3] / 2.0
        w_curr = rect_curr[2]
        h_curr = rect_curr[3]

        if exist_past and len(rect_past) >= 4:
            cx_past = rect_past[0] + rect_past[2] / 2.0
            cy_past = rect_past[1] + rect_past[3] / 2.0
        else:
            cx_past, cy_past = cx_curr, cy_curr
            exist_past = 0

        if exist_fut and len(rect_fut) >= 4:
            cx_fut = rect_fut[0] + rect_fut[2] / 2.0
            cy_fut = rect_fut[1] + rect_fut[3] / 2.0
        else:
            cx_fut, cy_fut = cx_curr, cy_curr
            exist_fut = 0

        # GMC Affine Alignment across t-1 -> t and t+1 -> t
        p_curr_raw = raw_seq_dir / f"{curr_idx + 1:06d}.jpg"
        p_past_raw = raw_seq_dir / f"{past_idx + 1:06d}.jpg"
        p_fut_raw = raw_seq_dir / f"{fut_idx + 1:06d}.jpg"

        if p_curr_raw.exists() and p_past_raw.exists() and p_fut_raw.exists():
            for p, idx in [(p_curr_raw, curr_idx), (p_past_raw, past_idx), (p_fut_raw, fut_idx)]:
                if idx not in frame_cache:
                    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    frame_cache[idx] = im

            im_c = frame_cache[curr_idx]
            im_p = frame_cache[past_idx]
            im_f = frame_cache[fut_idx]

            if im_c is not None and im_p is not None:
                H_p2c = aligner.compute_affine(im_p, im_c)
                cx_past, cy_past = aligner.warp_point((cx_past, cy_past), H_p2c)

            if im_c is not None and im_f is not None:
                H_f2c = aligner.compute_affine(im_f, im_c)
                cx_fut, cy_fut = aligner.warp_point((cx_fut, cy_fut), H_f2c)

        # Normalize
        norm_cx_curr = np.clip(cx_curr / img_w, 0.0, 1.0)
        norm_cy_curr = np.clip(cy_curr / img_h, 0.0, 1.0)
        norm_w_curr = np.clip(w_curr / img_w, 0.0, 1.0)
        norm_h_curr = np.clip(h_curr / img_h, 0.0, 1.0)

        norm_cx_past = np.clip(cx_past / img_w, 0.0, 1.0)
        norm_cy_past = np.clip(cy_past / img_h, 0.0, 1.0)
        norm_cx_fut = np.clip(cx_fut / img_w, 0.0, 1.0)
        norm_cy_fut = np.clip(cy_fut / img_h, 0.0, 1.0)

        # 1. Standard 5-column YOLO label (Guarantees compatibility with official YOLO dataloader)
        line_std = f"0 {norm_cx_curr:.6f} {norm_cy_curr:.6f} {norm_w_curr:.6f} {norm_h_curr:.6f}\n"
        out_std_file.write_text(line_std, encoding="utf-8")

        # 2. Extended Multi-timeframe label
        line_mth = (
            f"0 {norm_cx_curr:.6f} {norm_cy_curr:.6f} {norm_w_curr:.6f} {norm_h_curr:.6f} "
            f"{norm_cx_past:.6f} {norm_cy_past:.6f} {norm_cx_fut:.6f} {norm_cy_fut:.6f} "
            f"{int(exist_past)} {int(exist_fut)}\n"
        )
        out_mth_file.write_text(line_mth, encoding="utf-8")
        success_cnt += 1

        if len(frame_cache) > 30:
            frame_cache.clear()

    return {"seq": seq_name, "success": success_cnt, "status": "ok"}


def process_split(
    split: str,
    ref_median: Path,
    raw_root: Path,
    out_dir: Path,
    workers: int,
    link_images: bool,
):
    print(f"\n==================== Building MTH Split [{split}] ====================")
    ref_img_dir = ref_median / "images" / split
    out_img_dir = out_dir / "images" / split
    out_lbl_dir = out_dir / "labels" / split
    out_mth_dir = out_dir / "mth_labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    out_mth_dir.mkdir(parents=True, exist_ok=True)

    if not ref_img_dir.is_dir():
        print(f"[WARN] Reference split not found: {ref_img_dir}, skipping.")
        return

    img_files = [p for p in ref_img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    print(f"[{split}] Discovered {len(img_files)} images.")

    if link_images:
        print(f"[{split}] Creating image symlinks from {ref_img_dir}...")
        for p in img_files:
            dst_link = out_img_dir / p.name
            if not dst_link.exists():
                try:
                    os.symlink(p.resolve(), dst_link)
                except Exception:
                    pass

    seq_groups: dict[str, list[str]] = {}
    for p in img_files:
        seq, _ = parse_seq_and_frame(p.name)
        seq_groups.setdefault(seq, []).append(p.name)

    print(f"[{split}] Grouped into {len(seq_groups)} sequences. Processing labels with {workers} workers...")

    raw_seq_map = {}
    for p in raw_root.rglob("*"):
        if p.is_dir() and (p / "IR_label.json").exists():
            raw_seq_map[p.name] = p

    t0 = time.time()
    total_success = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for seq_name, names in seq_groups.items():
            raw_dir = raw_seq_map.get(seq_name)
            if raw_dir is None:
                cand = raw_root / "train" / seq_name
                raw_dir = cand if cand.is_dir() else raw_root / seq_name

            futures.append(
                executor.submit(
                    process_sequence_mth,
                    seq_name=seq_name,
                    im_names=names,
                    ref_img_dir=ref_img_dir,
                    out_lbl_dir=out_lbl_dir,
                    out_mth_dir=out_mth_dir,
                    raw_seq_dir=raw_dir,
                )
            )

        for fut in futures:
            res = fut.result()
            total_success += res["success"]

    elapsed = time.time() - t0
    print(f"[{split}] Complete! Standard & MTH Labels generated for {total_success} frames in {elapsed:.1f}s.")


def main():
    args = parse_args()
    ref_median = Path(args.ref_median)
    raw_root = Path(args.raw_root)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        process_split(
            split=split,
            ref_median=ref_median,
            raw_root=raw_root,
            out_dir=out_dir,
            workers=args.workers,
            link_images=args.link_images,
        )

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"""path: {out_dir.resolve()}
train: images/train
val: images/val

names:
  0: uav
""",
        encoding="utf-8",
    )
    print(f"\n[SUCCESS] Multi-Timeframe Dataset created at: {out_dir.resolve()}")
    print(f"data.yaml written to: {data_yaml.resolve()}\n")


if __name__ == "__main__":
    main()
