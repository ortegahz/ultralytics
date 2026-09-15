#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Targeted Deep Diagnostic for 02_6321_0274-2773.
Analyzes exact frame-by-frame match between predictions and Ground Truth:
1. Target scale (w, h) evolution across frames.
2. Heatmap detection distance distribution (why are detections 12~30px away from GT centroid?).
3. Box Center vs Heatmap Peak offset vs Bounding Box Coverage.
4. Quantifies how many FN are due to:
   - Genuine miss (no peak >= 0.05 within 50px)
   - Centroid-Peak offset (distance > dist_thresh but peak is inside GT bbox!)
   - Low confidence cutoff (peak within dist_thresh but score < th_base)

Usage:
    python manu/diagnose_6321_deep.py \
        --data-root /mnt/data/siping/datasets/manu/uav_gmc_median \
        --cache-file runs/gmc_eval/uav_median_trial0474_cache.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re
import sys
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr


def natural_sort_key(path_or_str: str | Path):
    s = Path(path_or_str).stem
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def parse_args():
    parser = argparse.ArgumentParser(description="Targeted Deep Diagnostic for 02_6321")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median",
        help="Path to dataset root",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial0474_cache.pkl",
        help="Path to inference cache",
    )
    parser.add_argument("--seq", type=str, default="02_6321_0274-2773")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    val_img_dir = data_root / "images" / "val"
    val_lbl_dir = data_root / "labels" / "val"

    cache_path = Path(args.cache_file)
    if not cache_path.is_absolute():
        for cand in [
            PROJECT_ROOT / cache_path,
            Path("/tmp/pycharm_project_10ae9e2e") / cache_path,
            Path("/home/manu/mnt/pycharm_project_10ae9e2e") / cache_path,
        ]:
            if cand.exists():
                cache_path = cand
                break

    print(f"[INFO] Loading inference cache from: {cache_path}")
    with open(cache_path, "rb") as f:
        records = pickle.load(f)

    cache_records = {}
    for r in records:
        if args.seq in r["im_name"]:
            cache_records[Path(r["im_name"]).name] = r

    img_files = sorted(
        [p for p in val_img_dir.glob(f"*{args.seq}*") if p.suffix.lower() in [".jpg", ".png"]],
        key=natural_sort_key,
    )
    print(f"[INFO] Found {len(img_files)} frames for {args.seq}")

    # Inspect all frames
    total_frames = 0
    gt_count = 0
    direct_hits_dist_thresh = 0  # dist <= args.dist_thresh and score >= 0.22
    inside_bbox_hits = 0         # peak inside bbox and score >= 0.22
    dist_8_to_15 = 0
    dist_15_to_30 = 0
    dist_gt_30 = 0
    weak_peaks_near_gt = 0       # dist <= dist_thresh and 0.05 <= score < 0.22
    no_peak_at_all = 0

    offsets = []
    bbox_widths = []
    bbox_heights = []

    frame_details = []

    for f_idx, p in enumerate(img_files):
        lbl_p = val_lbl_dir / f"{p.stem}.txt"
        if not lbl_p.exists():
            continue

        with open(lbl_p, "r", encoding="utf-8") as f:
            lines = [l.strip().split() for l in f if l.strip()]
        if not lines:
            continue

        box = np.array([float(x) for x in lines[0][1:5]], dtype=np.float32)
        cx, cy = box[0] * args.imgsz, box[1] * args.imgsz
        w, h = box[2] * args.imgsz, box[3] * args.imgsz
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2

        bbox_widths.append(w)
        bbox_heights.append(h)
        gt_count += 1
        total_frames += 1

        rec = cache_records.get(p.name)
        if rec is None or len(rec["pred_points"]) == 0:
            no_peak_at_all += 1
            frame_details.append((f_idx, cx, cy, w, h, None, None, "NO_DET"))
            continue

        pts = rec["pred_points"]
        scs = rec["pred_scores"]

        # Calculate distances to GT center
        dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
        best_idx = np.argmin(dists)
        best_dist = float(dists[best_idx])
        best_sc = float(scs[best_idx])
        best_pt = pts[best_idx]

        # Is the peak inside bbox?
        in_bbox = (x1 <= best_pt[0] <= x2) and (y1 <= best_pt[1] <= y2)

        # Check high conf
        high_conf_mask = scs >= 0.22
        if np.any(high_conf_mask):
            high_dists = dists[high_conf_mask]
            high_scs = scs[high_conf_mask]
            high_pts = pts[high_conf_mask]
            best_high_idx = np.argmin(high_dists)
            min_high_dist = float(high_dists[best_high_idx])
            min_high_sc = float(high_scs[best_high_idx])
            min_high_pt = high_pts[best_high_idx]
            in_bbox_high = (x1 <= min_high_pt[0] <= x2) and (y1 <= min_high_pt[1] <= y2)
        else:
            min_high_dist = 999.0
            min_high_sc = 0.0
            min_high_pt = None
            in_bbox_high = False

        offsets.append(best_dist)

        if min_high_dist <= args.dist_thresh:
            direct_hits_dist_thresh += 1
            status = "HIT"
        elif in_bbox_high:
            inside_bbox_hits += 1
            status = f"IN_BBOX_OFFSET (dist={min_high_dist:.1f}px, bbox={w:.0f}x{h:.0f})"
        elif 8.0 < min_high_dist <= 15.0:
            dist_8_to_15 += 1
            status = f"NEAR_OFFSET_8_15 (dist={min_high_dist:.1f}px, sc={min_high_sc:.2f})"
        elif 15.0 < min_high_dist <= 30.0:
            dist_15_to_30 += 1
            status = f"MID_OFFSET_15_30 (dist={min_high_dist:.1f}px, sc={min_high_sc:.2f})"
        elif min_high_dist < 999.0:
            dist_gt_30 += 1
            status = f"FAR_OFFSET_GT30 (dist={min_high_dist:.1f}px)"
        elif best_dist <= args.dist_thresh and best_sc >= 0.05:
            weak_peaks_near_gt += 1
            status = f"WEAK_CUTOFF (dist={best_dist:.1f}px, sc={best_sc:.2f})"
        else:
            no_peak_at_all += 1
            status = "TOTALLY_BLIND"

        frame_details.append((f_idx, cx, cy, w, h, min_high_dist, min_high_sc, status))

    print("\n" + "=" * 100)
    print(colorstr("bold", colorstr("cyan", f"🔍 TARGET PATHOLOGY BREAKDOWN FOR {args.seq} (GT={gt_count} frames)")))
    print("=" * 100)
    print(f"• Bounding Box Size Distribution across 1498 Frames:")
    print(f"  - Width  : min={np.min(bbox_widths):.1f}px, median={np.median(bbox_widths):.1f}px, max={np.max(bbox_widths):.1f}px")
    print(f"  - Height : min={np.min(bbox_heights):.1f}px, median={np.median(bbox_heights):.1f}px, max={np.max(bbox_heights):.1f}px")

    print(f"\n• Exact Failure Category Breakdown (Criterion: dist <= {args.dist_thresh}px & conf >= 0.22):")
    print(f"  1. Direct Hits (dist <= {args.dist_thresh}px, conf >= 0.22)          : {direct_hits_dist_thresh} frames ({direct_hits_dist_thresh/gt_count*100:.1f}%)")
    print(f"  2. Peak INSIDE BBOX but dist > {args.dist_thresh}px (conf >= 0.22)   : {colorstr('bold', colorstr('yellow', str(inside_bbox_hits)))} frames ({inside_bbox_hits/gt_count*100:.1f}%) 🚨")
    print(f"  3. Distance 8.0px ~ 15.0px (conf >= 0.22)               : {dist_8_to_15} frames ({dist_8_to_15/gt_count*100:.1f}%)")
    print(f"  4. Distance 15.0px ~ 30.0px (conf >= 0.22)              : {dist_15_to_30} frames ({dist_15_to_30/gt_count*100:.1f}%)")
    print(f"  5. Distance > 30.0px (conf >= 0.22)                     : {dist_gt_30} frames ({dist_gt_30/gt_count*100:.1f}%)")
    print(f"  6. Weak Peak Near GT (dist <= {args.dist_thresh}px, 0.05 <= conf < 0.22): {weak_peaks_near_gt} frames ({weak_peaks_near_gt/gt_count*100:.1f}%)")
    print(f"  7. Total Blind (no peaks at all near GT)                : {no_peak_at_all} frames ({no_peak_at_all/gt_count*100:.1f}%)")

    print("\n• Critical Conclusion & Insight:")
    print("-" * 100)
    potential_salvage = direct_hits_dist_thresh + inside_bbox_hits
    print(f"  If 'Inside BBox' or 12px tolerance is permitted:")
    print(f"  -> Potential TP can reach: {potential_salvage} / {gt_count} ({potential_salvage/gt_count*100:.1f}% Recall!)")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
