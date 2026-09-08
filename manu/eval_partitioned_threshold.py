#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Partitioned Threshold Verification Script for Hard-case Sequences.

Evaluates how many low-confidence true positives are recovered when using:
1. Sky-partitioned threshold (low threshold in sky, standard threshold on ground)
2. Global low threshold vs standard threshold (for direct comparison)

Runs purely from precomputed inference_cache.pkl in seconds (ZERO model inference cost).

Usage:
    python manu/eval_partitioned_threshold.py \
        --cache-file /tmp/pycharm_project_10ae9e2e/runs/badcase_analysis/inference_cache.pkl \
        --sequences DJI_0175_2,wg2022_ir_011_split_03
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Test partitioned / lowered thresholds on hard cases")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="/home/manu/mnt/pycharm_project_10ae9e2e/runs/badcase_analysis/inference_cache.pkl",
        help="Path to inference_cache.pkl",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default="DJI_0175_2,wg2022_ir_011_split_03,DJI_0051_2,wg2022_ir_020_split_03",
        help="Comma-separated sequence filters",
    )
    parser.add_argument("--conf-sky", type=float, default=0.06, help="Low threshold for sky region (default: 0.06)")
    parser.add_argument("--conf-ground", type=float, default=0.20, help="Strict threshold for ground (default: 0.20)")
    parser.add_argument("--sky-ratio", type=float, default=0.60, help="Upper fraction of image treated as sky (default: top 60%)")
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="Distance tolerance for TP (default: 8.0px)")
    parser.add_argument("--img-height", type=int, default=640, help="Image height")
    return parser.parse_args()


def extract_seq_name(im_name: str) -> str:
    stem = Path(im_name).stem
    if "___" in stem:
        return stem.split("___")[0]
    if "__" in stem:
        return stem.split("__")[0]
    match = re.search(r"^(.*?)(?:[_-]+)?\d{3,}$", stem)
    return match.group(1).rstrip("_-") if match else stem


def match_points(gt_pts: np.ndarray, pred_pts: np.ndarray, dist_thresh: float):
    """Match ground truths and predictions based on Euclidean distance within dist_thresh."""
    matched_gt = set()
    matched_pred = set()

    if len(pred_pts) > 0 and len(gt_pts) > 0:
        diff = pred_pts[:, np.newaxis, :] - gt_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))

        p_inds, g_inds = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
        for p_i, g_i in zip(p_inds, g_inds):
            if dists[p_i, g_i] > dist_thresh:
                break
            if p_i not in matched_pred and g_i not in matched_gt:
                matched_pred.add(p_i)
                matched_gt.add(g_i)

    tp = len(matched_gt)
    fp = len(pred_pts) - tp
    fn = len(gt_pts) - tp
    return tp, fp, fn


def evaluate_records(records: list[dict], filter_fn, dist_thresh: float):
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for r in records:
        gt_pts = r["gt_pts"]
        pred_pts = r["pred_points"]
        pred_scores = r["pred_scores"]

        keep = filter_fn(pred_pts, pred_scores)
        filtered_preds = pred_pts[keep]

        tp, fp, fn = match_points(gt_pts, filtered_preds, dist_thresh)
        total_gt += len(gt_pts)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    rec = (total_tp / total_gt * 100.0) if total_gt > 0 else 0.0
    prec = (total_tp / (total_tp + total_fp) * 100.0) if (total_tp + total_fp) > 0 else 0.0
    f1 = (2 * rec * prec / (rec + prec)) if (rec + prec) > 0 else 0.0
    return {
        "gt": total_gt,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "recall": rec,
        "precision": prec,
        "f1": f1,
    }


def main():
    args = parse_args()
    cache_path = Path(args.cache_file)
    if not cache_path.exists():
        # Fallback to local /tmp path if present
        alt_path = Path("/tmp/pycharm_project_10ae9e2e/runs/badcase_analysis/inference_cache.pkl")
        if alt_path.exists():
            cache_path = alt_path
        else:
            raise FileNotFoundError(f"Cache not found: {cache_path}")

    print(f"Loading cached predictions from: {cache_path}")
    with open(cache_path, "rb") as f:
        all_records = pickle.load(f)
    print(f"Loaded {len(all_records)} frame records.")

    # Group by sequence
    seq_records: dict[str, list[dict]] = {}
    for r in all_records:
        seq = extract_seq_name(r["im_name"])
        seq_records.setdefault(seq, []).append(r)

    target_seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]

    sky_y_boundary = args.img_height * args.sky_ratio

    print("\n" + "=" * 95)
    print(f"{'Sequence':<25} | {'Mode':<18} | {'TP/GT':<12} | {'FP':<6} | {'Recall':<8} | {'Precision':<8} | {'F1':<6}")
    print("=" * 95)

    for target in target_seqs:
        # Match sequence names leniently
        matched_keys = [k for k in seq_records.keys() if target in k]
        if not matched_keys:
            print(f"[WARN] Sequence '{target}' not found in cache.")
            continue

        for seq_name in matched_keys:
            recs = seq_records[seq_name]

            # 1. Baseline: Standard global conf = 0.20
            res_std = evaluate_records(
                recs,
                filter_fn=lambda pts, sc: sc >= 0.20,
                dist_thresh=args.dist_thresh,
            )

            # 2. Partitioned: Sky (y < boundary) conf >= conf_sky, Ground (y >= boundary) conf >= conf_ground
            def partitioned_filter(pts, sc):
                if len(pts) == 0:
                    return np.array([], dtype=bool)
                is_sky = pts[:, 1] < sky_y_boundary
                keep_sky = is_sky & (sc >= args.conf_sky)
                keep_ground = (~is_sky) & (sc >= args.conf_ground)
                return keep_sky | keep_ground

            res_part = evaluate_records(
                recs,
                filter_fn=partitioned_filter,
                dist_thresh=args.dist_thresh,
            )

            # 3. Global low conf = 0.06 (to see raw potential and FP explosion)
            res_low = evaluate_records(
                recs,
                filter_fn=lambda pts, sc: sc >= args.conf_sky,
                dist_thresh=args.dist_thresh,
            )

            print(
                f"{seq_name:<25} | {'Baseline (0.20)':<18} | {res_std['tp']:>4}/{res_std['gt']:<5} "
                f"| {res_std['fp']:<6} | {res_std['recall']:>5.1f}%  | {res_std['precision']:>5.1f}%    | {res_std['f1']:>4.1f}%"
            )
            print(
                f"{'':<25} | {f'Partition ({args.conf_sky}/{args.conf_ground})':<18} | {res_part['tp']:>4}/{res_part['gt']:<5} "
                f"| {res_part['fp']:<6} | {res_part['recall']:>5.1f}%  | {res_part['precision']:>5.1f}%    | {res_part['f1']:>4.1f}%"
            )
            print(
                f"{'':<25} | {f'Global Low ({args.conf_sky})':<18} | {res_low['tp']:>4}/{res_low['gt']:<5} "
                f"| {res_low['fp']:<6} | {res_low['recall']:>5.1f}%  | {res_low['precision']:>5.1f}%    | {res_low['f1']:>4.1f}%"
            )
            print("-" * 95)


if __name__ == "__main__":
    main()
