#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Parameter grid search / sweep for Bidirectional Spatio-Temporal Smoothed Track Fusion (Scheme 1)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr
from manu.eval_bidirectional_track_fusion import (
    evaluate_sequence_bidirectional,
    extract_seq_name,
    calc_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Tune Bidirectional Track Fusion Parameters")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial22_cache.pkl",
        help="Path to Trial 22 inference cache",
    )
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="Evaluation tolerance (default: 8.0px)")
    return parser.parse_args()


def main():
    args = parse_args()
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

    if not cache_path.exists():
        print(colorstr("red", f"[ERROR] Cache not found: {args.cache_file}"))
        sys.exit(1)

    print(f"Loading cache: {cache_path}")
    with open(cache_path, "rb") as f:
        records = pickle.load(f)
    print(f"Loaded {len(records)} predictions.\n")

    seq_records: Dict[str, List[Dict]] = {}
    for r in records:
        seq = extract_seq_name(r["im_name"])
        seq_records.setdefault(seq, []).append(r)

    # Systematic multi-dimensional grid to squeeze F1 beyond 0.9160+:
    # (th_base, th_salvage, th_ground, stitch_gap, infill_gap, min_hits_infill)
    grid = [
        # Baseline reference that achieved 0.9143
        (0.22, 0.06, 0.32, 4, 3, 4),
        # 1. Stricter infill gap (reduce FP leakage during occlusion)
        (0.22, 0.06, 0.32, 4, 2, 4),
        (0.22, 0.06, 0.32, 4, 1, 4),
        # 2. Require higher track maturity before allowing infill (min_hits_infill = 5 or 6)
        (0.22, 0.06, 0.32, 4, 2, 5),
        (0.22, 0.06, 0.32, 4, 3, 5),
        (0.22, 0.06, 0.32, 4, 2, 6),
        # 3. Ground suppression fine-tuning (th_ground = 0.34, 0.36 to crush ground clutter)
        (0.22, 0.06, 0.34, 4, 2, 5),
        (0.22, 0.06, 0.36, 4, 2, 5),
        (0.22, 0.06, 0.35, 4, 3, 5),
        # 4. Base threshold sweep (th_base = 0.23, 0.24, 0.25)
        (0.23, 0.06, 0.32, 4, 2, 4),
        (0.24, 0.06, 0.32, 4, 2, 4),
        (0.24, 0.06, 0.34, 4, 2, 5),
        (0.25, 0.06, 0.32, 4, 2, 4),
        (0.25, 0.06, 0.32, 4, 3, 4),
        # 5. Salvage threshold sensitivity (th_salvage = 0.07, 0.08)
        (0.22, 0.07, 0.32, 4, 2, 5),
        (0.22, 0.08, 0.32, 4, 2, 4),
        (0.24, 0.07, 0.34, 4, 2, 5),
        # 6. Aggressive stitching (stitch_gap = 5, 6)
        (0.22, 0.06, 0.34, 5, 2, 5),
        (0.24, 0.06, 0.34, 5, 2, 5),
    ]

    print("=" * 110)
    print(f"{'Config (base/salv/grnd/stitch/infill/min_h)':<45} | {'TP / GT':<14} | {'FP':<6} | {'Recall':<8} | {'Prec':<8} | {'F1-Score':<8}")
    print("=" * 110)

    best_f1 = -1.0
    best_cfg = None
    best_stats = None

    for th_b, th_s, th_g, s_gap, i_gap, m_h_inf in grid:
        tracker_config = {
            "max_age": 3,
            "min_hits": 3,
            "match_dist": 12.0,
            "max_match_dist": 18.0,
            "min_track_score": 0.08,
            "instant_conf": 0.25,
            "min_displacement": 2.5,
            "sky_ratio": 0.60,
            "img_h": 640,
        }
        smoother_config = {
            "stitch_max_gap": s_gap,
            "stitch_max_dist": 25.0,
            "min_hits_for_infill": m_h_inf,
            "max_infill_gap": i_gap,
            "min_track_hits": 3,
            "min_track_score": 0.08,
            "instant_conf": 0.25,
        }

        total_tp = 0
        total_fp = 0
        total_gt = 0

        for seq_name, recs in seq_records.items():
            res = evaluate_sequence_bidirectional(
                records=recs,
                dist_thresh=args.dist_thresh,
                th_base=th_b,
                th_salvage=th_s,
                th_ground=th_g,
                sky_ratio=0.60,
                img_h=640,
                tracker_config=tracker_config,
                smoother_config=smoother_config,
            )
            m = res["bidirectional"]
            total_tp += int(m["tp"])
            total_fp += int(m["fp"])
            total_gt += int(m["gt"])

        metrics = calc_metrics(total_tp, total_fp, total_gt)
        cfg_str = f"b={th_b:.2f}/s={th_s:.2f}/g={th_g:.2f}/st={s_gap}/inf={i_gap}/h={m_h_inf}"

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_cfg = cfg_str
            best_stats = metrics

        line = f"{cfg_str:<45} | {metrics['tp']:>5} / {metrics['gt']:<6} | {metrics['fp']:<6} | {metrics['recall']:>6.2f}% | {metrics['precision']:>6.2f}% | {metrics['f1']:>6.4f}"
        if metrics["f1"] >= 0.9114:
            print(colorstr("bold", colorstr("green", line)))
        else:
            print(line)

    print("=" * 110)
    print(colorstr("bold", f"BEST CONFIG: {best_cfg} -> F1: {best_f1:.4f} (Recall: {best_stats['recall']:.2f}%, Prec: {best_stats['precision']:.2f}%)"))
    print("=" * 110)


if __name__ == "__main__":
    main()
