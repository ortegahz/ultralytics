#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultra-fast Offline Grid Sweeper for P0 Highway Bidirectional Tracking System.

Loads runs/gmc_eval/uav_median_p0_highway_cache.pkl and sweeps:
- th_base: [0.22, 0.23, 0.24, 0.25]
- th_salvage: [0.06, 0.07, 0.08, 0.09]
- th_ground: [0.32, 0.35, 0.38]
- min_hits_infill: [4, 5, 6]

Goal: Find configurations where Recall >= 90.00% AND F1-Score >= 0.9155 (Beat Trial 22 SOTA).
Execution: Runs in memory on pre-cached pickle; takes ~30-60 seconds for 100+ configs!
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
import time
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr
from manu.eval_bidirectional_track_fusion import (
    evaluate_sequence_bidirectional,
    extract_seq_name,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep P0 Highway Fusion Parameters")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_p0_highway_cache.pkl",
        help="Path to P0 Highway pickle cache",
    )
    parser.add_argument("--dist-thresh", type=float, default=8.0)
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

    print(colorstr("bold", colorstr("green", f"\n>>> Loading P0 Highway cache from: {cache_path}")))
    with open(cache_path, "rb") as f:
        records = pickle.load(f)
    print(f"Loaded {len(records)} image predictions.\n")

    seq_records: Dict[str, List[Dict]] = {}
    for r in records:
        seq = extract_seq_name(r["im_name"])
        seq_records.setdefault(seq, []).append(r)

    total_gt = sum(len(r["gt_pts"]) for r in records)
    all_seq_names = sorted(seq_records.keys())

    # Parameter Grid
    th_base_list = [0.22, 0.23, 0.24, 0.25]
    th_salvage_list = [0.06, 0.07, 0.08, 0.09]
    th_ground_list = [0.32, 0.35, 0.38]
    min_hits_infill_list = [4, 5, 6]

    print("=" * 110)
    print(f"🚀 Sweeping {len(th_base_list) * len(th_salvage_list) * len(th_ground_list) * len(min_hits_infill_list)} parameter configurations...")
    print(f"Target: Recall >= 90.00% (TP >= 22,600) AND F1 >= 0.9150")
    print("=" * 110)

    results = []
    t0 = time.time()

    for th_base in th_base_list:
        for th_salvage in th_salvage_list:
            if th_salvage >= th_base:
                continue
            for th_ground in th_ground_list:
                for min_h_inf in min_hits_infill_list:
                    tracker_cfg = {
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
                    smoother_cfg = {
                        "stitch_max_gap": 4,
                        "stitch_max_dist": 25.0,
                        "min_hits_for_infill": min_h_inf,
                        "max_infill_gap": 3,
                        "min_track_hits": 3,
                        "min_track_score": 0.08,
                        "instant_conf": 0.25,
                    }

                    total_tp = 0
                    total_fp = 0

                    for s_name in all_seq_names:
                        recs = seq_records[s_name]
                        res = evaluate_sequence_bidirectional(
                            records=recs,
                            dist_thresh=args.dist_thresh,
                            th_base=th_base,
                            th_salvage=th_salvage,
                            th_ground=th_ground,
                            sky_ratio=0.60,
                            img_h=640,
                            tracker_config=tracker_cfg,
                            smoother_config=smoother_cfg,
                        )
                        total_tp += res["bidirectional"]["tp"]
                        total_fp += res["bidirectional"]["fp"]

                    rec = (total_tp / total_gt) * 100.0
                    prec = (total_tp / max(1, total_tp + total_fp)) * 100.0
                    f1 = (2 * rec * prec / max(1e-6, rec + prec))
                    far = total_fp / len(records)

                    cfg_record = {
                        "th_base": th_base,
                        "th_salvage": th_salvage,
                        "th_ground": th_ground,
                        "min_hits_infill": min_h_inf,
                        "tp": total_tp,
                        "fp": total_fp,
                        "gt": total_gt,
                        "recall": rec,
                        "precision": prec,
                        "f1": f1,
                        "far": far,
                    }
                    results.append(cfg_record)

                    if rec >= 90.00 and f1 >= 0.9140:
                        print(
                            f"★ [HIGH SOTA] base={th_base:.2f}, salv={th_salvage:.2f}, grnd={th_ground:.2f}, inf_h={min_h_inf} | "
                            f"F1: {f1:.4f} | Recall: {rec:.2f}% (TP: {total_tp}) | Prec: {prec:.2f}% | FP: {total_fp}"
                        )

    dur = time.time() - t0
    print("=" * 110)
    print(f"Sweep completed in {dur:.1f}s across {len(results)} evaluated configurations.")

    # 1. Top F1 where Recall >= 90.00%
    rec_90_cfgs = [c for c in results if c["recall"] >= 90.00]
    if rec_90_cfgs:
        best_rec90 = max(rec_90_cfgs, key=lambda c: c["f1"])
        print("\n" + colorstr("bold", colorstr("green", "★ BEST CONFIG WITH RECALL >= 90.00% (The Holy Grail):")))
        print(
            f"Parameters : th_base={best_rec90['th_base']}, th_salvage={best_rec90['th_salvage']}, "
            f"th_ground={best_rec90['th_ground']}, min_hits_infill={best_rec90['min_hits_infill']}\n"
            f"F1-Score   : {best_rec90['f1']:.4f}\n"
            f"Recall     : {best_rec90['recall']:.2f}% (TP: {best_rec90['tp']} / {best_rec90['gt']})\n"
            f"Precision  : {best_rec90['precision']:.2f}% (FP: {best_rec90['fp']})\n"
            f"FAR        : {best_rec90['far']:.4f} per frame"
        )
    else:
        print(colorstr("yellow", "\nNo config met Recall >= 90.00% in this sweep."))

    # 2. Overall Best F1 config
    best_overall_f1 = max(results, key=lambda c: c["f1"])
    print("\n" + colorstr("bold", colorstr("cyan", "★ BEST OVERALL F1-SCORE CONFIG:")))
    print(
        f"Parameters : th_base={best_overall_f1['th_base']}, th_salvage={best_overall_f1['th_salvage']}, "
        f"th_ground={best_overall_f1['th_ground']}, min_hits_infill={best_overall_f1['min_hits_infill']}\n"
        f"F1-Score   : {best_overall_f1['f1']:.4f}\n"
        f"Recall     : {best_overall_f1['recall']:.2f}% (TP: {best_overall_f1['tp']} / {best_overall_f1['gt']})\n"
        f"Precision  : {best_overall_f1['precision']:.2f}% (FP: {best_overall_f1['fp']})\n"
        f"FAR        : {best_overall_f1['far']:.4f} per frame\n"
    )


if __name__ == "__main__":
    main()
