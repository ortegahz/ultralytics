#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Grid Optimizer for Rigid Static Pruner Parameters.

Explores:
- min_rigid_disp: [0.8, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5]
- max_rigid_var: [0.15, 0.25, 0.35, 0.50, 0.65]
- min_duration_for_prune: [5, 8, 12, 16] (Only prune if track persists long enough to prove it is a static sensor defect)

Goal: Achieve MAXIMUM TP (close to 22,616) while keeping FP as low as possible (~1370), maximizing F1.

Usage:
    python manu/tune_rigid_pruner.py \
        --cache-file runs/gmc_eval/uav_median_trial0474_cache.pkl \
        --th-base 0.22 \
        --th-salvage 0.06 \
        --th-ground 0.35 \
        --min-hits-infill 5
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
    calc_metrics,
    evaluate_sequence_bidirectional,
    extract_seq_name,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Tune Rigid Static Pruner Parameters")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial0474_cache.pkl",
    )
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    parser.add_argument("--th-base", type=float, default=0.22)
    parser.add_argument("--th-salvage", type=float, default=0.06)
    parser.add_argument("--th-ground", type=float, default=0.35)
    parser.add_argument("--min-hits-infill", type=int, default=5)
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

    with open(cache_path, "rb") as f:
        records = pickle.load(f)

    seq_records: Dict[str, List[Dict]] = {}
    for r in records:
        seq = extract_seq_name(r["im_name"])
        seq_records.setdefault(seq, []).append(r)

    all_seq_names = sorted(seq_records.keys())

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

    # Grid search space
    disp_candidates = [1.0, 1.2, 1.5, 1.8, 2.0]
    var_candidates = [0.20, 0.35, 0.50]
    min_hits_prune_candidates = [3, 5, 8]

    # Baseline without pruner (min_rigid_disp=0.0)
    print("\n" + "=" * 115)
    print("🔬 RUNNING RIGID PRUNER GRID OPTIMIZATION")
    print(f"Total Sequences: {len(all_seq_names)} | Total Cache Frames: {len(records)}")
    print("=" * 115)

    base_smoother_cfg = {
        "stitch_max_gap": 4,
        "stitch_max_dist": 25.0,
        "min_hits_for_infill": args.min_hits_infill,
        "max_infill_gap": 3,
        "min_track_hits": 3,
        "min_track_score": 0.08,
        "instant_conf": 0.25,
        "min_rigid_displacement": 0.0,
        "max_rigid_variance": 0.0,
        "min_hits_for_prune": 1,
    }

    tot_tp, tot_fp, tot_gt = 0, 0, 0
    for s_name in all_seq_names:
        res = evaluate_sequence_bidirectional(
            records=seq_records[s_name],
            dist_thresh=args.dist_thresh,
            th_base=args.th_base,
            th_salvage=args.th_salvage,
            th_ground=args.th_ground,
            sky_ratio=0.60,
            img_h=640,
            tracker_config=tracker_cfg,
            smoother_config=base_smoother_cfg,
        )
        bidi = res["bidirectional"]
        tot_tp += bidi["tp"]
        tot_fp += bidi["fp"]
        tot_gt += bidi["gt"]

    m_no_prune = calc_metrics(tot_tp, tot_fp, tot_gt)
    print(
        colorstr(
            "bold",
            f"0. BASELINE (NO PRUNER)       | F1: {m_no_prune['f1']:.4f} | Rec: {m_no_prune['recall']:.2f}% | "
            f"Prec: {m_no_prune['precision']:.2f}% | TP: {tot_tp:,} | FP: {tot_fp:,}",
        )
    )
    print("-" * 115)

    header = f"{'Rank':<4} | {'MinDisp':<7} | {'MaxVar':<6} | {'MinHitsPrune':<12} | {'F1-Score':<8} | {'Recall':<7} | {'Prec':<7} | {'TP':<6} | {'FP':<5} | {'ΔTP':<5} | {'ΔFP':<5}"
    print(header)
    print("-" * 115)

    results = []
    trial_idx = 0
    t0 = time.time()

    for d_cand in disp_candidates:
        for v_cand in var_candidates:
            for h_cand in min_hits_prune_candidates:
                trial_idx += 1
                curr_smoother_cfg = {
                    "stitch_max_gap": 4,
                    "stitch_max_dist": 25.0,
                    "min_hits_for_infill": args.min_hits_infill,
                    "max_infill_gap": 3,
                    "min_track_hits": 3,
                    "min_track_score": 0.08,
                    "instant_conf": 0.25,
                    "min_rigid_displacement": d_cand,
                    "max_rigid_variance": v_cand,
                    "min_hits_for_prune": h_cand,
                }

                tp_acc, fp_acc, gt_acc = 0, 0, 0
                for s_name in all_seq_names:
                    res = evaluate_sequence_bidirectional(
                        records=seq_records[s_name],
                        dist_thresh=args.dist_thresh,
                        th_base=args.th_base,
                        th_salvage=args.th_salvage,
                        th_ground=args.th_ground,
                        sky_ratio=0.60,
                        img_h=640,
                        tracker_config=tracker_cfg,
                        smoother_config=curr_smoother_cfg,
                    )
                    bidi = res["bidirectional"]
                    tp_acc += bidi["tp"]
                    fp_acc += bidi["fp"]
                    gt_acc += bidi["gt"]

                m = calc_metrics(tp_acc, fp_acc, gt_acc)
                delta_tp = tp_acc - tot_tp
                delta_fp = fp_acc - tot_fp

                results.append({
                    "disp": d_cand,
                    "var": v_cand,
                    "hits_prune": h_cand,
                    "f1": m["f1"],
                    "recall": m["recall"],
                    "precision": m["precision"],
                    "tp": tp_acc,
                    "fp": fp_acc,
                    "delta_tp": delta_tp,
                    "delta_fp": delta_fp,
                })

    # Sort results by F1 descending
    results.sort(key=lambda x: (x["f1"], x["tp"]), reverse=True)

    for i, r in enumerate(results[:15], 1):
        f1_str = f"{r['f1']:.4f}"
        rec_str = f"{r['recall']:.2f}%"
        prec_str = f"{r['precision']:.2f}%"
        dtp_str = f"{r['delta_tp']:+d}"
        dfp_str = f"{r['delta_fp']:+d}"
        line = (
            f"{i:<4} | {r['disp']:<7.1f} | {r['var']:<6.2f} | {r['hits_prune']:<12} | "
            f"{f1_str:<8} | {rec_str:<7} | {prec_str:<7} | {r['tp']:<6} | {r['fp']:<5} | "
            f"{dtp_str:<5} | {dfp_str:<5}"
        )
        if i == 1:
            print(colorstr("bold", colorstr("green", line)))
        elif r["delta_tp"] >= -5:
            print(colorstr("cyan", line))
        else:
            print(line)

    print("=" * 115)
    best = results[0]
    print(
        colorstr(
            "bold",
            colorstr(
                "magenta",
                f"★ OPTIMAL CONFIG: min_rigid_disp={best['disp']}, max_rigid_var={best['var']}, min_hits_for_prune={best['hits_prune']}\n"
                f"  F1: {best['f1']:.4f} (Rec: {best['recall']:.2f}%, Prec: {best['precision']:.2f}%, TP: {best['tp']}, FP: {best['fp']})\n"
                f"  Gain vs Baseline: ΔF1 = {best['f1'] - m_no_prune['f1']:+.4f} | ΔTP = {best['delta_tp']:+d} | ΔFP = {best['delta_fp']:+d} (Total time: {time.time()-t0:.1f}s)",
            ),
        )
    )
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
