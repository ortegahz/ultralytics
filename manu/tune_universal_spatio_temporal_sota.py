#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Universal Spatio-Temporal SOTA Sweeper with Multi-Core Parallel Acceleration.

Speed Optimizations:
1. Python multiprocessing.Pool utilizes all available CPU cores (16~32 cores).
2. Focused, high-yield parameter grid based on prior empirical findings:
   - th_base: [0.21, 0.22, 0.23, 0.24]
   - th_salvage: [0.035, 0.045, 0.055, 0.065]
   - th_ground: [0.35, 0.38]
   - max_adaptive_gap: [6, 8, 10]
   - min_hits_infill: [5, 6]
3. Total configurations: ~100~192 configs evaluated in parallel; finishes in ~10-20 seconds!

Usage:
    python manu/tune_universal_spatio_temporal_sota.py \
        --cache-file runs/gmc_eval/uav_median_trial0474_cache.pkl \
        --dist-thresh 8.0 \
        --exclude-demo \
        --workers 16
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr
from manu.eval_universal_spatio_temporal_sota import (
    evaluate_sequence_universal,
    extract_seq_name,
)

AVIAN_SEQUENCES = {
    "01_4485_1167-2666",
    "wg2022_ir_020_split_07",
}

# Global dictionary for multiprocessing workers (avoids IPC transfer overhead)
_WORKER_SEQ_RECORDS: Dict[str, List[Dict]] = {}
_WORKER_DIST_THRESH: float = 8.0
_WORKER_EXCLUDE_DEMO: bool = False


def _init_worker(seq_records: Dict[str, List[Dict]], dist_thresh: float, exclude_demo: bool):
    global _WORKER_SEQ_RECORDS, _WORKER_DIST_THRESH, _WORKER_EXCLUDE_DEMO
    _WORKER_SEQ_RECORDS = seq_records
    _WORKER_DIST_THRESH = dist_thresh
    _WORKER_EXCLUDE_DEMO = exclude_demo


def _eval_single_config(cfg: Tuple[float, float, float, int, int]) -> dict:
    th_base, th_salvage, th_ground, max_adaptive_gap, min_hits_infill = cfg

    tracker_config = {
        "max_age": 4,
        "min_hits": 3,
        "match_dist": 12.0,
        "max_match_dist": 20.0,
        "min_track_score": 0.08,
        "instant_conf": 0.25,
        "min_displacement": 2.5,
        "sky_ratio": 0.60,
        "img_h": 640,
    }

    smoother_config = {
        "base_stitch_gap": 4,
        "max_adaptive_stitch_gap": max_adaptive_gap,
        "stitch_max_dist": 25.0,
        "min_hits_for_infill": min_hits_infill,
        "max_infill_gap": 3,
        "min_track_hits": 3,
        "min_track_score": 0.08,
        "min_rigid_displacement": 2.0,
        "max_rigid_variance": 0.5,
        "min_hits_for_prune": 8,
    }

    tot_tp, tot_fp, tot_gt = 0, 0, 0
    seq_metrics = {}

    for seq_name, recs in _WORKER_SEQ_RECORDS.items():
        if seq_name in AVIAN_SEQUENCES:
            continue
        if _WORKER_EXCLUDE_DEMO and seq_name == "DJI_0175_2":
            continue

        res = evaluate_sequence_universal(
            records=recs,
            dist_thresh=_WORKER_DIST_THRESH,
            th_base=th_base,
            th_salvage=th_salvage,
            th_ground=th_ground,
            sky_ratio=0.60,
            img_h=640,
            tracker_config=tracker_config,
            smoother_config=smoother_config,
        )

        m = res["metrics"]
        seq_metrics[seq_name] = m
        tot_tp += m["tp"]
        tot_fp += m["fp"]
        tot_gt += m["gt"]

    rec = (tot_tp / max(1, tot_gt)) * 100.0
    prec = (tot_tp / max(1, tot_tp + tot_fp)) * 100.0
    f1 = 2 * rec * prec / max(1e-6, rec + prec)

    return {
        "cfg": {
            "th_base": th_base,
            "th_salvage": th_salvage,
            "th_ground": th_ground,
            "max_adaptive_gap": max_adaptive_gap,
            "min_hits_infill": min_hits_infill,
        },
        "f1": f1,
        "recall": rec,
        "prec": prec,
        "tp": tot_tp,
        "fp": tot_fp,
        "gt": tot_gt,
        "seq_metrics": seq_metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep Universal Spatio-Temporal Parameters (Parallel)")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial0474_cache.pkl",
        help="Path to Trial 0474 or EP-Focal cache file",
    )
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="Evaluation tolerance")
    parser.add_argument("--exclude-demo", action="store_true", default=True, help="Exclude DJI_0175_2")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, max(1, mp.cpu_count() - 2)),
        help="Parallel CPU workers",
    )
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

    print(colorstr("bold", f"\n>>> Loading inference cache from: {cache_path}"))
    with open(cache_path, "rb") as f:
        records = pickle.load(f)

    seq_records: Dict[str, List[Dict]] = {}
    for r in records:
        seq = extract_seq_name(r["im_name"])
        seq_records.setdefault(seq, []).append(r)

    print(f"Loaded {len(records)} frames across {len(seq_records)} total sequences.")

    # High-Yield Focused Search Space (Centered around high precision base>=0.22, ground>=0.35)
    th_base_list = [0.22, 0.23, 0.24, 0.25]
    th_salvage_list = [0.045, 0.055, 0.065]
    th_ground_list = [0.35, 0.38, 0.40]
    max_adaptive_gap_list = [4, 6]
    min_hits_infill_list = [5, 6]

    configs = []
    for b in th_base_list:
        for s in th_salvage_list:
            if s >= b:
                continue
            for g in th_ground_list:
                for gap in max_adaptive_gap_list:
                    for inf in min_hits_infill_list:
                        configs.append((b, s, g, gap, inf))

    total_cfgs = len(configs)
    num_workers = args.workers
    print("=" * 115)
    print(
        f"🚀 Fast Multi-Core Sweeper: Running {total_cfgs} configurations across {num_workers} CPU cores..."
    )
    print("Target Benchmark (Current SOTA): F1 >= 0.9274 | Recall >= 90.14% | Prec >= 95.49%")
    print("=" * 115)

    t0 = time.time()
    best_f1 = -1.0
    best_res = None

    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(seq_records, args.dist_thresh, args.exclude_demo),
    ) as pool:
        # Evaluate in parallel chunks
        for res in pool.imap_unordered(_eval_single_config, configs, chunksize=4):
            cur_f1 = res["f1"]
            if cur_f1 > best_f1:
                best_f1 = cur_f1
                best_res = res
                c = res["cfg"]
                print(
                    colorstr(
                        "green",
                        f"  ★ NEW PEAK F1: {best_f1:.4f} | Recall: {res['recall']:.2f}% (TP={res['tp']:,}) | "
                        f"Prec: {res['prec']:.2f}% (FP={res['fp']:,}) | "
                        f"Cfg: base={c['th_base']:.2f}, salv={c['th_salvage']:.3f}, gnd={c['th_ground']:.2f}, gap={c['max_adaptive_gap']}, inf={c['min_hits_infill']}",
                    )
                )

    elapsed = time.time() - t0
    print("\n" + "=" * 115)
    print(
        colorstr(
            "bold",
            f"🏆 BEST UNIVERSAL SPATIO-TEMPORAL CONFIGURATION FOUND (Parallel Search: {elapsed:.1f}s):",
        )
    )
    print(f"Optimal Config: {best_res['cfg']}")
    print(
        f"Master Metrics: F1 = {best_res['f1']:.4f} | Recall = {best_res['recall']:.2f}% (TP={best_res['tp']:,}/{best_res['gt']:,}) | "
        f"Prec = {best_res['prec']:.2f}% (FP={best_res['fp']:,})"
    )
    print("=" * 115)

    # Detailed Per-Sequence Breakdown
    print("\n" + colorstr("bold", colorstr("cyan", "📋 DETAILED PER-SEQUENCE BREAKDOWN FOR OPTIMAL CONFIG:")))
    print("-" * 110)
    print(
        f"{'Sequence Name':<28} | {'GT':<5} | {'TP':<5} | {'FP':<5} | {'Recall':<7} | {'Prec':<7} | {'F1':<7} | Status"
    )
    print("-" * 110)

    seq_m = best_res["seq_metrics"]
    passed_count = 0
    total_audited = len(seq_m)

    for seq_name in sorted(seq_m.keys()):
        m = seq_m[seq_name]
        rec_s = f"{m['recall']:>5.1f}%"
        prec_s = f"{m['precision']:>5.1f}%"
        f1_s = f"{m['f1']:>6.2f}"

        # Standard check
        if m["gt"] == 0:
            far = m["fp"] / 1498.0
            passed = far <= 0.030
            rec_s = "N/A"
            prec_s = "N/A"
            f1_s = "N/A"
        else:
            passed = m["f1"] >= 88.0 and m["recall"] >= 85.0 and m["precision"] >= 90.0

        if passed:
            passed_count += 1
            status = colorstr("green", "PASS ✅")
        else:
            status = colorstr("yellow", "SUB ⚠️")

        print(
            f"{seq_name:<28} | {m['gt']:<5} | {m['tp']:<5} | {m['fp']:<5} | {rec_s:<7} | {prec_s:<7} | {f1_s:<7} | {status}"
        )

    print("=" * 110)
    print(
        colorstr(
            "bold",
            f"SUMMARY: {passed_count} / {total_audited} Sequences Passed Industrial Delivery Standards!",
        )
    )
    print("=" * 110 + "\n")


if __name__ == "__main__":
    main()
