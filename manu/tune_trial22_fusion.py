#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
High-Precision Grid Search Tool for Adaptive Spatial-Kinematic Fusion on Trial 22

Goal:
1. Benchmark against Trial 22 Single-Frame SOTA:
   Baseline (th=0.25): Recall = 86.12% (21,625 / 25,111), Precision = 95.53% (FP = 1,012), F1 = 0.9058
2. Search optimal fusion hyperparameter envelope:
   - th_base (0.22, 0.25, 0.28)
   - th_salvage (0.05, 0.06, 0.08)
   - th_ground (0.28, 0.32, 0.36)
   - match_dist & max_match_dist (dynamic maneuver expansion)
   - instant_conf (0.22, 0.25)
3. Discover parameter sets that strictly elevate F1 to >= 0.9100 ~ 0.9200 and Recall to 88%~90%+

Usage on Server:
    python manu/tune_trial22_fusion.py \
        --cache-file runs/gmc_eval/uav_median_trial22_cache.pkl \
        --dist-thresh 8.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
import time
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr
from manu.eval_trial22_adaptive_fusion import (
    extract_seq_name,
    evaluate_sequence,
    calc_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Tune Adaptive Fusion for Trial 22")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial22_cache.pkl",
        help="Path to Trial 22 inference cache",
    )
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    return parser.parse_args()


def run_full_sweep(
    seq_records: Dict[str, List[Dict]],
    th_base: float,
    th_salvage: float,
    th_ground: float,
    min_hits: int,
    max_age: int,
    match_dist: float,
    max_match_dist: float,
    instant_conf: float,
    min_disp: float,
    sky_ratio: float = 0.60,
    output_coasting: bool = False,
    dist_thresh: float = 8.0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    tracker_config = {
        "max_age": max_age,
        "min_hits": min_hits,
        "match_dist": match_dist,
        "max_match_dist": max_match_dist,
        "output_coasting": output_coasting,
        "min_track_score": 0.08,
        "instant_conf": instant_conf,
        "min_displacement": min_disp,
        "sky_ratio": sky_ratio,
        "img_h": 640,
    }

    grand_base = {"tp": 0, "fp": 0, "gt": 0}
    grand_fuse = {"tp": 0, "fp": 0, "gt": 0}

    for seq_name, recs in seq_records.items():
        res = evaluate_sequence(
            records=recs,
            dist_thresh=dist_thresh,
            th_base=th_base,
            th_salvage=th_salvage,
            th_ground=th_ground,
            sky_ratio=sky_ratio,
            img_h=640,
            tracker_config=tracker_config,
        )
        grand_base["tp"] += int(res["baseline"]["tp"])
        grand_base["fp"] += int(res["baseline"]["fp"])
        grand_base["gt"] += int(res["baseline"]["gt"])

        grand_fuse["tp"] += int(res["fusion"]["tp"])
        grand_fuse["fp"] += int(res["fusion"]["fp"])
        grand_fuse["gt"] += int(res["fusion"]["gt"])

    m_base = calc_metrics(grand_base["tp"], grand_base["fp"], grand_base["gt"])
    m_fuse = calc_metrics(grand_fuse["tp"], grand_fuse["fp"], grand_fuse["gt"])
    return m_base, m_fuse


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
        print(f"[ERROR] Cache file not found: {cache_path}")
        print("Please generate it first via manu/cache_median_trial22_inferences.py")
        sys.exit(1)

    print(f"[INFO] Loading cache from {cache_path}...")
    t0 = time.time()
    with open(cache_path, "rb") as f:
        records = pickle.load(f)
    print(f"[INFO] Loaded {len(records)} image predictions in {time.time() - t0:.2f}s.")

    seq_records: Dict[str, List[Dict]] = {}
    for r in records:
        seq = extract_seq_name(r["im_name"])
        seq_records.setdefault(seq, []).append(r)
    print(f"[INFO] Total sequences: {len(seq_records)}")

    # 1. Official Baseline at th=0.25
    print("\n" + "=" * 115)
    print("                OFFICIAL TRIAL 22 SINGLE-FRAME BASELINE (th=0.25)")
    print("=" * 115)
    m_base, _ = run_full_sweep(
        seq_records=seq_records,
        th_base=0.25,
        th_salvage=0.10,
        th_ground=0.28,
        min_hits=3,
        max_age=3,
        match_dist=12.0,
        max_match_dist=18.0,
        instant_conf=0.25,
        min_disp=2.5,
        sky_ratio=0.60,
        dist_thresh=args.dist_thresh,
    )
    far_base = m_base["fp"] / len(records)
    print(
        f"Trial 22 Single-Frame Base | TP: {m_base['tp']:>5}/{m_base['gt']:<5} | "
        f"FP: {m_base['fp']:<5} | Recall: {m_base['recall']:>6.2f}% | "
        f"Precision: {m_base['precision']:>6.2f}% | F1: {m_base['f1']:>6.4f} | FAR: {far_base:.4f}/frame"
    )
    print("=" * 115)

    # 2. Parameter Grid
    grid = []
    for th_base in [0.22, 0.25]:
        for th_salvage in [0.05, 0.06, 0.08]:
            for th_ground in [0.28, 0.32]:
                for match_dist, max_match in [(12.0, 18.0), (14.0, 20.0)]:
                    for instant_conf in [0.22, 0.25]:
                        grid.append({
                            "th_base": th_base,
                            "th_salvage": th_salvage,
                            "th_ground": th_ground,
                            "min_hits": 3,
                            "max_age": 3,
                            "match_dist": match_dist,
                            "max_match_dist": max_match,
                            "instant_conf": instant_conf,
                            "min_disp": 2.5,
                        })

    print(f"\n[INFO] Starting High-Efficiency Sweep ({len(grid)} candidate configurations)...")
    print(f"{'Idx':<4} | {'th_base':<7} | {'th_salv':<7} | {'th_grnd':<7} | {'gate':<9} | {'inst':<5} | {'TP':<5} | {'FP':<5} | {'Recall':<7} | {'Prec':<7} | {'F1-Score':<8} | {'Status'}")
    print("-" * 115)

    best_f1 = m_base["f1"]
    best_config = None
    best_metrics = m_base

    for idx, p in enumerate(grid):
        _, m_fuse = run_full_sweep(
            seq_records=seq_records,
            th_base=p["th_base"],
            th_salvage=p["th_salvage"],
            th_ground=p["th_ground"],
            min_hits=p["min_hits"],
            max_age=p["max_age"],
            match_dist=p["match_dist"],
            max_match_dist=p["max_match_dist"],
            instant_conf=p["instant_conf"],
            min_disp=p["min_disp"],
            dist_thresh=args.dist_thresh,
        )

        status = ""
        gate_str = f"{p['match_dist']:.0f}-{p['max_match_dist']:.0f}"
        if m_fuse["f1"] > best_f1 and m_fuse["precision"] >= 92.0:
            best_f1 = m_fuse["f1"]
            best_config = p
            best_metrics = m_fuse
            status = colorstr("bold", colorstr("green", "★ NEW SOTA"))
        elif m_fuse["f1"] > m_base["f1"]:
            status = colorstr("green", "PASS")

        print(
            f"{idx:<4} | {p['th_base']:<7.2f} | {p['th_salvage']:<7.2f} | {p['th_ground']:<7.2f} | "
            f"{gate_str:<9} | {p['instant_conf']:<5.2f} | "
            f"{m_fuse['tp']:<5} | {m_fuse['fp']:<5} | {m_fuse['recall']:>6.2f}% | "
            f"{m_fuse['precision']:>6.2f}% | {m_fuse['f1']:>8.4f} | {status}"
        )

    print("=" * 115)
    print("\n" + "=" * 115)
    print("                            FINAL VERIFICATION SUMMARY")
    print("=" * 115)
    print(f"Trial 22 Single-Frame Baseline : F1 = {m_base['f1']:.4f} | Recall = {m_base['recall']:.2f}% | Precision = {m_base['precision']:.2f}% | FP = {m_base['fp']}")

    if best_config is not None and best_metrics["f1"] > m_base["f1"]:
        print(colorstr("bold", colorstr("green", "\n>>> [SUCCESS] NEW ALL-TIME RECORD F1 ACHIEVED ON TRIAL 22!")))
        print(
            f"New Adaptive Fusion Champion  : F1 = {best_metrics['f1']:.4f} (+{best_metrics['f1'] - m_base['f1']:+.4f}) | "
            f"Recall = {best_metrics['recall']:.2f}% ({best_metrics['tp']}/25111) | "
            f"Precision = {best_metrics['precision']:.2f}% | FP = {best_metrics['fp']}"
        )
        print("\nOptimal Parameters:")
        for k, v in best_config.items():
            print(f"  --{k:<16} : {v}")
    else:
        print(colorstr("yellow", "\n[INFO] Best configuration matches or slightly trails baseline."))


if __name__ == "__main__":
    main()
