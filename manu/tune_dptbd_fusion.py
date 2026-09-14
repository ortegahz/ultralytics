#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
High-Throughput Offline Grid Optimizer for DP-TBD Spatio-Temporal Parameters.

Sweeps DP-TBD parameters:
- max_vel: [14.0, 18.0, 22.0]
- gamma: [0.75, 0.82, 0.88]
- boost_scale: [0.18, 0.25, 0.32]
- th_base: [0.20, 0.22, 0.24]
- th_salvage: [0.05, 0.06, 0.07]
- th_ground: [0.32, 0.35, 0.38]

Runs directly against cached candidate pickle (~1-2 seconds per config).
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
from manu.eval_dptbd_track_fusion import (
    DynamicProgrammingTBD,
    evaluate_sequence_dptbd,
    extract_seq_name,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep DP-TBD Fusion Parameters")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial0474_cache.pkl",
        help="Path to Trial 0474 pickle cache",
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

    print(colorstr("bold", colorstr("green", f"\n>>> Loading cache from: {cache_path}")))
    with open(cache_path, "rb") as f:
        records = pickle.load(f)
    print(f"Loaded {len(records)} predictions.\n")

    seq_records: Dict[str, List[Dict]] = {}
    for r in records:
        seq = extract_seq_name(r["im_name"])
        seq_records.setdefault(seq, []).append(r)

    total_gt = sum(len(r["gt_pts"]) for r in records)
    all_seq_names = sorted(seq_records.keys())

    # Parameter space for rapid exploration
    boost_scale_list = [0.18, 0.25, 0.32]
    gamma_list = [0.75, 0.82]
    max_vel_list = [16.0, 20.0]
    th_base_list = [0.22, 0.24]
    th_ground_list = [0.35, 0.38]
    th_salvage = 0.06
    min_hits_infill = 5

    total_cfgs = len(boost_scale_list) * len(gamma_list) * len(max_vel_list) * len(th_base_list) * len(th_ground_list)

    print("=" * 110)
    print(f"🚀 Sweeping {total_cfgs} DP-TBD parameter configurations...")
    print(f"Current System SOTA to Beat: F1 = 0.9172 | Recall = 90.06% (TP: 22,616) | Prec = 93.44% (FP: 1,589)")
    print("=" * 110)

    results = []
    t0 = time.time()
    cfg_idx = 0

    # Cache pre-computed DP-boosted records per (boost_scale, gamma, max_vel)
    dp_cache: Dict[Tuple, Dict[str, List[Dict]]] = {}

    for boost_scale in boost_scale_list:
        for gamma in gamma_list:
            for max_vel in max_vel_list:
                dp_key = (boost_scale, gamma, max_vel)
                dp_tbd = DynamicProgrammingTBD(
                    max_velocity=max_vel,
                    gamma=gamma,
                    penalty_dist=0.08,
                    penalty_turn=0.06,
                    boost_scale=boost_scale,
                    min_cand_score=0.03,
                    sky_ratio=0.60,
                    img_h=640,
                )

                # Precompute DP transformation across all sequences once
                processed_seqs = {}
                for s_name in all_seq_names:
                    recs = seq_records[s_name]
                    processed_seqs[s_name] = dp_tbd.run_sequence(recs)
                dp_cache[dp_key] = processed_seqs

                for th_base in th_base_list:
                    for th_ground in th_ground_list:
                        cfg_idx += 1
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
                            "min_hits_for_infill": min_hits_infill,
                            "max_infill_gap": 3,
                            "min_track_hits": 3,
                            "min_track_score": 0.08,
                            "instant_conf": 0.25,
                        }

                        total_tp = 0
                        total_fp = 0

                        for s_name in all_seq_names:
                            proc_recs = dp_cache[dp_key][s_name]
                            res = evaluate_sequence_dptbd(
                                records=proc_recs,
                                dp_tbd=None,  # Already precomputed
                                dist_thresh=args.dist_thresh,
                                th_base=th_base,
                                th_salvage=th_salvage,
                                th_ground=th_ground,
                                sky_ratio=0.60,
                                img_h=640,
                                tracker_config=tracker_cfg,
                                smoother_config=smoother_cfg,
                            )
                            total_tp += res["dptbd_bidirectional"]["tp"]
                            total_fp += res["dptbd_bidirectional"]["fp"]

                        rec = (total_tp / total_gt) * 100.0
                        prec = (total_tp / max(1, total_tp + total_fp)) * 100.0
                        f1 = (2 * rec * prec / max(1e-6, rec + prec)) / 100.0
                        far = total_fp / len(records)

                        rec_data = {
                            "boost_scale": boost_scale,
                            "gamma": gamma,
                            "max_vel": max_vel,
                            "th_base": th_base,
                            "th_ground": th_ground,
                            "tp": total_tp,
                            "fp": total_fp,
                            "recall": rec,
                            "precision": prec,
                            "f1": f1,
                            "far": far,
                        }
                        results.append(rec_data)

                        if f1 >= 0.9160:
                            print(
                                f"[{cfg_idx:02d}/{total_cfgs:02d}] boost={boost_scale:.2f}, gamma={gamma:.2f}, vel={max_vel:.1f}, "
                                f"base={th_base:.2f}, grnd={th_ground:.2f} | "
                                f"F1: {f1:.4f} | Recall: {rec:.2f}% (TP: {total_tp}) | Prec: {prec:.2f}% (FP: {total_fp})"
                            )

    dur = time.time() - t0
    print("=" * 110)
    print(f"Optimization finished in {dur:.1f}s.")

    best_overall = max(results, key=lambda c: c["f1"])
    print("\n" + colorstr("bold", colorstr("magenta", "★ TOP F1 DP-TBD CONFIGURATION:")))
    print(
        f"   F1-Score  : {best_overall['f1']:.4f}\n"
        f"   Recall    : {best_overall['recall']:.2f}% (TP: {best_overall['tp']} / {total_gt})\n"
        f"   Precision : {best_overall['precision']:.2f}% (FP: {best_overall['fp']})\n"
        f"   FAR       : {best_overall['far']:.4f} fps\n"
        f"   Params    : boost_scale={best_overall['boost_scale']}, gamma={best_overall['gamma']}, max_vel={best_overall['max_vel']}, "
        f"th_base={best_overall['th_base']}, th_ground={best_overall['th_ground']}"
    )
    print("=" * 110 + "\n")


if __name__ == "__main__":
    main()
