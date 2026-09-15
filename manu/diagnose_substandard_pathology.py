#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Targeted Pathological Slice Diagnostics for Top 10 Substandard UAV Sequences.

Diagnostic Dimensions:
1. Missed Targets (FNs) Activation Spectrum (Distance <= 8.0px to Ground Truth):
   - Tier A [0.10 <= score < 0.22]: Confident sub-peak, directly cut off by threshold.
   - Tier B [0.05 <= score < 0.10]: Weak pulse, salvageable by track guidance.
   - Tier C [0.02 <= score < 0.05]: Sub-visual faint impulse.
   - Tier D [score < 0.02 or dist > 8px]: True physical dead zone / sensor blind.
2. False Alarms (FPs) Kinematic Profiling:
   - Rigid Static Jitter (||P_end - P_start|| < 2.0px & Var < 0.5px²): Glints / Bad pixels.
   - Ground Clutter / Transient Drift (2.0px <= Net Disp < 8.0px): Moving foliage/edges.
   - Persistent Moving Tracklet (Net Disp >= 8.0px & Lifetime >= 8): Unlabeled targets / birds.
   - Near-GT Halo (8.0px < Dist_GT <= 15.0px): Annotator box jitter.

Usage:
    python manu/diagnose_substandard_pathology.py \
        --cache-file runs/gmc_eval/uav_median_trial0474_cache.pkl \
        --th-base 0.22 \
        --th-salvage 0.06 \
        --th-ground 0.35 \
        --min-hits-infill 5
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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
)

# Explicit list of the 10 substandard sequences
TARGET_SEQUENCES = [
    "wg2022_ir_020_split_03",
    "DJI_0051_2",
    "wg2022_ir_011_split_03",
    "DJI_0175_2",
    "02_6321_0274-2773",
    "wg2022_ir_020_split_07",
    "01_4485_1167-2666",
    "wg2022_ir_012_split_08",
    "3700000000002_153918_1",
    "wg2022_ir_020_split_05",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Targeted Pathological Slice Diagnostics")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial0474_cache.pkl",
        help="Path to Trial 0474 pickle cache",
    )
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    parser.add_argument("--th-base", type=float, default=0.22)
    parser.add_argument("--th-salvage", type=float, default=0.06)
    parser.add_argument("--th-ground", type=float, default=0.35)
    parser.add_argument("--min-hits-infill", type=int, default=5)
    parser.add_argument("--img-h", type=int, default=640)
    parser.add_argument("--img-w", type=int, default=640)
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
        print(colorstr("red", f"[ERROR] Cache file not found: {args.cache_file}"))
        sys.exit(1)

    with open(cache_path, "rb") as f:
        records = pickle.load(f)

    seq_records: Dict[str, List[Dict]] = {}
    for r in records:
        seq = extract_seq_name(r["im_name"])
        seq_records.setdefault(seq, []).append(r)

    tracker_config = {
        "max_age": 3,
        "min_hits": 3,
        "match_dist": 12.0,
        "max_match_dist": 18.0,
        "min_track_score": 0.08,
        "instant_conf": 0.25,
        "min_displacement": 2.5,
        "sky_ratio": 0.60,
        "img_h": args.img_h,
    }

    smoother_config = {
        "stitch_max_gap": 4,
        "stitch_max_dist": 25.0,
        "min_hits_for_infill": args.min_hits_infill,
        "max_infill_gap": 3,
        "min_track_hits": 3,
        "min_track_score": 0.08,
        "instant_conf": 0.25,
    }

    print("\n" + "=" * 130)
    print("🔬 TARGETED PATHOLOGICAL SLICE DIAGNOSTICS FOR 10 SUBSTANDARD SEQUENCES")
    print(f"Tolerance: {args.dist_thresh}px | th_base={args.th_base} | th_salvage={args.th_salvage} | th_ground={args.th_ground}")
    print("=" * 130)

    # Global accumulation for substandard subset
    sub_fn_tiers = {"tier_a": 0, "tier_b": 0, "tier_c": 0, "tier_dead": 0, "total_fn": 0}
    sub_fp_types = {"static_dead": 0, "clutter_drift": 0, "persistent_flyer": 0, "halo": 0, "total_fp": 0}

    seq_diagnostics = []

    for seq_name in TARGET_SEQUENCES:
        if seq_name not in seq_records:
            continue
        recs = seq_records[seq_name]
        eval_res = evaluate_sequence_bidirectional(
            records=recs,
            dist_thresh=args.dist_thresh,
            th_base=args.th_base,
            th_salvage=args.th_salvage,
            th_ground=args.th_ground,
            sky_ratio=0.60,
            img_h=args.img_h,
            tracker_config=tracker_config,
            smoother_config=smoother_config,
        )

        bidi = eval_res["bidirectional"]
        tp = bidi["tp"]
        fp = bidi["fp"]
        gt = bidi["gt"]
        fn = gt - tp
        bidi_frame_dets = eval_res["bidi_frame_dets"]
        records_sorted = eval_res["records_sorted"]

        # Track-level kinematic profile dictionary: track_id -> dict
        track_points = defaultdict(list)
        for f_idx, dets in enumerate(bidi_frame_dets):
            for d in dets:
                tid = d["track_id"]
                track_points[tid].append((f_idx, d["pos"]))

        track_kinematics = {}
        for tid, pts_info in track_points.items():
            pts = np.array([p[1] for p in pts_info], dtype=np.float32)
            frames = [p[0] for p in pts_info]
            net_disp = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) > 1 else 0.0
            pos_var = float(np.var(pts[:, 0]) + np.var(pts[:, 1])) if len(pts) > 1 else 0.0
            lifetime = len(pts)
            track_kinematics[tid] = {
                "net_disp": net_disp,
                "pos_var": pos_var,
                "lifetime": lifetime,
            }

        # Sequence-level breakdown
        seq_fn = {"tier_a": 0, "tier_b": 0, "tier_c": 0, "tier_dead": 0, "total": fn}
        seq_fp = {"static_dead": 0, "clutter_drift": 0, "persistent_flyer": 0, "halo": 0, "total": fp}

        for f_idx, r in enumerate(records_sorted):
            gt_pts = np.asarray(r["gt_pts"], dtype=np.float32)
            bidi_dets = bidi_frame_dets[f_idx]
            raw_peaks = np.asarray(r["pred_points"], dtype=np.float32)
            raw_scores = np.asarray(r["pred_scores"], dtype=np.float32)

            pred_pts = np.array([d["pos"] for d in bidi_dets], dtype=np.float32) if len(bidi_dets) > 0 else np.zeros((0, 2), dtype=np.float32)
            pred_matched = [False] * len(pred_pts)
            gt_matched = [False] * len(gt_pts)

            if len(pred_pts) > 0 and len(gt_pts) > 0:
                dists = np.linalg.norm(pred_pts[:, None, :] - gt_pts[None, :, :], axis=2)
                for _ in range(min(len(pred_pts), len(gt_pts))):
                    min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                    p_i, g_j = min_idx[0], min_idx[1]
                    d_val = dists[p_i, g_j]
                    if d_val <= args.dist_thresh:
                        pred_matched[p_i] = True
                        gt_matched[g_j] = True
                        dists[p_i, :] = 1e9
                        dists[:, g_j] = 1e9
                    else:
                        break

            # Analyze False Positives
            for p_i, d in enumerate(bidi_dets):
                if not pred_matched[p_i]:
                    pos = d["pos"]
                    tid = d["track_id"]
                    t_info = track_kinematics.get(tid, {"net_disp": 0.0, "pos_var": 0.0, "lifetime": 1})
                    min_d_gt = float(np.min(np.linalg.norm(gt_pts - pos, axis=1))) if len(gt_pts) > 0 else 999.0

                    if 8.0 < min_d_gt <= 15.0:
                        seq_fp["halo"] += 1
                    elif t_info["net_disp"] < 2.0 and t_info["pos_var"] < 0.5:
                        seq_fp["static_dead"] += 1
                    elif t_info["net_disp"] >= 8.0 and t_info["lifetime"] >= 8:
                        seq_fp["persistent_flyer"] += 1
                    else:
                        seq_fp["clutter_drift"] += 1

            # Analyze False Negatives (Missed GTs)
            for g_j, g_pos in enumerate(gt_pts):
                if not gt_matched[g_j]:
                    best_sc = 0.0
                    best_d = 999.0
                    if len(raw_peaks) > 0:
                        d_to_raw = np.linalg.norm(raw_peaks - g_pos, axis=1)
                        within_8px = d_to_raw <= args.dist_thresh
                        if np.any(within_8px):
                            idx_within = np.where(within_8px)[0]
                            max_sc_idx = idx_within[np.argmax(raw_scores[idx_within])]
                            best_sc = float(raw_scores[max_sc_idx])
                            best_d = float(d_to_raw[max_sc_idx])

                    if best_sc >= 0.10:
                        seq_fn["tier_a"] += 1
                    elif best_sc >= 0.05:
                        seq_fn["tier_b"] += 1
                    elif best_sc >= 0.02:
                        seq_fn["tier_c"] += 1
                    else:
                        seq_fn["tier_dead"] += 1

        # Accumulate to global
        for k in ["tier_a", "tier_b", "tier_c", "tier_dead"]:
            sub_fn_tiers[k] += seq_fn[k]
        sub_fn_tiers["total_fn"] += fn

        for k in ["static_dead", "clutter_drift", "persistent_flyer", "halo"]:
            sub_fp_types[k] += seq_fp[k]
        sub_fp_types["total_fp"] += fp

        seq_diagnostics.append({
            "seq": seq_name,
            "gt": gt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "rec": bidi["recall"],
            "prec": bidi["precision"],
            "f1": bidi["f1"],
            "fn_diag": seq_fn,
            "fp_diag": seq_fp,
        })

    # ==============================================================================
    # Report Section 1: FN Spectrum
    # ==============================================================================
    print("\n" + colorstr("bold", colorstr("cyan", "📉 1. MISSED TARGET (FN) ACTIVATION SPECTRUM BREAKDOWN")))
    print("=" * 130)
    print(
        f"{'Sequence Name':<26} | {'Total FN':<8} | {'Tier A (0.10~0.22)':<19} | {'Tier B (0.05~0.10)':<19} | "
        f"{'Tier C (0.02~0.05)':<19} | {'Tier D (Dead Zone)':<18} | {'Recoverable Ratio'}"
    )
    print("-" * 130)

    for s in seq_diagnostics:
        fn_d = s["fn_diag"]
        t_fn = max(1, s["fn"])
        if s["fn"] == 0:
            print(f"{s['seq']:<26} | {0:<8} | {'-':<19} | {'-':<19} | {'-':<19} | {'-':<18} | 0.0%")
            continue

        a_str = f"{fn_d['tier_a']} ({fn_d['tier_a']/t_fn*100:4.1f}%)"
        b_str = f"{fn_d['tier_b']} ({fn_d['tier_b']/t_fn*100:4.1f}%)"
        c_str = f"{fn_d['tier_c']} ({fn_d['tier_c']/t_fn*100:4.1f}%)"
        d_str = f"{fn_d['tier_dead']} ({fn_d['tier_dead']/t_fn*100:4.1f}%)"
        recov_pct = (fn_d["tier_a"] + fn_d["tier_b"]) / t_fn * 100.0

        recov_color = "green" if recov_pct >= 50.0 else ("yellow" if recov_pct >= 25.0 else "red")
        line = (
            f"{s['seq']:<26} | {s['fn']:<8} | {a_str:<19} | {b_str:<19} | {c_str:<19} | {d_str:<18} | "
            f"{colorstr(recov_color, f'{recov_pct:>5.1f}% (Tier A+B)')}"
        )
        print(line)

    tot_fn = max(1, sub_fn_tiers["total_fn"])
    tot_recov = sub_fn_tiers["tier_a"] + sub_fn_tiers["tier_b"]
    print("-" * 130)
    print(
        colorstr(
            "bold",
            f"{'SUBSTANDARD TOTAL':<26} | {tot_fn:<8} | "
            f"{sub_fn_tiers['tier_a']} ({sub_fn_tiers['tier_a']/tot_fn*100:4.1f}%)       | "
            f"{sub_fn_tiers['tier_b']} ({sub_fn_tiers['tier_b']/tot_fn*100:4.1f}%)       | "
            f"{sub_fn_tiers['tier_c']} ({sub_fn_tiers['tier_c']/tot_fn*100:4.1f}%)       | "
            f"{sub_fn_tiers['tier_dead']} ({sub_fn_tiers['tier_dead']/tot_fn*100:4.1f}%)      | "
            f"{tot_recov / tot_fn * 100:5.1f}% (RECOVERABLE: {tot_recov:,} FNs!)",
        )
    )

    # ==============================================================================
    # Report Section 2: FP Kinematic Breakdown
    # ==============================================================================
    print("\n" + colorstr("bold", colorstr("magenta", "📊 2. FALSE ALARM (FP) KINEMATIC & NATURE BREAKDOWN")))
    print("=" * 130)
    print(
        f"{'Sequence Name':<26} | {'Total FP':<8} | {'Rigid (<2px, static)':<20} | {'Clutter Drift':<15} | "
        f"{'Persistent Flyer':<18} | {'Near-GT Halo':<14} | {'Actionable Strategy'}"
    )
    print("-" * 130)

    for s in seq_diagnostics:
        fp_d = s["fp_diag"]
        t_fp = max(1, s["fp"])
        if s["fp"] == 0:
            print(f"{s['seq']:<26} | {0:<8} | {'-':<20} | {'-':<15} | {'-':<18} | {'-':<14} | -")
            continue

        r_str = f"{fp_d['static_dead']} ({fp_d['static_dead']/t_fp*100:4.1f}%)"
        c_str = f"{fp_d['clutter_drift']} ({fp_d['clutter_drift']/t_fp*100:4.1f}%)"
        f_str = f"{fp_d['persistent_flyer']} ({fp_d['persistent_flyer']/t_fp*100:4.1f}%)"
        h_str = f"{fp_d['halo']} ({fp_d['halo']/t_fp*100:4.1f}%)"

        if fp_d["static_dead"] / t_fp > 0.40:
            strategy = "✂️ Rigid Static Pruner"
        elif fp_d["halo"] / t_fp > 0.30:
            strategy = "🎯 Label Jitter Inhibit"
        elif fp_d["persistent_flyer"] / t_fp > 0.40:
            strategy = "✈️ Flying Track / Unlabeled"
        else:
            strategy = "💨 Spatial Variance Filter"

        line = f"{s['seq']:<26} | {s['fp']:<8} | {r_str:<20} | {c_str:<15} | {f_str:<18} | {h_str:<14} | {strategy}"
        print(line)

    tot_fp = max(1, sub_fp_types["total_fp"])
    print("-" * 130)
    print(
        colorstr(
            "bold",
            f"{'SUBSTANDARD TOTAL':<26} | {tot_fp:<8} | "
            f"{sub_fp_types['static_dead']} ({sub_fp_types['static_dead']/tot_fp*100:4.1f}%)      | "
            f"{sub_fp_types['clutter_drift']} ({sub_fp_types['clutter_drift']/tot_fp*100:4.1f}%)     | "
            f"{sub_fp_types['persistent_flyer']} ({sub_fp_types['persistent_flyer']/tot_fp*100:4.1f}%)      | "
            f"{sub_fp_types['halo']} ({sub_fp_types['halo']/tot_fp*100:4.1f}%)    | "
            f"STATIC+DRIFT PRUNABLE: {sub_fp_types['static_dead'] + sub_fp_types['clutter_drift']:,} FPs",
        )
    )
    print("=" * 130 + "\n")


if __name__ == "__main__":
    main()
