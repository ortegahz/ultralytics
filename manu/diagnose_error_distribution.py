#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Objective Morphological & Kinematic Forensic Profiler for UAV Infrared Detection Errors.

Strict Principles:
1. NO subjective hand-drawn horizontal lines (NO y < 384 sky assumption).
2. Purely objective spatial & temporal kinematic indicators:
   - Trajectory Total Net Displacement: ||P_last - P_start||
   - Sub-pixel Trajectory Spatial Variance: Var(X) + Var(Y) along track lifetime
   - Tracklet Lifetime (Hits & Age)
   - Local Neighbor Clutter Density: Number of concurrent candidate pulses within R=25px
   - Proximity to Ground Truth: exact Euclidean distance to target centroid
3. Missed Target (FN) Sub-threshold Activation Spectrum:
   - 0.10 <= conf < 0.22 (Direct threshold cutoff)
   - 0.05 <= conf < 0.10 (Suppressed by loss/penalty)
   - 0.02 <= conf < 0.05 (Sub-visual faint impulse)
   - conf < 0.02 (True SCR < 1.0 physical dead zone)

Usage:
    python manu/diagnose_error_distribution.py \
        --cache-file runs/gmc_eval/uav_median_trial0474_cache.pkl \
        --dist-thresh 8.0 \
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
from typing import Dict, List, Tuple

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
    parser = argparse.ArgumentParser(description="Objective Morphological & Kinematic Error Diagnosis")
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


def run_objective_diagnostics(records: List[Dict], args):
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

    print("=" * 105)
    print("🔬 RUNNING OBJECTIVE FORENSIC ANALYSIS (NO SUBJECTIVE POSITION ASSUMPTIONS)...")
    print("=" * 105)

    fp_records = []
    fn_records = []
    tp_count = 0
    gt_total = 0

    for seq_name in sorted(seq_records.keys()):
        recs = seq_records[seq_name]
        res = evaluate_sequence_bidirectional(
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

        bidi = res["bidirectional"]
        tp_count += bidi["tp"]
        gt_total += bidi["gt"]
        bidi_frame_dets = res["bidi_frame_dets"]
        records_sorted = res["records_sorted"]

        # Track-level kinematic profile dictionary: track_id -> dict
        # We compute spatial trajectory variance and net displacement for every track
        track_kinematics = {}
        # Collect track points over time
        track_points = defaultdict(list)
        for f_idx, dets in enumerate(bidi_frame_dets):
            for d in dets:
                tid = d["track_id"]
                track_points[tid].append((f_idx, d["pos"]))

        for tid, pts_info in track_points.items():
            pts = np.array([p[1] for p in pts_info], dtype=np.float32)
            frames = [p[0] for p in pts_info]
            net_disp = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) > 1 else 0.0
            pos_var = float(np.var(pts[:, 0]) + np.var(pts[:, 1])) if len(pts) > 1 else 0.0
            lifetime = len(pts)
            duration_frames = max(frames) - min(frames) + 1 if frames else 1
            track_kinematics[tid] = {
                "net_disp": net_disp,
                "pos_var": pos_var,
                "lifetime": lifetime,
                "duration": duration_frames,
            }

        # Frame-by-frame objective matching
        for f_idx, r in enumerate(records_sorted):
            gt_pts = r["gt_pts"].astype(np.float32)
            bidi_dets = bidi_frame_dets[f_idx]
            raw_peaks = r["pred_points"].astype(np.float32)
            raw_scores = r["pred_scores"].astype(np.float32)

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

            # 1. Inspect False Positives
            for p_i, d in enumerate(bidi_dets):
                if not pred_matched[p_i]:
                    pos = d["pos"]
                    tid = d["track_id"]
                    t_info = track_kinematics.get(tid, {"net_disp": 0.0, "pos_var": 0.0, "lifetime": 1, "duration": 1})

                    # Calculate distance to nearest GT in this frame (if any)
                    min_d_gt = float(np.min(np.linalg.norm(gt_pts - pos, axis=1))) if len(gt_pts) > 0 else 999.0

                    # Calculate local neighbor candidate density (within R=25px) from raw peaks
                    local_density = 0
                    if len(raw_peaks) > 0:
                        d_raw = np.linalg.norm(raw_peaks - pos, axis=1)
                        local_density = int(np.sum((d_raw <= 25.0) & (raw_scores >= 0.05)))

                    # Boundary distance
                    min_edge_dist = float(min(pos[0], args.img_w - pos[0], pos[1], args.img_h - pos[1]))

                    fp_records.append({
                        "seq": seq_name,
                        "frame": f_idx,
                        "pos": pos,
                        "track_id": tid,
                        "score": float(d.get("score", 1.0)),
                        "net_disp": t_info["net_disp"],
                        "pos_var": t_info["pos_var"],
                        "lifetime": t_info["lifetime"],
                        "min_d_gt": min_d_gt,
                        "local_density": local_density,
                        "min_edge_dist": min_edge_dist,
                    })

            # 2. Inspect Missed GTs (FNs)
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
                        else:
                            closest_idx = int(np.argmin(d_to_raw))
                            best_d = float(d_to_raw[closest_idx])
                            best_sc = float(raw_scores[closest_idx])

                    fn_records.append({
                        "seq": seq_name,
                        "frame": f_idx,
                        "gt_pos": g_pos,
                        "best_subpeak_score": best_sc,
                        "best_subpeak_dist": best_d,
                    })

    print(f"[INFO] Analyzed {len(fp_records)} FPs and {len(fn_records)} FNs (Total GT: {gt_total}, TP: {tp_count}).\n")
    return fp_records, fn_records


def print_objective_report(fp_records: List[Dict], fn_records: List[Dict]):
    total_fps = len(fp_records)
    total_fns = len(fn_records)

    print("=" * 105)
    print(colorstr("bold", colorstr("magenta", f"📊 1. OBJECTIVE KINEMATIC & MORPHOLOGICAL FP FORENSICS (TOTAL: {total_fps:,})")))
    print("=" * 105)

    # 1.1 Trajectory Net Displacement Profile (Is it actually moving or completely static?)
    static_dead = sum(1 for fp in fp_records if fp["net_disp"] < 1.0)
    micro_vibrate = sum(1 for fp in fp_records if 1.0 <= fp["net_disp"] < 3.0)
    medium_move = sum(1 for fp in fp_records if 3.0 <= fp["net_disp"] < 10.0)
    high_flight = sum(1 for fp in fp_records if fp["net_disp"] >= 10.0)

    print("• Trajectory Net Displacement (||P_end - P_start|| across track):")
    print(f"  - Completely Rigid / Dead Static (< 1.0px)   : {static_dead:>5} ({static_dead / total_fps * 100:>5.1f}%)  <-- 🛑 Sensor Bad Pixels / Static Glints")
    print(f"  - Micro-Jitter (1.0px ~ 3.0px)               : {micro_vibrate:>5} ({micro_vibrate / total_fps * 100:>5.1f}%)  <-- 🛑 Background edge vibration / Subpixel jitter")
    print(f"  - Short Local Drift (3.0px ~ 10.0px)         : {medium_move:>5} ({medium_move / total_fps * 100:>5.1f}%)  <-- Cloud edge morph / Short ghost drift")
    print(f"  - True Continuous Flight (>= 10.0px)         : {high_flight:>5} ({high_flight / total_fps * 100:>5.1f}%)  <-- ✈️ True Kinematic Flight (Birds / Moving Targets / Unlabeled Drones!)")

    # 1.2 Spatial Variance Profile (Var(X) + Var(Y))
    var_zero = sum(1 for fp in fp_records if fp["pos_var"] < 0.25)
    var_subpixel = sum(1 for fp in fp_records if 0.25 <= fp["pos_var"] < 1.0)
    var_dynamic = sum(1 for fp in fp_records if fp["pos_var"] >= 1.0)
    print("\n• Sub-pixel Coordinate Variance Var(X) + Var(Y):")
    print(f"  - Absolute Frozen Grid Point (Var < 0.25px²) : {var_zero:>5} ({var_zero / total_fps * 100:>5.1f}%)  <-- Physically Frozen Sensor Dead Points")
    print(f"  - Subpixel Thermal Fluctuations (0.25 ~ 1.0) : {var_subpixel:>5} ({var_subpixel / total_fps * 100:>5.1f}%)")
    print(f"  - Kinematically Dynamic (Var >= 1.0px²)       : {var_dynamic:>5} ({var_dynamic / total_fps * 100:>5.1f}%)")

    # 1.3 Local Clutter Density (Number of candidate pulses within R=25px)
    dense_clutter = sum(1 for fp in fp_records if fp["local_density"] >= 3)
    sparse_isolated = sum(1 for fp in fp_records if fp["local_density"] <= 1)
    print("\n• Local Morphological Clutter Density (within R=25px neighborhood):")
    print(f"  - High-Density Clutter Patch (>= 3 pulses)   : {dense_clutter:>5} ({dense_clutter / total_fps * 100:>5.1f}%)  <-- Complex Textured Terrains / Cloud Clusters")
    print(f"  - Clean Isolated Impulse (<= 1 pulse)        : {sparse_isolated:>5} ({sparse_isolated / total_fps * 100:>5.1f}%)  <-- Discrete Point-like Impulses")

    # 1.4 Correlation with Ground Truth (Halo & Multipath)
    halo_fps = sum(1 for fp in fp_records if 8.0 < fp["min_d_gt"] <= 15.0)
    unlabeled_flyers = sum(1 for fp in fp_records if fp["min_d_gt"] > 20.0 and fp["net_disp"] >= 10.0 and fp["lifetime"] >= 10)
    print("\n• Ground Truth Proximity & Potential Unlabeled Targets:")
    print(f"  - Near-GT Halo / Label Jitter (8px < d <= 15px): {halo_fps:>5} ({halo_fps / total_fps * 100:>5.1f}%)  <-- Sub-pixel Halo / Annotator Label Inaccuracies")
    print(f"  - Long Persistent Flying Tracks (disp>=10, L>=10): {unlabeled_flyers:>5} ({unlabeled_flyers / total_fps * 100:>5.1f}%)  <-- 🎯 Likely Unlabeled Flying Drones / Birds!")

    # 1.5 Top Sequence Kinematic Profiles
    seq_kinematics = defaultdict(lambda: {"total": 0, "rigid_static": 0, "flying": 0, "halo": 0})
    for fp in fp_records:
        s = fp["seq"]
        seq_kinematics[s]["total"] += 1
        if fp["net_disp"] < 2.5:
            seq_kinematics[s]["rigid_static"] += 1
        if fp["net_disp"] >= 10.0:
            seq_kinematics[s]["flying"] += 1
        if 8.0 < fp["min_d_gt"] <= 15.0:
            seq_kinematics[s]["halo"] += 1

    print("\n• Kinematic Breakdown of Top FP Sequences:")
    print(f"  {'Sequence Name':<28} | {'Total FP':<8} | {'Rigid (<2.5px)':<14} | {'Flying (>=10px)':<15} | {'Near-GT Halo':<12} | {'Objective Nature'}")
    print("  " + "-" * 100)
    for s_name, stats in sorted(seq_kinematics.items(), key=lambda x: x[1]["total"], reverse=True)[:8]:
        if stats["rigid_static"] / max(1, stats["total"]) > 0.40:
            nat = "🛑 High Static / Dead Point Leakage"
        elif stats["flying"] / max(1, stats["total"]) > 0.40:
            nat = "✈️ Persistent Moving Tracklets"
        elif stats["halo"] / max(1, stats["total"]) > 0.30:
            nat = "🎯 GT Label Jitter / Halo"
        else:
            nat = "💨 Transient Moving Noise"
        print(f"  {s_name:<28} | {stats['total']:<8} | {stats['rigid_static']:<14} | {stats['flying']:<15} | {stats['halo']:<12} | {nat}")

    print("\n" + "=" * 105)
    print(colorstr("bold", colorstr("cyan", f"📉 2. OBJECTIVE MISSED TARGETS (FN) SPECTRUM (TOTAL: {total_fns:,})")))
    print("=" * 105)

    within_8px_active = [fn for fn in fn_records if fn["best_subpeak_dist"] <= 8.0 and fn["best_subpeak_score"] >= 0.02]
    tier_trunc = sum(1 for fn in within_8px_active if fn["best_subpeak_score"] >= 0.10)
    tier_suppressed = sum(1 for fn in within_8px_active if 0.05 <= fn["best_subpeak_score"] < 0.10)
    tier_faint = sum(1 for fn in within_8px_active if 0.02 <= fn["best_subpeak_score"] < 0.05)
    tier_dead = total_fns - len(within_8px_active)

    print("• Network Activation Strength at Target Centroid (Distance <= 8.0px):")
    print(f"  - Confident Sub-peak (0.10 <= conf < 0.22)   : {tier_trunc:>5} ({tier_trunc / total_fns * 100:>5.1f}%)  <-- 🎯 Direct Threshold Cutoff")
    print(f"  - Weak Sub-peak (0.05 <= conf < 0.10)        : {tier_suppressed:>5} ({tier_suppressed / total_fns * 100:>5.1f}%)  <-- 🎯 Loss Penalty Suppression")
    print(f"  - Barely Visible (0.02 <= conf < 0.05)       : {tier_faint:>5} ({tier_faint / total_fns * 100:>5.1f}%)  <-- Faint Energy Leaks")
    print(f"  - Zero Energy (conf < 0.02 or dist > 8.0px) : {tier_dead:>5} ({tier_dead / total_fns * 100:>5.1f}%)  <-- 🛑 Absolute Physical Dead Zone (SCR < 1.0)")

    print("=" * 105 + "\n")


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

    fp_records, fn_records = run_objective_diagnostics(records, args)
    print_objective_report(fp_records, fn_records)


if __name__ == "__main__":
    main()
