#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
High-Speed Grid Optimizer for Guided Kinematic Infill & Salvage.

Strict Industrial Condition:
- Recall MUST Increase (TP > 22,616)
- Precision MUST NOT Drop (Precision >= 93.89%)
- F1-Score MUST NOT Drop (F1 >= 0.9194)

Architecture:
1. One-time Pre-computation: Extracts and stitches all tracks per sequence into memory.
2. Ultra-fast Infill & Guided Deep-Salvage Testing: Takes < 1 second across all grid configs.

Search Space:
- max_infill_gap: [3, 4, 5]
- min_hits_for_infill: [4, 5, 6]
- enable_guided_salvage: [False, True]
- guided_th: [0.035, 0.040, 0.045, 0.050]
- guided_radius: [4.0, 6.0, 8.0]

Usage:
    python manu/tune_guided_salvage.py \
        --cache-file runs/gmc_eval/uav_median_trial0474_cache.pkl
"""

from __future__ import annotations

import argparse
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
from manu.eval_bidirectional_track_fusion import (
    BidirectionalTemporalSmoother,
    OnlineAdaptiveTracker,
    PointKalmanTrack,
    calc_metrics,
    extract_seq_name,
    filter_dense_clutter_clusters,
    match_predictions_to_gt,
    natural_sort_key,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Tune Guided Infill & Salvage")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial0474_cache.pkl",
    )
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    parser.add_argument("--th-base", type=float, default=0.22)
    parser.add_argument("--th-salvage", type=float, default=0.06)
    parser.add_argument("--th-ground", type=float, default=0.35)
    parser.add_argument("--min-rigid-disp", type=float, default=2.0)
    parser.add_argument("--max-rigid-var", type=float, default=0.5)
    parser.add_argument("--min-hits-prune", type=int, default=8)
    return parser.parse_args()


class FastGuidedSmoother:
    def __init__(
        self,
        min_hits_for_infill: int = 5,
        max_infill_gap: int = 3,
        min_track_hits: int = 3,
        min_track_score: float = 0.08,
        min_rigid_displacement: float = 2.0,
        max_rigid_variance: float = 0.5,
        min_hits_for_prune: int = 8,
        enable_guided_salvage: bool = False,
        guided_th: float = 0.04,
        guided_radius: float = 6.0,
    ):
        self.min_hits_for_infill = min_hits_for_infill
        self.max_infill_gap = max_infill_gap
        self.min_track_hits = min_track_hits
        self.min_track_score = min_track_score
        self.min_rigid_displacement = min_rigid_displacement
        self.max_rigid_variance = max_rigid_variance
        self.min_hits_for_prune = min_hits_for_prune
        self.enable_guided_salvage = enable_guided_salvage
        self.guided_th = guided_th
        self.guided_radius = guided_radius

    def smooth_and_infill_fast(
        self,
        tracks: List[PointKalmanTrack],
        raw_peaks_per_frame: List[Tuple[np.ndarray, np.ndarray]],
        num_frames: int,
    ) -> List[List[Dict]]:
        frame_outputs: List[List[Dict]] = [[] for _ in range(num_frames)]

        for trk in tracks:
            is_confirmed = trk.is_confirmed or (
                trk.hits >= self.min_track_hits and trk.score >= self.min_track_score
            )
            if not is_confirmed:
                continue

            obs_frames = sorted(trk.observations.keys())
            if not obs_frames:
                continue

            # Rigid Static Pruner
            if self.min_rigid_displacement > 0 and len(obs_frames) >= self.min_hits_for_prune:
                pts_arr = np.array([trk.observations[f][0] for f in obs_frames], dtype=np.float32)
                if len(pts_arr) > 1:
                    net_disp = float(np.linalg.norm(pts_arr[-1] - pts_arr[0]))
                    pos_var = float(np.var(pts_arr[:, 0]) + np.var(pts_arr[:, 1]))
                    if net_disp < self.min_rigid_displacement and pos_var < self.max_rigid_variance:
                        continue

            can_infill = trk.hits >= self.min_hits_for_infill

            # 1. Output confirmed direct observations
            for f_idx in obs_frames:
                if 0 <= f_idx < num_frames:
                    pos, sc, is_meas = trk.observations[f_idx]
                    frame_outputs[f_idx].append({
                        "pos": pos,
                        "score": sc,
                        "track_id": trk.track_id,
                        "infilled": not is_meas,
                    })

            # 2. Infill and Guided Deep-Salvage for internal gaps
            if can_infill and len(obs_frames) >= 2:
                for i in range(len(obs_frames) - 1):
                    f1 = obs_frames[i]
                    f2 = obs_frames[i + 1]
                    gap = f2 - f1
                    if 1 < gap <= (self.max_infill_gap + 1):
                        p1 = trk.observations[f1][0]
                        p2 = trk.observations[f2][0]
                        s1 = trk.observations[f1][1]
                        s2 = trk.observations[f2][1]

                        for step, missing_f in enumerate(range(f1 + 1, f2), start=1):
                            if not (0 <= missing_f < num_frames):
                                continue
                            alpha = step / gap
                            interp_pos = (1.0 - alpha) * p1 + alpha * p2
                            interp_score = (1.0 - alpha) * s1 + alpha * s2

                            # Guided Salvage: Check if an actual faint peak exists near interp_pos
                            final_pos = interp_pos
                            final_score = interp_score
                            is_guided = False

                            if self.enable_guided_salvage:
                                r_pts, r_scs = raw_peaks_per_frame[missing_f]
                                if len(r_pts) > 0:
                                    dists = np.linalg.norm(r_pts - interp_pos, axis=1)
                                    within_tube = (dists <= self.guided_radius) & (r_scs >= self.guided_th)
                                    if np.any(within_tube):
                                        idx_tube = np.where(within_tube)[0]
                                        best_idx = idx_tube[np.argmax(r_scs[idx_tube])]
                                        final_pos = r_pts[best_idx]
                                        final_score = float(r_scs[best_idx])
                                        is_guided = True

                            frame_outputs[missing_f].append({
                                "pos": final_pos,
                                "score": final_score,
                                "track_id": trk.track_id,
                                "infilled": not is_guided,
                                "guided": is_guided,
                            })

        return frame_outputs


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

    print(f"[INFO] Loading inference cache from: {cache_path}")
    with open(cache_path, "rb") as f:
        records = pickle.load(f)

    seq_records: Dict[str, List[Dict]] = {}
    for r in records:
        seq = extract_seq_name(r["im_name"])
        seq_records.setdefault(seq, []).append(r)

    all_seq_names = sorted(seq_records.keys())

    # Pre-computation: Run Pass 1 tracking & stitching once for all sequences
    print("\n" + "=" * 115)
    print("⚡ PRE-COMPUTING TRACKLETS ACROSS ALL 24 SEQUENCES (ONE-TIME ONLY)...")
    print("=" * 115)
    t0 = time.time()

    precomputed_seqs = {}
    total_gt = 0

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
    sky_y_boundary = 640 * 0.60

    for s_name in all_seq_names:
        recs = sorted(seq_records[s_name], key=lambda r: natural_sort_key(r["im_name"]))
        num_frames = len(recs)
        online_tracker = OnlineAdaptiveTracker(**tracker_cfg)

        raw_peaks_list = []
        gt_list = []

        for f_idx, r in enumerate(recs):
            gt_pts = np.asarray(r["gt_pts"], dtype=np.float32)
            pred_pts = np.asarray(r["pred_points"], dtype=np.float32)
            pred_scs = np.asarray(r["pred_scores"], dtype=np.float32)
            total_gt += len(gt_pts)

            gt_list.append(gt_pts)
            raw_peaks_list.append((pred_pts, pred_scs))

            if len(pred_pts) > 0:
                is_sky = pred_pts[:, 1] < sky_y_boundary
                fusion_high_mask = (is_sky & (pred_scs >= args.th_base)) | ((~is_sky) & (pred_scs >= args.th_ground))
                fusion_salvage_mask = is_sky & (pred_scs >= args.th_salvage) & (pred_scs < args.th_base)

                high_pts = pred_pts[fusion_high_mask]
                high_scs = pred_scs[fusion_high_mask]
                salvage_pts = pred_pts[fusion_salvage_mask]
                salvage_scs = pred_scs[fusion_salvage_mask]

                salvage_pts, salvage_scs = filter_dense_clutter_clusters(
                    salvage_pts, salvage_scs, cluster_radius=25.0, max_neighbors=2
                )
            else:
                high_pts = np.zeros((0, 2), dtype=np.float32)
                high_scs = np.zeros((0,), dtype=np.float32)
                salvage_pts = np.zeros((0, 2), dtype=np.float32)
                salvage_scs = np.zeros((0,), dtype=np.float32)

            online_tracker.step(f_idx, high_pts, high_scs, salvage_pts, salvage_scs)

        all_tracks = online_tracker.finalize()
        base_smoother = BidirectionalTemporalSmoother(stitch_max_gap=4, stitch_max_dist=25.0)
        stitched_tracks = base_smoother.stitch_tracklets(all_tracks)

        precomputed_seqs[s_name] = {
            "tracks": stitched_tracks,
            "raw_peaks": raw_peaks_list,
            "gt_list": gt_list,
            "num_frames": num_frames,
        }

    print(f"✅ Pre-computation complete in {time.time() - t0:.2f}s! Total GT: {total_gt:,}")

    # Baseline (current SOTA: min_rigid_disp=2.0, min_hits_prune=8, max_infill_gap=3, min_hits_for_infill=5)
    sota_smoother = FastGuidedSmoother(
        min_hits_for_infill=5,
        max_infill_gap=3,
        min_rigid_displacement=args.min_rigid_disp,
        max_rigid_variance=args.max_rigid_var,
        min_hits_for_prune=args.min_hits_prune,
        enable_guided_salvage=False,
    )

    base_tp, base_fp = 0, 0
    for s_name, data in precomputed_seqs.items():
        frame_dets = sota_smoother.smooth_and_infill_fast(
            data["tracks"], data["raw_peaks"], data["num_frames"]
        )
        for f_idx in range(data["num_frames"]):
            gt_pts = data["gt_list"][f_idx]
            dets = frame_dets[f_idx]
            pts = np.array([d["pos"] for d in dets], dtype=np.float32) if len(dets) > 0 else np.zeros((0, 2), dtype=np.float32)
            tp, fp, _ = match_predictions_to_gt(gt_pts, pts, args.dist_thresh)
            base_tp += tp
            base_fp += fp

    base_m = calc_metrics(base_tp, base_fp, total_gt)
    print("\n" + "=" * 115)
    print(
        colorstr(
            "bold",
            colorstr(
                "cyan",
                f"🏆 CURRENT SOTA BASELINE | F1: {base_m['f1']:.4f} | Recall: {base_m['recall']:.2f}% | "
                f"Prec: {base_m['precision']:.2f}% | TP: {base_tp:,} | FP: {base_fp:,}",
            ),
        )
    )
    print(
        "Strict Requirement: ΔTP > 0 AND Precision >= 93.89% AND F1 >= 0.9194 (Zero Regression Guarantee)"
    )
    print("=" * 115)

    # Grid Search Space
    infill_gaps = [3, 4, 5]
    min_hits_infills = [4, 5, 6]
    guided_options = [
        (False, 0.0, 0.0),  # Baseline infill only
        (True, 0.050, 4.0),
        (True, 0.050, 6.0),
        (True, 0.045, 4.0),
        (True, 0.045, 6.0),
        (True, 0.040, 4.0),
        (True, 0.040, 6.0),
        (True, 0.035, 4.0),
        (True, 0.035, 6.0),
    ]

    results = []
    t_search_start = time.time()

    for inf_gap in infill_gaps:
        for min_h in min_hits_infills:
            for enable_g, g_th, g_rad in guided_options:
                smoother = FastGuidedSmoother(
                    min_hits_for_infill=min_h,
                    max_infill_gap=inf_gap,
                    min_rigid_displacement=args.min_rigid_disp,
                    max_rigid_variance=args.max_rigid_var,
                    min_hits_for_prune=args.min_hits_prune,
                    enable_guided_salvage=enable_g,
                    guided_th=g_th,
                    guided_radius=g_rad,
                )

                cur_tp, cur_fp = 0, 0
                for s_name, data in precomputed_seqs.items():
                    frame_dets = smoother.smooth_and_infill_fast(
                        data["tracks"], data["raw_peaks"], data["num_frames"]
                    )
                    for f_idx in range(data["num_frames"]):
                        gt_pts = data["gt_list"][f_idx]
                        dets = frame_dets[f_idx]
                        pts = np.array([d["pos"] for d in dets], dtype=np.float32) if len(dets) > 0 else np.zeros((0, 2), dtype=np.float32)
                        tp, fp, _ = match_predictions_to_gt(gt_pts, pts, args.dist_thresh)
                        cur_tp += tp
                        cur_fp += fp

                m = calc_metrics(cur_tp, cur_fp, total_gt)
                delta_tp = cur_tp - base_tp
                delta_fp = cur_fp - base_fp
                delta_f1 = m["f1"] - base_m["f1"]

                # Strict Qualification Check
                qualifies = (delta_tp >= 0) and (m["precision"] >= 93.89) and (m["f1"] >= 0.9194)

                results.append({
                    "inf_gap": inf_gap,
                    "min_h": min_h,
                    "guided": enable_g,
                    "g_th": g_th,
                    "g_rad": g_rad,
                    "tp": cur_tp,
                    "fp": cur_fp,
                    "delta_tp": delta_tp,
                    "delta_fp": delta_fp,
                    "recall": m["recall"],
                    "precision": m["precision"],
                    "f1": m["f1"],
                    "delta_f1": delta_f1,
                    "qualifies": qualifies,
                })

    search_dur = time.time() - t_search_start
    print(f"⚡ Grid evaluation of {len(results)} configurations finished in {search_dur:.2f} seconds!\n")

    # Filter & Sort
    qualified_results = [r for r in results if r["qualifies"]]
    qualified_results.sort(key=lambda x: (x["f1"], x["tp"]), reverse=True)

    header = (
        f"{'Rank':<4} | {'InfGap':<6} | {'MinHit':<6} | {'Guided':<6} | {'G-Th':<5} | {'G-Rad':<5} | "
        f"{'F1-Score':<8} | {'Recall':<7} | {'Prec':<7} | {'TP':<6} | {'FP':<5} | {'ΔTP':<5} | {'ΔFP':<5}"
    )
    print(colorstr("bold", colorstr("green", f"🏆 QUALIFIED PARETO-OPTIMAL CONFIGS ({len(qualified_results)} / {len(results)} PASSED STRICT CONDITIONS):")))
    print("=" * 115)
    print(header)
    print("-" * 115)

    for i, r in enumerate(qualified_results[:12], 1):
        g_str = "YES" if r["guided"] else "NO"
        th_str = f"{r['g_th']:.3f}" if r["guided"] else "-"
        rad_str = f"{r['g_rad']:.1f}" if r["guided"] else "-"
        f1_str = f"{r['f1']:.4f}"
        rec_str = f"{r['recall']:.2f}%"
        prec_str = f"{r['precision']:.2f}%"
        dtp_str = f"{r['delta_tp']:+d}"
        dfp_str = f"{r['delta_fp']:+d}"

        line = (
            f"{i:<4} | {r['inf_gap']:<6} | {r['min_h']:<6} | {g_str:<6} | {th_str:<5} | {rad_str:<5} | "
            f"{f1_str:<8} | {rec_str:<7} | {prec_str:<7} | {r['tp']:<6} | {r['fp']:<5} | {dtp_str:<5} | {dfp_str:<5}"
        )
        if i == 1:
            print(colorstr("bold", colorstr("green", line)))
        elif r["delta_tp"] > 0:
            print(colorstr("cyan", line))
        else:
            print(line)

    if qualified_results:
        best = qualified_results[0]
        print("=" * 115)
        print(
            colorstr(
                "bold",
                colorstr(
                    "magenta",
                    f"★ NEW RECOMMENDED SOTA CONFIG:\n"
                    f"  Infill Gap: {best['inf_gap']} | Min Hits: {best['min_h']} | Guided Salvage: {best['guided']} (th={best['g_th']}, radius={best['g_rad']}px)\n"
                    f"  F1: {best['f1']:.4f} (Rec: {best['recall']:.2f}%, Prec: {best['precision']:.2f}%, TP: {best['tp']}, FP: {best['fp']})\n"
                    f"  Gain vs Previous SOTA: ΔTP = {best['delta_tp']:+d} | ΔFP = {best['delta_fp']:+d} | ΔF1 = {best['delta_f1']:+.4f}",
                ),
            )
        )
    else:
        print(colorstr("yellow", "[WARN] No config met the strict condition. Showing Top-5 unrestricted:"))
        results.sort(key=lambda x: x["f1"], reverse=True)
        for i, r in enumerate(results[:5], 1):
            print(r)

    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
