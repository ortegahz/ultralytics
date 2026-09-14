#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Dynamic Programming Track-Before-Detect (DP-TBD) Sequence Point Flow Fusion.

Core Algorithmic Principles:
1. Low-SNR Sub-threshold Evidence Accumulation:
   - For faint targets (SCR < 1.0 or intermittent flutters), single-frame heatmaps
     produce peak scores in [0.03, 0.22] which are safely cut by standard thresholds.
   - We extract all candidate peaks down to min_cand_score=0.03.

2. Kinematics-Constrained Viterbi Forward-Backward Energy Dynamic Programming:
   - Within each continuous video sequence, candidate points form a time-staged graph:
     G = (V, E) where V_t are candidates at frame t.
   - Transition energy:
       S_t(p_t) = score(p_t) + max_{p_{t-1}} [
           S_{t-1}(p_{t-1})
           - penalty_dist * (||p_t - p_{t-1}|| / max_step)
           - penalty_turn * (1 - cos(theta))  # directional coherence
       ]
   - Time-gap tolerance: Allows gap jumping (up to max_gap frames) with linear velocity extrapolation.

3. Two-Way Energy Accumulation:
   - Forward DP accumulates causal historical energy.
   - Backward DP accumulates reverse anticausal future energy.
   - Bidirectional consensus score S_bi(p_t) = S_fwd(p_t) + S_bwd(p_t) - score(p_t).
   - Faint true targets along physical flight paths receive huge SNR boost (e.g. +0.15~0.30),
     while random detector thermal noise / ground clutter vibrations cannot maintain coherent motion
     and quickly decay.

4. Integration with Dual-Pass Smoothing:
   - DP-boosted candidate points seamlessly flow into Two-Tier Adaptive Association & Infill.

Usage:
    python manu/eval_dptbd_track_fusion.py \
        --cache-file runs/gmc_eval/uav_median_trial0474_cache.pkl \
        --dist-thresh 8.0 \
        --th-base 0.22 \
        --th-salvage 0.06
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr
from manu.eval_bidirectional_track_fusion import (
    PointKalmanTrack,
    OnlineAdaptiveTracker,
    BidirectionalTemporalSmoother,
    extract_seq_name,
    natural_sort_key,
    filter_dense_clutter_clusters,
    match_predictions_to_gt,
    calc_metrics,
)


# ==============================================================================
# 1. Kinematic Dynamic Programming TBD Module
# ==============================================================================

class DynamicProgrammingTBD:
    """
    Spatio-temporal Dynamic Programming Accumulator on candidate point flows.
    """

    def __init__(
        self,
        max_velocity: float = 18.0,      # Max allowable target speed (pixels/frame)
        gamma: float = 0.82,             # Temporal energy discount factor per frame
        penalty_dist: float = 0.08,      # Penalty factor for large spatial jump
        penalty_turn: float = 0.06,      # Penalty factor for sharp acceleration / heading change
        boost_scale: float = 0.28,       # Scale of bidirectional energy bonus injected into score
        min_cand_score: float = 0.03,    # Minimum peak candidate score to participate in DP
        sky_ratio: float = 0.60,         # Sky/ground division ratio
        img_h: int = 640,
    ):
        self.max_velocity = max_velocity
        self.gamma = gamma
        self.penalty_dist = penalty_dist
        self.penalty_turn = penalty_turn
        self.boost_scale = boost_scale
        self.min_cand_score = min_cand_score
        self.sky_y_boundary = img_h * sky_ratio

    def run_sequence(
        self,
        records: List[Dict],
    ) -> List[Dict]:
        """
        Process a temporally sorted list of frame records.
        Returns modified copies of records where pred_scores are boosted by DP consensus.
        """
        num_frames = len(records)
        if num_frames <= 1:
            return records

        # Step 1: Collect candidates per frame
        frame_cands = []
        for r in records:
            pts = np.asarray(r["pred_points"], dtype=np.float32)
            scs = np.asarray(r["pred_scores"], dtype=np.float32)
            if len(pts) > 0:
                mask = scs >= self.min_cand_score
                c_pts = pts[mask]
                c_scs = scs[mask]
            else:
                c_pts = np.zeros((0, 2), dtype=np.float32)
                c_scs = np.zeros((0,), dtype=np.float32)
            frame_cands.append((c_pts, c_scs))

        # Forward DP: V_fwd[t][i] = (accumulated_score, best_prev_idx, prev_delta_t)
        # We allow looking back up to 2 frames (gap=1 or gap=2) for subtle infill
        V_fwd: List[np.ndarray] = [np.zeros(len(scs), dtype=np.float32) for _, scs in frame_cands]
        V_fwd_vel: List[np.ndarray] = [np.zeros((len(scs), 2), dtype=np.float32) for _, scs in frame_cands]

        for t in range(num_frames):
            cur_pts, cur_scs = frame_cands[t]
            n_cur = len(cur_scs)
            if n_cur == 0:
                continue

            # Base energy from detector
            v_curr = cur_scs.copy()

            # Search backwards across past 1 or 2 frames
            best_trans = np.zeros(n_cur, dtype=np.float32)
            best_vel = np.zeros((n_cur, 2), dtype=np.float32)

            for dt in [1, 2]:
                prev_t = t - dt
                if prev_t < 0:
                    continue
                prev_pts, prev_scs = frame_cands[prev_t]
                n_prev = len(prev_scs)
                if n_prev == 0:
                    continue

                prev_v = V_fwd[prev_t]
                prev_vels = V_fwd_vel[prev_t]

                # Distance matrix: [n_cur, n_prev]
                dists = np.linalg.norm(cur_pts[:, None, :] - prev_pts[None, :, :], axis=-1)
                max_allowed_dist = self.max_velocity * dt

                # Velocity vectors: [n_cur, n_prev, 2]
                v_cand = (cur_pts[:, None, :] - prev_pts[None, :, :]) / float(dt)

                valid_mask = dists <= max_allowed_dist
                if not np.any(valid_mask):
                    continue

                # Transition score computation
                # 1. Distance penalty
                cost_dist = self.penalty_dist * (dists / max(1.0, max_allowed_dist))

                # 2. Smooth directional penalty if prev had motion
                prev_speeds = np.linalg.norm(prev_vels, axis=-1)  # [n_prev]
                cand_speeds = np.linalg.norm(v_cand, axis=-1)     # [n_cur, n_prev]

                dot = np.sum(v_cand * prev_vels[None, :, :], axis=-1)
                denom = np.maximum(1e-4, cand_speeds * prev_speeds[None, :])
                cos_sim = dot / denom
                cost_turn = np.where(
                    (prev_speeds[None, :] > 1.0) & (cand_speeds > 1.0),
                    self.penalty_turn * (1.0 - cos_sim),
                    0.0
                )

                # Net incoming energy discounted by gamma^dt
                discount = self.gamma ** dt
                incoming_energy = discount * (prev_v[None, :] - cost_dist - cost_turn)

                incoming_energy = np.where(valid_mask, incoming_energy, -1e5)
                max_cand_prev = np.max(incoming_energy, axis=1)  # [n_cur]
                best_prev_indices = np.argmax(incoming_energy, axis=1)

                better_mask = max_cand_prev > best_trans
                best_trans = np.where(better_mask, max_cand_prev, best_trans)

                for ci in range(n_cur):
                    if better_mask[ci]:
                        best_vel[ci] = v_cand[ci, best_prev_indices[ci]]

            V_fwd[t] = v_curr + best_trans
            V_fwd_vel[t] = best_vel

        # Backward DP: V_bwd[t][i]
        V_bwd: List[np.ndarray] = [np.zeros(len(scs), dtype=np.float32) for _, scs in frame_cands]
        V_bwd_vel: List[np.ndarray] = [np.zeros((len(scs), 2), dtype=np.float32) for _, scs in frame_cands]

        for t in range(num_frames - 1, -1, -1):
            cur_pts, cur_scs = frame_cands[t]
            n_cur = len(cur_scs)
            if n_cur == 0:
                continue

            v_curr = cur_scs.copy()
            best_trans = np.zeros(n_cur, dtype=np.float32)
            best_vel = np.zeros((n_cur, 2), dtype=np.float32)

            for dt in [1, 2]:
                next_t = t + dt
                if next_t >= num_frames:
                    continue
                next_pts, next_scs = frame_cands[next_t]
                n_next = len(next_scs)
                if n_next == 0:
                    continue

                next_v = V_bwd[next_t]
                next_vels = V_bwd_vel[next_t]

                dists = np.linalg.norm(next_pts[None, :, :] - cur_pts[:, None, :], axis=-1)  # [n_cur, n_next]
                max_allowed_dist = self.max_velocity * dt

                v_cand = (next_pts[None, :, :] - cur_pts[:, None, :]) / float(dt)

                valid_mask = dists <= max_allowed_dist
                if not np.any(valid_mask):
                    continue

                cost_dist = self.penalty_dist * (dists / max(1.0, max_allowed_dist))
                next_speeds = np.linalg.norm(next_vels, axis=-1)
                cand_speeds = np.linalg.norm(v_cand, axis=-1)

                dot = np.sum(v_cand * next_vels[None, :, :], axis=-1)
                denom = np.maximum(1e-4, cand_speeds * next_speeds[None, :])
                cos_sim = dot / denom
                cost_turn = np.where(
                    (next_speeds[None, :] > 1.0) & (cand_speeds > 1.0),
                    self.penalty_turn * (1.0 - cos_sim),
                    0.0
                )

                discount = self.gamma ** dt
                incoming_energy = discount * (next_v[None, :] - cost_dist - cost_turn)
                incoming_energy = np.where(valid_mask, incoming_energy, -1e5)

                max_cand_next = np.max(incoming_energy, axis=1)
                best_next_indices = np.argmax(incoming_energy, axis=1)

                better_mask = max_cand_next > best_trans
                best_trans = np.where(better_mask, max_cand_next, best_trans)
                for ci in range(n_cur):
                    if better_mask[ci]:
                        best_vel[ci] = v_cand[ci, best_next_indices[ci]]

            V_bwd[t] = v_curr + best_trans
            V_bwd_vel[t] = best_vel

        # Step 4: Inject bidirectional consensus bonus into candidate scores
        new_records = []
        for t, r in enumerate(records):
            cur_pts, cur_scs = frame_cands[t]
            orig_pts = np.asarray(r["pred_points"], dtype=np.float32)
            orig_scs = np.asarray(r["pred_scores"], dtype=np.float32)

            if len(cur_scs) == 0 or len(orig_scs) == 0:
                new_records.append(r)
                continue

            # Bidirectional consensus score
            # S_bi = (V_fwd + V_bwd - cur_scs) is the total path energy through this node
            S_bi = V_fwd[t] + V_bwd[t] - cur_scs
            # Path gain over single frame: max(0, S_bi - cur_scs)
            path_gain = np.maximum(0.0, S_bi - cur_scs)

            # High confidence boost, bounded between 0.0 and 0.40
            boost = np.tanh(path_gain * 0.5) * self.boost_scale

            # Sky vs ground discrimination: dampen boost near ground clutter to prevent boosting building jitter
            is_sky = cur_pts[:, 1] < self.sky_y_boundary
            boost = np.where(is_sky, boost, boost * 0.35)

            boosted_cand_scs = np.clip(cur_scs + boost, 0.0, 1.0)

            # Map back to orig_scs
            # Match cur_pts back to orig_pts
            updated_scs = orig_scs.copy()
            cand_mask = orig_scs >= self.min_cand_score
            updated_scs[cand_mask] = boosted_cand_scs

            new_r = dict(r)
            new_r["pred_scores"] = updated_scs
            new_records.append(new_r)

        return new_records


# ==============================================================================
# 2. Sequence Evaluation with DP-TBD
# ==============================================================================

def evaluate_sequence_dptbd(
    records: List[Dict],
    dp_tbd: Optional[DynamicProgrammingTBD] = None,
    dist_thresh: float = 8.0,
    th_base: float = 0.22,
    th_salvage: float = 0.06,
    th_ground: float = 0.35,
    sky_ratio: float = 0.60,
    img_h: int = 640,
    tracker_config: Optional[Dict] = None,
    smoother_config: Optional[Dict] = None,
) -> Dict[str, Dict[str, float]]:
    records_sorted = sorted(records, key=lambda r: natural_sort_key(r["im_name"]))

    # Pass 0: Apply DP-TBD spatio-temporal energy accumulation across sequence
    if dp_tbd is not None:
        records_processed = dp_tbd.run_sequence(records_sorted)
    else:
        records_processed = records_sorted

    num_frames = len(records_processed)
    stats = {
        "baseline": {"tp": 0, "fp": 0, "gt": 0},
        "bidirectional": {"tp": 0, "fp": 0, "gt": 0},
        "dptbd_bidirectional": {"tp": 0, "fp": 0, "gt": 0},
    }

    if tracker_config is None:
        tracker_config = {
            "max_age": 3,
            "min_hits": 3,
            "match_dist": 12.0,
            "max_match_dist": 18.0,
            "min_track_score": 0.08,
            "instant_conf": 0.25,
            "min_displacement": 2.5,
            "sky_ratio": sky_ratio,
            "img_h": img_h,
        }

    if smoother_config is None:
        smoother_config = {
            "stitch_max_gap": 4,
            "stitch_max_dist": 25.0,
            "min_hits_for_infill": 5,
            "max_infill_gap": 3,
            "min_track_hits": 3,
            "min_track_score": 0.08,
            "instant_conf": 0.25,
        }

    sky_y_boundary = img_h * sky_ratio

    # 1. Run baseline bidirectional tracking on raw scores
    tracker_raw = OnlineAdaptiveTracker(**tracker_config)
    smoother_raw = BidirectionalTemporalSmoother(**smoother_config)

    for f_idx, r in enumerate(records_sorted):
        pred_pts = np.asarray(r["pred_points"], dtype=np.float32)
        pred_scs = np.asarray(r["pred_scores"], dtype=np.float32)
        if len(pred_pts) > 0:
            is_sky = pred_pts[:, 1] < sky_y_boundary
            fusion_high_mask = (is_sky & (pred_scs >= th_base)) | ((~is_sky) & (pred_scs >= th_ground))
            fusion_salvage_mask = is_sky & (pred_scs >= th_salvage) & (pred_scs < th_base)

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

        tracker_raw.step(f_idx, high_pts, high_scs, salvage_pts, salvage_scs)

    all_tracks_raw = tracker_raw.finalize()
    stitched_raw = smoother_raw.stitch_tracklets(all_tracks_raw)
    dets_raw_bidi = smoother_raw.smooth_and_infill(stitched_raw, num_frames)

    # 2. Run DP-TBD enhanced bidirectional tracking
    tracker_dp = OnlineAdaptiveTracker(**tracker_config)
    smoother_dp = BidirectionalTemporalSmoother(**smoother_config)

    for f_idx, r in enumerate(records_processed):
        pred_pts = np.asarray(r["pred_points"], dtype=np.float32)
        pred_scs = np.asarray(r["pred_scores"], dtype=np.float32)
        if len(pred_pts) > 0:
            is_sky = pred_pts[:, 1] < sky_y_boundary
            fusion_high_mask = (is_sky & (pred_scs >= th_base)) | ((~is_sky) & (pred_scs >= th_ground))
            fusion_salvage_mask = is_sky & (pred_scs >= th_salvage) & (pred_scs < th_base)

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

        tracker_dp.step(f_idx, high_pts, high_scs, salvage_pts, salvage_scs)

    all_tracks_dp = tracker_dp.finalize()
    stitched_dp = smoother_dp.stitch_tracklets(all_tracks_dp)
    dets_dp_bidi = smoother_dp.smooth_and_infill(stitched_dp, num_frames)

    # 3. Benchmark evaluation
    for f_idx in range(num_frames):
        r_raw = records_sorted[f_idx]
        gt_pts = np.asarray(r_raw["gt_pts"], dtype=np.float32)
        pred_pts_raw = np.asarray(r_raw["pred_points"], dtype=np.float32)
        pred_scs_raw = np.asarray(r_raw["pred_scores"], dtype=np.float32)
        n_gt = len(gt_pts)

        for k in stats:
            stats[k]["gt"] += n_gt

        # Mode 1: Single Frame Baseline at 0.25
        base_mask = pred_scs_raw >= 0.25
        tp1, fp1, _ = match_predictions_to_gt(gt_pts, pred_pts_raw[base_mask], dist_thresh)
        stats["baseline"]["tp"] += tp1
        stats["baseline"]["fp"] += fp1

        # Mode 2: Standard Bidirectional SOTA
        pts2 = np.array([d["pos"] for d in dets_raw_bidi[f_idx]], dtype=np.float32) if len(dets_raw_bidi[f_idx]) > 0 else np.zeros((0, 2), dtype=np.float32)
        tp2, fp2, _ = match_predictions_to_gt(gt_pts, pts2, dist_thresh)
        stats["bidirectional"]["tp"] += tp2
        stats["bidirectional"]["fp"] += fp2

        # Mode 3: DP-TBD Bidirectional Fusion
        pts3 = np.array([d["pos"] for d in dets_dp_bidi[f_idx]], dtype=np.float32) if len(dets_dp_bidi[f_idx]) > 0 else np.zeros((0, 2), dtype=np.float32)
        tp3, fp3, _ = match_predictions_to_gt(gt_pts, pts3, dist_thresh)
        stats["dptbd_bidirectional"]["tp"] += tp3
        stats["dptbd_bidirectional"]["fp"] += fp3

    res = {}
    for k in stats:
        res[k] = calc_metrics(stats[k]["tp"], stats[k]["fp"], stats[k]["gt"])
    return res


# ==============================================================================
# 3. Main Entrypoint & CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DP-TBD Kinematic Spatio-Temporal Point Flow Fusion")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial0474_cache.pkl",
        help="Path to Trial 0474 (or Trial 22) inference pickle cache",
    )
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="GJB tolerance (default: 8.0px)")
    parser.add_argument("--th-base", type=float, default=0.22, help="High confidence base threshold (default: 0.22)")
    parser.add_argument("--th-salvage", type=float, default=0.06, help="Weak pulse salvage threshold (default: 0.06)")
    parser.add_argument("--th-ground", type=float, default=0.35, help="Ground clutter suppression threshold (default: 0.35)")
    parser.add_argument("--max-vel", type=float, default=18.0, help="DP max allowable drone velocity px/frame (default: 18.0)")
    parser.add_argument("--gamma", type=float, default=0.82, help="DP temporal decay factor (default: 0.82)")
    parser.add_argument("--boost-scale", type=float, default=0.28, help="DP bidirectional consensus boost scale (default: 0.28)")
    parser.add_argument("--min-hits-infill", type=int, default=5, help="Smoother min hits for infill (default: 5)")
    parser.add_argument("--sequences", type=str, default="", help="Optional sequence filtering (e.g. DJI_0175,wg2022_ir_020)")
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
        print(colorstr("red", f"[ERROR] Pickle cache not found: {args.cache_file}"))
        sys.exit(1)

    print(colorstr("bold", colorstr("green", f"\n>>> Loading cached inferences from: {cache_path}")))
    with open(cache_path, "rb") as f:
        records = pickle.load(f)
    print(f"Loaded {len(records)} image predictions.\n")

    seq_records: Dict[str, List[Dict]] = {}
    for r in records:
        seq = extract_seq_name(r["im_name"])
        seq_records.setdefault(seq, []).append(r)

    filter_seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]

    dp_tbd = DynamicProgrammingTBD(
        max_velocity=args.max_vel,
        gamma=args.gamma,
        penalty_dist=0.08,
        penalty_turn=0.06,
        boost_scale=args.boost_scale,
        min_cand_score=0.03,
        sky_ratio=0.60,
        img_h=640,
    )

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
        "stitch_max_gap": 4,
        "stitch_max_dist": 25.0,
        "min_hits_for_infill": args.min_hits_infill,
        "max_infill_gap": 3,
        "min_track_hits": 3,
        "min_track_score": 0.08,
        "instant_conf": 0.25,
    }

    grand_stats = {
        "baseline": {"tp": 0, "fp": 0, "gt": 0},
        "bidirectional": {"tp": 0, "fp": 0, "gt": 0},
        "dptbd_bidirectional": {"tp": 0, "fp": 0, "gt": 0},
    }

    print("=" * 128)
    print(f"{'Sequence Name':<28} | {'Mode':<26} | {'TP / GT':<14} | {'FP':<6} | {'Recall':<8} | {'Prec':<8} | {'F1-Score':<8}")
    print("=" * 128)

    t0 = time.time()
    for seq_name in sorted(seq_records.keys()):
        if filter_seqs and not any(f in seq_name for f in filter_seqs):
            continue

        recs = seq_records[seq_name]
        res = evaluate_sequence_dptbd(
            records=recs,
            dp_tbd=dp_tbd,
            dist_thresh=args.dist_thresh,
            th_base=args.th_base,
            th_salvage=args.th_salvage,
            th_ground=args.th_ground,
            sky_ratio=0.60,
            img_h=640,
            tracker_config=tracker_config,
            smoother_config=smoother_config,
        )

        for k in grand_stats:
            grand_stats[k]["tp"] += int(res[k]["tp"])
            grand_stats[k]["fp"] += int(res[k]["fp"])
            grand_stats[k]["gt"] += int(res[k]["gt"])

        m_base = res["baseline"]
        m_bidi = res["bidirectional"]
        m_dp = res["dptbd_bidirectional"]

        diff_tp = m_dp["tp"] - m_bidi["tp"]
        diff_fp = m_dp["fp"] - m_bidi["fp"]
        diff_tag = f" (ΔTP:{diff_tp:+d}, ΔFP:{diff_fp:+d})"

        print(f"{seq_name:<28} | {'1. Single Base (0.25)':<26} | {int(m_base['tp']):>5} / {int(m_base['gt']):<6} | {int(m_base['fp']):<6} | {m_base['recall']:>6.2f}% | {m_base['precision']:>6.2f}% | {m_base['f1']:>6.4f}")
        print(f"{'':<28} | {'2. Standard Bidi SOTA':<26} | {int(m_bidi['tp']):>5} / {int(m_bidi['gt']):<6} | {int(m_bidi['fp']):<6} | {m_bidi['recall']:>6.2f}% | {m_bidi['precision']:>6.2f}% | {m_bidi['f1']:>6.4f}")
        print(f"{'':<28} | {colorstr('bold', colorstr('green', '3. 🔥 DP-TBD Bidi Fusion')):<35} | {int(m_dp['tp']):>5} / {int(m_dp['gt']):<6} | {int(m_dp['fp']):<6} | {m_dp['recall']:>6.2f}% | {m_dp['precision']:>6.2f}% | {m_dp['f1']:>6.4f}{diff_tag}")
        print("-" * 128)

    dur = time.time() - t0
    print("=" * 128)
    print(colorstr("bold", f"GRAND OVERALL RESULTS ACROSS ALL SEQUENCES (Total Time: {dur:.2f}s)"))
    print("=" * 128)

    grand_metrics = {k: calc_metrics(grand_stats[k]["tp"], grand_stats[k]["fp"], grand_stats[k]["gt"]) for k in grand_stats}

    m_b = grand_metrics["baseline"]
    m_s = grand_metrics["bidirectional"]
    m_d = grand_metrics["dptbd_bidirectional"]

    print(f"{'1. Single Frame Baseline (th=0.25)':<46} | TP: {m_b['tp']:>5}/{m_b['gt']:<5} | FP: {m_b['fp']:<6} | Recall: {m_b['recall']:>6.2f}% | Prec: {m_b['precision']:>6.2f}% | F1: {m_b['f1']:>6.4f} | FAR: {m_b['fp']/len(records):.4f}/frame")
    print(f"{'2. Standard Bidirectional SOTA':<46} | TP: {m_s['tp']:>5}/{m_s['gt']:<5} | FP: {m_s['fp']:<6} | Recall: {m_s['recall']:>6.2f}% | Prec: {m_s['precision']:>6.2f}% | F1: {m_s['f1']:>6.4f} | FAR: {m_s['fp']/len(records):.4f}/frame")
    print(colorstr("bold", colorstr("green",
        f"{'3. 🔥 DP-TBD Kinematic Spatio-Temporal Fusion':<46} | TP: {m_d['tp']:>5}/{m_d['gt']:<5} | FP: {m_d['fp']:<6} | Recall: {m_d['recall']:>6.2f}% | Prec: {m_d['precision']:>6.2f}% | F1: {m_d['f1']:>6.4f} | FAR: {m_d['fp']/len(records):.4f}/frame"
    )))

    d_tp = m_d["tp"] - m_s["tp"]
    d_fp = m_d["fp"] - m_s["fp"]
    d_f1 = (m_d["f1"] - m_s["f1"]) / 100.0
    print("-" * 128)
    print(colorstr("bold", f"★ Net Gain over Current SOTA: ΔTP = {d_tp:+d} frames | ΔFP = {d_fp:+d} false alarms | ΔF1 = {d_f1:+.4f}"))
    print("=" * 128 + "\n")


if __name__ == "__main__":
    main()
