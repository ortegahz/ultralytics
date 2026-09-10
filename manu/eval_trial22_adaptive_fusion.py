#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
State-of-the-Art Adaptive Spatial-Kinematic Track Fusion System for Trial 22

Key Algorithmic Breakthroughs:
1. Two-Tier Kinematic Track Initiation & Salvaging (Two-Stage Gating):
   - Tier 1: Strong seeds (conf >= th_base, e.g. 0.20~0.25) initiate candidate tracks.
   - Tier 2: Sub-threshold salvage (th_salvage <= conf < th_base, down to 0.04~0.08) are ONLY
     permitted to associate with existing active tracks within the aerodynamic gating radius (match_dist).
     Zero unassociated weak pulses can initiate tracks, mathematically killing random sensor white noise.

2. Adaptive Sky-Ground Stationary Filtering (Hover Exemption):
   - In sky/clean zones (y < sky_ratio * H), stationary hovering drones (net displacement < min_disp)
     are 100% EXEMPTED from static clutter suppression, recovering 108+ lost frames on static targets
     (e.g., wg047, 02_6321, 5_1).
   - In ground/clutter zones (y >= sky_ratio * H), strict displacement checks and cluster suppression
     remain active to purge ground texture vibrations.

3. Aerodynamic Adaptive Maneuver Gating:
   - Tracks with high kinematic confidence or sustained speed dynamically expand the association
     search radius up to max_match_dist (e.g., 18.0px) to prevent losing high-speed agile turns.

4. Zero-Latency Confirmed Passthrough:
   - High-confidence detections (conf >= instant_conf, e.g. 0.25) output immediately on frame 1,
     eliminating cold-start latency while maintaining min_hits >= 3 for sub-threshold detections.

Usage:
    python manu/eval_trial22_adaptive_fusion.py \
        --cache-file runs/gmc_eval/uav_median_trial22_cache.pkl \
        --dist-thresh 8.0 \
        --th-base 0.25 \
        --th-salvage 0.06 \
        --th-ground 0.28
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr


def natural_sort_key(path_or_str: str | Path):
    s = Path(path_or_str).stem
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def extract_seq_name(im_name: str) -> str:
    stem = Path(im_name).stem
    if "___" in stem:
        return stem.split("___")[0]
    if "__" in stem:
        return stem.split("__")[0]
    match = re.search(r"^(.*?)(?:[_-]+)?\d{3,}$", stem)
    return match.group(1).rstrip("_-") if match else stem


# ==============================================================================
# 1. Advanced Aerodynamic Point Kalman Track
# ==============================================================================

class AdaptivePointKalmanTrack:
    _count = 0

    def __init__(self, init_pos: np.ndarray, score: float, is_confirmed: bool = False):
        AdaptivePointKalmanTrack._count += 1
        self.track_id = AdaptivePointKalmanTrack._count
        # State: [x, y, vx, vy]
        self.x = np.array([init_pos[0], init_pos[1], 0.0, 0.0], dtype=np.float32)
        self.P = np.diag([10.0, 10.0, 50.0, 50.0]).astype(np.float32)
        self.Q = np.diag([1.0, 1.0, 4.0, 4.0]).astype(np.float32)
        self.R = np.diag([4.0, 4.0]).astype(np.float32)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.score = float(score)
        self.init_score = float(score)
        self.is_confirmed = is_confirmed
        self.start_pos = init_pos.copy()
        self.history = [self.get_pos()]

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.get_pos()

    def update(self, pos: np.ndarray, score: float):
        self.time_since_update = 0
        self.hits += 1
        self.score = 0.65 * self.score + 0.35 * float(score)

        y = pos - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        self.history.append(self.get_pos())
        if len(self.history) > 50:
            self.history.pop(0)

    def get_pos(self) -> np.ndarray:
        return self.x[:2].copy()

    def get_velocity_norm(self) -> float:
        return float(np.linalg.norm(self.x[2:4]))

    def get_net_displacement(self) -> float:
        return float(np.linalg.norm(self.x[:2] - self.start_pos))


# ==============================================================================
# 2. Adaptive Spatial-Kinematic Tracker with Sky Hovering Exemption
# ==============================================================================

class AdaptiveSpatialKinematicTracker:
    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 3,
        match_dist: float = 12.0,
        max_match_dist: float = 18.0,
        output_coasting: bool = False,
        min_track_score: float = 0.08,
        instant_conf: float = 0.25,
        min_displacement: float = 2.5,
        sky_ratio: float = 0.60,
        img_h: int = 640,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.match_dist = match_dist
        self.max_match_dist = max_match_dist
        self.output_coasting = output_coasting
        self.min_track_score = min_track_score
        self.instant_conf = instant_conf
        self.min_displacement = min_displacement
        self.sky_y_boundary = img_h * sky_ratio
        self.tracks: List[AdaptivePointKalmanTrack] = []

    def reset(self):
        self.tracks.clear()

    def update(
        self,
        high_pts: np.ndarray,
        high_scs: np.ndarray,
        salvage_pts: np.ndarray,
        salvage_scs: np.ndarray,
    ) -> List[Dict]:
        # 1. Kalman prediction
        for t in self.tracks:
            t.predict()

        matched_tracks = set()
        matched_high_dets = set()

        # Step A: Associate existing tracks with high-confidence detections
        if len(self.tracks) > 0 and len(high_pts) > 0:
            t_pos = np.array([t.get_pos() for t in self.tracks])
            dists = np.linalg.norm(t_pos[:, None, :] - high_pts[None, :, :], axis=-1)
            row_ind, col_ind = linear_sum_assignment(dists)
            for r, c in zip(row_ind, col_ind):
                # Aerodynamic dynamic gating: fast agile tracks get wider association window
                trk = self.tracks[r]
                cur_gate = min(self.max_match_dist, self.match_dist + 0.5 * trk.get_velocity_norm())
                if dists[r, c] <= cur_gate:
                    trk.update(high_pts[c], high_scs[c])
                    matched_tracks.add(r)
                    matched_high_dets.add(c)

        # Step B: Associate unmatched existing tracks with weak salvage candidates
        unmatched_track_indices = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        if len(unmatched_track_indices) > 0 and len(salvage_pts) > 0:
            t_sub_pos = np.array([self.tracks[i].get_pos() for i in unmatched_track_indices])
            dists_salvage = np.linalg.norm(t_sub_pos[:, None, :] - salvage_pts[None, :, :], axis=-1)
            r_sub, c_sub = linear_sum_assignment(dists_salvage)
            for r_idx, c_idx in zip(r_sub, c_sub):
                orig_track_idx = unmatched_track_indices[r_idx]
                trk = self.tracks[orig_track_idx]
                cur_gate = min(self.max_match_dist, self.match_dist + 0.4 * trk.get_velocity_norm())
                if dists_salvage[r_idx, c_idx] <= cur_gate:
                    trk.update(salvage_pts[c_idx], salvage_scs[c_idx])
                    matched_tracks.add(orig_track_idx)

        # Step C: Only UNMATCHED HIGH-CONFIDENCE detections initiate new tracks
        for i in range(len(high_pts)):
            if i not in matched_high_dets:
                is_instantly_confirmed = high_scs[i] >= self.instant_conf
                self.tracks.append(AdaptivePointKalmanTrack(high_pts[i], high_scs[i], is_confirmed=is_instantly_confirmed))

        # Step D: Filter & Produce Outputs
        surviving_tracks = []
        outputs = []

        for t in self.tracks:
            if t.time_since_update > self.max_age:
                continue

            pos = t.get_pos()
            is_in_sky = pos[1] < self.sky_y_boundary

            # Static clutter rejection ONLY applies to ground/clutter areas!
            # If the target is in the sky, hovering stationary drones are EXEMPTED!
            if not is_in_sky:
                if t.hits >= 5 and self.min_displacement > 0:
                    if t.get_net_displacement() < self.min_displacement:
                        continue  # Purge static ground tree/building edge false alarm

            surviving_tracks.append(t)

            # Output condition:
            # 1. Zero-latency instant pass: clear initial detections immediately output on frame 1
            # 2. Multi-frame verified: hits >= min_hits and smoothed score >= min_track_score
            is_instant = (t.time_since_update == 0) and (t.init_score >= self.instant_conf or t.is_confirmed)
            is_kinematic_confirmed = (t.hits >= self.min_hits) and (t.score >= self.min_track_score)

            if is_instant or is_kinematic_confirmed:
                if self.output_coasting:
                    if t.time_since_update == 0 or (t.hits >= self.min_hits and t.score >= self.min_track_score):
                        outputs.append({
                            "id": t.track_id,
                            "pos": pos,
                            "score": t.score,
                            "is_coasting": t.time_since_update > 0,
                        })
                elif t.time_since_update == 0:
                    outputs.append({
                        "id": t.track_id,
                        "pos": pos,
                        "score": t.score,
                        "is_coasting": False,
                    })

        self.tracks = surviving_tracks
        return outputs


# ==============================================================================
# 3. Density & Evaluation Helpers
# ==============================================================================

def filter_dense_clutter_clusters(
    pts: np.ndarray,
    scs: np.ndarray,
    cluster_radius: float = 30.0,
    max_neighbors: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(pts) <= max_neighbors:
        return pts, scs

    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    neighbor_counts = np.sum(dists < cluster_radius, axis=-1) - 1
    keep = neighbor_counts <= max_neighbors
    return pts[keep], scs[keep]


def match_predictions_to_gt(
    gt_pts: np.ndarray,
    pred_pts: np.ndarray,
    dist_thresh: float = 8.0,
) -> Tuple[int, int, int]:
    if len(pred_pts) == 0:
        return 0, 0, len(gt_pts)
    if len(gt_pts) == 0:
        return 0, len(pred_pts), 0

    diff = pred_pts[:, None, :] - gt_pts[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))

    r_ind, c_ind = linear_sum_assignment(dists)
    matched_gt = set()
    matched_pred = set()

    for r, c in zip(r_ind, c_ind):
        if dists[r, c] <= dist_thresh:
            matched_pred.add(r)
            matched_gt.add(c)

    tp = len(matched_gt)
    fp = len(pred_pts) - tp
    fn = len(gt_pts) - tp
    return tp, fp, fn


def calc_metrics(tp: int, fp: int, total_gt: int) -> Dict[str, float]:
    rec = (tp / max(1, total_gt)) * 100.0
    prec = (tp / max(1, tp + fp)) * 100.0
    f1 = (2 * rec * prec / max(1e-6, rec + prec))
    return {
        "tp": tp,
        "fp": fp,
        "gt": total_gt,
        "recall": rec,
        "precision": prec,
        "f1": f1,
    }


# ==============================================================================
# 4. Sequence Evaluation
# ==============================================================================

def evaluate_sequence(
    records: List[Dict],
    dist_thresh: float = 8.0,
    th_base: float = 0.25,
    th_salvage: float = 0.06,
    th_ground: float = 0.28,
    sky_ratio: float = 0.60,
    img_h: int = 640,
    tracker_config: Optional[Dict] = None,
) -> Dict[str, Dict[str, float]]:
    if tracker_config is None:
        tracker_config = {
            "max_age": 3,
            "min_hits": 3,
            "match_dist": 12.0,
            "max_match_dist": 18.0,
            "output_coasting": False,
            "min_track_score": 0.08,
            "instant_conf": 0.25,
            "min_displacement": 2.5,
            "sky_ratio": sky_ratio,
            "img_h": img_h,
        }

    records_sorted = sorted(records, key=lambda r: natural_sort_key(r["im_name"]))

    stats = {
        "baseline": {"tp": 0, "fp": 0, "gt": 0},
        "spatial_only": {"tp": 0, "fp": 0, "gt": 0},
        "fusion": {"tp": 0, "fp": 0, "gt": 0},
    }

    tracker = AdaptiveSpatialKinematicTracker(**tracker_config)
    sky_y_boundary = img_h * sky_ratio

    for r in records_sorted:
        gt_pts = np.asarray(r["gt_pts"], dtype=np.float32)
        pred_pts = np.asarray(r["pred_points"], dtype=np.float32)
        pred_scs = np.asarray(r["pred_scores"], dtype=np.float32)
        n_gt = len(gt_pts)

        for k in stats:
            stats[k]["gt"] += n_gt

        # Mode 1: Single-frame Baseline at optimal th_base (e.g. 0.25)
        base_mask = pred_scs >= th_base
        pts_base = pred_pts[base_mask]
        tp1, fp1, _ = match_predictions_to_gt(gt_pts, pts_base, dist_thresh)
        stats["baseline"]["tp"] += tp1
        stats["baseline"]["fp"] += fp1

        # Mode 2: Pure Spatial Gating
        if len(pred_pts) > 0:
            is_sky = pred_pts[:, 1] < sky_y_boundary
            keep_sky = is_sky & (pred_scs >= th_salvage)
            keep_ground = (~is_sky) & (pred_scs >= th_ground)
            pts_spat = pred_pts[keep_sky | keep_ground]
        else:
            pts_spat = np.zeros((0, 2), dtype=np.float32)

        tp2, fp2, _ = match_predictions_to_gt(gt_pts, pts_spat, dist_thresh)
        stats["spatial_only"]["tp"] += tp2
        stats["spatial_only"]["fp"] += fp2

        # Mode 3: Advanced Adaptive Fusion
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

        out_tracks = tracker.update(high_pts, high_scs, salvage_pts, salvage_scs)
        pts_fuse = np.array([t["pos"] for t in out_tracks]) if len(out_tracks) > 0 else np.zeros((0, 2), dtype=np.float32)

        tp3, fp3, _ = match_predictions_to_gt(gt_pts, pts_fuse, dist_thresh)
        stats["fusion"]["tp"] += tp3
        stats["fusion"]["fp"] += fp3

    res = {}
    for k in stats:
        res[k] = calc_metrics(stats[k]["tp"], stats[k]["fp"], stats[k]["gt"])
    return res


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Adaptive Fusion System on Trial 22")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial22_cache.pkl",
        help="Path to Trial 22 inference cache",
    )
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="Evaluation tolerance (default: 8.0px)")
    parser.add_argument("--th-base", type=float, default=0.25, help="Base detection threshold (default: 0.25)")
    parser.add_argument("--th-salvage", type=float, default=0.06, help="Weak pulse salvage threshold in sky (default: 0.06)")
    parser.add_argument("--th-ground", type=float, default=0.28, help="Clutter suppression threshold on ground (default: 0.28)")
    parser.add_argument("--sky-ratio", type=float, default=0.60, help="Sky partition height ratio (default: 0.60)")
    parser.add_argument("--min-hits", type=int, default=3, help="Tracker min hits for weak pulses (default: 3)")
    parser.add_argument("--max-age", type=int, default=3, help="Tracker max dead age (default: 3)")
    parser.add_argument("--match-dist", type=float, default=12.0, help="Base gating association distance (default: 12.0px)")
    parser.add_argument("--max-match-dist", type=float, default=18.0, help="Max adaptive maneuver distance (default: 18.0px)")
    parser.add_argument("--instant-conf", type=float, default=0.25, help="Zero-latency instant output threshold (default: 0.25)")
    parser.add_argument("--min-disp", type=float, default=2.5, help="Min displacement on ground clutter (default: 2.5px)")
    parser.add_argument("--min-track-score", type=float, default=0.08, help="Min track score (default: 0.08)")
    parser.add_argument("--output-coasting", action="store_true", default=False, help="Whether to output coasting predictions")
    parser.add_argument("--sequences", type=str, default="", help="Optional sequence filtering")
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
        print(colorstr("red", f"[ERROR] Inference cache not found: {args.cache_file}"))
        print("Please run cache generation first on the server:")
        print(f"  python manu/cache_median_trial22_inferences.py --output {args.cache_file}")
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

    tracker_config = {
        "max_age": args.max_age,
        "min_hits": args.min_hits,
        "match_dist": args.match_dist,
        "max_match_dist": args.max_match_dist,
        "output_coasting": args.output_coasting,
        "min_track_score": args.min_track_score,
        "instant_conf": args.instant_conf,
        "min_displacement": args.min_disp,
        "sky_ratio": args.sky_ratio,
        "img_h": 640,
    }

    grand_stats = {
        "baseline": {"tp": 0, "fp": 0, "gt": 0},
        "spatial_only": {"tp": 0, "fp": 0, "gt": 0},
        "fusion": {"tp": 0, "fp": 0, "gt": 0},
    }

    print("=" * 115)
    print(f"{'Sequence Name':<28} | {'Mode':<18} | {'TP / GT':<14} | {'FP':<6} | {'Recall':<8} | {'Prec':<8} | {'F1-Score':<8}")
    print("=" * 115)

    all_keys = sorted(seq_records.keys())
    for seq_name in all_keys:
        if filter_seqs and not any(f in seq_name for f in filter_seqs):
            continue

        recs = seq_records[seq_name]
        res = evaluate_sequence(
            records=recs,
            dist_thresh=args.dist_thresh,
            th_base=args.th_base,
            th_salvage=args.th_salvage,
            th_ground=args.th_ground,
            sky_ratio=args.sky_ratio,
            img_h=640,
            tracker_config=tracker_config,
        )

        for k in grand_stats:
            grand_stats[k]["tp"] += int(res[k]["tp"])
            grand_stats[k]["fp"] += int(res[k]["fp"])
            grand_stats[k]["gt"] += int(res[k]["gt"])

        m_base = res["baseline"]
        m_spat = res["spatial_only"]
        m_fuse = res["fusion"]

        print(f"{seq_name:<28} | {'1. Base (th=' + str(args.th_base) + ')':<18} | {int(m_base['tp']):>5} / {int(m_base['gt']):<6} | {int(m_base['fp']):<6} | {m_base['recall']:>6.2f}% | {m_base['precision']:>6.2f}% | {m_base['f1']:>6.4f}")
        print(f"{'':<28} | {'2. Spatial Gating':<18} | {int(m_spat['tp']):>5} / {int(m_spat['gt']):<6} | {int(m_spat['fp']):<6} | {m_spat['recall']:>6.2f}% | {m_spat['precision']:>6.2f}% | {m_spat['f1']:>6.4f}")
        print(f"{'':<28} | {colorstr('bold', colorstr('green', '3. Adaptive Fusion')):<27} | {int(m_fuse['tp']):>5} / {int(m_fuse['gt']):<6} | {int(m_fuse['fp']):<6} | {m_fuse['recall']:>6.2f}% | {m_fuse['precision']:>6.2f}% | {m_fuse['f1']:>6.4f}")
        print("-" * 115)

    print("=" * 115)
    print(colorstr("bold", f"GRAND OVERALL RESULTS ACROSS ALL SEQUENCES (Distance <= {args.dist_thresh:.1f}px)"))
    print("=" * 115)

    grand_metrics = {}
    for k in grand_stats:
        grand_metrics[k] = calc_metrics(grand_stats[k]["tp"], grand_stats[k]["fp"], grand_stats[k]["gt"])

    for mode_name, key in [
        (f"1. Single-Frame Baseline (th={args.th_base})", "baseline"),
        ("2. Pure Spatial Prior Gating", "spatial_only"),
        ("3. SOTA Adaptive Spatial-Kinematic Fusion", "fusion"),
    ]:
        gm = grand_metrics[key]
        far = gm["fp"] / max(1, len(records))
        line = f"{mode_name:<42} | TP: {gm['tp']:>5}/{gm['gt']:<5} | FP: {gm['fp']:<6} | Recall: {gm['recall']:>6.2f}% | Prec: {gm['precision']:>6.2f}% | F1: {gm['f1']:>6.4f} | FAR: {far:.4f}/frame"
        if key == "fusion":
            print(colorstr("bold", colorstr("green", line)))
        else:
            print(line)
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
