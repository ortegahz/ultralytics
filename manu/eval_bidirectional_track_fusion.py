#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Bidirectional Spatio-Temporal Smoothed Track Fusion System for Trial 22 (Scheme 1 SOTA)

Key Algorithmic Breakthroughs over Baseline Fusion:
1. Two-Tier Kinematic Online Association & Hover Exemption (Inherited from Trial 22 SOTA):
   - Tier 1: Strong seeds (conf >= th_base) initiate tracks.
   - Tier 2: Sub-threshold salvage (th_salvage <= conf < th_base) only associate with active tracks.
   - Hover Exemption: Static targets in sky (y < sky_ratio * H) are never purged.

2. Tracklet Stitching (Kinematic Fragment Stitching):
   - When a drone darkens or briefly loses detection for delta_t (e.g. 1~5 frames),
     the online tracker might kill the track and later start a new one.
   - We check terminal kinematics (position extrapolation, velocity alignment, and time gap <= max_gap).
   - If physically coherent, the fragmented tracklets are stitched into one continuous track.

3. Confirmed Track Infill / Interpolation (Rauch-Tung-Striebel inspired interpolation):
   - For confirmed long-life tracks (hits >= min_hits_infill, e.g. 5~8 frames),
     internal missed frames (gaps of 1~3 frames) during momentary flutters or cloud crossings
     are safely infilled using linear or kinematic motion interpolation.
   - Zero infill for unconfirmed or short-lived clutter to guarantee zero false alarm leakage.

4. Backward Verification / Prefix Recovery:
   - For tracks confirmed by hits >= min_hits, their initial warmup detections
     (frames 1 and 2 before hitting min_hits threshold) are retroactively recovered and outputted,
     eliminating cold-start latency without lowering precision.

Usage:
    python manu/eval_bidirectional_track_fusion.py \
        --cache-file runs/gmc_eval/uav_median_trial22_cache.pkl \
        --dist-thresh 8.0 \
        --th-base 0.22 \
        --th-salvage 0.06 \
        --th-ground 0.32
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
# 1. Advanced Aerodynamic Point Kalman Track (With Full Temporal History)
# ==============================================================================

class PointKalmanTrack:
    _count = 0

    def __init__(self, init_pos: np.ndarray, score: float, frame_idx: int, is_confirmed: bool = False):
        PointKalmanTrack._count += 1
        self.track_id = PointKalmanTrack._count
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

        self.start_frame = frame_idx
        self.last_frame = frame_idx
        self.last_observed_frame = frame_idx
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.score = float(score)
        self.init_score = float(score)
        self.is_confirmed = is_confirmed
        self.start_pos = init_pos.copy()

        # Keyframe history: frame_idx -> (pos, score, is_measured)
        self.observations: Dict[int, Tuple[np.ndarray, float, bool]] = {
            frame_idx: (init_pos.copy(), float(score), True)
        }

    def predict(self, frame_idx: int) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        self.last_frame = frame_idx
        return self.get_pos()

    def update(self, pos: np.ndarray, score: float, frame_idx: int):
        self.time_since_update = 0
        self.hits += 1
        self.score = 0.65 * self.score + 0.35 * float(score)
        self.last_frame = frame_idx
        self.last_observed_frame = frame_idx

        y = pos - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        self.observations[frame_idx] = (pos.copy(), float(score), True)

    def get_pos(self) -> np.ndarray:
        return self.x[:2].copy()

    def get_velocity(self) -> np.ndarray:
        return self.x[2:4].copy()

    def get_velocity_norm(self) -> float:
        return float(np.linalg.norm(self.x[2:4]))

    def get_net_displacement(self) -> float:
        return float(np.linalg.norm(self.x[:2] - self.start_pos))


# ==============================================================================
# 2. Online Adaptive Gating Tracker (Pass 1)
# ==============================================================================

class OnlineAdaptiveTracker:
    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 3,
        match_dist: float = 12.0,
        max_match_dist: float = 18.0,
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
        self.min_track_score = min_track_score
        self.instant_conf = instant_conf
        self.min_displacement = min_displacement
        self.sky_y_boundary = img_h * sky_ratio

        self.active_tracks: List[PointKalmanTrack] = []
        self.finished_tracks: List[PointKalmanTrack] = []

    def reset(self):
        self.active_tracks.clear()
        self.finished_tracks.clear()

    def step(
        self,
        frame_idx: int,
        high_pts: np.ndarray,
        high_scs: np.ndarray,
        salvage_pts: np.ndarray,
        salvage_scs: np.ndarray,
    ):
        # 1. Kalman prediction
        for t in self.active_tracks:
            t.predict(frame_idx)

        matched_tracks = set()
        matched_high_dets = set()

        # Step A: Associate existing tracks with high-confidence detections
        if len(self.active_tracks) > 0 and len(high_pts) > 0:
            t_pos = np.array([t.get_pos() for t in self.active_tracks])
            dists = np.linalg.norm(t_pos[:, None, :] - high_pts[None, :, :], axis=-1)
            row_ind, col_ind = linear_sum_assignment(dists)
            for r, c in zip(row_ind, col_ind):
                trk = self.active_tracks[r]
                cur_gate = min(self.max_match_dist, self.match_dist + 0.5 * trk.get_velocity_norm())
                if dists[r, c] <= cur_gate:
                    trk.update(high_pts[c], high_scs[c], frame_idx)
                    matched_tracks.add(r)
                    matched_high_dets.add(c)

        # Step B: Associate unmatched active tracks with salvage candidates
        unmatched_track_indices = [i for i in range(len(self.active_tracks)) if i not in matched_tracks]
        if len(unmatched_track_indices) > 0 and len(salvage_pts) > 0:
            t_sub_pos = np.array([self.active_tracks[i].get_pos() for i in unmatched_track_indices])
            dists_salvage = np.linalg.norm(t_sub_pos[:, None, :] - salvage_pts[None, :, :], axis=-1)
            r_sub, c_sub = linear_sum_assignment(dists_salvage)
            for r_idx, c_idx in zip(r_sub, c_sub):
                orig_track_idx = unmatched_track_indices[r_idx]
                trk = self.active_tracks[orig_track_idx]
                cur_gate = min(self.max_match_dist, self.match_dist + 0.4 * trk.get_velocity_norm())
                if dists_salvage[r_idx, c_idx] <= cur_gate:
                    trk.update(salvage_pts[c_idx], salvage_scs[c_idx], frame_idx)
                    matched_tracks.add(orig_track_idx)

        # Step C: Only UNMATCHED HIGH-CONFIDENCE detections initiate new tracks
        for i in range(len(high_pts)):
            if i not in matched_high_dets:
                is_instantly_confirmed = high_scs[i] >= self.instant_conf
                self.active_tracks.append(
                    PointKalmanTrack(high_pts[i], high_scs[i], frame_idx, is_confirmed=is_instantly_confirmed)
                )

        # Step D: Cull dead tracks & enforce ground clutter suppression
        surviving = []
        for t in self.active_tracks:
            if t.time_since_update > self.max_age:
                # Track is dying, check if it was a false alarm ground vibration
                pos = t.get_pos()
                is_in_sky = pos[1] < self.sky_y_boundary
                if not is_in_sky and t.hits >= 5 and self.min_displacement > 0:
                    if t.get_net_displacement() < self.min_displacement:
                        continue  # Purge static ground tree/building edge vibration
                self.finished_tracks.append(t)
            else:
                surviving.append(t)
        self.active_tracks = surviving

    def finalize(self) -> List[PointKalmanTrack]:
        for t in self.active_tracks:
            pos = t.get_pos()
            is_in_sky = pos[1] < self.sky_y_boundary
            if not is_in_sky and t.hits >= 5 and self.min_displacement > 0:
                if t.get_net_displacement() < self.min_displacement:
                    continue
            self.finished_tracks.append(t)
        self.active_tracks.clear()
        return self.finished_tracks


# ==============================================================================
# 3. Bidirectional Temporal Smoother & Infill Engine (Pass 2)
# ==============================================================================

class BidirectionalTemporalSmoother:
    def __init__(
        self,
        stitch_max_gap: int = 4,
        stitch_max_dist: float = 25.0,
        min_hits_for_infill: int = 4,
        max_infill_gap: int = 3,
        min_track_hits: int = 3,
        min_track_score: float = 0.08,
        instant_conf: float = 0.25,
    ):
        self.stitch_max_gap = stitch_max_gap
        self.stitch_max_dist = stitch_max_dist
        self.min_hits_for_infill = min_hits_for_infill
        self.max_infill_gap = max_infill_gap
        self.min_track_hits = min_track_hits
        self.min_track_score = min_track_score
        self.instant_conf = instant_conf

    def stitch_tracklets(self, tracks: List[PointKalmanTrack]) -> List[PointKalmanTrack]:
        """
        Merge fragments belonging to the same physical drone flight trajectory.
        A fragmented tracklet starts shortly (1~stitch_max_gap frames) after an earlier tracklet ended.
        """
        if len(tracks) <= 1:
            return tracks

        # Sort tracks by start frame
        tracks_sorted = sorted(tracks, key=lambda t: t.start_frame)
        merged: List[PointKalmanTrack] = []

        for trk in tracks_sorted:
            matched_merged = False
            for prev in merged:
                # Time gap between end of prev observation and start of trk
                gap = trk.start_frame - prev.last_observed_frame
                if 1 <= gap <= self.stitch_max_gap:
                    # Extrapolate prev pos to trk.start_frame
                    prev_last_pos = prev.observations[prev.last_observed_frame][0]
                    prev_v = prev.get_velocity()
                    extrapolated_pos = prev_last_pos + prev_v * gap

                    trk_first_pos = trk.observations[trk.start_frame][0]
                    spatial_dist = np.linalg.norm(extrapolated_pos - trk_first_pos)

                    # Dynamic allowable distance based on gap
                    allowed_dist = max(self.stitch_max_dist, 10.0 * gap)
                    if spatial_dist <= allowed_dist:
                        # Stitch trk into prev
                        for f_idx, obs in trk.observations.items():
                            prev.observations[f_idx] = obs
                        prev.hits += trk.hits
                        prev.last_frame = max(prev.last_frame, trk.last_frame)
                        prev.last_observed_frame = max(prev.last_observed_frame, trk.last_observed_frame)
                        prev.score = max(prev.score, trk.score)
                        prev.is_confirmed = prev.is_confirmed or trk.is_confirmed
                        matched_merged = True
                        break

            if not matched_merged:
                merged.append(trk)

        return merged

    def smooth_and_infill(
        self,
        tracks: List[PointKalmanTrack],
        num_frames: int,
    ) -> List[List[Dict]]:
        """
        Produce smoothed frame-by-frame outputs:
        1. Prefix recovery: for confirmed tracks (hits >= min_hits), output warmup frames.
        2. Infill: for confirmed stable tracks, fill small gaps (<= max_infill_gap).
        """
        # frame_outputs[f] = list of detections: {"pos": (x, y), "score": s, "track_id": id}
        frame_outputs: List[List[Dict]] = [[] for _ in range(num_frames)]

        for trk in tracks:
            # Check confirmation status
            is_confirmed = trk.is_confirmed or (
                trk.hits >= self.min_track_hits and trk.score >= self.min_track_score
            )
            if not is_confirmed:
                continue

            # Frame indices where target was observed
            obs_frames = sorted(trk.observations.keys())
            if not obs_frames:
                continue

            # Determine whether this track qualifies for infill
            can_infill = trk.hits >= self.min_hits_for_infill

            # 1. Output all direct observations (including warmup prefix frames)
            for f_idx in obs_frames:
                if 0 <= f_idx < num_frames:
                    pos, sc, is_meas = trk.observations[f_idx]
                    frame_outputs[f_idx].append({
                        "pos": pos,
                        "score": sc,
                        "track_id": trk.track_id,
                        "infilled": not is_meas,
                    })

            # 2. Infill internal gaps if qualified
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
                        # Linear interpolation between p1 and p2
                        for step, missing_f in enumerate(range(f1 + 1, f2), start=1):
                            alpha = step / gap
                            interp_pos = (1.0 - alpha) * p1 + alpha * p2
                            interp_score = (1.0 - alpha) * s1 + alpha * s2
                            if 0 <= missing_f < num_frames:
                                frame_outputs[missing_f].append({
                                    "pos": interp_pos,
                                    "score": interp_score,
                                    "track_id": trk.track_id,
                                    "infilled": True,
                                })

        return frame_outputs


# ==============================================================================
# 4. Evaluation Helpers
# ==============================================================================

def filter_dense_clutter_clusters(
    pts: np.ndarray,
    scs: np.ndarray,
    cluster_radius: float = 25.0,
    max_neighbors: int = 2,
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

    r_ind, col_ind = linear_sum_assignment(dists)
    matched_gt = set()
    matched_pred = set()

    for r, c in zip(r_ind, col_ind):
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
# 5. Full Sequence Evaluation Engine
# ==============================================================================

def evaluate_sequence_bidirectional(
    records: List[Dict],
    dist_thresh: float = 8.0,
    th_base: float = 0.22,
    th_salvage: float = 0.06,
    th_ground: float = 0.32,
    sky_ratio: float = 0.60,
    img_h: int = 640,
    tracker_config: Optional[Dict] = None,
    smoother_config: Optional[Dict] = None,
) -> Dict[str, Dict[str, float]]:
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
            "min_hits_for_infill": 4,
            "max_infill_gap": 3,
            "min_track_hits": 3,
            "min_track_score": 0.08,
            "instant_conf": 0.25,
        }

    records_sorted = sorted(records, key=lambda r: natural_sort_key(r["im_name"]))
    num_frames = len(records_sorted)

    stats = {
        "baseline": {"tp": 0, "fp": 0, "gt": 0},
        "online_fusion": {"tp": 0, "fp": 0, "gt": 0},
        "bidirectional": {"tp": 0, "fp": 0, "gt": 0},
    }

    online_tracker = OnlineAdaptiveTracker(**tracker_config)
    smoother = BidirectionalTemporalSmoother(**smoother_config)
    sky_y_boundary = img_h * sky_ratio

    # Pass 1: Run online tracking across all frames
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

        online_tracker.step(f_idx, high_pts, high_scs, salvage_pts, salvage_scs)

    all_tracks = online_tracker.finalize()

    # Pass 2: Tracklet stitching & Bidirectional Infill
    stitched_tracks = smoother.stitch_tracklets(all_tracks)
    bidi_frame_dets = smoother.smooth_and_infill(stitched_tracks, num_frames)

    # Pass 3: Evaluate metrics
    for f_idx, r in enumerate(records_sorted):
        gt_pts = np.asarray(r["gt_pts"], dtype=np.float32)
        pred_pts = np.asarray(r["pred_points"], dtype=np.float32)
        pred_scs = np.asarray(r["pred_scores"], dtype=np.float32)
        n_gt = len(gt_pts)

        for k in stats:
            stats[k]["gt"] += n_gt

        # Mode 1: Single-frame Baseline at th_base
        base_mask = pred_scs >= th_base
        pts_base = pred_pts[base_mask]
        tp1, fp1, _ = match_predictions_to_gt(gt_pts, pts_base, dist_thresh)
        stats["baseline"]["tp"] += tp1
        stats["baseline"]["fp"] += fp1

        # Mode 2: Online Fusion (raw output from online confirmed tracks)
        # Re-derive online output for this frame
        online_pts_list = []
        for t in all_tracks:
            if f_idx in t.observations:
                is_conf = t.is_confirmed or (t.hits >= tracker_config["min_hits"] and t.score >= tracker_config["min_track_score"])
                if is_conf:
                    online_pts_list.append(t.observations[f_idx][0])
        pts_online = np.array(online_pts_list, dtype=np.float32) if len(online_pts_list) > 0 else np.zeros((0, 2), dtype=np.float32)
        tp2, fp2, _ = match_predictions_to_gt(gt_pts, pts_online, dist_thresh)
        stats["online_fusion"]["tp"] += tp2
        stats["online_fusion"]["fp"] += fp2

        # Mode 3: Bidirectional Smoothed & Infilled Output
        bidi_dets = bidi_frame_dets[f_idx]
        pts_bidi = np.array([d["pos"] for d in bidi_dets], dtype=np.float32) if len(bidi_dets) > 0 else np.zeros((0, 2), dtype=np.float32)
        tp3, fp3, _ = match_predictions_to_gt(gt_pts, pts_bidi, dist_thresh)
        stats["bidirectional"]["tp"] += tp3
        stats["bidirectional"]["fp"] += fp3

    res = {}
    for k in stats:
        res[k] = calc_metrics(stats[k]["tp"], stats[k]["fp"], stats[k]["gt"])
    return res


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Bidirectional Spatio-Temporal Smoothed Track Fusion")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial22_cache.pkl",
        help="Path to Trial 22 inference cache",
    )
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="Evaluation tolerance (default: 8.0px)")
    parser.add_argument("--th-base", type=float, default=0.22, help="Base detection threshold (default: 0.22)")
    parser.add_argument("--th-salvage", type=float, default=0.06, help="Weak pulse salvage threshold in sky (default: 0.06)")
    parser.add_argument("--th-ground", type=float, default=0.32, help="Clutter suppression threshold on ground (default: 0.32)")
    parser.add_argument("--sky-ratio", type=float, default=0.60, help="Sky partition height ratio (default: 0.60)")
    parser.add_argument("--min-hits", type=int, default=3, help="Tracker min hits for confirmation (default: 3)")
    parser.add_argument("--max-age", type=int, default=3, help="Tracker max dead age (default: 3)")
    parser.add_argument("--match-dist", type=float, default=12.0, help="Base gating association distance (default: 12.0px)")
    parser.add_argument("--max-match-dist", type=float, default=18.0, help="Max adaptive maneuver distance (default: 18.0px)")
    parser.add_argument("--instant-conf", type=float, default=0.25, help="Zero-latency instant output threshold (default: 0.25)")
    parser.add_argument("--min-disp", type=float, default=2.5, help="Min displacement on ground clutter (default: 2.5px)")
    parser.add_argument("--min-track-score", type=float, default=0.08, help="Min track score (default: 0.08)")
    parser.add_argument("--stitch-gap", type=int, default=4, help="Max frame gap for tracklet stitching (default: 4)")
    parser.add_argument("--infill-gap", type=int, default=3, help="Max internal gap for infill (default: 3)")
    parser.add_argument("--min-hits-infill", type=int, default=4, help="Min track hits to qualify for infill (default: 4)")
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
        "min_track_score": args.min_track_score,
        "instant_conf": args.instant_conf,
        "min_displacement": args.min_disp,
        "sky_ratio": args.sky_ratio,
        "img_h": 640,
    }

    smoother_config = {
        "stitch_max_gap": args.stitch_gap,
        "stitch_max_dist": 25.0,
        "min_hits_for_infill": args.min_hits_infill,
        "max_infill_gap": args.infill_gap,
        "min_track_hits": args.min_hits,
        "min_track_score": args.min_track_score,
        "instant_conf": args.instant_conf,
    }

    grand_stats = {
        "baseline": {"tp": 0, "fp": 0, "gt": 0},
        "online_fusion": {"tp": 0, "fp": 0, "gt": 0},
        "bidirectional": {"tp": 0, "fp": 0, "gt": 0},
    }

    print("=" * 120)
    print(f"{'Sequence Name':<28} | {'Mode':<22} | {'TP / GT':<14} | {'FP':<6} | {'Recall':<8} | {'Prec':<8} | {'F1-Score':<8}")
    print("=" * 120)

    all_keys = sorted(seq_records.keys())
    for seq_name in all_keys:
        if filter_seqs and not any(f in seq_name for f in filter_seqs):
            continue

        recs = seq_records[seq_name]
        res = evaluate_sequence_bidirectional(
            records=recs,
            dist_thresh=args.dist_thresh,
            th_base=args.th_base,
            th_salvage=args.th_salvage,
            th_ground=args.th_ground,
            sky_ratio=args.sky_ratio,
            img_h=640,
            tracker_config=tracker_config,
            smoother_config=smoother_config,
        )

        for k in grand_stats:
            grand_stats[k]["tp"] += int(res[k]["tp"])
            grand_stats[k]["fp"] += int(res[k]["fp"])
            grand_stats[k]["gt"] += int(res[k]["gt"])

        m_base = res["baseline"]
        m_onl = res["online_fusion"]
        m_bidi = res["bidirectional"]

        print(f"{seq_name:<28} | {'1. Single Base':<22} | {int(m_base['tp']):>5} / {int(m_base['gt']):<6} | {int(m_base['fp']):<6} | {m_base['recall']:>6.2f}% | {m_base['precision']:>6.2f}% | {m_base['f1']:>6.4f}")
        print(f"{'':<28} | {'2. Online Fusion':<22} | {int(m_onl['tp']):>5} / {int(m_onl['gt']):<6} | {int(m_onl['fp']):<6} | {m_onl['recall']:>6.2f}% | {m_onl['precision']:>6.2f}% | {m_onl['f1']:>6.4f}")
        print(f"{'':<28} | {colorstr('bold', colorstr('green', '3. Bidirectional SOTA')):<31} | {int(m_bidi['tp']):>5} / {int(m_bidi['gt']):<6} | {int(m_bidi['fp']):<6} | {m_bidi['recall']:>6.2f}% | {m_bidi['precision']:>6.2f}% | {m_bidi['f1']:>6.4f}")
        print("-" * 120)

    print("=" * 120)
    print(colorstr("bold", f"GRAND OVERALL RESULTS ACROSS ALL SEQUENCES (Distance <= {args.dist_thresh:.1f}px)"))
    print("=" * 120)

    grand_metrics = {}
    for k in grand_stats:
        grand_metrics[k] = calc_metrics(grand_stats[k]["tp"], grand_stats[k]["fp"], grand_stats[k]["gt"])

    for mode_name, key in [
        (f"1. Single-Frame Baseline (th={args.th_base})", "baseline"),
        ("2. Online Adaptive Fusion (Previous SOTA)", "online_fusion"),
        ("3. 🔥 Bidirectional Smoothed & Infilled Fusion", "bidirectional"),
    ]:
        gm = grand_metrics[key]
        far = gm["fp"] / max(1, len(records))
        line = f"{mode_name:<46} | TP: {gm['tp']:>5}/{gm['gt']:<5} | FP: {gm['fp']:<6} | Recall: {gm['recall']:>6.2f}% | Prec: {gm['precision']:>6.2f}% | F1: {gm['f1']:>6.4f} | FAR: {far:.4f}/frame"
        if key == "bidirectional":
            print(colorstr("bold", colorstr("green", line)))
        else:
            print(line)
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
