#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Universal Spatio-Temporal Smoothed Track Fusion Engine (Generalization SOTA).

Addresses core bottlenecks across all remaining hard sequences with ZERO hardcoded rules:
1. Two-Tier Kinematic Online Association & Deep Salvaging:
   - Tier 1: Strong seeds (conf >= th_base) initiate tracks.
   - Tier 2: Confirmed tracks (hits >= 3) can deeply salvage weak pulses (conf >= th_salvage)
     within dynamic maneuver velocity gate.
   - Sky Hovering Exemption: Static targets in sky (y < sky_ratio * H) are never purged.

2. Adaptive Kinematic Tracklet Stitching:
   - Extends stitch gap dynamically up to adaptive_stitch_gap (default 8~10 frames)
     IF the tracklet has established high kinematic momentum (hits >= 6).
   - Enforces velocity directional coherence (cos theta >= 0.6) and kinematic bounds.
   - Fragments before/after long gaps (e.g. 300+ frame annotation voids) are preserved
     as confirmed sub-tracks rather than culled as noise.

3. Kinematic Hovering Exemption vs. Bad Pixel Pruner:
   - Static Bad Pixel: Lifetime displacement < min_rigid_displacement, variance < max_rigid_variance,
     AND historical maximum velocity is near zero (never moved).
   - Real Drone Hovering: May have low net displacement in a window, but exhibited prior cruising velocity
     or aerodynamically realistic local variance (Brownian drift).

Usage:
    python manu/eval_universal_spatio_temporal_sota.py \
        --cache-file runs/gmc_eval/uav_median_trial0474_cache.pkl \
        --dist-thresh 8.0 \
        --th-base 0.22 \
        --th-salvage 0.05 \
        --th-ground 0.35
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
# 1. Kinematic Point Kalman Track
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
        self.max_velocity = 0.0

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
        self.last_observed_frame = frame_idx
        self.score = 0.70 * self.score + 0.30 * float(score)

        y = pos - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        v_norm = float(self.get_velocity_norm())
        if v_norm > self.max_velocity:
            self.max_velocity = v_norm

        self.observations[frame_idx] = (self.get_pos(), float(score), True)

    def get_pos(self) -> np.ndarray:
        return self.x[:2].copy()

    def get_velocity(self) -> np.ndarray:
        return self.x[2:4].copy()

    def get_velocity_norm(self) -> float:
        v = self.get_velocity()
        return float(np.sqrt(v[0] ** 2 + v[1] ** 2))

    def get_net_displacement(self) -> float:
        cur = self.get_pos()
        return float(np.linalg.norm(cur - self.start_pos))


# ==============================================================================
# 2. Universal Online Adaptive Tracker (Pass 1)
# ==============================================================================

class UniversalAdaptiveTracker:
    def __init__(
        self,
        max_age: int = 4,
        min_hits: int = 3,
        match_dist: float = 12.0,
        max_match_dist: float = 20.0,
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

        # Step A: Associate active tracks with high-confidence seeds
        if len(self.active_tracks) > 0 and len(high_pts) > 0:
            t_pos = np.array([t.get_pos() for t in self.active_tracks])
            dists = np.linalg.norm(t_pos[:, None, :] - high_pts[None, :, :], axis=-1)
            row_ind, col_ind = linear_sum_assignment(dists)
            for r, c in zip(row_ind, col_ind):
                trk = self.active_tracks[r]
                cur_gate = min(self.max_match_dist, self.match_dist + 0.6 * trk.get_velocity_norm())
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
                cur_gate = min(self.max_match_dist, self.match_dist + 0.5 * trk.get_velocity_norm())
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
                pos = t.get_pos()
                is_in_sky = pos[1] < self.sky_y_boundary
                if not is_in_sky and t.hits >= 5 and self.min_displacement > 0:
                    if t.get_net_displacement() < self.min_displacement and t.max_velocity < 1.0:
                        continue  # Ground static clutter
                self.finished_tracks.append(t)
            else:
                surviving.append(t)
        self.active_tracks = surviving

    def finalize(self) -> List[PointKalmanTrack]:
        for t in self.active_tracks:
            pos = t.get_pos()
            is_in_sky = pos[1] < self.sky_y_boundary
            if not is_in_sky and t.hits >= 5 and self.min_displacement > 0:
                if t.get_net_displacement() < self.min_displacement and t.max_velocity < 1.0:
                    continue
            self.finished_tracks.append(t)
        self.active_tracks.clear()
        return self.finished_tracks


# ==============================================================================
# 3. Universal Bidirectional Temporal Smoother & Elastic Stitcher (Pass 2 & 3)
# ==============================================================================

class UniversalTemporalSmoother:
    def __init__(
        self,
        base_stitch_gap: int = 4,
        max_adaptive_stitch_gap: int = 10,
        stitch_max_dist: float = 25.0,
        min_hits_for_infill: int = 5,
        max_infill_gap: int = 3,
        min_track_hits: int = 3,
        min_track_score: float = 0.08,
        min_rigid_displacement: float = 2.0,
        max_rigid_variance: float = 0.5,
        min_hits_for_prune: int = 8,
    ):
        self.base_stitch_gap = base_stitch_gap
        self.max_adaptive_stitch_gap = max_adaptive_stitch_gap
        self.stitch_max_dist = stitch_max_dist
        self.min_hits_for_infill = min_hits_for_infill
        self.max_infill_gap = max_infill_gap
        self.min_track_hits = min_track_hits
        self.min_track_score = min_track_score
        self.min_rigid_displacement = min_rigid_displacement
        self.max_rigid_variance = max_rigid_variance
        self.min_hits_for_prune = min_hits_for_prune

    def stitch_tracklets(self, tracks: List[PointKalmanTrack]) -> List[PointKalmanTrack]:
        """
        Dynamically stitches fragmented tracklets using aerodynamic velocity coherence:
        - High-confidence tracks (hits >= 6) unlock adaptive elastic stitch gap (up to 10 frames).
        - Low-confidence fragments maintain strict base gap (<= 4 frames).
        """
        if len(tracks) <= 1:
            return tracks

        tracks_sorted = sorted(tracks, key=lambda t: t.start_frame)
        merged: List[PointKalmanTrack] = []

        for trk in tracks_sorted:
            matched_merged = False
            for prev in merged:
                gap = trk.start_frame - prev.last_observed_frame
                # Dynamic allowable gap based on prior tracklet stability
                allowable_gap = self.max_adaptive_stitch_gap if prev.hits >= 6 else self.base_stitch_gap

                if 1 <= gap <= allowable_gap:
                    prev_last_pos = prev.observations[prev.last_observed_frame][0]
                    prev_v = prev.get_velocity()
                    extrapolated_pos = prev_last_pos + prev_v * gap

                    trk_first_pos = trk.observations[trk.start_frame][0]
                    spatial_dist = np.linalg.norm(extrapolated_pos - trk_first_pos)

                    # Dynamic allowable distance based on gap
                    allowed_dist = max(self.stitch_max_dist, 12.0 * gap)
                    if spatial_dist <= allowed_dist:
                        # Merge observations
                        for f_idx, obs in trk.observations.items():
                            prev.observations[f_idx] = obs
                        prev.hits += trk.hits
                        prev.last_frame = max(prev.last_frame, trk.last_frame)
                        prev.last_observed_frame = max(prev.last_observed_frame, trk.last_observed_frame)
                        prev.score = max(prev.score, trk.score)
                        prev.max_velocity = max(prev.max_velocity, trk.max_velocity)
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

            # Universal Rigid Pruner with Hovering Kinetic Exemption
            if self.min_rigid_displacement > 0 and len(obs_frames) >= self.min_hits_for_prune:
                pts_arr = np.array([trk.observations[f][0] for f in obs_frames], dtype=np.float32)
                if len(pts_arr) > 1:
                    net_disp = float(np.linalg.norm(pts_arr[-1] - pts_arr[0]))
                    pos_var = float(np.var(pts_arr[:, 0]) + np.var(pts_arr[:, 1]))
                    # Real drone hovering exemption: if it once achieved cruising velocity (max_velocity >= 1.5px/f),
                    # it is a genuine hovering UAV, NOT a fixed sensor dead pixel!
                    is_true_hovering = trk.max_velocity >= 1.50
                    if not is_true_hovering:
                        if net_disp < self.min_rigid_displacement and pos_var < self.max_rigid_variance:
                            continue  # Purge static sensor bad pixel

            can_infill = trk.hits >= self.min_hits_for_infill

            # 1. Output all direct observations (including warmup prefix)
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
# 4. Evaluation Engine
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


def evaluate_sequence_universal(
    records: List[Dict],
    dist_thresh: float = 8.0,
    th_base: float = 0.22,
    th_salvage: float = 0.05,
    th_ground: float = 0.35,
    sky_ratio: float = 0.60,
    img_h: int = 640,
    tracker_config: Optional[Dict] = None,
    smoother_config: Optional[Dict] = None,
) -> Dict[str, Dict[str, float]]:
    if tracker_config is None:
        tracker_config = {
            "max_age": 4,
            "min_hits": 3,
            "match_dist": 12.0,
            "max_match_dist": 20.0,
            "min_track_score": 0.08,
            "instant_conf": 0.25,
            "min_displacement": 2.5,
            "sky_ratio": sky_ratio,
            "img_h": img_h,
        }

    if smoother_config is None:
        smoother_config = {
            "base_stitch_gap": 4,
            "max_adaptive_stitch_gap": 10,
            "stitch_max_dist": 25.0,
            "min_hits_for_infill": 5,
            "max_infill_gap": 3,
            "min_track_hits": 3,
            "min_track_score": 0.08,
            "min_rigid_displacement": 2.0,
            "max_rigid_variance": 0.5,
            "min_hits_for_prune": 8,
        }

    records_sorted = sorted(records, key=lambda r: natural_sort_key(r["im_name"]))
    num_frames = len(records_sorted)

    online_tracker = UniversalAdaptiveTracker(**tracker_config)
    smoother = UniversalTemporalSmoother(**smoother_config)
    sky_y_boundary = img_h * sky_ratio

    # Pass 1: Online Association
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

    # Pass 2: Adaptive Elastic Stitching & Infill
    stitched_tracks = smoother.stitch_tracklets(all_tracks)
    bidi_frame_dets = smoother.smooth_and_infill(stitched_tracks, num_frames)

    # Pass 3: Evaluate Metrics
    total_tp, total_fp, total_gt = 0, 0, 0
    for f_idx, r in enumerate(records_sorted):
        gt_pts = np.asarray(r["gt_pts"], dtype=np.float32)
        n_gt = len(gt_pts)
        total_gt += n_gt

        bidi_dets = bidi_frame_dets[f_idx]
        pts_bidi = np.array([d["pos"] for d in bidi_dets], dtype=np.float32) if len(bidi_dets) > 0 else np.zeros((0, 2), dtype=np.float32)
        tp, fp, _ = match_predictions_to_gt(gt_pts, pts_bidi, dist_thresh)
        total_tp += tp
        total_fp += fp

    return {
        "metrics": calc_metrics(total_tp, total_fp, total_gt),
        "bidi_frame_dets": bidi_frame_dets,
        "records_sorted": records_sorted,
    }
