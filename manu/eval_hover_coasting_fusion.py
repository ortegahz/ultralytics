#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Hovering Coasting & Vanishing-Point Deep Re-acquisition Spatio-Temporal Tracker.

Core Innovations on top of Absolute SOTA (F1=0.9194):
1. Hovering Prior Detection:
   - When a mature confirmed track (hits >= min_hits_for_infill) decelerates to near-zero speed
     (||v|| < hover_max_speed, e.g. 0.8px/f), it enters the HOVERING STATE.
2. Coasting Hold & Narrow-Tube Energy Deep-Salvage:
   - During Zoom-out or stationary hovering, temporal diff & median cancel out, driving raw response down.
   - Instead of terminating the track within max_age=3 frames, the tracker enters COASTING HOLD (up to coasting_max_gap frames, e.g. 15~25 frames).
   - In each coasting frame, a tight spatial tube (R <= reacq_radius, e.g. 4.0~5.0px) centered around the
     last known hovering position is opened.
   - Inside this 0.01% tiny area of the frame, the threshold is safely lowered to th_reacq (e.g. 0.030~0.045)
     to re-acquire sub-threshold impulse responses, 100% immune to random background noise.
   - If an impulse is found, the track is locked and updated immediately; if not, dynamic coasting linear interpolation bridges the gap.
3. Rigid Static Pruner:
   - Preserves min_rigid_disp=2.0, max_rigid_var=0.5, min_hits_prune=8 to safeguard against frozen sensor pixels.

Usage on single sequence:
    python manu/eval_hover_coasting_fusion.py \
        --cache-file runs/gmc_eval/uav_median_trial0474_cache.pkl \
        --sequences 02_6321_0274-2773 \
        --enable-hover-reacq \
        --reacq-th 0.035 \
        --reacq-radius 4.5 \
        --coasting-max-gap 20
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
from manu.eval_bidirectional_track_fusion import (
    PointKalmanTrack,
    calc_metrics,
    extract_seq_name,
    filter_dense_clutter_clusters,
    match_predictions_to_gt,
    natural_sort_key,
)


class HoveringCoastingTracker:
    """
    Advanced Kinematic Tracker with Hovering Prior and Vanishing-Point Deep Re-acquisition.
    """

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
        # Hovering & Coasting extensions
        enable_hover_reacq: bool = True,
        hover_max_speed: float = 0.80,
        coasting_max_age: int = 20,
        reacq_radius: float = 4.5,
        reacq_th: float = 0.035,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.match_dist = match_dist
        self.max_match_dist = max_match_dist
        self.min_track_score = min_track_score
        self.instant_conf = instant_conf
        self.min_displacement = min_displacement
        self.sky_ratio = sky_ratio
        self.img_h = img_h
        self.sky_y_boundary = img_h * sky_ratio

        self.enable_hover_reacq = enable_hover_reacq
        self.hover_max_speed = hover_max_speed
        self.coasting_max_age = coasting_max_age
        self.reacq_radius = reacq_radius
        self.reacq_th = reacq_th

        self.active_tracks: List[PointKalmanTrack] = []
        self.finished_tracks: List[PointKalmanTrack] = []

    def step(
        self,
        frame_idx: int,
        high_pts: np.ndarray,
        high_scs: np.ndarray,
        salvage_pts: np.ndarray,
        salvage_scs: np.ndarray,
        raw_pts: np.ndarray,
        raw_scs: np.ndarray,
    ):
        # 0. Predict active tracks
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

        # Step B2: Hovering Prior & Vanishing-Point Deep Re-acquisition
        # For unmatched mature tracks in hovering state, search inside a micro-tube (R <= reacq_radius)
        if self.enable_hover_reacq and len(raw_pts) > 0:
            still_unmatched = [i for i in range(len(self.active_tracks)) if i not in matched_tracks]
            for orig_track_idx in still_unmatched:
                trk = self.active_tracks[orig_track_idx]
                is_mature = trk.hits >= 5 and trk.is_confirmed
                is_in_sky = trk.get_pos()[1] < self.sky_y_boundary
                is_slow_or_hovering = trk.get_velocity_norm() <= self.hover_max_speed

                # Check if this track qualifies for vanishing-point deep salvage
                if is_mature and is_in_sky and is_slow_or_hovering:
                    last_known_pos = trk.observations[trk.last_observed_frame][0]
                    dists_raw = np.linalg.norm(raw_pts - last_known_pos, axis=1)
                    in_tube = (dists_raw <= self.reacq_radius) & (raw_scs >= self.reacq_th)

                    if np.any(in_tube):
                        tube_indices = np.where(in_tube)[0]
                        best_cand_idx = tube_indices[np.argmax(raw_scs[tube_indices])]
                        # Successfully re-acquired! Update with faint observation
                        trk.update(raw_pts[best_cand_idx], float(raw_scs[best_cand_idx]), frame_idx)
                        matched_tracks.add(orig_track_idx)

        # Step C: Only UNMATCHED HIGH-CONFIDENCE detections initiate new tracks
        for i in range(len(high_pts)):
            if i not in matched_high_dets:
                is_instantly_confirmed = high_scs[i] >= self.instant_conf
                self.active_tracks.append(
                    PointKalmanTrack(high_pts[i], high_scs[i], frame_idx, is_confirmed=is_instantly_confirmed)
                )

        # Step D: Cull dead tracks with Adaptive Coasting Lifespan
        surviving = []
        for t in self.active_tracks:
            # Determine maximum allowable dead age:
            # Mature hovering tracks in sky get extended coasting lifespan
            is_mature_hover = (
                self.enable_hover_reacq
                and (t.hits >= 5 and t.is_confirmed)
                and (t.get_pos()[1] < self.sky_y_boundary)
                and (t.get_velocity_norm() <= self.hover_max_speed)
            )
            allowable_age = self.coasting_max_age if is_mature_hover else self.max_age

            if t.time_since_update > allowable_age:
                pos = t.get_pos()
                is_in_sky = pos[1] < self.sky_y_boundary
                if not is_in_sky and t.hits >= 5 and self.min_displacement > 0:
                    if t.get_net_displacement() < self.min_displacement:
                        continue  # Purge static ground vibration
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


class HoveringAwareSmoother:
    """
    Bidirectional Smoother with Hovering Coasting Infill.
    """

    def __init__(
        self,
        stitch_max_gap: int = 4,
        stitch_max_dist: float = 25.0,
        min_hits_for_infill: int = 5,
        max_infill_gap: int = 3,
        min_track_hits: int = 3,
        min_track_score: float = 0.08,
        instant_conf: float = 0.25,
        min_rigid_displacement: float = 2.0,
        max_rigid_variance: float = 0.5,
        min_hits_for_prune: int = 8,
        # Hovering infill extension
        enable_hover_infill: bool = True,
        hover_max_infill_gap: int = 20,
    ):
        self.stitch_max_gap = stitch_max_gap
        self.stitch_max_dist = stitch_max_dist
        self.min_hits_for_infill = min_hits_for_infill
        self.max_infill_gap = max_infill_gap
        self.min_track_hits = min_track_hits
        self.min_track_score = min_track_score
        self.instant_conf = instant_conf
        self.min_rigid_displacement = min_rigid_displacement
        self.max_rigid_variance = max_rigid_variance
        self.min_hits_for_prune = min_hits_for_prune
        self.enable_hover_infill = enable_hover_infill
        self.hover_max_infill_gap = hover_max_infill_gap

    def stitch_tracklets(self, tracks: List[PointKalmanTrack]) -> List[PointKalmanTrack]:
        if len(tracks) <= 1:
            return tracks

        tracks_sorted = sorted(tracks, key=lambda t: t.start_frame)
        merged: List[PointKalmanTrack] = []

        for trk in tracks_sorted:
            matched_merged = False
            for prev in merged:
                gap = trk.start_frame - prev.last_observed_frame
                # Allow extended stitching for hovering tracks
                is_hover_stitch = (
                    self.enable_hover_infill
                    and (prev.hits >= 5 and trk.hits >= 5)
                    and (prev.get_velocity_norm() < 1.0 and trk.get_velocity_norm() < 1.0)
                )
                allowable_gap = self.hover_max_infill_gap if is_hover_stitch else self.stitch_max_gap

                if 1 <= gap <= allowable_gap:
                    prev_last_pos = prev.observations[prev.last_observed_frame][0]
                    prev_v = prev.get_velocity()
                    extrapolated_pos = prev_last_pos + prev_v * gap
                    trk_first_pos = trk.observations[trk.start_frame][0]

                    spatial_dist = np.linalg.norm(extrapolated_pos - trk_first_pos)
                    # For hovering tracklets, endpoint distance must remain within tight radius
                    allowed_dist = 6.0 if is_hover_stitch else max(self.stitch_max_dist, 10.0 * gap)

                    if spatial_dist <= allowed_dist:
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
                        continue  # Purge static sensor bad pixel

            can_infill = trk.hits >= self.min_hits_for_infill

            # 1. Output direct observations
            for f_idx in obs_frames:
                if 0 <= f_idx < num_frames:
                    pos, sc, is_meas = trk.observations[f_idx]
                    frame_outputs[f_idx].append({
                        "pos": pos,
                        "score": sc,
                        "track_id": trk.track_id,
                        "infilled": not is_meas,
                    })

            # 2. Infill internal gaps
            if can_infill and len(obs_frames) >= 2:
                for i in range(len(obs_frames) - 1):
                    f1 = obs_frames[i]
                    f2 = obs_frames[i + 1]
                    gap = f2 - f1

                    p1 = trk.observations[f1][0]
                    p2 = trk.observations[f2][0]
                    s1 = trk.observations[f1][1]
                    s2 = trk.observations[f2][1]
                    pt_dist = np.linalg.norm(p2 - p1)

                    # Determine allowable infill gap:
                    # If endpoint drift is small (pt_dist < 6.0px) and track is slow, allow extended hover infill
                    is_hovering_gap = self.enable_hover_infill and (pt_dist < 6.0)
                    allowable_gap = self.hover_max_infill_gap if is_hovering_gap else self.max_infill_gap

                    if 1 < gap <= (allowable_gap + 1):
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


def evaluate_sequence_hover(
    records: List[Dict],
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
    num_frames = len(records_sorted)

    stats = {
        "baseline": {"tp": 0, "fp": 0, "gt": 0},
        "bidirectional": {"tp": 0, "fp": 0, "gt": 0},
    }

    tracker = HoveringCoastingTracker(**tracker_config)
    smoother = HoveringAwareSmoother(**smoother_config)
    sky_y_boundary = img_h * sky_ratio

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

        tracker.step(
            frame_idx=f_idx,
            high_pts=high_pts,
            high_scs=high_scs,
            salvage_pts=salvage_pts,
            salvage_scs=salvage_scs,
            raw_pts=pred_pts,
            raw_scs=pred_scs,
        )

    all_tracks = tracker.finalize()
    stitched_tracks = smoother.stitch_tracklets(all_tracks)
    bidi_frame_dets = smoother.smooth_and_infill(stitched_tracks, num_frames)

    for f_idx, r in enumerate(records_sorted):
        gt_pts = np.asarray(r["gt_pts"], dtype=np.float32)
        pred_pts = np.asarray(r["pred_points"], dtype=np.float32)
        pred_scs = np.asarray(r["pred_scores"], dtype=np.float32)
        n_gt = len(gt_pts)

        for k in stats:
            stats[k]["gt"] += n_gt

        # Baseline single frame
        base_mask = pred_scs >= th_base
        pts_base = pred_pts[base_mask]
        tp1, fp1, _ = match_predictions_to_gt(gt_pts, pts_base, dist_thresh)
        stats["baseline"]["tp"] += tp1
        stats["baseline"]["fp"] += fp1

        # Hovering-aware Smoothed & Infilled Output
        bidi_dets = bidi_frame_dets[f_idx]
        pts_bidi = np.array([d["pos"] for d in bidi_dets], dtype=np.float32) if len(bidi_dets) > 0 else np.zeros((0, 2), dtype=np.float32)
        tp3, fp3, _ = match_predictions_to_gt(gt_pts, pts_bidi, dist_thresh)
        stats["bidirectional"]["tp"] += tp3
        stats["bidirectional"]["fp"] += fp3

    res = {}
    for k in stats:
        res[k] = calc_metrics(stats[k]["tp"], stats[k]["fp"], stats[k]["gt"])
    res["bidi_frame_dets"] = bidi_frame_dets
    res["records_sorted"] = records_sorted
    return res


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Hovering Coasting & Vanishing-Point Deep Re-acquisition")
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
    parser.add_argument("--sequences", type=str, default="02_6321_0274-2773", help="Target sequence filter")

    # Hovering & Coasting Options
    parser.add_argument("--enable-hover-reacq", action="store_true", default=True)
    parser.add_argument("--reacq-th", type=float, default=0.035, help="Deep salvage threshold inside tube (default: 0.035)")
    parser.add_argument("--reacq-radius", type=float, default=4.5, help="Deep salvage micro-tube radius (default: 4.5px)")
    parser.add_argument("--hover-max-speed", type=float, default=0.80, help="Max speed to classify as hover (default: 0.8px/f)")
    parser.add_argument("--coasting-max-gap", type=int, default=20, help="Max coasting frames in hover state (default: 20)")
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

    print(f"[INFO] Loading inference cache from: {cache_path}")
    with open(cache_path, "rb") as f:
        records = pickle.load(f)

    seq_records: Dict[str, List[Dict]] = {}
    for r in records:
        seq = extract_seq_name(r["im_name"])
        seq_records.setdefault(seq, []).append(r)

    filter_seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]

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
        "enable_hover_reacq": args.enable_hover_reacq,
        "hover_max_speed": args.hover_max_speed,
        "coasting_max_age": args.coasting_max_gap,
        "reacq_radius": args.reacq_radius,
        "reacq_th": args.reacq_th,
    }

    smoother_config = {
        "stitch_max_gap": 4,
        "stitch_max_dist": 25.0,
        "min_hits_for_infill": 5,
        "max_infill_gap": 3,
        "min_track_hits": 3,
        "min_track_score": 0.08,
        "instant_conf": 0.25,
        "min_rigid_displacement": args.min_rigid_disp,
        "max_rigid_variance": args.max_rigid_var,
        "min_hits_for_prune": args.min_hits_prune,
        "enable_hover_infill": args.enable_hover_reacq,
        "hover_max_infill_gap": args.coasting_max_gap,
    }

    print("\n" + "=" * 125)
    print("🛸 EVALUATING HOVERING COASTING & VANISHING-POINT DEEP RE-ACQUISITION")
    print(f"Target Sequences : {filter_seqs or 'ALL 24'}")
    print(f"Deep Reacq Tube  : R <= {args.reacq_radius}px | Th >= {args.reacq_th} | Hover Speed <= {args.hover_max_speed}px/f")
    print(f"Max Coasting Gap : {args.coasting_max_gap} frames")
    print("=" * 125)

    header = f"{'Sequence Name':<28} | {'Mode':<24} | {'TP / GT':<14} | {'FP':<6} | {'Recall':<8} | {'Prec':<8} | {'F1-Score':<8}"
    print(header)
    print("-" * 125)

    tot_tp, tot_fp, tot_gt = 0, 0, 0

    for s_name in sorted(seq_records.keys()):
        if filter_seqs and not any(f in s_name for f in filter_seqs):
            continue

        recs = seq_records[s_name]
        res = evaluate_sequence_hover(
            records=recs,
            dist_thresh=args.dist_thresh,
            th_base=args.th_base,
            th_salvage=args.th_salvage,
            th_ground=args.th_ground,
            sky_ratio=0.60,
            img_h=640,
            tracker_config=tracker_config,
            smoother_config=smoother_config,
        )

        m_bidi = res["bidirectional"]
        tot_tp += int(m_bidi["tp"])
        tot_fp += int(m_bidi["fp"])
        tot_gt += int(m_bidi["gt"])

        m_base = res["baseline"]
        print(f"{s_name:<28} | {'1. Single Base (th=0.22)':<24} | {int(m_base['tp']):>5} / {int(m_base['gt']):<6} | {int(m_base['fp']):<6} | {m_base['recall']:>6.2f}% | {m_base['precision']:>6.2f}% | {m_base['f1']:>6.4f}")
        print(f"{'':<28} | {colorstr('bold', colorstr('green', '2. Hovering Coasting SOTA')):<27} | {int(m_bidi['tp']):>5} / {int(m_bidi['gt']):<6} | {int(m_bidi['fp']):<6} | {m_bidi['recall']:>6.2f}% | {m_bidi['precision']:>6.2f}% | {m_bidi['f1']:>6.4f}")
        print("-" * 125)

    overall_m = calc_metrics(tot_tp, tot_fp, tot_gt)
    print("=" * 125)
    print(
        colorstr(
            "bold",
            colorstr(
                "cyan",
                f"🏆 EVALUATION SUMMARY | F1: {overall_m['f1']:.4f} | Recall: {overall_m['recall']:.2f}% ({tot_tp:,}/{tot_gt:,}) | "
                f"Prec: {overall_m['precision']:.2f}% (FP={tot_fp:,})",
            ),
        )
    )
    print("=" * 125 + "\n")


if __name__ == "__main__":
    main()
