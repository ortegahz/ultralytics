#!/usr/bin/env python3
"""Attribute WHERE the tracked tier loses coverage against the single-frame tier.

For one sequence it compares, frame by frame:
  * the merged candidate pool (what the single-frame tier draws),
  * the tracker's own output,
  * the confirmation gate's output,
and attributes each lost frame to one of three causes:
  1. ``no_track``      -- the candidate never became a confirmed track (min_hits / instant_conf).
  2. ``static_prune``  -- the track was killed as a frozen static artefact (rigid static pruner).
  3. ``confirm_gate``  -- the track existed but never reached confirm_consec consecutive frames.

That last one is the expected, intended behaviour (a one-frame flash is a false alarm); the middle
one is the one that can also kill a genuinely hovering target.
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, "/tmp/pycharm_project_10ae9e2e")

from manu.evaluation.eval_bidirectional_track_fusion import (  # noqa: E402
    BidirectionalTemporalSmoother,
    OnlineAdaptiveTracker,
)
from manu.evaluation.system_sota_longquan import (  # noqa: E402
    apply_consecutive_confirmation,
    build_smoother_config,
    build_tracker_config,
)

PARAMS_JSON = (
    Path("/tmp/pycharm_project_10ae9e2e")
    / "manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json"
)


def main() -> None:
    cache_path = Path(sys.argv[1])
    th = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20
    params = dict(json.loads(PARAMS_JSON.read_text(encoding="utf-8"))["params"])

    cache = pickle.loads(cache_path.read_bytes())
    meta = cache["meta"]
    frames = int(meta["frames"])
    scores_all = cache["scores"].astype(np.float32)
    points_all = cache["points"]
    by_frame: dict[int, list[int]] = {}
    for position, stored_frame in enumerate(cache["frame_index"].tolist()):
        by_frame.setdefault(int(stored_frame), []).append(position)

    cand_points, cand_scores = [], []
    for frame in range(frames):
        positions = by_frame.get(frame, [])
        if positions:
            cand_points.append(points_all[positions])
            cand_scores.append(scores_all[positions])
        else:
            cand_points.append(np.zeros((0, 2), np.float32))
            cand_scores.append(np.zeros(0, np.float32))

    cand_frames = {i for i in range(frames) if len(cand_scores[i])}
    print(f"=== {cache_path.stem} frames={frames} candidate_frames={len(cand_frames)} ({len(cand_frames)/max(frames,1):.1%}) ===")

    tracker_config = build_tracker_config(params, 512)
    smoother_config = build_smoother_config(params, 512)
    tracker = OnlineAdaptiveTracker(**tracker_config)
    sky_y_boundary = 512 * float(params["sky_ratio"])

    for f_idx in range(frames):
        pred_pts, pred_scs = cand_points[f_idx], cand_scores[f_idx]
        if len(pred_pts):
            is_sky = pred_pts[:, 1] < sky_y_boundary
            high_mask = (is_sky & (pred_scs >= th)) | ((~is_sky) & (pred_scs >= th))
            salvage_mask = is_sky & (pred_scs >= 0.06) & (pred_scs < th)
            high_pts, high_scs = pred_pts[high_mask], pred_scs[high_mask]
            salvage_pts, salvage_scs = pred_pts[salvage_mask], pred_scs[salvage_mask]
        else:
            high_pts = np.zeros((0, 2), np.float32)
            high_scs = np.zeros(0, np.float32)
            salvage_pts = np.zeros((0, 2), np.float32)
            salvage_scs = np.zeros(0, np.float32)
        tracker.step(f_idx, high_pts, high_scs, salvage_pts, salvage_scs)

    all_tracks = tracker.finalize()
    smoother = BidirectionalTemporalSmoother(**smoother_config)
    stitched = smoother.stitch_tracklets(all_tracks)
    tracker_out = smoother.smooth_and_infill(stitched, frames)
    confirmed_out = apply_consecutive_confirmation(
        tracker_out,
        confirm_consec=int(params["confirm_consec"]),
        confirm_gap_tol=int(params["confirm_gap_tol"]),
        confirm_trim_prefix=int(params["confirm_trim_prefix"]),
    )

    tracker_frames = {i for i in range(frames) if tracker_out[i]}
    confirmed_frames = {i for i in range(frames) if confirmed_out[i]}

    # How many tracks were killed by the rigid static pruner?
    pruned = 0
    for trk in all_tracks:
        obs = sorted(trk.observations)
        if len(obs) < smoother_config["min_hits_for_prune"]:
            continue
        pts = np.array([trk.observations[f][0] for f in obs], dtype=np.float32)
        net_disp = float(np.linalg.norm(pts[-1] - pts[0]))
        pos_var = float(np.var(pts[:, 0]) + np.var(pts[:, 1]))
        eff = smoother_config["min_rigid_displacement"]
        if smoother_config["min_rigid_disp_sky"] is not None and pts[:, 1].mean() < sky_y_boundary:
            eff = smoother_config["min_rigid_disp_sky"]
        if eff > 0 and net_disp < eff and pos_var < smoother_config["max_rigid_variance"]:
            pruned += 1

    lost_to_track = cand_frames - tracker_frames
    lost_to_confirm = tracker_frames - confirmed_frames
    print(f"  tracks created          = {len(all_tracks)}  (stitched {len(stitched)})")
    print(f"  tracks killed by rigid static pruner = {pruned}")
    print(f"  candidate frames        = {len(cand_frames)}")
    print(f"  after tracker+smoother  = {len(tracker_frames)}   (lost {len(lost_to_track)})")
    print(f"  after confirm_consec    = {len(confirmed_frames)}   (lost {len(lost_to_confirm)})")
    if cand_frames:
        print(f"  net coverage            = {len(confirmed_frames)/len(cand_frames):.1%} of the candidate frames survive")
        print(f"  attribution: no_track {len(lost_to_track)} / confirm_gate {len(lost_to_confirm)}")


if __name__ == "__main__":
    main()
