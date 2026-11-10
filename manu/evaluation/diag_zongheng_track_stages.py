#!/usr/bin/env python3
"""Stage-by-stage trace of the 龙泉山 tracker on one 纵横AI sequence.

Replicates Pass 1 (online tracker) and Pass 2 (smoother) with counters, so the exact stage that
drops the tracks is visible instead of inferred.
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
    build_smoother_config,
    build_tracker_config,
)

LONGQUAN_PARAMS_JSON = (
    Path("/tmp/pycharm_project_10ae9e2e")
    / "manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json"
)


def main() -> None:
    cache_path = Path(sys.argv[1])
    params = dict(json.loads(LONGQUAN_PARAMS_JSON.read_text(encoding="utf-8"))["params"])
    th_base = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20
    th_ground = float(sys.argv[3]) if len(sys.argv) > 3 else 0.20

    cache = pickle.loads(cache_path.read_bytes())
    meta = cache["meta"]
    frames = int(meta["frames"])
    scores_all = cache["scores"].astype(np.float32)
    points_all = cache["points"]
    row_frames = cache["frame_index"].tolist()

    by_frame = {}
    for position, stored_frame in enumerate(row_frames):
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

    total = sum(len(s) for s in cand_scores)
    print(f"=== {cache_path.stem} frames={frames} candidate_rows={total} ===")

    # How stable is the candidate position frame to frame? A tracker needs a locally continuous target.
    tops = [p[0] for p in cand_points if len(p)]
    if len(tops) > 3:
        jumps = [float(np.linalg.norm(tops[i + 1] - tops[i])) for i in range(len(tops) - 1)]
        print(f"  per-frame top-candidate jump: p50={np.percentile(jumps,50):.1f}px "
              f"p90={np.percentile(jumps,90):.1f}px  (match_dist=16px, max_match_dist=24px)")
        ys = [float(p[0][1]) for p in cand_points if len(p)]
        print(f"  top-candidate y: p10={np.percentile(ys,10):.1f} p50={np.percentile(ys,50):.1f} "
              f"p90={np.percentile(ys,90):.1f}  (native h={meta['native_size'][1]}, "
              f"sky_ratio={params['sky_ratio']} -> sky boundary y={512*float(params['sky_ratio']):.0f})")

    tracker_config = build_tracker_config(params, 512)
    smoother_config = build_smoother_config(params, 512)

    tracker = OnlineAdaptiveTracker(**tracker_config)
    sky_y_boundary = 512 * float(params["sky_ratio"])
    n_high = n_salvage = 0
    for f_idx in range(frames):
        pred_pts = cand_points[f_idx]
        pred_scs = cand_scores[f_idx]
        if len(pred_pts):
            is_sky = pred_pts[:, 1] < sky_y_boundary
            high_mask = (is_sky & (pred_scs >= th_base)) | ((~is_sky) & (pred_scs >= th_ground))
            salvage_mask = is_sky & (pred_scs >= 0.06) & (pred_scs < th_base)
            high_pts, high_scs = pred_pts[high_mask], pred_scs[high_mask]
            salvage_pts, salvage_scs = pred_pts[salvage_mask], pred_scs[salvage_mask]
        else:
            high_pts = np.zeros((0, 2), np.float32)
            high_scs = np.zeros(0, np.float32)
            salvage_pts = np.zeros((0, 2), np.float32)
            salvage_scs = np.zeros(0, np.float32)
        n_high += len(high_pts)
        n_salvage += len(salvage_pts)
        tracker.step(f_idx, high_pts, high_scs, salvage_pts, salvage_scs)

    print(f"  pass1: high_pts={n_high} salvage_pts={n_salvage}")
    tracks = tracker.finalize()
    print(f"  pass1 finished_tracks={len(tracks)}")
    if tracks:
        hits = sorted(t.hits for t in tracks)
        print(f"    hits: min={hits[0]} p50={hits[len(hits)//2]} max={hits[-1]}")
        confirmed = sum(1 for t in tracks if t.is_confirmed or (t.hits >= tracker_config["min_hits"] and t.score >= tracker_config["min_track_score"]))
        print(f"    confirmed={confirmed}")
        print(f"    score: min={min(t.score for t in tracks):.3f} max={max(t.score for t in tracks):.3f}")
        print(f"    min_track_score={tracker_config['min_track_score']} min_hits={tracker_config['min_hits']}")

    smoother = BidirectionalTemporalSmoother(**smoother_config)
    stitched = smoother.stitch_tracklets(tracks)
    print(f"  pass2 stitched_tracks={len(stitched)}")
    out = smoother.smooth_and_infill(stitched, frames)
    print(f"  pass2 frames_with_dets={sum(1 for d in out if d)} detections={sum(len(d) for d in out)}")


if __name__ == "__main__":
    main()
