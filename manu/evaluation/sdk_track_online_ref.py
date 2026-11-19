#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Reference wrapper: SDK candidate stream -> ONLINE fused track output.

This is the reference for the **online** half of the SDK's spatio-temporal
track fusion (the half an RT pipeline consumes frame by frame): the online
adaptive tracker (Pass 1) and, per frame, the confirmed tracks that were
OBSERVED in that frame.

It does NOT re-implement the tracker: it drives the production
:class:`manu.evaluation.eval_bidirectional_track_fusion.OnlineAdaptiveTracker`
with the production parameter mapping
(:func:`manu.evaluation.system_sota_longquan.build_tracker_config`), i.e. the
same code the offline reference runs.

THE ONE DIFFERENCE FROM THE OFFLINE REFERENCE
---------------------------------------------
The production reference reports the online fusion from ``finalize()``'s track
list, i.e. from ALL the tracks the sequence ever had.  The SDK's online view
reports the tracks that are ALIVE in that frame (the C++ ``OnlineFused()``
walks the live track list).  The two agree on every frame in which the track
is alive; they differ only on the frame where a track is culled (the ground
static-clutter filter drops it).  This script implements the SDK's semantics,
because that is what the RT pipeline has to match.

Input ``candidates.tsv`` (one frame per line)::

    <frame_idx> <n> [<x1> <y1> <score1> ...]

Output ``online.tsv`` (one frame per line)::

    <frame_idx> <n> [<track_id1> <x1> <y1> <score1> <infilled1> ...]

Usage::

    python sdk_track_online_ref.py --cands candidates.tsv \
        --params system_sota_longquan_trial0080_delivery_params.json --out online.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ULTRALYTICS_ROOT = "/media/manu/1TB-Volume/workspace/ultralytics"
if ULTRALYTICS_ROOT not in sys.path:
    sys.path.insert(0, ULTRALYTICS_ROOT)

from manu.evaluation.eval_bidirectional_track_fusion import (  # noqa: E402
    OnlineAdaptiveTracker,
    filter_dense_clutter_clusters,
)
from manu.evaluation.sdk_track_fusion_ref import load_params, parse_candidates  # noqa: E402
from manu.evaluation.system_sota_longquan import build_tracker_config  # noqa: E402

FLOAT_FMT = "%.9g"


def main() -> int:
    parser = argparse.ArgumentParser(description="SDK candidate stream -> online fused track output")
    parser.add_argument("--cands", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--img-h", type=int, default=512)
    args = parser.parse_args()

    frames = parse_candidates(args.cands)
    params = load_params(args.params)
    tracker_config = build_tracker_config(params, args.img_h)

    th_base = float(params["th_base"])
    th_salvage = float(params["th_salvage"])
    th_ground = float(params["th_ground"])
    sky_ratio = float(params["sky_ratio"])
    sky_y_boundary = args.img_h * sky_ratio
    th_deep = float(tracker_config.get("th_deep_salvage", 0.0))

    tracker = OnlineAdaptiveTracker(**tracker_config)
    min_hits = int(tracker_config["min_hits"])
    min_track_score = float(tracker_config["min_track_score"])

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for f_idx, frame in enumerate(frames):
            pred_pts = np.asarray(frame["points"], dtype=np.float32)
            pred_scs = np.asarray(frame["scores"], dtype=np.float32)
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
                if th_deep > 0.0:
                    deep_mask = is_sky & (pred_scs >= th_deep) & (pred_scs < min(th_salvage, th_base))
                    deep_pts = pred_pts[deep_mask]
                    deep_scs = pred_scs[deep_mask]
                    deep_pts, deep_scs = filter_dense_clutter_clusters(
                        deep_pts, deep_scs, cluster_radius=25.0, max_neighbors=2
                    )
                else:
                    deep_pts = np.zeros((0, 2), dtype=np.float32)
                    deep_scs = np.zeros((0,), dtype=np.float32)
            else:
                high_pts = np.zeros((0, 2), dtype=np.float32)
                high_scs = np.zeros((0,), dtype=np.float32)
                salvage_pts = np.zeros((0, 2), dtype=np.float32)
                salvage_scs = np.zeros((0,), dtype=np.float32)
                deep_pts = np.zeros((0, 2), dtype=np.float32)
                deep_scs = np.zeros((0,), dtype=np.float32)

            tracker.step(f_idx, high_pts, high_scs, salvage_pts, salvage_scs, deep_pts, deep_scs)

            # THE SDK's online semantics: the tracks that are ALIVE in this
            # frame, that have an observation in it, and that are confirmed.
            online = []
            for t in tracker.active_tracks:
                if f_idx not in t.observations:
                    continue
                is_conf = t.is_confirmed or (t.hits >= min_hits and t.score >= min_track_score)
                if not is_conf:
                    continue
                pos, score, _measured = t.observations[f_idx]
                online.append((int(t.track_id), float(pos[0]), float(pos[1]), float(score)))
            # stable sort by descending score (the C++ uses std::stable_sort)
            online.sort(key=lambda d: -d[3])

            parts = [str(f_idx), str(len(online))]
            for track_id, x, y, score in online:
                parts.append(str(track_id))
                parts.append(FLOAT_FMT % x)
                parts.append(FLOAT_FMT % y)
                parts.append(FLOAT_FMT % score)
                parts.append("0")
            handle.write(" ".join(parts) + "\n")
            total += len(online)

    print(f"frames     : {len(frames)}")
    print(f"detections : {total}")
    print(f"output     : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
