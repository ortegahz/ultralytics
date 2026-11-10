#!/usr/bin/env python3
"""Diagnose why the 龙泉山 tracker produces no tracks on the 纵横AI domain.

Prints the funnel stage by stage: candidates -> tracker output -> confirmation, so the gate that
drops everything is visible instead of guessed.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/tmp/pycharm_project_10ae9e2e")

from manu.evaluation.eval_bidirectional_track_fusion import evaluate_sequence_bidirectional  # noqa: E402
from manu.evaluation.system_sota_longquan import (  # noqa: E402
    apply_consecutive_confirmation,
    build_smoother_config,
    build_tracker_config,
)

LONGQUAN_PARAMS_JSON = (
    Path("/tmp/pycharm_project_10ae9e2e")
    / "manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json"
)


def load_longquan_params() -> dict:
    data = json.loads(LONGQUAN_PARAMS_JSON.read_text(encoding="utf-8"))
    return dict(data.get("params", data))


def main() -> None:
    cache_root = Path(sys.argv[1])
    params = load_longquan_params()
    print(f"longquan params: confirm_consec={params['confirm_consec']} min_hits={params['min_hits']} "
          f"max_age={params['max_age']} min_disp={params['min_disp']} sky_ratio={params['sky_ratio']}")

    for cache_path in sorted(cache_root.glob("*.pkl")):
        import pickle

        cache = pickle.loads(cache_path.read_bytes())
        meta = cache["meta"]
        frames = meta["frames"]
        scores_all = cache["scores"].astype(np.float32)
        points_all = cache["points"]

        # Rebuild the candidate pool from the cache rows (the merged rows the renderer stored).
        offset = cache["offset"]
        frame_index = cache["frame_index"]
        records = []
        for frame in range(frames):
            lo, hi = offset[frame], offset[frame + 1] if frame + 1 < len(offset) else len(offset)
            # The cache stores only frames that had rows; map by the stored frame_index list.
            records.append((frame, lo, hi))

        # rows are grouped by the stored frame_index order
        row_frames = frame_index.tolist()
        by_frame = {}
        for position, stored_frame in enumerate(row_frames):
            by_frame.setdefault(stored_frame, []).append(position)

        cand_scores, cand_points = [], []
        for frame in range(frames):
            positions = by_frame.get(frame, [])
            if positions:
                cand_scores.append(scores_all[positions])
                cand_points.append(points_all[positions])
            else:
                cand_scores.append(np.zeros(0, np.float32))
                cand_points.append(np.zeros((0, 2), np.float32))

        total = sum(len(s) for s in cand_scores)
        print(f"\n=== {cache_path.stem}: frames={frames} candidate rows={total} "
              f"frames_with_cand={sum(1 for s in cand_scores if len(s))} ===")
        if total == 0:
            print("  NO CANDIDATES -> the detection stage produced nothing")
            continue
        allsc = np.concatenate(cand_scores)
        print(f"  score pct: min={allsc.min():.3f} p50={np.percentile(allsc,50):.3f} "
              f"p90={np.percentile(allsc,90):.3f} max={allsc.max():.3f}")

        recs = []
        for frame in range(frames):
            recs.append(
                {
                    "im_name": f"{cache_path.stem}__{frame:06d}.jpg",
                    "frame_index": frame,
                    "pred_points": cand_points[frame],
                    "pred_scores": cand_scores[frame],
                    "gt_pts": np.zeros((0, 2), np.float32),
                    "gt_bboxes": np.zeros((0, 4), np.float32),
                }
            )

        # Stage 1: the tracker alone (before the consecutive-frame confirmation).
        res = evaluate_sequence_bidirectional(
            records=recs,
            dist_thresh=8.0,
            th_base=0.20,
            th_salvage=0.06,
            th_ground=0.20,
            sky_ratio=float(params["sky_ratio"]),
            img_h=512,
            tracker_config=build_tracker_config(params, 512),
            smoother_config=build_smoother_config(params, 512),
            match_mode="bbox",
        )
        bidi = res["bidi_frame_dets"]
        print(f"  tracker+smoother output: frames_with_dets={sum(1 for d in bidi if d)} "
              f"detections={sum(len(d) for d in bidi)}")
        for key in ("track1", "track2", "track3"):
            if key in res:
                print(f"    {key}: {res[key]}")

        # Stage 2: the consecutive-frame confirmation, at several confirm_consec values.
        for consec in (1, 2, 3, 5, 7):
            dets = apply_consecutive_confirmation(
                bidi, confirm_consec=consec, confirm_gap_tol=int(params["confirm_gap_tol"]),
                confirm_trim_prefix=int(params["confirm_trim_prefix"]),
            )
            print(f"  confirm_consec={consec}: frames_with_dets={sum(1 for d in dets if d)} "
                  f"detections={sum(len(d) for d in dets)}")


if __name__ == "__main__":
    main()
