#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Attribute a board-side FAR blow-up to the candidate stage or to the fusion engine.

WHAT THIS DOES
--------------
The RK3588 SDK feeds the track-fusion engine every *connected component* of the
multi-scale back-projection (``mosaic::Comp``, ``tag`` = 0 native / 1 thumb160 /
2 thumb80).  The server-side reference instead builds its candidate pool with
``manu.evaluation.eval_multiscale_cache_th.merge_scales`` -- a cross-scale
de-duplication that keeps ONE candidate per target before the fusion engine ever
sees it.  When one physical target therefore reaches the fusion engine as 2..3
candidates, the engine (correctly) opens 2..3 tracks and the extra ones become
false positives.

This script answers "which of the two is responsible" by running the SAME Python
fusion reference on three different candidate streams:

* ``A``  server cache, per-scale rows pooled WITHOUT the cross-scale merge
* ``B``  the board's own candidate stream (``mosaic_track_dump`` ``_cands.tsv``)
* ``C``  server cache WITH the cross-scale merge  (= the calibrated delivery caliber)

and then sweeps the merge radius over the board's candidate stream, so the
embedded team gets a *usable* radius instead of the server's 768 px (which is
larger than the image and therefore collapses the whole frame to one candidate).

The matcher is the same one the server uses
(``manu.evaluation.eval_multiscale_cache_th.count_matches`` / ``metrics``).

Usage::

    cd /run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/ultralytics
    PYTHONPATH=. /run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/.conda-envs/lqcache/bin/python \
        manu/evaluation/board_candidate_attribution.py \
        --board-cands /tmp/boardwork/board_td_cands.tsv \
        --cache-root /run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/lqcache_rebuild/lookahead_residual/data/ms_cache \
        --labels-root /run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/lqcache_rebuild/lookahead_residual/data/labels \
        --params-json manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json \
        --sequence VIDEO00007_19700101_001324 --window 2500 \
        --out-json /tmp/boardwork/attrib.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.eval_multiscale_cache_th import merge_scales  # noqa: E402
from manu.evaluation.rt_fps_jitter_search import run_pipeline, score_run  # noqa: E402
from manu.evaluation.system_sota_longquan import (  # noqa: E402
    SCALE_TAGS,
    load_ground_truth,
    load_predictions,
    make_records,
)


def parse_board_cands(path: Path) -> dict[int, tuple]:
    """Parse a ``mosaic_track_dump`` ``_cands.tsv`` (``<fid> <n> cx cy peak ...``)."""
    out: dict[int, tuple] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if not parts or raw.startswith("#"):
            continue
        frame_id, count = int(parts[0]), int(parts[1])
        values = parts[2:]
        if len(values) != 3 * count:
            raise ValueError(f"{path}: frame {frame_id} declares {count} candidates "
                             f"but carries {len(values)} values")
        scores, points = [], []
        for k in range(count):
            base = 3 * k
            scores.append(float(values[base + 2]))
            points.append((float(values[base]), float(values[base + 1])))
        out[frame_id] = (np.asarray(scores, dtype=np.float32),
                         np.asarray(points, dtype=np.float32).reshape(-1, 2))
    return out


def pool_rows_unmerged(by_tag: dict) -> dict[int, tuple]:
    """Pool the per-scale cache rows WITHOUT the cross-scale merge (the board's view)."""
    out: dict[int, tuple] = {}
    for frame_id in sorted({f for rows in by_tag.values() for f in rows}):
        scores: list[float] = []
        points: list[tuple[float, float]] = []
        for tag in sorted(by_tag):
            score_row, point_row = by_tag[tag].get(
                frame_id, (np.zeros(0), np.zeros((0, 2))))
            for k in range(len(score_row)):
                scores.append(float(score_row[k]))
                points.append((float(point_row[k][0]), float(point_row[k][1])))
        if scores:
            out[frame_id] = (np.asarray(scores, dtype=np.float32),
                             np.asarray(points, dtype=np.float32))
    return out


def merge_candidates(cands: dict[int, tuple], radius: float) -> dict[int, tuple]:
    """Score-descending greedy de-duplication inside ``radius`` (same rule as merge_scales)."""
    radius_squared = radius * radius
    out: dict[int, tuple] = {}
    for frame_id, (scores, points) in cands.items():
        kept: list[int] = []
        for index in np.argsort(-scores):
            x, y = points[index]
            if any((x - points[j][0]) ** 2 + (y - points[j][1]) ** 2 <= radius_squared
                   for j in kept):
                continue
            kept.append(int(index))
        if kept:
            index = np.asarray(kept)
            out[frame_id] = (scores[index], points[index])
    return out


def evaluate(cands: dict[int, tuple], truth: dict, params: dict, sequence: str,
              window: int) -> dict:
    cands = {f: v for f, v in cands.items() if f < window}
    records = make_records(cands, truth, sequence)
    run = run_pipeline(sequence, params, None, dense_time_base=False, records=records)
    run["total_frames"] = window
    row = score_run(run, None)
    row["mean_candidates_per_frame"] = (
        sum(len(v[0]) for v in cands.values()) / max(1, len(cands)))
    row["frames_with_candidates"] = len(cands)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-cands", type=Path, default=None,
                        help="mosaic_track_dump _cands.tsv from the board")
    parser.add_argument("--cache-root", type=Path, default=None,
                        help="ms_cache directory (server-side multi-scale cache)")
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--params-json", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--window", type=int, default=2500)
    parser.add_argument("--merge-radii", default="0,4,6,8,10,12,16,20,24,32,768")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    params = json.loads(args.params_json.read_text(encoding="utf-8"))["params"]
    truth = load_ground_truth(args.labels_root, args.sequence, 640, 512)
    rows: list[dict] = []

    if args.cache_root:
        _meta, by_tag = load_predictions(args.cache_root, args.sequence, list(SCALE_TAGS), 0.0)
        for tag, cands in (
            ("A_cache_per_scale_UNMERGED", pool_rows_unmerged(by_tag)),
            ("C_cache_merged_768 (=server caliber)", merge_scales(by_tag, 768.0)),
        ):
            row = evaluate(cands, truth, params, args.sequence, args.window)
            row["arm"] = tag
            rows.append(row)
            print(f"[{tag}] mean/frame={row['mean_candidates_per_frame']:.2f} "
                  f"F1={row['f1_all']:.7f} R={row['recall_all']:.6f} "
                  f"P={row['precision_all']:.6f} FAR={row['far_all']:.7f} "
                  f"tp={row['tp_all']} fp={row['fp_all']} fn={row['fn_all']}")

    if args.board_cands:
        board = parse_board_cands(args.board_cands)
        row = evaluate(board, truth, params, args.sequence, args.window)
        row["arm"] = "B_board_raw_candidates"
        rows.append(row)
        print(f"[B_board_raw_candidates] mean/frame={row['mean_candidates_per_frame']:.2f} "
              f"F1={row['f1_all']:.7f} R={row['recall_all']:.6f} "
              f"P={row['precision_all']:.6f} FAR={row['far_all']:.7f} "
              f"tp={row['tp_all']} fp={row['fp_all']} fn={row['fn_all']}")

        print(f"\n{'radius':>8} {'mean/frame':>11} {'F1':>10} {'R':>9} {'P':>9} {'FAR':>10}")
        sweep = []
        for radius in [float(x) for x in args.merge_radii.split(",")]:
            merged = board if radius <= 0 else merge_candidates(board, radius)
            row = evaluate(merged, truth, params, args.sequence, args.window)
            row["merge_radius"] = radius
            sweep.append(row)
            print(f"{radius:>8.0f} {row['mean_candidates_per_frame']:>11.2f} "
                  f"{row['f1_all']:>10.7f} {row['recall_all']:>9.6f} "
                  f"{row['precision_all']:>9.6f} {row['far_all']:>10.7f}")
        rows.extend(sweep)

    if args.out_json:
        args.out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
        print(f"[OUT] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
