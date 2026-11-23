#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Reference wrapper: SDK candidate stream -> fused track output, via the *production* pipeline.

This is the golden reference for the C++ port of the spatio-temporal fusion stage.  It does **not**
re-implement the algorithm: it feeds the candidate stream into
:func:`manu.evaluation.system_sota_longquan.run_system_sota` (the same code the Python production
path uses) and serialises the result in a simple, diff-friendly text format.

Input ``candidates.tsv`` (whitespace separated, one frame per line)::

    <frame_idx> <n> [<x1> <y1> <score1> <x2> <y2> <score2> ...]

* ``frame_idx`` -- 0-based, strictly increasing.
* ``x, y``      -- native 640x512 frame pixel coordinates.
* ``score``     -- heat-map peak value (float32).

Output ``fused.tsv`` (one frame per line, always emitted, even for empty frames)::

    <frame_idx> <n> [<track_id1> <x1> <y1> <score1> <infilled1> <track_id2> ...]

* detections of a frame are sorted by descending score (that is what
  ``apply_consecutive_confirmation`` produces and what the business-domain matcher expects),
* ``infilled`` is ``0`` for a real observation and ``1`` for a synthesised (interpolated) frame,
* ``x``, ``y`` and ``score`` are printed with ``%.9g`` (9 significant digits round-trip float32
  exactly).

Cross-scale candidate merge
---------------------------
``--dedup-radius R`` enables the **candidate construction stage** cross-scale merge,
the counterpart of ``mosaic::MergeCandidatesByRadius``
(``manu/pipeline/npu/mosaic_track_fusion.cpp``) and of the historical server rule
``eval_multiscale_cache_th._merge_scale_rows``.  The merge runs on the candidate pool
of every frame **before** the fusion engine sees it (in C++ it happens in the
candidate construction stage, i.e. before ``TrackFusionEngine::Step``), so the engine
must be fed the merged candidates in the merged order.  ``R <= 0`` (the default)
disables the stage completely -- no sorting, no dropping -- which keeps the output
byte-identical to the behaviour before the stage existed.

Usage::

    /home/manu/anaconda3/envs/uav/bin/python sdk_track_fusion_ref.py \
        --cands candidates.tsv \
        --params system_sota_longquan_trial0080_delivery_params.json \
        --out fused.tsv [--img-h 512] [--seq NAME] [--dedup-radius 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ULTRALYTICS_ROOT = "/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/ultralytics"
if ULTRALYTICS_ROOT not in sys.path:
    sys.path.insert(0, ULTRALYTICS_ROOT)

from manu.evaluation.system_sota_longquan import run_system_sota, IMG_H  # noqa: E402
from manu.evaluation.eval_bidirectional_track_fusion import natural_sort_key  # noqa: E402

FLOAT_FMT = "%.9g"


def parse_candidates(path: Path) -> list[dict]:
    """Parse the candidate TSV into per-frame records.

    Returns one dict per frame with the keys ``frame_idx``, ``points`` (float32 ``(n, 2)``) and
    ``scores`` (float32 ``(n,)``).
    """
    frames: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{lineno}: expected at least '<frame_idx> <n>', got {line!r}")
            frame_idx = int(parts[0])
            count = int(parts[1])
            values = parts[2:]
            if len(values) != 3 * count:
                raise ValueError(
                    f"{path}:{lineno}: frame {frame_idx} declares {count} candidates "
                    f"but carries {len(values)} values (expected {3 * count})"
                )
            points = np.zeros((count, 2), dtype=np.float32)
            scores = np.zeros((count,), dtype=np.float32)
            for k in range(count):
                points[k, 0] = float(values[3 * k + 0])
                points[k, 1] = float(values[3 * k + 1])
                scores[k] = float(values[3 * k + 2])
            frames.append({"frame_idx": frame_idx, "points": points, "scores": scores})
    return frames


def load_params(path: Path) -> dict:
    """Load the parameter block from a params JSON file (the ``"params"`` sub-dictionary)."""
    blob = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(blob, dict) and "params" in blob:
        params = blob["params"]
    else:
        params = blob
    if not isinstance(params, dict):
        raise ValueError(f"{path}: expected a JSON object with a 'params' sub-dictionary")
    return params


def merge_candidates_by_radius(
    points: np.ndarray, scores: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-scale candidate merge -- the bit-exact mirror of ``mosaic::MergeCandidatesByRadius``.

    The C++ counterpart lives in ``manu/pipeline/npu/mosaic_track_fusion.cpp``; the
    historical server rule it mirrors is ``eval_multiscale_cache_th._merge_scale_rows``.
    The rules, one frame at a time:

    1. ``radius <= 0`` => the pool is returned untouched (no sort, no drop) -- bit-identical to the
       behaviour before the stage existed;
    2. the pool is sorted by DESCENDING score with a **stable** sort (C++: ``std::stable_sort`` with
       ``a.score > b.score``; Python's ``sorted`` is stable as well, so candidates with
       equal scores keep their input order on both sides);
    3. greedy keep: a candidate is dropped when its squared distance to an ALREADY KEPT candidate is
       ``<= radius * radius``.

    The distance is computed in double precision on the float32 coordinates promoted to double
    (``float(points[i, 0])``), exactly like the C++ ``(double)cand.x - (double)k.x``, so the ``<=``
    comparison is decided identically on both sides.

    Returns the merged ``(points, scores)`` in the order the fusion engine has to see
    them: the engine may be order sensitive (it associates the candidates of a frame
    in array order).
    """
    count = int(scores.shape[0])
    if count == 0 or radius <= 0.0:
        return points, scores
    radius_squared = radius * radius
    # `-float(score)` is the exact negation of the float32 score promoted to double => the ordering
    # matches the C++ `a.score > b.score` comparison bit for bit.
    order = sorted(range(count), key=lambda index: -float(scores[index]))
    kept: list[int] = []
    kept_xy: list[tuple[float, float]] = []
    for index in order:
        x = float(points[index, 0])
        y = float(points[index, 1])
        covered = False
        for px, py in kept_xy:
            dx = x - px
            dy = y - py
            if dx * dx + dy * dy <= radius_squared:
                covered = True
                break
        if not covered:
            kept.append(index)
            kept_xy.append((x, y))
    selection = np.asarray(kept, dtype=np.intp)
    return points[selection], scores[selection]


def build_records(frames: list[dict], seq: str) -> list[dict]:
    """Build the record list consumed by ``run_system_sota``.

    ``im_name`` uses a zero padded frame index so that ``natural_sort_key`` orders the records
    exactly like the input file (``evaluate_sequence_bidirectional`` sorts the records before
    running the tracker).
    """
    records: list[dict] = []
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        records.append(
            {
                "im_name": f"{seq}__{frame_idx:06d}.jpg",
                "frame_index": frame_idx,
                "pred_points": frame["points"],
                "pred_scores": frame["scores"],
                "gt_pts": np.zeros((0, 2), dtype=np.float32),
                "gt_bboxes": np.zeros((0, 4), dtype=np.float32),
            }
        )
    return records


def format_frame(frame_idx: int, dets: list[dict]) -> str:
    """Serialise one frame of detections."""
    parts = [str(frame_idx), str(len(dets))]
    for det in dets:
        pos = det["pos"]
        parts.append(str(int(det["track_id"])))
        parts.append(FLOAT_FMT % float(pos[0]))
        parts.append(FLOAT_FMT % float(pos[1]))
        parts.append(FLOAT_FMT % float(det["score"]))
        parts.append("1" if bool(det["infilled"]) else "0")
    return " ".join(parts)


def run(
    cands_path: Path,
    params_path: Path,
    out_path: Path,
    img_h: int,
    seq: str,
    physical_frame_index: bool = False,
    dedup_radius: float = 0.0,
) -> dict:
    """Run the reference pipeline and write the fused TSV.

    ``physical_frame_index`` feeds the tracker the CAMERA frame numbers of the
    candidate stream instead of a dense update counter: the tracker then sees
    the REAL frame gap (dn) -- the delta-n semantics of the C++ engine.  It is
    OFF by default, so the legacy dense alignment is untouched.

    ``dedup_radius`` enables the cross-scale candidate merge of the **candidate
    construction stage** (the counterpart of ``mosaic::MergeCandidatesByRadius``).
    It is applied to the candidate pool of every frame BEFORE ``build_records``,
    i.e. before the fusion engine sees the candidates -- the same position the
    merge occupies in C++.  ``0`` (the default) disables the stage and keeps the
    output byte-identical.
    """
    frames = parse_candidates(cands_path)
    params = load_params(params_path)

    # ---- 候选构造阶段：跨尺度合并（C++ 侧 mosaic::MergeCandidatesByRadius 的等价物）----
    candidates_in = sum(int(frame["scores"].shape[0]) for frame in frames)
    frames_with_drop = 0
    if dedup_radius > 0.0:
        for frame in frames:
            merged_points, merged_scores = merge_candidates_by_radius(
                frame["points"], frame["scores"], dedup_radius
            )
            if int(merged_scores.shape[0]) != int(frame["scores"].shape[0]):
                frames_with_drop += 1
            frame["points"] = merged_points
            frame["scores"] = merged_scores
    candidates_kept = sum(int(frame["scores"].shape[0]) for frame in frames)

    records = build_records(frames, seq)

    dets, _res = run_system_sota(
        records, params, img_h=img_h, use_physical_frame_index=physical_frame_index
    )

    # `dets` is ordered like `records_sorted` inside `evaluate_sequence_bidirectional`, i.e. sorted
    # by `natural_sort_key(im_name)`.  Map it back to the original frame indices explicitly so the
    # output never depends on the sort order being the identity.
    by_frame: dict[int, list[dict]] = {}
    if physical_frame_index:
        # `dets` is indexed by the CAMERA frame number and covers EVERY frame of
        # the axis -- including the frames the detector never delivered, which
        # carry the interpolated (infilled) detections.  Building the map from
        # the records instead would drop them.
        by_frame = {index: list(dets[index]) for index in range(len(dets))}
    else:
        order = sorted(range(len(records)), key=lambda i: natural_sort_key(records[i]["im_name"]))
        for slot, record_idx in enumerate(order):
            by_frame[int(records[record_idx]["frame_index"])] = list(dets[slot])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_dets = 0
    # In the physical mode the output axis IS the camera frame number, so EVERY
    # frame between 0 and the largest one is written (the frames the detector
    # never delivered may still carry an interpolated detection).  That is the
    # same axis the C++ engine uses => the two files can be compared directly.
    if physical_frame_index:
        axis = range(max((int(f["frame_idx"]) for f in frames), default=-1) + 1)
    else:
        axis = (int(frame["frame_idx"]) for frame in frames)
    with out_path.open("w", encoding="utf-8") as handle:
        for frame_idx in axis:
            frame_dets = by_frame.get(frame_idx, [])
            total_dets += len(frame_dets)
            handle.write(format_frame(frame_idx, frame_dets) + "\n")

    return {
        "frames": len(frames),
        "detections": total_dets,
        "out": str(out_path),
        "dedup_radius": float(dedup_radius),
        "candidates_in": candidates_in,
        "candidates_kept": candidates_kept,
        "candidates_dropped": candidates_in - candidates_kept,
        "frames_with_drop": frames_with_drop,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SDK candidate stream -> fused track output (reference, uses the production pipeline)"
    )
    parser.add_argument("--cands", required=True, type=Path, help="Input candidates TSV")
    parser.add_argument("--params", required=True, type=Path, help="Params JSON (uses the 'params' sub-dict)")
    parser.add_argument("--out", required=True, type=Path, help="Output fused TSV")
    parser.add_argument("--img-h", type=int, default=IMG_H, help=f"Frame height (default: {IMG_H})")
    parser.add_argument("--seq", type=str, default=None, help="Sequence name (default: candidates file stem)")
    parser.add_argument(
        "--physical-frame-index",
        action="store_true",
        help="feed the tracker the CAMERA frame numbers of the candidate stream (the delta-n "
             "semantics); off by default (the legacy dense alignment)",
    )
    parser.add_argument(
        "--dedup-radius",
        type=float,
        default=0.0,
        help="cross-scale candidate merge radius (px) of the candidate construction stage, 0 = off "
             "(same semantics as mosaic::MergeCandidatesByRadius / "
             "eval_multiscale_cache_th._merge_scale_rows)",
    )
    args = parser.parse_args()

    seq = args.seq if args.seq else args.cands.stem
    try:
        stats = run(
            args.cands,
            args.params,
            args.out,
            args.img_h,
            seq,
            physical_frame_index=args.physical_frame_index,
            dedup_radius=args.dedup_radius,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    print(f"candidates : {args.cands}")
    print(f"params     : {args.params}")
    print(f"seq        : {seq}")
    print(f"img_h      : {args.img_h}")
    print(f"frames     : {stats['frames']}")
    print(f"detections : {stats['detections']}")
    print(f"output     : {stats['out']}")
    print(
        f"[i] dedup    : radius={stats['dedup_radius']:g} "
        f"cands_in={stats['candidates_in']} cands_kept={stats['candidates_kept']} "
        f"dropped={stats['candidates_dropped']} frames_with_drop={stats['frames_with_drop']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
