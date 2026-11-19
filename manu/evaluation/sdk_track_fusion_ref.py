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

Usage::

    /home/manu/anaconda3/envs/uav/bin/python sdk_track_fusion_ref.py \
        --cands candidates.tsv \
        --params system_sota_longquan_trial0080_delivery_params.json \
        --out fused.tsv [--img-h 512] [--seq NAME]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ULTRALYTICS_ROOT = "/media/manu/1TB-Volume/workspace/ultralytics"
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


def run(cands_path: Path, params_path: Path, out_path: Path, img_h: int, seq: str) -> dict:
    """Run the reference pipeline and write the fused TSV."""
    frames = parse_candidates(cands_path)
    params = load_params(params_path)
    records = build_records(frames, seq)

    dets, _res = run_system_sota(records, params, img_h=img_h)

    # `dets` is ordered like `records_sorted` inside `evaluate_sequence_bidirectional`, i.e. sorted
    # by `natural_sort_key(im_name)`.  Map it back to the original frame indices explicitly so the
    # output never depends on the sort order being the identity.
    order = sorted(range(len(records)), key=lambda i: natural_sort_key(records[i]["im_name"]))
    by_frame: dict[int, list[dict]] = {}
    for slot, record_idx in enumerate(order):
        by_frame[int(records[record_idx]["frame_index"])] = list(dets[slot])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_dets = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for frame in frames:
            frame_idx = int(frame["frame_idx"])
            frame_dets = by_frame.get(frame_idx, [])
            total_dets += len(frame_dets)
            handle.write(format_frame(frame_idx, frame_dets) + "\n")

    return {
        "frames": len(frames),
        "detections": total_dets,
        "out": str(out_path),
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
    args = parser.parse_args()

    seq = args.seq if args.seq else args.cands.stem
    try:
        stats = run(args.cands, args.params, args.out, args.img_h, seq)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
