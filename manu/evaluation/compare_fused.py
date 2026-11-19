#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Compare two fused-track TSV files (e.g. the Python reference vs. the C++ port) frame by frame.

Both files use the format produced by :mod:`sdk_track_fusion_ref`::

    <frame_idx> <n> [<track_id1> <x1> <y1> <score1> <infilled1> <track_id2> ...]

Comparison rules
----------------
* **Discrete** items -- the number of frames, the number of detections per frame, every
  ``track_id`` and every ``infilled`` flag -- must match **exactly**.
* **Numeric** items -- ``(x, y)`` and ``score`` -- are compared with a tolerance
  (``--max-pos-diff`` / ``--max-score-diff``).

Exit code
---------
``1`` when at least one *discrete* mismatch is found, ``0`` otherwise.  Numeric deviations beyond
the tolerance are reported as well but do not change the exit code unless ``--strict`` is given.
``2`` is reserved for malformed / unreadable input files.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

MAX_DETAIL_LINES = 20


def parse_fused(path: Path) -> tuple[dict[int, list[dict]], list[int]]:
    """Parse a fused TSV into ``{frame_idx: [det, ...]}`` plus the frame order of appearance."""
    frames: dict[int, list[dict]] = {}
    order: list[int] = []
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
            if len(values) != 5 * count:
                raise ValueError(
                    f"{path}:{lineno}: frame {frame_idx} declares {count} detections "
                    f"but carries {len(values)} values (expected {5 * count})"
                )
            dets: list[dict] = []
            for k in range(count):
                base = 5 * k
                dets.append(
                    {
                        "track_id": int(values[base + 0]),
                        "x": float(values[base + 1]),
                        "y": float(values[base + 2]),
                        "score": float(values[base + 3]),
                        "infilled": int(values[base + 4]),
                    }
                )
            if frame_idx in frames:
                raise ValueError(f"{path}:{lineno}: duplicate frame index {frame_idx}")
            frames[frame_idx] = dets
            order.append(frame_idx)
    return frames, order


def compare(
    frames_a: dict[int, list[dict]],
    frames_b: dict[int, list[dict]],
    max_pos_diff: float,
    max_score_diff: float,
) -> dict:
    """Compare two parsed fused files."""
    discrete: list[dict] = []
    numeric: list[dict] = []
    max_dx = 0.0
    max_dy = 0.0
    max_dist = 0.0
    max_score = 0.0
    total_dets = 0

    keys_a = set(frames_a)
    keys_b = set(frames_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    for frame_idx in only_a:
        discrete.append(
            {"kind": "frame_missing_in_b", "frame": frame_idx, "detail": "only present in A"}
        )
    for frame_idx in only_b:
        discrete.append(
            {"kind": "frame_missing_in_a", "frame": frame_idx, "detail": "only present in B"}
        )

    for frame_idx in sorted(keys_a & keys_b):
        dets_a = frames_a[frame_idx]
        dets_b = frames_b[frame_idx]
        total_dets += len(dets_a)

        if len(dets_a) != len(dets_b):
            discrete.append(
                {
                    "kind": "det_count",
                    "frame": frame_idx,
                    "detail": f"A={len(dets_a)} vs B={len(dets_b)}",
                }
            )

        for k in range(min(len(dets_a), len(dets_b))):
            da = dets_a[k]
            db = dets_b[k]
            if da["track_id"] != db["track_id"]:
                discrete.append(
                    {
                        "kind": "track_id",
                        "frame": frame_idx,
                        "det": k,
                        "detail": f"A={da['track_id']} vs B={db['track_id']}",
                    }
                )
            if da["infilled"] != db["infilled"]:
                discrete.append(
                    {
                        "kind": "infilled",
                        "frame": frame_idx,
                        "det": k,
                        "detail": f"A={da['infilled']} vs B={db['infilled']}",
                    }
                )

            dx = da["x"] - db["x"]
            dy = da["y"] - db["y"]
            dist = math.hypot(dx, dy)
            max_dx = max(max_dx, abs(dx))
            max_dy = max(max_dy, abs(dy))
            max_dist = max(max_dist, dist)
            if abs(dx) > max_pos_diff or abs(dy) > max_pos_diff:
                numeric.append(
                    {
                        "kind": "pos",
                        "frame": frame_idx,
                        "det": k,
                        "detail": (
                            f"A=({da['x']:.9g}, {da['y']:.9g}) "
                            f"B=({db['x']:.9g}, {db['y']:.9g}) "
                            f"dx={dx:.6g} dy={dy:.6g}"
                        ),
                    }
                )

            dscore = da["score"] - db["score"]
            max_score = max(max_score, abs(dscore))
            if abs(dscore) > max_score_diff:
                numeric.append(
                    {
                        "kind": "score",
                        "frame": frame_idx,
                        "det": k,
                        "detail": f"A={da['score']:.9g} B={db['score']:.9g} d={dscore:.6g}",
                    }
                )

    return {
        "discrete": discrete,
        "numeric": numeric,
        "max_dx": max_dx,
        "max_dy": max_dy,
        "max_dist": max_dist,
        "max_score": max_score,
        "total_dets": total_dets,
        "frames_a": len(frames_a),
        "frames_b": len(frames_b),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two fused-track TSV files")
    parser.add_argument("--a", required=True, type=Path, help="File A (e.g. the Python reference)")
    parser.add_argument("--b", required=True, type=Path, help="File B (e.g. the C++ port)")
    parser.add_argument("--max-pos-diff", type=float, default=1e-4, help="Max |Δx| / |Δy| (default: 1e-4)")
    parser.add_argument("--max-score-diff", type=float, default=1e-5, help="Max |Δscore| (default: 1e-5)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also exit 1 when a numeric difference exceeds the tolerance",
    )
    args = parser.parse_args()

    try:
        frames_a, _order_a = parse_fused(args.a)
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot read {args.a}: {exc}")
        return 2
    try:
        frames_b, _order_b = parse_fused(args.b)
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot read {args.b}: {exc}")
        return 2
    stats = compare(frames_a, frames_b, args.max_pos_diff, args.max_score_diff)

    discrete = stats["discrete"]
    numeric = stats["numeric"]

    print("=== fused.tsv comparison ===")
    print(f"A                : {args.a}")
    print(f"B                : {args.b}")
    print(f"frames A         : {stats['frames_a']}")
    print(f"frames B         : {stats['frames_b']}")
    print(f"detections A     : {sum(len(v) for v in frames_a.values())}")
    print(f"detections B     : {sum(len(v) for v in frames_b.values())}")
    print(f"detections cmpd  : {stats['total_dets']}")
    print(f"discrete mismat. : {len(discrete)}")
    print(f"numeric mismat.  : {len(numeric)}")
    print(f"max |Δx|         : {stats['max_dx']:.9g}")
    print(f"max |Δy|         : {stats['max_dy']:.9g}")
    print(f"max |Δpos| (L2)  : {stats['max_dist']:.9g}")
    print(f"max |Δscore|     : {stats['max_score']:.9g}")
    print(f"tol pos / score  : {args.max_pos_diff:.9g} / {args.max_score_diff:.9g}")

    if discrete:
        print(f"\n--- discrete mismatches ({len(discrete)}) ---")
        for item in discrete[:MAX_DETAIL_LINES]:
            frame = item.get("frame", "-")
            det = item.get("det", "-")
            print(f"  [{item['kind']:20s}] frame={frame} det={det}  {item.get('detail', '')}")
        if len(discrete) > MAX_DETAIL_LINES:
            print(f"  ... and {len(discrete) - MAX_DETAIL_LINES} more")

    if numeric:
        print(f"\n--- numeric mismatches ({len(numeric)}) ---")
        for item in numeric[:MAX_DETAIL_LINES]:
            frame = item.get("frame", "-")
            det = item.get("det", "-")
            print(f"  [{item['kind']:20s}] frame={frame} det={det}  {item.get('detail', '')}")
        if len(numeric) > MAX_DETAIL_LINES:
            print(f"  ... and {len(numeric) - MAX_DETAIL_LINES} more")

    if discrete:
        print(f"\nRESULT: FAIL ({len(discrete)} discrete mismatch(es))")
        return 1
    if numeric and args.strict:
        print(f"\nRESULT: FAIL (strict mode, {len(numeric)} numeric mismatch(es))")
        return 1
    if numeric:
        print(f"\nRESULT: PASS (discrete items identical; {len(numeric)} numeric diff(s) beyond tolerance)")
        return 0
    print("\nRESULT: PASS (identical)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
