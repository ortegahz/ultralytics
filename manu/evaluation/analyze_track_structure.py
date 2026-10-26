#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Track-structure statistics of a system-SOTA prediction file.

Why
---
Threshold sweeps look monotone in aggregate metrics, but the *mechanism* behind a change is
invisible there: the system pipeline is not a per-frame filter, it is a tracker plus a
bidirectional smoother plus a consecutive-frame confirmation gate.  Removing low-confidence
detections therefore does not simply "delete boxes" -- it can re-shape the track graph
(tracks split, fragments survive the confirmation gate on their own, infilled gaps appear),
and the false-positive count can move *up* while the threshold moves *up*.

This tool makes that structure measurable: for one prediction file it reports the number of
tracks, the number of real vs. infilled detections, the track-length histogram and the share
of short tracks, next to the file's own TP/FP/FN.  Comparing those numbers across working
points is how a non-monotone FAR response gets explained instead of guessed.

Usage
-----
    python manu/evaluation/analyze_track_structure.py \
        --preds-dir runs/optuna_mix_ratio/trials/trial_0080/lq_eval/preds_track_v2 \
        --sequence VIDEO00012_19700101_002251

    # compare several working points for the same sequence (one --compare per directory)
    python manu/evaluation/analyze_track_structure.py \
        --compare preds_track=0.30/6/0.22 \
        --compare preds_track_th045=0.45/6/0.22 \
        --compare preds_track_v2=0.42/7/0.26 \
        --sequence VIDEO00012_19700101_002251

Execution machine: any machine that can read the prediction directory (CPU only).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

# Bucket boundaries on the number of *real* (non-infilled) observations of a track.
BUCKETS = ((1, 1), (2, 3), (4, 6), (7, 15), (16, 10**9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track-structure statistics of a system-SOTA prediction file")
    parser.add_argument("--preds-dir", default="", help="Directory of <sequence>.json prediction files")
    parser.add_argument("--sequence", default="", help="Sequence name (default: every file in the dir)")
    parser.add_argument("--compare", action="append", default=[], metavar="DIR=LABEL",
                        help="Directory to compare, e.g. preds_track_v2=0.42/7/0.26; repeatable")
    parser.add_argument("--json", default="", help="Optional path for the machine readable report")
    return parser.parse_args()


def bucket_of(real: int) -> str:
    for low, high in BUCKETS:
        if low <= real <= high:
            return f"{low}-{high}" if high < 10**9 else f"{low}+"
    return "0"


def analyse(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    per_frame = data.get("per_frame", {})
    metrics = data.get("metrics", {})
    config = data.get("config", {})
    params = config.get("params", {}) or {}

    # The system pipeline applies ``th_base`` above the horizon and ``th_ground`` below it
    # (eval_bidirectional_track_fusion.py:814), so the sky/ground split decides which threshold
    # gates a detection.  Counting the detections per side shows where the clutter lives.
    img_h = int(config.get("img_h") or 0)
    sky_ratio = float(params.get("sky_ratio", 0.6))
    boundary = img_h * sky_ratio if img_h else None

    lengths: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0])  # track_id -> [total, infilled]
    n_real = 0
    n_inf = 0
    side_counts = {"sky": 0, "ground": 0, "unknown": 0}
    side_scores: dict[str, list[float]] = {"sky": [], "ground": [], "unknown": []}
    for dets in per_frame.values():
        for det in dets:
            track_id = int(det["track_id"])
            lengths[track_id][0] += 1
            if det.get("infilled", False):
                lengths[track_id][1] += 1
                n_inf += 1
            else:
                n_real += 1
            if boundary is None:
                side = "unknown"
            else:
                side = "sky" if float(det["pos"][1]) < boundary else "ground"
            side_counts[side] += 1
            side_scores[side].append(float(det["score"]))

    histogram: collections.Counter = collections.Counter()
    for total, infilled in lengths.values():
        histogram[bucket_of(total - infilled)] += 1

    short = sum(count for bucket, count in histogram.items() if bucket in ("1-1", "2-3"))

    def median(values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    return {
        "sequence": path.stem,
        "path": str(path),
        "metrics": {key: metrics.get(key) for key in ("tp", "fp", "fn", "det", "gt", "recall")},
        "params": {key: params.get(key)
                   for key in ("th_base", "th_ground", "confirm_consec", "min_track_score", "sky_ratio")},
        "tracks": len(lengths),
        "detections": n_real + n_inf,
        "real_detections": n_real,
        "infilled_detections": n_inf,
        "track_length_histogram": dict(sorted(histogram.items())),
        "short_track_share": round(short / len(lengths), 4) if lengths else 0.0,
        "sky_boundary_y": boundary,
        "detections_by_side": side_counts,
        "median_score_by_side": {side: round(median(values), 4) for side, values in side_scores.items()},
    }


def print_row(row: dict) -> None:
    metrics = row["metrics"]
    params = row["params"]
    boundary = row.get("sky_boundary_y")
    print(f"--- {row['sequence']}  ({params.get('th_base')}/{params.get('th_ground')}/"
          f"confirm={params.get('confirm_consec')}/sky={params.get('sky_ratio')})  {row['path']}")
    print(f"  TP={metrics.get('tp')} FP={metrics.get('fp')} FN={metrics.get('fn')} det={metrics.get('det')}")
    print(f"  轨迹数={row['tracks']}  检出={row['detections']} (真实={row['real_detections']} "
          f"插补={row['infilled_detections']})  短轨迹占比={row['short_track_share'] * 100:.1f}%")
    print(f"  轨迹长度分布(按真实观测帧数): {row['track_length_histogram']}")
    if boundary:
        sides = row.get("detections_by_side", {})
        scores = row.get("median_score_by_side", {})
        print(f"  地平线 y={boundary:.0f} | 天空区检出={sides.get('sky')} (中位分 {scores.get('sky')}) "
              f"| 地面区检出={sides.get('ground')} (中位分 {scores.get('ground')})")


def main() -> int:
    args = parse_args()
    targets: list[tuple[Path, str]] = []

    if args.compare:
        for item in args.compare:
            if "=" not in item:
                raise SystemExit(f"--compare must look like DIR=LABEL -- got {item!r}")
            raw_dir, label = item.split("=", 1)
            directory = Path(raw_dir)
            if args.sequence:
                targets.append((directory / f"{args.sequence}.json", label))
            else:
                targets.extend((path, label) for path in sorted(directory.glob("*.json"))
                               if path.stem != "summary")
    elif args.preds_dir:
        directory = Path(args.preds_dir)
        if args.sequence:
            targets.append((directory / f"{args.sequence}.json", args.preds_dir))
        else:
            targets.extend((path, args.preds_dir) for path in sorted(directory.glob("*.json"))
                           if path.stem != "summary")
    else:
        raise SystemExit("give --preds-dir or --compare")

    rows = []
    for path, label in targets:
        if not path.is_file():
            print(f"[WARN] missing prediction file: {path}")
            continue
        row = analyse(path)
        row["label"] = label
        rows.append(row)

    if not rows:
        print("[FAIL] nothing to analyse")
        return 2

    for row in rows:
        print_row(row)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report -> {out.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
