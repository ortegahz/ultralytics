#!/usr/bin/env python3
"""Find single-frame detection segments that the tracking pipeline removed *entirely*.

The tracking tier's value proposition is "drop isolated flicker, keep real trajectories".
But the same gates can also delete a *persistent* segment: a target the detector sees
continuously for 30+ frames can still be dropped if it moves rigidly (static-pruning),
drifts out of the association gate, or scores below the confirmation gate.

A run of detections that the single-frame tier kept for many consecutive frames but the
tracked tier dropped completely is the risky case: a real, easy-to-see target that the
reviewer would never see. This script enumerates exactly those cases.

Per sequence it reports:
  * runs in the single-frame tier grouped by length bucket (1 / 2-6 / 7+)
  * for each run with len >= min_run: how many of its frames survived in the tracked tier
  * a "FULLY DROPPED" flag when a long run survived 0 frames
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


def load(path: Path) -> list[list[dict]]:
    out = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line).get("detections") or [])
    return out


def runs_of(hit_frames: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = prev = None
    for idx in hit_frames:
        if prev is not None and idx == prev + 1:
            prev = idx
            continue
        if start is not None:
            runs.append((start, prev))
        start = prev = idx
    if start is not None:
        runs.append((start, prev))
    return runs


def run_drift(single: list[list[dict]], start: int, end: int) -> tuple[float, float]:
    """Chase the detection through the run and report (max_displacement, net_displacement).

    Nearest-neighbour chaining inside the run: start from the detection of the first frame,
    then in every following frame pick the detection closest to the previous position.
    A run whose target never moves is a static structure (lamp post, tree edge, building
    edge) and dropping it is the intended behaviour; a run that moves is a candidate real
    target and needs a human look.
    """
    if end <= start:
        return 0.0, 0.0
    prev = None
    max_d = 0.0
    first = None
    last = None
    for idx in range(start, end + 1):
        dets = single[idx]
        if not dets:
            continue
        if prev is None:
            cur = dets[0]
        else:
            cur = min(dets, key=lambda d: (d["cx"] - prev[0]) ** 2 + (d["cy"] - prev[1]) ** 2)
        px, py = float(cur["cx"]), float(cur["cy"])
        if first is None:
            first = (px, py)
        if prev is not None:
            d = ((px - prev[0]) ** 2 + (py - prev[1]) ** 2) ** 0.5
            max_d = max(max_d, d)
        prev = (px, py)
        last = (px, py)
    if first is None or last is None:
        return 0.0, 0.0
    net = ((last[0] - first[0]) ** 2 + (last[1] - first[1]) ** 2) ** 0.5
    return max_d, net


def main() -> None:
    single_dir = Path(sys.argv[1])
    tracked_dir = Path(sys.argv[2])
    min_run = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    static_px = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    json_out = Path(sys.argv[5]) if len(sys.argv) > 5 else None

    fully_dropped: list[dict] = []
    partially: list[dict] = []
    seq_rows: list[dict] = []

    for path in sorted(single_dir.glob("*.jsonl")):
        tp = tracked_dir / path.name
        if not tp.exists():
            continue
        single = load(path)
        tracked = load(tp)
        n = min(len(single), len(tracked))
        single, tracked = single[:n], tracked[:n]

        hit = [i for i, dets in enumerate(single) if dets]
        thr = [bool(dets) for dets in tracked]
        runs = runs_of(hit)

        buckets = {"1": 0, "2-6": 0, "7+": 0}
        seq_full: list[dict] = []
        seq_part: list[dict] = []
        for start, end in runs:
            length = end - start + 1
            key = "1" if length == 1 else ("2-6" if length < 7 else "7+")
            buckets[key] += 1
            if length < min_run:
                continue
            kept = sum(1 for i in range(start, end + 1) if thr[i])
            ratio = kept / length
            max_d, net_d = run_drift(single, start, end)
            rec = {
                "sequence": path.stem,
                "start": start,
                "end": end,
                "length": length,
                "kept": kept,
                "kept_ratio": round(ratio, 3),
                "score_at_start": round(float(single[start][0].get("score", -1)), 3),
                "cx": round(float(single[start][0].get("cx", -1)), 1),
                "cy": round(float(single[start][0].get("cy", -1)), 1),
                "max_step_px": round(max_d, 2),
                "net_disp_px": round(net_d, 2),
                "motion": "static" if net_d <= static_px else "moving",
            }
            if kept == 0:
                fully_dropped.append(rec)
                seq_full.append(rec)
            elif ratio < 0.5:
                partially.append(rec)
                seq_part.append(rec)

        seq_rows.append(
            {
                "sequence": path.stem,
                "frames": n,
                "single_hit_frames": len(hit),
                "tracked_hit_frames": sum(thr),
                "runs": len(runs),
                "run_buckets": buckets,
                "long_runs": sum(1 for s, e in runs if e - s + 1 >= min_run),
                "fully_dropped_long_runs": len(seq_full),
                "partially_dropped_long_runs": len(seq_part),
            }
        )

    print(f"sequences compared : {len(seq_rows)}   (long-run threshold = {min_run} frames)")
    print(f"long runs fully dropped : {len(fully_dropped)}")
    print(f"long runs >50% dropped  : {len(partially)}\n")

    if fully_dropped:
        n_static = sum(1 for r in fully_dropped if r["motion"] == "static")
        n_moving = len(fully_dropped) - n_static
        print("--- FULLY DROPPED long runs (single-frame had >=%d consecutive dets, tracked kept 0) ---" % min_run)
        print(f"    static (net disp <= {static_px}px): {n_static}    moving: {n_moving}")
        print(f"{'sequence':<34}{'frames':>14}{'len':>6}{'score':>8}{'cx':>8}{'cy':>8}{'netpx':>8}{'motion':>9}")
        for rec in fully_dropped:
            print(
                f"{rec['sequence']:<34}{rec['start']:>7}-{rec['end']:<6}{rec['length']:>6}"
                f"{rec['score_at_start']:>8.3f}{rec['cx']:>8.1f}{rec['cy']:>8.1f}"
                f"{rec['net_disp_px']:>8.1f}{rec['motion']:>9}"
            )

    if partially:
        print(f"\n--- PARTIALLY DROPPED long runs (>50% of frames lost) : {len(partially)} ---")
        for rec in partially:
            print(
                f"{rec['sequence']:<34}{rec['start']:>7}-{rec['end']:<6}len={rec['length']:>5}"
                f" kept={rec['kept']:>5} ({rec['kept_ratio']:.0%}) score={rec['score_at_start']:.3f}"
            )

    by_seq = {r["sequence"]: r for r in seq_rows}
    print(f"\n{'sequence':<34}{'runs':>6}{'run=1':>8}{'run2-6':>9}{'run7+':>8}{'fullydrop':>11}{'partdrop':>10}")
    for name, rec in by_seq.items():
        b = rec["run_buckets"]
        print(
            f"{name:<34}{rec['runs']:>6}{b['1']:>8}{b['2-6']:>9}{b['7+']:>8}"
            f"{rec['fully_dropped_long_runs']:>11}{rec['partially_dropped_long_runs']:>10}"
        )

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(
                {
                    "min_run": min_run,
                    "n_sequences": len(seq_rows),
                    "fully_dropped_long_runs": fully_dropped,
                    "partially_dropped_long_runs": partially,
                    "sequences": seq_rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {json_out}")


if __name__ == "__main__":
    main()
