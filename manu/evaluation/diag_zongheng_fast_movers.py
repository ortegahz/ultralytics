#!/usr/bin/env python3
"""Quantify how much of the tracked tier's coverage loss is *structural* (gate-limited).

The Longquan tracker associates a detection with an existing track only when the distance
is <= min(max_match_dist, match_dist + 0.5 * |v|).  With the delivery-v3 parameters that is a
hard ceiling of `max_match_dist` = 24 px per frame.

Any single-frame detection run whose typical per-frame step exceeds that ceiling can never
be linked into one track, so it can never reach `confirm_consec` = 7 and is deleted outright
by the tracked tier -- no matter how confident the detector was.

This script measures that directly:
  * single-frame runs are chased with nearest-neighbour chaining
  * the median per-frame step of each run decides whether it is *linkable* or *un-linkable*
  * the frame cost of each category is reported, plus how much of the total tracked-tier
    coverage loss it explains

`--params` points at the Longquan delivery-v3 parameter file, so the gate is read from the
frozen upstream config instead of being hard-coded here.
"""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys


def load_dets(path: Path) -> list[list[dict]]:
    out = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line).get("detections") or [])
    return out


def runs_of(hits: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = prev = None
    for idx in hits:
        if prev is not None and idx == prev + 1:
            prev = idx
            continue
        if start is not None:
            runs.append((start, prev))
        start = prev = idx
    if start is not None:
        runs.append((start, prev))
    return runs


def chain(single: list[list[dict]], start: int, end: int) -> list[tuple[float, float]]:
    """Nearest-neighbour chain of positions through a run."""
    prev = None
    pts: list[tuple[float, float]] = []
    for idx in range(start, end + 1):
        dets = single[idx]
        if not dets:
            continue
        if prev is None:
            cur = dets[0]
        else:
            cur = min(dets, key=lambda d: (d["cx"] - prev[0]) ** 2 + (d["cy"] - prev[1]) ** 2)
        px, py = float(cur["cx"]), float(cur["cy"])
        pts.append((px, py))
        prev = (px, py)
    return pts


def median_step(pts: list[tuple[float, float]]) -> float:
    if len(pts) < 2:
        return 0.0
    steps = [
        ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in zip(pts, pts[1:])
    ]
    return statistics.median(steps)


def main() -> None:
    single_dir = Path(sys.argv[1])
    tracked_dir = Path(sys.argv[2])
    params_path = Path(sys.argv[3])
    json_out = Path(sys.argv[4]) if len(sys.argv) > 4 else None

    params = json.loads(params_path.read_text(encoding="utf-8")).get("params", {})
    gate = float(params.get("max_match_dist", 24.0))
    base_gate = float(params.get("match_dist", 16.0))
    confirm_consec = int(params.get("confirm_consec", 7))

    unlinkable: list[dict] = []
    linkable_long: list[dict] = []
    seq_rows: list[dict] = []

    total_single_hit = 0
    total_tracked_hit = 0
    total_frames = 0
    frames_unlinkable = 0
    frames_short = 0

    for path in sorted(single_dir.glob("*.jsonl")):
        tp = tracked_dir / path.name
        if not tp.exists():
            continue
        single = load_dets(path)
        tracked = load_dets(tp)
        n = min(len(single), len(tracked))
        single, tracked = single[:n], tracked[:n]
        total_frames += n
        hits = [i for i, d in enumerate(single) if d]
        thr = [bool(d) for d in tracked]
        total_single_hit += len(hits)
        total_tracked_hit += sum(thr)

        seq_unlinkable = 0
        seq_short = 0
        for start, end in runs_of(hits):
            length = end - start + 1
            step = median_step(chain(single, start, end))
            rec = {
                "sequence": path.stem,
                "start": start,
                "end": end,
                "length": length,
                "median_step_px": round(step, 2),
                "kept": sum(1 for i in range(start, end + 1) if thr[i]),
            }
            if length < confirm_consec:
                seq_short += length
                frames_short += length
                continue
            if step > gate:
                seq_unlinkable += length
                frames_unlinkable += length
                unlinkable.append(rec)
            else:
                linkable_long.append(rec)

        seq_rows.append(
            {
                "sequence": path.stem,
                "frames": n,
                "single_hit_frames": len(hits),
                "tracked_hit_frames": sum(thr),
                "frames_removed": len(hits) - sum(thr),
                "frames_in_unlinkable_runs": seq_unlinkable,
                "frames_in_short_runs": seq_short,
            }
        )

    removed = total_single_hit - total_tracked_hit
    print(f"gate: max_match_dist={gate}px  (match_dist={base_gate}px)  confirm_consec={confirm_consec}")
    print(f"sequences: {len(seq_rows)}   frames: {total_frames}")
    print(f"single-tier hit frames : {total_single_hit}")
    print(f"tracked-tier hit frames: {total_tracked_hit}")
    print(f"frames removed by tracking: {removed}\n")
    print(f"  frames inside runs shorter than confirm_consec ({confirm_consec}) : {frames_short}")
    print(f"  frames inside runs whose median step > gate ({gate}px)        : {frames_unlinkable}")
    print(f"  residual (linkable long runs that still lost frames)           : "
          f"{removed - frames_short - frames_unlinkable}")
    if removed:
        print(f"\nun-linkable share of total loss: "
              f"{frames_unlinkable / removed:.1%}   short-run share: {frames_short / removed:.1%}")

    if unlinkable:
        print(f"\n--- un-linkable runs (>= {confirm_consec} frames, median step > {gate}px) : "
              f"{len(unlinkable)} runs, {frames_unlinkable} frames ---")
        print(f"{'sequence':<34}{'frames':>14}{'len':>6}{'step_px':>10}{'kept':>7}")
        for rec in sorted(unlinkable, key=lambda r: -r["length"])[:40]:
            print(
                f"{rec['sequence']:<34}{rec['start']:>7}-{rec['end']:<6}{rec['length']:>6}"
                f"{rec['median_step_px']:>10.1f}{rec['kept']:>7}"
            )

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(
                {
                    "gate_px": gate,
                    "base_gate_px": base_gate,
                    "confirm_consec": confirm_consec,
                    "n_sequences": len(seq_rows),
                    "total_frames": total_frames,
                    "single_hit_frames": total_single_hit,
                    "tracked_hit_frames": total_tracked_hit,
                    "frames_removed": removed,
                    "frames_in_short_runs": frames_short,
                    "frames_in_unlinkable_runs": frames_unlinkable,
                    "unlinkable_runs": unlinkable,
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
