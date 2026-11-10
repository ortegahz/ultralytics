#!/usr/bin/env python3
"""Compare the tracked tier against the th=0.20 single-frame tier, sequence by sequence.

The tracked tier is only worth its extra cost if it *adds* coverage (trajectory stitching recovers
intermittent detections) without silently *losing* them. This prints both, plus the delta, so the
sequences where tracking removed detections are visible instead of averaged away.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


def coverage(frames_dir: Path) -> dict[str, tuple[int, int]]:
    """Return {sequence: (frames_with_detection, total_frames)} for one tier."""
    out = {}
    for path in sorted(frames_dir.glob("*.jsonl")):
        with_det = total = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                total += 1
                if json.loads(line).get("detections"):
                    with_det += 1
        out[path.stem] = (with_det, total)
    return out


def main() -> None:
    single_dir = Path(sys.argv[1])
    tracked_dir = Path(sys.argv[2])
    single = coverage(single_dir)
    tracked = coverage(tracked_dir)

    common = sorted(set(single) & set(tracked))
    print(f"single tier : {len(single)} sequences")
    print(f"tracked tier: {len(tracked)} sequences")
    print(f"common      : {len(common)}\n")

    header = f"{'sequence':<34} {'single':>12} {'tracked':>12} {'delta':>9}"
    print(header)
    print("-" * len(header))

    gained, lost, same = [], [], []
    for name in common:
        s_hit, s_tot = single[name]
        t_hit, t_tot = tracked[name]
        s_pct = 100 * s_hit / max(s_tot, 1)
        t_pct = 100 * t_hit / max(t_tot, 1)
        delta = t_pct - s_pct
        print(f"{name:<34} {s_pct:>11.1f}% {t_pct:>11.1f}% {delta:>+8.1f}")
        (gained if delta > 1 else lost if delta < -1 else same).append((name, delta))

    print()
    print(f"gained >1pp : {len(gained)}   lost >1pp : {len(lost)}   ~unchanged : {len(same)}")
    if lost:
        print("\n=== tracking REMOVED coverage (need a look) ===")
        for name, delta in sorted(lost, key=lambda kv: kv[1]):
            print(f"  {name:<34} {delta:>+7.1f} pp")
    if gained:
        print("\n=== tracking ADDED coverage (stitching/infill working) ===")
        for name, delta in sorted(gained, key=lambda kv: -kv[1])[:15]:
            print(f"  {name:<34} {delta:>+7.1f} pp")

    # Sequences the tracked tier emptied completely: the rigid-static pruner may have cut a real target.
    emptied = [n for n in common if tracked[n][0] == 0 and single[n][0] > 0]
    if emptied:
        print(f"\n=== tracked to ZERO but single-frame had detections ({len(emptied)}) ===")
        for name in emptied:
            print(f"  {name}: single had {single[name][0]}/{single[name][1]} frames")


if __name__ == "__main__":
    main()
