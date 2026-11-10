#!/usr/bin/env python3
"""Are the frames the tracked tier drops really one-frame flashes (false alarms) or real targets?

The tracker drops a candidate unless it becomes a confirmed track and then survives the
``confirm_consec`` consecutive-frame gate. So the decisive question is: are the dropped detections
isolated flashes, or are they part of long runs that the tracker should have kept?

This measures the **run-length distribution** of the single-frame detections: a detection run of
length 1 is a flash (very likely a false alarm), whereas a run of length >= confirm_consec is a
sustained target the tracker was supposed to keep. If the lost frames are dominated by short runs,
the tracker's deletions are the intended flash suppression; if they are dominated by long runs,
something is wrong with the wiring.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


def runs(frames: list[int]) -> list[int]:
    """Lengths of the maximal consecutive runs in a sorted frame list."""
    if not frames:
        return []
    lengths = []
    current = 1
    for index in range(1, len(frames)):
        if frames[index] == frames[index - 1] + 1:
            current += 1
        else:
            lengths.append(current)
            current = 1
    lengths.append(current)
    return lengths


def main() -> None:
    single_dir = Path(sys.argv[1])
    tracked_dir = Path(sys.argv[2])
    confirm_consec = int(sys.argv[3]) if len(sys.argv) > 3 else 7

    total_lost = 0
    lost_in_flash = 0
    lost_in_short = 0
    lost_in_long = 0
    kept = 0
    total_single = 0
    all_runs = []
    lost_runs = []

    for single_path in sorted(single_dir.glob("*.jsonl")):
        tracked_path = tracked_dir / single_path.name
        if not tracked_path.exists():
            continue
        single_hits, tracked_hits = set(), set()
        with single_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("detections"):
                    single_hits.add(record["frame"])
        with tracked_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("detections"):
                    tracked_hits.add(record["frame"])
        total_single += len(single_hits)
        kept += len(tracked_hits)
        lost = single_hits - tracked_hits
        total_lost += len(lost)

        single_runs = runs(sorted(single_hits))
        all_runs.extend(single_runs)
        # Which runs lost frames? A run is "kept" if ANY of its frames survived.
        lost_runs.append((len(lost), len(single_runs)))

        # Attribute each lost frame by the length of the run it belongs to.
        for run_len in single_runs:
            pass
        # Rebuild runs with their frame spans so we can attribute per frame.
        spans = []
        if single_hits:
            start = prev = min(single_hits)
            for f in sorted(single_hits)[1:]:
                if f == prev + 1:
                    prev = f
                else:
                    spans.append((start, prev))
                    start = prev = f
            spans.append((start, prev))
        for start, end in spans:
            run_len = end - start + 1
            lost_in_run = sum(1 for f in range(start, end + 1) if f in lost)
            if lost_in_run == 0:
                continue
            if run_len == 1:
                lost_in_flash += lost_in_run
            elif run_len < confirm_consec:
                lost_in_short += lost_in_run
            else:
                lost_in_long += lost_in_run
            lost_runs.append((run_len, lost_in_run))

    print(f"single-frame detections : {total_single} frames")
    print(f"tracked detections     : {kept} frames")
    print(f"dropped by tracking    : {total_lost} frames ({total_lost/max(total_single,1):.1%})")
    print()
    print("attribution of the DROPPED frames by the length of the run they belonged to:")
    print(f"  run length 1 (isolated flash)      : {lost_in_flash:>7} frames ({lost_in_flash/max(total_lost,1):>5.1%})")
    print(f"  run length 2..{confirm_consec-1} (short run)   : {lost_in_short:>7} frames ({lost_in_short/max(total_lost,1):>5.1%})")
    print(f"  run length >= {confirm_consec} (sustained)     : {lost_in_long:>7} frames ({lost_in_long/max(total_lost,1):>5.1%})")
    print()
    if all_runs:
        hist: dict[int, int] = {}
        for length in all_runs:
            hist[length] = hist.get(length, 0) + 1
        print("run-length histogram of ALL single-frame detections:")
        for length in sorted(hist)[:12]:
            print(f"   run={length:>4}: {hist[length]:>6} runs")
        flash_runs = sum(v for k, v in hist.items() if k == 1)
        print(f"  => isolated flashes (run=1): {flash_runs}/{len(all_runs)} runs "
              f"({flash_runs/max(len(all_runs),1):.1%})")


if __name__ == "__main__":
    main()
