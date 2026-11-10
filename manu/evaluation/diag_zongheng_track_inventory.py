#!/usr/bin/env python3
"""Inventory the tracked tier: how many tracks survived, and which sequences ended with none.

The tracked tier applies a rigid-static pruning stage plus a consecutive-frame confirmation gate.
Both can silently delete *everything* on a sequence (e.g. a genuinely static pole that barely
moves between frames). For a manual-review tier that is fatal: the reviewer would see an empty
video and conclude "no target", not "our post-processing removed the target".

This script therefore prints, per sequence:
  * single-frame tier  : frames that had at least one raw detection (th=0.20)
  * tracked tier       : frames with a confirmed system detection, distinct track ids, longest track
  * the delta          : frames that the tracking pipeline removed

Sequences whose tracked tier is completely empty are listed separately, because they need a
human decision (static real target vs. frozen detector) before the tier can be trusted.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


def read_single(frames_dir: Path) -> dict[str, tuple[int, int, int]]:
    """{sequence: (frames_with_detection, total_frames, total_detections)} for the single-frame tier."""
    out: dict[str, tuple[int, int, int]] = {}
    for path in sorted(frames_dir.glob("*.jsonl")):
        with_det = total = n_det = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                total += 1
                dets = json.loads(line).get("detections") or []
                if dets:
                    with_det += 1
                    n_det += len(dets)
        out[path.stem] = (with_det, total, n_det)
    return out


def read_tracked(frames_dir: Path) -> dict[str, dict]:
    """Per-sequence tracked-tier statistics from the rendered jsonl records."""
    out: dict[str, dict] = {}
    for path in sorted(frames_dir.glob("*.jsonl")):
        total = with_det = n_det = n_infilled = 0
        track_len: dict[int, int] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                total += 1
                dets = json.loads(line).get("detections") or []
                if dets:
                    with_det += 1
                    n_det += len(dets)
                for det in dets:
                    tid = det.get("track_id")
                    if tid is not None:
                        track_len[tid] = track_len.get(tid, 0) + 1
                    if det.get("infilled"):
                        n_infilled += 1
        out[path.stem] = {
            "frames_with_detection": with_det,
            "total_frames": total,
            "n_detections": n_det,
            "n_infilled": n_infilled,
            "n_tracks": len(track_len),
            "longest_track": max(track_len.values()) if track_len else 0,
            "track_ids": sorted(track_len),
        }
    return out


def main() -> None:
    single_dir = Path(sys.argv[1])
    tracked_dir = Path(sys.argv[2])
    json_out = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    single = read_single(single_dir)
    tracked = read_tracked(tracked_dir)

    common = sorted(set(single) & set(tracked))
    missing = sorted(set(single) - set(tracked))

    header = (
        f"{'sequence':<34}{'single_f%':>10}{'track_f%':>10}{'tracks':>8}"
        f"{'longest':>9}{'infil':>7}{'lost_f':>8}"
    )
    print(f"single frames dir : {single_dir}")
    print(f"tracked frames dir: {tracked_dir}")
    print(f"single sequences : {len(single)}")
    print(f"tracked sequences: {len(tracked)}")
    print(f"common           : {len(common)}")
    if missing:
        print(f"not yet rendered  : {len(missing)} -> {', '.join(missing[:6])}"
              f"{' ...' if len(missing) > 6 else ''}\n")

    print()
    print(header)
    print("-" * len(header))

    empty: list[str] = []
    wiped: list[str] = []
    rows = []
    for name in common:
        s_hit, s_tot, s_det = single[name]
        t = tracked[name]
        s_pct = 100 * s_hit / max(s_tot, 1)
        t_pct = 100 * t["frames_with_detection"] / max(t["total_frames"], 1)
        lost = s_hit - t["frames_with_detection"]
        print(
            f"{name:<34}{s_pct:>9.1f}%{t_pct:>9.1f}%{t['n_tracks']:>8}"
            f"{t['longest_track']:>9}{t['n_infilled']:>7}{lost:>8}"
        )
        if t["n_tracks"] == 0:
            empty.append(name)
            if s_hit > 0:
                wiped.append(name)
        rows.append(
            {
                "sequence": name,
                "single_frames_with_detection": s_hit,
                "single_total_frames": s_tot,
                "single_detections": s_det,
                "tracked_frames_with_detection": t["frames_with_detection"],
                "tracked_total_frames": t["total_frames"],
                "tracked_detections": t["n_detections"],
                "tracked_n_tracks": t["n_tracks"],
                "tracked_longest_track": t["longest_track"],
                "tracked_n_infilled": t["n_infilled"],
                "frames_removed_by_tracking": lost,
            }
        )

    print()
    print(f"sequences with 0 confirmed track : {len(empty)}")
    for name in empty:
        s_hit = single[name][0]
        flag = "  <-- single tier had detections, tracking removed ALL" if s_hit > 0 else ""
        print(f"  {name:<34} single_frames_with_det={s_hit}{flag}")

    totals_single = sum(single[n][0] for n in common)
    totals_tracked = sum(tracked[n]["frames_with_detection"] for n in common)
    print()
    print(f"coverage  single : {totals_single} / {sum(single[n][1] for n in common)} frames")
    print(f"coverage  tracked: {totals_tracked} / {sum(tracked[n]['total_frames'] for n in common)} frames")
    print(f"frames removed by tracking pipeline: {totals_single - totals_tracked}")

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "single_frames_dir": str(single_dir),
            "tracked_frames_dir": str(tracked_dir),
            "n_common": len(common),
            "zero_track_sequences": empty,
            "zero_track_but_single_had_detections": wiped,
            "totals": {
                "single_frames_with_detection": totals_single,
                "tracked_frames_with_detection": totals_tracked,
                "frames_removed_by_tracking": totals_single - totals_tracked,
            },
            "sequences": rows,
        }
        json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {json_out}")


if __name__ == "__main__":
    main()
