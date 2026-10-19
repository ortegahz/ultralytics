#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Per-sequence (merge radius x threshold) hyperparameter search over a multiscale cache.

`eval_multiscale_cache_th.py` sweeps the radius/threshold grid once to pick ONE global working point, then
reports per-sequence numbers only at that global radius with a per-sequence threshold search. That answers
"where does the whole domain sit", not "what can each sequence actually reach" -- the two differ a lot when a
domain mixes a few threshold-limited sequences with several capability-limited ones.

This script reuses that module's own ``load_ground_truth`` / ``load_predictions`` / ``merge_scales`` /
``count_matches`` / ``metrics`` rather than reimplementing them, so the per-sequence search and the headline
benchmark table are guaranteed to run identical operators (see the "no second implementation" rule in
memory.md 5d). Nothing here re-runs inference; it only reads the cached sparse detections.

Reported per sequence:
  * ``best``       -- the best (merge_radius, th) for that sequence alone, i.e. its reachable ceiling
  * ``at_global``  -- the same sequence measured at the global working point, for the gap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.eval_multiscale_cache_th import (  # noqa: E402
    build_frames,
    count_matches,
    load_ground_truth,
    load_predictions,
    merge_scales,
    metrics,
    resolve_sources,
)

FIELDS = ("tp", "fp", "fn", "fn_absent", "fn_compete", "det")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-root", required=True, help="Directory of <sequence>.pkl written by the cache script")
    parser.add_argument("--labels-root", required=True, help="Flat Final_Labels directory of {sequence}__{index:06d}.txt")
    parser.add_argument("--tag", default="all", help="'all' = every scale row, '255' = baked fused row, or one index")
    parser.add_argument("--match-mode", default="inbox", choices=["dist", "inbox"])
    parser.add_argument("--tol", type=float, default=8.0, help="Matching tolerance in native pixels, dist mode only")
    parser.add_argument("--radii", default="0,16,32,48,64,96,128,192", help="Comma-separated merge radii to search")
    parser.add_argument("--th-min", type=float, default=0.0)
    parser.add_argument("--th-max", type=float, default=0.90)
    parser.add_argument("--th-step", type=float, default=0.01)
    parser.add_argument(
        "--global-point",
        default="",
        help="Reference 'radius,th' to report the at_global column against, e.g. 64,0.30. Empty disables it",
    )
    parser.add_argument("--out-json", default="", help="Optional JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root)
    labels_root = Path(args.labels_root)
    if not cache_root.is_dir():
        raise NotADirectoryError(cache_root)
    if not labels_root.is_dir():
        raise NotADirectoryError(labels_root)

    thresholds = np.round(np.arange(args.th_min, args.th_max + 1e-9, args.th_step), 6)
    radii = sorted({float(item) for item in args.radii.split(",") if item.strip()})
    sequences = sorted(path.stem for path in cache_root.glob("*.pkl"))
    if not sequences:
        raise FileNotFoundError(f"No .pkl caches under {cache_root}")

    started = time.perf_counter()
    print(f"[SEARCH] sequences={len(sequences)} | radii={radii} | thresholds={len(thresholds)} | mode={args.match_mode}", flush=True)

    # One merged frame list per (sequence, radius); the threshold sweep then reuses it instead of re-merging.
    merged: dict[tuple[str, float], list] = {}
    gt_frames: dict[str, int] = {}
    total_frames: dict[str, int] = {}
    fused_tag, scale_tags = 255, []

    for position, sequence in enumerate(sequences, 1):
        cache = pickle.loads((cache_root / f"{sequence}.pkl").read_bytes())
        cache_meta = cache["meta"]
        fused_tag = cache_meta.get("fused_tag", 255)
        _, by_tag = load_predictions(cache_root, sequence, _unique_tags(cache), args.th_min)
        if not scale_tags:
            scale_tags = [tag for tag in range(len(cache_meta["scales"])) if tag in by_tag]
        truth = load_ground_truth(labels_root, sequence, cache_meta["native_size"][0], cache_meta["native_size"][1])
        total_frames[sequence] = cache_meta["frames"]
        gt_frames[sequence] = sum(1 for boxes in truth.values() if boxes is not None and len(boxes))
        sources = dict(enumerate(resolve_sources(by_tag, args.tag, fused_tag, scale_tags)))
        for radius in radii:
            merged[(sequence, radius)] = build_frames(truth, merge_scales(sources, radius))
        if position <= 3 or position == len(sequences):
            print(f"  [{position}/{len(sequences)}] {sequence} | frames={cache_meta['frames']} | gt_frames={gt_frames[sequence]}", flush=True)

    rows = []
    for sequence in sequences:
        best = None
        for radius in radii:
            frames = merged[(sequence, radius)]
            for threshold in thresholds:
                row = metrics(count_matches(frames, float(threshold), args.tol, args.match_mode), total_frames[sequence])
                row["th"] = float(threshold)
                row["merge_radius"] = radius
                # tie-break toward fewer detections: same F1 with a cleaner point beats the same F1 with a messier one
                if best is None or (row["f1"], row["precision"], -row["det"]) > (best["f1"], best["precision"], -best["det"]):
                    best = row
        record = {
            "sequence": sequence,
            "frames": total_frames[sequence],
            "gt_frames": gt_frames[sequence],
            "best": best,
        }
        if args.global_point:
            radius, threshold = (float(v) for v in args.global_point.split(","))
            at_global = metrics(
                count_matches(merged[(sequence, radius)], threshold, args.tol, args.match_mode), total_frames[sequence]
            )
            at_global["th"] = threshold
            at_global["merge_radius"] = radius
            record["at_global"] = at_global
        rows.append(record)
        extra = ""
        if args.global_point:
            extra = (
                f" | global@{record['at_global']['merge_radius']:.0f}/{record['at_global']['th']:.2f} "
                f"F1={record['at_global']['f1'] * 100:.2f}% ΔF1={(record['at_global']['f1'] - best['f1']) * 100:+.2f}"
            )
        print(
            f"{sequence:<32} frames={total_frames[sequence]:>6} gt={gt_frames[sequence]:>6} | "
            f"best r={best['merge_radius']:>5.0f} th={best['th']:>5.2f} "
            f"R={best['recall'] * 100:6.2f}% P={best['precision'] * 100:6.2f}% F1={best['f1'] * 100:7.3f}% "
            f"FP={best['fp']:>6} FN={best['fn']:>6} FAR={best['far_per_frame']:.4f}{extra}",
            flush=True,
        )

    print(
        f"\n[TOTAL] 序列数={len(rows)} | "
        f"逐序列最优 F1 之和/平均 = {sum(r['best']['f1'] for r in rows) * 100 / len(rows):.3f}% (平均, 非全局指标)",
        flush=True,
    )

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "config": {
                        "cache_root": str(cache_root),
                        "labels_root": str(labels_root),
                        "tag": args.tag,
                        "match_mode": args.match_mode,
                        "radii": radii,
                        "th_min": args.th_min,
                        "th_max": args.th_max,
                        "th_step": args.th_step,
                        "global_point": args.global_point,
                    },
                    "per_sequence": rows,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[SUCCESS] JSON -> {out.resolve()}", flush=True)
    print(f"[SUCCESS] elapsed {time.perf_counter() - started:.1f}s", flush=True)


def _unique_tags(cache: dict) -> list[int]:
    """Distinct cached row tags, i.e. the scale indices actually present plus the fused row."""
    return [int(value) for value in np.unique(cache["tag"])]


if __name__ == "__main__":
    main()
