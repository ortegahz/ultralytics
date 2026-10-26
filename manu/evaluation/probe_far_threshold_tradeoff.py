#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""FAR / F1 trade-off probe on an existing Longquanshan heatmap cache.

What it does
------------
The system-level delivery criterion is a **hard triple** (F1 >= 88 %, Recall >= 85 %,
FAR <= 0.030 per frame).  A model can raise F1 and still miss the FAR bound, and the usual
reaction -- tightening the global threshold -- costs recall on every other sequence.

This probe answers one question with numbers instead of guesses:

    *Along one axis (or a small grid of axes), what does it cost to get FAR back under the
    bound, and does the triple still hold at that point?*

Why it is cheap
---------------
The swept parameters are **post-processing settings**: they never touch the heatmap cache, the
ground truth or the model.  ``build_records`` produces the *same* candidate pool for every
parameter combination that leaves the de-duplication radius and the persistence filter alone,
so the records are built **once per record-affecting configuration** and reused for every
candidate.  Only the tracker / smoother run is repeated.  No GPU, no re-inference.

It also reports the two sequences that dominate the FAR budget, so it stays obvious whether a
parameter move actually touches the offending sequence or just trims healthy ones.

Usage
-----
    # one axis
    python manu/evaluation/probe_far_threshold_tradeoff.py \
        --cache-root runs/optuna_mix_ratio/trials/trial_0080/lq_eval/ms_cache \
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
        --params-json manu/evaluation/system_sota_longquan_delivery_params.json \
        --scan th_base=0.40,0.45,0.50 \
        --out-json runs/optuna_mix_ratio/trials/trial_0080/lq_eval/far_tradeoff_probe.json

    # two axes (grid)
    python manu/evaluation/probe_far_threshold_tradeoff.py ... \
        --scan th_base=0.42,0.45,0.48 --scan confirm_consec=5,6,7

    # robustness: perturb the delivery point one axis at a time around it
    python manu/evaluation/probe_far_threshold_tradeoff.py ... \
        --params-json manu/evaluation/system_sota_longquan_trial0080_delivery_params.json \
        --scan th_base=0.42,0.45,0.48 --scan th_ground=0.18,0.22,0.26
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.system_sota_longquan import (  # noqa: E402
    IMG_H,
    aggregate_metrics,
    build_records,
    evaluate_biz,
    load_cache_meta,
    run_system_sota,
)
from manu.evaluation.eval_multiscale_cache_th import COUNT_KEYS  # noqa: E402

TARGET_F1 = 0.88
TARGET_RECALL = 0.85
TARGET_FAR = 0.030

# The sequences that carry the FAR budget in the trial_0080 measurement (2026-10-22).
WATCH = ("VIDEO00012_19700101_002251", "VIDEO00006_19700101_000322")

# The parameters that change the candidate pool itself: changing one of them means the records
# have to be rebuilt, every other parameter can reuse them.
RECORD_AFFECTING = ("dedup_radius", "persist_min_hits", "persist_window", "persist_tol")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FAR / F1 trade-off probe over post-processing parameters")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--params-json", required=True)
    parser.add_argument("--scan", action="append", default=[], metavar="KEY=V1,V2,...",
                        help="Axis to sweep; repeatable, the axes form a grid")
    parser.add_argument("--seqs", default="", help="Optional comma separated sequence subset")
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def parse_scan(entries: list[str]) -> dict[str, list]:
    """Turn ``["th_base=0.4,0.45", "confirm_consec=5,6"]`` into ``{"th_base": [...], ...}``."""
    axes: dict[str, list] = {}
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"--scan must look like KEY=V1,V2 -- got {entry!r}")
        key, _, values = entry.partition("=")
        key = key.strip()
        parsed = []
        for raw in values.split(","):
            raw = raw.strip()
            if not raw:
                continue
            parsed.append(int(raw) if raw.lstrip("-").isdigit() else float(raw))
        if not parsed:
            raise SystemExit(f"--scan {key}= has no value")
        axes[key] = parsed
    return axes


def build_combinations(base: dict, axes: dict[str, list]) -> list[tuple[str, dict]]:
    """Cartesian product of the axes; the label names the axes that actually differ from the base."""
    if not axes:
        return [("base", dict(base))]
    keys = list(axes)
    combos = []
    for values in _product([axes[key] for key in keys]):
        params = dict(base)
        params.update(dict(zip(keys, values)))
        changed = {key: params[key] for key in keys if params[key] != base.get(key)}
        combos.append((",".join(f"{k}={v}" for k, v in changed.items()) or "base", params))
    return combos


def _product(pools: list[list]):
    if not pools:
        yield ()
        return
    for head in pools[0]:
        for tail in _product(pools[1:]):
            yield (head,) + tail


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root)
    labels_root = Path(args.labels_root)

    base = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
    params = dict(base.get("params", base))
    axes = parse_scan(args.scan)
    unknown = [key for key in axes if key not in params]
    if unknown:
        raise SystemExit(f"--scan names parameters that are not in the params file: {unknown}")

    combinations = build_combinations(params, axes)

    if args.seqs:
        sequences = [name.strip() for name in args.seqs.split(",") if name.strip()]
    else:
        sequences = sorted(path.stem for path in cache_root.glob("*.pkl"))
    if not sequences:
        raise SystemExit(f"no cache under {cache_root}")

    # One counter set per candidate; each holds the summed per-sequence counts.
    counters = {label: {key: 0 for key in COUNT_KEYS} for label, _ in combinations}
    per_sequence = {label: {} for label, _ in combinations}
    num_frames = 0

    started = time.perf_counter()
    for position, sequence in enumerate(sequences, 1):
        # The records only depend on the record-affecting parameters, so they are built once per
        # distinct record configuration and reused for every other candidate.
        record_groups: dict[tuple, list[tuple[str, dict]]] = {}
        for label, trial in combinations:
            key = tuple(trial[name] for name in RECORD_AFFECTING)
            record_groups.setdefault(key, []).append((label, trial))

        for key, members in record_groups.items():
            sample = members[0][1]
            records = build_records(
                cache_root,
                labels_root,
                sequence,
                float(sample["dedup_radius"]),
                0.0,
                int(sample["persist_min_hits"]),
                int(sample["persist_window"]),
                float(sample["persist_tol"]),
            )
            frames = load_cache_meta(cache_root, sequence)["frames"]
            num_frames += frames

            for label, trial in members:
                dets, _ = run_system_sota(records, trial, img_h=IMG_H)
                row = evaluate_biz(records, dets, num_frames=frames)
                per_sequence[label][sequence] = row
                for name in COUNT_KEYS:
                    counters[label][name] += int(row[name])

        print(f"[{position}/{len(sequences)}] {sequence} frames={frames}", flush=True)

    report = {
        "cache_root": str(cache_root),
        "labels_root": str(labels_root),
        "base_params": params,
        "axes": axes,
        "criteria": {"f1_min": TARGET_F1, "recall_min": TARGET_RECALL, "far_max": TARGET_FAR},
        "num_frames": num_frames,
        "overall": {},
        "per_sequence": {label: per_sequence[label] for label, _ in combinations},
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    for label, _ in combinations:
        report["overall"][label] = aggregate_metrics([per_sequence[label][s] for s in sequences], num_frames)

    print()
    print("=" * 110)
    header = f"{'candidates':<34} {'F1%':>9} {'R%':>8} {'P%':>8} {'FP':>7} {'FAR':>9} " \
             f"{'F1>=88':>7} {'R>=85':>6} {'FAR<=.03':>9} {'002251FP':>9}"
    print(header)
    print("-" * 110)
    rows = []
    for label, _ in combinations:
        row = report["overall"][label]
        f1_ok = row["f1"] >= TARGET_F1
        r_ok = row["recall"] >= TARGET_RECALL
        far_ok = row["far_per_frame"] <= TARGET_FAR
        w1 = per_sequence[label].get(WATCH[0], {}).get("fp", 0)
        rows.append((label, row, f1_ok, r_ok, far_ok, w1))
    # The best triple-passing candidate first, then by F1.
    rows.sort(key=lambda item: (not (item[2] and item[3] and item[4]), -item[1]["f1"]))
    for label, row, f1_ok, r_ok, far_ok, w1 in rows:
        flag = "  <== 三条全过" if (f1_ok and r_ok and far_ok) else ""
        print(f"{label:<34} {row['f1'] * 100:>9.4f} {row['recall'] * 100:>8.3f} "
              f"{row['precision'] * 100:>8.3f} {row['fp']:>7} {row['far_per_frame']:>9.5f} "
              f"{'OK' if f1_ok else 'FAIL':>7} {'OK' if r_ok else 'FAIL':>6} "
              f"{'OK' if far_ok else 'FAIL':>9} {w1:>9}{flag}")
    print("=" * 110)
    print(f"elapsed {report['elapsed_seconds']}s | frames={num_frames} | sequences={len(sequences)} "
          f"| candidates={len(combinations)}")

    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report -> {path.resolve()}")


if __name__ == "__main__":
    main()
