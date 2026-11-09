#!/usr/bin/env python3
"""Sweep the merged-row score threshold on the real Longquan RC2 cache (CPU-only, no re-inference).

This is the measurement behind every working point chosen for this project. It reuses the shipped
evaluation helpers so the numbers are the same ones the delivery pipeline would produce.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp/pycharm_project_10ae9e2e")

from manu.evaluation.eval_multiscale_cache_th import load_predictions, merge_scales  # noqa: E402
from manu.evaluation.system_sota_longquan import (  # noqa: E402
    aggregate_metrics,
    build_records,
    evaluate_biz,
    single_frame_dets,
)

CACHE = Path("/tmp/pycharm_project_10ae9e2e/runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache")
LABELS = Path("/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010")
RADIUS = 768.0
THRESHOLDS = [0.16, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.44]


def main() -> None:
    sequences = sorted(path.stem for path in CACHE.glob("*.pkl"))
    total_frames = 0
    records = {}
    for sequence in sequences:
        meta, by_tag = load_predictions(CACHE, sequence, [0, 1, 2], 0.0)
        merged = merge_scales(by_tag, RADIUS)
        total_frames += int(meta["frames"])
        # build_records builds the candidate pool from the same cache and merge radius, so the
        # single-frame arm below sees exactly the rows the delivery pipeline would score.
        records[sequence] = build_records(CACHE, LABELS, sequence, RADIUS, 0.0, 0, 0, 4.0)
        # The merged rows are re-injected so the threshold is applied to the *same* pool the
        # delivery arm would use (per-scale rows, cross-scale dedup at RADIUS).
        for record in records[sequence]:
            frame_index = int(record["frame_index"])
            scores, points = merged.get(frame_index, (np.zeros(0), np.zeros((0, 2))))
            record["pred_scores"] = np.asarray(scores, dtype=np.float32)
            record["pred_points"] = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    header = "th      TP       FP      FN   Recall    Prec      F1    FAR/frame"
    print(header)
    print("-" * len(header))
    for threshold in THRESHOLDS:
        tp = fp = fn = 0
        for sequence in sequences:
            dets = single_frame_dets(records[sequence], threshold)
            row = evaluate_biz(records[sequence], dets, num_frames=0)
            tp += int(row["tp"])
            fp += int(row["fp"])
            fn += int(row["fn"])
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        far = fp / max(total_frames, 1)
        print(
            f"{threshold:5.2f} {tp:>8} {fp:>6} {fn:>7} {recall:>8.2%} {precision:>7.2%} {f1:>7.2%} {far:>9.4f}"
        )


if __name__ == "__main__":
    main()
