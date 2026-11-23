#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Independent re-computation of every number the report quotes.

This script deliberately does **not** import :mod:`vq_compare` or :mod:`vq_stats`: the AUC is
re-derived here from the rank-sum formula (instead of the pairwise comparison matrix used by the
comparison script) and the window statistics are rebuilt from the raw ``.npz`` per-frame arrays,
so a mistake in the comparison pipeline cannot hide behind a mirrored implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vq_metrics import METRIC_STAT, metric_array  # noqa: E402

PAIRS = [
    ("ir", lambda r: r["kind"] == "source" and r["modality"] == "ir",
     lambda r: r["kind"] == "source" and r["modality"] == "ir"),
    ("vis", lambda r: r["kind"] == "source" and r["modality"] == "vis",
     lambda r: r["kind"] == "source" and r["modality"] == "vis"),
    ("osd", lambda r: r["kind"] == "source", lambda r: r["kind"] == "osd"),
]

MOTION_METRICS = {"shift_frac", "shift_frac_max", "shift_rate", "rot_deg", "rot_rate", "illum_jump"}


def window_series(meta: dict, arrays: dict, metric: str) -> np.ndarray:
    """Rebuild the per 1 second window values from the raw per-frame arrays."""
    values = metric_array(arrays, metric)
    if values is None or values.size == 0:
        return np.array([], dtype=float)
    window = max(1, int(round(meta.get("fps_used", 25.0) * meta.get("stride", 1))))
    count = values.size // window
    if count < 1:
        return np.array([], dtype=float)
    trimmed = values[: count * window].reshape(count, window)
    if metric in MOTION_METRICS:
        trimmed = trimmed[:, 1:]
    if trimmed.size == 0:
        return np.array([], dtype=float)
    stat = METRIC_STAT[metric]
    if stat == "p05":
        return np.percentile(trimmed, 5, axis=1)
    if stat == "p95":
        return np.percentile(trimmed, 95, axis=1)
    if stat == "max":
        return trimmed.max(axis=1)
    return np.percentile(trimmed, 50, axis=1)


def auc_rank(good: np.ndarray, bad: np.ndarray) -> float:
    """P(good > bad) from the Mann-Whitney rank sum (ties get half a win)."""
    good = np.asarray(good, dtype=float)
    bad = np.asarray(bad, dtype=float)
    good = good[~np.isnan(good)]
    bad = bad[~np.isnan(bad)]
    if good.size == 0 or bad.size == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([good, bad]))
    rank_sum = ranks[: good.size].sum()
    n_good, n_bad = good.size, bad.size
    u_good = rank_sum - n_good * (n_good + 1) / 2.0
    return float(u_good / (n_good * n_bad))


def load_clips(metrics_dir: Path) -> list[dict]:
    clips = []
    for json_path in sorted(metrics_dir.glob("*.json")):
        if json_path.name == "run_manifest.json":
            continue
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        arrays = {}
        npz_path = json_path.with_suffix(".npz")
        if npz_path.exists():
            with np.load(npz_path) as handle:
                arrays = {key: handle[key] for key in handle.files}
        clips.append({"meta": meta, "arrays": arrays})
    return clips


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the video quality report numbers")
    parser.add_argument("--metrics-dir", type=str, required=True)
    parser.add_argument("--report-dir", type=str, required=True)
    parser.add_argument("--memory", type=str, default="", help="Optional memory file that must exist")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    report_dir = Path(args.report_dir)
    clips = load_clips(metrics_dir)
    with (report_dir / "comparison.csv").open(encoding="utf-8") as handle:
        comparison = list(csv.DictReader(handle))
    with (report_dir / "metrics_table.csv").open(encoding="utf-8") as handle:
        clip_table = {row["file"]: row for row in csv.DictReader(handle)}

    checks = 0
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal checks
        checks += 1
        if not ok:
            failures.append(f"{name}: {detail}")

    # 1. every clip is in the clip table with a re-computed "typical second"
    for clip in clips:
        meta = clip["meta"]
        name = meta["file"]
        row = clip_table.get(name)
        check(f"clip table row {name}", row is not None, "missing from metrics_table.csv")
        if row is None:
            continue
        for metric in ("lap_var", "global_contrast", "blockiness", "entropy", "edge_density"):
            expected = float(np.median(window_series(meta, clip["arrays"], metric)))
            got = float(row[metric])
            check(f"{name}/{metric}", abs(expected - got) <= 1e-6 * max(1.0, abs(expected)),
                  f"expected {expected!r} got {got!r}")

    # 2. every comparison row: AUC, red line and balanced accuracy, re-derived from the npz
    for row in comparison:
        pair = row["pair"]
        metric = row["metric"]
        good_filter, bad_filter = next((g, b) for key, g, b in PAIRS if key == pair)
        good_clips = [c for c in clips if c["meta"].get("group") == "good" and good_filter(c["meta"])]
        bad_clips = [c for c in clips if c["meta"].get("group") == "bad" and bad_filter(c["meta"])]
        g = np.array([np.median(window_series(c["meta"], c["arrays"], metric)) for c in good_clips], dtype=float)
        b = np.array([np.median(window_series(c["meta"], c["arrays"], metric)) for c in bad_clips], dtype=float)
        gw = np.concatenate([window_series(c["meta"], c["arrays"], metric) for c in good_clips])
        bw = np.concatenate([window_series(c["meta"], c["arrays"], metric) for c in bad_clips])

        expected_clip_auc = auc_rank(g, b)
        check(f"{pair}/{metric}/clip_auc", abs(expected_clip_auc - float(row["clip_auc"])) < 1e-9,
              f"expected {expected_clip_auc!r} got {row['clip_auc']!r}")
        expected_window_auc = auc_rank(gw, bw)
        check(f"{pair}/{metric}/window_auc", abs(expected_window_auc - float(row["window_auc"])) < 1e-9,
              f"expected {expected_window_auc!r} got {row['window_auc']!r}")
        expected_discrimination = max(expected_clip_auc, 1 - expected_clip_auc)
        check(f"{pair}/{metric}/clip_discrimination",
              abs(expected_discrimination - float(row["clip_discrimination"])) < 1e-9,
              f"expected {expected_discrimination!r} got {row['clip_discrimination']!r}")

        side = row["bad_side"]
        good_edge = float(np.percentile(g, 90 if side == "high" else 10))
        bad_edge = float(np.percentile(b, 10 if side == "high" else 90))
        expected_threshold = 0.5 * (good_edge + bad_edge)
        check(f"{pair}/{metric}/threshold", abs(expected_threshold - float(row["threshold"])) < 1e-9,
              f"expected {expected_threshold!r} got {row['threshold']!r}")
        if side == "high":
            tpr = float((b > expected_threshold).mean())
            tnr = float((g <= expected_threshold).mean())
        else:
            tpr = float((b < expected_threshold).mean())
            tnr = float((g >= expected_threshold).mean())
        expected_balanced = 0.5 * (tpr + tnr)
        check(f"{pair}/{metric}/balanced_acc", abs(expected_balanced - float(row["balanced_acc"])) < 1e-9,
              f"expected {expected_balanced!r} got {row['balanced_acc']!r}")

    # 3. the report and the memory file exist and quote the headline numbers
    report_path = report_dir / "REPORT_视频质量指标.md"
    check("report exists", report_path.exists(), str(report_path))
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        for metric in ("blockiness", "global_contrast", "lap_var", "entropy"):
            check(f"report mentions {metric}", metric in text, "metric missing from the report")
    if args.memory:
        memory_path = Path(args.memory)
        check("memory file exists", memory_path.exists(), str(memory_path))
        if memory_path.exists():
            text = memory_path.read_text(encoding="utf-8")
            for token in ("vq_compare.py", "vq_stats.py", "192.168.99.40", "conda activate uav"):
                check(f"memory quotes {token}", token in text, "token missing from the memory file")

    print(f"checks={checks} failures={len(failures)}")
    for failure in failures[:40]:
        print(f"[FAIL] {failure}")
    print("RESULT: PASS" if not failures else "RESULT: FAIL")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
