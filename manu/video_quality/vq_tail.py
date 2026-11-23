#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Failure-slice comparison: bad clips' failure window vs good clips' global behaviour.

WHY THIS EXISTS
---------------
The reference method says a bad clip is often stable for most of its length and only falls
apart in a short window (shake / blur / noise burst).  Averaging over the whole clip dilutes
that failure and makes good and bad look alike.  The reference therefore recommends slicing
out the failure window and comparing it against the good clips' global statistics.

We do not have the "frame index where the tracker lost the target" annotation, so this script
computes two slices per bad clip:

``tail``  - the LAST ``--slice-seconds`` of the clip.  The slice is chosen WITHOUT looking at
            any metric, therefore it carries NO selection bias.  This is the headline number.
``worst`` - the ``--slice-seconds`` window with the largest shift_frac (shake).  It is chosen
            per clip by an instability metric, so every *shake* metric is biased upward by
            construction; the report must flag it as such.

Both are compared against the good clips' GLOBAL statistics, exactly as the reference asks.

Outputs (in --out):
  ``tail_slices.csv``  - one row per clip: global vs tail vs worst slice values.
  ``tail_compare.csv`` - one row per (pair, metric): good global median vs bad slice median,
                        the separation ratio and the reference method's verdict.
  ``tail_compare.json``- same content as JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vq_metrics import METRIC_DIRECTION, METRIC_LABEL_CN, METRIC_STAT, metric_array  # noqa: E402

# Metrics whose first sample is a placeholder (no previous frame -> displacement is 0).
NO_PREV = {"shift_frac", "shift_frac_max", "shift_rate", "rot_deg", "rot_rate", "illum_jump"}

PAIRS = [
    ("ir", "红外 IR (源视频)", lambda r: r["kind"] == "source" and r["modality"] == "ir",
     lambda r: r["kind"] == "source" and r["modality"] == "ir"),
    ("vis", "可见光 VIS (源视频)", lambda r: r["kind"] == "source" and r["modality"] == "vis",
     lambda r: r["kind"] == "source" and r["modality"] == "vis"),
    ("osd", "挂飞记录 OSD (地面站录屏)", lambda r: r["kind"] == "source",
     lambda r: r["kind"] == "osd"),
]


def quantile(values: np.ndarray, stat: str) -> float:
    if values.size == 0:
        return float("nan")
    if stat == "p05":
        return float(np.percentile(values, 5))
    if stat == "p95":
        return float(np.percentile(values, 95))
    if stat == "max":
        return float(values.max())
    return float(np.percentile(values, 50))


def slice_stat(values: np.ndarray, stat: str, metric: str, start: int, end: int) -> float:
    chunk = values[start:end]
    if chunk.size == 0:
        return float("nan")
    if metric in NO_PREV:
        chunk = chunk[1:]
    return quantile(chunk, stat)


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


def clip_slices(clip: dict, slice_seconds: float) -> dict:
    meta = clip["meta"]
    arrays = clip["arrays"]
    fps = float(meta.get("fps_used", 25.0)) * float(meta.get("stride", 1))
    slice_frames = max(1, int(round(fps * slice_seconds)))
    shift = arrays.get("shift_frac")
    n_frames = int(shift.size) if shift is not None else 0
    if n_frames == 0:
        for values in arrays.values():
            n_frames = max(n_frames, int(values.size))
    if n_frames == 0:
        return {}

    tail_start = max(0, n_frames - slice_frames)
    worst_start = 0
    if shift is not None and shift.size > slice_frames:
        step = max(1, slice_frames // 4)
        candidates = []
        for start in range(0, n_frames - slice_frames + 1, step):
            chunk = shift[start:start + slice_frames]
            if chunk.size > 1:
                candidates.append((float(np.percentile(chunk[1:], 95)), start))
        if candidates:
            worst_start = max(candidates)[1]

    out = {"n_frames": n_frames, "slice_frames": slice_frames,
           "tail_start": tail_start, "worst_start": worst_start}
    for metric in METRIC_DIRECTION:
        values = metric_array(arrays, metric)
        if values is None or values.size == 0:
            continue
        stat = METRIC_STAT[metric]
        out[metric] = {
            "global": slice_stat(values, stat, metric, 0, values.size),
            "tail": slice_stat(values, stat, metric, tail_start, values.size),
            "worst": slice_stat(values, stat, metric, worst_start,
                                min(values.size, worst_start + slice_frames)),
        }
    return out


def verdict(ratio: float) -> str:
    """Verdict from the *fold* distance between the failure window and the good baseline.

    The ratio is computed on the direction-aligned scale (larger = better), so the failure window
    can be either below 1 (the metric got worse) or above 1 (the metric got worse in the other
    direction).  The fold distance ``max(ratio, 1/ratio)`` folds both into one number >= 1.
    """
    if not np.isfinite(ratio) or ratio <= 0:
        return "无效"
    if ratio >= 2.0 or ratio <= 0.5:
        return "保留·强区分"
    if ratio >= 1.2 or ratio <= 0.83:
        return "参考·弱区分"
    return "剔除·无区分度"


def main() -> None:
    parser = argparse.ArgumentParser(description="Failure-slice (tail/worst) comparison")
    parser.add_argument("--metrics-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--slice-seconds", type=float, default=5.0)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = load_clips(metrics_dir)
    per_clip = []
    for clip in clips:
        meta = clip["meta"]
        slices = clip_slices(clip, args.slice_seconds)
        if not slices:
            continue
        row = {
            "file": meta.get("file", "?"),
            "group": meta.get("group", "?"),
            "kind": meta.get("kind", "?"),
            "modality": meta.get("modality", "?"),
            "fps": float(meta.get("fps_used", 0)),
            "n_frames": slices["n_frames"],
            "slice_frames": slices["slice_frames"],
            "tail_start_frame": slices["tail_start"],
            "worst_start_frame": slices["worst_start"],
        }
        for metric, values in slices.items():
            if isinstance(values, dict):
                row[f"{metric}__global"] = values["global"]
                row[f"{metric}__tail"] = values["tail"]
                row[f"{metric}__worst"] = values["worst"]
        per_clip.append(row)

    with (out_dir / "tail_slices.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in per_clip for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_clip)
    print(f"[INFO] per-clip slices -> {out_dir}/tail_slices.csv ({len(per_clip)} clips)")

    compare_rows = []
    for pair, title, good_filter, bad_filter in PAIRS:
        good_clips = [r for r in per_clip if r["group"] == "good" and good_filter(r)]
        bad_clips = [r for r in per_clip if r["group"] == "bad" and bad_filter(r)]
        if not good_clips or not bad_clips:
            print(f"[SKIP] {pair}: good={len(good_clips)} bad={len(bad_clips)}")
            continue
        for metric, direction in METRIC_DIRECTION.items():
            if direction == 0:
                continue
            stat = METRIC_STAT[metric]
            good_global = np.array([r.get(f"{metric}__global", np.nan) for r in good_clips], dtype=float) * direction
            bad_tail = np.array([r.get(f"{metric}__tail", np.nan) for r in bad_clips], dtype=float) * direction
            bad_worst = np.array([r.get(f"{metric}__worst", np.nan) for r in bad_clips], dtype=float) * direction
            g_med = float(np.nanmedian(good_global))
            t_med = float(np.nanmedian(bad_tail))
            w_med = float(np.nanmedian(bad_worst))
            ratio_tail = abs(t_med) / (abs(g_med) + 1e-9)
            ratio_worst = abs(w_med) / (abs(g_med) + 1e-9)
            fold_tail = max(ratio_tail, 1.0 / ratio_tail) if ratio_tail > 0 else float("nan")
            fold_worst = max(ratio_worst, 1.0 / ratio_worst) if ratio_worst > 0 else float("nan")
            compare_rows.append({
                "pair": pair,
                "pair_title": title,
                "metric": metric,
                "label_cn": METRIC_LABEL_CN.get(metric, metric),
                "stat": stat,
                "direction": direction,
                "n_good_clips": len(good_clips),
                "n_bad_clips": len(bad_clips),
                "good_global_median": g_med / direction,
                "bad_tail_median": t_med / direction,
                "bad_worst_median": w_med / direction,
                "ratio_tail": ratio_tail,
                "ratio_worst": ratio_worst,
                "fold_tail": fold_tail,
                "fold_worst": fold_worst,
                "verdict_tail": verdict(fold_tail),
                "verdict_worst": verdict(fold_worst),
                "note": "tail=切片无选择偏差; worst=按晃动选窗, 晃动类指标有构造性上偏",
            })

    if not compare_rows:
        print("[WARN] no comparable pair found")
        return

    compare_rows.sort(key=lambda r: (r["pair"], -r["ratio_tail"]))
    (out_dir / "tail_compare.json").write_text(
        json.dumps(compare_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "tail_compare.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compare_rows[0].keys()))
        writer.writeheader()
        writer.writerows(compare_rows)
    print(f"[INFO] slice comparison -> {out_dir}/tail_compare.csv / tail_compare.json")
    for row in compare_rows[:12]:
        print(f"[{row['pair']:<4}] {row['metric']:<22} good_global={row['good_global_median']:>12.4g} "
              f"bad_tail={row['bad_tail_median']:>12.4g} ratio_tail={row['ratio_tail']:.2f} "
              f"({row['verdict_tail']})")


if __name__ == "__main__":
    sys.exit(main())
