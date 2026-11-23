#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Grid trackability vs the quality red lines of report/REPORT_视频质量指标.md.

Reads the per-clip grid artefacts (``vq_grid.py``) plus the per-second quality metrics
(``report/windows.csv``) and produces

  * ``grid_summary.csv``   - one row per clip: grid statistics + the red-line tier of the clip
  * ``grid_windows.csv``   - one row per (clip, 1 second window): the grid statistics of that
                             second next to the quality metrics of the same second
  * ``grid_compare.csv``   - AUC of every grid metric for good/bad (clip and window level) and
                             for "the red lines of this second are violated" (window level)
  * ``grid_crosstab.csv``  - red-line tier x grid statistic (the cross table of the task)
  * ``grid_thresholds.json`` - the per-cell detail threshold and its derivation

Definitions used here
  TGR_raw  - the forward/backward template matching closure test of the task document, on the
             raw grey picture ("how often does an arbitrary click stay locked").
  TGR_dog  - the same test on a band-pass (DoG) picture plus a peak-uniqueness test.
  DET      - the fraction of cells whose per-cell Laplacian variance reaches the detail floor
             tau_det ("how much of the picture has usable fine detail at all").
  UGR      - DET and TGR_dog at the same time ("usable grid ratio": the share of the picture the
             operator could both lock onto and keep locked).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vq_grid import (  # noqa: E402
    DOG_CLOSURE_THRESH,
    DOG_NCC_THRESH,
    DOG_UNIQ_THRESH,
    RAW_CLOSURE_THRESH,
    RAW_NCC_THRESH,
    LazyArrays,
    decide,
)

# --- red lines of report/REPORT_视频质量指标.md section 3 -------------------------
# (metric, direction) - "low" means "a smaller value is worse"
CORE_LINES = {
    "ir": [("blockiness", "high"), ("entropy", "low"), ("global_contrast", "low"),
           ("edge_density", "low"), ("lap_var", "low")],
    "vis": [("lap_var", "low"), ("edge_density", "low"), ("global_contrast", "low")],
}
CORE_THRESH = {
    "ir": {"blockiness": 1.631, "entropy": 6.417, "global_contrast": 0.4693,
           "edge_density": 0.02016, "lap_var": 57.32},
    "vis": {"lap_var": 534.0, "edge_density": 0.03637, "global_contrast": 0.4756},
}
WEAK_LINES = {"vis": [("illum_jump", "high"), ("stripe_idx_row", "high")]}
WEAK_THRESH = {"vis": {"illum_jump": 0.00168, "stripe_idx_row": 33.53}}

GRID_METRICS = [
    "tgr_raw", "tgr_dog", "det_ratio", "ugr", "ugr_raw",
    "tgr_dog_textured", "tgr_dog_best_block", "tgr_dog_best_row",
    "flat_cells", "ncc_raw_p50", "ncc_dog_p50", "peak2_dog_p50",
    "closure_raw_p50", "closure_dog_p50", "lambda_min_p50", "std_p50",
    "lap_var_cell_p50", "edge_cell_p50",
]
QUALITY_METRICS = ["lap_var", "edge_density", "global_contrast", "blockiness", "entropy"]


def auc_score(positive: np.ndarray, negative: np.ndarray) -> float:
    """Mann-Whitney AUC of ``positive`` against ``negative`` (1.0 = perfect separation)."""
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    joined = np.concatenate([positive, negative])
    ranks = np.empty(joined.size, dtype=float)
    order = np.argsort(joined, kind="mergesort")
    sorted_values = joined[order]
    index = 0
    while index < sorted_values.size:
        stop = index
        while stop + 1 < sorted_values.size and sorted_values[stop + 1] == sorted_values[index]:
            stop += 1
        ranks[order[index:stop + 1]] = (index + stop) / 2.0 + 1.0
        index = stop + 1
    u_positive = ranks[: positive.size].sum() - positive.size * (positive.size + 1) / 2.0
    return float(u_positive / (positive.size * negative.size))


def violated(direction: str, value: float, threshold: float) -> bool:
    if not np.isfinite(value):
        return False
    return bool(value > threshold) if direction == "high" else bool(value < threshold)


def clip_tier(modality: str, video_values: dict) -> tuple[str, int, int, list[str]]:
    """Apply the grading rule of report section 3.1 to one clip."""
    core_hits, weak_hits, details = 0, 0, []
    if modality not in CORE_LINES:
        return "不适用", 0, 0, []
    for metric, direction in CORE_LINES[modality]:
        value = video_values.get(metric, float("nan"))
        if violated(direction, value, CORE_THRESH[modality][metric]):
            core_hits += 1
            details.append(f"{metric}={value:.4g}")
    for metric, direction in WEAK_LINES.get(modality, []):
        value = video_values.get(metric, float("nan"))
        if violated(direction, value, WEAK_THRESH[modality][metric]):
            weak_hits += 1
            details.append(f"{metric}={value:.4g}")
    total = len(CORE_LINES[modality])
    if core_hits == total:
        tier = "拒绝接入"
    elif core_hits >= 1:
        tier = "人工复核"
    elif weak_hits >= 1:
        tier = "通过（弱红线告警）"
    else:
        tier = "通过"
    return tier, core_hits, weak_hits, details


def load_windows(path: Path) -> dict:
    data: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            data.setdefault(row["file"], []).append(row)
    for rows in data.values():
        rows.sort(key=lambda item: int(item["window"]))
    return data


def load_clips(grid_dir: Path) -> list[dict]:
    clips = []
    for json_path in sorted(grid_dir.glob("*.json")):
        if json_path.name == "run_manifest.json":
            continue
        npz_path = grid_dir / f"{json_path.stem}.npz"
        if not npz_path.exists():
            continue
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        meta["arrays"] = LazyArrays(npz_path)
        clips.append(meta)
    return clips


def best_block(mask: np.ndarray) -> float:
    rows, cols = mask.shape
    best = 0
    for r in range(rows - 1):
        for c in range(cols - 1):
            best = max(best, int(mask[r:r + 2, c:c + 2].sum()))
    return best / 4.0


def window_rows_for(clip: dict, tau_det: float) -> list[dict]:
    """One row per evaluated second: grid statistics of that second."""
    arrays = clip["arrays"]
    count = arrays["window"].shape[0]
    rows = []
    for index in range(count):
        cells = {name: arrays[name][index] for name in
                 ("ncc_raw", "peak2_raw", "closure_raw", "ncc_dog", "peak2_dog", "closure_dog")}
        flat = arrays["flat"][index].astype(bool)
        raw_mask = decide(cells, flat, "raw", RAW_NCC_THRESH, RAW_CLOSURE_THRESH, None)
        dog_mask = decide(cells, flat, "dog", DOG_NCC_THRESH, DOG_CLOSURE_THRESH, DOG_UNIQ_THRESH)
        detail = arrays["lap_var_cell"][index] >= tau_det
        textured = int((~flat).sum())
        rows.append({
            "window": int(arrays["window"][index]),
            "pair": int(arrays["pair"][index]),
            "tgr_raw": float(raw_mask.mean()),
            "tgr_dog": float(dog_mask.mean()),
            "det_ratio": float(detail.mean()),
            "ugr": float((detail & dog_mask).mean()),
            "ugr_raw": float((detail & raw_mask).mean()),
            "tgr_dog_textured": (float(dog_mask.sum()) / textured) if textured else float("nan"),
            "tgr_dog_best_block": best_block(dog_mask),
            "tgr_dog_best_row": float(max((dog_mask[row].mean() for row in range(dog_mask.shape[0])),
                                          default=0.0)),
            "tgr_dog_best_col": float(max((dog_mask[:, col].mean() for col in range(dog_mask.shape[1])),
                                          default=0.0)),
            "flat_cells": int(flat.sum()),
            "ncc_raw_p50": (float(np.nanmedian(cells["ncc_raw"]))
                            if np.isfinite(cells["ncc_raw"]).any() else float("nan")),
            "ncc_dog_p50": (float(np.nanmedian(cells["ncc_dog"]))
                            if np.isfinite(cells["ncc_dog"]).any() else float("nan")),
            "peak2_dog_p50": (float(np.nanmedian(cells["peak2_dog"]))
                              if np.isfinite(cells["peak2_dog"]).any() else float("nan")),
            "closure_raw_p50": (float(np.nanmedian(cells["closure_raw"]))
                                if np.isfinite(cells["closure_raw"]).any() else float("nan")),
            "closure_dog_p50": (float(np.nanmedian(cells["closure_dog"]))
                                if np.isfinite(cells["closure_dog"]).any() else float("nan")),
            "lambda_min_p50": float(np.nanmedian(arrays["lam"][index])),
            "std_p50": float(np.nanmedian(arrays["std"][index])),
            "lap_var_cell_p50": float(np.nanmedian(arrays["lap_var_cell"][index])),
            "edge_cell_p50": float(np.nanmedian(arrays["edge_cell"][index])),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid trackability vs quality red lines")
    parser.add_argument("--grid-dir", type=str, required=True)
    parser.add_argument("--windows-csv", type=str, required=True)
    parser.add_argument("--report-dir", type=str, required=True)
    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    windows = load_windows(Path(args.windows_csv))
    clips = load_clips(grid_dir)

    # ---- detail floor tau_det: P10 of the per-cell Laplacian variance of the GOOD clips ----
    tau_det: dict[str, float] = {}
    threshold_info: dict[str, dict] = {}
    for modality in ("ir", "vis"):
        pooled = []
        for clip in clips:
            if clip.get("group") != "good" or clip.get("kind") != "source":
                continue
            if clip.get("modality") != modality:
                continue
            values = clip["arrays"]["lap_var_cell"].ravel()
            pooled.append(values[np.isfinite(values)])
        if not pooled:
            continue
        pooled_values = np.concatenate(pooled)
        tau = float(np.percentile(pooled_values, 10))
        tau_det[modality] = tau
        threshold_info[modality] = {
            "tau_det": tau,
            "rule": "P10 of the pooled per-cell Laplacian variance of the good clips of this modality",
            "good_clips": [clip["file"] for clip in clips
                           if clip.get("group") == "good" and clip.get("kind") == "source"
                           and clip.get("modality") == modality],
            "good_cells": int(pooled_values.size),
            "good_cell_p50": float(np.percentile(pooled_values, 50)),
            "good_cell_p90": float(np.percentile(pooled_values, 90)),
        }

    clip_rows: list[dict] = []
    window_rows: list[dict] = []

    for clip in clips:
        file_name = clip["file"]
        modality = clip.get("modality", "?")
        kind = clip.get("kind", "?")
        group = clip.get("group", "?")
        tau = tau_det.get(modality, 0.0)
        rows = window_rows_for(clip, tau)

        quality = {int(item["window"]): item for item in windows.get(file_name, [])}
        for row in rows:
            quality_row = quality.get(row["window"])
            if quality_row is None:
                continue
            merged = dict(row)
            merged.update({"file": file_name, "group": group, "kind": kind, "modality": modality})
            for metric in QUALITY_METRICS + ["illum_jump", "stripe_idx_row", "shift_frac"]:
                merged[metric] = float(quality_row[metric])
            core_hits = sum(1 for metric, direction in CORE_LINES.get(modality, [])
                            if violated(direction, merged[metric], CORE_THRESH[modality][metric]))
            weak_hits = sum(1 for metric, direction in WEAK_LINES.get(modality, [])
                            if violated(direction, merged[metric], WEAK_THRESH[modality][metric]))
            total_core = len(CORE_LINES.get(modality, []))
            merged["core_violations"] = core_hits
            merged["weak_violations"] = weak_hits
            merged["all_core_violated"] = int(total_core > 0 and core_hits == total_core)
            merged["any_core_violated"] = int(core_hits >= 1)
            window_rows.append(merged)

        def stats(key: str, fn) -> float:
            values = [row[key] for row in rows if np.isfinite(row[key])]
            return float(fn(values)) if values else float("nan")

        video_values = {}
        if file_name in windows:
            for metric in QUALITY_METRICS + ["illum_jump", "stripe_idx_row"]:
                series = [float(item[metric]) for item in windows[file_name]
                          if item[metric] not in ("", "nan")]
                video_values[metric] = float(np.median(series)) if series else float("nan")
        tier, core_hits, weak_hits, details = clip_tier(modality, video_values)

        clip_rows.append({
            "file": file_name, "group": group, "kind": kind, "modality": modality,
            "evaluations": len(rows), "tau_det": tau,
            "tgr_raw_p50": stats("tgr_raw", np.median), "tgr_raw_mean": stats("tgr_raw", np.mean),
            "tgr_dog_p50": stats("tgr_dog", np.median), "tgr_dog_mean": stats("tgr_dog", np.mean),
            "tgr_dog_p05": stats("tgr_dog", lambda v: np.percentile(v, 5)),
            "tgr_dog_p95": stats("tgr_dog", lambda v: np.percentile(v, 95)),
            "det_ratio_p50": stats("det_ratio", np.median),
            "ugr_p50": stats("ugr", np.median), "ugr_mean": stats("ugr", np.mean),
            "ugr_p05": stats("ugr", lambda v: np.percentile(v, 5)),
            "ugr_raw_p50": stats("ugr_raw", np.median),
            "tgr_dog_textured_p50": stats("tgr_dog_textured", np.median),
            "tgr_dog_best_row_p50": stats("tgr_dog_best_row", np.median),
            "tgr_dog_best_block_p50": stats("tgr_dog_best_block", np.median),
            "flat_cells_p50": stats("flat_cells", np.median),
            "ncc_raw_p50": stats("ncc_raw_p50", np.median),
            "ncc_dog_p50": stats("ncc_dog_p50", np.median),
            "peak2_dog_p50": stats("peak2_dog_p50", np.median),
            "closure_raw_p50": stats("closure_raw_p50", np.median),
            "closure_dog_p50": stats("closure_dog_p50", np.median),
            "lambda_min_p50": stats("lambda_min_p50", np.median),
            "std_p50": stats("std_p50", np.median),
            "lap_var_cell_p50": stats("lap_var_cell_p50", np.median),
            "edge_cell_p50": stats("edge_cell_p50", np.median),
            "core_violations": core_hits, "weak_violations": weak_hits,
            "tier": tier, "violated": "; ".join(details),
            **{metric: video_values.get(metric, float("nan")) for metric in QUALITY_METRICS},
        })

    # clip level aliases: the AUC table uses the same metric names for clips and windows
    for row in clip_rows:
        row["tgr_raw"] = row["tgr_raw_p50"]
        row["tgr_dog"] = row["tgr_dog_p50"]
        row["det_ratio"] = row["det_ratio_p50"]
        row["ugr"] = row["ugr_p50"]
        row["ugr_raw"] = row["ugr_raw_p50"]
        row["tgr_dog_textured"] = row["tgr_dog_textured_p50"]
        row["tgr_dog_best_block"] = row["tgr_dog_best_block_p50"]
        row["tgr_dog_best_row"] = row["tgr_dog_best_row_p50"]
        row["flat_cells"] = row["flat_cells_p50"]
        for metric in QUALITY_METRICS + ["illum_jump", "stripe_idx_row"]:
            row.setdefault(metric, float("nan"))

    # ------------------------------------------------------------------ AUC table
    compare_rows = []
    for modality in ("ir", "vis"):
        good_clips = [row for row in clip_rows if row["group"] == "good" and row["kind"] == "source"
                      and row["modality"] == modality]
        bad_clips = [row for row in clip_rows if row["group"] == "bad" and row["kind"] == "source"
                     and row["modality"] == modality]
        good_windows = [row for row in window_rows if row["group"] == "good" and row["kind"] == "source"
                        and row["modality"] == modality]
        bad_windows = [row for row in window_rows if row["group"] == "bad" and row["kind"] == "source"
                       and row["modality"] == modality]
        violated_windows = [row for row in bad_windows if row["all_core_violated"]]
        clean_windows = [row for row in good_windows if not row["any_core_violated"]]
        for metric in GRID_METRICS + QUALITY_METRICS:
            good = np.asarray([row[metric] for row in good_clips], dtype=float)
            bad = np.asarray([row[metric] for row in bad_clips], dtype=float)
            good_w = np.asarray([row[metric] for row in good_windows], dtype=float)
            bad_w = np.asarray([row[metric] for row in bad_windows], dtype=float)
            compare_rows.append({
                "modality": modality, "metric": metric, "kind": "grid" if metric in GRID_METRICS else "quality",
                "clip_auc_good_vs_bad": auc_score(good, bad),
                "clip_n_good": len(good), "clip_n_bad": len(bad),
                "window_auc_good_vs_bad": auc_score(good_w, bad_w),
                "window_n_good": len(good_w), "window_n_bad": len(bad_w),
                "window_auc_redline_violated": auc_score(
                    np.asarray([row[metric] for row in violated_windows], dtype=float),
                    np.asarray([row[metric] for row in clean_windows], dtype=float)),
                "window_n_clean": len(clean_windows), "window_n_violated": len(violated_windows),
                "good_median": float(np.median(good)) if good.size else float("nan"),
                "bad_median": float(np.median(bad)) if bad.size else float("nan"),
                "good_window_median": float(np.median(good_w)) if good_w.size else float("nan"),
                "bad_window_median": float(np.median(bad_w)) if bad_w.size else float("nan"),
            })

    # ------------------------------------------------------------------ cross table
    crosstab = []
    for tier in ("通过", "通过（弱红线告警）", "人工复核", "拒绝接入"):
        subset = [row for row in clip_rows if row["tier"] == tier]
        if not subset:
            continue
        for metric in ("ugr", "tgr_dog", "tgr_raw", "det_ratio", "tgr_dog_best_row",
                       "flat_cells", "lap_var_cell_p50"):
            values = [row[metric] for row in subset if np.isfinite(row[metric])]
            crosstab.append({
                "tier": tier, "clips": len(subset), "metric": metric,
                "median": float(np.median(values)) if values else float("nan"),
                "min": float(np.min(values)) if values else float("nan"),
                "max": float(np.max(values)) if values else float("nan"),
                "files": "; ".join(sorted(row["file"] for row in subset)),
            })

    # ---- sensitivity of the detail floor: DET/UGR for a sweep of tau_det -------------
    def cached_masks(clip: dict):
        """(dog closure mask, per-cell Laplacian variance) per evaluation - tau independent."""
        arrays = clip["arrays"]
        count = arrays["window"].shape[0]
        for index in range(count):
            cells = {name: arrays[name][index] for name in
                     ("ncc_raw", "peak2_raw", "closure_raw", "ncc_dog", "peak2_dog", "closure_dog")}
            flat = arrays["flat"][index].astype(bool)
            dog_mask = decide(cells, flat, "dog", DOG_NCC_THRESH, DOG_CLOSURE_THRESH, DOG_UNIQ_THRESH)
            yield dog_mask, arrays["lap_var_cell"][index]

    sensitivity = []
    for modality in ("ir", "vis"):
        for factor in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            tau = tau_det.get(modality, 0.0) * factor
            good = []
            bad = []
            for clip in clips:
                if clip.get("modality") != modality:
                    continue
                values = []
                for dog_mask, lap_var_cell in cached_masks(clip):
                    detail = lap_var_cell >= tau
                    values.append(float((detail & dog_mask).mean()))
                if clip.get("group") == "good" and clip.get("kind") == "source":
                    good.extend(values)
                if clip.get("group") == "bad" and clip.get("kind") == "source":
                    bad.extend(values)
            sensitivity.append({
                "modality": modality, "tau_det": tau, "factor": factor,
                "good_ugr_median": float(np.median(good)) if good else float("nan"),
                "bad_ugr_median": float(np.median(bad)) if bad else float("nan"),
                "window_auc": auc_score(np.asarray(bad, dtype=float), np.asarray(good, dtype=float)),
                "good_n": len(good), "bad_n": len(bad),
            })

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        fields = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(report_dir / "grid_summary.csv", clip_rows)
    write_csv(report_dir / "grid_windows.csv", window_rows)
    write_csv(report_dir / "grid_compare.csv", compare_rows)
    write_csv(report_dir / "grid_crosstab.csv", crosstab)
    write_csv(report_dir / "grid_sensitivity.csv", sensitivity)
    (report_dir / "grid_thresholds.json").write_text(
        json.dumps({"tau_det": tau_det, "detail": threshold_info}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print(json.dumps({"clips": len(clip_rows), "windows": len(window_rows),
                      "tau_det": tau_det}, ensure_ascii=False))
    for row in compare_rows:
        if row["metric"] in ("tgr_raw", "tgr_dog", "det_ratio", "ugr", "lap_var"):
            print(f"[AUC] {row['modality']:3} {row['metric']:10} clip={row['clip_auc_good_vs_bad']:.3f} "
                  f"window={row['window_auc_good_vs_bad']:.3f} "
                  f"redline={row['window_auc_redline_violated']:.3f} "
                  f"good_med={row['good_median']:.4g} bad_med={row['bad_median']:.4g}")


if __name__ == "__main__":
    sys.exit(main())
