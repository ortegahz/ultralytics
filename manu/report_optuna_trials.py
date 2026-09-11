#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Universal Optuna Study Reporter for Tiny Object Heatmap Tuning.

Supports:
1. Auto-detection of active/recent study in runs/ (defaults to runs/optuna_soft_iou_search or runs/optuna_median_search).
2. Direct reading from SQLite study.db and/or optuna_summary.csv.
3. Full hyperparameter display: soft_iou_weight, focal_beta, lr0, best_th, TP, FP, etc.
4. Flexible sorting by F1, recall, precision, or trial number.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

try:
    import optuna
except ImportError:
    optuna = None

DEFAULT_CANDIDATE_ROOTS = [
    Path("runs/optuna_soft_iou_search"),
    Path("runs/optuna_median_search"),
    Path("runs/optuna_heatmap_stride2_640"),
    Path("/tmp/pycharm_project_10ae9e2e/runs/optuna_soft_iou_search"),
    Path("/tmp/pycharm_project_10ae9e2e/runs/optuna_median_search"),
]


def resolve_study_root(root_arg: str | None) -> Path:
    if root_arg:
        p = Path(root_arg).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified root directory does not exist: {p}")

    for cand in DEFAULT_CANDIDATE_ROOTS:
        if (cand / "study.db").exists() or (cand / "optuna_summary.csv").exists():
            return cand.resolve()

    return Path("runs/optuna_soft_iou_search").resolve()


def read_summary_csv(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    return reader


def read_results_csv(trial_dir: Path) -> dict:
    csv_file = trial_dir / "results.csv"
    if not csv_file.exists():
        return {}
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        # Find row with maximum F1
        best = max(rows, key=lambda r: float(r.get("metrics/f1(B)", 0.0)))
        return {
            "epoch": best.get("epoch", "N/A"),
            "f1": float(best.get("metrics/f1(B)", 0.0)),
            "recall": float(best.get("metrics/recall(B)", 0.0)),
            "precision": float(best.get("metrics/precision(B)", 0.0)),
            "best_th": float(best.get("metrics/best_th", 0.25)),
            "tp": int(best.get("metrics/tp", 0)),
            "fp": int(best.get("metrics/fp", 0)),
            "gt": int(best.get("metrics/gt", 0)),
        }
    except Exception:
        return {}


def format_number(value, digits: int = 4) -> str:
    if value in (None, "", "N/A"):
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def print_table(rows: list[dict], show_soft_iou: bool = True) -> None:
    headers = [
        "Rank", "Trial", "State", "Epoch", "F1", "Recall", "Precision", "TP", "FP", "Thresh"
    ]
    if show_soft_iou:
        headers.append("soft_iou")
    headers.extend(["focal_beta", "lr0", "radius"])

    table = [headers]
    for row in rows:
        line = [
            str(row["rank"]),
            str(row["trial"]),
            str(row["state"]),
            str(row.get("epoch", "N/A")),
            format_number(row.get("f1", "N/A")),
            format_number(row.get("recall", "N/A")),
            format_number(row.get("precision", "N/A")),
            str(row.get("tp", "N/A")),
            str(row.get("fp", "N/A")),
            format_number(row.get("best_th", "N/A"), 2),
        ]
        if show_soft_iou:
            line.append(format_number(row.get("soft_iou_weight", "N/A"), 4))
        line.extend([
            format_number(row.get("focal_beta", "N/A"), 2),
            format_number(row.get("lr0", "N/A"), 6),
            str(row.get("min_radius", "N/A")),
        ])
        table.append(line)

    widths = [max(len(str(table[r][c])) for r in range(len(table))) for c in range(len(headers))]
    sep = "-+-".join("-" * w for w in widths)

    print(" | ".join(str(val).ljust(w) for val, w in zip(table[0], widths)))
    print(sep)
    for row_vals in table[1:]:
        print(" | ".join(str(val).ljust(w) for val, w in zip(row_vals, widths)))


def main():
    parser = argparse.ArgumentParser(description="Report Optuna UAV Study Results")
    parser.add_argument("--root", type=str, default=None, help="Root folder of the optuna study")
    parser.add_argument("--study-name", type=str, default=None, help="Optuna study name (auto-detected if None)")
    parser.add_argument("--sort-by", type=str, choices=["f1", "recall", "precision", "trial"], default="f1")
    parser.add_argument("--top", type=int, default=30, help="Show top N trials")
    args = parser.parse_args()

    root = resolve_study_root(args.root)
    summary_csv = root / "optuna_summary.csv"
    db_file = root / "study.db"

    records = []
    has_soft_iou = False

    # 1. Prefer reading from optuna_summary.csv if available
    if summary_csv.exists():
        csv_records = read_summary_csv(summary_csv)
        if csv_records:
            has_soft_iou = "soft_iou_weight" in csv_records[0]
            for row in csv_records:
                records.append({
                    "trial": row.get("trial", "").replace("trial_", ""),
                    "state": row.get("status", "COMPLETE"),
                    "epoch": row.get("epoch", "N/A"),
                    "f1": float(row.get("f1", 0.0)) if row.get("f1") else "N/A",
                    "recall": float(row.get("recall", 0.0)) if row.get("recall") else "N/A",
                    "precision": float(row.get("precision", 0.0)) if row.get("precision") else "N/A",
                    "best_th": float(row.get("best_th", 0.25)) if row.get("best_th") else "N/A",
                    "tp": row.get("tp", "N/A"),
                    "fp": row.get("fp", "N/A"),
                    "soft_iou_weight": row.get("soft_iou_weight", "N/A"),
                    "focal_beta": row.get("focal_beta", "N/A"),
                    "lr0": row.get("lr0", "N/A"),
                    "min_radius": row.get("min_radius", "N/A"),
                })

    # 2. Fallback to study.db if CSV is missing
    elif db_file.exists():
        storage = f"sqlite:///{db_file.resolve()}"
        study_summaries = optuna.study.get_all_study_summaries(storage)
        study_name = args.study_name or (study_summaries[0].study_name if study_summaries else "soft_iou_recall_bias_study")
        study = optuna.load_study(study_name=study_name, storage=storage)

        for trial in study.trials:
            trial_dir = root / f"trial_{trial.number:04d}"
            metrics = read_results_csv(trial_dir)
            f1 = trial.value if trial.value is not None else metrics.get("f1", "N/A")
            rec = metrics.get("recall", "N/A")
            prec = metrics.get("precision", "N/A")

            p = trial.params
            if "soft_iou_weight" in p:
                has_soft_iou = True

            records.append({
                "trial": f"{trial.number:04d}",
                "state": trial.state.name,
                "epoch": metrics.get("epoch", "N/A"),
                "f1": f1,
                "recall": rec,
                "precision": prec,
                "best_th": metrics.get("best_th", "N/A"),
                "tp": metrics.get("tp", "N/A"),
                "fp": metrics.get("fp", "N/A"),
                "soft_iou_weight": p.get("soft_iou_weight", "N/A"),
                "focal_beta": p.get("focal_beta", "N/A"),
                "lr0": p.get("lr0", "N/A"),
                "min_radius": p.get("min_radius", "N/A"),
            })
    else:
        print(f"[ERROR] Neither study.db nor optuna_summary.csv found in {root}")
        return

    if not records:
        print(f"[INFO] No trial records found in {root}.")
        return

    # Sorting
    def sort_key(r):
        val = r.get(args.sort_by, 0.0)
        try:
            return float(val)
        except (ValueError, TypeError):
            return -1.0

    records.sort(key=lambda r: (r["state"] == "COMPLETE", sort_key(r)), reverse=True)

    for idx, r in enumerate(records, start=1):
        r["rank"] = idx

    print("=" * 125)
    print(f"📊 Optuna Study Summary Report: {root.name}")
    print(f"Directory : {root}")
    print(f"Total Trials: {len(records)} | Sorted by: {args.sort_by}")
    print("=" * 125)

    print_table(records[:args.top], show_soft_iou=has_soft_iou)

    complete_records = [r for r in records if r["state"] == "COMPLETE"]
    if complete_records:
        best = complete_records[0]
        print("\n" + "=" * 80)
        print("🏆 Best COMPLETE Trial")
        print("=" * 80)
        print(f"Trial       : trial_{best['trial']}")
        print(f"F1-Score    : {format_number(best['f1'])}")
        print(f"Recall      : {format_number(best['recall'])} (TP: {best.get('tp', 'N/A')})")
        print(f"Precision   : {format_number(best['precision'])} (FP: {best.get('fp', 'N/A')})")
        print(f"Best Thresh : {format_number(best.get('best_th', 'N/A'), 2)}")
        print("Hyperparameters:")
        if has_soft_iou:
            print(f"  soft_iou_weight : {best.get('soft_iou_weight')}")
        print(f"  focal_beta      : {best.get('focal_beta')}")
        print(f"  lr0             : {best.get('lr0')}")
        print(f"  min_radius      : {best.get('min_radius')}")
        trial_str = str(best['trial'])
        print(f"Weights     : {root / f'trial_{trial_str}' / 'weights' / 'best.pt'}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
