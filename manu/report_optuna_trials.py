#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import optuna

DEFAULT_ROOT = Path(
    "/tmp/pycharm_project_10ae9e2e/runs/optuna_uav_recall"
)

DEFAULT_STUDY_NAME = "uav_fold4_yolo26n_p2_recall"


def read_csv_metrics(results_csv: Path) -> dict:
    """Read the best Recall row from a trial results.csv."""
    if not results_csv.exists():
        return {}

    try:
        with results_csv.open(
                "r",
                encoding="utf-8",
                newline="",
        ) as file:
            rows = list(csv.DictReader(file))
    except Exception:
        return {}

    if not rows:
        return {}

    rows = [
        {
            key.strip(): value.strip()
            if isinstance(value, str)
            else value
            for key, value in row.items()
        }
        for row in rows
    ]

    valid_rows = []

    for row in rows:
        recall = row.get("metrics/recall(B)")

        if recall in (None, ""):
            continue

        try:
            row["_recall"] = float(recall)
        except ValueError:
            continue

        valid_rows.append(row)

    if not valid_rows:
        return {}

    # 产品目标：选择 Recall 最高的 epoch
    best = max(
        valid_rows,
        key=lambda row: row["_recall"],
    )

    return {
        "epoch": best.get("epoch", "N/A"),
        "recall": best.get("metrics/recall(B)", "N/A"),
        "ap50": best.get("metrics/mAP50(B)", "N/A"),
        "map50_95": best.get("metrics/mAP50-95(B)", "N/A"),
        "precision": best.get("metrics/precision(B)", "N/A"),
    }


def get_trial_metrics(
        trial: optuna.trial.FrozenTrial,
        root: Path,
) -> dict:
    """Read metrics from Optuna user attributes or results.csv."""
    metrics = trial.user_attrs.get("metrics")

    if isinstance(metrics, dict) and metrics.get("recall") is not None:
        return {
            "epoch": metrics.get("epoch", "N/A"),
            "recall": metrics.get("recall", "N/A"),
            "ap50": metrics.get("ap50", "N/A"),
            "map50_95": metrics.get("map50_95", "N/A"),
            "precision": metrics.get("precision", "N/A"),
        }

    trial_dir = root / f"trial_{trial.number:04d}"
    results_csv = trial_dir / "results.csv"

    return read_csv_metrics(results_csv)


def format_number(value, digits: int = 4) -> str:
    """Format numeric values for table output."""
    if value in (None, "", "N/A"):
        return "N/A"

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def print_table(rows: list[dict]) -> None:
    """Print trial results as an aligned table."""
    headers = [
        "Rank",
        "Trial",
        "State",
        "Epoch",
        "Recall",
        "AP50",
        "mAP50-95",
        "Precision",
        "lr0",
        "momentum",
        "weight_decay",
        "warmup",
        "scale",
        "mosaic",
    ]

    table = [headers]

    for row in rows:
        table.append(
            [
                str(row["rank"]),
                str(row["trial"]),
                str(row["state"]),
                str(row["epoch"]),
                format_number(row["recall"]),
                format_number(row["ap50"]),
                format_number(row["map50_95"]),
                format_number(row["precision"]),
                format_number(row["lr0"], 6),
                format_number(row["momentum"], 6),
                format_number(row["weight_decay"], 6),
                format_number(row["warmup_epochs"], 4),
                format_number(row["scale"], 4),
                format_number(row["mosaic"], 4),
            ]
        )

    widths = [
        max(len(str(table[row][column])) for row in range(len(table)))
        for column in range(len(headers))
    ]

    separator = "-+-".join("-" * width for width in widths)

    print(
        " | ".join(
            str(value).ljust(width)
            for value, width in zip(table[0], widths)
        )
    )
    print(separator)

    for values in table[1:]:
        print(
            " | ".join(
                str(value).ljust(width)
                for value, width in zip(values, widths)
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="Report Optuna UAV hyperparameter trials."
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Optuna output directory.",
    )

    parser.add_argument(
        "--study-name",
        type=str,
        default=DEFAULT_STUDY_NAME,
        help="Optuna study name.",
    )

    parser.add_argument(
        "--sort-by",
        choices=[
            "recall",
            "ap50",
            "map50_95",
            "precision",
        ],
        default="recall",
        help="Metric used to sort trials.",
    )

    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    database = root / "study.db"
    storage = f"sqlite:///{database}"

    if not database.exists():
        raise FileNotFoundError(
            f"Optuna database not found: {database}"
        )

    study = optuna.load_study(
        study_name=args.study_name,
        storage=storage,
    )

    records = []

    for trial in study.trials:
        metrics = get_trial_metrics(
            trial=trial,
            root=root,
        )

        record = {
            "trial": trial.number,
            "state": trial.state.name,
            "epoch": metrics.get("epoch", "N/A"),
            "recall": metrics.get("recall", "N/A"),
            "ap50": metrics.get("ap50", "N/A"),
            "map50_95": metrics.get("map50_95", "N/A"),
            "precision": metrics.get("precision", "N/A"),
            "lr0": trial.params.get("lr0", "N/A"),
            "momentum": trial.params.get("momentum", "N/A"),
            "weight_decay": trial.params.get(
                "weight_decay",
                "N/A",
            ),
            "warmup_epochs": trial.params.get(
                "warmup_epochs",
                "N/A",
            ),
            "scale": trial.params.get("scale", "N/A"),
            "mosaic": trial.params.get("mosaic", "N/A"),
        }

        records.append(record)

    def sort_key(record):
        value = record.get(args.sort_by, "N/A")

        try:
            return float(value)
        except (TypeError, ValueError):
            return float("-inf")

    # 只把 COMPLETE trial 放在前面，再按目标指标降序
    records.sort(
        key=lambda record: (
            record["state"] == "COMPLETE",
            sort_key(record),
        ),
        reverse=True,
    )

    for rank, record in enumerate(records, start=1):
        record["rank"] = rank

    print("=" * 150)
    print("Optuna UAV Trial Summary")
    print("=" * 150)
    print(f"Database : {database}")
    print(f"Study    : {args.study_name}")
    print(f"Trials   : {len(records)}")
    print(f"Sort by  : {args.sort_by}")
    print("=" * 150)

    print_table(records)

    complete = [
        record
        for record in records
        if record["state"] == "COMPLETE"
    ]

    if complete:
        best = complete[0]

        print("\n" + "=" * 80)
        print("Best COMPLETE Trial")
        print("=" * 80)
        print(f"Trial       : {best['trial']}")
        print(f"Epoch       : {best['epoch']}")
        print(f"Recall      : {format_number(best['recall'])}")
        print(f"AP50        : {format_number(best['ap50'])}")
        print(f"mAP50-95    : {format_number(best['map50_95'])}")
        print(f"Precision   : {format_number(best['precision'])}")

        print("\nBest parameters:")
        print(f"  lr0           : {best['lr0']}")
        print(f"  momentum      : {best['momentum']}")
        print(f"  weight_decay  : {best['weight_decay']}")
        print(f"  warmup_epochs : {best['warmup_epochs']}")
        print(f"  scale         : {best['scale']}")
        print(f"  mosaic        : {best['mosaic']}")

        best_model = (
                root
                / f"trial_{best['trial']:04d}"
                / "weights"
                / "best.pt"
        )

        print(f"\nBest model path:")
        print(f"  {best_model}")
    else:
        print("\nNo COMPLETE trial found.")


if __name__ == "__main__":
    main()
