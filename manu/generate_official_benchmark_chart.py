#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Generate and Update the Official UAV Point Detection Benchmark Bar Chart (compare_best_metrics.png).

Includes all official milestones tested under the unified Distance <= 8.0px standard:
1. trial_0028 (YOLO)          : Rec=0.7561, Prec=0.7703, F1=0.7631
2. uav_gpu23_heatmap          : Rec=0.8319, Prec=0.8342, F1=0.8331
3. uav_gpu23_heatmap_stride2  : Rec=0.8435, Prec=0.9355, F1=0.8871
4. trial_0031 (Baseline)       : Rec=0.8451, Prec=0.9396, F1=0.8898
5. trial_0028_ (3-frame)       : Rec=0.8230, Prec=0.9644, F1=0.8881
6. trial_0031 + GMC (NEW SOTA) : Rec=0.8464, Prec=0.9389, F1=0.8902 (Breakthrough)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate official compare_best_metrics.png")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/compare_results",
        help="Directory to save the generated chart",
    )
    return parser.parse_args()


def render_benchmark_chart(output_path: Path):
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    models = [
        {"name": "trial_0028", "rec": 0.7561, "prec": 0.7703, "f1": 0.7631, "color": "#1f77b4"},
        {"name": "uav_gpu23_heatmap", "rec": 0.8319, "prec": 0.8342, "f1": 0.8331, "color": "#2ca02c"},
        {"name": "uav_gpu23_heatmap_stride2", "rec": 0.8435, "prec": 0.9355, "f1": 0.8871, "color": "#8c564b"},
        {"name": "trial_0031", "rec": 0.8451, "prec": 0.9396, "f1": 0.8898, "color": "#7f7f7f"},
        {"name": "trial_0028_", "rec": 0.8230, "prec": 0.9644, "f1": 0.8881, "color": "#17becf"},
        {"name": "trial_0031 (GMC)", "rec": 0.8464, "prec": 0.9389, "f1": 0.8902, "color": "#d62728"},
    ]

    metric_names = ["Recall", "Precision", "F1-Score"]
    n_models = len(models)
    n_metrics = len(metric_names)

    fig, ax = plt.subplots(figsize=(12.5, 6.8), dpi=300)
    x = np.arange(n_metrics)
    total_width = 0.78
    bar_width = total_width / n_models

    for i, m in enumerate(models):
        values = [m["rec"], m["prec"], m["f1"]]
        offsets = x - (total_width / 2) + (i + 0.5) * bar_width
        bars = ax.bar(
            offsets,
            values,
            bar_width,
            label=m["name"],
            color=m["color"],
            alpha=0.92,
            edgecolor="black",
            linewidth=0.5,
        )

        for bar, val in zip(bars, values):
            is_new = "GMC" in m["name"]
            ax.annotate(
                f"{val:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold" if is_new else "normal",
                color="#b30000" if is_new else "#222222",
            )

    ax.set_title("UAV 极小目标模型点检测基准对比 (Distance <= 8.0px)", fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12.5, fontweight="bold")
    ax.set_ylabel("Metric Value (0 ~ 1.0)", fontsize=12, fontweight="bold")
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300)
    plt.close()
    print(f"[SUCCESS] Updated benchmark chart saved to: {output_path.resolve()}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    chart_p = out_dir / "compare_best_metrics.png"
    render_benchmark_chart(chart_p)


if __name__ == "__main__":
    main()
