#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Generate Detailed Sequence Diagnostics and Updated Official SOTA Comparison Bar Chart
Loads uav_median_trial22_cache.pkl, evaluates each sequence with the Champion Adaptive Fusion config:
  th_base = 0.22, th_salvage = 0.06, th_ground = 0.32, match_dist = 12.0, max_match_dist = 18.0
Outputs:
1. Detailed per-sequence metrics table (console & CSV).
2. Publication-grade updated comparison bar chart: compare_best_metrics.png.

Usage on Server:
    python manu/report_trial22_fusion_diagnostics.py \
        --cache-file runs/gmc_eval/uav_median_trial22_cache.pkl \
        --dist-thresh 8.0 \
        --output-dir runs/compare_results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import pickle
import sys
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr
from manu.eval_trial22_adaptive_fusion import (
    extract_seq_name,
    evaluate_sequence,
    calc_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Report Sequence Diagnostics and Plot Benchmark Chart")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial22_cache.pkl",
        help="Path to Trial 22 inference cache",
    )
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    parser.add_argument("--output-dir", type=str, default="runs/compare_results")
    return parser.parse_args()


def plot_official_comparison_chart(
    m_base: Dict[str, float],
    m_fuse: Dict[str, float],
    output_png: Path,
):
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    models = [
        {"name": "trial_0028 (YOLO)", "rec": 0.7561, "prec": 0.7703, "f1": 0.7631, "color": "#1f77b4"},
        {"name": "uav_gpu23_heatmap", "rec": 0.8319, "prec": 0.8342, "f1": 0.8331, "color": "#2ca02c"},
        {"name": "uav_gpu23_stride2", "rec": 0.8435, "prec": 0.9355, "f1": 0.8871, "color": "#8c564b"},
        {"name": "trial_0031 (Baseline)", "rec": 0.8451, "prec": 0.9396, "f1": 0.8898, "color": "#7f7f7f"},
        {"name": "trial_0028_ (3-frame)", "rec": 0.8230, "prec": 0.9644, "f1": 0.8881, "color": "#17becf"},
        {"name": "exp_6ep_median", "rec": 0.8568, "prec": 0.9422, "f1": 0.8975, "color": "#ff7f0e"},
        {"name": "Trial 22 (Single-Frame)", "rec": m_base["recall"] / 100.0, "prec": m_base["precision"] / 100.0, "f1": m_base["f1"] / 100.0, "color": "#9467bd"},
        {
            "name": "Trial 22 + Fusion (NEW SOTA)",
            "rec": m_fuse["recall"] / 100.0,
            "prec": m_fuse["precision"] / 100.0,
            "f1": m_fuse["f1"] / 100.0,
            "color": "#d62728",
        },
    ]

    metric_names = ["Recall", "Precision", "F1-Score"]
    n_models = len(models)
    n_metrics = len(metric_names)

    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=300)
    width = 0.095
    x = np.arange(n_metrics)

    for i, m in enumerate(models):
        values = [m["rec"], m["prec"], m["f1"]]
        offset = (i - (n_models - 1) / 2) * width
        is_new = "NEW SOTA" in m["name"]
        rects = ax.bar(
            x + offset,
            values,
            width,
            label=m["name"],
            color=m["color"],
            edgecolor="black" if is_new else "none",
            linewidth=1.2 if is_new else 0.5,
        )

        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.0,
                rotation=0,
                fontweight="bold" if is_new else "normal",
            )

    ax.set_ylabel("Metric Value (0 ~ 1.0)", fontsize=13, fontweight="bold")
    ax.set_title("Infrared Tiny UAV Detection: Full Evolution Benchmark (Distance <= 8.0px)", fontsize=16, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=13, fontweight="bold")
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95, ncol=2)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_png), dpi=300)
    plt.close()
    print(colorstr("bold", colorstr("green", f"[SUCCESS] New Official SOTA Chart exported -> {output_png.resolve()}")))


def main():
    args = parse_args()
    cache_path = Path(args.cache_file)
    if not cache_path.is_absolute():
        for cand in [
            PROJECT_ROOT / cache_path,
            Path("/tmp/pycharm_project_10ae9e2e") / cache_path,
            Path("/home/manu/mnt/pycharm_project_10ae9e2e") / cache_path,
        ]:
            if cand.exists():
                cache_path = cand
                break

    if not cache_path.exists():
        print(f"[ERROR] Cache file not found: {cache_path}")
        sys.exit(1)

    print(f"[INFO] Loading cache from {cache_path}...")
    with open(cache_path, "rb") as f:
        records = pickle.load(f)
    print(f"Loaded {len(records)} image predictions.")

    seq_records: Dict[str, List[Dict]] = {}
    for r in records:
        seq = extract_seq_name(r["im_name"])
        seq_records.setdefault(seq, []).append(r)

    # Optimal parameters (Trial 22 Champion)
    tracker_config = {
        "max_age": 3,
        "min_hits": 3,
        "match_dist": 12.0,
        "max_match_dist": 18.0,
        "output_coasting": False,
        "min_track_score": 0.08,
        "instant_conf": 0.25,
        "min_displacement": 2.5,
        "sky_ratio": 0.60,
        "img_h": 640,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / "trial22_adaptive_fusion_per_sequence.csv"

    grand_base = {"tp": 0, "fp": 0, "gt": 0}
    grand_fuse = {"tp": 0, "fp": 0, "gt": 0}
    rows = []

    print("\n" + "=" * 125)
    print(f"{'Sequence Name':<28} | {'Single-Frame Base (th=0.25)':<28} | {'Adaptive Fusion (SOTA)':<28} | {'Recall Delta'}")
    print(f"{'':<28} | {'TP/GT':<12} {'Rec%':<7} {'Prec%':<7} | {'TP/GT':<12} {'Rec%':<7} {'Prec%':<7} | {'ΔTP':<6} {'ΔRec%'}")
    print("=" * 125)

    for seq_name in sorted(seq_records.keys()):
        recs = seq_records[seq_name]
        res = evaluate_sequence(
            records=recs,
            dist_thresh=args.dist_thresh,
            th_base=0.22,
            th_salvage=0.06,
            th_ground=0.32,
            sky_ratio=0.60,
            img_h=640,
            tracker_config=tracker_config,
        )

        b = res["baseline"]
        f = res["fusion"]

        grand_base["tp"] += int(b["tp"])
        grand_base["fp"] += int(b["fp"])
        grand_base["gt"] += int(b["gt"])

        grand_fuse["tp"] += int(f["tp"])
        grand_fuse["fp"] += int(f["fp"])
        grand_fuse["gt"] += int(f["gt"])

        delta_tp = int(f["tp"]) - int(b["tp"])
        delta_rec = f["recall"] - b["recall"]

        b_str = f"{int(b['tp']):>5}/{int(b['gt']):<5} {b['recall']:>5.1f}% {b['precision']:>5.1f}%"
        f_str = f"{int(f['tp']):>5}/{int(f['gt']):<5} {f['recall']:>5.1f}% {f['precision']:>5.1f}%"
        delta_str = f"{delta_tp:>+5d} ({delta_rec:>+5.2f}%)"

        if delta_rec > 0:
            delta_str = colorstr("green", delta_str)
        elif delta_rec < 0:
            delta_str = colorstr("red", delta_str)

        print(f"{seq_name:<28} | {b_str:<28} | {f_str:<28} | {delta_str}")

        rows.append({
            "sequence": seq_name,
            "gt_count": int(b["gt"]),
            "base_tp": int(b["tp"]),
            "base_fp": int(b["fp"]),
            "base_rec": f"{b['recall']:.2f}%",
            "base_prec": f"{b['precision']:.2f}%",
            "base_f1": f"{b['f1']:.4f}",
            "fuse_tp": int(f["tp"]),
            "fuse_fp": int(f["fp"]),
            "fuse_rec": f"{f['recall']:.2f}%",
            "fuse_prec": f"{f['precision']:.2f}%",
            "fuse_f1": f"{f['f1']:.4f}",
            "delta_tp": delta_tp,
            "delta_rec": f"{delta_rec:+.2f}%",
        })

    print("=" * 125)

    m_base = calc_metrics(grand_base["tp"], grand_base["fp"], grand_base["gt"])
    m_fuse = calc_metrics(grand_fuse["tp"], grand_fuse["fp"], grand_fuse["gt"])

    delta_tp_total = m_fuse["tp"] - m_base["tp"]
    delta_rec_total = m_fuse["recall"] - m_base["recall"]
    delta_summary_str = f"{delta_tp_total:>+5d} ({delta_rec_total:>+5.2f}%)"
    print(f"{'GRAND TOTAL':<28} | "
          f"{m_base['tp']:>5}/{m_base['gt']:<5} {m_base['recall']:>5.1f}% {m_base['precision']:>5.1f}% | "
          f"{m_fuse['tp']:>5}/{m_fuse['gt']:<5} {m_fuse['recall']:>5.1f}% {m_fuse['precision']:>5.1f}% | "
          f"{colorstr('bold', colorstr('green', delta_summary_str))}")
    print("=" * 125 + "\n")

    # Export CSV
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(colorstr("bold", colorstr("green", f"[SUCCESS] Per-sequence diagnostic table exported -> {csv_file.resolve()}")))

    # Plot official comparison bar chart
    chart_file = out_dir / "compare_best_metrics.png"
    plot_official_comparison_chart(m_base, m_fuse, chart_file)


if __name__ == "__main__":
    main()
