#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Figures for the grid trackability report."""
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
    LazyArrays,
    decide,
)

GROUP_COLOR = {"good": "#2E7D32", "bad": "#C62828"}
TIER_COLOR = {
    "通过": "#2E7D32",
    "通过（弱红线告警）": "#9E9D24",
    "人工复核": "#EF6C00",
    "拒绝接入": "#C62828",
}


def setup_fonts():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for candidate in (
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def short(name: str) -> str:
    name = name.replace(".h264", "").replace(".mp4", "")
    for prefix in ("VIDEO000", "TS120BT-04", "TS130CT-02", "120BT-04", "130CT-02"):
        if name.startswith(prefix):
            return name
    return name


def make_perclip(plt, clip_rows: list[dict], out_dir: Path) -> str:
    rows = sorted(clip_rows, key=lambda row: (row["group"], row["modality"], row["file"]))
    labels = [short(row["file"]) for row in rows]
    positions = np.arange(len(rows))
    figure, axes = plt.subplots(3, 1, figsize=(17, 11), sharex=True)
    panels = [("tgr_raw_p50", "TGR（原始灰度域，文档原方法）", "#1565C0"),
              ("tgr_dog_p50", "TGR（DoG 带通域 + 峰值唯一性）", "#6A1B9A"),
              ("ugr_p50", "UGR（可用格占比＝有细节 且 锁得住）", "#C62828")]
    for axis, (key, title, color) in zip(axes, panels):
        values = [float(row[key]) for row in rows]
        colors = [GROUP_COLOR.get(row["group"], "#555555") for row in rows]
        axis.bar(positions, values, color=colors, alpha=0.85)
        axis.set_ylabel(title, fontsize=10)
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR["good"]),
               plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR["bad"])]
    axes[0].legend(handles, ["好视频组", "坏视频组"], loc="upper right")
    figure.suptitle("每条视频的网格可跟踪率（48 格均匀网格，每秒 2 对帧）", fontsize=14)
    figure.tight_layout()
    path = out_dir / "chart_grid_perclip.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return str(path)


def make_heatmap(plt, grid_dir: Path, tau_det: dict, out_dir: Path) -> str:
    groups = {"good-ir": ("good", "ir"), "good-vis": ("good", "vis"),
              "bad-ir": ("bad", "ir"), "bad-vis": ("bad", "vis")}
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    for axis, (title, (group, modality)) in zip(axes, groups.items()):
        accumulator = np.zeros((6, 8))
        count = 0
        for json_path in sorted(grid_dir.glob("*.json")):
            if json_path.name == "run_manifest.json":
                continue
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            if meta.get("group") != group or meta.get("kind") != "source" or meta.get("modality") != modality:
                continue
            arrays = LazyArrays(grid_dir / f"{json_path.stem}.npz")
            tau = tau_det.get(modality, 0.0)
            for index in range(arrays["window"].shape[0]):
                cells = {name: arrays[name][index] for name in
                         ("ncc_raw", "peak2_raw", "closure_raw", "ncc_dog", "peak2_dog", "closure_dog")}
                flat = arrays["flat"][index].astype(bool)
                dog_mask = decide(cells, flat, "dog", DOG_NCC_THRESH, DOG_CLOSURE_THRESH, DOG_UNIQ_THRESH)
                detail = arrays["lap_var_cell"][index] >= tau
                accumulator += (detail & dog_mask).astype(float)
                count += 1
        image = accumulator / max(count, 1)
        axis.imshow(image, cmap="RdYlGn", vmin=0, vmax=1)
        axis.set_title(f"{title}\n可用格占比均值（{count} 次评估）", fontsize=10)
        axis.set_xticks(range(8))
        axis.set_yticks(range(6))
        axis.tick_params(labelsize=7)
        for row in range(6):
            for col in range(8):
                axis.text(col, row, f"{image[row, col]:.2f}", ha="center", va="center", fontsize=6.5)
    figure.suptitle("6×8 网格的空间分布：好片 vs 坏片（格内数字＝该格可用比例）", fontsize=13)
    figure.tight_layout()
    path = out_dir / "chart_grid_heatmap.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return str(path)


def make_auc(plt, compare_rows: list[dict], out_dir: Path) -> str:
    grid_metrics = ["tgr_raw", "tgr_dog", "det_ratio", "ugr", "tgr_dog_best_block", "flat_cells",
                    "lambda_min_p50", "lap_var_cell_p50"]
    quality_metrics = ["lap_var", "edge_density", "global_contrast", "blockiness", "entropy"]
    figure, axes = plt.subplots(1, 2, figsize=(16, 5.2))
    for axis, modality in zip(axes, ("ir", "vis")):
        rows = {row["metric"]: row for row in compare_rows if row["modality"] == modality}
        labels, clip_auc, window_auc = [], [], []
        for metric in grid_metrics + quality_metrics:
            row = rows.get(metric)
            if row is None:
                continue
            labels.append(metric)
            clip_auc.append(row["clip_auc_good_vs_bad"])
            window_auc.append(row["window_auc_good_vs_bad"])
        positions = np.arange(len(labels))
        axis.plot(positions, clip_auc, "o-", label="视频级 AUC")
        axis.plot(positions, window_auc, "s--", label="窗口级 AUC")
        axis.axhline(0.5, color="#888888", linewidth=1, linestyle=":")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        axis.set_ylim(0, 1.05)
        axis.set_title(f"{modality.upper()}：网格指标 vs 画质指标的区分度", fontsize=11)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)
    figure.suptitle("判别力对比：AUC=0.5 表示没有区分度", fontsize=13)
    figure.tight_layout()
    path = out_dir / "chart_grid_auc.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return str(path)


def make_redline(plt, window_rows: list[dict], out_dir: Path) -> str:
    figure, axes = plt.subplots(1, 2, figsize=(16, 5.2))
    for axis, metric, label in ((axes[0], "ugr", "UGR（可用格占比）"),
                                (axes[1], "tgr_dog", "TGR（DoG 域）")):
        for group, color in (("good", GROUP_COLOR["good"]), ("bad", GROUP_COLOR["bad"])):
            subset = [row for row in window_rows if row["group"] == group and row["kind"] == "source"]
            x = [float(row[metric]) for row in subset]
            y = [int(row["core_violations"]) for row in subset]
            axis.scatter(x, y, s=6, alpha=0.25, color=color,
                         label="好视频组" if group == "good" else "坏视频组")
        axis.set_xlabel(label)
        axis.set_ylabel("该秒命中的核心红线条数")
        axis.set_yticks([0, 1, 2, 3, 4, 5])
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=9)
    figure.suptitle("画质红线 vs 任意点选可用率（每个点＝1 秒）", fontsize=13)
    figure.tight_layout()
    path = out_dir / "chart_grid_redline.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return str(path)


def make_crosstab(plt, clip_rows: list[dict], out_dir: Path) -> str:
    tiers = ["通过", "通过（弱红线告警）", "人工复核", "拒绝接入"]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    for axis, metric, title in ((axes[0], "ugr_p50", "UGR（可用格占比）"),
                                (axes[1], "tgr_dog_p50", "TGR（DoG 域）")):
        for index, tier in enumerate(tiers):
            subset = [row for row in clip_rows if row["tier"] == tier]
            if not subset:
                continue
            values = [float(row[metric]) for row in subset]
            axis.scatter([index] * len(values), values, s=70, color=TIER_COLOR[tier], alpha=0.85,
                         label=f"{tier}（{len(values)} 条）")
            axis.hlines(float(np.median(values)), index - 0.25, index + 0.25, color="#212121", linewidth=2)
        axis.set_xticks(range(len(tiers)))
        axis.set_xticklabels(tiers, fontsize=9)
        axis.set_ylim(-0.03, 1.05)
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle("画质红线分档 × 任意点选可用率（每条视频一个点）", fontsize=13)
    figure.tight_layout()
    path = out_dir / "chart_grid_crosstab.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return str(path)


def make_timeline(plt, window_rows: list[dict], out_dir: Path, names: list[str]) -> str:
    figure, axes = plt.subplots(len(names), 1, figsize=(15, 3.0 * len(names)), sharex=False)
    if len(names) == 1:
        axes = [axes]
    for axis, name in zip(axes, names):
        subset = sorted((row for row in window_rows if row["file"] == name), key=lambda row: row["window"])
        axis.plot([int(row["window"]) for row in subset], [float(row["ugr"]) for row in subset],
                  linewidth=1.0, color="#C62828", label="UGR（可用格占比）")
        axis.plot([int(row["window"]) for row in subset], [float(row["tgr_dog"]) for row in subset],
                  linewidth=1.0, color="#1565C0", alpha=0.7, label="TGR（DoG 域）")
        axis.set_ylabel("占比", fontsize=9)
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.25)
        axis.set_title(short(name), fontsize=10)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("秒")
    figure.suptitle("网格可跟踪率时间线（每秒 2 对帧）", fontsize=13)
    figure.tight_layout()
    path = out_dir / "chart_grid_timeline.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid trackability figures")
    parser.add_argument("--grid-dir", type=str, required=True)
    parser.add_argument("--report-dir", type=str, required=True)
    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)
    report_dir = Path(args.report_dir)
    plt = setup_fonts()

    clip_rows = read_csv(report_dir / "grid_summary.csv")
    window_rows = read_csv(report_dir / "grid_windows.csv")
    compare_rows = read_csv(report_dir / "grid_compare.csv")
    thresholds = json.loads((report_dir / "grid_thresholds.json").read_text(encoding="utf-8"))
    tau_det = thresholds["tau_det"]

    charts = [
        make_perclip(plt, clip_rows, report_dir),
        make_heatmap(plt, grid_dir, tau_det, report_dir),
        make_auc(plt, compare_rows, report_dir),
        make_redline(plt, window_rows, report_dir),
        make_crosstab(plt, clip_rows, report_dir),
    ]
    names = [row["file"] for row in clip_rows
             if row["file"] in ("VIDEO00012_19700101_010308VIS.h264", "VIDEO00007_19700101_012038IR.h264",
                                "120BT-04-红外02.mp4", "120BT-04-可见光.mp4")]
    if names:
        charts.append(make_timeline(plt, window_rows, report_dir, names))
    print(json.dumps({"charts": charts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
