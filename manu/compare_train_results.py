#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare YOLO26 and Heatmap UAV models fairly under unified Point-Detection Criteria (Distance <= 4.0px).

Supports:
1. Standard YOLO26 weights (.pt) -> Converts bbox predictions to center points (cx, cy)
2. Heatmap weights (.pt) -> Directly outputs peak points
3. Evaluates all models on the validation dataset under the exact same distance threshold
4. Searches best threshold, calculates Recall, Precision, and F1-Score
5. Plots comparison bar charts and prints/saves Markdown and CSV reports
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import font_manager
from tqdm import tqdm

# Add repo root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG, colorstr
from manu.heatmap_model import YOLO26HeatmapDetector
from manu.heatmap_evaluate import extract_peaks, evaluate_point_detections


def configure_chinese_font():
    """选择系统中可用的中文字体，避免图例和标签显示为方框。"""
    preferred_fonts = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
    ]
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next((f for f in preferred_fonts if f in installed_fonts), None)
    if selected_font is not None:
        mpl.rcParams["font.family"] = selected_font
    else:
        mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["axes.unicode_minus"] = False


configure_chinese_font()


# ======================== 用户配置区 ========================
# 支持传入两种类型的模型权重路径或目录：
# 1. 原始 YOLO26 模型（如 yolo26np2.pt、best.pt 等）
# 2. Heatmap 模型（如 best_recall.pt、best_f1.pt）
DEFAULT_MODELS = [
    # 原始 YOLO26 系列（可填入你的历史最优权重路径）
    Path("runs/optuna_uav_recall_sgpu/trial_0028/weights/best.pt"),
    # Heatmap 模型系列
    Path("runs/heatmap_uav/uav_gpu23_heatmap/weights/best_recall.pt"),
    Path("runs/heatmap_uav_s2/uav_gpu23_heatmap_stride2/weights/best_recall.pt"),
]

# 显示别名（留空则自动从文件名或目录名生成）
DEFAULT_LABELS = []

# 默认数据集路径
DEFAULT_DATA = "/mnt/data/siping/datasets/manu/uav/data.yaml"

# 默认输出图表目录
DEFAULT_OUTPUT_DIR = Path("runs/compare_eval")
# ============================================================


def merge_points_ensemble(
    pts_yolo: np.ndarray,
    scores_yolo: np.ndarray,
    pts_hm: np.ndarray,
    scores_hm: np.ndarray,
    match_dist: float = 4.0,
    boost_weight: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """非对称互证策略 (Heatmap-Anchor + YOLO Verification)：

    1. 以高精度的 Heatmap 为主基准（保留其 100% 极小目标捕获力，拒绝 YOLO 单边噪点倒灌）
    2. 当 YOLO 在 match_dist 像素内同样给出响应时，强力增强其置信度（极大拉高 Precision）
    """
    if len(pts_hm) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    if len(pts_yolo) == 0:
        return pts_hm, scores_hm

    diff = pts_hm[:, np.newaxis, :] - pts_yolo[np.newaxis, :, :]  # (M, N, 2)
    dists = np.sqrt(np.sum(diff**2, axis=-1))

    merged_scores = scores_hm.copy()
    # 查找每个 Heatmap 点距离最近的 YOLO 点
    min_dists = np.min(dists, axis=1)  # (M,)
    closest_yolo = np.argmin(dists, axis=1)

    matched = min_dists <= match_dist
    # 双方互证增强：置信度跃升
    merged_scores[matched] = np.minimum(
        1.0, merged_scores[matched] + scores_yolo[closest_yolo[matched]] * boost_weight
    )

    return pts_hm, merged_scores


def get_cache_path(cache_dir: Path, model_path: Path, data_path: str, imgsz: int) -> Path:
    """根据模型路径、数据路径与图像尺寸生成唯一的缓存文件名。"""
    stem = model_path.stem
    parent_name = model_path.parent.parent.name if model_path.parent.name == "weights" else model_path.parent.name
    data_stem = Path(data_path).stem
    return cache_dir / f"pred_{parent_name}_{stem}_{data_stem}_{imgsz}.json"


def save_preds_cache(cache_path: Path, model_type: str, all_points_raw: list[dict[str, np.ndarray]]):
    """序列化并保存模型的预测点与置信度到 JSON 文件。"""
    serialized = []
    for p in all_points_raw:
        serialized.append({
            "points": p["points"].tolist(),
            "scores": p["scores"].tolist(),
        })
    data = {
        "model_type": model_type,
        "preds": serialized,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"[Cache] Saved inference predictions to: {cache_path}")


def load_preds_cache(cache_path: Path) -> tuple[str, list[dict[str, np.ndarray]]]:
    """从 JSON 缓存文件恢复模型的预测点与置信度。"""
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    model_type = data["model_type"]
    all_points_raw = []
    for item in data["preds"]:
        all_points_raw.append({
            "points": np.array(item["points"], dtype=np.float32) if item["points"] else np.zeros((0, 2), dtype=np.float32),
            "scores": np.array(item["scores"], dtype=np.float32) if item["scores"] else np.zeros((0,), dtype=np.float32),
        })
    print(f"[Cache] Loaded cached predictions from: {cache_path}")
    return model_type, all_points_raw


def get_ground_truth_and_sizes(val_loader, imgsz: int) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """仅从 DataLoader 中提取 Ground-Truth 框与图像尺寸（无需加载模型推理）。"""
    val_gt_list = []
    val_sizes_list = []
    for batch in val_loader:
        bboxes = batch["bboxes"]
        b_idx = batch["batch_idx"]
        bs = batch["img"].shape[0]
        for b in range(bs):
            mask_b = b_idx == b
            val_gt_list.append(bboxes[mask_b].cpu().numpy())
            val_sizes_list.append((imgsz, imgsz))
    return val_gt_list, val_sizes_list


def is_heatmap_model(ckpt: dict) -> bool:
    """自动判断权重文件是 Heatmap 模型还是原生 YOLO 模型。"""
    state_dict = ckpt.get("model", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    if isinstance(state_dict, dict):
        keys = list(state_dict.keys())
        if any("head.heatmap" in k or "fuse_p2" in k or "fuse_p1" in k for k in keys):
            return True
    return False


def load_eval_data(data_path: str, imgsz: int = 640, batch: int = 64):
    """构建统一的验证集 DataLoader。"""
    data_dict = check_det_dataset(data_path)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = imgsz
    cfg.data = data_path
    val_dataset = build_yolo_dataset(cfg, data_dict["val"], batch=batch, data=data_dict, mode="val", stride=32)
    val_loader = build_dataloader(val_dataset, batch=batch, workers=8, shuffle=False)
    return val_dataset, val_loader


def run_inference_as_points(
    model_path: Path,
    val_loader,
    device: str = "2",
    imgsz: int = 640,
) -> tuple[str, list[dict[str, np.ndarray]], list[np.ndarray], list[tuple[int, int]]]:
    """
    统一推理函数：无论输入是 YOLO 还是 Heatmap，输出都统一转换为点 (cx, cy) 格式。
    返回: (model_type, all_points_raw, val_gt_list, val_sizes_list)
    """
    dev = torch.device(f"cuda:{device}" if torch.cuda.is_available() and device != "cpu" else "cpu")
    ckpt = torch.load(model_path, map_location="cpu")

    all_points_raw = []
    val_gt_list = []
    val_sizes_list = []

    if is_heatmap_model(ckpt):
        model_type = "Heatmap"
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        stride = ckpt.get("stride", 4)
        model = YOLO26HeatmapDetector(stride=stride, num_classes=1)
        model.load_state_dict(state_dict)
        model.to(dev)
        model.eval()

        print(colorstr("bold", f"[{model_path.name}] Evaluated as Heatmap Model (stride={stride})"))
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Infer {model_path.stem[:15]}"):
                imgs = batch["img"].to(dev, non_blocking=True).float() / 255.0
                bboxes = batch["bboxes"]
                b_idx = batch["batch_idx"]
                bs = imgs.shape[0]

                preds = model(imgs)
                peaks = extract_peaks(
                    heatmap=preds["heatmap"],
                    offset=preds["offset"],
                    stride=stride,
                    conf_thresh=0.05,
                    top_k=100,
                )
                all_points_raw.extend(peaks)

                for b in range(bs):
                    mask_b = b_idx == b
                    val_gt_list.append(bboxes[mask_b].cpu().numpy())
                    val_sizes_list.append((imgsz, imgsz))

    else:
        model_type = "YOLO26"
        model = YOLO(str(model_path))
        print(colorstr("bold", f"[{model_path.name}] Evaluated as YOLO26 (Bbox converted to Center Points)"))

        for batch in tqdm(val_loader, desc=f"Infer {model_path.stem[:15]}"):
            imgs = batch["img"].float() / 255.0
            bboxes = batch["bboxes"]
            b_idx = batch["batch_idx"]
            bs = imgs.shape[0]

            results = model.predict(imgs, conf=0.01, verbose=False, device=str(dev))

            for b in range(bs):
                res = results[b]
                boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) > 0 else np.zeros((0, 4))
                confs = res.boxes.conf.cpu().numpy() if len(res.boxes) > 0 else np.zeros((0,))

                if len(boxes) > 0:
                    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
                    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
                    pts = np.stack([cx, cy], axis=1)
                else:
                    pts = np.zeros((0, 2), dtype=np.float32)

                all_points_raw.append({"points": pts, "scores": confs})

                mask_b = b_idx == b
                val_gt_list.append(bboxes[mask_b].cpu().numpy())
                val_sizes_list.append((imgsz, imgsz))

    return model_type, all_points_raw, val_gt_list, val_sizes_list


def evaluate_model_at_thresholds(
    all_points_raw: list[dict[str, np.ndarray]],
    val_gt_list: list[np.ndarray],
    val_sizes_list: list[tuple[int, int]],
    dist_thresh: float = 4.0,
    thresholds: list[float] | None = None,
) -> dict:
    """在统一距离阈值下，扫描置信度并获得最佳 F1、Recall 与 Precision。"""
    if thresholds is None:
        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]

    best_f1 = -1.0
    best_th = 0.20
    best_metrics = {}
    curve_data = []

    for th in thresholds:
        th_preds = []
        for p in all_points_raw:
            keep = p["scores"] >= th
            th_preds.append({
                "points": p["points"][keep],
                "scores": p["scores"][keep],
            })

        m = evaluate_point_detections(
            predictions=th_preds,
            gt_boxes_list=val_gt_list,
            img_sizes=val_sizes_list,
            distance_threshold=dist_thresh,
        )

        curve_data.append({"threshold": th, **m})

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_th = th
            best_metrics = m

    best_metrics["best_th"] = best_th
    best_metrics["curve"] = curve_data
    return best_metrics


def plot_metrics_barchart(summary_df: pd.DataFrame, output_dir: Path):
    """参考 compare_train_results 风格绘制各模型最佳点检测指标对比柱状图。"""
    metrics = ["Recall", "Precision", "F1-Score"]
    n_metrics = len(metrics)
    n_models = len(summary_df)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    x = np.arange(n_metrics)
    total_width = 0.8
    bar_width = total_width / n_models

    # 使用柔和调色板
    colors = plt.cm.get_cmap("tab10", n_models)

    for i, (_, row) in enumerate(summary_df.iterrows()):
        values = [row["Recall"], row["Precision"], row["F1-Score"]]
        offsets = x - (total_width / 2) + (i + 0.5) * bar_width
        bars = ax.bar(offsets, values, bar_width, label=row["Model"], color=colors(i), alpha=0.9)

        # 柱子上方标注数值
        for bar, val in zip(bars, values):
            ax.annotate(
                f"{val:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_title("UAV 极小目标模型点检测基准对比 (Distance <= 4.0px)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric Value (0 ~ 1.0)", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9)

    plt.tight_layout()
    chart_path = output_dir / "compare_best_metrics.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"[Chart] Best metrics bar chart saved to: {chart_path}")


def main():
    parser = argparse.ArgumentParser(description="Fair Comparison between YOLO26 and Heatmap Point Detections")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of model checkpoint paths (.pt) to compare",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Custom labels for each model",
    )
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to data.yaml")
    parser.add_argument("--device", type=str, default="1", help="CUDA device index or cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--dist_thresh", type=float, default=4.0, help="Distance threshold (pixels) for TP")
    parser.add_argument("--solo_thresh", type=float, default=0.35, help="Solo detection threshold for Ensemble")
    parser.add_argument("--boost_weight", type=float, default=0.35, help="Boost weight when YOLO verifies Heatmap detection")
    parser.add_argument("--ensemble", action="store_true", help="Enable Heterogeneous Ensemble fusion evaluation (default: disabled)")
    parser.add_argument("--cache_dir", type=str, default="runs/compare_eval/cache", help="Directory to save/load prediction cache JSON")
    parser.add_argument("--no_cache", action="store_true", help="Force re-inference without using cache")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    # 处理模型路径列表
    model_paths = [Path(p) for p in (args.models or DEFAULT_MODELS)]
    valid_models = []
    for p in model_paths:
        if p.is_dir():
            best_pt = p / "weights/best_recall.pt"
            if not best_pt.exists():
                best_pt = p / "weights/best.pt"
            if best_pt.exists():
                valid_models.append(best_pt)
            else:
                print(f"[WARN] No .pt found in {p}, skipping.")
        elif p.exists():
            valid_models.append(p)
        else:
            print(f"[WARN] File not found: {p}, skipping.")

    if len(valid_models) == 0:
        print("[ERROR] No valid model checkpoints found for comparison!")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 模型标签
    labels = args.labels or DEFAULT_LABELS
    if not labels or len(labels) != len(valid_models):
        labels = [m.parent.parent.name if m.parent.name == "weights" else m.stem for m in valid_models]

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否全部命中缓存
    all_cached = (not args.no_cache) and all(
        get_cache_path(cache_dir, p, args.data, args.imgsz).exists() for p in valid_models
    )

    # 仅在需要实际跑模型推理，或者需要提取 GT 评测时才构建 DataLoader
    print(colorstr("bold", f"Loading validation dataset: {args.data}"))
    val_dataset, val_loader = load_eval_data(args.data, imgsz=args.imgsz, batch=args.batch)

    summary_rows = []
    collected_preds = {}

    print("\n" + "=" * 80)
    print("Running Unified Point-Detection Evaluation (Apple-to-Apple)")
    print("=" * 80)

    gt_list_ref = None
    sizes_list_ref = None

    for path, label in zip(valid_models, labels):
        c_path = get_cache_path(cache_dir, path, args.data, args.imgsz)
        if not args.no_cache and c_path.exists():
            m_type, raw_preds = load_preds_cache(c_path)
            if gt_list_ref is None or sizes_list_ref is None:
                gt_list_ref, sizes_list_ref = get_ground_truth_and_sizes(val_loader, imgsz=args.imgsz)
            gt_list = gt_list_ref
            sizes_list = sizes_list_ref
        else:
            m_type, raw_preds, gt_list, sizes_list = run_inference_as_points(
                path, val_loader, device=args.device, imgsz=args.imgsz
            )
            save_preds_cache(c_path, m_type, raw_preds)
            gt_list_ref = gt_list
            sizes_list_ref = sizes_list

        collected_preds[label] = {"type": m_type, "preds": raw_preds}

        res = evaluate_model_at_thresholds(
            raw_preds, gt_list, sizes_list, dist_thresh=args.dist_thresh
        )

        summary_rows.append({
            "Model": label,
            "Type": m_type,
            "Best_Th": res["best_th"],
            "Recall": res["recall"],
            "Precision": res["precision"],
            "F1-Score": res["f1"],
            "TP": res["tp"],
            "FP": res["fp"],
            "GT": res["total_gt"],
            "Weights": str(path),
        })

    # 当启用 --ensemble 且存在至少一个 YOLO 模型和一个 Heatmap 模型时，进行异构 Ensemble 融合对比
    if args.ensemble:
        yolo_labels = [lbl for lbl, item in collected_preds.items() if item["type"] == "YOLO26"]
        hm_labels = [lbl for lbl, item in collected_preds.items() if item["type"] == "Heatmap"]

        if yolo_labels and hm_labels:
            best_yolo_lbl = yolo_labels[0]
            best_hm_lbl = hm_labels[0]
            ens_label = f"Ensemble({best_yolo_lbl[:10]}+{best_hm_lbl[:10]})"
            print(colorstr("bold", f"\n[Ensemble] Fusing {best_yolo_lbl} and {best_hm_lbl} (Heatmap-Anchor + YOLO Verification)..."))

            yolo_raw = collected_preds[best_yolo_lbl]["preds"]
            hm_raw = collected_preds[best_hm_lbl]["preds"]

            ens_raw = []
            for yp, hp in zip(yolo_raw, hm_raw):
                m_pts, m_scores = merge_points_ensemble(
                    pts_yolo=yp["points"],
                    scores_yolo=yp["scores"],
                    pts_hm=hp["points"],
                    scores_hm=hp["scores"],
                    match_dist=args.dist_thresh,
                    boost_weight=args.boost_weight,
                )
                ens_raw.append({"points": m_pts, "scores": m_scores})

            res_ens = evaluate_model_at_thresholds(
                ens_raw, gt_list_ref, sizes_list_ref, dist_thresh=args.dist_thresh
            )

            summary_rows.append({
                "Model": ens_label,
                "Type": "Ensemble",
                "Best_Th": res_ens["best_th"],
                "Recall": res_ens["recall"],
                "Precision": res_ens["precision"],
                "F1-Score": res_ens["f1"],
                "TP": res_ens["tp"],
                "FP": res_ens["fp"],
                "GT": res_ens["total_gt"],
                "Weights": "Fused in memory",
            })

    summary_df = pd.DataFrame(summary_rows)

    # 打印对比结果表格
    print("\n" + "=" * 95)
    print(f"{'Model Label':<28} | {'Type':<8} | {'Best_Th':<8} | {'Recall':<8} | {'Precision':<10} | {'F1':<8} | {'TP':<6} | {'FP':<6}")
    print("-" * 95)
    for _, r in summary_df.iterrows():
        print(f"{r['Model'][:28]:<28} | {r['Type']:<8} | {r['Best_Th']:<8.2f} | {r['Recall']:<8.4f} | {r['Precision']:<10.4f} | {r['F1-Score']:<8.4f} | {r['TP']:<6} | {r['FP']:<6}")
    print("=" * 95)

    # 保存 CSV
    csv_file = output_dir / "comparison_point_metrics.csv"
    summary_df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"\n[Summary] Metrics table saved to: {csv_file}")

    # 保存 Markdown 报告
    md_file = output_dir / "comparison_report.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# UAV 极小目标模型统一评测对比报告 (Apple-to-Apple)\n\n")
        f.write(f"- **判决基准**: 预测点与真实目标中心欧式距离 $\\le {args.dist_thresh:.1f}$ 像素即为 True Positive\n")
        f.write(f"- **测试数据集**: `{args.data}` (图像总数: {len(val_dataset)})\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n")
    print(f"[Summary] Markdown report saved to: {md_file}")

    # 绘制对比柱状图
    plot_metrics_barchart(summary_df, output_dir)


if __name__ == "__main__":
    main()
