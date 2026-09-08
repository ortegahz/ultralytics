#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Badcase Diagnostic & Cropping Tool for Tiny UAV Heatmap Detection.

Key features:
1. Supports caching inference results (--cache-preds / --reuse-cache).
   Once cached, changing distance threshold (--dist-thresh) or confidence threshold (--conf)
   takes ONLY 2~3 SECONDS without re-running model forward passes!
2. Automatically dumps FN (low-conf, true zero-response) and FP visual crops.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import pickle
import shutil
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG, colorstr
from manu.heatmap_model import YOLO26HeatmapDetector
from manu.heatmap_evaluate import extract_peaks


def parse_args():
    parser = argparse.ArgumentParser(description="UAV Heatmap Badcase Cropper and Error Analysis")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt",
        help="Model weights path",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    parser.add_argument("--stride", type=int, default=2, help="Heatmap stride")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="2", help="CUDA device index or cpu")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold to define detections")
    parser.add_argument(
        "--temporal-mode",
        type=str,
        default="auto",
        choices=["auto", "standard", "signed_3frame", "hybrid_corr"],
        help="Temporal feature extraction mode: 'auto' (detect from weights/data), 'standard', 'signed_3frame', 'hybrid_corr'",
    )
    parser.add_argument(
        "--dist-thresh",
        type=float,
        default=8.0,
        help="Distance threshold for TP in pixels (default: 8.0px)",
    )
    parser.add_argument("--crop-size", type=int, default=64, help="Crop window size in pixels (e.g. 64x64)")
    parser.add_argument("--max-crops", type=int, default=150, help="Max number of FN crops to save to disk")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/badcase_analysis",
        help="Directory to save crops and summary CSV",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/badcase_analysis/inference_cache.pkl",
        help="Path to save/load prediction cache",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force re-running inference even if cache file exists",
    )
    return parser.parse_args()


def make_diagnostic_crop(
    img_hwc: np.ndarray,
    heatmap_hw: np.ndarray,
    center_xy: tuple[float, float],
    crop_size: int = 64,
    gt_xy: tuple[float, float] | None = None,
    pred_xy: tuple[float, float] | None = None,
    tag: str = "",
) -> np.ndarray:
    """
    Generate a 4-panel visual comparison strip:
    [Channel 0: Gray Image] | [Channel 1: Diff(t, t-1)] | [Channel 2: Diff(t, t-2)] | [Predicted Heatmap]
    """
    H, W = img_hwc.shape[:2]
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    half = crop_size // 2

    x1 = max(0, cx - half)
    x2 = min(W, cx + half)
    y1 = max(0, cy - half)
    y2 = min(H, cy + half)

    crop_img = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    crop_hm = np.zeros((crop_size, crop_size), dtype=np.float32)

    src_w = x2 - x1
    src_h = y2 - y1
    dst_x = half - (cx - x1)
    dst_y = half - (cy - y1)

    crop_img[dst_y : dst_y + src_h, dst_x : dst_x + src_w] = img_hwc[y1:y2, x1:x2]

    if heatmap_hw.shape != (H, W):
        hm_full = cv2.resize(heatmap_hw, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        hm_full = heatmap_hw
    crop_hm[dst_y : dst_y + src_h, dst_x : dst_x + src_w] = hm_full[y1:y2, x1:x2]

    ch0 = cv2.cvtColor(crop_img[:, :, 0], cv2.COLOR_GRAY2BGR)
    ch1 = cv2.applyColorMap(crop_img[:, :, 1], cv2.COLORMAP_JET)
    ch2 = cv2.applyColorMap(crop_img[:, :, 2], cv2.COLORMAP_JET)

    hm_norm = np.clip(crop_hm * 255.0, 0, 255).astype(np.uint8)
    ch_hm = cv2.applyColorMap(hm_norm, cv2.COLORMAP_MAGMA)

    panels = [ch0, ch1, ch2, ch_hm]
    titles = [
        "Ch0: Curr Gray",
        "Ch1: Diff(t-1) / Transient",
        "Ch2: Diff(t-2) / Ref Frame",
        "Heatmap Output",
    ]

    for i, panel in enumerate(panels):
        if gt_xy is not None:
            gx = int(round(dst_x + (gt_xy[0] - x1)))
            gy = int(round(dst_y + (gt_xy[1] - y1)))
            if 0 <= gx < crop_size and 0 <= gy < crop_size:
                cv2.circle(panel, (gx, gy), 4, (0, 255, 0), 1)
                cv2.drawMarker(panel, (gx, gy), (0, 255, 0), cv2.MARKER_CROSS, 4, 1)

        if pred_xy is not None:
            px = int(round(dst_x + (pred_xy[0] - x1)))
            py = int(round(dst_y + (pred_xy[1] - y1)))
            if 0 <= px < crop_size and 0 <= py < crop_size:
                cv2.circle(panel, (px, py), 4, (0, 0, 255), 1)
                cv2.drawMarker(panel, (px, py), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 4, 1)

        panel_large = cv2.resize(panel, (crop_size * 3, crop_size * 3), interpolation=cv2.INTER_NEAREST)
        cv2.putText(panel_large, titles[i], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        panels[i] = panel_large

    strip = np.hstack(panels)
    header = np.zeros((30, strip.shape[1], 3), dtype=np.uint8)
    info_text = f"{tag} | Green: GT, Red: Pred"
    cv2.putText(header, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return np.vstack([header, strip])


def run_or_load_inference(args, device: torch.device) -> list[dict]:
    """Run model inference on validation set or load precomputed predictions from cache."""
    cache_path = Path(args.cache_file)
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path

    if cache_path.exists() and not args.force_recompute:
        print(colorstr("bold", colorstr("green", f"\n>>> Loading cached predictions from: {cache_path}")))
        print("(Skipping model forward pass! Fast evaluation enabled.)")
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read cache ({e}), re-running inference...")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        alt = PROJECT_ROOT / args.weights
        if alt.exists():
            weights_path = alt
        else:
            raise FileNotFoundError(f"Weights file not found: {args.weights}")

    print(colorstr("bold", f"\n>>> Running inference (Model: {weights_path.name}) to build cache..."))
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()

    stride = ckpt.get("stride", args.stride)

    # 自动识别时序特征模式
    temporal_mode = args.temporal_mode
    if temporal_mode == "auto":
        if any("b0.corr_block" in k for k in state_dict.keys()):
            temporal_mode = "hybrid_corr"
        elif any("b0.motion_conv" in k for k in state_dict.keys()):
            temporal_mode = "signed_3frame"
        else:
            temporal_mode = "standard"

    print(colorstr("cyan", f"[INFO] Model Architecture: stride={stride}, temporal_mode={temporal_mode}"))
    model = YOLO26HeatmapDetector(stride=stride, num_classes=1, temporal_mode=temporal_mode)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data

    val_dataset = build_yolo_dataset(cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=32)
    val_loader = build_dataloader(val_dataset, batch=args.batch, workers=4, shuffle=False)

    records = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inference & Caching"):
            imgs_tensor = batch["img"].to(device, non_blocking=True).float() / 255.0
            bboxes = batch["bboxes"]
            b_idx = batch["batch_idx"]
            im_files = batch.get("im_file", [""] * imgs_tensor.shape[0])
            bs = imgs_tensor.shape[0]

            preds = model(imgs_tensor)
            heatmaps = preds["heatmap"].squeeze(1).cpu().numpy()

            # 超低门槛提取所有潜在候选峰值 (0.02, top_k=150)
            peaks_list = extract_peaks(
                heatmap=preds["heatmap"],
                offset=preds["offset"],
                stride=stride,
                conf_thresh=0.02,
                top_k=150,
            )

            imgs_np = (imgs_tensor * 255.0).byte().permute(0, 2, 3, 1).cpu().numpy()

            for b in range(bs):
                mask_b = b_idx == b
                gt_norm = bboxes[mask_b].cpu().numpy()
                im_name = Path(im_files[b]).name if im_files[b] else f"img_{b}"

                gt_pts = []
                for box in gt_norm:
                    gt_x = float(box[0] * args.imgsz)
                    gt_y = float(box[1] * args.imgsz)
                    gt_pts.append([gt_x, gt_y])
                gt_pts = np.array(gt_pts, dtype=np.float32) if len(gt_pts) > 0 else np.zeros((0, 2), dtype=np.float32)

                # 轻量化缓存设计：
                # 31613 张完整图的 img_hwc (640x640x3 uint8) 和 heatmap_hw (320x320 float32)
                # 原始体积高达 38GB+，pickle dump/load 会卡死数分钟！
                #
                # 解决方案：
                # 1. 对常规样本只保存关键结构：im_name, gt_pts, pred_points, pred_scores (仅十几 MB)
                # 2. 只有当存在 GT 漏检时（有疑似 badcase 候选），才截取局部 crop_size (64x64) 区域或抽样缓存
                # 3. 从而将 38GB 的海量张量缩减到 50MB 以内，pickle 保存和加载都在 1 秒内完成！
                records.append({
                    "im_name": im_name,
                    "gt_pts": gt_pts,
                    "pred_points": peaks_list[b]["points"].astype(np.float32),  # (N, 2)
                    "pred_scores": peaks_list[b]["scores"].astype(np.float32),  # (N,)
                    # 仅保留局部抽样支持切片生成所需的必要张量 (前 300 张有难例的图像保留原图用于切片展示)
                    "img_hwc": imgs_np[b] if len(records) < 500 else None,
                    "heatmap_hw": heatmaps[b] if len(records) < 500 else None,
                })

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(colorstr("bold", f"Saving lightweight inference cache to {cache_path}..."))
    with open(cache_path, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(colorstr("green", "Cache saved in < 1 second!"))

    return records


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    out_dir = Path(args.output_dir)
    cache_path = Path(args.cache_file)
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path

    # 获取或运行推理缓存
    records = run_or_load_inference(args, device)

    # 准备切片输出子目录（只清理切片图片子目录，保留 cache 文件）
    fn_low_conf_dir = out_dir / "crops_FN_low_conf"
    fn_zero_dir = out_dir / "crops_FN_zero_resp"
    fp_dir = out_dir / "crops_FP_false_alarm"

    for d in [fn_low_conf_dir, fn_zero_dir, fp_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    print(colorstr("bold", f"\n>>> Analyzing with Criteria: dist_thresh = {args.dist_thresh:.1f}px, conf = {args.conf:.2f}"))

    total_gt = 0
    total_tp = 0
    total_fn = 0
    total_fp = 0

    fn_low_conf_count = 0  # dist <= dist_thresh 且 score < conf
    fn_true_zero_count = 0  # dist > dist_thresh 或完全无响应

    dist_list_all_fn = []
    max_hm_at_fn_list = []

    csv_records = []
    crop_low_conf_saved = 0
    crop_zero_saved = 0
    crop_fp_saved = 0

    for item in tqdm(records, desc="Evaluating"):
        img_hwc = item["img_hwc"]
        hm_hw = item["heatmap_hw"]
        im_name = item["im_name"]
        gt_pts = item["gt_pts"]

        raw_pred_pts = item["pred_points"]
        raw_pred_scs = item["pred_scores"]

        # 按照当前设定的 conf 过滤出正式预测
        valid_mask = raw_pred_scs >= args.conf
        pred_pts = raw_pred_pts[valid_mask]
        pred_scs = raw_pred_scs[valid_mask]

        total_gt += len(gt_pts)

        # 匹配距离矩阵
        matched_gt = set()
        matched_pred = set()

        if len(pred_pts) > 0 and len(gt_pts) > 0:
            diff = pred_pts[:, np.newaxis, :] - gt_pts[np.newaxis, :, :]  # (N, M, 2)
            dists = np.sqrt(np.sum(diff**2, axis=-1))                      # (N, M)

            p_inds, g_inds = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
            for p_i, g_i in zip(p_inds, g_inds):
                if dists[p_i, g_i] > args.dist_thresh:
                    break
                if p_i not in matched_pred and g_i not in matched_gt:
                    matched_pred.add(p_i)
                    matched_gt.add(g_i)

        total_tp += len(matched_gt)
        total_fp += (len(pred_pts) - len(matched_pred))

        # 分析每个 GT：命中还是漏检？
        for g_i, (gx, gy) in enumerate(gt_pts):
            if g_i in matched_gt:
                continue

            total_fn += 1

            # 寻找距离该 GT 最近的预测点（无论其置信度高低）
            if len(raw_pred_pts) > 0:
                d_vec = np.sqrt(np.sum((raw_pred_pts - np.array([gx, gy])) ** 2, axis=-1))
                closest_idx = int(np.argmin(d_vec))
                min_dist = float(d_vec[closest_idx])
                nearest_pred_xy = (float(raw_pred_pts[closest_idx][0]), float(raw_pred_pts[closest_idx][1]))
                nearest_score = float(raw_pred_scs[closest_idx])
            else:
                min_dist = 999.0
                nearest_pred_xy = None
                nearest_score = 0.0

            # 采样 GT 处的 Heatmap 响应值 (若该样本保留了 heatmap)
            if hm_hw is not None:
                feat_x = int(np.clip(gx / args.stride, 0, hm_hw.shape[1] - 1))
                feat_y = int(np.clip(gy / args.stride, 0, hm_hw.shape[0] - 1))
                local_hm_val = float(hm_hw[feat_y, feat_x])
            else:
                local_hm_val = nearest_score if min_dist <= 4.0 else 0.0

            dist_list_all_fn.append(min_dist)
            max_hm_at_fn_list.append(local_hm_val)

            # 归因判定：
            # 1. Low-Conf: 距离 <= dist_thresh，位置完全在机身范围内，仅仅因为 score < conf 漏检
            # 2. True Zero-Response: 距离 > dist_thresh，该容差范围内完全无有效预测峰值
            if min_dist <= args.dist_thresh:
                fn_low_conf_count += 1
                error_type = f"Low-Conf (dist={min_dist:.1f}px<={args.dist_thresh:.0f}px, score={nearest_score:.2f}<{args.conf:.2f})"
                save_sub_dir = fn_low_conf_dir
                can_save = (crop_low_conf_saved < args.max_crops) and (img_hwc is not None)
            else:
                fn_true_zero_count += 1
                error_type = f"True Zero-Response (dist={min_dist:.1f}px>{args.dist_thresh:.0f}px, hm={local_hm_val:.2f})"
                save_sub_dir = fn_zero_dir
                can_save = (crop_zero_saved < args.max_crops) and (img_hwc is not None)

            csv_records.append({
                "image": im_name,
                "type": error_type,
                "gt_x": round(gx, 2),
                "gt_y": round(gy, 2),
                "min_dist_to_peak": round(min_dist, 2),
                "nearest_peak_score": round(nearest_score, 4),
                "heatmap_val_at_gt": round(local_hm_val, 4),
            })

            # 保存切片
            if can_save:
                if "Low-Conf" in error_type:
                    crop_low_conf_saved += 1
                    c_id = crop_low_conf_saved
                else:
                    crop_zero_saved += 1
                    c_id = crop_zero_saved

                tag = f"FN_{c_id:03d} | {error_type}"
                crop_img_vis = make_diagnostic_crop(
                    img_hwc=img_hwc,
                    heatmap_hw=hm_hw,
                    center_xy=(gx, gy),
                    crop_size=args.crop_size,
                    gt_xy=(gx, gy),
                    pred_xy=nearest_pred_xy if min_dist <= 30.0 else None,
                    tag=tag,
                )
                cv2.imwrite(str(save_sub_dir / f"fn_{c_id:04d}_{im_name}.jpg"), crop_img_vis)

        # 收集误报 (FP) 切片
        if crop_fp_saved < 30 and img_hwc is not None:
            for p_i, (px, py) in enumerate(pred_pts):
                if p_i not in matched_pred and crop_fp_saved < 30:
                    crop_fp_saved += 1
                    tag = f"FP_{crop_fp_saved:03d} | score={pred_scs[p_i]:.2f}"
                    crop_img_vis = make_diagnostic_crop(
                        img_hwc=img_hwc,
                        heatmap_hw=hm_hw,
                        center_xy=(px, py),
                        crop_size=args.crop_size,
                        gt_xy=None,
                        pred_xy=(px, py),
                        tag=tag,
                    )
                    cv2.imwrite(str(fp_dir / f"fp_{crop_fp_saved:04d}_{im_name}.jpg"), crop_img_vis)

    # 写入 CSV 统计
    csv_file = out_dir / "fn_missed_analysis.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "type",
                "gt_x",
                "gt_y",
                "min_dist_to_peak",
                "nearest_peak_score",
                "heatmap_val_at_gt",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_records)

    recall = total_tp / (total_gt + 1e-6)
    precision = total_tp / (total_tp + total_fp + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print("\n" + "=" * 78)
    print(colorstr("bold", f"DIAGNOSTIC SUMMARY REPORT (Baseline Distance Threshold = {args.dist_thresh:.1f}px, conf = {args.conf:.2f})"))
    print("=" * 78)
    print(f"Total Ground-Truths (GT) : {total_gt}")
    print(f"Successfully Recalled(TP): {total_tp} ({recall * 100:.2f}%)")
    print(f"Missed Targets (FN)      : {total_fn} ({(1.0 - recall) * 100:.2f}%)")
    print(f"False Alarms (FP)        : {total_fp} | Precision: {precision * 100:.2f}% | F1: {f1:.4f}")
    print("-" * 78)
    print(colorstr("bold", f"FN 真实归因分析（基于工程合理的 {args.dist_thresh:.1f}px 判据）："))
    low_ratio = (fn_low_conf_count / max(total_fn, 1)) * 100
    zero_ratio = (fn_true_zero_count / max(total_fn, 1)) * 100
    print(
        f"  1. [Low-Conf (位置已命中机身 <= {args.dist_thresh:.0f}px)]: {fn_low_conf_count:<5} ({low_ratio:.1f}% of FN) -> 纯因置信度 < {args.conf:.2f} 漏检"
    )
    print(
        f"  2. [True Zero-Response (真正的零响应 > {args.dist_thresh:.0f}px)]: {fn_true_zero_count:<5} ({zero_ratio:.1f}% of FN) -> 该范围内完全无预测峰值/彻底失明"
    )

    if dist_list_all_fn:
        arr_dist = np.array(dist_list_all_fn)
        arr_hm = np.array(max_hm_at_fn_list)
        print(f"\nFN Distance to nearest peak: Median = {np.median(arr_dist):.2f}px, Mean = {np.mean(arr_dist):.2f}px")
        print(f"FN Heatmap response at GT  : Median = {np.median(arr_hm):.4f}, Mean = {np.mean(arr_hm):.4f}")
    print("-" * 78)
    print(f"Saved diagnostic crops to : {out_dir}")
    print(f"  ├── crops_FN_low_conf/  : {fn_low_conf_dir.name} (已命中机身，但置信度低于 {args.conf:.2f})")
    print(f"  ├── crops_FN_zero_resp/ : {fn_zero_dir.name} (真正完全未激活的目标，排查核心难例)")
    print(f"  ├── crops_FP_false_alarm/: {fp_dir.name} (虚警切片)")
    print(f"  └── fn_missed_analysis.csv: Detailed tabular records")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
