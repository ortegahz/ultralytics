#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Target-Level Heatmap Probe for Infrared Small UAV Sequences.
Directly inspects raw unthresholded heatmap logits & scores at Ground-Truth locations,
and compares target response against full-frame maximum background noise.

Usage on server:
    python manu/probe_target_heatmap.py \
        --weights runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt \
        --seq wg2022_ir_020_split_03 \
        --max-frames 60 \
        --device 0
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manu.heatmap_model import YOLO26HeatmapDetector


def natural_sort_key(path: Path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.stem)]


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def parse_args():
    parser = argparse.ArgumentParser(description="Probe raw heatmap values at GT target position.")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt",
        help="Path to best.pt weights",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav",
        help="UAV dataset root containing images/val and labels/val",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default="wg2022_ir_020_split_03",
        help="Sequence name prefix in images/val",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    parser.add_argument("--stride", type=int, default=2, help="Heatmap stride")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or cpu")
    parser.add_argument("--radius", type=int, default=3, help="Search radius (in feature pixels) around GT")
    parser.add_argument("--max-frames", type=int, default=50, help="Number of frames to probe")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        # Fallback candidate for local mounts
        cand = Path("/home/manu/mnt/pycharm_project_10ae9e2e") / args.weights
        if cand.exists():
            weights_path = cand

    print(f"[INFO] 加载模型权重: {weights_path}")
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    stride = ckpt.get("stride", args.stride)
    imgsz = ckpt.get("imgsz", args.imgsz)

    use_temporal = any("b0.motion_conv" in k for k in state_dict.keys())
    model = YOLO26HeatmapDetector(stride=stride, num_classes=1, use_temporal_stem=use_temporal)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 数据集检索
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        cand_data = Path("/home/manu/mnt/datasets/manu/uav")
        if cand_data.exists():
            data_dir = cand_data

    img_dir = data_dir / "images" / "val"
    lbl_dir = data_dir / "labels" / "val"

    # 寻找匹配该序列的所有图片
    all_imgs = sorted(list(img_dir.glob(f"{args.seq}*.jpg")), key=natural_sort_key)
    if not all_imgs:
        print(f"[ERROR] 在 {img_dir} 下未找到匹配序列: {args.seq}")
        sys.exit(1)

    print(f"[INFO] 找到 {args.seq} 验证集图像共 {len(all_imgs)} 帧，测试前 {min(args.max_frames, len(all_imgs))} 帧...")
    test_imgs = all_imgs[:args.max_frames]

    print("\n" + "=" * 105)
    print(f"{'帧号 / 文件名':<35} | {'GT坐标 (原图)':<16} | {'GT处最大Score':<14} | {'全图最大Score (Max FP)':<22} | {'GT在全图排位':<12}")
    print("=" * 105)

    gt_scores = []
    bg_max_scores = []
    gt_ranks = []
    snr_ratios = []

    for img_p in test_imgs:
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue

        # 读取 GT 框 (class, cx, cy, w, h)
        gt_boxes = []
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    gt_boxes.append([float(x) for x in parts[1:5]])

        if not gt_boxes:
            continue

        # 读取三通道输入 (Channel 0: Gray, Channel 1: Diff1, Channel 2: Diff2)
        orig_img = cv2.imread(str(img_p))
        if orig_img is None:
            continue
        h0, w0 = orig_img.shape[:2]

        # 预处理 Letterbox
        img_resized, r, (dw, dh) = letterbox(orig_img, (imgsz, imgsz))
        img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(img_t)
            # heatmap: [1, 1, H_feat, W_feat]
            hm = preds["heatmap"][0, 0].cpu().numpy()

        feat_h, feat_w = hm.shape

        # 遍历所有 GT 计算响应
        for cx_norm, cy_norm, w_norm, h_norm in gt_boxes:
            gx_orig = cx_norm * w0
            gy_orig = cy_norm * h0

            # 映射到特征图坐标
            gx_feat = (gx_orig * r + dw) / stride
            gy_feat = (gy_orig * r + dh) / stride

            fx = int(round(gx_feat))
            fy = int(round(gy_feat))

            # 提取 GT 邻域局部窗口 (半径 r)
            rad = args.radius
            x1 = max(0, fx - rad)
            x2 = min(feat_w, fx + rad + 1)
            y1 = max(0, fy - rad)
            y2 = min(feat_h, fy + rad + 1)

            gt_patch = hm[y1:y2, x1:x2]
            gt_score = float(np.max(gt_patch)) if gt_patch.size > 0 else 0.0

            # 计算全图背景最大响应（排除 GT 邻域自身）
            hm_mask = np.ones_like(hm, dtype=bool)
            hm_mask[max(0, fy - 6):min(feat_h, fy + 7), max(0, fx - 6):min(feat_w, fx + 7)] = False
            bg_max = float(np.max(hm[hm_mask]))

            # 计算 GT 处得分在整幅热图所有像素（102400个像素点）中的百分比排名
            rank = int(np.sum(hm > gt_score)) + 1
            total_pixels = feat_h * feat_w

            gt_scores.append(gt_score)
            bg_max_scores.append(bg_max)
            gt_ranks.append(rank)
            snr_ratios.append(gt_score / (bg_max + 1e-6))

            print(
                f"{img_p.name:<35} | ({gx_orig:5.1f}, {gy_orig:5.1f})    | "
                f"{gt_score:12.5f} | "
                f"{bg_max:20.5f} | "
                f"Top {rank:5d} / {total_pixels}"
            )

    print("=" * 105)
    print("\n【统计定量诊断总结】")
    if gt_scores:
        avg_gt_score = np.mean(gt_scores)
        max_gt_score = np.max(gt_scores)
        min_gt_score = np.min(gt_scores)
        avg_bg_max = np.mean(bg_max_scores)
        avg_rank = np.mean(gt_ranks)
        avg_snr = np.mean(snr_ratios)

        print(f"1. GT 真实位置平均响应得分: {avg_gt_score:.5f}  (Min: {min_gt_score:.5f}, Max: {max_gt_score:.5f})")
        print(f"2. 背景最大伪响应平均得分: {avg_bg_max:.5f}")
        print(f"3. 目标 / 背景最大杂波峰值比 (Peak SNR): {avg_snr:.2f} 倍")
        print(f"4. GT 像素在全图 102,400 像素中的平均排位: 第 {avg_rank:.1f} 名")

        print("\n【物理结论判定】")
        if avg_gt_score < 0.02:
            print("❌ [诊断结果：彻底熄灭 (Dead Zone)]")
            print("   -> GT 处响应得分 < 0.02，甚至接近 0。说明输入端差分自抵消或前向特征严重平滑，目标在网络内部根本没有形成激活！")
            print("   -> 必须在前端引入时序特征局部相关层（Local Correlation Volume）或者增大差分时序跨度，后处理滤波在此类数据上无法救活！")
        elif avg_gt_score >= 0.02 and avg_gt_score < 0.20:
            if avg_snr >= 0.5:
                print("⚠️ [诊断结果：微弱弱脉冲激活，但被高阈值截断 (Sub-threshold Signal)]")
                print("   -> 目标确实激发了局部能量凸起（得分在 0.03~0.15 区间），但是被常规 0.20~0.30 全局门限硬生生切断！")
                print("   -> 此时若后接时序动态规划相干积累（TBD）或空域分区动态门限，有希望从噪声中成串打捞！")
            else:
                print("❌ [诊断结果：目标能量完全淹没在地面/杂波背景中]")
                print("   -> 目标有微弱激活，但背景杂波响应（地物边缘）远高于目标（SNR < 0.5），单纯降阈值会导致虚警雪崩。")
        else:
            print("✅ [诊断结果：正常强响应，无需特殊处理]")
    else:
        print("[WARN] 未能提取到有效的 GT 数据。")


if __name__ == "__main__":
    main()
