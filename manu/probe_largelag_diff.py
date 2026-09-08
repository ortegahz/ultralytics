#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Large-Lag Frame Difference Probe for Infrared Small UAV Detection.
Simulates multi-lag frame differences:
    [I_t, |I_t - I_{t-lag1}|, |I_t - I_{t-lag2}|]
directly from raw sequence images and feeds them into the champion model (trial_0031)
to test if separating temporal lag pulls the tiny target out of the dead zone.

Usage on server:
    # 默认对比测试 lag=(1, 2) [原基线] vs lag=(15, 30) [大跨度]
    python manu/probe_largelag_diff.py \
        --weights runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt \
        --seq wg2022_ir_020_split_03 \
        --lags 1,2 5,10 15,30 25,50 \
        --max-frames 50 \
        --device 0
"""

from __future__ import annotations

import argparse
import json
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

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


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


def load_gt_from_sequence(seq_dir: Path) -> dict[int, list[float]]:
    """Load frame-indexed GT annotations from sequence directory."""
    json_candidates = [
        seq_dir / "IR_label.json",
        seq_dir / "label.json",
        seq_dir / f"{seq_dir.name}.json",
    ]
    for json_p in json_candidates:
        if json_p.is_file():
            try:
                with open(json_p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "gt_rect" in data:
                    gt_rects = data["gt_rect"]
                    exists = data.get("exist", [1] * len(gt_rects))
                    gt_dict = {}
                    for idx, (rect, exist) in enumerate(zip(gt_rects, exists)):
                        if exist and rect and len(rect) == 4:
                            x, y, w, h = [float(v) for v in rect]
                            if 0 < w < 200 and 0 < h < 200:
                                gt_dict[idx] = [x, y, w, h]
                    return gt_dict
            except Exception as e:
                print(f"[WARN] Failed to parse {json_p}: {e}")
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description="Probe large lag temporal differences on heatmap response.")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt",
        help="Path to best.pt weights",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/anti-uav",
        help="Root directory containing raw sequences",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default="wg2022_ir_020_split_03",
        help="Sequence name",
    )
    parser.add_argument(
        "--lags",
        nargs="+",
        default=["1,2", "5,10", "15,30", "25,50"],
        help="List of lag pairs to test, e.g. 1,2 5,10 15,30 25,50",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    parser.add_argument("--stride", type=int, default=2, help="Heatmap stride")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or cpu")
    parser.add_argument("--radius", type=int, default=3, help="Search radius (in feature pixels) around GT")
    parser.add_argument("--max-frames", type=int, default=50, help="Number of frames to probe")
    parser.add_argument("--start-frame", type=int, default=0, help="Starting frame index (0 for auto-detect valid segment)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        cand = Path("/home/manu/mnt/pycharm_project_10ae9e2e") / args.weights
        if cand.exists():
            weights_path = cand

    print(f"[INFO] 加载模型: {weights_path} 到设备 {device}")
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    stride = ckpt.get("stride", args.stride)
    imgsz = ckpt.get("imgsz", args.imgsz)

    use_temporal = any("b0.motion_conv" in k for k in state_dict.keys())
    model = YOLO26HeatmapDetector(stride=stride, num_classes=1, use_temporal_stem=use_temporal)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 寻找原始序列目录
    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        candidates = [
            Path("/mnt/data/siping/datasets/manu/anti-uav"),
            Path("/home/manu/mnt/datasets/manu/anti-uav"),
            Path("/media/manu/1TB-Volume/data/anti-uav"),
        ]
        for c in candidates:
            if c.exists():
                raw_root = c
                break

    # 递归查找序列目录
    seq_dir = raw_root / args.seq
    if not seq_dir.is_dir():
        for p in raw_root.rglob(args.seq):
            if p.is_dir():
                seq_dir = p
                break

    if not seq_dir.is_dir():
        print(f"[ERROR] 找不到序列目录: {args.seq} in {raw_root}")
        sys.exit(1)

    print(f"[INFO] 序列路径: {seq_dir}")

    # 读取所有帧
    image_paths = sorted([p for p in seq_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES], key=natural_sort_key)
    total_imgs = len(image_paths)
    print(f"[INFO] 找到全序列图片: {total_imgs} 帧")

    # 加载标注
    gt_dict = load_gt_from_sequence(seq_dir)
    print(f"[INFO] 加载有效标注: {len(gt_dict)} 帧")

    # 自动搜寻具有充足前序历史帧且目标连续存在的目标区间
    valid_indices = sorted(list(gt_dict.keys()))
    if not valid_indices:
        print("[ERROR] 序列中没有任何有效 GT 目标！")
        sys.exit(1)

    # 寻找第一个 index >= 60 的连续段
    chosen_start = None
    for idx in valid_indices:
        if idx >= 60 and (idx + 10 in gt_dict):
            chosen_start = idx
            break

    if chosen_start is None:
        # 如果大于 60 的没有，则选取第一个至少有历史帧的目标
        for idx in valid_indices:
            if idx >= 30:
                chosen_start = idx
                break
        if chosen_start is None:
            chosen_start = valid_indices[0]

    eval_start = args.start_frame if args.start_frame > 0 else chosen_start
    eval_end = min(total_imgs, eval_start + args.max_frames)
    max_history = 60

    print(f"[INFO] 自动选定测试区间: 帧 {eval_start} ~ {eval_end - 1} (该区间内包含有效目标)")
    print(f"[INFO] 预加载灰度帧缓存 (帧 {max(0, eval_start - max_history)} ~ {eval_end - 1})...")
    gray_cache: dict[int, np.ndarray] = {}
    for idx in range(max(0, eval_start - max_history), eval_end):
        if 0 <= idx < total_imgs:
            im = cv2.imread(str(image_paths[idx]), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                gray_cache[idx] = im

    # 预热第一帧尺寸
    first_h, first_w = gray_cache[eval_start].shape[:2]

    # 解析测试的 lag 组合
    lag_groups: list[tuple[int, int]] = []
    for s in args.lags:
        p1, p2 = [int(v.strip()) for v in s.split(",")]
        lag_groups.append((p1, p2))

    print(f"\n[INFO] 开始测试不同时间跨度 (Lags) 组合: {lag_groups}")
    print("=" * 115)

    results_summary = []

    for lag1, lag2 in lag_groups:
        tag = f"Lag({lag1:2d}, {lag2:2d})"
        gt_scores = []
        bg_max_scores = []
        gt_ranks = []
        target_diff_mags = []

        for frame_idx in range(eval_start, eval_end):
            if frame_idx not in gt_dict:
                continue

            curr_gray = gray_cache.get(frame_idx)
            prev1_gray = gray_cache.get(frame_idx - lag1)
            prev2_gray = gray_cache.get(frame_idx - lag2)

            if curr_gray is None or prev1_gray is None or prev2_gray is None:
                continue

            # 构建 3 通道输入: [I_t, |I_t - I_{t-lag1}|, |I_t - I_{t-lag2}|]
            diff1 = cv2.absdiff(curr_gray, prev1_gray)
            diff2 = cv2.absdiff(curr_gray, prev2_gray)

            # 统计 GT 目标处的差分信号幅值
            gx, gy, gw, gh = gt_dict[frame_idx]
            cx, cy = int(round(gx + gw / 2.0)), int(round(gy + gh / 2.0))
            patch_d1 = diff1[max(0, cy - 2):min(first_h, cy + 3), max(0, cx - 2):min(first_w, cx + 3)]
            patch_d2 = diff2[max(0, cy - 2):min(first_h, cy + 3), max(0, cx - 2):min(first_w, cx + 3)]
            max_diff_val = max(float(np.max(patch_d1)) if patch_d1.size else 0.0,
                               float(np.max(patch_d2)) if patch_d2.size else 0.0)
            target_diff_mags.append(max_diff_val)

            input_3ch = np.stack([curr_gray, diff1, diff2], axis=-1)

            # Letterbox 预处理
            img_resized, r, (dw, dh) = letterbox(input_3ch, (imgsz, imgsz))
            img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_t = img_t.unsqueeze(0).to(device)

            with torch.no_grad():
                preds = model(img_t)
                hm = preds["heatmap"][0, 0].cpu().numpy()

            feat_h, feat_w = hm.shape

            # 映射 GT 坐标到特征图
            gx_feat = ((gx + gw / 2.0) * r + dw) / stride
            gy_feat = ((gy + gh / 2.0) * r + dh) / stride
            fx = int(round(gx_feat))
            fy = int(round(gy_feat))

            rad = args.radius
            x1 = max(0, fx - rad)
            x2 = min(feat_w, fx + rad + 1)
            y1 = max(0, fy - rad)
            y2 = min(feat_h, fy + rad + 1)

            gt_patch = hm[y1:y2, x1:x2]
            gt_score = float(np.max(gt_patch)) if gt_patch.size > 0 else 0.0

            # 背景最大响应（排除目标自身）
            hm_mask = np.ones_like(hm, dtype=bool)
            hm_mask[max(0, fy - 6):min(feat_h, fy + 7), max(0, fx - 6):min(feat_w, fx + 7)] = False
            bg_max = float(np.max(hm[hm_mask]))

            rank = int(np.sum(hm > gt_score)) + 1

            gt_scores.append(gt_score)
            bg_max_scores.append(bg_max)
            gt_ranks.append(rank)

        avg_score = float(np.mean(gt_scores)) if gt_scores else 0.0
        max_score = float(np.max(gt_scores)) if gt_scores else 0.0
        avg_bg = float(np.mean(bg_max_scores)) if bg_max_scores else 0.0
        avg_rank = float(np.mean(gt_ranks)) if gt_ranks else 0.0
        avg_diff = float(np.mean(target_diff_mags)) if target_diff_mags else 0.0
        snr = avg_score / (avg_bg + 1e-6)

        results_summary.append({
            "tag": tag,
            "avg_score": avg_score,
            "max_score": max_score,
            "avg_bg": avg_bg,
            "avg_rank": avg_rank,
            "avg_diff": avg_diff,
            "snr": snr,
        })

        print(
            f"{tag:<14} | 目标差分输入幅值: {avg_diff:4.1f}/255 | "
            f"GT平均Score: {avg_score:10.5f} (Max: {max_score:8.5f}) | "
            f"背景MaxFP: {avg_bg:8.5f} | "
            f"全图平均排位: Top {avg_rank:6.1f} / 102400"
        )

    print("=" * 115)
    print("\n【大跨度差分测试结论对比分析】")
    base_res = results_summary[0]
    print(f"• 基准短差分 {base_res['tag']}: 目标处差分幅值仅 {base_res['avg_diff']:.1f} 灰度级, 响应只有 {base_res['avg_score']:.5f}")

    best_res = max(results_summary, key=lambda x: x["avg_score"])
    print(f"• 最优差分跨度 {best_res['tag']}:")
    print(f"  - 目标差分幅值从 {base_res['avg_diff']:.1f} -> {best_res['avg_diff']:.1f} (物理位移彻底拉开)")
    print(f"  - GT 响应得分从 {base_res['avg_score']:.5f} -> {best_res['avg_score']:.5f} (提升 {best_res['avg_score'] / (base_res['avg_score'] + 1e-6):.1f} 倍)")
    print(f"  - 全图平均排名从 Top {base_res['avg_rank']:.0f} -> Top {best_res['avg_rank']:.0f}")

    if best_res["avg_score"] > 0.15:
        print("\n🎉 [结论：物理位移拉开直接激活了目标！]")
        print("   -> 证明网络权重完全具备识别该弱目标的能力，之前的彻底熄灭纯粹是短差分位移不足导致的物理自抵消！")
    elif best_res["avg_score"] > 0.03:
        print("\n⚡ [结论：目标响应有明显复苏，但因 trial_0031 在短差分上训练，大差分的正负波前未被最优解耦]")
        print("   -> 目标从彻底死区复苏至亚门限区间，进一步印证扩大时序跨度是正确突破口！")
    else:
        print("\n⚠️ [结论：即使差分幅值拉开，原模型响应依然低]")
        print("   -> 说明当前模型的卷积核在训练时强烈拟合了短差分的纹理分布，必须配合新时序方案重训。")


if __name__ == "__main__":
    main()
