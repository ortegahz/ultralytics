#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Generate SCR (Signal-to-Clutter Ratio) Normalized Temporal Difference Videos.

Features:
1. Fast local background variance calculation using O(1) box filtering.
2. Computes adaptive SCR motion saliency:
       Delta_I = |I_t - I_{t - lag}|
       sigma = sqrt(max(E[I^2] - E[I]^2, 0))
       SCR = Delta_I / (sigma + eps)
3. Generates:
   - Enhanced single-view video (Grayscale or Jet/Magma pseudo-color)
   - 4-panel side-by-side comparison video:
     [1. Raw Frame + GT] | [2. Standard Frame Diff] | [3. Local Clutter Std (Sigma)] | [4. SCR Normalized Output]

Usage:
    # 1. 基础并排对比视频 (默认 lag=3, 包含4宫格对比与GT标记)
    python manu/visualize_scr_normalization.py \
        --data-root /mnt/data/siping/datasets/manu/anti-uav \
        --seq wg2022_ir_020_split_03 \
        --draw-gt

    # 2. 调整时序步长与局部滤波窗口
    python manu/visualize_scr_normalization.py \
        --seq wg2022_ir_020_split_03 \
        --lag 4 \
        --ksize 15 \
        --draw-gt \
        --colormap magma
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_sort_key(path: Path):
    """Sort filenames naturally (e.g., 000001.jpg, 000002.jpg)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.stem)]


def find_sequence_dir(data_root: Path, seq_name: str) -> Path:
    """Find sequence directory under data_root."""
    direct_path = data_root / seq_name
    if direct_path.is_dir():
        return direct_path

    for path in data_root.rglob(seq_name):
        if path.is_dir():
            return path

    raise FileNotFoundError(f"未在 {data_root} 下找到序列目录: {seq_name}")


def load_gt_annotations(seq_dir: Path) -> dict[int, list[float]] | None:
    """Load ground truth bounding boxes if JSON exists."""
    for json_name in ["IR_label.json", "label.json", f"{seq_dir.name}.json"]:
        json_path = seq_dir / json_name
        if json_path.is_file():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "gt_rect" in data:
                    gt_dict = {}
                    gt_rects = data["gt_rect"]
                    exists = data.get("exist", [1] * len(gt_rects))
                    for idx, (rect, exist) in enumerate(zip(gt_rects, exists)):
                        if exist and rect and len(rect) == 4 and rect[2] > 0 and rect[3] > 0:
                            gt_dict[idx] = [float(v) for v in rect]
                    return gt_dict
            except Exception as e:
                print(f"[WARN] 读取标注失败 {json_path}: {e}")
    return None


def compute_local_std(img_gray: np.ndarray, ksize: int = 15) -> np.ndarray:
    """
    Compute local standard deviation sigma efficiently using box filter:
    sigma = sqrt(max(E[X^2] - (E[X])^2, 0))
    """
    img_f = img_gray.astype(np.float32)
    mean = cv2.boxFilter(img_f, ddepth=-1, ksize=(ksize, ksize), borderType=cv2.BORDER_REFLECT)
    sqr_mean = cv2.boxFilter(img_f * img_f, ddepth=-1, ksize=(ksize, ksize), borderType=cv2.BORDER_REFLECT)
    variance = np.maximum(sqr_mean - mean * mean, 0.0)
    return np.sqrt(variance)


def compute_scr_map(
    curr_gray: np.ndarray,
    prev_gray: np.ndarray,
    ksize: int = 15,
    eps: float = 1.5,
    clip_max: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute raw diff, local background standard deviation, and SCR normalized response.
    
    Returns:
        diff_u8: Standard absolute frame difference in [0, 255]
        sigma_u8: Local clutter standard deviation visualization in [0, 255]
        scr_u8: SCR normalized response scaled to [0, 255]
    """
    # 1. 绝对差分
    diff = np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32))

    # 2. 局部背景标准差 (Clutter Sigma)
    sigma = compute_local_std(curr_gray, ksize=ksize)

    # 3. 局部信噪比自适应拉伸
    # 平坦背景下 sigma 较小，微小运动差值被强行放大；复杂杂波区 sigma 较大，边缘差值被压制
    scr = diff / (sigma + eps)

    # 归一化映射到 [0, 255]
    diff_u8 = np.clip(diff * 2.0, 0, 255).astype(np.uint8)  # 乘以2方便人眼观察微弱差分
    sigma_u8 = np.clip(sigma * 4.0, 0, 255).astype(np.uint8)
    scr_norm = np.clip(scr / clip_max, 0.0, 1.0)
    scr_u8 = (scr_norm * 255.0).astype(np.uint8)

    return diff_u8, sigma_u8, scr_u8


def draw_gt_on_image(img_bgr: np.ndarray, rect: list[float] | None):
    """Draw green bounding box and red cross marker for GT target."""
    if rect is None:
        return
    gx, gy, gw, gh = rect
    x1, y1 = int(round(gx)), int(round(gy))
    x2, y2 = int(round(gx + gw)), int(round(gy + gh))
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cx, cy = int(round(gx + gw / 2.0)), int(round(gy + gh / 2.0))
    cv2.drawMarker(img_bgr, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 6, 1)


def add_label(img: np.ndarray, text: str, bg_color=(0, 0, 0)):
    """Add a header title banner on top of the image."""
    h, w = img.shape[:2]
    header = np.zeros((26, w, 3), dtype=np.uint8)
    if bg_color != (0, 0, 0):
        header[:] = bg_color
    cv2.putText(header, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, img])


def main():
    parser = argparse.ArgumentParser(description="Visualize SCR Normalized Frame Difference")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/anti-uav",
        help="Path to Anti-UAV root directory",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default="wg2022_ir_020_split_03",
        help="Sequence folder name",
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=3,
        help="Frame difference lag step (e.g. 2, 3, 5)",
    )
    parser.add_argument(
        "--ksize",
        type=int,
        default=15,
        help="Local window size for background variance estimation (odd integer)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1.5,
        help="Epsilon to prevent noise amplification in completely static pixels",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=6.0,
        help="Maximum SCR value mapped to 255",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Output video FPS",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/scr_videos",
        help="Directory to save output videos",
    )
    parser.add_argument(
        "--draw-gt",
        action="store_true",
        help="Draw ground truth target markers",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="magma",
        choices=["magma", "jet", "gray"],
        help="Color map for SCR normalized response: magma, jet, or gray",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        alt_root = Path("/media/manu/1TB-Volume/data/anti-uav")
        if alt_root.exists():
            print(f"[INFO] 路径 {data_root} 不存在，自动切换为本地目录: {alt_root}")
            data_root = alt_root
        else:
            print(f"[ERROR] 找不到数据集目录: {data_root}")
            sys.exit(1)

    seq_dir = find_sequence_dir(data_root, args.seq)
    print(f"[INFO] 找到序列路径: {seq_dir}")

    image_paths = [p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    image_paths.sort(key=natural_sort_key)
    total_frames = len(image_paths)

    if total_frames <= args.lag:
        print(f"[ERROR] 序列帧数 ({total_frames}) 小于等于 lag 跨度 ({args.lag})")
        sys.exit(1)

    gt_dict = load_gt_annotations(seq_dir) if args.draw_gt else None
    if args.draw_gt and gt_dict:
        print(f"[INFO] 成功加载 GT 标注，共覆盖 {len(gt_dict)} 帧")

    first_img = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
    H, W = first_img.shape[:2]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 对比视频路径 (2x2 网格)
    comp_path = out_dir / f"{args.seq}_scr_lag{args.lag}_compare.mp4"
    # 2. 单独增强视频路径
    enh_path = out_dir / f"{args.seq}_scr_lag{args.lag}_enhanced.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # 2x2 网格分辨率: (2 * W, 2 * (H + 26))
    writer_comp = cv2.VideoWriter(str(comp_path), fourcc, args.fps, (W * 2, (H + 26) * 2))
    writer_enh = cv2.VideoWriter(str(enh_path), fourcc, args.fps, (W, H))

    print(f"[INFO] 正在生成增强视频: {enh_path}")
    print(f"[INFO] 正在生成并排对比视频: {comp_path}")

    # 维护滑动窗口灰度图
    gray_buffer: list[np.ndarray] = []

    for idx, p in enumerate(image_paths):
        curr_bgr = cv2.imread(str(p))
        if curr_bgr is None:
            continue
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
        gray_buffer.append(curr_gray)

        if len(gray_buffer) <= args.lag:
            # 初始帧不足 lag 时，差分用自身（无响应）
            prev_gray = curr_gray
        else:
            prev_gray = gray_buffer[-1 - args.lag]
            # 释放过旧缓存保持内存精简
            if len(gray_buffer) > args.lag + 5:
                gray_buffer.pop(0)

        # 计算 SCR 归一化
        diff_u8, sigma_u8, scr_u8 = compute_scr_map(
            curr_gray,
            prev_gray,
            ksize=args.ksize,
            eps=args.eps,
            clip_max=args.clip_max,
        )

        # 伪彩色映射
        if args.colormap == "magma":
            scr_color = cv2.applyColorMap(scr_u8, cv2.COLORMAP_MAGMA)
        elif args.colormap == "jet":
            scr_color = cv2.applyColorMap(scr_u8, cv2.COLORMAP_JET)
        else:
            scr_color = cv2.cvtColor(scr_u8, cv2.COLOR_GRAY2BGR)

        diff_bgr = cv2.cvtColor(diff_u8, cv2.COLOR_GRAY2BGR)
        sigma_bgr = cv2.applyColorMap(sigma_u8, cv2.COLORMAP_BONE)

        gt_box = gt_dict.get(idx) if gt_dict else None

        # 写入增强单视频 (在增强结果上标出真实 GT 框)
        enh_frame = scr_color.copy()
        if args.draw_gt:
            draw_gt_on_image(enh_frame, gt_box)
        cv2.putText(
            enh_frame,
            f"SCR-Norm (Lag={args.lag}) | F:{idx:05d}/{total_frames:05d}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer_enh.write(enh_frame)

        # 构建 4 宫格对比画面
        # 1. 原图 + GT
        raw_panel = curr_bgr.copy()
        if args.draw_gt:
            draw_gt_on_image(raw_panel, gt_box)
        p1 = add_label(raw_panel, f"1. Raw IR Frame (F:{idx:05d}) [Green Box: GT]")

        # 2. 传统帧差
        p2 = add_label(diff_bgr, f"2. Standard Diff |I(t) - I(t-{args.lag})| (Noise & Clutter Dominated)")

        # 3. 局部背景杂波度 Sigma
        p3 = add_label(sigma_bgr, f"3. Local Clutter Std Dev (Sigma ksize={args.ksize})")

        # 4. SCR 归一化输出
        scr_panel = scr_color.copy()
        if args.draw_gt:
            draw_gt_on_image(scr_panel, gt_box)
        p4 = add_label(scr_panel, f"4. SCR Normalized = Diff / (Sigma + {args.eps}) [Target Highlighted]")

        row_top = np.hstack([p1, p2])
        row_bot = np.hstack([p3, p4])
        grid = np.vstack([row_top, row_bot])

        writer_comp.write(grid)

    writer_enh.release()
    writer_comp.release()

    print(f"\n[SUCCESS] 生成完毕！")
    print(f"1. 独立增强视频: {enh_path.resolve()}")
    print(f"2. 四宫格对比视频: {comp_path.resolve()}")


if __name__ == "__main__":
    main()
