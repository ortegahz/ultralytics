#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Temporal Coherence Energy Accumulator for Extreme Weak/Dark UAVs.

Physics Principle:
1. Random noise (shot noise, IR sensor FPN) is spatially uncorrelated across time.
2. A real UAV target follows a continuous physical trajectory across consecutive frames.
3. We compute:
   - Long-term Running Median Background subtraction (removes fixed patterns & slow clouds).
   - Multi-frame forward-backward persistence gating (suppresses random 1-frame spikes).
   - Motion Energy Pipeline Integration (accumulates energy strictly along moving directions).

Usage:
    python manu/visualize_temporal_accumulator.py \
        --data-root /mnt/data/siping/datasets/manu/anti-uav \
        --seq wg2022_ir_020_split_03 \
        --draw-gt
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
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.stem)]


def find_sequence_dir(data_root: Path, seq_name: str) -> Path:
    direct_path = data_root / seq_name
    if direct_path.is_dir():
        return direct_path
    for path in data_root.rglob(seq_name):
        if path.is_dir():
            return path
    raise FileNotFoundError(f"未在 {data_root} 下找到序列目录: {seq_name}")


def load_gt_annotations(seq_dir: Path) -> dict[int, list[float]] | None:
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


def draw_gt_on_image(img_bgr: np.ndarray, rect: list[float] | None):
    if rect is None:
        return
    gx, gy, gw, gh = rect
    x1, y1 = int(round(gx)), int(round(gy))
    x2, y2 = int(round(gx + gw)), int(round(gy + gh))
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cx, cy = int(round(gx + gw / 2.0)), int(round(gy + gh / 2.0))
    cv2.drawMarker(img_bgr, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 6, 1)


def add_label(img: np.ndarray, text: str):
    h, w = img.shape[:2]
    header = np.zeros((26, w, 3), dtype=np.uint8)
    cv2.putText(header, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, img])


def main():
    parser = argparse.ArgumentParser(description="Temporal Coherence Energy Accumulator")
    parser.add_argument("--data-root", type=str, default="/mnt/data/siping/datasets/manu/anti-uav")
    parser.add_argument("--seq", type=str, default="wg2022_ir_020_split_03")
    parser.add_argument("--window", type=int, default=7, help="Temporal coherence window (e.g. 5, 7, 9)")
    parser.add_argument("--decay", type=float, default=0.75, help="Decay factor for historical trajectory energy")
    parser.add_argument("--threshold", type=float, default=2.0, help="Minimum noise clipping threshold (gray levels)")
    parser.add_argument("--fps", type=float, default=25.0, help="Output video FPS")
    parser.add_argument("--output-dir", type=str, default="runs/scr_videos")
    parser.add_argument("--draw-gt", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        alt_root = Path("/media/manu/1TB-Volume/data/anti-uav")
        if alt_root.exists():
            data_root = alt_root
        else:
            print(f"[ERROR] 找不到数据集目录: {data_root}")
            sys.exit(1)

    seq_dir = find_sequence_dir(data_root, args.seq)
    print(f"[INFO] 找到序列路径: {seq_dir}")

    image_paths = [p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    image_paths.sort(key=natural_sort_key)
    total_frames = len(image_paths)

    gt_dict = load_gt_annotations(seq_dir) if args.draw_gt else None

    first_img = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
    H, W = first_img.shape[:2]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_video_path = out_dir / f"{args.seq}_temporal_coherent_energy.mp4"
    out_comp_path = out_dir / f"{args.seq}_temporal_coherent_compare.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_enh = cv2.VideoWriter(str(out_video_path), fourcc, args.fps, (W, H))
    writer_comp = cv2.VideoWriter(str(out_comp_path), fourcc, args.fps, (W * 2, H + 26))

    print(f"[INFO] 正在生成时空相干能量累加视频 -> {out_video_path.resolve()}")

    # 图像滑动窗口
    frame_buffer: list[np.ndarray] = []
    # 累加的运动能量场
    energy_map = np.zeros((H, W), dtype=np.float32)

    # 形态学核，用于时序空间关联膨胀（容忍目标 1~2 像素的运动位移）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for idx, p in enumerate(image_paths):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        frame_buffer.append(gray)

        if len(frame_buffer) > args.window:
            frame_buffer.pop(0)

        curr = frame_buffer[-1]

        if len(frame_buffer) < 3:
            # 初始帧预热
            diff_motion = np.zeros_like(curr)
        else:
            # 1. 计算当前帧与历史多帧的绝对差分
            diffs = [np.abs(curr - prev) for prev in frame_buffer[:-1]]
            
            # 2. 噪声硬阈值门限：凡是小于 threshold（如 2.0 灰度级）的微小抖动全部抹为 0
            diffs_clean = [np.where(d > args.threshold, d, 0.0) for d in diffs]

            # 3. 时序一致性检验 (Temporal Persistence)：
            # 随机噪点在历史帧差中只会出现 1 次；真实运动目标在连续差分中会持续触发
            active_count = np.sum([d > 0.0 for d in diffs_clean], axis=0)
            
            # 至少在 2 帧以上有运动响应，否则认定为孤立白噪点并置零
            coherent_mask = (active_count >= 2).astype(np.float32)
            
            # 加权瞬时有效运动分量
            instant_motion = diffs_clean[-1] * coherent_mask

            # 4. 空间邻域能量传递与累加：
            # 上一帧的能量在空间做微小膨胀（容许位移），并乘上时间衰减率 decay
            energy_propagated = cv2.dilate(energy_map, kernel) * args.decay
            
            # 当前帧运动能量累积
            energy_map = np.maximum(instant_motion, energy_propagated * 0.85 + instant_motion * 0.5)

        # 归一化显示
        disp_energy = np.clip(energy_map * 15.0, 0, 255).astype(np.uint8)
        color_energy = cv2.applyColorMap(disp_energy, cv2.COLORMAP_MAGMA)

        gt_box = gt_dict.get(idx) if gt_dict else None

        # 增强单独视频
        enh_frame = color_energy.copy()
        if args.draw_gt:
            draw_gt_on_image(enh_frame, gt_box)
        cv2.putText(
            enh_frame,
            f"Coherent Motion Energy | F:{idx:05d}/{total_frames:05d}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer_enh.write(enh_frame)

        # 左右对比视频
        raw_panel = bgr.copy()
        if args.draw_gt:
            draw_gt_on_image(raw_panel, gt_box)
        p_left = add_label(raw_panel, f"Raw IR Image (Frame {idx:05d}) [Green: GT]")
        p_right = add_label(enh_frame, "Temporal Coherence Filtered Energy (Noise Purged)")

        writer_comp.write(np.hstack([p_left, p_right]))

    writer_enh.release()
    writer_comp.release()

    print(f"\n[SUCCESS] 生成完毕！")
    print(f"1. 独立时空能量视频: {out_video_path.resolve()}")
    print(f"2. 左右对比视频: {out_comp_path.resolve()}")


if __name__ == "__main__":
    main()
