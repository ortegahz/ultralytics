#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Convert an Anti-UAV infrared image sequence to an MP4 video with optional GT annotations and zooming.

Usage:
    # 1. 基础用法：生成纯原图视频
    python manu/make_sequence_video.py --seq wg2022_ir_020_split_03

    # 2. 覆盖默认路径与帧率
    python manu/make_sequence_video.py \
        --data-root /mnt/data/siping/datasets/manu/anti-uav \
        --seq wg2022_ir_020_split_03 \
        --fps 25 \
        --output-dir runs/seq_videos

    # 3. 如果序列目录下包含 IR_label.json，则自动画出真实目标 GT 框
    python manu/make_sequence_video.py --seq wg2022_ir_020_split_03 --draw-gt
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
    """Find sequence directory under data_root (supporting train/val/test subdirectories or direct child)."""
    # 1. 直接检查根目录下是否存在
    direct_path = data_root / seq_name
    if direct_path.is_dir():
        return direct_path

    # 2. 递归查找子目录
    for path in data_root.rglob(seq_name):
        if path.is_dir():
            return path

    raise FileNotFoundError(f"未在 {data_root} 下找到序列目录: {seq_name}")


def load_gt_annotations(seq_dir: Path) -> dict[int, list[float]] | None:
    """
    Load ground truth bounding boxes if JSON / TXT annotation files exist in the sequence folder.
    Supports Anti-UAV IR_label.json format or frame-indexed list.
    """
    json_candidates = [
        seq_dir / "IR_label.json",
        seq_dir / "label.json",
        seq_dir / f"{seq_dir.name}.json",
    ]

    for json_path in json_candidates:
        if json_path.is_file():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Anti-UAV 常见格式: {"exist": [1, 1, ...], "gt_rect": [[x, y, w, h], ...]}
                if isinstance(data, dict) and "gt_rect" in data:
                    gt_dict = {}
                    gt_rects = data["gt_rect"]
                    exists = data.get("exist", [1] * len(gt_rects))
                    for idx, (rect, exist) in enumerate(zip(gt_rects, exists)):
                        if exist and rect and len(rect) == 4 and rect[2] > 0 and rect[3] > 0:
                            gt_dict[idx] = [float(v) for v in rect]
                    return gt_dict
            except Exception as e:
                print(f"[WARN] 解析标注文件 {json_path} 失败: {e}")

    return None


def main():
    parser = argparse.ArgumentParser(description="Convert an Anti-UAV sequence to MP4 video.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/anti-uav",
        help="Path to Anti-UAV root directory containing sequence folders",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default="wg2022_ir_020_split_03",
        help="Sequence folder name (e.g. wg2022_ir_020_split_03)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Video frame rate (default: 25.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/seq_videos",
        help="Directory to save the generated MP4 file",
    )
    parser.add_argument(
        "--draw-gt",
        action="store_true",
        help="Draw ground truth box if annotation JSON exists",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Resize scale factor (e.g. 1.0 or 2.0)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        # 如果 /mnt/ 不在本地，自动尝试回退检查 /media/ 本地目录
        alt_root = Path("/media/manu/1TB-Volume/data/anti-uav")
        if alt_root.exists():
            print(f"[INFO] 路径 {data_root} 不存在，自动切换为本地目录: {alt_root}")
            data_root = alt_root
        else:
            print(f"[ERROR] 找不到数据集根目录: {data_root}")
            sys.exit(1)

    # 1. 查找序列目录
    seq_dir = find_sequence_dir(data_root, args.seq)
    print(f"[INFO] 找到序列路径: {seq_dir}")

    # 2. 读取并排序所有图片
    image_paths = [
        p for p in seq_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    image_paths.sort(key=natural_sort_key)

    if not image_paths:
        print(f"[ERROR] 目录中没有找到图片: {seq_dir}")
        sys.exit(1)

    total_frames = len(image_paths)
    print(f"[INFO] 该序列共包含 {total_frames} 帧图像")

    # 3. 读取第一帧以获取宽高
    first_img = cv2.imread(str(image_paths[0]))
    if first_img is None:
        print(f"[ERROR] 无法读取图片: {image_paths[0]}")
        sys.exit(1)

    h, w = first_img.shape[:2]
    out_w, out_h = int(w * args.scale), int(h * args.scale)

    # 4. 加载标注信息（若开启 --draw-gt）
    gt_dict = None
    if args.draw_gt:
        gt_dict = load_gt_annotations(seq_dir)
        if gt_dict:
            print(f"[INFO] 成功加载 GT 标注，共覆盖 {len(gt_dict)} 帧")
        else:
            print(f"[INFO] 未在序列目录下找到匹配的标注文件，将生成纯图像视频")

    # 5. 初始化视频写入器
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{args.seq}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (out_w, out_h))

    if not writer.isOpened():
        print(f"[ERROR] 无法初始化 VideoWriter，目标路径: {video_path}")
        sys.exit(1)

    print(f"[INFO] 正在合成视频 -> {video_path} ({out_w}x{out_h} @ {args.fps}fps)...")

    for idx, img_path in enumerate(image_paths):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        # 绘制标注框
        if gt_dict and idx in gt_dict:
            gx, gy, gw, gh = gt_dict[idx]
            x1, y1 = int(round(gx)), int(round(gy))
            x2, y2 = int(round(gx + gw)), int(round(gy + gh))
            # 绿色框标出目标，并画红十字中心
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cx, cy = int(round(gx + gw / 2.0)), int(round(gy + gh / 2.0))
            cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 6, 1)

        # 添加 OSD 帧号水印
        cv2.putText(
            frame,
            f"{args.seq} | F:{idx:05d}/{total_frames:05d}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if args.scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        writer.write(frame)

    writer.release()
    print(f"[SUCCESS] 视频生成成功: {video_path.resolve()}")


if __name__ == "__main__":
    main()
