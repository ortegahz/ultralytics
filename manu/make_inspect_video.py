#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Generate a diagnostic video for Anti-UAV infrared sequences.
Splits the view into:
1. Left: Full original infrared frame with optional non-occluding bracket markers and trajectory.
2. Right: High-resolution zoomed-in crop centered on the ground-truth target.
   - Clean raw zoom (no pixels altered on the actual target)
   - Contrast enhanced (CLAHE) zoom
   - Crosshair / Reticle in margin pointing to center without covering the target pixel.
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
    """Sort filenames naturally (e.g. 000001.jpg, 000002.jpg)."""
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


def load_gt_annotations(seq_dir: Path) -> dict[int, list[float]]:
    """
    Load ground truth bounding boxes from IR_label.json or label.json.
    Filters out full-frame / absent markers.
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

                if isinstance(data, dict) and "gt_rect" in data:
                    gt_dict = {}
                    gt_rects = data["gt_rect"]
                    exists = data.get("exist", [1] * len(gt_rects))
                    for idx, (rect, exist) in enumerate(zip(gt_rects, exists)):
                        # 过滤无效框以及占满整幅画面的伪标注
                        if exist and rect and len(rect) == 4:
                            x, y, w, h = [float(v) for v in rect]
                            if w > 0 and h > 0 and w < 200 and h < 200:
                                gt_dict[idx] = [x, y, w, h]
                    return gt_dict
            except Exception as e:
                print(f"[WARN] 解析标注文件 {json_path} 失败: {e}")

    return {}


def draw_corner_brackets(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color=(0, 255, 0), margin: int = 10, arm_len: int = 6):
    """
    在外围画四个直角括号标记，完全不接触和遮挡目标本体。
    margin: 目标外扩的留白距离（确保目标周围背景也清晰可见）
    arm_len: 角标直角臂的长度
    """
    bx1 = max(0, x1 - margin)
    by1 = max(0, y1 - margin)
    bx2 = min(img.shape[1] - 1, x2 + margin)
    by2 = min(img.shape[0] - 1, y2 + margin)

    # 左上角
    cv2.line(img, (bx1, by1), (bx1 + arm_len, by1), color, 1)
    cv2.line(img, (bx1, by1), (bx1, by1 + arm_len), color, 1)
    # 右上角
    cv2.line(img, (bx2, by1), (bx2 - arm_len, by1), color, 1)
    cv2.line(img, (bx2, by1), (bx2, by1 + arm_len), color, 1)
    # 左下角
    cv2.line(img, (bx1, by2), (bx1 + arm_len, by2), color, 1)
    cv2.line(img, (bx1, by2), (bx1, by2 - arm_len), color, 1)
    # 右下角
    cv2.line(img, (bx2, by2), (bx2 - arm_len, by2), color, 1)
    cv2.line(img, (bx2, by2), (bx2, by2 - arm_len), color, 1)


def create_zoom_panel(
    raw_frame: np.ndarray,
    center_xy: tuple[float, float],
    crop_size: int = 48,
    target_wh: tuple[int, int] = (512, 512),
    draw_pointers: bool = True,
) -> np.ndarray:
    """
    提取以 center_xy 为中心的局部放大面板。
    panel 上半部分：Raw Pixel 原汁原味双三次插值放大（不做任何对比度改变，带外围指示准星）
    panel 下半部分：CLAHE 自适应直方图均衡增强放大（便于暗弱特征裸眼确认）
    """
    H, W = raw_frame.shape[:2]
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    half = crop_size // 2

    # 提取安全 patch
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(W, cx + half)
    y2 = min(H, cy + half)

    patch = raw_frame[y1:y2, x1:x2]
    # 若边缘尺寸不够进行补齐
    if patch.shape[0] != crop_size or patch.shape[1] != crop_size:
        pad_top = y1 - (cy - half)
        pad_bottom = (cy + half) - y2
        pad_left = x1 - (cx - half)
        pad_right = (cx + half) - x2
        patch = cv2.copyMakeBorder(patch, max(0, pad_top), max(0, pad_bottom), max(0, pad_left), max(0, pad_right), cv2.BORDER_REFLECT)
        patch = patch[:crop_size, :crop_size]

    panel_w, panel_h = target_wh
    half_h = panel_h // 2

    # 1. 上半部：纯原始无损放大（Raw Zoom）
    raw_zoom = cv2.resize(patch, (panel_w, half_h), interpolation=cv2.INTER_NEAREST)

    # 2. 下半部：自适应直方图增强放大（Enhanced Zoom）
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_patch = clahe.apply(gray_patch)
    enhanced_zoom = cv2.resize(enhanced_patch, (panel_w, half_h), interpolation=cv2.INTER_NEAREST)
    if len(raw_zoom.shape) == 3 and len(enhanced_zoom.shape) == 2:
        enhanced_zoom = cv2.cvtColor(enhanced_zoom, cv2.COLOR_GRAY2BGR)

    # 在两部分的外围边框绘制引导准星（边缘线段指向中心，但中心目标区域完全留白镂空！）
    def draw_margin_guides(img_panel: np.ndarray, title: str):
        ph, pw = img_panel.shape[:2]
        pcx, pcy = pw // 2, ph // 2
        # 镂空半径：中心 24 像素内绝对不画任何线条！
        inner_gap = 24
        pointer_color = (0, 255, 255)  # 黄色高亮边线

        # 上、下、左、右指示短线
        cv2.line(img_panel, (pcx, 2), (pcx, pcy - inner_gap), pointer_color, 1)
        cv2.line(img_panel, (pcx, ph - 2), (pcx, pcy + inner_gap), pointer_color, 1)
        cv2.line(img_panel, (2, pcy), (pcx - inner_gap, pcy), pointer_color, 1)
        cv2.line(img_panel, (pw - 2, pcy), (pcx + inner_gap, pcy), pointer_color, 1)

        # 标题标签
        cv2.rectangle(img_panel, (0, 0), (pw, 22), (30, 30, 30), -1)
        cv2.putText(img_panel, title, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    if draw_pointers:
        draw_margin_guides(raw_zoom, f"RAW ZOOM (Crop {crop_size}x{crop_size} | Nearest)")
        draw_margin_guides(enhanced_zoom, "CLAHE ENHANCED ZOOM (Local Contrast Boost)")

    # 上下拼接为完整的放大面板
    combined_zoom = np.vstack([raw_zoom, enhanced_zoom])
    # 中间加一条分割线
    cv2.line(combined_zoom, (0, half_h), (panel_w, half_h), (80, 80, 80), 1)
    return combined_zoom


def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic video with raw sequence and target zoomed view.")
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
        help="Sequence folder name",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Video frame rate (default: 25.0)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=40,
        help="Crop patch size around target (default: 40 pixels)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/seq_videos_inspect",
        help="Directory to save the generated MP4 file",
    )
    parser.add_argument(
        "--hide-brackets",
        action="store_true",
        help="If set, do not draw corner brackets on the left full overview frame",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames to process (0 for all)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        # 尝试自动切换常见备选目录（兼容本地测试、本地SSHFS挂载、服务器原生路径）
        candidates = [
            Path("/mnt/data/siping/datasets/manu/anti-uav"),
            Path("/mnt/data/siping/datasets/anti-uav"),
            Path("/home/manu/mnt/datasets/manu/anti-uav"),
            Path("/home/manu/mnt/datasets/anti-uav"),
            Path("/media/manu/1TB-Volume/data/anti-uav"),
        ]
        for c in candidates:
            if c.exists():
                data_root = c
                break

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
    if args.max_frames > 0:
        image_paths = image_paths[:args.max_frames]
        total_frames = len(image_paths)

    print(f"[INFO] 该序列共包含 {total_frames} 帧图像")

    # 3. 加载 GT 标注
    gt_dict = load_gt_annotations(seq_dir)
    print(f"[INFO] 成功加载有效 GT 标注: {len(gt_dict)} 帧")

    # 4. 获取第一帧图像信息
    first_img = cv2.imread(str(image_paths[0]))
    if first_img is None:
        print(f"[ERROR] 无法读取图片: {image_paths[0]}")
        sys.exit(1)

    h, w = first_img.shape[:2]
    # 右侧放大面板尺寸匹配左侧主图高度
    zoom_w = h
    out_w = w + zoom_w
    out_h = h

    # 5. 初始化视频写入
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{args.seq}_inspect_zoom.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"[ERROR] 无法创建视频文件: {video_path}")
        sys.exit(1)

    print(f"[INFO] 正在生成画中画/并排放大视频 -> {video_path} ({out_w}x{out_h} @ {args.fps}fps)...")

    # 轨迹记录（前 30 帧中心历史）
    history_centers: list[tuple[int, int]] = []
    last_known_center = (w // 2, h // 2)

    for idx, img_path in enumerate(image_paths):
        raw_frame = cv2.imread(str(img_path))
        if raw_frame is None:
            continue

        annotated_frame = raw_frame.copy()

        # 检查当前帧是否有 GT
        has_gt = idx in gt_dict
        if has_gt:
            gx, gy, gw, gh = gt_dict[idx]
            cx, cy = gx + gw / 2.0, gy + gh / 2.0
            last_known_center = (cx, cy)
            history_centers.append((int(round(cx)), int(round(cy))))
            if len(history_centers) > 40:
                history_centers.pop(0)

            # 在左侧主图绘制外围角标（不碰触任何目标内部像素）
            if not args.hide_brackets:
                x1, y1 = int(round(gx)), int(round(gy))
                x2, y2 = int(round(gx + gw)), int(round(gy + gh))
                draw_corner_brackets(annotated_frame, x1, y1, x2, y2, color=(0, 255, 0), margin=8, arm_len=6)

                # 绘制历史尾迹（浅绿色细线，展示运动学相干连续性）
                for i in range(1, len(history_centers)):
                    alpha = i / len(history_centers)
                    color = (0, int(180 * alpha), 0)
                    cv2.line(annotated_frame, history_centers[i - 1], history_centers[i], color, 1)

        # 生成右侧放大面板
        zoom_panel = create_zoom_panel(
            raw_frame=raw_frame,
            center_xy=last_known_center,
            crop_size=args.crop_size,
            target_wh=(zoom_w, h),
            draw_pointers=has_gt,
        )

        # 左图叠加极简非侵入 OSD
        target_info = f"GT: ({last_known_center[0]:.1f}, {last_known_center[1]:.1f})" if has_gt else "GT: LOST/NONE"
        status_color = (0, 255, 0) if has_gt else (100, 100, 100)
        cv2.putText(
            annotated_frame,
            f"{args.seq} | Frame: {idx:05d}/{total_frames:05d} | {target_info}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_color,
            1,
            cv2.LINE_AA,
        )

        # 拼接左右视图：[完整主视图 | 局部放大视图]
        combined = np.hstack([annotated_frame, zoom_panel])
        # 左右分界线
        cv2.line(combined, (w, 0), (w, h), (100, 100, 100), 1)

        writer.write(combined)

    writer.release()
    print(f"\n[SUCCESS] 视频合成完成: {video_path.resolve()}")


if __name__ == "__main__":
    main()
