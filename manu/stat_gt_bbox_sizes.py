#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Inspect Ground-Truth Bounding Box Dimensions for Specific Problematic Sequences.
Computes original image dimensions, w, h, area, diagonal, and 640-scale resized dimensions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.data.utils import check_det_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze GT Bbox sizes for specific sequences")
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default="wg2022_ir_020_split_03,DJI_0051_2,02_6321_0274-2773,DJI_0175_2",
        help="Comma-separated sequence name substrings to inspect",
    )
    return parser.parse_args()


def find_label_file(img_path: Path) -> Path | None:
    """Find label .txt corresponding to img_path."""
    s = str(img_path)
    candidates = []
    if "/images/" in s:
        candidates.append(Path(s.replace("/images/", "/labels/")).with_suffix(".txt"))
    candidates.append(img_path.with_suffix(".txt"))

    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    args = parse_args()
    target_seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]

    data_dict = check_det_dataset(args.data)
    val_source = data_dict["val"]

    # 收集验证集图像
    val_dirs = [Path(val_source)] if isinstance(val_source, (str, Path)) else [Path(p) for p in val_source]
    img_paths = []
    for d in val_dirs:
        if d.is_file():
            with open(d, "r", encoding="utf-8") as f:
                for line in f:
                    p = Path(line.strip())
                    if p.exists():
                        img_paths.append(p)
        elif d.is_dir():
            for p in d.rglob("*.*"):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    img_paths.append(p)

    print(f"Total validation images found: {len(img_paths)}")

    # 针对每个指定的序列进行统计
    for target in target_seqs:
        matched_imgs = [p for p in img_paths if target in p.name or target in str(p.parent)]
        matched_imgs.sort()

        if not matched_imgs:
            print(f"\n[WARN] No images found for sequence keyword: '{target}'")
            continue

        orig_w_list = []
        orig_h_list = []
        orig_diag_list = []
        orig_area_list = []

        scale640_w_list = []
        scale640_h_list = []
        scale640_area_list = []

        img_wh_sample = None

        for p in matched_imgs:
            lbl_p = find_label_file(p)
            if not lbl_p or not lbl_p.exists():
                continue

            # 读取图像实际宽高（只读头部或第一张即可获知序列宽高）
            if img_wh_sample is None:
                im = cv2.imread(str(p))
                if im is not None:
                    img_wh_sample = (im.shape[1], im.shape[0])  # (W, H)

            if img_wh_sample is None:
                continue

            W_orig, H_orig = img_wh_sample

            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # cls, cx, cy, bw_norm, bh_norm
                        bw_norm = float(parts[3])
                        bh_norm = float(parts[4])

                        # 原图实际像素尺寸
                        bw_px = bw_norm * W_orig
                        bh_px = bh_norm * H_orig
                        diag_px = np.sqrt(bw_px**2 + bh_px**2)
                        area_px = bw_px * bh_px

                        orig_w_list.append(bw_px)
                        orig_h_list.append(bh_px)
                        orig_diag_list.append(diag_px)
                        orig_area_list.append(area_px)

                        # 在 640x640 输入下的像素尺寸
                        s640_w = bw_norm * 640.0
                        s640_h = bh_norm * 640.0
                        scale640_w_list.append(s640_w)
                        scale640_h_list.append(s640_h)
                        scale640_area_list.append(s640_w * s640_h)

        print("\n" + "=" * 80)
        print(f"SEQUENCE: {target} (Frames found: {len(matched_imgs)}, GT Boxes: {len(orig_w_list)})")
        if img_wh_sample:
            print(f"Original Resolution: {img_wh_sample[0]} x {img_wh_sample[1]}")
        print("=" * 80)

        if not orig_w_list:
            print("  No GT boxes found in labels.")
            continue

        ow = np.array(orig_w_list)
        oh = np.array(orig_h_list)
        odiag = np.array(orig_diag_list)
        oarea = np.array(orig_area_list)

        s640w = np.array(scale640_w_list)
        s640h = np.array(scale640_h_list)
        s640area = np.array(scale640_area_list)

        print("【原图物理尺寸 (Original Resolution Pixels)】:")
        print(f"  Width (宽)    : Min = {ow.min():.1f}px, Median = {np.median(ow):.1f}px, Mean = {ow.mean():.1f}px, Max = {ow.max():.1f}px")
        print(f"  Height (高)   : Min = {oh.min():.1f}px, Median = {np.median(oh):.1f}px, Mean = {oh.mean():.1f}px, Max = {oh.max():.1f}px")
        print(f"  Diagonal (对角线): Median = {np.median(odiag):.1f}px, Mean = {np.mean(odiag):.1f}px")
        print(f"  Area (像素面积): Median = {np.median(oarea):.1f} px², Mean = {np.mean(oarea):.1f} px²")

        print("\n【模型输入 640x640 下的等比尺寸 (Resized to 640)】:")
        print(f"  Width (宽)    : Min = {s640w.min():.2f}px, Median = {np.median(s640w):.2f}px, Mean = {s640w.mean():.2f}px, Max = {s640w.max():.2f}px")
        print(f"  Height (高)   : Min = {s640h.min():.2f}px, Median = {np.median(s640h):.2f}px, Mean = {s640h.mean():.2f}px, Max = {s640h.max():.2f}px")
        print(f"  Area (像素面积): Median = {np.median(s640area):.2f} px², Mean = {np.mean(s640area):.2f} px²")

        # 统计极小目标占比
        sub_4px_count = np.sum((s640w <= 4.0) | (s640h <= 4.0))
        sub_2px_count = np.sum((s640w <= 2.0) | (s640h <= 2.0))
        print(f"\n【极小目标占比统计 (在 640 尺度下)】:")
        print(f"  单边 <= 4.0px 的目标数: {sub_4px_count} / {len(s640w)} ({sub_4px_count / len(s640w) * 100:.1f}%)")
        print(f"  单边 <= 2.0px 的目标数: {sub_2px_count} / {len(s640w)} ({sub_2px_count / len(s640w) * 100:.1f}%)")
        print("-" * 80)


if __name__ == "__main__":
    main()
