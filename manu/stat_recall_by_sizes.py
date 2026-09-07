#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Fine-grained Recall Analysis Stratified by Target Pixel Size.

Features:
1. Excludes the specified Top-N worst/extreme hardcase sequences (or custom list).
2. Uses the cached inference predictions (runs/badcase_analysis/inference_cache.pkl) for instant computation.
3. Maps each Ground-Truth target to its real bounding box size in the original image.
4. Stratifies Recall / Precision / F1 into representative tiny/small target size buckets:
   - Ultra-tiny Point (<= 3x3 px, Area <= 9 px²)
   - Very Small (4x4 ~ 6x6 px, Area 10 ~ 36 px²)
   - Small Point (7x7 ~ 10x10 px, Area 37 ~ 100 px²)
   - Medium Small (11x11 ~ 20x20 px, Area 101 ~ 400 px²)
   - Typical UAV (21x21 ~ 40x40 px, Area 401 ~ 1600 px²)
   - Large / Near (> 40x40 px, Area > 1600 px²)
5. Computes and prints clean formatted ASCII summary tables, both in Original Resolution and 640 Resized Scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.data.utils import check_det_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Size-stratified Recall Analysis excluding extreme hardcases")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/badcase_analysis/inference_cache.pkl",
        help="Path to inference_cache.pkl",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav/data.yaml",
        help="Path to data.yaml to resolve label paths",
    )
    parser.add_argument(
        "--exclude-seqs",
        type=str,
        default="wg2022_ir_020_split_03,DJI_0051_2,02_6321_0274-2773,DJI_0175_2,wg2022_ir_011_split_03",
        help="Comma-separated sequence names or substrings to exclude (the top extreme hard cases)",
    )
    parser.add_argument(
        "--dist-thresh",
        type=float,
        default=8.0,
        help="Distance tolerance in pixels (default: 8.0px)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Confidence threshold for predictions (default: 0.20)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (default: 640)",
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


def build_image_and_label_lookup(val_source: str | Path | list) -> tuple[dict[str, Path], dict[str, Path]]:
    """Index image filenames to disk paths and their label paths."""
    print("Indexing validation images and labels on disk...")
    img_lookup = {}
    lbl_lookup = {}

    val_dirs = [Path(val_source)] if isinstance(val_source, (str, Path)) else [Path(p) for p in val_source]

    for d in val_dirs:
        if d.is_file():
            with open(d, "r", encoding="utf-8") as f:
                for line in f:
                    p = Path(line.strip())
                    if p.exists():
                        img_lookup[p.name] = p
                        lbl = find_label_file(p)
                        if lbl:
                            lbl_lookup[p.name] = lbl
        elif d.is_dir():
            for p in d.rglob("*.*"):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    img_lookup[p.name] = p
                    lbl = find_label_file(p)
                    if lbl:
                        lbl_lookup[p.name] = lbl

    print(f"Indexed {len(img_lookup)} images and {len(lbl_lookup)} label files.")
    return img_lookup, lbl_lookup


def define_size_bins():
    """
    Define size bins based on max(width, height) in pixels.
    Tailored for UAV tiny target detection.
    """
    return [
        {"name": "极微小点目标 (<= 3x3 px)", "max_side_min": 0.0, "max_side_max": 3.01},
        {"name": "超小目标 (4x4 ~ 6x6 px)", "max_side_min": 3.01, "max_side_max": 6.01},
        {"name": "弱小目标 (7x7 ~ 10x10 px)", "max_side_min": 6.01, "max_side_max": 10.01},
        {"name": "中微目标 (11x11 ~ 20x20 px)", "max_side_min": 10.01, "max_side_max": 20.01},
        {"name": "中近距无人机 (21x21 ~ 40x40 px)", "max_side_min": 20.01, "max_side_max": 40.01},
        {"name": "近距/大目标 (> 40x40 px)", "max_side_min": 40.01, "max_side_max": 9999.0},
    ]


def main():
    args = parse_args()
    cache_path = Path(args.cache_file)
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    exclude_list = [s.strip() for s in args.exclude_seqs.split(",") if s.strip()]

    print(f"\n>>> Loading cached inferences from: {cache_path}")
    with open(cache_path, "rb") as f:
        records = pickle.load(f)
    print(f"Loaded {len(records)} image predictions.")

    data_dict = check_det_dataset(args.data)
    img_lookup, lbl_lookup = build_image_and_label_lookup(data_dict["val"])

    # 1. 过滤排除掉指定的极难/异常序列
    filtered_records = []
    excluded_counts: dict[str, int] = {k: 0 for k in exclude_list}

    for r in records:
        im_name = r["im_name"]
        is_excluded = False
        for exc in exclude_list:
            if exc in im_name:
                is_excluded = True
                excluded_counts[exc] += 1
                break
        if not is_excluded:
            filtered_records.append(r)

    print("\n" + "=" * 80)
    print("EXCLUSION SUMMARY (排除的极端 Hard Case 视频序列):")
    print("=" * 80)
    total_excluded_frames = len(records) - len(filtered_records)
    for exc, cnt in excluded_counts.items():
        print(f"  - Excluded sequence: {exc:<35} -> {cnt} frames")
    print(f"Total frames excluded: {total_excluded_frames} / {len(records)} ({total_excluded_frames/len(records)*100:.1f}%)")
    print(f"Remaining evaluated frames: {len(filtered_records)}")
    print("=" * 80)

    # 2. 准备尺寸分桶统计
    size_bins_orig = define_size_bins()
    size_bins_640 = define_size_bins()

    for b in size_bins_orig:
        b["total_gt"] = 0
        b["tp"] = 0
    for b in size_bins_640:
        b["total_gt"] = 0
        b["tp"] = 0

    total_gt = 0
    total_tp = 0
    total_fp = 0

    # 缓存分辨率查找避免重复 cv2.imread
    seq_res_cache: dict[str, tuple[int, int]] = {}

    print(f"\nAnalyzing recall stratified by bbox sizes @ dist_thresh={args.dist_thresh:.1f}px, conf={args.conf:.2f}...")

    for r in filtered_records:
        im_name = r["im_name"]
        gt_pts_640 = r["gt_pts"]  # (M, 2) in 640x640 coords
        raw_pred_pts = r["pred_points"]
        raw_pred_scs = r["pred_scores"]

        # 过滤预测置信度
        keep_p = raw_pred_scs >= args.conf
        pred_pts = raw_pred_pts[keep_p]

        # 读取该图的 GT Bbox 实际尺寸 (宽, 高)
        lbl_p = lbl_lookup.get(im_name)
        if not lbl_p:
            for ext in [".jpg", ".png", ".jpeg"]:
                alt_lbl = lbl_lookup.get(im_name + ext)
                if alt_lbl:
                    lbl_p = alt_lbl
                    break

        # 获取原图分辨率
        img_p = img_lookup.get(im_name)
        if not img_p:
            for ext in [".jpg", ".png", ".jpeg"]:
                alt_img = img_lookup.get(im_name + ext)
                if alt_img:
                    img_p = alt_img
                    break

        seq_id = im_name.split("___")[0] if "___" in im_name else im_name[:15]
        if seq_id in seq_res_cache:
            W_orig, H_orig = seq_res_cache[seq_id]
        elif img_p and img_p.exists():
            im = cv2.imread(str(img_p))
            if im is not None:
                W_orig, H_orig = im.shape[1], im.shape[0]
                seq_res_cache[seq_id] = (W_orig, H_orig)
            else:
                W_orig, H_orig = 640, 512
        else:
            W_orig, H_orig = 640, 512

        gt_boxes_orig = []
        gt_boxes_640 = []
        if lbl_p and lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        bw_norm = float(parts[3])
                        bh_norm = float(parts[4])
                        # 过滤掉大于整图 80% 的伪标签/全屏脏框
                        if bw_norm > 0.8 and bh_norm > 0.8:
                            continue
                        gt_boxes_orig.append((bw_norm * W_orig, bh_norm * H_orig))
                        gt_boxes_640.append((bw_norm * args.imgsz, bh_norm * args.imgsz))

        num_gt = len(gt_pts_640)
        total_gt += num_gt

        # 执行匹配
        matched_gt = set()
        matched_pred = set()

        if len(pred_pts) > 0 and num_gt > 0:
            diff = pred_pts[:, np.newaxis, :] - gt_pts_640[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))

            p_inds, g_inds = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
            for p_i, g_i in zip(p_inds, g_inds):
                if dists[p_i, g_i] > args.dist_thresh:
                    break
                if p_i not in matched_pred and g_i not in matched_gt:
                    matched_pred.add(p_i)
                    matched_gt.add(g_i)

        total_tp += len(matched_gt)
        total_fp += (len(pred_pts) - len(matched_pred))

        # 将每个 GT 归入相应的尺寸区间
        for g_i in range(num_gt):
            is_hit = g_i in matched_gt

            # 原图尺寸
            if g_i < len(gt_boxes_orig):
                w_orig, h_orig = gt_boxes_orig[g_i]
                max_side_orig = max(w_orig, h_orig)
            else:
                max_side_orig = 4.0

            for b in size_bins_orig:
                if b["max_side_min"] <= max_side_orig < b["max_side_max"]:
                    b["total_gt"] += 1
                    if is_hit:
                        b["tp"] += 1
                    break

            # 640 尺度下的尺寸
            if g_i < len(gt_boxes_640):
                w_640, h_640 = gt_boxes_640[g_i]
                max_side_640 = max(w_640, h_640)
            else:
                max_side_640 = 4.0

            for b in size_bins_640:
                if b["max_side_min"] <= max_side_640 < b["max_side_max"]:
                    b["total_gt"] += 1
                    if is_hit:
                        b["tp"] += 1
                    break

    # 3. 格式化输出报表
    overall_recall = total_tp / (total_gt + 1e-6) * 100
    overall_precision = total_tp / (total_tp + total_fp + 1e-6) * 100
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall + 1e-6)

    print("\n" + "=" * 90)
    print(f"REMAINING DATASET EVALUATION RESULTS (Distance <= {args.dist_thresh:.1f}px, Conf >= {args.conf:.2f}):")
    print("=" * 90)
    print(f"Total Evaluated Ground-Truths : {total_gt}")
    print(f"Successfully Recalled (TP)    : {total_tp}")
    print(f"False Alarms (FP)             : {total_fp}")
    print(f"OVERALL RECALL                : {overall_recall:.2f}%")
    print(f"OVERALL PRECISION             : {overall_precision:.2f}%")
    print(f"OVERALL F1-SCORE              : {overall_f1/100:.4f}")
    print("=" * 90)

    # 4. 原图物理分辨率分桶表格
    table_orig = []
    for b in size_bins_orig:
        gt_cnt = b["total_gt"]
        tp_cnt = b["tp"]
        fn_cnt = gt_cnt - tp_cnt
        rec = (tp_cnt / max(gt_cnt, 1)) * 100
        prop = (gt_cnt / max(total_gt, 1)) * 100
        table_orig.append([
            b["name"],
            f"{b['max_side_min']:.0f} ~ {b['max_side_max']:.0f} px",
            gt_cnt,
            f"{prop:.1f}%",
            tp_cnt,
            fn_cnt,
            f"{rec:6.2f}%",
        ])

    print("\n" + "-" * 90)
    print("【表 1：按原图物理分辨率尺寸统计 (Original Resolution Bbox)】")
    print("-" * 90)
    headers = ["目标尺度区间", "像素跨度", "总目标(GT)", "样本占比", "成功召回(TP)", "漏检(FN)", "召回率 (Recall)"]
    print(tabulate(table_orig, headers=headers, tablefmt="github"))

    # 5. 640 Resized 尺寸分桶表格
    table_640 = []
    for b in size_bins_640:
        gt_cnt = b["total_gt"]
        tp_cnt = b["tp"]
        fn_cnt = gt_cnt - tp_cnt
        rec = (tp_cnt / max(gt_cnt, 1)) * 100
        prop = (gt_cnt / max(total_gt, 1)) * 100
        table_640.append([
            b["name"],
            f"{b['max_side_min']:.0f} ~ {b['max_side_max']:.0f} px",
            gt_cnt,
            f"{prop:.1f}%",
            tp_cnt,
            fn_cnt,
            f"{rec:6.2f}%",
        ])

    print("\n" + "-" * 90)
    print("【表 2：按送入网络 640x640 实际尺寸统计 (Resized 640x640 Bbox)】")
    print("-" * 90)
    print(tabulate(table_640, headers=headers, tablefmt="github"))
    print("-" * 90 + "\n")


if __name__ == "__main__":
    main()
