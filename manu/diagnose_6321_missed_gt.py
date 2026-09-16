#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Specialized Diagnostic for 02_6321_0274-2773:
Analyzes Ground Truth Bbox sizes, Heatmap peak responses, and scores for missed frames.

Specifically answers:
1. When target is missed (under Distance <= 8px or Point-in-BBox), what are its GT width/height?
2. What is the nearest Heatmap peak score? (Does the model have faint response or is it totally 0.00?)
3. What is the distance distribution from GT center to the nearest Heatmap peak?
4. How do detection and miss patterns correlate with the video timeline (e.g. frame index, zoom-out phase)?
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr


def natural_sort_key(path_or_str: str | Path):
    s = Path(path_or_str).stem
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze 02_6321 missed frames vs GT sizes and HM scores")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median",
        help="Path to dataset root",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial0474_cache.pkl",
        help="Path to inference cache (e.g. Trial 0474 cache)",
    )
    parser.add_argument("--seq", type=str, default="02_6321_0274-2773", help="Target sequence name")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--th-hit", type=float, default=0.22, help="Threshold to consider a peak as valid hit")
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="Distance threshold in pixels")
    parser.add_argument(
        "--max-search-radius",
        type=float,
        default=50.0,
        help="Max radius (px) to search for nearby Heatmap peaks",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    val_img_dir = data_root / "images" / "val"
    val_lbl_dir = data_root / "labels" / "val"

    # Fallback paths for remote / local mounts
    if not val_lbl_dir.exists():
        for cand in [
            Path("/mnt/data/siping/datasets/manu/uav_gmc_median"),
            Path("/home/manu/mnt/datasets/manu/uav_gmc_median"),
            PROJECT_ROOT / "datasets/uav_gmc_median",
        ]:
            if (cand / "labels" / "val").exists():
                val_img_dir = cand / "images" / "val"
                val_lbl_dir = cand / "labels" / "val"
                break

    cache_path = Path(args.cache_file)
    if not cache_path.is_absolute():
        for cand in [
            PROJECT_ROOT / cache_path,
            Path("/tmp/pycharm_project_10ae9e2e") / cache_path,
            Path("/home/manu/mnt/pycharm_project_10ae9e2e") / cache_path,
        ]:
            if cand.exists():
                cache_path = cand
                break

    print(colorstr("bold", f"\n>>> Target Sequence Diagnostic: {args.seq}"))
    print(f"Dataset root : {val_lbl_dir.parent}")
    print(f"Cache file   : {cache_path}")
    print(f"Hit criteria : (dist <= {args.dist_thresh:.1f}px OR Point-in-BBox) AND score >= {args.th_hit:.2f}\n")

    if not cache_path.exists():
        print(colorstr("red", f"[ERROR] Cache file not found: {cache_path}"))
        sys.exit(1)

    with open(cache_path, "rb") as f:
        records = pickle.load(f)

    # Index cache records by image filename stem
    cache_by_stem = {}
    for r in records:
        stem = Path(r["im_name"]).stem
        if args.seq in stem:
            cache_by_stem[stem] = r

    # Collect all image/label pairs for target sequence
    lbl_files = sorted(
        [p for p in val_lbl_dir.glob(f"*{args.seq}*.txt")],
        key=natural_sort_key,
    )
    if not lbl_files:
        print(colorstr("red", f"[ERROR] No label files found matching {args.seq} in {val_lbl_dir}"))
        sys.exit(1)

    print(f"[INFO] Found {len(lbl_files)} annotated frames for {args.seq}")

    # Analysis storage
    all_frames = []
    hits = []
    misses = []

    for f_idx, lbl_p in enumerate(lbl_files):
        with open(lbl_p, "r", encoding="utf-8") as f:
            lines = [l.strip().split() for l in f if l.strip()]
        if not lines:
            continue

        # Extract first ground truth box (normalized cx, cy, w, h)
        box = [float(x) for x in lines[0][1:5]]
        cx = box[0] * args.imgsz
        cy = box[1] * args.imgsz
        w = box[2] * args.imgsz
        h = box[3] * args.imgsz
        diag = np.sqrt(w**2 + h**2)
        max_dim = max(w, h)
        x1, y1 = cx - w / 2.0, cy - h / 2.0
        x2, y2 = cx + w / 2.0, cy + h / 2.0

        rec = cache_by_stem.get(lbl_p.stem)
        if rec is None or len(rec.get("pred_points", [])) == 0:
            pred_pts = np.zeros((0, 2), dtype=np.float32)
            pred_scs = np.zeros((0,), dtype=np.float32)
        else:
            pred_pts = np.asarray(rec["pred_points"], dtype=np.float32)
            pred_scs = np.asarray(rec["pred_scores"], dtype=np.float32)

        is_hit = False
        hit_pt = None
        hit_sc = 0.0
        hit_dist = 999.0
        hit_type = "NONE"

        # Check if any peak qualifies as a HIT (score >= th_hit and inside bbox or dist <= dist_thresh)
        if len(pred_pts) > 0:
            dists = np.linalg.norm(pred_pts - np.array([cx, cy]), axis=1)
            in_bbox = (pred_pts[:, 0] >= x1) & (pred_pts[:, 0] <= x2) & (pred_pts[:, 1] >= y1) & (pred_pts[:, 1] <= y2)
            qualifies = (pred_scs >= args.th_hit) & ((dists <= args.dist_thresh) | in_bbox)

            if np.any(qualifies):
                is_hit = True
                valid_idx = np.where(qualifies)[0]
                # Pick the closest valid hit
                best_idx = valid_idx[np.argmin(dists[valid_idx])]
                hit_pt = pred_pts[best_idx]
                hit_sc = float(pred_scs[best_idx])
                hit_dist = float(dists[best_idx])
                if hit_dist <= args.dist_thresh:
                    hit_type = "DIST"
                else:
                    hit_type = "IN_BBOX_OFFSET"

        # Also find the NEAREST peak regardless of score (within search radius)
        nearest_sc = 0.0
        nearest_dist = 999.0
        nearest_in_bbox = False
        if len(pred_pts) > 0:
            dists_all = np.linalg.norm(pred_pts - np.array([cx, cy]), axis=1)
            min_idx = np.argmin(dists_all)
            nearest_dist = float(dists_all[min_idx])
            nearest_sc = float(pred_scs[min_idx])
            nearest_pt = pred_pts[min_idx]
            nearest_in_bbox = bool((nearest_pt[0] >= x1) & (nearest_pt[0] <= x2) & (nearest_pt[1] >= y1) & (nearest_pt[1] <= y2))

        # Best peak inside GT Bbox (if any)
        in_bbox_mask = (pred_pts[:, 0] >= x1) & (pred_pts[:, 0] <= x2) & (pred_pts[:, 1] >= y1) & (pred_pts[:, 1] <= y2) if len(pred_pts) > 0 else []
        max_in_bbox_sc = float(np.max(pred_scs[in_bbox_mask])) if np.any(in_bbox_mask) else 0.0

        # Best peak within dist_thresh
        near_mask = (dists_all <= args.dist_thresh) if len(pred_pts) > 0 else []
        max_near_sc = float(np.max(pred_scs[near_mask])) if np.any(near_mask) else 0.0

        item = {
            "frame_idx": f_idx,
            "stem": lbl_p.stem,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "max_dim": max_dim,
            "diag": diag,
            "is_hit": is_hit,
            "hit_type": hit_type,
            "hit_sc": hit_sc,
            "hit_dist": hit_dist,
            "nearest_sc": nearest_sc,
            "nearest_dist": nearest_dist,
            "nearest_in_bbox": nearest_in_bbox,
            "max_in_bbox_sc": max_in_bbox_sc,
            "max_near_sc": max_near_sc,
        }
        all_frames.append(item)
        if is_hit:
            hits.append(item)
        else:
            misses.append(item)

    total_gt = len(all_frames)
    total_hits = len(hits)
    total_misses = len(misses)
    rec = total_hits / total_gt * 100.0

    print("=" * 100)
    print(colorstr("bold", f"📊 SUMMARY FOR {args.seq}"))
    print("=" * 100)
    print(f"Total Annotated GT Frames : {total_gt}")
    print(f"Direct Single-Frame Hits  : {total_hits} ({rec:.2f}%)")
    print(f"Missed Frames (FN)        : {total_misses} ({total_misses/total_gt*100:.2f}%)")
    print("=" * 100 + "\n")

    # =========================================================================
    # 1. Bbox Size Analysis: HITS vs MISSES
    # =========================================================================
    print(colorstr("bold", "【1. GT 尺寸分布分析 (GT Bbox Size: Hits vs Misses)】"))
    size_bins = [
        ("极小/微弱脉冲 (Tiny)", 0.0, 8.0),
        ("小尺寸目标 (Small)", 8.0, 16.0),
        ("中等尺寸目标 (Medium)", 16.0, 32.0),
        ("大尺寸目标 (Large)", 32.0, 64.0),
        ("超大尺寸机体 (Extra Large)", 64.0, 999.0),
    ]

    size_table = []
    for label, min_s, max_s in size_bins:
        sub_all = [x for x in all_frames if min_s <= x["max_dim"] < max_s]
        sub_hits = [x for x in hits if min_s <= x["max_dim"] < max_s]
        sub_misses = [x for x in misses if min_s <= x["max_dim"] < max_s]
        n_all = len(sub_all)
        n_hit = len(sub_hits)
        n_mis = len(sub_misses)
        sub_rec = (n_hit / n_all * 100.0) if n_all > 0 else 0.0
        prop = (n_all / total_gt * 100.0) if total_gt > 0 else 0.0
        size_table.append([
            label,
            f"{min_s:.0f} ~ {max_s:.0f} px",
            n_all,
            f"{prop:.1f}%",
            n_hit,
            n_mis,
            f"{sub_rec:.2f}%",
        ])

    print(tabulate(size_table, headers=["尺度区间", "像素跨度 (max(w,h))", "总GT帧数", "全序列占比", "命中数 (TP)", "漏检数 (FN)", "单帧召回率"], tablefmt="github"))
    print()

    # =========================================================================
    # 2. Score Breakdown for Missed Frames: Is it 0.00 or faint response?
    # =========================================================================
    print(colorstr("bold", "【2. 漏检样本 (369+ 帧) 的 Heatmap 置信度峰值分布分析】"))
    print("探究问题：模型在漏检处是完全致盲(0.00)，还是有微弱脉冲被门限截断？")

    # For misses: look at max response in proximity (within max(w,h)/2 or 10px)
    score_bins = [
        ("强弱响应 (0.15 <= Score < 0.22, 临界被卡)", 0.15, args.th_hit),
        ("中弱响应 (0.08 <= Score < 0.15, 明显脉冲)", 0.08, 0.15),
        ("微弱响应 (0.03 <= Score < 0.08, 深潜可捞)", 0.03, 0.08),
        ("底噪响应 (0.01 <= Score < 0.03, 极度微弱)", 0.01, 0.03),
        ("完全致盲 (Score < 0.01 或 25px 内无峰)", 0.0, 0.01),
    ]

    score_table = []
    for label, min_sc, max_sc in score_bins:
        # Check within target vicinity (dist <= 15px or inside bbox)
        matched = []
        for m in misses:
            sc = max(m["max_in_bbox_sc"], m["max_near_sc"])
            if min_sc == 0.0:
                if sc < max_sc or m["nearest_dist"] > 25.0:
                    matched.append(m)
            else:
                if (min_sc <= sc < max_sc) and m["nearest_dist"] <= 25.0:
                    matched.append(m)

        n_cnt = len(matched)
        prop_mis = (n_cnt / total_misses * 100.0) if total_misses > 0 else 0.0
        score_table.append([label, n_cnt, f"{prop_mis:.1f}%"])

    print(tabulate(score_table, headers=["热图响应等级 (目标近邻区域极大值)", "漏检帧数 (FN)", "在漏检中的占比"], tablefmt="github"))
    print()

    # =========================================================================
    # 3. Distance Distribution of Peaks for Missed Frames
    # =========================================================================
    print(colorstr("bold", "【3. 漏检样本近邻热图峰值与 GT 中心的空间距离分布】"))
    dist_bins = [
        ("峰值在 GT Bbox 内部 (但 score < 0.22)", lambda m: m["nearest_in_bbox"] and m["nearest_sc"] > 0),
        ("峰值在 dist <= 8.0px (但 score < 0.22)", lambda m: m["nearest_dist"] <= 8.0 and not m["nearest_in_bbox"]),
        ("峰值在 8.0px < dist <= 15.0px", lambda m: 8.0 < m["nearest_dist"] <= 15.0),
        ("峰值在 15.0px < dist <= 30.0px", lambda m: 15.0 < m["nearest_dist"] <= 30.0),
        ("25px 范围内彻底无任何峰值 (Totally Blind)", lambda m: m["nearest_dist"] > 25.0 or m["nearest_sc"] < 0.01),
    ]

    dist_table = []
    for label, cond in dist_bins:
        matched = [m for m in misses if cond(m)]
        dist_table.append([label, len(matched), f"{len(matched)/total_misses*100:.1f}%"])

    print(tabulate(dist_table, headers=["空间距离类别", "漏检帧数 (FN)", "在漏检中的占比"], tablefmt="github"))
    print()

    # =========================================================================
    # 4. Timeline Breakdown: Where do misses happen in the video?
    # =========================================================================
    print(colorstr("bold", "【4. 视频时序演进分析 (Timeline Breakdown: 前程机动 vs 后程远处悬停)】"))
    chunk_size = 250
    timeline_table = []
    for start_f in range(0, total_gt, chunk_size):
        end_f = min(start_f + chunk_size, total_gt)
        chunk_frames = all_frames[start_f:end_f]
        chunk_hits = [x for x in chunk_frames if x["is_hit"]]
        chunk_miss = [x for x in chunk_frames if not x["is_hit"]]
        c_tot = len(chunk_frames)
        c_hit = len(chunk_hits)
        c_mis = len(chunk_miss)
        c_rec = (c_hit / c_tot * 100.0) if c_tot > 0 else 0.0
        avg_w = np.mean([x["w"] for x in chunk_frames])
        avg_h = np.mean([x["h"] for x in chunk_frames])
        avg_sc = np.mean([x["nearest_sc"] for x in chunk_frames])
        timeline_table.append([
            f"Frame #{start_f:04d} ~ #{end_f-1:04d}",
            c_tot,
            f"{avg_w:.1f} x {avg_h:.1f} px",
            c_hit,
            c_mis,
            f"{c_rec:.2f}%",
            f"{avg_sc:.3f}",
        ])

    print(tabulate(timeline_table, headers=["时序区间", "帧数", "平均 GT 尺寸 (w x h)", "命中数 (TP)", "漏检数 (FN)", "区间召回率", "近邻平均 Score"], tablefmt="github"))
    print()

    # =========================================================================
    # 5. Representative Missed Frames Sampling
    # =========================================================================
    print(colorstr("bold", "【5. 典型漏检帧抽样展示 (Top 10 Representative Misses)】"))
    sample_table = []
    # Pick a few evenly spaced misses
    step = max(1, len(misses) // 10)
    for m in misses[::step][:10]:
        sample_table.append([
            m["stem"],
            f"{m['w']:.1f} x {m['h']:.1f}",
            f"{m['nearest_dist']:.1f} px",
            f"{m['nearest_sc']:.4f}",
            "YES" if m["nearest_in_bbox"] else "NO",
            f"{m['max_in_bbox_sc']:.4f}",
        ])

    print(tabulate(sample_table, headers=["帧名称", "GT尺寸 (w x h)", "最近峰值距离", "最近峰值分", "最近峰在BBox内?", "BBox内最高分"], tablefmt="github"))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
