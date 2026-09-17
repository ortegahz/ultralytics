#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Cross-Sequence Response / Input Statistics Comparator (Pure Measurement).

Purpose
-------
Compare, using only hard statistics, why one sequence fails while others succeed.
No semantic interpretation (no "cloud", no "flicker", no causal claims).

Two independent measurement blocks:
  A. RESPONSE statistics (cache-only, no images):
     For every GT point, find the nearest predicted peak in the SAME cache coordinate
     space (640x640 letterbox) and record its distance, score and within-frame score rank.
     Reports how often the model produces a peak at/near the target at all.
  B. INPUT statistics (optional, reads native images at native coordinates):
     Robust target/background contrast and the diff / median channel levels at the GT.

Coordinate contract
-------------------
Cache `gt_pts` and `pred_points` are both in the 640x640 letterbox space -> safe to compare
directly. Input pixel sampling uses NATIVE coordinates (box[0]*W, box[1]*H). The two blocks
never mix spaces.

Usage on server:
    python manu/compare_sequence_response.py \
        --cache-file runs/gmc_eval/uav_median_trial0474_cache.pkl \
        --data-root /mnt/data/siping/datasets/manu/uav_gmc_median
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr

DEFAULT_SEQUENCES = [
    "wg2022_ir_020_split_03",
    "wg2022_ir_011_split_03",
    "DJI_0175_2",
    "02_6321_0274-2773",
    "wg2022_ir_011_split_02",
    "wg2022_ir_020_split_01",
    "wg2022_ir_047_split_01",
    "5_1",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-sequence response/input statistics comparator")
    parser.add_argument("--cache-file", type=str, default="runs/gmc_eval/uav_median_trial0474_cache.pkl")
    parser.add_argument("--data-root", type=str, default="/mnt/data/siping/datasets/manu/uav_gmc_median")
    parser.add_argument("--sequences", type=str, default="", help="Comma-separated sequence names (default: built-in list)")
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    parser.add_argument("--th-detected", type=float, default=0.22)
    parser.add_argument("--th-weak", type=float, default=0.06)
    parser.add_argument("--no-input-stats", action="store_true", help="Skip image-based input statistics (cache-only)")
    parser.add_argument("--max-frames", type=int, default=0, help="Cap frames read per sequence for input stats (0 = all)")
    return parser.parse_args()


def extract_seq_name(im_name: str) -> str:
    stem = Path(im_name).stem
    if "___" in stem:
        return stem.split("___")[0]
    if "__" in stem:
        return stem.split("__")[0]
    match = re.search(r"^(.*?)(?:[_-]+)?\d{3,}$", stem)
    return match.group(1).rstrip("_-") if match else stem


def patch_metrics(channel: np.ndarray, cx: int, cy: int) -> Dict[str, float]:
    h, w = channel.shape[:2]
    if not (10 <= cx < w - 10 and 10 <= cy < h - 10):
        return {"signed": 0.0, "contrast": 0.0, "mad_z": 0.0}
    target = channel[cy - 2 : cy + 3, cx - 2 : cx + 3].astype(np.float32)
    outer = channel[cy - 10 : cy + 11, cx - 10 : cx + 11].astype(np.float32)
    mask = np.ones((21, 21), dtype=bool)
    mask[5:16, 5:16] = False
    background = outer[mask]
    bg_median = float(np.median(background))
    bg_mad = float(np.median(np.abs(background - bg_median)))
    bg_mean = float(np.mean(background))
    bg_std = float(np.std(background))
    peak = float(np.max(target))
    trough = float(np.min(target))
    signed = peak - bg_median if (peak - bg_median) >= (bg_median - trough) else trough - bg_median
    return {
        "signed": signed,
        "contrast": signed / (bg_std + 1e-4),
        "mad_z": abs(signed) / (1.4826 * bg_mad + 1e-4),
    }


def response_stats(records: List[dict], dist_thresh: float, th_det: float, th_weak: float) -> Dict[str, float]:
    n_gt = 0
    frames_with_gt = 0
    frames_hit_weak = 0
    frames_hit_det = 0
    dists, scores, ranks = [], [], []
    peaks_per_frame, top1_scores = [], []

    for rec in records:
        gt = np.asarray(rec["gt_pts"], dtype=np.float32)
        pred = np.asarray(rec["pred_points"], dtype=np.float32)
        scs = np.asarray(rec["pred_scores"], dtype=np.float32)
        if gt.size == 0:
            continue
        frames_with_gt += 1
        peaks_per_frame.append(int(pred.shape[0]))
        top1_scores.append(float(scs.max()) if scs.size else 0.0)

        frame_hit_weak = False
        frame_hit_det = False
        if pred.shape[0] == 0:
            for _ in range(gt.shape[0]):
                dists.append(np.inf)
                scores.append(0.0)
                ranks.append(0)
                n_gt += 1
        else:
            order = np.argsort(-scs)
            rank_of = np.empty(pred.shape[0], dtype=np.int32)
            rank_of[order] = np.arange(1, pred.shape[0] + 1)
            for g in gt:
                d = np.linalg.norm(pred - g, axis=1)
                i = int(np.argmin(d))
                d_i, s_i = float(d[i]), float(scs[i])
                dists.append(d_i)
                scores.append(s_i)
                ranks.append(int(rank_of[i]))
                if d_i <= dist_thresh and s_i >= th_weak:
                    frame_hit_weak = True
                if d_i <= dist_thresh and s_i >= th_det:
                    frame_hit_det = True
                n_gt += 1
        if frame_hit_weak:
            frames_hit_weak += 1
        if frame_hit_det:
            frames_hit_det += 1

    dists = np.asarray(dists, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    ranks = np.asarray(ranks, dtype=np.float32)
    near = dists <= dist_thresh
    return {
        "frames_with_gt": frames_with_gt,
        "n_gt": n_gt,
        "pct_near": float(np.mean(near) * 100) if n_gt else 0.0,
        "pct_det": float(np.mean(near & (scores >= th_det)) * 100) if n_gt else 0.0,
        "pct_weak": float(np.mean(near & (scores >= th_weak)) * 100) if n_gt else 0.0,
        "median_nearest_dist": float(np.median(dists)) if n_gt else 0.0,
        "median_nearest_score": float(np.median(scores)) if n_gt else 0.0,
        "median_rank": float(np.median(ranks)) if n_gt else 0.0,
        "median_peaks_per_frame": float(np.median(peaks_per_frame)) if peaks_per_frame else 0.0,
        "median_top1_score": float(np.median(top1_scores)) if top1_scores else 0.0,
        "pct_frames_hit_weak": float(frames_hit_weak / frames_with_gt * 100) if frames_with_gt else 0.0,
        "pct_frames_hit_det": float(frames_hit_det / frames_with_gt * 100) if frames_with_gt else 0.0,
    }


def input_stats(img_dir: Path, lbl_dir: Path, seq: str, max_frames: int) -> Dict[str, float]:
    import cv2

    files = sorted(
        [p for p in img_dir.glob(f"{seq}__*") if p.suffix.lower() in (".jpg", ".png")],
        key=lambda p: p.stem,
    )
    if max_frames:
        files = files[:max_frames]

    raw_signed, raw_contrast, raw_madz = [], [], []
    diff_peak, median_peak = [], []
    resolutions: Dict[str, int] = {}

    for p in files:
        lbl = lbl_dir / f"{p.stem}.txt"
        if not lbl.exists():
            continue
        boxes = []
        with lbl.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    boxes.append([float(v) for v in parts[1:5]])
        if not boxes:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape[:2]
        resolutions[f"{w}x{h}"] = resolutions.get(f"{w}x{h}", 0) + 1
        if img.ndim == 3 and img.shape[2] >= 3:
            ch0, ch1, ch2 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        else:
            ch0, ch1, ch2 = img, np.zeros_like(img), np.zeros_like(img)

        for b in boxes:
            cx, cy = int(round(b[0] * w)), int(round(b[1] * h))
            m0 = patch_metrics(ch0, cx, cy)
            raw_signed.append(m0["signed"])
            raw_contrast.append(m0["contrast"])
            raw_madz.append(m0["mad_z"])
            diff_peak.append(patch_metrics(ch1, cx, cy)["signed"])
            median_peak.append(patch_metrics(ch2, cx, cy)["signed"])

    def med(v):
        return float(np.median(v)) if v else 0.0

    return {
        "n_boxes": len(raw_madz),
        "resolutions": resolutions,
        "raw_signed_med": med(raw_signed),
        "raw_contrast_med": med(raw_contrast),
        "raw_madz_med": med(raw_madz),
        "diff_med": med(diff_peak),
        "median_med": med(median_peak),
    }


def main():
    args = parse_args()
    sequences = [s.strip() for s in args.sequences.split(",") if s.strip()] or DEFAULT_SEQUENCES

    cache_path = Path(args.cache_file)
    if not cache_path.is_absolute():
        for cand in [PROJECT_ROOT / cache_path, Path("/tmp/pycharm_project_10ae9e2e") / cache_path]:
            if cand.exists():
                cache_path = cand
                break
    if not cache_path.exists():
        print(colorstr("red", f"[ERROR] Cache not found: {args.cache_file}"))
        sys.exit(1)

    print(f"[INFO] Loading cache: {cache_path}")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    grouped: Dict[str, List[dict]] = {}
    for rec in cache:
        grouped.setdefault(extract_seq_name(rec["im_name"]), []).append(rec)
    print(f"[INFO] Cache contains {len(cache):,} frames across {len(grouped)} sequences.")

    data_root = Path(args.data_root)
    img_dir = data_root / "images" / "val"
    lbl_dir = data_root / "labels" / "val"
    do_input = not args.no_input_stats and img_dir.exists()

    rows = []
    for seq in sequences:
        recs = grouped.get(seq)
        if not recs:
            print(colorstr("yellow", f"[WARN] Sequence not found in cache: {seq}"))
            continue
        r = response_stats(recs, args.dist_thresh, args.th_detected, args.th_weak)
        r["seq"] = seq
        if do_input:
            r.update(input_stats(img_dir, lbl_dir, seq, args.max_frames))
        rows.append(r)

    if not rows:
        print(colorstr("red", "[ERROR] No sequences processed."))
        return

    print("\n" + "=" * 140)
    print(colorstr("bold", colorstr("cyan", "A. RESPONSE STATISTICS (cache-only, 640x640 letterbox space)")))
    print("=" * 140)
    print(
        f"{'Sequence':<28} | {'GT':<6} | {'Frames':<7} | {'%near8px':<9} | {'%det>=0.22':<11} | {'%weak>=0.06':<11} | "
        f"{'medNearDist':<11} | {'medNearScore':<12} | {'medRank':<8} | {'peaks/f':<8} | {'top1Score':<10}"
    )
    print("-" * 140)
    for r in rows:
        print(
            f"{r['seq']:<28} | {r['n_gt']:<6} | {r['frames_with_gt']:<7} | {r['pct_near']:>8.1f}% | "
            f"{r['pct_det']:>10.1f}% | {r['pct_weak']:>10.1f}% | {r['median_nearest_dist']:>11.1f} | "
            f"{r['median_nearest_score']:>12.3f} | {r['median_rank']:>8.0f} | {r['median_peaks_per_frame']:>8.0f} | "
            f"{r['median_top1_score']:>10.3f}"
        )
    print("=" * 140)
    print("Columns: %near8px = GT with any predicted peak within 8px | %det = peak within 8px AND score>=th_detected")
    print("         medNearDist/Score = median distance/score of the nearest peak to GT | medRank = median rank of that")
    print("         peak among the frame's peaks (1 = highest score) | top1Score = median frame-best score")

    if do_input and rows and "n_boxes" in rows[0]:
        print("\n" + "=" * 140)
        print(colorstr("bold", colorstr("cyan", "B. INPUT STATISTICS (native coordinates, polarity-agnostic)")))
        print("=" * 140)
        print(
            f"{'Sequence':<28} | {'boxes':<6} | {'resolution':<12} | {'rawSignedMed':<13} | {'rawContrastMed':<14} | "
            f"{'rawMadZMed':<11} | {'diffMed':<8} | {'medianMed':<10}"
        )
        print("-" * 140)
        for r in rows:
            res = ",".join(f"{k}" for k in r.get("resolutions", {})) or "-"
            print(
                f"{r['seq']:<28} | {r.get('n_boxes', 0):<6} | {res:<12} | {r.get('raw_signed_med', 0.0):>13.2f} | "
                f"{r.get('raw_contrast_med', 0.0):>14.2f} | {r.get('raw_madz_med', 0.0):>11.2f} | "
                f"{r.get('diff_med', 0.0):>8.2f} | {r.get('median_med', 0.0):>10.2f}"
            )
        print("=" * 140)
        print("rawSignedMed = median(peak - local bg median), sign preserved (negative = dark target)")
        print("diffMed/medianMed = same signed statistic measured on channel 1 (GMC diff) and channel 2 (median residual)")
    print()


if __name__ == "__main__":
    main()
