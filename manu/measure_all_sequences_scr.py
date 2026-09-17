#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Batch Robust Signal-to-Clutter / Detectability Profiler for All Validation Sequences.

Motivation
----------
The previous profiler measured target signal as `max(5x5)` against the mean/std of an
annulus. That statistic is inflated: `max` of 25 pixels picks up a lucky bright noise pixel,
while `mean/std` is not robust to cloud gradients. It ranked wg2022_ir_020_split_03 as
"SCR=4.66, mid-pack", which contradicts the per-frame probe showing a noise-floor target.

Robust Infrared Standard Formulation
------------------------------------
For each ground truth target location (x, y):
1. Target Region (T): max over a 5x5 patch centered at GT -> I_T
2. Background Annulus (B): ring between 11x11 and 21x21 (excludes halo/transition edge)
3. Metrics:
   - Classic SCR   = |I_T - mean(B)| / std(B)                  (legacy, non-robust)
   - Robust MAD-Z  = (I_T - median(B)) / (1.4826 * MAD(B))     (primary, robust)
   - Peak Contrast = (I_T - mean(B)) / std(B)
   - Delta_I       = |I_T - median(B)|
The MAD-Z is the detection statistic: MAD-Z < 3 means the target peak does not stand out
from background noise, regardless of how large the legacy SCR looks.

Outputs
-------
- Per-sequence robust leaderboard sorted by median MAD-Z ascending (worst first).
- Legacy ranking retained side-by-side so rank flips are visible.
- Stratification by MAD-Z regimes and sub-3.0 / sub-2.0 / sub-1.0 fractions.

Usage
-----
    python manu/measure_all_sequences_scr.py \
        --data-root /mnt/data/siping/datasets/manu/uav_gmc_median
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr


def natural_sort_key(path_or_str: str | Path):
    s = Path(path_or_str).stem
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def extract_seq_name(im_name: str) -> str:
    stem = Path(im_name).stem
    if "___" in stem:
        return stem.split("___")[0]
    if "__" in stem:
        return stem.split("__")[0]
    match = re.search(r"^(.*?)(?:[_-]+)?\d{3,}$", stem)
    return match.group(1).rstrip("_-") if match else stem


def parse_args():
    parser = argparse.ArgumentParser(description="Measure Robust Detectability Across All UAV Sequences")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median",
        help="Path to dataset root containing images/val and labels/val",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--target-size", type=int, default=5, help="Target kernel size (default: 5x5)")
    parser.add_argument("--bg-inner", type=int, default=11, help="Background annulus inner size (default: 11x11)")
    parser.add_argument("--bg-outer", type=int, default=21, help="Background annulus outer size (default: 21x21)")
    return parser.parse_args()


def calculate_frame_metrics(
    img_gray: np.ndarray,
    gt_boxes: np.ndarray,
    target_size: int = 5,
    bg_inner: int = 11,
    bg_outer: int = 21,
) -> List[Dict[str, float]]:
    """
    Compute robust and legacy detectability metrics for each GT box in the frame.

    Returns a list of dicts with keys: scr_classic, mad_z, contrast, delta_i, peak,
    bg_mean, bg_std, bg_median, robust_sigma.
    """
    h, w = img_gray.shape[:2]
    results = []
    t_r = target_size // 2
    in_r = bg_inner // 2
    out_r = bg_outer // 2

    for box in gt_boxes:
        cx = int(round(box[0] * w))
        cy = int(round(box[1] * h))

        if cx - out_r < 0 or cx + out_r + 1 > w or cy - out_r < 0 or cy + out_r + 1 > h:
            continue

        target_patch = img_gray[cy - t_r : cy + t_r + 1, cx - t_r : cx + t_r + 1].astype(np.float32)
        if target_patch.size == 0:
            continue
        peak = float(np.max(target_patch))

        outer_patch = img_gray[cy - out_r : cy + out_r + 1, cx - out_r : cx + out_r + 1].astype(np.float32)
        mask = np.ones_like(outer_patch, dtype=bool)
        start_in = out_r - in_r
        end_in = out_r + in_r + 1
        mask[start_in:end_in, start_in:end_in] = False
        bg_pixels = outer_patch[mask]
        if len(bg_pixels) == 0:
            continue

        bg_mean = float(np.mean(bg_pixels))
        bg_std = float(np.std(bg_pixels))
        bg_median = float(np.median(bg_pixels))
        bg_mad = float(np.median(np.abs(bg_pixels - bg_median)))
        robust_sigma = 1.4826 * bg_mad + 1e-4

        results.append({
            "scr_classic": abs(peak - bg_mean) / (bg_std + 1e-4),
            "mad_z": (peak - bg_median) / robust_sigma,
            "contrast": (peak - bg_mean) / (bg_std + 1e-4),
            "delta_i": abs(peak - bg_median),
            "peak": peak,
            "bg_mean": bg_mean,
            "bg_std": bg_std,
            "bg_median": bg_median,
            "robust_sigma": robust_sigma,
        })

    return results


def classify_regime(median_mad_z: float) -> Tuple[str, str]:
    if median_mad_z < 1.5:
        return "🚨 NOISE-FLOOR (MAD-Z < 1.5)", "red"
    if median_mad_z < 3.0:
        return "⚠️ SUB-DETECTION (1.5 <= MAD-Z < 3.0)", "yellow"
    if median_mad_z < 5.0:
        return "⚡ MARGINAL (3.0 <= MAD-Z < 5.0)", "cyan"
    return "✅ DETECTABLE (MAD-Z >= 5.0)", "green"


def main():
    args = parse_args()
    data_root = Path(args.data_root)

    val_img_dir = data_root / "images" / "val"
    val_lbl_dir = data_root / "labels" / "val"

    if not val_img_dir.exists():
        alt_root = Path("/home/manu/mnt/datasets/manu/uav_gmc_median")
        if (alt_root / "images" / "val").exists():
            val_img_dir = alt_root / "images" / "val"
            val_lbl_dir = alt_root / "labels" / "val"
        else:
            print(colorstr("red", f"[ERROR] Validation images not found at: {val_img_dir}"))
            sys.exit(1)

    print("\n" + "=" * 150)
    print("🔬 ROBUST INFRARED DETECTABILITY PROFILER (MAD-Z PRIMARY, CLASSIC SCR FOR COMPARISON)")
    print(f"Dataset Images Root : {val_img_dir}")
    print(f"Target Kernel       : {args.target_size}x{args.target_size} px | Background Annulus: {args.bg_inner}x{args.bg_inner} ~ {args.bg_outer}x{args.bg_outer} px")
    print("=" * 150)

    img_files = sorted(list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png")), key=natural_sort_key)
    ext_counts: Dict[str, int] = {}
    for p in img_files:
        ext_counts[p.suffix.lower()] = ext_counts.get(p.suffix.lower(), 0) + 1
    print(f"[INFO] Found {len(img_files):,} validation image frames. File types: {ext_counts}")
    print("[INFO] Note: .jpg is lossy; a ~1 sigma point target can be altered by compression. Check extensions above.")

    seq_frames: Dict[str, List[Path]] = {}
    for p in img_files:
        seq = extract_seq_name(p.name)
        seq_frames.setdefault(seq, []).append(p)

    seq_stats = {}
    t0 = time.time()

    pbar = tqdm(sorted(seq_frames.keys()), desc="Measuring detectability per Sequence", dynamic_ncols=True, file=sys.stdout)

    for seq_name in pbar:
        paths = seq_frames[seq_name]
        scr_list, madz_list, contrast_list, delta_list, sigma_list = [], [], [], [], []
        total_gt_boxes = 0

        for p in paths:
            lbl_p = val_lbl_dir / f"{p.stem}.txt"
            if not lbl_p.exists():
                continue

            with open(lbl_p, "r", encoding="utf-8") as f:
                lines = [l.strip().split() for l in f if l.strip()]
            if not lines:
                continue

            gt_boxes = np.array([[float(x) for x in l[1:5]] for l in lines], dtype=np.float32)
            total_gt_boxes += len(gt_boxes)

            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            img_gray = img[:, :, 0] if img.ndim == 3 else img

            for m in calculate_frame_metrics(
                img_gray=img_gray,
                gt_boxes=gt_boxes,
                target_size=args.target_size,
                bg_inner=args.bg_inner,
                bg_outer=args.bg_outer,
            ):
                scr_list.append(m["scr_classic"])
                madz_list.append(m["mad_z"])
                contrast_list.append(m["contrast"])
                delta_list.append(m["delta_i"])
                sigma_list.append(m["robust_sigma"])

        if len(madz_list) > 0:
            madz = np.array(madz_list)
            seq_stats[seq_name] = {
                "frames": len(paths),
                "gt_boxes": total_gt_boxes,
                "measured_targets": len(madz_list),
                "median_mad_z": float(np.median(madz)),
                "mean_mad_z": float(np.mean(madz)),
                "p10_mad_z": float(np.percentile(madz, 10)),
                "p90_mad_z": float(np.percentile(madz, 90)),
                "mean_scr": float(np.mean(scr_list)),
                "median_scr": float(np.median(scr_list)),
                "mean_contrast": float(np.mean(contrast_list)),
                "mean_delta_i": float(np.mean(delta_list)),
                "mean_robust_sigma": float(np.mean(sigma_list)),
                "pct_sub3": float(np.mean(madz < 3.0) * 100),
                "pct_sub2": float(np.mean(madz < 2.0) * 100),
                "pct_sub1": float(np.mean(madz < 1.0) * 100),
            }
        else:
            seq_stats[seq_name] = {
                "frames": len(paths),
                "gt_boxes": 0,
                "measured_targets": 0,
                "median_mad_z": 999.0,
                "mean_mad_z": 999.0,
                "p10_mad_z": 999.0,
                "p90_mad_z": 999.0,
                "mean_scr": 999.0,
                "median_scr": 999.0,
                "mean_contrast": 0.0,
                "mean_delta_i": 0.0,
                "mean_robust_sigma": 0.0,
                "pct_sub3": 0.0,
                "pct_sub2": 0.0,
                "pct_sub1": 0.0,
            }

    pbar.close()

    target_seqs = [s for s in seq_stats.items() if s[1]["gt_boxes"] > 0]
    pure_neg_seqs = [s for s in seq_stats.items() if s[1]["gt_boxes"] == 0]

    robust_sorted = sorted(target_seqs, key=lambda x: x[1]["median_mad_z"])
    legacy_sorted = sorted(target_seqs, key=lambda x: x[1]["mean_scr"])
    robust_rank = {name: i for i, (name, _) in enumerate(robust_sorted, 1)}
    legacy_rank = {name: i for i, (name, _) in enumerate(legacy_sorted, 1)}

    print("\n" + "=" * 150)
    print(colorstr("bold", colorstr("cyan", "📊 ROBUST DETECTABILITY LEADERBOARD (SORTED BY MEDIAN MAD-Z ASCENDING, WORST FIRST)")))
    print("=" * 150)
    print(
        f"{'Rank':<4} | {'Sequence Name':<28} | {'GT':<5} | {'Med MAD-Z':<10} | {'P10':<7} | {'P90':<9} | "
        f"{'%<3.0':<7} | {'%<2.0':<7} | {'%<1.0':<7} | {'Legacy Mean SCR':<15} | {'Legacy Rank':<11} | {'Regime'}"
    )
    print("-" * 150)

    for i, (seq_name, stat) in enumerate(robust_sorted, 1):
        regime, color = classify_regime(stat["median_mad_z"])
        line = (
            f"{i:<4} | {seq_name:<28} | {stat['gt_boxes']:<5} | {stat['median_mad_z']:<10.2f} | "
            f"{stat['p10_mad_z']:<7.2f} | {stat['p90_mad_z']:<9.2f} | {stat['pct_sub3']:>6.1f}% | "
            f"{stat['pct_sub2']:>6.1f}% | {stat['pct_sub1']:>6.1f}% | {stat['mean_scr']:<15.2f} | "
            f"#{legacy_rank[seq_name]:<10} | {regime}"
        )
        if seq_name == "wg2022_ir_020_split_03":
            print(colorstr("bold", colorstr("magenta", f"👉 {line} 👈 [TARGET PROBE]")))
        elif color == "red":
            print(colorstr("bold", colorstr("red", line)))
        elif color == "yellow":
            print(colorstr("yellow", line))
        else:
            print(line)

    if pure_neg_seqs:
        print("-" * 150)
        for s_name, s in pure_neg_seqs:
            print(f" -   | {s_name:<28} | {0:<5} | Pure Negative Sequence (GT=0, Zero Target Radiation)")

    print("=" * 150)

    probe = "wg2022_ir_020_split_03"
    print("\n" + colorstr("bold", "🎯 ROBUST VERIFICATION CONCLUSION:"))
    if probe in seq_stats and seq_stats[probe]["gt_boxes"] > 0:
        p = seq_stats[probe]
        print(f"• {probe} Robust Rank  : #{robust_rank[probe]} / {len(robust_sorted)} (Median MAD-Z = {p['median_mad_z']:.2f})")
        print(f"• {probe} Legacy Rank  : #{legacy_rank[probe]} / {len(legacy_sorted)} (Mean Classic SCR = {p['mean_scr']:.2f})")
        print(f"• Frames with MAD-Z < 3.0 : {p['pct_sub3']:.1f}%  |  < 2.0 : {p['pct_sub2']:.1f}%  |  < 1.0 : {p['pct_sub1']:.1f}%")
        if robust_rank[probe] == 1:
            print(colorstr("bold", colorstr("green", "✅ CONFIRMED: wg2022_ir_020_split_03 is the WORST detectability sequence under robust MAD-Z.")))
        else:
            print(colorstr("bold", colorstr("yellow", f"⚠️ {probe} is NOT rank #1 under MAD-Z; {robust_sorted[0][0]} is the worst.")))
        if legacy_rank[probe] != robust_rank[probe]:
            print(colorstr("yellow", f"• Rank mismatch: legacy SCR ranked it #{legacy_rank[probe]}, robust MAD-Z ranks it #{robust_rank[probe]}. The legacy 5x5-max statistic was misleading."))

    print(f"\nExecution time: {time.time() - t0:.1f}s.\n")


if __name__ == "__main__":
    main()
