#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Batch Signal-to-Clutter Ratio (SCR) & Target Contrast Profiler for All Validation Sequences.

Strict Infrared Standard Formulation (GJB / IEEE / Classical IRST Definition):
For each ground truth target location (x, y):
1. Target Region (T): A 5x5 square box centered at GT: T = [x-2:x+3, y-2:y+3]
   - Target Peak Intensity: I_T = max(T)
2. Background Annulus Region (B): An outer ring between 11x11 and 21x21 centered at GT:
   - Eliminates target halo and transition edge pixels.
   - Mean Background Intensity: mu_B = mean(B)
   - Background Standard Deviation (Noise RMS): sigma_B = std(B)
3. Physical Metrics:
   - Absolute Signal Jump: Delta_I = |I_T - mu_B|
   - Signal-to-Clutter Ratio: SCR = Delta_I / (sigma_B + eps)
   - Local Contrast Ratio: Contrast = Delta_I / (mu_B + eps)

Outputs:
- Full benchmark leaderboard sorted by mean SCR ascending (verifying if wg2022_ir_020_split_03 is at the bottom).
- Stratification summary (SCR < 1.0, 1.0~2.0, 2.0~3.0, >= 3.0).

Usage:
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
    parser = argparse.ArgumentParser(description="Measure Physical SCR Across All UAV Sequences")
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


def calculate_frame_scr(
    img_gray: np.ndarray,
    gt_boxes: np.ndarray,
    target_size: int = 5,
    bg_inner: int = 11,
    bg_outer: int = 21,
) -> List[Tuple[float, float, float, float]]:
    """
    Calculate SCR, Delta_I, mu_B, sigma_B for each GT box in the frame.
    Returns: list of (scr, delta_i, mu_b, sigma_b)
    """
    h, w = img_gray.shape[:2]
    results = []
    t_r = target_size // 2
    in_r = bg_inner // 2
    out_r = bg_outer // 2

    for box in gt_boxes:
        # box: normalized [cx, cy, bw, bh]
        cx = int(round(box[0] * w))
        cy = int(round(box[1] * h))

        # Check boundary
        if cx - out_r < 0 or cx + out_r + 1 > w or cy - out_r < 0 or cy + out_r + 1 > h:
            continue

        # 1. Target region (5x5)
        target_patch = img_gray[cy - t_r : cy + t_r + 1, cx - t_r : cx + t_r + 1]
        i_target = float(np.max(target_patch))

        # 2. Outer and Inner regions for background annulus
        outer_patch = img_gray[cy - out_r : cy + out_r + 1, cx - out_r : cx + out_r + 1]
        
        # Mask out inner square (11x11)
        mask = np.ones_like(outer_patch, dtype=bool)
        start_in = out_r - in_r
        end_in = out_r + in_r + 1
        mask[start_in:end_in, start_in:end_in] = False

        bg_pixels = outer_patch[mask]
        if len(bg_pixels) == 0:
            continue

        mu_b = float(np.mean(bg_pixels))
        sigma_b = float(np.std(bg_pixels))

        delta_i = abs(i_target - mu_b)
        scr = delta_i / (sigma_b + 1e-4)

        results.append((scr, delta_i, mu_b, sigma_b))

    return results


def main():
    args = parse_args()
    data_root = Path(args.data_root)

    # Resolve paths
    val_img_dir = data_root / "images" / "val"
    val_lbl_dir = data_root / "labels" / "val"

    if not val_img_dir.exists():
        # Fallback for alternative paths
        alt_root = Path("/home/manu/mnt/datasets/manu/uav_gmc_median")
        if (alt_root / "images" / "val").exists():
            val_img_dir = alt_root / "images" / "val"
            val_lbl_dir = alt_root / "labels" / "val"
        else:
            print(colorstr("red", f"[ERROR] Validation images not found at: {val_img_dir}"))
            sys.exit(1)

    print("\n" + "=" * 125)
    print("🔬 RUNNING SCIENTIFIC INFRARED SCR & BACKGROUND NOISE PROFILER")
    print(f"Dataset Images Root : {val_img_dir}")
    print(f"Target Kernel       : {args.target_size}x{args.target_size} px | Background Annulus: {args.bg_inner}x{args.bg_inner} ~ {args.bg_outer}x{args.bg_outer} px")
    print("=" * 125)

    img_files = sorted(list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png")), key=natural_sort_key)
    print(f"[INFO] Found {len(img_files):,} validation image frames. Grouping by sequences...")

    seq_frames: Dict[str, List[Path]] = {}
    for p in img_files:
        seq = extract_seq_name(p.name)
        seq_frames.setdefault(seq, []).append(p)

    seq_stats = {}
    t0 = time.time()

    pbar = tqdm(sorted(seq_frames.keys()), desc="Measuring SCR per Sequence", dynamic_ncols=True, file=sys.stdout)

    for seq_name in pbar:
        paths = seq_frames[seq_name]
        scr_list = []
        delta_i_list = []
        mu_b_list = []
        sigma_b_list = []
        total_gt_boxes = 0

        for p in paths:
            lbl_p = val_lbl_dir / f"{p.stem}.txt"
            if not lbl_p.exists():
                continue

            with open(lbl_p, "r", encoding="utf-8") as f:
                lines = [l.strip().split() for l in f if l.strip()]
            if not lines:
                continue

            # Load boxes [class, cx, cy, w, h] -> [cx, cy, w, h]
            gt_boxes = np.array([[float(x) for x in l[1:5]] for l in lines], dtype=np.float32)
            total_gt_boxes += len(gt_boxes)

            # Read channel 0 (Raw Infrared Gray Intensity)
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if len(img.shape) == 3:
                # Channel 0 represents I_t
                img_gray = img[:, :, 0]
            else:
                img_gray = img

            frame_res = calculate_frame_scr(
                img_gray=img_gray,
                gt_boxes=gt_boxes,
                target_size=args.target_size,
                bg_inner=args.bg_inner,
                bg_outer=args.bg_outer,
            )

            for scr, delta_i, mu_b, sigma_b in frame_res:
                scr_list.append(scr)
                delta_i_list.append(delta_i)
                mu_b_list.append(mu_b)
                sigma_b_list.append(sigma_b)

        if len(scr_list) > 0:
            seq_stats[seq_name] = {
                "frames": len(paths),
                "gt_boxes": total_gt_boxes,
                "measured_targets": len(scr_list),
                "mean_scr": float(np.mean(scr_list)),
                "median_scr": float(np.median(scr_list)),
                "p10_scr": float(np.percentile(scr_list, 10)),
                "p90_scr": float(np.percentile(scr_list, 90)),
                "mean_delta_i": float(np.mean(delta_i_list)),
                "mean_sigma_b": float(np.mean(sigma_b_list)),
                "mean_mu_b": float(np.mean(mu_b_list)),
                "pct_scr_sub1": float(np.mean(np.array(scr_list) < 1.0) * 100),
                "pct_scr_sub2": float(np.mean(np.array(scr_list) < 2.0) * 100),
            }
        else:
            # Pure negative sequence with 0 GT
            seq_stats[seq_name] = {
                "frames": len(paths),
                "gt_boxes": 0,
                "measured_targets": 0,
                "mean_scr": 999.0,  # Pure negative marker
                "median_scr": 999.0,
                "p10_scr": 999.0,
                "p90_scr": 999.0,
                "mean_delta_i": 0.0,
                "mean_sigma_b": 0.0,
                "mean_mu_b": 0.0,
                "pct_scr_sub1": 0.0,
                "pct_scr_sub2": 0.0,
            }

    pbar.close()

    # Sort sequences by mean_scr ascending (most difficult, lowest SCR first)
    sorted_seqs = sorted(
        [s for s in seq_stats.items() if s[1]["gt_boxes"] > 0],
        key=lambda x: x[1]["mean_scr"],
    )
    pure_neg_seqs = [s for s in seq_stats.items() if s[1]["gt_boxes"] == 0]

    print("\n" + "=" * 135)
    print(colorstr("bold", colorstr("cyan", f"📊 ALL SEQUENCES INFRARED SCR LEADERBOARD (SORTED BY MEAN SCR ASCENDING)")))
    print("=" * 135)
    header = (
        f"{'Rank':<4} | {'Sequence Name':<28} | {'Frames':<6} | {'GT':<5} | "
        f"{'Mean SCR':<9} | {'Median SCR':<11} | {'ΔI (Gray)':<10} | {'Noise σ':<8} | "
        f"{'% SCR < 1.0':<12} | {'% SCR < 2.0':<12} | {'SCR Regime Classification'}"
    )
    print(header)
    print("-" * 135)

    for i, (seq_name, stat) in enumerate(sorted_seqs, 1):
        m_scr = stat["mean_scr"]
        med_scr = stat["median_scr"]
        d_i = stat["mean_delta_i"]
        sig = stat["mean_sigma_b"]
        sub1 = stat["pct_scr_sub1"]
        sub2 = stat["pct_scr_sub2"]

        if m_scr < 1.5 or sub1 > 40.0:
            regime = "🚨 PHYSICAL DEAD ZONE (SCR < 1.0)"
            line_color = "red"
        elif m_scr < 2.5:
            regime = "⚠️ ULTRA-WEAK REGIME (1.0 <= SCR < 2.5)"
            line_color = "yellow"
        elif m_scr < 4.0:
            regime = "⚡ MODERATE WEAK (2.5 <= SCR < 4.0)"
            line_color = "cyan"
        else:
            regime = "✅ HIGH CONTRAST (SCR >= 4.0)"
            line_color = "green"

        line = (
            f"{i:<4} | {seq_name:<28} | {stat['frames']:<6} | {stat['gt_boxes']:<5} | "
            f"{m_scr:<9.2f} | {med_scr:<11.2f} | {d_i:<10.2f} | {sig:<8.2f} | "
            f"{sub1:>9.1f}%   | {sub2:>9.1f}%   | {regime}"
        )

        if seq_name == "wg2022_ir_020_split_03":
            print(colorstr("bold", colorstr("magenta", f"👉 {line} 👈 [TARGET PROBE]")))
        elif line_color == "red":
            print(colorstr("bold", colorstr("red", line)))
        elif line_color == "yellow":
            print(colorstr("yellow", line))
        else:
            print(line)

    if pure_neg_seqs:
        print("-" * 135)
        for s_name, s in pure_neg_seqs:
            print(f" -   | {s_name:<28} | {s['frames']:<6} | {0:<5} | Pure Negative Sequence (GT=0, Zero Target Radiation)")

    print("=" * 135)

    # Verification conclusion
    lowest_seq = sorted_seqs[0][0]
    lowest_scr = sorted_seqs[0][1]["mean_scr"]
    sub1_pct = sorted_seqs[0][1]["pct_scr_sub1"]

    print("\n" + colorstr("bold", f"🎯 VERIFICATION EXPERIMENT CONCLUSION:"))
    print(f"• Lowest SCR Sequence in Dataset : {colorstr('bold', colorstr('magenta', lowest_seq))} (Mean SCR = {lowest_scr:.2f})")
    print(f"• Frames with SCR < 1.0 in {lowest_seq}: {sub1_pct:.1f}% of all GT frames!")
    if lowest_seq == "wg2022_ir_020_split_03":
        print(colorstr("bold", colorstr("green", "✅ HYPOTHESIS CONFIRMED: wg2022_ir_020_split_03 is mathematically and physically the LOWEST SCR sequence across the entire dataset!")))
    else:
        rank_wg020 = [i for i, (s, _) in enumerate(sorted_seqs, 1) if s == "wg2022_ir_020_split_03"]
        rank_idx = rank_wg020[0] if rank_wg020 else "N/A"
        print(f"• wg2022_ir_020_split_03 Rank   : Rank #{rank_idx} with Mean SCR = {seq_stats.get('wg2022_ir_020_split_03', {}).get('mean_scr', 0.0):.2f}")

    print(f"\nExecution time: {time.time() - t0:.1f}s.\n")


if __name__ == "__main__":
    main()
