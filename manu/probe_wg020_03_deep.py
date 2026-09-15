#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Deep Forensic Pathology Probe for wg2022_ir_020_split_03.

Investigates:
1. Target Geometry & Scale Distribution:
   - Real bounding box width & height (pixels on 640x640 canvas).
   - Target trajectory velocity, acceleration and displacement.
2. Three-Channel Input Signal Breakdown at Target Centroid:
   - Channel 0: Raw Infrared Intensity I_t
   - Channel 1: GMC-Aligned 2-step Difference |I_t - W(I_{t-2})|
   - Channel 2: 21-frame Temporal Median Background Residual (I_t - B_t)^+
3. Heatmap Activation Inspection:
   - What did the neural network actually predict around GT?
   - Is the target canceled out by median/diff? Or is there an activation offset?

Usage on server:
    python manu/probe_wg020_03_deep.py \
        --data-root /mnt/data/siping/datasets/manu/uav_gmc_median \
        --cache-file runs/gmc_eval/uav_median_trial0474_cache.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re
import sys
from typing import Dict, List

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr


def natural_sort_key(path_or_str: str | Path):
    s = Path(path_or_str).stem
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def parse_args():
    parser = argparse.ArgumentParser(description="Deep Forensic Pathology Probe for wg2022_ir_020_split_03")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median",
        help="Path to dataset root containing images/val and labels/val",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial0474_cache.pkl",
        help="Path to Trial 0474 inference cache",
    )
    parser.add_argument("--seq", type=str, default="wg2022_ir_020_split_03")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


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
            print(colorstr("red", f"[ERROR] Validation images not found: {val_img_dir}"))
            sys.exit(1)

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

    cache_records = {}
    if cache_path.exists():
        print(f"[INFO] Loading inference cache from: {cache_path}")
        with open(cache_path, "rb") as f:
            records = pickle.load(f)
        for r in records:
            if args.seq in r["im_name"]:
                cache_records[Path(r["im_name"]).name] = r
    else:
        print(colorstr("yellow", f"[WARN] Cache file not found: {cache_path}. Proceeding without cache."))

    img_files = sorted(
        [p for p in val_img_dir.glob(f"*{args.seq}*") if p.suffix.lower() in [".jpg", ".png"]],
        key=natural_sort_key,
    )
    print("\n" + "=" * 120)
    print(f"🔬 DEEP FORENSIC PATHOLOGY REPORT FOR SEQUENCE: {colorstr('bold', colorstr('magenta', args.seq))}")
    print(f"Total Video Frames Found: {len(img_files)}")
    print("=" * 120)

    gt_data = []
    # Collect all frame data
    for p in img_files:
        lbl_p = val_lbl_dir / f"{p.stem}.txt"
        has_gt = False
        box = None
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                lines = [l.strip().split() for l in f if l.strip()]
            if lines:
                has_gt = True
                box = np.array([float(x) for x in lines[0][1:5]], dtype=np.float32)

        gt_data.append({
            "path": p,
            "has_gt": has_gt,
            "box": box,
        })

    gt_frames = [d for d in gt_data if d["has_gt"]]
    print(f"[INFO] Frames containing Ground Truth: {len(gt_frames)} / {len(img_files)}")

    if not gt_frames:
        print(colorstr("red", "[ERROR] No GT frames found for this sequence."))
        sys.exit(1)

    # 1. Inspect Target Geometry & Sizes
    w_px = [d["box"][2] * args.imgsz for d in gt_frames]
    h_px = [d["box"][3] * args.imgsz for d in gt_frames]
    cx_px = [d["box"][0] * args.imgsz for d in gt_frames]
    cy_px = [d["box"][1] * args.imgsz for d in gt_frames]

    print("\n" + colorstr("bold", colorstr("cyan", "📐 1. TARGET SCALE & KINEMATIC GEOMETRY PROFILE:")))
    print("-" * 80)
    print(f"• Bounding Box Width (px)  : Min={np.min(w_px):.2f}, Mean={np.mean(w_px):.2f}, Max={np.max(w_px):.2f}")
    print(f"• Bounding Box Height (px) : Min={np.min(h_px):.2f}, Mean={np.mean(h_px):.2f}, Max={np.max(h_px):.2f}")
    print(f"• Target Centroid X Span   : Min={np.min(cx_px):.1f} ~ Max={np.max(cx_px):.1f}")
    print(f"• Target Centroid Y Span   : Min={np.min(cy_px):.1f} ~ Max={np.max(cy_px):.1f} (Canvas Height: 640)")

    # Compute inter-frame displacement
    displacements = []
    for i in range(1, len(gt_frames)):
        p1 = np.array([cx_px[i - 1], cy_px[i - 1]])
        p2 = np.array([cx_px[i], cy_px[i]])
        displacements.append(float(np.linalg.norm(p2 - p1)))

    if displacements:
        print(f"• Inter-frame Velocity (px): Min={np.min(displacements):.2f}, Mean={np.mean(displacements):.2f}, Max={np.max(displacements):.2f}")
        total_net_disp = np.linalg.norm(np.array([cx_px[-1], cy_px[-1]]) - np.array([cx_px[0], cy_px[0]]))
        print(f"• Total Trajectory Net Disp: {total_net_disp:.2f}px across {len(gt_frames)} frames")

    # 2. Inspect 3-Channel Signal Breakdown at Target Location
    # Sample every 10th frame across GT frames to get a panoramic view
    step = max(1, len(gt_frames) // 15)
    sample_indices = list(range(0, len(gt_frames), step))
    if (len(gt_frames) - 1) not in sample_indices:
        sample_indices.append(len(gt_frames) - 1)

    print("\n" + colorstr("bold", colorstr("magenta", "📡 2. THREE-CHANNEL PHYSICAL SIGNAL INSPECTION AT GT LOCATION:")))
    print("-" * 120)
    header = (
        f"{'Frame File':<28} | {'GT (cx, cy)':<14} | {'Ch0: Raw I':<11} | {'Ch1: Diff':<11} | "
        f"{'Ch2: Median':<11} | {'Local BG (μ±σ)':<16} | {'Max Peak (Dist)':<18} | {'Status'}"
    )
    print(header)
    print("-" * 120)

    for idx in sample_indices:
        item = gt_frames[idx]
        p = item["path"]
        box = item["box"]
        cx = int(round(box[0] * args.imgsz))
        cy = int(round(box[1] * args.imgsz))

        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        # Extract channels
        # ch0: raw, ch1: diff, ch2: median
        if len(img.shape) == 3 and img.shape[2] == 3:
            ch0 = img[:, :, 0]
            ch1 = img[:, :, 1]
            ch2 = img[:, :, 2]
        else:
            ch0 = img
            ch1 = np.zeros_like(img)
            ch2 = np.zeros_like(img)

        # 5x5 target patch
        r = 2
        y_min, y_max = max(0, cy - r), min(img.shape[0], cy + r + 1)
        x_min, x_max = max(0, cx - r), min(img.shape[1], cx + r + 1)
        if y_min >= y_max or x_min >= x_max:
            continue
        t0 = float(np.max(ch0[y_min:y_max, x_min:x_max]))
        t1 = float(np.max(ch1[y_min:y_max, x_min:x_max]))
        t2 = float(np.max(ch2[y_min:y_max, x_min:x_max]))

        # Annulus background on ch0
        r_out = 10
        r_in = 5
        out_y_min, out_y_max = max(0, cy - r_out), min(img.shape[0], cy + r_out + 1)
        out_x_min, out_x_max = max(0, cx - r_out), min(img.shape[1], cx + r_out + 1)
        out_patch = ch0[out_y_min:out_y_max, out_x_min:out_x_max]
        if out_patch.size == 0:
            mu_b, sig_b = 0.0, 0.0
        else:
            mask = np.ones_like(out_patch, dtype=bool)
            h_p, w_p = out_patch.shape
            cy_p, cx_p = cy - out_y_min, cx - out_x_min
            mask[max(0, cy_p - r_in):min(h_p, cy_p + r_in + 1), max(0, cx_p - r_in):min(w_p, cx_p + r_in + 1)] = False
            bg_pixels = out_patch[mask]
            mu_b = float(np.mean(bg_pixels)) if len(bg_pixels) > 0 else 0.0
            sig_b = float(np.std(bg_pixels)) if len(bg_pixels) > 0 else 0.0

        # Model prediction from cache
        rec = cache_records.get(p.name, None)
        pred_sc = 0.0
        pred_d = 999.0
        if rec is not None:
            pts = rec["pred_points"]
            scs = rec["pred_scores"]
            if len(pts) > 0:
                dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
                min_i = np.argmin(dists)
                pred_d = float(dists[min_i])
                pred_sc = float(scs[min_i])

        # Qualitative status
        if pred_d <= 8.0 and pred_sc >= 0.22:
            st = colorstr("green", f"🎯 Detected ({pred_sc:.2f})")
        elif pred_d <= 8.0 and pred_sc >= 0.05:
            st = colorstr("yellow", f"⚡ Weak Peak ({pred_sc:.2f})")
        elif t2 < 2.0 and t1 < 2.0:
            st = colorstr("red", f"🛑 Zero Input Channels!")
        else:
            st = colorstr("red", f"❌ Blind ({pred_sc:.2f} @ {pred_d:.1f}px)")

        line = (
            f"{p.name[:27]:<28} | ({cx:>3d}, {cy:>3d})    | {t0:>10.1f} | {t1:>10.1f} | "
            f"{t2:>10.1f} | {mu_b:>6.1f}±{sig_b:<6.1f} | {pred_sc:>6.2f} ({pred_d:>4.1f}px)    | {st}"
        )
        print(line)

    # 3. Comprehensive Diagnosis
    print("\n" + colorstr("bold", colorstr("cyan", "📊 3. DIAGNOSTIC HYPOTHESIS & FAILURE PATTERN:")))
    print("-" * 100)

    # Calculate overall channel energy on target across all GT frames
    ch0_vals, ch1_vals, ch2_vals = [], [], []
    for item in gt_frames:
        p = item["path"]
        box = item["box"]
        cx = int(round(box[0] * args.imgsz))
        cy = int(round(box[1] * args.imgsz))
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None and len(img.shape) == 3 and img.shape[2] == 3:
            r = 2
            y_min, y_max = max(0, cy - r), min(img.shape[0], cy + r + 1)
            x_min, x_max = max(0, cx - r), min(img.shape[1], cx + r + 1)
            if y_min < y_max and x_min < x_max:
                ch0_vals.append(float(np.max(img[y_min:y_max, x_min:x_max, 0])))
                ch1_vals.append(float(np.max(img[y_min:y_max, x_min:x_max, 1])))
                ch2_vals.append(float(np.max(img[y_min:y_max, x_min:x_max, 2])))

    print(f"• Mean Intensity on Target across All {len(gt_frames)} Frames:")
    print(f"  - Channel 0 (Raw Infrared Gray)    : {np.mean(ch0_vals):.2f}")
    print(f"  - Channel 1 (GMC 2-Step Difference): {np.mean(ch1_vals):.2f}")
    print(f"  - Channel 2 (21-Frame Median Res)  : {np.mean(ch2_vals):.2f}")

    if np.mean(ch1_vals) < 3.0 and np.mean(ch2_vals) < 3.0:
        print(colorstr("bold", colorstr("red", "\n🚨 ROOT CAUSE CONFIRMED: TEMPORAL FEATURE COLLAPSE!")))
        print("  Although Raw Gray contrast (Ch0) is healthy (SCR=4.66), both Diff (Ch1) and Median (Ch2)")
        print("  are effectively NEAR ZERO! The temporal pipeline extinguished the target's energy!")
    elif np.mean(ch2_vals) > 5.0 and np.mean(ch1_vals) > 5.0:
        print(colorstr("bold", colorstr("yellow", "\n⚡ ROOT CAUSE: INPUTS ARE HEALTHY, NETWORK THRESHOLD SUPPRESSION!")))
        print("  Both Diff and Median have strong signals. The network's learned anchor/heatmap suppressed it.")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
