#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Comprehensive Ground Truth (GT) Trajectory & Motion Analysis for wg2022_ir_020_split_03.

Detailed Diagnostics:
1. Full trajectory path inspection:
   - Frame indices where GT is present.
   - Cumulative path length (Path Length = sum(||P_{t+1} - P_t||)).
   - Max bounding box span across X and Y (Width_Span = X_max - X_min, Height_Span = Y_max - Y_min).
   - Instantaneous speed distribution (frame-to-frame velocity in pixels/frame).
   - Acceleration and turning angle distribution.
2. Segment-by-segment kinematic phase dissection (Is it looping, moving back and forth, or cruising?).
3. Visual representation of trajectory coordinates (every 25 frames).

Usage:
    python manu/analyze_wg020_03_gt_trajectory.py \
        --data-root /mnt/data/siping/datasets/manu/uav_gmc_median
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr


def natural_sort_key(path_or_str: str | Path):
    s = Path(path_or_str).stem
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze GT Trajectory for wg020_03")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median",
        help="Path to dataset root",
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

    img_files = sorted(
        [p for p in val_img_dir.glob(f"*{args.seq}*") if p.suffix.lower() in [".jpg", ".png"]],
        key=natural_sort_key,
    )

    print("\n" + "=" * 120)
    print(f"🔬 RIGOROUS GROUND TRUTH TRAJECTORY KINEMATIC AUDIT: {args.seq}")
    print(f"Total Video Frames Found: {len(img_files)}")
    print("=" * 120)

    gt_records = []
    for f_idx, p in enumerate(img_files):
        lbl_p = val_lbl_dir / f"{p.stem}.txt"
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                lines = [l.strip().split() for l in f if l.strip()]
            if lines:
                box = [float(x) for x in lines[0][1:5]]
                cx = box[0] * args.imgsz
                cy = box[1] * args.imgsz
                bw = box[2] * args.imgsz
                bh = box[3] * args.imgsz
                gt_records.append({
                    "seq_frame_idx": f_idx,
                    "stem": p.stem,
                    "pos": np.array([cx, cy], dtype=np.float32),
                    "size": np.array([bw, bh], dtype=np.float32),
                })

    total_gt = len(gt_records)
    print(f"[INFO] Frames with Valid Ground Truth: {total_gt} / {len(img_files)}")

    if total_gt == 0:
        print(colorstr("red", "[ERROR] No Ground Truth annotations found."))
        sys.exit(1)

    # 1. Spatial Coordinate Spans & Kinematic Extents
    positions = np.array([r["pos"] for r in gt_records])  # [N, 2]
    sizes = np.array([r["size"] for r in gt_records])        # [N, 2]
    frame_indices = [r["seq_frame_idx"] for r in gt_records]

    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    widths = sizes[:, 0]
    heights = sizes[:, 1]

    x_min, x_max = float(np.min(x_coords)), float(np.max(x_coords))
    y_min, y_max = float(np.min(y_coords)), float(np.max(y_coords))
    x_span = x_max - x_min
    y_span = y_max - y_min

    # Instantaneous step displacements (inter-frame velocities)
    step_disps = []
    step_times = []
    for i in range(1, total_gt):
        dt = frame_indices[i] - frame_indices[i - 1]
        dp = np.linalg.norm(positions[i] - positions[i - 1])
        step_disps.append(float(dp))
        step_times.append(dt)

    step_disps = np.array(step_disps)
    step_times = np.array(step_times)
    velocities = step_disps / np.maximum(step_times, 1)

    # Cumulative trajectory path length (Odometry)
    cumulative_path_length = float(np.sum(step_disps))
    # Direct start-to-end displacement
    net_start_end_disp = float(np.linalg.norm(positions[-1] - positions[0]))
    # Max pairwise distance across ANY two points in the trajectory
    # Sample down if total_gt is large to avoid O(N^2) memory
    sample_sub = positions[::max(1, total_gt // 200)]
    pairwise_dists = np.linalg.norm(sample_sub[:, None, :] - sample_sub[None, :, :], axis=-1)
    max_pairwise_disp = float(np.max(pairwise_dists))

    print("\n" + colorstr("bold", colorstr("cyan", "🗺️ 1. GLOBAL TRAJECTORY SPATIAL ENVELOPE:")))
    print("-" * 80)
    print(f"• Trajectory Frame Span           : Frame #{frame_indices[0]} ~ Frame #{frame_indices[-1]} ({frame_indices[-1] - frame_indices[0] + 1} frames)")
    print(f"• X Coordinate Dynamic Range       : [{x_min:6.1f}px, {x_max:6.1f}px] -> X Total Span = {colorstr('bold', colorstr('green', f'{x_span:.1f}px'))}")
    print(f"• Y Coordinate Dynamic Range       : [{y_min:6.1f}px, {y_max:6.1f}px] -> Y Total Span = {colorstr('bold', colorstr('green', f'{y_span:.1f}px'))}")
    print(f"• Maximum Bounding Extent (Diagonal): {np.sqrt(x_span**2 + y_span**2):.1f}px across canvas")
    print(f"• Maximum Pairwise Separation      : {colorstr('bold', colorstr('magenta', f'{max_pairwise_disp:.1f}px'))} (Max distance between any 2 moments)")
    print(f"• Cumulative Path Length (Odometer): {colorstr('bold', colorstr('yellow', f'{cumulative_path_length:.1f}px'))} total distance traveled!")
    print(f"• Net Start-to-End Displacement   : {net_start_end_disp:.2f}px (P_last - P_start, may be close if it flew a loop/round-trip!)")

    print("\n" + colorstr("bold", colorstr("cyan", "⚡ 2. INSTANTANEOUS VELOCITY & MOTION SPECTRUM:")))
    print("-" * 80)
    print(f"• Velocity (px/frame)              : Min={np.min(velocities):.2f}, Mean={np.mean(velocities):.2f}, Median={np.median(velocities):.2f}, Max={np.max(velocities):.2f}")
    print(f"• Frames with Speed > 1.0 px/frame : {np.sum(velocities > 1.0):>4} ({np.mean(velocities > 1.0)*100:>5.1f}%)  <-- ✈️ Actively Moving!")
    print(f"• Frames with Speed > 2.0 px/frame : {np.sum(velocities > 2.0):>4} ({np.mean(velocities > 2.0)*100:>5.1f}%)  <-- 🚀 High Speed Cruise!")
    print(f"• Frames with Speed < 0.3 px/frame : {np.sum(velocities < 0.3):>4} ({np.mean(velocities < 0.3)*100:>5.1f}%)  <-- Quasi-static moments")

    print("\n" + colorstr("bold", colorstr("cyan", "📏 3. TARGET BOUNDING BOX SIZE BREAKDOWN:")))
    print("-" * 80)
    print(f"• Bounding Box Width  (pixels)    : Min={np.min(widths):.1f}px, Mean={np.mean(widths):.1f}px, Median={np.median(widths):.1f}px, Max={np.max(widths):.1f}px")
    print(f"• Bounding Box Height (pixels)    : Min={np.min(heights):.1f}px, Mean={np.mean(heights):.1f}px, Median={np.median(heights):.1f}px, Max={np.max(heights):.1f}px")

    # 4. Trajectory Waypoints Dissection (Sample every 25 frames)
    print("\n" + colorstr("bold", colorstr("magenta", "📍 4. CHRONOLOGICAL WAYPOINTS (EVERY ~25 FRAMES):")))
    print("-" * 110)
    header = f"{'Sample #':<10} | {'Seq Frame':<12} | {'Pos (X, Y)':<18} | {'Dist to Start':<16} | {'Step Disp (v)':<16} | {'Target Box (W x H)'}"
    print(header)
    print("-" * 110)

    start_p = positions[0]
    sample_step = max(1, total_gt // 20)
    sample_sub_indices = list(range(0, total_gt, sample_step))
    if (total_gt - 1) not in sample_sub_indices:
        sample_sub_indices.append(total_gt - 1)

    for i, idx in enumerate(sample_sub_indices, 1):
        r = gt_records[idx]
        pos = r["pos"]
        dist_start = np.linalg.norm(pos - start_p)
        v_str = f"{velocities[idx-1]:.2f} px/f" if idx > 0 else "0.00 (Start)"
        box_str = f"{r['size'][0]:.1f} x {r['size'][1]:.1f} px"
        print(f"Point {i:<4} | Frame #{r['seq_frame_idx']:<6} | ({pos[0]:>6.1f}, {pos[1]:>6.1f})   | {dist_start:>11.1f} px   | {v_str:<16} | {box_str}")

    # 5. Diagnostic Qualitative Conclusion
    print("\n" + "=" * 120)
    print(colorstr("bold", colorstr("green", "🎯 AUDIT CONCLUSION ON TRAJECTORY DYNAMICS:")))
    print("-" * 120)
    if x_span > 20.0 or y_span > 20.0 or cumulative_path_length > 100.0:
        print(colorstr("bold", colorstr("green", f"✅ CONFIRMED: THE DRONE IS DEFINITELY MOVING SIGNIFICANTLY!")))
        print(f"  • Cumulative Flight Distance : {cumulative_path_length:.1f} pixels across canvas")
        print(f"  • Max Spatial Extent Span    : X Span = {x_span:.1f}px, Y Span = {y_span:.1f}px, Max Distance = {max_pairwise_disp:.1f}px")
        if net_start_end_disp < 10.0:
            print(colorstr("yellow", f"  • Note: Net Start-to-End is only {net_start_end_disp:.2f}px because the drone flew a CLOSED-LOOP / CIRCULAR / OUT-AND-BACK path!"))
            print(colorstr("yellow", f"    It returned near its starting position, which deceived simple ||P_last - P_start|| metrics!"))
    else:
        print("  • Target remains tightly localized within < 20px.")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
