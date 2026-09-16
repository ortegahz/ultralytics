#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Paper-Style Multi-Panel Diagnostic Video Generator for 02_6321_0274-2773
Reference Style: DJI_0175_2_paper_diagnostic_panel.mp4

Key Features:
1. Pure Post-Processing / Instant Rendering:
   - Reads directly from SOTA inference cache (.pkl, e.g. Trial 0474 cache).
   - Generates 1280x640 dual-panel video in ~30 seconds without running heavy model forward pass.
2. Dual-Panel Paper Layout (1280x640):
   - Left Panel (640x640): True infrared image with:
     * Ground Truth (Green corner-brackets + Bbox)
     * Model Detections (Red / Yellow marks + confidence scores)
     * Inset 1 (Bottom-Left): 4x raw infrared uncompressed target zoom
     * Inset 2 (Bottom-Right): 4x CLAHE enhanced contrast zoom
   - Right Panel (640x640): High-resolution synthetic heatmap energy surface (Magma colormap)
     * Reconstructed using Gaussian point diffusion from predicted scores & positions
     * Displays target peaks and background clutter cleanly
3. Top HUD Banner:
   - Real-time frame index, instantaneous TP / FP / FN status, Bbox size, running Recall and Precision.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_sort_key(path_or_str: str | Path):
    s = Path(path_or_str).stem
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 02_6321 Paper Diagnostic Panel Video")
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
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/badcase_analysis/paper_videos",
        help="Output directory to save diagnostic video",
    )
    parser.add_argument("--seq", type=str, default="02_6321_0274-2773", help="Target sequence name")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--fps", type=float, default=25.0, help="Video framerate")
    parser.add_argument("--conf", type=float, default=0.22, help="Detection threshold")
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="Distance threshold in pixels")
    return parser.parse_args()


def draw_corner_brackets(
    img: np.ndarray,
    cx: int,
    cy: int,
    w: int,
    h: int,
    color: tuple = (0, 255, 0),
    thickness: int = 1,
    arm_len: int = 6,
):
    """Draw corner brackets around the bounding box."""
    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = int(round(cx + w / 2.0))
    y2 = int(round(cy + h / 2.0))

    x1 = max(0, min(img.shape[1] - 1, x1))
    y1 = max(0, min(img.shape[0] - 1, y1))
    x2 = max(0, min(img.shape[1] - 1, x2))
    y2 = max(0, min(img.shape[0] - 1, y2))

    arm_x = max(3, min(arm_len, (x2 - x1) // 3))
    arm_y = max(3, min(arm_len, (y2 - y1) // 3))

    # Top-Left
    cv2.line(img, (x1, y1), (x1 + arm_x, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + arm_y), color, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - arm_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + arm_y), color, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + arm_x, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - arm_y), color, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - arm_x, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - arm_y), color, thickness)

    # Center micro-crosshair
    cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, 6, 1)


def create_paper_insets(
    raw_bgr: np.ndarray,
    center_xy: tuple[int, int],
    crop_size: int = 40,
    box_w: int = 140,
    box_h: int = 140,
) -> tuple[np.ndarray, np.ndarray]:
    """Create raw zoom and CLAHE enhanced zoom insets."""
    H, W = raw_bgr.shape[:2]
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    half = crop_size // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(W, cx + half)
    y2 = min(H, cy + half)

    crop = raw_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (box_w, box_h), interpolation=cv2.INTER_NEAREST)

    # Inset 1: Raw Zoom
    inset_raw = crop.copy()
    cv2.rectangle(inset_raw, (0, 0), (box_w - 1, box_h - 1), (0, 255, 0), 1)
    cv2.putText(inset_raw, "RAW ZOOM", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

    # Inset 2: CLAHE Contrast Enhanced Zoom
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enh_gray = clahe.apply(gray)
    inset_enh = cv2.cvtColor(enh_gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(inset_enh, (0, 0), (box_w - 1, box_h - 1), (0, 255, 255), 1)
    cv2.putText(inset_enh, "CLAHE ENHANCED", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

    return inset_raw, inset_enh


def render_synthetic_heatmap(
    points: np.ndarray,
    scores: np.ndarray,
    img_h: int = 640,
    img_w: int = 640,
    radius: int = 6,
) -> np.ndarray:
    """Render a synthetic Gaussian heatmap surface for visualization from predicted points."""
    hm = np.zeros((img_h, img_w), dtype=np.float32)
    if len(points) == 0:
        hm_disp = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        return hm_disp

    y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
    kernel = np.exp(-(x_grid**2 + y_grid**2) / (2.0 * (radius / 2.5)**2))

    for (px, py), sc in zip(points, scores):
        ix = int(round(px))
        iy = int(round(py))
        if 0 <= ix < img_w and 0 <= iy < img_h:
            x1 = max(0, ix - radius)
            y1 = max(0, iy - radius)
            x2 = min(img_w, ix + radius + 1)
            y2 = min(img_h, iy + radius + 1)

            kx1 = x1 - (ix - radius)
            ky1 = y1 - (iy - radius)
            kx2 = kx1 + (x2 - x1)
            ky2 = ky1 + (y2 - y1)

            k_slice = kernel[ky1:ky2, kx1:kx2] * sc
            hm[y1:y2, x1:x2] = np.maximum(hm[y1:y2, x1:x2], k_slice)

    hm_clipped = np.clip(hm * 255.0 * 2.0, 0, 255).astype(np.uint8)
    hm_bgr = cv2.applyColorMap(hm_clipped, cv2.COLORMAP_MAGMA)
    return hm_bgr


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

    if not cache_path.exists():
        print(colorstr("red", f"[ERROR] Cache file not found: {cache_path}"))
        sys.exit(1)

    print(colorstr("bold", f"\n>>> Rendering SOTA Paper Diagnostic Panel for: {args.seq}"))
    print(f"Dataset root : {val_lbl_dir.parent}")
    print(f"Cache file   : {cache_path}")
    print(f"Threshold    : conf >= {args.conf:.2f}")

    with open(cache_path, "rb") as f:
        records = pickle.load(f)

    cache_by_stem = {}
    for r in records:
        stem = Path(r["im_name"]).stem
        if args.seq in stem:
            cache_by_stem[stem] = r

    # Collect frames
    img_files = sorted(
        [p for p in val_img_dir.glob(f"*{args.seq}*") if p.suffix.lower() in IMAGE_SUFFIXES],
        key=natural_sort_key,
    )
    if not img_files:
        print(colorstr("red", f"[ERROR] No image files found for {args.seq} in {val_img_dir}"))
        sys.exit(1)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_path = out_dir / f"{args.seq}_paper_diagnostic_panel.mp4"

    out_w = args.imgsz * 2
    out_h = args.imgsz
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(vid_path), fourcc, args.fps, (out_w, out_h))
    print(f"[INFO] Initialized VideoWriter -> {vid_path} ({out_w}x{out_h} @ {args.fps}fps)")

    total_frames = len(img_files)
    running_tp = 0
    running_fp = 0
    running_fn = 0
    running_gt = 0

    last_target_pos = (args.imgsz // 2, args.imgsz // 2)

    for f_idx, img_p in enumerate(tqdm(img_files, desc=f"Rendering {args.seq}")):
        lbl_p = val_lbl_dir / f"{img_p.stem}.txt"
        gt_boxes = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                lines = [l.strip().split() for l in f if l.strip()]
            for l in lines:
                # class cx cy w h (normalized)
                b = [float(x) for x in l[1:5]]
                gt_boxes.append([
                    b[0] * args.imgsz,
                    b[1] * args.imgsz,
                    b[2] * args.imgsz,
                    b[3] * args.imgsz,
                ])
        gt_boxes = np.array(gt_boxes, dtype=np.float32) if len(gt_boxes) > 0 else np.zeros((0, 4), dtype=np.float32)
        n_gt = len(gt_boxes)
        running_gt += n_gt

        # Read image
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            img_bgr = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
        else:
            if img_bgr.shape[0] != args.imgsz or img_bgr.shape[1] != args.imgsz:
                img_bgr = cv2.resize(img_bgr, (args.imgsz, args.imgsz))

        # Retrieve predictions
        rec = cache_by_stem.get(img_p.stem)
        if rec is None or len(rec.get("pred_points", [])) == 0:
            pred_pts = np.zeros((0, 2), dtype=np.float32)
            pred_scs = np.zeros((0,), dtype=np.float32)
        else:
            pred_pts = np.asarray(rec["pred_points"], dtype=np.float32)
            pred_scs = np.asarray(rec["pred_scores"], dtype=np.float32)

        # Filter by threshold
        valid_mask = pred_scs >= args.conf
        filt_pts = pred_pts[valid_mask]
        filt_scs = pred_scs[valid_mask]

        # Match to GT (Point-in-BBox OR dist <= dist_thresh)
        frame_tp = 0
        frame_fp = 0
        matched_gt_indices = set()
        det_status = []  # (pt, sc, is_tp)

        for p_idx, (px, py) in enumerate(filt_pts):
            sc = filt_scs[p_idx]
            is_hit = False
            for g_idx, g_box in enumerate(gt_boxes):
                if g_idx in matched_gt_indices:
                    continue
                gcx, gcy, gw, gh = g_box
                x1, y1 = gcx - gw / 2.0, gcy - gh / 2.0
                x2, y2 = gcx + gw / 2.0, gcy + gh / 2.0
                in_bbox = (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)
                dist = np.sqrt((px - gcx) ** 2 + (py - gcy) ** 2)
                if in_bbox or dist <= args.dist_thresh:
                    is_hit = True
                    matched_gt_indices.add(g_idx)
                    break

            if is_hit:
                frame_tp += 1
                det_status.append(((px, py), sc, True))
            else:
                frame_fp += 1
                det_status.append(((px, py), sc, False))

        frame_fn = n_gt - frame_tp
        running_tp += frame_tp
        running_fp += frame_fp
        running_fn += frame_fn

        # Track target position for Inset Zoom
        if n_gt > 0:
            last_target_pos = (gt_boxes[0][0], gt_boxes[0][1])
        elif len(filt_pts) > 0:
            last_target_pos = (filt_pts[0][0], filt_pts[0][1])

        # -------------------------------------------------------------
        # PANEL 1 (LEFT): True Raw Infrared Frame
        # -------------------------------------------------------------
        panel_left = img_bgr.copy()

        # Draw GT Bounding Boxes / Brackets
        for gcx, gcy, gw, gh in gt_boxes:
            draw_corner_brackets(
                panel_left,
                int(round(gcx)),
                int(round(gcy)),
                int(round(gw)),
                int(round(gh)),
                color=(0, 255, 0),
                thickness=1,
                arm_len=8,
            )
            lbl_gt = f"GT {gw:.0f}x{gh:.0f}"
            cv2.putText(panel_left, lbl_gt, (int(gcx - gw / 2), int(gcy - gh / 2) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)

        # Draw Model Detections
        for (px, py), sc, is_tp in det_status:
            ix, iy = int(round(px)), int(round(py))
            color = (0, 255, 255) if is_tp else (0, 0, 255)  # Yellow for TP hit, Red for FP false alarm
            cv2.circle(panel_left, (ix, iy), 5, color, 1)
            cv2.drawMarker(panel_left, (ix, iy), color, cv2.MARKER_CROSS, 8, 1)
            txt = f"{sc:.2f}"
            cv2.putText(panel_left, txt, (ix + 8, iy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

        # If missed, mark with FN alert
        if frame_fn > 0 and n_gt > 0:
            gcx, gcy = int(round(gt_boxes[0][0])), int(round(gt_boxes[0][1]))
            cv2.putText(panel_left, "MISSED (FN)", (gcx - 30, gcy + int(gt_boxes[0][3] / 2) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1, cv2.LINE_AA)

        # Bottom-Left Insets: Raw Zoom vs CLAHE Enhanced Zoom
        inset_raw, inset_enh = create_paper_insets(
            raw_bgr=img_bgr,
            center_xy=last_target_pos,
            crop_size=40,
            box_w=140,
            box_h=140,
        )
        iy1, iy2 = args.imgsz - 150, args.imgsz - 10
        panel_left[iy1:iy2, 10:150] = inset_raw
        panel_left[iy1:iy2, args.imgsz - 150:args.imgsz - 10] = inset_enh

        # -------------------------------------------------------------
        # PANEL 2 (RIGHT): Synthetic Heatmap Surface
        # -------------------------------------------------------------
        panel_right = render_synthetic_heatmap(pred_pts, pred_scs, img_h=args.imgsz, img_w=args.imgsz)

        # Overlay GT center on right heatmap for visual reference
        for gcx, gcy, gw, gh in gt_boxes:
            cv2.circle(panel_right, (int(round(gcx)), int(round(gcy))), 4, (0, 255, 0), 1)
            cv2.drawMarker(panel_right, (int(round(gcx)), int(round(gcy))), (0, 255, 0), cv2.MARKER_CROSS, 6, 1)

        for (px, py), sc, is_tp in det_status:
            ix, iy = int(round(px)), int(round(py))
            cv2.circle(panel_right, (ix, iy), 5, (0, 255, 255), 1)

        # -------------------------------------------------------------
        # Top HUD Banner & Stats
        # -------------------------------------------------------------
        hud_h = 36
        def add_banner(panel: np.ndarray, title: str, subtitle: str, bg_color=(20, 20, 20)):
            overlay = panel.copy()
            cv2.rectangle(overlay, (0, 0), (panel.shape[1], hud_h), bg_color, -1)
            cv2.addWeighted(overlay, 0.82, panel, 0.18, 0, panel)
            cv2.putText(panel, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
            ts = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]
            cv2.putText(panel, subtitle, (panel.shape[1] - ts[0] - 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)

        curr_rec = (running_tp / max(1, running_gt)) * 100.0
        curr_prec = (running_tp / max(1, running_tp + running_fp)) * 100.0
        status_str = f"TP:{frame_tp} FP:{frame_fp} FN:{frame_fn}"

        target_size_str = f"{gt_boxes[0][2]:.0f}x{gt_boxes[0][3]:.0f}px" if n_gt > 0 else "NO_GT"
        add_banner(
            panel_left,
            f"SOTA DETECTOR | {args.seq} | F:{f_idx:04d}/{total_frames:04d} | Size:{target_size_str}",
            f"[{status_str}] Rec:{curr_rec:.1f}% Prec:{curr_prec:.1f}%",
        )
        add_banner(
            panel_right,
            "HEATMAP ENERGY SURFACE (Trial 0474 P0-NAS S2)",
            f"th={args.conf:.2f} | CumTP:{running_tp}/{running_gt}",
        )

        combined = np.hstack([panel_left, panel_right])
        cv2.line(combined, (args.imgsz, 0), (args.imgsz, args.imgsz), (80, 80, 80), 2)
        video_writer.write(combined)

    video_writer.release()
    print(colorstr("bold", f"\n✅ Video generation complete: {vid_path.resolve()}"))
    print(f"Total Evaluated GT Frames : {running_gt}")
    print(f"Total True Positives (TP) : {running_tp} (Recall: {running_tp/running_gt*100:.2f}%)")
    print(f"Total False Alarms   (FP) : {running_fp} (Precision: {running_tp/max(1, running_tp+running_fp)*100:.2f}%)")
    print(f"Total Missed Frames  (FN) : {running_fn}")


if __name__ == "__main__":
    main()
