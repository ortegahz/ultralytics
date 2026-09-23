#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Generate Multi-Panel Diagnostic Videos for Academic Collaboration & Problem Demonstration.

Layout (Dual 640x640 Panels -> 1280x640 Output Video):
-----------------------------------------------------------------------------------------------
Left Panel (640x640): Raw Infrared Video
  - Ground Truth Corner Brackets & True Bbox (Green)
  - Target Pixel Size OSD (e.g. GT: 2.1x2.3 px, Area: 4.8 px^2)
  - Bottom-Left Inset: 4x Raw Uncompressed Infrared Zoom
  - Bottom-Right Inset: 4x CLAHE Contrast Enhanced Zoom
  - Top HUD Banner: Sequence Name, Current Frame, GT Size, Status

Right Panel (640x640): Physical / Feature Inspection (Configurable)
  Mode A: "features" (Default) - Displays the 3-Channel Input Features or GMC Temporal Difference
          * Sub-split or highlighted view: Ch0 (Raw IR), Ch1 (GMC Diff), Ch2 (Temporal Median Residual)
  Mode B: "heatmap"  - Displays Heatmap Energy Surface reconstructed from predicted peaks/scores
-----------------------------------------------------------------------------------------------

Target Hard Cases:
1. wg2022_ir_020_split_03: Sub-noise weak target (medEnergy=40, ~3.3 sigma vs 5.4 sigma noise)
2. DJI_0051_2            : Dim small target over complex building/parallax background
3. wg2022_ir_011_split_03: Cold sky weak impulse, intermittent cloud edge clutter
4. 02_6321_0274-2773     : Scale variation (10px hovering -> 100px roof-edge overlap)
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
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.eval_bidirectional_track_fusion import match_predictions_to_gt

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

TARGET_HARD_CASES = [
    "wg2022_ir_020_split_03",
    "DJI_0051_2",
    "wg2022_ir_011_split_03",
    "02_6321_0274-2773",
]


def natural_sort_key(path_or_str: str | Path):
    s = Path(path_or_str).stem
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Collaboration Diagnostic Videos for Hard Cases")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median",
        help="Path to dataset root (containing images/<split> and labels/<split>)",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial0474_cache.pkl",
        help="Path to prediction cache (optional, used if right-panel is heatmap or to show model detections)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/collaboration_videos",
        help="Directory to save output mp4 videos",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default="",
        help="Sequence name or filename prefix; empty means all sequences in the dataset",
    )
    parser.add_argument(
        "--right-panel",
        type=str,
        default="features",
        choices=["features", "heatmap"],
        help="Content for right panel: 'features' (3-channel input decomposition) or 'heatmap' (energy surface)",
    )
    parser.add_argument(
        "--show-dets",
        action="store_true",
        default=False,
        help="Whether to overlay algorithm detection markers on the video",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Video panel resolution (640x640)")
    parser.add_argument("--fps", type=float, default=25.0, help="Video framerate")
    parser.add_argument("--conf", type=float, default=0.22, help="Fallback detection threshold")
    parser.add_argument("--search-threshold", action="store_true", help="Search the best threshold independently for each sequence")
    parser.add_argument("--th-min", type=float, default=0.05, help="Minimum threshold for per-sequence search")
    parser.add_argument("--th-max", type=float, default=0.50, help="Maximum threshold for per-sequence search")
    parser.add_argument("--th-step", type=float, default=0.01, help="Threshold step for per-sequence search")
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="Matching distance in letterbox pixels")
    return parser.parse_args()


def draw_corner_brackets(
    img: np.ndarray,
    cx: int,
    cy: int,
    w: int,
    h: int,
    color: tuple = (0, 255, 0),
    thickness: int = 1,
    min_box_size: int = 24,
    bracket_ratio: float = 0.25,
):
    """
    Draw an open/gapped bounding bracket box with a clear center gap.
    Leaves the center completely unobstructed so small targets/pixels remain 100% visible.
    
    Structure:
      x1,y1 ───            ─── x2,y1
        │                        │
        
        │                        │
      x1,y2 ───            ─── x2,y2
    Center area has NO crosshairs or solid lines.
    """
    H, W = img.shape[:2]

    # Ensure box has sufficient visual margin around tiny 1~2px targets
    box_w = max(float(w), float(min_box_size))
    box_h = max(float(h), float(min_box_size))

    x1 = int(round(cx - box_w / 2.0))
    y1 = int(round(cy - box_h / 2.0))
    x2 = int(round(cx + box_w / 2.0))
    y2 = int(round(cy + box_h / 2.0))

    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))

    # Corner segment length: 25% of box side, leaving 50% open center gap
    arm_x = max(3, int(round((x2 - x1) * bracket_ratio)))
    arm_y = max(3, int(round((y2 - y1) * bracket_ratio)))

    # 1. Top-Left corner
    cv2.line(img, (x1, y1), (x1 + arm_x, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + arm_y), color, thickness)

    # 2. Top-Right corner
    cv2.line(img, (x2, y1), (x2 - arm_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + arm_y), color, thickness)

    # 3. Bottom-Left corner
    cv2.line(img, (x1, y2), (x1 + arm_x, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - arm_y), color, thickness)

    # 4. Bottom-Right corner
    cv2.line(img, (x2, y2), (x2 - arm_x, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - arm_y), color, thickness)

    # NOTE: DO NOT draw center crosshair or dot to prevent covering delicate infrared pixels


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
    cv2.rectangle(inset_raw, (0, 0), (box_w - 1, box_h - 1), (0, 255, 0), 2)
    cv2.rectangle(inset_raw, (0, 0), (box_w - 1, 18), (30, 30, 30), -1)
    cv2.putText(inset_raw, "RAW ZOOM (4x)", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

    # Corner brackets in inset with large center opening to keep target completely unblocked
    pcx, pcy = box_w // 2, box_h // 2
    draw_corner_brackets(inset_raw, pcx, pcy, w=28, h=28, color=(0, 255, 0), thickness=1, min_box_size=24, bracket_ratio=0.25)

    # Inset 2: CLAHE Contrast Enhanced Zoom
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
    enh_gray = clahe.apply(gray)
    inset_enh = cv2.cvtColor(enh_gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(inset_enh, (0, 0), (box_w - 1, box_h - 1), (0, 255, 255), 2)
    cv2.rectangle(inset_enh, (0, 0), (box_w - 1, 18), (30, 30, 30), -1)
    cv2.putText(inset_enh, "CLAHE ENHANCED", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
    draw_corner_brackets(inset_enh, pcx, pcy, w=28, h=28, color=(0, 255, 255), thickness=1, min_box_size=24, bracket_ratio=0.25)

    return inset_raw, inset_enh


def render_feature_panel(
    raw_img: np.ndarray,
    target_pos: tuple[int, int],
    panel_size: int = 640,
) -> np.ndarray:
    """
    Render 3-Channel Feature Decomposition for Right Panel:
    Top: Ch0 Raw IR (normalized)
    Bottom-Left: Ch1 GMC / Frame Difference
    Bottom-Right: Ch2 Temporal Median Residual
    """
    # raw_img is 3-channel BGR from uav_gmc_median:
    # Channel B = I_t (Raw IR)
    # Channel G = |I_t - W(I_{t-2})| (GMC Difference)
    # Channel R = (I_t - B_t)^+ (Temporal Median Residual)
    if raw_img.ndim == 2 or raw_img.shape[2] == 1:
        ch0 = raw_img if raw_img.ndim == 2 else raw_img[:, :, 0]
        ch1 = np.zeros_like(ch0)
        ch2 = np.zeros_like(ch0)
    else:
        ch0 = raw_img[:, :, 0]
        ch1 = raw_img[:, :, 1]
        ch2 = raw_img[:, :, 2]

    half_s = panel_size // 2

    # Top-Left: Ch0 Raw IR with Jet/Magma colormap
    c0_color = cv2.applyColorMap(ch0, cv2.COLORMAP_BONE)
    c0_small = cv2.resize(c0_color, (half_s, half_s))
    cv2.putText(c0_small, "Ch0: Raw IR (I_t)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    # Top-Right: Ch1 Motion Residual |I_t - W(I_{t-2})| with JET
    c1_color = cv2.applyColorMap(cv2.equalizeHist(ch1), cv2.COLORMAP_JET)
    c1_small = cv2.resize(c1_color, (half_s, half_s))
    cv2.putText(c1_small, "Ch1: GMC Motion Diff |I_t - W(I_t-2)|", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    # Bottom-Left: Ch2 Background Subtraction (I_t - B_t)^+ with INFERNO
    c2_color = cv2.applyColorMap(cv2.equalizeHist(ch2), cv2.COLORMAP_INFERNO)
    c2_small = cv2.resize(c2_color, (half_s, half_s))
    cv2.putText(c2_small, "Ch2: Median Residual (I_t - B_t)+", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    # Bottom-Right: 3-Channel Composite RGB preview
    comp = cv2.resize(raw_img, (half_s, half_s))
    cv2.putText(comp, "3-Channel Composite", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)

    top_row = np.hstack([c0_small, c1_small])
    bot_row = np.hstack([c2_small, comp])
    panel = np.vstack([top_row, bot_row])

    # Draw grid dividers
    cv2.line(panel, (half_s, 0), (half_s, panel_size), (60, 60, 60), 2)
    cv2.line(panel, (0, half_s), (panel_size, half_s), (60, 60, 60), 2)

    return panel


def render_synthetic_heatmap(
    points: np.ndarray,
    scores: np.ndarray,
    img_h: int = 640,
    img_w: int = 640,
    radius: int = 6,
) -> np.ndarray:
    """Render a synthetic Gaussian heatmap surface from prediction points."""
    hm = np.zeros((img_h, img_w), dtype=np.float32)
    if len(points) == 0:
        return np.zeros((img_h, img_w, 3), dtype=np.uint8)

    y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
    kernel = np.exp(-(x_grid**2 + y_grid**2) / (2.0 * (radius / 2.5)**2))

    for (px, py), sc in zip(points, scores):
        ix, iy = int(round(px)), int(round(py))
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
    return cv2.applyColorMap(hm_clipped, cv2.COLORMAP_MAGMA)


def add_banner(panel: np.ndarray, title: str, subtitle: str, bg_color=(25, 25, 25)):
    hud_h = 36
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (panel.shape[1], hud_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.85, panel, 0.15, 0, panel)
    cv2.putText(panel, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    ts = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
    cv2.putText(panel, subtitle, (panel.shape[1] - ts[0] - 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)


def evaluate_threshold(records: list[dict], threshold: float, dist_thresh: float) -> dict:
    stats = {"tp": 0, "fp": 0, "gt": 0}
    for record in records:
        gt_pts = np.asarray(record.get("gt_pts", []), dtype=np.float32)
        gt_bboxes = np.asarray(record.get("gt_bboxes", []), dtype=np.float32)
        points = np.asarray(record.get("pred_points", []), dtype=np.float32)
        scores = np.asarray(record.get("pred_scores", []), dtype=np.float32)
        points = points[scores >= threshold]
        tp, fp, _ = match_predictions_to_gt(gt_pts, points, dist_thresh, gt_bboxes=gt_bboxes, match_mode="bbox")
        stats["tp"] += tp
        stats["fp"] += fp
        stats["gt"] += len(gt_pts)
    stats["recall"] = 100.0 * stats["tp"] / max(1, stats["gt"])
    stats["precision"] = 100.0 * stats["tp"] / max(1, stats["tp"] + stats["fp"])
    stats["f1"] = 2.0 * stats["recall"] * stats["precision"] / max(1e-6, stats["recall"] + stats["precision"])
    return stats


def generate_sequence_video(
    seq_name: str,
    data_root: Path,
    out_dir: Path,
    cache_by_stem: Dict,
    right_panel_mode: str = "features",
    show_dets: bool = False,
    imgsz: int = 640,
    fps: float = 25.0,
    conf_thresh: float = 0.22,
    search_threshold: bool = False,
    th_min: float = 0.05,
    th_max: float = 0.50,
    th_step: float = 0.01,
    dist_thresh: float = 8.0,
    split: str = "val",
):
    val_img_dir = data_root / "images" / split
    val_lbl_dir = data_root / "labels" / split

    img_files = sorted(
        [p for p in val_img_dir.glob(f"*{seq_name}*") if p.suffix.lower() in IMAGE_SUFFIXES],
        key=natural_sort_key,
    )

    if not img_files:
        print(f"[WARN] No image files found for {seq_name} in {val_img_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if search_threshold and cache_by_stem:
        sequence_records = [record for stem, record in cache_by_stem.items() if stem.startswith(f"{seq_name}__")]
        thresholds = np.arange(th_min, th_max + th_step * 0.5, th_step)
        threshold_results = [(float(threshold), evaluate_threshold(sequence_records, float(threshold), dist_thresh)) for threshold in thresholds]
        conf_thresh, best_stats = max(threshold_results, key=lambda item: (item[1]["f1"], item[1]["recall"], item[1]["precision"]))
        print(
            f"[THRESHOLD] {seq_name}: th={conf_thresh:.2f} TP={best_stats['tp']} FP={best_stats['fp']} "
            f"GT={best_stats['gt']} R={best_stats['recall']:.2f}% P={best_stats['precision']:.2f}% F1={best_stats['f1']:.4f}"
        )
    else:
        best_stats = None
    vid_path = out_dir / f"{seq_name}_diagnostic_demo_th{conf_thresh:.2f}.mp4"

    out_w = imgsz * 2
    out_h = imgsz
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(vid_path), fourcc, fps, (out_w, out_h))
    print(f"\n[INFO] Rendering: {seq_name} ({len(img_files)} frames) -> {vid_path}")

    total_gt = 0
    total_frames = len(img_files)
    last_target_pos = (imgsz // 2, imgsz // 2)

    for f_idx, img_p in enumerate(tqdm(img_files, desc=f"Video: {seq_name}")):
        lbl_p = val_lbl_dir / f"{img_p.stem}.txt"
        source_img = cv2.imread(str(img_p))
        source_h, source_w = source_img.shape[:2] if source_img is not None else (imgsz, imgsz)
        gt_boxes = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                lines = [l.strip().split() for l in f if l.strip()]
            for l in lines:
                # class cx cy w h (normalized)
                b = [float(x) for x in l[1:5]]
                gt_boxes.append([
                    b[0] * source_w,
                    b[1] * source_h,
                    b[2] * source_w,
                    b[3] * source_h,
                ])
        gt_boxes = np.array(gt_boxes, dtype=np.float32) if len(gt_boxes) > 0 else np.zeros((0, 4), dtype=np.float32)
        n_gt = len(gt_boxes)
        total_gt += n_gt

        # Read image
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            img_bgr = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        else:
            orig_h, orig_w = img_bgr.shape[:2]
            scale = min(imgsz / orig_w, imgsz / orig_h)
            resized_w, resized_h = round(orig_w * scale), round(orig_h * scale)
            img_bgr = cv2.resize(img_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
            pad_x = (imgsz - resized_w) // 2
            pad_y = (imgsz - resized_h) // 2
            img_bgr = cv2.copyMakeBorder(
                img_bgr,
                pad_y,
                imgsz - resized_h - pad_y,
                pad_x,
                imgsz - resized_w - pad_x,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
            gt_boxes[:, 0] = gt_boxes[:, 0] * scale + pad_x
            gt_boxes[:, 1] = gt_boxes[:, 1] * scale + pad_y
            gt_boxes[:, 2:] *= scale

        # Channel 0 is raw IR
        raw_ir_bgr = cv2.cvtColor(img_bgr[:, :, 0], cv2.COLOR_GRAY2BGR)

        # Retrieve prediction cache if needed
        rec = cache_by_stem.get(img_p.stem, {})
        pred_pts = np.asarray(rec.get("pred_points", []), dtype=np.float32)
        pred_scs = np.asarray(rec.get("pred_scores", []), dtype=np.float32)

        if n_gt > 0:
            last_target_pos = (gt_boxes[0][0], gt_boxes[0][1])
        elif len(pred_pts) > 0:
            last_target_pos = (pred_pts[0][0], pred_pts[0][1])

        # -----------------------------------------------------------------
        # Panel 1 (Left): True Raw Infrared Frame
        # -----------------------------------------------------------------
        panel_left = raw_ir_bgr.copy()

        gt_size_str = "No GT"
        for gcx, gcy, gw, gh in gt_boxes:
            draw_corner_brackets(
                panel_left,
                int(round(gcx)),
                int(round(gcy)),
                int(round(gw)),
                int(round(gh)),
                color=(0, 255, 0),
                thickness=1,
                min_box_size=24,
                bracket_ratio=0.25,
            )
            area_px = gw * gh
            gt_size_str = f"GT: {gw:.1f}x{gh:.1f}px (Area: {area_px:.1f}px^2)"
            # Put text safely above the open box with offset so it does not block the target
            box_h_used = max(float(gh), 24.0)
            text_y = max(16, int(round(gcy - box_h_used / 2.0)) - 6)
            text_x = max(4, int(round(gcx - 20)))
            cv2.putText(panel_left, f"{gw:.0f}x{gh:.0f}px", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)

        if show_dets and len(pred_pts) > 0:
            mask = pred_scs >= conf_thresh
            for (px, py), sc in zip(pred_pts[mask], pred_scs[mask]):
                ix, iy = int(round(px)), int(round(py))
                cv2.circle(panel_left, (ix, iy), 5, (0, 255, 255), 1)
                cv2.drawMarker(panel_left, (ix, iy), (0, 255, 255), cv2.MARKER_CROSS, 8, 1)
                cv2.putText(panel_left, f"{sc:.2f}", (ix + 8, iy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

        # Inset Zoom (Bottom Left & Right)
        inset_raw, inset_enh = create_paper_insets(
            raw_bgr=raw_ir_bgr,
            center_xy=last_target_pos,
            crop_size=40,
            box_w=140,
            box_h=140,
        )
        iy1, iy2 = imgsz - 150, imgsz - 10
        panel_left[iy1:iy2, 10:150] = inset_raw
        panel_left[iy1:iy2, imgsz - 150:imgsz - 10] = inset_enh

        # -----------------------------------------------------------------
        # Panel 2 (Right): Feature Decomposition or Synthetic Heatmap
        # -----------------------------------------------------------------
        if right_panel_mode == "features":
            panel_right = render_feature_panel(img_bgr, last_target_pos, panel_size=imgsz)
            right_title = "SPATIO-TEMPORAL INPUT FEATURES (Ch0, Ch1, Ch2)"
            right_sub = "Ch0:IR | Ch1:Motion | Ch2:Median"
        else:
            panel_right = render_synthetic_heatmap(pred_pts, pred_scs, img_h=imgsz, img_w=imgsz)
            # overlay GT center
            for gcx, gcy, gw, gh in gt_boxes:
                cv2.circle(panel_right, (int(round(gcx)), int(round(gcy))), 4, (0, 255, 0), 1)
            right_title = "HEATMAP ENERGY SURFACE"
            right_sub = f"Pred Peaks: {len(pred_pts)}"

        # -----------------------------------------------------------------
        # HUD Banners
        # -----------------------------------------------------------------
        add_banner(
            panel_left,
            f"INFRARED RAW | {seq_name} | Frame {f_idx:04d}/{total_frames:04d}",
            f"{gt_size_str} | th={conf_thresh:.2f}",
        )
        add_banner(
            panel_right,
            right_title,
            right_sub,
        )

        combined = np.hstack([panel_left, panel_right])
        cv2.line(combined, (imgsz, 0), (imgsz, imgsz), (80, 80, 80), 2)
        video_writer.write(combined)

    video_writer.release()
    print(f"[OK] Saved: {vid_path.resolve()}")


def main():
    args = parse_args()
    data_root = Path(args.data_root)

    # Auto fallback for remote or local mounts
    if not (data_root / "images" / args.split).exists():
        for cand in [
            Path("/mnt/data/siping/datasets/manu/uav_gmc_median"),
            Path("/home/manu/mnt/datasets/manu/uav_gmc_median"),
            PROJECT_ROOT / "datasets/uav_gmc_median",
        ]:
            if (cand / "images" / args.split).exists():
                data_root = cand
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

    cache_by_stem = {}
    if cache_path.exists():
        print(f"[INFO] Loading prediction cache: {cache_path}")
        with open(cache_path, "rb") as f:
            records = pickle.load(f)
        for r in records:
            stem = Path(r["im_name"]).stem
            cache_by_stem[stem] = r
    else:
        print(f"[INFO] Prediction cache not found or omitted. Proceeding in pure feature demo mode.")

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    if args.seq:
        seqs_to_run = [args.seq]
    else:
        seqs_to_run = sorted(
            {re.sub(r"___seg\d+$", "", Path(p).stem.split("__", 1)[0]) for p in cache_by_stem}
            or {"longquanshan_ir_1"},
            key=natural_sort_key,
        )

    for s in seqs_to_run:
        generate_sequence_video(
            seq_name=s,
            data_root=data_root,
            out_dir=out_dir,
            cache_by_stem=cache_by_stem,
            right_panel_mode=args.right_panel,
            show_dets=args.show_dets,
            imgsz=args.imgsz,
            fps=args.fps,
            conf_thresh=args.conf,
            search_threshold=args.search_threshold,
            th_min=args.th_min,
            th_max=args.th_max,
            th_step=args.th_step,
            dist_thresh=args.dist_thresh,
            split=args.split,
        )


if __name__ == "__main__":
    main()
