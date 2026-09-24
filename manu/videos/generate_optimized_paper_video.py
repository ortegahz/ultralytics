#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Paper-Style Multi-Panel Diagnostic Video Generator with Per-Sequence Parameter Optimization.

Solves the OSD Coordinate Misalignment Bug:
- The raw infrared dataset contains mixed resolutions (e.g. 640x512 and 512x512).
- The cached model predictions (pred_points) and ground truths (gt_pts) are strictly
  in the 640x640 Letterbox coordinate space (scaled with padding).
- This script uses official letterbox padding (NOT stretch resize) so image pixels,
  targets, and detection markers align with 100% pixel-perfect precision!

Layout (Dual 640x640 Panels -> 1280x640 Output Video):
-----------------------------------------------------------------------------------------------
Left Panel (640x640): Letterboxed Infrared Video + SOTA Tracked Detections
  - Ground Truth Open-Gapped Brackets (Green) with NO center obstruction
  - System SOTA Tracked Detections (Yellow for Hit TP, Red for False Alarm FP, Orange for Coasting)
  - Bottom-Left Inset: 4x Raw Infrared Uncompressed Zoom
  - Bottom-Right Inset: 4x CLAHE Contrast-Enhanced Zoom
  - Top HUD Banner: Sequence Name, Current Frame, GT Size, Running Recall & Precision

Right Panel (640x640):
  - Mode A (Default): "heatmap" - High-Resolution Synthetic Heatmap Energy Surface (Magma colormap)
  - Mode B: "features" - 3-Channel Input Decomposition (Ch0 Raw, Ch1 Motion, Ch2 Median)
-----------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import colorstr
from manu.evaluation.eval_bidirectional_track_fusion import (
    evaluate_sequence_bidirectional,
    extract_seq_name,
    natural_sort_key,
)
from manu.postprocess.optimize_per_sequence_video import optimize_sequence_params

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Paper Diagnostic Video with Sequence Parameter Optimization")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial0474_cache.pkl",
        help="Path to Trial 0474 cache file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml",
        help="Path to data.yaml (or dataset root) to locate original images and labels",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default="DJI_0175_2",
        help="Target sequence name, or 'all' to render all validation sequences",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/paper_diagnostic_videos",
        help="Directory to save output MP4 video(s)",
    )
    parser.add_argument(
        "--right-panel",
        type=str,
        default="heatmap",
        choices=["heatmap", "features"],
        help="Right panel mode: 'heatmap' (Magma surface) or 'features' (3-channel input decomposition)",
    )
    parser.add_argument(
        "--use-baseline-params",
        action="store_true",
        help="Force using Global Baseline SOTA params instead of per-sequence search",
    )
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="GJB pixel tolerance (default: 8.0px)")
    parser.add_argument("--imgsz", type=int, default=640, help="Panel dimension (640x640)")
    parser.add_argument("--fps", type=float, default=25.0, help="Video framerate")
    parser.add_argument("--no-video", action="store_true", help="Only run optimization & check metrics without writing video")
    return parser.parse_args()


def letterbox_image(image: np.ndarray, target_size: int = 640, color: tuple = (114, 114, 114)) -> np.ndarray:
    """
    Standard Letterbox transform identical to Ultralytics official DataLoader:
    Resizes image preserving aspect ratio and pads remaining borders.
    Ensures 100% pixel-perfect alignment with cache coordinates!
    """
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


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
    Target center remains 100% clean and unblocked!
    """
    H, W = img.shape[:2]
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

    arm_x = max(3, int(round((x2 - x1) * bracket_ratio)))
    arm_y = max(3, int(round((y2 - y1) * bracket_ratio)))

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
    cv2.line(img, (x2, y2), (x2 - arm_y, y2), color, thickness)


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
    if crop.size == 0 or crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        crop_clean = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        if crop.size > 0:
            ch, cw = crop.shape[:2]
            crop_clean[:ch, :cw] = crop
        crop = cv2.resize(crop_clean, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
    else:
        crop = cv2.resize(crop, (box_w, box_h), interpolation=cv2.INTER_NEAREST)

    # Inset 1: Raw Zoom
    inset_raw = crop.copy()
    cv2.rectangle(inset_raw, (0, 0), (box_w - 1, box_h - 1), (0, 255, 0), 2)
    cv2.rectangle(inset_raw, (0, 0), (box_w - 1, 18), (30, 30, 30), -1)
    cv2.putText(inset_raw, "RAW ZOOM (4x)", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)
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
    panel_size: int = 640,
) -> np.ndarray:
    """Render 3-Channel Feature Decomposition for Right Panel."""
    if raw_img.ndim == 2 or raw_img.shape[2] == 1:
        ch0 = raw_img if raw_img.ndim == 2 else raw_img[:, :, 0]
        ch1 = np.zeros_like(ch0)
        ch2 = np.zeros_like(ch0)
    else:
        ch0 = raw_img[:, :, 0]
        ch1 = raw_img[:, :, 1]
        ch2 = raw_img[:, :, 2]

    half_s = panel_size // 2
    c0_color = cv2.applyColorMap(ch0, cv2.COLORMAP_BONE)
    c0_small = cv2.resize(c0_color, (half_s, half_s))
    cv2.putText(c0_small, "Ch0: Raw IR (I_t)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    c1_color = cv2.applyColorMap(cv2.equalizeHist(ch1), cv2.COLORMAP_JET)
    c1_small = cv2.resize(c1_color, (half_s, half_s))
    cv2.putText(c1_small, "Ch1: GMC Motion Diff", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    c2_color = cv2.applyColorMap(cv2.equalizeHist(ch2), cv2.COLORMAP_INFERNO)
    c2_small = cv2.resize(c2_color, (half_s, half_s))
    cv2.putText(c2_small, "Ch2: Median Residual (I_t - B_t)+", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    comp = cv2.resize(raw_img, (half_s, half_s))
    cv2.putText(comp, "3-Channel Composite", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)

    top_row = np.hstack([c0_small, c1_small])
    bot_row = np.hstack([c2_small, comp])
    panel = np.vstack([top_row, bot_row])
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

    hm_clipped = np.clip(hm * 255.0 * 2.5, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(hm_clipped, cv2.COLORMAP_MAGMA)


def add_banner(panel: np.ndarray, title: str, subtitle: str, bg_color=(20, 20, 20)):
    hud_h = 36
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (panel.shape[1], hud_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.82, panel, 0.18, 0, panel)
    cv2.putText(panel, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    ts = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
    cv2.putText(panel, subtitle, (panel.shape[1] - ts[0] - 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)


def build_image_lookup(val_source: str | Path | list) -> dict[str, Path]:
    lookup = {}
    val_dirs = [Path(val_source)] if isinstance(val_source, (str, Path)) else [Path(p) for p in val_source]
    for d in val_dirs:
        if d.is_file():
            with open(d, "r", encoding="utf-8") as f:
                for line in f:
                    p = Path(line.strip())
                    if p.exists():
                        lookup[p.name] = p
        elif d.is_dir():
            for p in d.rglob("*.*"):
                if p.suffix.lower() in IMAGE_SUFFIXES:
                    lookup[p.name] = p
    return lookup


def generate_paper_video_for_seq(
    seq_name: str,
    seq_records: List[Dict],
    data_path: Path,
    out_dir: Path,
    right_panel_mode: str = "heatmap",
    dist_thresh: float = 8.0,
    imgsz: int = 640,
    fps: float = 25.0,
    use_baseline_params: bool = False,
    no_video: bool = False,
):
    print(colorstr("bold", f"\n========================================================================================="))
    print(colorstr("bold", f">>> Processing Sequence: {seq_name} ({len(seq_records)} frames)"))
    print(colorstr("bold", f"========================================================================================="))

    # 1. Run local parameter optimization
    t0 = time.time()
    b_metrics, best_m, best_params, best_eval_res = optimize_sequence_params(
        seq_records=seq_records,
        dist_thresh=dist_thresh,
        img_h=imgsz,
        quick_search=False,
    )
    t_search = time.time() - t0

    if use_baseline_params:
        chosen_eval_res = best_eval_res  # will re-run baseline
        chosen_params = {"th_base": 0.22, "th_salvage": 0.06, "th_ground": 0.35, "min_hits_infill": 5, "min_rigid_disp": 2.0}
        active_metrics = b_metrics
        mode_label = "GLOBAL BASELINE"
    else:
        chosen_eval_res = best_eval_res
        chosen_params = best_params
        active_metrics = best_m
        mode_label = "OPTIMIZED SOTA"

    d_f1 = best_m["f1"] - b_metrics["f1"]
    d_tp = int(best_m["tp"]) - int(b_metrics["tp"])
    d_fp = int(best_m["fp"]) - int(b_metrics["fp"])

    p_str = f"b={chosen_params['th_base']:.2f}, s={chosen_params['th_salvage']:.2f}, g={chosen_params['th_ground']:.2f}, inf={chosen_params['min_hits_infill']}"
    print(
        f"Search Complete ({t_search:.1f}s) | "
        f"Baseline F1: {b_metrics['f1']:.2f}% (R:{b_metrics['recall']:.1f}%, P:{b_metrics['precision']:.1f}%) -> "
        f"Optimized F1: {best_m['f1']:.2f}% (R:{best_m['recall']:.1f}%, P:{best_m['precision']:.1f}%) | "
        f"Gain: ΔF1={d_f1:+.2f}%, ΔTP={d_tp:+d}, ΔFP={d_fp:+d}"
    )
    print(f"Optimal Parameters: [{p_str}]")

    if no_video:
        return b_metrics, best_m, best_params

    # 2. Prepare Video Rendering
    bidi_frame_dets = chosen_eval_res["bidi_frame_dets"]
    records_sorted = chosen_eval_res["records_sorted"]

    # Image lookup for disk loading
    img_lookup = {}
    if data_path.is_file() and data_path.suffix in (".yaml", ".yml"):
        data_dict = check_det_dataset(str(data_path))
        val_source = data_dict.get("val")
        if val_source:
            img_lookup = build_image_lookup(val_source)

    if not img_lookup:
        for cand in [
            Path("/mnt/data/siping/datasets/manu/uav_gmc_median/images/val"),
            Path("/home/manu/mnt/datasets/manu/uav_gmc_median/images/val"),
            PROJECT_ROOT / "datasets/uav_gmc_median/images/val",
        ]:
            if cand.exists():
                img_lookup = build_image_lookup(cand)
                break

    out_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = out_dir / f"{seq_name}_paper_diagnostic_panel.mp4"
    out_w = imgsz * 2
    out_h = imgsz
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (out_w, out_h))
    print(f"[INFO] Rendering dual-panel video (Letterbox Aligned) -> {out_video_path}...")

    cum_tp, cum_fp, cum_gt = 0, 0, 0
    total_frames = len(records_sorted)
    last_target_pos = (imgsz // 2, imgsz // 2)

    for f_idx, r in enumerate(tqdm(records_sorted, desc=f"Rendering {seq_name}")):
        im_name = r["im_name"]
        gt_pts = np.asarray(r["gt_pts"], dtype=np.float32)
        gt_bboxes = np.asarray(r.get("gt_bboxes", np.zeros((len(gt_pts), 4))), dtype=np.float32)
        sys_dets = bidi_frame_dets[f_idx]

        # Raw image on disk
        img_raw = None
        p = img_lookup.get(im_name)
        if p is None or not p.exists():
            for ext in [".jpg", ".png", ".jpeg"]:
                alt = img_lookup.get(Path(im_name).stem + ext)
                if alt and alt.exists():
                    p = alt
                    break
        if p and p.exists():
            img_raw = cv2.imread(str(p))

        if img_raw is None:
            img_raw = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

        # -------------------------------------------------------------
        # CRITICAL FIX: Standard Letterbox Padding (NO STRETCH RESIZE!)
        # -------------------------------------------------------------
        img_letterboxed = letterbox_image(img_raw, target_size=imgsz)
        raw_ir_bgr = cv2.cvtColor(img_letterboxed[:, :, 0], cv2.COLOR_GRAY2BGR) if img_letterboxed.ndim == 3 else img_letterboxed

        # Match detections to GT in 640x640 Letterbox space
        matched_gt = set()
        matched_pred = set()
        pred_pts = np.array([d["pos"] for d in sys_dets], dtype=np.float32) if len(sys_dets) > 0 else np.zeros((0, 2), dtype=np.float32)
        scores = np.array([d["score"] for d in sys_dets], dtype=np.float32) if len(sys_dets) > 0 else np.zeros((0,), dtype=np.float32)
        infilled = np.array([d.get("infilled", False) for d in sys_dets], dtype=bool) if len(sys_dets) > 0 else np.zeros((0,), dtype=bool)
        tids = [d.get("track_id", 0) for d in sys_dets]

        if len(pred_pts) > 0 and len(gt_pts) > 0:
            diff = pred_pts[:, np.newaxis, :] - gt_pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))
            p_inds, g_inds = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
            for p_i, g_i in zip(p_inds, g_inds):
                # Point-in-BBox or dist <= dist_thresh
                gcx, gcy, gw, gh = gt_bboxes[g_i] if len(gt_bboxes) > g_i else (gt_pts[g_i][0], gt_pts[g_i][1], 0, 0)
                in_bbox = (pred_pts[p_i][0] >= gcx - gw/2) and (pred_pts[p_i][0] <= gcx + gw/2) and \
                          (pred_pts[p_i][1] >= gcy - gh/2) and (pred_pts[p_i][1] <= gcy + gh/2)
                if in_bbox or dists[p_i, g_i] <= dist_thresh:
                    if p_i not in matched_pred and g_i not in matched_gt:
                        matched_pred.add(p_i)
                        matched_gt.add(g_i)

        frame_tp = len(matched_gt)
        frame_fp = len(pred_pts) - frame_tp
        frame_fn = len(gt_pts) - frame_tp
        cum_tp += frame_tp
        cum_fp += frame_fp
        cum_gt += len(gt_pts)

        if len(gt_pts) > 0:
            last_target_pos = (gt_pts[0][0], gt_pts[0][1])
        elif len(pred_pts) > 0:
            last_target_pos = (pred_pts[0][0], pred_pts[0][1])

        # -------------------------------------------------------------
        # PANEL 1 (LEFT): True Raw Infrared Frame
        # -------------------------------------------------------------
        panel_left = raw_ir_bgr.copy()

        # 1. Draw GT with Open-Gapped Brackets (100% clean center!)
        gt_size_str = "No GT"
        for g_i, (gx, gy) in enumerate(gt_pts):
            gw, gh = gt_bboxes[g_i, 2:4] if len(gt_bboxes) > g_i else (4.0, 4.0)
            is_hit = g_i in matched_gt
            color = (0, 255, 0) if is_hit else (0, 165, 255)  # Green for Hit, Orange for Missed

            draw_corner_brackets(
                panel_left,
                int(round(gx)),
                int(round(gy)),
                w=int(round(gw)),
                h=int(round(gh)),
                color=color,
                thickness=1,
                min_box_size=24,
                bracket_ratio=0.25,
            )
            gt_size_str = f"GT: {gw:.1f}x{gh:.1f}px"
            tag_str = f"{gw:.0f}x{gh:.0f}" if is_hit else f"{gw:.0f}x{gh:.0f}[FN]"
            text_y = max(16, int(round(gy - max(gh, 24.0)/2.0)) - 6)
            cv2.putText(panel_left, tag_str, (max(4, int(round(gx - 18))), text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        # 2. Draw Model Detections
        for p_i, ((px, py), sc, inf, tid) in enumerate(zip(pred_pts, scores, infilled, tids)):
            ix, iy = int(round(px)), int(round(py))
            is_tp = p_i in matched_pred
            color = (0, 255, 255) if is_tp else (0, 0, 255)  # Yellow for Hit, Red for False Alarm
            if inf:
                color = (255, 180, 0)  # Cyan/Orange for infill
            cv2.circle(panel_left, (ix, iy), 4, color, 1)
            cv2.drawMarker(panel_left, (ix, iy), color, cv2.MARKER_CROSS, 8, 1)
            sc_str = "Inf" if inf else f"{sc:.2f}"
            cv2.putText(panel_left, f"T{tid}:{sc_str}", (ix + 6, iy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        # 3. Bottom Insets: 4x Raw Zoom and 4x CLAHE Enhanced Zoom
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

        # -------------------------------------------------------------
        # PANEL 2 (RIGHT): Heatmap Surface or Feature Decomposition
        # -------------------------------------------------------------
        if right_panel_mode == "features":
            panel_right = render_feature_panel(img_letterboxed, panel_size=imgsz)
            right_title = "FEATURE DECOMPOSITION (Ch0/Ch1/Ch2)"
            right_sub = "Raw IR | Motion GMC | Median Residual"
        else:
            raw_pred_pts = np.asarray(r.get("pred_points", []), dtype=np.float32)
            raw_pred_scs = np.asarray(r.get("pred_scores", []), dtype=np.float32)
            panel_right = render_synthetic_heatmap(raw_pred_pts, raw_pred_scs, img_h=imgsz, img_w=imgsz)
            for gx, gy in gt_pts:
                cv2.circle(panel_right, (int(round(gx)), int(round(gy))), 4, (0, 255, 0), 1)
            for (px, py), _, _, _ in zip(pred_pts, scores, infilled, tids):
                cv2.circle(panel_right, (int(round(px)), int(round(py))), 4, (0, 255, 255), 1)
            right_title = "HEATMAP ENERGY SURFACE (Trial 0474 P0-NAS)"
            right_sub = f"Peaks: {len(pred_pts)} | Conf: {chosen_params['th_base']:.2f}"

        # -------------------------------------------------------------
        # HUD Banners
        # -------------------------------------------------------------
        cur_rec = cum_tp / max(1, cum_gt) * 100.0
        cur_prec = cum_tp / max(1, cum_tp + cum_fp) * 100.0
        cur_f1 = 2 * cur_rec * cur_prec / max(1e-6, cur_rec + cur_prec)

        add_banner(
            panel_left,
            f"INFRARED SOTA | {seq_name} | F:{f_idx:04d}/{total_frames:04d} | {gt_size_str}",
            f"[{mode_label}] Rec:{cur_rec:.1f}% Prec:{cur_prec:.1f}% F1:{cur_f1:.2f}",
        )
        add_banner(
            panel_right,
            right_title,
            f"Params: [{p_str}]",
        )

        combined = np.hstack([panel_left, panel_right])
        cv2.line(combined, (imgsz, 0), (imgsz, imgsz), (80, 80, 80), 2)
        video_writer.write(combined)

    video_writer.release()
    print(colorstr("green", f"✅ [SAVED VIDEO] -> {out_video_path.resolve()}\n"))
    return b_metrics, best_m, best_params


def main():
    args = parse_args()
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
        print(colorstr("red", f"[ERROR] Cache file not found: {args.cache_file}"))
        sys.exit(1)

    print(colorstr("bold", f"\n>>> Loading cached inferences from: {cache_path}"))
    with open(cache_path, "rb") as f:
        records = pickle.load(f)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    label_dir = data_path.parent / "labels" / "val" if data_path.suffix in (".yaml", ".yml") else data_path / "labels" / "val"
    if not label_dir.exists():
        label_dir = Path("/mnt/data/siping/datasets/manu/uav_gmc_median/labels/val")

    # Enrich gt_bboxes for Point-in-BBox evaluation
    for record in records:
        if "gt_bboxes" not in record:
            boxes = []
            label_path = label_dir / f"{Path(record['im_name']).stem}.txt"
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) >= 5:
                        _, cx, cy, width, height = map(float, parts[:5])
                        boxes.append([cx * 640.0, cy * 640.0, width * 640.0, height * 640.0])
            record["gt_bboxes"] = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    seq_records: Dict[str, List[Dict]] = {}
    for r in records:
        seq = extract_seq_name(r["im_name"])
        seq_records.setdefault(seq, []).append(r)

    if args.seq == "all":
        seqs_to_process = sorted(seq_records.keys(), key=natural_sort_key)
    else:
        seqs_to_process = [args.seq] if args.seq in seq_records else []

    if not seqs_to_process:
        print(colorstr("red", f"[ERROR] Sequence '{args.seq}' not found in cache!"))
        sys.exit(1)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    for s in seqs_to_process:
        generate_paper_video_for_seq(
            seq_name=s,
            seq_records=seq_records[s],
            data_path=data_path,
            out_dir=out_dir,
            right_panel_mode=args.right_panel,
            dist_thresh=args.dist_thresh,
            imgsz=args.imgsz,
            fps=args.fps,
            use_baseline_params=args.use_baseline_params,
            no_video=args.no_video,
        )


if __name__ == "__main__":
    main()
