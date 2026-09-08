#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Paper-Style Multi-Panel Diagnostic Video Generator for Weak Infrared UAVs.
Uses TRUE RAW INFRARED VIDEO FRAMES from /mnt/data/siping/datasets/manu/anti-uav
combined with official dataset predictions and Kinematic Tracker.

Demonstrates to leadership:
1. Left Panel (640x640): Real raw infrared frame with full contrast, natural background (water, land, sky),
   GT targeting brackets, Kalman smooth trajectory, and Coasting status.
2. Right Panel (640x640): High-resolution predicted heatmap energy surface (Magma/Jet colormap).
3. Bottom-Left Inset: Raw uncompressed local patch zoom.
4. Bottom-Right Inset: CLAHE enhanced local patch zoom to show the extremely faint target dot.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG, colorstr
from manu.heatmap_model import YOLO26HeatmapDetector
from manu.heatmap_evaluate import extract_peaks


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_sort_key(path: Path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.stem)]


def find_sequence_dir(raw_root: Path, seq_name: str) -> Path | None:
    direct_path = raw_root / seq_name
    if direct_path.is_dir():
        return direct_path
    for path in raw_root.rglob(seq_name):
        if path.is_dir():
            return path
    return None


class PointKalmanTrack:
    _count = 0

    def __init__(self, init_pos: np.ndarray, score: float):
        PointKalmanTrack._count += 1
        self.track_id = PointKalmanTrack._count
        self.x = np.array([init_pos[0], init_pos[1], 0.0, 0.0], dtype=np.float32)
        self.P = np.diag([10.0, 10.0, 100.0, 100.0]).astype(np.float32)
        self.Q = np.diag([1.0, 1.0, 4.0, 4.0]).astype(np.float32)
        self.R = np.diag([4.0, 4.0]).astype(np.float32)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.score = score
        self.history = [self.get_pos()]

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.get_pos()

    def update(self, pos: np.ndarray, score: float):
        self.time_since_update = 0
        self.hits += 1
        self.score = 0.7 * self.score + 0.3 * score

        y = pos - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        self.history.append(self.get_pos())
        if len(self.history) > 40:
            self.history.pop(0)

    def get_pos(self) -> np.ndarray:
        return self.x[:2].copy()


class PointTracker:
    def __init__(self, max_age: int = 4, min_hits: int = 4, match_dist: float = 12.0):
        self.max_age = max_age
        self.min_hits = min_hits
        self.match_dist = match_dist
        self.tracks: list[PointKalmanTrack] = []

    def update(self, detections: np.ndarray, scores: np.ndarray) -> list[dict]:
        for t in self.tracks:
            t.predict()

        matched_tracks, matched_dets = set(), set()
        if len(self.tracks) > 0 and len(detections) > 0:
            track_positions = np.array([t.get_pos() for t in self.tracks])
            dists = np.linalg.norm(track_positions[:, None, :] - detections[None, :, :], axis=-1)

            row_ind, col_ind = linear_sum_assignment(dists)
            for r, c in zip(row_ind, col_ind):
                if dists[r, c] <= self.match_dist:
                    self.tracks[r].update(detections[c], float(scores[c]))
                    matched_tracks.add(r)
                    matched_dets.add(c)

        for i in range(len(detections)):
            if i not in matched_dets:
                self.tracks.append(PointKalmanTrack(detections[i], float(scores[i])))

        active_outputs = []
        surviving_tracks = []
        for t in self.tracks:
            if t.time_since_update <= self.max_age:
                surviving_tracks.append(t)
                if t.hits >= self.min_hits:
                    active_outputs.append({
                        "id": t.track_id,
                        "pos": t.get_pos(),
                        "score": t.score,
                        "is_coasting": t.time_since_update > 0,
                        "history": list(t.history),
                    })
        self.tracks = surviving_tracks
        return active_outputs


def draw_corner_brackets(img: np.ndarray, cx: int, cy: int, size: int = 24, arm: int = 6, color=(0, 255, 0), thickness: int = 1):
    half = size // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half
    H, W = img.shape[:2]
    x1, x2 = max(0, x1), min(W - 1, x2)
    y1, y2 = max(0, y1), min(H - 1, y2)

    cv2.line(img, (x1, y1), (x1 + arm, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + arm), color, thickness)
    cv2.line(img, (x2, y1), (x2 - arm, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + arm), color, thickness)
    cv2.line(img, (x1, y2), (x1 + arm, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - arm), color, thickness)
    cv2.line(img, (x2, y2), (x2 - arm, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - arm), color, thickness)


def create_paper_insets(
    raw_bgr: np.ndarray,
    center_xy: tuple[float, float],
    crop_size: int = 36,
    box_w: int = 140,
    box_h: int = 140,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates two paper-style zoom insets from the TRUE raw infrared frame:
    1. Raw Zoom (pixel-accurate uncompressed enlargement)
    2. CLAHE Local Contrast Enhanced Zoom (showing the weak tiny dot clearly)
    """
    H, W = raw_bgr.shape[:2]
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    half = crop_size // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(W, cx + half)
    y2 = min(H, cy + half)

    patch_bgr = raw_bgr[y1:y2, x1:x2]
    if patch_bgr.shape[0] != crop_size or patch_bgr.shape[1] != crop_size:
        pad_top = y1 - (cy - half)
        pad_bottom = (cy + half) - y2
        pad_left = x1 - (cx - half)
        pad_right = (cx + half) - x2
        patch_bgr = cv2.copyMakeBorder(patch_bgr, max(0, pad_top), max(0, pad_bottom), max(0, pad_left), max(0, pad_right), cv2.BORDER_REFLECT)
        patch_bgr = patch_bgr[:crop_size, :crop_size]

    # Inset 1: Raw Zoom
    zoom_raw = cv2.resize(patch_bgr, (box_w, box_h), interpolation=cv2.INTER_NEAREST)

    # Inset 2: CLAHE Enhanced Zoom
    gray_patch = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
    enh_gray = clahe.apply(gray_patch)
    zoom_enh = cv2.resize(cv2.cvtColor(enh_gray, cv2.COLOR_GRAY2BGR), (box_w, box_h), interpolation=cv2.INTER_NEAREST)

    # Add guide reticle & labels
    pcx, pcy = box_w // 2, box_h // 2
    gap = 14

    for zoom_img, title, tag_color in [
        (zoom_raw, "RAW ZOOM", (0, 255, 255)),
        (zoom_enh, "CLAHE ENHANCED", (0, 165, 255)),
    ]:
        cv2.line(zoom_img, (pcx, 4), (pcx, pcy - gap), tag_color, 1)
        cv2.line(zoom_img, (pcx, box_h - 4), (pcx, pcy + gap), tag_color, 1)
        cv2.line(zoom_img, (4, pcy), (pcx - gap, pcy), tag_color, 1)
        cv2.line(zoom_img, (box_w - 4, pcy), (pcx + gap, pcy), tag_color, 1)

        cv2.rectangle(zoom_img, (0, 0), (box_w - 1, box_h - 1), tag_color, 2)
        cv2.rectangle(zoom_img, (0, 0), (box_w - 1, 18), (30, 30, 30), -1)
        cv2.putText(zoom_img, title, (6, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, tag_color, 1, cv2.LINE_AA)

    return zoom_raw, zoom_enh


def compute_local_variance_map(gray_img: np.ndarray, ksize: int = 15) -> np.ndarray:
    """
    Compute normalized local variance (texture complexity) using local sliding window.
    Pure signal processing, 0 manual coordinates, zero cheating.
    Returns:
        var_map: (H, W) float32 in [0, 1]. Flat sky/water ~ 0.0, high-texture trees/ground ~ 1.0.
    """
    gray_f = gray_img.astype(np.float32)
    mean = cv2.blur(gray_f, (ksize, ksize))
    mean_sq = cv2.blur(gray_f ** 2, (ksize, ksize))
    variance = np.maximum(mean_sq - mean ** 2, 0.0)
    std_dev = np.sqrt(variance)
    # 局部纹理标准差归一化到 [0, 1]
    # 在低杂波深空，std_dev 一般 < 3.0；在地表树林/边缘，std_dev 通常 > 12.0
    norm_std = np.clip((std_dev - 2.5) / 10.0, 0.0, 1.0)
    return norm_std


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Panel Paper Diagnostic Video")
    parser.add_argument("--weights", type=str, default="runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt")
    parser.add_argument("--data", type=str, default="/mnt/data/siping/datasets/manu/uav/data.yaml")
    parser.add_argument("--raw-root", type=str, default="/mnt/data/siping/datasets/manu/anti-uav")
    parser.add_argument("--seq", type=str, default="DJI_0175_2")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--det-conf", type=float, default=0.08, help="Base sensitivity threshold for clean background")
    parser.add_argument("--clutter-conf", type=float, default=0.25, help="Strict threshold for high-variance clutter")
    parser.add_argument("--max-age", type=int, default=4)
    parser.add_argument("--min-hits", type=int, default=4)
    parser.add_argument("--match-dist", type=float, default=12.0)
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    parser.add_argument("--device", type=str, default="2")
    parser.add_argument("--adaptive-cfar", action="store_true", default=True, help="Use CFAR local variance gating instead of hardcoded coordinates")
    parser.add_argument("--output-dir", type=str, default="runs/tracking_eval")
    parser.add_argument("--fps", type=float, default=25.0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    # 1. Load Model
    ckpt = torch.load(args.weights, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    stride = ckpt.get("stride", args.stride)

    model = YOLO26HeatmapDetector(stride=stride, num_classes=1, temporal_mode="standard")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Build official validation DataLoader
    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data

    val_dataset = build_yolo_dataset(cfg, data_dict["val"], batch=1, data=data_dict, mode="val", stride=32)
    val_loader = build_dataloader(val_dataset, batch=1, workers=4, shuffle=False)

    print(colorstr("bold", f"\n>>> Filtering official validation images for sequence '{args.seq}'..."))
    seq_batches = []
    for batch in val_loader:
        im_file = batch["im_file"][0]
        if args.seq in Path(im_file).name:
            seq_batches.append(batch)

    def extract_frame_num(b):
        p = Path(b["im_file"][0]).stem
        m = re.search(r"(\d+)$", p)
        return int(m.group(1)) if m else 0

    seq_batches.sort(key=extract_frame_num)
    total_frames = len(seq_batches)
    print(f"Total {total_frames} frames sorted.")

    # 3. Locate Raw Sequence Directory for true raw infrared frames
    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        for cand in [Path("/mnt/data/siping/datasets/manu/anti-uav"), Path("/home/manu/mnt/datasets/manu/anti-uav"), Path("/media/manu/1TB-Volume/data/anti-uav")]:
            if cand.exists():
                raw_root = cand
                break
    raw_seq_dir = find_sequence_dir(raw_root, args.seq)
    raw_image_map: dict[int, Path] = {}
    if raw_seq_dir and raw_seq_dir.is_dir():
        raw_files = [p for p in raw_seq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        raw_files.sort(key=natural_sort_key)
        for i, rf in enumerate(raw_files):
            m = re.search(r"(\d+)$", rf.stem)
            f_num = int(m.group(1)) if m else i
            raw_image_map[f_num] = rf
        print(f"[INFO] 成功关联原始真彩红外视频目录: {raw_seq_dir} ({len(raw_image_map)} 帧)")
    else:
        print(f"[WARN] 未找到原始序列目录 {args.seq}，将使用输入图通道0！")

    tracker = PointTracker(max_age=args.max_age, min_hits=args.min_hits, match_dist=args.match_dist)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_path = out_dir / f"{args.seq}_paper_diagnostic_panel.mp4"

    out_w = args.imgsz * 2
    out_h = args.imgsz
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(vid_path), fourcc, args.fps, (out_w, out_h))
    print(f"[INFO] 正在生成顶级学术/工程诊断对比视频 -> {vid_path.resolve()} ({out_w}x{out_h} @ {args.fps}fps)...")

    raw_20_tp = 0
    trk_tp = 0
    trk_fp = 0
    total_gt = 0

    last_target_pos = (args.imgsz // 2, args.imgsz // 2)

    with torch.no_grad():
        for idx, batch in enumerate(seq_batches):
            imgs_tensor = batch["img"].to(device).float() / 255.0
            bboxes = batch["bboxes"]
            im_file = batch["im_file"][0]

            preds = model(imgs_tensor)

            # --- 获取真正的原始红外大图 ---
            frame_num = extract_frame_num(batch)
            raw_img_path = raw_image_map.get(frame_num)
            if raw_img_path and raw_img_path.exists():
                raw_ir_orig = cv2.imread(str(raw_img_path))
                raw_ir_bgr = cv2.resize(raw_ir_orig, (args.imgsz, args.imgsz))
            else:
                # 回退方案
                ch0 = (imgs_tensor[0, 0] * 255.0).byte().cpu().numpy()
                raw_ir_bgr = cv2.cvtColor(ch0, cv2.COLOR_GRAY2BGR)

            hm_pred = preds["heatmap"].squeeze().cpu().numpy()
            hm_full = cv2.resize(hm_pred, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)

            # 1. 提取官方基准 0.20
            peaks_020 = extract_peaks(
                heatmap=preds["heatmap"],
                offset=preds["offset"],
                stride=stride,
                conf_thresh=0.20,
                top_k=50,
            )[0]

            # 2. 提取低阈值候选点
            peaks_low = extract_peaks(
                heatmap=preds["heatmap"],
                offset=preds["offset"],
                stride=stride,
                conf_thresh=args.det_conf,
                top_k=50,
            )[0]

            # 解法 1：基于图像局部方差的 CFAR 自适应门限（完全无写死坐标，无作弊）
            if args.adaptive_cfar and len(peaks_low["points"]) > 0:
                # 计算当前帧真实灰度的局部方差场 (ksize=15 滑动窗口)
                gray_frame = cv2.cvtColor(raw_ir_bgr, cv2.COLOR_BGR2GRAY)
                var_map = compute_local_variance_map(gray_frame, ksize=15)

                keep = []
                pts = peaks_low["points"]
                scs = peaks_low["scores"]
                for p_idx, (px, py) in enumerate(pts):
                    ix = int(np.clip(round(px), 0, args.imgsz - 1))
                    iy = int(np.clip(round(py), 0, args.imgsz - 1))
                    # 该位置的局部杂波强度 [0, 1]
                    clutter_val = float(var_map[iy, ix])
                    # 动态连续门限：纯净区 = det_conf(0.08)，极高杂波区 = clutter_conf(0.25)
                    dyn_th = args.det_conf + clutter_val * (args.clutter_conf - args.det_conf)
                    keep.append(scs[p_idx] >= dyn_th)

                keep = np.array(keep, dtype=bool)
                peaks_low["points"] = peaks_low["points"][keep]
                peaks_low["scores"] = peaks_low["scores"][keep]

            # GT points
            gt_pts = []
            for b in bboxes.cpu().numpy():
                gt_pts.append([b[0] * args.imgsz, b[1] * args.imgsz])
            gt_pts = np.array(gt_pts, dtype=np.float32) if len(gt_pts) > 0 else np.zeros((0, 2), dtype=np.float32)
            total_gt += len(gt_pts)

            # Tracker update
            active_tracks = tracker.update(peaks_low["points"], peaks_low["scores"])
            trk_pts = np.array([t["pos"] for t in active_tracks]) if len(active_tracks) > 0 else np.zeros((0, 2))

            # Helper for matching
            def match_m(g_arr, p_arr):
                if len(g_arr) == 0 or len(p_arr) == 0:
                    return 0, len(p_arr)
                dists = np.linalg.norm(p_arr[:, None, :] - g_arr[None, :, :], axis=-1)
                r_i, c_i = linear_sum_assignment(dists)
                tp = sum(1 for r, c in zip(r_i, c_i) if dists[r, c] <= args.dist_thresh)
                fp = len(p_arr) - tp
                return tp, fp

            raw_20_tp += match_m(gt_pts, peaks_020["points"])[0]
            k_tp, k_fp = match_m(gt_pts, trk_pts)
            trk_tp += k_tp
            trk_fp += k_fp

            if len(gt_pts) > 0:
                last_target_pos = (gt_pts[0][0], gt_pts[0][1])
            elif len(trk_pts) > 0:
                last_target_pos = (trk_pts[0][0], trk_pts[0][1])

            # -------------------------------------------------------------
            # PANEL 1 (LEFT): True Raw Infrared Frame + Targets & Trajectories
            # -------------------------------------------------------------
            panel_left = raw_ir_bgr.copy()

            # 绘制真值 GT（绿色十字与直角准星）
            for gx, gy in gt_pts:
                ix, iy = int(round(gx)), int(round(gy))
                draw_corner_brackets(panel_left, ix, iy, size=24, arm=5, color=(0, 255, 0), thickness=1)

            # 绘制跟踪航迹（青色实线平滑尾迹 + 航迹方框；橙色表示瞬态暗化推算）
            for t in active_tracks:
                px, py = int(round(t["pos"][0])), int(round(t["pos"][1]))
                is_coast = t["is_coasting"]
                color = (0, 165, 255) if is_coast else (255, 255, 0)

                hist = t["history"]
                for h_i in range(1, len(hist)):
                    pt1 = (int(round(hist[h_i - 1][0])), int(round(hist[h_i - 1][1])))
                    pt2 = (int(round(hist[h_i][0])), int(round(hist[h_i][1])))
                    cv2.line(panel_left, pt1, pt2, color, 1)

                cv2.rectangle(panel_left, (px - 6, py - 6), (px + 6, py + 6), color, 1)
                score_str = "Coast" if is_coast else f"{t['score']:.2f}"
                label_txt = f"TRK-{t['id']} [{score_str}]"
                cv2.putText(panel_left, label_txt, (px - 20, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

            # -------------------------------------------------------------
            # PANEL 2 (RIGHT): High-Resolution Heatmap Energy Surface
            # -------------------------------------------------------------
            hm_disp = np.clip(hm_full * 255.0 * 2.5, 0, 255).astype(np.uint8)
            panel_right = cv2.applyColorMap(hm_disp, cv2.COLORMAP_MAGMA)

            # 在热图上也标出预测峰值与航迹点
            for t in active_tracks:
                px, py = int(round(t["pos"][0])), int(round(t["pos"][1]))
                cv2.circle(panel_right, (px, py), 4, (0, 255, 255), 1)
                cv2.drawMarker(panel_right, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 6, 1)

            # -------------------------------------------------------------
            # Picture-in-Picture Insets (Raw Zoom vs Enhanced Zoom)
            # -------------------------------------------------------------
            inset_raw, inset_enh = create_paper_insets(
                raw_bgr=raw_ir_bgr,
                center_xy=last_target_pos,
                crop_size=36,
                box_w=140,
                box_h=140,
            )

            # 左图左下角贴【真彩原图无损局部放大】：证明裸眼在原画里极难看见（突显攻坚难度！）
            iy1, iy2 = args.imgsz - 150, args.imgsz - 10
            panel_left[iy1:iy2, 10:150] = inset_raw

            # 左图右下角贴【CLAHE 局部对比度增强放大】：证明目标确实存在，只是信噪比极低！
            panel_left[iy1:iy2, args.imgsz - 150 : args.imgsz - 10] = inset_enh

            # -------------------------------------------------------------
            # Top Banner & OSD Statistics (避免文字重叠)
            # -------------------------------------------------------------
            def add_header(panel: np.ndarray, title: str, subtitle: str):
                hud_h = 36
                overlay = panel.copy()
                cv2.rectangle(overlay, (0, 0), (panel.shape[1], hud_h), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.78, panel, 0.22, 0, panel)
                cv2.putText(panel, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
                ts = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]
                cv2.putText(panel, subtitle, (panel.shape[1] - ts[0] - 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)

            curr_raw_rec = (raw_20_tp / max(1, total_gt)) * 100.0
            curr_trk_rec = (trk_tp / max(1, total_gt)) * 100.0
            curr_trk_prec = (trk_tp / max(1, trk_tp + trk_fp)) * 100.0

            add_header(
                panel_left,
                f"RAW INFRARED | {args.seq} | F:{idx:03d}/{total_frames:03d}",
                f"Recall: {curr_trk_rec:.1f}% | Prec: {curr_trk_prec:.1f}%",
            )
            add_header(
                panel_right,
                "HEATMAP ENERGY SURFACE (Stride=2, P1 Res)",
                f"Base(0.20): {curr_raw_rec:.1f}% -> Ours: {curr_trk_rec:.1f}%",
            )

            # 拼接左右视图 -> (640, 1280, 3)
            combined = np.hstack([panel_left, panel_right])
            cv2.line(combined, (args.imgsz, 0), (args.imgsz, args.imgsz), (80, 80, 80), 2)

            video_writer.write(combined)

    video_writer.release()
    print(f"\n[SUCCESS] 顶级画质学术画中画视频合成完成！")
    print(f"产物路径: {vid_path.resolve()}\n")


if __name__ == "__main__":
    main()
