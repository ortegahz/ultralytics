#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Scheme B Evaluation: Full Multi-Sequence & Grand Dataset GMC + Spatial-Kinematic Fusion.

Integrates:
1. Online Global Motion Compensation (GMC) on raw infrared frames to generate
   jitter-free difference channels: [I_t, |I_t - W(I_{t-1})|, |I_t - W(I_{t-2})|]
2. Online heatmap inference using the best checkpoint (trial_0031)
3. Four-tier evaluation per sequence and grand total:
   - Mode 1: Baseline Unaligned (th=0.20)
   - Mode 2: GMC Alone (th=0.20)
   - Mode 3: Kinematic Tracking alone on GMC predictions
   - Mode 4: Full Fusion (GMC + Spatial Prior Gating + Dual-Threshold Kinematic Tracking)

Usage on Server:
    # 1. Quick test on the 4 Hard Cases:
    python manu/eval_gmc_fusion_benchmark.py \
        --weights runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt \
        --raw-root /mnt/data/siping/datasets/manu/anti-uav \
        --sequences DJI_0051_2,DJI_0175_2,wg2022_ir_011_split_03,wg2022_ir_020_split_03 \
        --gmc-method sparseOptFlow \
        --device 0

    # 2. Grand Full Dataset Evaluation across all validation sequences:
    python manu/eval_gmc_fusion_benchmark.py \
        --weights runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt \
        --raw-root /mnt/data/siping/datasets/manu/anti-uav \
        --ref-dataset /mnt/data/siping/datasets/manu/uav \
        --gmc-method sparseOptFlow \
        --device 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.utils import colorstr
from manu.heatmap_model import YOLO26HeatmapDetector
from manu.heatmap_evaluate import extract_peaks


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def natural_sort_key(p: Path | str):
    stem = Path(p).stem
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", stem)]


def letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2.0
    dh /= 2.0

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


# ==============================================================================
# 1. GMC Estimator & Difference Processor
# ==============================================================================

class GlobalMotionEstimator:
    """Computes affine transformation matrix H from frame_prev to frame_curr."""

    def __init__(self, method: str = "sparseOptFlow", downscale: int = 2):
        self.method = method
        self.downscale = downscale
        self.feature_params = {
            "maxCorners": 800,
            "qualityLevel": 0.01,
            "minDistance": 4,
            "blockSize": 3,
        }

    def compute_affine(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        H = np.eye(2, 3, dtype=np.float32)
        h, w = curr_gray.shape[:2]
        ds = self.downscale
        if ds > 1:
            prev_small = cv2.resize(prev_gray, (w // ds, h // ds))
            curr_small = cv2.resize(curr_gray, (w // ds, h // ds))
        else:
            prev_small = prev_gray
            curr_small = curr_gray

        if self.method == "sparseOptFlow":
            pts_prev = cv2.goodFeaturesToTrack(prev_small, mask=None, **self.feature_params)
            if pts_prev is None or len(pts_prev) < 6:
                return H

            pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_small, curr_small, pts_prev, None, winSize=(15, 15), maxLevel=2
            )
            good = (status.ravel() == 1)
            p0 = pts_prev[good]
            p1 = pts_curr[good]

            if len(p0) >= 6:
                M, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if M is not None:
                    H = M.astype(np.float32)
                    if ds > 1:
                        H[0, 2] *= ds
                        H[1, 2] *= ds

        elif self.method == "ecc":
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4)
            H_small = np.eye(2, 3, dtype=np.float32)
            try:
                _, H_small = cv2.findTransformECC(
                    prev_small, curr_small, H_small, cv2.MOTION_EUCLIDEAN, criteria, None, 1
                )
                H = H_small.astype(np.float32)
                if ds > 1:
                    H[0, 2] *= ds
                    H[1, 2] *= ds
            except Exception:
                pass

        return H


def align_and_diff(
    curr_gray: np.ndarray,
    prev_gray: np.ndarray,
    gmc_estimator: Optional[GlobalMotionEstimator],
) -> np.ndarray:
    if gmc_estimator is None:
        return cv2.absdiff(curr_gray, prev_gray)

    h, w = curr_gray.shape[:2]
    H = gmc_estimator.compute_affine(prev_gray, curr_gray)
    warped_prev = cv2.warpAffine(
        prev_gray,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return cv2.absdiff(curr_gray, warped_prev)


# ==============================================================================
# 2. Kinematic Tracker with Two-Stage Initiation & Salvaging Logic
# ==============================================================================

class PointKalmanTrack:
    _count = 0

    def __init__(self, init_pos: np.ndarray, score: float, is_confirmed: bool = False):
        PointKalmanTrack._count += 1
        self.track_id = PointKalmanTrack._count
        self.x = np.array([init_pos[0], init_pos[1], 0.0, 0.0], dtype=np.float32)
        self.P = np.diag([10.0, 10.0, 50.0, 50.0]).astype(np.float32)
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
        self.score = float(score)
        self.init_score = float(score)
        self.is_confirmed = is_confirmed
        self.start_pos = init_pos.copy()
        self.history = [self.get_pos()]

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.get_pos()

    def update(self, pos: np.ndarray, score: float):
        self.time_since_update = 0
        self.hits += 1
        self.score = 0.65 * self.score + 0.35 * float(score)
        y = pos - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        self.history.append(self.get_pos())

    def get_pos(self) -> np.ndarray:
        return self.x[:2].copy()

    def net_displacement(self) -> float:
        curr = self.get_pos()
        return float(np.sqrt(np.sum((curr - self.start_pos) ** 2)))


class PointTracker:
    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 3,
        match_dist: float = 12.0,
        output_coasting: bool = False,
        min_track_score: float = 0.07,
        instant_conf: float = 0.30,
        min_displacement: float = 2.5,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.match_dist = match_dist
        self.output_coasting = output_coasting
        self.min_track_score = min_track_score
        self.instant_conf = instant_conf
        self.min_displacement = min_displacement
        self.tracks: List[PointKalmanTrack] = []

    def update(
        self,
        high_points: np.ndarray,
        high_scores: np.ndarray,
        salvage_points: Optional[np.ndarray] = None,
        salvage_scores: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        predicted_positions = []
        for trk in self.tracks:
            predicted_positions.append(trk.predict())

        matched_tracks = set()
        matched_high = set()

        if len(self.tracks) > 0 and len(high_points) > 0:
            pred_arr = np.array(predicted_positions)
            diff = pred_arr[:, None, :] - high_points[None, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            r_ind, c_ind = linear_sum_assignment(dists)

            for r, c in zip(r_ind, c_ind):
                if dists[r, c] <= self.match_dist:
                    self.tracks[r].update(high_points[c], high_scores[c])
                    self.tracks[r].is_confirmed = True
                    matched_tracks.add(r)
                    matched_high.add(c)

        if salvage_points is not None and len(salvage_points) > 0:
            unmatched_trks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
            if len(unmatched_trks) > 0:
                pred_arr = np.array([predicted_positions[i] for i in unmatched_trks])
                diff = pred_arr[:, None, :] - salvage_points[None, :, :]
                dists = np.sqrt(np.sum(diff ** 2, axis=-1))
                r_ind, c_ind = linear_sum_assignment(dists)

                for r, c in zip(r_ind, c_ind):
                    trk_idx = unmatched_trks[r]
                    if dists[r, c] <= self.match_dist:
                        self.tracks[trk_idx].update(salvage_points[c], salvage_scores[c])
                        matched_tracks.add(trk_idx)

        for j in range(len(high_points)):
            if j not in matched_high:
                is_instantly_confirmed = high_scores[j] >= self.instant_conf
                new_trk = PointKalmanTrack(high_points[j], high_scores[j], is_confirmed=is_instantly_confirmed)
                self.tracks.append(new_trk)

        outputs = []
        surviving_tracks = []
        for trk in self.tracks:
            if trk.time_since_update > self.max_age:
                continue

            if trk.age >= 5 and trk.net_displacement() < self.min_displacement:
                continue

            surviving_tracks.append(trk)

            is_valid_detection = False
            if trk.time_since_update == 0:
                if (trk.hits >= self.min_hits or trk.is_confirmed) and trk.score >= self.min_track_score:
                    is_valid_detection = True
            elif self.output_coasting:
                if trk.hits >= (self.min_hits + 1) and trk.score >= (self.min_track_score + 0.05):
                    is_valid_detection = True

            if is_valid_detection:
                outputs.append({
                    "track_id": trk.track_id,
                    "pos": trk.get_pos(),
                    "score": trk.score,
                    "is_coasting": trk.time_since_update > 0,
                })

        self.tracks = surviving_tracks
        return outputs


def filter_dense_clutter_clusters(
    pts: np.ndarray,
    scs: np.ndarray,
    cluster_radius: float = 30.0,
    max_neighbors: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(pts) <= max_neighbors:
        return pts, scs

    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    neighbor_counts = np.sum(dists < cluster_radius, axis=-1) - 1
    keep = neighbor_counts <= max_neighbors
    return pts[keep], scs[keep]


def match_predictions_to_gt(
    gt_pts: np.ndarray,
    pred_pts: np.ndarray,
    dist_thresh: float = 8.0,
) -> Tuple[int, int, int]:
    if len(pred_pts) == 0:
        return 0, 0, len(gt_pts)
    if len(gt_pts) == 0:
        return 0, len(pred_pts), 0

    diff = pred_pts[:, None, :] - gt_pts[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))

    r_ind, c_ind = linear_sum_assignment(dists)
    matched_gt = set()
    matched_pred = set()

    for r, c in zip(r_ind, c_ind):
        if dists[r, c] <= dist_thresh:
            matched_pred.add(r)
            matched_gt.add(c)

    tp = len(matched_gt)
    fp = len(pred_pts) - tp
    fn = len(gt_pts) - tp
    return tp, fp, fn


def calc_metrics(tp: int, fp: int, total_gt: int) -> Dict[str, float]:
    rec = (tp / max(1, total_gt)) * 100.0
    prec = (tp / max(1, tp + fp)) * 100.0
    f1 = (2 * rec * prec / max(1e-6, rec + prec)) / 100.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "gt": int(total_gt),
        "recall": float(rec),
        "precision": float(prec),
        "f1": float(f1),
    }


# ==============================================================================
# 3. Sequence Search & Annotation Loaders
# ==============================================================================

def find_sequence_dir(raw_root: Path, seq_name: str) -> Optional[Path]:
    cand = raw_root / seq_name
    if cand.is_dir():
        return cand
    for sub in [raw_root / "Data" / "val" / seq_name, raw_root / "val" / seq_name, raw_root / "train" / seq_name]:
        if sub.is_dir():
            return sub
    for p in raw_root.rglob(seq_name):
        if p.is_dir():
            return p
    return None


def load_gt_annotations(seq_dir: Path) -> dict[int, list[float]]:
    for json_name in ["IR_label.json", "label.json", f"{seq_dir.name}.json"]:
        json_path = seq_dir / json_name
        if json_path.is_file():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "gt_rect" in data:
                    gt_dict = {}
                    gt_rects = data["gt_rect"]
                    exists = data.get("exist", [1] * len(gt_rects))
                    for idx, (rect, exist) in enumerate(zip(gt_rects, exists)):
                        if exist and rect and len(rect) == 4 and rect[2] > 0 and rect[3] > 0:
                            gt_dict[idx] = [float(v) for v in rect]
                    return gt_dict
            except Exception as e:
                pass
    return {}


def discover_sequences(ref_dataset: Optional[Path], raw_root: Path, target_seqs: List[str]) -> List[str]:
    if target_seqs:
        return target_seqs

    # Discover from reference dataset images/val if provided
    discovered = set()
    if ref_dataset and (ref_dataset / "images" / "val").is_dir():
        val_dir = ref_dataset / "images" / "val"
        for p in val_dir.iterdir():
            if p.suffix.lower() in IMAGE_SUFFIXES:
                stem = p.stem
                if "___" in stem:
                    discovered.add(stem.split("___")[0])
                elif "__" in stem:
                    discovered.add(stem.split("__")[0])
                else:
                    m = re.search(r"^(.*?)(?:[_-]+)?\d{3,}$", stem)
                    if m:
                        discovered.add(m.group(1).rstrip("_-"))

    if not discovered:
        # Fallback to scanning raw_root directly
        for p in raw_root.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                discovered.add(p.name)

    return sorted(list(discovered), key=natural_sort_key)


# ==============================================================================
# 4. Core Unified Forward Loop per Sequence
# ==============================================================================

def evaluate_single_sequence(
    seq_name: str,
    seq_dir: Path,
    model: YOLO26HeatmapDetector,
    device: torch.device,
    args,
    tracker_config: dict,
) -> dict:
    image_paths = sorted(
        [p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
        key=natural_sort_key,
    )
    total_imgs = len(image_paths)
    if total_imgs == 0:
        return {}

    gt_dict = load_gt_annotations(seq_dir)

    # Initialize GMC Estimators
    gmc_estimator = GlobalMotionEstimator(method=args.gmc_method, downscale=2)

    # Trackers for Modes 3 and 4
    trk_gmc = PointTracker(**tracker_config)
    trk_fusion = PointTracker(**tracker_config)

    # Accumulator for stats across 4 modes
    stats = {
        "m1_baseline": {"tp": 0, "fp": 0, "gt": 0},
        "m2_gmc_raw": {"tp": 0, "fp": 0, "gt": 0},
        "m3_gmc_trk": {"tp": 0, "fp": 0, "gt": 0},
        "m4_gmc_fusion": {"tp": 0, "fp": 0, "gt": 0},
    }

    frame_cache: dict[int, np.ndarray] = {}

    for i, p in enumerate(image_paths):
        m = re.search(r"(\d+)$", p.stem)
        f_idx = int(m.group(1)) if m else i

        im_curr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im_curr is None:
            continue
        h_orig, w_orig = im_curr.shape[:2]
        frame_cache[f_idx] = im_curr

        f_prev1 = f_idx - 1 if (f_idx - 1) in frame_cache else f_idx
        f_prev2 = f_idx - 2 if (f_idx - 2) in frame_cache else f_prev1

        im_prev1 = frame_cache[f_prev1]
        im_prev2 = frame_cache[f_prev2]

        # 1. Unaligned Difference (Mode 1 Baseline)
        diff1_unalign = cv2.absdiff(im_curr, im_prev1)
        diff2_unalign = cv2.absdiff(im_curr, im_prev2)
        inp_unalign = np.stack([im_curr, diff1_unalign, diff2_unalign], axis=-1)

        # 2. GMC Aligned Difference (Modes 2, 3, 4)
        diff1_gmc = align_and_diff(im_curr, im_prev1, gmc_estimator)
        diff2_gmc = align_and_diff(im_curr, im_prev2, gmc_estimator)
        inp_gmc = np.stack([im_curr, diff1_gmc, diff2_gmc], axis=-1)

        # Forward passes
        # Batching 2 inputs: [Unaligned, GMC] together
        lb_unalign, r, (dw, dh) = letterbox(inp_unalign, (args.imgsz, args.imgsz))
        lb_gmc, _, _ = letterbox(inp_gmc, (args.imgsz, args.imgsz))

        batch_t = torch.from_numpy(np.stack([lb_unalign, lb_gmc], axis=0)).permute(0, 3, 1, 2).float() / 255.0
        batch_t = batch_t.to(device)

        with torch.no_grad():
            preds = model(batch_t)
            peaks_list = extract_peaks(
                preds["heatmap"],
                preds["offset"],
                stride=args.stride,
                conf_thresh=0.03,  # Extract down to 0.03 to allow two-stage gating
                top_k=150,
            )

        # Parse Ground Truth points for this frame
        if f_idx in gt_dict:
            gx, gy, gw, gh = gt_dict[f_idx]
            cx, cy = gx + gw / 2.0, gy + gh / 2.0
            gt_pts = np.array([[cx, cy]], dtype=np.float32)
        else:
            gt_pts = np.zeros((0, 2), dtype=np.float32)

        total_gt = len(gt_pts)
        for k in stats:
            stats[k]["gt"] += total_gt

        # Transform predicted points back to original image coordinates
        def rescale_pts(pts_arr):
            if len(pts_arr) == 0:
                return np.zeros((0, 2), dtype=np.float32)
            pts_out = pts_arr.copy()
            pts_out[:, 0] = np.clip((pts_out[:, 0] - dw) / r, 0, w_orig - 1)
            pts_out[:, 1] = np.clip((pts_out[:, 1] - dh) / r, 0, h_orig - 1)
            return pts_out

        # Mode 1: Baseline (th=0.20 on Unaligned input)
        pts_m1_raw = rescale_pts(peaks_list[0]["points"])
        scs_m1_raw = peaks_list[0]["scores"]
        mask_m1 = scs_m1_raw >= args.th_base
        pts_m1 = pts_m1_raw[mask_m1]
        tp1, fp1, _ = match_predictions_to_gt(gt_pts, pts_m1, args.dist_thresh)
        stats["m1_baseline"]["tp"] += tp1
        stats["m1_baseline"]["fp"] += fp1

        # GMC Predictions
        pts_gmc_raw = rescale_pts(peaks_list[1]["points"])
        scs_gmc_raw = peaks_list[1]["scores"]

        # Mode 2: GMC Raw Detection (th=0.20 on GMC input)
        mask_m2 = scs_gmc_raw >= args.th_base
        pts_m2 = pts_gmc_raw[mask_m2]
        tp2, fp2, _ = match_predictions_to_gt(gt_pts, pts_m2, args.dist_thresh)
        stats["m2_gmc_raw"]["tp"] += tp2
        stats["m2_gmc_raw"]["fp"] += fp2

        # Mode 3: GMC + Kinematic Tracking (th_high=0.20, th_salvage=0.06)
        mask_m3_high = scs_gmc_raw >= args.th_base
        mask_m3_salvage = (scs_gmc_raw >= args.th_salvage) & (~mask_m3_high)
        out_m3 = trk_gmc.update(
            pts_gmc_raw[mask_m3_high],
            scs_gmc_raw[mask_m3_high],
            pts_gmc_raw[mask_m3_salvage],
            scs_gmc_raw[mask_m3_salvage],
        )
        pts_m3 = np.array([t["pos"] for t in out_m3], dtype=np.float32) if out_m3 else np.zeros((0, 2), dtype=np.float32)
        tp3, fp3, _ = match_predictions_to_gt(gt_pts, pts_m3, args.dist_thresh)
        stats["m3_gmc_trk"]["tp"] += tp3
        stats["m3_gmc_trk"]["fp"] += fp3

        # Mode 4: Full Fusion (GMC + Sky/Ground Prior Gating + Density Filtering + Kinematics)
        sky_limit_y = args.sky_ratio * h_orig
        is_sky = pts_gmc_raw[:, 1] < sky_limit_y
        is_ground = ~is_sky

        m4_high_mask = (scs_gmc_raw >= args.th_base) | (is_ground & (scs_gmc_raw >= args.th_ground))
        m4_salvage_mask = is_sky & (scs_gmc_raw >= args.th_salvage) & (scs_gmc_raw < args.th_base)

        pts_m4_high = pts_gmc_raw[m4_high_mask]
        scs_m4_high = scs_gmc_raw[m4_high_mask]
        pts_m4_salvage = pts_gmc_raw[m4_salvage_mask]
        scs_m4_salvage = scs_gmc_raw[m4_salvage_mask]

        pts_m4_salvage, scs_m4_salvage = filter_dense_clutter_clusters(
            pts_m4_salvage, scs_m4_salvage, cluster_radius=25.0, max_neighbors=2
        )

        out_m4 = trk_fusion.update(pts_m4_high, scs_m4_high, pts_m4_salvage, scs_m4_salvage)
        pts_m4 = np.array([t["pos"] for t in out_m4], dtype=np.float32) if out_m4 else np.zeros((0, 2), dtype=np.float32)
        tp4, fp4, _ = match_predictions_to_gt(gt_pts, pts_m4, args.dist_thresh)
        stats["m4_gmc_fusion"]["tp"] += tp4
        stats["m4_gmc_fusion"]["fp"] += fp4

        # Evict old cached frames
        if (f_idx - 5) in frame_cache:
            del frame_cache[f_idx - 5]

    metrics = {}
    for k in stats:
        metrics[k] = calc_metrics(stats[k]["tp"], stats[k]["fp"], stats[k]["gt"])
    metrics["frames"] = total_imgs
    return metrics


# ==============================================================================
# 5. Main Script Runner
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Scheme B: Full GMC + Spatial-Kinematic Fusion Benchmark")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt",
        help="Path to trained trial_0031 weights",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/anti-uav",
        help="Root path to raw sequences",
    )
    parser.add_argument(
        "--ref-dataset",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav",
        help="Reference YOLO dataset path (to discover all 24 validation sequences)",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default="",
        help="Comma-separated sequence names to test. Empty = test all validation sequences",
    )
    parser.add_argument("--gmc-method", type=str, default="sparseOptFlow", choices=["sparseOptFlow", "ecc"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="Distance threshold (GJB standard 8.0px)")
    parser.add_argument("--th-base", type=float, default=0.20, help="Base detection threshold (default: 0.20)")
    parser.add_argument("--th-salvage", type=float, default=0.06, help="Sky weak pulse salvage threshold (default: 0.06)")
    parser.add_argument("--th-ground", type=float, default=0.28, help="Ground strict threshold (default: 0.28)")
    parser.add_argument("--sky-ratio", type=float, default=0.60, help="Sky upper partition ratio (default: 0.60)")
    parser.add_argument("--min-hits", type=int, default=3, help="Tracker minimum hits (default: 3)")
    parser.add_argument("--max-age", type=int, default=3, help="Tracker max age (default: 3)")
    parser.add_argument("--match-dist", type=float, default=12.0, help="Tracker gating distance (default: 12.0px)")
    parser.add_argument("--min-track-score", type=float, default=0.07, help="Min track score (default: 0.07)")
    parser.add_argument("--instant-conf", type=float, default=0.30, help="0-latency pass conf (default: 0.30)")
    parser.add_argument("--min-disp", type=float, default=2.5, help="Min net displacement to kill static clutter")
    parser.add_argument("--output-coasting", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output-dir", type=str, default="runs/gmc_fusion_eval")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 115)
    print("   UAV Tiny Object Detection: Scheme B Full GMC + Spatial-Kinematic Fusion Benchmark")
    print(f"   Checkpoint: {args.weights} | GMC Method: {args.gmc_method} | Tol <= {args.dist_thresh:.1f}px")
    print("=" * 115)

    # 1. Resolve raw root
    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        for cand in [
            Path("/mnt/data/siping/datasets/manu/anti-uav"),
            Path("/mnt/data/siping/datasets/anti-uav"),
            Path("/home/manu/mnt/datasets/manu/anti-uav"),
            Path("/home/manu/mnt/datasets/anti-uav"),
            Path("/media/manu/1TB-Volume/data/anti-uav"),
        ]:
            if cand.exists():
                raw_root = cand
                break
    print(f"[INFO] Raw Sequences Root : {raw_root}")

    ref_dir = Path(args.ref_dataset) if args.ref_dataset else None
    target_seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]
    seq_list = discover_sequences(ref_dir, raw_root, target_seqs)
    print(f"[INFO] Total sequences queued for evaluation: {len(seq_list)}")

    # 2. Load Model
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[INFO] Using Device: {device}")
    model = YOLO26HeatmapDetector(stride=args.stride, num_classes=1, temporal_mode="standard")
    ckpt = torch.load(args.weights, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tracker_config = {
        "max_age": args.max_age,
        "min_hits": args.min_hits,
        "match_dist": args.match_dist,
        "output_coasting": args.output_coasting,
        "min_track_score": args.min_track_score,
        "instant_conf": args.instant_conf,
        "min_displacement": args.min_disp,
    }

    grand_stats = {
        "m1_baseline": {"tp": 0, "fp": 0, "gt": 0},
        "m2_gmc_raw": {"tp": 0, "fp": 0, "gt": 0},
        "m3_gmc_trk": {"tp": 0, "fp": 0, "gt": 0},
        "m4_gmc_fusion": {"tp": 0, "fp": 0, "gt": 0},
    }
    grand_frames = 0
    all_sequence_results = []

    print("\n" + "=" * 115)
    print(f"{'Sequence Name':<28} | {'Mode':<20} | {'TP / GT':<13} | {'FP':<6} | {'Recall':<8} | {'Prec':<8} | {'F1-Score':<8}")
    print("=" * 115)

    for seq_name in seq_list:
        seq_dir = find_sequence_dir(raw_root, seq_name)
        if seq_dir is None:
            print(f"[WARN] Sequence '{seq_name}' directory not found, skipping.")
            continue

        res = evaluate_single_sequence(
            seq_name=seq_name,
            seq_dir=seq_dir,
            model=model,
            device=device,
            args=args,
            tracker_config=tracker_config,
        )
        if not res:
            continue

        grand_frames += res["frames"]
        for k in grand_stats:
            grand_stats[k]["tp"] += res[k]["tp"]
            grand_stats[k]["fp"] += res[k]["fp"]
            grand_stats[k]["gt"] += res[k]["gt"]

        m1 = res["m1_baseline"]
        m2 = res["m2_gmc_raw"]
        m3 = res["m3_gmc_trk"]
        m4 = res["m4_gmc_fusion"]

        print(f"{seq_name:<28} | {'1. Baseline (Unalign)':<20} | {m1['tp']:>5} / {m1['gt']:<5} | {m1['fp']:<6} | {m1['recall']:>6.2f}% | {m1['precision']:>6.2f}% | {m1['f1']:>6.4f}")
        print(f"{'':<28} | {'2. GMC Raw (0.20)':<20} | {m2['tp']:>5} / {m2['gt']:<5} | {m2['fp']:<6} | {m2['recall']:>6.2f}% | {m2['precision']:>6.2f}% | {m2['f1']:>6.4f}")
        print(f"{'':<28} | {'3. GMC + Kinematic':<20} | {m3['tp']:>5} / {m3['gt']:<5} | {m3['fp']:<6} | {m3['recall']:>6.2f}% | {m3['precision']:>6.2f}% | {m3['f1']:>6.4f}")
        print(f"{'':<28} | {colorstr('bold', colorstr('green', '4. GMC + Fusion F1')):29} | {m4['tp']:>5} / {m4['gt']:<5} | {m4['fp']:<6} | {m4['recall']:>6.2f}% | {m4['precision']:>6.2f}% | {m4['f1']:>6.4f}")
        print("-" * 115)

        all_sequence_results.append({
            "sequence": seq_name,
            "metrics": res,
        })

    # Print Grand Overall Table
    print("\n" + "=" * 115)
    print(colorstr("bold", f"GRAND OVERALL RESULTS ACROSS ALL EVALUATED SEQUENCES (Tol <= {args.dist_thresh:.1f}px)"))
    print("=" * 115)

    grand_metrics = {}
    for k in grand_stats:
        grand_metrics[k] = calc_metrics(grand_stats[k]["tp"], grand_stats[k]["fp"], grand_stats[k]["gt"])

    mode_labels = [
        ("1. Standard Baseline (Unaligned th=0.20)", "m1_baseline"),
        ("2. GMC Pure Detection (Aligned th=0.20)", "m2_gmc_raw"),
        ("3. GMC + Pure Kinematic Filtering", "m3_gmc_trk"),
        ("4. SCHEME B: GMC + Spatial-Kinematic FUSION", "m4_gmc_fusion"),
    ]

    for label, key in mode_labels:
        gm = grand_metrics[key]
        far = gm["fp"] / max(1, grand_frames)
        line = (
            f"{label:<42} | TP: {gm['tp']:>5}/{gm['gt']:<5} | "
            f"FP: {gm['fp']:<6} | Recall: {gm['recall']:>6.2f}% | "
            f"Prec: {gm['precision']:>6.2f}% | F1: {gm['f1']:>6.4f} | "
            f"FAR: {far:.4f}/frame"
        )
        if key == "m4_gmc_fusion":
            print(colorstr("bold", colorstr("green", line)))
        else:
            print(line)
    print("=" * 115 + "\n")

    # Save grand evaluation json
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "scheme_b_grand_evaluation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "grand_metrics": grand_metrics,
            "grand_frames": grand_frames,
            "sequences": all_sequence_results,
        }, f, indent=2)
    print(f"[INFO] Complete Grand Results exported to: {summary_file.resolve()}")


if __name__ == "__main__":
    main()
