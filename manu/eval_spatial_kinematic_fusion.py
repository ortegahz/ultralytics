#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Unified Evaluation Script: Spatial Prior Gating + Kinematic Track Filtering
Evaluates scheme 2 (Spatial/Variance Gating) and scheme 5 (Kinematic Hypotheses Filtering)
on the entire validation dataset or specified sequences using precomputed inference cache
or direct model forward pass.

Key algorithmic designs:
1. Two-stage Peak Extraction & Gating:
   - High threshold (th_high, default 0.20~0.25): Confirmed strong detections, seeds new tracks.
   - Low threshold (th_low, default 0.05~0.08): Weak candidate detections.
2. Spatial / CFAR Prior Gating:
   - If raw image is available, dynamic threshold tau(x, y) = tau_low + norm_var * (tau_high - tau_low)
   - Alternatively, height-based sky/ground geometric partition:
     Sky (y < sky_ratio * H): allow weak pulses down to th_low
     Ground (y >= sky_ratio * H): enforce strict th_high
3. Two-Stage Kinematic Track Filtering (System-level FP suppression):
   - Confirmed tracks can salvage weak pulses in gating window (match_dist).
   - Unconfirmed weak pulses cannot initiate standalone tracks (preventing random white-noise false alarms).
   - Strict consecutive hits (min_hits >= 3) to filter isolated false alarms.
   - Controlled coasting (extrapolation) with strict lifetime and score decay.

Usage:
    # 1. Fast evaluation from inference_cache.pkl (takes ~5 seconds across 31,613 images!):
    python manu/eval_spatial_kinematic_fusion.py \
        --cache-file runs/badcase_analysis/inference_cache.pkl \
        --dist-thresh 8.0

    # 2. Test specific sequences (e.g. hard cases):
    python manu/eval_spatial_kinematic_fusion.py \
        --cache-file runs/badcase_analysis/inference_cache.pkl \
        --sequences DJI_0175_2,wg2022_ir_011_split_03,DJI_0051_2,wg2022_ir_020_split_03

    # 3. Direct model inference on validation set:
    python manu/eval_spatial_kinematic_fusion.py \
        --weights runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt \
        --data /mnt/data/siping/datasets/manu/uav/data.yaml \
        --device 2
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
import torch
from scipy.optimize import linear_sum_assignment
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


# ==============================================================================
# 1. Kinematic Tracker with Two-Stage Initiation & Salvaging Logic
# ==============================================================================

class PointKalmanTrack:
    _count = 0

    def __init__(self, init_pos: np.ndarray, score: float, is_confirmed: bool = False):
        PointKalmanTrack._count += 1
        self.track_id = PointKalmanTrack._count
        # State: [x, y, vx, vy]
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
        if len(self.history) > 50:
            self.history.pop(0)

    def get_pos(self) -> np.ndarray:
        return self.x[:2].copy()

    def get_net_displacement(self) -> float:
        """Calculate net distance between current position and start position."""
        return float(np.linalg.norm(self.x[:2] - self.start_pos))


class TwoStageKinematicTracker:
    """
    Two-stage Kinematic Tracker with:
    1. Zero-latency instant output for confident detections (score >= instant_conf, e.g. 0.30).
       Prevents losing frames 1-2 on high-confidence targets in clean sequences.
    2. Multi-frame kinematic verification for weak pulses (hits >= min_hits).
    3. Stationary clutter rejection: cuts off tracks that persist for multiple frames
       without net displacement (e.g. static ground tree/building edge false alarms).
    """

    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 3,
        match_dist: float = 12.0,
        output_coasting: bool = False,
        min_track_score: float = 0.08,
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

    def reset(self):
        self.tracks.clear()

    def update(
        self,
        high_pts: np.ndarray,
        high_scs: np.ndarray,
        salvage_pts: np.ndarray,
        salvage_scs: np.ndarray,
    ) -> List[Dict]:
        # 1. Kalman prediction
        for t in self.tracks:
            t.predict()

        matched_tracks = set()
        matched_high_dets = set()

        # Step A: Associate existing tracks with high-confidence detections first
        if len(self.tracks) > 0 and len(high_pts) > 0:
            t_pos = np.array([t.get_pos() for t in self.tracks])
            dists = np.linalg.norm(t_pos[:, None, :] - high_pts[None, :, :], axis=-1)
            row_ind, col_ind = linear_sum_assignment(dists)
            for r, c in zip(row_ind, col_ind):
                if dists[r, c] <= self.match_dist:
                    self.tracks[r].update(high_pts[c], high_scs[c])
                    matched_tracks.add(r)
                    matched_high_dets.add(c)

        # Step B: Associate unmatched existing tracks with weak salvage detections
        unmatched_track_indices = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        if len(unmatched_track_indices) > 0 and len(salvage_pts) > 0:
            t_sub_pos = np.array([self.tracks[i].get_pos() for i in unmatched_track_indices])
            dists_salvage = np.linalg.norm(t_sub_pos[:, None, :] - salvage_pts[None, :, :], axis=-1)
            r_sub, c_sub = linear_sum_assignment(dists_salvage)
            for r_idx, c_idx in zip(r_sub, c_sub):
                if dists_salvage[r_idx, c_idx] <= self.match_dist:
                    orig_track_idx = unmatched_track_indices[r_idx]
                    self.tracks[orig_track_idx].update(salvage_pts[c_idx], salvage_scs[c_idx])
                    matched_tracks.add(orig_track_idx)

        # Step C: Only UNMATCHED HIGH-CONFIDENCE detections can initiate new tracks
        for i in range(len(high_pts)):
            if i not in matched_high_dets:
                is_instantly_confirmed = high_scs[i] >= self.instant_conf
                self.tracks.append(PointKalmanTrack(high_pts[i], high_scs[i], is_confirmed=is_instantly_confirmed))

        # Step D: Filter stationary clutter & generate outputs
        surviving_tracks = []
        outputs = []
        for t in self.tracks:
            if t.time_since_update > self.max_age:
                continue

            # Stationary clutter suppression: If a track has lasted for >= 5 hits
            # but hasn't moved more than min_displacement pixels, it's ground edge vibration
            if t.hits >= 5 and self.min_displacement > 0:
                if t.get_net_displacement() < self.min_displacement:
                    continue  # Kill static false alarm

            surviving_tracks.append(t)

            # Output condition:
            # 1. Zero-latency instant pass: If the track is fresh (hits < min_hits) but
            #    the detection itself is strong (init_score >= instant_conf) and alive now,
            #    output immediately! (Eliminates the 2-frame cold start delay on clear targets)
            # 2. Multi-frame kinematic confirmed: hits >= min_hits and score >= min_track_score
            is_instant = (t.time_since_update == 0) and (t.init_score >= self.instant_conf or t.is_confirmed)
            is_kinematic_confirmed = (t.hits >= self.min_hits) and (t.score >= self.min_track_score)

            if is_instant or is_kinematic_confirmed:
                if self.output_coasting:
                    # Allow coasting output if track was firmly confirmed (hits >= min_hits)
                    if t.time_since_update == 0 or (t.hits >= self.min_hits and t.score >= self.min_track_score):
                        outputs.append({
                            "id": t.track_id,
                            "pos": t.get_pos(),
                            "score": t.score,
                            "is_coasting": t.time_since_update > 0,
                        })
                elif t.time_since_update == 0:
                    outputs.append({
                        "id": t.track_id,
                        "pos": t.get_pos(),
                        "score": t.score,
                        "is_coasting": False,
                    })

        self.tracks = surviving_tracks
        return outputs


# ==============================================================================
# 2. Advanced Prior Filters (Density & Cluster Suppression)
# ==============================================================================

def filter_dense_clutter_clusters(
    pts: np.ndarray,
    scs: np.ndarray,
    cluster_radius: float = 30.0,
    max_neighbors: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Suppresses dense false alarm clusters (e.g. tree lines or building boundaries).
    A real tiny drone in infrared is an ISOLATED point.
    If there are more than `max_neighbors` points within `cluster_radius`,
    it is almost certainly structured ground clutter.
    """
    if len(pts) <= max_neighbors:
        return pts, scs

    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    neighbor_counts = np.sum(dists < cluster_radius, axis=-1) - 1  # exclude self

    keep = neighbor_counts <= max_neighbors
    return pts[keep], scs[keep]


def match_predictions_to_gt(
    gt_pts: np.ndarray,
    pred_pts: np.ndarray,
    dist_thresh: float = 8.0,
) -> Tuple[int, int, int]:
    """Returns (tp, fp, fn)"""
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
    f1 = (2 * rec * prec / max(1e-6, rec + prec))
    return {
        "tp": tp,
        "fp": fp,
        "gt": total_gt,
        "recall": rec,
        "precision": prec,
        "f1": f1,
    }


# ==============================================================================
# 3. Batch Evaluation per Sequence
# ==============================================================================

def evaluate_sequence(
    records: List[Dict],
    dist_thresh: float,
    th_base: float = 0.20,
    th_salvage: float = 0.06,
    th_ground: float = 0.25,
    sky_ratio: float = 0.60,
    img_h: int = 640,
    tracker_config: Optional[Dict] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates 4 modes on one sequence:
      1. Baseline: Single-frame global threshold (th=th_base)
      2. Pure Spatial Gating: Sky (y < sky_ratio*H) th=th_salvage, Ground th=th_ground
      3. Pure Kinematic Filtering: Global th_base for seed, th_salvage for track association
      4. Fusion (Scheme 2 + Scheme 5): Spatial gating + Two-stage Kinematic Filtering
    """
    if tracker_config is None:
        tracker_config = {
            "max_age": 3,
            "min_hits": 3,
            "match_dist": 12.0,
            "output_coasting": False,
            "min_track_score": 0.07,
        }

    # Sort records chronologically by frame number
    records_sorted = sorted(records, key=lambda r: natural_sort_key(r["im_name"]))

    stats = {
        "baseline": {"tp": 0, "fp": 0, "gt": 0},
        "spatial_only": {"tp": 0, "fp": 0, "gt": 0},
        "tracker_only": {"tp": 0, "fp": 0, "gt": 0},
        "fusion": {"tp": 0, "fp": 0, "gt": 0},
    }

    trk_pure = TwoStageKinematicTracker(**tracker_config)
    trk_fusion = TwoStageKinematicTracker(**tracker_config)

    sky_y_boundary = img_h * sky_ratio

    for r in records_sorted:
        gt_pts = np.asarray(r["gt_pts"], dtype=np.float32)
        pred_pts = np.asarray(r["pred_points"], dtype=np.float32)
        pred_scs = np.asarray(r["pred_scores"], dtype=np.float32)
        n_gt = len(gt_pts)

        for k in stats:
            stats[k]["gt"] += n_gt

        # -------------------------------------------------------------
        # Mode 1: Gold Standard Baseline (Fixed at Optimal th=0.30)
        # Directly mirrors the champion trial_0031 milestone
        # -------------------------------------------------------------
        base_mask = pred_scs >= 0.30
        pts_base = pred_pts[base_mask]
        tp1, fp1, _ = match_predictions_to_gt(gt_pts, pts_base, dist_thresh)
        stats["baseline"]["tp"] += tp1
        stats["baseline"]["fp"] += fp1

        # -------------------------------------------------------------
        # Mode 2: Pure Spatial Gating
        # -------------------------------------------------------------
        if len(pred_pts) > 0:
            is_sky = pred_pts[:, 1] < sky_y_boundary
            keep_sky = is_sky & (pred_scs >= th_salvage)
            keep_ground = (~is_sky) & (pred_scs >= th_ground)
            spatial_mask = keep_sky | keep_ground
            pts_spatial = pred_pts[spatial_mask]
        else:
            pts_spatial = np.zeros((0, 2), dtype=np.float32)

        tp2, fp2, _ = match_predictions_to_gt(gt_pts, pts_spatial, dist_thresh)
        stats["spatial_only"]["tp"] += tp2
        stats["spatial_only"]["fp"] += fp2

        # -------------------------------------------------------------
        # Mode 3: Pure Two-Stage Kinematic Filtering (without spatial prior)
        # High detections: sc >= th_base
        # Salvage detections: th_salvage <= sc < th_base
        # -------------------------------------------------------------
        mask_high = pred_scs >= th_base
        mask_salvage = (pred_scs >= th_salvage) & (pred_scs < th_base)

        high_pts_m3 = pred_pts[mask_high]
        high_scs_m3 = pred_scs[mask_high]
        salvage_pts_m3 = pred_pts[mask_salvage]
        salvage_scs_m3 = pred_scs[mask_salvage]

        # Filter dense clutter clusters from salvage candidates (drones are isolated)
        salvage_pts_m3, salvage_scs_m3 = filter_dense_clutter_clusters(
            salvage_pts_m3, salvage_scs_m3, cluster_radius=25.0, max_neighbors=2
        )

        out_tracks_m3 = trk_pure.update(high_pts_m3, high_scs_m3, salvage_pts_m3, salvage_scs_m3)
        pts_m3 = np.array([t["pos"] for t in out_tracks_m3]) if len(out_tracks_m3) > 0 else np.zeros((0, 2), dtype=np.float32)

        tp3, fp3, _ = match_predictions_to_gt(gt_pts, pts_m3, dist_thresh)
        stats["tracker_only"]["tp"] += tp3
        stats["tracker_only"]["fp"] += fp3

        # -------------------------------------------------------------
        # Mode 4: Fusion (Scheme 2 Spatial/CFAR Prior + Scheme 5 Kinematic Gating)
        # -------------------------------------------------------------
        if len(pred_pts) > 0:
            var_map = r.get("var_map")
            if var_map is not None:
                # CFAR Continuous Variance Gating
                pts_round = np.clip(np.round(pred_pts).astype(int), 0, 639)
                clutter_vals = var_map[pts_round[:, 1], pts_round[:, 0]]
                dyn_ths = th_salvage + clutter_vals * (th_ground - th_salvage)
                fusion_high_mask = pred_scs >= np.maximum(th_base, dyn_ths)
                fusion_salvage_mask = (pred_scs >= dyn_ths) & (~fusion_high_mask)
            else:
                # Geometric Partition Gating
                is_sky = pred_pts[:, 1] < sky_y_boundary
                fusion_high_mask = (is_sky & (pred_scs >= th_base)) | ((~is_sky) & (pred_scs >= th_ground))
                fusion_salvage_mask = is_sky & (pred_scs >= th_salvage) & (pred_scs < th_base)

            high_pts_m4 = pred_pts[fusion_high_mask]
            high_scs_m4 = pred_scs[fusion_high_mask]
            salvage_pts_m4 = pred_pts[fusion_salvage_mask]
            salvage_scs_m4 = pred_scs[fusion_salvage_mask]

            salvage_pts_m4, salvage_scs_m4 = filter_dense_clutter_clusters(
                salvage_pts_m4, salvage_scs_m4, cluster_radius=25.0, max_neighbors=2
            )
        else:
            high_pts_m4 = np.zeros((0, 2), dtype=np.float32)
            high_scs_m4 = np.zeros((0,), dtype=np.float32)
            salvage_pts_m4 = np.zeros((0, 2), dtype=np.float32)
            salvage_scs_m4 = np.zeros((0,), dtype=np.float32)

        out_tracks_m4 = trk_fusion.update(high_pts_m4, high_scs_m4, salvage_pts_m4, salvage_scs_m4)
        pts_m4 = np.array([t["pos"] for t in out_tracks_m4]) if len(out_tracks_m4) > 0 else np.zeros((0, 2), dtype=np.float32)

        tp4, fp4, _ = match_predictions_to_gt(gt_pts, pts_m4, dist_thresh)
        stats["fusion"]["tp"] += tp4
        stats["fusion"]["fp"] += fp4

    res = {}
    for k in stats:
        res[k] = calc_metrics(stats[k]["tp"], stats[k]["fp"], stats[k]["gt"])
    return res


# ==============================================================================
# 4. Main Entry Point
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fusion of Scheme 2 (Spatial Gating) & Scheme 5 (Kinematics)")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_gmc_fusion_cache.pkl",
        help="Path to precomputed inference_cache.pkl (supports uav_gmc_fusion_cache.pkl or legacy badcase cache)",
    )
    parser.add_argument("--weights", type=str, default="runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt")
    parser.add_argument("--data", type=str, default="/mnt/data/siping/datasets/manu/uav/data.yaml")
    parser.add_argument(
        "--sequences",
        type=str,
        default="",
        help="Comma-separated sequence names to filter (empty = evaluate all 24 validation sequences)",
    )
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="GJB/Industry evaluation tolerance (default: 8.0px)")
    parser.add_argument("--th-base", type=float, default=0.20, help="Base detection threshold (default: 0.20)")
    parser.add_argument("--th-salvage", type=float, default=0.06, help="Weak pulse salvage threshold in sky (default: 0.06)")
    parser.add_argument("--th-ground", type=float, default=0.28, help="Clutter suppression threshold on ground (default: 0.28)")
    parser.add_argument("--sky-ratio", type=float, default=0.60, help="Upper fraction of image treated as sky (default: 0.60)")
    parser.add_argument("--min-hits", type=int, default=3, help="Tracker minimum consecutive hits (default: 3)")
    parser.add_argument("--max-age", type=int, default=3, help="Tracker max dead age (default: 3)")
    parser.add_argument("--match-dist", type=float, default=12.0, help="Tracker gating association distance (default: 12.0px)")
    parser.add_argument("--min-track-score", type=float, default=0.07, help="Minimum track smoothed score (default: 0.07)")
    parser.add_argument("--instant-conf", type=float, default=0.30, help="High confidence threshold for 0-latency instant output (default: 0.30)")
    parser.add_argument("--min-disp", type=float, default=2.5, help="Min net displacement in pixels for aging tracks to suppress static clutter (default: 2.5px)")
    parser.add_argument("--cfar-gating", action="store_true", default=False, help="Use continuous CFAR texture variance gating")
    parser.add_argument("--output-coasting", action="store_true", default=False, help="Whether to output coasting predictions")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch", type=int, default=32, help="Inference batch size")
    parser.add_argument("--force-forward", action="store_true", default=False, help="Force full model forward pass on data")
    parser.add_argument("--save-cache", type=str, default="", help="Optional path to save records as new cache pickle")
    return parser.parse_args()


def load_or_create_records(args) -> List[Dict]:
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

    if cache_path.exists() and not getattr(args, "force_forward", False):
        print(colorstr("bold", colorstr("green", f"\n>>> Loading cached inferences from: {cache_path}")))
        with open(cache_path, "rb") as f:
            records = pickle.load(f)
        print(f"Loaded {len(records)} frame predictions in < 1 second.\n")
        return records

    print(colorstr("yellow", f"[INFO] Running direct Model Forward pass on dataset: {args.data}..."))
    print(colorstr("cyan", f"[INFO] Target checkpoint: {args.weights}"))

    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    from manu.heatmap_model import YOLO26HeatmapDetector
    from manu.heatmap_evaluate import extract_peaks

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    ckpt = torch.load(args.weights, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    stride = ckpt.get("stride", 2)

    model = YOLO26HeatmapDetector(stride=stride, num_classes=1, temporal_mode="standard")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = 640
    cfg.data = args.data
    val_dataset = build_yolo_dataset(cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=32)
    val_loader = build_dataloader(val_dataset, batch=args.batch, workers=4, shuffle=False)

    records = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Model Forward @ {args.data}"):
            imgs_tensor = batch["img"].to(device).float() / 255.0
            bboxes = batch["bboxes"]
            b_idx = batch["batch_idx"]
            im_files = batch.get("im_file", [""] * imgs_tensor.shape[0])
            bs = imgs_tensor.shape[0]

            preds = model(imgs_tensor)
            peaks_list = extract_peaks(
                heatmap=preds["heatmap"],
                offset=preds["offset"],
                stride=stride,
                conf_thresh=0.02,  # 下探到 0.02，保留深空极暗弱信号打捞潜力
                top_k=150,
            )

            for b in range(bs):
                mask_b = b_idx == b
                gt_norm = bboxes[mask_b].cpu().numpy()
                im_name = Path(im_files[b]).name if im_files[b] else f"img_{b}"
                gt_pts = []
                for box in gt_norm:
                    gt_pts.append([float(box[0] * 640), float(box[1] * 640)])
                gt_pts = np.array(gt_pts, dtype=np.float32) if len(gt_pts) > 0 else np.zeros((0, 2), dtype=np.float32)

                # Compute normalized variance if cfar gating is requested
                var_map = None
                if getattr(args, "cfar_gating", False):
                    ch0 = (imgs_tensor[b, 0] * 255.0).byte().cpu().numpy()
                    gray_f = ch0.astype(np.float32)
                    mean = cv2.blur(gray_f, (15, 15))
                    mean_sq = cv2.blur(gray_f ** 2, (15, 15))
                    var = np.maximum(mean_sq - mean ** 2, 0.0)
                    std_dev = np.sqrt(var)
                    var_map = np.clip((std_dev - 2.5) / 10.0, 0.0, 1.0)

                records.append({
                    "im_name": im_name,
                    "gt_pts": gt_pts,
                    "pred_points": peaks_list[b]["points"].astype(np.float32),
                    "pred_scores": peaks_list[b]["scores"].astype(np.float32),
                    "var_map": var_map,
                })

    if getattr(args, "save_cache", ""):
        save_p = Path(args.save_cache)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        with open(save_p, "wb") as f:
            pickle.dump(records, f)
        print(colorstr("bold", colorstr("green", f"\n[SUCCESS] Cached inferences saved to: {save_p.resolve()}\n")))
    return records


def main():
    args = parse_args()

    records = load_or_create_records(args)

    # Group by sequence
    seq_records: Dict[str, List[Dict]] = {}
    for r in records:
        seq = extract_seq_name(r["im_name"])
        seq_records.setdefault(seq, []).append(r)

    filter_seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]

    tracker_config = {
        "max_age": args.max_age,
        "min_hits": args.min_hits,
        "match_dist": args.match_dist,
        "output_coasting": args.output_coasting,
        "min_track_score": args.min_track_score,
        "instant_conf": args.instant_conf,
        "min_displacement": args.min_disp,
    }

    # Grand totals
    grand_stats = {
        "baseline": {"tp": 0, "fp": 0, "gt": 0},
        "spatial_only": {"tp": 0, "fp": 0, "gt": 0},
        "tracker_only": {"tp": 0, "fp": 0, "gt": 0},
        "fusion": {"tp": 0, "fp": 0, "gt": 0},
    }

    print("=" * 115)
    print(f"{'Sequence Name':<28} | {'Mode':<18} | {'TP / GT':<14} | {'FP':<6} | {'Recall':<8} | {'Prec':<8} | {'F1-Score':<8}")
    print("=" * 115)

    all_keys = sorted(seq_records.keys())
    for seq_name in all_keys:
        if filter_seqs and not any(f in seq_name for f in filter_seqs):
            continue

        recs = seq_records[seq_name]
        res = evaluate_sequence(
            records=recs,
            dist_thresh=args.dist_thresh,
            th_base=args.th_base,
            th_salvage=args.th_salvage,
            th_ground=args.th_ground,
            sky_ratio=args.sky_ratio,
            img_h=640,
            tracker_config=tracker_config,
        )

        for k in grand_stats:
            grand_stats[k]["tp"] += int(res[k]["tp"])
            grand_stats[k]["fp"] += int(res[k]["fp"])
            grand_stats[k]["gt"] += int(res[k]["gt"])

        m_base = res["baseline"]
        m_spat = res["spatial_only"]
        m_trk = res["tracker_only"]
        m_fuse = res["fusion"]

        print(f"{seq_name:<28} | {'1. Gold Base (0.30)':<18} | {int(m_base['tp']):>5} / {int(m_base['gt']):<6} | {int(m_base['fp']):<6} | {m_base['recall']:>6.2f}% | {m_base['precision']:>6.2f}% | {m_base['f1']:>6.4f}")
        print(f"{'':<28} | {'2. Spatial Gating':<18} | {int(m_spat['tp']):>5} / {int(m_spat['gt']):<6} | {int(m_spat['fp']):<6} | {m_spat['recall']:>6.2f}% | {m_spat['precision']:>6.2f}% | {m_spat['f1']:>6.4f}")
        print(f"{'':<28} | {'3. Kinematic Trk':<18} | {int(m_trk['tp']):>5} / {int(m_trk['gt']):<6} | {int(m_trk['fp']):<6} | {m_trk['recall']:>6.2f}% | {m_trk['precision']:>6.2f}% | {m_trk['f1']:>6.4f}")
        print(f"{'':<28} | {colorstr('bold', colorstr('green', '4. Fusion (2 + 5)')):<27} | {int(m_fuse['tp']):>5} / {int(m_fuse['gt']):<6} | {int(m_fuse['fp']):<6} | {m_fuse['recall']:>6.2f}% | {m_fuse['precision']:>6.2f}% | {m_fuse['f1']:>6.4f}")
        print("-" * 115)

    # Print Grand Overall Results
    print("=" * 115)
    print(colorstr("bold", f"GRAND OVERALL RESULTS ACROSS EVALUATED SEQUENCES (Distance <= {args.dist_thresh:.1f}px)"))
    print("=" * 115)

    grand_metrics = {}
    for k in grand_stats:
        grand_metrics[k] = calc_metrics(grand_stats[k]["tp"], grand_stats[k]["fp"], grand_stats[k]["gt"])

    for mode_name, key in [
        ("1. Gold Standard Baseline (Optimal th=0.30)", "baseline"),
        ("2. Pure Spatial Prior Gating", "spatial_only"),
        ("3. Pure Kinematic Filtering", "tracker_only"),
        ("4. FUSION: Spatial Gating + Kinematics", "fusion"),
    ]:
        gm = grand_metrics[key]
        far = gm["fp"] / max(1, len(records))
        line = f"{mode_name:<38} | TP: {gm['tp']:>5}/{gm['gt']:<5} | FP: {gm['fp']:<6} | Recall: {gm['recall']:>6.2f}% | Prec: {gm['precision']:>6.2f}% | F1: {gm['f1']:>6.4f} | FAR: {far:.4f}/frame"
        if key == "fusion":
            print(colorstr("bold", colorstr("green", line)))
        else:
            print(line)
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
