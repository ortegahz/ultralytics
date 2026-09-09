#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Temporal Median Background Residual Probe for Infrared Ultra-Weak UAVs.

Key Objectives:
1. Replaces channel 3 with aligned long-term temporal median background residual:
     Input: [I_t, |I_t - W(I_{t-2})|, (I_t - B_t)^+]
   where B_t = median{ W(I_{t-k}) } over a sliding window (e.g., N=16 or 24 frames).
2. Uses pure GMC alignment so camera jitters are compensated before median computation.
3. Directly loads champion checkpoint (trial_0031) with ZERO training.
4. Evaluates whether the ultimate dead-zone sequence (wg2022_ir_020_split_03) gets resurrected:
   - Target diff magnitude (Delta I)
   - GT Heatmap response score (0.00 -> ?)
   - Target ranking vs full-frame background peak noise.

Usage on Server:
    python manu/probe_temporal_median_diff.py \
        --weights runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt \
        --seq wg2022_ir_020_split_03 \
        --window 21 \
        --max-frames 60 \
        --device 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.utils import colorstr
from manu.heatmap_model import YOLO26HeatmapDetector

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def natural_sort_key(path: Path | str):
    stem = Path(path).stem
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", stem)]


def letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
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
                print(f"[WARN] Failed to read {json_path}: {e}")
    return {}


class FastGMCAligner:
    """Lightweight Optical Flow GMC Aligner to warp historical frames into current frame coordinates."""

    def __init__(self, downscale: int = 2):
        self.downscale = downscale
        self.feature_params = {
            "maxCorners": 600,
            "qualityLevel": 0.01,
            "minDistance": 4,
            "blockSize": 3,
        }

    def compute_affine(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        h, w = curr_gray.shape[:2]
        ds = self.downscale
        H = np.eye(2, 3, dtype=np.float32)

        if ds > 1:
            p_small = cv2.resize(prev_gray, (w // ds, h // ds))
            c_small = cv2.resize(curr_gray, (w // ds, h // ds))
        else:
            p_small, c_small = prev_gray, curr_gray

        pts_prev = cv2.goodFeaturesToTrack(p_small, mask=None, **self.feature_params)
        if pts_prev is not None and len(pts_prev) >= 6:
            pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
                p_small, c_small, pts_prev, None, winSize=(15, 15), maxLevel=2
            )
            good = status.ravel() == 1
            p0 = pts_prev[good]
            p1 = pts_curr[good]
            if len(p0) >= 6:
                M, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if M is not None:
                    H = M.astype(np.float32)
                    if ds > 1:
                        H[0, 2] *= ds
                        H[1, 2] *= ds
        return H

    def warp(self, img: np.ndarray, H: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        return cv2.warpAffine(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def parse_args():
    parser = argparse.ArgumentParser(description="Probe temporal median background subtraction on infrared weak targets")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt",
        help="Path to trial_0031 checkpoint",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/anti-uav",
        help="Root path containing raw sequence videos",
    )
    parser.add_argument("--seq", type=str, default="wg2022_ir_020_split_03", help="Target sequence to probe")
    parser.add_argument("--window", type=int, default=21, help="Temporal sliding window size for median background (e.g. 15, 21, 29)")
    parser.add_argument("--stride-step", type=int, default=2, help="Temporal sampling step in window (default: 2, e.g. t, t-2, t-4...)")
    parser.add_argument("--max-frames", type=int, default=60, help="Number of frames to evaluate")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 105)
    print("   Temporal Median Background Residual Probe for Extreme Low-SCR Targets")
    print(f"   Checkpoint: {args.weights} | Target: {args.seq} | Window: {args.window} frames")
    print("=" * 105)

    # 1. Resolve raw directory
    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        for cand in [
            Path("/mnt/data/siping/datasets/manu/anti-uav"),
            Path("/mnt/data/siping/datasets/anti-uav"),
            Path("/home/manu/mnt/datasets/manu/anti-uav"),
            Path("/media/manu/1TB-Volume/data/anti-uav"),
        ]:
            if cand.exists():
                raw_root = cand
                break

    seq_dir = raw_root / args.seq
    if not seq_dir.is_dir():
        for p in raw_root.rglob(args.seq):
            if p.is_dir():
                seq_dir = p
                break

    if not seq_dir.is_dir():
        print(f"[ERROR] Sequence directory '{args.seq}' not found in {raw_root}")
        sys.exit(1)

    print(f"[INFO] Sequence path: {seq_dir}")
    gt_dict = load_gt_annotations(seq_dir)
    print(f"[INFO] Loaded GT annotations: {len(gt_dict)} frames")

    image_paths = sorted(
        [p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
        key=natural_sort_key,
    )
    total_imgs = len(image_paths)
    print(f"[INFO] Found {total_imgs} total frames in sequence.")

    # 2. Load trial_0031 model
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    weights_path = Path(args.weights)
    if not weights_path.exists():
        alt = REPO_ROOT / args.weights
        if alt.exists():
            weights_path = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()

    model = YOLO26HeatmapDetector(stride=args.stride, num_classes=1, temporal_mode="standard")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Find test interval with valid GTs
    valid_indices = sorted(list(gt_dict.keys()))
    if not valid_indices:
        print("[ERROR] No valid GT found in annotations!")
        sys.exit(1)

    # Pick start frame ensuring sufficient history window
    eval_start = None
    for idx in valid_indices:
        if idx >= args.window * args.stride_step + 5:
            eval_start = idx
            break
    if eval_start is None:
        eval_start = valid_indices[0]

    eval_end = min(total_imgs, eval_start + args.max_frames)
    print(f"[INFO] Evaluation Segment: Frame {eval_start} -> {eval_end - 1} ({eval_end - eval_start} frames)")

    # Pre-cache gray frames
    buf_start = max(0, eval_start - args.window * args.stride_step - 5)
    print(f"[INFO] Caching raw frames ({buf_start} ~ {eval_end})...")
    gray_cache: dict[int, np.ndarray] = {}
    for idx in range(buf_start, eval_end):
        if 0 <= idx < total_imgs:
            im = cv2.imread(str(image_paths[idx]), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                gray_cache[idx] = im

    aligner = FastGMCAligner(downscale=2)

    # 4. Probe two modes on the exact same frames:
    # Mode A: Standard Baseline 3-ch [I_t, |I_t - W(I_{t-1})|, |I_t - W(I_{t-2})|]
    # Mode B: Temporal Median Residual 3-ch [I_t, |I_t - W(I_{t-2})|, (I_t - B_t)^+]
    records_base = []
    records_median = []

    print("\n" + "=" * 115)
    print(f"{'Frame':<6} | {'Target Delta-I':<15} | {'Mode A (Baseline) Score':<25} | {'Mode B (Median) Score':<25} | {'Gain / Ratio':<15}")
    print("-" * 115)

    for f_idx in range(eval_start, eval_end):
        if f_idx not in gt_dict or f_idx not in gray_cache:
            continue

        im_curr = gray_cache[f_idx]
        h_orig, w_orig = im_curr.shape[:2]
        gx, gy, gw, gh = gt_dict[f_idx]
        cx, cy = int(round(gx + gw / 2.0)), int(round(gy + gh / 2.0))

        # Fetch short history
        im_prev1 = gray_cache.get(f_idx - 1, im_curr)
        im_prev2 = gray_cache.get(f_idx - 2, im_prev1)

        # 1. Mode A: Standard GMC Short Differences
        H1 = aligner.compute_affine(im_prev1, im_curr)
        H2 = aligner.compute_affine(im_prev2, im_curr)
        diff1_gmc = cv2.absdiff(im_curr, aligner.warp(im_prev1, H1))
        diff2_gmc = cv2.absdiff(im_curr, aligner.warp(im_prev2, H2))
        inp_base = np.stack([im_curr, diff1_gmc, diff2_gmc], axis=-1)

        # 2. Mode B: Long-term Temporal Median Background Subtraction
        history_frames = []
        for step in range(1, args.window + 1):
            hist_idx = f_idx - step * args.stride_step
            if hist_idx in gray_cache:
                im_hist = gray_cache[hist_idx]
                H_hist = aligner.compute_affine(im_hist, im_curr)
                im_warped = aligner.warp(im_hist, H_hist)
                history_frames.append(im_warped)

        if len(history_frames) >= 5:
            hist_stack = np.stack(history_frames, axis=0)  # (N, H, W)
            bg_median = np.median(hist_stack, axis=0).astype(np.float32)
            # Positive residual: only targets brighter than median background
            res_pos = np.clip(im_curr.astype(np.float32) - bg_median, 0, 255).astype(np.uint8)
        else:
            res_pos = diff2_gmc

        inp_median = np.stack([im_curr, diff2_gmc, res_pos], axis=-1)

        # Compute physical contrast Delta I at target center
        patch_curr = im_curr[max(0, cy - 2):min(h_orig, cy + 3), max(0, cx - 2):min(w_orig, cx + 3)]
        patch_diff2 = diff2_gmc[max(0, cy - 2):min(h_orig, cy + 3), max(0, cx - 2):min(w_orig, cx + 3)]
        patch_res = res_pos[max(0, cy - 2):min(h_orig, cy + 3), max(0, cx - 2):min(w_orig, cx + 3)]

        d_short = float(np.max(patch_diff2)) if patch_diff2.size else 0.0
        d_med = float(np.max(patch_res)) if patch_res.size else 0.0

        # Run inference in a 2-sample batch
        lb_a, r_a, (dw_a, dh_a) = letterbox(inp_base, (args.imgsz, args.imgsz))
        lb_b, _, _ = letterbox(inp_median, (args.imgsz, args.imgsz))

        batch_t = torch.from_numpy(np.stack([lb_a, lb_b], axis=0)).permute(0, 3, 1, 2).float() / 255.0
        batch_t = batch_t.to(device)

        with torch.no_grad():
            preds = model(batch_t)
            hm_a = preds["heatmap"][0, 0].cpu().numpy()
            hm_b = preds["heatmap"][1, 0].cpu().numpy()

        feat_h, feat_w = hm_a.shape
        gx_feat = int(round(((gx + gw / 2.0) * r_a + dw_a) / args.stride))
        gy_feat = int(round(((gy + gh / 2.0) * r_a + dh_a) / args.stride))

        rad = 3
        y1, y2 = max(0, gy_feat - rad), min(feat_h, gy_feat + rad + 1)
        x1, x2 = max(0, gx_feat - rad), min(feat_w, gx_feat + rad + 1)

        sc_a = float(np.max(hm_a[y1:y2, x1:x2])) if hm_a[y1:y2, x1:x2].size else 0.0
        sc_b = float(np.max(hm_b[y1:y2, x1:x2])) if hm_b[y1:y2, x1:x2].size else 0.0

        # Background peak excluding target
        mask_bg = np.ones_like(hm_a, dtype=bool)
        mask_bg[max(0, gy_feat - 6):min(feat_h, gy_feat + 7), max(0, gx_feat - 6):min(feat_w, gx_feat + 7)] = False
        bg_max_a = float(np.max(hm_a[mask_bg]))
        bg_max_b = float(np.max(hm_b[mask_bg]))

        rank_a = int(np.sum(hm_a > sc_a)) + 1
        rank_b = int(np.sum(hm_b > sc_b)) + 1

        records_base.append({"score": sc_a, "bg_max": bg_max_a, "rank": rank_a, "delta": d_short})
        records_median.append({"score": sc_b, "bg_max": bg_max_b, "rank": rank_b, "delta": d_med})

        gain_str = f"{sc_b - sc_a:+.4f}"
        if sc_b > sc_a:
            gain_str = colorstr("green", gain_str)

        print(
            f"{f_idx:<6d} | Short:{d_short:3.1f} Med:{d_med:3.1f} | "
            f"Score: {sc_a:.5f} (Bg:{bg_max_a:.3f}) | "
            f"Score: {sc_b:.5f} (Bg:{bg_max_b:.3f}) | {gain_str}"
        )

    print("=" * 115)
    mean_sc_a = float(np.mean([r["score"] for r in records_base]))
    mean_sc_b = float(np.mean([r["score"] for r in records_median]))
    max_sc_a = float(np.max([r["score"] for r in records_base]))
    max_sc_b = float(np.max([r["score"] for r in records_median]))
    mean_bg_a = float(np.mean([r["bg_max"] for r in records_base]))
    mean_bg_b = float(np.mean([r["bg_max"] for r in records_median]))

    print(colorstr("bold", "\n【时域中值残差 vs 短时差分 对比总结】"))
    print(f"1. 原始基准 (Mode A - 短差分) : 平均GT得分 = {mean_sc_a:.5f} | 最高得分 = {max_sc_a:.5f} | 背景假警峰值 = {mean_bg_a:.5f}")
    print(f"2. 时域中值 (Mode B - 中值残差): 平均GT得分 = {mean_sc_b:.5f} | 最高得分 = {max_sc_b:.5f} | 背景假警峰值 = {mean_bg_b:.5f}")

    if mean_sc_b > mean_sc_a:
        diff_pct = ((mean_sc_b - mean_sc_a) / max(1e-6, mean_sc_a)) * 100.0
        print(colorstr("bold", colorstr("green", f"\n>>> [SUCCESS] 目标响应获得正向提升: +{diff_pct:.1f}% (得分由 {mean_sc_a:.4f} 提升至 {mean_sc_b:.4f})")))
        if mean_sc_b >= 0.10:
            print(colorstr("bold", colorstr("green", ">>> [突破定性] 目标已越过亚门限并激活！长时中值背景彻底打破了短差分自抵消盲区！")))
    else:
        print(colorstr("yellow", "\n>>> [OBSERVATION] 目标响应未能显著激活，说明模型卷积核强依赖短差分的特定梯度分布，需极轻量微调适配。"))
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
