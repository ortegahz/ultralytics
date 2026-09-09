#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Zero-Training GMC Alignment Evaluation on Hard Case DJI_0051_2.

Evaluates YOLO26HeatmapDetector (trial_0031) on sequence DJI_0051_2 with:
  Mode 1 (Baseline)  : [I_t, |I_t - I_{t-1}|, |I_t - I_{t-2}|] (Raw unaligned difference)
  Mode 2 (GMC-Sparse): [I_t, |I_t - W(I_{t-1})|, |I_t - W(I_{t-2})|] (LK Optical Flow Affine)
  Mode 3 (GMC-ECC)   : [I_t, |I_t - W(I_{t-1})|, |I_t - W(I_{t-2})|] (ECC Correlation Affine)

Verifies whether camera vibration false alarms (270+ FPs) can be eliminated mathematically
without retraining the neural network.

Usage on Server:
    python manu/eval_gmc_dji0051.py \
        --weights runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt \
        --raw-root /mnt/data/siping/datasets/manu/anti-uav \
        --device 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add repository root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manu.heatmap_model import YOLO26HeatmapDetector
from manu.heatmap_evaluate import extract_peaks, evaluate_point_detections


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GMC Alignment on DJI_0051_2 sequence")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt",
        help="Path to trained trial_0031 heatmap weights",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/anti-uav",
        help="Root directory of raw anti-uav sequences",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default="DJI_0051_2",
        help="Target sequence name",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    parser.add_argument("--stride", type=int, default=2, help="Feature stride (2 for P1 trial_0031)")
    parser.add_argument("--conf-thresh", type=float, default=0.20, help="Heatmap peak detection threshold")
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="Tolerance distance (GJB 8.0px standard)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--save-vis", action="store_true", default=True, help="Save qualitative visual comparisons")
    parser.add_argument("--output-dir", type=str, default="runs/gmc_eval", help="Output directory for diagnostics")
    return parser.parse_args()


def natural_sort_key(p: Path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", p.stem)]


def letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = img.shape[:2]  # [h, w]
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


def find_sequence_dir(raw_root: Path, seq_name: str) -> Path | None:
    direct = raw_root / seq_name
    if direct.is_dir():
        return direct
    for p in raw_root.rglob(seq_name):
        if p.is_dir():
            return p
    return None


def load_gt_annotations(seq_dir: Path) -> dict[int, list[float]]:
    """Loads ground-truth bounding boxes [x, y, w, h] from sequence json."""
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
                print(f"[WARN] Failed to parse annotation {json_path}: {e}")
    return {}


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
        """Estimates 2x3 Affine transformation matrix mapping prev -> curr."""
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
                M, inliers = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if M is not None:
                    H = M.astype(np.float32)
                    if ds > 1:
                        H[0, 2] *= ds
                        H[1, 2] *= ds

        elif self.method == "ecc":
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
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
    gmc_estimator: GlobalMotionEstimator | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        diff: Absolute difference |curr - W(prev)|
        warped_prev: Warped previous frame
    """
    h, w = curr_gray.shape[:2]
    if gmc_estimator is None:
        return cv2.absdiff(curr_gray, prev_gray), prev_gray

    H = gmc_estimator.compute_affine(prev_gray, curr_gray)
    warped_prev = cv2.warpAffine(
        prev_gray,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    diff = cv2.absdiff(curr_gray, warped_prev)
    return diff, warped_prev


def run_evaluation_mode(
    mode_name: str,
    gmc_method: str | None,
    image_paths: list[Path],
    gt_dict: dict[int, list[float]],
    model: YOLO26HeatmapDetector,
    args,
    device: torch.device,
    save_vis_indices: set[int],
    vis_dir: Path,
) -> dict:
    """Runs sequence evaluation under a specific differential mode."""
    total_imgs = len(image_paths)
    estimator = GlobalMotionEstimator(method=gmc_method, downscale=2) if gmc_method else None

    # Cache for grayscale raw frames
    frame_cache: dict[int, np.ndarray] = {}

    all_preds = []
    all_gts_norm = []
    all_sizes = []

    diff1_energies = []
    diff2_energies = []

    t0 = time.time()
    for i, p in enumerate(image_paths):
        # Extract frame index
        m = re.search(r"(\d+)$", p.stem)
        f_idx = int(m.group(1)) if m else i

        im_curr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im_curr is None:
            continue
        h_orig, w_orig = im_curr.shape[:2]
        frame_cache[f_idx] = im_curr

        # Fetch history frames with clamp fallback
        f_prev1 = f_idx - 1 if (f_idx - 1) in frame_cache else f_idx
        f_prev2 = f_idx - 2 if (f_idx - 2) in frame_cache else f_prev1

        im_prev1 = frame_cache[f_prev1]
        im_prev2 = frame_cache[f_prev2]

        # Compute differences (Baseline vs GMC)
        diff1, warped_p1 = align_and_diff(im_curr, im_prev1, estimator)
        diff2, warped_p2 = align_and_diff(im_curr, im_prev2, estimator)

        diff1_energies.append(float(np.mean(diff1)))
        diff2_energies.append(float(np.mean(diff2)))

        # Assemble 3-channel input: [I_t, diff1, diff2]
        merged_3ch = np.stack([im_curr, diff1, diff2], axis=-1)

        # Letterbox and forward
        img_lb, r, (dw, dh) = letterbox(merged_3ch, (args.imgsz, args.imgsz))
        img_t = torch.from_numpy(img_lb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_t = img_t.to(device)

        with torch.no_grad():
            preds_dict = model(img_t)
            hm = preds_dict["heatmap"]
            offset = preds_dict["offset"]
            peaks = extract_peaks(
                hm,
                offset,
                stride=args.stride,
                conf_thresh=args.conf_thresh,
                top_k=100,
            )[0]

        # Inverse letterbox back to original image space
        pts = peaks["points"]
        if len(pts) > 0:
            pts[:, 0] = (pts[:, 0] - dw) / r
            pts[:, 1] = (pts[:, 1] - dh) / r
            # Clip to valid bounds
            pts[:, 0] = np.clip(pts[:, 0], 0, w_orig - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h_orig - 1)
            peaks["points"] = pts

        all_preds.append(peaks)
        all_sizes.append((h_orig, w_orig))

        # Ground truth bounding box for this frame
        if f_idx in gt_dict:
            gx, gy, gw, gh = gt_dict[f_idx]
            cx, cy = gx + gw / 2.0, gy + gh / 2.0
            norm_box = np.array([[cx / w_orig, cy / h_orig, gw / w_orig, gh / h_orig]], dtype=np.float32)
            all_gts_norm.append(norm_box)
        else:
            all_gts_norm.append(np.zeros((0, 4), dtype=np.float32))

        # Save diagnostic visualization for selected key frames
        if args.save_vis and i in save_vis_indices:
            vis_canvas = np.zeros((h_orig, w_orig * 3), dtype=np.uint8)
            vis_canvas[:, :w_orig] = im_curr
            vis_canvas[:, w_orig:w_orig * 2] = diff1
            vis_canvas[:, w_orig * 2:] = diff2
            cv2.putText(vis_canvas, f"Mode: {mode_name} | Frame {f_idx}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
            cv2.putText(vis_canvas, "Raw Infrared (I_t)", (20, h_orig - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            cv2.putText(vis_canvas, "Diff1 (|I_t - W(I_t-1)|)", (w_orig + 20, h_orig - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            cv2.putText(vis_canvas, "Diff2 (|I_t - W(I_t-2)|)", (w_orig * 2 + 20, h_orig - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            out_p = vis_dir / f"{mode_name}_frame_{f_idx:04d}.jpg"
            cv2.imwrite(str(out_p), vis_canvas)

        # Evict old frames from cache to conserve memory
        if (f_idx - 5) in frame_cache:
            del frame_cache[f_idx - 5]

    eval_time = time.time() - t0

    # Calculate metrics at standard Distance <= args.dist_thresh
    metrics = evaluate_point_detections(
        all_preds,
        all_gts_norm,
        all_sizes,
        distance_threshold=args.dist_thresh,
    )
    metrics["avg_diff1_energy"] = float(np.mean(diff1_energies))
    metrics["avg_diff2_energy"] = float(np.mean(diff2_energies))
    metrics["fps"] = float(total_imgs / (eval_time + 1e-6))
    return metrics


def main():
    args = parse_args()
    print("=" * 90)
    print("   UAV Tiny Object Detection: Zero-Training GMC Alignment Evaluation")
    print(f"   Target Sequence: {args.seq} | Weights: {args.weights}")
    print("=" * 90)

    # 1. Resolve raw dataset path
    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        candidates = [
            Path("/mnt/data/siping/datasets/manu/anti-uav"),
            Path("/mnt/data/siping/datasets/anti-uav"),
            Path("/home/manu/mnt/datasets/manu/anti-uav"),
            Path("/home/manu/mnt/datasets/anti-uav"),
            Path("/media/manu/1TB-Volume/data/anti-uav"),
        ]
        for c in candidates:
            if c.exists():
                raw_root = c
                break

    seq_dir = find_sequence_dir(raw_root, args.seq)
    if seq_dir is None:
        print(f"[ERROR] Sequence directory '{args.seq}' not found under {raw_root}")
        sys.exit(1)
    print(f"[INFO] Found sequence path: {seq_dir}")

    # 2. Collect image paths
    image_paths = sorted(
        [p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
        key=natural_sort_key,
    )
    total_frames = len(image_paths)
    if total_frames == 0:
        print(f"[ERROR] No valid images found in {seq_dir}")
        sys.exit(1)
    print(f"[INFO] Total frames found: {total_frames}")

    # 3. Load GT annotations
    gt_dict = load_gt_annotations(seq_dir)
    print(f"[INFO] Loaded {len(gt_dict)} Ground Truth annotations")

    # 4. Load Model
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[INFO] Using Device: {device}")

    weights_p = Path(args.weights)
    if not weights_p.exists():
        print(f"[ERROR] Checkpoint weights not found: {weights_p}")
        sys.exit(1)

    print(f"[INFO] Loading trial_0031 checkpoint: {weights_p}...")
    model = YOLO26HeatmapDetector(stride=args.stride, num_classes=1, temporal_mode="standard")
    ckpt = torch.load(weights_p, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Pick representative key frames for qualitative diff comparison
    save_vis_indices = {total_frames // 4, total_frames // 2, 3 * total_frames // 4}
    vis_dir = Path(args.output_dir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 5. Run Evaluations
    modes = [
        ("Baseline (Unaligned)", None),
        ("GMC (SparseOptFlow)", "sparseOptFlow"),
        ("GMC (ECC-Euclidean)", "ecc"),
    ]

    results_table = []
    print("\n[INFO] Starting comparative evaluation across modes...")

    for mode_name, gmc_method in modes:
        print(f"\n---> Running mode: {mode_name} ...")
        res = run_evaluation_mode(
            mode_name,
            gmc_method,
            image_paths,
            gt_dict,
            model,
            args,
            device,
            save_vis_indices,
            vis_dir,
        )
        res["mode"] = mode_name
        results_table.append(res)
        print(f"     [Result] Recall: {res['recall'] * 100:5.2f}% | "
              f"Precision: {res['precision'] * 100:5.2f}% | "
              f"F1: {res['f1']:6.4f} | "
              f"TP: {res['tp']:3d} | FP: {res['fp']:3d} | "
              f"Diff1 Energy: {res['avg_diff1_energy']:5.2f} | Speed: {res['fps']:5.1f} FPS")

    # 6. Print Summary Table
    print("\n" + "=" * 105)
    print(f"             FINAL COMPARATIVE REPORT ON {args.seq} (Tol <= {args.dist_thresh}px, conf={args.conf_thresh:.2f})")
    print("=" * 105)
    header = f"{'Differential Mode':<24} | {'Recall':<9} | {'Precision':<9} | {'F1-Score':<8} | {'TP':<5} | {'FP':<5} | {'Diff1 Energy':<12} | {'FPS':<6}"
    print(header)
    print("-" * 105)

    base_fp = results_table[0]["fp"]
    for r in results_table:
        fp_change = f"({r['fp'] - base_fp:+d})" if r != results_table[0] else ""
        fp_str = f"{r['fp']} {fp_change}"
        print(f"{r['mode']:<24} | "
              f"{r['recall'] * 100:6.2f}%  | "
              f"{r['precision'] * 100:6.2f}%  | "
              f"{r['f1']:8.4f} | "
              f"{r['tp']:<5d} | "
              f"{fp_str:<12} | "
              f"{r['avg_diff1_energy']:12.2f} | "
              f"{r['fps']:5.1f}")
    print("=" * 105)

    # Save summary json
    summary_path = Path(args.output_dir) / f"{args.seq}_gmc_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)
    print(f"[INFO] Detailed summary saved to: {summary_path.resolve()}")
    if args.save_vis:
        print(f"[INFO] Diagnostic visual comparison frames saved to: {vis_dir.resolve()}")


if __name__ == "__main__":
    main()
