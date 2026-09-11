#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Test-Time Augmentation (TTA) in Heatmap Probability Domain for Infrared Tiny UAV Detection.

Core Principles:
1. Pure Inference-Time Enhancement (Zero retraining, zero architecture modifications).
2. TTA Views:
   - View 0: Original input (I_t, Diff, Median)
   - View 1: Horizontal Flip (flip left-right)
   - View 2 (Optional): Vertical Flip (flip up-down)
   - View 3 (Optional): Sub-pixel / 1px circular shift (+1, -1)
3. Probabilistic Heatmap Domain Soft Fusion:
   - Flip / invert predictions back to original coordinate space.
   - Heatmap fusion: Mean, Soft-Max, or Trimmed Mean in probability space [0, 1].
   - Offset fusion: Invert dx / dy signs appropriately, then average.
4. Fast Batch Inference:
   - Evaluates full 24 sequences (31,613 frames, GT = 25,111).
   - Distance tolerance <= 8.0px (GJB / Industry IR Standard).
   - Scans full threshold spectrum [0.10 ~ 0.60] to benchmark against Trial 22 baseline (F1=0.9057).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, colorstr
from manu.heatmap_evaluate import extract_peaks, evaluate_point_detections
from manu.heatmap_model import YOLO26HeatmapDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Heatmap Test-Time Augmentation (TTA)")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/optuna_median_search/trial_0022/weights/best.pt",
        help="Model weights path (default: Trial 22 SOTA)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--tta-mode",
        type=str,
        default="hflip",
        choices=["none", "hflip", "hvflip", "shift"],
        help="TTA mode: 'none' (baseline), 'hflip' (2 views: orig+hflip), 'hvflip' (3 views: orig+hflip+vflip), 'shift' (orig+hflip+shift)",
    )
    parser.add_argument(
        "--fusion-method",
        type=str,
        default="mean",
        choices=["mean", "max", "geom_mean", "top2_mean"],
        help="Fusion operation across TTA views in probability domain",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--stride", type=int, default=2, help="Feature stride (2 for P1 high-res)")
    parser.add_argument("--scale", type=str, default="n", choices=["n", "s"], help="Model scale ('n' for Trial 22, 's' for YOLO26s)")
    parser.add_argument("--batch", type=int, default=32, help="Inference batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index, e.g. '0' or '0,1'")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="TP distance match threshold (px)")
    parser.add_argument("--conf-extract", type=float, default=0.04, help="Low extraction threshold for scanning")
    return parser.parse_args()


def predict_tta(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    tta_mode: str = "hflip",
    fusion: str = "mean",
) -> dict[str, torch.Tensor]:
    """
    Run forward inference with TTA in Heatmap probability domain.

    Args:
        model: Heatmap detector model
        imgs: (B, 3, H, W) normalized image tensor in [0, 1]
        tta_mode: 'none', 'hflip', 'hvflip', 'shift'
        fusion: 'mean', 'max', 'geom_mean', 'top2_mean'

    Returns:
        dict with fused 'heatmap' (B, 1, H_feat, W_feat) and 'offset' (B, 2, H_feat, W_feat)
    """
    if tta_mode == "none":
        return model(imgs)

    heatmaps = []
    offsets = []

    # View 0: Original View
    out_orig = model(imgs)
    heatmaps.append(out_orig["heatmap"])
    offsets.append(out_orig["offset"])

    # View 1: Horizontal Flip
    if tta_mode in ("hflip", "hvflip", "shift"):
        # Flip along W (dimension 3)
        imgs_hflip = torch.flip(imgs, dims=[3])
        out_hflip = model(imgs_hflip)

        # Invert heatmap back to original coordinate system
        hm_hflip_back = torch.flip(out_hflip["heatmap"], dims=[3])

        # Invert offset: dx has sign inverted because pixel relative position is reversed horizontally
        # offset is (B, 2, H_feat, W_feat): offset[:, 0] is dx, offset[:, 1] is dy
        off_hflip_back = torch.flip(out_hflip["offset"], dims=[3])
        # In cell-normalized [0, 1] or relative dx, horizontal flip inverts dx: dx_orig = -dx_flipped
        # Specifically: x_orig = W - 1 - x_flipped. If x_flipped = cell + dx, x_orig = W - 1 - (cell + dx) = (W - 1 - cell) - dx
        off_hflip_back = torch.stack([-off_hflip_back[:, 0], off_hflip_back[:, 1]], dim=1)

        heatmaps.append(hm_hflip_back)
        offsets.append(off_hflip_back)

    # View 2: Vertical Flip (if requested)
    if tta_mode == "hvflip":
        imgs_vflip = torch.flip(imgs, dims=[2])
        out_vflip = model(imgs_vflip)

        hm_vflip_back = torch.flip(out_vflip["heatmap"], dims=[2])
        off_vflip_back = torch.flip(out_vflip["offset"], dims=[2])
        off_vflip_back = torch.stack([off_vflip_back[:, 0], -off_vflip_back[:, 1]], dim=1)

        heatmaps.append(hm_vflip_back)
        offsets.append(off_vflip_back)

    # View 3: Sub-pixel / 1px shift (if requested)
    if tta_mode == "shift":
        # Shift 2 pixels right and down (+2px in image space corresponds to +1px in P1 feature space)
        imgs_shift = torch.roll(imgs, shifts=(2, 2), dims=(2, 3))
        out_shift = model(imgs_shift)

        # Un-shift in feature map (stride=2 -> shift 1 cell)
        hm_shift_back = torch.roll(out_shift["heatmap"], shifts=(-1, -1), dims=(2, 3))
        off_shift_back = torch.roll(out_shift["offset"], shifts=(-1, -1), dims=(2, 3))

        heatmaps.append(hm_shift_back)
        offsets.append(off_shift_back)

    # Stack all views: (NumViews, B, C, H, W)
    stacked_hm = torch.stack(heatmaps, dim=0)
    stacked_off = torch.stack(offsets, dim=0)

    # Fusion Strategy
    if fusion == "mean":
        fused_hm = torch.mean(stacked_hm, dim=0)
        fused_off = torch.mean(stacked_off, dim=0)
    elif fusion == "max":
        fused_hm, max_idx = torch.max(stacked_hm, dim=0)
        # Select offset from the view with maximum heatmap response
        max_idx_off = max_idx.expand(-1, 2, -1, -1).unsqueeze(0)
        fused_off = torch.gather(stacked_off, dim=0, index=max_idx_off).squeeze(0)
    elif fusion == "geom_mean":
        # Geometric mean: exp(mean(log(hm)))
        log_hm = torch.log(torch.clamp(stacked_hm, min=1e-5))
        fused_hm = torch.exp(torch.mean(log_hm, dim=0))
        fused_off = torch.mean(stacked_off, dim=0)
    elif fusion == "top2_mean":
        if stacked_hm.shape[0] >= 2:
            top2_hm, _ = torch.topk(stacked_hm, k=2, dim=0)
            fused_hm = torch.mean(top2_hm, dim=0)
        else:
            fused_hm = stacked_hm[0]
        fused_off = torch.mean(stacked_off, dim=0)
    else:
        fused_hm = torch.mean(stacked_hm, dim=0)
        fused_off = torch.mean(stacked_off, dim=0)

    return {"heatmap": fused_hm, "offset": fused_off}


def main():
    args = parse_args()

    # Determine device
    device_str = args.device.strip()
    gpu_ids = [int(x) for x in device_str.split(",") if x.isdigit()]
    primary_device = torch.device(f"cuda:{gpu_ids[0]}" if (torch.cuda.is_available() and len(gpu_ids) > 0) else "cpu")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        alt_paths = [
            REPO_ROOT / args.weights,
            Path("/tmp/pycharm_project_10ae9e2e") / args.weights,
            Path("/home/manu/mnt/pycharm_project_10ae9e2e") / args.weights,
        ]
        for alt in alt_paths:
            if alt.exists():
                weights_path = alt
                break

    print("=" * 86)
    print(f"🔬 [TTA EVALUATION] Heatmap-Domain Test-Time Augmentation")
    print(f"Model Checkpoint : {weights_path}")
    print(f"TTA Mode         : {args.tta_mode.upper()} | Fusion: {args.fusion_method}")
    print(f"Device           : {primary_device} | Stride: {args.stride} (P1 High-Res)")
    print(f"Match Tolerance  : Distance <= {args.dist_thresh:.1f}px (GJB / Standard IR Protocol)")
    print(f"Dataset          : {args.data}")
    print("=" * 86)

    # 1. Build Model
    model = YOLO26HeatmapDetector(
        stride=args.stride,
        num_classes=1,
        weights=weights_path if weights_path.exists() else None,
        scale=args.scale,
    )
    model.to(primary_device)
    model.eval()

    # 2. Build Dataset & Dataloader
    data_dict = check_det_dataset(args.data)
    val_path = data_dict["val"]

    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    val_dataset = build_yolo_dataset(cfg, img_path=val_path, batch=args.batch, data=data_dict, mode="val", rect=False)
    val_loader = build_dataloader(val_dataset, batch=args.batch, workers=args.workers, shuffle=False)

    print(f"[DATA] Loaded {len(val_dataset)} validation frames.")

    # 3. Inference with TTA
    all_peaks_raw = []
    val_gt_list = []
    val_sizes_list = []

    t0 = time.time()
    pbar = tqdm(val_loader, desc=f"TTA [{args.tta_mode}+{args.fusion_method}]", dynamic_ncols=True)

    with torch.no_grad():
        for batch in pbar:
            imgs = batch["img"].to(primary_device, non_blocking=True).float() / 255.0
            bboxes = batch["bboxes"]
            b_idx = batch["batch_idx"]
            bs = imgs.shape[0]

            preds = predict_tta(
                model=model,
                imgs=imgs,
                tta_mode=args.tta_mode,
                fusion=args.fusion_method,
            )

            peaks = extract_peaks(
                heatmap=preds["heatmap"],
                offset=preds["offset"],
                stride=args.stride,
                conf_thresh=args.conf_extract,
                top_k=100,
            )
            all_peaks_raw.extend(peaks)

            b_idx_cpu = b_idx.long().cpu().view(-1)
            bboxes_cpu = bboxes.cpu().numpy()
            for b in range(bs):
                mask_b = (b_idx_cpu == b).numpy()
                gt_b = bboxes_cpu[mask_b] if mask_b.any() else np.zeros((0, 4), dtype=np.float32)
                val_gt_list.append(gt_b)
                val_sizes_list.append((args.imgsz, args.imgsz))

    total_infer_time = time.time() - t0
    fps = len(val_dataset) / max(total_infer_time, 0.1)
    print(f"\n[INFERENCE] Completed in {total_infer_time:.1f}s ({fps:.1f} FPS)\n")

    # 4. Sweep full threshold spectrum
    thresholds = [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50]
    best_f1 = 0.0
    best_th = 0.25
    best_metrics = {}

    print("=" * 86)
    print(f"{'Threshold':<11} | {'Recall':<11} | {'Precision':<11} | {'F1-Score':<11} | {'TP':<7} | {'FP':<7} | {'GT':<7}")
    print("-" * 86)

    for th in thresholds:
        th_preds = []
        for p in all_peaks_raw:
            keep = p["scores"] >= th
            th_preds.append({
                "points": p["points"][keep],
                "scores": p["scores"][keep],
            })

        m = evaluate_point_detections(
            predictions=th_preds,
            gt_boxes_list=val_gt_list,
            img_sizes=val_sizes_list,
            distance_threshold=args.dist_thresh,
        )

        rec = m["recall"]
        prec = m["precision"]
        f1 = m["f1"]

        mark = ""
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_metrics = m
            mark = " ★ (Best)"

        print(f"th = {th:<6.2f} | {rec * 100:<9.2f}% | {prec * 100:<9.2f}% | {f1:<11.4f} | {m['tp']:<7} | {m['fp']:<7} | {m['total_gt']:<7}{mark}")

    print("=" * 86)
    print(colorstr("bold", colorstr("green", f"\n[SUMMARY RESULT] TTA Mode: {args.tta_mode} (Fusion: {args.fusion_method})")))
    print(f"Target Baseline (Trial 22 Single-View) : F1 = 0.9057 | Recall = 86.11% | Precision = 95.51% | FP = 1012")
    print(f"TTA Evaluated Peak                    : F1 = {best_f1:.4f} (@ th={best_th:.2f}) | Recall = {best_metrics.get('recall', 0.0)*100:.2f}% | Precision = {best_metrics.get('precision', 0.0)*100:.2f}%")
    delta_f1 = best_f1 - 0.9057
    color_code = "green" if delta_f1 >= 0 else "yellow"
    print(colorstr(color_code, f"Delta vs Single-View Baseline         : {delta_f1:+.4f} (TP: {best_metrics.get('tp', 0)}, FP: {best_metrics.get('fp', 0)})\n"))


if __name__ == "__main__":
    main()
