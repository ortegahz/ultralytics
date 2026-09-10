#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Zero-Training Probe & Diagnostics for Dual-Domain Residual Mode (Stage 2 Verification).

Compares:
1. Baseline: Trial 22 on official uav_gmc_median (Distance <= 8.0px, 31,613 frames)
2. Dual-Residual Probe: Trial 22 zero-shot on uav_dual_residual [TopHat, ShortDiff, MedianRes]

Outputs:
1. Full Threshold Scanning (0.05 ~ 0.60) across all 31,613 validation frames.
2. Per-sequence Breakdown on the 4 key hard cases:
   - wg2022_ir_011_split_03 (Weak pulse cutoff)
   - DJI_0175_2 (Threshold transition)
   - DJI_0051_2 (Shake & ground clutter)
   - wg2022_ir_020_split_03 (SCR < 1.0 ultimate dead zone)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, colorstr
from manu.heatmap_evaluate import evaluate_point_detections, extract_peaks
from manu.heatmap_model import YOLO26HeatmapDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-training Probe for Dual-Residual Mode")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/optuna_median_search/trial_0022/weights/best.pt",
        help="Path to Trial 22 champion checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_dual_residual/data.yaml",
        help="Path to uav_dual_residual data.yaml",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    return parser.parse_args()


def parse_seq_name(im_path_str: str) -> str:
    stem = Path(im_path_str).stem
    if "___" in stem:
        return stem.split("___")[0]
    if "__" in stem:
        return stem.split("__")[0]
    match = re.search(r"^(.*?)(?:[_-]+)?\d{3,}$", stem)
    return match.group(1).rstrip("_-") if match else stem


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        alt = PROJECT_ROOT / args.weights
        if alt.exists():
            weights_path = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

    print("=" * 90)
    print("   Stage 2 Probe: Dual-Domain Residual Zero-Training Diagnostic Evaluation")
    print(f"   Checkpoint        : {weights_path}")
    print(f"   Dataset           : {args.data}")
    print(f"   Device            : {device} | imgsz: {args.imgsz} | stride: {args.stride}")
    print(f"   Distance Tolerance: <= {args.dist_thresh:.1f}px")
    print("=" * 90)

    # 1. Load Model
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()

    ckpt_stride = ckpt.get("stride", args.stride)
    ckpt_upsample = ckpt.get("upsample_mode", "nearest")

    model = YOLO26HeatmapDetector(stride=ckpt_stride, num_classes=1, upsample_mode=ckpt_upsample)
    # Filter state dict for direct loading
    own_state = model.state_dict()
    matched = 0
    for k, v in state_dict.items():
        clean_k = k.replace("model.model.", "").replace("model.", "")
        if clean_k in own_state and own_state[clean_k].shape == v.shape:
            own_state[clean_k].copy_(v)
            matched += 1
    print(f"[INFO] Loaded {matched}/{len(own_state)} layers from {weights_path.name}")
    model.to(device)
    model.eval()

    # 2. Build DataLoader
    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data

    val_dataset = build_yolo_dataset(cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=32)
    val_loader = build_dataloader(val_dataset, batch=args.batch, workers=8, shuffle=False)
    print(f"[INFO] Evaluating on {len(val_dataset)} images from {args.data}...")

    all_peaks_raw = []
    val_gt_list = []
    val_sizes_list = []
    im_names_list = []

    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Dual-Residual Infer"):
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            bboxes = batch["bboxes"]
            b_idx = batch["batch_idx"]
            bs = imgs.shape[0]
            im_files = batch["im_file"]

            preds = model(imgs)

            peaks = extract_peaks(
                heatmap=preds["heatmap"],
                offset=preds["offset"],
                stride=ckpt_stride,
                conf_thresh=0.03,  # Low cutoff to preserve full curve
                top_k=100,
            )
            all_peaks_raw.extend(peaks)

            for b in range(bs):
                mask_b = b_idx == b
                val_gt_list.append(bboxes[mask_b].cpu().numpy())
                val_sizes_list.append((args.imgsz, args.imgsz))
                im_names_list.append(im_files[b])

    infer_time = time.time() - t0
    print(f"[INFO] Inference finished in {infer_time:.1f}s ({len(val_dataset) / max(0.1, infer_time):.1f} imgs/s)")

    # 3. Grand Threshold Scanning
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    best_f1 = 0.0
    best_th = 0.25
    best_metrics = {}

    print("\n" + "=" * 90)
    print(f"{'Threshold':<10} | {'Recall':<10} | {'Precision':<10} | {'F1-Score':<10} | {'TP':<7} | {'FP':<7} | {'GT':<7}")
    print("-" * 90)

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

        r = m["recall"]
        p = m["precision"]
        f1 = m["f1"]

        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_metrics = m

        print(f"{th:<10.2f} | {r:<10.4f} | {p:<10.4f} | {f1:<10.4f} | {m['tp']:<7} | {m['fp']:<7} | {m['total_gt']:<7}")

    print("=" * 90)
    print(colorstr("bold", colorstr("green", f"\n>>> [Best F1-Score]  : {best_f1:.4f} @ Threshold = {best_th:.2f}")))
    print(f"    Recall    : {best_metrics['recall']:.4f}")
    print(f"    Precision : {best_metrics['precision']:.4f}")
    print(f"    TP: {best_metrics['tp']}, FP: {best_metrics['fp']}, GT: {best_metrics['total_gt']}")

    # 4. Compare with Trial 22 Reference on uav_gmc_median
    print("\n" + "=" * 90)
    print(">>> COMPARISON WITH TRIAL 22 ABSOLUTE SOTA (Distance <= 8.0px, GT = 25,111)")
    print("=" * 90)
    print(f"{'Metric':<22} | {'Trial 22 Benchmark (Median)':<30} | {'Dual-Residual Zero-Shot Probe':<30}")
    print("-" * 90)
    probe_f1_str = f"{best_f1:.4f} (@ th={best_th:.2f})"
    probe_rec_str = f"{best_metrics['recall'] * 100:.2f}%"
    probe_prec_str = f"{best_metrics['precision'] * 100:.2f}%"
    probe_tp_str = f"{best_metrics['tp']}"
    probe_fp_str = f"{best_metrics['fp']}"
    print(f"{'Best F1-Score':<22} | {'0.9058 (@ th=0.25)':<30} | {probe_f1_str:<30}")
    print(f"{'Recall':<22} | {'86.12%':<30} | {probe_rec_str:<30}")
    print(f"{'Precision':<22} | {'95.53%':<30} | {probe_prec_str:<30}")
    print(f"{'True Positives (TP)':<22} | {'21,625':<30} | {probe_tp_str:<30}")
    print(f"{'False Positives (FP)':<22} | {'1,012':<30} | {probe_fp_str:<30}")
    print("=" * 90)

    # 5. Diagnostic Breakdown for 4 Hard Cases at th=0.25
    hard_seqs = ["wg2022_ir_011_split_03", "DJI_0175_2", "DJI_0051_2", "wg2022_ir_020_split_03"]
    print(colorstr("bold", "\n>>> HARD CASE DIAGNOSTIC BREAKDOWN (@ th=0.25):"))
    print("-" * 90)
    print(f"{'Sequence Name':<28} | {'TP / GT':<14} | {'FP':<6} | {'Recall':<8} | {'Precision':<8} | {'F1':<8}")
    print("-" * 90)

    seq_data: dict[str, dict] = {}
    for i, im_path in enumerate(im_names_list):
        s_name = parse_seq_name(im_path)
        if s_name not in seq_data:
            seq_data[s_name] = {"preds": [], "gts": [], "sizes": []}

        p = all_peaks_raw[i]
        keep = p["scores"] >= 0.25
        seq_data[s_name]["preds"].append({
            "points": p["points"][keep],
            "scores": p["scores"][keep],
        })
        seq_data[s_name]["gts"].append(val_gt_list[i])
        seq_data[s_name]["sizes"].append(val_sizes_list[i])

    for hs in hard_seqs:
        matched_seq = next((s for s in seq_data if hs in s), None)
        if matched_seq:
            sd = seq_data[matched_seq]
            sm = evaluate_point_detections(
                predictions=sd["preds"],
                gt_boxes_list=sd["gts"],
                img_sizes=sd["sizes"],
                distance_threshold=args.dist_thresh,
            )
            print(f"{hs:<28} | {sm['tp']:>5} / {sm['total_gt']:<6} | {sm['fp']:<6} | {sm['recall']*100:>6.2f}% | {sm['precision']*100:>6.2f}% | {sm['f1']:>6.4f}")
        else:
            print(f"{hs:<28} | [Not Found in Val Split]")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
