#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Evaluation and Benchmark Script for IRSTD-UNet.

Computes official GJB/Security metrics (Recall, Precision, F1, TP, FP, FAR)
across multiple distance tolerances (4.0px and 8.0px).

Example server command:
    python manu/eval_irstd_unet.py --weights runs/irstd_unet/exp/weights/best_f1.pt --data /mnt/data/siping/datasets/manu/uav/data.yaml --device 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG, colorstr

from manu.irstd_unet_model import IRSTDNet
from manu.heatmap_evaluate import extract_peaks, evaluate_point_detections, find_best_f1_threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate IRSTD-UNet Model (Multi-GPU Parallel)")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--data", type=str, default="/mnt/data/siping/datasets/manu/uav/data.yaml", help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--batch", type=int, default=24, help="Batch size per GPU (default 24)")
    parser.add_argument("--device", type=str, default="0,1,2,3", help="CUDA device(s), e.g. 0,1,2,3 or 0 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="Workers")
    return parser.parse_args()


def main():
    args = parse_args()

    # Device configuration
    device_str = args.device.strip()
    if device_str != "cpu" and torch.cuda.is_available():
        gpu_ids = [int(x) for x in device_str.split(",") if x.isdigit()]
        primary_gpu = gpu_ids[0]
        device = torch.device(f"cuda:{primary_gpu}")
        torch.cuda.set_device(primary_gpu)
    else:
        gpu_ids = []
        device = torch.device("cpu")

    print(f"Loading checkpoint: {args.weights} on {device} (GPUs: {gpu_ids or 'CPU'})")

    ckpt = torch.load(args.weights, map_location="cpu")
    stride = ckpt.get("stride", 2)
    base_channels = ckpt.get("base_channels", 24)

    model = IRSTDNet(in_channels=3, out_stride=stride, base_channels=base_channels, num_classes=1)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if len(gpu_ids) > 1:
        model_module = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=primary_gpu)
    else:
        model_module = model

    # Dataset loading
    data_dict = check_det_dataset(args.data)
    val_path = data_dict["val"]

    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data

    total_batch = args.batch * max(1, len(gpu_ids))
    print(f"Eval Total Batch Size: {total_batch} across {len(gpu_ids) or 1} device(s) (per-GPU: {args.batch})")

    val_dataset = build_yolo_dataset(cfg, val_path, batch=total_batch, data=data_dict, mode="val", stride=32)
    val_loader = build_dataloader(val_dataset, batch=total_batch, workers=args.workers, shuffle=False)

    print("Running parallel inference across validation set...")
    val_preds_list = []
    val_gt_list = []
    val_sizes_list = []

    if device.type == "cuda":
        torch.cuda.empty_cache()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Eval"):
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            bboxes = batch["bboxes"]
            b_idx = batch["batch_idx"]
            bs = imgs.shape[0]

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                preds = model_module(imgs)
            peaks = extract_peaks(
                heatmap=preds["heatmap"],
                offset=preds["offset"],
                stride=stride,
                conf_thresh=0.10,
                top_k=80,
            )
            val_preds_list.extend(peaks)

            b_idx_cpu = b_idx.long().cpu().view(-1)
            bboxes_cpu = bboxes.cpu().numpy()
            for b in range(bs):
                mask_b = (b_idx_cpu == b).numpy()
                gt_b = bboxes_cpu[mask_b] if mask_b.any() else np.zeros((0, 4), dtype=np.float32)
                val_gt_list.append(gt_b)
                val_sizes_list.append((args.imgsz, args.imgsz))

    if device.type == "cuda":
        torch.cuda.empty_cache()

    total_frames = len(val_sizes_list)

    # 1. Official Distance <= 8.0px
    print(colorstr("bold", "\n=== Benchmark Results (Tolerance Distance <= 8.0px) ==="))
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    best_m8 = find_best_f1_threshold(val_preds_list, val_gt_list, val_sizes_list, distance_threshold=8.0, thresholds=thresholds)
    far8 = best_m8["fp"] / total_frames
    print(f"Optimal Threshold : {best_m8['best_th']:.2f}")
    print(f"Recall (召回率)   : {best_m8['recall'] * 100:.2f}% ({best_m8['tp']}/{best_m8['total_gt']})")
    print(f"Precision (精确率): {best_m8['precision'] * 100:.2f}%")
    print(f"F1-Score          : {best_m8['f1']:.4f}")
    print(f"Total FP (虚警数) : {best_m8['fp']}")
    print(f"FAR (单帧虚警率)  : {far8:.4f} false alarms / frame")

    # 2. Strict Distance <= 4.0px
    print(colorstr("bold", "\n=== Benchmark Results (Tolerance Distance <= 4.0px) ==="))
    best_m4 = find_best_f1_threshold(val_preds_list, val_gt_list, val_sizes_list, distance_threshold=4.0, thresholds=thresholds)
    far4 = best_m4["fp"] / total_frames
    print(f"Optimal Threshold : {best_m4['best_th']:.2f}")
    print(f"Recall (召回率)   : {best_m4['recall'] * 100:.2f}% ({best_m4['tp']}/{best_m4['total_gt']})")
    print(f"Precision (精确率): {best_m4['precision'] * 100:.2f}%")
    print(f"F1-Score          : {best_m4['f1']:.4f}")
    print(f"Total FP (虚警数) : {best_m4['fp']}")
    print(f"FAR (单帧虚警率)  : {far4:.4f} false alarms / frame")


if __name__ == "__main__":
    main()
