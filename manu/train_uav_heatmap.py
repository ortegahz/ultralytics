#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Full Training Script for Tiny UAV Point Detection via Online Gaussian Heatmap Regression.
Compatible with standard YOLO dataset format (data.yaml), uses YOLO26-P2 pretrained weights,
and supports multi-GPU training with Recall-maximizing checkpoint saving.

Run on server:
    python manu/train_uav_heatmap.py --device 0,1,2,3 --batch 32 --epochs 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr

from manu.heatmap_model import YOLO26HeatmapDetector
from manu.heatmap_loss import HeatmapLoss, generate_heatmaps_and_targets
from manu.heatmap_evaluate import extract_peaks, evaluate_point_detections, find_best_f1_threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Train UAV Tiny Object Heatmap Detector")
    parser.add_argument("--data", type=str, default="/mnt/data/siping/datasets/manu/uav/data.yaml", help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default="yolo26np2.pt", help="Initial weights (e.g. yolo26np2.pt)")
    parser.add_argument("--stride", type=int, default=4, choices=[2, 4], help="Feature stride (4 for P2, 2 for P1 high-res)")
    parser.add_argument("--imgsz", type=int, default=640, help="Train image size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final lr factor (lr0 * lrf)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--device", type=str, default="0", help="CUDA device(s), e.g. 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--project", type=str, default="runs/heatmap_uav", help="Save project directory")
    parser.add_argument("--name", type=str, default="exp", help="Save experiment name")
    parser.add_argument("--min_radius", type=int, default=1, help="Minimum Gaussian radius for point targets")
    parser.add_argument("--conf_thresh", type=float, default=0.20, help="Peak confidence threshold for evaluation")
    parser.add_argument("--dist_thresh", type=float, default=4.0, help="Distance threshold (pixels) for TP matching")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup directories
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device_str = args.device.strip()
    if device_str != "cpu" and torch.cuda.is_available():
        gpu_ids = [int(x) for x in device_str.split(",") if x.isdigit()]
        primary_gpu = gpu_ids[0]
        device = torch.device(f"cuda:{primary_gpu}")
        torch.cuda.set_device(primary_gpu)
    else:
        gpu_ids = []
        device = torch.device("cpu")

    print(colorstr("bold", f"Starting UAV Heatmap training on {device} (GPUs: {gpu_ids or 'CPU'})"))
    print(f"Output directory: {save_dir}")

    # 1. Dataset loading via Ultralytics data pipeline
    data_dict = check_det_dataset(args.data)
    train_path = data_dict["train"]
    val_path = data_dict["val"]

    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data
    # Specific augmentations for tiny point-like targets
    cfg.hsv_h = 0.0
    cfg.hsv_s = 0.0
    cfg.hsv_v = 0.0
    cfg.degrees = 0.0
    cfg.shear = 0.0
    cfg.perspective = 0.0
    cfg.translate = 0.1
    cfg.scale = 0.2  # subtle scale to protect 3x3 dots
    cfg.fliplr = 0.5
    cfg.flipud = 0.0
    cfg.mosaic = 0.2
    cfg.mixup = 0.0
    cfg.copy_paste = 0.0

    print("Building datasets...")
    train_dataset = build_yolo_dataset(cfg, train_path, batch=args.batch, data=data_dict, mode="train", stride=32)
    val_dataset = build_yolo_dataset(cfg, val_path, batch=args.batch, data=data_dict, mode="val", stride=32)

    total_batch = args.batch * max(1, len(gpu_ids))
    train_loader = build_dataloader(train_dataset, batch=total_batch, workers=args.workers, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch=total_batch, workers=args.workers, shuffle=False)

    # 2. Build Model
    model = YOLO26HeatmapDetector(stride=args.stride, weights=args.weights, num_classes=1)
    model.to(device)

    # Multi-GPU DataParallel support if multiple GPUs specified
    if len(gpu_ids) > 1:
        model_module = nn.DataParallel(model, device_ids=gpu_ids, output_device=primary_gpu)
    else:
        model_module = model

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr0, weight_decay=args.weight_decay)
    lf = lambda epoch: ((1 + math.cos(epoch * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    criterion = HeatmapLoss(hm_weight=1.0, offset_weight=0.5)

    feat_h = args.imgsz // args.stride
    feat_w = args.imgsz // args.stride
    print(f"Heatmap target resolution: {feat_h} x {feat_w} (stride={args.stride})")
    print(f"Total Batch Size: {total_batch} across {len(gpu_ids) or 1} device(s)")

    best_recall = 0.0
    best_f1 = 0.0

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_hm_accum = 0.0
        train_off_accum = 0.0
        num_batches = len(train_loader)

        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d}", total=num_batches, dynamic_ncols=True)
        for batch_i, batch in enumerate(pbar):
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            bboxes = batch["bboxes"].to(device, non_blocking=True)
            b_idx = batch["batch_idx"].to(device, non_blocking=True)
            bs = imgs.shape[0]

            # Generate target heatmaps & offsets on-the-fly
            targets = generate_heatmaps_and_targets(
                batch_bboxes=bboxes,
                batch_idx=b_idx,
                batch_size=bs,
                feat_shape=(feat_h, feat_w),
                stride=args.stride,
                min_radius=args.min_radius,
                device=device,
            )

            optimizer.zero_grad()
            with autocast(enabled=(device.type == "cuda")):
                preds = model_module(imgs)
                loss, loss_items = criterion(preds, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += loss_items["loss_total"]
            train_hm_accum += loss_items["loss_hm"]
            train_off_accum += loss_items["loss_offset"]

            pbar.set_postfix({
                "loss": f"{loss_items['loss_total']:.4f}",
                "hm": f"{loss_items['loss_hm']:.4f}",
                "off": f"{loss_items['loss_offset']:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })

        scheduler.step()
        epoch_time = time.time() - t0

        avg_loss = train_loss_accum / max(num_batches, 1)
        avg_hm_loss = train_hm_accum / max(num_batches, 1)
        avg_off_loss = train_off_accum / max(num_batches, 1)

        # 5. Validation Evaluation
        model.eval()
        val_preds_list = []
        val_gt_list = []
        val_sizes_list = []

        val_pbar = tqdm(val_loader, desc=f"Val {epoch:02d}", total=len(val_loader), dynamic_ncols=True)
        with torch.no_grad():
            for batch in val_pbar:
                imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
                bboxes = batch["bboxes"]
                b_idx = batch["batch_idx"]
                bs = imgs.shape[0]

                preds = model(imgs)
                # Peak detection: extract candidates with low threshold (0.10) for F1 threshold searching
                peaks = extract_peaks(
                    heatmap=preds["heatmap"],
                    offset=preds["offset"],
                    stride=args.stride,
                    conf_thresh=0.10,
                    top_k=80,
                )
                val_preds_list.extend(peaks)

                for b in range(bs):
                    mask_b = b_idx == b
                    gt_b = bboxes[mask_b].cpu().numpy()
                    val_gt_list.append(gt_b)
                    val_sizes_list.append((args.imgsz, args.imgsz))

        # Evaluate across confidence thresholds to find peak F1 and optimal threshold
        metrics = find_best_f1_threshold(
            predictions_raw=val_preds_list,
            gt_boxes_list=val_gt_list,
            img_sizes=val_sizes_list,
            distance_threshold=args.dist_thresh,
            thresholds=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        )

        rec = metrics["recall"]
        prec = metrics["precision"]
        f1 = metrics["f1"]
        best_th = metrics.get("best_th", 0.20)

        log_str = (
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Loss: {avg_loss:.4f} (HM: {avg_hm_loss:.4f}, Off: {avg_off_loss:.4f}) | "
            f"Best F1: {f1:.4f} @ th={best_th:.2f} | Recall: {rec:.4f} | Prec: {prec:.4f} | "
            f"TP: {metrics['tp']}, FP: {metrics['fp']}, GT: {metrics['total_gt']} | "
            f"Time: {epoch_time:.1f}s"
        )
        print(log_str)

        # Log CSV results for easy comparison
        csv_path = save_dir / "results.csv"
        csv_header = "epoch,train/loss,train/loss_hm,train/loss_off,metrics/best_th,metrics/recall(B),metrics/precision(B),metrics/f1(B),metrics/tp,metrics/fp,metrics/gt,lr\n"
        if not csv_path.exists():
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(csv_header)
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{avg_loss:.6f},{avg_hm_loss:.6f},{avg_off_loss:.6f},"
                f"{best_th:.4f},{rec:.6f},{prec:.6f},{f1:.6f},"
                f"{metrics['tp']},{metrics['fp']},{metrics['total_gt']},"
                f"{optimizer.param_groups[0]['lr']:.8f}\n"
            )

        # Save latest
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "stride": args.stride,
            "imgsz": args.imgsz,
        }
        torch.save(ckpt, weights_dir / "last.pt")

        # Save best recall
        if rec > best_recall:
            best_recall = rec
            torch.save(ckpt, weights_dir / "best_recall.pt")
            print(colorstr("green", f"  --> New Best Recall: {best_recall:.4f} saved to best_recall.pt"))

        # Save best F1
        if f1 > best_f1:
            best_f1 = f1
            torch.save(ckpt, weights_dir / "best_f1.pt")
            print(colorstr("magenta", f"  --> New Best F1: {best_f1:.4f} (@ th={best_th:.2f}) saved to best_f1.pt"))

    print(colorstr("bold", f"\nTraining Complete! Best Recall: {best_recall:.4f}, Best F1: {best_f1:.4f}"))
    print(f"Weights saved at: {weights_dir}")


if __name__ == "__main__":
    import math
    main()
