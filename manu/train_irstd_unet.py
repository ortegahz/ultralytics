#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Dedicated Training Script for IRSTD-UNet (Infrared Small Target Detection U-Net).

Features:
1. Pure PyTorch U-Net architecture (manu/irstd_unet_model.py) without modifying YOLO26 internals.
2. PixelShuffle Sub-pixel reconstruction + Asymmetric Context Modulation (ACM).
3. Soft-IoU + Focal Loss + RegL1 Multi-Task supervision for tiny point targets.
4. Fully compatible with existing YOLO dataset format (data.yaml) and 3-channel temporal inputs.
5. Server multi-GPU DataParallel and Optuna/AMP acceleration support.

Example server command:
    python manu/train_irstd_unet.py --data /mnt/data/siping/datasets/manu/uav/data.yaml --device 0,1,2,3 --batch 32 --epochs 30 --stride 2
"""

from __future__ import annotations

import argparse
import math
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
from ultralytics.utils import DEFAULT_CFG, colorstr

from manu.irstd_unet_model import IRSTDNet
from manu.irstd_loss import IRSTDLoss
from manu.heatmap_loss import generate_heatmaps_and_targets
from manu.heatmap_evaluate import extract_peaks, find_best_f1_threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Train Dedicated IRSTD-UNet for Infrared Tiny UAVs")
    parser.add_argument("--data", type=str, default="/mnt/data/siping/datasets/manu/uav/data.yaml", help="Path to data.yaml")
    parser.add_argument("--stride", type=int, default=2, choices=[1, 2], help="Decoder output stride (2: 320x320, 1: 640x640)")
    parser.add_argument("--base_channels", type=int, default=24, help="Base channel dimension for level 1 (e.g. 24 -> 24, 48, 96, 192)")
    parser.add_argument("--imgsz", type=int, default=640, help="Train image size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size per GPU for training")
    parser.add_argument("--val_batch", type=int, default=24, help="Batch size per GPU for validation")
    parser.add_argument("--lr0", type=float, default=0.0003, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final lr factor (lr0 * lrf)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--w_iou", type=float, default=1.0, help="Weight of Soft-IoU loss")
    parser.add_argument("--w_bce", type=float, default=1.0, help="Weight of BCEWithLogits loss")
    parser.add_argument("--w_off", type=float, default=0.1, help="Weight of Sub-pixel Offset loss (default 0.1)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device(s), e.g. 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--project", type=str, default="runs/irstd_unet", help="Save project directory")
    parser.add_argument("--name", type=str, default="exp", help="Save experiment name")
    parser.add_argument("--min_radius", type=int, default=1, help="Minimum Gaussian radius for point targets")
    parser.add_argument("--dist_thresh", type=float, default=8.0, help="Distance threshold (pixels) for TP matching")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    return parser.parse_args()


def main():
    args = parse_args()

    # Directories
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

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

    print(colorstr("bold", f"=== Starting Dedicated IRSTD-UNet Training on {device} (GPUs: {gpu_ids or 'CPU'}) ==="))
    print(f"Output directory: {save_dir}")
    print(f"Model Configuration: Output Stride={args.stride}, Base Channels={args.base_channels}")

    # 1. Dataset loading
    data_dict = check_det_dataset(args.data)
    train_path = data_dict["train"]
    val_path = data_dict["val"]

    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data
    # Augmentations designed for subtle point-like infrared targets
    cfg.hsv_h = 0.0
    cfg.hsv_s = 0.0
    cfg.hsv_v = 0.0
    cfg.degrees = 0.0
    cfg.shear = 0.0
    cfg.perspective = 0.0
    cfg.translate = 0.1
    cfg.scale = 0.2
    cfg.fliplr = 0.5
    cfg.flipud = 0.0
    cfg.mosaic = 0.1
    cfg.mixup = 0.0
    cfg.copy_paste = 0.0

    print("Building datasets...")
    train_dataset = build_yolo_dataset(cfg, train_path, batch=args.batch, data=data_dict, mode="train", stride=32)
    val_dataset = build_yolo_dataset(cfg, val_path, batch=args.batch, data=data_dict, mode="val", stride=32)

    total_batch = args.batch * max(1, len(gpu_ids))
    val_total_batch = args.val_batch * max(1, len(gpu_ids))
    train_loader = build_dataloader(train_dataset, batch=total_batch, workers=args.workers, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch=val_total_batch, workers=args.workers, shuffle=False)

    # 2. Build IRSTD-UNet Model
    model = IRSTDNet(
        in_channels=3,
        out_stride=args.stride,
        base_channels=args.base_channels,
        num_classes=1,
    )
    model.to(device)

    # Parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"IRSTD-UNet Model Parameters: Total = {total_params / 1e6:.2f}M, Trainable = {trainable_params / 1e6:.2f}M")

    if len(gpu_ids) > 1:
        model_module = nn.DataParallel(model, device_ids=gpu_ids, output_device=primary_gpu)
    else:
        model_module = model

    # 3. Optimizer & Step-level Warmup Scheduler & Multi-Task Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr0, weight_decay=args.weight_decay)
    
    num_batches = len(train_loader)
    total_steps = args.epochs * num_batches
    warmup_steps = min(500, max(50, int(0.05 * total_steps)))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup from 0.01 * lr0 to lr0
            return max(0.01, float(current_step) / float(max(1, warmup_steps)))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return ((1.0 + math.cos(progress * math.pi)) / 2.0) * (1.0 - args.lrf) + args.lrf

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    criterion = IRSTDLoss(w_iou=args.w_iou, w_bce=args.w_bce, w_off=args.w_off)

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
        train_iou_accum = 0.0
        train_bce_accum = 0.0
        train_off_accum = 0.0
        num_batches = len(train_loader)

        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d}", total=num_batches, dynamic_ncols=True)
        for batch_i, batch in enumerate(pbar):
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            bboxes = batch["bboxes"].to(device, non_blocking=True)
            b_idx = batch["batch_idx"].to(device, non_blocking=True)
            bs = imgs.shape[0]

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

            if torch.isnan(loss) or torch.isinf(loss):
                print(colorstr("red", f"\n[Warning] NaN/Inf loss encountered at Epoch {epoch}, Batch {batch_i}! Skipping step."))
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step() # Step-level warmup & cosine annealing

            train_loss_accum += loss_items["loss_total"]
            train_iou_accum += loss_items["loss_iou"]
            train_bce_accum += loss_items["loss_bce"]
            train_off_accum += loss_items["loss_offset"]

            pbar.set_postfix({
                "loss": f"{loss_items['loss_total']:.4f}",
                "iou": f"{loss_items['loss_iou']:.4f}",
                "bce": f"{loss_items['loss_bce']:.4f}",
                "off": f"{loss_items['loss_offset']:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })

        epoch_time = time.time() - t0

        avg_loss = train_loss_accum / max(num_batches, 1)
        avg_iou = train_iou_accum / max(num_batches, 1)
        avg_bce = train_bce_accum / max(num_batches, 1)
        avg_off = train_off_accum / max(num_batches, 1)

        # 5. Validation Evaluation
        model.eval()
        if device.type == "cuda":
            torch.cuda.empty_cache()
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

                with autocast(enabled=(device.type == "cuda")):
                    preds = model_module(imgs)
                peaks = extract_peaks(
                    heatmap=preds["heatmap"],
                    offset=preds["offset"],
                    stride=args.stride,
                    conf_thresh=0.10,
                    top_k=80,
                )
                val_preds_list.extend(peaks)

                # Ensure b_idx is 1D integer CPU tensor for robust indexing
                b_idx_cpu = b_idx.long().cpu().view(-1)
                bboxes_cpu = bboxes.cpu().numpy()
                for b in range(bs):
                    mask_b = (b_idx_cpu == b).numpy()
                    gt_b = bboxes_cpu[mask_b] if mask_b.any() else np.zeros((0, 4), dtype=np.float32)
                    val_gt_list.append(gt_b)
                    val_sizes_list.append((args.imgsz, args.imgsz))

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Evaluate across confidence thresholds at official tolerance dist_thresh (default 8.0px)
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
            f"Loss: {avg_loss:.4f} (IoU: {avg_iou:.4f}, BCE: {avg_bce:.4f}, Off: {avg_off:.4f}) | "
            f"Best F1: {f1:.4f} @ th={best_th:.2f} | Recall: {rec:.4f} | Prec: {prec:.4f} | "
            f"TP: {metrics['tp']}, FP: {metrics['fp']}, GT: {metrics['total_gt']} | "
            f"Time: {epoch_time:.1f}s"
        )
        print(log_str)

        # Log CSV results
        csv_path = save_dir / "results.csv"
        csv_header = "epoch,train/loss,train/loss_iou,train/loss_bce,train/loss_off,metrics/best_th,metrics/recall(B),metrics/precision(B),metrics/f1(B),metrics/tp,metrics/fp,metrics/gt,lr\n"
        if not csv_path.exists():
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(csv_header)
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{avg_loss:.6f},{avg_iou:.6f},{avg_bce:.6f},{avg_off:.6f},"
                f"{best_th:.4f},{rec:.6f},{prec:.6f},{f1:.6f},"
                f"{metrics['tp']},{metrics['fp']},{metrics['total_gt']},"
                f"{optimizer.param_groups[0]['lr']:.8f}\n"
            )

        # Save weights
        if not math.isnan(avg_loss) and not math.isnan(rec):
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": metrics,
                "stride": args.stride,
                "base_channels": args.base_channels,
                "imgsz": args.imgsz,
            }
            torch.save(ckpt, weights_dir / "last.pt")

            if rec > best_recall:
                best_recall = rec
                torch.save(ckpt, weights_dir / "best_recall.pt")
                print(colorstr("green", f"  --> New Best Recall: {best_recall:.4f} saved to best_recall.pt"))

            if f1 > best_f1:
                best_f1 = f1
                torch.save(ckpt, weights_dir / "best_f1.pt")
                print(colorstr("magenta", f"  --> New Best F1: {best_f1:.4f} (@ th={best_th:.2f}) saved to best_f1.pt"))

    print(colorstr("bold", f"\nTraining Complete! Best Recall: {best_recall:.4f}, Best F1: {best_f1:.4f}"))
    print(f"Weights saved at: {weights_dir}")


if __name__ == "__main__":
    main()
