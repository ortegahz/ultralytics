#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
2-GPU Parallel Controlled A/B Benchmark: YOLO26n vs YOLO26s (Stride=4 P2 Heatmap Output).

Core Design (Strict Single-Variable Controlled Comparison):
1. Scales: Supports --scale n (Nano) or --scale s (Small).
2. Initialization:
   - For scale 'n': loads official backbone from yolo26n.pt (or yolo26np2.pt).
   - For scale 's': loads official backbone from yolo26s.pt.
3. Architecture:
   - YOLO26HeatmapDetector with Stride=4 (P2 layer, 160x160 heatmap for 640x640 input).
4. Dataset: Official GMC Temporal Median Dataset:
   /mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml
   Input channels: [I_t, |I_t - W(I_{t-2})|, (I_t - B_t)^+]
5. SOTA Hyperparameters (Frozen from Trial 22 Optuna optimum):
   - focal_beta = 2.40 (negative attenuation)
   - offset_weight = 0.50
   - min_radius = 1
   - lr0 = 0.0001, lrf = 0.01 (cosine annealing)
   - epochs = 5
   - distance_threshold = 8.0px (official standard)
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, colorstr
from manu.heatmap_evaluate import extract_peaks, find_best_f1_threshold
from manu.heatmap_loss import HeatmapLoss, generate_heatmaps_and_targets
from manu.heatmap_model import YOLO26HeatmapDetector


def parse_args():
    parser = argparse.ArgumentParser(description="2-GPU Controlled A/B Training: YOLO26n vs YOLO26s")
    parser.add_argument("--scale", type=str, required=True, choices=["n", "s"], help="Model scale: 'n' or 's'")
    parser.add_argument("--data", type=str, default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml")
    parser.add_argument("--weights", type=str, default=None, help="Official initial weights (e.g. yolo26n.pt or yolo26s.pt)")
    parser.add_argument("--device", type=str, default="0,1", help="CUDA device IDs, e.g. '0,1' or '2,3'")
    parser.add_argument("--batch", type=int, default=32, help="Batch size per GPU (total = batch * num_gpus)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (strictly 5)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--stride", type=int, default=4, help="Feature stride (strictly 4 for P2)")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers per GPU")
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="TP distance match threshold (px)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max grad norm clipping")

    # SOTA Frozen Hyperparameters
    parser.add_argument("--lr0", type=float, default=0.0001, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final lr factor")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument("--focal-beta", type=float, default=2.40, help="Focal loss negative exponent beta")
    parser.add_argument("--offset-weight", type=float, default=0.50, help="Offset L1 loss weight")
    parser.add_argument("--min-radius", type=int, default=1, help="Min Gaussian radius on feature map")
    parser.add_argument("--scale-aug", type=float, default=0.15, help="Mosaic/scale augmentation")
    parser.add_argument("--mosaic", type=float, default=0.10, help="Mosaic probability")
    parser.add_argument("--translate", type=float, default=0.08, help="Translation augmentation")

    parser.add_argument("--project", type=str, default="runs/scale_ab_test", help="Save directory root")
    parser.add_argument("--name", type=str, default=None, help="Experiment name (default: auto scale name)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine default weights if not specified
    if args.weights is None:
        args.weights = f"yolo26{args.scale}.pt"

    exp_name = args.name or f"yolo26{args.scale}_p2_5ep"
    save_dir = Path(args.project) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Setup devices
    device_str = args.device.strip()
    gpu_ids = [int(x) for x in device_str.split(",") if x.isdigit()]
    assert len(gpu_ids) > 0, "At least one GPU is required for training."
    primary_device = torch.device(f"cuda:{gpu_ids[0]}")
    torch.cuda.set_device(gpu_ids[0])

    total_batch = args.batch * len(gpu_ids)

    print("=" * 90)
    print(f"[A/B BENCHMARK] Scale: YOLO26{args.scale.upper()} (P2 Stride=4 Heatmap)")
    print(f"GPUs: {gpu_ids} (Primary: cuda:{gpu_ids[0]}), Total Batch: {total_batch}")
    print(f"Init Weights: {args.weights}")
    print(f"Hyperparameters: lr0={args.lr0}, lrf={args.lrf}, beta={args.focal_beta}, off_w={args.offset_weight}")
    print(f"Dataset: {args.data}")
    print(f"Save Directory: {save_dir}")
    print("=" * 90)

    # 1. Dataset & Dataloaders
    data_dict = check_det_dataset(args.data)
    train_path = data_dict["train"]
    val_path = data_dict["val"]

    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.scale = args.scale_aug
    cfg.mosaic = args.mosaic
    cfg.translate = args.translate
    cfg.degrees = 0.0
    cfg.shear = 0.0
    cfg.perspective = 0.0
    cfg.flipud = 0.5
    cfg.fliplr = 0.5

    train_dataset = build_yolo_dataset(cfg, img_path=train_path, batch=total_batch, data=data_dict, mode="train", rect=False)
    val_dataset = build_yolo_dataset(cfg, img_path=val_path, batch=total_batch, data=data_dict, mode="val", rect=False)

    train_loader = build_dataloader(train_dataset, batch=total_batch, workers=args.workers * len(gpu_ids), shuffle=True)
    val_loader = build_dataloader(val_dataset, batch=total_batch, workers=args.workers * len(gpu_ids), shuffle=False)

    # 2. Build Model
    model = YOLO26HeatmapDetector(
        stride=args.stride,
        weights=args.weights,
        num_classes=1,
        scale=args.scale,
    )
    model.to(primary_device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] YOLO26{args.scale.upper()} constructed. Total Params: {total_params:,}, Trainable: {trainable_params:,}")

    # Wrap DataParallel if multiple GPUs
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    # 3. Loss & Optimizer
    criterion = HeatmapLoss(
        hm_weight=1.0,
        offset_weight=args.offset_weight,
        soft_iou_weight=0.0,
        focal_alpha=2.0,
        focal_beta=args.focal_beta,
    )

    raw_model = model.module if hasattr(model, "module") else model
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.lr0,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Cosine Annealing LR Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr0 * args.lrf,
    )
    scaler = GradScaler()

    feat_h = args.imgsz // args.stride
    feat_w = args.imgsz // args.stride

    best_f1 = 0.0
    best_recall = 0.0
    best_metrics = {}

    csv_path = save_dir / "results.csv"
    csv_header = "epoch,train/loss,train/loss_hm,train/loss_off,metrics/best_th,metrics/recall(B),metrics/precision(B),metrics/f1(B),metrics/tp,metrics/fp,metrics/gt,lr\n"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_header)

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_accum = 0.0
        train_hm_accum = 0.0
        train_off_accum = 0.0
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"Train Ep {epoch:02d}/{args.epochs:02d}", total=num_batches, dynamic_ncols=True)
        for batch in pbar:
            imgs = batch["img"].to(primary_device, non_blocking=True).float() / 255.0
            bboxes = batch["bboxes"].to(primary_device)
            b_idx = batch["batch_idx"].to(primary_device)
            bs = imgs.shape[0]

            targets = generate_heatmaps_and_targets(
                batch_bboxes=bboxes,
                batch_idx=b_idx,
                batch_size=bs,
                feat_shape=(feat_h, feat_w),
                stride=args.stride,
                min_radius=args.min_radius,
                device=primary_device,
            )

            optimizer.zero_grad()
            with autocast():
                preds = model(imgs)
                loss, loss_items = criterion(preds, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += loss_items["loss_total"]
            train_hm_accum += loss_items["loss_hm"]
            train_off_accum += loss_items["loss_offset"]

            pbar.set_postfix({
                "loss": f"{loss_items['loss_total']:.4f}",
                "hm": f"{loss_items['loss_hm']:.4f}",
                "off": f"{loss_items['loss_offset']:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
            })

        scheduler.step()
        epoch_time = time.time() - t0

        avg_loss = train_loss_accum / max(num_batches, 1)
        avg_hm_loss = train_hm_accum / max(num_batches, 1)
        avg_off_loss = train_off_accum / max(num_batches, 1)

        # 5. Full Validation Scanning
        model.eval()
        val_preds_list = []
        val_gt_list = []
        val_sizes_list = []

        val_pbar = tqdm(val_loader, desc=f"Val Ep {epoch:02d}/{args.epochs:02d}", total=len(val_loader), dynamic_ncols=True)
        with torch.no_grad():
            for batch in val_pbar:
                imgs = batch["img"].to(primary_device, non_blocking=True).float() / 255.0
                bboxes = batch["bboxes"]
                b_idx = batch["batch_idx"]
                bs = imgs.shape[0]

                preds = model(imgs)
                peaks = extract_peaks(
                    heatmap=preds["heatmap"],
                    offset=preds["offset"],
                    stride=args.stride,
                    conf_thresh=0.08,
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
        best_th = metrics.get("best_th", 0.30)

        log_str = (
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Loss: {avg_loss:.4f} (HM: {avg_hm_loss:.4f}, Off: {avg_off_loss:.4f}) | "
            f"Best F1: {f1:.4f} @ th={best_th:.2f} | Recall: {rec:.4f} | Prec: {prec:.4f} | "
            f"TP: {metrics['tp']}, FP: {metrics['fp']}, GT: {metrics['total_gt']} | "
            f"Time: {epoch_time:.1f}s"
        )
        print(log_str)

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{avg_loss:.6f},{avg_hm_loss:.6f},{avg_off_loss:.6f},"
                f"{best_th:.4f},{rec:.6f},{prec:.6f},{f1:.6f},"
                f"{metrics['tp']},{metrics['fp']},{metrics['total_gt']},"
                f"{optimizer.param_groups[0]['lr']:.8f}\n"
            )

        ckpt = {
            "epoch": epoch,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "scale": args.scale,
            "stride": args.stride,
            "imgsz": args.imgsz,
        }
        torch.save(ckpt, weights_dir / "last.pt")

        if rec > best_recall:
            best_recall = rec
            torch.save(ckpt, weights_dir / "best_recall.pt")

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = metrics
            torch.save(ckpt, weights_dir / "best_f1.pt")
            print(colorstr("magenta", f"  --> New Best F1: {best_f1:.4f} (@ th={best_th:.2f}) saved to best_f1.pt"))

    print("=" * 90)
    print(colorstr("bold", colorstr("green", f"\n[SUCCESS] Scale YOLO26{args.scale.upper()} 5-Epoch Benchmark Complete!")))
    print(f"Best F1: {best_f1:.4f} | Recall: {best_metrics.get('recall', 0.0):.4f} | Precision: {best_metrics.get('precision', 0.0):.4f} | Best th: {best_metrics.get('best_th', 0.0)}")
    print(f"Checkpoints: {weights_dir.resolve()}\n")
    print("=" * 90)


if __name__ == "__main__":
    main()
