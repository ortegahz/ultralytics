#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Dedicated 4-GPU Parallel 6-Epoch Fine-Tuning with Soft-IoU + Focal Loss Recall Bias.

Engineering Rationale (P2 from claude-opus-5.md & memory.md):
1. Backbone & Weights: YOLO26HeatmapDetector (Stride=2, P1 High-Res 320x320 Heatmap Output).
   Warm-starts directly from the NEW CHAMPION checkpoint:
   `runs/optuna_median_search/trial_0022/weights/best.pt`
2. Dataset: Official Aligned Temporal Median Mode:
   `/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml`
   Inputs: [I_t, |I_t - W(I_{t-2})|, (I_t - B_t)^+]
3. Loss Configuration:
   - Focal Loss with beta=2.40 (Preserving Trial 22 negative attenuation)
   - Soft-IoU Loss (Region overlap friendly to tiny point targets, default weight=1.0)
   - Offset RegL1Loss (Sub-pixel fine tuning, weight=0.45)
   Total Loss = hm_weight * FocalLoss + soft_iou_weight * SoftIoULoss + offset_weight * RegL1Loss
4. Training Protocol:
   - 6 Epochs max to strictly prevent Gaussian overfitting.
   - Initial lr0=1.0e-4 (gentle cosine annealing to 1.0e-5).
   - Real-time threshold sweeping (0.15~0.60) every epoch at Distance <= 8.0px.
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
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26 Heatmap with Soft-IoU Recall Bias")
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/optuna_median_search/trial_0022/weights/best.pt",
        help="Path to Trial 22 champion checkpoint",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=6, help="Strictly 6 epochs to prevent overfitting")
    parser.add_argument("--batch", type=int, default=32, help="Batch size per GPU (total = batch * num_gpus)")
    parser.add_argument("--lr0", type=float, default=0.0001, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.1, help="Final lr factor (lr0 * lrf)")
    parser.add_argument("--weight_decay", type=float, default=3.5e-5, help="Weight decay (matched Trial 22)")
    parser.add_argument("--focal_beta", type=float, default=2.4, help="Focal beta for negative softening")
    parser.add_argument("--soft_iou_weight", type=float, default=1.0, help="Soft-IoU loss weight")
    parser.add_argument("--hm_weight", type=float, default=1.0, help="Focal loss weight")
    parser.add_argument("--offset_weight", type=float, default=0.45, help="Offset L1 loss weight")
    parser.add_argument("--min_radius", type=int, default=1, help="Min Gaussian radius")
    parser.add_argument("--device", type=str, default="0,1,2,3", help="CUDA devices (e.g. '0,1,2,3')")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--project", type=str, default="runs/finetune_soft_iou")
    parser.add_argument("--name", type=str, default="exp_trial22_softiou")
    parser.add_argument("--dist_thresh", type=float, default=8.0, help="GJB evaluation tolerance (default: 8.0px)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 1. Device and DataParallel configuration
    device_str = args.device.strip()
    if device_str != "cpu" and torch.cuda.is_available():
        gpu_ids = [int(x) for x in device_str.split(",") if x.strip().isdigit()]
        primary_gpu = gpu_ids[0]
        device = torch.device(f"cuda:{primary_gpu}")
        torch.cuda.set_device(primary_gpu)
    else:
        gpu_ids = []
        device = torch.device("cpu")

    print("=" * 100)
    print("   UAV Tiny Object Detection: 4-GPU 6-Epoch Fine-Tuning with Soft-IoU Loss (P2)")
    print(f"   Checkpoint        : {args.weights}")
    print(f"   Dataset           : {args.data}")
    print(f"   Devices           : {gpu_ids or 'CPU'} (Primary: {device})")
    print(f"   Epochs            : {args.epochs} | lr0: {args.lr0} | Stride: {args.stride}")
    print(f"   Focal beta        : {args.focal_beta} | Soft-IoU weight: {args.soft_iou_weight} | Offset weight: {args.offset_weight}")
    print("=" * 100)

    # 2. Build Datasets & DataLoaders
    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data
    # Augmentation set tuned for infrared tiny targets
    cfg.hsv_h = 0.0
    cfg.hsv_s = 0.0
    cfg.hsv_v = 0.0
    cfg.degrees = 0.0
    cfg.shear = 0.0
    cfg.perspective = 0.0
    cfg.translate = 0.04
    cfg.scale = 0.05
    cfg.fliplr = 0.5
    cfg.flipud = 0.0
    cfg.mosaic = 0.0
    cfg.mixup = 0.0
    cfg.copy_paste = 0.0

    print("[INFO] Building Train & Validation Datasets...")
    train_dataset = build_yolo_dataset(cfg, data_dict["train"], batch=args.batch, data=data_dict, mode="train", stride=32)
    val_dataset = build_yolo_dataset(cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=32)

    total_batch = args.batch * max(1, len(gpu_ids))
    train_loader = build_dataloader(train_dataset, batch=total_batch, workers=args.workers, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch=total_batch, workers=args.workers, shuffle=False)

    print(f"[INFO] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"[INFO] Effective Batch Size: {total_batch} ({args.batch} x {len(gpu_ids)} GPUs)")

    # 3. Model warm-start loading
    weights_path = Path(args.weights)
    if not weights_path.exists():
        alt = REPO_ROOT / args.weights
        if alt.exists():
            weights_path = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

    print(colorstr("bold", f"\n[INFO] Warm-starting model from Trial 22: {weights_path}"))
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()

    model = YOLO26HeatmapDetector(stride=args.stride, num_classes=1, temporal_mode="standard")
    model.load_state_dict(state_dict)
    model.to(device)

    if len(gpu_ids) > 1:
        model_module = nn.DataParallel(model, device_ids=gpu_ids, output_device=primary_gpu)
    else:
        model_module = model

    # 4. Optimizer, Scheduler & Criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr0, weight_decay=args.weight_decay)
    lf = lambda epoch: ((1 + math.cos(epoch * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    criterion = HeatmapLoss(
        hm_weight=args.hm_weight,
        offset_weight=args.offset_weight,
        soft_iou_weight=args.soft_iou_weight,
        focal_beta=args.focal_beta,
    )

    feat_h = args.imgsz // args.stride
    feat_w = args.imgsz // args.stride

    best_recall = 0.0
    best_f1 = 0.0
    best_metrics = {}

    csv_path = save_dir / "results.csv"
    csv_header = "epoch,train/loss,train/loss_hm,train/loss_iou,train/loss_off,metrics/best_th,metrics/recall(B),metrics/precision(B),metrics/f1(B),metrics/tp,metrics/fp,metrics/gt,lr\n"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_header)

    # 5. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_hm_accum = 0.0
        train_iou_accum = 0.0
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
                print(colorstr("red", f"\n[WARN] NaN/Inf loss at Epoch {epoch}, Batch {batch_i}. Skipping step."))
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += loss_items["loss_total"]
            train_hm_accum += loss_items["loss_hm"]
            train_iou_accum += loss_items.get("loss_iou", 0.0)
            train_off_accum += loss_items["loss_offset"]

            pbar.set_postfix({
                "loss": f"{loss_items['loss_total']:.4f}",
                "hm": f"{loss_items['loss_hm']:.4f}",
                "iou": f"{loss_items.get('loss_iou', 0.0):.4f}",
                "off": f"{loss_items['loss_offset']:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
            })

        scheduler.step()
        epoch_time = time.time() - t0

        avg_loss = train_loss_accum / max(num_batches, 1)
        avg_hm_loss = train_hm_accum / max(num_batches, 1)
        avg_iou_loss = train_iou_accum / max(num_batches, 1)
        avg_off_loss = train_off_accum / max(num_batches, 1)

        # 6. Full Validation Evaluation
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
        best_th = metrics.get("best_th", 0.25)

        log_str = (
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Loss: {avg_loss:.4f} (HM: {avg_hm_loss:.4f}, IoU: {avg_iou_loss:.4f}, Off: {avg_off_loss:.4f}) | "
            f"Best F1: {f1:.4f} @ th={best_th:.2f} | Recall: {rec:.4f} | Prec: {prec:.4f} | "
            f"TP: {metrics['tp']}, FP: {metrics['fp']}, GT: {metrics['total_gt']} | "
            f"Time: {epoch_time:.1f}s"
        )
        print(log_str)

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{avg_loss:.6f},{avg_hm_loss:.6f},{avg_iou_loss:.6f},{avg_off_loss:.6f},"
                f"{best_th:.4f},{rec:.6f},{prec:.6f},{f1:.6f},"
                f"{metrics['tp']},{metrics['fp']},{metrics['total_gt']},"
                f"{optimizer.param_groups[0]['lr']:.8f}\n"
            )

        # Checkpoint saving
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "stride": args.stride,
            "imgsz": args.imgsz,
        }
        torch.save(ckpt, weights_dir / "last.pt")

        if rec > best_recall:
            best_recall = rec
            torch.save(ckpt, weights_dir / "best_recall.pt")
            print(colorstr("green", f"  --> New Best Recall: {best_recall:.4f} saved to best_recall.pt"))

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = metrics
            torch.save(ckpt, weights_dir / "best_f1.pt")
            print(colorstr("magenta", f"  --> New Best F1: {best_f1:.4f} (@ th={best_th:.2f}) saved to best_f1.pt"))

    print("=" * 100)
    print(colorstr("bold", colorstr("green", f"\n[SUCCESS] 6-Epoch Soft-IoU Fine-Tuning Complete!")))
    print(f"Target Baseline (Trial 22) : F1 = 0.9057 | Recall = 86.11% | Precision = 95.51%")
    print(f"Best Fine-Tuned Model      : F1 = {best_f1:.4f} | Recall = {best_metrics.get('recall', 0.0):.4f} | Precision = {best_metrics.get('precision', 0.0):.4f}")
    print(f"Checkpoints Saved to       : {weights_dir.resolve()}\n")
    print("=" * 100)


if __name__ == "__main__":
    main()
