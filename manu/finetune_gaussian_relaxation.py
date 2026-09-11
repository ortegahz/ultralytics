#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Dedicated 4-GPU 5-Epoch Fine-Tuning with Gaussian Relaxation Focal Loss on Trial 22 SOTA Base.

Core Hypothesis (Option 1 Single-Frame Breakthrough):
- Current SOTA (Trial 22) achieves F1=0.9057 (Recall=86.11%, Prec=95.51%), with ~14% misses concentrated
  in extremely weak IR impulses (SCR < 1.5).
- Standard CenterNet Focal Loss enforces harsh negative penalties on Gaussian shoulders, over-suppressing
  diffuse/weak targets.
- By introducing `relaxation_tau > 0.0` (Gaussian relaxation), we soften the background penalty in the
  immediate transition zone around the impulse center, allowing low-SCR targets to emerge above threshold.
- We also allow `pos_weight` (> 1.0) to give gentle boost to rare positive spikes.
- Strict 5-6 epoch conservative fine-tuning with cosine annealing and low lr0 prevents overfitting.

Usage on Server:
    python manu/finetune_gaussian_relaxation.py \
        --data /mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml \
        --weights runs/optuna_median_search/trial_0022/weights/best.pt \
        --device 0,1,2,3 \
        --batch 32 \
        --epochs 5 \
        --lr0 5e-5 \
        --relaxation-tau 0.25 \
        --pos-weight 1.2 \
        --focal-beta 2.4 \
        --project runs/finetune_relaxation \
        --name exp_relaxation_tau025
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
    parser = argparse.ArgumentParser(description="Fine-tune Trial 22 with Gaussian Relaxation Focal Loss")
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
        help="Path to Trial 22 SOTA checkpoint",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5, help="5 epochs conservative fine-tuning")
    parser.add_argument("--batch", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--lr0", type=float, default=5e-5, help="Gentle initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.1, help="Final lr factor (lr0 * lrf)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--focal-beta", type=float, default=2.4, help="Trial 22 golden parameter (2.4)")
    parser.add_argument("--relaxation-tau", type=float, default=0.25, help="Gaussian relaxation factor tau (e.g. 0.15~0.35)")
    parser.add_argument("--pos-weight", type=float, default=1.2, help="Positive center focus weight (default: 1.2)")
    parser.add_argument("--offset-weight", type=float, default=0.45, help="L1 offset loss weight")
    parser.add_argument("--device", type=str, default="0,1,2,3", help="CUDA devices (e.g. '0,1,2,3')")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--project", type=str, default="runs/finetune_relaxation")
    parser.add_argument("--name", type=str, default="exp_relaxation_tau025")
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
    print("   UAV Tiny Object Detection: Gaussian Relaxation Focal Loss Fine-Tuning")
    print(f"   Base Checkpoint : {args.weights} (Trial 22 SOTA)")
    print(f"   Dataset         : {args.data}")
    print(f"   Devices         : {gpu_ids or 'CPU'} (Primary: {device})")
    print(f"   Focal Beta      : {args.focal_beta} | Relaxation Tau: {args.relaxation_tau} | Pos Weight: {args.pos_weight}")
    print(f"   Epochs          : {args.epochs} | lr0: {args.lr0} | Stride: {args.stride}")
    print("=" * 100)

    # 2. Build Datasets & DataLoaders
    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data
    cfg.hsv_h = 0.0
    cfg.hsv_s = 0.0
    cfg.hsv_v = 0.0
    cfg.degrees = 0.0
    cfg.shear = 0.0
    cfg.perspective = 0.0
    cfg.translate = 0.08
    cfg.scale = 0.15
    cfg.fliplr = 0.5
    cfg.flipud = 0.0
    cfg.mosaic = 0.10
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

    # 3. Model warm-start loading from Trial 22
    weights_path = Path(args.weights)
    if not weights_path.exists():
        alt = REPO_ROOT / args.weights
        if alt.exists():
            weights_path = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

    print(colorstr("bold", f"\n[INFO] Warm-starting model from Golden Baseline: {weights_path}"))
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

    # 4. Optimizer, Scheduler & Criterion with Gaussian Relaxation
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr0, weight_decay=args.weight_decay)
    lf = lambda epoch: ((1 + math.cos(epoch * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    criterion = HeatmapLoss(
        hm_weight=1.0,
        offset_weight=args.offset_weight,
        focal_alpha=2.0,
        focal_beta=args.focal_beta,
        relaxation_tau=args.relaxation_tau,
        pos_weight=args.pos_weight,
    )

    feat_h = args.imgsz // args.stride
    feat_w = args.imgsz // args.stride

    best_recall = 0.0
    best_f1 = 0.0
    best_metrics = {}

    # 5. Training Loop
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

            targets = generate_heatmaps_and_targets(
                batch_bboxes=bboxes,
                batch_idx=b_idx,
                batch_size=bs,
                feat_shape=(feat_h, feat_w),
                stride=args.stride,
                min_radius=1,
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
            thresholds=[0.10, 0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50],
        )

        rec = metrics["recall"]
        prec = metrics["precision"]
        f1 = metrics["f1"]
        best_th = metrics.get("best_th", 0.25)

        log_str = (
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Loss: {avg_loss:.4f} (HM: {avg_hm_loss:.4f}, Off: {avg_off_loss:.4f}) | "
            f"Best F1: {f1:.4f} @ th={best_th:.2f} | Recall: {rec:.4f} | Prec: {prec:.4f} | "
            f"TP: {metrics['tp']}, FP: {metrics['fp']}, GT: {metrics['total_gt']} | "
            f"Time: {epoch_time:.1f}s"
        )
        print(log_str)

        # Log CSV results
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

        # Checkpoint saving
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "stride": args.stride,
            "imgsz": args.imgsz,
            "relaxation_tau": args.relaxation_tau,
            "pos_weight": args.pos_weight,
            "focal_beta": args.focal_beta,
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
    print(colorstr("bold", colorstr("green", f"\n[SUCCESS] Gaussian Relaxation Fine-Tuning Complete!")))
    print(f"Target Baseline (Trial 22) : F1 = 0.9057 | Recall = 86.11% | Precision = 95.51% (TP=21,622, FP=1,012)")
    print(f"Best Relaxed Model         : F1 = {best_f1:.4f} | Recall = {best_metrics.get('recall', 0.0):.4f} | Precision = {best_metrics.get('precision', 0.0):.4f}")
    print(f"Checkpoints Saved to       : {weights_dir.resolve()}\n")
    print("=" * 100)


if __name__ == "__main__":
    main()
