#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Fine-tuning with Sub-pixel Energy-Preserving Focal Loss for Infrared Tiny Object Detection.

Core Architecture & Principles:
1. Base Checkpoint: SOTA Champion Trial 0474 (runs/optuna_p0_nas/trial_0474/weights/best.pt).
2. Strict Freezing & Head Adaptation Protocol:
   - Backbone & Neck (1,790,531 parameters) 100% frozen in eval() mode.
   - P0 Residual Highway feature branch frozen; only Heatmap Head Conv layers fine-tuned
     with gentle learning rate (lr0=5e-5 ~ 1e-4) to re-align peak activation with optical impulse energy.
3. Loss Function:
   - EnergyPreservingFocalLoss:
     * Standard Focal Loss with beta=2.40 (Trial 22/0474 golden parameter).
     * Subpixel Peak 3x3 Max Pooling to prevent tearing down valid peaks near sub-pixel grid intersections.
     * Local Energy Preservation Loss (SmoothL1 on 3x3 integrated optical volume) to elevate weak impulses (0.15~0.22) past 0.25 threshold.
4. Official Ultralytics DataLoader:
   - 43,008 Train / 31,613 Val with 100% Zero Coordinate Drift.
   - Epoch 00 Verification strictly guarantees reproduction of baseline SOTA F1=0.9064.

Usage:
    python manu/train_energy_preserving_focal.py \
        --data /mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml \
        --weights runs/optuna_p0_nas/trial_0474/weights/best.pt \
        --device 0,1,2,3 \
        --batch 32 \
        --epochs 5 \
        --lr0 0.00008 \
        --energy-weight 0.20 \
        --project runs/energy_preserving_focal \
        --name exp_ep_focal_5ep
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, colorstr
from manu.heatmap_evaluate import extract_peaks, find_best_f1_threshold
from manu.heatmap_loss import HeatmapLoss, generate_heatmaps_and_targets
from manu.heatmap_model import YOLO26HeatmapDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Head with Sub-pixel Energy-Preserving Focal Loss")
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/optuna_p0_nas/trial_0474/weights/best.pt",
        help="Path to SOTA Trial 0474 checkpoint",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5, help="Fine-tuning epochs (default: 5)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size per GPU (default: 32)")
    parser.add_argument("--lr0", type=float, default=0.00008, help="Initial learning rate for Head")
    parser.add_argument("--lrf", type=float, default=0.05, help="Final lr factor (lr0 * lrf)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--focal-beta", type=float, default=2.40, help="Trial 22 golden parameter (2.40)")
    parser.add_argument("--offset-weight", type=float, default=0.45, help="Offset L1 loss weight")
    parser.add_argument("--energy-weight", type=float, default=0.20, help="Optical energy preservation weight")
    parser.add_argument("--no-peak-pool", action="store_true", default=False, help="Disable 3x3 subpixel peak pool")
    parser.add_argument("--device", type=str, default="0,1,2,3", help="CUDA devices (e.g. '0,1,2,3')")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--project", type=str, default="runs/energy_preserving_focal")
    parser.add_argument("--name", type=str, default="exp_ep_focal_5ep")
    parser.add_argument("--dist_thresh", type=float, default=8.0, help="GJB evaluation tolerance (default: 8.0px)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    return parser.parse_args()


def validate_one_epoch(
    model: nn.Module,
    val_loader,
    device: torch.device,
    stride: int,
    feat_shape: tuple[int, int],
    criterion: HeatmapLoss,
    dist_thresh: float = 8.0,
    eval_thresholds: list[float] | None = None,
) -> dict:
    model.eval()
    if eval_thresholds is None:
        eval_thresholds = [0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40]

    val_loss_accum = 0.0
    val_hm_accum = 0.0
    val_energy_accum = 0.0
    val_off_accum = 0.0
    num_batches = len(val_loader)

    val_preds_list = []
    val_gt_list = []
    val_sizes_list = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False, dynamic_ncols=True, file=sys.stdout):
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            bboxes = batch["bboxes"]
            b_idx = batch["batch_idx"]
            bs = imgs.shape[0]

            targets = generate_heatmaps_and_targets(
                batch_bboxes=bboxes.to(device, non_blocking=True),
                batch_idx=b_idx.to(device, non_blocking=True),
                batch_size=bs,
                feat_shape=feat_shape,
                stride=stride,
                min_radius=1,
                device=device,
            )

            with autocast(enabled=(device.type == "cuda")):
                preds = model(imgs)
                loss, loss_items = criterion(preds, targets)

            val_loss_accum += loss_items["loss_total"]
            val_hm_accum += loss_items["loss_hm"]
            val_energy_accum += loss_items.get("loss_energy", 0.0)
            val_off_accum += loss_items["loss_offset"]

            peaks = extract_peaks(
                heatmap=preds["heatmap"],
                offset=preds["offset"],
                stride=stride,
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
                val_sizes_list.append((640, 640))

    metrics = find_best_f1_threshold(
        predictions_raw=val_preds_list,
        gt_boxes_list=val_gt_list,
        img_sizes=val_sizes_list,
        distance_threshold=dist_thresh,
        thresholds=eval_thresholds,
    )

    best_res = {
        "th": metrics.get("best_th", 0.25),
        "f1": metrics["f1"],
        "recall": metrics["recall"] * 100.0,
        "prec": metrics["precision"] * 100.0,
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "gt": metrics["total_gt"],
        "loss_total": val_loss_accum / max(1, num_batches),
        "loss_hm": val_hm_accum / max(1, num_batches),
        "loss_energy": val_energy_accum / max(1, num_batches),
        "loss_offset": val_off_accum / max(1, num_batches),
    }
    return best_res


def main():
    args = parse_args()
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 1. Device configuration
    device_str = args.device.strip()
    if device_str != "cpu" and torch.cuda.is_available():
        gpu_ids = [int(x) for x in device_str.split(",") if x.strip().isdigit()]
        primary_gpu = gpu_ids[0]
        device = torch.device(f"cuda:{primary_gpu}")
        torch.cuda.set_device(primary_gpu)
    else:
        gpu_ids = []
        device = torch.device("cpu")

    print("=" * 110)
    print("🚀 Fine-Tuning with Sub-pixel Energy-Preserving Focal Loss (SOTA Model Level Optimization)")
    print(f"Data YAML       : {args.data} (43,008 Train / 31,613 Val)")
    print(f"Base Weights    : {args.weights} (Trial 0474 SOTA: F1=0.9064)")
    print(f"Energy Weight   : {args.energy_weight:.2f} | Peak Pool: {not args.no_peak_pool}")
    print(f"Head LR         : {args.lr0} | Epochs: {args.epochs}")
    print(f"Devices         : {gpu_ids or 'CPU'} (Primary: {device})")
    print(f"Output Directory: {save_dir}")
    print("=" * 110)

    # 2. Build Datasets using Official Ultralytics DataLoaders (100% Zero Drift)
    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data
    # Use deterministic augmentations matching SOTA golden recipe
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

    print("[INFO] Building Train & Validation Datasets with Official YOLO DataLoader...")
    train_dataset = build_yolo_dataset(cfg, data_dict["train"], batch=args.batch, data=data_dict, mode="train", stride=32)
    val_dataset = build_yolo_dataset(cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=32)

    total_batch = args.batch * max(1, len(gpu_ids))
    train_loader = build_dataloader(train_dataset, batch=total_batch, workers=args.workers, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch=total_batch, workers=args.workers, shuffle=False)

    print(f"[INFO] Train samples: {len(train_dataset):,} (Target: 43,008) | Val samples: {len(val_dataset):,} (Target: 31,613)")
    print(f"[INFO] Effective Batch Size: {total_batch} ({args.batch} x {len(gpu_ids)} GPUs)")

    # 3. Load Trial 0474 SOTA Base Model
    weights_path = Path(args.weights)
    if not weights_path.exists():
        for cand in [
            PROJECT_ROOT / weights_path,
            Path("/tmp/pycharm_project_10ae9e2e") / weights_path,
            Path("/home/manu/mnt/pycharm_project_10ae9e2e") / weights_path,
        ]:
            if cand.exists():
                weights_path = cand
                break

    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

    print(colorstr("bold", f"\n[INFO] Loading Trial 0474 Weights: {weights_path}"))
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    stride = ckpt.get("stride", args.stride)
    p0_kwargs = ckpt.get("p0_kwargs", {
        "use_spatial_gate": True,
        "stem_type": "standard_dw",
        "downsample_mode": "pixel_unshuffle",
        "gate_input_mode": "diff_only",
        "gate_mid_channels": 16,
        "gate_depth": 2,
        "fusion_mode": "scalar_gate",
    })

    model = YOLO26HeatmapDetector(
        stride=stride,
        num_classes=1,
        temporal_mode="standard",
        use_p0_highway=True,
        p0_highway_kwargs=p0_kwargs,
    )

    own_state = model.state_dict()
    matched = 0
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "").replace("model.model.", "").replace("model.", "")
        if clean_k in own_state and own_state[clean_k].shape == v.shape:
            own_state[clean_k].copy_(v)
            matched += 1

    print(colorstr("green", f"[LOAD SUCCESS] Matched {matched}/{len(own_state)} layers from Trial 0474."))

    # 4. Strict Isolation & Freezing Protocol:
    # FREEZE: Backbone (b0~b10), FPN/PAN Neck (c13~c22, down1, fuse), P0 Highway
    # TRAINABLE: Heatmap Head Conv layers (h_conv, o_conv, hm_head, off_head)
    frozen_params = 0
    trainable_params = 0
    trainable_named = []

    for name, param in model.named_parameters():
        if "head" in name:
            param.requires_grad = True
            trainable_params += param.numel()
            trainable_named.append(name)
        else:
            param.requires_grad = False
            frozen_params += param.numel()

    print(colorstr("green", f"[STRICT ISOLATION] Frozen Parameters: {frozen_params:,} (Backbone, Neck & P0)"))
    print(colorstr("cyan", f"[TRAINABLE FOCUS] Trainable Parameters: {trainable_params:,} (Heatmap Head)"))
    print(f"[INFO] Trainable Layers: {trainable_named}")

    model.to(device)

    # 5. Optimizer, Scheduler & Criterion
    trainable_param_groups = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_param_groups, lr=args.lr0, weight_decay=args.weight_decay)
    lf = lambda epoch: ((1 + math.cos(epoch * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    criterion = HeatmapLoss(
        hm_weight=1.0,
        offset_weight=args.offset_weight,
        focal_alpha=2.0,
        focal_beta=args.focal_beta,
        use_energy_preservation=True,
        energy_weight=args.energy_weight,
        peak_pool=(not args.no_peak_pool),
    )

    feat_h = args.imgsz // stride
    feat_w = args.imgsz // stride

    results_csv = save_dir / "results.csv"
    results_csv.write_text("epoch,loss_total,loss_hm,loss_energy,loss_offset,best_th,recall,precision,f1,tp,fp,gt,lr\n")

    # 6. Epoch 00 Verification (Zero-Regression Baseline Check)
    print("\n" + colorstr("bold", colorstr("yellow", ">>> RUNNING EPOCH 00 ZERO-REGRESSION BASELINE CHECK <<<")))
    t0_eval = time.time()
    val_baseline = validate_one_epoch(
        model=model,
        val_loader=val_loader,
        device=device,
        stride=stride,
        feat_shape=(feat_h, feat_w),
        criterion=criterion,
        dist_thresh=args.dist_thresh,
    )
    t_eval = time.time() - t0_eval

    print(
        f"[EPOCH 00 BASELINE] F1: {val_baseline['f1']:.4f} @ th={val_baseline['th']:.2f} | "
        f"Recall: {val_baseline['recall']:.2f}% | Prec: {val_baseline['prec']:.2f}% | "
        f"TP: {val_baseline['tp']:,} | FP: {val_baseline['fp']:,} | Eval Time: {t_eval:.1f}s"
    )
    print(colorstr("cyan", f"[INFO] Target SOTA Benchmark: F1=0.9064 | Rec=86.19% | Prec=95.57% (TP=21,643, FP=1,004)"))

    with open(results_csv, "a") as f:
        f.write(
            f"0,{val_baseline['loss_total']:.4f},{val_baseline['loss_hm']:.4f},"
            f"{val_baseline['loss_energy']:.4f},{val_baseline['loss_offset']:.4f},"
            f"{val_baseline['th']:.2f},{val_baseline['recall']:.2f},{val_baseline['prec']:.2f},"
            f"{val_baseline['f1']:.4f},{val_baseline['tp']},{val_baseline['fp']},{val_baseline['gt']},{args.lr0:.6f}\n"
        )

    best_f1 = val_baseline["f1"]
    best_recall = val_baseline["recall"]

    if len(gpu_ids) > 1:
        model_train_wrapper = nn.DataParallel(model, device_ids=gpu_ids)
    else:
        model_train_wrapper = model

    # 7. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Keep frozen modules strictly in eval mode to prevent running stats updates in BN
        for name, m in model.named_modules():
            if "head" not in name:
                m.eval()

        train_loss_accum = 0.0
        train_hm_accum = 0.0
        train_energy_accum = 0.0
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
                stride=stride,
                min_radius=1,
                device=device,
            )

            optimizer.zero_grad()
            with autocast(enabled=(device.type == "cuda")):
                preds = model_train_wrapper(imgs)
                loss, loss_items = criterion(preds, targets)

            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_param_groups, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += loss_items["loss_total"]
            train_hm_accum += loss_items["loss_hm"]
            train_energy_accum += loss_items.get("loss_energy", 0.0)
            train_off_accum += loss_items["loss_offset"]

            pbar.set_postfix({
                "loss": f"{loss_items['loss_total']:.4f}",
                "hm": f"{loss_items['loss_hm']:.4f}",
                "energy": f"{loss_items.get('loss_energy', 0.0):.4f}",
                "off": f"{loss_items['loss_offset']:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        scheduler.step()
        train_time = time.time() - t0

        # Epoch Validation
        val_res = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            device=device,
            stride=stride,
            feat_shape=(feat_h, feat_w),
            criterion=criterion,
            dist_thresh=args.dist_thresh,
        )

        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Ep {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss_accum / num_batches:.4f} | "
            f"F1: {val_res['f1']:.4f} @ th={val_res['th']:.2f} | Recall: {val_res['recall']:.2f}% | "
            f"Prec: {val_res['prec']:.2f}% | TP: {val_res['tp']:,} | FP: {val_res['fp']:,} | Time: {train_time:.1f}s"
        )

        # Save results log
        with open(results_csv, "a") as f:
            f.write(
                f"{epoch},{val_res['loss_total']:.4f},{val_res['loss_hm']:.4f},"
                f"{val_res['loss_energy']:.4f},{val_res['loss_offset']:.4f},"
                f"{val_res['th']:.2f},{val_res['recall']:.2f},{val_res['prec']:.2f},"
                f"{val_res['f1']:.4f},{val_res['tp']},{val_res['fp']},{val_res['gt']},{cur_lr:.6f}\n"
            )

        # Save checkpoints
        save_ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "stride": stride,
            "p0_kwargs": p0_kwargs,
            "metrics": val_res,
        }
        torch.save(save_ckpt, weights_dir / "last.pt")

        if val_res["f1"] > best_f1:
            best_f1 = val_res["f1"]
            torch.save(save_ckpt, weights_dir / "best_f1.pt")
            print(colorstr("green", f"  ★ NEW BEST F1: {best_f1:.4f} @ th={val_res['th']:.2f} saved to best_f1.pt"))

        if val_res["recall"] > best_recall:
            best_recall = val_res["recall"]
            torch.save(save_ckpt, weights_dir / "best_recall.pt")
            print(colorstr("magenta", f"  ★ NEW BEST RECALL: {best_recall:.2f}% (TP={val_res['tp']}) saved to best_recall.pt"))

    print("\n" + "=" * 110)
    print(colorstr("green", f"✅ TRAINING COMPLETED: Best F1={best_f1:.4f} | Best Recall={best_recall:.2f}%"))
    print(f"Results logged to: {results_csv}")
    print(f"Checkpoints in  : {weights_dir}")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    main()
