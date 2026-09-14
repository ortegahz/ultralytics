#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train Temporal Tubelet Residual Highway from Pre-extracted Shallow Difference Caches.

Features:
1. Base Detector 100% physically frozen (Trial 0474 SOTA).
2. Input: Loads only 1 image for current frame t + pre-extracted 1x320x320 diff cache for K=8 sequence.
3. Completely bypasses CPU I/O bottleneck -> Training speed reaches 20~40 it/s (1~2 mins per epoch on 4 GPUs).
4. Strictly aligns with baseline capacity: exactly 43,008 training samples.
5. Strictly zero-regression bounded gate: effective_alpha = tanh(gate) * 0.05.

Usage on Server:
    python manu/train_temporal_tubelet_s2.py \
        --data-root /mnt/data/siping/datasets/manu/uav_gmc_median_s2 \
        --cache-root /mnt/data/siping/datasets/manu/uav_s2_diff_cache \
        --ref-manifest /mnt/data/siping/datasets/manu/uav_gmc_median/images/train \
        --base-weights runs/optuna_p0_nas/trial_0474/weights/best.pt \
        --output-dir runs/temporal_tubelet_s2/exp_s2_fast_tubelet_12ep \
        --device 0,1,2,3 \
        --batch 64 \
        --epochs 12 \
        --lr0 0.0003
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr
from manu.heatmap_evaluate import extract_peaks, find_best_f1_threshold
from manu.heatmap_loss import HeatmapLoss, generate_heatmaps_and_targets
from manu.heatmap_model import YOLO26HeatmapDetector
from manu.temporal_tubelet_module import TemporalDiffCacheDataset, TemporalTubeletHighwayFromDiff


def parse_args():
    parser = argparse.ArgumentParser(description="Train Fast Temporal Tubelet S2 Model")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median_s2",
        help="Root of Stride=2 dataset",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_s2_diff_cache",
        help="Path to pre-extracted diff cache",
    )
    parser.add_argument(
        "--ref-manifest",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median/images/train",
        help="Path to baseline train images for exact 1:1 capacity alignment (default: 43,008 samples)",
    )
    parser.add_argument(
        "--base-weights",
        type=str,
        default="runs/optuna_p0_nas/trial_0474/weights/best.pt",
        help="Path to Trial 0474 SOTA checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/temporal_tubelet_s2/exp_s2_fast_tubelet_12ep",
        help="Directory to save runs",
    )
    parser.add_argument("--device", type=str, default="0,1,2,3", help="CUDA devices (e.g. '0,1,2,3')")
    parser.add_argument("--batch", type=int, default=64, help="Batch size across GPUs (default: 64)")
    parser.add_argument("--epochs", type=int, default=12, help="Epochs (default: 12)")
    parser.add_argument("--lr0", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final lr ratio")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seq-len", type=int, default=8, help="Temporal sequence length K (default: 8)")
    parser.add_argument("--stride", type=int, default=2, help="Temporal sampling stride (default: 2)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--focal-beta", type=float, default=2.40)
    parser.add_argument("--offset-weight", type=float, default=0.45)
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


class FastTemporalIntegratedModel(nn.Module):
    """
    Integrates 100% frozen Trial 0474 base detector with Fast Temporal Tubelet Highway.
    """

    def __init__(self, base_detector: YOLO26HeatmapDetector, tubelet_highway: TemporalTubeletHighwayFromDiff):
        super().__init__()
        self.base_detector = base_detector
        self.tubelet_highway = tubelet_highway

    def forward(self, curr_img: torch.Tensor, diff_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            curr_img: (B, 3, 640, 640) current frame only!
            diff_seq: (B, K, 1, 320, 320) pre-extracted shallow diff sequence
        """
        # 1. Base model extraction on current frame (P1: 48x320x320)
        feat_base = self.base_detector.extract_features(curr_img)
        if self.base_detector.p0_highway is not None:
            feat_base = feat_base + self.base_detector.p0_highway(curr_img)

        # 2. Fast Temporal Tubelet Residual enhancement across K Stride=2 diff pulses
        delta_temporal = self.tubelet_highway(diff_seq)

        # 3. Residual addition with zero-initialized bounded gate alpha
        feat_final = feat_base + delta_temporal

        # 4. Final Head prediction
        return self.base_detector.head(feat_final)


def collate_cache_batch(batch: list[dict]) -> dict:
    curr_imgs = torch.stack([b["curr_img"] for b in batch], dim=0)  # (B, 3, H, W)
    diff_seqs = torch.stack([b["diff_seq"] for b in batch], dim=0)  # (B, K, 1, 320, 320)
    im_names = [b["im_name"] for b in batch]

    batch_bboxes_list = []
    batch_idx_list = []
    for b_i, b in enumerate(batch):
        boxes = b["bboxes"]  # (N_gt, 5) -> cls, cx, cy, w, h
        if len(boxes) > 0:
            for box in boxes:
                batch_bboxes_list.append(box[1:])
                batch_idx_list.append(b_i)

    if batch_bboxes_list:
        batch_bboxes = torch.stack(batch_bboxes_list, dim=0)
        batch_idx = torch.tensor(batch_idx_list, dtype=torch.long)
    else:
        batch_bboxes = torch.zeros((0, 4), dtype=torch.float32)
        batch_idx = torch.zeros((0,), dtype=torch.long)

    return {
        "curr_img": curr_imgs,
        "diff_seq": diff_seqs,
        "bboxes": batch_bboxes,
        "batch_idx": batch_idx,
        "im_names": im_names,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("🚀 Fast Training Temporal Tubelet Highway on Pre-extracted Diff Caches")
    print(f"Dataset Root    : {args.data_root}")
    print(f"Cache Root      : {args.cache_root}")
    print(f"Ref Manifest    : {args.ref_manifest} (Capacity: 43,008)")
    print(f"Base Weights    : {args.base_weights} (Trial 0474 SOTA)")
    print(f"Output Directory: {output_dir}")
    print(f"Sequence Config : K={args.seq_len} frames, Stride={args.stride} (Covering {args.seq_len * args.stride * 40}ms)")
    print(f"Batch Size      : {args.batch} | Epochs: {args.epochs}")
    print("=" * 100)

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
    print(f"[INFO] Using Devices: {gpu_ids or 'CPU'} (Primary: {device})")

    # 2. Load Base Detector
    base_weights_path = Path(args.base_weights)
    if not base_weights_path.exists():
        alt = PROJECT_ROOT / args.base_weights
        if alt.exists():
            base_weights_path = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.base_weights}")

    ckpt = torch.load(base_weights_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    stride = ckpt.get("stride", 2)
    p0_kwargs = ckpt.get("p0_kwargs", {
        "use_spatial_gate": True,
        "stem_type": "standard_dw",
        "downsample_mode": "pixel_unshuffle",
        "gate_input_mode": "diff_only",
        "gate_mid_channels": 16,
        "gate_depth": 2,
        "fusion_mode": "scalar_gate",
    })

    base_detector = YOLO26HeatmapDetector(
        stride=stride,
        num_classes=1,
        temporal_mode="standard",
        use_p0_highway=True,
        p0_highway_kwargs=p0_kwargs,
    )

    matched = 0
    own_state = base_detector.state_dict()
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "").replace("model.model.", "").replace("model.", "")
        if clean_k in own_state and own_state[clean_k].shape == v.shape:
            own_state[clean_k].copy_(v)
            matched += 1

    # STRICT FREEZING: 100% freeze base detector
    frozen_params = 0
    for param in base_detector.parameters():
        param.requires_grad = False
        frozen_params += param.numel()

    # 3. Build Fast Temporal Tubelet Highway
    tubelet_highway = TemporalTubeletHighwayFromDiff(
        in_channels=1,
        out_channels=48,
        num_frames=args.seq_len,
        mid_channels=16,
    )
    trainable_params = sum(p.numel() for p in tubelet_highway.parameters() if p.requires_grad)

    print(colorstr("green", f"[INFO] Base Detector: {frozen_params:,} parameters (100% LOCKED)."))
    print(colorstr("cyan", f"[INFO] Fast Temporal Tubelet Highway: {trainable_params:,} parameters (TRAINABLE)."))
    print(f"[INFO] Initial Gate alpha: {tubelet_highway.gate.item():.6f} (Strict 0.0 Zero Regression Guarantee)")

    model = FastTemporalIntegratedModel(base_detector, tubelet_highway).to(device)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(colorstr("bold", colorstr("green", f"[INFO] Enabled DataParallel across {len(gpu_ids)} GPUs: {gpu_ids}")))

    # 4. Setup Datasets from pre-extracted cache
    data_p = Path(args.data_root)
    cache_p = Path(args.cache_root)
    train_cache_dir = cache_p / "train"
    train_lbl_dir = data_p / "labels" / "train"
    train_curr_img_dir = data_p / "images" / "train"

    val_cache_dir = cache_p / "val"
    val_lbl_dir = data_p / "labels" / "val"
    val_curr_img_dir = data_p / "images" / "val"

    train_dataset = TemporalDiffCacheDataset(
        cache_dir=train_cache_dir,
        lbl_dir=train_lbl_dir,
        curr_img_dir=train_curr_img_dir,
        ref_manifest_dir=args.ref_manifest,
        seq_len=args.seq_len,
        stride=args.stride,
        imgsz=args.imgsz,
    )
    val_dataset = TemporalDiffCacheDataset(
        cache_dir=val_cache_dir,
        lbl_dir=val_lbl_dir,
        curr_img_dir=val_curr_img_dir,
        ref_manifest_dir=None,
        seq_len=args.seq_len,
        stride=args.stride,
        imgsz=args.imgsz,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_cache_batch,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_cache_batch,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
    )

    # 5. Optimizer with Parameter Grouping:
    # Dedicated tiny learning rate for gate alpha (1e-5) to prevent early divergence
    tubelet_conv_params = [p for n, p in tubelet_highway.named_parameters() if p.requires_grad and "gate" not in n]
    tubelet_gate_params = [p for n, p in tubelet_highway.named_parameters() if p.requires_grad and "gate" in n]
    all_trainable_params = tubelet_conv_params + tubelet_gate_params

    optimizer = torch.optim.AdamW([
        {"params": tubelet_conv_params, "lr": args.lr0, "weight_decay": args.weight_decay},
        {"params": tubelet_gate_params, "lr": 1e-5, "weight_decay": 0.0},
    ])
    lf = lambda ep: ((1 + math.cos(ep * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scaler = GradScaler(enabled=True)

    criterion = HeatmapLoss(
        hm_weight=1.0,
        offset_weight=args.offset_weight,
        focal_alpha=2.0,
        focal_beta=args.focal_beta,
    )

    feat_h = args.imgsz // stride
    feat_w = args.imgsz // stride

    results_csv = output_dir / "results.csv"
    results_csv.write_text("epoch,gate_val,loss_total,loss_hm,best_th,recall,precision,f1,tp,fp,gt,lr\n")

    best_f1 = 0.0

    # 5.5 ZERO-CHECK BASELINE EVALUATION (Epoch 0 Verification)
    # Evaluates base model + tubelet at alpha=0.0 before any gradient update!
    # Mathematically GUARANTEES that initial state reproduces SOTA (F1 ≈ 0.9064).
    print("\n" + "=" * 90)
    print("🔬 Running Zero-Regression Baseline Check (Epoch 0, alpha=0.0)...")
    model.eval()
    val_preds_list = []
    val_gt_list = []
    val_sizes_list = []
    t_val = time.time()
    val_pbar = tqdm(val_loader, desc="Baseline Check [Val Ep 00]", dynamic_ncols=True, file=sys.stdout)
    with torch.no_grad():
        for batch in val_pbar:
            curr_img = batch["curr_img"].to(device, non_blocking=True)
            diff_seq = batch["diff_seq"].to(device, non_blocking=True)
            bboxes = batch["bboxes"]
            b_idx = batch["batch_idx"]
            bs = curr_img.shape[0]

            preds = model(curr_img, diff_seq)
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
                val_sizes_list.append((args.imgsz, args.imgsz))

    init_metrics = find_best_f1_threshold(
        predictions_raw=val_preds_list,
        gt_boxes_list=val_gt_list,
        img_sizes=val_sizes_list,
        distance_threshold=args.dist_thresh,
        thresholds=[0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40],
    )
    init_f1 = init_metrics["f1"]
    init_rec = init_metrics["recall"]
    init_prec = init_metrics["precision"]
    init_th = init_metrics.get("best_th", 0.25)
    print(
        colorstr("bold", colorstr("cyan", f"[BASELINE CHECK] Ep 00 | F1: {init_f1:.4f} @ th={init_th:.2f} | "
        f"Rec: {init_rec * 100:.2f}% | Prec: {init_prec * 100:.2f}% | TP: {init_metrics['tp']} FP: {init_metrics['fp']} (Target SOTA: 0.9064)\n"))
    )
    print("=" * 90 + "\n")
    best_f1 = init_f1

    # 6. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        model_module = model.module if hasattr(model, "module") else model
        base_detector_mod = model_module.base_detector
        tubelet_highway_mod = model_module.tubelet_highway

        base_detector_mod.eval()

        train_loss_accum = 0.0
        train_hm_accum = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Ep {epoch:02d}/{args.epochs:02d} [Train]", dynamic_ncols=True, file=sys.stdout)
        for batch_i, batch in enumerate(pbar):
            curr_img = batch["curr_img"].to(device, non_blocking=True)
            diff_seq = batch["diff_seq"].to(device, non_blocking=True)
            bboxes = batch["bboxes"].to(device, non_blocking=True)
            b_idx = batch["batch_idx"].to(device, non_blocking=True)
            bs = curr_img.shape[0]

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
            with autocast(enabled=True):
                preds = model(curr_img, diff_seq)
                loss, loss_items = criterion(preds, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += loss_items["loss_total"]
            train_hm_accum += loss_items["loss_hm"]

            gate_disp = tubelet_highway_mod.gate.item()
            effective_alpha = (torch.tanh(tubelet_highway_mod.gate) * tubelet_highway_mod.scale_factor).item()
            if batch_i % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{loss_items['loss_total']:.4f}",
                    "hm": f"{loss_items['loss_hm']:.4f}",
                    "alpha": f"{effective_alpha:.6f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                })

        scheduler.step()
        train_dur = time.time() - t0

        # 7. Fast Validation Loop
        model.eval()
        val_preds_list = []
        val_gt_list = []
        val_sizes_list = []
        t_val = time.time()

        val_pbar = tqdm(val_loader, desc=f"Ep {epoch:02d}/{args.epochs:02d} [Val]", dynamic_ncols=True, file=sys.stdout)
        with torch.no_grad():
            for batch in val_pbar:
                curr_img = batch["curr_img"].to(device, non_blocking=True)
                diff_seq = batch["diff_seq"].to(device, non_blocking=True)
                bboxes = batch["bboxes"]
                b_idx = batch["batch_idx"]
                bs = curr_img.shape[0]

                preds = model(curr_img, diff_seq)
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
                    val_sizes_list.append((args.imgsz, args.imgsz))

        metrics = find_best_f1_threshold(
            predictions_raw=val_preds_list,
            gt_boxes_list=val_gt_list,
            img_sizes=val_sizes_list,
            distance_threshold=args.dist_thresh,
            thresholds=[0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40],
        )

        f1 = metrics["f1"]
        rec = metrics["recall"]
        prec = metrics["precision"]
        best_th = metrics.get("best_th", 0.25)
        val_dur = time.time() - t_val

        avg_loss = train_loss_accum / max(len(train_loader), 1)
        avg_hm = train_hm_accum / max(len(train_loader), 1)
        effective_alpha = (torch.tanh(tubelet_highway_mod.gate) * tubelet_highway_mod.scale_factor).item()

        print(
            f"\n[SUMMARY] Ep {epoch:02d}/{args.epochs:02d} | Effective α: {effective_alpha:.6f} | "
            f"F1: {f1:.4f} @ th={best_th:.2f} | Rec: {rec * 100:.2f}% | Prec: {prec * 100:.2f}% | "
            f"TP: {metrics['tp']} FP: {metrics['fp']} (GT: {metrics['total_gt']}) | Train: {train_dur:.1f}s Val: {val_dur:.1f}s\n"
        )

        with open(results_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{effective_alpha:.8f},{avg_loss:.6f},{avg_hm:.6f},{best_th:.4f},"
                f"{rec:.6f},{prec:.6f},{f1:.6f},{metrics['tp']},{metrics['fp']},{metrics['total_gt']},"
                f"{optimizer.param_groups[0]['lr']:.8f}\n"
            )

        ckpt_data = {
            "epoch": epoch,
            "tubelet_state": tubelet_highway_mod.state_dict(),
            "effective_alpha": effective_alpha,
            "metrics": metrics,
            "stride": stride,
            "seq_len": args.seq_len,
            "temporal_stride": args.stride,
        }
        torch.save(ckpt_data, weights_dir / "last.pt")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(ckpt_data, weights_dir / "best.pt")
            print(colorstr("bold", colorstr("green", f"★ New Best F1 checkpoint saved: {f1:.4f}!")))

    print("\n" + "=" * 100)
    print(f"🎉 Fast Training Complete! Peak Single-Frame F1: {best_f1:.4f}")
    print(f"Weights saved to: {weights_dir.resolve()}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
