#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train RK3588-Friendly Local Cross-Window Attention Temporal Model with Official YOLO Dataset Alignment.

Strict Protocols:
1. Base Detector 100% physically frozen (Trial 0474 SOTA: F1 = 0.9064).
2. Uses official Ultralytics `build_yolo_dataset` + `build_dataloader`:
   - Exact 43,008 samples for training from uav_gmc_median/data.yaml.
   - Exact 31,613 samples for validation from uav_gmc_median/data.yaml.
   - Zero Letterbox / RGB channel drift.
3. Zero-check on Epoch 0 MUST reproduce baseline SOTA (F1 ≈ 0.9064).
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

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, colorstr
from manu.heatmap_evaluate import extract_peaks, find_best_f1_threshold
from manu.heatmap_loss import HeatmapLoss, generate_heatmaps_and_targets
from manu.heatmap_model import YOLO26HeatmapDetector
from manu.local_cross_window_module import LocalCrossWindowAttentionHighway
from manu.official_aligned_dataset import OfficialAlignedCrossAttentionDataset, collate_aligned_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Train Aligned RK3588 Cross-Attention Temporal Model")
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml",
        help="Path to official baseline data.yaml (strictly 43,008 train / 31,613 val)",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_s2_diff_cache",
        help="Path to pre-extracted diff cache",
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
        default="runs/local_cross_attention/exp_rk3588_aligned_6ep",
        help="Directory to save runs",
    )
    parser.add_argument("--device", type=str, default="0,1,2,3", help="CUDA devices (e.g. '0,1,2,3')")
    parser.add_argument("--batch", type=int, default=16, help="Batch size per GPU (default: 16, effective: 64 across 4 GPUs)")
    parser.add_argument("--epochs", type=int, default=6, help="Epochs (default: 6)")
    parser.add_argument("--lr0", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final lr ratio")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seq-len", type=int, default=4, help="Historical sequence length K (default: 4)")
    parser.add_argument("--stride", type=int, default=2, help="Temporal sampling stride (default: 2)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--focal-beta", type=float, default=2.40)
    parser.add_argument("--offset-weight", type=float, default=0.45)
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


class LocalCrossAttentionIntegratedModel(nn.Module):
    def __init__(
        self,
        base_detector: YOLO26HeatmapDetector,
        cross_attn_highway: LocalCrossWindowAttentionHighway,
    ):
        super().__init__()
        self.base_detector = base_detector
        self.cross_attn_highway = cross_attn_highway

    def forward(self, curr_img: torch.Tensor, diff_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        feat_base = self.base_detector.extract_features(curr_img)
        if self.base_detector.p0_highway is not None:
            feat_base = feat_base + self.base_detector.p0_highway(curr_img)

        delta_temporal = self.cross_attn_highway(feat_base, diff_seq)
        feat_final = feat_base + delta_temporal
        return self.base_detector.head(feat_final)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("🚀 Training Official-Aligned RK3588 Local Cross-Window Attention Temporal Model")
    print(f"Data YAML       : {args.data} (Strict 43,008 Train / 31,613 Val)")
    print(f"Cache Root      : {args.cache_root}")
    print(f"Base Weights    : {args.base_weights} (Trial 0474 SOTA: F1=0.9064)")
    print(f"Output Directory: {output_dir}")
    print(f"Sequence Config : K={args.seq_len} frames, Stride={args.stride}")
    print(f"Batch Size      : {args.batch} per GPU | Epochs: {args.epochs}")
    print("=" * 110)

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
    print(f"[INFO] Using Devices: {gpu_ids or 'CPU'} (Primary: {device})")

    # 2. Build Datasets using Official YOLO format
    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data
    # Disable augmentations for deterministic training & validation
    cfg.hsv_h = 0.0
    cfg.hsv_s = 0.0
    cfg.hsv_v = 0.0
    cfg.degrees = 0.0
    cfg.shear = 0.0
    cfg.perspective = 0.0
    cfg.translate = 0.0
    cfg.scale = 0.0
    cfg.fliplr = 0.0
    cfg.flipud = 0.0
    cfg.mosaic = 0.0
    cfg.mixup = 0.0
    cfg.copy_paste = 0.0

    print("[INFO] Building Base YOLO Datasets...")
    base_train_ds = build_yolo_dataset(cfg, data_dict["train"], batch=args.batch, data=data_dict, mode="train", stride=32)
    base_val_ds = build_yolo_dataset(cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=32)

    cache_p = Path(args.cache_root)
    train_aligned_ds = OfficialAlignedCrossAttentionDataset(
        yolo_dataset=base_train_ds,
        cache_dir=cache_p / "train",
        seq_len=args.seq_len,
        stride=args.stride,
    )
    val_aligned_ds = OfficialAlignedCrossAttentionDataset(
        yolo_dataset=base_val_ds,
        cache_dir=cache_p / "val",
        seq_len=args.seq_len,
        stride=args.stride,
    )

    total_batch = args.batch * max(1, len(gpu_ids))
    train_loader = DataLoader(
        train_aligned_ds,
        batch_size=total_batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_aligned_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_aligned_ds,
        batch_size=total_batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_aligned_batch,
        pin_memory=True,
    )

    print(f"[INFO] Train samples: {len(train_aligned_ds)} (Target: 43,008)")
    print(f"[INFO] Val samples  : {len(val_aligned_ds)} (Target: 31,613)")
    print(f"[INFO] Effective Batch Size: {total_batch}")

    # 3. Load Base Model and strictly freeze it
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

    # 4. Build RK3588-Friendly Local Cross-Window Attention Highway
    cross_attn_highway = LocalCrossWindowAttentionHighway(
        in_diff_channels=1,
        feat_channels=48,
        num_hist_frames=args.seq_len,
        mid_channels=16,
    )
    trainable_params = sum(p.numel() for p in cross_attn_highway.parameters() if p.requires_grad)

    print(colorstr("green", f"[INFO] Base Detector: {frozen_params:,} parameters (100% LOCKED & FROZEN)."))
    print(colorstr("cyan", f"[INFO] RK3588 Local Cross-Attention Highway: {trainable_params:,} parameters (TRAINABLE)."))
    print(f"[INFO] Initial Gate alpha: {cross_attn_highway.gate.item():.6f} (Strict 0.0 Zero Regression Guarantee)")

    model = LocalCrossAttentionIntegratedModel(base_detector, cross_attn_highway).to(device)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    # 5. Optimizer with Dedicated Learning Rates
    conv_params = [p for n, p in cross_attn_highway.named_parameters() if p.requires_grad and "gate" not in n]
    gate_params = [p for n, p in cross_attn_highway.named_parameters() if p.requires_grad and "gate" in n]
    all_trainable = conv_params + gate_params

    optimizer = torch.optim.AdamW([
        {"params": conv_params, "lr": args.lr0, "weight_decay": args.weight_decay},
        {"params": gate_params, "lr": 1e-5, "weight_decay": 0.0},
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

    # 5.5 ZERO-CHECK BASELINE EVALUATION (Epoch 0 Verification)
    print("\n" + "=" * 90)
    print("🔬 Running Zero-Regression Baseline Check (Epoch 0, alpha=0.0)...")
    torch.cuda.empty_cache()
    model.eval()
    val_preds_list = []
    val_gt_list = []
    val_sizes_list = []
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

    # Clean cache before training starts
    torch.cuda.empty_cache()

    # 6. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        model_module = model.module if hasattr(model, "module") else model
        base_detector_mod = model_module.base_detector
        cross_attn_highway_mod = model_module.cross_attn_highway

        # STRICT PROTOCOL: Base detector ALWAYS in eval mode to freeze BatchNorm running stats
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
            torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += loss_items["loss_total"]
            train_hm_accum += loss_items["loss_hm"]

            effective_alpha = (torch.tanh(cross_attn_highway_mod.gate) * cross_attn_highway_mod.scale_factor).item()
            if batch_i % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{loss_items['loss_total']:.4f}",
                    "hm": f"{loss_items['loss_hm']:.4f}",
                    "alpha": f"{effective_alpha:.6f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                })

        scheduler.step()
        train_dur = time.time() - t0

        # 7. Validation Loop
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
        effective_alpha = (torch.tanh(cross_attn_highway_mod.gate) * cross_attn_highway_mod.scale_factor).item()

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
            "highway_state": cross_attn_highway_mod.state_dict(),
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

    print("\n" + "=" * 110)
    print(f"🎉 Aligned Training Complete! Peak Single-Frame F1: {best_f1:.4f}")
    print(f"Weights saved to: {weights_dir.resolve()}")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    main()
