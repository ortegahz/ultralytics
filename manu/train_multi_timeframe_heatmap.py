#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Multi-Timeframe Heatmap Pure-Ablation Fine-tuning Script.

Strict Single-Variable Isolation:
- No Soft-IoU loss (Zero Soft-IoU interference).
- Uses PureMultiTimeframeLoss (Focal beta=2.4 on [H_{t-1}, H_t, H_{t+1}]).
- Trains on newly aligned /mnt/data/siping/datasets/manu/uav_gmc_mth dataset.
- Evaluates Channel 0 at Distance <= 8.0px against Trial 22 Single-Frame SOTA baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, colorstr

from manu.heatmap_evaluate import extract_peaks, find_best_f1_threshold
from manu.multi_timeframe_model import YOLO26MultiTimeframeDetector
from manu.multi_timeframe_loss import generate_mth_heatmaps_and_targets, PureMultiTimeframeLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Multi-Timeframe Heatmap Model")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/optuna_median_search/trial_0022/weights/best.pt",
        help="Initial model weights (Trial 22)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_mth/data.yaml",
        help="Multi-Timeframe data.yaml path",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Fine-tune epochs (default: 3)")
    parser.add_argument("--batch", type=int, default=32, help="Training batch size")
    parser.add_argument("--val-batch", type=int, default=32, help="Validation batch size")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--lr0", type=float, default=0.000144, help="Initial learning rate (strictly Trial 22: 0.000144)")
    parser.add_argument("--lrf", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=3.5e-5)
    parser.add_argument("--focal-beta", type=float, default=2.4)
    parser.add_argument("--temporal-weight", type=float, default=0.35)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    parser.add_argument("--project", type=str, default="runs/multi_timeframe_train")
    parser.add_argument("--name", type=str, default="exp_mth_pure")
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.project) / args.name
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("   UAV Detection: Multi-Timeframe Heatmap Pure Supervision Training")
    print(f"   Initial Checkpoint : {args.weights}")
    print(f"   MTH Dataset        : {args.data}")
    print(f"   Output Directory   : {save_dir}")
    print(f"   Epochs / Batch     : {args.epochs} / {args.batch}")
    print(f"   Focal beta         : {args.focal_beta:.2f} (Softened, beta=2.4)")
    print(f"   Temporal weight    : {args.temporal_weight:.2f}")
    print("   Ablation Control   : Soft-IoU DISABLED (Single-variable test)")
    print("=" * 85)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    # 1. Model
    model = YOLO26MultiTimeframeDetector(stride=args.stride, num_timeframes=3, temporal_mode="standard")
    model.load_from_single_frame(args.weights)
    model.to(device)

    # 2. Loss
    criterion = PureMultiTimeframeLoss(
        hm_weight=1.0,
        offset_weight=0.45,
        temporal_weight=args.temporal_weight,
        focal_alpha=2.0,
        focal_beta=args.focal_beta,
    )

    # 3. Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr0, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr0 * args.lrf)

    # 4. DataLoaders
    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data
    cfg.scale = 0.05
    cfg.translate = 0.04
    cfg.mosaic = 0.0

    train_dataset = build_yolo_dataset(cfg, data_dict["train"], batch=args.batch, data=data_dict, mode="train", stride=32)
    train_loader = build_dataloader(train_dataset, batch=args.batch, workers=4, shuffle=True)

    val_dataset = build_yolo_dataset(cfg, data_dict["val"], batch=args.val_batch, data=data_dict, mode="val", stride=32)
    val_loader = build_dataloader(val_dataset, batch=args.val_batch, workers=4, shuffle=False)

    feat_h = args.imgsz // args.stride
    feat_w = args.imgsz // args.stride

    best_f1 = 0.0
    results_csv = save_dir / "results.csv"
    with open(results_csv, "w", encoding="utf-8") as f:
        f.write("epoch,loss,loss_hm,loss_temporal,best_th,recall,precision,f1,tp,fp,gt,lr\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_acc = 0.0
        hm_acc = 0.0
        temp_acc = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch:02d}/{args.epochs:02d} [Train]")
        for batch in pbar:
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            bboxes = batch["bboxes"].to(device)
            b_idx = batch["batch_idx"].to(device)
            im_files = batch.get("im_file", [])
            bs = imgs.shape[0]

            targets = generate_mth_heatmaps_and_targets(
                batch_bboxes=bboxes,
                batch_idx=b_idx,
                batch_size=bs,
                feat_shape=(feat_h, feat_w),
                stride=args.stride,
                im_files=im_files,
                min_radius=1,
                device=device,
            )

            optimizer.zero_grad()
            preds = model(imgs)
            loss_dict = criterion(preds, targets)

            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            loss_acc += loss.item()
            hm_acc += loss_dict["loss_hm"].item()
            temp_acc += loss_dict["loss_temporal"].item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "hm": f"{loss_dict['loss_hm'].item():.4f}",
                "tmp": f"{loss_dict['loss_temporal'].item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        scheduler.step()
        train_dur = time.time() - t0

        # ----------------- Validation Phase -----------------
        model.eval()
        val_preds_list = []
        val_gt_list = []
        val_sizes_list = []
        t_val = time.time()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch:02d}/{args.epochs:02d} [Val]"):
                imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
                bboxes = batch["bboxes"]
                b_idx = batch["batch_idx"]
                bs = imgs.shape[0]

                preds = model(imgs)
                # Channel 0: current frame
                peaks = extract_peaks(
                    heatmap=preds["heatmap"][:, 0:1],
                    offset=preds["offset"],
                    stride=args.stride,
                    conf_thresh=0.05,
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
            thresholds=[0.15, 0.20, 0.22, 0.25, 0.30, 0.35],
        )

        f1 = metrics["f1"]
        rec = metrics["recall"]
        prec = metrics["precision"]
        best_th = metrics.get("best_th", 0.25)
        val_dur = time.time() - t_val

        avg_loss = loss_acc / max(1, num_batches)
        avg_hm = hm_acc / max(1, num_batches)
        avg_tmp = temp_acc / max(1, num_batches)

        print(
            f"\n[Ep {epoch:02d}/{args.epochs:02d}] "
            f"Loss: {avg_loss:.4f} (HM: {avg_hm:.4f}, Temp: {avg_tmp:.4f}) | "
            f"Best F1: {f1:.4f} @ th={best_th:.2f} | Recall: {rec:.4f} | Precision: {prec:.4f} | "
            f"TP: {metrics['tp']} FP: {metrics['fp']} | Train: {train_dur:.1f}s Val: {val_dur:.1f}s\n"
        )

        with open(results_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{avg_loss:.6f},{avg_hm:.6f},{avg_tmp:.6f},"
                f"{best_th:.4f},{rec:.6f},{prec:.6f},{f1:.6f},"
                f"{metrics['tp']},{metrics['fp']},{metrics['total_gt']},"
                f"{optimizer.param_groups[0]['lr']:.8f}\n"
            )

        ckpt_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "metrics": metrics,
            "stride": args.stride,
            "imgsz": args.imgsz,
        }
        torch.save(ckpt_data, weights_dir / "last.pt")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(ckpt_data, weights_dir / "best.pt")
            print(colorstr("bold", colorstr("green", f"★ New Best Multi-Timeframe Checkpoint Saved: F1={best_f1:.4f}")))

    print(colorstr("bold", colorstr("green", f"\n[COMPLETE] Multi-Timeframe Training Finished. Best F1: {best_f1:.4f}\n")))


if __name__ == "__main__":
    main()
