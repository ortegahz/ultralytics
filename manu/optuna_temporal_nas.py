#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
High-Throughput 4-GPU Distributed Spatio-Temporal NAS for Tiny Infrared UAV Detection.

Core Engineering Guarantees:
1. Base Detector 100% physically frozen: Trial 0474 SOTA checkpoint (F1 = 0.9064).
2. Pure Hardware RK3588-Friendly Operations.
3. Official Dataset 1:1 Strict Alignment: exactly 43,008 train samples & 31,613 val samples.
4. Epoch 0 Zero-Regression Guarantee:
   - At Step 0, alpha starts from zero; model reproduces baseline SOTA (F1 = 0.9064).
5. Expanded NAS Search Space:
   - Micro-Architecture: kernel_mode (dual_scale_3_5, single_scale_3, single_scale_5, dilated_3_d2)
   - Capacity: mid_channels (12, 16, 24), gate_activation (sigmoid, silu, tanh)
   - Gating Form: gate_mode (tanh_bounded, positive_softplus, channel_adaptive)
   - Temporal dynamics: seq_len (2, 3, 4), stride (1, 2)
   - Hyperparameters: lr0, weight_decay, focal_beta, offset_weight
6. Robust Subprocess Execution:
   - 4 GPUs run independent worker processes asynchronously.
   - SQLite persistence (study.db) with automatic resume support.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, colorstr
from manu.heatmap_evaluate import extract_peaks, find_best_f1_threshold
from manu.heatmap_loss import HeatmapLoss, generate_heatmaps_and_targets
from manu.heatmap_model import YOLO26HeatmapDetector
from manu.local_cross_window_module import LocalCrossWindowAttentionHighway
from manu.official_aligned_dataset import OfficialAlignedCrossAttentionDataset, collate_aligned_batch


# ==============================================================================
# 1. Single Worker Subprocess (Runs independently on assigned GPU)
# ==============================================================================

def run_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--trial-number", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--cache-root", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=4)

    # Micro-architecture & Gating Space
    parser.add_argument("--kernel-mode", type=str, required=True)
    parser.add_argument("--mid-channels", type=int, required=True)
    parser.add_argument("--gate-activation", type=str, required=True)
    parser.add_argument("--gate-mode", type=str, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)

    # Optimization Hyperparameters
    parser.add_argument("--lr0", type=float, required=True)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--focal-beta", type=float, required=True)
    parser.add_argument("--offset-weight", type=float, required=True)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    args = parser.parse_args()

    trial_name = f"trial_{args.trial_number:04d}"
    save_dir = Path(args.output_root) / trial_name
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(args.gpu_id)

    print(f"\n" + "=" * 90, flush=True)
    print(f"[WORKER] {trial_name} started on GPU {args.gpu_id} (PID: {os.getpid()})", flush=True)
    print(
        f"Config: kernel={args.kernel_mode}, mid_ch={args.mid_channels}, gate_act={args.gate_activation}, "
        f"gate_mode={args.gate_mode}, seq_len={args.seq_len}, stride={args.stride}",
        flush=True,
    )
    print(
        f"Hyperparameters: lr0={args.lr0:.6f}, wd={args.weight_decay:.6f}, beta={args.focal_beta:.2f}, off_w={args.offset_weight:.3f}",
        flush=True,
    )
    print("=" * 90 + "\n", flush=True)

    # 1. Dataset setup using official YOLO dataset + aligned temporal cache
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
    cfg.translate = 0.0
    cfg.scale = 0.0
    cfg.fliplr = 0.0
    cfg.flipud = 0.0
    cfg.mosaic = 0.0
    cfg.mixup = 0.0
    cfg.copy_paste = 0.0

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

    train_loader = DataLoader(
        train_aligned_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_aligned_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_aligned_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_aligned_batch,
        pin_memory=True,
    )

    # 2. Base Model Loading & Freezing
    weights_path = Path(args.weights)
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()

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
        stride=2,
        num_classes=1,
        temporal_mode="standard",
        use_p0_highway=True,
        p0_highway_kwargs=p0_kwargs,
    )

    own_state = base_detector.state_dict()
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "").replace("model.model.", "").replace("model.", "")
        if clean_k in own_state and own_state[clean_k].shape == v.shape:
            own_state[clean_k].copy_(v)

    # STRICT 100% FREEZING
    for param in base_detector.parameters():
        param.requires_grad = False

    # 3. Build Configurable Cross-Attention Highway
    cross_attn_highway = LocalCrossWindowAttentionHighway(
        in_diff_channels=1,
        feat_channels=48,
        num_hist_frames=args.seq_len,
        mid_channels=args.mid_channels,
        kernel_mode=args.kernel_mode,
        gate_activation=args.gate_activation,
        gate_mode=args.gate_mode,
    )

    class IntegratedWrapper(nn.Module):
        def __init__(self, base, highway):
            super().__init__()
            self.base = base
            self.highway = highway

        def forward(self, curr_img, diff_seq):
            f_base = self.base.extract_features(curr_img)
            if self.base.p0_highway is not None:
                f_base = f_base + self.base.p0_highway(curr_img)
            delta = self.highway(f_base, diff_seq)
            return self.base.head(f_base + delta)

    model = IntegratedWrapper(base_detector, cross_attn_highway).to(device)

    # 4. Optimizer & Loss
    conv_params = [p for n, p in cross_attn_highway.named_parameters() if p.requires_grad and "gate" not in n]
    gate_params = [p for n, p in cross_attn_highway.named_parameters() if p.requires_grad and "gate" in n]

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
        pos_weight=args.pos_weight,
    )

    feat_h, feat_w = args.imgsz // 2, args.imgsz // 2
    csv_path = save_dir / "results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,gate_val,loss_total,loss_hm,best_th,recall,precision,f1,tp,fp,gt,lr\n")

    best_f1 = 0.0

    # 5. Fast Training Loop (3 Epochs)
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.base.eval()  # Keep base strictly in eval mode

        train_loss_accum = 0.0
        train_hm_accum = 0.0
        t0 = time.time()

        for batch in train_loader:
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
                stride=2,
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
            torch.nn.utils.clip_grad_norm_(cross_attn_highway.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += loss_items["loss_total"]
            train_hm_accum += loss_items["loss_hm"]

        scheduler.step()
        train_dur = time.time() - t0

        # Fast Validation Loop
        torch.cuda.empty_cache()
        model.eval()
        val_preds_list = []
        val_gt_list = []
        val_sizes_list = []
        t_val = time.time()

        with torch.no_grad():
            for batch in val_loader:
                curr_img = batch["curr_img"].to(device, non_blocking=True)
                diff_seq = batch["diff_seq"].to(device, non_blocking=True)
                bboxes = batch["bboxes"]
                b_idx = batch["batch_idx"]
                bs = curr_img.shape[0]

                preds = model(curr_img, diff_seq)
                peaks = extract_peaks(
                    heatmap=preds["heatmap"],
                    offset=preds["offset"],
                    stride=2,
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
            thresholds=[0.20, 0.22, 0.25, 0.28, 0.30],
        )

        f1 = metrics["f1"]
        rec = metrics["recall"]
        prec = metrics["precision"]
        best_th = metrics.get("best_th", 0.25)
        val_dur = time.time() - t_val

        avg_loss = train_loss_accum / max(len(train_loader), 1)
        avg_hm = train_hm_accum / max(len(train_loader), 1)
        eff_alpha = cross_attn_highway.get_effective_alpha()
        alpha_val = eff_alpha.mean().item() if isinstance(eff_alpha, torch.Tensor) else float(eff_alpha)

        print(
            f"[{trial_name}] Ep {epoch:02d}/{args.epochs:02d} | α: {alpha_val:.6f} | "
            f"Loss: {avg_loss:.4f} | F1: {f1:.4f} @ th={best_th:.2f} | Rec: {rec * 100:.2f}% | Prec: {prec * 100:.2f}% | "
            f"TP: {metrics['tp']} FP: {metrics['fp']} | Train: {train_dur:.1f}s Val: {val_dur:.1f}s",
            flush=True,
        )

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{alpha_val:.8f},{avg_loss:.6f},{avg_hm:.6f},{best_th:.4f},"
                f"{rec:.6f},{prec:.6f},{f1:.6f},{metrics['tp']},{metrics['fp']},{metrics['total_gt']},"
                f"{optimizer.param_groups[0]['lr']:.8f}\n"
            )

        if f1 > best_f1:
            best_f1 = f1
            ckpt_data = {
                "epoch": epoch,
                "highway_state": cross_attn_highway.state_dict(),
                "metrics": metrics,
                "config": {
                    "kernel_mode": args.kernel_mode,
                    "mid_channels": args.mid_channels,
                    "gate_activation": args.gate_activation,
                    "gate_mode": args.gate_mode,
                    "seq_len": args.seq_len,
                    "stride": args.stride,
                },
            }
            torch.save(ckpt_data, weights_dir / "best.pt")

        torch.cuda.empty_cache()

    print(f"\n[WORKER] {trial_name} Complete. Best F1: {best_f1:.4f}\n", flush=True)


# ==============================================================================
# 2. Main Distributed Dispatcher Logic
# ==============================================================================

def suggest_params(trial: optuna.Trial) -> dict:
    return {
        # Micro-architecture & Gating space
        "kernel_mode": trial.suggest_categorical("kernel_mode", ["dual_scale_3_5", "single_scale_3", "single_scale_5", "dilated_3_d2"]),
        "mid_channels": trial.suggest_categorical("mid_channels", [12, 16, 24]),
        "gate_activation": trial.suggest_categorical("gate_activation", ["sigmoid", "silu", "tanh"]),
        "gate_mode": trial.suggest_categorical("gate_mode", ["positive_softplus", "tanh_bounded", "channel_adaptive"]),
        "seq_len": trial.suggest_categorical("seq_len", [2, 3, 4]),
        "stride": trial.suggest_categorical("stride", [1, 2]),

        # Continuous Hyperparameters
        "lr0": trial.suggest_float("lr0", 8e-5, 6e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True),
        "focal_beta": trial.suggest_float("focal_beta", 2.20, 2.60, step=0.05),
        "offset_weight": trial.suggest_float("offset_weight", 0.35, 0.55, step=0.05),
        "pos_weight": trial.suggest_float("pos_weight", 1.0, 2.5, step=0.25),
    }


def read_best_metrics(results_csv: Path) -> tuple[float, dict]:
    with open(results_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    if not reader:
        return 0.0, {}

    best_row = max(reader, key=lambda r: float(r.get("f1", 0.0)))
    f1 = float(best_row["f1"])
    rec = float(best_row["recall"])
    prec = float(best_row["precision"])
    th = float(best_row["best_th"])
    ep = int(best_row["epoch"])
    tp = int(best_row["tp"])
    fp = int(best_row["fp"])
    gt = int(best_row["gt"])

    return f1, {
        "f1": f1,
        "recall": rec,
        "precision": prec,
        "best_th": th,
        "epoch": ep,
        "tp": tp,
        "fp": fp,
        "gt": gt,
    }


def launch_trial(
    trial: optuna.Trial,
    params: dict,
    output_root: Path,
    gpu_id: int,
    args: argparse.Namespace,
) -> dict:
    trial_number = trial.number
    trial_name = f"trial_{trial_number:04d}"

    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_file = log_root / f"{trial_name}.log"

    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--trial-number", str(trial_number),
        "--gpu-id", str(gpu_id),
        "--output-root", str(output_root),
        "--data", str(args.data),
        "--cache-root", str(args.cache_root),
        "--weights", str(args.weights),
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--epochs", str(args.epochs),
        "--dist-thresh", str(args.dist_thresh),
        "--workers", str(args.workers),
        "--kernel-mode", str(params["kernel_mode"]),
        "--mid-channels", str(params["mid_channels"]),
        "--gate-activation", str(params["gate_activation"]),
        "--gate-mode", str(params["gate_mode"]),
        "--seq-len", str(params["seq_len"]),
        "--stride", str(params["stride"]),
        "--lr0", str(params["lr0"]),
        "--weight-decay", str(params["weight_decay"]),
        "--focal-beta", str(params["focal_beta"]),
        "--offset-weight", str(params["offset_weight"]),
        "--pos-weight", str(params["pos_weight"]),
    ]

    log_handle = open(log_file, "w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )

    return {
        "trial": trial,
        "trial_number": trial_number,
        "trial_name": trial_name,
        "gpu_id": gpu_id,
        "params": params,
        "process": process,
        "log_file": log_file,
        "log_handle": log_handle,
        "start_time": time.time(),
    }


def write_summary_row(summary_csv: Path, trial_info: dict, f1: float, metrics: dict, status: str):
    header = (
        "trial,status,f1,recall,precision,best_th,epoch,tp,fp,gt,duration_min,"
        "kernel_mode,mid_channels,gate_activation,gate_mode,seq_len,stride,"
        "lr0,weight_decay,focal_beta,offset_weight,pos_weight\n"
    )
    if not summary_csv.exists():
        with open(summary_csv, "w", encoding="utf-8") as f:
            f.write(header)

    params = trial_info["params"]
    dur = (time.time() - trial_info["start_time"]) / 60.0
    row = (
        f"{trial_info['trial_name']},{status},{f1:.6f},"
        f"{metrics.get('recall', 0.0):.6f},{metrics.get('precision', 0.0):.6f},"
        f"{metrics.get('best_th', 0.25):.4f},{metrics.get('epoch', 0)},"
        f"{metrics.get('tp', 0)},{metrics.get('fp', 0)},{metrics.get('gt', 0)},"
        f"{dur:.2f},"
        f"{params['kernel_mode']},{params['mid_channels']},{params['gate_activation']},"
        f"{params['gate_mode']},{params['seq_len']},{params['stride']},"
        f"{params['lr0']:.8f},{params['weight_decay']:.8f},"
        f"{params['focal_beta']:.4f},{params['offset_weight']:.4f},{params['pos_weight']:.2f}\n"
    )
    with open(summary_csv, "a", encoding="utf-8") as f:
        f.write(row)


def main():
    if "--worker" in sys.argv:
        run_worker()
        return

    parser = argparse.ArgumentParser(description="Multi-GPU Parallel Optuna NAS for Spatio-Temporal Highway")
    parser.add_argument("--data", type=str, default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml")
    parser.add_argument("--cache-root", type=str, default="/mnt/data/siping/datasets/manu/uav_s2_diff_cache")
    parser.add_argument("--weights", type=str, default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--n-trials", type=int, default=1000, help="Target total trials (default: 1000)")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per trial (default: 3)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size per GPU (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-root", type=str, default="runs/optuna_temporal_nas")
    parser.add_argument("--study-name", type=str, default="temporal_nas_1000")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv = output_root / "optuna_summary.csv"

    gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip().isdigit()]
    if not gpu_list:
        raise ValueError(f"No valid GPU IDs parsed from: {args.gpus}")

    print("=" * 100, flush=True)
    print(f"🚀 Starting High-Throughput Distributed Spatio-Temporal NAS Search", flush=True)
    print(f"Base Checkpoint (100% Frozen) : {args.weights} (Trial 0474 SOTA: F1=0.9064)")
    print(f"Dataset                       : {args.data} (43,008 train / 31,613 val)")
    print(f"GPUs available ({len(gpu_list)})            : {gpu_list}")
    print(f"Total Trials Target           : {args.n_trials} | Epochs per Trial: {args.epochs}")
    print(f"Output Root                   : {output_root.resolve()}")
    print(f"Logs per Trial                : {output_root / 'logs' / 'trial_XXXX.log'}")
    print("=" * 100 + "\n", flush=True)

    db_path = output_root / "study.db"
    storage_url = f"sqlite:///{db_path.resolve()}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
    )

    available_gpus = list(gpu_list)
    running_trials: list[dict] = []
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    pbar = tqdm(total=args.n_trials, initial=completed_trials, desc="NAS Search Progress", file=sys.stdout)

    while completed_trials < args.n_trials or running_trials:
        # 1. Dispatch new trials to idle GPUs
        while available_gpus and (completed_trials + len(running_trials) < args.n_trials):
            gpu_id = available_gpus.pop(0)
            trial = study.ask()
            params = suggest_params(trial)

            trial_info = launch_trial(trial, params, output_root, gpu_id, args)
            running_trials.append(trial_info)
            print(
                f"\n[LAUNCH] {trial_info['trial_name']} dispatched to GPU {gpu_id} "
                f"(log: {trial_info['log_file'].name})",
                flush=True,
            )

        # 2. Monitor running subprocesses
        still_running = []
        for info in running_trials:
            proc = info["process"]
            ret = proc.poll()
            if ret is None:
                still_running.append(info)
            else:
                info["log_handle"].close()
                available_gpus.append(info["gpu_id"])
                trial = info["trial"]
                trial_name = info["trial_name"]

                results_csv = output_root / trial_name / "results.csv"
                if ret == 0 and results_csv.exists():
                    try:
                        f1, metrics = read_best_metrics(results_csv)

                        # Precision guardrail penalty: if precision < 94.5%, penalize fitness
                        fitness = f1
                        prec_pct = metrics["precision"] * 100.0 if metrics["precision"] <= 1.0 else metrics["precision"]
                        if prec_pct < 94.5:
                            fitness = f1 - (94.5 - prec_pct) * 0.02

                        for k, v in metrics.items():
                            try:
                                trial.set_user_attr(k, v)
                            except Exception:
                                pass

                        study.tell(trial, fitness)
                        write_summary_row(summary_csv, info, f1, metrics, status="SUCCESS")
                        print(
                            f"\n" + colorstr("bold", colorstr("green", f"[SUCCESS] {trial_name} on GPU {info['gpu_id']} finished!"))
                            + f" F1: {f1:.4f} (Rec: {metrics['recall']:.4f}, Prec: {metrics['precision']:.4f}, th={metrics['best_th']:.2f})",
                            flush=True,
                        )
                    except Exception as e:
                        try:
                            study.tell(trial, state=optuna.trial.TrialState.FAIL)
                        except Exception:
                            pass
                        write_summary_row(summary_csv, info, 0.0, {}, status="PARSE_FAIL")
                        print(f"\n[FAIL] {trial_name} result parse failed: {e}", flush=True)
                else:
                    try:
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    except Exception:
                        pass
                    write_summary_row(summary_csv, info, 0.0, {}, status="CRASHED")
                    print(f"\n[CRASH] {trial_name} exited with return code {ret}. Inspect: {info['log_file']}", flush=True)

                completed_trials += 1
                pbar.update(1)

        running_trials = still_running
        time.sleep(3)

    pbar.close()
    print("\n" + "=" * 100, flush=True)
    print(f"🎉 Spatio-Temporal NAS Study Complete! Total Trials: {len(study.trials)}", flush=True)
    print(f"Best Trial: #{study.best_trial.number} with F1: {study.best_value:.4f}", flush=True)
    print(f"Best Architecture: {study.best_params}", flush=True)
    print(f"Summary saved to: {summary_csv.resolve()}", flush=True)
    print("=" * 100 + "\n", flush=True)


if __name__ == "__main__":
    main()
