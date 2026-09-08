#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-GPU Parallel Optuna Hyperparameter Search for 3-Frame Raw Temporal Heatmap Detector.

Tailored for:
1. 10 Epochs Fine-Tuning on [I_t, I_{t-4}, I_{t-12}] dataset from current best checkpoint.
2. 4-card parallel scheduling (each trial on a single GPU).
3. Highly optimized search space designed specifically for temporal fine-tuning:
   - lr0: Conservative fine-tuning range [5e-5, 6e-4] (prevents catastrophic forgetting of backbone).
   - weight_decay: [1e-5, 3e-4].
   - offset_weight: [0.25, 0.70] (balances CenterNet heatmap vs sub-pixel coordinate alignment).
   - focal_beta: [2.5, 4.2] (controls suppression strength on background clutter).
   - Spatial augmentations: scale [0.05, 0.20], mosaic [0.0, 0.15], translate [0.05, 0.15].

Usage:
    python manu/optuna_parallel_3frame_heatmap.py \
        --data /mnt/data/siping/datasets/manu/uav_temporal_3frame/data.yaml \
        --weights runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt \
        --gpus 0,1,2,3 \
        --n-trials 60 \
        --epochs 10 \
        --batch 32 \
        --stride 2 \
        --output-root runs/optuna_3frame_temporal_10ep
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

import optuna
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================
# 搜索空间定义 (严格契合 10 Epoch 微调物理特性)
# =========================
def suggest_params(trial: optuna.Trial) -> dict:
    """采样面向 3 帧时序输入微调的科学参数区间。"""
    return {
        # 1. 学习率区间: 既要让第一层 b0 适应 3 帧新输入，又不能把已收敛的骨干冲垮
        # 经验甜点区在 8e-5 ~ 4e-4 之间
        "lr0": trial.suggest_float("lr0", 5e-5, 6e-4, log=True),
        # 2. 权重衰减: 防止微调过拟合
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 3e-4, log=True),
        # 3. 偏移量 Loss 权重: 平衡点目标召回与定位精度
        "offset_weight": trial.suggest_float("offset_weight", 0.25, 0.70),
        # 4. Focal Loss 负样本惩罚指数 beta:
        # beta 越小，对难例负样本关注度越高；beta 越大，训练越稳定。
        "focal_beta": trial.suggest_float("focal_beta", 2.5, 4.2),
        # 5. 极弱小目标保护型轻量空间增广
        "scale": trial.suggest_float("scale", 0.05, 0.20),
        "mosaic": trial.suggest_float("mosaic", 0.0, 0.15),
        "translate": trial.suggest_float("translate", 0.05, 0.15),
    }


def read_best_metrics(results_csv: Path) -> tuple[float, dict]:
    """读取单 trial 的 results.csv 中最优 F1-Score 及其各项指标。"""
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv}")

    with results_csv.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError(f"results.csv is empty: {results_csv}")

    valid_rows = []
    for row in rows:
        f1_str = row.get("metrics/f1(B)")
        if f1_str is not None and f1_str != "":
            try:
                row["_f1"] = float(f1_str)
                valid_rows.append(row)
            except ValueError:
                continue

    if not valid_rows:
        raise RuntimeError(f"No valid F1 metrics found in {results_csv}")

    best_row = max(valid_rows, key=lambda r: r["_f1"])
    metrics = {
        "epoch": int(best_row.get("epoch", 0)),
        "f1": float(best_row["_f1"]),
        "recall": float(best_row.get("metrics/recall(B)", 0.0)),
        "precision": float(best_row.get("metrics/precision(B)", 0.0)),
        "best_th": float(best_row.get("metrics/best_th", 0.20)),
        "tp": int(best_row.get("metrics/tp", 0)),
        "fp": int(best_row.get("metrics/fp", 0)),
        "gt": int(best_row.get("metrics/gt", 0)),
    }
    return metrics["f1"], metrics


# =========================
# Worker 单卡训练子进程逻辑
# =========================
def run_worker():
    from torch.cuda.amp import GradScaler, autocast
    from tqdm import tqdm
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG

    from manu.heatmap_evaluate import extract_peaks, find_best_f1_threshold
    from manu.heatmap_loss import HeatmapLoss, generate_heatmaps_and_targets
    from manu.heatmap_model import YOLO26HeatmapDetector

    parser = argparse.ArgumentParser(description="Worker process for 3-frame temporal heatmap training")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--trial-number", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--dist-thresh", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=4)

    # 超参数参数输入
    parser.add_argument("--lr0", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--offset-weight", type=float, required=True)
    parser.add_argument("--focal-beta", type=float, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--mosaic", type=float, required=True)
    parser.add_argument("--translate", type=float, required=True)
    args = parser.parse_args()

    trial_name = f"trial_{args.trial_number:04d}"
    save_dir = Path(args.output_root) / trial_name
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(args.gpu_id)

    print(f"\n[WORKER] {trial_name} started on GPU {args.gpu_id}")
    print(
        f"Hyperparameters: lr0={args.lr0:.6f}, wd={args.weight_decay:.6f}, "
        f"off_w={args.offset_weight:.3f}, beta={args.focal_beta:.2f}, "
        f"scale={args.scale:.3f}, mosaic={args.mosaic:.3f}"
    )

    # 1. 数据配置
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
    cfg.translate = args.translate
    cfg.scale = args.scale
    cfg.mosaic = args.mosaic
    cfg.fliplr = 0.5
    cfg.flipud = 0.0
    cfg.mixup = 0.0
    cfg.copy_paste = 0.0

    train_dataset = build_yolo_dataset(cfg, data_dict["train"], batch=args.batch, data=data_dict, mode="train", stride=32)
    val_dataset = build_yolo_dataset(cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=32)
    train_loader = build_dataloader(train_dataset, batch=args.batch, workers=args.workers, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch=args.batch, workers=args.workers, shuffle=False)

    # 2. 构建模型并 100% 匹配加载已有预训练权重（保持网络完全不变）
    model = YOLO26HeatmapDetector(stride=args.stride, num_classes=1, use_temporal_stem=False)
    weights_path = Path(args.weights)
    if weights_path.exists():
        ckpt = torch.load(weights_path, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
        if hasattr(state_dict, "state_dict"):
            state_dict = state_dict.state_dict()
        model.load_state_dict(state_dict, strict=True)
        print(f"[WORKER] Loaded pretrained checkpoint with 100% strict match: {weights_path.name}")
    else:
        print(f"[WARN] Checkpoint not found: {weights_path}, training from scratch!")

    model.to(device)

    # 3. 优化器、余弦衰减与损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr0, weight_decay=args.weight_decay)
    lf = lambda epoch: ((1 + math.cos(epoch * math.pi / args.epochs)) / 2) * (1 - 0.01) + 0.01
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scaler = GradScaler(enabled=True)

    criterion = HeatmapLoss(hm_weight=1.0, offset_weight=args.offset_weight)
    criterion.focal_loss.beta = args.focal_beta

    feat_h = args.imgsz // args.stride
    feat_w = args.imgsz // args.stride

    csv_path = save_dir / "results.csv"
    csv_header = "epoch,train/loss,train/loss_hm,train/loss_off,metrics/best_th,metrics/recall(B),metrics/precision(B),metrics/f1(B),metrics/tp,metrics/fp,metrics/gt,lr\n"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_header)

    best_f1 = 0.0

    # 4. 10 轮微调训练
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_hm_accum = 0.0
        train_off_accum = 0.0
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"[{trial_name}] Ep {epoch:02d}/{args.epochs:02d}", total=num_batches, dynamic_ncols=True, file=sys.stdout)
        for batch in pbar:
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
            with autocast(enabled=True):
                preds = model(imgs)
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
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
            })

        scheduler.step()

        # 5. 验证评估 (动态搜索最优 F1 门限)
        model.eval()
        val_preds_list = []
        val_gt_list = []
        val_sizes_list = []

        val_pbar = tqdm(val_loader, desc=f"[{trial_name}] Val {epoch:02d}", total=len(val_loader), dynamic_ncols=True, file=sys.stdout)
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
                    conf_thresh=0.10,
                    top_k=80,
                )
                val_preds_list.extend(peaks)

                for b in range(bs):
                    mask_b = b_idx == b
                    val_gt_list.append(bboxes[mask_b].cpu().numpy())
                    val_sizes_list.append((args.imgsz, args.imgsz))

        metrics = find_best_f1_threshold(
            predictions_raw=val_preds_list,
            gt_boxes_list=val_gt_list,
            img_sizes=val_sizes_list,
            distance_threshold=args.dist_thresh,
            thresholds=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        )

        f1 = metrics["f1"]
        rec = metrics["recall"]
        prec = metrics["precision"]
        best_th = metrics.get("best_th", 0.20)

        avg_loss = train_loss_accum / max(num_batches, 1)
        avg_hm = train_hm_accum / max(num_batches, 1)
        avg_off = train_off_accum / max(num_batches, 1)

        print(
            f"[{trial_name}] Ep {epoch:02d}/{args.epochs:02d} | "
            f"Loss: {avg_loss:.4f} | F1: {f1:.4f} @ th={best_th:.2f} | Rec: {rec:.4f} | Prec: {prec:.4f}"
        )

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{avg_loss:.6f},{avg_hm:.6f},{avg_off:.6f},"
                f"{best_th:.4f},{rec:.6f},{prec:.6f},{f1:.6f},"
                f"{metrics['tp']},{metrics['fp']},{metrics['total_gt']},"
                f"{optimizer.param_groups[0]['lr']:.8f}\n"
            )

        if f1 > best_f1:
            best_f1 = f1
            ckpt_data = {
                "epoch": epoch,
                "model": model.state_dict(),
                "metrics": metrics,
                "stride": args.stride,
                "imgsz": args.imgsz,
            }
            torch.save(ckpt_data, weights_dir / "best.pt")

    print(f"[WORKER] {trial_name} complete. Best F1: {best_f1:.4f}")


# =========================
# 主控调度逻辑
# =========================
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
        str(Path(__file__).resolve()),
        "--worker",
        "--trial-number",
        str(trial_number),
        "--gpu-id",
        str(gpu_id),
        "--output-root",
        str(output_root),
        "--data",
        str(args.data),
        "--weights",
        str(args.weights),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--epochs",
        str(args.epochs),
        "--stride",
        str(args.stride),
        "--dist-thresh",
        str(args.dist_thresh),
        "--workers",
        str(args.workers),
        "--lr0",
        str(params["lr0"]),
        "--weight-decay",
        str(params["weight_decay"]),
        "--offset-weight",
        str(params["offset_weight"]),
        "--focal-beta",
        str(params["focal_beta"]),
        "--scale",
        str(params["scale"]),
        "--mosaic",
        str(params["mosaic"]),
        "--translate",
        str(params["translate"]),
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
        "lr0,weight_decay,offset_weight,focal_beta,scale,mosaic,translate\n"
    )
    if not summary_csv.exists():
        with open(summary_csv, "w", encoding="utf-8") as f:
            f.write(header)

    params = trial_info["params"]
    dur = (time.time() - trial_info["start_time"]) / 60.0
    row = (
        f"{trial_info['trial_name']},{status},{f1:.6f},"
        f"{metrics.get('recall', 0.0):.6f},{metrics.get('precision', 0.0):.6f},"
        f"{metrics.get('best_th', 0.20):.4f},{metrics.get('epoch', 0)},"
        f"{metrics.get('tp', 0)},{metrics.get('fp', 0)},{metrics.get('gt', 0)},"
        f"{dur:.2f},"
        f"{params['lr0']:.8f},{params['weight_decay']:.8f},"
        f"{params['offset_weight']:.4f},{params['focal_beta']:.4f},"
        f"{params['scale']:.4f},{params['mosaic']:.4f},{params['translate']:.4f}\n"
    )
    with open(summary_csv, "a", encoding="utf-8") as f:
        f.write(row)


def main():
    if "--worker" in sys.argv:
        run_worker()
        return

    parser = argparse.ArgumentParser(description="Multi-GPU Parallel Optuna for 3-Frame Temporal Heatmap")
    parser.add_argument("--data", type=str, default="/mnt/data/siping/datasets/manu/uav_temporal_3frame/data.yaml")
    parser.add_argument("--weights", type=str, default="runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="GPUs to use, e.g. '0,1,2,3'")
    parser.add_argument("--n-trials", type=int, default=60, help="Total number of trials to run")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per trial")
    parser.add_argument("--batch", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--dist-thresh", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-root", type=str, default="runs/optuna_3frame_temporal_10ep")
    parser.add_argument("--study-name", type=str, default="uav_3frame_temporal_tuning")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv = output_root / "optuna_summary.csv"

    gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip().isdigit()]
    if not gpu_list:
        raise ValueError(f"No valid GPU IDs parsed from: {args.gpus}")

    print("=" * 80)
    print(f"Starting Multi-GPU Optuna Tuning on 3-Frame Raw Temporal Heatmap Detector")
    print(f"Dataset: {args.data}")
    print(f"Pretrained Weights: {args.weights}")
    print(f"GPUs available ({len(gpu_list)}): {gpu_list}")
    print(f"Total Trials: {args.n_trials} | Epochs per Trial: {args.epochs}")
    print(f"Output Root: {output_root.resolve()}")
    print("=" * 80)

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
    completed_trials = len(study.trials)

    pbar = tqdm(total=args.n_trials, initial=completed_trials, desc="Overall Progress")

    while completed_trials < args.n_trials or running_trials:
        # 1. 派发新任务
        while available_gpus and (completed_trials + len(running_trials) < args.n_trials):
            gpu_id = available_gpus.pop(0)
            trial = study.ask()
            params = suggest_params(trial)

            trial_info = launch_trial(trial, params, output_root, gpu_id, args)
            running_trials.append(trial_info)
            print(f"\n[LAUNCH] {trial_info['trial_name']} dispatched to GPU {gpu_id}")

        # 2. 轮询监控运行中的进程
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
                        study.tell(trial, f1)
                        write_summary_row(summary_csv, info, f1, metrics, status="SUCCESS")
                        print(f"\n[SUCCESS] {trial_name} on GPU {info['gpu_id']} finished! F1: {f1:.4f} (Rec: {metrics['recall']:.4f}, Prec: {metrics['precision']:.4f})")
                    except Exception as e:
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                        write_summary_row(summary_csv, info, 0.0, {}, status="PARSE_FAIL")
                        print(f"\n[FAIL] {trial_name} result parse failed: {e}")
                else:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    write_summary_row(summary_csv, info, 0.0, {}, status="CRASHED")
                    print(f"\n[CRASH] {trial_name} exited with return code {ret}. See {info['log_file']}")

                completed_trials += 1
                pbar.update(1)

        running_trials = still_running
        time.sleep(5)

    pbar.close()
    print("\n" + "=" * 80)
    print(f"All {args.n_trials} trials completed!")
    print(f"Best Trial: #{study.best_trial.number} with F1: {study.best_value:.4f}")
    print(f"Best Hyperparameters: {study.best_params}")
    print(f"Summary saved to: {summary_csv.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
