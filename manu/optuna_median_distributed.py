#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
High-Throughput 4-GPU Distributed Optuna Hyperparameter Search on Aligned Median SOTA Model.

Architecture (Matches the reliable optuna_parallel_3frame_heatmap.py structure):
1. Main Process dispatches trials across GPUs [0, 1, 2, 3] as independent subprocesses.
2. Each Trial writes to its OWN separate log file:
     runs/optuna_median_search/logs/trial_XXXX.log
   Allowing you to run: tail -f runs/optuna_median_search/logs/trial_0000.log at any time!
3. Subprocess worker runs 3 epochs with explicit progress prints per epoch:
     - Epoch Train Loss & LR
     - Epoch Val Recall, Precision, and F1
     - Real-time logging with flush=True
4. Main process collects results, records summary in optuna_summary.csv and study.db.
5. Injects Trial 0 with Golden Baseline Seed (SOTA F1 = 0.8975).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import optuna
from optuna.samplers import TPESampler
import torch
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


# ==============================================================================
# 1. Worker Execution Loop (Runs in separate subprocess per GPU)
# ==============================================================================

def run_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--trial-number", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=4)

    # Hyperparameters
    parser.add_argument("--lr0", type=float, required=True)
    parser.add_argument("--lrf", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--focal-beta", type=float, required=True)
    parser.add_argument("--offset-weight", type=float, required=True)
    parser.add_argument("--min-radius", type=int, required=True)
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

    print(f"\n" + "=" * 80, flush=True)
    print(f"[WORKER] {trial_name} started on GPU {args.gpu_id} (PID: {os.getpid()})", flush=True)
    print(
        f"Hyperparameters: lr0={args.lr0:.6f}, lrf={args.lrf:.3f}, wd={args.weight_decay:.6f}, "
        f"beta={args.focal_beta:.2f}, off_w={args.offset_weight:.3f}, radius={args.min_radius}, "
        f"scale={args.scale:.3f}, mosaic={args.mosaic:.3f}, translate={args.translate:.3f}",
        flush=True,
    )
    print("=" * 80 + "\n", flush=True)

    # 1. Dataset setup
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

    # 2. Model setup (100% warm-start from current SOTA)
    model = YOLO26HeatmapDetector(stride=args.stride, num_classes=1, temporal_mode="standard")
    weights_path = Path(args.weights)
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"[WORKER] Successfully warm-started from {weights_path.name}", flush=True)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr0, weight_decay=args.weight_decay)
    lf = lambda ep: ((1 + math.cos(ep * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scaler = GradScaler(enabled=True)

    criterion = HeatmapLoss(
        hm_weight=1.0,
        offset_weight=args.offset_weight,
        focal_beta=args.focal_beta,
    )

    feat_h = args.imgsz // args.stride
    feat_w = args.imgsz // args.stride

    csv_path = save_dir / "results.csv"
    csv_header = "epoch,train/loss,train/loss_hm,train/loss_off,metrics/best_th,metrics/recall(B),metrics/precision(B),metrics/f1(B),metrics/tp,metrics/fp,metrics/gt,lr\n"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_header)

    best_f1 = 0.0

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_hm_accum = 0.0
        train_off_accum = 0.0
        num_batches = len(train_loader)
        t0 = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"[{trial_name}] Ep {epoch:02d}/{args.epochs:02d}",
            total=num_batches,
            dynamic_ncols=True,
            file=sys.stdout,
        )

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
                min_radius=args.min_radius,
                device=device,
            )

            optimizer.zero_grad()
            with autocast(enabled=True):
                preds = model(imgs)
                loss, loss_items = criterion(preds, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += loss_items["loss_total"]
            train_hm_accum += loss_items["loss_hm"]
            train_off_accum += loss_items["loss_offset"]

            pbar.set_postfix({
                "loss": f"{loss_items['loss_total']:.4f}",
                "hm": f"{loss_items['loss_hm']:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
            })

        scheduler.step()
        train_dur = time.time() - t0

        # 5. Validation Loop
        model.eval()
        val_preds_list = []
        val_gt_list = []
        val_sizes_list = []
        t_val = time.time()

        val_pbar = tqdm(
            val_loader,
            desc=f"[{trial_name}] Val {epoch:02d}",
            total=len(val_loader),
            dynamic_ncols=True,
            file=sys.stdout,
        )

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
            thresholds=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        )

        f1 = metrics["f1"]
        rec = metrics["recall"]
        prec = metrics["precision"]
        best_th = metrics.get("best_th", 0.25)
        val_dur = time.time() - t_val

        avg_loss = train_loss_accum / max(num_batches, 1)
        avg_hm = train_hm_accum / max(num_batches, 1)
        avg_off = train_off_accum / max(num_batches, 1)

        print(
            f"[{trial_name}] Ep {epoch:02d}/{args.epochs:02d} | "
            f"Loss: {avg_loss:.4f} (HM: {avg_hm:.4f}) | "
            f"F1: {f1:.4f} @ th={best_th:.2f} | Rec: {rec:.4f} | Prec: {prec:.4f} | "
            f"TP: {metrics['tp']} FP: {metrics['fp']} | Train: {train_dur:.1f}s Val: {val_dur:.1f}s",
            flush=True,
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

    print(f"\n[WORKER] {trial_name} Complete. Best F1: {best_f1:.4f}\n", flush=True)


# ==============================================================================
# 2. Main Distributed Dispatcher Logic
# ==============================================================================

def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "lr0": trial.suggest_float("lr0", 1.5e-5, 1.5e-4, log=True),
        "lrf": trial.suggest_float("lrf", 0.05, 0.5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True),
        "focal_beta": trial.suggest_float("focal_beta", 2.0, 4.0, step=0.2),
        "offset_weight": trial.suggest_float("offset_weight", 0.3, 0.6, step=0.05),
        "min_radius": trial.suggest_int("min_radius", 1, 2),
        "translate": trial.suggest_float("translate", 0.02, 0.12, step=0.02),
        "scale": trial.suggest_float("scale", 0.05, 0.20, step=0.05),
        "mosaic": trial.suggest_float("mosaic", 0.0, 0.20, step=0.05),
    }


def read_best_metrics(results_csv: Path) -> tuple[float, dict]:
    with open(results_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    if not reader:
        return 0.0, {}

    best_row = max(reader, key=lambda r: float(r.get("metrics/f1(B)", 0.0)))
    f1 = float(best_row["metrics/f1(B)"])
    rec = float(best_row["metrics/recall(B)"])
    prec = float(best_row["metrics/precision(B)"])
    th = float(best_row["metrics/best_th"])
    ep = int(best_row["epoch"])
    tp = int(best_row["metrics/tp"])
    fp = int(best_row["metrics/fp"])
    gt = int(best_row["metrics/gt"])

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
        "-u",  # Unbuffered stdout/stderr to ensure real-time tail visibility
        str(Path(__file__).resolve()),
        "--worker",
        "--trial-number", str(trial_number),
        "--gpu-id", str(gpu_id),
        "--output-root", str(output_root),
        "--data", str(args.data),
        "--weights", str(args.weights),
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--epochs", str(args.epochs),
        "--stride", str(args.stride),
        "--dist-thresh", str(args.dist_thresh),
        "--workers", str(args.workers),
        "--lr0", str(params["lr0"]),
        "--lrf", str(params["lrf"]),
        "--weight-decay", str(params["weight_decay"]),
        "--focal-beta", str(params["focal_beta"]),
        "--offset-weight", str(params["offset_weight"]),
        "--min-radius", str(params["min_radius"]),
        "--scale", str(params["scale"]),
        "--mosaic", str(params["mosaic"]),
        "--translate", str(params["translate"]),
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
        "lr0,lrf,weight_decay,focal_beta,offset_weight,min_radius,scale,mosaic,translate\n"
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
        f"{params['lr0']:.8f},{params['lrf']:.4f},{params['weight_decay']:.8f},"
        f"{params['focal_beta']:.4f},{params['offset_weight']:.4f},{params['min_radius']},"
        f"{params['scale']:.4f},{params['mosaic']:.4f},{params['translate']:.4f}\n"
    )
    with open(summary_csv, "a", encoding="utf-8") as f:
        f.write(row)


def main():
    if "--worker" in sys.argv:
        run_worker()
        return

    parser = argparse.ArgumentParser(description="Multi-GPU Parallel Optuna for Median SOTA")
    parser.add_argument("--data", type=str, default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml")
    parser.add_argument("--weights", type=str, default="runs/finetune_median/exp_6ep_median/weights/best_f1.pt")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-root", type=str, default="runs/optuna_median_search")
    parser.add_argument("--project", type=str, default="runs/optuna_median_search", help="Alias for output-root")
    parser.add_argument("--study-name", type=str, default="median_sota_search")
    args = parser.parse_args()

    # Support both --project and --output-root
    output_dir_str = args.output_root if args.output_root != "runs/optuna_median_search" else args.project
    output_root = Path(output_dir_str)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv = output_root / "optuna_summary.csv"

    gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip().isdigit()]
    if not gpu_list:
        raise ValueError(f"No valid GPU IDs parsed from: {args.gpus}")

    print("=" * 90, flush=True)
    print(f"Starting Multi-GPU Optuna Tuning on Aligned Temporal Median SOTA Detector", flush=True)
    print(f"Dataset: {args.data}", flush=True)
    print(f"Pretrained Weights (Starting Benchmark): {args.weights}", flush=True)
    print(f"GPUs available ({len(gpu_list)}): {gpu_list}", flush=True)
    print(f"Total Trials: {args.n_trials} | Epochs per Trial: {args.epochs}", flush=True)
    print(f"Output Root: {output_root.resolve()}", flush=True)
    print(f"Per-trial logs will be saved to: {output_root / 'logs' / 'trial_XXXX.log'}", flush=True)
    print("=" * 90 + "\n", flush=True)

    db_path = output_root / "study.db"
    storage_url = f"sqlite:///{db_path.resolve()}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
    )

    # Inject baseline seed parameters (Trial 0)
    golden_seed = {
        "lr0": 0.0001,
        "lrf": 0.1,
        "weight_decay": 0.0001,
        "focal_beta": 4.0,
        "offset_weight": 0.5,
        "min_radius": 1,
        "translate": 0.08,
        "scale": 0.15,
        "mosaic": 0.10,
    }
    try:
        study.enqueue_trial(golden_seed)
        print(colorstr("green", "[INFO] Pre-injected Golden SOTA Seed (Trial 0) into Study queue."), flush=True)
    except Exception:
        pass

    available_gpus = list(gpu_list)
    running_trials: list[dict] = []
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    pbar = tqdm(total=args.n_trials, initial=completed_trials, desc="Overall Study Progress", file=sys.stdout)

    while completed_trials < args.n_trials or running_trials:
        # 1. 派发新任务到空闲 GPU
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

        # 2. 轮询监控运行中的子进程
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
                        
                        # Precision guardrail penalty: if precision < 93.0%, penalize fitness score
                        fitness = f1
                        if metrics["precision"] < 93.0:
                            fitness = f1 - (93.0 - metrics["precision"]) * 0.01

                        # CRITICAL FIX: Set user attributes BEFORE calling study.tell()!
                        # In Optuna, once tell() is called, the trial state becomes COMPLETE and immutable.
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
    print("\n" + "=" * 90, flush=True)
    print(f"All {args.n_trials} trials completed!", flush=True)
    print(f"Best Trial: #{study.best_trial.number} with F1: {study.best_value:.4f}", flush=True)
    print(f"Best Hyperparameters: {study.best_params}", flush=True)
    print(f"Summary saved to: {summary_csv.resolve()}", flush=True)
    print("=" * 90 + "\n", flush=True)


if __name__ == "__main__":
    main()
