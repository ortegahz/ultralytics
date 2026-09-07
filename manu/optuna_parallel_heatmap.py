#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-GPU Parallel Optuna Hyperparameter Search for Tiny UAV Point Detection via Heatmap (Stride=2, imgsz=640).

Features:
1. Multi-GPU parallel scheduling (each trial runs on a dedicated single GPU, e.g. GPUs 1, 2, 3).
2. Loads pretrained weights from the best Stride=2 Heatmap checkpoint to fine-tune.
3. Optimizes point-detection F1-score under Distance <= 4.0px criteria.
4. Auto-logs metrics to SQLite database (study.db) and CSV for easy recovery and inspection.
5. Search parameters tailored for Heatmap tiny target detection (lr0, weight_decay, offset_weight, focal_beta, scale, mosaic, translate).
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

# =========================
# 路径与默认配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_WEIGHTS = PROJECT_ROOT / "runs/heatmap_uav/uav_gpu23_heatmap_stride2/weights/best_recall.pt"
DEFAULT_DATA = "/mnt/data/siping/datasets/manu/uav/data.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "runs/optuna_heatmap_stride2_640"
LOG_ROOT = OUTPUT_ROOT / "logs"

FITNESS_KEY = "f1"  # Optuna 优化目标：Peak F1-Score


# =========================
# 搜索空间定义
# =========================
def suggest_params(trial: optuna.Trial) -> dict:
    """生成一组面向 640 尺度下 Heatmap 微调的候选超参数。"""
    return {
        # 初始学习率 (基于已有模型微调，对数分布采样)
        "lr0": trial.suggest_float("lr0", 1e-4, 1.2e-3, log=True),
        # 权重衰减
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True),
        # Heatmap Loss vs Offset Loss 权重配比
        "offset_weight": trial.suggest_float("offset_weight", 0.2, 0.8),
        # Focal Loss 难例负样本抑制因子 (控制虚警的核心旋钮，默认 4.0，适当调低可收窄高斯峰)
        "focal_beta": trial.suggest_float("focal_beta", 2.0, 4.0),
        # 极小目标保护型空间增广
        "scale": trial.suggest_float("scale", 0.05, 0.25),
        "mosaic": trial.suggest_float("mosaic", 0.0, 0.15),
        "translate": trial.suggest_float("translate", 0.05, 0.15),
    }


# =========================
# 读取单 Trial 的最优指标
# =========================
def read_best_metrics(results_csv: Path) -> tuple[float, dict]:
    """读取 results.csv 中最佳 F1-Score 及对应指标。"""
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
# 子进程：执行单卡 Trial 训练
# =========================
def worker_main(args: argparse.Namespace) -> None:
    """在指定独立单卡上运行单个 Trial 的完整训练与评测。"""
    from torch.cuda.amp import GradScaler, autocast
    from tqdm import tqdm
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG, colorstr
    from manu.heatmap_evaluate import extract_peaks, find_best_f1_threshold
    from manu.heatmap_loss import HeatmapLoss, generate_heatmaps_and_targets
    from manu.heatmap_model import YOLO26HeatmapDetector

    trial_name = f"trial_{args.trial_number:04d}"
    save_dir = Path(args.output_root) / trial_name
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 注意：父进程已经设置了 CUDA_VISIBLE_DEVICES=str(args.gpu_id)，
    # 因此在当前子进程的环境中，该物理卡已被重映射为 cuda:0。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(colorstr("bold", f"[WORKER] Starting {trial_name} on allocated GPU (physical GPU {args.gpu_id})"))

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

    # 2. 构建模型并加载已有权重微调
    model = YOLO26HeatmapDetector(stride=args.stride, num_classes=1)
    weights_path = Path(args.weights)
    if weights_path.exists():
        ckpt = torch.load(weights_path, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
        if hasattr(state_dict, "state_dict"):
            state_dict = state_dict.state_dict()
        model.load_state_dict(state_dict)
        print(f"[WORKER] Loaded pretrained checkpoint: {weights_path.name}")
    else:
        print(f"[WARN] Checkpoint not found: {weights_path}, training from scratch.")

    model.to(device)

    # 3. 优化器、调度器与 Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr0, weight_decay=args.weight_decay)
    lf = lambda epoch: ((1 + math.cos(epoch * math.pi / args.epochs)) / 2) * (1 - 0.01) + 0.01
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    criterion = HeatmapLoss(hm_weight=1.0, offset_weight=args.offset_weight)
    criterion.focal_loss.beta = args.focal_beta

    feat_h = args.imgsz // args.stride
    feat_w = args.imgsz // args.stride

    csv_path = save_dir / "results.csv"
    csv_header = "epoch,train/loss,train/loss_hm,train/loss_off,metrics/best_th,metrics/recall(B),metrics/precision(B),metrics/f1(B),metrics/tp,metrics/fp,metrics/gt,lr\n"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_header)

    best_f1 = 0.0

    # 4. 训练循环
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
            with autocast(enabled=(device.type == "cuda")):
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
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })

        scheduler.step()

        # 5. 验证评估
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
            thresholds=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
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
# 父进程：拉起子进程 Trial
# =========================
def launch_trial(
    trial: optuna.Trial,
    params: dict,
    output_root: Path,
    gpu_id: int,
    args: argparse.Namespace,
) -> dict:
    """在指定 GPU 上启动 worker 子进程。"""
    trial_number = trial.number
    trial_name = f"trial_{trial_number:04d}"

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_file = LOG_ROOT / f"{trial_name}.log"

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
        "--workers",
        str(args.workers),
        "--dist-thresh",
        str(args.dist_thresh),
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

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"

    log_handle = log_file.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )

    print(f"[START] {trial_name} on GPU {gpu_id} (pid={process.pid}, log={log_file})")

    return {
        "trial": trial,
        "params": params,
        "trial_number": trial_number,
        "gpu_id": gpu_id,
        "process": process,
        "log_handle": log_handle,
        "log_file": log_file,
    }


# =========================
# 等待 Trial 完成并提取指标
# =========================
def finish_trial(item: dict, output_root: Path) -> float:
    """等待单个 Trial 子进程结束，解析最优指标。"""
    process = item["process"]
    process.wait()
    item["log_handle"].close()

    trial = item["trial"]
    trial_number = item["trial_number"]

    if process.returncode != 0:
        raise RuntimeError(f"Trial {trial_number} failed with exit code {process.returncode}. Log: {item['log_file']}")

    trial_dir = output_root / f"trial_{trial_number:04d}"
    results_csv = trial_dir / "results.csv"
    best_f1, metrics = read_best_metrics(results_csv)

    trial.set_user_attr("metrics", metrics)
    trial.set_user_attr("recall", metrics["recall"])
    trial.set_user_attr("precision", metrics["precision"])
    trial.set_user_attr("results_csv", str(results_csv))

    print(
        f"[DONE] {trial_dir.name} (GPU {item['gpu_id']}) -> "
        f"F1={best_f1:.4f} | Recall={metrics['recall']:.4f} | Precision={metrics['precision']:.4f}"
    )
    return best_f1


# =========================
# 调度器主入口
# =========================
def scheduler_main(args: argparse.Namespace) -> None:
    """并行多 GPU 调度器，满载管理可用 GPU。"""
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    database_path = output_root / "study.db"
    storage_url = f"sqlite:///{database_path}"

    study = optuna.create_study(
        study_name="heatmap_stride2_640",
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=4),
    )

    gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not gpu_list:
        raise ValueError("No valid GPU IDs specified in --gpus")

    print("=" * 75)
    print("UAV Heatmap Multi-GPU Optuna Search")
    print("=" * 75)
    print(f"Total trials       : {args.n_trials}")
    print(f"Allocated GPUs     : {gpu_list}")
    print(f"Initial weights    : {args.weights}")
    print(f"Input size         : {args.imgsz}")
    print(f"Epochs per trial   : {args.epochs}")
    print(f"Batch per GPU      : {args.batch}")
    print(f"Fitness objective  : Peak F1-Score (Distance <= {args.dist_thresh}px)")
    print(f"Output directory   : {output_root}")
    print(f"Database           : {database_path}")
    print("=" * 75)

    completed_before = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if completed_before:
        print(f"[INFO] Reusing {completed_before} completed trials from database.")

    # 维护当前运行中的任务字典: {gpu_id: item}
    running_tasks: dict[int, dict] = {}
    free_gpus = list(gpu_list)

    while len(study.trials) < args.n_trials or len(running_tasks) > 0:
        # 1. 如果有空闲卡且还有剩余 trial 额度，立即派发新任务
        while free_gpus and len(study.trials) < args.n_trials:
            gpu_id = free_gpus.pop(0)
            trial = study.ask()
            params = suggest_params(trial)
            item = launch_trial(trial, params, output_root, gpu_id, args)
            running_tasks[gpu_id] = item

        if not running_tasks:
            break

        # 2. 轮询检查是否有已完成的 GPU 任务
        time.sleep(2.0)
        done_gpus = []
        for gpu_id, item in running_tasks.items():
            if item["process"].poll() is not None:
                done_gpus.append(gpu_id)

        # 3. 收集完成任务，释放 GPU 并通知 Optuna
        for gpu_id in done_gpus:
            item = running_tasks.pop(gpu_id)
            trial = item["trial"]
            try:
                f1_score = finish_trial(item, output_root)
                study.tell(trial, f1_score)
            except Exception as exc:
                print(f"[FAILED] Trial {trial.number} on GPU {gpu_id}: {exc}")
                study.tell(trial, state=optuna.trial.TrialState.FAIL)

            free_gpus.append(gpu_id)

    best_trial = study.best_trial
    print("\n" + "=" * 75)
    print("OPTUNA SEARCH COMPLETED SUCCESSFULLY")
    print("=" * 75)
    print(f"Best Trial Number : {best_trial.number}")
    print(f"Best F1-Score     : {best_trial.value:.6f}")
    print("\nBest Parameters:")
    for k, v in best_trial.params.items():
        print(f"  {k:<16}: {v}")
    print("\nBest Trial Detailed Metrics:")
    for k, v in best_trial.user_attrs.get("metrics", {}).items():
        print(f"  {k:<16}: {v}")
    print("=" * 75)


# =========================
# 命令行参数解析
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-GPU Optuna Search for UAV Heatmap Detector")
    parser.add_argument("--worker", action="store_true", help="Internal worker mode for single trial")
    parser.add_argument("--gpus", type=str, default="1,2,3", help="Comma-separated GPU IDs, e.g. 1,2,3")
    parser.add_argument("--n-trials", type=int, default=50, help="Total number of trials to run")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per trial")
    parser.add_argument("--batch", type=int, default=32, help="Batch size per GPU (default: 32, matches train_uav_heatmap)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--stride", type=int, default=2, help="Feature stride (2 for P1 high-res)")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Initial pretrained weights (.pt)")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to data.yaml")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers per worker process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Optuna sampler")
    parser.add_argument("--dist-thresh", type=float, default=4.0, help="Distance threshold (pixels) for TP matching")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT), help="Output directory")

    # Worker 专属参数
    parser.add_argument("--trial-number", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--offset-weight", type=float, default=0.5)
    parser.add_argument("--focal-beta", type=float, default=3.0)
    parser.add_argument("--scale", type=float, default=0.15)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.10)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker:
        worker_main(args)
    else:
        scheduler_main(args)


if __name__ == "__main__":
    main()
