#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-GPU Parallel Optuna Hyperparameter Search & Deep Fine-tuning for Tiny UAV Point Detection (Heatmap Stride=2).

Features:
1. Multi-GPU parallel scheduling (1 trial per GPU, e.g. GPUs 0, 1, 2, 3).
2. Deep fine-tuning (default 40 epochs) initialized from the champion checkpoint (trial_0031).
3. Pre-enqueues Top-4 golden parameter sets from historical search:
   - Rank 1: Trial 0031 (Champion baseline)
   - Rank 2: Trial 0079
   - Rank 3: Trial 0059
   - Rank 4: Trial 0044
4. Exploration space is tailored for deep fine-tuning (stable learning rates, delicate augmentation).
5. Seamless worker loop: whenever a GPU finishes its 40 epochs, it immediately takes the next trial from Optuna.
6. Auto-logs metrics to SQLite database (study.db) and CSV for easy recovery and real-time monitoring.
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

DEFAULT_WEIGHTS = PROJECT_ROOT / "runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt"
DEFAULT_DATA = "/mnt/data/siping/datasets/manu/uav/data.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "runs/optuna_heatmap_stride2_fine40"
LOG_ROOT = OUTPUT_ROOT / "logs"

FITNESS_KEY = "f1"

# =========================
# 历史 Top-4 黄金参数池 (Top 1 trial_0031 + Top 2~4)
# =========================
HISTORICAL_TOP_TRIALS = [
    {
        "name": "trial_0031 (Rank 1 Champion)",
        "params": {
            "lr0": 0.00031257055949452447,
            "weight_decay": 1.5054113785546842e-05,
            "offset_weight": 0.3435252197465767,
            "focal_beta": 2.189633389327753,
            "scale": 0.09183976220206519,
            "mosaic": 0.08647445067666594,
            "translate": 0.14208511242153077,
        },
    },
    {
        "name": "trial_0079 (Rank 2)",
        "params": {
            "lr0": 0.0002641331893454064,
            "weight_decay": 1.945748107086009e-05,
            "offset_weight": 0.3860817965528071,
            "focal_beta": 2.077952599914415,
            "scale": 0.07147326356488404,
            "mosaic": 0.0585460347134236,
            "translate": 0.12285348485094467,
        },
    },
    {
        "name": "trial_0059 (Rank 3)",
        "params": {
            "lr0": 0.0008173483634663189,
            "weight_decay": 2.6172819531935735e-05,
            "offset_weight": 0.34760862002424275,
            "focal_beta": 2.2655625688176393,
            "scale": 0.15752748298337615,
            "mosaic": 0.14607541056000975,
            "translate": 0.08248928365613215,
        },
    },
    {
        "name": "trial_0044 (Rank 4)",
        "params": {
            "lr0": 0.0003631615545046237,
            "weight_decay": 1.7931312119915866e-05,
            "offset_weight": 0.2793122300602119,
            "focal_beta": 2.752327304306524,
            "scale": 0.07607383081524063,
            "mosaic": 0.06853532868072451,
            "translate": 0.060727389431225076,
        },
    },
]


# =========================
# 微调搜索空间定义 (针对 40 Epochs 长周期微调收紧优化)
# =========================
def suggest_params(trial: optuna.Trial) -> dict:
    """生成面向在 trial_0031 基础上进行 40 轮深层微调的候选参数。"""
    return {
        # 初始学习率：收紧在温和微调区间 (8e-5 ~ 6e-4)，避免破坏已学好的精细特征
        "lr0": trial.suggest_float("lr0", 8e-5, 6e-4, log=True),
        # 权重衰减：聚焦在 1e-5 ~ 8e-5 黄金区间
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 8e-5, log=True),
        # Heatmap Loss 与 Offset Loss 权重比
        "offset_weight": trial.suggest_float("offset_weight", 0.25, 0.45),
        # Focal Loss 难例负样本抑制因子：Top 结果高度集中在 2.0 ~ 2.8
        "focal_beta": trial.suggest_float("focal_beta", 2.0, 2.8),
        # 空间增广：温和微调尺度，保护弱小目标像素能量
        "scale": trial.suggest_float("scale", 0.05, 0.15),
        "mosaic": trial.suggest_float("mosaic", 0.02, 0.12),
        "translate": trial.suggest_float("translate", 0.06, 0.15),
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
        raise RuntimeError(f"No valid metrics in {results_csv}")

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
# 子进程：执行单卡 Trial 训练 (40 Epochs)
# =========================
def worker_main(args: argparse.Namespace) -> None:
    """在指定独立单卡上运行单个 Trial 的完整微调训练与评测。"""
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
        print(f"[WORKER] Loaded pretrained checkpoint: {weights_path}")
    else:
        raise FileNotFoundError(f"Initial weights file not found: {weights_path}")

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

    # 4. 训练循环 (40 Epochs)
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

        # 5. 验证评估 (每轮评估)
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

        # 保存最优与最新权重
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

        # 同时也保存最新的 checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "metrics": metrics,
                "stride": args.stride,
                "imgsz": args.imgsz,
            },
            weights_dir / "last.pt",
        )

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
        f"[DONE] Trial {trial_number:04d} on GPU {item['gpu_id']} finished! "
        f"F1={best_f1:.4f} (Rec={metrics['recall']:.4f}, Prec={metrics['precision']:.4f}) @ Ep {metrics['epoch']}"
    )

    return best_f1


# =========================
# 调度器主函数
# =========================
def scheduler_main(args: argparse.Namespace) -> None:
    """并行多 GPU 调度器，满载管理可用 GPU。"""
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    database_path = output_root / "study.db"
    storage_url = f"sqlite:///{database_path}"

    study = optuna.create_study(
        study_name="heatmap_stride2_fine40",
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.startup_trials),
    )

    # 预先压入历史最优 Top-4 组参数（确保首轮 4 张 GPU 立即启动这 4 组黄金参数）
    if len(study.trials) == 0:
        print("[INFO] Enqueueing historical Top-4 candidate parameter sets into study queue:")
        for top_item in HISTORICAL_TOP_TRIALS:
            print(f"  -> {top_item['name']}: {top_item['params']}")
            study.enqueue_trial(top_item["params"])

    gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not gpu_list:
        raise ValueError("No valid GPU IDs specified in --gpus")

    print("=" * 75)
    print("UAV Heatmap Multi-GPU Deep Fine-tuning Search (40 Epochs)")
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

        # 3. 收集完成任务，释放 GPU 并通知 Optuna，空闲 GPU 将在下轮循环立即领取新任务
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
    print("OPTUNA DEEP FINE-TUNING COMPLETED SUCCESSFULLY")
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
    parser = argparse.ArgumentParser(description="Multi-GPU Deep Fine-tuning Search for UAV Heatmap Detector")
    parser.add_argument("--worker", action="store_true", help="Internal worker mode for single trial")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma-separated GPU IDs, e.g. 0,1,2,3")
    parser.add_argument("--n-trials", type=int, default=20, help="Total number of trials to run")
    parser.add_argument("--epochs", type=int, default=40, help="Epochs per trial (default: 40 for deep fine-tuning)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--stride", type=int, default=2, help="Feature stride (2 for P1 high-res)")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Initial pretrained weights (.pt)")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to data.yaml")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers per worker process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Optuna sampler")
    parser.add_argument("--startup-trials", type=int, default=4, help="Number of random startup trials if queue is empty")
    parser.add_argument("--dist-thresh", type=float, default=4.0, help="Distance threshold (pixels) for TP matching")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT), help="Output directory")

    # Worker 专属参数
    parser.add_argument("--trial-number", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=1.5e-5)
    parser.add_argument("--offset-weight", type=float, default=0.35)
    parser.add_argument("--focal-beta", type=float, default=2.2)
    parser.add_argument("--scale", type=float, default=0.09)
    parser.add_argument("--mosaic", type=float, default=0.08)
    parser.add_argument("--translate", type=float, default=0.14)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker:
        worker_main(args)
    else:
        scheduler_main(args)


if __name__ == "__main__":
    main()
