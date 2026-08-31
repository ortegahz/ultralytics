#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import optuna

# =========================
# 固定路径
# =========================

PROJECT_ROOT = Path(
    "/tmp/pycharm_project_10ae9e2e"
).resolve()

MODEL_PATH = PROJECT_ROOT / "yolo26n.pt"
DATA_PATH = PROJECT_ROOT / "datasets/uav/data.yaml"

OUTPUT_ROOT = PROJECT_ROOT / "runs/optuna_uav"
LOG_ROOT = OUTPUT_ROOT / "logs"

# 每个 trial 使用全部四张 GPU
GPU_IDS = "0,1,2,3"


# =========================
# 搜索空间
# =========================

def suggest_params(trial: optuna.Trial) -> dict:
    """Generate one hyperparameter configuration."""
    return {
        # SGD 初始学习率
        "lr0": trial.suggest_float(
            "lr0",
            0.005,
            0.02,
        ),

        # SGD momentum
        "momentum": trial.suggest_float(
            "momentum",
            0.90,
            0.97,
        ),

        # 权重衰减
        "weight_decay": trial.suggest_float(
            "weight_decay",
            1e-4,
            1e-3,
            log=True,
        ),

        # warmup epoch
        "warmup_epochs": trial.suggest_float(
            "warmup_epochs",
            0.5,
            3.0,
        ),

        # Ultralytics scale
        "scale": trial.suggest_float(
            "scale",
            0.2,
            0.4,
        ),

        # Mosaic 概率
        "mosaic": trial.suggest_float(
            "mosaic",
            0.3,
            0.7,
        ),
    }


# =========================
# 读取训练结果
# =========================

def read_best_metrics(results_csv: Path) -> tuple[float, dict]:
    """
    Read the best Recall and corresponding metrics.

    Recall is used as the product-oriented objective.
    """
    if not results_csv.exists():
        raise FileNotFoundError(
            f"results.csv not found: {results_csv}"
        )

    with results_csv.open(
            "r",
            encoding="utf-8",
            newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError(
            f"results.csv is empty: {results_csv}"
        )

    rows = [
        {
            key.strip(): value.strip()
            if isinstance(value, str)
            else value
            for key, value in row.items()
        }
        for row in rows
    ]

    fitness_key = "metrics/recall(B)"
    valid_rows = []

    for row in rows:
        raw_value = row.get(fitness_key)

        if raw_value is None or raw_value == "":
            continue

        try:
            row["_fitness"] = float(raw_value)
        except ValueError:
            continue

        valid_rows.append(row)

    if not valid_rows:
        raise RuntimeError(
            f"No valid {fitness_key} found in {results_csv}"
        )

    best_row = max(
        valid_rows,
        key=lambda row: row["_fitness"],
    )

    metrics = {
        "epoch": best_row.get("epoch"),
        "ap50": best_row.get("metrics/mAP50(B)"),
        "map50_95": best_row.get("metrics/mAP50-95(B)"),
        "precision": best_row.get("metrics/precision(B)"),
        "recall": best_row.get("metrics/recall(B)"),
    }

    return best_row["_fitness"], metrics


# =========================
# 子进程：训练一个 trial
# =========================

def worker_main(args: argparse.Namespace) -> None:
    """Run one four-GPU YOLO training trial."""
    from ultralytics import YOLO

    trial_name = f"trial_{args.trial_number:04d}"
    output_root = Path(args.output_root).resolve()

    print(f"[WORKER] Starting {trial_name}")
    print(f"[WORKER] output: {output_root / trial_name}")
    print(f"[WORKER] params:")
    print(f"  lr0={args.lr0}")
    print(f"  momentum={args.momentum}")
    print(f"  weight_decay={args.weight_decay}")
    print(f"  warmup_epochs={args.warmup_epochs}")
    print(f"  scale={args.scale}")
    print(f"  mosaic={args.mosaic}")

    model = YOLO(str(MODEL_PATH))

    model.train(
        data=str(DATA_PATH),

        # 快速参数搜索
        epochs=10,
        patience=0,

        # 每个 trial 使用四张 GPU
        device=[0, 1, 2, 3],
        batch=32,
        imgsz=640,

        # 论文使用 SGD
        optimizer="SGD",
        lr0=args.lr0,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,

        # Frame Difference 输入固定设置
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        bgr=0.0,

        # 论文相关固定配置
        mixup=0.1,
        close_mosaic=3,

        # 当前版本如果 detection pipeline 不使用该参数，
        # 它不会影响本次搜索
        erasing=0.4,

        # 搜索参数
        scale=args.scale,
        mosaic=args.mosaic,

        # 其他增强固定
        degrees=0.0,
        translate=0.1,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        copy_paste=0.0,

        # 单类别 UAV
        single_cls=True,

        # 复现性
        seed=42,
        deterministic=True,

        # 搜索阶段减少 IO
        plots=False,
        save=True,
        save_period=10,

        # 每个 trial 独立目录
        project=str(output_root),
        name=trial_name,
        exist_ok=False,

        # 四个 trial 同时运行时，降低 DataLoader worker 数
        workers=4,

        verbose=True,
    )


# =========================
# 父进程：启动一个 trial
# =========================

def launch_trial(
        trial: optuna.Trial,
        params: dict,
        output_root: Path,
):
    """Launch one training subprocess."""
    trial_number = trial.number
    trial_name = f"trial_{trial_number:04d}"

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = LOG_ROOT / f"{trial_name}.log"

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--trial-number",
        str(trial_number),
        "--output-root",
        str(output_root),
        "--lr0",
        str(params["lr0"]),
        "--momentum",
        str(params["momentum"]),
        "--weight-decay",
        str(params["weight_decay"]),
        "--warmup-epochs",
        str(params["warmup_epochs"]),
        "--scale",
        str(params["scale"]),
        "--mosaic",
        str(params["mosaic"]),
    ]

    env = os.environ.copy()

    # 每个 trial 都使用四张 GPU
    env["CUDA_VISIBLE_DEVICES"] = GPU_IDS

    # 防止每个 DDP 子进程过度争抢 CPU 线程
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    log_handle = log_file.open(
        "w",
        encoding="utf-8",
    )

    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )

    print(
        f"[START] {trial_name} "
        f"pid={process.pid} "
        f"log={log_file}"
    )

    return {
        "trial": trial,
        "params": params,
        "trial_number": trial_number,
        "process": process,
        "log_handle": log_handle,
        "log_file": log_file,
    }


# =========================
# 完成一个 trial 并反馈 Optuna
# =========================

def finish_trial(item: dict, output_root: Path) -> float:
    """Wait for a trial and return its best mAP50-95."""
    process = item["process"]
    process.wait()

    item["log_handle"].close()

    trial = item["trial"]
    trial_number = item["trial_number"]

    if process.returncode != 0:
        raise RuntimeError(
            f"Trial {trial_number} failed with return code "
            f"{process.returncode}. Log: {item['log_file']}"
        )

    trial_dir = output_root / f"trial_{trial_number:04d}"
    results_csv = trial_dir / "results.csv"

    _, metrics = read_best_metrics(results_csv)

    trial.set_user_attr("metrics", metrics)
    trial.set_user_attr("results_csv", str(results_csv))

    # 以 Recall 作为 Optuna 的 fitness
    fitness = float(metrics["recall"])

    print(
        f"[DONE] trial={trial_number} "
        f"Recall={fitness:.6f} "
        f"AP50={metrics['ap50']} "
        f"mAP50-95={metrics['map50_95']} "
        f"Precision={metrics['precision']}"
    )

    return fitness


# =========================
# 调度器
# =========================

def scheduler_main(args: argparse.Namespace) -> None:
    """Run four concurrent four-GPU trials."""
    output_root = Path(args.output_root).resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    database_path = output_root / "study.db"
    storage_url = f"sqlite:///{database_path}"

    study = optuna.create_study(
        study_name="uav_fold4_yolo26n",
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=args.seed,
            # 第一批参数先随机探索
            # 之后 TPE 根据结果调整搜索方向
            n_startup_trials=4,
        ),
    )

    print("=" * 70)
    print("Optuna parallel search")
    print("=" * 70)
    print(f"Total trials       : {args.n_trials}")
    print(f"Parallel trials    : {args.parallel}")
    print("GPUs per trial     : 4")
    print("Epochs per trial   : 10")
    print("Close mosaic       : 3")
    print(f"Output directory   : {output_root}")
    print(f"Optuna database    : {database_path}")
    print("=" * 70)

    completed_before = len(study.trials)

    while len(study.trials) < args.n_trials:
        active = []

        remaining = args.n_trials - len(study.trials)
        batch_size = min(
            args.parallel,
            remaining,
        )

        # 生成当前批次的参数
        for _ in range(batch_size):
            trial = study.ask()
            params = suggest_params(trial)

            item = launch_trial(
                trial=trial,
                params=params,
                output_root=output_root,
            )
            active.append(item)

        # 当前批次全部结束后，统一反馈给 Optuna
        for item in active:
            trial = item["trial"]

            try:
                fitness = finish_trial(
                    item=item,
                    output_root=output_root,
                )
                study.tell(
                    trial,
                    fitness,
                )
            except Exception as exc:
                print(
                    f"[FAILED] trial={trial.number}: {exc}"
                )
                study.tell(
                    trial,
                    state=optuna.trial.TrialState.FAIL,
                )

        print(
            f"[INFO] trials in study: "
            f"{len(study.trials)}/{args.n_trials}"
        )

    best_trial = study.best_trial

    print("\n" + "=" * 70)
    print("OPTUNA SEARCH FINISHED")
    print("=" * 70)
    print(f"Best trial number : {best_trial.number}")
    print(f"Best fitness      : {best_trial.value:.6f}")
    print("\nBest parameters:")

    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    print("\nBest trial metrics:")
    for key, value in best_trial.user_attrs.get(
            "metrics",
            {},
    ).items():
        print(f"  {key}: {value}")

    print(f"\nOptuna database: {database_path}")
    print("=" * 70)

    if completed_before:
        print(
            f"\n[INFO] Existing trials were reused: "
            f"{completed_before}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--worker",
        action="store_true",
        help="Run one trial instead of the scheduler.",
    )

    parser.add_argument(
        "--trial-number",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=8,
        help="Total number of trials.",
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of simultaneous four-GPU trials.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument("--lr0", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-epochs", type=float)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--mosaic", type=float)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.worker:
        worker_main(args)
    else:
        scheduler_main(args)


if __name__ == "__main__":
    main()
