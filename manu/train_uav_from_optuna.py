#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 Optuna 最佳超参数进行 UAV 正式训练。

功能：
1. 自动执行 report_optuna_trials.py；
2. 从 Optuna study.best_trial 读取真正的最佳参数；
3. 使用最佳参数进行 20 epoch 正式训练；
4. 使用 4 张 GPU；
5. patience=0，不提前停止；
6. 最后 5 个 epoch 关闭 Mosaic。

运行：

python manu/train_uav_from_optuna.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import optuna

from ultralytics import YOLO

# ======================== 用户配置区 ========================

# 项目根目录
PROJECT_ROOT = Path(
    "/tmp/pycharm_project_10ae9e2e"
).resolve()

# Optuna 输出目录
OPTUNA_ROOT = PROJECT_ROOT / "runs/optuna_uav_ap50"

# Optuna 数据库
OPTUNA_DATABASE = OPTUNA_ROOT / "study.db"

# Optuna study 名称
OPTUNA_STUDY_NAME = "uav_fold4_yolo26n_ap50"

# Optuna 报告脚本
REPORT_SCRIPT = PROJECT_ROOT / "manu/report_optuna_trials.py"

# 预训练模型
MODEL_PATH = PROJECT_ROOT / "yolo26n.pt"

# 数据集配置
DATA_PATH = PROJECT_ROOT / "datasets/uav/data.yaml"

# 正式训练输出目录
TRAIN_PROJECT = PROJECT_ROOT / "runs/detect"

# 正式训练名称
TRAIN_NAME = "train_uav_optuna_best"

# 正式训练时用于选择 best.pt 的 fitness。
FITNESS_METRIC = "metrics/recall(B)"

# 使用的 GPU
DEVICE = [0, 1, 2, 3]

# 正式训练 epoch
EPOCHS = 20

# Batch size。
# Ultralytics 在多卡训练时，通常表示每张 GPU 的 batch size。
BATCH = 32

# 数据加载线程
WORKERS = 8

# 随机种子
SEED = 42

# 是否重新覆盖同名正式训练目录
EXIST_OK = False


# ============================================================


def run_report() -> None:
    """
    自动调用 report_optuna_trials.py，打印 Optuna 结果。
    """
    if not REPORT_SCRIPT.exists():
        raise FileNotFoundError(
            f"找不到报告脚本：{REPORT_SCRIPT}"
        )

    print("=" * 80)
    print("运行 Optuna trial 报告")
    print("=" * 80)

    command = [
        sys.executable,
        str(REPORT_SCRIPT),
        "--root",
        str(OPTUNA_ROOT),
        "--study-name",
        OPTUNA_STUDY_NAME,
        "--sort-by",
        "ap50",
    ]

    subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def load_best_trial() -> optuna.trial.FrozenTrial:
    """
    从 Optuna 数据库中读取真正的 best_trial。

    使用 report_optuna_trials.py 对应的 Optuna AP50 最优 trial。
    """
    if not OPTUNA_DATABASE.exists():
        raise FileNotFoundError(
            f"找不到 Optuna 数据库：{OPTUNA_DATABASE}"
        )

    storage = f"sqlite:///{OPTUNA_DATABASE}"

    study = optuna.load_study(
        study_name=OPTUNA_STUDY_NAME,
        storage=storage,
    )

    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    if not complete_trials:
        raise RuntimeError(
            f"Optuna study 中没有 COMPLETE trial："
            f"{OPTUNA_STUDY_NAME}"
        )

    best_trial = study.best_trial

    print()
    print("=" * 80)
    print("Optuna 最佳 Trial")
    print("=" * 80)
    print(f"Study       : {OPTUNA_STUDY_NAME}")
    print(f"Trial       : {best_trial.number}")
    print(f"Optuna AP50 : {best_trial.value:.6f}")
    print(f"Params      :")

    for key, value in best_trial.params.items():
        print(f"  {key:<16}: {value}")

    print("=" * 80)

    return best_trial


def train_with_best_params(
        best_trial: optuna.trial.FrozenTrial,
) -> None:
    """
    使用 Optuna 最佳参数进行正式训练。
    """
    params = best_trial.params

    required_params = [
        "lr0",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "scale",
        "mosaic",
    ]

    missing_params = [
        key
        for key in required_params
        if key not in params
    ]

    if missing_params:
        raise KeyError(
            "Optuna 最佳 trial 缺少参数："
            + ", ".join(missing_params)
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{MODEL_PATH}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到数据集配置：{DATA_PATH}"
        )

    print()
    print("=" * 80)
    print("开始正式训练")
    print("=" * 80)
    print(f"Model       : {MODEL_PATH}")
    print(f"Data        : {DATA_PATH}")
    print(f"Project     : {TRAIN_PROJECT}")
    print(f"Name        : {TRAIN_NAME}")
    print(f"Device      : {DEVICE}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Batch       : {BATCH}")
    print(f"Patience    : 0")
    print(f"Close mosaic: 5")
    print("=" * 80)

    model = YOLO(str(MODEL_PATH))

    model.train(
        # 数据集
        data=str(DATA_PATH),

        # 输入尺寸
        imgsz=640,

        # 正式训练 20 epoch
        epochs=EPOCHS,

        # 四卡训练
        device=DEVICE,

        # 每张 GPU 的 batch size
        batch=BATCH,

        # Optuna 搜索得到的优化器参数
        optimizer="SGD",
        lr0=float(params["lr0"]),
        momentum=float(params["momentum"]),
        weight_decay=float(params["weight_decay"]),
        warmup_epochs=float(params["warmup_epochs"]),

        # Optuna 搜索得到的数据增强参数
        scale=float(params["scale"]),
        mosaic=float(params["mosaic"]),

        # 最后 5 个 epoch 关闭 Mosaic
        close_mosaic=5,

        # 不提前停止
        patience=0,

        # 正式训练按 Recall 选择 best.pt
        fitness_metric=FITNESS_METRIC,

        # 数据加载
        workers=WORKERS,

        # 保存策略
        save=True,
        save_period=10,

        # 输出训练曲线和混淆矩阵等图像
        plots=True,

        # Frame Difference 输入关闭 HSV 增强
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        # 空间增强
        degrees=0.0,
        translate=0.1,
        shear=0.0,
        perspective=0.0,

        # 翻转
        fliplr=0.5,
        flipud=0.0,

        # 关闭 Copy-Paste
        copy_paste=0.0,

        # 正式训练沿用 train_uav.py 的 MixUp 配置
        mixup=0.0,

        # 单类别 UAV 检测
        single_cls=True,

        # 复现性
        seed=SEED,
        deterministic=True,

        # 输出目录
        project=str(TRAIN_PROJECT),
        name=TRAIN_NAME,
        exist_ok=EXIST_OK,

        # 打印详细训练信息
        verbose=True,
    )

    print()
    print("=" * 80)
    print("正式训练完成")
    print("=" * 80)
    print(f"训练目录：{TRAIN_PROJECT / TRAIN_NAME}")
    print(
        f"最佳权重："
        f"{TRAIN_PROJECT / TRAIN_NAME / 'weights/best.pt'}"
    )
    print(
        f"最终权重："
        f"{TRAIN_PROJECT / TRAIN_NAME / 'weights/last.pt'}"
    )
    print("=" * 80)


def main() -> None:
    run_report()
    best_trial = load_best_trial()
    train_with_best_params(best_trial)


if __name__ == "__main__":
    main()
