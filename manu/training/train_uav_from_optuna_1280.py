#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train the UAV model at 1280 resolution with the current best parameters."""

from __future__ import annotations

from train_uav_from_optuna import (
    BATCH,
    DATA_PATH,
    DEVICE,
    EPOCHS,
    EXIST_OK,
    FITNESS_METRIC,
    MODEL_PATH,
    PROJECT_ROOT,
    SEED,
    WORKERS,
    load_best_trial,
    run_report,
)
from ultralytics import YOLO

IMG_SIZE = 1280
TRAIN_PROJECT = PROJECT_ROOT / "runs/detect"
TRAIN_NAME = "train_uav_optuna_best_1280"


def train_with_best_params_1280(best_trial) -> None:
    """Train from yolo26np2.pt with the best Recall-search parameters."""
    params = best_trial.params
    required_params = [
        "lr0",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "scale",
        "mosaic",
    ]
    missing_params = [key for key in required_params if key not in params]
    if missing_params:
        raise KeyError(
            "Optuna 最佳 trial 缺少参数：" + ", ".join(missing_params)
        )

    for path, label in ((MODEL_PATH, "模型"), (DATA_PATH, "数据集配置")):
        if not path.exists():
            raise FileNotFoundError(f"找不到{label}：{path}")

    print("=" * 80)
    print("开始 1280 分辨率正式训练")
    print("=" * 80)
    print(f"Pretrained  : {MODEL_PATH}")
    print(f"Data        : {DATA_PATH}")
    print(f"Project     : {TRAIN_PROJECT}")
    print(f"Name        : {TRAIN_NAME}")
    print(f"Image size  : {IMG_SIZE}")
    print(f"Device      : {DEVICE}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Batch       : {BATCH}")
    print("Close mosaic: 5")
    print("=" * 80)

    model = YOLO(str(MODEL_PATH))
    model.train(
        data=str(DATA_PATH),
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        device=DEVICE,
        batch=BATCH,
        optimizer="SGD",
        lr0=float(params["lr0"]),
        momentum=float(params["momentum"]),
        weight_decay=float(params["weight_decay"]),
        warmup_epochs=float(params["warmup_epochs"]),
        scale=float(params["scale"]),
        mosaic=float(params["mosaic"]),
        close_mosaic=5,
        patience=0,
        fitness_metric=FITNESS_METRIC,
        workers=WORKERS,
        save=True,
        save_period=10,
        plots=True,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        bgr=0.0,
        degrees=0.0,
        translate=0.1,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        copy_paste=0.0,
        mixup=0.1,
        erasing=0.4,
        single_cls=True,
        seed=SEED,
        deterministic=True,
        project=str(TRAIN_PROJECT),
        name=TRAIN_NAME,
        exist_ok=EXIST_OK,
        verbose=True,
    )

    output_dir = TRAIN_PROJECT / TRAIN_NAME
    print("=" * 80)
    print("1280 分辨率正式训练完成")
    print(f"训练目录：{output_dir}")
    print(f"最佳权重：{output_dir / 'weights/best.pt'}")
    print(f"最终权重：{output_dir / 'weights/last.pt'}")
    print("=" * 80)


def main() -> None:
    run_report()
    best_trial = load_best_trial()
    train_with_best_params_1280(best_trial)


if __name__ == "__main__":
    main()
