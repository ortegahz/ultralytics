#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Frozen Trial 0474 velocity-highway NAS."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG
from manu.heatmap_evaluate import extract_peaks, find_best_f1_threshold
from manu.heatmap_loss import HeatmapLoss, generate_heatmaps_and_targets
from manu.heatmap_model import YOLO26HeatmapDetector
from manu.velocity_highway_dataset import OfficialVelocityDataset, collate_velocity_batch
from manu.velocity_highway_module import VelocityResidualHighway

BASELINE = {"recall": 0.8619, "precision": 0.9557, "f1": 0.9064}


def load_base(path: str, p0_kwargs: dict):
    model = YOLO26HeatmapDetector(stride=2, num_classes=1, temporal_mode="standard", use_p0_highway=True, p0_highway_kwargs=p0_kwargs)
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state = state.state_dict() if hasattr(state, "state_dict") else state
    own = model.state_dict()
    for key, value in state.items():
        clean = key.replace("module.", "").replace("model.model.", "").replace("model.", "")
        if clean in own and own[clean].shape == value.shape:
            own[clean].copy_(value)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def make_loaders(data_path: str, cache_root: str, feature_channels: int, batch: int, workers: int):
    data_dict = check_det_dataset(data_path)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = 640
    cfg.data = data_path
    for key, value in {"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "degrees": 0.0, "shear": 0.0, "perspective": 0.0, "translate": 0.0, "scale": 0.0, "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0, "fliplr": 0.0, "flipud": 0.0}.items():
        setattr(cfg, key, value)
    train_base = build_yolo_dataset(cfg, data_dict["train"], batch=batch, data=data_dict, mode="train", stride=32)
    val_base = build_yolo_dataset(cfg, data_dict["val"], batch=batch, data=data_dict, mode="val", stride=32)
    train = OfficialVelocityDataset(train_base, Path(cache_root) / "train", feature_channels=feature_channels)
    val = OfficialVelocityDataset(val_base, Path(cache_root) / "val", feature_channels=feature_channels)
    return build_dataloader(train, batch=batch, workers=workers, shuffle=True), build_dataloader(val, batch=batch, workers=workers, shuffle=False)


def evaluate(model, loader, device, dist_thresh, trial_number, desc="val"):
    model.eval()
    predictions, ground_truth, sizes = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"trial {trial_number:04d} {desc}", leave=False):
            images = batch["img"].to(device, non_blocking=True).float() / 255.0
            velocity = batch["velocity_seq"].to(device, non_blocking=True)
            output = model(images, velocity)
            predictions.extend(extract_peaks(output["heatmap"], output["offset"], stride=2, conf_thresh=0.08, top_k=80))
            boxes = batch["bboxes"].cpu().numpy()
            indices = batch["batch_idx"].cpu().numpy()
            for index in range(images.shape[0]):
                ground_truth.append(boxes[indices == index] if len(boxes) else np.zeros((0, 4), dtype=np.float32))
                sizes.append((640, 640))
    return find_best_f1_threshold(predictions, ground_truth, sizes, distance_threshold=dist_thresh, thresholds=[0.10, 0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50])


def worker(args):
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(args.gpu_id)
    output = Path(args.output_root) / f"trial_{args.trial_number:04d}"
    (output / "weights").mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = make_loaders(args.data, args.cache_root, args.feature_channels, args.batch, args.workers)
    p0_kwargs = {"use_spatial_gate": True, "stem_type": "standard_dw", "downsample_mode": "pixel_unshuffle", "gate_input_mode": "diff_only", "gate_mid_channels": 16, "gate_depth": 2, "fusion_mode": "scalar_gate"}
    base = load_base(args.weights, p0_kwargs).to(device)
    highway = VelocityResidualHighway(args.feature_channels, 48, args.mid_channels, args.kernel_mode, args.depth, args.gate_mode).to(device)

    class Model(nn.Module):
        def __init__(self, detector, branch):
            super().__init__()
            self.detector = detector
            self.branch = branch

        def forward(self, images, velocity):
            if velocity.shape[-2:] != (320, 320):
                velocity = torch.nn.functional.interpolate(velocity, size=(320, 320), mode="bilinear", align_corners=False)
            features = self.detector.extract_features(images) + self.detector.p0_highway(images)
            return self.detector.head(features + self.branch(velocity))

    model = Model(base, highway).to(device)
    optimizer = torch.optim.AdamW(highway.parameters(), lr=args.lr0, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr0 * 0.1)
    scaler = GradScaler(enabled=True)
    criterion = HeatmapLoss(hm_weight=1.0, offset_weight=0.45, focal_alpha=2.0, focal_beta=2.4)
    csv_path = output / "results.csv"
    baseline_metrics = evaluate(model, val_loader, device, args.dist_thresh, args.trial_number, "baseline val")
    print(
        f"[trial {args.trial_number:04d}] Epoch 0 baseline | "
        f"F1={baseline_metrics['f1']:.6f} @ th={baseline_metrics['best_th']:.2f} | "
        f"Recall={baseline_metrics['recall']:.6f} | Precision={baseline_metrics['precision']:.6f} | "
        f"TP={baseline_metrics['tp']} FP={baseline_metrics['fp']} GT={baseline_metrics['total_gt']}",
        flush=True,
    )
    if abs(baseline_metrics["f1"] - BASELINE["f1"]) > 0.001 or abs(baseline_metrics["recall"] - BASELINE["recall"]) > 0.001 or abs(baseline_metrics["precision"] - BASELINE["precision"]) > 0.001:
        raise RuntimeError(f"Epoch-0 baseline drift: {baseline_metrics} expected={BASELINE}")
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("epoch,recall,precision,f1,tp,fp,gt,best_th,gate_norm\n")
        handle.write(f"0,{baseline_metrics['recall']:.8f},{baseline_metrics['precision']:.8f},{baseline_metrics['f1']:.8f},{baseline_metrics['tp']},{baseline_metrics['fp']},{baseline_metrics['total_gt']},{baseline_metrics['best_th']:.4f},0.00000000\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        base.eval()
        for batch in tqdm(train_loader, desc=f"trial {args.trial_number:04d} epoch {epoch} train", leave=False):
            images = batch["img"].to(device, non_blocking=True).float() / 255.0
            velocity = batch["velocity_seq"].to(device, non_blocking=True)
            boxes = batch["bboxes"].to(device, non_blocking=True)
            indices = batch["batch_idx"].to(device, non_blocking=True)
            targets = generate_heatmaps_and_targets(
                batch_bboxes=boxes,
                batch_idx=indices,
                batch_size=images.shape[0],
                feat_shape=(320, 320),
                stride=2,
                min_radius=1,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                loss, _ = criterion(model(images, velocity), targets)
            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(highway.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        scheduler.step()
        metrics = evaluate(model, val_loader, device, args.dist_thresh, args.trial_number, f"epoch {epoch} val")
        gate_norm = float(highway.gate.detach().norm().cpu())
        with csv_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{epoch},{metrics['recall']:.8f},{metrics['precision']:.8f},{metrics['f1']:.8f},{metrics['tp']},{metrics['fp']},{metrics['total_gt']},{metrics['best_th']:.4f},{gate_norm:.8f}\n")
        print(f"trial={args.trial_number:04d} epoch={epoch} recall={metrics['recall']:.6f} precision={metrics['precision']:.6f} f1={metrics['f1']:.6f} tp={metrics['tp']} fp={metrics['fp']}", flush=True)
        if metrics["recall"] > BASELINE["recall"] and metrics["precision"] > BASELINE["precision"] and metrics["f1"] > BASELINE["f1"]:
            torch.save({"highway": highway.state_dict(), "metrics": metrics, "config": vars(args), "base_weights": args.weights}, output / "weights" / "candidate.pt")
    del model, base, highway
    torch.cuda.empty_cache()


def suggest(trial):
    return {"feature_channels": 5, "mid_channels": trial.suggest_categorical("mid_channels", [8, 16, 24]), "kernel_mode": trial.suggest_categorical("kernel_mode", ["dw3", "dw5", "dilated3"]), "depth": trial.suggest_categorical("depth", [1, 2]), "gate_mode": trial.suggest_categorical("gate_mode", ["scalar", "channel", "spatial"]), "lr0": trial.suggest_float("lr0", 1e-4, 8e-4, log=True), "weight_decay": trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--trial-number", type=int)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    parser.add_argument("--trials", type=int, default=256)
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--feature-channels", type=int)
    parser.add_argument("--mid-channels", type=int)
    parser.add_argument("--kernel-mode")
    parser.add_argument("--depth", type=int)
    parser.add_argument("--gate-mode")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=47), storage=f"sqlite:///{output / 'study.db'}", load_if_exists=True)
    stale_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.RUNNING]
    for trial in stale_trials:
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
    if not study.trials:
        study.enqueue_trial({"feature_channels": 5, "mid_channels": 16, "kernel_mode": "dw3", "depth": 1, "gate_mode": "scalar", "lr0": 3e-4, "weight_decay": 1e-5})
    completed = len([trial for trial in study.trials if trial.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL)])
    gpu_list = [int(item) for item in args.gpus.split(",") if item.strip()]
    if not gpu_list:
        raise ValueError("--gpus must contain at least one GPU id")
    gpu_count = len(gpu_list)
    running = []
    print(f"[NAS] resume: completed={completed}/{args.trials}, recovered_stale={len(stale_trials)}, gpus={gpu_list}", flush=True)
    while completed < args.trials or running:
        while len(running) < gpu_count and completed + len(running) < args.trials:
            trial = study.ask()
            params = suggest(trial)
            gpu = gpu_list[len(running) % len(gpu_list)]
            trial_dir = output / f"trial_{trial.number:04d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            log = (trial_dir / "worker.log").open("w", encoding="utf-8")
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--trial-number", str(trial.number), "--gpu-id", str(gpu), "--output-root", str(output), "--data", args.data, "--cache-root", args.cache_root, "--weights", args.weights, "--batch", str(args.batch), "--epochs", str(args.epochs), "--workers", str(args.workers), "--dist-thresh", str(args.dist_thresh), "--feature-channels", str(params["feature_channels"]), "--mid-channels", str(params["mid_channels"]), "--kernel-mode", params["kernel_mode"], "--depth", str(params["depth"]), "--gate-mode", params["gate_mode"], "--lr0", str(params["lr0"]), "--weight-decay", str(params["weight_decay"])]
            running.append((trial, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
        still_running = []
        for trial, process, log in running:
            if process.poll() is None:
                still_running.append((trial, process, log))
                continue
            log.close()
            result_file = output / f"trial_{trial.number:04d}" / "results.csv"
            rows = list(csv.DictReader(result_file.open(encoding="utf-8"))) if result_file.exists() else []
            row = max(rows, key=lambda item: float(item["f1"])) if rows else None
            valid = row and float(row["recall"]) > BASELINE["recall"] and float(row["precision"]) > BASELINE["precision"] and float(row["f1"]) > BASELINE["f1"]
            study.tell(trial, float(row["f1"]) if valid else -1.0)
            completed += 1
            print(f"[NAS] completed {completed}/{args.trials} trial={trial.number} valid={bool(valid)}", flush=True)
        running = still_running
        time.sleep(2)


if __name__ == "__main__":
    main()
