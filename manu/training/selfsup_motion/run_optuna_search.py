#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Single-GPU worker for a resumable multi-process Optuna study."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import optuna
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_cache import MotionMemmapDataset, worker_init_fn
from disentangle_model import MotionDisentangler, compute_loss
from unsupervised_metrics import saliency_fitness


def stage_boundaries(epochs, stage1_ratio, stage2_ratio):
    if epochs <= 1:
        return 1, 1
    first = min(max(1, int(round(epochs * stage1_ratio))), epochs - 1)
    second = min(first + max(1, int(round(epochs * stage2_ratio))), epochs)
    return first, second


def set_stage(model, stage):
    ego_trainable = stage in (1, 3)
    saliency_trainable = stage in (2, 3)
    for parameter in model.ego.parameters():
        parameter.requires_grad = ego_trainable
    for parameter in model.saliency.parameters():
        parameter.requires_grad = saliency_trainable
    model.ego.train(ego_trainable)
    model.saliency.train(saliency_trainable)


def objective(trial, args):
    device = torch.device("cuda:0")
    model = MotionDisentangler(
        trial.suggest_categorical("base_channels", [16, 32, 64]),
        trial.suggest_categorical("saliency_kernel_size", [3, 5, 7]),
    ).to(device)
    lr_cam = trial.suggest_float("lr_cam", 1e-4, 5e-3, log=True)
    lr_mot = trial.suggest_float("lr_mot", 1e-4, 5e-3, log=True)
    optimizer = torch.optim.AdamW([
        {"params": model.ego.parameters(), "lr": lr_cam},
        {"params": model.saliency.parameters(), "lr": lr_mot},
    ])
    lambdas = {name: trial.suggest_float(name, low, high, log=True) for name, low, high in [("lambda_sparse", 1e-4, 1e-1), ("lambda_smooth", 1e-5, 1e-2), ("lambda_reg", 1e-4, 1e-1)]}
    loss_type = trial.suggest_categorical("loss_type", ["L1", "SmoothL1", "Charbonnier"])
    stage1_ratio = trial.suggest_float("stage1_ratio", 0.1, 0.4)
    stage2_ratio = trial.suggest_float("stage2_ratio", 0.2, 0.5)
    joint_lr_scale = trial.suggest_float("joint_lr_scale", 0.01, 0.5, log=True)
    stage1_end, stage2_end = stage_boundaries(args.epochs, stage1_ratio, stage2_ratio)
    loader = DataLoader(MotionMemmapDataset(args.train_cache), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(MotionMemmapDataset(args.val_cache), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0, worker_init_fn=worker_init_fn)
    best = -100.0
    current_stage = 0
    for epoch in range(args.epochs):
        stage = 1 if epoch < stage1_end else (2 if epoch < stage2_end else 3)
        if stage != current_stage:
            set_stage(model, stage)
            if stage == 3:
                optimizer.param_groups[0]["lr"] = lr_cam * joint_lr_scale
                optimizer.param_groups[1]["lr"] = lr_mot * joint_lr_scale
            current_stage = stage
        for frames in tqdm(loader, desc=f"Trial {trial.number} stage {stage} epoch {epoch + 1}/{args.epochs}", unit="batch", leave=False):
            frames = frames.to(device, non_blocking=True)
            output = model(frames)
            terms = compute_loss(output, frames[:, 2:3], None, loss_type=loss_type, use_saliency=stage >= 2, **lambdas)
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
        scores = []
        collapsed_count = 0
        total_batches = 0
        with torch.no_grad():
            for frames in val_loader:
                output = model(frames.to(device, non_blocking=True))
                saliency = output["saliency"]
                score, bad = saliency_fitness(saliency)
                total_batches += 1
                if bad:
                    collapsed_count += 1
                else:
                    scores.append(score)
                if args.test_trial and total_batches == 1:
                    print("[SANITY] stage={} max={:.6f} mean={:.6f} score={:.6f}".format(stage, float(saliency.max()), float(saliency.mean()), float(score)), flush=True)
                if args.test_trial and total_batches >= args.test_val_batches:
                    break
        collapse_fraction = collapsed_count / max(1, total_batches)
        fitness = float(torch.stack(scores).mean()) if scores else -100.0
        collapsed = not scores or collapse_fraction > 0.5
        if stage == 1:
            print("[GPU]", args.gpu_id, "trial", trial.number, "epoch", epoch + 1, "/", args.epochs, "stage", stage, "ego_warmup", flush=True)
            continue
        if collapsed and not args.test_trial:
            raise optuna.TrialPruned()
        trial.report(fitness, epoch + 1)
        if trial.should_prune() and not args.test_trial:
            raise optuna.TrialPruned()
        if fitness > best:
            best = fitness
            if args.checkpoint_dir:
                checkpoint_dir = Path(args.checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "trial": trial.number, "fitness": fitness, "params": dict(trial.params)}, checkpoint_dir / f"trial_{trial.number:05d}_best.pt")
        print("[GPU]", args.gpu_id, "trial", trial.number, "epoch", epoch + 1, "/", args.epochs, "stage", stage, "fitness", fitness, "collapse_fraction", collapse_fraction, flush=True)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", "--gpu-id", type=int, required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--study-name", default="motion_disentangler")
    parser.add_argument("--storage", default="sqlite:///optuna_national_day.db?timeout=60")
    parser.add_argument("--n-trials", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--test-trial", action="store_true")
    parser.add_argument("--test-val-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    torch.cuda.set_device(0)
    if args.test_trial:
        args.n_trials = 1
        args.workers = 0
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.HyperbandPruner(min_resource=4, max_resource=max(4, args.epochs), reduction_factor=3)
    study = optuna.create_study(study_name=args.study_name, storage=args.storage, load_if_exists=True, direction="maximize", sampler=sampler, pruner=pruner)
    print(json.dumps({"study": args.study_name, "gpu_id": args.gpu_id, "n_trials": args.n_trials, "epochs": args.epochs}, ensure_ascii=False), flush=True)
    try:
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, gc_after_trial=True, show_progress_bar=False)
    finally:
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
