#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Self-supervised infrared background reconstruction with optional Optuna workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import cv2
import optuna
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.models.bg_refine_unet import BackgroundReconUNet


class TripletDataset(Dataset):
    def __init__(self, index_path: str, image_size: int = 256, input_mode: str = "context"):
        self.records = [json.loads(line) for line in Path(index_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.image_size = image_size
        self.input_mode = input_mode

    def __len__(self):
        return len(self.records)

    def _read(self, path: str):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Cannot decode frame: {path}")
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(image).float().div(255.0)

    def __getitem__(self, index):
        record = self.records[index]
        previous = self._read(record["prev"])
        center = self._read(record["center"])
        following = self._read(record["next"])
        if self.input_mode == "triplet":
            inputs = torch.stack([previous, center, following])
        else:
            inputs = torch.stack([previous, following])
        return inputs, center


def charbonnier(prediction, target, epsilon=1e-3):
    return torch.sqrt((prediction - target).pow(2) + epsilon**2).mean()


def sobel_loss(prediction, target):
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=prediction.device, dtype=prediction.dtype).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(2, 3)
    pred_x = F.conv2d(prediction, kernel_x, padding=1)
    pred_y = F.conv2d(prediction, kernel_y, padding=1)
    target_x = F.conv2d(target, kernel_x, padding=1)
    target_y = F.conv2d(target, kernel_y, padding=1)
    return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


def ssim_loss(prediction, target):
    window = 7
    mu_x = F.avg_pool2d(prediction, window, stride=1, padding=window // 2)
    mu_y = F.avg_pool2d(target, window, stride=1, padding=window // 2)
    sigma_x = F.avg_pool2d(prediction * prediction, window, stride=1, padding=window // 2) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, window, stride=1, padding=window // 2) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(prediction * target, window, stride=1, padding=window // 2) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-6)
    return (1 - ssim.clamp(-1, 1)).mean()


def losses(prediction, target, ssim_weight, edge_weight):
    return charbonnier(prediction, target) + ssim_weight * ssim_loss(prediction, target) + edge_weight * sobel_loss(prediction, target)


def evaluate(model, loader, device, ssim_weight, edge_weight):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for inputs, target in loader:
            prediction = model(inputs.to(device))
            target = target.to(device).unsqueeze(1)
            total += float(losses(prediction, target, ssim_weight, edge_weight))
            count += 1
    return total / max(count, 1)


def train_trial(args, params, trial_number=0, gpu_id=0):
    if torch.cuda.is_available() and args.device != "cpu":
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    dataset = TripletDataset(args.index, args.image_size, params["input_mode"])
    if len(dataset) < 4:
        raise ValueError(f"Triplet index too small: {len(dataset)}")
    val_size = max(1, round(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_set, batch_size=params["batch"], shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True)
    val_loader = DataLoader(val_set, batch_size=params["batch"], shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    in_channels = 3 if params["input_mode"] == "triplet" else 2
    model = BackgroundReconUNet(in_channels, params["base_channels"], params["depth"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, args.epochs), eta_min=params["lr"] * 0.05)
    best_val = float("inf")
    output = Path(args.output_root) / f"trial_{trial_number:04d}"
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for inputs, target in tqdm(train_loader, desc=f"trial {trial_number} epoch {epoch + 1} train", leave=False):
            inputs, target = inputs.to(device, non_blocking=True), target.to(device, non_blocking=True).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = losses(prediction, target, params["ssim_weight"], params["edge_weight"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.detach())
        scheduler.step()
        val_loss = evaluate(model, val_loader, device, params["ssim_weight"], params["edge_weight"])
        train_loss /= max(len(train_loader), 1)
        print(f"[TRIAL {trial_number}] epoch={epoch + 1}/{args.epochs} train={train_loss:.6f} val={val_loss:.6f} lr={scheduler.get_last_lr()[0]:.3e}", flush=True)
        torch.save({"model": model.state_dict(), "params": params, "epoch": epoch + 1, "val_loss": val_loss}, output / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "params": params, "epoch": epoch + 1, "val_loss": val_loss}, output / "best.pt")
    return best_val


def suggest_params(trial, args):
    return {
        "input_mode": trial.suggest_categorical("input_mode", ["context", "triplet"]),
        "base_channels": trial.suggest_categorical("base_channels", [16, 24, 32, 48]),
        "depth": trial.suggest_categorical("depth", [2, 3, 4]),
        "batch": trial.suggest_categorical("batch", [args.batch, max(1, args.batch // 2)]),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "ssim_weight": trial.suggest_float("ssim_weight", 0.05, 0.5),
        "edge_weight": trial.suggest_float("edge_weight", 0.02, 0.3),
    }


def worker(args):
    params = json.loads(args.params)
    return train_trial(args, params, args.trial_number, args.gpu_id)


def main():
    parser = argparse.ArgumentParser(description="Self-supervised infrared background reconstruction")
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-root", default="runs/selfsup_bg_unet")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--trial-number", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--params", default="")
    parser.add_argument("--optuna", action="store_true")
    parser.add_argument("--n-trials", type=int, default=4)
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    if args.params:
        train_trial(args, json.loads(args.params), args.trial_number, args.gpu_id)
        return
    if not args.optuna:
        params = {"input_mode": "context", "base_channels": 32, "depth": 3, "batch": args.batch, "lr": 1e-3, "weight_decay": 1e-4, "ssim_weight": 0.2, "edge_weight": 0.1}
        train_trial(args, params, 0, args.gpu_id)
        return
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(direction="minimize", study_name="selfsup_bg_unet", storage=f"sqlite:///{(output / 'study.db').resolve()}", load_if_exists=True)
    gpu_ids = [int(item) for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")
    next_trial = 0
    while next_trial < args.n_trials:
        active = []
        for gpu_id in gpu_ids:
            if next_trial >= args.n_trials:
                break
            trial = study.ask()
            params = suggest_params(trial, args)
            log_path = output / "logs" / f"trial_{trial.number:04d}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w", encoding="utf-8")
            command = [
                sys.executable, "-u", str(Path(__file__).resolve()), "--index", args.index,
                "--output-root", args.output_root, "--epochs", str(args.epochs),
                "--image-size", str(args.image_size), "--workers", str(args.workers),
                "--val-fraction", str(args.val_fraction), "--seed", str(args.seed),
                "--trial-number", str(trial.number), "--gpu-id", str(gpu_id),
                "--params", json.dumps(params),
            ]
            process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
            active.append((trial, process, log_handle))
            next_trial += 1
        for trial, process, log_handle in active:
            return_code = process.wait()
            log_handle.close()
            checkpoint = output / f"trial_{trial.number:04d}" / "best.pt"
            if return_code or not checkpoint.exists():
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                print(f"[FAIL] trial={trial.number} return_code={return_code}", flush=True)
            else:
                value = float(torch.load(checkpoint, map_location="cpu")["val_loss"])
                study.tell(trial, value)
                print(f"[DONE] trial={trial.number} val_loss={value:.6f}", flush=True)
    print(f"[SUCCESS] Best trial: {study.best_trial.number} value={study.best_value:.6f} params={study.best_trial.params}")


if __name__ == "__main__":
    main()
