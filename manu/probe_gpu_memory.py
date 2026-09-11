#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Quick GPU Memory & Throughput Probe for YOLO26s (Stride=2 P1 High-Res 320x320 Heatmap).

Purpose:
Test exact GPU VRAM footprint for forward + backward pass under different batch sizes
(e.g., batch=16, 24, 32 per GPU) before starting full training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time

import torch
from torch.cuda.amp import autocast, GradScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manu.heatmap_model import YOLO26HeatmapDetector
from manu.heatmap_loss import HeatmapLoss, generate_heatmaps_and_targets


def probe_vram(device_id: int = 0, batch_size: int = 16, imgsz: int = 640, stride: int = 2):
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_id)

    print("=" * 80)
    print(f"[PROBE] Target: YOLO26s Heatmap (Stride={stride} P1, {imgsz // stride}x{imgsz // stride} map)")
    print(f"[PROBE] GPU: {torch.cuda.get_device_name(device_id)} | Device ID: {device_id}")
    print(f"[PROBE] Batch Size per GPU: {batch_size} | Input: ({batch_size}, 3, {imgsz}, {imgsz})")
    print("=" * 80)

    # 1. Instantiate Model
    t0 = time.time()
    model = YOLO26HeatmapDetector(
        stride=stride,
        scale="s",
        num_classes=1,
    ).to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[1/4] Model created in {time.time() - t0:.2f}s | Params: {total_params:,}")
    mem_model = torch.cuda.memory_allocated(device_id) / 1024**2
    print(f"      Static Model VRAM: {mem_model:.1f} MB")

    # 2. Setup Loss & Optimizer
    criterion = HeatmapLoss(hm_weight=1.0, offset_weight=0.5, focal_alpha=2.0, focal_beta=2.4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    # 3. Simulate Inputs & Heatmap Targets
    feat_dim = imgsz // stride
    x = torch.randn(batch_size, 3, imgsz, imgsz, device=device)
    # Simulate 5 random bounding boxes per image
    bboxes = torch.rand(batch_size * 5, 4, device=device)
    batch_idx = torch.repeat_interleave(torch.arange(batch_size, device=device), 5)

    targets = generate_heatmaps_and_targets(
        batch_bboxes=bboxes,
        batch_idx=batch_idx,
        batch_size=batch_size,
        feat_shape=(feat_dim, feat_dim),
        stride=stride,
        device=device,
    )

    print(f"[2/4] Simulating Forward + Backward step (Autocast FP16)...")
    torch.cuda.synchronize(device_id)
    t_step0 = time.time()

    # 4. Forward + Backward test
    optimizer.zero_grad()
    with autocast():
        preds = model(x)
        loss, _ = criterion(preds, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device_id)
    step_time = (time.time() - t_step0) * 1000.0

    peak_mem = torch.cuda.max_memory_allocated(device_id) / 1024**2
    total_mem = torch.cuda.get_device_properties(device_id).total_memory / 1024**2
    mem_percent = (peak_mem / total_mem) * 100.0

    print(f"[3/4] Step completed in {step_time:.1f} ms ({batch_size / (step_time / 1000.0):.1f} img/s)")
    print(f"[4/4] Peak VRAM Allocated: {peak_mem:.1f} MB / {total_mem:.1f} MB ({mem_percent:.1f}%)")
    print("=" * 80)

    if peak_mem < total_mem * 0.85:
        print(f"✅ [VERDICT] Batch size {batch_size} is completely SAFE and FEASIBLE on this GPU!")
    else:
        print(f"⚠️ [VERDICT] Peak VRAM exceeds 85%. Consider lowering batch size to avoid OOM.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16, help="Batch size per GPU to test")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--stride", type=int, default=2, help="Feature stride (2 for P1 high-res)")
    args = parser.parse_args()
    probe_vram(device_id=args.device, batch_size=args.batch, stride=args.stride)
