#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Generate ultra-fast lightweight inference cache (.pkl) directly from the official GMC Median dataset
using the new P0 Residual Highway enhanced model (best_recall.pt).

Key design:
1. Load champion P0 Residual Highway model weights (with use_p0_highway=True).
2. Deep extraction down to conf=0.02 (top_k=100) to capture all sub-threshold candidate pulses.
3. Strict YOLO DataLoader alignment with official validation set (/mnt/data/siping/datasets/manu/uav_gmc_median).
4. Save compact float16 detections (< 20MB) enabling instant multi-strategy track fusion sweeps (<2s per run).

Usage on Server:
    python manu/cache_p0_highway_inferences.py \
        --weights runs/p0_residual_highway/exp_p0_highway_8ep/weights/best_recall.pt \
        --data /mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml \
        --output runs/gmc_eval/uav_median_p0_highway_cache.pkl \
        --device 0 \
        --batch 32
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, colorstr
from manu.heatmap_evaluate import extract_peaks
from manu.heatmap_model import YOLO26HeatmapDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Inference Cache for P0 Residual Highway Model")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/p0_residual_highway/exp_p0_highway_8ep/weights/best_recall.pt",
        help="Path to P0 Highway checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml",
        help="Path to uav_gmc_median data.yaml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/gmc_eval/uav_median_p0_highway_cache.pkl",
        help="Destination pickle path",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--conf-thresh", type=float, default=0.02, help="Deep harvest threshold (default: 0.02)")
    parser.add_argument("--top-k", type=int, default=100, help="Max candidate peaks to store per frame (default: 100)")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("   UAV Tiny Object Detection: Offline Inference Cacher for P0 Residual Highway")
    print(f"   Checkpoint  : {args.weights}")
    print(f"   Dataset     : {args.data}")
    print(f"   Output PKL  : {args.output}")
    print(f"   Conf Gating : {args.conf_thresh:.2f} (top {args.top_k} peaks)")
    print("=" * 80)

    # 1. Resolve paths
    weights_path = Path(args.weights)
    if not weights_path.exists():
        alt = REPO_ROOT / args.weights
        if alt.exists():
            weights_path = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Load Model
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[INFO] Using Device: {device}")

    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    stride = ckpt.get("stride", args.stride)
    gate_val = ckpt.get("gate_alpha", None)

    # Instantiate model with use_p0_highway=True
    model = YOLO26HeatmapDetector(
        stride=stride,
        num_classes=1,
        temporal_mode="standard",
        use_p0_highway=True,
    )

    # Load trained state_dict directly
    matched, skipped = 0, 0
    own_state = model.state_dict()
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "").replace("model.model.", "").replace("model.", "")
        if clean_k in own_state and own_state[clean_k].shape == v.shape:
            own_state[clean_k].copy_(v)
            matched += 1
        else:
            skipped += 1

    actual_gate = model.p0_highway.gate.item() if hasattr(model, "p0_highway") and model.p0_highway is not None else 0.0
    print(f"[INFO] Loaded State Dict: {matched} layers matched, {skipped} skipped.")
    print(colorstr("cyan", f"[INFO] Active Gate α Value: {actual_gate:.6f} (Recorded in ckpt: {gate_val})"))

    model.to(device)
    model.eval()

    # 3. Setup DataLoader
    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data

    val_dataset = build_yolo_dataset(cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=32)
    val_loader = build_dataloader(val_dataset, batch=args.batch, workers=4, shuffle=False)
    print(colorstr("bold", f"[INFO] Processing {len(val_dataset)} images in official val split..."))

    records = []
    t0 = time.time()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="P0 Highway Inference & Caching"):
            imgs_tensor = batch["img"].to(device, non_blocking=True).float() / 255.0
            bboxes = batch["bboxes"]
            b_idx = batch["batch_idx"]
            im_files = batch.get("im_file", [""] * imgs_tensor.shape[0])
            bs = imgs_tensor.shape[0]

            preds = model(imgs_tensor)

            peaks_list = extract_peaks(
                heatmap=preds["heatmap"],
                offset=preds["offset"],
                stride=stride,
                conf_thresh=args.conf_thresh,
                top_k=args.top_k,
            )

            for b in range(bs):
                mask_b = b_idx == b
                gt_norm = bboxes[mask_b].cpu().numpy()
                im_name = Path(im_files[b]).name if im_files[b] else f"img_{b}"

                gt_pts = []
                for box in gt_norm:
                    gt_x = float(box[0] * args.imgsz)
                    gt_y = float(box[1] * args.imgsz)
                    gt_pts.append([gt_x, gt_y])
                gt_pts = np.array(gt_pts, dtype=np.float32) if len(gt_pts) > 0 else np.zeros((0, 2), dtype=np.float32)

                pts_fp16 = peaks_list[b]["points"].astype(np.float16)
                scs_fp16 = peaks_list[b]["scores"].astype(np.float16)
                gt_fp16 = gt_pts.astype(np.float16)

                records.append({
                    "im_name": im_name,
                    "gt_pts": gt_fp16,
                    "pred_points": pts_fp16,
                    "pred_scores": scs_fp16,
                })

    elapsed = time.time() - t0
    fps = len(records) / max(0.1, elapsed)
    print(f"\n[INFO] Inference finished in {elapsed:.1f}s ({fps:.1f} imgs/s).")

    print(f"[INFO] Writing {len(records)} records to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(colorstr("bold", colorstr("green", f"[SUCCESS] Inference cache saved ({file_size_mb:.1f} MB) -> {output_path.resolve()}\n")))


if __name__ == "__main__":
    main()
