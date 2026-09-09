#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Generate ultra-fast lightweight inference cache (.pkl) directly from the official GMC dataset:
/mnt/data/siping/datasets/manu/uav_gmc

Key advantages:
1. Pure YOLO DataLoader format: 100% strictly aligned with official validation sets (no coordinate drift).
2. Deep Extraction down to conf=0.02 (top_k=150) to preserve all sub-threshold weak pulses.
3. Computes and caches CFAR local texture variance per frame for dynamic spatial gating.
4. Output file: runs/gmc_eval/uav_gmc_fusion_cache.pkl (~45MB), enables instant grid searches (<2s per run).

Usage on Server:
    python manu/cache_gmc_inferences.py \
        --weights runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt \
        --data /mnt/data/siping/datasets/manu/uav_gmc/data.yaml \
        --output runs/gmc_eval/uav_gmc_fusion_cache.pkl \
        --device 0 \
        --batch 32
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
import time

import cv2
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
    parser = argparse.ArgumentParser(description="Generate Inference Cache from GMC Dataset")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt",
        help="Path to trial_0031 checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc/data.yaml",
        help="Path to uav_gmc data.yaml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/gmc_eval/uav_gmc_fusion_cache.pkl",
        help="Destination pickle path",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--conf-thresh", type=float, default=0.02, help="Deep harvest threshold (default: 0.02)")
    parser.add_argument("--top-k", type=int, default=80, help="Max candidate peaks to store per frame (default: 80)")
    parser.add_argument("--calc-variance", action="store_true", default=False, help="Compute 2D variance map (default: False to strictly keep file < 25MB)")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("   UAV Tiny Object Detection: Offline Inference Cacher for GMC Dataset")
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

    model = YOLO26HeatmapDetector(stride=stride, num_classes=1, temporal_mode="standard")
    model.load_state_dict(state_dict)
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
        for batch in tqdm(val_loader, desc="Inference & Caching"):
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

                var_map = None
                if getattr(args, "calc_variance", False):
                    ch0 = (imgs_tensor[b, 0] * 255.0).byte().cpu().numpy()
                    gray_f = ch0.astype(np.float32)
                    mean = cv2.blur(gray_f, (15, 15))
                    mean_sq = cv2.blur(gray_f**2, (15, 15))
                    var = np.maximum(mean_sq - mean**2, 0.0)
                    std_dev = np.sqrt(var)
                    var_map = np.clip((std_dev - 2.5) / 10.0, 0.0, 1.0).astype(np.float16)

                # 极限瘦身：只保留纯点坐标与得分（float16），31613 帧总大小严格控制在 15~25 MB
                pts_fp16 = peaks_list[b]["points"].astype(np.float16)
                scs_fp16 = peaks_list[b]["scores"].astype(np.float16)
                gt_fp16 = gt_pts.astype(np.float16)

                records.append({
                    "im_name": im_name,
                    "gt_pts": gt_fp16,
                    "pred_points": pts_fp16,
                    "pred_scores": scs_fp16,
                    "var_map": var_map,
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
