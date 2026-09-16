#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Generate ultra-lightweight YOLO26 Bbox expert cache (.pkl) for Phase-A Size-Gated Dual-Expert Fusion.

Purpose:
    The Heatmap expert (Trial 0474) is a point-impulse specialist that fails on large UAV bodies
    overlapping strong building edges (Hard Case 5, Frame #750~#1250). A standard YOLO26 Bbox
    model (e.g. the original bbox-paradigm trial_0028) detects these large bodies natively.
    This script extracts compact bbox candidate caches to be merged by
    `eval_bidirectional_track_fusion.py --bbox-cache ...` (size-gated, conf-gated, deduped).

Key design:
1. Official YOLO DataLoader alignment (zero Letterbox/color drift, identical frame order to heatmap cache).
2. Deep harvest at conf=0.01, top_k=50 boxes per frame.
3. Compact float16 storage: cx, cy, w, h (pixel coords, 640 basis) + scores. Expected size < 10MB.

Usage on Server:
    python manu/cache_yolo26_bbox_inferences.py \
        --weights runs/optuna_uav_recall_sgpu/trial_0028/weights/best.pt \
        --data /mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml \
        --output runs/gmc_eval/uav_median_bbox_trial0028_cache.pkl \
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

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, colorstr


def parse_args():
    parser = argparse.ArgumentParser(description="Generate YOLO26 Bbox Expert Cache for Dual-Expert Fusion")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/optuna_uav_recall_sgpu/trial_0028/weights/best.pt",
        help="Path to YOLO26 Bbox paradigm checkpoint (bbox-paradigm trial_0028 recommended)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml",
        help="Path to data.yaml (MUST be the same val split as the heatmap cache for frame alignment)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/gmc_eval/uav_median_bbox_trial0028_cache.pkl",
        help="Destination pickle path",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--conf-thresh", type=float, default=0.01, help="Deep harvest confidence (default: 0.01)")
    parser.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold (default: 0.70)")
    parser.add_argument("--top-k", type=int, default=50, help="Max boxes stored per frame (default: 50)")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("   UAV Tiny Object Detection: YOLO26 Bbox Expert Cache Builder (Phase-A Dual-Expert)")
    print(f"   Checkpoint  : {args.weights}")
    print(f"   Dataset     : {args.data}")
    print(f"   Output PKL  : {args.output}")
    print(f"   Conf Gating : {args.conf_thresh:.2f} (top {args.top_k}, iou={args.iou:.2f})")
    print("=" * 80)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        alt = REPO_ROOT / args.weights
        if alt.exists():
            weights_path = alt
        else:
            raise FileNotFoundError(f"Bbox checkpoint not found: {args.weights}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    print(f"[INFO] Using Device: {device}")

    model = YOLO(str(weights_path))

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
        for batch in tqdm(val_loader, desc="YOLO26 Bbox Inference & Caching"):
            imgs_tensor = batch["img"].to(device, non_blocking=True).float() / 255.0
            im_files = batch.get("im_file", [""] * imgs_tensor.shape[0])
            bs = imgs_tensor.shape[0]

            results = model.predict(imgs_tensor, conf=args.conf_thresh, iou=args.iou, verbose=False, device=device)

            for b in range(bs):
                res = results[b]
                if len(res.boxes) > 0:
                    boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
                    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0
                    w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
                    h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
                    boxes_cxcywh = np.stack([cx, cy, w, h], axis=1)
                    # Keep top-k by confidence
                    if len(boxes_cxcywh) > args.top_k:
                        order = np.argsort(-confs)[: args.top_k]
                        boxes_cxcywh = boxes_cxcywh[order]
                        confs = confs[order]
                else:
                    boxes_cxcywh = np.zeros((0, 4), dtype=np.float32)
                    confs = np.zeros((0,), dtype=np.float32)

                im_name = Path(im_files[b]).name if im_files[b] else f"img_{b}"

                records.append({
                    "im_name": im_name,
                    "pred_boxes": boxes_cxcywh.astype(np.float16),
                    "box_scores": confs.astype(np.float16),
                })

    elapsed = time.time() - t0
    fps = len(records) / max(0.1, elapsed)
    print(f"\n[INFO] Inference finished in {elapsed:.1f}s ({fps:.1f} imgs/s).")

    print(f"[INFO] Writing {len(records)} records to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(colorstr("bold", colorstr("green", f"[SUCCESS] Bbox expert cache saved ({file_size_mb:.1f} MB) -> {output_path.resolve()}\n")))


if __name__ == "__main__":
    main()
