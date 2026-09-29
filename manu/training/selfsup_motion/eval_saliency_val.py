#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Evaluate unsupervised saliency peaks against official Anti-UAV validation labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_cache import MotionMemmapDataset, worker_init_fn
from disentangle_model import MotionDisentangler


def extract_candidates(saliency, low_threshold, top_k):
    pooled = F.max_pool2d(saliency, 3, 1, 1)
    mask = (saliency == pooled) & (saliency >= low_threshold)
    batches = []
    for item in range(saliency.shape[0]):
        points = torch.nonzero(mask[item, 0], as_tuple=False)
        if len(points) == 0:
            batches.append((np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)))
            continue
        scores = saliency[item, 0, points[:, 0], points[:, 1]]
        if len(points) > top_k:
            keep = torch.topk(scores, top_k).indices
            points, scores = points[keep], scores[keep]
        batches.append((points[:, [1, 0]].cpu().numpy().astype(np.float32), scores.cpu().numpy().astype(np.float32)))
    return batches


def match(pred, gt, distance):
    used = set()
    tp = 0
    for point in pred:
        best_dist, best_index = None, None
        for index, target in enumerate(gt):
            if index in used:
                continue
            dist = float(np.linalg.norm(point - target))
            if best_dist is None or dist < best_dist:
                best_dist, best_index = dist, index
        if best_dist is not None and best_dist <= distance:
            used.add(best_index)
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def metrics(tp, fp, fn):
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-7, precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--study", default="")
    parser.add_argument("--storage", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--distance", type=float, default=8.0)
    parser.add_argument("--low-threshold", type=float, default=0.02)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--visualize-dir", default="")
    parser.add_argument("--max-visualizations", type=int, default=24)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    if not args.checkpoint:
        if not args.checkpoint_dir or not args.study or not args.storage:
            raise ValueError("Provide --checkpoint or --checkpoint-dir with --study and --storage")
        import optuna

        study = optuna.load_study(study_name=args.study, storage=args.storage)
        args.checkpoint = str(Path(args.checkpoint_dir) / f"trial_{study.best_trial.number:05d}_best.pt")
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    params = checkpoint["params"]
    model = MotionDisentangler(params["base_channels"], params["saliency_kernel_size"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = MotionMemmapDataset(args.val_cache)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, worker_init_fn=worker_init_fn)
    records = [json.loads(line) for line in (Path(args.val_cache) / "records.jsonl").read_text(encoding="utf-8").splitlines() if line]
    thresholds = [round(x, 3) for x in np.arange(0.02, 0.61, 0.02)]
    stats = {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in thresholds}
    visualize_pending = []
    with torch.no_grad():
        for batch_index, frames in enumerate(tqdm(loader, desc="Evaluating Anti-UAV val", unit="batch")):
            output = model(frames.to(device, non_blocking=True))
            candidates = extract_candidates(output["saliency"], args.low_threshold, args.top_k)
            for offset, (points, scores) in enumerate(candidates):
                record = records[batch_index * args.batch_size + offset]
                scale_x = 640.0 / record["native_width"]
                scale_y = 512.0 / record["native_height"]
                gt = np.asarray([[box[0] * record["native_width"] * scale_x, box[1] * record["native_height"] * scale_y] for box in record.get("gt", [])], dtype=np.float32).reshape(-1, 2)
                for threshold in thresholds:
                    selected = points[scores >= threshold]
                    tp, fp, fn = match(selected, gt, args.distance)
                    stats[threshold]["tp"] += tp
                    stats[threshold]["fp"] += fp
                    stats[threshold]["fn"] += fn
                if args.visualize_dir and len(visualize_pending) < args.max_visualizations and len(gt):
                    visualize_pending.append((points, scores, gt, record["sequence"], frames[offset, 2].numpy(), output["saliency"][offset, 0].cpu().numpy()))
    rows = [{"threshold": threshold, **metrics(**stats[threshold])} for threshold in thresholds]
    best = max(rows, key=lambda row: row["f1"])
    if args.visualize_dir and visualize_pending:
        Path(args.visualize_dir).mkdir(parents=True, exist_ok=True)
        for index, (points, scores, gt, sequence, gray, saliency) in enumerate(visualize_pending):
            selected = points[scores >= best["threshold"]]
            heat = cv2.applyColorMap((saliency * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            panel = cv2.addWeighted(cv2.cvtColor((gray * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), 0.65, heat, 0.35, 0)
            for x, y in gt.astype(int):
                cv2.drawMarker(panel, (int(x), int(y)), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
            for x, y in selected.astype(int):
                cv2.circle(panel, (int(x), int(y)), 4, (0, 0, 255), 1)
            cv2.imwrite(str(Path(args.visualize_dir) / f"{index:05d}_{sequence}.jpg"), panel)
    result = {"checkpoint": args.checkpoint, "distance": args.distance, "frames": len(dataset), "best": best, "sweep": rows, "checkpoint_fitness": checkpoint.get("fitness"), "trial": checkpoint.get("trial")}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "checkpoint_fitness": checkpoint.get("fitness"), "frames": len(dataset)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
