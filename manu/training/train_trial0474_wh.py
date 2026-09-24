#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train a frozen Trial 0474 with a sparse CenterNet-style width-height head."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG
from manu.evaluation.heatmap_evaluate import evaluate_point_detections, extract_peaks
from manu.models.heatmap_model import YOLO26HeatmapDetector


class WhHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a sparse Wh head on frozen Trial 0474 features")
    parser.add_argument("--data", default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml")
    parser.add_argument("--weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr0", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--wh-loss", choices=["smoothl1", "l1"], default="smoothl1")
    parser.add_argument("--device", default="0,1,2,3")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    parser.add_argument("--sota-recall", type=float, default=0.8619)
    parser.add_argument("--sota-precision", type=float, default=0.9557)
    parser.add_argument("--sota-f1", type=float, default=0.9064)
    parser.add_argument("--hm-thresh", type=float, default=0.03)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--project", default="runs/trial0474_wh")
    parser.add_argument("--name", default="frozen_trial0474_wh")
    return parser.parse_args()


def resolve_weights(path: str) -> Path:
    candidates = [Path(path), PROJECT_ROOT / path, Path("/home/manu/mnt/pycharm_project_10ae9e2e") / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


def load_frozen_model(path: Path, device: torch.device, stride: int):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint.get("state_dict", checkpoint)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    p0_kwargs = checkpoint.get(
        "p0_kwargs",
        {
            "use_spatial_gate": True,
            "stem_type": "standard_dw",
            "downsample_mode": "pixel_unshuffle",
            "gate_input_mode": "diff_only",
            "gate_mid_channels": 16,
            "gate_depth": 2,
            "fusion_mode": "scalar_gate",
        },
    )
    model = YOLO26HeatmapDetector(
        stride=checkpoint.get("stride", stride),
        num_classes=1,
        temporal_mode="standard",
        use_p0_highway=True,
        p0_highway_kwargs=p0_kwargs,
    )
    own_state = model.state_dict()
    matched = 0
    for key, value in state_dict.items():
        clean_key = key.replace("module.", "").replace("model.model.", "").replace("model.", "")
        if clean_key in own_state and own_state[clean_key].shape == value.shape:
            own_state[clean_key].copy_(value)
            matched += 1
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, checkpoint.get("stride", stride), matched


def make_wh_targets(bboxes: torch.Tensor, batch_idx: torch.Tensor, batch_size: int, height: int, width: int, imgsz: int, device):
    targets = []
    for batch_number in range(batch_size):
        boxes = bboxes[batch_idx == batch_number]
        if boxes.numel() == 0:
            targets.append(torch.empty((0, 4), device=device))
            continue
        centers_x = torch.clamp((boxes[:, 0] * width).floor(), 0, width - 1).long()
        centers_y = torch.clamp((boxes[:, 1] * height).floor(), 0, height - 1).long()
        log_width = torch.log(torch.clamp(boxes[:, 2] * imgsz, min=1e-3))
        log_height = torch.log(torch.clamp(boxes[:, 3] * imgsz, min=1e-3))
        targets.append(torch.stack([centers_x.float(), centers_y.float(), log_width, log_height], dim=1))
    return targets


def sparse_wh_loss(wh_map: torch.Tensor, targets: list[torch.Tensor], loss_type: str):
    values = []
    for batch_number, target in enumerate(targets):
        if target.numel() == 0:
            continue
        x = target[:, 0].long()
        y = target[:, 1].long()
        prediction = wh_map[batch_number, :, y, x].transpose(0, 1)
        target_log_wh = target[:, 2:4]
        values.append(F.smooth_l1_loss(prediction, target_log_wh) if loss_type == "smoothl1" else F.l1_loss(prediction, target_log_wh))
    return torch.stack(values).mean() if values else wh_map.sum() * 0.0


def extract_boxes(heatmap: torch.Tensor, offset: torch.Tensor, wh_map: torch.Tensor, stride: int, threshold: float, top_k: int, imgsz: int):
    pooled = F.max_pool2d(heatmap, 3, 1, 1)
    keep = (heatmap == pooled) & (heatmap >= threshold)
    results = []
    for batch_number in range(heatmap.shape[0]):
        indices = torch.nonzero(keep[batch_number, 0], as_tuple=False)
        if indices.numel() == 0:
            results.append((np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)))
            continue
        scores = heatmap[batch_number, 0, indices[:, 0], indices[:, 1]]
        if len(scores) > top_k:
            scores, selected = torch.topk(scores, top_k)
            indices = indices[selected]
        y, x = indices[:, 0], indices[:, 1]
        centers_x = (x.float() + offset[batch_number, 0, y, x]) * stride
        centers_y = (y.float() + offset[batch_number, 1, y, x]) * stride
        widths = torch.exp(wh_map[batch_number, 0, y, x]).clamp(1.0, float(imgsz))
        heights = torch.exp(wh_map[batch_number, 1, y, x]).clamp(1.0, float(imgsz))
        boxes = torch.stack(
            [centers_x - widths / 2, centers_y - heights / 2, centers_x + widths / 2, centers_y + heights / 2], dim=1
        ).clamp(0, imgsz)
        results.append((boxes.detach().cpu().numpy(), scores.detach().cpu().numpy()))
    return results


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = np.prod(np.maximum(0.0, bottom_right - top_left), axis=2)
    area_a = np.prod(np.maximum(0.0, boxes_a[:, 2:] - boxes_a[:, :2]), axis=1)
    area_b = np.prod(np.maximum(0.0, boxes_b[:, 2:] - boxes_b[:, :2]), axis=1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-9)


def average_precision(predictions, ground_truths, iou_threshold: float):
    detections = []
    total_gt = sum(len(boxes) for boxes in ground_truths)
    for image_id, (boxes, scores) in enumerate(predictions):
        for box, score in zip(boxes, scores):
            detections.append((float(score), image_id, box))
    detections.sort(reverse=True, key=lambda item: item[0])
    matched = [np.zeros(len(boxes), dtype=bool) for boxes in ground_truths]
    true_positive = np.zeros(len(detections), dtype=np.float32)
    false_positive = np.zeros(len(detections), dtype=np.float32)
    for index, (_, image_id, box) in enumerate(detections):
        gt = ground_truths[image_id]
        if len(gt) == 0:
            false_positive[index] = 1
            continue
        overlaps = box_iou(box[None], gt)[0]
        best = int(np.argmax(overlaps))
        if overlaps[best] >= iou_threshold and not matched[image_id][best]:
            matched[image_id][best] = True
            true_positive[index] = 1
        else:
            false_positive[index] = 1
    if total_gt == 0 or len(detections) == 0:
        return 0.0
    precision = np.cumsum(true_positive) / np.maximum(np.cumsum(true_positive + false_positive), 1e-9)
    recall = np.cumsum(true_positive) / total_gt
    precision = np.concatenate(([1.0], precision, [0.0]))
    recall = np.concatenate(([0.0], recall, [1.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def dist_context():
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return distributed, rank, world_size, local_rank


def gather_objects(value, distributed):
    if not distributed:
        return value
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return [item for group in gathered for item in group]


def assert_frozen_unchanged(model, reference_state):
    for key, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), reference_state[key]):
            raise RuntimeError(f"Frozen Trial 0474 tensor changed: {key}")


def baseline_metrics(model, loader, device, stride, imgsz, dist_thresh, rank, distributed):
    model.eval()
    predictions, ground_truths = [], []
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, total=len(loader), desc="SOTA audit", leave=False, disable=rank != 0)):
            images = batch["img"].to(device, non_blocking=True).float() / 255.0
            frozen_predictions = model(images)
            predictions.extend(extract_peaks(frozen_predictions["heatmap"], frozen_predictions["offset"], stride, 0.03, 120))
            batch_idx = batch["batch_idx"].to(device)
            bboxes = batch["bboxes"].cpu().numpy()
            for image_number in range(images.shape[0]):
                mask = (batch_idx.cpu().numpy() == image_number)
                ground_truths.append(bboxes[mask])
    predictions = gather_objects(predictions, distributed)
    ground_truths = gather_objects(ground_truths, distributed)
    filtered = [{"points": item["points"][item["scores"] >= 0.25], "scores": item["scores"][item["scores"] >= 0.25]} for item in predictions]
    metrics = evaluate_point_detections(filtered, ground_truths, [(imgsz, imgsz)] * len(ground_truths), dist_thresh)
    return metrics


def run_epoch(model, wh_head, loader, device, stride, imgsz, loss_type, optimizer=None, hm_threshold=0.03, top_k=100):
    training = optimizer is not None
    wh_head.train(training)
    model.eval()
    total_loss = 0.0
    predictions, ground_truths = [], []
    width_errors, height_errors, relative_errors = [], [], []
    for batch_index, batch in enumerate(tqdm(loader, total=len(loader), desc="Train" if training else "Val", leave=False)):
        images = batch["img"].to(device, non_blocking=True).float() / 255.0
        bboxes = batch["bboxes"].to(device, non_blocking=True)
        batch_idx = batch["batch_idx"].to(device, non_blocking=True)
        with torch.no_grad():
            features = model.extract_features(images)
            frozen_predictions = model.head(features)
        wh_map = wh_head(features.detach())
        targets = make_wh_targets(bboxes, batch_idx, images.shape[0], wh_map.shape[2], wh_map.shape[3], imgsz, device)
        loss = sparse_wh_loss(wh_map, targets, loss_type)
        for batch_number, target in enumerate(targets):
            if target.numel() == 0:
                continue
            x = target[:, 0].long()
            y = target[:, 1].long()
            predicted_wh = torch.exp(wh_map[batch_number, :, y, x].transpose(0, 1)).detach().cpu().numpy()
            target_wh = torch.exp(target[:, 2:4]).detach().cpu().numpy()
            absolute_error = np.abs(predicted_wh - target_wh)
            width_errors.extend(absolute_error[:, 0].tolist())
            height_errors.extend(absolute_error[:, 1].tolist())
            relative_errors.extend((absolute_error / np.maximum(target_wh, 1e-6)).reshape(-1).tolist())
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu())
        boxes = extract_boxes(frozen_predictions["heatmap"], frozen_predictions["offset"], wh_map.detach(), stride, hm_threshold, top_k, imgsz)
        predictions.extend(boxes)
        for image_number in range(images.shape[0]):
            image_gt = bboxes[batch_idx == image_number].detach().cpu().numpy()
            gt_boxes = np.zeros((len(image_gt), 4), dtype=np.float32)
            if len(image_gt):
                gt_boxes[:, 0] = (image_gt[:, 0] - image_gt[:, 2] / 2) * imgsz
                gt_boxes[:, 1] = (image_gt[:, 1] - image_gt[:, 3] / 2) * imgsz
                gt_boxes[:, 2] = (image_gt[:, 0] + image_gt[:, 2] / 2) * imgsz
                gt_boxes[:, 3] = (image_gt[:, 1] + image_gt[:, 3] / 2) * imgsz
            ground_truths.append(gt_boxes)
    predictions = gather_objects(predictions, dist.is_initialized())
    ground_truths = gather_objects(ground_truths, dist.is_initialized())
    if dist.is_initialized():
        width_errors = [item for group in gather_objects(width_errors, True) for item in group]
        height_errors = [item for group in gather_objects(height_errors, True) for item in group]
        relative_errors = [item for group in gather_objects(relative_errors, True) for item in group]
    loss_tensor = torch.tensor(total_loss / max(1, len(loader)), device=device)
    if dist.is_initialized():
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= dist.get_world_size()
    map50_95 = float(np.mean([average_precision(predictions, ground_truths, threshold) for threshold in np.arange(0.50, 0.951, 0.05)]))
    error_values = np.asarray(width_errors + height_errors, dtype=np.float32)
    relative_values = np.asarray(relative_errors, dtype=np.float32)
    return {
        "loss": float(loss_tensor.cpu()),
        "map50_95": map50_95,
        "width_mae": float(np.mean(width_errors)) if width_errors else 0.0,
        "height_mae": float(np.mean(height_errors)) if height_errors else 0.0,
        "size_abs_median": float(np.median(error_values)) if error_values.size else 0.0,
        "size_rel_median": float(np.median(relative_values)) if relative_values.size else 0.0,
        "size_rel_p10": float(np.mean(relative_values <= 0.10)) if relative_values.size else 0.0,
        "size_rel_p25": float(np.mean(relative_values <= 0.25)) if relative_values.size else 0.0,
    }


def main():
    args = parse_args()
    distributed, rank, world_size, local_rank = dist_context()
    if torch.cuda.is_available() and args.device != "cpu":
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device(f"cuda:{args.device.split(',')[0]}")
    else:
        device = torch.device("cpu")
    model, stride, matched = load_frozen_model(resolve_weights(args.weights), device, args.stride)
    reference_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    feature_channels = model.head.feat_conv[0].conv.in_channels
    wh_head = WhHead(feature_channels, args.hidden_channels).to(device)
    if distributed:
        wh_head = DistributedDataParallel(wh_head, device_ids=[local_rank] if device.type == "cuda" else None)
    optimizer = torch.optim.AdamW(wh_head.parameters(), lr=args.lr0, weight_decay=args.weight_decay)
    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data
    train_dataset = build_yolo_dataset(cfg, data_dict["train"], batch=args.batch, data=data_dict, mode="train", stride=32)
    val_dataset = build_yolo_dataset(cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=32)
    train_loader = build_dataloader(train_dataset, batch=args.batch, workers=args.workers, shuffle=True, rank=rank if distributed else -1, device=device)
    val_loader = build_dataloader(val_dataset, batch=args.batch, workers=args.workers, shuffle=False, rank=rank if distributed else -1, device=device)
    output_dir = Path(args.project) / args.name
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    if rank == 0:
        print(f"[INFO] DDP world_size={world_size}; Frozen Trial 0474 tensors matched: {matched}")
        print(f"[INFO] Trainable Wh parameters: {sum(parameter.numel() for parameter in wh_head.parameters())}")
    initial = baseline_metrics(model, val_loader, device, stride, args.imgsz, args.dist_thresh, rank, distributed)
    audit_ok = torch.ones(1, device=device, dtype=torch.int32)
    if rank == 0:
        print(f"[SOTA AUDIT][Epoch 0] Recall={initial['recall']:.6f} Precision={initial['precision']:.6f} F1={initial['f1']:.6f} TP={initial['tp']} FP={initial['fp']} GT={initial['total_gt']}")
        if any(abs(initial[key] - expected) > 1e-4 for key, expected in (("recall", args.sota_recall), ("precision", args.sota_precision), ("f1", args.sota_f1))):
            audit_ok[0] = 0
    if distributed:
        dist.broadcast(audit_ok, src=0)
    if not bool(audit_ok.item()):
        raise RuntimeError("Trial 0474 Epoch 0 SOTA audit failed; stop before Wh training")
    if distributed:
        dist.barrier()
    best_map50_95 = -1.0
    for epoch in range(args.epochs):
        if distributed and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)
        train_metrics = run_epoch(model, wh_head, train_loader, device, stride, args.imgsz, args.wh_loss, optimizer, args.hm_thresh, args.top_k)
        val_metrics = run_epoch(model, wh_head, val_loader, device, stride, args.imgsz, args.wh_loss, None, args.hm_thresh, args.top_k)
        if rank == 0:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | train_wh={train_metrics['loss']:.5f} | "
                f"val_wh={val_metrics['loss']:.5f} | mAP50-95={val_metrics['map50_95']:.4f} | "
                f"W_MAE={val_metrics['width_mae']:.2f}px | H_MAE={val_metrics['height_mae']:.2f}px | "
                f"AbsMed={val_metrics['size_abs_median']:.2f}px | RelMed={val_metrics['size_rel_median'] * 100:.1f}% | "
                f"Rel<=10%={val_metrics['size_rel_p10'] * 100:.1f}% | Rel<=25%={val_metrics['size_rel_p25'] * 100:.1f}%"
            )
            checkpoint = {
                "wh_head": wh_head.module.state_dict() if distributed else wh_head.state_dict(),
                "epoch": epoch + 1,
                "stride": stride,
                "imgsz": args.imgsz,
                "base_weights": str(resolve_weights(args.weights)),
                "val_map50_95": val_metrics["map50_95"],
                "val_width_mae": val_metrics["width_mae"],
                "val_height_mae": val_metrics["height_mae"],
            }
            torch.save(checkpoint, output_dir / "last.pt")
            if val_metrics["map50_95"] > best_map50_95:
                best_map50_95 = val_metrics["map50_95"]
                torch.save(checkpoint, output_dir / "best.pt")
        if distributed:
            dist.barrier()
    assert_frozen_unchanged(model, reference_state)
    final = baseline_metrics(model, val_loader, device, stride, args.imgsz, args.dist_thresh, rank, distributed)
    final_ok = torch.ones(1, device=device, dtype=torch.int32)
    if rank == 0:
        print(f"[SOTA AUDIT][Final] Recall={final['recall']:.6f} Precision={final['precision']:.6f} F1={final['f1']:.6f} TP={final['tp']} FP={final['fp']} GT={final['total_gt']}")
        if any(abs(final[key] - expected) > 1e-4 for key, expected in (("recall", args.sota_recall), ("precision", args.sota_precision), ("f1", args.sota_f1))):
            final_ok[0] = 0
    if distributed:
        dist.broadcast(final_ok, src=0)
    if not bool(final_ok.item()):
        raise RuntimeError("Trial 0474 final SOTA audit failed; Wh branch changed frozen baseline behavior")
    if rank == 0:
        print(f"[SUCCESS] Outputs saved to {output_dir}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
