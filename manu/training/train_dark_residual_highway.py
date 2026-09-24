#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, colorstr
from manu.models.dark_residual_highway_module import DarkResidualHighway
from manu.heatmap_evaluate import extract_peaks, find_best_f1_threshold
from manu.models.heatmap_loss import HeatmapLoss, generate_heatmaps_and_targets
from manu.models.heatmap_model import YOLO26HeatmapDetector


class DarkResidualIntegratedModel(nn.Module):
    def __init__(self, base_detector: YOLO26HeatmapDetector, highway: DarkResidualHighway):
        super().__init__()
        self.base_detector = base_detector
        self.highway = highway

    def forward(self, x_4ch: torch.Tensor) -> dict[str, torch.Tensor]:
        x_base = x_4ch[:, :3]
        feat = self.base_detector.extract_features(x_base)
        if self.base_detector.p0_highway is not None:
            feat = feat + self.base_detector.p0_highway(x_base)
        feat = feat + self.highway(x_4ch[:, 3:4])
        return self.base_detector.head(feat)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a frozen Trial 0474 dark residual highway")
    parser.add_argument("--data", type=str, default="/mnt/data/siping/datasets/manu/uav_gmc_median_signed/data.yaml")
    parser.add_argument("--weights", type=str, default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr0", type=float, default=0.0003)
    parser.add_argument("--gate-lr", type=float, default=1e-5)
    parser.add_argument("--scale-factor", type=float, default=0.05)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-beta", type=float, default=2.40)
    parser.add_argument("--offset-weight", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="0,1,2,3")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default="runs/dark_residual_highway")
    parser.add_argument("--name", type=str, default="exp_dark_residual_6ep")
    return parser.parse_args()


def load_base(weights_path: Path, stride: int):
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    p0_kwargs = ckpt.get(
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
    base = YOLO26HeatmapDetector(
        stride=ckpt.get("stride", stride),
        num_classes=1,
        temporal_mode="standard",
        use_p0_highway=True,
        p0_highway_kwargs=p0_kwargs,
    )
    own_state = base.state_dict()
    matched = 0
    for key, value in state_dict.items():
        clean_key = key.replace("module.", "").replace("model.model.", "").replace("model.", "")
        if clean_key in own_state and own_state[clean_key].shape == value.shape:
            own_state[clean_key].copy_(value)
            matched += 1
    for parameter in base.parameters():
        parameter.requires_grad = False
    base.eval()
    print(f"[INFO] Frozen Trial 0474 layers loaded: {matched}")
    return base, ckpt.get("stride", stride)


def evaluate(model, loader, device, stride, imgsz):
    predictions, ground_truths = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", dynamic_ncols=True):
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            preds = model(imgs)
            predictions.extend(
                extract_peaks(
                    heatmap=preds["heatmap"],
                    offset=preds["offset"],
                    stride=stride,
                    conf_thresh=0.08,
                    top_k=80,
                )
            )
            bboxes = batch["bboxes"].cpu().numpy()
            batch_idx = batch["batch_idx"].long().cpu().numpy()
            for index in range(imgs.shape[0]):
                ground_truths.append(bboxes[batch_idx == index])
    sizes = [(imgsz, imgsz)] * len(ground_truths)
    return find_best_f1_threshold(
        predictions_raw=predictions,
        gt_boxes_list=ground_truths,
        img_sizes=sizes,
        distance_threshold=8.0,
        thresholds=[0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40],
    )


def main():
    args = parse_args()
    device_ids = [int(value) for value in args.device.split(",") if value.strip().isdigit()]
    device = torch.device(f"cuda:{device_ids[0]}" if device_ids and torch.cuda.is_available() else "cpu")
    save_dir = Path(args.project) / args.name
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    data_dict = check_det_dataset(args.data)
    if data_dict.get("channels", 3) != 4:
        raise ValueError(f"Signed dataset must declare channels: 4, got {data_dict.get('channels', 3)}")
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data
    cfg.hsv_h = cfg.hsv_s = cfg.hsv_v = 0.0
    cfg.degrees = cfg.shear = cfg.perspective = 0.0
    cfg.translate = 0.08
    cfg.scale = 0.15
    cfg.fliplr = 0.5
    cfg.flipud = 0.0
    cfg.mosaic = 0.10
    cfg.mixup = cfg.copy_paste = 0.0
    train_dataset = build_yolo_dataset(cfg, data_dict["train"], batch=args.batch, data=data_dict, mode="train", stride=32)
    val_dataset = build_yolo_dataset(cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=32)
    total_batch = args.batch * max(1, len(device_ids))
    train_loader = build_dataloader(train_dataset, batch=total_batch, workers=args.workers, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch=total_batch, workers=args.workers, shuffle=False)
    print(f"[INFO] Samples: train={len(train_dataset):,}, val={len(val_dataset):,}, channels={data_dict['channels']}")

    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path
    base, stride = load_base(weights_path, args.stride)
    highway = DarkResidualHighway(out_channels=48, scale_factor=args.scale_factor)
    print(
        "[INFO] Channel contract: x[:, :3] = [(I-B)^+, |I-W(I2)|, I_t] (official model RGB order), "
        "x[:, 3] = (B-I)^+ dark residual"
    )
    model = DarkResidualIntegratedModel(base, highway).to(device)
    model_module = model
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    conv_params = [parameter for name, parameter in highway.named_parameters() if parameter.requires_grad and name != "gate"]
    gate_params = [highway.gate]
    trainable = conv_params + gate_params
    optimizer = torch.optim.AdamW(
        [
            {"params": conv_params, "lr": args.lr0, "weight_decay": args.weight_decay},
            {"params": gate_params, "lr": args.gate_lr, "weight_decay": 0.0},
        ]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: ((1 + math.cos(epoch * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    )
    scaler = GradScaler(enabled=device.type == "cuda")
    criterion = HeatmapLoss(hm_weight=1.0, offset_weight=args.offset_weight, focal_alpha=2.0, focal_beta=args.focal_beta)
    feat_shape = (args.imgsz // stride, args.imgsz // stride)
    print("[INFO] Running Epoch 0 zero-regression check before any optimizer step")
    baseline_metrics = evaluate(model, val_loader, device, stride, args.imgsz)
    print(
        f"[BASELINE] F1={baseline_metrics['f1']:.4f} R={baseline_metrics['recall']:.4f} "
        f"P={baseline_metrics['precision']:.4f} TP={baseline_metrics['tp']} FP={baseline_metrics['fp']} "
        f"alpha={highway.effective_alpha():.6f}"
    )
    if baseline_metrics["f1"] < 0.905:
        raise RuntimeError(
            f"Epoch 0 baseline F1={baseline_metrics['f1']:.4f} deviates from Trial 0474 SOTA 0.9064 "
            "(expect float-level reproduction on the official dataset extension); stop before training."
        )
    best_f1 = baseline_metrics["f1"]

    for epoch in range(args.epochs):
        model.train()
        model_module.base_detector.eval()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True):
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            bboxes = batch["bboxes"].to(device, non_blocking=True)
            batch_idx = batch["batch_idx"].to(device, non_blocking=True)
            targets = generate_heatmaps_and_targets(
                batch_bboxes=bboxes,
                batch_idx=batch_idx,
                batch_size=imgs.shape[0],
                feat_shape=feat_shape,
                stride=stride,
                min_radius=1,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                loss, _ = criterion(model(imgs), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        metrics = evaluate(model, val_loader, device, stride, args.imgsz)
        alpha = highway.effective_alpha()
        print(f"[SUMMARY] epoch={epoch + 1} alpha={alpha:.6f} F1={metrics['f1']:.4f} R={metrics['recall']:.4f} P={metrics['precision']:.4f}")
        if metrics["f1"] < baseline_metrics["f1"] - 0.01:
            print(
                f"[WARN] Residual branch caused F1 regression: {metrics['f1']:.4f} vs baseline "
                f"{baseline_metrics['f1']:.4f}; do not promote this checkpoint."
            )
        checkpoint = {"highway_state": highway.state_dict(), "metrics": metrics, "stride": stride, "channels": 4, "effective_alpha": alpha}
        torch.save(checkpoint, weights_dir / "last.pt")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(checkpoint, weights_dir / "best.pt")

    print(colorstr("green", f"[SUCCESS] Best F1: {best_f1:.4f}; highway weights: {weights_dir / 'best.pt'}"))


if __name__ == "__main__":
    main()
