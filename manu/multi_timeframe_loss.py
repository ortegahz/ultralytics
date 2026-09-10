#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Multi-Timeframe Heatmap Target Generator & Pure Focal Loss.

Strict Single-Variable Isolation (Ablation-Pure):
1. Completely removes Soft-IoU loss (Zero Soft-IoU interference).
2. Uses purely CenterNet-style Focal Loss with beta=2.4 on 3-channel targets:
   - Channel 0: Current frame t (Gaussian peak at (x_curr, y_curr))
   - Channel 1: Past frame t-1 (Gaussian peak at GMC-aligned (x_past, y_past))
   - Channel 2: Future frame t+1 (Gaussian peak at GMC-aligned (x_fut, y_fut))
3. Dynamically reads companion mth_labels/{split}/*.txt if present, or falls back to center point.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

from manu.heatmap_loss import draw_gaussian, FocalLoss, RegL1Loss


def load_mth_label_for_image(im_file: str) -> tuple[float, float, float, float] | None:
    """
    Attempts to read (x_past, y_past, x_fut, y_fut) from companion mth_labels folder:
    images/{split}/{stem}.jpg -> mth_labels/{split}/{stem}.txt
    """
    p = Path(im_file)
    parts = list(p.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "mth_labels"
            break
    mth_file = Path(*parts).with_suffix(".txt")
    if mth_file.exists():
        try:
            content = mth_file.read_text(encoding="utf-8").strip()
            if content:
                row = content.splitlines()[0].split()
                if len(row) >= 9:
                    # class_id, x_curr, y_curr, w, h, x_past, y_past, x_fut, y_fut, ...
                    return float(row[5]), float(row[6]), float(row[7]), float(row[8])
        except Exception:
            pass
    return None


def generate_mth_heatmaps_and_targets(
    batch_bboxes: torch.Tensor,
    batch_idx: torch.Tensor,
    batch_size: int,
    feat_shape: tuple[int, int],
    stride: int,
    im_files: list[str] | None = None,
    min_radius: int = 1,
    max_objects: int = 64,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """
    Generates 3-channel ground-truth heatmaps:
    - Channel 0: current frame (cx, cy)
    - Channel 1: past frame (cx_past, cy_past) with GMC alignment
    - Channel 2: future frame (cx_fut, cy_fut) with GMC alignment
    """
    feat_h, feat_w = feat_shape
    if device is None:
        device = batch_bboxes.device

    heatmap_3ch = torch.zeros((batch_size, 3, feat_h, feat_w), dtype=torch.float32, device=device)
    gt_offsets = torch.zeros((batch_size, max_objects, 2), dtype=torch.float32, device=device)
    indices = torch.zeros((batch_size, max_objects), dtype=torch.int64, device=device)
    mask = torch.zeros((batch_size, max_objects), dtype=torch.bool, device=device)

    for b in range(batch_size):
        obj_mask = batch_idx == b
        boxes = batch_bboxes[obj_mask]
        num_objs = min(boxes.shape[0], max_objects)

        mth_coords = None
        if im_files is not None and b < len(im_files) and im_files[b]:
            mth_coords = load_mth_label_for_image(im_files[b])

        for i in range(num_objs):
            box = boxes[i]
            # Center frame coords in feature pixel space
            cx = float(box[0] * (feat_w * stride))
            cy = float(box[1] * (feat_h * stride))

            cx_feat = cx / stride
            cy_feat = cy / stride
            ct_int = [int(cx_feat), int(cy_feat)]

            # Draw center Gaussian (Channel 0)
            draw_gaussian(heatmap_3ch[b, 0], ct_int, radius=min_radius)

            if mth_coords is not None:
                cx_past_feat = (mth_coords[0] * (feat_w * stride)) / stride
                cy_past_feat = (mth_coords[1] * (feat_h * stride)) / stride
                ct_past = [int(cx_past_feat), int(cy_past_feat)]
                draw_gaussian(heatmap_3ch[b, 1], ct_past, radius=min_radius)

                cx_fut_feat = (mth_coords[2] * (feat_w * stride)) / stride
                cy_fut_feat = (mth_coords[3] * (feat_h * stride)) / stride
                ct_fut = [int(cx_fut_feat), int(cy_fut_feat)]
                draw_gaussian(heatmap_3ch[b, 2], ct_fut, radius=min_radius)
            else:
                # Replicate center to past and future if no extended coords
                draw_gaussian(heatmap_3ch[b, 1], ct_int, radius=min_radius)
                draw_gaussian(heatmap_3ch[b, 2], ct_int, radius=min_radius)

            # Sub-pixel offset regression on center frame t
            ct_x = min(max(ct_int[0], 0), feat_w - 1)
            ct_y = min(max(ct_int[1], 0), feat_h - 1)
            gt_offsets[b, i, 0] = cx_feat - ct_x
            gt_offsets[b, i, 1] = cy_feat - ct_y
            indices[b, i] = ct_y * feat_w + ct_x
            mask[b, i] = True

    return {
        "heatmap": heatmap_3ch,
        "offset": gt_offsets,
        "indices": indices,
        "mask": mask,
    }


class PureMultiTimeframeLoss(nn.Module):
    """
    Pure Multi-Timeframe Loss (CenterNet-style with Softened Focal beta=2.4):
    Loss = hm_weight * (L_focal_t + temporal_weight * [L_focal_{t-1} + L_focal_{t+1}])
         + offset_weight * RegL1Loss
    """

    def __init__(
        self,
        hm_weight: float = 1.0,
        offset_weight: float = 0.45,
        temporal_weight: float = 0.35,
        focal_alpha: float = 2.0,
        focal_beta: float = 2.4,
    ):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, beta=focal_beta)
        self.offset_loss = RegL1Loss()
        self.hm_weight = hm_weight
        self.offset_weight = offset_weight
        self.temporal_weight = temporal_weight

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        pred_hm = preds["heatmap"]
        gt_hm = targets["heatmap"]

        # Center frame loss
        loss_hm_center = self.focal_loss(pred_hm[:, 0:1], gt_hm[:, 0:1])

        # Auxiliary past and future temporal tube loss
        loss_past = self.focal_loss(pred_hm[:, 1:2], gt_hm[:, 1:2])
        loss_future = self.focal_loss(pred_hm[:, 2:3], gt_hm[:, 2:3])
        loss_hm_temporal = (loss_past + loss_future) * 0.5

        # Sub-pixel offset loss
        loss_offset = self.offset_loss(
            preds["offset"],
            targets["offset"],
            targets["indices"],
            targets["mask"],
        )

        total_loss = (
            self.hm_weight * (loss_hm_center + self.temporal_weight * loss_hm_temporal)
            + self.offset_weight * loss_offset
        )

        return {
            "loss": total_loss,
            "loss_hm": loss_hm_center,
            "loss_temporal": loss_hm_temporal,
            "loss_offset": loss_offset,
        }
