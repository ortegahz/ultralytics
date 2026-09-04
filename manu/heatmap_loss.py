#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Online Gaussian Heatmap generator and Loss functions for UAV tiny/point object detection.
Supports:
1. Online Gaussian heatmap construction with min_radius protection (for 3x3 point targets).
2. Modified Focal Loss (CenterNet style) with configurable focus for high recall.
3. L1 / Smooth-L1 Loss for sub-pixel offset regression.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian2d(shape: tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    """Generate a 2D Gaussian kernel."""
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(
    heatmap: torch.Tensor,
    center: tuple[int, int] | list[int],
    radius: int,
    k: float = 1.0,
) -> torch.Tensor:
    """
    Draw a 2D Gaussian circle onto a heatmap tensor in-place.
    
    Args:
        heatmap: Tensor of shape (H, W)
        center: (x, y) coordinates on the feature map
        radius: Gaussian kernel radius
        k: Peak value scale (default 1.0)
    """
    diameter = 2 * radius + 1
    # Standard rule of thumb: 3*sigma = radius => sigma = diameter / 6
    sigma = diameter / 6.0
    gaussian = gaussian2d((diameter, diameter), sigma=sigma)
    gaussian = torch.from_numpy(gaussian).to(device=heatmap.device, dtype=heatmap.dtype)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0], heatmap.shape[1]

    left = min(x, radius)
    right = min(width - x - 1, radius)
    top = min(y, radius)
    bottom = min(height - y - 1, radius)

    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return heatmap

    masked_heatmap = heatmap[y - top : y + bottom + 1, x - left : x + right + 1]
    masked_gaussian = gaussian[radius - top : radius + bottom + 1, radius - left : radius + right + 1]

    if masked_gaussian.numel() > 0 and masked_heatmap.numel() > 0:
        torch.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


def generate_heatmaps_and_targets(
    batch_bboxes: torch.Tensor,
    batch_idx: torch.Tensor,
    batch_size: int,
    feat_shape: tuple[int, int],
    stride: int,
    min_radius: int = 1,
    max_objects: int = 128,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """
    Convert normalized bounding boxes from YOLO dataloader to ground truth heatmaps and offset maps.
    
    Args:
        batch_bboxes: (N, 4) tensor in cx, cy, w, h format (normalized to [0, 1])
        batch_idx: (N,) tensor indicating which image index each box belongs to
        batch_size: batch size B
        feat_shape: (H_feat, W_feat)
        stride: downsampling stride (e.g. 4 for P2)
        min_radius: guaranteed minimum Gaussian radius (crucial for 3x3 tiny targets)
        max_objects: max number of point targets tracked per image for offset regression
    """
    H, W = feat_shape
    if device is None:
        device = batch_bboxes.device

    # Initialize outputs
    gt_heatmap = torch.zeros((batch_size, 1, H, W), dtype=torch.float32, device=device)
    gt_offset = torch.zeros((batch_size, max_objects, 2), dtype=torch.float32, device=device)
    gt_ind = torch.zeros((batch_size, max_objects), dtype=torch.long, device=device)
    gt_mask = torch.zeros((batch_size, max_objects), dtype=torch.bool, device=device)

    if batch_bboxes.numel() == 0:
        return {
            "heatmap": gt_heatmap,
            "offset": gt_offset,
            "ind": gt_ind,
            "mask": gt_mask,
        }

    for b in range(batch_size):
        mask_b = batch_idx == b
        if not mask_b.any():
            continue

        boxes = batch_bboxes[mask_b]  # (M, 4): cx, cy, w, h normalized
        # Convert cx, cy to feature map coordinates
        cx_feat = boxes[:, 0] * W
        cy_feat = boxes[:, 1] * H
        w_feat = boxes[:, 2] * W
        h_feat = boxes[:, 3] * H

        num_objs = min(len(boxes), max_objects)
        for i in range(num_objs):
            ct_x = cx_feat[i]
            ct_y = cy_feat[i]
            ct_x_int = int(torch.clamp(ct_x.floor(), 0, W - 1).item())
            ct_y_int = int(torch.clamp(ct_y.floor(), 0, H - 1).item())

            # For 3x3 pixel objects, radius would collapse to 0. Enforce min_radius.
            # If target has measurable area on feature map, estimate radius
            box_sz = max(w_feat[i].item(), h_feat[i].item())
            radius = max(min_radius, int(round(box_sz / 2.0)))

            draw_gaussian(gt_heatmap[b, 0], (ct_x_int, ct_y_int), radius=radius)

            # Sub-pixel offset target
            offset_x = (ct_x - ct_x_int).item()
            offset_y = (ct_y - ct_y_int).item()

            gt_offset[b, i, 0] = offset_x
            gt_offset[b, i, 1] = offset_y
            gt_ind[b, i] = ct_y_int * W + ct_x_int
            gt_mask[b, i] = True

    return {
        "heatmap": gt_heatmap,
        "offset": gt_offset,
        "ind": gt_ind,
        "mask": gt_mask,
    }


class FocalLoss(nn.Module):
    """
    Modified Focal Loss for Heatmap Regression (CenterNet / CornerNet style).
    
    L = - 1/N * sum(
          (1 - y_pred)^alpha * log(y_pred)                  if y_true == 1
          (1 - y_true)^beta * (y_pred)^alpha * log(1-y_pred) otherwise
        )
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) in [0, 1]
            gt:   (B, 1, H, W) in [0, 1]
        """
        pos_inds = gt.eq(1.0).float()
        neg_inds = gt.lt(1.0).float()

        neg_weights = torch.pow(1.0 - gt, self.beta)

        pos_loss = torch.log(pred.clamp(min=1e-6)) * torch.pow(1.0 - pred, self.alpha) * pos_inds
        neg_loss = torch.log((1.0 - pred).clamp(min=1e-6)) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / num_pos


class RegL1Loss(nn.Module):
    """L1 Loss masked by target positions."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_offset: torch.Tensor,
        gt_offset: torch.Tensor,
        ind: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_offset: (B, 2, H, W)
            gt_offset:   (B, max_objs, 2)
            ind:         (B, max_objs) flat indices
            mask:        (B, max_objs) bool
        """
        B, C, H, W = pred_offset.shape
        # Flatten spatial dimensions: (B, 2, H*W) -> (B, H*W, 2)
        pred_offset = pred_offset.view(B, C, -1).permute(0, 2, 1)

        # Gather predictions at object positions
        ind_expanded = ind.unsqueeze(-1).expand(-1, -1, 2)  # (B, max_objs, 2)
        pred_at_obj = torch.gather(pred_offset, dim=1, index=ind_expanded)

        mask_expanded = mask.unsqueeze(-1).expand_as(pred_at_obj).float()
        loss = F.l1_loss(pred_at_obj * mask_expanded, gt_offset * mask_expanded, reduction="sum")
        num_pos = mask.float().sum()
        return loss / (num_pos * 2 + 1e-6)


class HeatmapLoss(nn.Module):
    """
    Combined Loss for Tiny Object Heatmap Detection:
    Loss = hm_weight * FocalLoss + offset_weight * RegL1Loss
    """

    def __init__(self, hm_weight: float = 1.0, offset_weight: float = 0.5):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=2.0, beta=4.0)
        self.offset_loss = RegL1Loss()
        self.hm_weight = hm_weight
        self.offset_weight = offset_weight

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_hm = self.focal_loss(preds["heatmap"], targets["heatmap"])
        loss_offset = self.offset_loss(
            preds["offset"], targets["offset"], targets["ind"], targets["mask"]
        )

        total_loss = self.hm_weight * loss_hm + self.offset_weight * loss_offset
        loss_items = {
            "loss_total": float(total_loss.detach().item()),
            "loss_hm": float(loss_hm.detach().item()),
            "loss_offset": float(loss_offset.detach().item()),
        }
        return total_loss, loss_items
