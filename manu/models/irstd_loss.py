#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Loss Functions specifically engineered for Infrared Small Target Detection (IRSTD).

Adopts the de facto standard IRSTD formulation (as in ACMNet, DNA-Net, UIUNet):
1. Soft-IoU Loss: Measures region overlap for point/tiny targets.
   Includes background-energy penalty when no target is present.
2. Weighted BCEWithLogitsLoss: Uses PyTorch native optimized CUDA kernel.
   Per-pixel normalized, immune to huge spatial-pixel summations (guaranteed O(0.01 ~ 1.0)).
3. RegL1Loss: Sub-pixel coordinate fine-tuning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftIoULoss(nn.Module):
    """
    Soft-IoU loss function for infrared small target detection.
    
    L_soft_iou = 1 - (sum(P * G) + eps) / (sum(P) + sum(G) - sum(P * G) + eps)
    If pure background (sum(G) == 0), penalizes mean false-alarm activations.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) in [0, 1]
            gt:   (B, 1, H, W) in [0, 1]
        """
        pred = pred.float()
        gt = gt.float()
        B, _, H, W = pred.shape

        intersection = torch.sum(pred * gt, dim=(2, 3))
        union = torch.sum(pred, dim=(2, 3)) + torch.sum(gt, dim=(2, 3)) - intersection
        
        has_target = (torch.sum(gt, dim=(2, 3)) > 1e-3).float()
        
        iou = (intersection + self.eps) / (union + self.eps)
        loss_pos = 1.0 - iou
        loss_neg = torch.sum(pred, dim=(2, 3)) / float(H * W)

        loss = has_target * loss_pos + (1.0 - has_target) * loss_neg
        return loss.mean()


class RegL1Loss(nn.Module):
    """Sub-pixel coordinate offset loss masked at target center positions."""

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
            ind:         (B, max_objs) flat 1D indices (y * W + x)
            mask:        (B, max_objs) bool
        """
        B, C, H, W = pred_offset.shape
        pred_offset = pred_offset.view(B, C, -1).permute(0, 2, 1)

        ind_expanded = ind.unsqueeze(-1).expand(-1, -1, 2)
        pred_at_obj = torch.gather(pred_offset, dim=1, index=ind_expanded)

        mask_expanded = mask.unsqueeze(-1).expand_as(pred_at_obj).float()
        num_pos = mask.float().sum()
        if num_pos == 0:
            return torch.tensor(0.0, device=pred_offset.device, dtype=pred_offset.dtype)

        loss = F.l1_loss(pred_at_obj * mask_expanded, gt_offset * mask_expanded, reduction="sum")
        return loss / (num_pos * 2 + 1e-6)


class IRSTDLoss(nn.Module):
    """
    Unified Numerically Stable Multi-Task Loss for IRSTD-UNet:
    Loss = w_iou * SoftIoU(prob, gt) + w_bce * BCEWithLogits(logits, gt) + w_off * RegL1Loss
    """

    def __init__(
        self,
        w_iou: float = 1.0,
        w_bce: float = 1.0,
        w_off: float = 0.5,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.soft_iou = SoftIoULoss()
        # BCE with logits is per-pixel mean reduced, guaranteed O(1e-3 ~ 1.0)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.reg_l1 = RegL1Loss()

        self.w_iou = w_iou
        self.w_bce = w_bce
        self.w_off = w_off

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Ensure pos_weight is on the correct device
        if self.bce.pos_weight is not None and self.bce.pos_weight.device != preds["logits"].device:
            self.bce.pos_weight = self.bce.pos_weight.to(preds["logits"].device)

        loss_iou = self.soft_iou(preds["heatmap"], targets["heatmap"])
        loss_bce = self.bce(preds["logits"], targets["heatmap"])
        loss_off = self.reg_l1(
            preds["offset"], targets["offset"], targets["ind"], targets["mask"]
        )

        total_loss = self.w_iou * loss_iou + self.w_bce * loss_bce + self.w_off * loss_off

        loss_dict = {
            "loss_total": float(total_loss.detach().item()),
            "loss_iou": float(loss_iou.detach().item()),
            "loss_bce": float(loss_bce.detach().item()),
            "loss_offset": float(loss_off.detach().item()),
        }

        return total_loss, loss_dict
