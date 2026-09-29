#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Self-supervised camera-motion and independent-motion disentangler."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class EgoMotionNet(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, base_channels, 5, 2, 2), nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, 5, 2, 2), nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 5, 2, 2), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(base_channels * 4, 8)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, previous, current):
        params = self.head(self.encoder(torch.cat((previous, current), dim=1)).flatten(1))
        matrix = torch.eye(3, device=params.device, dtype=params.dtype).expand(params.shape[0], -1, -1).clone()
        matrix[:, 0, 0] += 0.1 * torch.tanh(params[:, 0])
        matrix[:, 0, 1] = 0.1 * torch.tanh(params[:, 1])
        matrix[:, 0, 2] = 0.2 * torch.tanh(params[:, 2])
        matrix[:, 1, 0] = 0.1 * torch.tanh(params[:, 3])
        matrix[:, 1, 1] += 0.1 * torch.tanh(params[:, 4])
        matrix[:, 1, 2] = 0.2 * torch.tanh(params[:, 5])
        matrix[:, 2, 0] = 0.01 * torch.tanh(params[:, 6])
        matrix[:, 2, 1] = 0.01 * torch.tanh(params[:, 7])
        return matrix


def warp_perspective(image, matrix):
    batch, _, height, width = image.shape
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=image.device, dtype=image.dtype),
        torch.linspace(-1, 1, width, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    target = torch.stack((x.flatten(), y.flatten(), torch.ones_like(x).flatten()), dim=0).unsqueeze(0).expand(batch, -1, -1)
    source = torch.bmm(torch.linalg.inv(matrix), target)
    source = source[:, :2] / source[:, 2:3].clamp_min(1e-7)
    grid = source.transpose(1, 2).reshape(batch, height, width, 2)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)


class MotionSaliencyNet(nn.Module):
    def __init__(self, base_channels=32, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(2, base_channels, kernel_size, padding=pad), nn.GroupNorm(4, base_channels), nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1), nn.GroupNorm(4, base_channels * 2), nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(base_channels, 1, 1), nn.Sigmoid(),
        )
        nn.init.constant_(self.net[-2].bias, -2.8)

    def forward(self, current, residual):
        return self.net(torch.cat((current, residual), dim=1))


class MotionDisentangler(nn.Module):
    def __init__(self, base_channels=32, kernel_size=5):
        super().__init__()
        self.ego = EgoMotionNet(base_channels)
        self.saliency = MotionSaliencyNet(base_channels, kernel_size)

    def forward(self, frames):
        current = frames[:, 2:3]
        previous = frames[:, 1:2]
        matrix = self.ego(previous, current)
        warped = warp_perspective(previous, matrix)
        residual = (current - warped).abs()
        saliency = self.saliency(current, residual)
        return {"saliency": saliency, "warped": warped, "residual": residual, "matrix": matrix}


def photometric_error(current, warped, loss_type="L1"):
    if loss_type == "SmoothL1":
        return F.smooth_l1_loss(current, warped, reduction="none")
    if loss_type == "Charbonnier":
        return torch.sqrt((current - warped).square() + 1e-7)
    return (current - warped).abs()


def compute_loss(
    output,
    current,
    previous_matrix,
    lambda_sparse,
    lambda_smooth,
    lambda_reg,
    loss_type="L1",
    mean_target=0.05,
    mean_barrier=2.0,
    use_saliency=True,
):
    saliency, warped, matrix = output["saliency"], output["warped"], output["matrix"]
    photo = photometric_error(current, warped, loss_type)
    zero = torch.zeros((), device=current.device, dtype=current.dtype)
    reg = zero if previous_matrix is None else (matrix - 2 * previous_matrix[0] + previous_matrix[1]).square().mean()
    if not use_saliency:
        return {"photo": photo.mean(), "sparse": zero, "smooth": zero, "reg": reg, "barrier": zero, "peak_floor": zero, "total": photo.mean() + lambda_reg * reg}
    smooth = (saliency[:, :, :, 1:] - saliency[:, :, :, :-1]).abs().mean() + (saliency[:, :, 1:, :] - saliency[:, :, :-1, :]).abs().mean()
    mean = saliency.mean()
    flat = saliency.flatten(1)
    peak_count = max(1, flat.shape[1] // 1000)
    peak = flat.topk(peak_count, dim=1).values.mean()
    barrier = mean_barrier * F.softplus(mean - mean_target).square()
    peak_floor = 0.2 * F.relu(0.05 - peak).square()
    terms = {"photo": ((1 - saliency) * photo).mean(), "sparse": mean, "smooth": smooth, "reg": reg, "barrier": barrier, "peak_floor": peak_floor}
    terms["total"] = terms["photo"] + lambda_sparse * mean + lambda_smooth * smooth + lambda_reg * reg + barrier + peak_floor
    return terms
