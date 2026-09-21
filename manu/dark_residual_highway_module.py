#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class DarkResidualHighway(nn.Module):
    """Extract a bounded trainable residual from the aligned dark temporal channel.

    The final projection is zero-initialized, so the residual is exactly zero at step 0 regardless of the gate and the
    model output is bit-identical to the frozen base. The gate starts fully open (tanh(1) * scale) so gradients reach
    the projection at step 0; a zero gate together with a zero projection would leave the branch permanently dead.
    """

    def __init__(self, mid_channels: int = 16, out_channels: int = 48, scale_factor: float = 0.05):
        super().__init__()
        self.scale_factor = scale_factor
        self.stem = nn.Sequential(
            Conv(1, mid_channels, k=3, s=1),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.proj = nn.Sequential(Conv(mid_channels, out_channels, k=1, s=1), Conv(out_channels, out_channels, k=3, s=1))
        self.gate = nn.Parameter(torch.ones(1))
        nn.init.zeros_(self.proj[-1].conv.weight)
        if self.proj[-1].conv.bias is not None:
            nn.init.zeros_(self.proj[-1].conv.bias)

    def forward(self, dark_residual: torch.Tensor) -> torch.Tensor:
        delta = self.proj(self.downsample(self.stem(dark_residual)))
        return torch.tanh(self.gate) * self.scale_factor * delta

    def effective_alpha(self) -> float:
        return (torch.tanh(self.gate) * self.scale_factor).item()

    def residual_magnitude(self, dark_residual: torch.Tensor) -> float:
        return self(dark_residual).abs().max().item()
