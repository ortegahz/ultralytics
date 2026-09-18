#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Searchable zero-init residual highway for GMC velocity features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VelocityResidualHighway(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 48, mid_channels: int = 16, kernel_mode: str = "dw3", depth: int = 1, gate_mode: str = "scalar"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.kernel_mode = kernel_mode
        self.depth = depth
        self.gate_mode = gate_mode
        layers: list[nn.Module] = [nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False), nn.BatchNorm2d(mid_channels), nn.SiLU(inplace=True)]
        for _ in range(depth - 1):
            if kernel_mode == "dw5":
                layers.extend([nn.Conv2d(mid_channels, mid_channels, 5, padding=2, groups=mid_channels, bias=False), nn.BatchNorm2d(mid_channels), nn.SiLU(inplace=True)])
            elif kernel_mode == "dilated3":
                layers.extend([nn.Conv2d(mid_channels, mid_channels, 3, padding=2, dilation=2, groups=mid_channels, bias=False), nn.BatchNorm2d(mid_channels), nn.SiLU(inplace=True)])
            else:
                layers.extend([nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels, bias=False), nn.BatchNorm2d(mid_channels), nn.SiLU(inplace=True)])
        layers.extend([nn.MaxPool2d(2, 2), nn.Conv2d(mid_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels)])
        self.body = nn.Sequential(*layers)
        if gate_mode == "channel":
            self.gate = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        elif gate_mode == "spatial":
            self.gate_net = nn.Conv2d(in_channels, 1, 3, padding=1)
            nn.init.zeros_(self.gate_net.weight)
            nn.init.zeros_(self.gate_net.bias)
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.gate = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        for module in self.body.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def effective_gate(self):
        return torch.tanh(self.gate) * 0.05

    def forward(self, velocity: torch.Tensor) -> torch.Tensor:
        delta = self.body(velocity)
        if delta.shape[-2:] != (320, 320):
            delta = F.interpolate(delta, size=(320, 320), mode="bilinear", align_corners=False)
        gate = self.effective_gate()
        if self.gate_mode == "spatial":
            mask = torch.sigmoid(self.gate_net(velocity))
            if mask.shape[-2:] != delta.shape[-2:]:
                mask = F.interpolate(mask, size=delta.shape[-2:], mode="bilinear", align_corners=False)
            return gate * mask * delta
        return gate * delta
