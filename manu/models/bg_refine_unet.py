#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Compact U-Net-like background reconstruction network for infrared frames."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BackgroundReconUNet(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 32, depth: int = 3):
        super().__init__()
        if depth not in (2, 3, 4):
            raise ValueError("depth must be 2, 3, or 4")
        channels = [base_channels * (2**index) for index in range(depth)]
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        previous = in_channels
        for index, current in enumerate(channels):
            self.encoders.append(ConvBlock(previous, current))
            if index < depth - 1:
                self.downsamples.append(nn.Conv2d(current, current, 3, stride=2, padding=1))
            previous = current
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for index in range(depth - 2, -1, -1):
            self.upconvs.append(nn.ConvTranspose2d(channels[index + 1], channels[index], 2, stride=2))
            self.decoders.append(ConvBlock(channels[index] * 2, channels[index]))
        self.out = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x):
        skips = []
        current = x
        for index, encoder in enumerate(self.encoders):
            current = encoder(current)
            skips.append(current)
            if index < len(self.downsamples):
                current = self.downsamples[index](current)
        for index, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            current = upconv(current)
            skip = skips[-index - 2]
            if current.shape[-2:] != skip.shape[-2:]:
                current = nn.functional.interpolate(current, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            current = decoder(torch.cat([current, skip], dim=1))
        return torch.sigmoid(self.out(current))
