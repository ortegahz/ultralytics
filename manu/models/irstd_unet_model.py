#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
IRSTD-UNet: Dedicated Infrared Small Target Detection Network.

Key Architectural Principles for 1~3px Point Targets:
1. High-Resolution Detail Preservation:
   - Preserves high-frequency details from Stride 1 and Stride 2 via direct dense skip-connections.
   - Avoids deep feature dilution typical in standard object detection backbones.
2. Directional & Point-Spread Response Blocks:
   - Residual Bottleneck with channel attention (Squeeze-and-Excitation / Gate) to isolate isotropic
     Gaussian pulses from anisotropic terrestrial clutter.
3. Sub-Pixel Convolution (PixelShuffle) Reconstruction Head:
   - Reconstructs sharp probability peaks instead of smooth bilinear upsampling.
   - Outputs:
     - 'prob_map': Dense probability map (B, 1, H, W) or (B, 1, H/2, W/2).
     - 'offset': Sub-pixel coordinate fine-tuning (B, 2, H, W).
"""

from __future__ import annotations

import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Standard Conv2d + BatchNorm2d + LeakyReLU/SiLU."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None, act: bool = True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Residual Block with optional Channel Attention."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, k=3, s=1)
        self.conv2 = ConvBNAct(channels, channels, k=3, s=1, act=False)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        # Lightweight Channel Gate
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(8, channels // 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, channels // 4), channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        feat = self.conv2(self.conv1(x))
        feat = feat * self.se(feat)
        return self.act(res + feat)


class SubPixelUpsampleBlock(nn.Module):
    """
    Sub-Pixel Convolution (PixelShuffle) for sharp, high-frequency boundary reconstruction.
    Preferred over bilinear/nearest interpolation to prevent peak flattening on tiny dots.
    """

    def __init__(self, in_ch: int, out_ch: int, scale_factor: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pixel_shuffle(self.conv(x))
        return self.act(self.bn(x))


class AsymmetricContextModulation(nn.Module):
    """
    Asymmetric Context Modulation (inspired by ACMNet).
    Uses deep context to selectively gate and modulate shallow high-resolution detail.
    """

    def __init__(self, shallow_ch: int, deep_ch: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(deep_ch, shallow_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(shallow_ch),
            nn.Sigmoid(),
        )
        self.fuse = ConvBNAct(shallow_ch + deep_ch, shallow_ch, k=3)

    def forward(self, shallow: torch.Tensor, deep: torch.Tensor) -> torch.Tensor:
        # deep feature is already upsampled to match shallow spatial dimension
        mod_shallow = shallow * self.gate(deep)
        out = self.fuse(torch.cat([mod_shallow, deep], dim=1))
        return out


class IRSTDNet(nn.Module):
    """
    U-Net style Infrared Small Target Detector with Sub-pixel Reconstruction.
    
    Architecture (for 640x640 3-ch difference input):
    - Stride 1 (640x640): C1 = 24
    - Stride 2 (320x320): C2 = 48
    - Stride 4 (160x160): C3 = 96
    - Stride 8 (80x80):   C4 = 192 (Deep Context)
    
    Decoder:
    - ACM context modulation + SubPixelUpsample
    - Stride 2 output (P1, 320x320) or Stride 1 (640x640)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_stride: int = 2,  # 2 for Stride 2 (320x320), 1 for Stride 1 (640x640)
        base_channels: int = 24,
        num_classes: int = 1,
    ):
        super().__init__()
        assert out_stride in (1, 2), f"out_stride must be 1 or 2, got {out_stride}"
        self.out_stride = out_stride
        self.num_classes = num_classes

        c1 = base_channels       # 24
        c2 = base_channels * 2   # 48
        c3 = base_channels * 4   # 96
        c4 = base_channels * 8   # 192

        # ----------------- Encoder -----------------
        # Level 1: Stride 1 (640x640)
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, c1, k=3, s=1),
            ResBlock(c1),
        )

        # Level 2: Stride 2 (320x320)
        self.down1 = ConvBNAct(c1, c2, k=3, s=2)
        self.enc2 = ResBlock(c2)

        # Level 3: Stride 4 (160x160)
        self.down2 = ConvBNAct(c2, c3, k=3, s=2)
        self.enc3 = ResBlock(c3)

        # Level 4: Stride 8 (80x80) Context Bottleneck
        self.down3 = ConvBNAct(c3, c4, k=3, s=2)
        self.enc4 = nn.Sequential(
            ResBlock(c4),
            ResBlock(c4),
        )

        # ----------------- Decoder -----------------
        # Up from Level 4 to Level 3 (80x80 -> 160x160)
        self.up3 = SubPixelUpsampleBlock(c4, c3, scale_factor=2)
        self.acm3 = AsymmetricContextModulation(shallow_ch=c3, deep_ch=c3)

        # Up from Level 3 to Level 2 (160x160 -> 320x320)
        self.up2 = SubPixelUpsampleBlock(c3, c2, scale_factor=2)
        self.acm2 = AsymmetricContextModulation(shallow_ch=c2, deep_ch=c2)

        if out_stride == 1:
            # Up from Level 2 to Level 1 (320x320 -> 640x640)
            self.up1 = SubPixelUpsampleBlock(c2, c1, scale_factor=2)
            self.acm1 = AsymmetricContextModulation(shallow_ch=c1, deep_ch=c1)
            head_in_ch = c1
        else:
            head_in_ch = c2

        # ----------------- Output Heads -----------------
        # Probability / Heatmap Head
        self.prob_head = nn.Sequential(
            ConvBNAct(head_in_ch, head_in_ch, k=3),
            nn.Conv2d(head_in_ch, num_classes, kernel_size=1),
        )

        # Sub-pixel Offset Head (dx, dy in [0, 1])
        self.offset_head = nn.Sequential(
            ConvBNAct(head_in_ch, head_in_ch, k=3),
            nn.Conv2d(head_in_ch, 2, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Prior probability bias initialization for the final classification layer
        # bias = -2.19 gives sigmoid(-2.19) ≈ 0.1, preventing background gradient explosion
        nn.init.constant_(self.prob_head[-1].bias, -2.19)
        nn.init.constant_(self.offset_head[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Encoder
        x1 = self.stem(x)         # Stride 1
        x2 = self.enc2(self.down1(x1)) # Stride 2
        x3 = self.enc3(self.down2(x2)) # Stride 4
        x4 = self.enc4(self.down3(x3)) # Stride 8

        # Decoder
        d3 = self.up3(x4)
        d3 = self.acm3(x3, d3)

        d2 = self.up2(d3)
        d2 = self.acm2(x2, d2)

        if self.out_stride == 1:
            d1 = self.up1(d2)
            out_feat = self.acm1(x1, d1)
        else:
            out_feat = d2

        # Heads
        raw_logits = self.prob_head(out_feat)
        prob = torch.sigmoid(raw_logits)
        # Sub-pixel offsets are physically in [0, 1] relative to the cell
        offset = torch.sigmoid(self.offset_head(out_feat))

        return {
            "logits": raw_logits, # (B, 1, H_out, W_out) for stable BCEWithLogitsLoss
            "heatmap": prob,      # (B, 1, H_out, W_out) in [0, 1] for SoftIoU & peak extraction
            "offset": offset,     # (B, 2, H_out, W_out) in [0, 1]
        }
