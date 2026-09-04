#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Heatmap Model Architecture for Tiny / Point-like Object Detection (e.g. UAV 3x3 pixels).

Replaces standard YOLO bounding box regression with high-resolution Gaussian Heatmap
probability regression and sub-pixel offset regression.
"""

from __future__ import annotations

import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3k2, SPPF, C2PSA


class HeatmapHead(nn.Module):
    """
    CenterNet / SPIRE-style Heatmap & Offset regression head.
    Designed for point-like tiny objects (e.g., 3x3 UAVs).
    
    Outputs:
    - heatmap: (B, 1, H, W), logits/probabilities for target center
    - offset:  (B, 2, H, W), sub-pixel offset (dx, dy) within the stride cell
    """

    def __init__(self, in_channels: int, head_conv: int = 64, num_classes: int = 1):
        super().__init__()
        # Feature refining convolution layers
        self.feat_conv = nn.Sequential(
            Conv(in_channels, head_conv, k=3),
            Conv(head_conv, head_conv, k=3),
        )

        # 1. Heatmap branch (probability of object presence)
        self.heatmap = nn.Sequential(
            nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_classes, kernel_size=1),
        )

        # 2. Offset branch (sub-pixel shift dx, dy in [0, 1])
        self.offset = nn.Sequential(
            nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        # CenterNet standard initialization: bias = -2.19 corresponds to sigmoid(bias) ≈ 0.1
        # This prevents gradient explosions at the beginning of training on heavily imbalanced background
        self.heatmap[-1].bias.data.fill_(-2.19)
        self.offset[-1].bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.feat_conv(x)
        hm = torch.sigmoid(self.heatmap(feat))
        # Clamp hm to avoid log(0) in loss
        hm = torch.clamp(hm, min=1e-6, max=1.0 - 1e-6)
        offset = self.offset(feat)
        return {"heatmap": hm, "offset": offset}


class YOLO26HeatmapDetector(nn.Module):
    """
    Detector integrating YOLO26 backbone + P2 Neck with Heatmap Head.
    
    Default Stride is 4 (P2 layer, 160x160 for 640x640 input).
    Can optionally upsample P2 to P1 (Stride 2, 320x320 for 640x640 input) for extreme small target recall.
    """

    def __init__(
        self,
        stride: int = 4,
        weights: str | Path | None = None,
        num_classes: int = 1,
    ):
        super().__init__()
        assert stride in (2, 4), f"Only stride 4 (P2) or stride 2 (P1) is supported, got {stride}."
        self.stride = stride
        self.num_classes = num_classes

        # Build YOLO26n backbone & P2 neck
        # Backbone:
        self.b0 = Conv(3, 16, 3, 2)            # 0: P1 / 2 (320x320)
        self.b1 = Conv(16, 32, 3, 2)           # 1: P2 / 4 (160x160)
        self.b2 = C3k2(32, 64, n=1, c3k=False, e=0.25)  # 2: P2 / 4
        self.b3 = Conv(64, 64, 3, 2)           # 3: P3 / 8 (80x80)
        self.b4 = C3k2(64, 128, n=1, c3k=False, e=0.25) # 4: P3 / 8
        self.b5 = Conv(128, 128, 3, 2)         # 5: P4 / 16 (40x40)
        self.b6 = C3k2(128, 128, n=1, c3k=True) # 6: P4 / 16
        self.b7 = Conv(128, 256, 3, 2)         # 7: P5 / 32 (20x20)
        self.b8 = C3k2(256, 256, n=1, c3k=True) # 8: P5 / 32
        self.b9 = SPPF(256, 256, 5, 3, True)   # 9: P5 / 32
        self.b10 = C2PSA(256, 256, n=1)        # 10: P5 / 32

        # Neck (FPN top-down to P2):
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest") # 11: 40x40
        self.c13 = C3k2(256 + 128, 128, n=1, c3k=True)        # 13: 40x40

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest") # 14: 80x80
        self.c16 = C3k2(128 + 128, 64, n=1, c3k=True)         # 16: 80x80

        self.up3 = nn.Upsample(scale_factor=2, mode="nearest") # 17: 160x160
        self.c19 = C3k2(64 + 64, 32, n=1, c3k=True)           # 19: P2 feature (32 channels, stride 4)

        # Neck (PAN bottom-up):
        self.down1 = Conv(32, 32, 3, 2)                        # 20: 80x80
        self.c22 = C3k2(32 + 64, 64, n=1, c3k=True)           # 22: P3 feature

        # Final fusion onto P2 (re-injecting enriched semantic context from P3 to P2)
        self.up_p2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse_p2 = Conv(32 + 64, 64, 3, 1)                # 64 channels at stride 4

        if stride == 2:
            # Stride 2 branch: upsample P2 to P1 and fuse with backbone b0
            self.up_p1 = nn.Upsample(scale_factor=2, mode="nearest")
            self.fuse_p1 = Conv(64 + 16, 48, 3, 1)            # 48 channels at stride 2
            head_in_ch = 48
        else:
            head_in_ch = 64

        self.head = HeatmapHead(in_channels=head_in_ch, head_conv=64, num_classes=num_classes)

        if weights:
            self.load_pretrained(weights)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone forward
        p1 = self.b0(x)                 # /2
        p2_in = self.b2(self.b1(p1))    # /4
        p3_in = self.b4(self.b3(p2_in)) # /8
        p4_in = self.b6(self.b5(p3_in)) # /16
        p5_in = self.b10(self.b9(self.b8(self.b7(p4_in)))) # /32

        # FPN Top-down
        p4_fpn = self.c13(torch.cat([self.up1(p5_in), p4_in], dim=1))
        p3_fpn = self.c16(torch.cat([self.up2(p4_fpn), p3_in], dim=1))
        p2_fpn = self.c19(torch.cat([self.up3(p3_fpn), p2_in], dim=1))

        # Bottom-up enhancement
        p3_pan = self.c22(torch.cat([self.down1(p2_fpn), p3_fpn], dim=1))
        
        # P2 enriched feature
        p2_out = self.fuse_p2(torch.cat([p2_fpn, self.up_p2(p3_pan)], dim=1))

        if self.stride == 2:
            p1_out = self.fuse_p1(torch.cat([self.up_p1(p2_out), p1], dim=1))
            return p1_out
        return p2_out

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.extract_features(x)
        return self.head(feat)

    def load_pretrained(self, weights_path: str | Path):
        weights_path = Path(weights_path)
        if not weights_path.exists():
            print(f"[WARN] Pretrained weights file not found: {weights_path}, training from scratch.")
            return

        ckpt = torch.load(weights_path, map_location="cpu")
        state_dict = ckpt["model"].state_dict() if "model" in ckpt else ckpt.get("state_dict", ckpt)

        # Match layers between YOLO26 / YOLO26-P2 and this backbone
        # We match matching tensor shapes
        own_state = self.state_dict()
        transferred = 0
        skipped = 0

        for k, v in state_dict.items():
            # Strip model. prefix if exists
            clean_k = k.replace("model.model.", "").replace("model.", "")
            
            # Map YOLO layer indices to our module names
            # e.g., '0.' -> 'b0.', '1.' -> 'b1.', '2.' -> 'b2.' ...
            parts = clean_k.split(".", 1)
            if len(parts) == 2 and parts[0].isdigit():
                idx = int(parts[0])
                rest = parts[1]
                target_key = None
                if idx == 0: target_key = f"b0.{rest}"
                elif idx == 1: target_key = f"b1.{rest}"
                elif idx == 2: target_key = f"b2.{rest}"
                elif idx == 3: target_key = f"b3.{rest}"
                elif idx == 4: target_key = f"b4.{rest}"
                elif idx == 5: target_key = f"b5.{rest}"
                elif idx == 6: target_key = f"b6.{rest}"
                elif idx == 7: target_key = f"b7.{rest}"
                elif idx == 8: target_key = f"b8.{rest}"
                elif idx == 9: target_key = f"b9.{rest}"
                elif idx == 10: target_key = f"b10.{rest}"
                elif idx == 13: target_key = f"c13.{rest}"
                elif idx == 16: target_key = f"c16.{rest}"
                elif idx == 19: target_key = f"c19.{rest}"
                elif idx == 22: target_key = f"c22.{rest}"

                if target_key and target_key in own_state:
                    if own_state[target_key].shape == v.shape:
                        own_state[target_key].copy_(v)
                        transferred += 1
                        continue
            skipped += 1

        print(f"[INFO] Loaded pretrained weights from {weights_path.name}: {transferred} tensors matched, {skipped} skipped.")
