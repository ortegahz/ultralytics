#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Multi-Timeframe Heatmap Model Architecture for Tiny Object Detection.

Extends YOLO26HeatmapDetector with a Multi-Timeframe Head:
- Outputs 3 Heatmap channels: [H_t, H_{t-lag}, H_{t+lag}]
- Uses shared P1 / P2 backbones to enforce spatio-temporal coherence.
- Fully backwards-compatible with standard single-frame inference (evaluates channel 0).
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from manu.heatmap_model import YOLO26HeatmapDetector


class MultiTimeframeHeatmapHead(nn.Module):
    """
    Multi-Timeframe Heatmap & Offset regression head.
    Outputs:
    - heatmap: (B, 3, H, W) -> [H_t (center), H_{t-lag} (past), H_{t+lag} (future)]
    - offset:  (B, 2, H, W) -> Sub-pixel coordinate fine-tuning for frame t
    """

    def __init__(self, in_channels: int, head_conv: int = 64, num_timeframes: int = 3):
        super().__init__()
        self.feat_conv = nn.Sequential(
            Conv(in_channels, head_conv, k=3),
            Conv(head_conv, head_conv, k=3),
        )

        # Multi-timeframe heatmap output (3 channels)
        self.heatmap = nn.Sequential(
            nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_timeframes, kernel_size=1),
        )

        # Center frame sub-pixel offset
        self.offset = nn.Sequential(
            nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        # CenterNet standard initialization: bias = -2.19 corresponds to sigmoid(bias) ≈ 0.1
        self.heatmap[-1].bias.data.fill_(-2.19)
        self.offset[-1].bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.feat_conv(x)
        hm = torch.sigmoid(self.heatmap(feat))
        hm = torch.clamp(hm, min=1e-5, max=1.0 - 1e-5)
        offset = self.offset(feat)
        return {"heatmap": hm, "offset": offset}


class YOLO26MultiTimeframeDetector(YOLO26HeatmapDetector):
    """
    YOLO26 detector with Multi-Timeframe Heatmap Head.
    Supports warm-starting from Trial 22 single-frame weights.
    """

    def __init__(
        self,
        stride: int = 2,
        num_timeframes: int = 3,
        temporal_mode: str = "standard",
    ):
        super().__init__(stride=stride, num_classes=1, temporal_mode=temporal_mode)
        # Determine in_channels for the head: Stride 2 is 48 channels (from fuse_p1), Stride 4 is 64 channels
        in_c = 48 if stride == 2 else 64
        self.head = MultiTimeframeHeatmapHead(in_channels=in_c, head_conv=64, num_timeframes=num_timeframes)

    def load_from_single_frame(self, weights_path: str | Path):
        """
        Loads backbone and transfers single-frame head weights to the center timeframe (channel 0).
        Initializes past and future timeframe heads identically for smooth warm-start.
        """
        weights_path = Path(weights_path)
        if not weights_path.exists():
            print(f"[WARN] Pretrained weights not found: {weights_path}")
            return

        ckpt = torch.load(weights_path, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
        if hasattr(state_dict, "state_dict"):
            state_dict = state_dict.state_dict()

        own_state = self.state_dict()
        matched = 0

        for k, v in state_dict.items():
            clean_k = k.replace("model.model.", "").replace("model.", "")
            if clean_k in own_state:
                if own_state[clean_k].shape == v.shape:
                    own_state[clean_k].copy_(v)
                    matched += 1
                elif "head.heatmap.2" in clean_k:
                    # Target has 3 channels, source has 1 channel
                    if "weight" in clean_k and own_state[clean_k].shape[1:] == v.shape[1:]:
                        # Repeat weights across 3 timeframe channels
                        own_state[clean_k][0:1].copy_(v)
                        own_state[clean_k][1:2].copy_(v)
                        own_state[clean_k][2:3].copy_(v)
                        matched += 1
                    elif "bias" in clean_k and own_state[clean_k].shape[0] == 3:
                        own_state[clean_k][0:1].copy_(v)
                        own_state[clean_k][1:2].copy_(v)
                        own_state[clean_k][2:3].copy_(v)
                        matched += 1

        print(f"[INFO] Warm-started MultiTimeframe model from {weights_path.name}: {matched} layers transferred.")
