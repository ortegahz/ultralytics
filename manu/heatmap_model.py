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
        # Clamp hm to avoid log(0) and fp16 underflow/overflow
        hm = torch.clamp(hm, min=1e-5, max=1.0 - 1e-5)
        offset = self.offset(feat)
        return {"heatmap": hm, "offset": offset}


class LocalCorrelationBlock(nn.Module):
    """
    Local Feature Correlation Layer (inspired by RAFT) for infrared moving small target detection.
    
    Extracts high-resolution features from current frame I_t (channel 0) and long-lag reference frame
    I_{t-lag} (channel 2) using a shared lightweight stem, then computes normalized local dot-product 
    correlation volume across a (2R + 1) x (2R + 1) displacement search window.
    
    Parameters:
        radius (int): Search radius in feature pixels (radius=2 -> 5x5 = 25 velocity hypothesis channels).
        feat_dim (int): Intermediate feature dimension for correlation matching (default: 16).
        out_channels (int): Output projected channels (default: 16).
    """

    def __init__(self, radius: int = 2, feat_dim: int = 16, out_channels: int = 16):
        super().__init__()
        self.radius = radius
        self.num_displacements = (2 * radius + 1) * (2 * radius + 1)  # 25 channels for R=2
        
        # Lightweight shared spatial feature stem for matching (downsampled to stride 2: 320x320)
        self.match_stem = nn.Sequential(
            Conv(1, feat_dim, k=3, s=2),
            Conv(feat_dim, feat_dim, k=3, s=1),
        )
        
        # Dimension projection for the correlation volume + short-term difference
        self.proj = nn.Sequential(
            Conv(self.num_displacements, out_channels, k=1, s=1),
            Conv(out_channels, out_channels, k=3, s=1),
        )

    def forward(self, curr_gray: torch.Tensor, ref_gray: torch.Tensor) -> torch.Tensor:
        """
        Args:
            curr_gray: (B, 1, H, W) - Current frame I_t
            ref_gray:  (B, 1, H, W) - Long-term reference frame I_{t-lag}
        Returns:
            corr_feat: (B, out_channels, H/2, W/2) - Spatio-temporal motion correlation feature
        """
        B, _, H, W = curr_gray.shape
        r = self.radius

        # 1. Extract normalized feature maps: (B, C, H', W')
        f_curr = self.match_stem(curr_gray)
        f_ref = self.match_stem(ref_gray)

        # L2 normalize along channel dimension for cosine similarity
        f_curr = F.normalize(f_curr, p=2, dim=1)
        f_ref = F.normalize(f_ref, p=2, dim=1)

        # 2. Pad reference feature map by search radius
        # f_ref_pad: (B, C, H' + 2r, W' + 2r)
        f_ref_pad = F.pad(f_ref, (r, r, r, r), mode="replicate")

        # 3. Compute local dot-product correlation across (2r+1) x (2r+1) grid
        _, C, H_feat, W_feat = f_curr.shape
        corr_list = []
        for dy in range(2 * r + 1):
            for dx in range(2 * r + 1):
                # Slice the shifted patch from reference feature map
                ref_slice = f_ref_pad[:, :, dy : dy + H_feat, dx : dx + W_feat]
                # Dot product along channel dimension -> (B, 1, H', W')
                corr = torch.sum(f_curr * ref_slice, dim=1, keepdim=True)
                corr_list.append(corr)

        # Concat all displacement hypotheses: (B, 25, H', W')
        corr_volume = torch.cat(corr_list, dim=1)

        # 4. Project correlation volume to output feature space
        return self.proj(corr_volume)


class HybridCorrelationTemporalStem(nn.Module):
    """
    Hybrid Temporal Stem for 3-Channel Input: [I_t, |I_t - I_{t-2}|, I_{t-8}].
    
    1. Channel 0: I_t (Current frame gray) -> Static appearance stem.
    2. Channel 1: |I_t - I_{t-2}| (Short difference) -> High-frequency transient motion stem.
    3. Channel 0 & Channel 2: (I_t, I_{t-8}) -> LocalCorrelationBlock (5x5 velocity hypotheses).
    4. Fuse all 3 complementary representations into 16 channels at Stride 2 (P1 / 320x320).
    """

    def __init__(self, out_channels: int = 16, corr_radius: int = 2):
        super().__init__()
        # 1. Static appearance branch from current frame I_t
        self.spatial_stem = Conv(1, 8, k=3, s=2)  # (B, 8, H/2, W/2)

        # 2. Short-term transient difference branch |I_t - I_{t-2}|
        self.transient_stem = Conv(1, 8, k=3, s=2)  # (B, 8, H/2, W/2)

        # 3. Long-term local feature correlation branch (I_t, I_{t-8})
        self.corr_block = LocalCorrelationBlock(radius=corr_radius, feat_dim=16, out_channels=16)

        # 4. Multimodal fusion: (8 + 8 + 16 = 32 channels) -> out_channels (16)
        self.fuse = nn.Sequential(
            Conv(8 + 8 + 16, out_channels, k=1, s=1),
            Conv(out_channels, out_channels, k=3, s=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        curr = x[:, 0:1, :, :]      # I_t
        diff_short = x[:, 1:2, :, :] # |I_t - I_{t-2}|
        ref_long = x[:, 2:3, :, :]   # I_{t-8}

        feat_spatial = self.spatial_stem(curr)
        feat_transient = self.transient_stem(diff_short)
        feat_corr = self.corr_block(curr, ref_long)

        # Concat along channel dimension: (B, 32, H/2, W/2)
        fused = torch.cat([feat_spatial, feat_transient, feat_corr], dim=1)
        return self.fuse(fused)


class Learnable3FrameTemporalStem(nn.Module):
    """
    Learnable temporal difference stem for 3-frame raw inputs: [I_t, I_{t-4}, I_{t-12}].
    
    Architecture:
    1. Static spatial branch on current frame I_t (preserves high-res appearance & radiation profile).
    2. Multi-temporal signed motion difference branch [I_t - I_{t-4}, I_t - I_{t-12}].
       Preserves dipole signs (+bright / -dark) so conv kernels learn direction of flight.
    3. Temporal channel attention gate: dynamically attends to fast motion vs slow creep.
    4. Fused into 16 channels at stride 2 (P1 / 320x320) to seamlessly feed b1.
    """

    def __init__(self, out_channels: int = 16):
        super().__init__()
        # 1. Static appearance branch from current frame
        self.spatial_stem = Conv(1, 8, k=3, s=2)  # (B, 8, H/2, W/2)

        # 2. Signed motion difference branch
        self.motion_stem = nn.Sequential(
            Conv(2, 16, k=3, s=1),
            Conv(16, 16, k=3, s=2),  # (B, 16, H/2, W/2)
        )

        # 3. Velocity-aware temporal channel gate
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 1),
            nn.Sigmoid(),
        )

        # 4. Fused representation
        self.fuse = Conv(8 + 16, out_channels, k=3, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> [I_t, I_{t-4}, I_{t-12}]
        curr = x[:, 0:1, :, :]
        prev_mid = x[:, 1:2, :, :]
        prev_long = x[:, 2:3, :, :]

        # Static feature
        feat_spatial = self.spatial_stem(curr)

        # Signed temporal differences (no abs: network learns dipole positive/negative wavefronts)
        diff_mid = curr - prev_mid
        diff_long = curr - prev_long
        diff_stack = torch.cat([diff_mid, diff_long], dim=1)

        # Motion feature + temporal attention
        feat_motion = self.motion_stem(diff_stack)
        feat_motion = feat_motion * self.gate(feat_motion)

        # Fused P1 feature (B, 16, H/2, W/2)
        return self.fuse(torch.cat([feat_spatial, feat_motion], dim=1))


class PixelShuffleUpsample(nn.Module):
    """
    Sub-pixel convolution upsampling module for tiny point-like targets.
    Learns continuous sub-pixel interpolation instead of nearest neighbor staircase artifacts.
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        # Conv expands channels to out_channels * (scale_factor ** 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * (scale_factor**2),
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.ps = nn.PixelShuffle(scale_factor)
        self.act = nn.SiLU(inplace=True)
        self._init_weights(in_channels, out_channels)

    def _init_weights(self, in_channels: int, out_channels: int):
        # ICNR initialization or nearest-equivalent initialization for smooth start
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.ps(self.conv(x)))


class YOLO26HeatmapDetector(nn.Module):
    """
    Detector integrating YOLO26 backbone + P2 Neck with Heatmap Head.
    
    Default Stride is 4 (P2 layer, 160x160 for 640x640 input).
    Can optionally upsample P2 to P1 (Stride 2, 320x320 for 640x640 input) for extreme small target recall.
    Supports scale='n' (width=0.25) and scale='s' (width=0.50).
    """

    def __init__(
        self,
        stride: int = 4,
        weights: str | Path | None = None,
        num_classes: int = 1,
        use_temporal_stem: bool = False,
        temporal_mode: str = "standard",  # 'standard' (Conv 3ch), 'signed_3frame', 'hybrid_corr'
        upsample_mode: str = "nearest",  # 'nearest', 'pixelshuffle'
        scale: str = "n",  # 'n' or 's'
    ):
        super().__init__()
        assert stride in (2, 4), f"Only stride 4 (P2) or stride 2 (P1) is supported, got {stride}."
        assert scale in ("n", "s"), f"Only scale 'n' or 's' is supported, got {scale}."
        self.stride = stride
        self.scale = scale
        self.num_classes = num_classes
        self.use_temporal_stem = use_temporal_stem
        self.temporal_mode = temporal_mode
        self.upsample_mode = upsample_mode

        # Width scale multiplier (n: 0.25 -> 1x base; s: 0.50 -> 2x base)
        w = 1 if scale == "n" else 2

        c_p1 = 16 * w    # 16 or 32
        c_p2 = 32 * w    # 32 or 64
        c_p2_out = 64 * w  # 64 or 128
        c_p3 = 64 * w    # 64 or 128
        c_p3_out = 128 * w  # 128 or 256
        c_p4 = 128 * w   # 128 or 256
        c_p5 = 256 * w   # 256 or 512

        # Backbone:
        if temporal_mode == "hybrid_corr":
            self.b0 = HybridCorrelationTemporalStem(out_channels=c_p1, corr_radius=2)
        elif temporal_mode == "signed_3frame" or use_temporal_stem:
            self.b0 = Learnable3FrameTemporalStem(out_channels=c_p1)
        else:
            self.b0 = Conv(3, c_p1, 3, 2)                         # 0: P1 / 2
        self.b1 = Conv(c_p1, c_p2, 3, 2)                          # 1: P2 / 4
        self.b2 = C3k2(c_p2, c_p2_out, n=1, c3k=False, e=0.25)    # 2: P2 / 4
        self.b3 = Conv(c_p2_out, c_p3, 3, 2)                      # 3: P3 / 8
        self.b4 = C3k2(c_p3, c_p3_out, n=1, c3k=False, e=0.25)    # 4: P3 / 8
        self.b5 = Conv(c_p3_out, c_p4, 3, 2)                      # 5: P4 / 16
        self.b6 = C3k2(c_p4, c_p4, n=1, c3k=True)                 # 6: P4 / 16
        self.b7 = Conv(c_p4, c_p5, 3, 2)                          # 7: P5 / 32
        self.b8 = C3k2(c_p5, c_p5, n=1, c3k=True)                 # 8: P5 / 32
        self.b9 = SPPF(c_p5, c_p5, 5, 3, True)                    # 9: P5 / 32
        self.b10 = C2PSA(c_p5, c_p5, n=1)                         # 10: P5 / 32

        # Neck (FPN top-down to P2):
        if upsample_mode == "pixelshuffle":
            self.up1 = PixelShuffleUpsample(in_channels=c_p5, out_channels=c_p4, scale_factor=2)
            self.c13 = C3k2(c_p4 + c_p4, c_p4, n=1, c3k=True)
            self.up2 = PixelShuffleUpsample(in_channels=c_p4, out_channels=c_p3, scale_factor=2)
            self.c16 = C3k2(c_p3 + c_p3_out, c_p3, n=1, c3k=True)
            self.up3 = PixelShuffleUpsample(in_channels=c_p3, out_channels=c_p2, scale_factor=2)
            self.c19 = C3k2(c_p2 + c_p2_out, c_p2, n=1, c3k=True)
        else:
            self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
            self.c13 = C3k2(c_p5 + c_p4, c_p4, n=1, c3k=True)
            self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
            self.c16 = C3k2(c_p4 + c_p3_out, c_p3, n=1, c3k=True)
            self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
            self.c19 = C3k2(c_p3 + c_p2_out, c_p2, n=1, c3k=True)

        # Neck (PAN bottom-up):
        self.down1 = Conv(c_p2, c_p2, 3, 2)
        self.c22 = C3k2(c_p2 + c_p3, c_p3, n=1, c3k=True)

        # Final fusion onto P2 (re-injecting enriched semantic context from P3 to P2)
        if upsample_mode == "pixelshuffle":
            self.up_p2 = PixelShuffleUpsample(in_channels=c_p3, out_channels=c_p2_out, scale_factor=2)
            self.fuse_p2 = Conv(c_p2 + c_p2_out, c_p2_out, 3, 1)
        else:
            self.up_p2 = nn.Upsample(scale_factor=2, mode="nearest")
            self.fuse_p2 = Conv(c_p2 + c_p3, c_p2_out, 3, 1)

        if stride == 2:
            # Stride 2 branch: upsample P2 to P1 and fuse with backbone b0
            c_p1_fuse = 48 * w
            if upsample_mode == "pixelshuffle":
                self.up_p1 = PixelShuffleUpsample(in_channels=c_p2_out, out_channels=c_p2_out, scale_factor=2)
                self.fuse_p1 = Conv(c_p2_out + c_p1, c_p1_fuse, 3, 1)
            else:
                self.up_p1 = nn.Upsample(scale_factor=2, mode="nearest")
                self.fuse_p1 = Conv(c_p2_out + c_p1, c_p1_fuse, 3, 1)
            head_in_ch = c_p1_fuse
        else:
            head_in_ch = c_p2_out

        self.head = HeatmapHead(in_channels=head_in_ch, head_conv=64 * w, num_classes=num_classes)

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
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"].state_dict() if hasattr(ckpt["model"], "state_dict") else ckpt["model"]
        else:
            state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        # 1. 如果是同构/子集 Heatmap 权重（直接按名字与 shape 匹配）
        own_state = self.state_dict()
        if isinstance(state_dict, dict) and any(k in own_state for k in state_dict.keys()):
            transferred = 0
            skipped = 0
            for k, v in state_dict.items():
                clean_k = k.replace("module.", "").replace("model.model.", "").replace("model.", "")
                if clean_k in own_state:
                    if own_state[clean_k].shape == v.shape:
                        own_state[clean_k].copy_(v)
                        transferred += 1
                    else:
                        print(f"[WARN] Shape mismatch for {clean_k}: checkpoint {v.shape} vs model {own_state[clean_k].shape}, skipping.")
                        skipped += 1
                else:
                    skipped += 1
            print(f"[INFO] Loaded Heatmap pretrained weights from {weights_path.name}: {transferred}/{len(own_state)} layers matched directly, {skipped} skipped.")
            return

        # 2. 如果是原生 YOLO26 / YOLO26-P2 骨干权重（做层级索引映射）
        transferred = 0
        skipped = 0

        for k, v in state_dict.items():
            # Strip module. / model. prefix if exists
            clean_k = k.replace("module.", "").replace("model.model.", "").replace("model.", "")
            
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
