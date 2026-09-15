#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Searchable RK3588-Friendly Local Cross-Window Spatio-Temporal Attention Highway.

Supports High-Throughput Automated NAS Search:
1. Pure Hardware Operators: Conv2d, DepthwiseConv2d, BatchNorm2d, SiLU, Sigmoid, Add/Mul.
2. Receptive Field Kernel Search:
   - 'dual_scale_3_5': parallel 3x3 DWConv + 5x5 DWConv (near + maneuvering)
   - 'single_scale_3': compact 3x3 DWConv (near-range ultra lightweight)
   - 'single_scale_5': 5x5 DWConv (wide-range maneuver coverage)
   - 'dilated_3_d2': 3x3 DWConv with dilation=2 (wide field without param bloat)
3. Capacity & Depth Search:
   - mid_channels: [12, 16, 24]
   - gate_activation: 'sigmoid', 'silu', 'tanh'
4. Residual Gating Dynamics:
   - 'tanh_bounded': alpha = tanh(gate) * 0.05 (strictly zero-init)
   - 'positive_softplus': alpha = softplus(gate + init_bias) * 0.05 (strictly positive, prevents decaying into negative denoiser)
   - 'channel_adaptive': 48-dim channel vector alpha_c (channel-specific sensitivity)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class ConvLocalCrossAttentionBlock(nn.Module):
    """
    Searchable Pure-Convolutional Local Cross-Attention Block (RK3588 Native).
    """

    def __init__(
        self,
        feat_channels: int = 48,
        temporal_channels: int = 16,
        mid_channels: int = 24,
        kernel_mode: str = "dual_scale_3_5",
        gate_activation: str = "sigmoid",
    ):
        super().__init__()
        self.feat_channels = feat_channels
        self.temporal_channels = temporal_channels
        self.mid_channels = mid_channels
        self.kernel_mode = kernel_mode
        self.gate_activation = gate_activation

        # 1. Query, Key, Value projections
        self.q_proj = Conv(feat_channels, mid_channels, k=1, s=1)
        self.k_proj = Conv(temporal_channels, mid_channels, k=1, s=1)
        self.v_proj = nn.Sequential(
            Conv(temporal_channels, mid_channels, k=1, s=1),
            Conv(mid_channels, feat_channels, k=3, s=1),
        )

        # 2. Local Correlation Kernels
        corr_in_ch = mid_channels * 2
        self.corr_stem = Conv(corr_in_ch, mid_channels, k=1, s=1)

        if kernel_mode == "dual_scale_3_5":
            self.dw_branch1 = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True),
            )
            self.dw_branch2 = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True),
            )
            gate_in_ch = mid_channels * 2
        elif kernel_mode == "single_scale_5":
            self.dw_branch1 = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True),
            )
            self.dw_branch2 = None
            gate_in_ch = mid_channels
        elif kernel_mode == "dilated_3_d2":
            self.dw_branch1 = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True),
            )
            self.dw_branch2 = None
            gate_in_ch = mid_channels
        else:  # single_scale_3
            self.dw_branch1 = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True),
            )
            self.dw_branch2 = None
            gate_in_ch = mid_channels

        # 3. Soft gating activation
        if gate_activation == "tanh":
            act_layer = nn.Tanh()
        elif gate_activation == "silu":
            act_layer = nn.SiLU(inplace=True)
        else:
            act_layer = nn.Sigmoid()

        self.gate_proj = nn.Sequential(
            Conv(gate_in_ch, mid_channels, k=1, s=1),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),
            act_layer,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, feat_curr: torch.Tensor, feat_hist: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(feat_curr)
        k = self.k_proj(feat_hist)
        v = self.v_proj(feat_hist)

        qk_pair = torch.cat([q, k], dim=1)
        qk_feat = self.corr_stem(qk_pair)

        if self.dw_branch2 is not None:
            c1 = self.dw_branch1(qk_feat)
            c2 = self.dw_branch2(qk_feat)
            corr_feat = torch.cat([c1, c2], dim=1)
        else:
            corr_feat = self.dw_branch1(qk_feat)

        attn_mask = self.gate_proj(corr_feat)
        return attn_mask * v


class LocalCrossWindowAttentionHighway(nn.Module):
    """
    Configurable Temporal Cross-Attention Residual Highway for Automated NAS Search.
    """

    def __init__(
        self,
        in_diff_channels: int = 1,
        feat_channels: int = 48,
        num_hist_frames: int = 4,
        mid_channels: int = 16,
        kernel_mode: str = "dual_scale_3_5",
        gate_activation: str = "sigmoid",
        gate_mode: str = "tanh_bounded",
    ):
        super().__init__()
        self.num_hist_frames = num_hist_frames
        self.feat_channels = feat_channels
        self.mid_channels = mid_channels
        self.kernel_mode = kernel_mode
        self.gate_activation = gate_activation
        self.gate_mode = gate_mode

        # 1. Shallow temporal pulse stem
        self.hist_stem = nn.Sequential(
            Conv(in_diff_channels, mid_channels, k=3, s=1),
            Conv(mid_channels, mid_channels, k=3, s=1),
        )

        # 2. Local Cross-Attention Blocks for each historical time step
        self.cross_attn_blocks = nn.ModuleList([
            ConvLocalCrossAttentionBlock(
                feat_channels=feat_channels,
                temporal_channels=mid_channels,
                mid_channels=mid_channels,
                kernel_mode=kernel_mode,
                gate_activation=gate_activation,
            )
            for _ in range(num_hist_frames)
        ])

        # 3. Inter-frame temporal fusion
        self.temporal_fuse = nn.Sequential(
            Conv(feat_channels * num_hist_frames, feat_channels, k=1, s=1),
            Conv(feat_channels, feat_channels, k=3, s=1),
        )

        # 4. Searchable Residual Gate
        self.scale_factor = 0.05
        if gate_mode == "channel_adaptive":
            self.gate = nn.Parameter(torch.zeros(1, feat_channels, 1, 1))
        elif gate_mode == "positive_softplus":
            # Softplus(-6) ≈ 0.00247, scale * 0.00247 ≈ 0.0001 (effectively zero start, strictly positive)
            self.gate = nn.Parameter(torch.full((1,), -6.0))
        else:  # tanh_bounded
            self.gate = nn.Parameter(torch.zeros(1))

    def get_effective_alpha(self) -> float | torch.Tensor:
        if self.gate_mode == "positive_softplus":
            alpha = F.softplus(self.gate) * self.scale_factor
        elif self.gate_mode == "channel_adaptive":
            alpha = torch.tanh(self.gate) * self.scale_factor
        else:
            alpha = torch.tanh(self.gate) * self.scale_factor
        return alpha

    def forward(self, feat_curr: torch.Tensor, diff_seq: torch.Tensor) -> torch.Tensor:
        B, K, C, H, W = diff_seq.shape
        assert K == self.num_hist_frames, f"Expected {self.num_hist_frames} historical frames, got {K}"

        diff_flat = diff_seq.view(B * K, C, H, W)
        hist_feats_flat = self.hist_stem(diff_flat)
        _, c_h, _, _ = hist_feats_flat.shape
        hist_feats = hist_feats_flat.view(B, K, c_h, H, W)

        aligned_list = []
        for k in range(K):
            h_k = hist_feats[:, k, :, :, :]
            aligned_k = self.cross_attn_blocks[k](feat_curr, h_k)
            aligned_list.append(aligned_k)

        fused_temporal = self.temporal_fuse(torch.cat(aligned_list, dim=1))

        effective_alpha = self.get_effective_alpha()
        return effective_alpha * fused_temporal
