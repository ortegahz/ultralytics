#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Head 分支合并等价性自检 (`merge_head_branches`)。

背景（memory.md §二十六 / rules.md §四·9f）：
  trial0474 stride=2 @640 下，`head.heatmap.0` 与 `head.offset.0` 两个 3x3 C=64 卷积
  各占 3.775 G MAC（合计 7.55 G，全网 21.48 G 的 35%），且都要把同一份 320x320x64
  特征读一遍。本脚本验证把两者合并为单个 Conv2d(64,128,3) 后**数值完全等价**。

本脚本不依赖 rknn-toolkit2、不需要 checkpoint、不上板，只验证结构与数学等价。

用法：
  cd /media/manu/1TB-Volume/workspace/ultralytics
  /home/manu/anaconda3/envs/uav/bin/python manu/pipeline/npu/verify_merge_head_branches.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from manu.models.heatmap_model import YOLO26HeatmapDetector, remap_head_branches_state_dict  # noqa: E402


def conv_inventory(model: nn.Module, shape: tuple[int, ...]) -> tuple[int, float]:
    """返回 (卷积层数, 总 MAC)，MAC = H_out*W_out*C_out*(C_in/groups)*K*K。"""
    recs: list[tuple[nn.Module, tuple[int, ...]]] = []
    handles = []

    def hook(_m, _i, o, _n=None):
        t = o
        while isinstance(t, (list, tuple)) and t:
            t = t[0]
        if isinstance(t, torch.Tensor):
            recs.append((_m, tuple(t.shape)))

    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
            handles.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        model(torch.zeros(*shape))
    for h in handles:
        h.remove()

    total = 0.0
    for mod, s in recs:
        _, co, ho, wo = s
        kh, kw = mod.kernel_size
        total += ho * wo * co * (mod.in_channels // mod.groups) * kh * kw
    return len(recs), total / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2, choices=(2, 4))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--scale", default="n")
    ap.add_argument("--frames", type=int, default=8, help="随机输入帧数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    common = dict(stride=args.stride, num_classes=1, scale=args.scale,
                  upsample_mode="nearest", use_p0_highway=False)

    base = YOLO26HeatmapDetector(**common, merge_head_branches=False).eval()
    merged = YOLO26HeatmapDetector(**common, merge_head_branches=True).eval()

    # 权重重映射：未合并 state_dict -> 合并版
    remapped = remap_head_branches_state_dict(base.state_dict())
    missing, unexpected = merged.load_state_dict(remapped, strict=False)
    assert not missing, f"[GATE FAIL] 合并版缺权重: {missing}"
    assert not unexpected, f"[GATE FAIL] 合并版有多余权重: {unexpected}"
    print("[GATE OK] state_dict 重映射完整，无缺失/多余键")

    # 逐位等价性：随机输入帧
    worst_hm = worst_off = 0.0
    for i in range(args.frames):
        x = torch.randn(1, 3, args.imgsz, args.imgsz)
        with torch.no_grad():
            a, b = base(x), merged(x)
        worst_hm = max(worst_hm, (a["heatmap"] - b["heatmap"]).abs().max().item())
        worst_off = max(worst_off, (a["offset"] - b["offset"]).abs().max().item())
        print(f"  frame {i:02d}: heatmap max|d| = {(a['heatmap'] - b['heatmap']).abs().max().item():.3e}"
              f"   offset max|d| = {(a['offset'] - b['offset']).abs().max().item():.3e}")

    ok = worst_hm == 0.0 and worst_off == 0.0
    print(f"\n{'[GATE PASS]' if ok else '[GATE FAIL]'} "
          f"heatmap max|d|={worst_hm:.3e}  offset max|d|={worst_off:.3e}")
    print("  说明: PyTorch FP32 CPU 上逐位相同；**不代表 RKNN/NPU 上逐位相同**"
          "（Cout 64->128 会改变 NPU 分块调度），上板后仍须做逐帧张量对账。\n")

    n0, g0 = conv_inventory(base, (1, 3, args.imgsz, args.imgsz))
    n1, g1 = conv_inventory(merged, (1, 3, args.imgsz, args.imgsz))
    print(f"=== 结构对比 (stride={args.stride}, imgsz={args.imgsz}, scale={args.scale}) ===")
    print(f"  未合并: {n0} 个卷积, {g0:.2f} G MAC")
    print(f"  合并后: {n1} 个卷积, {g1:.2f} G MAC")
    print(f"  层数 {n0} -> {n1} ({n0 - n1} 个 3x3 被并成 1 个)")
    print(f"  MAC  {100 * (g0 - g1) / g0:+.2f}%  (预期 ~0: 合并只省输入读取，不省乘法)")
    print(f"  省下: 读 320x320x64 特征 1 次 ≈ {(args.imgsz // 2) ** 2 * 64 * 4 / 1024 ** 2:.1f} MiB (FP32)"
          if args.stride == 2 else "  (stride=4 时特征图为 160x160，收益按面积缩放)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
