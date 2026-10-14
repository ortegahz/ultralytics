#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Exact Conv/Linear MAC count for the Trial-0474 heatmap detector, plus the
NPU-efficiency view of where those MACs land.

Why this exists
---------------
The board measured 151.7 ms for `rknn_run` on a 7.4 MB checkpoint. That is
~20x slower than one would guess from parameter count alone, and the natural
reaction is "something fell back to CPU". Before blaming the port, the model
has to be priced properly.

Counting MACs from the ONNX graph is unreliable here: `onnx.shape_inference`
resolves only 11 of 91 Conv nodes (the rest sit behind ops it cannot propagate
through), which undercounts the network by an order of magnitude and quietly
produces a reassuring number. Registering forward hooks on the live PyTorch
modules reports the true shape of every Conv as it actually executes, so the
count cannot be wrong in the flattering direction.

Efficiency view
---------------
An NPU is not a generic multiplier. Its MAC units are tiled for wide channel
counts (64/128/256); a 3x3 convolution on a 16-channel tensor spends most of
its lanes on padding. So this script also reports the MAC-weighted channel
histogram, because "4.6 GFLOPs at 320x320 with C=16" and "4.6 GFLOPs at 20x20
with C=256" cost very different amounts of time on the same silicon.

Usage:
    python hm_flops.py [--weights PATH] [--imgsz 640]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# RK3588 NPU: 3 cores, ~2 TOPS int8 each (vendor figure). Used only to turn the
# MAC count into an order-of-magnitude expectation, never as a claim of
# achieved throughput.
RK3588_INT8_TOPS = 6.0


def hook_all(model: nn.Module) -> tuple[list, list]:
    """Collect (name, in_shape, out_shape, kernel, macs) for Conv2d and Linear.

    The model wraps every conv as `Conv -> [Conv2d, BatchNorm2d, act]`, so the
    hooks must go on the inner ``nn.Conv2d`` leaves. Iterating ``named_modules``
    and testing ``isinstance`` finds them, but the rows are only appended when a
    forward actually runs -- an earlier version of this script printed an empty
    table because the hooks were registered on a *copy* of the module list and
    the closure appended to a different list than the one returned.
    """
    conv_rows: list = []
    lin_rows: list = []

    for name, mod in model.named_modules():
        if not isinstance(mod, (nn.Conv2d, nn.Linear)):
            continue
        mod._flops_name = name
        if isinstance(mod, nn.Conv2d):
            def conv_hook(m, i, o):
                N, C, H, W = i[0].shape
                OC = o.shape[1]
                k = m.kernel_size
                conv_rows.append((m._flops_name, (N, C, H, W), tuple(o.shape),
                                  (k[0], k[1]), N * OC * H * W * C * k[0] * k[1]))

            def lin_hook(m, i, o):
                lin_rows.append((m._flops_name, tuple(i[0].shape), tuple(o.shape),
                                 o.numel() * i[0].shape[-1]))

            mod.register_forward_hook(conv_hook)
        else:
            mod.register_forward_hook(lin_hook)
    return conv_rows, lin_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",
                    default="/home/manu/mnt/pycharm_project_10ae9e2e/runs/optuna_p0_nas/"
                            "trial_0474/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from manu.pipeline.npu.hm_export_onnx import build_model

    model, ckpt = build_model(args.weights)
    assert not model.training, "FLOPs counting must run in eval mode"
    model.eval()

    n_param = sum(p.numel() for p in model.parameters())
    # Must take hook_all's return value: the hooks append to the lists created
    # inside hook_all, so a local `conv_rows, lin_rows = [], []` here would stay
    # empty forever and the script would print an all-zero table.
    conv_rows, lin_rows = hook_all(model)
    n_hooked = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))

    x = torch.zeros(1, 3, args.imgsz, args.imgsz, dtype=torch.float32)
    with torch.no_grad():
        t0 = time.perf_counter()
        model(x)
        cpu_ms = (time.perf_counter() - t0) * 1000.0

    # The emptiness check must come AFTER the forward: hooks only append when a
    # conv actually runs, so checking right after registration always fails.
    if not conv_rows:
        raise SystemExit(f"[FATAL] {n_hooked} Conv2d hooks registered but none fired "
                         f"-- the MAC table below would be silently zero")

    # A module invoked several times appends one row per call; that is intended
    # -- the cost is per execution, not per parameter.
    conv_macs = sum(r[4] for r in conv_rows)
    lin_macs = sum(r[3] for r in lin_rows)
    total_macs = conv_macs + lin_macs

    print(f"\n=== Trial 0474 计算量 ({args.imgsz}x{args.imgsz} int8 目标) ===")
    print(f"参数量           : {n_param/1e6:.2f} M")
    print(f"Conv2d 调用次数  : {len(conv_rows)}   MACs = {conv_macs/1e9:.3f} G")
    print(f"Linear 调用次数  : {len(lin_rows)}   MACs = {lin_macs/1e6:.1f} M")
    print(f"总 MACs          : {total_macs/1e9:.3f} G")
    print(f"总 FLOPs(2xMACs) : {2*total_macs/1e9:.3f} G")

    print(f"\n=== MACs 最大的 15 个 Conv 调用点 ===")
    print(f"{'module':<44}{'in':<22}{'out':<22}{'k':<8}{'M MAC':>9}")
    for nm, isz, osz, k, mac in sorted(conv_rows, key=lambda r: -r[4])[:15]:
        print(f"{nm[:43]:<44}{str(isz):<22}{str(osz):<22}{f'{k[0]}x{k[1]}':<8}{mac/1e6:9.1f}")

    by_res = defaultdict(float)
    for nm, isz, osz, k, mac in conv_rows:
        by_res[isz[2]] += mac
    print(f"\n=== MACs 按输入特征图分辨率分布 ===")
    for hw in sorted(by_res, reverse=True):
        print(f"  {hw:>4} x {hw:<4}: {by_res[hw]/1e9:7.3f} G  ({100*by_res[hw]/conv_macs:5.1f}%)")

    hi = by_res.get(args.imgsz, 0.0) + by_res.get(args.imgsz // 2, 0.0)
    print(f"\n  => {args.imgsz} 与 {args.imgsz//2} 分辨率合计占 {100*hi/conv_macs:.1f}% 的 Conv MACs")

    by_c = defaultdict(float)
    for nm, isz, osz, k, mac in conv_rows:
        by_c[isz[1]] += mac
    print(f"\n=== MACs 按输入通道数分布（NPU 效率的关键）===")
    print("  NPU 的 MAC 单元按 64/128/256 通道铺开；通道少的卷积大量算力落在 padding 上。")
    for c in sorted(by_c):
        w = ("窄 <=16  ← NPU 最不划算" if c <= 16 else
             "中 32-64" if c <= 64 else "宽 >=128")
        print(f"  C={c:<5}: {by_c[c]/1e9:7.3f} G  ({100*by_c[c]/conv_macs:5.1f}%)  {w}")

    print(f"\n=== 与板卡实测对照 ===")
    gflops = 2 * total_macs / 1e9
    ideal_ms = gflops / (RK3588_INT8_TOPS * 1e3) * 1e3
    print(f"  总 FLOPs               : {gflops:.2f} G")
    print(f"  RK3588 int8 理论下限   : {ideal_ms:.3f} ms  (假定 {RK3588_INT8_TOPS:.0f} TOPS 且 100% 利用率)")
    print(f"  板卡实测 rknn_run      : 151.7 ms")
    print(f"  等效算力               : {gflops/1e3/(151.7/1e3):.3f} TOPS  "
          f"= 理论峰值的 {100*gflops/1e3/151.7/RK3588_INT8_TOPS:.2f}%")
    print(f"  本机 CPU fp32 单次前向 : {cpu_ms:.1f} ms")


if __name__ == "__main__":
    main()
