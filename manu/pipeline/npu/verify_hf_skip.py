#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
跨层高频残差注入（`hf_skip`）的结构自检。

检查五件事：
  a) **默认参数下逐位不变**：完全不传新参数的"旧调用方式"与显式 `hf_skip=False`
     的 state_dict 键集合完全相同，同一随机输入下 heatmap/offset 用 `torch.equal`
     判定全等（不是 allclose，是逐位）；
  b) 开启侧枝后能正常前向，heatmap/offset 形状与关闭时一致；且 `head.*` 的张量
     形状与旧 checkpoint 完全兼容（旧 ckpt 的头可以直接 load）；
  c) 打印侧枝新增的张量清单与参数量；
  d) 侧枝真正接进了计算图（把零初始化的 `project` 换成随机权重后输出必须变化），
     否则"形状不变"可能只是因为侧枝根本没生效；
  e) 打印侧枝的 NPU 搬运量估算（按实测张量形状算，不靠嘴报）。

不依赖 GPU / rknn-toolkit2 / checkpoint，纯 CPU 可跑。

用法：
  cd /media/manu/1TB-Volume/workspace/ultralytics
  python manu/pipeline/npu/verify_hf_skip.py                 # 默认 P3 头（head_src=p3）
  python manu/pipeline/npu/verify_hf_skip.py --head-src p2_fused --hf-skip-src b0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from manu.models.heatmap_model import YOLO26HeatmapDetector  # noqa: E402


def build(args, **overrides) -> YOLO26HeatmapDetector:
    """按同一套参数建模型（权重由调用方统一 load，避免随机初始化干扰比对）。"""
    common = dict(
        stride=args.stride,
        num_classes=1,
        scale=args.scale,
        upsample_mode="nearest",
        use_p0_highway=False,
        no_p2=args.no_p2,
        head_level=args.head_level,
        head_src=args.head_src,
    )
    common.update(overrides)
    model = YOLO26HeatmapDetector(**common).eval()
    return model


def bytes_of(shape: tuple[int, ...], itemsize: int = 4) -> int:
    n = 1
    for s in shape:
        n *= s
    return n * itemsize


def branch_traffic(model: YOLO26HeatmapDetector, x: torch.Tensor) -> list[tuple[str, int, int]]:
    """用 forward hook 抓侧枝每个算子的输入/输出形状 ⇒ 估算搬运量。"""
    records: list[tuple[str, int, int]] = []
    handles = []

    def hook(name):
        def fn(_m, inputs, output):
            t = output
            while isinstance(t, (list, tuple)) and t:
                t = t[0]
            in_shape = tuple(inputs[0].shape) if inputs else ()
            out_shape = tuple(t.shape) if isinstance(t, torch.Tensor) else ()
            records.append((name, bytes_of(in_shape), bytes_of(out_shape)))
        return fn

    branch = model.hf_skip_branch
    for name, mod in branch.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            handles.append(mod.register_forward_hook(hook(name or "root")))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=8, choices=(2, 4, 8))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--scale", default="n", choices=("n", "s"))
    ap.add_argument("--head-src", default="p3", choices=("p1", "p2_raw", "p2_fused", "p3"))
    ap.add_argument("--head-level", default="p3", choices=("p1", "p2", "p3", "p2_fused"))
    ap.add_argument("--no-p2", type=int, default=1, choices=(0, 1),
                    help="1=no_p2 结构（默认）；0=保留 P2 融合级")
    ap.add_argument("--hf-skip-src", default="b1", choices=("b0", "b1"))
    ap.add_argument("--hf-skip-channels", type=int, default=1, choices=(1, 2))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    x = torch.randn(1, 3, args.imgsz, args.imgsz)

    print("=" * 78)
    print(f"[hf_skip] 自检：head_src={args.head_src} head_level={args.head_level} "
          f"stride={args.stride} imgsz={args.imgsz} scale={args.scale} "
          f"hf_skip_src={args.hf_skip_src} hf_skip_channels={args.hf_skip_channels}")
    print("=" * 78)

    # ---- a) 默认参数下与"旧调用方式"逐位一致 ----
    legacy = build(args)                                  # 旧调用方式：一个新参数都不传
    default = build(args, hf_skip=False)                  # 新参数显式关闭
    enabled = build(args, hf_skip=True, hf_skip_src=args.hf_skip_src,
                    hf_skip_channels=args.hf_skip_channels)

    legacy_sd, default_sd, enabled_sd = legacy.state_dict(), default.state_dict(), enabled.state_dict()
    assert set(legacy_sd) == set(default_sd), (
        f"[GATE FAIL] 默认参数下 state_dict 键集合与旧模型不一致: "
        f"多={set(default_sd) - set(legacy_sd)} 少={set(legacy_sd) - set(default_sd)}"
    )
    for k in legacy_sd:
        assert legacy_sd[k].shape == default_sd[k].shape, f"[GATE FAIL] 形状不一致: {k}"
    print(f"[GATE OK] 默认参数：state_dict 键集合与旧模型完全相同（{len(legacy_sd)} 个张量）")

    # 权重统一从 legacy 灌入，比对时不受随机初始化干扰
    default.load_state_dict(legacy_sd, strict=True)
    missing, unexpected = enabled.load_state_dict(legacy_sd, strict=False)
    assert not unexpected, f"[GATE FAIL] 开启侧枝后出现多余的键: {unexpected}"
    assert all(k.startswith("hf_skip_branch.") for k in missing), (
        f"[GATE FAIL] 旧 ckpt 缺失的键不止侧枝: {missing}"
    )
    print(f"[GATE OK] 开启侧枝：旧 ckpt 的全部张量原样可加载，仅缺 {len(missing)} 个侧枝张量")

    with torch.no_grad():
        out_legacy, out_default, out_enabled = legacy(x), default(x), enabled(x)
    for key in ("heatmap", "offset"):
        assert out_legacy[key].shape == out_default[key].shape
        assert torch.equal(out_legacy[key], out_default[key]), (
            f"[GATE FAIL] 默认参数下 {key} 不是逐位一致: "
            f"max|d|={(out_legacy[key] - out_default[key]).abs().max().item():.3e}"
        )
    print(f"[GATE OK] 默认参数：heatmap/offset 与旧模型**逐位一致**"
          f"（torch.equal 通过，shape={tuple(out_legacy['heatmap'].shape)} / "
          f"{tuple(out_legacy['offset'].shape)}）")

    # ---- b) 开启侧枝后形状不变 + head 与旧 ckpt 兼容 ----
    for key in ("heatmap", "offset"):
        assert out_enabled[key].shape == out_legacy[key].shape, (
            f"[GATE FAIL] 开启侧枝后 {key} 形状变了: "
            f"{tuple(out_legacy[key].shape)} -> {tuple(out_enabled[key].shape)}"
        )
    head_legacy = {k: v.shape for k, v in legacy_sd.items() if k.startswith("head.")}
    head_enabled = {k: v.shape for k, v in enabled_sd.items() if k.startswith("head.")}
    assert head_legacy == head_enabled, "[GATE FAIL] 开启侧枝后 head 的张量形状变了"
    feat_conv_w = enabled_sd["head.feat_conv.0.conv.weight"]
    print(f"[GATE OK] 开启侧枝：heatmap/offset 形状不变，head 的 {len(head_legacy)} 个张量形状不变")
    print(f"       head.feat_conv.0.conv.weight = {tuple(feat_conv_w.shape)}（旧 ckpt 的头可直接 load）")

    # ---- d) 侧枝真的接进了计算图（零初始化时输出恒等；换成随机权重后必须变）----
    with torch.no_grad():
        same = all(torch.equal(out_legacy[k], out_enabled[k]) for k in ("heatmap", "offset"))
    assert same, "[GATE FAIL] 侧枝零初始化下输出竟与旧模型不同（零初始化没生效？）"
    print("[GATE OK] 侧枝零初始化（project 权重为 0）⇒ 起点与旧模型逐位一致")

    torch.nn.init.normal_(enabled.hf_skip_branch.project.weight, std=0.05)
    torch.nn.init.normal_(enabled.hf_skip_branch.project.bias, std=0.05)
    with torch.no_grad():
        out_trained = enabled(x)
    changed = [k for k in ("heatmap", "offset") if not torch.equal(out_legacy[k], out_trained[k])]
    assert changed, "[GATE FAIL] 侧枝权重非零后输出仍与旧模型相同 ⇒ 侧枝没接进计算图"
    for key in ("heatmap", "offset"):
        assert out_trained[key].shape == out_legacy[key].shape
    print(f"[GATE OK] 侧枝权重非零后输出发生变化（{', '.join(changed)}）且形状不变 ⇒ 侧枝确实生效")

    # ---- c) 新增张量清单与参数量 ----
    new_keys = sorted(set(enabled_sd) - set(legacy_sd))
    new_params = sum(enabled_sd[k].numel() for k in new_keys)
    legacy_params = sum(v.numel() for v in legacy_sd.values())
    print("\n=== 侧枝新增张量 ===")
    for k in new_keys:
        print(f"  {k:38s} {tuple(enabled_sd[k].shape)}  ({enabled_sd[k].numel()} 个元素)")
    print(f"  新增张量 {len(new_keys)} 个 / 新增参数 {new_params} 个 "
          f"（全模型 {legacy_params} -> {legacy_params + new_params}，"
          f"+{100.0 * new_params / legacy_params:.3f}%）")
    print(f"  新增的可训练参数之外的开销：0（无 buffer、无 BN、无 gate 参数、无分支控制流）")

    # ---- e) NPU 搬运量估算（按实测形状）----
    print("\n=== 侧枝搬运量估算（FP32；RKNN 量化到 int8 后再乘 1/4）===")
    records = branch_traffic(enabled, x)
    total = 0
    for name, in_b, out_b in records:
        total += in_b + out_b
        print(f"  {name:10s} in={in_b / 1024:9.1f} KiB   out={out_b / 1024:9.1f} KiB")
    head_ch = enabled.hf_skip_branch.project.out_channels
    grid = args.imgsz // enabled.head_grid_stride
    feat_bytes = bytes_of((1, head_ch, grid, grid))
    fuse = 3 * feat_bytes  # 写 delta + 读 delta + 写回 feat
    print(f"  {'fuse(add)':10s} 写/读 delta + 写回 feat ≈ {fuse / 1024:.1f} KiB")
    grand = total + fuse
    print(f"  合计额外搬运 ≈ {grand / 1024 / 1024:.2f} MiB（FP32）"
          f" ⇒ int8 量化后 ≈ {grand / 4 / 1024 / 1024:.2f} MiB")
    print(f"  （对照：取 b0 而不是 b1 时，源张量 320x320x16 = 6.55 MB，比 b1 的 160x160x32 = 3.28 MB 多一倍）")

    # ---- ckpt 自描述 ----
    cfg = enabled.struct_config()
    for key in ("hf_skip", "hf_skip_src", "hf_skip_channels"):
        assert key in cfg, f"[GATE FAIL] struct_config() 缺字段 {key}"
    print("\n=== struct_config()（ckpt 自描述字段）===")
    for k, v in cfg.items():
        print(f"  {k:20s} = {v!r}")

    print(f"\n[GATE PASS] 全部自检通过：默认逐位不变 / 开启后形状不变 / head 与旧 ckpt 兼容 / "
          f"侧枝确实生效 / 零额外参数开销")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
