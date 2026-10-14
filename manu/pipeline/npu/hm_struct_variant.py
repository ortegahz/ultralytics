#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Build structural variants of the Trial-0474 heatmap detector for NPU latency.

Why this exists
---------------
Measured on RK3588 (0 CPU ops, v2 depthwise build): 103.7 ms for 25.54 GMAC.
Two independent methods agree on the MAC count (PyTorch forward hooks and the
RKNN build log's operator table), and they agree that the cost is concentrated
in a handful of 320x320 / C=64 3x3 convolutions:

    head.feat_conv.1   64->64 3x3 @320   3774.9 M   14.8%
    head.heatmap.0     64->64 3x3 @320   3774.9 M   14.8%
    head.offset.0      64->64 3x3 @320   3774.9 M   14.8%
    fuse_p1.conv       80->48 3x3 @320   3538.9 M   13.9%
    head.feat_conv.0   48->64 3x3 @320   2831.2 M   11.1%
    ------------------------------------------------------------
    the heatmap head alone                              10381 M   40.6%

That is the worst possible shape for an NPU: high spatial resolution (so every
operand is a full-frame DDR round trip) with a channel count (64) that leaves the
64-wide MAC units partially idle on 3x3 kernels.

This script enumerates structural cuts and emits one ONNX per variant so each
can be converted and timed on the board on its own. It does NOT care about
accuracy -- the point is to find the latency/structure curve, after which the
chosen shape is fine-tuned.

Variants
--------
    base          as-is (stride 2, head k=3, p0_highway on)
    no_p0         drop the P0 residual highway (a later-added side branch: see
                  memory/progressive_residual_training.md -- base frozen, branch
                  zero-init gated, measured gain +0.15pp recall, F1 -0.0002)
    head1x1       head's four 3x3 convs -> 1x1 (9x fewer MACs at same resolution)
    s4            stride 4: drop the P1 upsample, run the head at 160x160
    s4_head1x1    stride 4 + 1x1 head
    s4_head1x1_nop0   all three

Usage:
    python hm_struct_variant.py --list
    python hm_struct_variant.py --variant s4_head1x1_nop0 --out /path/x.onnx
    python hm_struct_variant.py --variant all --outdir /path/dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

DEFAULT_W = ("/home/manu/mnt/pycharm_project_10ae9e2e/runs/optuna_p0_nas/"
             "trial_0474/weights/best.pt")
WORK = Path("/media/manu/1TB-Volume/rknn_work/hm")

VARIANTS = {
    "base":              dict(stride=2, p0=True,  head_k3=True),
    "no_p0":             dict(stride=2, p0=False, head_k3=True),
    "head1x1":           dict(stride=2, p0=True,  head_k3=False),
    "no_p0_head1x1":     dict(stride=2, p0=False, head_k3=False),
    # stride 4 forces p0 off: P0ResidualHighway downsamples 640->320 exactly
    # once, so its residual is always stride 2 and cannot be added to a 160x160
    # neck feature. That coupling is itself a finding -- the highway pins the
    # whole network to stride 2 and therefore to 320x320 head convolutions.
    "s4":                dict(stride=4, p0=False, head_k3=True),
    "s4_head1x1":        dict(stride=4, p0=False, head_k3=False),
    # Input resolution is a first-class lever here: the network is fully
    # convolutional, so halving the input halves H and W everywhere and quarters
    # every conv. 320x320 input with stride 4 puts the head at 80x80.
    "in320_s4_head1x1":  dict(stride=4, p0=False, head_k3=False, imgsz=320),
    "in320_s2_head1x1":  dict(stride=2, p0=False, head_k3=False, imgsz=320),
}

DEFAULT_CFG = dict(stride=2, p0=True, head_k3=True, imgsz=640)


class OnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["heatmap"], out["offset"]


def set_head_kernel(head: nn.Module, k: int) -> None:
    """Rewrite the head's 3x3 convs as 1x1 (k=3 keeps them as-is).

    Only the spatial kernel changes; channels, activation order and the final 1x1
    classifiers are untouched. Weights are re-initialised because the shape no
    longer matches the checkpoint -- accuracy is explicitly out of scope here,
    the shape is what is being priced.

    nn.Sequential does not support item assignment, so each affected block is
    rebuilt as a new Sequential rather than patched in place.
    """
    if k == 3:
        return
    pad = 0
    # feat_conv holds ultralytics `Conv` wrappers (Conv2d + BN + act), while
    # heatmap/offset hold bare nn.Conv2d. Both shapes must be handled or the
    # 3x3s in feat_conv silently stay 3x3 -- the table would then claim a 9x
    # saving that was never applied.
    n_rewritten = 0
    for name in ("feat_conv", "heatmap", "offset"):
        seq = getattr(head, name)
        new_layers = []
        for m in seq:
            inner = getattr(m, "conv", None)  # Conv wrapper, else None
            target = inner if isinstance(inner, nn.Conv2d) else \
                (m if isinstance(m, nn.Conv2d) else None)
            if target is not None and tuple(target.kernel_size) == (3, 3):
                conv = nn.Conv2d(target.in_channels, target.out_channels,
                                 kernel_size=k, padding=pad)
                nn.init.kaiming_normal_(conv.weight, mode="fan_out",
                                        nonlinearity="relu")
                if conv.bias is not None:
                    nn.init.zeros_(conv.bias)
                if inner is not None:
                    m.conv = conv          # keep BN + act around it
                else:
                    m = conv              # bare conv -> replace the slot
                n_rewritten += 1
            new_layers.append(m)
        setattr(head, name, nn.Sequential(*new_layers))
    with torch.no_grad():
        head.heatmap[-1].bias.fill_(-2.19)
        head.offset[-1].bias.zero_()
    print(f"  [head ] rewrote {n_rewritten} conv(s) 3x3 -> 1x1")


def build_variant(weights: str, stride: int, use_p0: bool, head_k3: bool):
    """Construct the variant and report which checkpoint tensors actually load."""
    from manu.models.heatmap_model import YOLO26HeatmapDetector

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt

    model = YOLO26HeatmapDetector(
        stride=stride,
        num_classes=1,
        temporal_mode="standard",
        use_p0_highway=use_p0,
        p0_highway_kwargs=ckpt["p0_kwargs"] if use_p0 else None,
    )
    own = model.state_dict()

    # Strict loading is impossible by construction here: dropping p0_highway
    # leaves its tensors unmatched, and stride 4 changes the P1/head channel
    # widths. Report the load rate instead of pretending it is exact.
    #
    # NOTE: the loop variable comes from `own`, so the comparison MUST be
    # against sd[k]. An earlier version wrote `own[k].shape != v.shape`, where
    # v *is* own[k] -- always False -- which made the mismatch count read 0 and
    # turned the copy guard below into "always copy", so the stride-4 variant
    # died in copy_ with a 64-vs-48 size error while the report claimed the load
    # was perfect.
    loaded = missing = mism = 0
    for k in own:
        if k not in sd:
            missing += 1
        elif own[k].shape != sd[k].shape:
            mism += 1
        else:
            loaded += 1
    unexpected = sum(1 for k in sd if k not in own)
    print(f"  [load] matched {loaded}/{len(own)}  missing {missing}  "
          f"shape-mismatch {mism}  ckpt-unused {unexpected}")

    with torch.no_grad():
        skipped = 0
        for k, v in own.items():
            if k not in sd:
                continue
            if own[k].shape == sd[k].shape:
                own[k].copy_(sd[k])
            else:
                skipped += 1
    model.load_state_dict(own, strict=True)
    if skipped:
        print(f"  [load] {skipped} tensor(s) kept the model's own init "
              f"(shape differs from the stride-2 checkpoint)")
    set_head_kernel(model.head, 3 if head_k3 else 1)
    model.eval()
    assert not model.training
    return model


def export(model, out: Path, opset: int = 12, imgsz: int = 640) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    w = OnnxWrapper(model)
    torch.onnx.export(
        w,
        torch.zeros(1, 3, imgsz, imgsz),
        str(out),
        opset_version=opset,
        input_names=["images"],
        output_names=["heatmap", "offset"],
        do_constant_folding=True,
        dynamic_axes=None,
    )
    model.eval()  # export leaves it in train mode; undo that
    print(f"  [onnx ] {out}  {out.stat().st_size/1e6:.2f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_W)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--out", default=None)
    ap.add_argument("--outdir", default=str(WORK / "models/struct"))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, v in VARIANTS.items():
            print(f"  {k:<20} stride={v['stride']}  p0={v['p0']}  head_k3={v['head_k3']}")
        return

    names = list(VARIANTS) if args.variant == "all" else [args.variant]
    for name in names:
        if name not in VARIANTS:
            raise SystemExit(f"[FATAL] unknown variant {name}; try --list")
        cfg = dict(DEFAULT_CFG, **VARIANTS[name])
        print(f"\n=== {name}  stride={cfg['stride']} p0={cfg['p0']} "
              f"head_k3={cfg['head_k3']} imgsz={cfg['imgsz']} ===")
        model = build_variant(args.weights, cfg["stride"], cfg["p0"], cfg["head_k3"])
        out = Path(args.out) if (args.out and len(names) == 1) \
            else Path(args.outdir) / f"hm_{name}.onnx"
        export(model, out, imgsz=cfg["imgsz"])


if __name__ == "__main__":
    main()
