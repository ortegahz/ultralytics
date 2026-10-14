#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Export the Trial-0474 heatmap SOTA detector to ONNX for RKNN conversion.

Why a fresh export instead of reusing runs/onnx/yolo26s_p1_heatmap.onnx:
that file is dated 2026-09-11 and was exported by manu/inference/export_onnx.py
with a *different* checkpoint (runs/scale_ab_test/yolo26s_p2_5ep/.../best_f1.pt)
and without the P0 residual highway. The checkpoint itself carries the ground
truth, so nothing here is hard-coded:

    ckpt["stride"]      = 2
    ckpt["imgsz"]       = 640
    ckpt["p0_kwargs"]   = the P0 highway micro-architecture
    ckpt["metrics"]     = the frozen single-frame baseline we must reproduce

The script asserts that all 523 tensors load (523/523 matched => scale='n'),
exports a two-output graph, and then cross-checks ONNX Runtime against PyTorch
on a real pre-built feature JPEG before declaring success.

Usage:
    python hm_export_onnx.py --out runs/onnx/trial0474_hm.onnx
    python hm_export_onnx.py --check-image <path/to/feature.jpg>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_W = "/home/manu/mnt/pycharm_project_10ae9e2e/runs/optuna_p0_nas/trial_0474/weights/best.pt"
DEFAULT_IMG = ("/home/manu/mnt/data/siping/datasets/manu/uav_gmc_median/images/val/"
               "01_4485_1167-2666__000100.jpg")
DEFAULT_OUT = REPO_ROOT / "runs/onnx/trial0474_hm.onnx"


class OnnxWrapper(torch.nn.Module):
    """Flatten the dict output to a (heatmap, offset) tuple for a clean graph."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["heatmap"], out["offset"]


def build_model(ckpt_path: str, ablate_p0: bool = False):
    """Rebuild the detector from the checkpoint.

    ``ablate_p0=True`` reproduces ``manu/inference/cache_trial0474_inferences.py
    --ablate-p0``: the model is built with ``use_p0_highway=False``, every
    ``p0_highway.*`` tensor is skipped, and the detector becomes exactly the
    frozen base path. Verified bit-identical to that script's construction
    (494/494 tensors equal, heatmap/offset max|diff| = 0.0).

    The GATE is adopted from that script deliberately: an ablation that silently
    fails to take effect would still print a lower F1 and look like a real
    result. Failing loudly is the only safe behaviour.
    """
    import torch

    from manu.models.heatmap_model import YOLO26HeatmapDetector

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    p0_kwargs = ckpt["p0_kwargs"]
    stride = ckpt.get("stride", 2)
    sd = ckpt["model"] if "model" in ckpt else ckpt

    model = YOLO26HeatmapDetector(
        stride=stride,
        num_classes=1,
        temporal_mode="standard",
        use_p0_highway=not ablate_p0,
        p0_highway_kwargs=p0_kwargs,
    )
    own = model.state_dict()

    # Key normalisation matches cache_trial0474_inferences.py: a checkpoint saved
    # from a wrapped module may carry module./model.model./model. prefixes.
    matched = skipped = n_p0_skipped = 0
    with torch.no_grad():
        for k, v in sd.items():
            clean = k.replace("module.", "").replace("model.model.", "").replace("model.", "")
            if clean in own and own[clean].shape == v.shape:
                own[clean].copy_(v)
                matched += 1
            else:
                skipped += 1
                if clean.startswith("p0_highway"):
                    n_p0_skipped += 1

    if ablate_p0:
        if model.p0_highway is not None:
            raise RuntimeError("[ABLATE-P0 GATE] model.p0_highway is not None "
                               "-- ablation did not take effect")
        if n_p0_skipped == 0:
            raise RuntimeError("[ABLATE-P0 GATE] 0 p0_highway.* tensors were dropped "
                               "-- nothing was ablated")
        print(f"[ABLATE-P0 GATE] PASS: p0_highway=None, {n_p0_skipped} p0 tensors dropped")

    if not ablate_p0:
        missing = [k for k in own if k not in sd]
        mismatch = [k for k, v in sd.items() if k in own and own[k].shape != v.shape]
        if missing or mismatch:
            raise SystemExit(
                f"[FATAL] checkpoint does not fit the reconstructed model: "
                f"{len(missing)} missing, {len(mismatch)} shape-mismatched "
                f"(first missing: {missing[:3]}, first mismatch: {mismatch[:3]})"
            )
    model.load_state_dict(own, strict=True)
    model.eval()

    n_p0 = (sum(p.numel() for p in model.p0_highway.parameters())
            if model.p0_highway is not None else 0)
    n_all = sum(p.numel() for p in model.parameters())
    gate = (f"{float(model.p0_highway.gate):.6f}"
            if model.p0_highway is not None else "n/a (highway removed)")
    print(f"[model] {ckpt_path}")
    print(f"[model] stride={stride} imgsz={ckpt.get('imgsz')} p0_kwargs={p0_kwargs}")
    print(f"[model] ablate_p0={ablate_p0}")
    print(f"[model] gate alpha = {gate}")
    print(f"[model] P0 highway trainable params = {n_p0}  (total {n_all})")
    print(f"[model] metrics in checkpoint = {ckpt.get('metrics')}")
    print(f"[model] state_dict: matched {matched}, skipped {skipped} (own keys {len(own)})")
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_W)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--check-image", default=DEFAULT_IMG)
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--tol-hm", type=float, default=1e-4,
                    help="max allowed |ONNX - PyTorch| on the heatmap")
    ap.add_argument("--tol-off", type=float, default=1e-3,
                    help="max allowed |ONNX - PyTorch| on the offset")
    ap.add_argument("--ablate-p0", action="store_true",
                    help="drop the P0 residual highway (same semantics as "
                         "manu/inference/cache_trial0474_inferences.py --ablate-p0, "
                         "including its GATE). The detector becomes the frozen "
                         "base path; every p0_highway.* tensor is skipped.")
    ap.add_argument("--keep-pixel-unshuffle", action="store_true",
                    help="leave nn.PixelUnshuffle alone (default is to rewrite it "
                         "as a fixed stride-2 depthwise conv, which keeps the "
                         "shuffle on the NPU)")
    args = ap.parse_args()

    import numpy as np

    from manu.pipeline.npu.hm_common import letterbox_to_input
    from manu.pipeline.npu.hm_pixelshuffle import swap_to_strided_conv

    model, _ = build_model(args.weights, ablate_p0=args.ablate_p0)
    assert not model.training, "model must be in eval mode before anything else"

    # nn.PixelUnshuffle exports as reshape + permute + reshape. On (1,16,640,640)
    # rknn-toolkit2 needs a Transpose whose 2-D product is 640*640 = 409600,
    # against a hard limit of 16384, so it drops to the CPU:
    #     W RKNN: Transpose will fallback to CPU, because input shape has
    #            exceeded the max limit, height(4) * width(102400) = 409600,
    #            required product no larger than 16384!
    # Rewriting the same maths as strided slices does NOT help -- RKNN emulates
    # a strided Slice with a Transpose too, so it lands on (1,16,1280,320), the
    # same 409600 product, plus 6 extra CPU Slice ops (measured: 4 -> 10 CPU ops).
    # A stride-2 DEPTHWISE CONV is a native RKNN operator: it reads a window and
    # writes a smaller map without any transpose, and with groups=C its output
    # channels are already numbered c*f*f + i*f + j, matching PixelUnshuffle.
    # Bit-exact (hm_pixelshuffle.py --verify); the 0/1 weights are not learned.
    if args.keep_pixel_unshuffle:
        print("[onnx ] keeping nn.PixelUnshuffle as-is (--keep-pixel-unshuffle); "
              "expect an RKNN Transpose CPU fallback")
    else:
        n_swap = swap_to_strided_conv(model)
        print(f"[onnx ] rewrote {n_swap} nn.PixelUnshuffle -> fixed stride-2 "
              f"depthwise conv (bit-exact, keeps the shuffle on the NPU)")
        model.eval()  # the swap replaced modules; re-assert eval mode

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = OnnxWrapper(model)

    # ---- reference FIRST, while the model is provably in eval mode -----------
    # torch.onnx.export() leaves the module it was handed in TRAIN mode, so any
    # comparison taken after the export silently compares an eval-mode ONNX
    # graph against a train-mode PyTorch model -- i.e. BatchNorm running stats
    # vs. batch stats of the test image. That mismatch is ~0.28 on the heatmap
    # and ~1.7 on the offset, which reads exactly like "the export is broken"
    # and is not. Taking the reference first removes the trap entirely.
    ref = None
    if not args.skip_check:
        x = torch.from_numpy(letterbox_to_input(args.check_image))
        with torch.no_grad():
            ref = [t.detach().clone() for t in wrapper(x)]
        print(f"\n[ref  ] PyTorch eval-mode reference captured on {args.check_image}")

    print(f"[onnx ] exporting -> {out_path} (opset {args.opset})")
    torch.onnx.export(
        wrapper,
        torch.zeros(1, 3, 640, 640),
        str(out_path),
        opset_version=args.opset,
        input_names=["images"],
        output_names=["heatmap", "offset"],
        do_constant_folding=True,
        dynamic_axes=None,
    )
    if model.training:
        print("[warn ] torch.onnx.export left the module in TRAIN mode; forcing eval() back "
              "-- use this module for anything downstream, or BatchNorm will silently "
              "use per-batch statistics.")
        model.eval()
    print(f"[onnx ] wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")

    import onnx

    m = onnx.load(str(out_path))
    onnx.checker.check_model(m)
    for i in m.graph.input:
        dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in i.type.tensor_type.shape.dim]
        print(f"[onnx ] input  {i.name}: {dims}")
    for o in m.graph.output:
        dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in o.type.tensor_type.shape.dim]
        print(f"[onnx ] output {o.name}: {dims}")
    ops = sorted({n.op_type for n in m.graph.node})
    print(f"[onnx ] {len(m.graph.node)} nodes, {len(ops)} distinct ops")
    print(f"[onnx ] ops: {ops}")

    if args.skip_check:
        return

    # ---- ONNX Runtime vs the eval-mode PyTorch reference --------------------
    import onnxruntime as ort

    assert not model.training, "model must be in eval mode for the comparison"
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    got = sess.run(["heatmap", "offset"], {"images": x.numpy()})

    ok = True
    for name, r, g, tol in (("heatmap", ref[0].numpy(), got[0], args.tol_hm),
                            ("offset", ref[1].numpy(), got[1], args.tol_off)):
        assert g.shape == r.shape, f"{name}: shape {g.shape} != torch {r.shape}"
        d = np.abs(g - r)
        verdict = "PASS" if d.max() <= tol else "FAIL"
        ok &= d.max() <= tol
        print(f"[check] {name}: shape={r.shape} max_abs_diff={d.max():.3e} "
              f"mean_abs_diff={d.mean():.3e} tol={tol:.0e} -> {verdict}")
    if not ok:
        raise SystemExit("[FATAL] ONNX does not reproduce the eval-mode PyTorch model")
    print(f"[check] heatmap max={float(got[0].max()):.6f} "
          f"pixels>=0.25 = {int((got[0] >= 0.25).sum())}")
    print("[check] ONNX reproduces the eval-mode PyTorch model -> export is faithful")


if __name__ == "__main__":
    main()
