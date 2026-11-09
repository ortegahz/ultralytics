#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Export the STOCK Ultralytics YOLO26n detector to ONNX, for RK3588 / rknn-toolkit2.

Why this file exists
--------------------
The project's own detector is `YOLO26HeatmapDetector` (YOLO26 + P2 Neck +
320x320 heatmap head, 21.48 G MAC). The open question in the NPU-speed line of
work is: "how fast is a STOCK YOLO26n on the same RK3588 NPU?" That number is
the reference point against which every structural change is judged, so it has
to come from the real Ultralytics model -- not from a re-implementation.

Export path follows Ultralytics' own RKNN rules, not a hand-rolled guess:

  * `ultralytics/engine/exporter.py:681-684` -- for `format="rknn"` Ultralytics
    DISABLES the `end2end` branch, because RKNN has no `topk`. The exported
    graph is therefore the classic one2many detection head: a raw
    (B, 4+nc, N) tensor, with decode + NMS on the host. This is exactly how
    the Rockchip SDK's yolov8 example is wired, and it is why the board-side
    example in this repo does its own NMS.
  * `ultralytics/engine/exporter.py:1096-1104` -- for `rknn` + int8 +
    detect, Ultralytics wraps the model in `_NormalizeCoords`
    (`ultralytics/utils/export/engine.py:17`), which divides the 4 box
    channels by the input size. Pixel-valued coordinates have a dynamic range
    that per-tensor int8 quantization flattens; normalizing to [0,1] keeps the
    relative precision. The board-side C++ therefore scales boxes back by
    `imgsz`.

  * `end2end=False` does NOT change the graph's cost: `one2one_cv2/cv3` are
    `copy.deepcopy` of `cv2/cv3` (`ultralytics/nn/modules/head.py:140-141`),
    so both branches have identical layer counts and identical MACs. The timing
    measured for this export is therefore the timing of the stock network.

Input quantization contract (must match the board-side C++ exactly):
  mean 0 / std 255 -> the graph consumes plain uint8 RGB and RKNN applies
  (x - 0) / 255. Same convention as `hm_convert_rknn.py`, so the host-side
  PyTorch numbers and the board-side NPU numbers are directly comparable.

Usage:
    python yolo26n_export_onnx.py \
        --weights /media/manu/1TB-Volume/rknn_work/yolo26n/yolo26n.pt \
        --out     /media/manu/1TB-Volume/rknn_work/yolo26n/yolo26n.onnx \
        --imgsz 640 --opset 12
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.export.engine import _NormalizeCoords  # noqa: E402


def md5(path: str | Path) -> str:
    """MD5 of a file, so the artifact is auditable later."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="stock YOLO26n .pt (e.g. yolo26n.pt)")
    ap.add_argument("--out", required=True, help="output .onnx path")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument(
        "--no-normalize-coords",
        action="store_true",
        help="skip _NormalizeCoords (boxes stay in pixels; board C++ must then not scale)",
    )
    ap.add_argument(
        "--act-scope",
        choices=["all", "neck_head"],
        default="all",
        help="which layers the activation swap applies to. 'neck_head' keeps the "
        "backbone (model.0..model.10) on SiLU and only swaps the neck+head "
        "(model.11..), which is the only variant this project may legally use: "
        "b0..b4 are frozen with official SiLU weights, so changing their "
        "activation would change the frozen base model's behaviour.",
    )
    ap.add_argument(
        "--act",
        choices=["silu", "relu"],
        default="silu",
        help="activation. 'relu' is a TIMING PROBE ONLY, not an equivalent rewrite: "
        "Swish/SiLU fuses into RKNN's ConvExSwish, ReLU does not. Swapping it changes "
        "the numerics, so any accuracy claim from this arm is invalid -- it exists "
        "purely to price the activation on the NPU (the community reports a "
        "yolov5s_relu variant for exactly this reason).",
    )
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[weights] {args.weights}")
    yolo = YOLO(args.weights)
    net = yolo.model

    # --- confirm this really is the stock model ------------------------------
    yaml = net.yaml
    print(f"[model  ] class={type(net).__name__} scale={yaml.get('scale')} "
          f"end2end={yaml.get('end2end')} reg_max={yaml.get('reg_max')} nc={yaml.get('nc')}")
    params = sum(p.numel() for p in net.parameters())
    print(f"[model  ] parameters={params:,}")
    if yaml.get("scale") != "n":
        sys.exit(f"[FATAL] expected scale='n', got {yaml.get('scale')!r}")

    # --- RKNN rule: disable the end2end branch (no topk on RKNN) -------------
    was_end2end = net.end2end
    net.end2end = False
    print(f"[export ] end2end {was_end2end} -> {net.end2end} "
          f"(RKNN has no topk; Ultralytics does the same for format='rknn')")

    net.eval()
    net.float()
    net.fuse()

    if args.act == "relu":
        n_silu = 0
        if args.act_scope == "all":
            for mod in net.modules():
                for name, child in list(mod.named_children()):
                    if isinstance(child, torch.nn.SiLU):
                        setattr(mod, name, torch.nn.ReLU())
                        n_silu += 1
        else:
            # net.model is the flat ModuleList: indices 0..10 are the backbone
            # (b0..b10), 11.. are neck+head for the standard yolo26.yaml.
            for idx, layer in enumerate(net.model):
                if idx < 11:
                    continue
                for mod in layer.modules():
                    for name, child in list(mod.named_children()):
                        if isinstance(child, torch.nn.SiLU):
                            setattr(mod, name, torch.nn.ReLU())
                            n_silu += 1
        print(f"[export ] activation swapped SiLU -> ReLU on {n_silu} modules "
              f"(scope={args.act_scope}; TIMING PROBE ONLY, numerics changed)")

    head = net.model[-1]
    head.export = True
    head.format = "onnx"
    head.dynamic = False

    nc = int(yaml.get("nc"))
    wrapped = net
    if not args.no_normalize_coords:
        wrapped = _NormalizeCoords(net, args.imgsz, args.imgsz, "detect", nc, None)
        print("[export ] _NormalizeCoords ON (boxes normalized to [0,1]; board C++ scales by imgsz)")
    else:
        print("[export ] _NormalizeCoords OFF (boxes stay in pixels)")

    wrapped.eval()
    dummy = torch.zeros(args.batch, 3, args.imgsz, args.imgsz, dtype=torch.float32)

    print(f"[export ] -> {out}  (imgsz={args.imgsz}, opset={args.opset}, batch={args.batch})")
    torch.onnx.export(
        wrapped,
        dummy,
        str(out),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output0"],
        dynamo=False,
    )

    if not out.is_file():
        sys.exit(f"[FATAL] export reported success but {out} is absent")

    # --- verify the graph actually runs and report its contract --------------
    import onnx

    m = onnx.load(str(out))
    onnx.checker.check_model(m)
    inputs = [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
    outputs = [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
    ops = sorted({n.op_type for n in m.graph.node})
    print(f"[verify ] onnx.checker OK; inputs={inputs} outputs={outputs}")
    print(f"[verify ] distinct ops ({len(ops)}): {', '.join(ops)}")

    size_mb = out.stat().st_size / 1e6
    print(f"\n[OK] {out}")
    print(f"     {size_mb:.2f} MB   md5={md5(out)}")

    # Record the expected board-side decode contract, so the C++ author and the
    # reviewer read the same numbers.
    n_anchor = None
    with torch.no_grad():
        y = wrapped(dummy)
    if isinstance(y, (tuple, list)):
        y = y[0]
    print(f"[verify ] torch forward output shape={tuple(y.shape)}")
    if y.ndim == 3 and y.shape[1] >= 4 + nc:
        n_anchor = y.shape[2]
        print(f"[verify ] box channels = 4, class channels = {nc}, anchors = {n_anchor}")
        print(f"[verify ] board C++ must read: [0:4]=xyxy({'normalized' if not args.no_normalize_coords else 'pixels'}), "
              f"[4:{4 + nc}]=class scores")
    print(f"[verify ] output md5 note: N={n_anchor}, nc={nc}, imgsz={args.imgsz}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
