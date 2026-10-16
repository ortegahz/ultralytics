#!/usr/bin/env python3
"""
Rewrite the heatmap ONNX so its graph input is UINT8 and the /255 lives inside.

Why
---
The goal is a mixed int8/fp16 model whose INPUT stays int8, so rknn_inputs_set
keeps the int8 cost. Measured on this runtime, an int8 graph input costs 0.34 ms
while a FLOAT16 graph input costs 45.41 ms (the runtime then widens all
640*640*3 elements in user space) -- a 130x difference on a stage that is 28% of
an fp16 frame. That is the entire reason this file exists.

Why the toolkit cannot do it
----------------------------
rknn.config(quantized_hybrid_level=N) was tried first and does NOT preserve an
int8 input. Level 1 produced:

    0  InputOperator  FLOAT16  CPU  (1,3,640,640)
    188 OutputOperator INT8    CPU  heatmap
    Total Internal Memory Size: 105600KB      (fp16 is 105600KB, int8 is 20400KB)

i.e. the hybrid splits the BODY but leaves the input float, which forfeits the
whole point. The two-step hybrid API (hybrid_quantization_step1/2) would name the
layers precisely, but it crashes on this graph inside the toolkit's own quant
optimizer -- see the experiment log. So the input has to be fixed upstream, in
the ONNX.

What this does
--------------
  1. graph input `images` : FLOAT [1,3,640,640]  ->  UINT8 [1,3,640,640]
  2. insert, as the new first nodes:
         Cast(images -> float32) -> Div(/255) -> <old graph>
     feeding the ORIGINAL float tensor, so every downstream node, weight and
     initializer is bit-identical and the model stays the same network.

The values the NPU computes are therefore unchanged: the runtime used to do
`pixel/255` on our behalf for the float input, now the graph does it.

Keeping mean_values/std_values in rknn.config would apply the /255 a SECOND time
(0..255 becomes ~0..0.004 and detections collapse to zero). The caller must
therefore pass mean_values=None, std_values=None when building from this ONNX --
see build() in hm_convert_hybrid_rknn.py, which already does exactly that.

Usage:
    python make_uint8_input_onnx.py --onnx <in.onnx> --out <out.onnx>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=255.0)
    ap.add_argument("--cast-only", action="store_true",
                    help="diagnostic: emit only Cast(u8->f32), no Div and no injected "
                         "constant. Numerically WRONG on purpose -- it exists to tell a "
                         "bad injected constant apart from an unsupported uint8 graph input.")
    a = ap.parse_args()

    src, dst = Path(a.onnx), Path(a.out)
    if not src.is_file():
        sys.exit(f"[FATAL] no ONNX at {src}")
    model = onnx.load(str(src))
    g = model.graph

    if len(g.input) != 1:
        sys.exit(f"[FATAL] expected exactly 1 graph input, found {len(g.input)}")
    old_in = g.input[0]
    name = old_in.name
    old_type = old_in.type.tensor_type.elem_type
    dims = [d.dim_value for d in old_in.type.tensor_type.shape.dim]
    print(f"[in  ] {name}: {TensorProto.DataType.Name(old_type)} {dims}")
    if old_type == TensorProto.UINT8:
        print("[skip] already UINT8 input")
    if old_type != TensorProto.FLOAT:
        sys.exit(f"[FATAL] unexpected input dtype {TensorProto.DataType.Name(old_type)}")

    # float32, NOT float16: the rest of this ONNX is float32 throughout, and a
    # float16 Cast makes Conv's declared type disagree with its computed type
    # (onnxruntime: "Conv bound to different types (tensor(float16) and
    # tensor(float))"). The NPU picks its own compute dtype afterwards; what
    # matters here is that the graph stays self-consistent.
    f32 = TensorProto.FLOAT
    # Shape (1,), not a 0-d scalar: rknn::RKNNInitRNNConst rejects the model with
    # "Not Support Dtype: 2" when the injected constant is rank 0, so the rank is
    # pinned here rather than left to numpy's default.
    scale = numpy_helper.from_array(np.array([a.scale], dtype=np.float32),
                                    name="/model/pre/scale")
    cast_out = "/model/pre/cast_out"
    div_out = "/model/pre/div_out"

    # Rewrite the graph input as uint8, keeping shape.
    new_in = helper.make_tensor_value_info(name, TensorProto.UINT8, dims)
    g.input.pop(0)
    g.input.insert(0, new_in)

    cast = helper.make_node("Cast", [name], [cast_out], name="/model/pre/Cast", to=f32)
    div = helper.make_node("Div", [cast_out, scale.name], [div_out], name="/model/pre/Div")
    # New nodes must precede everything that consumed the original float tensor.
    g.node.insert(0, div)
    g.node.insert(0, cast)

    if a.cast_only:
        for n in g.node:
            if n.name != "/model/pre/Cast":
                continue
            for i, inp in enumerate(n.input):
                if inp == name:
                    n.input[i] = cast_out
        for n in g.node:
            if n.name == "/model/pre/Cast":
                n.output[0] = name
        g.node.remove(div)
        cast_out = name
        div_out = name
        print("[diag ] --cast-only: 无 Div、无注入常量（数值故意错误，仅用于判别）")

    # Re-point the first real consumer at the Div output instead of the input.
    # Compare by NAME, not identity: protobuf repeated fields hand back a fresh
    # wrapper object on every access, so `n is cast` is False even for the node
    # just inserted, and the loop would rewrite Cast's own input.
    injected = {"/model/pre/Cast", "/model/pre/Div"}
    if a.cast_only:
        injected = {"/model/pre/Cast"}
    rewired = 0
    for n in g.node:
        if n.name in injected:
            continue
        for i, inp in enumerate(n.input):
            if inp == name:
                n.input[i] = div_out
                rewired += 1
    print(f"[rewire] {rewired} consumer(s) moved from '{name}' to '{div_out}'")
    if rewired == 0:
        sys.exit("[FATAL] nothing consumed the input -- refusing to write a dead graph")

    if not a.cast_only:
        g.initializer.append(scale)

    # Output dtype changes are fine (both were FLOAT), but the model value_info
    # for the old input is now wrong; drop stale entries and let inference re-add.
    for vi in list(g.value_info):
        if vi.name == name:
            g.value_info.remove(vi)

    onnx.checker.check_model(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    print(f"[ok  ] {dst}  {dst.stat().st_size/1e6:.2f} MB")
    print(f"       graph input is now UINT8 {dims}; Cast->Div(/255) are the first two nodes")


if __name__ == "__main__":
    main()
