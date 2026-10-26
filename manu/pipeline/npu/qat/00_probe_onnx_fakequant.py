#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
00_probe_onnx_fakequant.py -- decide HOW the learned clips reach the ONNX graph.

THE QUESTION
------------
QAT learns a clip per quantised site.  The clip has to end up in the ONNX as a
real QuantizeLinear / DequantizeLinear pair, because that is the only construct
rknn-toolkit2 2.3.2 was measured to ADOPT (manu/pipeline/npu/ptq/17_probe_qdq_import.py
-> VERDICT: QDQ_ADOPTED).  A float "divide + round + multiply" chain is NOT a
quantisation request: with do_quantization=False the toolkit would compute it in
float16 and the model would carry no int8 at all.

TWO WAYS TO GET THE Q/DQ NODES
------------------------------
  A) export a module whose forward calls
         torch.fake_quantize_per_tensor_affine(x, scale, zp, qmin, qmax)
     and rely on torch's opset-13 symbolic to emit QuantizeLinear+DequantizeLinear.
     -> tested here.  Requires the symbolic to exist in torch 1.13.1.

  B) export the training module (which traces to Div/Clip/Round/Mul) and replace
     that sub-graph with Q/DQ by hand afterwards.
     -> always possible, but it needs the node names to be matched reliably.

This probe measures A, and -- if A works -- checks the three properties the rest
of the pipeline depends on:

  1. the ONNX really carries QuantizeLinear + DequantizeLinear;
  2. the scale reaches the graph as an INITIALIZER (a frozen constant), not as a
     graph input, and its rank/shape is something the RKNN toolkit accepts
     (the toolkit rejected a rank-0 UINT8 constant in an earlier experiment);
  3. ONNX Runtime reproduces the PyTorch numbers exactly (the Q/DQ and the
     fake-quant must be the SAME function).

Run on the server:
    source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
    python manu/pipeline/npu/qat/00_probe_onnx_fakequant.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class FQNet(nn.Module):
    """conv -> fake_quantize_per_tensor_affine, the shape the head sites have."""

    def __init__(self, scale: float):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, x):
        h = torch.relu(self.conv(x))
        return torch.fake_quantize_per_tensor_affine(h, self.scale, 0, 0, 255)


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/qat_probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[env ] torch={torch.__version__} opset-path={out_dir}")

    # ---- 1. does the opset-13 symbolic exist? --------------------------------
    try:
        from torch.onnx import symbolic_opset13 as sop13
        has_sym = any("fake_quantize_per_tensor_affine" in k for k in dir(sop13))
        print(f"[sym ] opset13 fake_quantize_per_tensor_affine symbol: {has_sym}")
    except Exception as exc:                                        # noqa: BLE001
        has_sym = False
        print(f"[sym ] import failed: {type(exc).__name__}: {exc}")

    torch.manual_seed(0)
    net = FQNet(0.05).eval()
    x = torch.rand(1, 3, 32, 32)

    with torch.no_grad():
        y_pt = net(x)

    path = out_dir / "fq_probe.onnx"
    ok = True
    try:
        torch.onnx.export(net, (x,), str(path), opset_version=13,
                          input_names=["images"], output_names=["out"],
                          dynamic_axes=None)
    except Exception as exc:                                        # noqa: BLE001
        ok = False
        print(f"[exp ] export FAILED: {type(exc).__name__}: {exc}")

    if ok:
        import onnx
        from onnx import numpy_helper

        m = onnx.load(str(path))
        print(f"[onnx] {path}  ({path.stat().st_size/1024:.1f} KB)")
        ops = {}
        for n in m.graph.node:
            ops[n.op_type] = ops.get(n.op_type, 0) + 1
        print(f"[onnx] op histogram: {ops}")

        inits = {i.name: i for i in m.graph.initializer}
        for n in m.graph.node:
            if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
                print(f"[onnx] {n.op_type}: in={list(n.input)} out={list(n.output)} "
                      f"axis={n.attribute}")
                for nm in n.input[1:]:
                    if nm in inits:
                        arr = numpy_helper.to_array(inits[nm])
                        print(f"[onnx]    {nm}: shape={arr.shape} dtype={arr.dtype} value={arr}")
                    else:
                        print(f"[onnx]    {nm}: NOT AN INITIALIZER (graph input?)")

        # ---- 3. does ONNX Runtime reproduce PyTorch exactly? ----------------
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            y_ort = sess.run(None, {"images": x.numpy()})[0]
            d = float(np.abs(y_ort - y_pt.numpy()).max())
            print(f"[num ] max|ORT - PyTorch| = {d:.3e}  (0 expected)")
            n_grid = len(np.unique(np.round(y_ort.reshape(-1) / 0.05)))
            print(f"[num ] distinct quantised levels seen: {n_grid} (grid step 0.05)")
        except Exception as exc:                                    # noqa: BLE001
            print(f"[num ] onnxruntime check failed: {type(exc).__name__}: {exc}")

        print(f"[verd] symbolic_emits_qdq = "
              f"{ops.get('QuantizeLinear', 0) > 0 and ops.get('DequantizeLinear', 0) > 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
