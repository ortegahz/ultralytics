#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
22_export_qdq_onnx.py -- freeze the QAT-learned clips into ONNX Q/DQ nodes.

WHAT THIS DOES
--------------
1. rebuilds the detector from the QAT checkpoint (trial_0080 epoch3.pt + the
   QAT fine-tune), p0_off variant;
2. wraps the three quantised head sites with the ONNX-native fake quantiser, so
   torch's opset-13 symbolic emits a REAL QuantizeLinear + DequantizeLinear pair
   per site (measured by 21_probe_onnx_fakequant.py: the pair is present and
   ONNX Runtime reproduces PyTorch with max|diff| = 0);
3. rewrites the Q/DQ constants into INITIALISERS with the dtypes the RKNN
   toolkit wants: y_scale float32 [1], y_zero_point uint8 [1]
   (uint8 zp = 0 is what the torch.ao route proved works on this board:
   "torch.ao writes uint8 zp=37, the toolkit re-centres it to int8 zp=-91, the
   scale is unchanged and the range is equivalent");
4. rewrites the graph input to UINT8 with Cast->Div(/255) INSIDE the graph, by
   calling the sanctioned manu/pipeline/npu/make_uint8_input_onnx.py -- so
   rknn.config() must be called with mean_values=None / std_values=None
   (a uint8 graph input is what the board's stbi_load() path produces; a float
   graph input makes the toolkit turn the input into INT8 and the board then
   misreads the 128..255 bytes);
5. prints the Q/DQ table (scale / zero_point per site) so the board read-out
   can be compared against it later.

THREE MODES (the A0 / A1 / A2 control triple)
----------------------------------------------
--mode qat   : the LSQ-learned clips (the A2 arm)
--mode ptq   : the pre-QAT calibration clips (the A1 arm) -- the same clips the
               RKNN toolkit's min/max rule would have picked, so A1 vs A2
               isolates "learned clip" from "calibrated clip".
--mode float : no Q/DQ at all (the A0 fp16 control).

--weights-source qat  : the QAT checkpoint's re-trained weights (default)
--weights-source base : the untouched trial_0080 weights, i.e. the control arm
                        that isolates the WEIGHT update from the clip.

The full ladder the delivery compares:

    A0   float   QAT weights          -> what the weight update alone does
    A1   int8    QAT weights + max-calibrated clips
    A2   int8    QAT weights + LSQ-learned clips   (the main candidate)
    A1b  int8    BASE weights + max-calibrated clips (the PTQ-style control)

Run (server):
    source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
    python manu/pipeline/npu/ptq/22_export_qdq_onnx.py \
        --ckpt /mnt/data/siping/ptq_work/qat/trial_0080/qat_epoch3.pt \
        --out  /mnt/data/siping/ptq_work/qat/trial_0080/qat_epoch3_qdq.onnx
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manu.pipeline.npu.ptq import ptq_qat_common as Q  # noqa: E402


class OnnxWrapper(nn.Module):
    """Flatten the dict output to a (heatmap, offset) tuple for a clean graph."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["heatmap"], out["offset"]


def promote_qdq_constants(model) -> list[dict]:
    """Turn the Q/DQ Constant nodes into INITIALISERS with toolkit-friendly dtypes.

    torch's exporter emits the scale / zero-point as Constant NODES.  The RKNN
    toolkit is much happier with initializers, and the zero-point has to be
    uint8 (ONNX allows int8/uint8/int32; torch emits int64, which some
    consumers reject).  The scale becomes float32 with shape [1] instead of a
    rank-0 scalar (a rank-0 constant was rejected by the toolkit in an earlier
    experiment on this project).

    Two things this has to get right:

    * the Constant node may ALREADY have been constant-folded into an
      initializer by the exporter (do_constant_folding=True); then the
      initializer's DTYPE is what has to be fixed, not a node to be removed;
    * the node list must NOT be mutated while it is being iterated -- doing so
      silently skips every second node, which would leave half of the Q/DQ
      pairs with an int64 zero-point that the toolkit rejects.
    """
    import onnx
    from onnx import numpy_helper

    graph = model.graph
    const_nodes = {n.output[0]: n for n in graph.node if n.op_type == "Constant"}
    init_names = {i.name for i in graph.initializer}
    table = []
    to_remove = []
    seen_init = set()

    for node in graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        for slot, label in ((1, "scale"), (2, "zero_point")):
            if len(node.input) <= slot:
                continue
            nm = node.input[slot]
            arr = None
            already_init = False
            if nm in init_names:
                arr = numpy_helper.to_array(
                    next(i for i in graph.initializer if i.name == nm))
                already_init = True
            elif nm in const_nodes:
                arr = numpy_helper.to_array(const_nodes[nm].attribute[0].t)
            if arr is None:
                continue
            if label == "scale":
                arr = arr.astype(np.float32).reshape(-1)[:1]
            else:
                arr = arr.astype(np.int64).reshape(-1)[:1]
                arr = (arr - 0).astype(np.uint8)   # zero-point 0 -> uint8 0
            new_t = numpy_helper.from_array(arr, name=nm)
            if already_init:
                for i, t in enumerate(graph.initializer):
                    if t.name == nm:
                        graph.initializer[i].CopyFrom(new_t)
                        break
            elif nm not in seen_init:
                graph.initializer.append(new_t)
                seen_init.add(nm)
            if nm in const_nodes and const_nodes[nm] not in to_remove:
                to_remove.append(const_nodes[nm])
            table.append({"node": node.name or node.op_type, "op": node.op_type,
                          "role": label, "value": float(arr[0]),
                          "dtype": str(arr.dtype), "shape": list(arr.shape),
                          "initializer": nm, "was_initializer": already_init})
    # remove the Constant nodes AFTER the walk (mutating the list while
    # iterating it would skip nodes)
    for cnode in to_remove:
        graph.node.remove(cnode)
    onnx.checker.check_model(model)
    return table


# which quantised site each Q/DQ node belongs to, matched on the torch-export
# node name (the module path of the wrapped block is part of it)
_SITE_PATHS = {
    "feat_conv1": "head/feat_conv/feat_conv.1",
    "heatmap0": "head/heatmap/heatmap.0",
    "offset0": "head/offset/offset.0",
}


def _site_of(node_name: str, input_tensor: str) -> str:
    for site, path in _SITE_PATHS.items():
        if path in node_name:
            return site
    # the heatmap that leaves the graph is quantised by a Q/DQ attached to the
    # head itself (torch names it /model/head/QuantizeLinear)
    if "heatmap" in input_tensor or node_name.startswith("/model/head/"):
        return "heatmap_out"
    return "?"


def read_qdq_table(onnx_path) -> list[dict]:
    """Read the Q/DQ scale / zero_point back OUT of the written ONNX.

    This is the number the board has to echo back, so it is read from the file
    that is actually handed to the toolkit -- never from memory.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))
    graph = model.graph
    inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    consts = {}
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    consts[node.output[0]] = numpy_helper.to_array(attr.t)

    def arr_of(name):
        return inits.get(name, consts.get(name))

    rows = []
    for node in graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        s = arr_of(node.input[1]) if len(node.input) > 1 else None
        z = arr_of(node.input[2]) if len(node.input) > 2 else None
        rows.append({
            "site": _site_of(node.name or "", node.input[0]),
            "node": node.name,
            "op": node.op_type,
            "input_tensor": node.input[0],
            "scale": float(s.reshape(-1)[0]) if s is not None else None,
            "n_scales": int(s.size) if s is not None else 0,
            "scale_dtype": str(s.dtype) if s is not None else None,
            "zero_point": int(z.reshape(-1)[0]) if z is not None else None,
            "zp_dtype": str(z.dtype) if z is not None else None,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="the QAT checkpoint (qat_epochN.pt)")
    ap.add_argument("--base-weights",
                    default=str(Q.SERVER_ROOT / "runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="qat", choices=["qat", "ptq", "float"])
    ap.add_argument("--weights-source", default="qat", choices=["qat", "base"],
                    help="'qat' = the QAT checkpoint's re-trained weights; "
                         "'base' = the untouched trial_0080 weights (the control "
                         "arm that isolates the weight update from the clip)")
    ap.add_argument("--ablate-p0", type=int, default=1)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--uint8-input", type=int, default=0,
                    help="1 = rewrite the graph input to UINT8 (Cast->Div inside the "
                         "graph).  The RKNN toolkit REJECTS a uint8 graph input "
                         "('Not Support Dtype: 2' / 'Missing quantization parameters "
                         "for images'), so the default is 0.")
    args = ap.parse_args()

    from manu.pipeline.npu.hm_export_onnx import build_model

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"[ckpt ] {args.ckpt}  epoch={ck.get('epoch')}  "
          f"train_loss={ck.get('metrics', {}).get('train_loss')}")

    model, _ = build_model(args.base_weights, ablate_p0=bool(args.ablate_p0))
    if args.weights_source == "qat":
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        if missing or unexpected:
            raise SystemExit(f"[FATAL] state dict mismatch: {len(missing)} missing "
                             f"(e.g. {missing[:3]}), {len(unexpected)} unexpected "
                             f"(e.g. {unexpected[:3]})")
        print(f"[ckpt ] weights loaded: {len(ck['model'])} tensors (QAT checkpoint)")
    else:
        print(f"[ckpt ] weights: BASE (untouched) {args.base_weights}")

    if args.mode == "qat":
        scales = dict(ck["qat_scales"])
    elif args.mode == "ptq":
        scales = dict(ck["calib_scales"])
    else:
        scales = None
    print(f"[scale] mode={args.mode}  weights-source={args.weights_source}  "
          f"{scales if scales is not None else '(no Q/DQ -> pure float)'}")

    if scales is not None:
        Q.attach_export_sites(model, scales)
        Q.attach_export_output_fq(model, float(ck.get("out_scale", Q.OUT_SCALE)))
    model.eval()

    wrapper = OnnxWrapper(model)
    x = torch.zeros(1, 3, Q.IMGSZ, Q.IMGSZ)
    tmp = Path(args.out).with_suffix(".float.onnx")
    tmp.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(wrapper, (x,), str(tmp), opset_version=args.opset,
                          input_names=["images"], output_names=["heatmap", "offset"],
                          dynamic_axes=None, do_constant_folding=True)
    print(f"[onnx ] float graph -> {tmp} ({tmp.stat().st_size/1e6:.2f} MB)")

    import onnx
    m = onnx.load(str(tmp))
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"[onnx ] op histogram: {ops}")
    table = promote_qdq_constants(m)
    for row in table:
        print(f"[qdq  ] {row['op']:<16} {row['role']:<11} = {row['value']!r} "
              f"({row['dtype']} {row['shape']})  <- {row['initializer']}")
    if args.mode != "float" and ops.get("QuantizeLinear", 0) == 0:
        raise SystemExit("[FATAL] the exported graph carries NO QuantizeLinear -- "
                         "the Q/DQ did not survive the export")
    onnx.save(m, str(tmp))

    # ---- the graph input --------------------------------------------------
    # The FLOAT input (x/255 applied by the toolkit's mean/std) is the one the
    # RKNN toolkit actually accepts.  The UINT8-input rewrite of
    # manu/pipeline/npu/make_uint8_input_onnx.py is rejected at build time:
    #   * uint8 graph input + no mean/std -> "Missing quantization parameters
    #     for images, leading to ambiguity"
    #   * uint8 graph input -> "Not Support Dtype: 2" (uint8) at build time
    # so it is OFF by default and kept only as a documented experiment.
    u8 = Path(args.out)
    if args.uint8_input:
        cmd = [sys.executable, str(REPO_ROOT / "manu/pipeline/npu/make_uint8_input_onnx.py"),
               "--onnx", str(tmp), "--out", str(u8)]
        print(f"[u8   ] {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    else:
        shutil.copyfile(tmp, u8)
        print(f"[in   ] graph input kept FLOAT32 (uint8_input=0); "
              f"the toolkit applies the /255 via mean_values/std_values")

    # ---- read the Q/DQ back OUT of the file that goes to the toolkit ---------
    final = read_qdq_table(u8)
    print(f"[qdq  ] === READ BACK FROM {u8} ===")
    for row in final:
        print(f"[qdq  ] site={row['site']:<12} {row['op']:<16} "
              f"scale={row['scale']!r} ({row['scale_dtype']}, n={row['n_scales']})  "
              f"zp={row['zero_point']!r} ({row['zp_dtype']})  "
              f"node={row['node']}")
    if args.mode != "float" and not final:
        raise SystemExit("[FATAL] no Q/DQ found in the written ONNX")

    meta = {
        "ckpt": args.ckpt, "mode": args.mode, "weights_source": args.weights_source,
        "onnx": str(u8), "opset": args.opset, "uint8_input": bool(args.uint8_input),
        "qdq": table, "qdq_final": final,
        "scales": scales, "out_scale": float(ck.get("out_scale", Q.OUT_SCALE)),
        "ablate_p0": bool(args.ablate_p0),
        "op_histogram": ops,
    }
    u8.with_suffix(".qdq_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"[ok   ] {u8}  ({u8.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
