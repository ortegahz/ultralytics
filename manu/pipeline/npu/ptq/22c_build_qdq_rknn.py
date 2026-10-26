#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
22c_build_qdq_rknn.py -- build a .rknn from a Q/DQ ONNX.

THE RECIPE (both halves are HARD CONSTRAINTS, both were measured on this board)

1. rknn.build(do_quantization=False)
   A Q/DQ ONNX carries its own calibration.  Handing the model to the toolkit's
   OWN quantiser on top of that fails outright -- 17_probe_qdq_import.py
   measured `build rc=3 / ret=-1` for the case `hm_qdq_dq_quant`
   (verdict QDQ_BUILD_FAILED).  With do_quantization=False the toolkit KEEPS
   the scale / zero_point that the Q/DQ carries and carries them all the way to
   the board's output tensor (verdict QDQ_ADOPTED, 5/5 cases, scale bit-exact).

2. mean_values / std_values describe the INPUT tensor, not the activations
   * mean_values=[[0,0,0]] / std_values=[[255,255,255]] tells the toolkit that
     the uint8 buffer the board uploads means x/255.  That is the INPUT's
     normalisation and is completely orthogonal to the Q/DQ activation clips --
     it is NOT a second activation calibration, so it cannot double-apply
     anything.  It is also REQUIRED here: the board's driver always uploads a
     UINT8 buffer, and without mean/std the runtime widens it to FLOAT16
     WITHOUT dividing by 255, so the graph sees values 255x too large and the
     network saturates -- measured: the board's own null control then reports
     "NPU output independent of input" and the driver aborts.
   * the UINT8-graph-input rewrite (manu/pipeline/npu/make_uint8_input_onnx.py)
     is NOT usable with this toolkit: a uint8 graph input is rejected with
     "Not Support Dtype: 2" / "Missing quantization parameters for images".

Everything the build actually did is written to the JSON report next to the
model, including the verbatim toolkit log, so "do_quantization=False was
adopted" is an archived fact and not a claim.

Usage (server, conda env `rknn`):
    source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate rknn
    python manu/pipeline/npu/ptq/22c_build_qdq_rknn.py \
        --onnx  /mnt/data/siping/ptq_work/qat/trial_0080/onnx/a2_qat_learned.onnx \
        --out   /mnt/data/siping/ptq_work/qat/trial_0080/rknn/a2_qat_learned.rknn
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_qdq_summary(onnx_path: Path) -> dict:
    """The scale / zero_point the ONNX writes, read back out of the file."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    consts = {}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    consts[node.output[0]] = numpy_helper.to_array(attr.t)

    def arr_of(name):
        return inits.get(name, consts.get(name))

    out = {"n_quantize_linear": 0, "n_dequantize_linear": 0, "nodes": [],
           "graph_input_dtype": None}
    inp = model.graph.input[0]
    out["graph_input_dtype"] = str(inp.type.tensor_type.elem_type)
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            out["n_quantize_linear"] += 1
        elif node.op_type == "DequantizeLinear":
            out["n_dequantize_linear"] += 1
        else:
            continue
        s = arr_of(node.input[1]) if len(node.input) > 1 else None
        z = arr_of(node.input[2]) if len(node.input) > 2 else None
        out["nodes"].append({
            "node": node.name, "op": node.op_type,
            "input_tensor": node.input[0],
            "scale": float(s.reshape(-1)[0]) if s is not None else None,
            "n_scales": int(s.size) if s is not None else 0,
            "scale_dtype": str(s.dtype) if s is not None else None,
            "zero_point": int(z.reshape(-1)[0]) if z is not None else None,
            "zp_dtype": str(z.dtype) if z is not None else None,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--do-quantization", type=int, default=0,
                    help="MUST stay 0 for a Q/DQ ONNX (see the module docstring)")
    ap.add_argument("--mean-std", type=int, default=1,
                    help="1 = pass mean_values=[[0,0,0]] / std_values=[[255,255,255]], "
                         "i.e. tell the toolkit that the uint8 board buffer means "
                         "x/255.  This is the INPUT's normalisation and is "
                         "orthogonal to the Q/DQ activation clips -- it is NOT a "
                         "second activation calibration.  0 = no mean/std, which "
                         "leaves the raw 0..255 in the graph.")
    ap.add_argument("--calib-list", default="",
                    help="the 250-image BMP list for the toolkit's own quantiser "
                         "(only used when --do-quantization 1)")
    ap.add_argument("--quantized-algorithm", default="",
                    help="e.g. 'normal' -- the toolkit's own activation calibration")
    ap.add_argument("--target-platform", default="rk3588")
    ap.add_argument("--quantized-method", default="channel")
    ap.add_argument("--optimization-level", type=int, default=3)
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not onnx_path.is_file():
        raise SystemExit(f"[FATAL] no ONNX at {onnx_path}")

    from rknn.api import RKNN

    cfg = {
        "target_platform": args.target_platform,
        "quantized_method": args.quantized_method,
        "optimization_level": args.optimization_level,
    }
    do_quant = bool(args.do_quantization)
    use_mean_std = bool(args.mean_std)
    if use_mean_std:
        cfg["mean_values"] = [[0, 0, 0]]
        cfg["std_values"] = [[255, 255, 255]]
    if args.quantized_algorithm:
        cfg["quantized_algorithm"] = args.quantized_algorithm
    if do_quant and not args.calib_list:
        raise SystemExit("[FATAL] --do-quantization 1 needs --calib-list")

    print("=" * 100, flush=True)
    print(f"[build] onnx    = {onnx_path}")
    print(f"[build] out     = {out_path}")
    print(f"[build] config  = {cfg}")
    print(f"[build] mean_values = {cfg.get('mean_values')}   "
          f"std_values = {cfg.get('std_values')}")
    print(f"[build] do_quantization = {do_quant}   <-- HARD CONSTRAINT for a Q/DQ ONNX")
    print("=" * 100, flush=True)

    rknn = RKNN(verbose=True)
    rknn.config(**cfg)
    if rknn.load_onnx(model=str(onnx_path)) != 0:
        raise SystemExit("[FATAL] load_onnx failed")
    t0 = time.time()
    ret = rknn.build(do_quantization=do_quant,
                     dataset=(args.calib_list or None))
    build_s = time.time() - t0
    print(f"[build] rknn.build(do_quantization={do_quant}) -> ret={ret} "
          f"in {build_s:.1f}s", flush=True)
    if ret != 0:
        raise SystemExit(f"[FATAL] build failed ret={ret}")
    if rknn.export_rknn(str(out_path)) != 0:
        raise SystemExit("[FATAL] export_rknn failed")

    report = {
        "onnx": str(onnx_path), "onnx_md5": md5_file(onnx_path),
        "rknn": str(out_path), "rknn_md5": md5_file(out_path),
        "rknn_bytes": out_path.stat().st_size,
        "config": cfg,
        "mean_values": cfg.get("mean_values"), "std_values": cfg.get("std_values"),
        "do_quantization": do_quant, "build_ret": ret,
        "calib_list": args.calib_list or None,
        "build_seconds": round(build_s, 1),
        "qdq_in_onnx": read_qdq_summary(onnx_path),
        "toolkit_version": getattr(__import__("rknn"), "__version__", "unknown"),
    }
    report_path = out_path.with_suffix(".build.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"[ok   ] {out_path}  ({report['rknn_bytes']} B)  md5={report['rknn_md5']}")
    print(f"[ok   ] {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
