#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
07_verify_u8_vs_float_onnx.py -- prove that the UINT8-input ONNX and the FLOAT32-input
ONNX are numerically equivalent, so that the int8 arm and the f16 arm are a CONTROLLED A/B.

Why
---
The board sweep compares two RKNN builds:

    int8 arm : built from `trial_0080_hm_nop0_u8.onnx`  (graph input = UINT8)
    f16  arm : built from `trial_0080_hm_nop0.onnx`     (graph input = FLOAT32)

The two arms differ in TWO respects at once: the graph input dtype AND the quantisation.
That is fine -- the /255 is a graph-level detail, not an experimental variable -- but
only if the two graphs really do compute the same function. The uint8 graph was made
by `manu/pipeline/npu/make_uint8_input_onnx.py`, which rewrites the graph input to
UINT8 and prepends `Cast(uint8->float32)` + `Div(/255)`; the body (weights, ops,
initialisers) is untouched. This script MEASURES that claim instead of trusting it.

What it does
------------
For every frame of the Tier-1 manifest it feeds **one and the same uint8 byte buffer**
to both graphs:

    float arm : nchw_u8.astype(float32) / 255.0     (the same bytes, rescaled)
    u8    arm : nchw_u8                             (the raw bytes)

and compares heatmap and offset with max abs diff / mean abs diff / cosine similarity.
The uint8 buffer is produced by ONE `cv2.imread` (the sanctioned reader
`C.bmp_to_input_nhwc`), so no decoder difference can hide inside the delta; the md5 of
that buffer is recorded per frame as the audit trail.

It also verifies the graph structure: the uint8 graph's input elem_type must be UINT8(2),
its first two nodes must be Cast + Div, the Div scale must be 255.0, and the float
graph's input elem_type must be FLOAT32(1).

Verdict
-------
    equivalent  <=>  heatmap max abs diff < 1e-4  AND offset max abs diff < 1e-4
                    AND min per-frame cos > 0.999999

Usage
-----
    python 07_verify_u8_vs_float_onnx.py
    python 07_verify_u8_vs_float_onnx.py --frames 20 --out /tmp/u8_vs_float_onnx.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
# repo root (so `manu.*` namespace packages import) + the path the sibling scripts use
for _p in (_HERE.parents[4], _HERE.parents[3]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402

ONNX_DIR = C.ONNX_DIR
WORK_ROOT = C.WORK_ROOT

DEFAULT_FLOAT_ONNX = ONNX_DIR / "trial_0080_hm_nop0.onnx"
DEFAULT_U8_ONNX = ONNX_DIR / "trial_0080_hm_nop0_u8.onnx"
DEFAULT_MANIFEST = WORK_ROOT / "fast" / "tier1_sensitive.json"
DEFAULT_OUT = WORK_ROOT / "u8_vs_float_onnx.json"

# equivalence gate (suggested by the brief)
TOL_ABS = 1e-4
MIN_COS = 0.999999


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _jsonable(v):
    """Make an arbitrary attribute value JSON-serialisable."""
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def _md5_array(a: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def _shape_of(vi) -> list:
    dims = []
    for d in vi.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.HasField("dim_value") else (d.dim_param or "?"))
    return dims


def graph_report(path: str | Path) -> dict:
    """Structural facts about one ONNX graph: input dtype, first nodes, Div scale."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"[FATAL] no ONNX at {p}")
    model = onnx.load(str(p))
    g = model.graph

    inp = g.input[0]
    tt = inp.type.tensor_type
    rep = {
        "path": str(p),
        "md5": C.md5_file(p),
        "n_nodes": len(g.node),
        "n_initializers": len(g.initializer),
        "n_graph_inputs": len(g.input),
        "input": {
            "name": inp.name,
            "elem_type": int(tt.elem_type),
            "elem_type_name": TensorProto.DataType.Name(tt.elem_type),
            "shape": _shape_of(inp),
        },
        "outputs": [
            {
                "name": o.name,
                "elem_type": int(o.type.tensor_type.elem_type),
                "elem_type_name": TensorProto.DataType.Name(o.type.tensor_type.elem_type),
                "shape": _shape_of(o),
            }
            for o in g.output
        ],
    }

    first_two = []
    for n in list(g.node)[:2]:
        attrs = {}
        for a in n.attribute:
            attrs[a.name] = _jsonable(helper.get_attribute_value(a))
        first_two.append({
            "op_type": n.op_type,
            "name": n.name,
            "inputs": list(n.input),
            "outputs": list(n.output),
            "attributes": attrs,
        })
    rep["first_two_nodes"] = first_two

    # the constant the Div divides by (initializer, or a Constant node's tensor)
    init_by_name = {i.name: i for i in g.initializer}
    const_by_name = {}
    for n in g.node:
        if n.op_type != "Constant" or not n.output:
            continue
        for a in n.attribute:
            if a.name == "value":
                const_by_name[n.output[0]] = numpy_helper.to_array(a.t)

    div_scale = None
    for n in g.node:
        if n.op_type != "Div" or len(n.input) < 2:
            continue
        iname = n.input[1]
        if iname in init_by_name:
            arr = numpy_helper.to_array(init_by_name[iname])
            div_scale = {
                "source": "initializer",
                "name": iname,
                "value": float(arr.reshape(-1)[0]) if arr.size else None,
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
            }
        elif iname in const_by_name:
            arr = const_by_name[iname]
            div_scale = {
                "source": "constant_node",
                "name": iname,
                "value": float(arr.reshape(-1)[0]) if arr.size else None,
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
            }
        else:
            div_scale = {"source": "not_a_constant", "name": iname, "value": None}
        break
    rep["div_scale"] = div_scale
    return rep


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """max abs diff / mean abs diff / cosine similarity between two tensors."""
    x = np.asarray(a, np.float64).ravel()
    y = np.asarray(b, np.float64).ravel()
    if x.shape != y.shape:
        raise SystemExit(f"[FATAL] shape mismatch {x.shape} vs {y.shape}")
    d = np.abs(x - y)
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    cos = float(x.dot(y) / (nx * ny)) if (nx > 0.0 and ny > 0.0) else 1.0
    return {
        "max_abs_diff": float(d.max()) if d.size else 0.0,
        "mean_abs_diff": float(d.mean()) if d.size else 0.0,
        "cos": cos,
    }


def _summarise(rows: list[dict], key: str) -> dict:
    mx = [r[key]["max_abs_diff"] for r in rows]
    mn = [r[key]["mean_abs_diff"] for r in rows]
    cs = [r[key]["cos"] for r in rows]
    return {
        "n_frames": len(rows),
        "max_abs_diff_max": max(mx) if mx else 0.0,
        "max_abs_diff_mean": float(np.mean(mx)) if mx else 0.0,
        "mean_abs_diff_max": max(mn) if mn else 0.0,
        "mean_abs_diff_mean": float(np.mean(mn)) if mn else 0.0,
        "cos_min": min(cs) if cs else 1.0,
        "cos_mean": float(np.mean(cs)) if cs else 1.0,
    }


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="uint8-input ONNX vs float32-input ONNX equivalence")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--float-onnx", default=str(DEFAULT_FLOAT_ONNX))
    ap.add_argument("--u8-onnx", default=str(DEFAULT_U8_ONNX))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--frames", type=int, default=100, help="0 = every frame in the manifest")
    ap.add_argument("--threads", type=int, default=4, help="ORT intra-op threads")
    args = ap.parse_args()

    t_start = time.time()

    # ---- structure ------------------------------------------------------------------
    g_float = graph_report(args.float_onnx)
    g_u8 = graph_report(args.u8_onnx)
    print(f"[onnx] float graph : input '{g_float['input']['name']}' "
          f"elem_type={g_float['input']['elem_type']} ({g_float['input']['elem_type_name']}) "
          f"shape={g_float['input']['shape']} nodes={g_float['n_nodes']} "
          f"inits={g_float['n_initializers']}")
    print(f"[onnx] u8    graph : input '{g_u8['input']['name']}' "
          f"elem_type={g_u8['input']['elem_type']} ({g_u8['input']['elem_type_name']}) "
          f"shape={g_u8['input']['shape']} nodes={g_u8['n_nodes']} "
          f"inits={g_u8['n_initializers']}")
    for i, n in enumerate(g_u8["first_two_nodes"]):
        print(f"[onnx] u8 node[{i}]  : op_type={n['op_type']} name={n['name']} "
              f"in={n['inputs']} out={n['outputs']} attrs={n['attributes']}")
    ds = g_u8.get("div_scale") or {}
    print(f"[onnx] u8 Div scale: source={ds.get('source')} name={ds.get('name')} "
          f"value={ds.get('value')} dtype={ds.get('dtype')} shape={ds.get('shape')}")

    import onnx
    chk = {
        "u8_input_is_uint8": g_u8["input"]["elem_type"] == onnx.TensorProto.UINT8,
        "float_input_is_float32": g_float["input"]["elem_type"] == onnx.TensorProto.FLOAT,
        "u8_first_two_nodes_are_cast_div": (
            len(g_u8["first_two_nodes"]) == 2
            and g_u8["first_two_nodes"][0]["op_type"] == "Cast"
            and g_u8["first_two_nodes"][1]["op_type"] == "Div"),
        "u8_div_scale_is_255": bool(ds.get("value") is not None
                                   and abs(float(ds["value"]) - 255.0) < 1e-6),
        "u8_cast_to_is_float32": bool(
            g_u8["first_two_nodes"]
            and g_u8["first_two_nodes"][0]["attributes"].get("to") == onnx.TensorProto.FLOAT),
        "same_input_name": g_u8["input"]["name"] == g_float["input"]["name"],
        "same_output_names": [o["name"] for o in g_u8["outputs"]] == [o["name"] for o in g_float["outputs"]],
    }
    for k, v in chk.items():
        print(f"[check] {k} -> {'PASS' if v else 'FAIL'}")

    # ---- frames ---------------------------------------------------------------------
    manifest = Path(args.manifest)
    if not manifest.is_file():
        raise SystemExit(f"[FATAL] no manifest at {manifest}")
    records = json.loads(manifest.read_text())
    if args.frames and args.frames < len(records):
        records = records[: args.frames]
    print(f"[run] {len(records)} frames from {manifest}")

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, int(args.threads))
    so.inter_op_num_threads = 1
    sess_f = ort.InferenceSession(str(args.float_onnx), so, providers=["CPUExecutionProvider"])
    sess_u = ort.InferenceSession(str(args.u8_onnx), so, providers=["CPUExecutionProvider"])
    name_f = sess_f.get_inputs()[0].name
    name_u = sess_u.get_inputs()[0].name
    out_names_f = [o.name for o in sess_f.get_outputs()]
    out_names_u = [o.name for o in sess_u.get_outputs()]
    print(f"[run] float session input '{name_f}' outputs {out_names_f}")
    print(f"[run] u8    session input '{name_u}' outputs {out_names_u}")
    if out_names_f != out_names_u:
        raise SystemExit("[FATAL] the two graphs do not expose the same output names")

    rows = []
    t0 = time.time()
    for i, rec in enumerate(records):
        # ONE cv2.imread for this frame; both arms are derived from this very buffer
        nhwc_u8 = C.bmp_to_input_nhwc(rec["bmp"])
        if nhwc_u8.dtype != np.uint8:
            raise SystemExit(f"[FATAL] bmp_to_input_nhwc must return uint8, got {nhwc_u8.dtype}")
        nchw_u8 = np.ascontiguousarray(nhwc_u8.transpose(0, 3, 1, 2))   # uint8, the raw bytes
        nchw_f32 = np.ascontiguousarray(nchw_u8.astype(np.float32) / 255.0)

        of = sess_f.run(None, {name_f: nchw_f32})
        ou = sess_u.run(None, {name_u: nchw_u8})

        hm = metrics(of[0], ou[0])       # heatmap  (output 0)
        off = metrics(of[1], ou[1])      # offset   (output 1)

        rows.append({
            "index": i,
            "seq": rec.get("seq"),
            "frame_id": rec.get("frame_id"),
            "bmp": rec.get("bmp"),
            "u8_bytes_md5": _md5_array(nchw_u8),
            "float_bytes_md5": _md5_array(nchw_f32),
            "heatmap": hm,
            "offset": off,
        })
        if (i + 1) % 10 == 0 or i + 1 == len(records):
            print(f"[run] {i + 1}/{len(records)} frames "
                  f"({(time.time() - t0) / (i + 1):.2f}s/frame)")

    # ---- summary --------------------------------------------------------------------
    hm_sum = _summarise(rows, "heatmap")
    off_sum = _summarise(rows, "offset")

    equivalent = bool(
        hm_sum["max_abs_diff_max"] < TOL_ABS
        and off_sum["max_abs_diff_max"] < TOL_ABS
        and hm_sum["cos_min"] > MIN_COS
        and off_sum["cos_min"] > MIN_COS
        and all(chk.values())
    )

    conclusion = (
        f"[结论] {'PASS' if equivalent else 'FAIL'} "
        f"{len(rows)} 帧: heatmap max_abs_diff={hm_sum['max_abs_diff_max']:.3e} "
        f"mean_abs_diff={hm_sum['mean_abs_diff_mean']:.3e} "
        f"cos(min/mean)={hm_sum['cos_min']:.9f}/{hm_sum['cos_mean']:.9f}; "
        f"offset max_abs_diff={off_sum['max_abs_diff_max']:.3e} "
        f"mean_abs_diff={off_sum['mean_abs_diff_mean']:.3e} "
        f"cos(min/mean)={off_sum['cos_min']:.9f}/{off_sum['cos_mean']:.9f} "
        f"=> uint8 输入 ONNX 与 float32 输入 ONNX "
        f"{'数值等价' if equivalent else '数值不等价'}"
    )

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "onnx": onnx.__version__,
        "manifest": str(manifest),
        "n_frames": len(rows),
        "tolerance": {"max_abs_diff": TOL_ABS, "min_cos": MIN_COS},
        "graphs": {"float32": g_float, "uint8": g_u8},
        "structure_checks": chk,
        "summary": {"heatmap": hm_sum, "offset": off_sum},
        "verdict": {
            "equivalent": equivalent,
            "conclusion": conclusion,
            "controlled_ab": equivalent,
        },
        "seconds": round(time.time() - t_start, 1),
        "frames": rows,
    }

    out = Path(args.out)
    C.write_json(out, report)
    print(f"[out ] {out}  ({out.stat().st_size / 1e6:.3f} MB, "
          f"{report['seconds']:.0f}s total)")
    print(f"[self] {Path(__file__).name} md5={C.md5_file(Path(__file__).resolve())}")
    print(conclusion)


if __name__ == "__main__":
    main()
