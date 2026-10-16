#!/usr/bin/env python3
"""
Hybrid (mixed int8/fp16) quantization of the Trial-0474 heatmap ONNX.

Why this script exists
----------------------
The 2026-10-16 controlled comparison measured the two extremes:

    int8  F1 0.882231   R 82.90%   -- 768 TP below the PC-FP32 reference
    fp16  F1 0.901401   R 85.81%   -- only  38 TP below

fp16 is essentially lossless but costs x1.85 in latency AND, on this runtime,
turns the model input into a FLOAT16 tensor, which makes rknn_inputs_set widen
all 640*640*3 elements in user space (45.4 ms/frame measured, versus 0.34 ms
when the graph input is int8). So neither extreme is acceptable: int8 is too
inaccurate for the business criterion (R >= 85%), fp16 is too slow and too
expensive to feed.

The goal is the middle: keep the graph INPUT int8 -- so inputs_set stays at the
int8 cost -- and let only the layers whose quantization error actually hurts run
in fp16.

Toolkit support
---------------
RKNN-Toolkit2 2.3.0 exposes the two-step hybrid API:

    rknn.hybrid_quantization_step1(dataset, proposal=True)
        -> runs calibration, measures per-layer quantization error, and writes
           a PROPOSAL of [input_name, output_name] pairs for the ops that should
           stay float. This is the toolkit's own per-layer loss analysis.

    rknn.hybrid_quantization_step2(model_input, data_input, model_quantization_cfg)
        -> rebuilds the model honouring that list.

The config accepted here is IDENTICAL to the int8/fp16 arms of that comparison
except that quantization is on and float_dtype stays float16, so the only thing
that varies across the three arms is which layers get quantized.

IMPORTANT: the calibration set is imported from the frozen hm_convert_rknn
rather than re-implemented. Re-deriving it here would silently make the hybrid
arm a different experiment from the int8 arm it is supposed to interpolate
between.

Usage:
    python hm_convert_hybrid_rknn.py --mode step1
    python hm_convert_hybrid_rknn.py --mode step2 --cfg <proposal.json> --out <model.rknn>
    python hm_convert_hybrid_rknn.py --mode build --hybrid-level 1 --out <model.rknn>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# The frozen calibration policy (pre-built FEATURE images, train split,
# stratified, deterministic). Imported, never re-implemented -- see module doc.
from hm_convert_rknn import WORK_ROOT, build_calib_list  # noqa: E402

from rknn.api import RKNN  # noqa: E402

DEFAULT_ONNX = WORK_ROOT / "models" / "trial0474_hm_nop0.onnx"
DEFAULT_CALIB = WORK_ROOT / "calib_train_features.txt"
DEFAULT_OUT = WORK_ROOT / "models" / "trial0474_hm_nop0_hybrid.rknn"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["step1", "step2", "build"], required=True)
    ap.add_argument("--onnx", default=str(DEFAULT_ONNX))
    ap.add_argument("--calib-split", default="train")
    ap.add_argument("--calib-n", type=int, default=200)
    ap.add_argument("--calib-list", default=str(DEFAULT_CALIB))
    # Proposal samples: how many calibration images feed the per-layer loss
    # estimate. 1 is the toolkit default and is almost certainly too noisy to
    # decide which layers matter.
    ap.add_argument("--proposal-dataset-size", type=int, default=50)
    ap.add_argument("--cfg", default=str(WORK_ROOT / "hybrid" / "hybrid_quantization_step1.json"))
    ap.add_argument("--data-input", default="")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--optimization-level", type=int, default=3)
    ap.add_argument("--compress-weight", type=int, default=1)
    ap.add_argument("--enable-flash-attention", type=int, default=1)
    ap.add_argument("--single-core-mode", type=int, default=0)
    # Hybrid level 0 = plain int8 (no float fallback). 1..3 progressively trade
    # size/latency for accuracy, which direction is exactly what has to be
    # measured, so all three are built and compared rather than assumed.
    ap.add_argument("--hybrid-level", type=int, default=0)
    # Diagnostic only: builds the uint8-graph-input ONNX with quantization OFF,
    # to separate "uint8 graph input unsupported" from "uint8 + quantization
    # unsupported". A model produced this way is NOT the deliverable.
    ap.add_argument("--no-quant", action="store_true")
    # w8a8 / w8a16 / w16a16 / w4a8 / w4a16 are the values the toolkit accepts.
    # w8a16 (int8 weights, int16 activations) is the candidate that could keep the
    # GRAPH INPUT at 1 byte/px -- which is what keeps rknn_inputs_set at the int8
    # cost -- while giving the body more headroom than int8 activations.
    ap.add_argument("--quantized-dtype", default="w8a8")
    return ap.parse_args()


def onnx_input_dtype(onnx_path):
    """elem_type of the graph's single input, or None if unreadable."""
    try:
        import onnx as _onnx
        g = _onnx.load(onnx_path).graph
        return g.input[0].type.tensor_type.elem_type
    except Exception:
        return None


def base_config(a):
    """hm_convert_rknn's config, with quantization ON.

    mean_values/std_values are applied ONLY when the graph input is FLOAT. For
    the uint8-input variant built by make_uint8_input_onnx.py the /255 already
    lives INSIDE the graph as Cast->Div, so passing std_values=[[255]] here
    would divide a second time: inputs land near 0.001 instead of 0..1 and every
    detection collapses. Getting this wrong is silent -- the build succeeds and
    the model just stops finding anything -- so it is decided from the ONNX
    itself rather than from a flag nobody remembers to set.
    """
    dt = onnx_input_dtype(a.onnx)
    # TensorProto.UINT8 is 2, not 1 (1 is FLOAT). Hardcoding the literal here
    # silently disabled this whole check and produced models that divide by 255
    # twice, so the enum is imported instead.
    import onnx as _onnx
    uint8_graph = (dt == _onnx.TensorProto.UINT8)
    if uint8_graph:
        print(f"[cfg  ] graph input is UINT8 -> omitting mean/std "
              f"(the /255 is already a graph node)")
    cfg = dict(target_platform="rk3588",
               optimization_level=a.optimization_level,
               compress_weight=bool(a.compress_weight),
               enable_flash_attention=bool(a.enable_flash_attention),
               single_core_mode=bool(a.single_core_mode),
               quantized_dtype=a.quantized_dtype,
               float_dtype="float16")
    if not uint8_graph:
        cfg["mean_values"] = [[0, 0, 0]]
        cfg["std_values"] = [[255, 255, 255]]
    if a.hybrid_level:
        cfg["quantized_hybrid_level"] = a.hybrid_level
    return cfg


def do_step1(a):
    calib = build_calib_list(a.calib_split, a.calib_n, Path(a.calib_list))
    print(f"[calib] {len(calib)} pre-built FEATURE images, split={a.calib_split}")
    print(f"[calib] list -> {a.calib_list}")

    print("[onnx ]", a.onnx, f"({os.path.getsize(a.onnx)/1e6:.2f} MB)")
    cfg = base_config(a)
    print("     ", ", ".join(f"{k}={v}" for k, v in cfg.items() if k != "mean_values"))

    rknn = RKNN(verbose=True)
    rknn.config(**cfg)
    ret = rknn.load_onnx(model=a.onnx)
    if ret != 0:
        sys.exit(f"[FATAL] load_onnx failed rc={ret}")

    print(f"[step1] hybrid_quantization_step1(proposal=True, "
          f"proposal_dataset_size={a.proposal_dataset_size})")
    ret = rknn.hybrid_quantization_step1(dataset=a.calib_list,
                                         proposal=True,
                                         proposal_dataset_size=a.proposal_dataset_size)
    print(f"[step1] rc={ret}")
    if ret != 0:
        sys.exit("[FATAL] hybrid_quantization_step1 failed")

    # The C++ layer writes next to the CWD; collect whatever it produced.
    print("[step1] artefacts in", os.getcwd())
    for p in sorted(Path(os.getcwd()).glob("hybrid_quantization_step1*")):
        print(f"    {p}  ({p.stat().st_size} B)")
    return 0


def do_step2(a):
    cfg_path = Path(a.cfg)
    if not cfg_path.is_file():
        sys.exit(f"[FATAL] no quantization cfg at {cfg_path} -- run --mode step1 first")

    onnx = Path(a.onnx)
    print("[onnx ]", onnx, f"({onnx.stat().st_size/1e6:.2f} MB)")
    data_input = a.data_input or a.calib_list
    print("[cfg  ]", cfg_path, f"({cfg_path.stat().st_size} B)")
    print("[data ]", data_input)

    rknn = RKNN(verbose=True)
    ret = rknn.hybrid_quantization_step2(str(onnx), data_input, str(cfg_path))
    print(f"[step2] rc={ret}")
    if ret != 0:
        sys.exit("[FATAL] hybrid_quantization_step2 failed")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ret = rknn.export_rknn(str(out))
    print(f"[export] rc={ret} -> {out}")
    if ret != 0:
        sys.exit("[FATAL] export_rknn failed")
    print(f"[ok] {out}  ({out.stat().st_size} B)")
    return 0


def do_build(a):
    """Plain config -> load_onnx -> build path, with quantized_hybrid_level.

    This is the SAME path that produced the frozen int8 and fp16 arms, so the
    only thing that varies is the hybrid level. The two-step hybrid API is not
    used here on purpose: on this graph it crashes in the toolkit's own quant
    optimizer (step2 raises KeyError on a Sigmoid tensor that the graph fusion
    already removed), which is a toolkit defect and not a property of the model.
    """
    calib = build_calib_list(a.calib_split, a.calib_n, Path(a.calib_list))
    print(f"[calib] {len(calib)} pre-built FEATURE images, split={a.calib_split}")

    onnx = Path(a.onnx)
    print("[onnx ]", onnx, f"({onnx.stat().st_size/1e6:.2f} MB)")
    cfg = base_config(a)
    print("     ", ", ".join(f"{k}={v}" for k, v in cfg.items() if k != "mean_values"))

    rknn = RKNN(verbose=True)
    rknn.config(**cfg)
    if rknn.load_onnx(model=str(onnx)) != 0:
        sys.exit("[FATAL] load_onnx failed")
    print("--> Building")
    ret = rknn.build(do_quantization=not a.no_quant,
                     dataset=None if a.no_quant else a.calib_list)
    if ret != 0:
        sys.exit(f"[FATAL] build failed rc={ret}")
    print("--> Exporting")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rknn.export_rknn(str(out)) != 0:
        sys.exit("[FATAL] export_rknn failed")
    rknn.release()
    if not out.is_file():
        sys.exit(f"[FATAL] export reported success but {out} is absent")
    with open(out, "rb") as f:
        magic = f.read(4)
    if magic != b"RKNN":
        sys.exit(f"[FATAL] {out} does not start with RKNN magic (got {magic!r})")
    print(f"[ok] {out}  {out.stat().st_size/1e6:.2f} MB  magic=RKNN")
    return 0


def main():
    a = parse_args()
    if a.mode == "step1":
        return do_step1(a)
    if a.mode == "step2":
        return do_step2(a)
    return do_build(a)


if __name__ == "__main__":
    sys.exit(main())
