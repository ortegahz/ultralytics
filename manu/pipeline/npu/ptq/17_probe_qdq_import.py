#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
17_probe_qdq_import.py -- the decisive probe for "is QAT worth doing tomorrow?".

The question
------------
    Does rknn-toolkit2 2.3.2 ADOPT the scale / zero_point that a QuantizeLinear /
    DequantizeLinear (Q/DQ) pair carries, when it imports an ONNX model?

Why the answer decides tomorrow
-------------------------------
QAT learns the activation clipping thresholds and writes them into the Q/DQ
nodes of the exported ONNX.  If the toolkit adopts them, those thresholds reach
the NPU and QAT buys what it promises.  If it ignores them, the toolkit
re-quantises the activations itself and every bit of activation-side QAT work is
thrown away on the way to the board -- so QAT would only ever be worth its
weight-side share (a small share).

How the answer is read
----------------------
The board driver (/root/rknn_heatmap_smoke) prints the quantisation parameters
of the deployed model's output tensors:

    out[0] dims=[1,1,320,320] fmt=NCHW type=INT8 zp=-128 scale=0.00378871

That line is the only readout of the deployed quantisation parameters that
exists (the toolkit's Python API does not expose them), so it is what this
probe compares against the numbers written into the Q/DQ.

Baselines already measured on this project (same board, same driver, same
heatmap model):

    do_quantization=False (float build)  -> out[0] type=FLOAT16 zp=0 scale=1.0
    do_quantization=True  (normal algo)  -> out[0] type=INT8 zp=-128 scale=0.00378871

so the read-out table is:

    out[0] carries the numbers written in the Q/DQ  -> QDQ_ADOPTED
    out[0] is identical to the baseline              -> QDQ_IGNORED
    the build refuses the model                       -> QDQ_BUILD_FAILED

The second, independent read-out (the value grid)
--------------------------------------------------
When the DequantizeLinear is the graph output, the output tensor is float and
its printed zp/scale cannot carry the answer on its own.  The board's --dump
option writes the DEQUANTISED float32 output, so this script also asks whether
those values sit on the grid the Q/DQ defines: an adopted Q/DQ leaves at most
256 distinct values (8-bit) spaced exactly `scale` apart; a graph the toolkit
re-quantised (or left in float) does not.

The shapes are not free
-----------------------
The board driver refuses anything that is not 1 input of 640x640x3 and 2
outputs of [1,1,H,W] + [1,2,H,W] with 640/H == 2, so every probe model here is
built to that contract.  A bare `y = DQ(Q(x))` graph cannot be run on the board
at all -- the minimal probe therefore carries a 1x1 stride-2 convolution that
produces the required [1,1,320,320] grid.

Where it runs
-------------
On the SERVER: it needs rknn-toolkit2 and the key-based `ptq-board` ssh alias.
The board is driven ONE process at a time -- the board holds at most two RKNN
contexts and a third one soft-resets the NPU.

Usage
-----
    source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate rknn
    python manu/pipeline/npu/ptq/17_probe_qdq_import.py
    python manu/pipeline/npu/ptq/17_probe_qdq_import.py --cases hm_qdq_q hm_qdq_dq
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Paths (overridable from the environment; the defaults are the server's own layout)
# --------------------------------------------------------------------------------------
SERVER_ROOT = Path(os.environ.get("PTQ_SERVER_ROOT", "/tmp/pycharm_project_10ae9e2e"))
HEATMAP_ONNX = Path(os.environ.get(
    "PTQ_PROBE_HEATMAP_ONNX", str(SERVER_ROOT / "runs/onnx/trial_0080_hm_nop0.onnx")))
WORK_ROOT = Path(os.environ.get("PTQ_WORK_ROOT", "/mnt/data/siping/ptq_work"))
OUT_DIR = Path(os.environ.get("PTQ_QDQ_PROBE_DIR", str(WORK_ROOT / "qdq_probe")))
CALIB_LIST = Path(os.environ.get("PTQ_QDQ_PROBE_CALIB", str(WORK_ROOT / "calib/set_A.txt")))
RESULT_JSON = OUT_DIR / "qdq_probe.json"

# the board: key-based ssh alias on the server, no password anywhere
BOARD_ALIAS = os.environ.get("PTQ_BOARD_ALIAS", "ptq-board")
BOARD_BIN = "/root/rknn_heatmap_smoke"
BOARD_WORK = "/root/ptq/qdq_probe"
BOARD_DUMP = BOARD_WORK + "/dump"
BOARD_BMP = ("/mnt/manu/ptq/bmp/VIDEO00007_19700101_001324/"
             "VIDEO00007_19700101_001324__000000.bmp")

# The numbers written into the Q/DQ nodes -- deliberately unmistakable.
PROBE_SCALE_HM = 0.001234
PROBE_ZP_HM = -100
PROBE_SCALE_PURE = 0.00555
PROBE_ZP_PURE = -77

# The board prints the tensor attributes once per model load:
#     input dims=[1,640,640,3] fmt=NHWC type=INT8
#     out[0] dims=[1,1,320,320] fmt=NCHW type=INT8 zp=-128 scale=0.00378871
# (same regexes as 04_run_board_fast_eval.py, which cannot be imported: its file
# name starts with a digit and is therefore not a legal module name.)
_RE_INPUT_ATTR = re.compile(r"input dims=\[([^\]]*)\]\s*fmt=(\S+)\s+type=(\S+)")
_RE_OUT_ATTR = re.compile(
    r"out\[(\d+)\] dims=\[([^\]]*)\]\s*fmt=(\S+)\s+type=(\S+)\s+zp=(-?\d+)\s+scale=([0-9eE+\-.]+)")


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def sh(cmd: list[str], timeout: int = 3600) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def md5_file(path: Path) -> str:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_board_attrs(stdout: str) -> dict:
    """Extract the input/output tensor attributes from the board's stdout."""
    out: dict = {"input_attr": None, "out_attrs": []}
    text = stdout or ""
    m = _RE_INPUT_ATTR.search(text)
    if m:
        out["input_attr"] = {
            "dims": [int(x) for x in m.group(1).split(",") if x.strip()],
            "fmt": m.group(2), "type": m.group(3), "raw": m.group(0).strip()}
    for mo in _RE_OUT_ATTR.finditer(text):
        out["out_attrs"].append({
            "index": int(mo.group(1)),
            "dims": [int(x) for x in mo.group(2).split(",") if x.strip()],
            "fmt": mo.group(3), "type": mo.group(4),
            "zp": int(mo.group(5)), "scale": float(mo.group(6)),
            "raw": mo.group(0).strip()})
    return out


# --------------------------------------------------------------------------------------
# ONNX construction
# --------------------------------------------------------------------------------------
def _qdq_initializers(scale: float, zp: int):
    from onnx import numpy_helper

    return (numpy_helper.from_array(np.array([scale], dtype=np.float32), "qdq_scale"),
            numpy_helper.from_array(np.array([zp], dtype=np.int8), "qdq_zp"))


def make_pure_probe(path: Path, scale: float, zp: int, q_only: bool, opset: int = 13) -> dict:
    """The minimal probe: images -> 1x1 stride-2 conv -> Q(/D Q) -> out.

    The convolution exists only to give the board the shapes it insists on
    ([1,1,320,320] and [1,2,320,320] out of a 640x640x3 input); the Q/DQ pair
    is the thing under test.
    """
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    w = np.array([0.0001, 0.0002, 0.0003], dtype=np.float32).reshape(1, 3, 1, 1)
    b = np.array([0.0], dtype=np.float32)
    s_init, z_init = _qdq_initializers(scale, zp)

    inits = [numpy_helper.from_array(w, "conv_w"),
             numpy_helper.from_array(b, "conv_b"),
             s_init, z_init]
    nodes = [
        helper.make_node("Conv", ["images", "conv_w", "conv_b"], ["conv_out"],
                         name="probe_conv", kernel_shape=[1, 1], strides=[2, 2],
                         pads=[0, 0, 0, 0], dilations=[1, 1], group=1),
        helper.make_node("QuantizeLinear", ["conv_out", "qdq_scale", "qdq_zp"], ["q_out"],
                         name="probe_quantize"),
    ]
    if q_only:
        out0_name, out0_elem = "q_out", TensorProto.INT8
    else:
        nodes.append(helper.make_node("DequantizeLinear", ["q_out", "qdq_scale", "qdq_zp"],
                                      ["dq_out"], name="probe_dequantize"))
        out0_name, out0_elem = "dq_out", TensorProto.FLOAT
    nodes.append(helper.make_node("Concat", [out0_name, out0_name], ["concat_out"],
                                  name="probe_concat", axis=1))

    graph = helper.make_graph(
        nodes, "qdq_pure_probe",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info(out0_name, out0_elem, [1, 1, 320, 320]),
         helper.make_tensor_value_info("concat_out", out0_elem, [1, 2, 320, 320])],
        initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return {"scale": scale, "zero_point": zp, "mode": "q" if q_only else "dq",
            "output_is_int8": bool(q_only)}


def insert_qdq_into_heatmap(src: Path, dst: Path, scale: float, zp: int,
                             mode: str) -> dict:
    """Put a Q(/D Q) pair with the given parameters on the heatmap output.

    mode="q"  : the graph output becomes the int8 QuantizeLinear output.
    mode="dq" : the graph output becomes the float DequantizeLinear output
                (this is what a QAT-exported ONNX actually looks like).
    """
    import onnx
    from onnx import helper, TensorProto

    model = onnx.load(str(src))
    graph = model.graph
    out0 = graph.output[0]
    name = out0.name                                   # "heatmap"
    producer = None
    for node in graph.node:
        if name in node.output:
            producer = node
            break
    if producer is None:
        raise RuntimeError("no producer node for the output tensor " + name)

    idx = list(producer.output).index(name)
    base = name + "_pre"
    producer.output[idx] = base
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == name:
                node.input[i] = base

    # stale value_info entries for the old name would contradict the new graph
    keep = [vi for vi in graph.value_info if vi.name not in (name, base)]
    del graph.value_info[:]
    graph.value_info.extend(keep)

    s_init, z_init = _qdq_initializers(scale, zp)
    graph.initializer.append(s_init)
    graph.initializer.append(z_init)

    q_out = name + "_q"
    new_nodes = [helper.make_node("QuantizeLinear", [base, "qdq_scale", "qdq_zp"], [q_out],
                                   name="qdq_probe_quantize")]
    if mode == "q":
        final, elem = q_out, TensorProto.INT8
    else:
        final = name + "_dq"
        new_nodes.append(helper.make_node("DequantizeLinear", [q_out, "qdq_scale", "qdq_zp"],
                                           [final], name="qdq_probe_dequantize"))
        elem = TensorProto.FLOAT

    at = list(graph.node).index(producer)
    for k, node in enumerate(new_nodes):
        graph.node.insert(at + 1 + k, node)

    out0.name = final
    out0.type.tensor_type.elem_type = elem
    onnx.save(model, str(dst))
    return {"scale": scale, "zero_point": zp, "mode": mode,
            "output_is_int8": bool(mode == "q")}


def make_torchao_probe(path: Path, opset: int = 13, per_tensor: bool = False) -> dict:
    """A small conv net quantized with torch.ao, exported as a Q/DQ ONNX.

    The net is shaped so that the exported model satisfies the board's contract
    (1 input 640x640x3, outputs [1,1,320,320] and [1,2,320,320]).
    """
    import torch
    import torch.nn as nn
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(3, 4, 3, stride=2, padding=1)
            self.c2 = nn.Conv2d(4, 2, 3, stride=1, padding=1)
            self.c3 = nn.Conv2d(4, 1, 3, stride=1, padding=1)

        def forward(self, x):
            h = self.c1(x)
            return self.c3(h), self.c2(h)

    torch.manual_seed(0)
    model = Tiny().eval()
    x = torch.zeros(1, 3, 640, 640)

    if per_tensor:
        try:
            from torch.ao.quantization import QconfigMapping
        except ImportError:                       # torch 2.4 moved it
            from torch.ao.quantization.qconfig import QconfigMapping
        from torch.ao.quantization import FakeQuantize, MinMaxObserver, QConfig
        qconfig = QConfig(activation=FakeQuantize.with_args(observer=MinMaxObserver),
                           weight=MinMaxObserver)
        mapping = QconfigMapping().set_global(qconfig)
    else:
        mapping = get_default_qconfig_mapping("x86")

    prepared = prepare_fx(model, mapping, (x,))
    for _ in range(4):
        prepared(torch.rand(1, 3, 640, 640))
    quantized = convert_fx(prepared)
    quantized.eval()
    torch.onnx.export(quantized, (x,), str(path), opset_version=opset,
                      input_names=["images"], output_names=["heatmap", "offset"],
                      do_constant_folding=True)
    return {"per_tensor_weights": bool(per_tensor), "opset": opset}


def read_output_qdq(onnx_path: Path) -> dict | None:
    """The scale/zero_point that the ONNX itself writes on its FIRST output.

    This is the number the board has to echo back for the Q/DQ to count as
    adopted.  For a graph whose output is a DequantizeLinear it is the DQ's
    own parameters; for a graph whose output is a QuantizeLinear it is that
    Q's.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))
    inits = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    # a torch.ao export writes the scale/zero_point as Constant NODES, not as
    # initializers, so both spellings have to be resolved
    consts: dict = {}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    consts[node.output[0]] = numpy_helper.to_array(attr.t)
    out_name = model.graph.output[0].name

    def params(node) -> dict | None:
        if len(node.input) < 2:
            return None
        s = inits.get(node.input[1], consts.get(node.input[1]))
        if s is None:
            return None
        z = None
        if len(node.input) > 2:
            z = inits.get(node.input[2], consts.get(node.input[2]))
        flat = s.reshape(-1)
        axis = None
        for attr in node.attribute:
            if attr.name == "axis":
                axis = int(attr.i)
        return {"scale": float(flat[0]), "n_scales": int(flat.size),
                "zero_point": (int(z.reshape(-1)[0]) if z is not None else None),
                "axis": axis, "node": node.name or node.output[0]}

    for node in model.graph.node:
        if node.op_type == "DequantizeLinear" and out_name in node.output:
            return params(node)
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear" and out_name in node.output:
            return params(node)
    return None


# --------------------------------------------------------------------------------------
# the build (run as a subprocess so the toolkit's C-level log is captured verbatim)
# --------------------------------------------------------------------------------------
def build_one(onnx_path: Path, out_rknn: Path, do_quant: bool, calib: Path | None,
              mean_std: bool) -> int:
    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    cfg: dict = {"target_platform": "rk3588", "optimization_level": 3,
                 "quantized_method": "channel"}
    if mean_std:
        cfg["mean_values"] = [[0, 0, 0]]
        cfg["std_values"] = [[255, 255, 255]]
    rknn.config(**cfg)

    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        print("LOAD_ONNX_FAILED ret=%d" % ret)
        return 2
    ret = rknn.build(do_quantization=do_quant,
                     dataset=str(calib) if (do_quant and calib) else None)
    if ret != 0:
        print("BUILD_FAILED ret=%d" % ret)
        return 3
    ret = rknn.export_rknn(str(out_rknn))
    if ret != 0:
        print("EXPORT_FAILED ret=%d" % ret)
        return 4
    print("BUILD_OK %s" % out_rknn)
    return 0


def run_build(onnx_path: Path, out_rknn: Path, do_quant: bool, calib: Path | None,
              mean_std: bool) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--build-one",
           "--onnx", str(onnx_path), "--out-rknn", str(out_rknn)]
    if do_quant:
        cmd += ["--do-quant"]
        if calib:
            cmd += ["--calib", str(calib)]
    if mean_std:
        cmd += ["--mean-std"]
    t0 = time.time()
    rc, out, err = sh(cmd, timeout=3600)
    return {"rc": rc, "seconds": round(time.time() - t0, 1),
            "stdout": out[-8000:], "stderr": err[-4000:]}


# --------------------------------------------------------------------------------------
# the board
# --------------------------------------------------------------------------------------
def board_leftovers() -> list[str]:
    rc, out, err = sh(["ssh", "-o", "StrictHostKeyChecking=no", BOARD_ALIAS,
                        "ps -eo args | grep rknn_heatmap_smoke | grep -v grep"])
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def board_prepare() -> tuple[int, str]:
    # The dump dir is wiped on every case: the board names its dump after the
    # INPUT file, so a leftover dump from the previous case would silently be
    # read as this case's output.
    return sh(["ssh", "-o", "StrictHostKeyChecking=no", BOARD_ALIAS,
               "rm -rf " + BOARD_DUMP + " && mkdir -p " + BOARD_WORK + " " + BOARD_DUMP +
               " && printf '%s\\n' " + BOARD_BMP + " > " + BOARD_WORK + "/list.txt"])


def board_run(remote_model: str, tag: str, dump: bool = True, attempts: int = 3) -> dict:
    # stdbuf: the board's stdout is a pipe, so it is fully buffered -- if the
    # process dies later (a probe model can drive the driver into its null
    # control or crash it) the attribute lines would be lost with the buffer.
    # Unbuffered stdout keeps the evidence that was printed before the death.
    run_cmd = ("stdbuf -o0 -e0 " + BOARD_BIN + " " + remote_model +
               " --batch " + BOARD_WORK + "/list.txt --th 0.3 --topk 100" +
               " --detections-out " + BOARD_WORK + "/det.tsv")
    if dump:
        run_cmd += " --dump " + BOARD_DUMP
    res: dict = {"rc": -1, "attempts": []}
    for attempt in range(1, attempts + 1):
        t0 = time.time()
        rc, out, err = sh(["ssh", "-o", "StrictHostKeyChecking=no", BOARD_ALIAS, run_cmd],
                          timeout=1800)
        res["attempts"].append({"attempt": attempt, "rc": rc,
                                "seconds": round(time.time() - t0, 1),
                                "stdout_len": len(out), "stderr": err[-400:]})
        if rc == 0 or out.strip():
            # rc==0 -> a clean run; a non-empty stdout means the board did print
            # the tensor attributes, which is all this probe needs.
            res["rc"] = rc
            res["seconds"] = round(time.time() - t0, 1)
            res["stdout"] = out[-16000:]
            res["stderr"] = err[-4000:]
            break
        # rc==255 with an EMPTY stdout is an sshd-level refusal (the probe
        # opens several ssh connections in a row and sshd throttles them), not
        # a board failure -- retry after a short pause.
        time.sleep(3)
    else:
        rc, out, err = sh(["ssh", "-o", "StrictHostKeyChecking=no", BOARD_ALIAS, run_cmd],
                          timeout=1800)
        res["rc"] = rc
        res["seconds"] = 0.0
        res["stdout"] = out[-16000:]
        res["stderr"] = err[-4000:]
    res.update(parse_board_attrs(res.get("stdout", "")))
    return res


def board_fetch_dump(local_path: Path) -> Path | None:
    rc, out, err = sh(["ssh", "-o", "StrictHostKeyChecking=no", BOARD_ALIAS,
                        "ls " + BOARD_DUMP + "/*_hm.bin 2>/dev/null | head -1"])
    remote = out.strip().splitlines()[0] if out.strip() else ""
    if not remote:
        return None
    rc, out, err = sh(["scp", "-o", "StrictHostKeyChecking=no", BOARD_ALIAS + ":" + remote,
                        str(local_path)])
    return local_path if rc == 0 and local_path.is_file() else None


def grid_test(bin_path: Path, scale: float, zp: int) -> dict | None:
    """Is the board's float output sitting on the grid the Q/DQ defines?"""
    vals = np.fromfile(str(bin_path), dtype=np.float32)
    if vals.size == 0:
        return None
    uniq = np.unique(vals)
    out = {"n_values": int(vals.size), "n_unique": int(uniq.size),
           "min": float(vals.min()), "max": float(vals.max())}
    if uniq.size >= 2:
        diffs = np.diff(uniq.astype(np.float64))
        s_impl = float(np.median(diffs[diffs > 0]))
        out["scale_implied"] = s_impl
        out["scale_ratio"] = (s_impl / scale) if scale else None
    r = vals.astype(np.float64) / scale + zp
    out["on_grid_fraction"] = float(np.mean(np.abs(r - np.round(r)) < 0.25))
    out["median_grid_error"] = float(np.median(np.abs(r - np.round(r))))
    out["grid_matches"] = bool(uniq.size <= 256 and
                                out.get("scale_ratio") is not None and
                                abs(out["scale_ratio"] - 1.0) < 0.02)
    return out


# --------------------------------------------------------------------------------------
# the cases
# --------------------------------------------------------------------------------------
def case_list() -> list[dict]:
    return [
        {"tag": "probe_pure_q",
         "desc": "minimal probe, Q output is int8 (s=0.00555 zp=-77)",
         "source": "pure", "mode": "q", "scale": PROBE_SCALE_PURE, "zp": PROBE_ZP_PURE,
         "do_quant": False, "mean_std": False},
        {"tag": "probe_pure_dq",
         "desc": "minimal probe, Q/DQ output is float (s=0.00555 zp=-77)",
         "source": "pure", "mode": "dq", "scale": PROBE_SCALE_PURE, "zp": PROBE_ZP_PURE,
         "do_quant": False, "mean_std": False},
        {"tag": "hm_qdq_q",
         "desc": "real heatmap ONNX, Q output is int8 (s=0.001234 zp=-100)",
         "source": "heatmap", "mode": "q", "scale": PROBE_SCALE_HM, "zp": PROBE_ZP_HM,
         "do_quant": False, "mean_std": False},
        {"tag": "hm_qdq_dq",
         "desc": "real heatmap ONNX, Q/DQ output is float (s=0.001234 zp=-100)",
         "source": "heatmap", "mode": "dq", "scale": PROBE_SCALE_HM, "zp": PROBE_ZP_HM,
         "do_quant": False, "mean_std": False},
        {"tag": "hm_plain",
         "desc": "control: real heatmap ONNX, no Q/DQ, same build recipe",
         "source": "heatmap", "mode": None, "do_quant": False, "mean_std": False},
        {"tag": "hm_plain_quant",
         "desc": "control: real heatmap ONNX, no Q/DQ, do_quantization=True "
                 "(must reproduce the known baseline zp=-128 scale=0.00378871)",
         "source": "heatmap", "mode": None, "do_quant": True, "mean_std": True},
        {"tag": "hm_qdq_dq_quant",
         "desc": "real heatmap ONNX + Q/DQ, built WITH do_quantization=True",
         "source": "heatmap", "mode": "dq", "scale": PROBE_SCALE_HM, "zp": PROBE_ZP_HM,
         "do_quant": True, "mean_std": True},
        {"tag": "torchao_qdq_pc",
         "desc": "torch.ao quantized small conv net (per-channel weights) -> Q/DQ ONNX",
         "source": "torchao", "per_tensor": False, "do_quant": False, "mean_std": False},
        {"tag": "torchao_qdq_pt",
         "desc": "torch.ao quantized small conv net (per-tensor weights) -> Q/DQ ONNX",
         "source": "torchao", "per_tensor": True, "do_quant": False, "mean_std": False},
    ]


def build_onnx_for_case(case: dict, out_dir: Path) -> tuple[Path, dict]:
    src = case["source"]
    if src == "pure":
        path = out_dir / (case["tag"] + ".onnx")
        info = make_pure_probe(path, case["scale"], case["zp"], case["mode"] == "q")
        return path, info
    if src == "heatmap":
        if case["mode"] is None:
            path = out_dir / (case["tag"] + ".onnx")
            info = {"mode": None, "note": "unmodified heatmap ONNX (control)"}
            path.write_bytes(HEATMAP_ONNX.read_bytes())
            return path, info
        path = out_dir / (case["tag"] + ".onnx")
        info = insert_qdq_into_heatmap(HEATMAP_ONNX, path, case["scale"], case["zp"],
                                       case["mode"])
        return path, info
    if src == "torchao":
        path = out_dir / (case["tag"] + ".onnx")
        info = make_torchao_probe(path, per_tensor=case.get("per_tensor", False))
        return path, info
    raise RuntimeError("unknown source " + str(src))


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-one", action="store_true",
                    help="internal: build a single model and exit")
    ap.add_argument("--onnx", help="internal: ONNX to build")
    ap.add_argument("--out-rknn", help="internal: RKNN to write")
    ap.add_argument("--do-quant", action="store_true", help="internal: quantise")
    ap.add_argument("--calib", help="internal: calibration list")
    ap.add_argument("--mean-std", action="store_true", help="internal: mean/std config")
    ap.add_argument("--cases", nargs="*", default=None, help="subset of case tags")
    ap.add_argument("--merge", action="store_true",
                    help="merge the re-run cases into the existing result JSON "
                         "instead of replacing it")
    ap.add_argument("--keep-board", action="store_true",
                    help="do not delete the board's scratch dir at the end")
    args = ap.parse_args()

    if args.build_one:
        sys.exit(build_one(Path(args.onnx), Path(args.out_rknn), args.do_quant,
                           Path(args.calib) if args.calib else None, args.mean_std))

    import importlib.metadata

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_dir = OUT_DIR / "onnx"
    rknn_dir = OUT_DIR / "rknn"
    dump_dir = OUT_DIR / "dump"
    for d in (onnx_dir, rknn_dir, dump_dir):
        d.mkdir(parents=True, exist_ok=True)

    cases = case_list()
    if args.cases:
        wanted = set(args.cases)
        cases = [c for c in cases if c["tag"] in wanted]

    report: dict = {
        "probe": "qdq_import",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": os.uname().nodename,
        "rknn_toolkit2_version": importlib.metadata.version("rknn-toolkit2"),
        "heatmap_onnx": str(HEATMAP_ONNX),
        "board_alias": BOARD_ALIAS,
        "board_bmp": BOARD_BMP,
        "calib_list": str(CALIB_LIST),
        "cases": [],
    }
    print("[probe] rknn-toolkit2 %s, heatmap ONNX %s"
          % (report["rknn_toolkit2_version"], HEATMAP_ONNX), flush=True)

    for case in cases:
        tag = case["tag"]
        print("\n[case] %s -- %s" % (tag, case["desc"]), flush=True)
        rec: dict = {"tag": tag, "description": case["desc"],
                     "do_quantization": case["do_quant"], "mean_std": case["mean_std"],
                     "qdq": None}
        try:
            onnx_path, qinfo = build_onnx_for_case(case, onnx_dir)
        except Exception as exc:                                     # noqa: BLE001
            rec["error"] = "onnx_build_failed: %s: %s" % (type(exc).__name__, exc)
            rec["verdict"] = "ONNX_EXPORT_FAILED"
            report["cases"].append(rec)
            print("        [FAIL] could not build the ONNX: %s" % rec["error"], flush=True)
            continue
        rec["onnx"] = str(onnx_path)
        rec["onnx_md5"] = md5_file(onnx_path)
        rec["qdq"] = qinfo
        try:
            rec["onnx_output_qdq"] = read_output_qdq(onnx_path)
        except Exception as exc:                                     # noqa: BLE001
            rec["onnx_output_qdq"] = {"error": "%s: %s" % (type(exc).__name__, exc)}
        print("        onnx: %s (md5 %s)" % (onnx_path.name, rec["onnx_md5"]), flush=True)
        print("        onnx output Q/DQ: %s" % json.dumps(rec["onnx_output_qdq"]),
              flush=True)

        rknn_path = rknn_dir / (tag + ".rknn")
        build = run_build(onnx_path, rknn_path, case["do_quant"], CALIB_LIST,
                          case["mean_std"])
        rec["build"] = build
        rec["build"]["rknn"] = str(rknn_path) if rknn_path.is_file() else None
        rec["build"]["rknn_md5"] = md5_file(rknn_path) if rknn_path.is_file() else None
        print("        build rc=%d (%.1fs)" % (build["rc"], build["seconds"]), flush=True)
        if build["rc"] != 0:
            tail = (build["stdout"] or "")[-900:].replace("\n", " | ")
            print("        [FAIL] build: %s" % tail, flush=True)
            rec["verdict"] = "QDQ_BUILD_FAILED"
            report["cases"].append(rec)
            continue

        # ---- the board: ONE process at a time ----
        pre = board_leftovers()
        if pre:
            print("        [WARN] leftover board processes before the run: %s" % pre,
                  flush=True)
        board_prepare()
        remote = BOARD_WORK + "/" + rknn_path.name
        rc, out, err = sh(["scp", "-o", "StrictHostKeyChecking=no", str(rknn_path),
                            BOARD_ALIAS + ":" + remote])
        if rc != 0:
            rec["board"] = {"rc": -1, "error": "scp failed: " + err[-300:]}
            rec["verdict"] = "QDQ_BUILD_FAILED"
            report["cases"].append(rec)
            continue
        run = board_run(remote, tag)
        rec["board"] = run
        for a in run.get("out_attrs", []):
            print("        BOARD %s" % a["raw"], flush=True)
        if run.get("input_attr"):
            print("        BOARD %s" % run["input_attr"]["raw"], flush=True)
        if run["rc"] != 0:
            print("        board rc=%d stderr: %s"
                  % (run["rc"], (run.get("stderr") or "")[-300:].replace("\n", " | ")),
                  flush=True)
        rec["board"]["leftovers_after"] = board_leftovers()

        # ---- the second read-out: is the float output on the Q/DQ's grid? ----
        if case["source"] in ("pure", "heatmap") and case["mode"] == "dq":
            local_bin = dump_dir / (tag + "_hm.bin")
            if local_bin.exists():
                local_bin.unlink()          # never read a stale dump
            fetched = board_fetch_dump(local_bin)
            if fetched is not None:
                gt = grid_test(fetched, case["scale"], case["zp"])
                rec["grid_test"] = gt
                print("        grid: %s" % json.dumps(gt), flush=True)
            else:
                rec["grid_test"] = {"note": "the board wrote no dump for this case"}

        rec["verdict"] = judge_case(case, rec)
        print("        -> %s" % rec["verdict"], flush=True)
        report["cases"].append(rec)

    if args.merge and RESULT_JSON.is_file():
        old = json.loads(RESULT_JSON.read_text())
        by_tag = {c["tag"]: c for c in old.get("cases", [])}
        order = [c["tag"] for c in old.get("cases", [])]
        for c in report["cases"]:
            by_tag[c["tag"]] = c
            if c["tag"] not in order:
                order.append(c["tag"])
        report["cases"] = [by_tag[t] for t in order]
        print("[probe] merged %d case(s) into the existing %s"
              % (len(report["cases"]), RESULT_JSON), flush=True)

    report["verdict"], report["verdict_reason"] = overall_verdict(report)
    report["result_json"] = str(RESULT_JSON)

    if not args.keep_board:
        sh(["ssh", "-o", "StrictHostKeyChecking=no", BOARD_ALIAS,
            "rm -rf " + BOARD_WORK])
    report["board_leftovers_final"] = board_leftovers()

    RESULT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n=========================================================")
    print("VERDICT: %s" % report["verdict"])
    print("%s" % report["verdict_reason"])
    print("json: %s" % RESULT_JSON)
    print("=========================================================")


def attr_matches(a: dict, scale: float, zp: int | None) -> bool:
    """Does this board-reported tensor attribute carry the Q/DQ's numbers?

    The board prints the scale with 8 decimals, so the comparison has to
    allow for that rounding.  A zero point may also come back shifted by 128:
    torch.ao writes a UINT8 quantisation (zp=37) and the toolkit re-centres it
    on the int8 grid (37-128 = -91) -- the same range, the same scale.
    """
    if scale is None or a.get("scale") is None:
        return False
    if abs(float(a["scale"]) - float(scale)) > 1e-8 + 1e-6 * abs(float(scale)):
        return False
    if zp is None:
        return True
    if int(a["zp"]) in (int(zp), int(zp) - 128, int(zp) + 128):
        return True
    return False


def judge_case(case: dict, rec: dict) -> str:
    """Did THIS case's board output carry the numbers written into the Q/DQ?

    The reference is the ONNX file itself: the scale/zero_point it writes on
    the Q/DQ that feeds the model's first output.  The board has to echo
    those numbers on one of the model's output tensors -- a DequantizeLinear
    output is float by construction, so the numbers then land on the
    quantised consumer tensor instead of on the float one.
    """
    if rec.get("build", {}).get("rc", 1) != 0:
        return "QDQ_BUILD_FAILED"
    if case.get("mode") is None and case.get("source") == "heatmap":
        return "CONTROL"          # the no-Q/DQ control build of the real model
    attrs = rec.get("board", {}).get("out_attrs", [])
    if not attrs:
        return "NO_BOARD_OUTPUT"

    want = rec.get("onnx_output_qdq") or {}
    scale = want.get("scale")
    zp = want.get("zero_point")
    if scale is None:
        scale, zp = case.get("scale"), case.get("zp")
    for a in attrs:
        if attr_matches(a, scale, zp):
            return "QDQ_ADOPTED"
    gt = rec.get("grid_test") or {}
    if gt.get("grid_matches"):
        return "QDQ_ADOPTED"
    return "QDQ_IGNORED"


def overall_verdict(report: dict) -> tuple[str, str]:
    adopted, ignored, failed, built, exported_failed = [], [], [], [], []
    for c in report["cases"]:
        v = c.get("verdict")
        if v == "QDQ_ADOPTED":
            adopted.append(c["tag"])
        elif v == "QDQ_IGNORED":
            ignored.append(c["tag"])
        elif v == "QDQ_BUILD_FAILED":
            failed.append(c["tag"])
        elif v == "ONNX_EXPORT_FAILED":
            exported_failed.append(c["tag"])
        elif v == "QDQ_BUILT_OK":
            built.append(c["tag"])
    if adopted:
        return ("QDQ_ADOPTED",
                "the board printed the scale/zero_point written into the Q/DQ for: "
                + ", ".join(adopted))
    if ignored:
        return ("QDQ_IGNORED",
                "the Q/DQ built fine but the board's out[0] never carried the written "
                "scale/zero_point (ignored in: " + ", ".join(ignored) + ")")
    if failed and not built:
        return ("QDQ_BUILD_FAILED",
                "every Q/DQ case failed to build: " + ", ".join(failed)
                + (" (ONNX export failures: " + ", ".join(exported_failed) + ")"
                   if exported_failed else ""))
    if failed:
        return ("QDQ_BUILD_FAILED",
                "the Q/DQ cases that carried a hand-written scale all failed to "
                "build: " + ", ".join(failed))
    return ("QDQ_IGNORED", "no case carried a hand-written scale to the board")


if __name__ == "__main__":
    main()
