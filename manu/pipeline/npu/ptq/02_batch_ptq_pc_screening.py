#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
02_batch_ptq_pc_screening.py -- build the RKNN PTQ matrix and screen it on the PC simulator.

What it does, per configuration
-------------------------------
    build  : rknn.config(target_platform='rk3588', quantized_method='channel',
                         optimization_level=3, quantized_algorithm=<algo>)
             -> rknn.build(do_quantization=True, dataset=<the 250-image BMP list>)
             -> rknn.export_rknn(...)

    screen : rknn.init_runtime(target=None)   # PC simulator
             feed the SAME uint8 NHWC buffer the board will feed
             -> per-layer cosine similarity + MSE against the float ONNX graph
             -> output-level cosine similarity + MSE
             -> end-to-end detections (peaks decoded with the SAME code as the board)
                scored against the PT reference on the same frames

Why the simulator, and what it can and cannot prove
---------------------------------------------------
The PC simulator is a CPU emulation of the NPU: same quantiser, same calibration,
same operator set -- but a different numeric implementation, so its numbers are
close to the board's, NOT equal. It is used here to RANK 16 candidates cheaply.
The board run (04_run_board_fast_eval.py) is what actually decides.

Two hard facts that shape this script
-------------------------------------
1. A model loaded with `load_rknn()` cannot be simulated ("not support inference
   on the simulator"). The simulator only accepts a model that was built in the
   SAME process. Therefore build+simulate cannot be split across two scripts.
   => every configuration is built AND screened inside one worker process.
2. The `mmse` algorithm makes the toolkit allocate tens of GB. Running all
   configurations at once will OOM the box, so the driver caps the parallelism
   and runs the memory-hungry algorithm in a smaller pool.

Usage
-----
    python 02_batch_ptq_pc_screening.py --variant p0_off --sets A B C D \
        --algos kl_divergence mmse normal --jobs 4 --out /mnt/data/siping/ptq_work/screen
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

# parents[4] is the REPO ROOT (manu/pipeline/npu/ptq/x.py -> repo root).  The old
# parent.parent.parent.parent pointed at .../<repo>/manu, so `import manu...`
# only resolved when PYTHONPATH happened to carry the repo root -- running this
# file directly (python manu/pipeline/npu/ptq/02_....py) died with
# ModuleNotFoundError: No module named 'manu'.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
ALGOS = ["kl_divergence", "mmse", "normal", "gdq"]
# mmse is a memory monster; keep it out of the wide pool
MEMORY_HEAVY = {"mmse"}
# gdq = Gradient-based Distillation Quantization: the toolkit's own
# QAT-flavoured algorithm (it optimises the quantisation scales with a
# gradient step on the calibration data instead of a closed-form clip).
# It is the official "quantisation-aware" knob available in toolkit 2.3.2,
# so it is the cheap stand-in for a hand-written QAT -- and it is the only
# algorithm in this family that was NOT covered by the A/B/C/D sweep.
MEMORY_HEAVY |= {"gdq"}
# algos that build a NON-quantized (float16) model
F16_ALGOS = {"none"}


# --------------------------------------------------------------------------------------
# Tier-1 sensitive frames: 50 dark-weak rigid-body frames + 50 surface-clutter frames
# --------------------------------------------------------------------------------------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float(a @ b / (na * nb))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(d * d))


def _cfg_suffix(quant_method, hybrid_level, auto_cos, auto_euc, quant_dtype,
                optimization_level=None, flash_attention=None,
                auto_hybrid=False, compress_weight=False) -> str:
    """Tag suffix that distinguishes builds differing ONLY in the PTQ knobs.

    Both the worker (which writes <tag>.rknn / <tag>.json) and the driver
    (which READS <tag>.json back) must compute the same suffix -- when they
    disagree the driver reports "no_json" for a build that actually
    succeeded.
    """
    suffix = ""
    if quant_method and quant_method != "channel":
        suffix += f"_{quant_method}"
    if hybrid_level:
        suffix += f"_hyb{hybrid_level}"
    if auto_hybrid:
        suffix += "_ah"
    if compress_weight:
        suffix += "_cw"
    if flash_attention:
        suffix += "_fa"
    if auto_cos is not None and abs(float(auto_cos) - 0.98) > 1e-9:
        suffix += f"_cos{auto_cos}"
    if auto_euc is not None:
        suffix += f"_euc{auto_euc}"
    if quant_dtype and quant_dtype != "w8a8":
        suffix += f"_{quant_dtype}"
    if optimization_level is not None and int(optimization_level) != 3:
        suffix += f"_olv{int(optimization_level)}"
    if flash_attention:
        suffix += "_fa"
    return suffix


_RE_INPUT_OP = re.compile(r"InputOperator\s+(\w+)")


def _input_op_dtype(log_path) -> str | None:
    """The dtype of the graph's INPUT operator, read back from the build log.

    This is the single most important property of a deliverable RKNN model:
    an INT8 input is uploaded as-is (inputs_set ~0.34 ms), a FLOAT16 input has
    to be widened by the runtime in user space (inputs_set ~45 ms on the
    small cores / ~14.8 ms on the big cores).  Measured on 2026-10-16
    (memory archive/02 section 28) and re-confirmed on the QAT arms on
    2026-10-23: every path that leaves "toolkit quantises the whole graph"
    (--no-quant, quantized_hybrid_level=1/2/3, QDQ + do_quantization=False)
    turns the input into FLOAT16.

    The operator table is only printed when the toolkit runs verbose, hence
    the --verbose-build switch.
    """
    try:
        text = Path(log_path).read_text(errors="ignore")
    except Exception:
        return None
    m = _RE_INPUT_OP.search(text)
    return m.group(1) if m else None


def worker(args) -> dict:
    import numpy as np

    (variant, set_key, algo, calib_list, onnx_path, frames_json, out_dir,
     topk, quant_dtype, quant_method, hybrid_level, auto_cos, auto_euc,
     verbose, optimization_level, flash_attention, auto_hybrid,
     compress_weight) = args
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{variant}_{set_key}_{algo}"
    # The tag must distinguish builds that differ ONLY in the PTQ knobs,
    # otherwise two configurations overwrite each other's .rknn / .json and
    # the comparison silently becomes a comparison of nothing.
    cfg_suffix = _cfg_suffix(quant_method, hybrid_level, auto_cos, auto_euc, quant_dtype,
                             auto_hybrid=auto_hybrid, compress_weight=compress_weight)
    tag = f"{tag}{cfg_suffix}"
    res = {"variant": variant, "set": set_key, "algo": algo, "tag": tag,
           "quantized_dtype": quant_dtype or "w8a8",
           "quantized_method": quant_method or "channel",
           "quantized_hybrid_level": int(hybrid_level or 0),
           "auto_hybrid_cos_thresh": auto_cos,
           "auto_hybrid_euc_thresh": auto_euc,
           "auto_hybrid": bool(auto_hybrid),
           "compress_weight": bool(compress_weight),
           "optimization_level": int(optimization_level if optimization_level is not None else 3),
           "enable_flash_attention": bool(flash_attention),
           "calib_list": calib_list, "onnx": onnx_path, "status": "running"}
    t0 = time.time()

    from rknn.api import RKNN

    C.selftest()          # the cheap gate: never start a 9-minute build on a broken helper
    frames = json.loads(Path(frames_json).read_text())

    # ---- the graph input dtype decides EVERYTHING downstream -----------------
    # FLOAT input ONNX  -> rknn.config(mean=0, std=255) bakes (x-0)/255 and the
    #                      toolkit CHANGES the input tensor to INT8. The board
    #                      feeds a uint8 buffer, so the runtime then misreads
    #                      128..255 as negative -> the heatmap collapses and
    #                      EVERY frame scores zero detections.
    # UINT8 input ONNX  -> the graph carries Cast->Div(/255) itself, so
    #                      mean_values/std_values MUST be None (passing them
    #                      applies /255 a SECOND time -- also zero detections)
    #                      and the input tensor stays UINT8, which is exactly
    #                      what the board's stb_image path produces.
    import onnx as _onnx
    _m = _onnx.load(onnx_path)
    _inp = _m.graph.input[0]
    u8_graph = (_inp.type.tensor_type.elem_type == _onnx.TensorProto.UINT8)
    res["onnx_input_dtype"] = "uint8" if u8_graph else "float32"
    res["mean_std"] = None if u8_graph else "0/255"

    # ---- float reference: ONNX Runtime on the SAME uint8 buffer the board sees ----
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    float_out = []
    inputs_u8 = []
    for f in frames:
        u8 = C.bmp_to_input_nhwc(f["bmp"])          # (1,640,640,3) uint8 NHWC
        inputs_u8.append(u8)
        nchw_u8 = np.ascontiguousarray(u8.transpose(0, 3, 1, 2))
        feed = nchw_u8 if u8_graph else nchw_u8.astype(np.float32) / 255.0
        float_out.append(sess.run(None, {"images": feed}))
    res["n_frames"] = len(frames)

    # ---- build ----
    rknn = RKNN(verbose=bool(verbose))
    cfg = dict(target_platform="rk3588",
               quantized_method=quant_method,
               optimization_level=int(optimization_level if optimization_level is not None else 3))
    # enable_flash_attention: the per-layer SQNR analysis of 2026-10-16
    # (memory archive/02 section 28.3) put the attention Softmax among the
    # three worst tensors (28.73 dB) -- i.e. exactly the kind of op a
    # different attention kernel would change.  Default is False; this is
    # the only remaining knob that can move the ACTIVATION quantisation
    # noise without turning the graph input into FLOAT16.
    if flash_attention:
        cfg["enable_flash_attention"] = True
    # compress_weight: the toolkit's hardware weight decompression. It changes
    # the weight representation (and therefore the numbers) without changing
    # the graph, so it is a real PTQ knob, not just a build speed knob.
    if compress_weight:
        cfg["compress_weight"] = True
    # ---------------------------------------------------------------------
    # PTQ knobs that were NEVER exercised before 2026-10-23.
    #
    # quantized_hybrid_level (default 0 = hybrid quantisation DISABLED):
    #   RKNN's own hybrid quantisation -- the layers whose quantisation error
    #   is large keep a float16 path instead of being forced to int8.  This is
    #   the toolkit's built-in answer to the mechanism measured on 2026-10-22:
    #   the int8 loss is an ADDITIVE noise floor on the heatmap, and a noise
    #   floor is exactly what a float escape-hatch removes.
    #   NOTE: the toolkit's documented value for AUTOMATIC hybrid is 1
    #   (0 = off).  Verified on 2026-10-23 against the installed
    #   rknn-toolkit2 2.3.2 config() signature.
    # auto_hybrid_cos_thresh (default 0.98): with hybrid enabled, a layer whose
    #   quantised output has a cosine similarity BELOW this threshold to the
    #   float output is automatically kept un-quantised.  RAISING the threshold
    #   makes MORE layers stay float16 (more conservative, slower); lowering it
    #   keeps fewer layers float.
    # quantized_method: 'channel' (per-channel, what we used) | 'layer'
    #   (per-tensor) | 'group{SIZE}' (groups of SIZE channels -- finer
    #   scales than per-channel, so a smaller weight quantisation error).
    #   NOTE: {SIZE} must be a multiple of 32 between 32 and 256 (toolkit
    #   docstring), e.g. group32 / group64.  'group4' is NOT a legal value.
    if hybrid_level:
        cfg["quantized_hybrid_level"] = int(hybrid_level)
    if auto_cos is not None:
        cfg["auto_hybrid_cos_thresh"] = float(auto_cos)
    if auto_euc is not None:
        cfg["auto_hybrid_euc_thresh"] = float(auto_euc)
    # quantized_dtype:
    #   w8a8  -- the plain INT8 model (8-bit weights AND 8-bit activations).
    #   w8a16 -- 8-bit weights but 16-bit ACTIVATIONS.  The toolkit's `gdq`
    #            algorithm is only offered for w4a16/w8a16.  NOTE: measured on
    #            2026-10-22 -- rk3588 does NOT support w8a16 ("The
    #            quantized_dtype = 'w8a16' not support in 'rk3588'!"), so this
    #            path is closed on this platform.
    if quant_dtype:
        cfg["quantized_dtype"] = quant_dtype
    if not u8_graph:
        cfg["mean_values"] = [[0, 0, 0]]
        cfg["std_values"] = [[255, 255, 255]]
    if algo != "none":
        cfg["quantized_algorithm"] = algo
    rknn.config(**cfg)
    if rknn.load_onnx(model=onnx_path) != 0:
        res["status"] = "load_onnx_failed"
        return res
    t_build = time.time()
    # auto_hybrid (toolkit 2.3.2): the per-layer cosine test keeps the sensitive
    # layers on a float16 path. UNLIKE quantized_hybrid_level (whose models have a
    # FLOAT16 INPUT and therefore blow the inputs_set budget), auto_hybrid is a
    # different code path -- the report records input_op_dtype for every arm so
    # the input dtype is never assumed.
    build_kwargs = dict(do_quantization=(algo != "none"),
                        dataset=calib_list if algo != "none" else None)
    if auto_hybrid:
        build_kwargs["auto_hybrid"] = True
    ret = rknn.build(**build_kwargs)
    res["build_seconds"] = round(time.time() - t_build, 1)
    if ret != 0:
        res["status"] = f"build_failed_ret{ret}"
        return res

    rknn_path = out_dir / f"{tag}.rknn"
    if rknn.export_rknn(str(rknn_path)) != 0:
        res["status"] = "export_failed"
        return res
    res["rknn"] = str(rknn_path)
    res["rknn_md5"] = C.md5_file(rknn_path)

    # ---- PC simulator ----
    if rknn.init_runtime(target=None) != 0:
        res["status"] = "init_runtime_failed"
        rknn.release()
        return res

    # A QUANTIZED model bakes (x-0)/255 into the graph, so it takes uint8 NHWC.
    # A NON-quantized (fp16) model keeps the float32 graph, so it takes the same
    # numbers PyTorch sees: uint8/255. Feeding uint8 to the fp16 arm would be a
    # 255x input error that looks like "f16 destroyed the model".
    # MEASURED (probe: 6 frames x 6 feed conventions, cos against ONNX Runtime):
    #   mean/std set + uint8  -> cos 0.999999   <- the board's convention, correct
    #   mean/std set + float32-> cos 0.108      <- 255x too small, garbage
    #   no mean/std  + float32-> cos 0.999999   <- equivalent, but NOT the board's
    # So BOTH arms take uint8 NHWC: the graph applies (x-0)/255 itself. The NHWC
    # vs NCHW choice does not change the result (the toolkit re-lays it out).
    quantized = (algo != "none")
    sim_out = []
    t_inf = time.time()
    for u8 in inputs_u8:
        sim_out.append(rknn.inference(inputs=[u8], data_format="nhwc"))
    res["sim_seconds"] = round(time.time() - t_inf, 1)
    res["sim_ms_per_frame"] = round(res["sim_seconds"] / max(1, len(inputs_u8)) * 1000, 1)

    # ---- per-layer accuracy analysis (best effort; the tool's own report) ----
    layers = []
    try:
        acc = rknn.accuracy_analysis(inputs=[inputs_u8[0]], output_details=True)
        res["accuracy_analysis_available"] = True
        if isinstance(acc, (list, tuple)):
            for row in acc:
                if isinstance(row, dict):
                    layers.append({k: (v if isinstance(v, (int, float, str)) else str(v))
                                   for k, v in row.items()})
    except Exception as exc:                                    # noqa: BLE001
        res["accuracy_analysis_available"] = False
        res["accuracy_analysis_error"] = f"{type(exc).__name__}: {exc}"[:300]
    res["per_layer"] = layers

    # ---- output-level cosine / MSE (computed here, not taken from the tool) ----
    per_frame = []
    cos_hm, cos_off, mse_hm, mse_off = [], [], [], []
    for i, (fo, so) in enumerate(zip(float_out, sim_out)):
        f_hm, f_off = fo[0], fo[1]
        s_hm, s_off = so[0], so[1]
        if s_hm.shape != f_hm.shape or s_off.shape != f_off.shape:
            res["status"] = f"shape_mismatch_frame{i}"
            return res
        c_hm, c_off = cosine(f_hm, s_hm), cosine(f_off, s_off)
        m_hm, m_off = mse(f_hm, s_hm), mse(f_off, s_off)
        cos_hm.append(c_hm); cos_off.append(c_off)
        mse_hm.append(m_hm); mse_off.append(m_off)
        per_frame.append({"frame": frames[i]["bmp"], "frame_id": frames[i]["frame_id"],
                          "seq": frames[i]["seq"],
                          "cos_heatmap": c_hm, "cos_offset": c_off,
                          "mse_heatmap": m_hm, "mse_offset": m_off})
    res["cos_heatmap_mean"] = float(np.mean(cos_hm))
    res["cos_heatmap_min"] = float(np.min(cos_hm))
    res["cos_offset_mean"] = float(np.mean(cos_off))
    res["cos_offset_min"] = float(np.min(cos_off))
    res["mse_heatmap_mean"] = float(np.mean(mse_hm))
    res["mse_offset_mean"] = float(np.mean(mse_off))
    res["per_frame"] = per_frame

    # ---- end-to-end detections: peaks -> points -> scored against the GT ----
    res["end_to_end"] = score_end_to_end(sim_out, frames, topk=topk)

    rknn.release()
    res["status"] = "ok"
    res["total_seconds"] = round(time.time() - t0, 1)
    C.write_json(out_dir / f"{tag}.json", res)
    return res


def score_end_to_end(sim_out, frames, topk: int = 100, th: float = C.TH_BASE):
    """Decode the simulator's outputs exactly like the board does and score them."""
    tp = fp = fn = 0
    n_det_total = 0
    per_seq = {}
    for so, f in zip(sim_out, frames):
        hm = np.asarray(so[0], np.float32).reshape(so[0].shape[-2], so[0].shape[-1])
        off = np.asarray(so[1], np.float32).reshape(2, *hm.shape)
        det = C.extract_peaks_np(hm, off, stride=C.STRIDE, conf_thresh=th, top_k=topk)
        pts = det["points"]
        n_det_total += len(pts)
        # geometry: the source frame is the BMP's own size
        import cv2
        img = cv2.imread(f["bmp"], cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        geom = C.letterbox_geometry((h, w))
        pts_src = C.canvas_to_src(pts, geom) if len(pts) else pts
        # GT is stored in SOURCE-image normalised coordinates
        gpts = []
        gp = Path(f["gt"])
        if gp.is_file():
            for line in gp.read_text().splitlines():
                fl = line.split()
                if len(fl) >= 5:
                    gpts.append([float(fl[1]) * w, float(fl[2]) * h])
        gpts = np.asarray(gpts, np.float32).reshape(-1, 2)
        d = np.concatenate([pts_src, det["scores"][:, None]], 1) if len(pts_src) else np.zeros((0, 3), np.float32)
        a, b, c = C.match_distance(d, gpts)
        tp += a; fp += b; fn += c
        s = per_seq.setdefault(f["seq"], {"tp": 0, "fp": 0, "fn": 0, "det": 0})
        s["tp"] += a; s["fp"] += b; s["fn"] += c; s["det"] += len(pts)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"tp": tp, "fp": fp, "fn": fn, "det": n_det_total,
            "precision": prec, "recall": rec, "f1": f1,
            "per_sequence": per_seq, "th": th}


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def _quant_method_arg(v: str) -> str:
    """Validate RKNN's quantized_method.

    The toolkit docstring says: "currently support: layer, channel, group{SIZE}.
    {SIZE} is the multiple value of 32 between 32 and 256, e.g group32."
    A value outside that set is accepted by rknn.config() and then either
    ignored or fails deep inside the quantiser -- 10 wasted minutes per build.
    """
    if v in ("layer", "channel"):
        return v
    if v.startswith("group"):
        try:
            size = int(v[len("group"):])
        except ValueError:
            raise argparse.ArgumentTypeError(f"bad quantized_method {v!r}")
        if size % 32 == 0 and 32 <= size <= 256:
            return v
        raise argparse.ArgumentTypeError(
            f"quantized_method {v!r}: {{SIZE}} must be a multiple of 32 between "
            f"32 and 256 (e.g. group32 / group64)")
    raise argparse.ArgumentTypeError(
        f"bad quantized_method {v!r} (expected layer / channel / group{{SIZE}})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="p0_off", choices=list(C.VARIANTS))
    ap.add_argument("--onnx-mode", default="u8", choices=["u8", "float"],
                    help="u8 = the uint8-input ONNX (BOARD-COMPATIBLE: the input tensor "
                         "stays UINT8 and the /255 lives inside the graph); "
                         "float = the original float-input ONNX (needs mean/std, and the "
                         "toolkit turns the input into INT8, which the board misreads)")
    ap.add_argument("--sets", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--algos", nargs="+", default=["kl_divergence", "mmse"],
                    help="'none' builds the float/fp16 arm")
    ap.add_argument("--calib-root", default=str(C.WORK_ROOT / "calib"))
    ap.add_argument("--out", default=str(C.WORK_ROOT / "screen"))
    ap.add_argument("--frames-json", default=str(C.WORK_ROOT / "fast" / "tier1_sensitive.json"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--quant-method", default="channel", type=_quant_method_arg,
                    help="RKNN quantized_method: 'channel' (per-channel, the "
                         "default we used), 'layer' (per-tensor) or 'group{SIZE}' "
                         "(finer weight scales than per-channel).  {SIZE} must be "
                         "a multiple of 32 between 32 and 256 (toolkit docstring).")
    ap.add_argument("--quant-hybrid-level", type=int, default=0,
                    help="RKNN quantized_hybrid_level; 0 (the default) DISABLES "
                         "hybrid quantisation, 1 enables the AUTOMATIC hybrid "
                         "(per-layer cosine test against auto_hybrid_cos_thresh, "
                         "the sensitive layers keep a float16 path).  This is the "
                         "toolkit's built-in answer to the additive heatmap noise "
                         "floor that causes the int8 false-positive explosion.")
    ap.add_argument("--auto-hybrid-cos", type=float, default=None,
                    help="RKNN auto_hybrid_cos_thresh; a layer whose quantised "
                         "output has a cosine similarity below this to the float "
                         "output is kept un-quantised.  Lower = more float layers. "
                         "(the toolkit default is 0.98; it only has an effect "
                         "when hybrid quantisation is enabled)")
    ap.add_argument("--auto-hybrid-euc", type=float, default=None,
                    help="RKNN auto_hybrid_euc_thresh (euclidean variant)")
    ap.add_argument("--compress-weight", action="store_true",
                    help="RKNN compress_weight (hardware weight decompression); "
                         "it changes the weight representation, so it is a real "
                         "accuracy knob, not only a build-speed one")
    ap.add_argument("--auto-hybrid", action="store_true",
                    help="enable RKNN's 2.3.2 auto_hybrid (per-layer cosine test); "
                         "the input tensor dtype is reported per arm so the "
                         "inputs_set budget can be checked before going to the board")
    ap.add_argument("--quant-dtype", default="",
                    help="RKNN quantized_dtype: '' (toolkit default w8a8), 'w8a8' or "
                         "'w8a16'.  'w8a16' keeps the weights at 8 bit and lifts the "
                         "ACTIVATIONS to 16 bit; it is the only dtype the 'gdq' "
                         "algorithm is offered for, and it is the direct remedy for "
                         "the spurious-heatmap-peak mechanism behind the int8 FP "
                         "explosion (FP 527 vs 184 for f16 on the same 2000 frames).")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--verbose-build", action="store_true",
                    help="run the toolkit verbose so the operator table is "
                         "printed into the build log; the table carries the "
                         "INPUT operator's dtype (INT8 vs FLOAT16), which is "
                         "the input-set latency gate and cannot be seen any "
                         "other way without spending board time")
    ap.add_argument("--optimization-level", type=int, default=3, choices=[0, 1, 2, 3],
                    help="RKNN optimization_level; 3 is the default we always "
                         "used.  It controls how hard the graph is optimised "
                         "(fusion), and the fusion result decides where the "
                         "quantisation nodes sit -- so it is a real axis for "
                         "the activation quantisation noise, and unlike hybrid "
                         "it does NOT change the dtype of the graph input.")
    ap.add_argument("--enable-flash-attention", action="store_true",
                    help="RKNN enable_flash_attention (default False).  The "
                         "per-layer SQNR analysis of 2026-10-16 ranked the "
                         "attention Softmax among the three worst tensors "
                         "(28.73 dB), i.e. an op whose numerics a different "
                         "attention kernel would change.")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--job-args", help=argparse.SUPPRESS)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.worker:
        job = json.loads(args.job_args)
        res = worker((job["variant"], job["set"], job["algo"], job["calib_list"],
                      job["onnx"], job["frames_json"], job["out_dir"], job["topk"],
                      job.get("quant_dtype", ""), job.get("quant_method", "channel"),
                      job.get("quant_hybrid_level", 0), job.get("auto_hybrid_cos"),
                      job.get("auto_hybrid_euc"), job.get("verbose_build", False),
                      job.get("optimization_level"), job.get("enable_flash_attention", False),
                      job.get("auto_hybrid", False),
                      job.get("compress_weight", False)))
        print(json.dumps({k: v for k, v in res.items() if k != "per_layer" and k != "per_frame"},
                         ensure_ascii=False))
        return

    variant = C.VARIANTS[args.variant]
    _onnx_key = "onnx_u8" if args.onnx_mode == "u8" else "onnx"
    onnx_path = str(C.require(variant[_onnx_key], f"{args.variant} ONNX"))
    # The float16 arm ALWAYS uses the float32 graph: rknn-toolkit2's fp16 build
    # mode rejects a UINT8 graph input with "Not Support Dtype: 2"
    # (2 == TensorProto.UINT8) and aborts before OpFusing completes.
    # The uint8 graph is numerically identical (Cast -> Div(255) lives inside
    # the graph), so the fp16 and int8 arms stay directly comparable.
    onnx_float_path = str(C.require(variant["onnx"], f"{args.variant} float ONNX"))

    # ---- Tier-1 sensitive frames ----
    frames_json = Path(args.frames_json)
    if not frames_json.is_file():
        print(f"[frames] building the Tier-1 sensitive frame list -> {frames_json}")
        frames = C.pick_sensitive_frames(n_dark=args.n_frames // 2,
                                         n_clutter=args.n_frames - args.n_frames // 2)
        C.write_json(frames_json, frames)
    frames = json.loads(frames_json.read_text())
    print(f"[frames] {len(frames)} sensitive frames from {frames_json}")

    # ---- job matrix ----
    jobs = []
    for set_key in args.sets:
        calib = C.require(Path(args.calib_root) / f"set_{set_key}.txt", f"calib set {set_key}")
        n_lines = len([l for l in calib.read_text().splitlines() if l.strip()])
        # 250 is the design target, but the frozen trial-0474 list has 200 and
        # the calibration-range A/B deliberately uses it as-is -> accept 150..400.
        # The range is a sanity bound on a broken list, not a design rule: the
        # CALIBRATION-SIZE experiment needs 100-frame sets to be legal.
        assert 100 <= n_lines <= 400, f"calib set {set_key} has {n_lines} lines, expected 100..400"
        for algo in args.algos:
            job_onnx = onnx_float_path if algo in F16_ALGOS else onnx_path
            jobs.append({"variant": args.variant, "set": set_key, "algo": algo,
                         "calib_list": str(calib), "onnx": job_onnx,
                         "frames_json": str(frames_json), "out_dir": str(out_root),
                         "topk": args.topk, "quant_dtype": args.quant_dtype,
                         "quant_method": args.quant_method,
                         "quant_hybrid_level": args.quant_hybrid_level,
                         "auto_hybrid_cos": args.auto_hybrid_cos,
                         "auto_hybrid_euc": args.auto_hybrid_euc,
                         "auto_hybrid": args.auto_hybrid,
                         "compress_weight": args.compress_weight,
                         "verbose_build": args.verbose_build,
                         "optimization_level": args.optimization_level,
                         "enable_flash_attention": args.enable_flash_attention})

    # mmse needs the whole box to itself-ish
    heavy = [j for j in jobs if j["algo"] in MEMORY_HEAVY]
    light = [j for j in jobs if j["algo"] not in MEMORY_HEAVY]
    light_jobs = max(1, min(args.jobs, 4))
    heavy_jobs = max(1, min(args.jobs, 2))

    print(f"\n[matrix] {len(jobs)} configurations  "
          f"(light pool {light_jobs} way, heavy/mmse pool {heavy_jobs} way)\n")

    results = []
    for group, pool in ((light, light_jobs), (heavy, heavy_jobs)):
        if not group:
            continue
        # chunk by the pool size: mmse alone wants tens of GB, so the whole
        # matrix at once would OOM the box
        for start in range(0, len(group), pool):
            chunk = group[start:start + pool]
            procs = []
            for j in chunk:
                cmd = [sys.executable, str(HERE / "02_batch_ptq_pc_screening.py"), "--worker",
                       "--job-args", json.dumps(j)]
                jtag = (f"{j['variant']}_{j['set']}_{j['algo']}"
                        + _cfg_suffix(j.get("quant_method", "channel"),
                                      j.get("quant_hybrid_level", 0),
                                      j.get("auto_hybrid_cos"),
                                      j.get("auto_hybrid_euc"),
                                      j.get("quant_dtype", ""),
                                      j.get("optimization_level"),
                                      j.get("enable_flash_attention", False),
                                      auto_hybrid=j.get("auto_hybrid", False),
                                      compress_weight=j.get("compress_weight", False)))
                log = (out_root / f"{jtag}.log").open("w")
                procs.append((j, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
                print(f"  launched {jtag}", flush=True)
            for j, p, log in procs:
                p.wait()
                log.close()
            tag = (f"{j['variant']}_{j['set']}_{j['algo']}"
                   + _cfg_suffix(j.get("quant_method", "channel"),
                                 j.get("quant_hybrid_level", 0),
                                 j.get("auto_hybrid_cos"),
                                 j.get("auto_hybrid_euc"),
                                 j.get("quant_dtype", ""),
                                 j.get("optimization_level"),
                                 j.get("enable_flash_attention", False),
                                 auto_hybrid=j.get("auto_hybrid", False),
                                 compress_weight=j.get("compress_weight", False)))
            rj = out_root / f"{tag}.json"
            if rj.is_file():
                r = json.loads(rj.read_text())
                in_dtype = _input_op_dtype(out_root / f"{tag}.log")
                results.append({k: r.get(k) for k in
                                ("variant", "set", "algo", "tag", "status", "cos_heatmap_mean",
                                 "cos_heatmap_min", "cos_offset_mean", "mse_heatmap_mean",
                                 "build_seconds", "sim_ms_per_frame", "rknn_md5",
                                 "quantized_method", "quantized_hybrid_level",
                                 "auto_hybrid_cos_thresh")})
                results[-1]["input_op_dtype"] = in_dtype
                e = r.get("end_to_end") or {}
                results[-1]["f1"] = e.get("f1")
                results[-1]["recall"] = e.get("recall")
                results[-1]["precision"] = e.get("precision")
                results[-1]["fp"] = e.get("fp")
                results[-1]["fn"] = e.get("fn")
                print(f"  done {tag}: status={r.get('status')} "
                      f"cos_hm={r.get('cos_heatmap_mean')} f1={e.get('f1')} "
                      f"build={r.get('build_seconds')}s")
            else:
                results.append({"variant": j["variant"], "set": j["set"], "algo": j["algo"],
                                "status": "no_json"})
                print(f"  FAILED {tag} (see {out_root / (tag + '.log')})")

    C.write_json(out_root / f"screen_{args.variant}.json",
                 {"variant": args.variant, "onnx": onnx_path, "results": results})
    print(f"\n[report] -> {out_root / f'screen_{args.variant}.json'}")

    ok = [r for r in results if r.get("status") == "ok"]
    ok.sort(key=lambda r: (-(r.get("cos_heatmap_mean") or 0)))
    print("\n=== ranking by mean heatmap cosine similarity ===")
    print(f"{'tag':34s} {'status':10s} {'cos_hm':>8s} {'cos_hm_min':>11s} "
          f"{'mse_hm':>10s} {'F1':>7s} {'build_s':>8s}")
    for r in results:
        print(f"{r.get('variant')}_{r.get('set')}_{r.get('algo'):26s} "
              f"{str(r.get('status')):10s} "
              f"{(r.get('cos_heatmap_mean') or 0):8.5f} "
              f"{(r.get('cos_heatmap_min') or 0):11.5f} "
              f"{(r.get('mse_heatmap_mean') or 0):10.3e} "
              f"{(r.get('f1') or 0):7.4f} {str(r.get('build_seconds')):>8s}")
    if ok:
        # With a single configuration (e.g. a one-model arm) ok[1] does not
        # exist -- the old unconditional ok[1] raised IndexError AFTER the
        # model, the .rknn and the .json were already on disk, so the arm
        # looked "failed" even though it had succeeded. Print what exists.
        rank = "  ".join(
            f"{i+1}) {r['variant']}_{r['set']}_{r['algo']}"
            for i, r in enumerate(ok[:2]))
        print(f"\n[Top {min(2, len(ok))}] {rank}")


if __name__ == "__main__":
    main()
