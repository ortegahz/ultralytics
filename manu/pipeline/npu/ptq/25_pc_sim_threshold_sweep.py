#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
25_pc_sim_threshold_sweep.py -- a BOARD-FREE threshold sweep for arbitrary RKNN arms.

WHY THIS TOOL EXISTS
--------------------
The board (RK3588) is a single shared resource.  Every accuracy question that
used to need the board can be answered on the RKNN PC simulator instead
(rknn.init_runtime(target=None)), which is a CPU emulation of the NPU: same
quantiser, same calibration, same operator set -- a different numeric
implementation, so its numbers are CLOSE TO the board's, not equal.

This script is the generic version of 22h_pc_sim_eval_qat_arms.py:

  * 22h hard-codes the QAT arm names and their directory layout, so it can
    only evaluate the four QAT arms it was written for.
  * this one takes ANY number of .rknn paths, rebuilds each of them in-process
    from its build recipe and sweeps the DETECTION threshold offline.

THE TOOLKIT LIMITATION THAT SHAPES THE DESIGN
---------------------------------------------
A model loaded with load_rknn() CANNOT be simulated:

    init_runtime: RKNN model that loaded by 'load_rknn' not support inference
    on the simulator, please set 'target' first!

(verified on rknn-toolkit2 2.3.2, the version on the server).  The simulator
only accepts a model that was built in the SAME process, so build+simulate
cannot be split across two scripts -- exactly what 22h documents.  Therefore:

    every arm is REBUILT in-process from its recipe, then simulated.

The recipe is the sidecar JSON that 02_batch_ptq_pc_screening.py writes next to
the .rknn (<stem>.json): it records the ONNX, the calibration list and every
quantisation knob.  The rebuilt model is exported to --rknn-out (NEVER into
the directory that holds the models the board runs) and its md5 is compared
with the md5 of the .rknn that was asked for, so the report says whether the
toolkit build is byte-reproducible for that recipe.  The toolkit is NOT
bit-reproducible (two builds of the same recipe gave 75d84fcd... and
cedaeb74...), but the NUMBERS are identical to the last digit, which is what
this tool is for.

THE THRESHOLD CURVE
-------------------
The peaks are decoded ONCE with a LOW collection threshold (0.05) and the
surviving peaks are then filtered per scoring threshold.  The 3x3 NMS and the
top-k cut do not depend on the collection threshold, so one pass yields the
whole curve -- and th=0.25 (which no existing board TSV can reproduce, because
the board TSVs were collected at th=0.30) becomes reachable without the board.

The decode / match口径 is NOT re-implemented here: the peak extractor, the
letterbox geometry and the greedy matcher come from ptq_common (the single
definition shared by 02/03/04/19), the threshold curve comes from 22h, and
--crosscheck re-scores the first N frames with 02's score_end_to_end() and
fails loudly if the two disagree.

Usage (server, conda env `rknn`):

    # one arm, one manifest
    python manu/pipeline/npu/ptq/25_pc_sim_threshold_sweep.py \
        --models /mnt/data/siping/ptq_work/screen_base/p0_off_A_normal.rknn \
        --manifest /mnt/data/siping/ptq_work/fast/001324_sub2000.json \
        --out /mnt/data/siping/ptq_work/sim25/base

    # several manifests in one process (the build is paid once)
    python .../25_pc_sim_threshold_sweep.py --models A.rknn \
        --manifest m1.json m2.json --out OUT

    # the PT float reference arm (onnxruntime, same uint8 buffer)
    python .../25_pc_sim_threshold_sweep.py --pt-reference \
        --manifest m1.json --out OUT

    # merge several sweep reports into one comparison table
    python .../25_pc_sim_threshold_sweep.py --merge-json a.json b.json --out OUT
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

# The repo root is parents[4] (ptq -> npu -> pipeline -> manu -> <root>).
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent

DEFAULT_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                      0.50, 0.55, 0.60, 0.70]


def _load_module(path: Path, name: str):
    """Load a sibling module whose file name starts with a digit."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 22h owns the threshold curve and pulls the cosine/mse helpers out of 02.
# 02 owns score_end_to_end(), the end-to-end scorer of the screening matrix.
_22h = _load_module(HERE / "22h_pc_sim_eval_qat_arms.py", "_ptq_22h")
_02 = _load_module(HERE / "02_batch_ptq_pc_screening.py", "_ptq_02")

score_curve = _22h.score_curve                  # one definition of the curve
cosine, mse = _22h.cosine, _22h.mse             # 02's, via 22h's helper


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------
def load_manifest(path: str) -> list[dict]:
    """Load a frame manifest.

    Two layouts are in use and BOTH are accepted:
      (a) a LIST of rows                    -- fast/tier1_sensitive.json
      (b) a DICT with an "items" list       -- fast/001324_sub2000.json,
                                                fast/golden_pool_sub1200.json
    Anything else is a hard error that names the offending type: a manifest
    that silently decodes to an empty frame list is how an evaluation turns
    into a "no detections anywhere" report.
    """
    raw = C.read_json(path)
    if isinstance(raw, list):
        pass
    elif isinstance(raw, dict):
        if not isinstance(raw.get("items"), list):
            raise SystemExit(
                f"[FATAL] manifest {path}: dict without an 'items' list "
                f"(keys={sorted(raw.keys())}) -- expected a list of rows or a "
                f"dict with an 'items' list")
    else:
        raise SystemExit(
            f"[FATAL] manifest {path}: expected a list of rows or a dict with an "
            f"'items' list, got {type(raw).__name__}")
    rows = _22h.load_manifest(path)          # the ONE definition of the contract
    for i, r in enumerate(rows[:8]):
        missing = [k for k in ("seq", "bmp", "gt") if k not in r]
        if missing:
            raise SystemExit(f"[FATAL] manifest {path}: row {i} misses {missing}")
    return rows


# --------------------------------------------------------------------------------------
# Build recipe -> rknn.config()
# --------------------------------------------------------------------------------------
def onnx_input_dtype(onnx_path: Path) -> str:
    """'uint8' or 'float32' -- the dtype of the graph's INPUT tensor.

    It decides the whole feed convention:
      UINT8 input -> the graph carries Cast->Div(/255) itself, so mean/std must
                      be None and the runtime gets the raw uint8 buffer;
      FLOAT input -> mean_values=[[0,0,0]] / std_values=[[255,255,255]] bake the
                      /255 into the graph, and the toolkit turns the input into
                      INT8 (or FLOAT16 for a non-quantised build).
    """
    import onnx

    m = onnx.load(str(onnx_path))
    t = m.graph.input[0].type.tensor_type.elem_type
    if t == onnx.TensorProto.UINT8:
        return "uint8"
    if t == onnx.TensorProto.FLOAT:
        return "float32"
    return f"elem_type_{t}"


def rknn_config(recipe: dict) -> tuple[dict, bool, str]:
    """Translate a screening recipe into (rknn.config kwargs, u8_graph, algo).

    This mirrors 02_batch_ptq_pc_screening.py's worker EXACTLY -- same knobs,
    same defaults -- because the rebuilt model has to be the model that was
    measured before, otherwise the comparison is against a different arm.
    """
    onnx_path = Path(recipe["onnx"])
    dtype = onnx_input_dtype(onnx_path)
    u8_graph = dtype == "uint8"

    cfg = {
        "target_platform": "rk3588",
        "quantized_method": recipe.get("quantized_method") or "channel",
        "optimization_level": int(recipe.get("optimization_level") or 3),
    }
    if recipe.get("enable_flash_attention"):
        cfg["enable_flash_attention"] = True
    hybrid = int(recipe.get("quantized_hybrid_level") or 0)
    if hybrid:
        cfg["quantized_hybrid_level"] = hybrid
    if recipe.get("auto_hybrid_cos_thresh") is not None:
        cfg["auto_hybrid_cos_thresh"] = float(recipe["auto_hybrid_cos_thresh"])
    if recipe.get("auto_hybrid_euc_thresh") is not None:
        cfg["auto_hybrid_euc_thresh"] = float(recipe["auto_hybrid_euc_thresh"])
    qd = recipe.get("quantized_dtype") or "w8a8"
    if qd != "w8a8":
        cfg["quantized_dtype"] = qd
    if not u8_graph:
        cfg["mean_values"] = [[0, 0, 0]]
        cfg["std_values"] = [[255, 255, 255]]
    algo = recipe.get("algo") or "normal"
    if algo != "none":
        cfg["quantized_algorithm"] = algo
    return cfg, u8_graph, algo


def load_recipe(rknn_path: Path, explicit: str | None) -> dict:
    """The build recipe of an arm: the sidecar JSON 02 writes next to the .rknn."""
    if explicit:
        rec = C.read_json(explicit)
        src = explicit
    else:
        side = rknn_path.with_suffix(".json")
        if not side.is_file():
            raise SystemExit(
                f"[FATAL] no build recipe for {rknn_path}: the simulator cannot "
                f"load a prebuilt .rknn, so the arm has to be rebuilt, and the "
                f"recipe is the sidecar {side} that "
                f"02_batch_ptq_pc_screening.py writes.  Pass --recipe explicitly "
                f"if the recipe lives elsewhere.")
        rec = C.read_json(side)
        src = str(side)
    for k in ("onnx",):
        if not rec.get(k):
            raise SystemExit(f"[FATAL] recipe {src} has no '{k}'")
    if not Path(rec["onnx"]).is_file():
        raise SystemExit(f"[FATAL] recipe {src}: ONNX not found: {rec['onnx']}")
    rec["_recipe_src"] = src
    return rec


# --------------------------------------------------------------------------------------
# Per-frame plumbing (the口径 of 02/22h: ptq_common does the actual work)
# --------------------------------------------------------------------------------------
def load_gt_points(gt_path: str, w: int, h: int) -> np.ndarray:
    """GT centres in SOURCE pixels (the labels are normalised by the source size)."""
    gpts = []
    gp = Path(gt_path)
    if gp.is_file():
        for line in gp.read_text().splitlines():
            fl = line.split()
            if len(fl) >= 5:
                gpts.append([float(fl[1]) * w, float(fl[2]) * h])
    return np.asarray(gpts, np.float32).reshape(-1, 2)


def seq_source_hw(rows: list[dict]) -> dict[str, tuple[int, int]]:
    """The source size of every sequence (read once per sequence, not per frame)."""
    import cv2

    out: dict[str, tuple[int, int]] = {}
    for r in rows:
        seq = r["seq"]
        if seq in out:
            continue
        img = cv2.imread(r["bmp"], cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"[FATAL] cannot read staged bmp: {r['bmp']}")
        out[seq] = (img.shape[0], img.shape[1])
    return out


def decode_frame(hm: np.ndarray, off: np.ndarray, geom: dict, gt: np.ndarray,
                 collect_th: float, topk: int):
    """heat map + offset -> (points in source pixels, scores).

    The peak extractor is ptq_common.extract_peaks_np -- the numpy port of the
    board's C++ extract_peaks (manu/pipeline/npu/rknn_heatmap_smoke.cpp), so the
    board's detections and the simulator's detections are decoded by ONE rule.
    """
    det = C.extract_peaks_np(hm, off, stride=C.STRIDE,
                             conf_thresh=collect_th, top_k=topk)
    pts = det["points"]
    pts_src = C.canvas_to_src(pts, geom) if len(pts) else pts
    return pts_src, det["scores"].astype(np.float32)


# --------------------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------------------
def build_arm(recipe: dict, rknn_out: Path, tag: str):
    """Build the arm in-process and start the simulator on it.

    The built model stays alive so that several manifests can be swept with ONE
    build (the build is ~10 minutes, the sweep is ~1 hour).
    """
    from rknn.api import RKNN

    cfg, u8_graph, algo = rknn_config(recipe)
    onnx_path = Path(recipe["onnx"])
    calib = recipe.get("calib_list")
    do_quant = algo != "none"
    if do_quant and not calib:
        raise SystemExit(f"[FATAL] recipe {recipe.get('_recipe_src')}: a quantised "
                         f"arm needs a calibration list")

    rknn = RKNN(verbose=False)
    rknn.config(**cfg)
    if rknn.load_onnx(model=str(onnx_path)) != 0:
        raise SystemExit("[FATAL] load_onnx failed")
    t0 = time.time()
    ret = rknn.build(do_quantization=do_quant,
                     dataset=calib if do_quant else None)
    build_s = time.time() - t0
    if ret != 0:
        raise SystemExit(f"[FATAL] build failed ret={ret}")
    rknn_out.parent.mkdir(parents=True, exist_ok=True)
    if rknn.export_rknn(str(rknn_out)) != 0:
        raise SystemExit("[FATAL] export_rknn failed")
    if rknn.init_runtime(target=None) != 0:
        raise SystemExit("[FATAL] init_runtime (simulator) failed")
    return {"rknn": rknn, "cfg": cfg, "u8_graph": u8_graph, "algo": algo,
            "build_seconds": round(build_s, 1), "rebuilt": rknn_out}


def simulate_arm(arm: dict, tag: str, rows: list[dict], src_hw: dict,
                 collect_th: float, topk: int, ort_sess, ort_u8: bool,
                 crosscheck_n: int, pt_stride: int = 1):
    """Simulate one arm over the manifest; return the per-frame peaks + stats.

    When an ONNX Runtime session is passed in, the PT float arm's peaks are
    collected IN THE SAME LOOP: one ORT forward per frame, decoded by the same
    rule, on the same uint8 buffer -- which is what "same frame, same decoder,
    same GT" means for the quantisation loss.

    pt_stride > 1 runs the ORT forward only on every pt_stride-th frame: the
    ORT forward is the expensive part of the loop and its result (the PT float
    curve) is IDENTICAL for every arm, so running it once per manifest (in a
    --pt-only job) and sampling it here is the cheap way to keep the heat-map
    cos/mse without paying for it on every frame.
    """
    rknn = arm["rknn"]
    per_frame: list[dict] = []
    pt_per_frame: list[dict] = []
    raw_for_crosscheck: list = []
    cross_rows: list[dict] = []
    cos_hm, mse_hm, cos_off, mse_off = [], [], [], []

    t_inf = time.time()
    for i, f in enumerate(rows):
        u8 = C.bmp_to_input_nhwc(f["bmp"])            # (1,640,640,3) uint8 NHWC
        out = rknn.inference(inputs=[u8], data_format="nhwc")
        hm = np.asarray(out[0], np.float32).reshape(out[0].shape[-2], out[0].shape[-1])
        off = np.asarray(out[1], np.float32).reshape(2, *hm.shape)

        h, w = src_hw[f["seq"]]
        geom = C.letterbox_geometry((h, w))
        gpts = load_gt_points(f["gt"], w, h)
        pts_src, scores = decode_frame(hm, off, geom, gpts, collect_th, topk)
        per_frame.append({"seq": f["seq"], "frame_id": f["frame_id"],
                          "points_src": pts_src, "scores": scores, "gt": gpts})

        if crosscheck_n and i < crosscheck_n:
            raw_for_crosscheck.append(out)
            cross_rows.append(f)

        if ort_sess is not None and (i % pt_stride == 0):
            feed = np.ascontiguousarray(u8.transpose(0, 3, 1, 2))
            if not ort_u8:
                feed = feed.astype(np.float32) / 255.0
            fo = ort_sess.run(None, {ort_sess.get_inputs()[0].name: feed})
            f_hm = np.asarray(fo[0], np.float32).reshape(hm.shape)
            f_off = np.asarray(fo[1], np.float32).reshape(off.shape)
            cos_hm.append(cosine(f_hm, hm))
            mse_hm.append(mse(f_hm, hm))
            cos_off.append(cosine(f_off, off))
            mse_off.append(mse(f_off, off))
            f_pts, f_scores = decode_frame(f_hm, f_off, geom, gpts, collect_th, topk)
            pt_per_frame.append({"seq": f["seq"], "frame_id": f["frame_id"],
                                 "points_src": f_pts, "scores": f_scores,
                                 "gt": gpts})

        if (i + 1) % 50 == 0:
            el = time.time() - t_inf
            print(f"[sim ] {tag}: {i+1}/{len(rows)} frames "
                  f"({el/(i+1):.2f} s/frame)", flush=True)

    # NOTE: the model is NOT released here -- one build serves every manifest.
    sim_s = time.time() - t_inf
    return {"per_frame": per_frame, "pt_per_frame": pt_per_frame,
            "cfg": arm["cfg"], "u8_graph": arm["u8_graph"],
            "build_seconds": arm["build_seconds"], "sim_seconds": round(sim_s, 1),
            "sim_ms_per_frame": round(sim_s / max(1, len(rows)) * 1000, 1),
            "cos_heatmap_mean": float(np.mean(cos_hm)) if cos_hm else None,
            "cos_heatmap_min": float(np.min(cos_hm)) if cos_hm else None,
            "cos_frames": len(cos_hm),
            "mse_heatmap_mean": float(np.mean(mse_hm)) if mse_hm else None,
            "cos_offset_mean": float(np.mean(cos_off)) if cos_off else None,
            "mse_offset_mean": float(np.mean(mse_off)) if mse_off else None,
            "_raw": raw_for_crosscheck, "_rows": cross_rows}


def run_pt_arm(onnx_path: Path, rows: list[dict], src_hw: dict, collect_th: float,
               topk: int, ort_threads: int = 0):
    """The PT float reference arm: onnxruntime on the SAME uint8 buffer."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    if ort_threads > 0:
        so.intra_op_num_threads = ort_threads
        so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), so,
                                providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    dtype = onnx_input_dtype(onnx_path)
    u8_graph = dtype == "uint8"
    per_frame: list[dict] = []
    t0 = time.time()
    for i, f in enumerate(rows):
        u8 = C.bmp_to_input_nhwc(f["bmp"])
        feed = np.ascontiguousarray(u8.transpose(0, 3, 1, 2))
        if not u8_graph:
            feed = feed.astype(np.float32) / 255.0
        fo = sess.run(None, {name: feed})
        hm = np.asarray(fo[0], np.float32).reshape(fo[0].shape[-2], fo[0].shape[-1])
        off = np.asarray(fo[1], np.float32).reshape(2, *hm.shape)
        h, w = src_hw[f["seq"]]
        geom = C.letterbox_geometry((h, w))
        gpts = load_gt_points(f["gt"], w, h)
        pts_src, scores = decode_frame(hm, off, geom, gpts, collect_th, topk)
        per_frame.append({"seq": f["seq"], "frame_id": f["frame_id"],
                          "points_src": pts_src, "scores": scores, "gt": gpts})
        if (i + 1) % 200 == 0:
            print(f"[pt  ] {i+1}/{len(rows)} frames "
                  f"({(time.time()-t0)/(i+1):.2f} s/frame)", flush=True)
    return {"per_frame": per_frame, "onnx": str(onnx_path),
            "onnx_input_dtype": dtype,
            "onnx_md5": C.md5_file(onnx_path),
            "seconds": round(time.time() - t0, 1)}


def crosscheck_with_02(raw: list, rows: list[dict], topk: int, th: float) -> dict:
    """Re-score the first N frames with 02's score_end_to_end() and compare.

    02 is the screening matrix's scorer; if this script's decode / geometry /
    GT / match pipeline ever drifts away from it, the numbers would silently
    stop being comparable with the ones already on disk.  This check is what
    makes that impossible.
    """
    if not raw:
        return {"checked": 0}
    res = _02.score_end_to_end(raw, rows, topk=topk, th=th)
    return {"checked": len(rows), "th": th, "tp": res["tp"], "fp": res["fp"],
            "fn": res["fn"], "det": res["det"], "precision": res["precision"],
            "recall": res["recall"], "f1": res["f1"]}


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def arm_entry(tag: str, curve: dict, extra: dict) -> dict:
    """One arm in the report: the curve plus the provenance of the numbers."""
    curve_rows = []
    for th, c in curve.items():
        curve_rows.append({
            "th": c["th"], "f1": c["f1"], "precision": c["precision"],
            "recall": c["recall"], "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
            "det": c["det"],
            "per_sequence": c.get("per_sequence", {}),
        })
    curve_rows.sort(key=lambda r: r["th"])
    best = max(curve_rows, key=lambda r: r["f1"]) if curve_rows else None
    return {"model": extra.get("model"), "model_md5": extra.get("model_md5"),
            "rebuilt_rknn": extra.get("rebuilt_rknn"),
            "rebuilt_rknn_md5": extra.get("rebuilt_rknn_md5"),
            "md5_match": extra.get("md5_match"),
            "build_config": extra.get("build_config"),
            "build_seconds": extra.get("build_seconds"),
            "sim_seconds": extra.get("sim_seconds"),
            "sim_ms_per_frame": extra.get("sim_ms_per_frame"),
            "cos_heatmap_mean": extra.get("cos_heatmap_mean"),
            "cos_heatmap_min": extra.get("cos_heatmap_min"),
            "cos_frames": extra.get("cos_frames"),
            "mse_heatmap_mean": extra.get("mse_heatmap_mean"),
            "cos_offset_mean": extra.get("cos_offset_mean"),
            "mse_offset_mean": extra.get("mse_offset_mean"),
            "best": best, "curve": curve_rows}


def write_report(report: dict, out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}_pcsim_threshold_sweep.json"
    C.write_json(json_path, report)

    arms = report.get("arms", {})
    thresholds = report.get("thresholds", [])
    lines = []
    lines.append(f"# PC-simulator threshold sweep -- {report.get('manifest_name')}")
    lines.append("")
    lines.append(f"- manifest: {report.get('manifest')}")
    lines.append(f"- frames: {report.get('frames')}  collect_th: {report.get('collect_th')}  "
                 f"topk: {report.get('topk')}")
    lines.append(f"- thresholds: {','.join(f'{t:.2f}' for t in thresholds)}")
    lines.append("")
    lines.append("## F1 per arm per threshold")
    lines.append("")
    lines.append("| arm | " + " | ".join(f"{t:.2f}" for t in thresholds) + " |")
    lines.append("|---|" + "---|" * len(thresholds))
    for name, arm in arms.items():
        by_th = {c["th"]: c for c in arm["curve"]}
        cells = []
        for t in thresholds:
            c = by_th.get(t)
            cells.append(f"{c['f1']:.4f}" if c else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## best threshold per arm")
    lines.append("")
    lines.append("| arm | best th | F1 | P | R | TP | FP | FN | cos_hm | mse_hm |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, arm in arms.items():
        b = arm.get("best")
        if not b:
            continue
        cos = arm.get("cos_heatmap_mean")
        mse_h = arm.get("mse_heatmap_mean")
        lines.append(
            f"| {name} | {b['th']:.2f} | {b['f1']:.4f} | {b['precision']:.4f} | "
            f"{b['recall']:.4f} | {b['tp']} | {b['fp']} | {b['fn']} | "
            f"{cos if cos is None else f'{cos:.5f}'} | "
            f"{mse_h if mse_h is None else f'{mse_h:.3e}'} |")
    lines.append("")
    lines.append("## full curve")
    lines.append("")
    for name, arm in arms.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| th | F1 | P | R | TP | FP | FN | det |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in arm["curve"]:
            lines.append(f"| {c['th']:.2f} | {c['f1']:.4f} | {c['precision']:.4f} | "
                         f"{c['recall']:.4f} | {c['tp']} | {c['fp']} | {c['fn']} | {c['det']} |")
        lines.append("")
    md_path = out_dir / f"{stem}_pcsim_threshold_sweep.md"
    md_path.write_text("\n".join(lines))
    return json_path, md_path


def merge_reports(paths: list[str], out_dir: Path) -> Path:
    """Merge several sweep reports into one comparison table (per manifest)."""
    reports = [(Path(p), C.read_json(p)) for p in paths]
    by_manifest: dict[str, list[tuple[Path, dict]]] = {}
    for p, r in reports:
        by_manifest.setdefault(r.get("manifest_name") or p.stem, []).append((p, r))

    out_dir.mkdir(parents=True, exist_ok=True)
    md = ["# PC-simulator threshold sweep -- merged comparison", ""]
    summary = {}
    for mname, items in by_manifest.items():
        base = items[0][1]
        thresholds = base.get("thresholds", [])
        md += [f"## {mname}", "",
               f"- manifest: {base.get('manifest')}",
               f"- frames: {base.get('frames')}  collect_th: {base.get('collect_th')}  "
               f"topk: {base.get('topk')}", ""]
        for th_label in thresholds:
            md += [f"### th = {th_label:.2f}", "",
                   "| arm | F1 | P | R | TP | FP | FN | det |",
                   "|---|---|---|---|---|---|---|---|"]
            rows = []
            for p, r in items:
                for name, arm in r.get("arms", {}).items():
                    c = next((c for c in arm["curve"] if abs(c["th"] - th_label) < 1e-9), None)
                    if not c:
                        continue
                    rows.append((name, c))
                    md.append(f"| {name} | {c['f1']:.4f} | {c['precision']:.4f} | "
                              f"{c['recall']:.4f} | {c['tp']} | {c['fp']} | "
                              f"{c['fn']} | {c['det']} |")
            # quantisation loss against the PT float arm, same frame / decoder / GT / th
            pt = next((c for n, c in rows if n == "pt_float"), None)
            if pt is not None:
                md += ["", "| arm | dF1 vs PT | dRecall | dFP | dTP |", "|---|---|---|---|---|"]
                for name, c in rows:
                    if name == "pt_float":
                        continue
                    md.append(f"| {name} | {c['f1']-pt['f1']:+.4f} | "
                              f"{c['recall']-pt['recall']:+.4f} | "
                              f"{c['fp']-pt['fp']:+d} | {c['tp']-pt['tp']:+d} |")
            md.append("")
        # best threshold + the loss there
        md += ["### best threshold per arm", "",
               "| arm | best th | F1 | P | R | TP | FP | FN | dF1 vs PT@best |",
               "|---|---|---|---|---|---|---|---|---|"]
        bests = {}
        for p, r in items:
            for name, arm in r.get("arms", {}).items():
                if arm.get("best"):
                    bests[name] = arm["best"]
        pt_best = bests.get("pt_float")
        for name, b in bests.items():
            d = f"{b['f1']-pt_best['f1']:+.4f}" if pt_best else "-"
            md.append(f"| {name} | {b['th']:.2f} | {b['f1']:.4f} | {b['precision']:.4f} | "
                      f"{b['recall']:.4f} | {b['tp']} | {b['fp']} | {b['fn']} | {d} |")
        md.append("")
        summary[mname] = {
            "manifest": base.get("manifest"),
            "frames": base.get("frames"),
            "arms": {n: {"best": b,
                         "at_0.30": next((c for p, r in items
                                          for nm, arm in r.get("arms", {}).items() if nm == n
                                          for c in arm["curve"]
                                          if abs(c["th"] - 0.30) < 1e-9), None)}
                     for n, b in bests.items()},
        }
    md_path = out_dir / "merged_pcsim_threshold_sweep.md"
    md_path.write_text("\n".join(md))
    C.write_json(out_dir / "merged_pcsim_threshold_sweep.json", summary)
    print(f"[merge] -> {md_path}")
    return md_path


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[],
                    help="RKNN models to sweep; the tag is the file stem and the "
                         "build recipe is the sidecar <stem>.json next to it")
    ap.add_argument("--recipe", default="",
                    help="explicit recipe JSON (overrides the sidecar lookup)")
    ap.add_argument("--manifest", nargs="*", default=[],
                    help="frame manifest(s): a list of rows, or a dict with 'items'")
    ap.add_argument("--thresholds", default="",
                    help="comma separated scoring thresholds")
    ap.add_argument("--collect-th", type=float, default=0.05,
                    help="the LOW collection threshold; the curve is filtered offline")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="only the first N frames")
    ap.add_argument("--pt-reference", action="store_true",
                    help="add the PT float arm (onnxruntime on the same uint8 buffer)")
    ap.add_argument("--pt-only", action="store_true",
                    help="ONLY the PT float arm (no simulator arm)")
    ap.add_argument("--pt-onnx", default="/tmp/pycharm_project_10ae9e2e/runs/onnx/trial_0080_hm_nop0.onnx",
                    help="the float ONNX of the PT reference arm")
    ap.add_argument("--ort-threads", type=int, default=0,
                    help="ONNX Runtime intra-op threads (0 = the toolkit default, "
                         "which is every core -- that starves the simulator)")
    ap.add_argument("--pt-ref-frames", type=int, default=0,
                    help="run the ORT forward only on every N-th frame (N = "
                         "frames/this value), i.e. a uniform sample of the "
                         "manifest, to keep the heat-map cos/mse cheap.  0 = "
                         "every frame.  The PT float CURVE is unaffected (it is "
                         "produced by a --pt-only job on all frames).")
    ap.add_argument("--no-pt-arm", action="store_true",
                    help="do not emit the pt_float arm from this job (use it "
                         "when the ORT forward was sampled, or when another "
                         "job already produced the PT float arm)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rknn-out", default="",
                    help="where the REBUILT .rknn is written (default <out>/rknn). "
                         "It must never be the directory the board runs from.")
    ap.add_argument("--crosscheck", type=int, default=100,
                    help="re-score the first N frames with 02's score_end_to_end "
                         "and fail if the two disagree (0 = off)")
    ap.add_argument("--merge-json", nargs="*", default=[],
                    help="merge existing sweep reports into one comparison table")
    args = ap.parse_args()

    out_dir = Path(args.out)
    rknn_out_dir = Path(args.rknn_out) if args.rknn_out else out_dir / "rknn"

    if args.merge_json:
        merge_reports(args.merge_json, out_dir)
        return 0

    if not args.manifest:
        raise SystemExit("[FATAL] --manifest is required")
    if not args.models and not args.pt_reference and not args.pt_only:
        raise SystemExit("[FATAL] nothing to do: pass --models and/or --pt-reference")

    if args.thresholds:
        thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    else:
        thresholds = list(DEFAULT_THRESHOLDS)

    ort_sess = None
    ort_u8 = False
    if args.pt_reference or args.pt_only:
        import onnxruntime as ort

        pt_onnx = Path(args.pt_onnx)
        if not pt_onnx.is_file():
            raise SystemExit(f"[FATAL] no PT ONNX at {pt_onnx}")
        so = ort.SessionOptions()
        if args.ort_threads > 0:
            # ONNX Runtime grabs every core by default, which would starve the
            # simulator processes that run next to it.
            so.intra_op_num_threads = args.ort_threads
            so.inter_op_num_threads = 1
        ort_sess = ort.InferenceSession(str(pt_onnx), so,
                                        providers=["CPUExecutionProvider"])
        ort_u8 = onnx_input_dtype(pt_onnx) == "uint8"

    # arms: (tag, model path, recipe)
    arms = []
    seen = set()
    for mp in args.models:
        mpath = Path(mp)
        C.require(mpath, "RKNN model")
        tag = mpath.stem
        if tag in seen:
            tag = f"{mpath.parent.name}/{tag}"
        seen.add(tag)
        arms.append((tag, mpath, load_recipe(mpath, args.recipe or None)))

    # ---- build every arm ONCE; the simulator needs the model in this process ----
    built = []
    for tag, mpath, recipe in arms:
        print(f"\n[arm] {tag}: rebuilding from {recipe['_recipe_src']}", flush=True)
        rebuilt = rknn_out_dir / f"{tag}.rknn"
        arm = build_arm(recipe, rebuilt, tag)
        arm.update({"tag": tag, "model": mpath, "rebuilt": rebuilt,
                    "model_md5": C.md5_file(mpath),
                    "rebuilt_md5": C.md5_file(rebuilt)})
        built.append(arm)

    for manifest in args.manifest:
        rows = load_manifest(manifest)
        if args.limit:
            rows = rows[: args.limit]
        src_hw = seq_source_hw(rows)
        mname = Path(manifest).stem
        print(f"\n[manifest] {manifest}: {len(rows)} frames "
              f"({len(src_hw)} sequences, sizes {sorted({v for v in src_hw.values()})})",
              flush=True)

        report = {
            "tool": "25_pc_sim_threshold_sweep.py",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "manifest": manifest, "manifest_name": mname, "frames": len(rows),
            "collect_th": args.collect_th, "topk": args.topk,
            "thresholds": thresholds, "arms": {},
        }
        pt_frames: dict = {"dtype": "uint8" if ort_u8 else "float32"}

        for arm in built:
            tag = arm["tag"]
            pt_stride = (max(1, len(rows) // args.pt_ref_frames)
                         if args.pt_ref_frames else 1)
            r = simulate_arm(arm, tag, rows, src_hw,
                             args.collect_th, args.topk, ort_sess, ort_u8,
                             args.crosscheck, pt_stride)
            curve = score_curve(r["per_frame"], thresholds)
            if r["pt_per_frame"]:
                pt_frames["per_frame"] = r["pt_per_frame"]
            entry = arm_entry(tag, curve, {
                "model": str(arm["model"]), "model_md5": arm["model_md5"],
                "rebuilt_rknn": str(arm["rebuilt"]),
                "rebuilt_rknn_md5": arm["rebuilt_md5"],
                "md5_match": arm["model_md5"] == arm["rebuilt_md5"],
                "build_config": r["cfg"], "build_seconds": r["build_seconds"],
                "sim_seconds": r["sim_seconds"],
                "sim_ms_per_frame": r["sim_ms_per_frame"],
                "cos_heatmap_mean": r["cos_heatmap_mean"],
                "cos_heatmap_min": r["cos_heatmap_min"],
                "cos_frames": r["cos_frames"],
                "mse_heatmap_mean": r["mse_heatmap_mean"],
                "cos_offset_mean": r["cos_offset_mean"],
                "mse_offset_mean": r["mse_offset_mean"],
            })
            report["arms"][tag] = entry
            # The report is written BEFORE the cross-check: the cross-check is a
            # guard, and a guard must never be able to throw away hours of
            # simulation.
            jp, mp_ = write_report(report, out_dir, mname)
            print(f"[report] -> {jp}", flush=True)
            if args.crosscheck and r["_raw"]:
                cc = crosscheck_with_02(r["_raw"], r["_rows"], args.topk,
                                        args.collect_th)
                # the SAME frame subset has to be scored on both sides: 02 scores
                # only the cross-check frames, so the curve has to be restricted
                # to them as well (a whole-manifest curve would not match).
                mine = score_curve(r["per_frame"][:cc["checked"]],
                                   [args.collect_th])[f"{args.collect_th:.2f}"]
                cc["mine"] = {"tp": mine["tp"], "fp": mine["fp"], "fn": mine["fn"],
                              "det": mine["det"], "f1": mine["f1"]}
                cc["match"] = (cc["tp"] == mine["tp"] and cc["fp"] == mine["fp"]
                               and cc["fn"] == mine["fn"] and cc["det"] == mine["det"])
                entry["crosscheck_vs_02"] = cc
                write_report(report, out_dir, mname)
                if not cc["match"]:
                    raise SystemExit(
                        f"[FATAL] {tag}: the curve disagrees with 02's "
                        f"score_end_to_end on the first {cc['checked']} frames: "
                        f"{cc} vs {cc['mine']} -- the decode口径 has drifted")
                print(f"[check] {tag}: identical to 02's score_end_to_end on "
                      f"{cc['checked']} frames @th={cc['th']:.2f}", flush=True)

        if ((args.pt_reference or args.pt_only) and not args.no_pt_arm
                and "pt_float" not in report["arms"]):
            if pt_frames.get("per_frame") and len(pt_frames["per_frame"]) == len(rows):
                # the ORT forward was already done inside the arm loop: the PT
                # float arm must see the SAME frames and the SAME decoder.
                curve = score_curve(pt_frames["per_frame"], thresholds)
                pt_meta = {"model": str(args.pt_onnx),
                           "model_md5": C.md5_file(args.pt_onnx),
                           "build_config": {"note": "onnxruntime float reference, "
                                                     "input dtype " + pt_frames["dtype"]},
                           "sim_seconds": pt_frames.get("seconds")}
            else:
                pt = run_pt_arm(Path(args.pt_onnx), rows, src_hw, args.collect_th,
                                args.topk, args.ort_threads)
                curve = score_curve(pt["per_frame"], thresholds)
                pt_meta = {"model": pt["onnx"], "model_md5": pt["onnx_md5"],
                           "build_config": {"note": "onnxruntime float reference, "
                                                     "input dtype " + pt["onnx_input_dtype"]},
                           "sim_seconds": pt["seconds"]}
            report["arms"]["pt_float"] = arm_entry("pt_float", curve, pt_meta)
            jp, mp_ = write_report(report, out_dir, mname)
            print(f"[report] -> {jp}", flush=True)

        for name, arm in report["arms"].items():
            b = arm.get("best")
            if not b:
                continue
            print(f"[arm  ] {name}: best F1={b['f1']:.4f} @ th={b['th']:.2f} "
                  f"TP={b['tp']} FP={b['fp']} FN={b['fn']}")

    for arm in built:
        arm["rknn"].release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
