#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
22h_pc_sim_eval_qat_arms.py -- PC-simulator evaluation of the QAT arms.

WHAT THIS IS
------------
The rknn-toolkit2 PC simulator is a CPU emulation of the RK3588 NPU: the same
quantiser, the same calibration, the same operator set -- a different numeric
implementation, so its numbers are close to the board's, not equal.  It is the
cheap way to rank arms; the board decides.

WHY IT HAS TO BUILD IN-PROCESS
-------------------------------
A model loaded with load_rknn() CANNOT be simulated ("not support inference on
the simulator") -- the simulator only accepts a model built in the SAME
process.  So build+simulate cannot be split across two scripts, and this
script does both.  The build recipe is the SAME one the board models were
built with, so the .rknn this writes is byte-identical to the one on the board
(the md5 is written to the report so that can be checked).

THE INPUT
---------
The board uploads a uint8 NHWC 640x640 letterbox buffer.  The simulator gets
the SAME buffer (C.bmp_to_input_nhwc) with data_format="nhwc", and the model's
own mean_values/std_values do the /255 -- that is the board's convention, and
it was the convention the toolkit measured as correct (cos 0.999999 against
ONNX Runtime; feeding float32 instead gives cos 0.108).

THE THRESHOLD CURVE
-------------------
The peaks are decoded ONCE with a LOW collection threshold (0.05) and the
surviving peaks are then filtered per scoring threshold.  That is the same
trick the board sweep uses: the 3x3 NMS and the top-k cut do not depend on the
collection threshold, so one pass yields the whole curve.

Usage (server, conda env `rknn`):
    python manu/pipeline/npu/ptq/22h_pc_sim_eval_qat_arms.py \
        --arm a2_qat_learned \
        --manifest /mnt/data/siping/ptq_work/fast/tier1_sensitive.json \
        --out /mnt/data/siping/ptq_work/qat/trial_0080/pc_sim/tier1_sensitive
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402


def _load_screening_helpers():
    """The cosine / mse helpers live in 02_batch_ptq_pc_screening.py, whose file
    name starts with a digit and therefore cannot be imported by name.  Load it
    by path so there is ONE definition of the metric rather than a copy."""
    import importlib.util
    p = Path(__file__).resolve().parent / "02_batch_ptq_pc_screening.py"
    spec = importlib.util.spec_from_file_location("_ptq_screening", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.cosine, mod.mse


cosine, mse = _load_screening_helpers()

#  arm name -> (onnx stem, rknn stem, build mode)
#  build mode:
#    "qdq"          the Q/DQ graph, rknn.build(do_quantization=False) + mean/std
#    "toolkit_int8" the toolkit's OWN full int8 quantiser -- the a2f control:
#                   the QAT weights under exactly the baseline's recipe, so the
#                   only difference from p0_off_A_normal is the weights
ARMS = {
    "a2_qat_learned": ("a2_qat_learned", "a2_qat_learned", "qdq"),
    "a1_qat_maxclip": ("a1_qat_maxclip", "a1_qat_maxclip", "qdq"),
    "a0_qat_float": ("a0_qat_float", "a0_qat_float", "qdq"),
    "a1b_base_maxclip": ("a1b_base_maxclip", "a1b_base_maxclip", "qdq"),
    "a2f_qatw_toolkit_int8": ("a0_qat_float", "a2f_qatw_toolkit_int8", "toolkit_int8"),
}


def load_manifest(path: str) -> list[dict]:
    raw = C.read_json(path)
    rows = raw if isinstance(raw, list) else raw["items"]
    if not rows:
        raise SystemExit(f"[FATAL] manifest {path} is empty")
    return rows


def build_in_process(onnx_path: Path, rknn_path: Path, mode: str = "qdq",
                     calib_list: str = "", quantized_algorithm: str = "normal",
                     verbose: bool = True):
    """The SAME build recipe the board models were built with.

    verbose=True prints the toolkit operator table, which is the ONLY place
    the toolkit says what the model's Input tensor dtype is (INT8 vs FLOAT16
    decides the host-side inputs_set cost -- see 22m_parse_build_log.py)."""
    from rknn.api import RKNN

    rknn = RKNN(verbose=verbose)
    cfg = {
        "target_platform": "rk3588",
        "quantized_method": "channel",
        "optimization_level": 3,
        # the INPUT's /255 -- orthogonal to the Q/DQ activation clips
        "mean_values": [[0, 0, 0]],
        "std_values": [[255, 255, 255]],
    }
    if mode == "toolkit_int8":
        # the BASELINE's own recipe: the toolkit quantises everything itself
        cfg["quantized_algorithm"] = quantized_algorithm
        rknn.config(**cfg)
        if rknn.load_onnx(model=str(onnx_path)) != 0:
            raise SystemExit("[FATAL] load_onnx failed")
        ret = rknn.build(do_quantization=True, dataset=(calib_list or None))
    else:
        rknn.config(**cfg)
        if rknn.load_onnx(model=str(onnx_path)) != 0:
            raise SystemExit("[FATAL] load_onnx failed")
        ret = rknn.build(do_quantization=False, dataset=None)   # HARD CONSTRAINT
    if ret != 0:
        raise SystemExit(f"[FATAL] build failed ret={ret}")
    rknn_path.parent.mkdir(parents=True, exist_ok=True)
    if rknn.export_rknn(str(rknn_path)) != 0:
        raise SystemExit("[FATAL] export_rknn failed")
    if rknn.init_runtime(target=None) != 0:
        raise SystemExit("[FATAL] init_runtime (simulator) failed")
    return rknn, cfg


def score_curve(per_frame: list[dict], thresholds: list[float]) -> dict:
    """Score the collected peaks at every threshold (the peaks are already in
    source-image coordinates, with their scores)."""
    out = {}
    for th in thresholds:
        tp = fp = fn = 0
        det_total = 0
        per_seq: dict[str, dict] = {}
        for fr in per_frame:
            keep = fr["scores"] >= th
            pts = fr["points_src"][keep]
            scores = fr["scores"][keep]
            gpts = fr["gt"]
            det_total += len(pts)
            d = (np.concatenate([pts, scores[:, None]], 1) if len(pts)
                 else np.zeros((0, 3), np.float32))
            a, b, c = C.match_distance(d, gpts)
            tp += a; fp += b; fn += c
            s = per_seq.setdefault(fr["seq"], {"tp": 0, "fp": 0, "fn": 0, "det": 0})
            s["tp"] += a; s["fp"] += b; s["fn"] += c; s["det"] += len(pts)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        out[f"{th:.2f}"] = {"th": th, "tp": tp, "fp": fp, "fn": fn, "det": det_total,
                            "precision": round(prec, 6), "recall": round(rec, 6),
                            "f1": round(f1, 6), "per_sequence": per_seq}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--onnx-dir", default="/mnt/data/siping/ptq_work/qat/trial_0080/onnx")
    ap.add_argument("--rknn-dir", default="/mnt/data/siping/ptq_work/qat/trial_0080/pc_sim_rknn",
                    help="WHERE THE BUILT .rknn IS WRITTEN.  It MUST NOT be the "
                         "directory that holds the models the board runs: the "
                         "simulator refuses a loaded model, so this script has "
                         "to build its own, and the toolkit build is NOT "
                         "bit-reproducible -- overwriting the board's model "
                         "breaks the board driver's md5 check (the driver "
                         "aborts with 'model md5 MISMATCH').")
    ap.add_argument("--thresholds", default="0.30,0.45")
    ap.add_argument("--collect-th", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--calib-list",
                    default="/mnt/data/siping/ptq_work/calib/set_A.txt",
                    help="the toolkit's own calibration list (toolkit_int8 mode)")
    ap.add_argument("--quantized-algorithm", default="normal")
    ap.add_argument("--threads", type=int, default=2,
                    help="CPU threads per process (this script runs 15x in parallel)")
    ap.add_argument("--verbose", type=int, default=1,
                    help="print the toolkit operator table (it carries the "
                         "InputOperator dtype)")
    ap.add_argument("--pt-ref", default="",
                    help="a PT float reference curve JSON "
                         "(22l_pt_float_reference.py) for the SAME frames; "
                         "the quantisation loss is computed against it")
    ap.add_argument("--ort-reference", type=int, default=1,
                    help="also run ONNX Runtime on the same frames and report cos/mse")
    args = ap.parse_args()

    onnx_name, rknn_name, build_mode = ARMS[args.arm]
    onnx_path = Path(args.onnx_dir) / f"{onnx_name}.onnx"
    rknn_path = Path(args.rknn_dir) / f"{rknn_name}.rknn"
    if not onnx_path.is_file():
        raise SystemExit(f"[FATAL] no ONNX at {onnx_path}")

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]

    import torch
    torch.set_num_threads(args.threads)

    t0 = time.time()
    rknn, cfg = build_in_process(onnx_path, rknn_path, mode=build_mode,
                                 calib_list=args.calib_list,
                                 quantized_algorithm=args.quantized_algorithm,
                                 verbose=bool(args.verbose))
    build_s = time.time() - t0
    rknn_md5 = C.md5_file(rknn_path)
    print(f"[sim ] {args.arm}: built in {build_s:.1f}s  rknn_md5={rknn_md5}")

    ort_sess = None
    if args.ort_reference:
        import onnxruntime as ort
        # CAP THE THREAD POOL.  ONNX Runtime grabs every core by default, and
        # this script is run 15x in parallel -- an uncapped session starves
        # every other cell (measured: 15 parallel cells at 50 frames in 6 min
        # instead of 2.6 s/frame).
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = 1
        ort_sess = ort.InferenceSession(str(onnx_path),
                                        sess_options=so,
                                        providers=["CPUExecutionProvider"])

    per_frame = []
    cos_hm, mse_hm = [], []
    t_inf = time.time()
    for i, f in enumerate(rows):
        u8 = C.bmp_to_input_nhwc(f["bmp"])
        out = rknn.inference(inputs=[u8], data_format="nhwc")
        hm = np.asarray(out[0], np.float32).reshape(out[0].shape[-2], out[0].shape[-1])
        off = np.asarray(out[1], np.float32).reshape(2, *hm.shape)

        det = C.extract_peaks_np(hm, off, stride=C.STRIDE,
                                 conf_thresh=args.collect_th, top_k=args.topk)
        pts = det["points"]
        scores = det["scores"]
        import cv2
        img = cv2.imread(f["bmp"], cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        geom = C.letterbox_geometry((h, w))
        pts_src = C.canvas_to_src(pts, geom) if len(pts) else pts

        gpts = []
        gp = Path(f["gt"])
        if gp.is_file():
            for line in gp.read_text().splitlines():
                fl = line.split()
                if len(fl) >= 5:
                    gpts.append([float(fl[1]) * w, float(fl[2]) * h])
        gpts = np.asarray(gpts, np.float32).reshape(-1, 2)

        per_frame.append({"seq": f["seq"], "frame_id": f["frame_id"],
                          "points_src": pts_src, "scores": scores, "gt": gpts})

        if ort_sess is not None:
            feed = np.ascontiguousarray(u8.transpose(0, 3, 1, 2)).astype(np.float32) / 255.0
            fo = ort_sess.run(None, {"images": feed})
            f_hm = np.asarray(fo[0], np.float32).reshape(hm.shape)
            cos_hm.append(float(cosine(f_hm, hm)))
            mse_hm.append(float(mse(f_hm, hm)))
        if (i + 1) % 50 == 0:
            print(f"[sim ] {args.arm}: {i+1}/{len(rows)} frames "
                  f"({(time.time()-t_inf)/(i+1):.2f} s/frame)", flush=True)

    rknn.release()
    curve = score_curve(per_frame, thresholds)
    # ---- the quantisation loss against the PT float reference --------------
    # MUST come before the report dict (the loss is part of the report) and
    # MUST read the curve from pt_ref["curve"] -- 22l writes {"curve": {...}}.
    pt_ref = {}
    if args.pt_ref and Path(args.pt_ref).is_file():
        pt_ref = json.loads(Path(args.pt_ref).read_text())
    ref_curve = pt_ref.get("curve", pt_ref) if pt_ref else {}
    loss = {}
    for th, c in curve.items():
        p = ref_curve.get(f"{float(th):.2f}")
        if p:
            loss[f"{float(th):.2f}"] = round(p["f1"] - c["f1"], 6)

    report = {
        "arm": args.arm,
        "onnx": str(onnx_path), "onnx_md5": C.md5_file(onnx_path),
        "rknn": str(rknn_path), "rknn_md5": rknn_md5,
        "manifest": args.manifest, "n_frames": len(rows),
        "collect_th": args.collect_th, "topk": args.topk,
        "thresholds": thresholds, "curve": curve,
        "build_config": cfg,
        "build_mode": build_mode,
        "do_quantization": (build_mode == "toolkit_int8"),
        "calib_list": (args.calib_list if build_mode == "toolkit_int8" else None),
        "pt_float_reference": args.pt_ref or None,
        "loss_vs_pt_float": loss,
        "build_seconds": round(build_s, 1),
        "sim_seconds": round(time.time() - t_inf, 1),
        "sim_ms_per_frame": round((time.time() - t_inf) / max(1, len(rows)) * 1000, 1),
        "cos_heatmap_mean": float(np.mean(cos_hm)) if cos_hm else None,
        "cos_heatmap_min": float(np.min(cos_hm)) if cos_hm else None,
        "mse_heatmap_mean": float(np.mean(mse_hm)) if mse_hm else None,
        "total_seconds": round(time.time() - t0, 1),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.arm}_pcsim.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in
                      ("arm", "n_frames", "rknn_md5", "cos_heatmap_mean",
                       "mse_heatmap_mean", "sim_ms_per_frame", "total_seconds")},
                     ensure_ascii=False, indent=1))
    for th in thresholds:
        c = curve[f"{th:.2f}"]
        l = loss.get(f"{th:.2f}")
        ls = f" loss={l:+.4f}" if l is not None else ""
        print(f"[sim ] {args.arm} @th={th:.2f}: F1={c['f1']:.4f} P={c['precision']:.4f} "
              f"R={c['recall']:.4f} TP={c['tp']} FP={c['fp']} FN={c['fn']}{ls}")
    print(f"[sim ] -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
