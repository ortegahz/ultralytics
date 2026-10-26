#!/usr/bin/env python3
"""10_probe_calib_peak.py -- why is the int8 heatmap output range so narrow?

THE MEASURED PROBLEM
--------------------
Board probe on the trial-0080 int8 model (p0_off_C_kl_divergence, 500 frames
of 001324, one model, same BMP bytes as the PT arm):

    th=0.30  frames_with_det=0    total_det=0     score_max=-
    th=0.10  frames_with_det=389  total_det=8529  score_max=0.2399
    th=0.05  frames_with_det=431  total_det=10768 score_max=0.2399

    board: out[0] dims=[1,1,320,320] type=INT8 zp=-128 scale=0.00094072

0.2399 == 255 * 0.00094072, i.e. the heatmap output tensor can only represent
[0, 0.2399] -- every real peak is HARD CLIPPED at 0.2399, which is below the
decision threshold 0.3 => zero detections on every frame.

For comparison, the trial-0474 int8 model that actually shipped has
out[0] scale=0.00365272, i.e. a representable range of [0, 0.93].

An RKNN int8 activation scale is calibrated from the activation range observed
on the CALIBRATION set. So the question this script answers is:

    what is the heatmap peak that the model produces ON THE CALIBRATION IMAGES?

If it is ~0.9 then the calibration images are fine and the toolkit is not
seeing what we think it sees.  If it is ~0.12 then the calibration set really
does under-cover the output range, and the fix is the calibration set itself
(which is the whole point of this project).

It also cross-checks the channel order: the toolkit reads images with
stb_image (RGB) while the PC path uses cv2 (BGR) + an explicit reversal, so
the same bytes can be fed in two orders.  If the "no reversal" order gives a
much smaller peak, the calibration files are in the wrong order for the
toolkit -- a second, independent failure mode with the same symptom.

Usage (server, conda env `uav`):
    python manu/pipeline/npu/ptq/10_probe_calib_peak.py --sets A B --frames 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402


def heatmap_max_batch(sess, nhwc_u8: np.ndarray) -> float:
    """Max over the heatmap of one frame, float32 ONNX, uint8/255 input."""
    nchw = np.ascontiguousarray(nhwc_u8.transpose(0, 3, 1, 2)).astype(np.float32) / 255.0
    out = sess.run(None, {"images": nchw})
    hm = np.asarray(out[0], np.float32)
    return float(hm.max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--calib-root", default="/mnt/data/siping/ptq_work/calib")
    ap.add_argument("--onnx", default="/tmp/pycharm_project_10ae9e2e/runs/onnx/trial_0080_hm_nop0.onnx")
    ap.add_argument("--out", default="/mnt/data/siping/ptq_work/probe_calib_peak.json")
    args = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    report = {"onnx": args.onnx, "groups": {}}
    for key in args.sets:
        lst = Path(args.calib_root) / f"set_{key}.txt"
        if not lst.is_file():
            print(f"[skip] {lst} missing")
            continue
        paths = [l.strip() for l in lst.read_text().splitlines() if l.strip()][: args.frames]
        pe_model, pe_raw, missing = [], [], 0
        for p in paths:
            if not Path(p).is_file():
                missing += 1
                continue
            nhwc = C.bmp_to_input_nhwc(p)              # board order: [res, diff, I_t]
            pe_model.append(heatmap_max_batch(sess, nhwc))
            # the order a reader that does NOT reverse would see in the same file
            raw = C.letterbox(C.load_feature_bmp(p), C.IMGSZ)
            pe_raw.append(heatmap_max_batch(sess, raw[None]))
        g = {
            "n": len(pe_model),
            "missing_files": missing,
            "peak_max_correct_order": _stats(pe_model),
            "peak_max_raw_order": _stats(pe_raw) if pe_raw else None,
            "implied_scale": (max(pe_model) / 127.0) if pe_model else None,
            "implied_representable_max": (255.0 * max(pe_model) / 127.0) if pe_model else None,
        }
        report["groups"][key] = g
        print(f"[{key}] n={g['n']} missing={missing}")
        print(f"     peak(correct order) {g['peak_max_correct_order']}")
        print(f"     implied out scale  {g['implied_scale']:.8f}  "
              f"-> representable max {g['implied_representable_max']:.4f}")
        if g["peak_max_raw_order"]:
            print(f"     peak(raw order)     {g['peak_max_raw_order']}")

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"[out] {args.out}")


def _stats(v: list[float]) -> dict:
    if not v:
        return {}
    a = np.asarray(v, np.float64)
    return {"min": float(a.min()), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "max": float(a.max()),
            "mean": float(a.mean())}


if __name__ == "__main__":
    main()
