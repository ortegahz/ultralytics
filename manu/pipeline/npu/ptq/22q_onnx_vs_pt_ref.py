#!/usr/bin/env python3
"""22q_onnx_vs_pt_ref.py -- prove an ONNX export == the torch checkpoint it came from.

WHY
---
The trial_0080 torch checkpoint (epoch3.pt) has been DELETED together with the whole
``runs/optuna_mix_ratio`` tree, but the 2026-10-22 PTQ work left behind two independent
artefacts that still pin the model's behaviour down:

  * ``pt_ref/<case>.json``          -- the torch (.pt) float reference curve
  * ``report/ort_float_<case>.json``-- the ONNX float curve of ``trial_0080_hm_nop0.onnx``

Both were produced by the SAME decoder (``ptq_common``), the SAME frames and the SAME GT,
so the numbers are directly comparable.  This script runs ONE ONNX graph over ONE manifest
and prints the per-threshold delta against the torch reference, so a rebuilt cache can be
justified without ever touching the missing .pt again.

Usage (server, conda env `uav`):
    python manu/pipeline/npu/ptq/22q_onnx_vs_pt_ref.py \
        --manifest /mnt/data/siping/ptq_work/fast/tier1_sensitive.json \
        --onnx /mnt/data/siping/ptq_work/qdq_probe/onnx/hm_plain.onnx \
        --pt-ref /mnt/data/siping/ptq_work/qat/trial_0080/pt_ref/tier1_sensitive.json \
        --provider CUDAExecutionProvider --gpu 0 \
        --out /mnt/data/siping/ptq_work/qat/trial_0080/report/onnx_vs_pt_ref_tier1.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402

THS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]


def run_onnx(onnx_path, rows, collect_th, top_k, provider, threads):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                providers=[provider, "CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    per_frame = []
    stats = []
    t0 = time.time()
    for r in rows:
        u8 = C.bmp_to_input_nhwc(r["bmp"])          # (1,640,640,3) uint8 NHWC
        feed = np.ascontiguousarray(u8.transpose(0, 3, 1, 2)).astype(np.float32) / 255.0
        out = sess.run(None, {in_name: feed})
        hm = np.asarray(out[0], np.float32).reshape(out[0].shape[-2], out[0].shape[-1])
        off = np.asarray(out[1], np.float32).reshape(2, *hm.shape)
        stats.append((float(hm.min()), float(hm.max()), float(hm.mean()),
                      float(np.percentile(hm, 99)),
                      int((hm > 0.3).sum()), int((hm > 0.5).sum())))
        det = C.extract_peaks_np(hm, off, stride=C.STRIDE, conf_thresh=collect_th, top_k=top_k)
        pts, scores = det["points"], det["scores"]
        import cv2
        img = cv2.imread(r["bmp"], cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        geom = C.letterbox_geometry((h, w))
        pts_src = C.canvas_to_src(pts, geom) if len(pts) else pts
        gpts = []
        gp = Path(r["gt"])
        if gp.is_file():
            for line in gp.read_text().splitlines():
                fl = line.split()
                if len(fl) >= 5:
                    gpts.append([float(fl[1]) * w, float(fl[2]) * h])
        per_frame.append({"seq": r["seq"], "frame_id": r["frame_id"],
                          "points_src": pts_src, "scores": scores,
                          "gt": np.asarray(gpts, np.float32).reshape(-1, 2),
                          "n_det": len(pts)})
    arr = np.asarray(stats, np.float64)
    return {
        "per_frame": per_frame,
        "seconds": round(time.time() - t0, 1),
        "heatmap": {
            "min": float(arr[:, 0].min()), "max": float(arr[:, 1].max()),
            "mean_of_frame_means": float(arr[:, 2].mean()),
            "max_of_frame_means": float(arr[:, 2].max()),
            "mean_p99": float(arr[:, 3].mean()),
            "mean_px_gt_0.3": float(arr[:, 4].mean()),
            "mean_px_gt_0.5": float(arr[:, 5].mean()),
            "total_px_gt_0.5": float(arr[:, 5].sum()),
        },
        "detections_total_at_collect_th": int(sum(f["n_det"] for f in per_frame)),
    }


def curve_of(per_frame):
    out = {}
    for th in THS:
        tp = fp = fn = 0
        det = 0
        for fr in per_frame:
            keep = fr["scores"] >= th
            pts, scores = fr["points_src"][keep], fr["scores"][keep]
            det += len(pts)
            d = (np.concatenate([pts, scores[:, None]], 1) if len(pts)
                 else np.zeros((0, 3), np.float32))
            a, b, c = C.match_distance(d, fr["gt"])
            tp += a; fp += b; fn += c
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fp + fn)
        out[f"{th:.2f}"] = {"th": th, "tp": tp, "fp": fp, "fn": fn, "det": det,
                            "precision": round(prec, 6), "recall": round(rec, 6),
                            "f1": round(2 * prec * rec / max(1e-9, prec + rec), 6)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--pt-ref", default="")
    ap.add_argument("--provider", default="CUDAExecutionProvider")
    ap.add_argument("--gpu", default="")
    ap.add_argument("--collect-th", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    raw = C.read_json(args.manifest)
    rows = raw if isinstance(raw, list) else raw["items"]
    if args.limit:
        rows = rows[: args.limit]

    print(f"[22q] {len(rows)} frames | onnx={args.onnx} | provider={args.provider}", flush=True)
    r = run_onnx(args.onnx, rows, args.collect_th, args.topk, args.provider, args.threads)
    curve = curve_of(r["per_frame"])
    h = r["heatmap"]
    best = max(curve.values(), key=lambda x: x["f1"])
    print(f"[22q] heatmap max={h['max']:.4f} mean_frame_mean={h['mean_of_frame_means']:.5f} "
          f"px>0.5/frame={h['mean_px_gt_0.5']:.1f} det@0.05={r['detections_total_at_collect_th']}")
    print(f"[22q] best F1={best['f1']:.6f} @ th={best['th']} | "
          f"@0.30 F1={curve['0.30']['f1']:.6f} TP={curve['0.30']['tp']} FP={curve['0.30']['fp']}")
    print(f"[22q] seconds={r['seconds']}")

    res = {"manifest": args.manifest, "onnx": args.onnx, "provider": args.provider,
           "n_frames": len(rows), "seconds": r["seconds"], "heatmap": h,
           "detections_total_at_collect_th": r["detections_total_at_collect_th"],
           "curve": curve}

    if args.pt_ref:
        ref = json.loads(Path(args.pt_ref).read_text())
        rcurve = ref.get("curve", {})
        deltas = {}
        for th in THS:
            key = f"{th:.2f}"
            a, b = curve.get(key), rcurve.get(key)
            if not a or not b:
                continue
            deltas[key] = {"f1_onnx": a["f1"], "f1_pt": b["f1"],
                           "df1": round(a["f1"] - b["f1"], 6),
                           "dtp": a["tp"] - b["tp"], "dfp": a["fp"] - b["fp"],
                           "dfn": a["fn"] - b["fn"]}
        max_df1 = max((abs(v["df1"]) for v in deltas.values()), default=0.0)
        any_count_diff = any(v["dtp"] or v["dfp"] or v["dfn"] for v in deltas.values())
        verdict = ("IDENTICAL" if max_df1 == 0.0 and not any_count_diff else
                   ("EQUIVALENT" if max_df1 <= 1e-6 and not any_count_diff else "DIFFERENT"))
        res["pt_ref"] = args.pt_ref
        res["pt_ref_best_f1"] = ref.get("best_f1")
        res["pt_ref_f1_at_0.30"] = ref.get("f1_at_0.30")
        res["deltas"] = deltas
        res["max_abs_df1"] = max_df1
        res["verdict"] = verdict
        print(f"[22q] vs PT ref ({Path(args.pt_ref).name}): verdict={verdict} "
              f"max|df1|={max_df1:.8f}")
        for key, v in deltas.items():
            flag = "" if (v["df1"] == 0.0 and not (v["dtp"] or v["dfp"] or v["dfn"])) else "  <-- DIFF"
            print(f"[22q]   th={key} onnx_f1={v['f1_onnx']:.6f} pt_f1={v['f1_pt']:.6f} "
                  f"df1={v['df1']:+.6f} dTP={v['dtp']:+d} dFP={v['dfp']:+d} dFN={v['dfn']:+d}{flag}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[22q] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
