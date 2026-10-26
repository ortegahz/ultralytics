#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
22p_ort_float_diagnosis.py -- FLOAT-side diagnosis of an ONNX pair.

WHY
---
The a2f arm (the QAT weights under the toolkit's OWN full int8 recipe)
explodes on the board: F1 0.2574 vs the int8 baseline's 0.6448 on
001324/2000, with 116,053 raw detections instead of ~20,000 -- 58 detections
per frame.  That magnitude smells like a bug, not like a precision loss, so
the FIRST cut has to separate the two possible causes:

    float explodes  -> the WEIGHTS (or the export) are wrong
    float is fine, only int8 explodes
                     -> the QUANTISATION RANGE is wrong (a per-tensor min/max
                        calibration blown up by outliers)

This script runs the two ONNX graphs through ONNX Runtime on the SAME frames,
in FLOAT, and reports for each:
    * the heatmap distribution (min / max / mean / p99 / how many pixels above
      0.3 and 0.5)
    * the decoded detections and the F1 curve over the thresholds
    * the per-frame detection counts (the frames that blow up)
so the two causes can be told apart with numbers.

Usage (server, conda env `uav`):
    python manu/pipeline/npu/ptq/22p_ort_float_diagnosis.py \
        --manifest /mnt/data/siping/ptq_work/fast/tier1_sensitive.json \
        --onnx-base runs/onnx/trial_0080_hm_nop0.onnx \
        --onnx-test /mnt/data/siping/ptq_work/qat/trial_0080/onnx/a0_qat_float.onnx \
        --label-base base_weights --label-test qat_weights \
        --out /mnt/data/siping/ptq_work/qat/trial_0080/report/ort_float_tier1.json
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

THS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]


def op_histogram(onnx_path: Path) -> dict:
    import onnx
    m = onnx.load(str(onnx_path))
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    inp = m.graph.input[0]
    return {
        "ops": dict(sorted(ops.items(), key=lambda kv: -kv[1])),
        "n_nodes": len(m.graph.node),
        "input_elem_type": int(inp.type.tensor_type.elem_type),
        "input_name": inp.name,
        "outputs": [o.name for o in m.graph.output],
        "output_elem_type": [int(o.type.tensor_type.elem_type) for o in m.graph.output],
    }


def run_onnx(onnx_path: Path, rows: list[dict], src_hw: dict, collect_th: float,
             topk: int, threads: int = 4) -> dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                providers=["CPUExecutionProvider"])
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
        det = C.extract_peaks_np(hm, off, stride=C.STRIDE,
                                 conf_thresh=collect_th, top_k=topk)
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
                          "bmp": r["bmp"], "points_src": pts_src, "scores": scores,
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


def curve_of(per_frame: list[dict]) -> dict:
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
    ap.add_argument("--onnx-base", required=True)
    ap.add_argument("--onnx-test", required=True)
    ap.add_argument("--label-base", default="base_weights")
    ap.add_argument("--label-test", default="qat_weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--collect-th", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    raw = C.read_json(args.manifest)
    rows = raw if isinstance(raw, list) else raw["items"]
    if args.limit:
        rows = rows[: args.limit]

    import cv2
    src_hw = {}
    for r in rows:
        if r["seq"] not in src_hw:
            img = cv2.imread(r["bmp"], cv2.IMREAD_COLOR)
            src_hw[r["seq"]] = img.shape[:2]

    pb, pt = Path(args.onnx_base), Path(args.onnx_test)
    print(f"[ort ] {len(rows)} frames")
    print(f"[ort ] base={pb}")
    print(f"[ort ] test={pt}")

    res = {"manifest": args.manifest, "n_frames": len(rows),
           "collect_th": args.collect_th, "topk": args.topk,
           "onnx_base": str(pb), "onnx_test": str(pt),
           "graph_base": op_histogram(pb), "graph_test": op_histogram(pt)}

    for tag, path, label in (("base", pb, args.label_base), ("test", pt, args.label_test)):
        print(f"[ort ] running {label} ...", flush=True)
        r = run_onnx(path, rows, src_hw, args.collect_th, args.topk,
                     threads=args.threads)
        c = curve_of(r["per_frame"])
        res[tag] = {"label": label, "onnx": str(path),
                    "heatmap": r["heatmap"], "seconds": r["seconds"],
                    "detections_total_at_collect_th": r["detections_total_at_collect_th"],
                    "curve": c}
        worst = sorted(r["per_frame"], key=lambda f: -f["n_det"])[:5]
        res[tag]["worst_frames"] = [{"bmp": f["bmp"], "n_det": f["n_det"]} for f in worst]
        h = r["heatmap"]
        best = max(c.values(), key=lambda x: x["f1"])
        at30 = c.get("0.30")
        print(f"[ort ] {label}: heatmap max={h['max']:.4f} mean_frame_mean={h['mean_of_frame_means']:.5f} "
              f"px>0.5/frame={h['mean_px_gt_0.5']:.1f}  det@0.05={r['detections_total_at_collect_th']}")
        print(f"[ort ] {label}: best F1={best['f1']:.4f} @ th={best['th']}"
              + (f" | @0.30 F1={at30['f1']:.4f} TP={at30['tp']} FP={at30['fp']}" if at30 else ""))
        for w in res[tag]["worst_frames"][:3]:
            print(f"[ort ]   worst frame: {Path(w['bmp']).name}  det={w['n_det']}")

    hb, ht = res["base"]["heatmap"], res["test"]["heatmap"]
    res["comparison"] = {
        "heatmap_max_ratio": (ht["max"] / hb["max"]) if hb["max"] else None,
        "mean_frame_mean_ratio": (ht["mean_of_frame_means"] / hb["mean_of_frame_means"]
                                  if hb["mean_of_frame_means"] else None),
        "px_gt_0.5_ratio": (ht["mean_px_gt_0.5"] / hb["mean_px_gt_0.5"]
                            if hb["mean_px_gt_0.5"] else None),
        "detections_ratio": (res["test"]["detections_total_at_collect_th"] /
                             res["base"]["detections_total_at_collect_th"]
                             if res["base"]["detections_total_at_collect_th"] else None),
        "same_graph_structure": (res["graph_base"]["ops"] == res["graph_test"]["ops"]),
        "verdict": ("FLOAT_ALSO_EXPLODES -> the WEIGHTS/EXPORT are wrong"
                    if ht["mean_px_gt_0.5"] > 3 * max(hb["mean_px_gt_0.5"], 1e-9)
                    else "float is fine -> the QUANTISATION RANGE is the problem"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[ort ] comparison: {json.dumps(res['comparison'], ensure_ascii=False)}")
    print(f"[ort ] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
