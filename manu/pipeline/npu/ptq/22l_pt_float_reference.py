#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
22l_pt_float_reference.py -- the PT float reference curve for ONE manifest.

WHY A SEPARATE STEP
-------------------
The quantisation loss is only meaningful against a float reference measured on
the SAME frames, with the SAME decoder and the SAME GT.  The PT reference does
not depend on the arm, so it is computed ONCE per manifest and then handed to
every arm's report (22h_pc_sim_eval_qat_arms.py --pt-ref), instead of being
recomputed five times.

The reference is the trial_0080 float model in the p0_off variant -- the same
model the board sweep's pt_float arm uses, so the numbers are comparable with
the existing tier2 reports.

Usage (server, conda env `uav` -- needs torch+CUDA):
    python manu/pipeline/npu/ptq/22l_pt_float_reference.py \
        --manifest /mnt/data/siping/ptq_work/fast/001324_sub2000.json \
        --device cuda:1 --variant p0_off \
        --out /mnt/data/siping/ptq_work/qat/trial_0080/pt_ref/001324_sub2000.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402


def _load_04():
    p = Path(__file__).resolve().parent / "04_run_board_fast_eval.py"
    spec = importlib.util.spec_from_file_location("_ptq_04", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--variant", default="p0_off")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--thresholds", default="")
    args = ap.parse_args()

    _04 = _load_04()
    thresholds = ([float(x) for x in args.thresholds.split(",") if x.strip()]
                  if args.thresholds else
                  [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                   0.50, 0.55, 0.60, 0.70])

    raw = C.read_json(args.manifest)
    rows = raw if isinstance(raw, list) else raw["items"]
    name = Path(args.manifest).stem
    print(f"[pt ] {name}: {len(rows)} frames", flush=True)

    src_hw = {}
    import cv2
    for r in rows:
        if r["seq"] in src_hw:
            continue
        img = cv2.imread(r["bmp"], cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"[FATAL] cannot read {r['bmp']}")
        src_hw[r["seq"]] = img.shape[:2]

    t0 = time.time()
    det = _04.run_pt_reference(rows, src_hw, args.variant, topk=args.topk,
                              device=args.device)
    seconds = round(time.time() - t0, 1)

    curve = {}
    for th in thresholds:
        filtered = {k: [d for d in v if d[2] >= th] for k, v in det.items()}
        s = _04.score_rows(filtered, rows, src_hw)
        curve[f"{th:.2f}"] = {"th": th, "f1": s["f1"], "precision": s["precision"],
                              "recall": s["recall"], "tp": s["tp"], "fp": s["fp"],
                              "fn": s["fn"], "det": s["det"]}
    best = max(curve.values(), key=lambda c: c["f1"])
    at30 = curve.get("0.30")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "manifest": args.manifest, "manifest_name": name, "n_frames": len(rows),
        "variant": args.variant, "device": args.device, "topk": args.topk,
        "seconds": seconds, "curve": curve,
        "best_f1": best["f1"], "best_th": best["th"],
        "f1_at_0.30": at30["f1"] if at30 else None,
    }, ensure_ascii=False, indent=1))
    print(f"[pt ] {name}: best F1={best['f1']:.4f} @ th={best['th']}  "
          f"({seconds}s)")
    if at30:
        print(f"[pt ] {name}: @0.30 F1={at30['f1']:.4f} TP={at30['tp']} FP={at30['fp']}")
    print(f"[pt ] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
