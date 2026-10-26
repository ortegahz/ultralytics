#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
19_board_threshold_sweep.py -- sweep the DETECTION threshold offline, on
detections the board already produced.

Why this script exists
----------------------
The int8 loss on this model is NOT a recall loss.  On the same 2000 frames:

    arm            F1      TP     FP
    PT float      0.7484   1282   171
    board fp16    0.7395   1273   184
    board int8    0.6448   1359   527      <- TP is HIGHER than fp16

The int8 arm finds MORE true targets than fp16 and loses 100% of the ground in
false positives.  That points at one of two very different situations:

  (a) the quantisation noise manufactures spurious heatmap peaks  -> raising
      the threshold is the fix, and the int8 curve is a shifted copy of the
      fp16 curve;
  (b) the int8 score distribution is simply offset             -> the fix is a
      threshold re-calibration, and the int8 F1 at some th' can be compared
      against fp16 at 0.3.

A single F1 number at one threshold cannot tell (a) from (b).  A threshold
CURVE can.  So this script collects the board detections ONCE at a LOW
collection threshold and then re-scores the SAME detections at every
threshold of interest, entirely offline.

    board run (one pass, --th <collect_th>)  ->  detections.tsv
                                                 |
    offline re-score at th = 0.05 .. 0.70      +--> F1/R/P/TP/FP/FN curve

Collection threshold vs scoring threshold
-----------------------------------------
The board's own `--th` is the ONLY thing that decides which peaks reach the
TSV, so the collection threshold has to be LOW (default 0.05) for the sweep to
have anything to sweep.  The scoring threshold is applied on the host, on the
same detections, exactly the way 04_run_board_fast_eval.py matches them
against the GT (same C.match_distance, same letterbox geometry, same DIST_THR).
The curve is therefore comparable with the 04 numbers at th=0.3 by
construction.

The board's peak decoder uses `thr` only to filter peaks; the 3x3 local-max
NMS and the top-k cut are independent of it, so a lower collection threshold
does not move the surviving peaks.

Usage
-----
    # (a) score the TSVs a previous run already left in /tmp/ptq_board_out
    python 19_board_threshold_sweep.py --manifest <fast/x.json> \\
        --tsv-dir /tmp/ptq_board_out/p0_off_A_normal --out <out>

    # (b) run the board once at a low collection threshold, then sweep
    python 19_board_threshold_sweep.py --manifest <fast/x.json> \\
        --models a.rknn b.rknn --collect-th 0.05 --out <out>
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
# 04_run_board_fast_eval.py starts with a digit, so it cannot be imported by
# name; load it by path instead of duplicating its board driver.
_spec = importlib.util.spec_from_file_location(
    "ptq_04", HERE / "04_run_board_fast_eval.py")
_04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_04)

parse_board_tsv = _04.parse_board_tsv
score_rows = _04.score_rows
run_on_board = _04.run_on_board
run_pt_reference = _04.run_pt_reference


def load_manifest(path: str) -> tuple[list[dict], str]:
    manifest = C.read_json(path)
    if isinstance(manifest, list):
        return manifest, Path(path).stem
    return manifest["items"], manifest.get("name", Path(path).stem)


def read_src_hw(rows: list[dict]) -> dict[str, tuple[int, int]]:
    import cv2
    src_hw: dict[str, tuple[int, int]] = {}
    for r in rows:
        seq = r["seq"]
        if seq in src_hw:
            continue
        img = cv2.imread(r["bmp"], cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"[FATAL] cannot read staged bmp: {r['bmp']}")
        src_hw[seq] = (img.shape[0], img.shape[1])
    return src_hw


def sweep(det_by_name: dict, rows: list[dict], src_hw: dict,
          thresholds: list[float]) -> list[dict]:
    """Re-score the SAME detections at every threshold."""
    curve = []
    for th in thresholds:
        filtered = {k: [d for d in v if d[2] >= th] for k, v in det_by_name.items()}
        s = score_rows(filtered, rows, src_hw)
        curve.append({
            "th": th,
            "f1": s["f1"], "precision": s["precision"], "recall": s["recall"],
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"], "det": s["det"],
        })
    return curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--models", nargs="*", default=[],
                    help="RKNN models to run on the board at --collect-th")
    ap.add_argument("--tsv-dir", default="",
                    help="score an existing board output dir instead of running")
    ap.add_argument("--collect-th", type=float, default=0.05,
                    help="the BOARD's collection threshold (low = more peaks)")
    ap.add_argument("--thresholds", default="",
                    help="comma separated scoring thresholds; default = a sweep")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--variant", default="p0_off", choices=list(C.VARIANTS))
    ap.add_argument("--pt-device", default="cuda:0")
    ap.add_argument("--skip-pt", action="store_true")
    ap.add_argument("--board-bin", default=_04.BOARD_BIN)
    ap.add_argument("--board-models-dir", default="/mnt/manu/ptq/models")
    ap.add_argument("--board-timeout", type=int, default=_04.BOARD_RUN_TIMEOUT,
                    help="hard cap on ONE board sequence run, in seconds")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    rows, mname = load_manifest(args.manifest)
    print(f"[manifest] {args.manifest}: {len(rows)} frames ({mname})")
    src_hw = read_src_hw(rows)
    print(f"[geom] {len(src_hw)} sequences, sizes "
          f"{sorted({v for v in src_hw.values()})}")

    if args.thresholds:
        thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    else:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                      0.50, 0.55, 0.60, 0.70]

    report = {
        "manifest": args.manifest, "manifest_name": mname,
        "variant": args.variant, "frames": len(rows), "topk": args.topk,
        "collect_th": args.collect_th, "thresholds": thresholds,
        "arms": {},
    }

    # ---------------- PT reference curve (the float anchor) ----------------
    if not args.skip_pt:
        print("[PT] running the PyTorch reference on the same BMPs ...", flush=True)
        t0 = time.time()
        det = run_pt_reference(rows, src_hw, args.variant, topk=args.topk,
                               device=args.pt_device)
        report["arms"]["pt_float"] = {
            "curve": sweep(det, rows, src_hw, thresholds),
            "seconds": round(time.time() - t0, 1),
        }
        best = max(report["arms"]["pt_float"]["curve"], key=lambda c: c["f1"])
        at30 = [c for c in report["arms"]["pt_float"]["curve"] if c["th"] == 0.30]
        print(f"[PT] best F1={best['f1']:.4f} @ th={best['th']}  "
              f"({report['arms']['pt_float']['seconds']}s)")
        if at30:
            print(f"[PT] @0.30 F1={at30[0]['f1']:.4f} TP={at30[0]['tp']} FP={at30[0]['fp']}")

    # ---------------- board arms ----------------
    if args.models:
        for mp in args.models:
            mpath = Path(mp)
            C.require(mpath, "RKNN model")
            md5 = C.md5_file(mpath)
            tag = mpath.stem
            print(f"\n[board] {tag}  md5={md5}  collect_th={args.collect_th}",
                  flush=True)
            run = run_on_board(mpath, md5, rows, C.BOARD_NFS, args.collect_th,
                               args.topk, tag, args.board_bin,
                               args.board_models_dir, timeout=args.board_timeout)
            det_dir = Path("/tmp/ptq_board_out") / tag
            det_by_name: dict = {}
            for tsv in sorted(det_dir.glob("*.tsv")):
                det_by_name.update(parse_board_tsv(tsv))
            n_raw = sum(len(v) for v in det_by_name.values())
            print(f"[board] {tag}: {len(det_by_name)} frames, {n_raw} raw detections "
                  f"at collect_th={args.collect_th}")
            curve = sweep(det_by_name, rows, src_hw, thresholds)
            best = max(curve, key=lambda c: c["f1"])
            at30 = [c for c in curve if c["th"] == 0.30]
            print(f"[board] {tag}: best F1={best['f1']:.4f} @ th={best['th']} "
                  f"TP={best['tp']} FP={best['fp']}")
            if at30:
                print(f"[board] {tag}: @0.30 F1={at30[0]['f1']:.4f} "
                      f"TP={at30[0]['tp']} FP={at30[0]['fp']}")
            report["arms"][f"board_{tag}"] = {
                "model": str(mpath), "model_md5": md5,
                "collect_th": args.collect_th, "raw_detections": n_raw,
                "board_run": run, "curve": curve,
            }
            # keep the raw detections so the curve can be re-derived later
            keep = out_root / f"dets_{tag}.tsv"
            with keep.open("w") as fh:
                for name, ds in sorted(det_by_name.items()):
                    fh.write(f"{name}\t{len(ds)}")
                    for x, y, s in ds:
                        fh.write(f"\t{x} {y} {s}")
                    fh.write("\n")
            print(f"[board] raw detections -> {keep}")

    # ---------------- existing TSV mode ----------------
    if args.tsv_dir:
        tdir = Path(args.tsv_dir)
        det_by_name: dict = {}
        for tsv in sorted(tdir.glob("*.tsv")):
            det_by_name.update(parse_board_tsv(tsv))
        n_raw = sum(len(v) for v in det_by_name.values())
        tag = tdir.name
        print(f"\n[tsv] {tdir}: {len(det_by_name)} frames, {n_raw} detections")
        curve = sweep(det_by_name, rows, src_hw, thresholds)
        best = max(curve, key=lambda c: c["f1"])
        print(f"[tsv] {tag}: best F1={best['f1']:.4f} @ th={best['th']} "
              f"TP={best['tp']} FP={best['fp']}")
        report["arms"][f"tsv_{tag}"] = {
            "tsv_dir": str(tdir), "raw_detections": n_raw, "curve": curve,
        }

    report_path = out_root / f"{mname}_threshold_sweep.json"
    C.write_json(report_path, report)
    print(f"\n[report] -> {report_path}")

    # a compact table for the log
    print("\n=== threshold sweep ===")
    hdr = f"{'arm':28s}" + "".join(f"{t:>9.2f}" for t in thresholds)
    print(hdr)
    for name, arm in report["arms"].items():
        by_th = {c["th"]: c for c in arm["curve"]}
        line = f"{name:28s}"
        for t in thresholds:
            c = by_th.get(t)
            line += f"{c['f1']:>9.4f}" if c else f"{'-':>9s}"
        print(line)
    print("\n=== FP at each threshold ===")
    print(hdr)
    for name, arm in report["arms"].items():
        by_th = {c["th"]: c for c in arm["curve"]}
        line = f"{name:28s}"
        for t in thresholds:
            c = by_th.get(t)
            line += f"{c['fp']:>9d}" if c else f"{'-':>9s}"
        print(line)


if __name__ == "__main__":
    main()
