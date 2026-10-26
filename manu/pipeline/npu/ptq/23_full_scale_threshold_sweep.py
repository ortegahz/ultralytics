#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
23_full_scale_threshold_sweep.py -- turn the overnight full-scale board run
into the numbers the delivery actually needs:

  1. the threshold CURVE of the final int8 model on all three tracks
     (the board's detections are re-scored offline at every threshold), and
  2. the matching PyTorch float REFERENCE arm on the same frames, so the
     quantisation loss is measured against float on exactly the same bytes.

WHY THIS IS NEEDED
------------------
Two facts from 2026-10-22 night make a single F1 number at th=0.30
insufficient:

  * the int8 loss is 100% false positives (TP is HIGHER than fp16, FP is
    527 vs 184 on 2000 frames), and
  * quantisation SHIFTS the optimal detection threshold by +0.10 (int8 peaks
    at 0.45, fp16 at 0.35).  Re-scoring the int8 detections at 0.45 recovered
    63% of the loss (0.6448 -> 0.7090 on the 2000-frame subset).

So the delivery needs, per track: the F1 at th=0.30 (comparable with every
earlier measurement in this study) AND the F1 at the model-optimal threshold
(the deployable number), both against the float reference.

The board's TSVs are the frame-level truth: 14_run_full_scale_nofs.py
verified that the number of TSV rows equals the number of frames of the
track, and that the frame-name set matches exactly.

Usage
-----
    python 23_full_scale_threshold_sweep.py \
        --run-dir /mnt/data/siping/ptq_work/full_scale_int8 \
        --bmp-root /mnt/data/siping/ptq_work/bmp \
        --anti-root /mnt/data/siping/ptq_work/pcfeat_bmp/images \
        --out /mnt/data/siping/ptq_work/full_scale_sweep \
        --pt --pt-device cuda:1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

# The repo root is parents[4] (ptq -> npu -> pipeline -> manu -> <root>).
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_s04 = _load("ptq_04", "04_run_board_fast_eval.py")

parse_board_tsv = _s04.parse_board_tsv
score_rows = _s04.score_rows
run_pt_reference = _s04.run_pt_reference

SWEEP_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# per-track: the frame tree and the GT label root that 14 used
TRACKS = {
    "blind": ("bmp", "GT"),
    "golden": ("bmp", "GT"),
    "anti": ("anti", "ANTI"),
}


def split_name(name: str) -> tuple[str, str]:
    """'25_1__000003.bmp' -> ('25_1', '000003');  the sequence name may
    itself contain single underscores, so split on the LAST '__'."""
    stem = name[:-4] if name.lower().endswith(".bmp") else name
    seq, _, fid = stem.rpartition("__")
    return seq, fid


def build_rows(track: str, tsv_path: Path, bmp_root: Path,
               anti_root: Path) -> tuple[list[dict], dict[str, tuple[int, int]]]:
    """Rebuild the frame rows from the board TSV (the frame set is the truth)
    and locate the source BMP + the GT label for each frame."""
    import cv2

    det = parse_board_tsv(tsv_path)
    names = sorted(det.keys())
    rows: list[dict] = []
    src_hw: dict[str, tuple[int, int]] = {}

    for name in names:
        seq, fid = split_name(name)
        if track == "anti":
            bmp = anti_root / name
        else:
            bmp = bmp_root / seq / name
        if not bmp.is_file():
            raise SystemExit(f"[FATAL] source BMP missing for {name}: {bmp}")
        if seq not in src_hw:
            img = cv2.imread(str(bmp), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f"[FATAL] cannot read {bmp}")
            src_hw[seq] = (img.shape[0], img.shape[1])
        rows.append({"seq": seq, "frame_id": fid, "bmp": str(bmp),
                     "gt": str(C.gt_path(seq, int(fid), root=C.GT_ROOT))})

    # the Anti-UAV track has its own label tree
    if track == "anti":
        for r in rows:
            r["gt"] = str(C.ANTI_GT_ROOT / f"{r['seq']}__{r['frame_id']}.txt")
    return rows, src_hw


def sweep(det_by_name: dict, rows: list[dict], src_hw: dict,
          thresholds: list[float]) -> list[dict]:
    curve = []
    for th in thresholds:
        filtered = {k: [d for d in v if d[2] >= th] for k, v in det_by_name.items()}
        s = score_rows(filtered, rows, src_hw)
        curve.append({"th": th, "f1": s["f1"], "precision": s["precision"],
                      "recall": s["recall"], "tp": s["tp"], "fp": s["fp"],
                      "fn": s["fn"], "det": s["det"]})
    return curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="/mnt/data/siping/ptq_work/full_scale_int8")
    ap.add_argument("--bmp-root", default="/mnt/data/siping/ptq_work/bmp")
    ap.add_argument("--anti-root",
                    default="/mnt/data/siping/ptq_work/pcfeat_bmp/images")
    ap.add_argument("--out", default="/mnt/data/siping/ptq_work/full_scale_sweep")
    ap.add_argument("--thresholds", default="")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--variant", default="p0_off", choices=list(C.VARIANTS))
    ap.add_argument("--pt", action="store_true", help="also run the float reference")
    ap.add_argument("--pt-device", default="cuda:1")
    ap.add_argument("--tracks", nargs="+", default=["anti", "golden", "blind"],
                    choices=list(TRACKS))
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    thresholds = ([float(x) for x in args.thresholds.split(",") if x.strip()]
                  or SWEEP_THRESHOLDS)

    report = {"run_dir": str(run_dir), "thresholds": thresholds,
              "topk": args.topk, "tracks": {}}

    for track in args.tracks:
        tsv = run_dir / f"{track}_board.tsv"
        if not tsv.is_file():
            print(f"[skip] {track}: no TSV at {tsv}")
            continue
        print(f"\n===== track {track} =====", flush=True)
        rows, src_hw = build_rows(track, tsv, Path(args.bmp_root),
                                  Path(args.anti_root))
        det = parse_board_tsv(tsv)
        n_raw = sum(len(v) for v in det.values())
        print(f"[rows] {len(rows)} frames, {len(src_hw)} sequences, "
              f"{n_raw} raw detections", flush=True)

        board_curve = sweep(det, rows, src_hw, thresholds)
        best = max(board_curve, key=lambda c: c["f1"])
        at30 = [c for c in board_curve if c["th"] == 0.30]
        print(f"[board] F1@0.30 = {at30[0]['f1']:.4f} (FP {at30[0]['fp']})"
              if at30 else "", flush=True)
        print(f"[board] best  F1 = {best['f1']:.4f} @ th={best['th']} "
              f"(TP {best['tp']} FP {best['fp']})", flush=True)

        entry = {"frames": len(rows), "raw_detections": n_raw,
                 "board_curve": board_curve,
                 "board_best": best,
                 "board_at_030": at30[0] if at30 else None}

        if args.pt:
            print("[PT] running the float reference on the same BMPs ...", flush=True)
            t0 = time.time()
            pt_det = run_pt_reference(rows, src_hw, args.variant,
                                      topk=args.topk, device=args.pt_device)
            pt_curve = sweep(pt_det, rows, src_hw, thresholds)
            pt_best = max(pt_curve, key=lambda c: c["f1"])
            pt30 = [c for c in pt_curve if c["th"] == 0.30]
            print(f"[PT] F1@0.30 = {pt30[0]['f1']:.4f} (FP {pt30[0]['fp']})",
                  flush=True)
            print(f"[PT] best  F1 = {pt_best['f1']:.4f} @ th={pt_best['th']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            entry["pt_curve"] = pt_curve
            entry["pt_best"] = pt_best
            entry["pt_at_030"] = pt30[0] if pt30 else None
            entry["pt_seconds"] = round(time.time() - t0, 1)

        report["tracks"][track] = entry
        C.write_json(out_root / f"{track}_sweep.json", entry)

    # ---------------- the delivery table ----------------
    print("\n================ FINAL TABLE ================")
    print(f"{'track':8s}{'frames':>8s}{'int8@0.30':>11s}{'int8@best':>11s}"
          f"{'@th':>6s}{'fp@0.30':>9s}{'fp@best':>9s}")
    for track, e in report["tracks"].items():
        b30 = e.get("board_at_030") or {}
        bb = e.get("board_best") or {}
        print(f"{track:8s}{e['frames']:>8d}{b30.get('f1', 0):>11.4f}"
              f"{bb.get('f1', 0):>11.4f}{bb.get('th', 0):>6.2f}"
              f"{b30.get('fp', 0):>9d}{bb.get('fp', 0):>9d}")
    if args.pt:
        print("\n--- float reference (PT) ---")
        print(f"{'track':8s}{'frames':>8s}{'PT@0.30':>11s}{'PT@best':>11s}"
              f"{'@th':>6s}{'loss@0.30':>11s}{'loss@best':>11s}")
        for track, e in report["tracks"].items():
            p30 = e.get("pt_at_030") or {}
            pb = e.get("pt_best") or {}
            b30 = e.get("board_at_030") or {}
            bb = e.get("board_best") or {}
            loss30 = (b30.get("f1", 0) - p30.get("f1", 0))
            lossb = (bb.get("f1", 0) - pb.get("f1", 0))
            print(f"{track:8s}{e['frames']:>8d}{p30.get('f1', 0):>11.4f}"
                  f"{pb.get('f1', 0):>11.4f}{pb.get('th', 0):>6.2f}"
                  f"{loss30:>11.4f}{lossb:>11.4f}")

    C.write_json(out_root / "full_scale_sweep.json", report)
    print(f"\n[report] -> {out_root/'full_scale_sweep.json'}")


if __name__ == "__main__":
    main()
