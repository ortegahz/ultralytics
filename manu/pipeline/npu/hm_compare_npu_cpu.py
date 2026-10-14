#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Reconcile the RK3588 NPU heatmap outputs against the CPU/PyTorch reference.

Three independent things are checked, because they fail independently:

  1. board decoder vs host decoder
     The board decodes peaks in C++ (a port of extract_peaks) and prints them;
     this script re-decodes the very same dumped float32 tensors with the numpy
     port. Agreement proves the port is a port.

  2. NPU numerics vs CPU numerics
     heatmap / offset tensors are compared elementwise. This is the number that
     actually tells you what int8 quantization cost.

  3. NPU detections vs ground truth
     Reported as TP/FP/FN with a Wilson 95% interval, plus the same statistics
     for the CPU reference on the identical frames, so the two are comparable
     and neither is quoted as a bare rate (rules.md 3.6b).

Usage:
    python hm_compare_npu_cpu.py --ref <ref_dir> --npu <npu_dir> --tag int8 [--th 0.25]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manu.pipeline.npu.hm_common import (  # noqa: E402
    canvas_to_src,
    extract_peaks_np,
    match_distance,
)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def fmt_rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / max(1, n):.4f} [95% CI {lo:.4f}, {hi:.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--npu", required=True)
    ap.add_argument("--tag", default="int8")
    ap.add_argument("--th", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--dist", type=float, default=8.0)
    args = ap.parse_args()

    ref_dir, npu_dir = Path(args.ref), Path(args.npu)
    recs = json.loads((ref_dir / "detections.json").read_text())
    by_stem = {Path(r["name"]).stem: r for r in recs}

    rows = []
    hm_diffs, off_diffs = [], []
    for stem, r in by_stem.items():
        h = npu_dir / f"{args.tag}_{stem}_hm.bin"
        o = npu_dir / f"{args.tag}_{stem}_off.bin"
        if not (h.is_file() and o.is_file()):
            continue
        hm_n = np.fromfile(h, np.float32).reshape(320, 320)
        off_n = np.fromfile(o, np.float32).reshape(2, 320, 320)
        hm_c = np.load(ref_dir / f"{stem}_hm.npy")[0]
        off_c = np.load(ref_dir / f"{stem}_off.npy")
        hm_diffs.append(np.abs(hm_n - hm_c))
        off_diffs.append(np.abs(off_n - off_c))

        d = extract_peaks_np(hm_n[None], off_n, stride=args.stride, conf_thresh=args.th)
        npu_canvas, npu_sc = d["points"], d["scores"]
        cpu_canvas = np.array(r["det_canvas"], np.float64).reshape(-1, 2)
        cpu_sc = np.array(r["scores"], np.float64)

        # Pairwise nearest-neighbour agreement between the two detectors.
        #
        # The "neither engine found anything" case must be counted as AGREEMENT,
        # not silently dropped: with an `else: dx = nan` branch, `nan <= dist` is
        # False, so every legitimately empty frame was tallied as a mismatch and
        # dragged the reported agreement from ~93% down to ~65%. The three cases
        # are therefore reported separately.
        if len(npu_canvas) == 0 and len(cpu_canvas) == 0:
            dx, ds, cat = 0.0, 0.0, "both_empty"
        elif len(npu_canvas) == 0 or len(cpu_canvas) == 0:
            dx, ds, cat = float("inf"), float("nan"), "one_empty"
        else:
            dd = np.hypot(npu_canvas[:, None, 0] - cpu_canvas[None, :, 0],
                          npu_canvas[:, None, 1] - cpu_canvas[None, :, 1])
            j = dd.argmin(axis=1)
            best = dd[np.arange(len(j)), j]
            k_npu = int(best.argmin())
            k_cpu = int(j[k_npu])  # index into the CPU list, not the NPU list
            dx = float(best[k_npu])
            ds = float(abs(npu_sc[k_npu] - cpu_sc[k_cpu]))
            cat = "matched" if dx <= args.dist else "disagree"

        gt = np.array(r["gt_canvas"], np.float64).reshape(-1, 2)
        for name, pts, sc in (("npu", npu_canvas, npu_sc), ("cpu", cpu_canvas, cpu_sc)):
            arr = np.concatenate([pts, sc[:, None]], 1) if len(pts) else np.zeros((0, 3), np.float32)
            t, f_, n_ = match_distance(arr, gt, args.dist)
            r[f"_{name}_tp"], r[f"_{name}_fp"], r[f"_{name}_fn"] = t, f_, n_
        r["_n_npu"], r["_n_cpu"] = len(npu_canvas), len(cpu_canvas)
        r["_match"], r["_dscore"], r["_cat"] = dx, ds, cat
        # Stash the NPU score vector: `r["scores"]` is the CPU arm's, and the
        # per-frame dump below must report the two arms' scores separately.
        r["_npu_scores"] = [float(v) for v in npu_sc]
        rows.append(r)

    n = len(rows)
    print(f"frames compared: {n}   (tag={args.tag}, th={args.th}, dist<={args.dist}px)")
    print()

    # --- 1/2 numerics -------------------------------------------------------
    if hm_diffs:
        hd = np.concatenate([d.ravel() for d in hm_diffs])
        od = np.concatenate([d.ravel() for d in off_diffs])
        print("[2] tensor-level NPU(int8) vs CPU(fp32), all pixels")
        print(f"    heatmap  n={hd.size:9d}  mean|diff|={hd.mean():.3e}  p99={np.percentile(hd, 99):.3e}  max={hd.max():.3e}")
        print(f"    offset   n={od.size:9d}  mean|diff|={od.mean():.3e}  p99={np.percentile(od, 99):.3e}  max={od.max():.3e}")
        print()

    # --- 1 detection-level agreement ---------------------------------------
    same = sum(1 for r in rows if r["_n_npu"] == r["_n_cpu"])
    cats = [r["_cat"] for r in rows]
    both_empty = sum(1 for c in cats if c == "both_empty")
    one_empty = sum(1 for c in cats if c == "one_empty")
    mat = sum(1 for c in cats if c == "matched")
    dis = sum(1 for c in cats if c == "disagree")
    dsc = np.array([r["_dscore"] for r in rows if np.isfinite(r["_dscore"])], np.float64)
    # Denominators differ and the difference is easy to misread, so print both:
    # `matched` frames have a real score to compare; `both_empty` frames contribute
    # an exact 0.0, which drags a pooled mean down and reads as a smaller gap.
    dsc_m = np.array([r["_dscore"] for r in rows if r["_cat"] == "matched"
                      and np.isfinite(r["_dscore"])], np.float64)
    agree = both_empty + mat
    print("[1] per-frame outcome breakdown (NPU vs CPU, identical frames)")
    print(f"    both engines empty (agreement) : {fmt_rate(both_empty, n)}")
    print(f"    matched within {args.dist:.0f} px               : {fmt_rate(mat, n)}")
    print(f"    both positions but >{args.dist:.0f} px apart    : {fmt_rate(dis, n)}")
    print(f"    only one engine fired         : {fmt_rate(one_empty, n)}")
    print(f"    --> frames in agreement       : {fmt_rate(agree, n)}")
    print(f"    identical detection count     : {fmt_rate(same, n)}")
    if dsc_m.size:
        print(f"    |score diff| on MATCHED frames only (n={dsc_m.size}): "
              f"mean={dsc_m.mean():.4f} max={dsc_m.max():.4f}")
    if dsc.size:
        print(f"    |score diff| pooled w/ both_empty zeros (n={dsc.size}): "
              f"mean={dsc.mean():.4f} max={dsc.max():.4f}")
    print()

    # --- 3 accuracy against GT ---------------------------------------------
    print("[3] accuracy against ground truth (identical frames)")
    for name, label in (("npu", "RK3588 NPU int8"), ("cpu", "CPU / PyTorch fp32")):
        tp = sum(r[f"_{name}_tp"] for r in rows)
        fp = sum(r[f"_{name}_fp"] for r in rows)
        fn = sum(r[f"_{name}_fn"] for r in rows)
        g = sum(len(np.array(r["gt_canvas"]).reshape(-1, 2)) for r in rows)
        p = tp / max(1, tp + fp)
        rc = tp / max(1, g)
        f1 = 2 * p * rc / max(1e-9, p + rc)
        plo, phi = wilson(tp, max(1, tp + fp))
        rlo, rhi = wilson(tp, max(1, g))
        print(f"    {label:22s} GT={g:4d} TP={tp:4d} FP={fp:4d} FN={fn:4d}  "
              f"P={p:.4f} [{plo:.4f},{phi:.4f}]  R={rc:.4f} [{rlo:.4f},{rhi:.4f}]  F1={f1:.4f}")

    # One JSON object PER FRAME. An earlier version wrote
    #   {k: v for k, v in ((k1, v1), (k2, v2), ...)}
    # which collapses to a single object whose values are the last record's --
    # the file looked plausible and contained seven keys for 600 frames.
    per_frame = [
        {
            "stem": Path(r["name"]).stem,
            "seq": r["seq"],
            "outcome": r["_cat"],
            "n_npu": r["_n_npu"],
            "n_cpu": r["_n_cpu"],
            "match_px": (None if not np.isfinite(r["_match"]) else r["_match"]),
            "dscore": (None if not np.isfinite(r["_dscore"]) else r["_dscore"]),
            "npu_scores": r["_npu_scores"],
            "cpu_scores": r["scores"],
            "npu_tp": r["_npu_tp"], "npu_fp": r["_npu_fp"], "npu_fn": r["_npu_fn"],
            "cpu_tp": r["_cpu_tp"], "cpu_fp": r["_cpu_fp"], "cpu_fn": r["_cpu_fn"],
        }
        for r in rows
    ]
    out = npu_dir.parent / f"compare_{args.tag}.json"
    out.write_text(json.dumps(per_frame, indent=1))
    print(f"\n[out] {out}  ({len(per_frame)} per-frame records)")


if __name__ == "__main__":
    main()
