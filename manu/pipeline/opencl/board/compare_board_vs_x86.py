#!/usr/bin/env python3
"""
compare_board_vs_x86.py -- cross-architecture accuracy comparison of the
gmc_stream_ocl feature channel.

What this compares
------------------
Two independent full runs of the SAME pipeline over the SAME BMP sequence:

  x86    Ubuntu OpenCV 4.10.0 (dispatch AVX2/AVX512) + PoCL on CPU
  board  cross-built OpenCV 4.10.0 (NEON)                 + Mali-G610

Each writes one .npy per push, shaped (3, H, W) = [Ch0, Ch1, Ch2]. The question
this answers is whether "run it on the board" changes the feature tensor.

Why the per-frame nature matters
--------------------------------
The pipeline is stateful: a median of 21 history frames and a chained transform
composition mean frame k depends on every frame before it. A single aggregate
MAE over 60 frames cannot distinguish "every frame is slightly off" from
"frame 3 diverged and dragged the median window with it". So the report shows
both the aggregate and the per-frame worst offenders, and a divergence index
first appearing at frame N is reported explicitly.

Interpretation of the numbers
-----------------------------
This is NOT expected to be bit-exact, and earlier work established why:
  * LK sub-pixel positions differ between the two architectures (max ~1e-4 px),
    already measured in opencv_arm64_parity.md and traced to architecture-
    specific SIMD plus a 6-year compiler gap, not to FMA.
  * x86 dispatches to AVX2/AVX512 code; the board uses NEON.
Those differences are inputs to the fit, so they propagate. The question this
script answers is whether the propagated result stays inside the project's
existing acceptance gate -- Max|Diff| <= 1 and MAE <= 0.05 per channel, the same
gate used for the CPU/GPU A/B.
"""

import argparse
import os
import sys

import numpy as np

GATE_MAXDIFF = 1.0
GATE_MAE = 0.05


def load(path):
    a = np.load(path)
    if a.ndim != 3:
        raise ValueError(f"{path}: expected (3,H,W), got {a.shape}")
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x86", required=True)
    ap.add_argument("--board", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    names = sorted(f for f in os.listdir(args.x86) if f.endswith("_feat.npy"))
    if not names:
        print(f"[ERROR] no *_feat.npy under {args.x86}")
        return 2

    print("=" * 74)
    print("Cross-architecture feature comparison -- gmc_stream_ocl")
    if args.label:
        print(args.label)
    print("=" * 74)
    print(f"frames compared : {len(names)}")
    print(f"gate            : Max|Diff| <= {GATE_MAXDIFF:g}  and  MAE <= {GATE_MAE:g} per channel")
    print()

    ch_names = ["Ch0", "Ch1", "Ch2"]
    agg = {c: {"max": 0, "sum_abs": 0.0, "n": 0, "exact": 0, "over_gate": 0}
           for c in ch_names}
    per_frame = []
    missing = []

    for nm in names:
        px = os.path.join(args.x86, nm)
        pb = os.path.join(args.board, nm)
        if not os.path.exists(pb):
            missing.append(nm)
            continue
        a = load(px).astype(np.int16)
        b = load(pb).astype(np.int16)
        if a.shape != b.shape:
            print(f"[ERROR] shape mismatch on {nm}: {a.shape} vs {b.shape}")
            return 2

        d = np.abs(a - b)
        row = {"name": nm}
        for i, c in enumerate(ch_names):
            di = d[i]
            mx = int(di.max())
            mae = float(di.sum()) / di.size
            agg[c]["max"] = max(agg[c]["max"], mx)
            agg[c]["sum_abs"] += float(di.sum())
            agg[c]["n"] += di.size
            agg[c]["exact"] += int((di == 0).sum())
            agg[c]["over_gate"] += int((di > GATE_MAXDIFF).sum())
            row[c] = (mx, mae)
        per_frame.append(row)

    if missing:
        print(f"[WARN] {len(missing)} frame(s) present on x86 but not on the board, "
              f"first: {missing[0]}")
        print()

    # ---- aggregate -------------------------------------------------------
    print(f"{'channel':<8}{'Max|Diff|':>12}{'MAE':>12}{'exact %':>12}"
          f"{'px over gate':>15}{'verdict':>12}")
    print("-" * 74)
    overall_pass = True
    for c in ch_names:
        s = agg[c]
        mae = s["sum_abs"] / max(1, s["n"])
        exact = 100.0 * s["exact"] / max(1, s["n"])
        ok = (s["max"] <= GATE_MAXDIFF) and (mae <= GATE_MAE)
        overall_pass &= ok
        print(f"{c:<8}{s['max']:>12d}{mae:>12.6f}{exact:>11.4f}%"
              f"{s['over_gate']:>15d}{'PASS' if ok else 'FAIL':>12}")
    print("-" * 74)

    # ---- where the divergence starts ------------------------------------
    print()
    nonzero = [r for r in per_frame if any(r[c][0] > 0 for c in ch_names)]
    print(f"frames with any difference : {len(nonzero)}/{len(per_frame)}")
    if nonzero:
        print(f"first diverging frame      : {nonzero[0]['name']}"
              f"  (index {per_frame.index(nonzero[0])})")
        worst = sorted(per_frame,
                       key=lambda r: -max(r[c][0] for c in ch_names))[:5]
        print("worst frames:")
        for r in worst:
            parts = "  ".join(f"{c}:max={r[c][0]},mae={r[c][1]:.5f}" for c in ch_names)
            print(f"  {r['name']}  {parts}")
    else:
        print("every compared frame is bit-identical")

    # ---- transform comparison, when dumped --------------------------------
    m_x = os.path.join(args.x86, names[0].replace("_feat.npy", "_mats.npy"))
    m_b = os.path.join(args.board, names[0].replace("_feat.npy", "_mats.npy"))
    if os.path.exists(m_x) and os.path.exists(m_b):
        # mats are (n_mats, 6) float32 rows of partial-affine coefficients, not
        # image planes -- load() would reject them, and rightly so.
        mx_ = np.load(m_x)
        mb_ = np.load(m_b)
        print()
        print(f"transform matrices ({os.path.basename(m_x)}):")
        print(f"  x86   shape {mx_.shape}  dtype {mx_.dtype}")
        print(f"  board shape {mb_.shape}  dtype {mb_.dtype}")
        if mx_.shape == mb_.shape:
            dm = np.abs(mx_.astype(np.float64) - mb_.astype(np.float64))
            print(f"  Max|Diff| {dm.max():.6g}   mean {dm.mean():.6g}")
            exact = int((dm == 0).sum())
            print(f"  exactly equal entries {exact}/{dm.size}")
        else:
            print("  SHAPE MISMATCH -- the two runs produced different transform counts")

    print()
    if overall_pass:
        print("VERDICT: PASS -- the board's feature tensor stays inside the "
              "project gate on every channel.")
    else:
        print("VERDICT: FAIL -- at least one channel exceeds the gate.")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
