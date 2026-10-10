#!/usr/bin/env python3
"""
compare_opencv_parity.py -- compare two opencv_parity output trees element by
element and issue a per-stage verdict.

Run on the x86 host. Both sides write their dumps to the NFS share, so this
needs no transfer back from the board.

Why per-stage
-------------
A single aggregate number cannot tell you *where* the two architectures part
company. If Shi-Tomasi finds 601 corners on x86 and 600 on arm64, every later
stage is already comparing different things and the "error" is meaningless.
So each operator is judged on its own, and a shape/count mismatch is reported
as its own failure class rather than as a large pixel error.

Why the control stage matters
-----------------------------
median_out is a 21-element selection over uint8 -- no floating point at all.
Given identical warped inputs it *must* come out bit-identical. If it differs,
the pipeline feeding it already diverged and any error downstream says nothing
about median. It is therefore reported separately and used as a sanity anchor.
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np

# depth id -> (name, numpy dtype, itemsize)
DEPTH = {
    0: ("CV_8U", np.uint8, 1),
    1: ("CV_8S", np.int8, 1),
    2: ("CV_16U", np.uint16, 2),
    3: ("CV_16S", np.int16, 2),
    4: ("CV_32S", np.int32, 4),
    5: ("CV_32F", np.float32, 4),
    6: ("CV_64F", np.float64, 8),
    7: ("CV_16F", np.float16, 2),
}

NAME_RE = re.compile(r"^(?P<name>.+)_f(?P<frame>\d+)\.(?P<ext>bin|txt)$")

# Stages in pipeline order, with what each one is expected to prove.
STAGE_ROLE = {
    "resize_prev": "cv::resize INTER_LINEAR 640x512 -> 320x256",
    "resize_curr": "cv::resize INTER_LINEAR 640x512 -> 320x256",
    "gftt": "cv::goodFeaturesToTrack Shi-Tomasi corners",
    "lk_curr": "cv::calcOpticalFlowPyrLK positions",
    "lk_status": "cv::calcOpticalFlowPyrLK status flags",
    "lk_err": "cv::calcOpticalFlowPyrLK min-Eig error",
    "lk_p0": "LK input points after status filter",
    "lk_p1": "LK tracked points after status filter",
    "affine_M": "cv::estimateAffinePartial2D RANSAC 2x3 CV_64F",
    "affine_inl": "RANSAC inlier mask",
    "fit_H": "fit converted to float32 and scaled by downscale",
    "warp_dst": "cv::warpAffine INTER_LINEAR BORDER_REFLECT",
    "median_out": "21-element uint8 median (CONTROL: integer-only)",
}


def read_header(path):
    meta = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k] = v
    return meta


def load(dirpath, name, frame):
    base = os.path.join(dirpath, f"{name}_f{frame:04d}")
    hdr, binp = base + ".txt", base + ".bin"
    if not (os.path.exists(hdr) and os.path.exists(binp)):
        return None
    meta = read_header(hdr)
    depth = int(meta["depth"])
    dname, dtype, _ = DEPTH[depth]
    rows, cols, ch = int(meta["rows"]), int(meta["cols"]), int(meta["channels"])
    # The header's `elems` is Mat::total(), which counts channels as part of one
    # element. fromfile needs a count of whole scalar values, so multi-channel
    # Mats (CV_32FC2 points) must be counted as rows*cols*channels.
    nscalars = rows * cols * ch
    arr = np.fromfile(binp, dtype=dtype, count=nscalars)
    if arr.size != nscalars:
        return None
    return arr.reshape(rows, cols, ch), meta


def ulp_distance(a, b):
    """Elementwise distance in representable steps, order-independent.

    Bit patterns are folded onto a monotonic unsigned key: values with the sign
    bit set are mirrored via `SIGN - bits`, which wraps around in unsigned
    arithmetic and so keeps -0.0 < -1.0 < ... < +0.0 < ... < +inf in order.
    Signed arithmetic cannot express this -- the mirror of INT64_MIN needs
    2^64, which does not fit -- so the unsigned wrap is the whole point.
    """
    def key(x):
        x = np.asarray(x)
        if x.dtype == np.float32:
            u = x.view(np.uint32)
            sign = np.uint32(1) << np.uint32(31)
            return np.where(u & sign != 0, sign - u, u).astype(np.uint64)
        u = x.view(np.uint64)
        sign = np.uint64(1) << np.uint64(63)
        return np.where(u & sign != 0, sign - u, u)

    ka, kb = key(a), key(b)
    # np.where evaluates both branches, and the losing branch WOULD underflow,
    # but it is discarded -- so the branch that is kept is always the
    # non-negative difference. Selecting first is what avoids the wrap.
    return np.where(ka >= kb, ka - kb, kb - ka)


def compare(a, b, meta):
    res = {}
    if a.shape != b.shape:
        res["class"] = "SHAPE_MISMATCH"
        res["a_shape"] = a.shape
        res["b_shape"] = b.shape
        return res

    if a.dtype.kind in "iu":
        diff = np.abs(a.astype(np.int64) - b.astype(np.int64))
        res["class"] = "INT"
        res["max_abs"] = int(diff.max()) if diff.size else 0
        res["n_diff"] = int((diff != 0).sum())
        res["total"] = int(diff.size)
        res["frac"] = res["n_diff"] / max(1, res["total"])
        res["exact"] = res["n_diff"] == 0
    else:
        finite = np.isfinite(a) & np.isfinite(b)
        n_nan_mismatch = int((np.isfinite(a) != np.isfinite(b)).sum())
        res["class"] = "FLOAT"
        res["n_nonfinite_mismatch"] = n_nan_mismatch
        if not finite.any():
            res["max_abs"] = float("nan")
            res["mae"] = float("nan")
            res["max_ulp"] = 0
            res["exact"] = n_nan_mismatch == 0
            return res
        d = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
        u = ulp_distance(a[finite], b[finite])
        res["max_abs"] = float(d.max())
        res["mae"] = float(d.mean())
        res["max_ulp"] = int(u.max())
        res["median_ulp"] = float(np.median(u))
        res["p99_ulp"] = float(np.percentile(u, 99))
        res["exact"] = (d.max() == 0) and n_nan_mismatch == 0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference dump dir")
    ap.add_argument("--test", required=True, help="test dump dir")
    ap.add_argument("--label", default=None,
                    help="describe the comparison; defaults to the two paths")
    ap.add_argument("--frames", type=int, default=1000)
    args = ap.parse_args()

    label = args.label or f"{args.ref}\n{' ' * len(args.ref)} vs {args.test}"

    # Discover the stages actually present rather than assuming.
    stages = {}
    for fn in os.listdir(args.ref):
        m = NAME_RE.match(fn)
        if m and m.group("ext") == "txt":
            stages.setdefault(m.group("name"), []).append(int(m.group("frame")))

    if not stages:
        print(f"[ERROR] no stage headers found in {args.ref}")
        return 2

    order = ["resize_prev", "resize_curr", "gftt", "lk_curr", "lk_status", "lk_err",
             "lk_p0", "lk_p1", "affine_M", "affine_inl", "fit_H", "warp_dst", "median_out"]
    names = [s for s in order if s in stages] + \
            [s for s in stages if s not in order]

    print("=" * 78)
    print("OpenCV 4.10.0 -- per-operator x86/arm64 comparison")
    print(label)
    print("=" * 78)

    summary = []
    first_bad = None

    for name in names:
        frames = sorted(f for f in stages[name] if f < args.frames)
        role = STAGE_ROLE.get(name, "")
        stats = {"exact_frames": 0, "compared": 0, "shape_mm": 0,
                 "max_abs": 0, "max_ulp": 0, "mae_sum": 0.0, "mae_n": 0,
                 "n_diff_total": 0, "elem_total": 0}
        mismatch_frames = []

        for f in frames:
            ra = load(args.ref, name, f)
            tb = load(args.test, name, f)
            if ra is None or tb is None:
                continue
            a, _ = ra
            b, _ = tb
            stats["compared"] += 1
            r = compare(a, b, None)
            if r["class"] == "SHAPE_MISMATCH":
                stats["shape_mm"] += 1
                mismatch_frames.append((f, f"{r['a_shape']} vs {r['b_shape']}"))
                continue
            if r.get("exact"):
                stats["exact_frames"] += 1
            stats["max_abs"] = max(stats["max_abs"], r["max_abs"])
            if r["class"] == "FLOAT":
                stats["max_ulp"] = max(stats["max_ulp"], r["max_ulp"])
                if not np.isnan(r["mae"]):
                    stats["mae_sum"] += r["mae"]
                    stats["mae_n"] += 1
            else:
                stats["n_diff_total"] += r["n_diff"]
                stats["elem_total"] += r["total"]

        if stats["compared"] == 0:
            continue

        mae = stats["mae_sum"] / stats["mae_n"] if stats["mae_n"] else float("nan")
        all_exact = stats["exact_frames"] == stats["compared"]
        verdict = "IDENTICAL" if all_exact else "DIFFERS"
        if stats["shape_mm"]:
            verdict = "SHAPE-MISMATCH"

        if not all_exact and first_bad is None:
            first_bad = name

        if stats["elem_total"]:
            det = f"{stats['n_diff_total']}/{stats['elem_total']} elems"
        else:
            det = f"max {stats['max_ulp']} ulp"

        print(f"\n{name:<13} {role}")
        print(f"  frames compared : {stats['compared']}")
        print(f"  bit-exact frames: {stats['exact_frames']}/{stats['compared']}")
        if stats["shape_mm"]:
            print(f"  SHAPE MISMATCH in {stats['shape_mm']} frame(s); first: {mismatch_frames[0]}")
        if not all_exact:
            print(f"  max |diff|       : {stats['max_abs']:.6g}")
            if not np.isnan(mae):
                print(f"  mean |diff|      : {mae:.6g}")
            print(f"  detail           : {det}")

        summary.append((name, verdict, stats, mae))

    print("\n" + "=" * 78)
    print("VERDICT TABLE")
    print("=" * 78)
    print(f"{'stage':<14}{'verdict':<16}{'max|diff|':>12}{'max ulp':>10}{'mean|diff|':>13}")
    for name, verdict, st, mae in summary:
        m = "-" if np.isnan(mae) else f"{mae:.3g}"
        print(f"{name:<14}{verdict:<16}{st['max_abs']:>12.6g}{st['max_ulp']:>10}{m:>13}")

    # A ULP count is only meaningful next to the absolute difference. Near
    # zero the representable steps are tiny, so two values that differ by
    # 1e-05 in absolute terms can be billions of ULP apart. Reporting the ULP
    # column alone would make a harmless stage look catastrophic.
    float_rows = [(n, st) for n, _, st, _ in summary if st["max_ulp"] > 0]
    misleading = [(n, st["max_ulp"], st["max_abs"]) for n, st in float_rows
                  if st["max_abs"] < 1e-3]
    if misleading:
        print("\nNOTE -- ULP counts above are dominated by near-zero elements:")
        for n, u, a in misleading:
            print(f"  {n:<13} max|diff| {a:.3g} but max {u} ulp")
        print("  Judge these stages on max|diff|; the ULP figure reflects how")
        print("  finely representable numbers are spaced near zero, not a large")
        print("  error.")

    print("\n" + "=" * 78)
    ctrl = dict((n, v) for n, v, _, _ in summary).get("median_out")
    if ctrl == "IDENTICAL":
        print("CONTROL median_out: IDENTICAL -- the integer-only stage matches,")
        print("  so the pipelines fed it the same warped inputs and the")
        print("  comparison above is measuring real operator differences.")
    else:
        print(f"CONTROL median_out: {ctrl} -- upstream already diverged;")
        print("  treat downstream numbers as unexplained, not as median error.")

    if first_bad:
        print(f"\nFirst stage to diverge: {first_bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
