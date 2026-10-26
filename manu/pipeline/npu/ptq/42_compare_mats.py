#!/usr/bin/env python3
# ===========================================================================
# 42_compare_mats.py -- frame-by-frame comparison of two GMC transform-chain
# dumps (the reference implementation's --mats-all format: a raw float32
# stream, 21 lags x 6 floats per frame, ascending lag order).
#
# This is the instrument that separates "the fit differs" from "the warp
# differs".  The transforms are the only thing the fit produces, so:
#   * mats identical  -> the fit is identical -> any feature difference is
#                        in the tail (warp / median / assembly)
#   * mats differ     -> the fit itself differs
#
# Usage:
#   python3 42_compare_mats.py --a REF.bin --b LIB.bin [--limit 60]
# ===========================================================================
import argparse

import numpy as np


def load(path, limit=None):
    raw = np.fromfile(path, dtype=np.float32)
    per = 21 * 6
    n = raw.size // per
    m = raw[: n * per].reshape(n, 21, 6)
    return m[:limit] if limit else m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--first", type=int, default=8,
                    help="how many frames to print in detail")
    args = ap.parse_args()

    A = load(args.a, args.limit)
    B = load(args.b, args.limit)
    n = min(A.shape[0], B.shape[0])
    A, B = A[:n], B[:n]
    print(f"frames: A={A.shape[0]} B={B.shape[0]} compared={n}")
    if n == 0:
        return 1

    exact = [i for i in range(n) if np.array_equal(A[i], B[i])]
    print(f"bit-identical frames: {len(exact)}/{n}")
    if exact and len(exact) < n:
        print(f"  first identical: {exact[:5]}   first differing: "
              f"{[i for i in range(n) if i not in set(exact)][:5]}")

    print("\nper-frame max |Δ| over all 21 lags x 6 floats:")
    for i in range(min(args.first, n)):
        d = np.abs(A[i] - B[i])
        print(f"  frame {i:3d}: max|d| = {d.max():.6g}   mean|d| = {d.mean():.3g}"
              f"   identical={np.array_equal(A[i], B[i])}")

    # Per-lag summary over all frames: which lag diverges first.
    print("\nper-lag max |Δ| over all compared frames:")
    for lag in range(A.shape[1]):
        d = np.abs(A[:, lag, :] - B[:, lag, :])
        print(f"  lag {lag * 2:3d}: max|d| = {d.max():.6g}  "
              f"frames with any Δ = {int((d.max(axis=1) > 0).sum()):3d}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
