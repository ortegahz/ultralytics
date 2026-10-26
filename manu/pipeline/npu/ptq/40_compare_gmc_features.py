#!/usr/bin/env python3
# ===========================================================================
# compare_gmc_features.py -- per-pixel comparison of three feature sources on
# the BOARD, for one Longquan sequence.
#
#   official : /mnt/manu/ptq/bmp/<SEQ>/<SEQ>__%06d.bmp
#              the frozen official feature dataset (losslessly re-encoded from
#              the server-side JPEG), built with the PER-LAG DIRECT FIT
#              semantics (build_full_median_dataset.py)
#   s2       : the on-board C++ GMC engine with --anchor-step 2
#              (anchors = every lag => per-lag direct fit) -> should match
#              the official semantics
#   s10      : the same engine with --anchor-step 10
#              (anchors {2,12,22,32,42} + bounded chain composition) -> the
#              "tree" semantics the e2e demo ships with
#
# WHAT IT ANSWERS
#   * s2 vs official  : is the ON-BOARD engine's per-lag semantics the same
#                       thing as the frozen baseline?  (the residual is the
#                       JPEG quantisation of the official dataset)
#   * s10 vs official : how far the "tree" semantics actually drifts
#   * s10 vs s2      : the pure semantic gap, with no codec in the way
#
# Usage:
#   python3 compare_gmc_features.py --seq <SEQ> --limit 60 \
#       --official /mnt/manu/ptq/bmp --s2 /mnt/manu/mosaic_out/gmcfeat_s2 \
#       --s10 /mnt/manu/mosaic_out/gmcfeat_s10
# ===========================================================================
import argparse
import os

import cv2
import numpy as np

CH_NAMES = ("Ch0 I_t", "Ch1 diff", "Ch2 residual")


def read(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None
    return im


def stats(a, b):
    """(MAE, max|diff|, frac(|diff|>2) per channel)."""
    out = []
    for c in range(3):
        d = np.abs(a[:, :, c].astype(np.int16) - b[:, :, c].astype(np.int16))
        out.append((float(d.mean()), int(d.max()), float((d > 2).mean())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--official", default="/mnt/manu/ptq/bmp")
    ap.add_argument("--s2", default="/mnt/manu/mosaic_out/gmcfeat_s2")
    ap.add_argument("--s10", default="/mnt/manu/mosaic_out/gmcfeat_s10")
    args = ap.parse_args()

    off_dir = os.path.join(args.official, args.seq)
    pairs = []
    for i in range(args.limit):
        off = read(os.path.join(off_dir, f"{args.seq}__{i:06d}.bmp"))
        s2 = read(os.path.join(args.s2, f"{args.seq}_{i:06d}.bmp"))
        s10 = read(os.path.join(args.s10, f"{args.seq}_{i:06d}.bmp"))
        if off is None or s2 is None or s10 is None:
            continue
        pairs.append((i, off, s2, s10))
    if not pairs:
        print("no comparable frames")
        return
    print(f"frames compared: {len(pairs)} (of {args.limit})")

    for label, ia, ib in (("s2  vs official", 2, 1), ("s10 vs official", 2, 3), ("s10 vs s2", 3, 2)):
        acc = np.zeros((3, 3), dtype=np.float64)
        for p in pairs:
            st = stats(p[ia], p[ib])
            for c in range(3):
                acc[c, 0] += st[c][0]
                acc[c, 1] = max(acc[c, 1], st[c][1])
                acc[c, 2] += st[c][2]
        n = len(pairs)
        print(f"\n=== {label} ===")
        print(f"{'channel':<16}{'MAE':>10}{'max|d|':>10}{'frac(|d|>2)':>16}")
        for c in range(3):
            print(f"{CH_NAMES[c]:<16}{acc[c, 0] / n:>10.4f}{int(acc[c, 1]):>10d}"
                  f"{acc[c, 2] / n * 100:>15.3f}%")


if __name__ == "__main__":
    main()
