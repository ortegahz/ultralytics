#!/usr/bin/env python3
# ===========================================================================
# 41_compare_feature_dirs.py -- per-pixel comparison of two GMC feature DUMPS.
#
# The MD5 stream cannot answer "how far apart are these two feature sets":
# one flipped bit changes the whole hash, so a 0.1%-of-pixels difference by
# ONE grey level looks exactly like a catastrophic one.  This script answers
# the question that matters: per channel, what is the mean absolute error,
# the worst pixel, and what fraction of pixels differ by more than 2 levels.
#
# Both directories must hold the SAME file naming convention:
#   <seq>_%06d.bmp   with pixel order [Ch0 I_t, Ch1 diff, Ch2 residual]
# which is what the reference implementation's --out-dir writes and what the
# library's mosaic_pipeline_set_feature_dump_out writes.
#
# Usage:
#   python3 41_compare_feature_dirs.py --a DIR_A --b DIR_B [--limit 60] \
#       [--seq VIDEO00007_19700101_001324]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference feature dir")
    ap.add_argument("--b", required=True, help="feature dir under test")
    ap.add_argument("--seq", default="VIDEO00007_19700101_001324")
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()

    pairs = []
    for i in range(args.limit):
        pa = os.path.join(args.a, f"{args.seq}_{i:06d}.bmp")
        pb = os.path.join(args.b, f"{args.seq}_{i:06d}.bmp")
        ia, ib = read(pa), read(pb)
        if ia is None or ib is None:
            continue
        pairs.append((i, ia, ib))
    if not pairs:
        print("no comparable frames")
        return 1

    # acc[c] = [sum|d|, max|d|, count(|d|>2)]
    acc = np.zeros((3, 3), dtype=np.float64)
    exact_frames = np.zeros(3, dtype=np.int64)
    first_diff = [-1, -1, -1]
    for idx, ia, ib in pairs:
        for c in range(3):
            d = np.abs(ia[:, :, c].astype(np.int16) - ib[:, :, c].astype(np.int16))
            acc[c, 0] += float(d.sum())
            acc[c, 1] = max(acc[c, 1], float(d.max()))
            acc[c, 2] += float((d > 2).sum())
            if d.max() == 0:
                exact_frames[c] += 1
            elif first_diff[c] < 0:
                first_diff[c] = idx

    n = len(pairs)
    print(f"frames compared: {n} (of {args.limit})")
    print(f"A = {args.a}")
    print(f"B = {args.b}")
    print(f"\n{'channel':<16}{'MAE':>10}{'max|d|':>10}{'frac(|d|>2)':>16}{'bit-exact frames':>20}")
    for c in range(3):
        print(f"{CH_NAMES[c]:<16}{acc[c, 0] / n / (512 * 640):>10.5f}{int(acc[c, 1]):>10d}"
              f"{acc[c, 2] / n / (512 * 640) * 100:>15.4f}%{exact_frames[c]:>14d}/{n}")
    print("\nfirst frame with a difference, per channel:",
          {CH_NAMES[c]: first_diff[c] for c in range(3)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
