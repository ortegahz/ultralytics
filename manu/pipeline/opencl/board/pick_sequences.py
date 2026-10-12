#!/usr/bin/env python3
"""
pick_sequences.py -- choose the benchmark sequences by MEASURED texture, not by name.

WHY THIS EXISTS
  The project's own hard-won rule is that any speed ratio is uninterpretable
  without the texture level it was measured at: GMC fit time is dominated by
  Shi-Tomasi corner count and LK iteration count, and the same cache + threading
  work delivered 4.30x on a 15.7-corner fit and 1.37x on a 291.9-corner one.
  Picking "4 representative sequences" by folder name would silently bake that
  confound into the headline numbers.

  So: sample frames from every 龙泉山 sequence, run the SAME corner detector the
  pipeline runs (goodFeaturesToTrack on the SAME half-res resize, downscale=2),
  and report the distribution. The four chosen for the board run then span the
  measured range instead of guessing.

MEASUREMENT MATCHES THE PIPELINE
  gmc_stream.cpp resizes with INTER_AREA to half size before gFTT (downscale=2),
  so this script must too -- measuring on full-res corners would report a
  different quantity than the one that drives LK.
"""
import argparse
import os
import sys

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sample", type=int, default=12,
                    help="frames sampled per sequence (spread across the sequence)")
    ap.add_argument("--max-corners", type=int, default=400)
    ap.add_argument("--quality", type=float, default=0.01)
    ap.add_argument("--min-dist", type=int, default=8)
    args = ap.parse_args()

    seqs = sorted(d for d in os.listdir(args.root)
                  if os.path.isdir(os.path.join(args.root, d)))
    if not seqs:
        sys.exit(f"no sequence dirs under {args.root}")

    rows = []
    for s in seqs:
        sd = os.path.join(args.root, s)
        names = sorted(f for f in os.listdir(sd) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
        if not names:
            continue
        # spread the sample across the whole sequence, not just the head:
        # the head of a video is often the least textured part.
        step = max(1, len(names) // args.sample)
        picks = names[::step][:args.sample]
        counts, valid = [], []
        for nm in picks:
            g = cv2.imread(os.path.join(sd, nm), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            if valid and g.shape != valid[0]:
                sys.exit(f"[FATAL] {s}/{nm} geometry {g.shape} != {valid[0]}: "
                         "mixed resolutions would make the corpus unrunnable")
            if not valid:
                valid.append(g.shape)
            half = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2),
                              interpolation=cv2.INTER_AREA)
            c = cv2.goodFeaturesToTrack(half, maxCorners=args.max_corners,
                                        qualityLevel=args.quality, minDistance=args.min_dist,
                                        blockSize=3, useHarrisDetector=False)
            counts.append(0 if c is None else len(c))
        if counts:
            rows.append((s, float(np.mean(counts)), min(counts), max(counts), len(names), g.shape))

    rows.sort(key=lambda r: r[1])
    print(f"{'sequence':<34}{'mean_corners':>14}{'min':>7}{'max':>7}{'frames':>9}  geometry")
    print("-" * 88)
    for s, mean, lo, hi, nf, shape in rows:
        print(f"{s:<34}{mean:>14.1f}{lo:>7}{hi:>7}{nf:>9}  {shape[1]}x{shape[0]}")

    if rows:
        vals = np.array([r[1] for r in rows])
        print("-" * 88)
        print(f"corpus: n_seq={len(rows)} mean={vals.mean():.1f} min={vals.min():.1f} "
              f"max={vals.max():.1f} spread={vals.max() / max(vals.min(), 1e-9):.2f}x")


if __name__ == "__main__":
    main()