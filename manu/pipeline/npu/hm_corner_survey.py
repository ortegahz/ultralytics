#!/usr/bin/env python3
"""
hm_corner_survey.py -- measure Shi-Tomasi corner counts across the whole val set
BEFORE committing to a full-length board run.

Why this must be measured rather than assumed
----------------------------------------------
`--fit-corners` is not a free speed knob. It is a ``goodFeaturesToTrack``
parameter, so changing it changes the fitted transform, which changes the
warped frames, which changes the emitted features. A run with ``--fit-corners
300`` therefore CANNOT be compared against a frozen feature set built at 600 --
the comparison would measure "clamp delta + quantisation delta" with no way to
separate them.

So the question "should the full run clamp corners?" has to be answered with the
corner-count distribution first. If the counts stay well under 600, no clamp is
needed and the full run can use the frozen configuration, keeping the accuracy
reconciliation clean.

The survey reuses the shipped estimator (``FastGMCEstimator``), so the counts
are the same ones the C++/Mali path would see at the same downscale.

Usage:
    python hm_corner_survey.py                       # all val sequences
    python hm_corner_survey.py --frames 20 --seqs 34_1 DJI_0051_2
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "data"))
from build_sample_median_dataset import FastGMCEstimator  # noqa: E402

RAW = "/home/manu/mnt/data/siping/datasets/manu/anti-uav/train"
FROZEN = "/home/manu/mnt/data/siping/datasets/manu/uav_gmc_median/images/val"


def natural(p):
    return int("".join(c for c in Path(p).stem if c.isdigit()) or 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=RAW)
    ap.add_argument("--frozen-root", default=FROZEN)
    ap.add_argument("--frames", type=int, default=16, help="frames per sequence to probe")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--max-corners", type=int, default=600)
    args = ap.parse_args()

    seqs = args.seqs or sorted({f.split("__")[0]
                                for f in os.listdir(args.frozen_root) if "__" in f})
    print(f"[survey] {len(seqs)} sequences, up to {args.frames} frames each, "
          f"maxCorners={args.max_corners}\n")
    hdr = f"| {'序列':26} | {'n':>3} | {'mean':>7} | {'p95':>5} | {'max':>5} | {'=cap':>5} | {'>400':>5} |"
    print(hdr)
    print("|" + "-" * (len(hdr) - 2) + "|")

    est = FastGMCEstimator(downscale=2)
    allc: list[int] = []
    worst = []
    for seq in seqs:
        d = Path(args.raw_root) / seq
        if not d.is_dir():
            print(f"| {seq:26} |  -- 无原始目录 -- |")
            continue
        fs = sorted((p for p in os.listdir(d)
                     if p.lower().endswith((".png", ".jpg", ".bmp", ".tif"))), key=natural)
        fs = fs[: args.frames]
        counts = []
        prev = None
        for f in fs:
            im = cv2.imread(str(d / f), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            if prev is not None:
                small = cv2.resize(prev, (im.shape[1] // 2, im.shape[0] // 2))
                pts = cv2.goodFeaturesToTrack(small, mask=None, **est.feature_params)
                counts.append(0 if pts is None else len(pts))
            prev = im
        if not counts:
            print(f"| {seq:26} | {0:3d} | 无有效帧 |")
            continue
        a = np.array(counts)
        allc.extend(counts)
        capped = int((a >= args.max_corners).sum())
        hi = int((a > 400).sum())
        worst.append((seq, int(a.max()), int(a.mean())))
        print(f"| {seq:26} | {len(counts):3d} | {a.mean():7.1f} | "
              f"{np.percentile(a, 95):5.0f} | {a.max():5d} | {capped:5d} | {hi:5d} |")

    if allc:
        a = np.array(allc)
        cap = int((a >= args.max_corners).sum())
        print(f"\n[survey] 全体角点数: n={len(a)}  mean={a.mean():.1f}  "
              f"p50={np.percentile(a,50):.0f}  p95={np.percentile(a,95):.0f}  "
              f"p99={np.percentile(a,99):.0f}  max={a.max()}")
        print(f"[survey] 触顶(=600)帧数 {cap}/{len(a)}   >400 帧数 {int((a>400).sum())}")
        worst.sort(key=lambda t: -t[1])
        print(f"[survey] 角点最多的 3 个序列: " +
              ", ".join(f"{s}(max={m}, mean={round(av,1)})" for s, m, av in worst[:3]))
        print("\n[survey] 判决："
              + ("角点分布远离 600 上限，**无需 clamp**；全量 run 可直接用冻结配置 600，"
                 "从而保持 L1 特征对齐干净。"
                 if cap == 0 else
                 f"有 {cap} 帧触顶，若 clamp 会同时改变 GMC 拟合与特征，"
                 f"精度对账须另设对照臂，不能与冻结集直接比。"))


if __name__ == "__main__":
    main()
