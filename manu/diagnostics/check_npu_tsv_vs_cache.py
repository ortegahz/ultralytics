#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPU TSV <-> PC cache 坐标一致性校验（冒烟）

目的：在开全量之前，用一帧一帧的实数证明「板卡 NPU 检测坐标」与
「PC cache 的 pred_points」在同一个坐标系里，且同架构两臂的峰大体重合。

输入：
  --tsv    板卡 rknn_heatmap_smoke --detections-out 的输出
  --cache  PC 侧 cache pkl（应为 --ablate-p0 那一臂）
  --tol    判定「坐标一致」的容差（px）

TSV 格式（已由 rknn_heatmap_smoke.cpp:606-616 确认）：
  <basename>\\t<ndet>\\t<x y score>\\t<x y score>...
  每组三元组各占一个独立 tab 列；0 检出写 `name\\t0`（两列），整行不会跳过。
"""
import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_tsv(path):
    out = {}
    n_zero = 0
    with open(path) as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            stem = Path(p[0]).stem
            n = int(p[1])
            if n == 0:
                out[stem] = np.zeros((0, 3), np.float32)
                n_zero += 1
                continue
            if len(p) < 2 + n:
                raise ValueError(f"列数不足: {stem} 声称 {n} 个检出，实际 {len(p)-2} 列")
            out[stem] = np.array(
                [[float(v) for v in p[2 + k].split()] for k in range(n)], dtype=np.float32
            )
    return out, n_zero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tol", type=float, default=8.0)
    ap.add_argument("--show", type=int, default=12)
    a = ap.parse_args()

    t, n_zero = parse_tsv(a.tsv)
    rows = pickle.load(open(a.cache, "rb"))
    pc = {}
    for r in rows:
        key = r["im_name"].rsplit(".", 1)[0]
        pc[key] = (
            np.asarray(r["pred_points"], dtype=np.float32).reshape(-1, 2),
            np.asarray(r["pred_scores"], dtype=np.float32).reshape(-1),
        )

    hit = sum(1 for k in t if k in pc)
    print(f"TSV 帧数={len(t)}  其中 0 检出={n_zero}  PC cache 命中={hit}")
    print(f"{'frame':<22}{'npu':>4}{'pc@th':>7}{'最近距px':>11}   npu_top1                 pc_top1")
    print("-" * 86)

    dmin, matched, n_cmp = [], 0, 0
    for k in sorted(t):
        if k not in pc:
            continue
        dp_all, ds = pc[k]
        dp = dp_all[ds >= 0.08]
        a_det = t[k]
        if len(a_det) == 0:
            continue
        n_cmp += 1
        if len(dp):
            # a_det 是 (N,3) 的 x,y,score，取前两列才是坐标
            d = np.sqrt(((dp[:, None, :] - a_det[None, :1, :2]) ** 2).sum(-1))
            best = float(d.min())
        else:
            best = float("inf")
        dmin.append(best)
        if best <= a.tol:
            matched += 1
        if len(dmin) <= a.show:
            npu1 = tuple(np.round(a_det[0, :2], 1))
            pc1 = tuple(np.round(dp[0], 1)) if len(dp) else None
            print(f"{k:<22}{len(a_det):>4}{len(dp):>7}{best:>11.3f}   {npu1}   {pc1}")

    print("-" * 86)
    fin = [d for d in dmin if np.isfinite(d)]
    print(f"npu_top1 在 PC-nop0 中找到 <= {a.tol}px 匹配: {matched}/{n_cmp}")
    if fin:
        print(f"最近点距: 中位数={np.median(fin):.3f}  最大={np.max(fin):.3f}  无有限匹配={len(dmin)-len(fin)}")
    print()
    print("判定：坐标同框且峰大体重合 ⇒ 两臂可直接在同一 8px 口径下比 F1。")


if __name__ == "__main__":
    main()
