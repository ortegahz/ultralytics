#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARM vs PC 双臂逐像素比对

用途
----
1. 验证索引映射 <seq>_<k>.bmp -> <seq>__<k+3>.bmp 是否正确。
   若偏移错 1，帧间运动会让像素差远大于「同帧跨架构浮点末位舍入」的量级（Max|D| 个位数）。
2. 量化 C++ -> aarch64/Mali 移植引入的残差（Max|D| / MAE / 分通道）。

两侧必须同容器格式（都 BMP 或都 PNG），否则差异里混入了压缩误差。
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="ARM vs PC feature pixel diff")
    p.add_argument("--arm", required=True, help="ARM dataset images/val dir")
    p.add_argument("--pc", required=True, help="PC dataset images/val dir")
    p.add_argument("--samples", type=int, default=300, help="frames to sample (0 = all)")
    p.add_argument("--per-seq", type=int, default=40, help="max samples per sequence when sampling")
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--warn-ma", type=float, default=1.0, help="MAE above this = suspicious")
    p.add_argument("--out-json", default="")
    return p.parse_args()


def list_names(d):
    return sorted(p.name for p in Path(d).iterdir()
                  if p.is_file() and p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"})


def main():
    a = parse_args()
    na, nb = set(list_names(a.arm)), set(list_names(a.pc))
    common = sorted(na & nb)
    print("=" * 88)
    print(f"  ARM : {a.arm}  ({len(na)} files)")
    print(f"  PC  : {a.pc}  ({len(nb)} files)")
    print(f"  交集: {len(common)}   仅ARM: {len(na-nb)}   仅PC: {len(nb-na)}")
    if not common:
        print("[FAIL] 无交集")
        return 2
    print("=" * 88)

    by_seq = defaultdict(list)
    for n in common:
        by_seq[n.split("__")[0]].append(n)

    if a.samples and a.samples < len(common):
        random.seed(a.seed)
        picked = []
        for s in sorted(by_seq):
            pool = by_seq[s]
            k = min(a.per_seq, len(pool))
            picked += random.sample(pool, k)
        picked = sorted(picked)
    else:
        picked = common

    print(f"[比对] {len(picked)} 帧（{len(by_seq)} 条序列）\n")
    print(f"{'seq':<26} {'n':>5} {'Max|D|':>8} {'MAE':>10} {'p99|D|':>8} {'>5比例':>9}")
    print("-" * 72)

    rows = []
    global_max = 0
    for s in sorted(by_seq):
        names = [n for n in picked if n.split("__")[0] == s]
        if not names:
            continue
        mx = 0
        acc = 0.0
        cnt = 0
        over5 = 0
        allv = []
        for n in names:
            ia = cv2.imread(str(Path(a.arm) / n), cv2.IMREAD_COLOR)
            ib = cv2.imread(str(Path(a.pc) / n), cv2.IMREAD_COLOR)
            if ia is None or ib is None:
                print(f"[WARN] 读取失败 {n}")
                continue
            if ia.shape != ib.shape:
                print(f"[FAIL] 形状不一致 {n}: {ia.shape} vs {ib.shape}")
                return 3
            d = cv2.absdiff(ia, ib)
            mx = max(mx, int(d.max()))
            acc += float(d.sum())
            cnt += d.size
            over5 += int((d > 5).sum())
            allv.append(d.reshape(-1))
        if cnt == 0:
            continue
        mae = acc / cnt
        v = np.concatenate(allv)
        p99 = float(np.percentile(v, 99))
        frac = over5 / v.size
        global_max = max(global_max, mx)
        flag = "  <-- 可疑" if mae > a.warn_ma else ""
        print(f"{s:<26} {len(names):>5} {mx:>8} {mae:>10.6f} {p99:>8.1f} {frac*100:>8.3f}%{flag}")
        rows.append({"seq": s, "n": len(names), "max_abs": mx, "mae": mae,
                     "p99": p99, "frac_gt5": frac})

    print("-" * 72)
    tot = sum(r["n"] for r in rows)
    wmae = sum(r["mae"] * r["n"] for r in rows) / max(tot, 1)
    print(f"{'合计':<26} {tot:>5} {global_max:>8} {wmae:>10.6f}")

    # 判定：以 MAE 与离群比例为主，Max|D| 只作参考。
    # 高运动序列上，21 帧中值的选取会被 warp 的末位浮点差异放大：
    # 极少数像素发生「中值换帧」，单点差可达数百，但整体 MAE 仍在 1e-3 量级。
    # 因此不能用 Max|D| 单一指标判死刑，否则会把正确映射误报成错误。
    frac5 = sum(r["frac_gt5"] * r["n"] for r in rows) / max(tot, 1)
    print(f"\n[判定] 加权 MAE = {wmae:.6f}   >5 像素占比 = {frac5*100:.4f}%   全局 Max|D| = {global_max}")
    if wmae < 0.01 and frac5 < 0.005:
        print("       -> 索引映射正确；差异属跨架构浮点末位舍入（中值换帧造成的孤立大偏差）")
        verdict = "PASS"
    elif wmae < 0.5 and frac5 < 0.05:
        print("       -> 基本正确，但离群比例偏高，建议做偏移扫描复核")
        verdict = "WARN"
    else:
        print("       -> 偏差过大，索引映射很可能有误（建议用偏移扫描复核）")
        verdict = "FAIL"

    if a.out_json:
        import json
        Path(a.out_json).write_text(json.dumps(
            {"frames": tot, "global_max_abs": global_max, "weighted_mae": wmae,
             "frac_gt5": frac5, "verdict": verdict, "per_seq": rows},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[WROTE] {a.out_json}")
    return 0 if verdict != "FAIL" else 4


if __name__ == "__main__":
    raise SystemExit(main())
