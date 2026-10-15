#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARM vs PC 纯 GMC 对照评估（A/B 双臂配对评估）

设计目标
--------
唯一变量是「GMC 特征由谁生成」。两侧必须：
  * 同一批帧（按 im_name 严格一对一配对）
  * 同一份 GT（逐项校验一致）
  * 同一容器格式（两侧都必须是 PNG 或都必须是 BMP 等无损格式）
  * 同一模型权重 / 同一抽峰口径 / 同一匹配容差

自校准闸门
----------
--expect-f1 / --expect-tp / --expect-fp 用来校验本脚本自身。
先拿 SOTA 参照臂跑一遍，必须复现冻结 headline，否则后续对比不可信。

匹配算法
--------
按分数降序贪心：每个预测点找 tol 内最近的、尚未被匹配的 GT 点。
tol 为原生像素距离（冻结口径 8.0px）。
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# 冻结口径的 15 档门限（覆盖峰值 0.20~0.25）
DEFAULT_THS = [round(0.05 * i, 2) for i in range(1, 16)]


def parse_args():
    p = argparse.ArgumentParser(description="ARM vs PC pure-GMC A/B evaluation")
    p.add_argument("--arm-a", required=True, help="cache .pkl for arm A (e.g. PC SOTA)")
    p.add_argument("--label-a", default="A")
    p.add_argument("--arm-b", required=True, help="cache .pkl for arm B (e.g. ARM board)")
    p.add_argument("--label-b", default="B")
    p.add_argument("--tol", type=float, default=8.0, help="matching tolerance in native px")
    p.add_argument("--ths", default="", help="comma separated thresholds; default = 15-point 0.05..0.75")
    p.add_argument("--expect-f1", type=float, default=None, help="self-calibration gate")
    p.add_argument("--expect-tp", type=int, default=None)
    p.add_argument("--expect-fp", type=int, default=None)
    p.add_argument("--out-json", default="")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def load_cache(path):
    with open(path, "rb") as f:
        rows = pickle.load(f)
    out = {}
    for r in rows:
        out[r["im_name"]] = {
            "gt_pts": np.asarray(r.get("gt_pts", np.zeros((0, 2))), dtype=np.float64).reshape(-1, 2),
            "pred_points": np.asarray(r.get("pred_points", np.zeros((0, 2))), dtype=np.float64).reshape(-1, 2),
            "pred_scores": np.asarray(r.get("pred_scores", np.zeros((0,))), dtype=np.float64).reshape(-1),
        }
    return out


def gt_key(pts):
    """GT 逐项一致性校验用的稳定 key（四舍五入到 1e-4）"""
    if len(pts) == 0:
        return ()
    return tuple(tuple(np.round(p, 4)) for p in pts[np.lexsort((pts[:, 1], pts[:, 0]))])


def match_at(rec, th, tol):
    """按分数降序贪心匹配，返回 (tp, fp, fn)"""
    scores = rec["pred_scores"]
    if len(scores) == 0:
        return 0, 0, len(rec["gt_pts"])
    keep = np.where(scores >= th)[0]
    if len(keep) == 0:
        return 0, 0, len(rec["gt_pts"])
    keep = keep[np.argsort(-scores[keep], kind="stable")]

    gts = rec["gt_pts"]
    used = np.zeros(len(gts), dtype=bool)
    tp = fp = 0
    tol2 = tol * tol
    for i in keep:
        p = rec["pred_points"][i]
        if len(gts):
            d2 = ((gts - p) ** 2).sum(axis=1)
            d2[used] = np.inf
            j = int(np.argmin(d2))
            if d2[j] <= tol2:
                used[j] = True
                tp += 1
                continue
        fp += 1
    return tp, fp, int((~used).sum())


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def sweep(cache, names, ths, tol):
    rows = []
    for th in ths:
        tp = fp = fn = 0
        for n in names:
            a, b, c = match_at(cache[n], th, tol)
            tp += a
            fp += b
            fn += c
        p, r, f = prf(tp, fp, fn)
        rows.append({"th": th, "tp": tp, "fp": fp, "fn": fn,
                     "precision": p, "recall": r, "f1": f})
    return rows


def main():
    a = parse_args()
    ths = ([float(x) for x in a.ths.split(",")] if a.ths else DEFAULT_THS)

    A, B = load_cache(a.arm_a), load_cache(a.arm_b)
    nA, nB = len(A), len(B)
    names = sorted(set(A) & set(B))

    if not a.quiet:
        print("=" * 92)
        print(f"  ARM vs PC pure-GMC A/B   tol={a.tol}px   thresholds={len(ths)}")
        print(f"  {a.label_a}: {a.arm_a}  ({nA} frames)")
        print(f"  {a.label_b}: {a.arm_b}  ({nB} frames)")
        print("=" * 92)

    if len(names) != nA or len(names) != nB:
        print(f"[WARN] 配对不完全: 交集={len(names)}  {a.label_a}独有={nA-len(names)}  {a.label_b}独有={nB-len(names)}")
    else:
        print(f"[OK]   严格同集配对: {len(names)} 帧一一对应")

    # GT 逐项校验
    bad = [n for n in names if gt_key(A[n]["gt_pts"]) != gt_key(B[n]["gt_pts"])]
    n_gt = sum(len(A[n]["gt_pts"]) for n in names)
    if bad:
        print(f"[FAIL] GT 不一致 {len(bad)}/{len(names)} 帧，例如 {bad[:3]}")
    else:
        print(f"[OK]   GT 逐项一致: {len(names)} 帧, 合计 GT={n_gt}")

    ra = sweep(A, names, ths, a.tol)
    rb = sweep(B, names, ths, a.tol)

    hdr = f"{'th':>5} | {a.label_a+' TP':>9} {'FP':>7} {'P':>8} {'R':>8} {'F1':>9} | " \
          f"{a.label_b+' TP':>9} {'FP':>7} {'P':>8} {'R':>8} {'F1':>9} | {'dF1':>9} {'dP':>9} {'dR':>9}"
    print("\n" + hdr)
    print("-" * len(hdr))
    neg = 0
    for x, y in zip(ra, rb):
        df = y["f1"] - x["f1"]
        dp = y["precision"] - x["precision"]
        dr = y["recall"] - x["recall"]
        if df < 0:
            neg += 1
        print(f"{x['th']:>5.2f} | {x['tp']:>9} {x['fp']:>7} {x['precision']:>8.5f} {x['recall']:>8.5f} "
              f"{x['f1']:>9.6f} | {y['tp']:>9} {y['fp']:>7} {y['precision']:>8.5f} {y['recall']:>8.5f} "
              f"{y['f1']:>9.6f} | {df:>+9.6f} {dp:>+9.5f} {dr:>+9.5f}")

    ba = max(ra, key=lambda r: r["f1"])
    bb = max(rb, key=lambda r: r["f1"])
    dfs = [y["f1"] - x["f1"] for x, y in zip(ra, rb)]
    print("-" * len(hdr))
    print(f"[PEAK] {a.label_a}: th={ba['th']:.2f}  F1={ba['f1']:.6f}  P={ba['precision']*100:.2f}%  "
          f"R={ba['recall']*100:.2f}%  TP={ba['tp']}  FP={ba['fp']}")
    print(f"[PEAK] {a.label_b}: th={bb['th']:.2f}  F1={bb['f1']:.6f}  P={bb['precision']*100:.2f}%  "
          f"R={bb['recall']*100:.2f}%  TP={bb['tp']}  FP={bb['fp']}")
    print(f"[DELTA] 峰值 dF1 = {bb['f1']-ba['f1']:+.6f}")
    print(f"[DELTA] 跨 {len(ths)} 档均值 = {np.mean(dfs):+.6f}   "
          f"区间 [{min(dfs):+.6f}, {max(dfs):+.6f}]   负档 {neg}/{len(ths)}")

    # 自校准闸门
    if a.expect_f1 is not None:
        got = ba["f1"]
        ok = abs(got - a.expect_f1) < 5e-6
        tp_ok = a.expect_tp is None or ba["tp"] == a.expect_tp
        fp_ok = a.expect_fp is None or ba["fp"] == a.expect_fp
        print(f"[GATE]  {a.label_a} 期望 F1={a.expect_f1:.6f} TP={a.expect_tp} FP={a.expect_fp} | "
              f"实测 F1={got:.6f} TP={ba['tp']} FP={ba['fp']}  -> "
              f"{'PASS' if (ok and tp_ok and fp_ok) else 'FAIL'}")
        if not (ok and tp_ok and fp_ok):
            print("[GATE]  未复现冻结 headline，后续对比不可信 —— 中止")
            sys.exit(2)

    if a.out_json:
        rep = {
            "tol": a.tol, "ths": ths, "frames": len(names), "gt": n_gt,
            "arm_a": {"label": a.label_a, "path": a.arm_a, "rows": ra, "peak": ba},
            "arm_b": {"label": a.label_b, "path": a.arm_b, "rows": rb, "peak": bb},
            "delta": {
                "peak_f1": bb["f1"] - ba["f1"],
                "mean_f1": float(np.mean(dfs)),
                "min_f1": float(min(dfs)),
                "max_f1": float(max(dfs)),
                "negative_bands": int(neg),
                "bands": len(ths),
            },
        }
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[WROTE] {a.out_json}")


if __name__ == "__main__":
    main()
