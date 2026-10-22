#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""混合比例搜索结果的**统计对账**（回答"哪个比例真的更好"+"到底有没有复现 RC1"）。

为什么要有这个脚本
------------------
单看 U 的极值会被噪声骗：同一档不同 trial 的 U 抖动 ±0.009，
而**同档中位数**才是该比例的真实水平。本脚本给出：
  1. 每档的 n / 中位数 / IQR / 极值（**U 一律重算**，不信 CSV 列）
  2. 区分「RC1 原值超参」的 trial 与「TPE 自选超参」的 trial
     （前者才是复现对照；后者是"比例 + 超参"的联合效果）
  3. **复现判据**：RC1 原值超参 + natural 档 ⇒ 与 RC1/Ep3 的差是否落在该档自身离散度内
  4. **效应量**：最佳档 vs natural 基线（同协议、同超参口径）的差 + bootstrap 置信区间
  5. 帕累托/红线视角下的"能不能上线"结论

用法（服务器）：
```bash
python manu/training/analyze_mix_spec_results.py
python manu/training/analyze_mix_spec_results.py --boot 5000
```
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manu.training.mix_ratio_judgement import (  # noqa: E402
    FROZEN_BASELINE,
    RC1_EPOCH3,
    hard_gate,
    reproduction_check,
    utility,
)

OUT_ROOT_DEFAULT = "runs/optuna_mix_ratio"

# 路线 B（RC1）的原始超参 —— 用来把"复现对照"与"TPE 自选超参"分开
RC1_HP = {
    "lr_scale": 1.0, "warmup_epochs": 0.5, "weight_decay": 1e-4,
    "focal_beta": 2.7, "offset_weight": 0.55, "mosaic": 0.05,
    "scale": 0.20, "translate": 0.10, "clip_max_norm": 5.0, "freeze_bn": 0.0,
}
HP_COLS = {
    "lr_scale": 1e-6, "warmup_epochs": 1e-3, "weight_decay": 1e-9,
    "focal_beta": 1e-6, "offset_weight": 1e-6, "mosaic": 1e-6,
    "scale": 1e-6, "translate": 1e-6, "clip_max_norm": 1e-6, "freeze_bn": 1e-6,
}


def load_rows(out_root: Path) -> list[dict]:
    p = out_root / "optuna_summary.csv"
    if not p.is_file():
        return []
    with p.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def f(r: dict, k: str, default=0.0) -> float:
    try:
        return float(r.get(k) or default)
    except Exception:
        return default


def i(r: dict, k: str, default=0) -> int:
    try:
        return int(float(r.get(k) or default))
    except Exception:
        return default


def metrics_of(r: dict) -> dict | None:
    if not r.get("t1_f1") or not r.get("t2_f1") or not r.get("t3_f1"):
        return None
    return {
        "track1_f1": f(r, "t1_f1"), "track1_fp": i(r, "t1_fp"),
        "track2_f1": f(r, "t2_f1"), "track2_fp": i(r, "t2_fp"),
        "track3_f1": f(r, "t3_f1"), "track3_fp": i(r, "t3_fp"),
    }


def is_rc1_hp(r: dict) -> bool:
    """该 trial 的超参是否与 RC1 完全一致（= 合格的复现对照）。"""
    for k, v in RC1_HP.items():
        if abs(f(r, k) - v) > HP_COLS[k]:
            return False
    return True


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def quartiles(xs):
    s = sorted(xs)
    n = len(s)
    if n < 4:
        return (float("nan"), float("nan"))
    return (s[n // 4], s[3 * n // 4])


def boot_ci(xs, stat=statistics.median, n_boot=4000, seed=20261022):
    """中位数的 bootstrap 百分位区间。"""
    if len(xs) < 3:
        return (float("nan"), float("nan"))
    rnd = random.Random(seed)
    n = len(xs)
    vals = []
    for _ in range(n_boot):
        vals.append(stat([xs[rnd.randrange(n)] for _ in range(n)]))
    vals.sort()
    return (vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)])


def main() -> int:
    ap = argparse.ArgumentParser(description="混合比例搜索结果统计对账")
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--epoch", type=int, default=3, help="对照的 RC1 测量 epoch")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    rows = load_rows(out_root)
    if not rows:
        print(f"[FATAL] 读不到 {out_root / 'optuna_summary.csv'}")
        return 1

    u_rc1 = utility(RC1_EPOCH3, FROZEN_BASELINE)[0]
    items = []
    for r in rows:
        m = metrics_of(r)
        if m is None:
            continue
        u = utility(m, FROZEN_BASELINE)[0]
        ok, reason, _ = hard_gate(m, FROZEN_BASELINE)
        items.append({
            "trial": r["trial"], "mix": r.get("mix_spec", "?"), "m": m, "u": u,
            "rc1_hp": is_rc1_hp(r), "gate_ok": ok, "gate_reason": reason,
            "measured": f(r, "measured_ratio"),
        })

    print("=" * 104)
    print(f"混合比例搜索 · 统计对账    RC1/Ep{args.epoch} 的 U = {u_rc1:+.6f}（及格线）")
    print(f"样本 {len(items)} 个有指标的 trial；U 一律重算（不信 CSV 列）")
    print("=" * 104)

    # ---------- 1. 分档统计 ----------
    by_mix: dict[str, list[dict]] = {}
    for it in items:
        by_mix.setdefault(it["mix"], []).append(it)

    print(f"\n【1】按混合比例分组（全部 trial，含 TPE 自选超参）")
    print(f"{'mix':<10}{'n':<4}{'U中位数':<12}{'95%CI(中位)':<22}{'U最好':<12}{'U最差':<12}"
          f"{'中位-擂主':<12}{'RC1超参n':<10}")
    for m in sorted(by_mix, key=lambda k: -med([x["u"] for x in by_mix[k]])):
        us = [x["u"] for x in by_mix[m]]
        lo, hi = boot_ci(us, n_boot=args.boot)
        n_rc1 = sum(1 for x in by_mix[m] if x["rc1_hp"])
        ci = f"[{lo:+.6f}, {hi:+.6f}]" if lo == lo else "(样本不足)"
        print(f"{m:<10}{len(us):<4}{med(us):<+12.6f}{ci:<22}{max(us):<+12.6f}{min(us):<+12.6f}"
              f"{med(us)-u_rc1:<+12.6f}{n_rc1:<10}")

    # ---------- 2. 只看 RC1 原值超参（干净的"比例效应"口径）----------
    rc1hp = [x for x in items if x["rc1_hp"]]
    if rc1hp:
        print(f"\n【2】只保留 **RC1 原值超参**的 trial（n={len(rc1hp)}）—— 比例是唯一变量")
        print(f"{'mix':<10}{'n':<4}{'U中位数':<12}{'U最好':<12}{'中位-擂主':<12}")
        g2: dict[str, list[dict]] = {}
        for x in rc1hp:
            g2.setdefault(x["mix"], []).append(x)
        for m in sorted(g2, key=lambda k: -med([y["u"] for y in g2[k]])):
            us = [y["u"] for y in g2[m]]
            print(f"{m:<10}{len(us):<4}{med(us):<+12.6f}{max(us):<+12.6f}{med(us)-u_rc1:<+12.6f}")

    # ---------- 3. 复现判据 ----------
    print(f"\n【3】复现判据：RC1 原值超参 + natural 档 ⇒ 与 RC1/Ep{args.epoch} 比")
    ctrl = [x for x in rc1hp if x["mix"] == "natural"]
    if not ctrl:
        print("    （没有 natural + RC1 原值超参的 trial）")
    for x in ctrl:
        rep = reproduction_check(x["m"], args.epoch)
        print(f"    {x['trial']}: {rep['verdict']}  U={x['u']:+.6f}（差 {x['u']-u_rc1:+.6f}）"
              f"  超容差={rep['outliers']}  出包络={rep['band_outliers']}")
    if len(ctrl) >= 2:
        us = [x["u"] for x in ctrl]
        spread = max(us) - min(us)
        gap = med(us) - u_rc1
        print(f"    ⇒ 该组 U 中位数 {med(us):+.6f}，离散度(极差) {spread:.6f}，与 RC1 的差 {gap:+.6f}"
              f" ⇒ 差/离散度 = {abs(gap)/spread if spread else float('nan'):.2f}"
              f"（<1 ⇒ 落在自身离散度内，属随机波动）")

    # ---------- 4. 效应量：最佳档 vs natural 基线 ----------
    print(f"\n【4】效应量：**比例效应** = 某档中位数 − natural 档中位数（同协议、同超参口径）")
    nat = [x["u"] for x in items if x["mix"] == "natural"]
    if nat:
        base = med(nat)
        print(f"    natural 档（n={len(nat)}）中位数 = {base:+.6f}")
        for m in sorted(by_mix, key=lambda k: -med([x["u"] for x in by_mix[k]])):
            if m == "natural":
                continue
            us = [x["u"] for x in by_mix[m]]
            d = med(us) - base
            lo, hi = boot_ci(us, n_boot=args.boot)
            sig = ""
            if lo == lo and hi == hi and lo > base:
                sig = "  ← 95%CI 下界仍高于 natural 中位数 ⇒ 显著"
            print(f"    {m:<10} n={len(us):<4} U中位数={med(us):+.6f}  Δ vs natural={d:+.6f}"
                  f"  CI=[{lo:+.6f}, {hi:+.6f}]{sig}")

    # ---------- 5. 上线视角 ----------
    print(f"\n【5】上线视角：最佳 trial 相对 RC1/Ep{args.epoch} 的三轨分解")
    best = max(items, key=lambda x: x["u"])
    bm = best["m"]
    rm = RC1_EPOCH3
    print(f"    最佳：{best['trial']}（mix={best['mix']} 实测1:{best['measured']:.2f}）U={best['u']:+.6f}"
          f"  差={best['u']-u_rc1:+.6f}  红线={best['gate_reason']}")
    for k, label in (("track1_f1", "T1 F1"), ("track1_fp", "T1 FP"),
                     ("track2_f1", "T2 F1"), ("track2_fp", "T2 FP"),
                     ("track3_f1", "T3 F1"), ("track3_fp", "T3 FP")):
        d = bm[k] - rm[k]
        rel = "" if k.endswith("_fp") else ""
        print(f"      {label:<7} 候选={bm[k]:<12.6f} RC1={rm[k]:<12.6f} Δ={d:+.6f}{rel}")
    print(f"    ⇒ U 构成：ΔF1_T3 {bm['track3_f1']-rm['track3_f1']:+.6f}"
          f" + 0.5·ΔF1_T2 {0.5*(bm['track2_f1']-rm['track2_f1']):+.6f}"
          f" − 2.5·max(0,−ΔF1_T1) {max(0.0, -(bm['track1_f1']-rm['track1_f1']))*2.5:+.6f}"
          f" − 0.0005·ΔFP_T2 {max(0.0, bm['track2_fp']-rm['track2_fp'])*0.0005:+.6f}")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
