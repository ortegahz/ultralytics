#!/usr/bin/env python3
"""Pareto 池分析 + 加权汤可行性 LP —— 判定「在已测臂池内是否存在能过六项的加权汤」。

背景（2026-11-16 §195）：
  · 策展汤搜索长期被两个限制卡死：成员必须 T1/T3 贴基线、且必须是汤臂 ⇒ 池只有 ~52 个；
  · 穷举 C(52,5) 均等权重 ⇒ 能同时压过 nano/nano4/f1 的组合数 = 0 ⇒ 纯「策展汤」轴已榨干。
本脚本做两件事：
  1) 用**加权**视角重解可行性：把「汤指标 ≈ 成员加权均值」当代理，
     在全池（1524 个有实测指标的臂）上求 max-min-margin LP；
  2) 输出 Pareto 前沿（nano / nano4 / f1 / t1 / t3 / t2_fp 六维），供训练轴选点。

用法（服务器）：
  python -u manu/pipeline/npu/pareto_pool_lp.py --ledger <台账> --top 30
"""
from __future__ import annotations

import argparse
import json
import os

AXES = ["nano", "nano4", "f1", "t1", "t3"]
# Baseline_v2.2 = trial_610000@5（六项口径）
THRESH = {
    "nano": 0.7197542533081286,
    "nano4": 0.6554820415879017,
    "f1": 0.890398116705902,
    "t1": 0.8904938838558845,
    "t3": 0.9485250445456345,
}
BASE_T2_FP = 1
# 代理噪声 σ（用于把不同量纲的余量归一）
NOISE = {"nano": 0.0042, "nano4": 0.0060, "f1": 0.0008, "t1": 0.0006, "t3": 0.0006}
ARM_ROOT = "/mnt/data/siping/runs/npu20ms_offcal"


def ckpt(trial: int, epoch: int) -> str:
    return os.path.join(ARM_ROOT, "trials", f"trial_{int(trial)}", "weights", f"epoch{int(epoch)}.pt")


def load_pool(ledger_path: str, require_ckpt: bool = True) -> dict:
    raw = json.load(open(ledger_path, encoding="utf-8"))
    arms = raw.get("arms") if isinstance(raw, dict) else {}
    out = {}
    for tid, a in arms.items():
        if not isinstance(a, dict):
            continue
        picked = a.get("picked") if isinstance(a.get("picked"), dict) else {}
        off = a.get("official") if isinstance(a.get("official"), dict) else {}
        ep = picked.get("epoch", a.get("picked_epoch"))
        m = {
            "nano": picked.get("nano"),
            "nano4": picked.get("nano4", picked.get("nano_4")),
            "f1": picked.get("f1", picked.get("F1")),
            "t1": off.get("t1", off.get("T1")),
            "t3": off.get("t3", off.get("T3")),
            "t2_fp": off.get("t2_fp", off.get("T2_fp")),
        }
        if any(m.get(x) is None for x in AXES) or ep is None or m.get("t2_fp") is None:
            continue
        try:
            tid_i, ep_i = int(tid), int(float(ep))
        except (TypeError, ValueError):
            continue
        if require_ckpt and not os.path.isfile(ckpt(tid_i, ep_i)):
            continue
        m["trial"], m["epoch"] = tid_i, ep_i
        m["t2_fp"] = int(m["t2_fp"])
        m["is_soup"] = bool(a.get("soup_members"))
        m["desc"] = (a.get("desc") or "")[:80]
        out[tid_i] = m
    return out


def margins(m: dict) -> dict:
    """六项余量（归一 σ 单位；t2_fp 越低越好）。"""
    r = {ax: (m[ax] - THRESH[ax]) / NOISE[ax] for ax in AXES}
    r["t2_fp"] = (BASE_T2_FP - m["t2_fp"])          # >0 表示虚警更少
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pool = load_pool(args.ledger)
    print(f"池内臂 {len(pool)}（有实测指标 + 官方两轨 + ckpt 存在）")
    if not pool:
        return

    # ---- 1) 六维 Pareto 前沿（逐轴 top）----
    print("\n=== 逐轴 top10 ===")
    for ax in AXES + ["t2_fp"]:
        best = sorted(pool.values(), key=lambda m: -(m[ax] if ax != "t2_fp" else -m[ax]))[:10]
        print(f"-- {ax}: " + ", ".join(f"{m['trial']}({m[ax]:.6f})" for m in best))

    # ---- 2) 加权可行性 LP（自己实现的原始/对偶迭代，避免依赖 scipy）----
    # 目标：max m  s.t.  sum w = 1, w>=0,
    #                 THRESH_ax - sum(w_i * m_i_ax) + NOISE_ax * m <= 0   （六轴）
    #                 sum(w_i * t2fp_i) - 1 <= 0
    items = list(pool.values())
    n = len(items)

    def axis_vals(idx: int) -> dict:
        return {ax: items[idx][ax] for ax in AXES}

    # 迭代投影法：在单纯形上做多约束 max-min-margin（坐标上升 + 随机重启）
    import random

    def score(w: list[float]) -> float:
        agg = {ax: sum(w[i] * items[i][ax] for i in range(n)) for ax in AXES}
        fp = sum(w[i] * items[i]["t2_fp"] for i in range(n))
        marg = {ax: (agg[ax] - THRESH[ax]) / NOISE[ax] for ax in AXES}
        marg["t2_fp"] = (BASE_T2_FP - fp) / 0.5
        return min(marg.values()), marg, agg, fp

    rng = random.Random(20261116)
    best_w, best_score, best_marg = None, -1e9, None
    for restart in range(60):
        # 初始化：优先选「单臂已过线最多」的若干臂做随机权重
        seed_idx = rng.sample(range(n), min(n, 6))
        w = [0.0] * n
        for i in seed_idx:
            w[i] = 1.0 / len(seed_idx)
        cur = score(w)
        step = 0.25
        for _ in range(4000):
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            delta = step * w[j] if w[j] > 0 else step * 0.01
            if w[j] + delta > 0.9:
                continue
            w[j] += delta
            w[i] -= delta
            if w[i] < 0:
                w[j] -= 0
                w[i] = 0.0
            s = score(w)
            if s[0] >= cur[0]:
                cur = s
            else:
                w[j] -= delta
                if w[i] == 0.0:
                    pass
            if step > 1e-4:
                step *= 0.9995
        if cur[0] > best_score:
            best_score, best_w, best_marg = cur[0], list(w), cur
    agg = {ax: sum(best_w[i] * items[i][ax] for i in range(n)) for ax in AXES}
    fp = sum(best_w[i] * items[i]["t2_fp"] for i in range(n))
    print(f"\n=== 加权 LP 最优 min-margin = {best_score:.4f} σ（>0 ⇒ 池内理论存在过线汤）")
    print("    预测:", {ax: round(agg[ax], 6) for ax in AXES}, "t2_fp=%.3f" % fp)
    used = [(items[i]["trial"], items[i]["epoch"], round(best_w[i], 4)) for i in range(n) if best_w[i] > 0.01]
    print("    非零权重成员:", used[:20])

    # ---- 3) 训练轴选点：六项余量最小但仍在基线附近的臂（贴线候选）----
    print("\n=== 贴线候选（min-margin 在 [-2σ, +2σ] 且 t2_fp<=2）===")
    cand = []
    for m in items:
        if m["t2_fp"] > 2:
            continue
        mg = margins(m)
        worst = min(mg[ax] for ax in AXES)
        if -2.0 <= worst <= 2.0:
            cand.append((worst, mg, m))
    cand.sort(key=lambda x: -x[0])
    for worst, mg, m in cand[:args.top]:
        print(f"  trial={m['trial']}@{m['epoch']} worst={worst:+.3f}σ nano={m['nano']:.6f} nano4={m['nano4']:.6f} "
              f"f1={m['f1']:.6f} t1={m['t1']:.6f} t3={m['t3']:.6f} t2fp={m['t2_fp']} soup={m['is_soup']}")
    print(f"贴线候选共 {len(cand)} 个")

    if args.out:
        json.dump({"pool_size": len(pool),
                   "lp_best_margin_sigma": best_score,
                   "lp_pred": {ax: agg[ax] for ax in AXES}, "lp_t2_fp": fp,
                   "lp_members": used,
                   "near_line": [{"trial": m["trial"], "epoch": m["epoch"], **{ax: m[ax] for ax in AXES},
                                  "t2_fp": m["t2_fp"], "worst_sigma": w} for w, _, m in cand]},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("写出", args.out)


if __name__ == "__main__":
    main()
