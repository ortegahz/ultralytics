#!/usr/bin/env python3
"""重建策展集合池 + 分片（原子写）—— 为「策展汤」通道持续供给高质量成员组合。

🟢 2026-11-16 新增（服务器 /tmp 被重启清空后重建搜索空间供给链）：
  · 输入 = live 台账（nano_search_ledger.json）+ 当前基线（默认 Baseline_v2.2 = trial_610000@5）
  · 输出 = curated/k5_sets.json + curated/k5_s{0..7}.json（以及 k4 家族）
  · 成员组合 = 「按 ΠΦ(z) 排序的高分臂」在**同家族/全池**上的 k 组合（成员替换轴）
  · 打分口径沿用 make_family_sets.py：mean + SOUP_DELTA → pred → z → ΠΦ(z)
  · 取样沿用「按 mean-nano 三层分层」对冲「回归均值」风险
  · 写文件用 tmp + os.replace ⇒ 通道读到的永远是完整 JSON

用法（服务器）：
  python -u manu/pipeline/npu/refresh_curated_sets.py \
      --ledger /mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json \
      --out-dir /mnt/data/siping/runs/npu20ms/nano_search/curated --shards 8 --per-shard 40
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random

AXES = ["nano", "nano4", "f1", "t1", "t3"]
# 汤 − 成员均值 的系统偏移（§191 标定：nano 不可预测 ⇒ 偏移置 0）
SOUP_DELTA = {"t1": +0.000059, "t3": -0.000053, "f1": +0.000052, "nano4": -0.000589, "nano": 0.0}
# Baseline_v2.2 = trial_610000@5（六项口径）
THRESH = {
    "nano": 0.7197542533081286,
    "nano4": 0.6554820415879017,
    "f1": 0.890398116705902,
    "t1": 0.8904938838558845,
    "t3": 0.9485250445456345,
}
BASE_T2_FP = 1
ARM_ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
# 🟢🟢 2026-11-16 16:15 修正：打分必须用「汤 − 成员均值」的**真实噪声 σ**，
#   绝不能用「全池离散度」——池子里大部分是弱臂（t1 低 0.02~0.03），
#   用池 σ 会把「高 nano 但 t1 崩」的臂排到前面 ⇒ 汤的 nano 掉到 0.6586（实测）。
NOISE = {"nano": 0.0042, "nano4": 0.0060, "f1": 0.0008, "t1": 0.0006, "t3": 0.0006}
# 成员资格：官方两轨（T1/T3）必须贴近基线（汤 ≈ 成员均值，delta 极小 ⇒ 低 t1 成员会拖垮整汤）
MEMBER_T1_MIN = THRESH["t1"] - 0.0008
MEMBER_T3_MIN = THRESH["t3"] - 0.0008


def arm_ckpt(trial: int, epoch: int) -> str:
    return os.path.join(ARM_ROOT, "trials", f"trial_{int(trial)}", "weights", f"epoch{int(epoch)}.pt")


def load_arms(ledger_path: str, soup_only: bool = True) -> dict:
    raw = json.load(open(ledger_path, encoding="utf-8"))
    arms = raw.get("arms") if isinstance(raw, dict) else None
    if not isinstance(arms, dict):
        arms = raw if isinstance(raw, dict) else {}
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
        if any(m.get(x) is None for x in AXES) or ep is None:
            continue
        try:
            tid_i, ep_i = int(tid), int(float(ep))
        except (TypeError, ValueError):
            continue
        if not os.path.isfile(arm_ckpt(tid_i, ep_i)):
            continue
        m["trial"], m["epoch"] = tid_i, ep_i
        m["t2_fp"] = int(m["t2_fp"]) if m["t2_fp"] is not None else 99
        if m["t1"] < MEMBER_T1_MIN or m["t3"] < MEMBER_T3_MIN:
            continue          # 官方两轨不贴基线 ⇒ 不能当汤成员
        m["is_soup"] = bool(a.get("soup_members"))
        if soup_only and not m["is_soup"]:
            # 🟢🟢 2026-11-16 16:20 实测：掺入原始训练臂会把汤拉崩
            #   （成员含 14963@4 的新集合 nano=0.6692；纯汤臂集合 nano=0.7193）
            #   ⇒ 权重平均只在**同一凸盆**内有效 ⇒ 成员必须是汤臂。
            continue
        out[tid_i] = m
    return out


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--per-shard", type=int, default=40)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--core", type=int, default=12, help="核心成员臂个数（做全组合）")
    ap.add_argument("--mid", type=int, default=30, help="外围成员臂个数（做替换组合）")
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--raw-arms", action="store_true",
                    help="允许原始训练臂入池（默认只允许**汤臂**）")
    args = ap.parse_args()

    arms = load_arms(args.ledger, soup_only=not args.raw_arms)
    if len(arms) < args.k:
        raise SystemExit(f"可用臂不足（{len(arms)} < {args.k}）")
    def arm_score(m: dict) -> float:
        s = 1.0
        for ax in AXES:
            s *= phi((m[ax] - THRESH[ax]) / NOISE[ax])
        return s

    ranked = sorted(arms.values(), key=lambda m: -arm_score(m))
    core = ranked[: args.core]
    mid = ranked[args.core: args.core + args.mid]
    print(f"可用臂 {len(arms)}｜核心 {len(core)}｜外围 {len(mid)}")

    rng = random.Random(args.seed)
    combos: list[list[dict]] = []
    # ① 核心全组合（同 k 家族 ⇒ eq{k} 家族）
    for c in itertools.combinations(core, args.k):
        combos.append(list(c))
    # ② 成员替换：核心 k-1 + 外围 1 ⇒ 同盆地新成员 ⇒ 新家族（零训练成本的新搜索点）
    for c in itertools.combinations(core, args.k - 1):
        for m in mid:
            combos.append(list(c) + [m])
    # ③ 外围互搭（补多样性，抽样控制规模）
    if len(mid) >= args.k:
        for _ in range(min(400, len(mid) * 4)):
            combos.append(rng.sample(mid, args.k))

    seen, cands = set(), []
    for mem in combos:
        key = "|".join(f"{m['trial']}@{m['epoch']}" for m in sorted(mem, key=lambda x: x["trial"]))
        if key in seen:
            continue
        seen.add(key)
        mean = {ax: sum(m[ax] for m in mem) / len(mem) for ax in AXES}
        pred = {ax: mean[ax] + SOUP_DELTA[ax] for ax in AXES}
        z = {ax: (pred[ax] - THRESH[ax]) / NOISE[ax] for ax in AXES}
        score = 1.0
        for ax in AXES:
            score *= phi((pred[ax] - THRESH[ax]) / NOISE[ax])
        feasible = all(pred[ax] >= THRESH[ax] - NOISE[ax] for ax in AXES)
        cands.append({"members": [[m["trial"], m["epoch"]] for m in mem], "mean": mean,
                      "pred": pred, "score": score, "feasible": feasible})
    cands.sort(key=lambda c: (c["feasible"], c["score"]), reverse=True)
    print(f"候选 {len(cands)} 组｜均值过线 {sum(1 for c in cands if c['feasible'])} 组")

    pool = [c for c in cands if c["feasible"]] or cands
    by_nano = sorted(pool, key=lambda c: -c["mean"]["nano"])
    n = len(by_nano)
    per_layer = max(1, (args.per_shard * args.shards) // 3)
    picked = []
    for lo, hi in ((0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)):
        layer = sorted(by_nano[lo:hi], key=lambda c: -c["score"])[:per_layer]
        picked.extend(layer)
    picked = picked[: args.per_shard * args.shards]

    base = {"nano": THRESH["nano"], "nano4": THRESH["nano4"], "f1": THRESH["f1"],
            "t1": THRESH["t1"], "t2_fp": BASE_T2_FP, "t3": THRESH["t3"],
            "ref": "Baseline_v2.2(trial_610000@5)"}
    os.makedirs(args.out_dir, exist_ok=True)
    total = len(picked)
    for s in range(args.shards):
        part = picked[s * args.per_shard: (s + 1) * args.per_shard]
        sets = [{"members": c["members"], "tag": f"rc{args.k}{s:02d}{i:03d}",
                 "score": round(c["score"], 6), "pred": {ax: round(c["pred"][ax], 8) for ax in AXES},
                 "mean": {ax: round(c["mean"][ax], 8) for ax in AXES}, "mean_fp": 0}
                for i, c in enumerate(part)]
        payload = {"base": base, "sets": sets}
        fp = os.path.join(args.out_dir, f"k{args.k}_s{s}.json")
        tmp = fp + ".tmp"
        json.dump(payload, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        os.replace(tmp, fp)
    fp = os.path.join(args.out_dir, f"k{args.k}_sets.json")
    tmp = fp + ".tmp"
    json.dump({"base": base, "sets": [{"members": c["members"], "tag": f"rc{args.k}all{i:04d}",
                                       "score": round(c["score"], 6),
                                       "pred": {ax: round(c["pred"][ax], 8) for ax in AXES},
                                       "mean": {ax: round(c["mean"][ax], 8) for ax in AXES},
                                       "mean_fp": 0} for i, c in enumerate(picked)]},
              open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, fp)
    print(f"写出 {args.out_dir}/k{args.k}_sets.json：{total} 组 / {args.shards} 片（每片 {args.per_shard}）")
    if picked:
        c = picked[0]
        print("示例:", c["members"], f"score={c['score']:.4f} feasible={c['feasible']}",
              {ax: round(c["mean"][ax], 6) for ax in AXES})


if __name__ == "__main__":
    main()
