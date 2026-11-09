#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成「同配方多种子」的补投臂计划（让 Model Soup 的交接口径成立）—— 2026-11-06。

为什么
----
基线 `cs_t0043` 本身就是**同配方 3 粒子 soup**（trial_0043/0049/0050）。
而项目自己的结论（memory.md §37 / add_seed_arms.py 的开篇）是：

  · 种子噪声带：同配方换种子 |ΔT1| 中位 **0.00246**、|ΔT3| 中位 **0.00300**；
  · **「交接口径（同超参多种子中位）是唯一能给出可信结论的办法」**，
    n=2 的中位仍然很脆 ⇒ **n=3 才可信**；
  · ⇒ 「给最近点各补一个种子（n=3）比再开新配置更值钱」。

而本轮（`mix_ratio_search_npu20ms_v2`）的 36 条臂里**只有 1 个配方有多种子**
（两条 p1 对照臂）⇒ **soup 口径无法成立**。
⇒ 本脚本补投：**母配方（lr3.5/ep20/w2.0/uf1）下，给重点结构各补 2 个种子**。

选哪些结构
----------
以**速度表 + 已验证的精度画像**排序，优先：
  ① **320×320 输出**（late_upsample）—— 与基线 cs_t0043 的输出分辨率一致
     ⇒ 门禁/评测口径与基线完全相同 ⇒ T1 退化最小（这是本轮的核心假设）；
  ② 其中**最快**的两个（p2f_hc32_late_sc025 = 19.83 ms、p2f_hc32_late_sc05 = 20.06 ms）
     —— 速度余量最大 ⇒ 后续还能叠「补 T1」的超参（w2.5 / uf0）。

用法（服务器，conda uav）：
    python manu/training/build_seed_arms.py \
        --speed-table manu/pipeline/npu/struct_speed_table.json \
        --out /mnt/data/siping/runs/npu20ms/seed_arms.json \
        --seed-base 3000 --seeds-per-struct 2 --limit-ms 21.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "manu" / "training"))
try:
    import optuna_mix_ratio_search as D
except Exception as _exc:  # pragma: no cover
    raise SystemExit(f"[FATAL] 导入 dispatcher 常量失败：{_exc}") from _exc

BASE = {
    "mix_spec": "w2.0", "lr_scale": 3.5, "warmup_epochs": 1.5, "weight_decay": 0.000269,
    "focal_beta": 2.75, "offset_weight": 0.40, "mosaic": 0.15, "scale": 0.10,
    "translate": 0.05, "clip_max_norm": 5.65, "freeze_bn": False,
}
# 母配方下要补种子的结构（按优先级；每项 = 结构描述）
# ⚠️ 与 npu20ms_plan.json 里已有的臂**同名结构、不同 seed** ⇒ 它们能凑成 soup。
TARGETS = [
    # (head_conv, late_upsample, p2_out_scale, 说明)
    # ★ 三个都是**母配方已有 1 条臂**的 late 结构 ⇒ 各补 2 个种子即可凑 **n=3**
    #   ⇒ 3 个独立的 soup 家族（对齐基线 cs_t0043 的形态）。
    (32, True, 0.25, "★320输出+最快(19.83ms)+neck收窄0.25"),
    (32, True, 0.5, "★320输出+20.06ms+neck收窄0.5"),
    (32, True, 1.0, "★320输出+20.83ms+neck不收窄"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed-table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-base", type=int, default=3000)
    ap.add_argument("--seeds-per-struct", type=int, default=2,
                    help="每个结构补几个种子（2 ⇒ 与原臂凑成 n=3）")
    ap.add_argument("--limit-ms", type=float, default=21.0)
    args = ap.parse_args()

    tbl = json.loads(Path(args.speed_table).read_text(encoding="utf-8"))
    by_key = {}
    for a in tbl.get("arms", []):
        sp = a.get("spec") or {}
        ms = a.get("ms")
        if ms is None or ms > args.limit_ms:
            continue
        if not bool(sp.get("ablate_p0", True)):
            continue
        k = (sp.get("head_src"), int(sp.get("head_conv", 64)), sp.get("act", "relu"),
             bool(sp.get("late_upsample", False)),
             round(float(sp.get("p2_out_scale", 1.0) or 1.0), 3))
        prev = by_key.get(k)
        # 优先保留**实测**（ms_source=measured）的项
        if prev is None or (prev.get("ms_source") != "measured"
                            and a.get("ms_source") == "measured"):
            by_key[k] = {"ms": float(ms), "spec": sp, "name": a["name"],
                         "ms_source": a.get("ms_source", "measured")}

    arms = []
    seed = args.seed_base
    for hc, late, sc, note in TARGETS:
        k = ("p2_fused", hc, "relu", late, round(float(sc), 3))
        hit = by_key.get(k)
        if hit is None:
            print(f"[SKIP] 结构 {k} 不在 ≤{args.limit_ms} ms 的速度表里 ⇒ 不补种子")
            continue
        for i in range(args.seeds_per_struct):
            q = dict(BASE)
            q.update({
                "no_p2": False,
                "head_level": "p2_fused",
                "act": "relu",
                "head_conv": hc,
                "late_upsample": late,
                "p2_out_scale": float(sc),
                # ⚠️ 2026-11-06 **回退到 True**（曾试过 False ⇒ 必须回退）：
                #   实测 uf=False（冻结 b0-b4）+ p2_fused/late 新头 ⇒ **训不动**
                #   （trial_0004：Ep13 loss 1.63 → Ep20 仍 1.44~1.50；而 uf=True 的
                #     trial_0002/0005 是 0.89~0.99）⇒ 种子臂必须用 uf=True 才能训出可用模型。
                #   ⚠️ 「uf=True 导致 FP +70」这个旧结论是**混淆变量**的：老轮从未测过
                #   「p2_fused + uf=False」，只测过「p1+uf=False(FP 39/42)」与
                #   「p2_fused+uf=True(FP 104)」⇒ 结构与 uf 未解耦。
                #   ⇒ 第一轮**同时**有 uf=True 与 uf=False 两族 ⇒ 周一按验收表挑，不要预先定论。
                "unfreeze_b0b4": True,
                "head_src": "p2_fused",
                "epochs": 20,
            })
            arms.append({
                "role": f"SEED_hc{hc}_late{int(late)}_sc{int(round(sc*100))}_s{seed}",
                "seed": seed,
                "note": (f"★种子臂{note} | 板卡 {hit['ms']:.2f} ms（{hit['ms_source']}）"
                         f" | 与第一轮 uf=True 母配方臂凑 n=3 soup（uf=True 才训得动）"),
                "params": q,
            })
            seed += 1

    # ---- 自检：取值必须在 dispatcher 分布内 ----
    dists = {
        "mix_spec": ("cat", list(D.MIX_SPECS)),
        "lr_scale": ("float", (D.LR_SCALE_LOW, D.LR_SCALE_HIGH)),
        "warmup_epochs": ("float", (D.WARMUP_EPOCHS_LOW, D.WARMUP_EPOCHS_HIGH)),
        "weight_decay": ("float", (1e-5, 5e-4)),
        "focal_beta": ("float", (2.2, 2.9)),
        "offset_weight": ("float", (0.40, 0.70)),
        "mosaic": ("float", (0.0, 0.15)),
        "scale": ("float", (0.10, 0.30)),
        "translate": ("float", (0.05, 0.15)),
        "clip_max_norm": ("float", (2.0, 8.0)),
        "freeze_bn": ("cat", [True, False]),
        "no_p2": ("cat", list(D.STRUCT_NO_P2_CHOICES)),
        "head_level": ("cat", list(D.STRUCT_HEAD_LEVEL_CHOICES)),
        "act": ("cat", list(D.STRUCT_ACT_CHOICES)),
        "head_conv": ("cat", list(D.STRUCT_HEAD_CONV_CHOICES)),
        "late_upsample": ("cat", list(D.STRUCT_LATE_CHOICES)),
        "unfreeze_b0b4": ("cat", list(D.STRUCT_UNFREEZE_CHOICES)),
        "head_src": ("cat", list(D.STRUCT_HEAD_SRC_CHOICES)),
        "p2_out_scale": ("cat", list(D.STRUCT_P2_OUT_SCALE_CHOICES)),
    }
    bad = []
    for a in arms:
        for k, v in (a.get("params") or {}).items():
            if k not in dists:
                continue
            kind, spec = dists[k]
            if kind == "cat":
                if v not in spec:
                    bad.append((a["role"], k, v, spec))
            else:
                lo, hi = spec
                if not (lo <= v <= hi):
                    bad.append((a["role"], k, v, (lo, hi)))
    if bad:
        print("\n[FATAL] 取值不在 dispatcher 分布内：")
        for role, k, v, spec in bad:
            print(f"   [{role}] {k}={v!r} ∉ {spec}")
        return 1

    Path(args.out).write_text(json.dumps(arms, indent=2, ensure_ascii=False) + "\n")
    print(f"[OK] {len(arms)} 条种子臂 -> {args.out}")
    for a in arms:
        q = a["params"]
        print(f"   {a['role']:<40} seed={a['seed']} hc={q['head_conv']} "
              f"late={int(q['late_upsample'])} sc={q['p2_out_scale']} "
              f"lr={q['lr_scale']} ep={q['epochs']} mix={q['mix_spec']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
