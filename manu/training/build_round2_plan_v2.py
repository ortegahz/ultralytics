#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二轮臂计划 v2（2026-11-06 18:15）——**止损版**：把 82 条「hc32 从零学」换成「诊断臂 + 老轮结构 relu 化」。

为什么推翻 v1（`build_round2_plan.py` 的 82 臂）
------------------------------------------------
v1 的 4 个家族全是 `hc32 / hc24 + p2_fused + relu + unfreeze=True` ⇒ **头的 conv 形状变了
⇒ cs_t0043 的头权重全部丢弃 ⇒ 从零学** ⇒ 实测**全线训不动**：

* 新轮 sc25 家族 4 条臂（trial_0004~0007）：最终 loss **1.07~1.28**（hm 0.90~1.13），
  T1 F1 **0.25~0.27**（ΔT1 = **−63 pp**），T2/T3 = 0 ⇒ 死模型；
* 老轮 late 臂 4 条（`p3/hc64/silu/late/uf=False`）：最终 loss **0.17~0.25**（hm 0.12~0.16），
  T1 F1 **0.874~0.889**（ΔT1 = −0.20 ~ −1.68 pp）⇒ 正常。

⇒ 差异不在「late」，而在**头**（hc64→hc32 / p3→p2_fused 把 conv 形状改了 ⇒ 权重丢）
⇒ **v1 的 82 臂大概率全灭 ⇒ 30 h GPU 白烧 ⇒ 必须止损。**

v2 的结构：**6 条诊断臂（隔离变量）+ 2 个候选家族（各 12~18 条种子）**
--------------------------------------------------------------
诊断臂（各 1 条，先跑 ⇒ 1.5 h 内出训练曲线，周一就能定论）：

| 臂 | 结构 | uf | act | 检验什么 |
|:--|:--|:--|:--|:--|
| D1 | p3/hc64/late/sc1.0 | False | relu | **主候选**（老轮结构只换 act ⇒ 头形状不变 ⇒ 权重可继承） |
| D2 | p3/hc64/late/sc1.0 | False | silu | 复现老轮（应重现 T1 0.874~0.889 ⇒ 证明「结构+配方」本身没问题） |
| D3 | p3/hc64/late/sc1.0 | True  | relu | 检验 unfreeze_b0b4 是否是主因 |
| D4 | p2f/hc32/late/sc1.0 | False | relu | 检验「换头（hc32）从零学」是否致命（uf=False + 不收窄） |
| D5 | p2f/hc32/late/sc0.25 | True | relu | = v1 死配置复现（对照） |
| D6 | p3/hc64/late/sc1.0 | False | relu, lr1.5 | 检验 LR（若 relu 难训 ⇒ 降 lr） |

家族（诊断臂结论出来后直接可用；与对应诊断臂**同配方**⇒同 params 配对）：
* `F_MAIN` = p3/hc64/relu/late/sc1.0/uf=False/lr3.5 × 12 种子（**与 D1 同配方**）
* `F_ALT`  = p2f/hc32/relu/late/sc1.0/uf=False/lr3.5 × 12 种子（**与 D4 同配方**）

★ 用户硬要求「all relu」⇒ `F_MAIN` 是交付主线；D2（silu）只作对照，不作交付。

用法（服务器，conda uav）：
    python manu/training/build_round2_plan_v2.py \
        --out /mnt/data/siping/runs/npu20ms_seed/seed_arms.json \
        --seed-base 4100 --seeds-per-struct 12
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

BASE = {
    "mix_spec": "w2.0", "lr_scale": 3.5, "warmup_epochs": 1.5, "weight_decay": 0.000269,
    "focal_beta": 2.75, "offset_weight": 0.40, "mosaic": 0.15, "scale": 0.10,
    "translate": 0.05, "clip_max_norm": 5.65, "freeze_bn": False,
}


def _params(head_src: str, hc: int, act: str, *, late: bool = True, sc: float = 1.0,
            uf: bool = False, lr: float = 3.5, epochs: int = 20) -> dict:
    q = dict(BASE)
    q.update({
        "lr_scale": float(lr),
        "no_p2": head_src in ("p3", "p2_raw"),
        "head_level": head_src, "act": act, "head_conv": int(hc),
        "late_upsample": bool(late), "p2_out_scale": float(sc),
        "unfreeze_b0b4": bool(uf), "head_src": head_src, "epochs": int(epochs),
    })
    return q


# ---- 诊断臂：(role, 参数覆盖, 说明) ----
DIAGS = [
    ("D1_p3hc64_relu_late_uf0", dict(head_src="p3", hc=64, act="relu", uf=False),
     "★主候选：老轮结构只换 act=relu（头形状不变⇒权重可继承）"),
    ("D2_p3hc64_silu_late_uf0", dict(head_src="p3", hc=64, act="silu", uf=False),
     "对照：复现老轮 late 臂（应重现 T1 0.874~0.889）"),
    ("D3_p3hc64_relu_late_uf1", dict(head_src="p3", hc=64, act="relu", uf=True),
     "检验 unfreeze_b0b4 是否主因"),
    ("D4_p2fhc32_relu_late_sc10_uf0", dict(head_src="p2_fused", hc=32, act="relu", sc=1.0, uf=False),
     "检验「换头 hc32 从零学」是否致命（不收窄 + 老配方）"),
    ("D5_p2fhc32_relu_late_sc025_uf1", dict(head_src="p2_fused", hc=32, act="relu", sc=0.25, uf=True),
     "对照：v1 死配置复现"),
    ("D6_p3hc64_relu_late_uf0_lr15", dict(head_src="p3", hc=64, act="relu", uf=False, lr=1.5),
     "检验 LR（relu 难训时降 lr）"),
]

# ---- 家族：(tag, 参数覆盖, 种子数) ----
FAMS = [
    ("F_MAIN_p3hc64_relu_late_uf0", dict(head_src="p3", hc=64, act="relu", uf=False), 12),
    ("F_ALT_p2fhc32_relu_late_sc10_uf0", dict(head_src="p2_fused", hc=32, act="relu", sc=1.0, uf=False), 12),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-base", type=int, default=4100)
    ap.add_argument("--seeds-per-struct", type=int, default=12)
    ap.add_argument("--seeds-main", type=int, default=36,
                    help="F_MAIN（p3/hc64/relu/late，★交付主线，13.131 ms）的种子数")
    ap.add_argument("--seeds-alt", type=int, default=24,
                    help="F_ALT（p2f/hc32/relu/late/sc1.0，20.829 ms）的种子数")
    ap.add_argument("--diag-prefix", default="",
                    help="诊断臂 role 前缀（★重跑时必须改，否则与 study 里旧臂 role 撞名 "
                         "⇒ soup 解析成员会指向没有产物的旧 trial）")
    args = ap.parse_args()

    arms: list[dict] = []
    seed = args.seed_base

    # 诊断臂先跑（它们的结论决定家族选谁）
    for role, kw, note in DIAGS:
        arms.append({"role": f"{args.diag_prefix}{role}", "seed": seed,
                     "note": f"★诊断臂 | {note}", "params": _params(**kw)})
        seed += 1

    # 家族：先每家族 3 个保底，再交错补齐（FIFO 下队尾家族不能 n=0）
    fam_specs = []
    for tag, kw, cnt in FAMS:
        n = args.seeds_main if tag.startswith("F_MAIN") else args.seeds_alt
        fam_specs.append((tag, kw, max(3, n)))
    per = max(n for _, _, n in fam_specs)
    for i in range(per):
        for tag, kw, n in fam_specs:
            if i >= n:
                continue
            arms.append({"role": f"SEED_{tag}_s{seed}", "seed": seed,
                         "note": f"★家族种子（与对应诊断臂同配方）| {tag}",
                         "params": _params(**kw)})
            seed += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(arms, ensure_ascii=False, indent=2), encoding="utf-8")

    roles = [a["role"] for a in arms]
    seeds = [a["seed"] for a in arms]
    assert len(roles) == len(set(roles)), "role 重复"
    assert len(seeds) == len(set(seeds)), "seed 重复"
    print(f"[OK] 写出 {len(arms)} 臂 -> {out}")
    print(f"     诊断臂 {sum(1 for a in arms if a['role'].startswith('D'))} 条")
    print(f"     家族分布：{dict(Counter(a['role'].split('_s')[0] for a in arms if a['role'].startswith('SEED_')))}")
    print(f"     seed 区间 {min(seeds)} ~ {max(seeds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
