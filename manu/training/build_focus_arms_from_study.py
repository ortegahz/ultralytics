#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从既有 Optuna study 里**精确**复制配方，生成/覆盖定点打靶臂。

为什么需要它
------------
「同超参多种子」对照要求配方**逐位相同**（否则就不是多种子对照，只是相近配方）。
但 ``optuna_summary.csv`` 里的超参是**写盘时格式化过的**（``lr_scale:.4f``、
``clip_max_norm:.2f``、``weight_decay:.8f`` …）⇒ **精度已经丢了**：

| 真实值（study.db） | CSV 里 |
|:--|:--|
| ``lr_scale = 2.922306436248727`` | ``2.9223`` |
| ``clip_max_norm = 5.563409172338667`` | ``5.56`` |
| ``warmup_epochs = 1.2055599542047064`` | ``1.205`` |
| ``weight_decay = 0.00014963699994362378`` | ``0.00014964`` |

⇒ 照抄 CSV 会让「多种子对照」退化成「近似配方对照」，**证据链失效**。
本脚本直接读 ``study.db`` 的 ``trial.params``（**原始 float**），
再按需覆盖少量维度（如骨干保护），产出精确的臂。

用法（服务器 ``conda activate uav``）：

```bash
# arms.json：[{"role": "...", "seed": 224, "trial": 77, "note": "...", "override": {...}}]
python manu/training/build_focus_arms_from_study.py \
  --study-db   /mnt/data/siping/runs/nogmc_weekend/study.db \
  --study-name mix_ratio_search_nogmc_weekend \
  --arms-json  manu/training/nogmc_focus_arms.json \
  --plan       manu/training/nogmc_focus_plan.json \
  --apply
```

``--apply`` 才会写回 plan（默认干跑：只打印将要写入的臂与差异）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna

# 臂里必须出现的维度（少一个就不是完整配方）
REQUIRED = ("mix_spec", "lr_scale", "warmup_epochs", "weight_decay", "focal_beta",
            "offset_weight", "mosaic", "scale", "translate", "clip_max_norm",
            "freeze_bn", "init_from", "epochs", "backbone_lr_scale",
            "freeze_backbone_epochs")


def _fmt(v) -> str:
    """按值的真实精度打印（float 用 repr ⇒ 不丢位）。"""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description="从 study.db 精确复制配方生成打靶臂")
    ap.add_argument("--study-db", required=True)
    ap.add_argument("--study-name", required=True)
    ap.add_argument("--arms-json", required=True, help="臂定义：[{role, seed, trial, note, override}]")
    ap.add_argument("--plan", required=True, help="要写回的 plan JSON")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    storage = f"sqlite:///{args.study_db}"
    study = optuna.load_study(study_name=args.study_name, storage=storage)
    by_number = {t.number: t for t in study.get_trials(deepcopy=False)}

    arms = json.loads(Path(args.arms_json).read_text(encoding="utf-8"))
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    built = []
    for spec in arms:
        num = int(spec["trial"])
        t = by_number.get(num)
        if t is None:
            print(f"[FAIL] study 里没有 trial_{num:04d}")
            return 3
        params = dict(t.params)
        params.update(spec.get("override") or {})
        missing = [k for k in REQUIRED if k not in params]
        if missing:
            print(f"[FAIL] trial_{num:04d} 缺维度：{missing}")
            return 3
        built.append({
            "role": str(spec["role"]),
            "seed": int(spec["seed"]),
            "note": str(spec.get("note", "")),
            "params": params,
        })

    # ---- 与现有 plan 对比（同一 role 则覆盖，否则追加）----------------
    existing = {a.get("role"): a for a in plan}
    changed = []
    for b in built:
        old = existing.get(b["role"])
        if not old:
            continue
        diffs = []
        for k in REQUIRED:
            ov = (old.get("params") or {}).get(k)
            nv = b["params"].get(k)
            same = (ov == nv)
            if not same and isinstance(ov, (int, float)) and isinstance(nv, (int, float)):
                same = float(ov) == float(nv)
            if not same:
                diffs.append(f"{k}: {_fmt(ov)} → {_fmt(nv)}")
        if diffs:
            changed.append((b["role"], diffs))

    print(f"[BUILD] 生成 {len(built)} 条臂（精确参数来自 study.db）")
    for b in built:
        p = b["params"]
        print(f"  · {b['role']} (seed={b['seed']}) ← trial 参数已精确复制")
    if changed:
        print(f"\n[DIFF] 与现有 plan 有 {len(changed)} 处差异（CSV 精度丢失导致）：")
        for role, diffs in changed:
            print(f"  {role}:")
            for d in diffs:
                print(f"      {d}")
    else:
        print("[DIFF] 与现有 plan 无差异")

    if not args.apply:
        print("[DRY-RUN] 未写回（加 --apply 生效）")
        return 0

    roles = {b["role"] for b in built}
    out = [a for a in plan if a.get("role") not in roles] + built
    plan_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] 写回 {plan_path}（共 {len(out)} 条臂）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
