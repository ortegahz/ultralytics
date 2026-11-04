#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把计划文件里**已入队**臂的 params 对齐到 study 里实际存的值（2026-11-02 新增）。

为什么需要它
------------
调度器的去重护栏用「参数集 + 种子」指纹，浮点按 `round(v, 6)` 归一
（`optuna_mix_ratio_search.py::_cfg_sig`）。于是：

    study 里存的   lr_scale = 2.7676
    计划文件里的   lr_scale = 2.7675790243131346
    ⇒ round(...,6) 分别是 2.7676 与 2.767579 ⇒ **指纹不同** ⇒ 去重失效

2026-11-02 实测：T0077_PROT8 / T0077_SEED2 / T0082_PROT8 / T0082_SEED2 这 4 条臂
在 10:32 那次入队时用的是**从 CSV 抄来的截断值**，之后计划文件被
`build_focus_arms_from_study.py` 修成了 study.db 的精确值 ⇒
**调度器重启时这 4 条会以"新臂"身份再入队一次** ⇒ 白烧 4 个 GPU 槽（约 5.5 卡时）。

修法：**以 study 为准**（study 里那份才是真正会跑的），把计划文件里这 4 条对齐回去。
差异本身是 1e-5 量级（lr/clip/warmup/weight_decay），对结果没有实质影响，
但能省下 4 个 GPU 槽，也避免"同一条臂跑两遍"污染多种子统计。

用法（服务器，conda `uav`）：
```bash
python manu/training/align_plan_with_study.py \
  --plan manu/training/nogmc_focus_plan.json \
  --out-root /mnt/data/siping/runs/nogmc_focus \
  --study-name mix_ratio_search_nogmc_focus [--apply]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

ALLOWED = {
    "mix_spec", "lr_scale", "warmup_epochs", "weight_decay", "focal_beta",
    "offset_weight", "mosaic", "scale", "translate", "clip_max_norm", "freeze_bn",
    "init_from", "init_from_x", "epochs", "backbone_lr_scale", "freeze_backbone_epochs",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out-root", default="/mnt/data/siping/runs/nogmc_focus")
    ap.add_argument("--study-name", default="mix_ratio_search_nogmc_focus")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    st = optuna.load_study(
        storage="sqlite:///" + str(Path(args.out_root) / "study.db"),
        study_name=args.study_name)
    role_trial = {}
    for t in st.trials:
        r = (t.user_attrs or {}).get("role")
        if r:
            role_trial[r] = t

    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    n_fixed = 0
    report = []
    for arm in plan:
        role = arm.get("role")
        t = role_trial.get(role)
        if t is None:
            continue
        fixed = (t.system_attrs or {}).get("fixed_params") or {}
        if not fixed:
            continue
        p = arm.get("params") or {}
        for k in sorted(set(fixed) & ALLOWED):
            if k not in p:
                continue
            a, b = fixed[k], p[k]
            same = False
            if isinstance(a, (int, float)) and not isinstance(a, bool) and \
               isinstance(b, (int, float)) and not isinstance(b, bool):
                same = abs(float(a) - float(b)) <= 1e-12
            else:
                same = a == b
            if not same:
                report.append("%s: %s  plan=%r -> study=%r" % (role, k, b, a))
                p[k] = a
                n_fixed += 1

    print("对齐检查：%d 条已入队臂，%d 个字段与 study 不一致" % (len(role_trial), n_fixed))
    for r in report:
        print("   - %s" % r)
    if n_fixed and args.apply:
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        print("✅ 已写回 %s" % plan_path)
    elif n_fixed:
        print("（未加 --apply，只报告不写回）")
    else:
        print("✅ 无需改动")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
