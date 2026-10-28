#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""向**运行中**的 Optuna study 投放**对照组 trial**（不新开 study、不改搜索空间）。

为什么要这个脚本
----------------
用户质疑："相同配置无法复现 RC1 的精度，这个不正常吧？"
要闭环这个问题，必须有一个**可判定的对照**：在**与 RC1 完全同口径**（超参原值 +
cosine 周期 5 epoch）下再跑一次 natural 档。如果它落回 RC1/Ep3 附近
（T3 F1 ≈ 0.8717 / T1 FP ≈ 941 / T2 FP ≈ 31 / U ≈ 0.0276），
则说明"复现失败"是判据用单点造成的错觉；否则说明管线里还有未发现的差异。

本脚本做什么
------------
1. 从 `optuna_mix_ratio_search.py` **用 AST 解析**出 `ROUTE_B_DEFAULTS` / `MIX_SPECS`
   —— 不复制常量 ⇒ 永远与搜索的真实参数空间一致，不会漂移。
2. 把 `ROUTE_B_DEFAULTS` 原值 + 指定 `mix_spec` 组合**入队**（`enqueue_trial`）。
   调度器用 `study.ask()` 取 trial，**入队的会被优先取走** ⇒ 无需重启调度器。
3. 给入队 trial 打 `user_attrs` 标记（可审计：CSV/DB 里一眼看出是对照组）。
   注意 `enqueue_trial` 的参数**必须经 `suggest_*` 取值才生效**，直接 `ask()`
   后 `trial.params` 是空的 —— 所以这里走 `enqueue_trial` 而非 `ask()`。

安全性
------
* **不改搜索空间**（`mix_spec` 取值仍在 `MIX_SPECS` 内）⇒ 不会触发
  `CategoricalDistribution does not support dynamic value space`。
* **不碰正在跑的 trial**：只是往 study 里加 WAITING 状态的 trial。
* **可重复执行**：`--dry-run` 先看要投什么。

用法（服务器，`ssh -p 32222 huangzhe@192.168.99.40` → `conda activate uav`）：
```bash
cd /tmp/pycharm_project_10ae9e2e
python manu/training/enqueue_control_trials.py --list
python manu/training/enqueue_control_trials.py --specs natural,w4.0 --dry-run
python manu/training/enqueue_control_trials.py --specs natural,w4.0
```
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import optuna

REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_MODULE = REPO_ROOT / "manu" / "training" / "optuna_mix_ratio_search.py"

# 对照组的语义标签
ROLE_BY_SPEC = {
    "natural": "RC1_REPRO_CONTROL",   # 复现对照：与 RC1 同超参
    "w4.0": "TARGET_RATIO_CONTROL",   # 目标比例对照：真正的 1:4
}
DEFAULT_SPECS = "natural,w4.0"

# 复现对照的两种强度
#   普通（默认）      ：超参与 RC1 一致，但随机种子/worker 数与 RC1 不同
#                       ⇒ 检验「同分布可复现」
#   位级复现（--exact）：**连 seed 与 dataloader worker 数都对齐 RC1**
#                       （RC1: seed=0, workers=16），且门禁量 Ep3
#                       ⇒ 5-epoch 训练里 Ep3 的权重**就是 RC1/Ep3 的等价物**
#                       ⇒ 数据流完全一致，只剩 GPU 非确定性 ⇒ 应当**逐位接近**
RC1_SEED = 0
RC1_WORKERS = 16
RC1_GATE_EPOCH = 3


def literal_assignment(module_path: Path, name: str):
    """用 AST 从源码里取出模块级字面量赋值（不 import ⇒ 无重依赖、不会漂移）。"""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(f"{module_path} 里找不到模块级常量 {name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="投放对照组 trial 到运行中的 Optuna study")
    ap.add_argument("--storage", default="sqlite:///runs/optuna_mix_ratio/study.db",
                    help="Optuna RDB storage URL（相对路径按 CWD 解析）")
    ap.add_argument("--study-name", default="mix_ratio_search")
    ap.add_argument("--specs", default=DEFAULT_SPECS,
                    help="要入队的 mix_spec（逗号分隔）")
    ap.add_argument("--dry-run", action="store_true", help="只打印要投的参数，不真的入队")
    ap.add_argument("--list", action="store_true", help="列出当前 WAITING 状态的 trial 后退出")
    ap.add_argument("--exact", action="store_true",
                    help="位级复现模式：seed=0 / workers=16 / 门禁量 Ep3（与 RC1 完全对齐）")
    ap.add_argument("--from-trial", default="",
                    help="复制某个已有 trial 的**精确超参**入队（用于对胜出配置做多种子确认重跑）。"
                         "例：--from-trial trial_0080")
    ap.add_argument("--summary-csv", default="runs/optuna_mix_ratio/optuna_summary.csv",
                    help="--from-trial 模式下从哪个汇总 CSV 取超参（默认=GMC 侧，旧行为不变；"
                         "no-GMC 侧传 runs/optuna_nogmc/optuna_summary.csv）")
    ap.add_argument("--count", type=int, default=1,
                    help="--from-trial 模式下的重复次数（每个副本换不同 trial 编号 ⇒ 不同随机种子）")
    ap.add_argument("--role-tag", default="",
                    help="覆盖入队 trial 的 role 标签（默认 = CONFIRM_<from_trial>）。"
                         "例：--role-tag RC2_RECIPE_TRANSPLANT —— 用于「配方移植」这类语义不同的情况")
    ap.add_argument("--role-note", default="",
                    help="覆盖入队 trial 的 note 文本（默认 = 确认重跑说明）")
    args = ap.parse_args()

    route_b = literal_assignment(SEARCH_MODULE, "ROUTE_B_DEFAULTS")
    mix_specs = literal_assignment(SEARCH_MODULE, "MIX_SPECS")
    print(f"[SRC] {SEARCH_MODULE}")
    print(f"      ROUTE_B_DEFAULTS = {json.dumps(route_b, ensure_ascii=False)}")
    print(f"      MIX_SPECS        = {mix_specs}")

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)

    if args.list:
        waiting = [t for t in study.get_trials(deepcopy=False)
                   if t.state == optuna.trial.TrialState.WAITING]
        print(f"\n[WAITING] {len(waiting)} 个待跑 trial")
        for t in waiting:
            print(f"  #{t.number} mix={t.params.get('mix_spec')} user_attrs={t.user_attrs}")
        done = [t for t in study.get_trials(deepcopy=False)
                if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
                               optuna.trial.TrialState.FAIL, optuna.trial.TrialState.RUNNING)]
        print(f"[TOTAL] study 共 {len(done)} 个已结算/在跑 trial")
        return 0

    specs = [s.strip() for s in args.specs.split(",") if s.strip()]
    bad = [s for s in specs if s not in mix_specs]
    if bad:
        print(f"[FATAL] mix_spec {bad} 不在搜索空间 {mix_specs} 内 —— 拒绝入队")
        return 1

    payloads = []
    if args.from_trial:
        # ---- 确认重跑：复制已有 trial 的精确超参（换 trial 编号 ⇒ 换随机种子）----
        import csv
        import io
        summary = Path(args.summary_csv)
        if not summary.is_file():
            print(f"[FATAL] 找不到 {summary}")
            return 1
        with summary.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        src = next((r for r in rows if r["trial"] == args.from_trial), None)
        if src is None:
            print(f"[FATAL] {args.from_trial} 不在 {summary} 里")
            return 1
        p = {
            "mix_spec": src["mix_spec"],
            "lr_scale": float(src["lr_scale"]),
            "warmup_epochs": float(src["warmup_epochs"]),
            "weight_decay": float(src["weight_decay"]),
            "focal_beta": float(src["focal_beta"]),
            "offset_weight": float(src["offset_weight"]),
            "mosaic": float(src["mosaic"]),
            "scale": float(src["scale"]),
            "translate": float(src["translate"]),
            "clip_max_norm": float(src["clip_max_norm"]),
            "freeze_bn": src["freeze_bn"].strip() in ("1", "True", "true"),
        }
        for k in ("mix_spec",):
            if p[k] not in mix_specs:
                print(f"[FATAL] {args.from_trial} 的 mix_spec={p[k]} 不在搜索空间内")
                return 1
        print(f"[SRC] 复制 {args.from_trial} 的超参：{json.dumps(p, ensure_ascii=False)}")
        for _ in range(max(1, args.count)):
            ua = {
                "role": args.role_tag or f"CONFIRM_{args.from_trial}",
                "note": args.role_note or f"确认重跑：复制 {args.from_trial} 的精确超参，换随机种子",
                "enqueued_by": "enqueue_control_trials.py",
                "src_trial": args.from_trial,
            }
            if args.exact:
                ua.update({"seed": RC1_SEED, "workers": RC1_WORKERS, "gate_epoch": RC1_GATE_EPOCH})
            payloads.append((p["mix_spec"], ua["role"], p, ua))
    else:
        for spec in specs:
            p = dict(route_b)
            p["mix_spec"] = spec
            role = ROLE_BY_SPEC.get(spec, "CONTROL")
            ua = {
                "role": role,
                "note": "对照组：与 RC1/Ep3 同超参（ROUTE_B_DEFAULTS 原值）+ cosine 周期 5 epoch",
                "enqueued_by": "enqueue_control_trials.py",
            }
            if args.exact:
                # 位级复现：连随机性都对齐 RC1
                ua["role"] = f"{role}_EXACT"
                ua["seed"] = RC1_SEED
                ua["workers"] = RC1_WORKERS
                ua["gate_epoch"] = RC1_GATE_EPOCH
                ua["note"] = ("位级复现：seed=0 / workers=16 / 门禁量 Ep3，"
                              "5-epoch 训练下 Ep3 权重即 RC1/Ep3 的等价物")
            payloads.append((spec, role, p, ua))

    print("\n[PLAN] 将入队的对照 trial：")
    for spec, role, p, ua in payloads:
        print(f"  mix_spec={spec:<9} role={role:<22} params={json.dumps(p, ensure_ascii=False)}")
        print(f"      user_attrs={json.dumps(ua, ensure_ascii=False)}")

    if args.dry_run:
        print("\n[DRY-RUN] 未入队（去掉 --dry-run 执行）")
        return 0

    for spec, role, p, ua in payloads:
        study.enqueue_trial(
            p,
            skip_if_exists=False,  # Optuna 4.x：True 会把重复参数组合直接跳过
            user_attrs=ua,
        )
        print(f"[ENQUEUE] mix_spec={spec} role={ua['role']} ✔")

    waiting = [t for t in study.get_trials(deepcopy=False)
               if t.state == optuna.trial.TrialState.WAITING]
    print(f"\n[OK] 当前 WAITING trial 共 {len(waiting)} 个：")
    for t in waiting:
        print(f"  #{t.number} mix={t.params.get('mix_spec')} role={t.user_attrs.get('role')}")
    print("[提示] 调度器用 study.ask() 取 trial，入队的会被优先取走；"
          "无需重启搜索，可用 status 脚本观察。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
