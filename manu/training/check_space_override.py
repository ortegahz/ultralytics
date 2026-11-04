#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""搜索空间覆盖自检：验证「固定参数是否落在**被覆盖后**的分布内」。

为什么需要它
------------
Optuna 的分布兼容性检查是 **study 级**的：某个参数名一旦在某个 study 里记录过分布，
之后该 study 的所有新 trial 都必须与该分布兼容 ⇒ 如果覆盖后的空间和 plan 里写死的
固定参数对不上，**入队/派单那一刻就会炸**（`ValueError: '25' not in (6, 8, 10, 15)`）。

本脚本在**不连数据库、不起训练、不碰 GPU**的前提下：
  1. 用**与调度器完全同一份**覆盖逻辑（`optuna_mix_ratio_search.apply_space_overrides`）
     算出「生效后的空间」；
  2. 逐条检查给定臂的固定参数是否落在该分布内（用 Optuna 自己的分布对象判定）；
  3. 额外用 `FixedTrial` 跑一遍调度器真实的 `suggest_params()` ⇒ 端到端验证；
  4. 打印「覆盖前 vs 覆盖后」的分布兼容性报告（提醒必须换新 study-name）。

用法
----
```bash
# 自带演示臂（lr=4.0 / lr=3.0 / epochs=25 / mix_spec=w2.5）
python manu/training/check_space_override.py --focus-space \\
    --lr-scale-high 4.0 --epoch-choices 6,8,10,15,20,25 \\
    --mix-choices natural,w2.0,w2.5,w3.0,w6.0 --demo

# 检查一份 plan（格式与 --enqueue-plan 一致：[{"role":..., "params":{...}}, ...]）
python manu/training/check_space_override.py --focus-space \\
    --lr-scale-high 4.0 --epoch-choices 6,8,10,15,20,25 --arms-json plan.json
```

退出码：0 = 全部合法；1 = 有越界/不一致。
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import optuna
from optuna import distributions as dist

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import manu.training.optuna_mix_ratio_search as M  # noqa: E402


# 固定维度（不随覆盖变化）——与 suggest_params 里的 suggest_* 一一对应
FIXED_FLOATS = {
    "warmup_epochs": (M.WARMUP_EPOCHS_LOW, M.WARMUP_EPOCHS_HIGH, False, None),
    "weight_decay": (1e-5, 5e-4, True, None),
    "focal_beta": (2.2, 2.9, False, 0.05),
    "offset_weight": (0.40, 0.70, False, 0.05),
    "mosaic": (0.0, 0.15, False, 0.05),
    "scale": (0.10, 0.30, False, 0.05),
    "translate": (0.05, 0.15, False, 0.05),
    "clip_max_norm": (2.0, 8.0, False, None),
}
# 固定维度里的 categorical
FIXED_CATS = {"freeze_bn": [True, False]}
# 可被覆盖的维度
SPACE_DIMS = ("mix_spec", "lr_scale", "epochs", "init_from", "init_from_x",
              "backbone_lr_scale", "freeze_backbone_epochs")


def snapshot_module() -> dict:
    """读当前模块级空间常量（覆盖**前**的快照）。"""
    return {
        "lr_scale": (M.FOCUS_LR_SCALE[0], M.FOCUS_LR_SCALE[1]),
        "epochs": list(M.FOCUS_CHOICES["epochs"]),
        "mix_spec": list(M.FOCUS_CHOICES["mix_spec"]),
        "init_from": list(M.FOCUS_CHOICES["init_from"]),
        "init_from_x": list(M.INIT_X_CHOICES),
        "backbone_lr_scale": list(M.FOCUS_CHOICES["backbone_lr_scale"]),
        "freeze_backbone_epochs": list(M.FOCUS_CHOICES["freeze_backbone_epochs"]),
    }


def make_distribution(space: dict, name: str):
    """按参数名构造 Optuna 分布对象（与 suggest_params 的 suggest_* 对应）。"""
    if name == "lr_scale":
        lo, hi = space["lr_scale"]
        return dist.FloatDistribution(float(lo), float(hi), log=True)
    if name in ("epochs", "mix_spec", "init_from", "init_from_x",
                "backbone_lr_scale", "freeze_backbone_epochs"):
        return dist.CategoricalDistribution(list(space[name]))
    if name in FIXED_FLOATS:
        lo, hi, log, step = FIXED_FLOATS[name]
        return dist.FloatDistribution(float(lo), float(hi), log=log, step=step)
    if name in FIXED_CATS:
        return dist.CategoricalDistribution(list(FIXED_CATS[name]))
    return None


def check_value(space: dict, name: str, value) -> tuple[bool, str]:
    """判定单个固定参数是否落在分布内（用 Optuna 自己的分布对象）。"""
    d = make_distribution(space, name)
    if d is None:
        return True, f"{name}={value!r}（未知维度，跳过）"
    if isinstance(d, dist.FloatDistribution):
        try:
            fv = float(value)
        except (TypeError, ValueError):
            return False, f"{name}={value!r} 不是数值"
        ok = d._contains(fv)
        lo, hi = space["lr_scale"] if name == "lr_scale" else (d.low, d.high)
        return ok, f"{name}={value!r} {'∈' if ok else '∉'} [{lo}, {hi}]"
    # CategoricalDistribution._contains 吃的是内部表示（int 下标）⇒ 先转
    try:
        internal = d.to_internal_repr(value)
    except ValueError as exc:
        return False, f"{name}={value!r} 不在 {list(d.choices)}（{exc}）"
    ok = d._contains(internal)
    return ok, f"{name}={value!r} {'∈' if ok else '∉'} {list(d.choices)}"


def build_full_params(space: dict, arm: dict) -> dict:
    """把臂补成「完整参数集」（缺的维度用首个取值/区间中点补齐）。

    目的：让 `FixedTrial` 能完整跑一遍调度器真实的 `suggest_params()`。
    """
    full = dict(arm)
    for name in ("mix_spec", "epochs", "init_from", "init_from_x",
                 "backbone_lr_scale", "freeze_backbone_epochs"):
        if name not in full and name in space:
            full[name] = space[name][0]
    if "lr_scale" not in full:
        lo, hi = space["lr_scale"]
        full["lr_scale"] = float(lo)
    if "freeze_bn" not in full:
        full["freeze_bn"] = False
    for name, (lo, hi, _log, _step) in FIXED_FLOATS.items():
        if name not in full:
            full[name] = float(lo)
    return full


def end_to_end(space: dict, arm: dict) -> tuple[bool, str]:
    """用 FixedTrial 跑调度器真实的 suggest_params()（端到端）。"""
    full = build_full_params(space, arm)
    fixed = optuna.trial.FixedTrial(full)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if "init_from_x" in full:
            # FixedTrial.set_system_attr 在 optuna 4.9 已 deprecated（仍可用）
            fixed.set_system_attr("fixed_params", {"init_from_x": full["init_from_x"]})
        try:
            M.suggest_params(fixed)
        except Exception as exc:  # noqa: BLE001
            return False, f"suggest_params 抛 {type(exc).__name__}: {exc}"
        bad = [str(w.message) for w in caught if "out of the range" in str(w.message)]
    if bad:
        return False, "FixedTrial 警告：" + "; ".join(bad)
    return True, "suggest_params() 端到端通过"


def compat_report(before: dict, after: dict) -> list[str]:
    """覆盖前 vs 覆盖后：哪些维度与旧分布**不兼容**（⇒ 必须换新 study-name）。"""
    rows = []
    for name in ("lr_scale", "epochs", "mix_spec", "init_from", "init_from_x"):
        old_d = make_distribution(before, name)
        new_d = make_distribution(after, name)
        if old_d is None or new_d is None:
            continue
        try:
            dist.check_distribution_compatibility(old_d, new_d)
            rows.append(f"  {name:<12} 兼容（同一个 study 里也能继续用）")
        except Exception as exc:  # noqa: BLE001
            rows.append(f"  {name:<12} 不兼容 ⇒ 必须换新 study-name（{type(exc).__name__}: {exc}）")
    return rows


def parse_arm(text: str) -> dict:
    """解析 "k=v,k=v" 形式的固定参数（值按 JSON 解析，失败则当字符串）。"""
    out = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"臂参数格式错误（应为 k=v）：{part!r}")
        k, _, v = part.partition("=")
        k = k.strip()
        v = v.strip()
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="搜索空间覆盖自检（不连库、不训练、不碰 GPU）")
    ap.add_argument("--focus-space", action="store_true",
                    help="按聚焦模式（FOCUS_CHOICES / FOCUS_LR_SCALE）检查")
    ap.add_argument("--lr-scale-low", type=float, default=None, help="覆盖 lr 下界")
    ap.add_argument("--lr-scale-high", type=float, default=None, help="覆盖 lr 上界")
    ap.add_argument("--epoch-choices", type=str, default="", help="覆盖 epoch 取值（逗号分隔）")
    ap.add_argument("--mix-choices", type=str, default="", help="覆盖 mix_spec 取值（逗号分隔）")
    ap.add_argument("--init-choices", type=str, default="", help="覆盖 init_from 取值（逗号分隔）")
    ap.add_argument("--init-x-choices", type=str, default="", help="覆盖 init_from_x 取值（逗号分隔）")
    ap.add_argument("--arm", action="append", default=[],
                    help='固定臂，格式 "k=v,k=v"（可重复）。例：--arm "lr_scale=4.0,epochs=25"')
    ap.add_argument("--arms-json", type=str, default="",
                    help='臂清单 JSON（[{"role":...,"params":{...}}, ...] 或 [{"lr_scale":...}, ...]）')
    ap.add_argument("--demo", action="store_true",
                    help="追加 4 条演示臂：lr=4.0 / lr=3.0 / epochs=25 / mix_spec=w2.5")
    args = ap.parse_args()

    # ---- 覆盖前快照（模块默认状态）----
    before = snapshot_module()

    # ---- 走调度器同一份覆盖逻辑 ----
    M.FOCUS_SPACE = bool(args.focus_space)
    M.EXTENDED_SPACE = bool(args.focus_space) or bool(M.EXTENDED_SPACE)
    snap = M.apply_space_overrides(args)
    space = snap["focus"] if args.focus_space else snap["legacy"]

    print("=== 生效后的空间（%s 模式）===" % ("focus" if args.focus_space else "default"))
    print(json.dumps(space, ensure_ascii=False, indent=2))
    print("overridden: %s" % json.dumps(snap["overridden"], ensure_ascii=False))

    if any(snap["overridden"].values()):
        print("\n=== 覆盖前 vs 覆盖后：分布兼容性（study 级检查）===")
        for row in compat_report(before, space):
            print(row)

    # ---- 组装待检查的臂 ----
    arms: list[tuple[str, dict]] = []
    if args.demo:
        arms += [
            ("demo-lr-high", {"lr_scale": 4.0}),
            ("demo-lr-mid", {"lr_scale": 3.0}),
            ("demo-epoch-25", {"epochs": 25}),
            ("demo-mix-w2.5", {"mix_spec": "w2.5"}),
        ]
    for i, text in enumerate(args.arm):
        arms.append((f"cli-{i}", parse_arm(text)))
    if args.arms_json:
        data = json.loads(Path(args.arms_json).read_text(encoding="utf-8"))
        for i, item in enumerate(data):
            if isinstance(item, dict) and "params" in item:
                arms.append((str(item.get("role") or item.get("name") or f"json-{i}"), dict(item["params"])))
            elif isinstance(item, dict):
                arms.append((f"json-{i}", dict(item)))
            else:
                raise ValueError(f"臂清单第 {i} 项格式不对：{item!r}")

    if not arms:
        print("\n（没有待检查的臂：用 --arm / --arms-json / --demo 提供）")
        return 0

    print("\n=== 逐条臂检查 ===")
    n_bad = 0
    for role, arm in arms:
        problems = []
        for k, v in arm.items():
            if k not in SPACE_DIMS and k not in FIXED_FLOATS and k not in FIXED_CATS:
                continue
            ok, msg = check_value(space, k, v)
            print("  %-22s %s %s" % (role, "✅" if ok else "❌", msg))
            if not ok:
                problems.append(msg)
        ok_e2e, msg_e2e = end_to_end(space, arm)
        print("  %-22s %s %s" % (role, "✅" if ok_e2e else "❌", msg_e2e))
        if not ok_e2e:
            problems.append(msg_e2e)
        if problems:
            n_bad += 1

    print("\n结论：%d/%d 条臂合法" % (len(arms) - n_bad, len(arms)))
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
