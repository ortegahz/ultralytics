#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定点打靶计划文件 × 搜索空间 体检器（2026-11-02 新增，防 bug 复发）。

为什么需要它
------------
2026-11-02 15:00 实测踩到：**Optuna 4.9 会校验固定参数**。
`optuna/trial/_trial.py::_suggest` → `_is_fixed_param()` → `distribution.to_internal_repr(value)`，
对 CategoricalDistribution 而言，**值不在 choices 里直接抛**
`ValueError: 'w2.5' not in ('w2.0','w3.0','w6.0','w8.0','natural')`。
⇒ 计划文件里任何一条用了 `--focus-space` 白名单外取值的臂，
**在 trial 被 ask 的瞬间就崩**，不产出任何结果，还白占一次调度。

而 `study.enqueue_trial()` **不做**这个校验 ⇒ 崩溃点被推迟到运行期，
在日志里表现为一条莫名其妙的 FAIL，很容易被漏掉（本轮就差点漏掉 8 条）。

本工具做三件事
--------------
1. **空间体检**：逐条臂校验 15 个维度是否落在当前搜索空间内
   （取值直接从 `optuna_mix_ratio_search` 模块读 ⇒ 与调度器永远同源）。
2. **已入队一致性**：已入队的臂，其 params 必须与 study 里的 fixed_params **逐位相同**
   （改了就等于偷偷开了一条新臂；没改又会被去重护栏跳过 ⇒ 两种情况都要能看见）。
3. **入队体检**：模拟调度器的 `allowed` 过滤，看每条臂真正会被入队的参数集。

用法（服务器，conda `uav`，项目根 /tmp/pycharm_project_10ae9e2e）：
```bash
python manu/training/check_plan_vs_search_space.py \
  --plan manu/training/nogmc_focus_plan.json \
  --out-root /mnt/data/siping/runs/nogmc_focus \
  --study-name mix_ratio_search_nogmc_focus
```
退出码：0 = 全部通过；1 = 有越界/不一致。
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def apply_overrides(M, o: dict) -> None:
    """把**调度器同款**的空间覆盖应用到模块常量上。

    ⚠️ **为什么要覆盖**（2026-11-03）：新 study（`mix_ratio_search_nogmc_t3`）要把
    `lr_scale` 扩到 4.0、`epochs` 扩到 20/25、`mix_spec` 收窄 —— 这些是**调度器运行期
    的 CLI 覆盖**，不写进模块常量 ⇒ 体检器必须能复现同一套覆盖，否则它会拿旧空间
    判新计划「越界」（假阳性）。
    ⇒ 优先调用调度器自己的 `apply_space_overrides()`（永不两处漂移）；
      调度器还没实现时退回到本地等价实现。
    """
    if not o:
        return
    fn = getattr(M, "apply_space_overrides", None)
    if callable(fn):
        ns = types.SimpleNamespace(**{
            "lr_scale_low": o.get("lr_scale_low"),
            "lr_scale_high": o.get("lr_scale_high"),
            "epoch_choices": o.get("epoch_choices") or "",
            "mix_choices": o.get("mix_choices") or "",
            "init_choices": o.get("init_choices") or "",
            "init_x_choices": o.get("init_x_choices") or "",
        })
        try:
            fn(ns)
            return
        except Exception as exc:  # 调度器版本还没带这个函数 ⇒ 用本地兜底
            print("[CHECK] 调度器的 apply_space_overrides 不可用（%s）⇒ 用本地等价实现" % exc)

    # ---- 本地兜底实现（与调度器的覆盖语义保持一致）----
    lo, hi = o.get("lr_scale_low"), o.get("lr_scale_high")
    if lo is not None or hi is not None:
        base = tuple(M.FOCUS_LR_SCALE)
        M.FOCUS_LR_SCALE = (base[0] if lo is None else float(lo),
                            base[1] if hi is None else float(hi))
    fc = dict(M.FOCUS_CHOICES)
    if o.get("epoch_choices"):
        fc["epochs"] = [int(x) for x in str(o["epoch_choices"]).split(",") if x.strip()]
        M.EPOCH_CHOICES = list(fc["epochs"])
    if o.get("mix_choices"):
        fc["mix_spec"] = [x.strip() for x in str(o["mix_choices"]).split(",") if x.strip()]
        M.MIX_SPECS = list(fc["mix_spec"])
    if o.get("init_choices"):
        fc["init_from"] = [x.strip() for x in str(o["init_choices"]).split(",") if x.strip()]
        M.INIT_CHOICES = list(fc["init_from"])
    if o.get("init_x_choices"):
        M.INIT_X_CHOICES = [x.strip() for x in str(o["init_x_choices"]).split(",") if x.strip()]
    M.FOCUS_CHOICES = fc


def load_space(overrides: dict | None = None) -> tuple[dict, dict, dict, dict]:
    """从调度器模块里读搜索空间（与调度器永远同源，避免手抄漂移）。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import manu.training.optuna_mix_ratio_search as M  # noqa: E402

    M.FOCUS_SPACE = True
    M.EXTENDED_SPACE = True
    apply_overrides(M, overrides or {})
    cat = dict(M.FOCUS_CHOICES)
    # 新参数名 init_from_x 走独立 choices（理由见 suggest_params 的注释）
    cat["init_from_x"] = list(M.INIT_X_CHOICES)
    flt = {
        "lr_scale": M.FOCUS_LR_SCALE,
        "warmup_epochs": (M.WARMUP_EPOCHS_LOW, M.WARMUP_EPOCHS_HIGH),
        "weight_decay": (1e-5, 5e-4),
        "focal_beta": (2.2, 2.9),
        "offset_weight": (0.40, 0.70),
        "mosaic": (0.0, 0.15),
        "scale": (0.10, 0.30),
        "translate": (0.05, 0.15),
        "clip_max_norm": (2.0, 8.0),
    }
    step = {"focal_beta": 0.05, "offset_weight": 0.05, "mosaic": 0.05,
            "scale": 0.05, "translate": 0.05}
    return cat, flt, step, M


def check_arm(params: dict, cat: dict, flt: dict, step: dict) -> list[str]:
    bad: list[str] = []
    for k, choices in cat.items():
        if k not in params:
            continue
        v = params[k]
        if v not in choices:
            bad.append("%s=%r 不在 %s" % (k, v, choices))
    for k, (lo, hi) in flt.items():
        if k not in params:
            continue
        v = float(params[k])
        if not (lo - 1e-9 <= v <= hi + 1e-9):
            bad.append("%s=%r 越界 [%s,%s]" % (k, v, lo, hi))
        st = step.get(k)
        if st:
            frac = (v - lo) / st
            if abs(round(frac) - frac) > 1e-6:
                bad.append("%s=%r 不在 step=%s 的格点上(lo=%s)" % (k, v, st, lo))
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out-root", default="/mnt/data/siping/runs/nogmc_focus")
    ap.add_argument("--study-name", default="mix_ratio_search_nogmc_focus")
    ap.add_argument("--skip-study", action="store_true",
                    help="不连 study（只做空间体检）")
    # ---- 空间覆盖（与调度器的 CLI 同名同义，2026-11-03 新 study 用）----
    ap.add_argument("--lr-scale-low", type=float, default=None)
    ap.add_argument("--lr-scale-high", type=float, default=None)
    ap.add_argument("--epoch-choices", default="")
    ap.add_argument("--mix-choices", default="")
    ap.add_argument("--init-choices", default="")
    ap.add_argument("--init-x-choices", default="")
    args = ap.parse_args()

    overrides = dict(lr_scale_low=args.lr_scale_low, lr_scale_high=args.lr_scale_high,
                     epoch_choices=args.epoch_choices, mix_choices=args.mix_choices,
                     init_choices=args.init_choices, init_x_choices=args.init_x_choices)
    cat, flt, step, M = load_space(overrides)
    print("[CHECK] 生效的搜索空间覆盖：%s" % {k: v for k, v in overrides.items() if v not in (None, "")})
    print("[CHECK] lr_scale 区间=%s | epochs=%s | mix_spec=%s | init_from=%s | init_from_x=%s" % (
        flt["lr_scale"], cat.get("epochs"), cat.get("mix_spec"),
        cat.get("init_from"), cat.get("init_from_x")))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    print("计划文件：%s（%d 条臂）" % (args.plan, len(plan)))

    allowed = {"mix_spec", "lr_scale", "warmup_epochs", "weight_decay", "focal_beta",
               "offset_weight", "mosaic", "scale", "translate", "clip_max_norm", "freeze_bn"}
    allowed |= {"init_from", "init_from_x", "epochs", "backbone_lr_scale",
                "freeze_backbone_epochs"}

    roles = [x.get("role") for x in plan]
    dup = sorted({r for r in roles if roles.count(r) > 1})
    problems: list[str] = []
    if dup:
        problems.append("角色重复：%s" % dup)

    print("\n%-4s %-38s %-10s %s" % ("#", "role", "空间", "问题"))
    n_bad = 0
    for i, arm in enumerate(plan):
        p = arm.get("params") or {}
        bad = check_arm(p, cat, flt, step)
        dropped = sorted(set(p) - allowed)
        if dropped:
            bad.append("调度器会丢掉这些键：%s" % dropped)
        if bad:
            n_bad += 1
            problems.append("%s: %s" % (arm.get("role"), "; ".join(bad)))
        print("%-4d %-38s %-10s %s" % (i, arm.get("role"),
                                       "❌" if bad else "✅",
                                       "; ".join(bad) if bad else ""))

    # ---- 已入队一致性 ----
    if not args.skip_study:
        db = Path(args.out_root) / "study.db"
        if db.exists():
            st = optuna.study.load_study(
                storage="sqlite:///" + str(db), study_name=args.study_name)
            role_trial = {}
            for t in st.trials:
                r = (t.user_attrs or {}).get("role")
                if r:
                    role_trial[r] = t
            print("\n=== 已入队一致性（study 共 %d 个 trial，带 role 的 %d 个）===" % (
                len(st.trials), len(role_trial)))
            enq_roles = {x.get("role") for x in plan} & set(role_trial)
            for r in sorted(enq_roles):
                t = role_trial[r]
                fixed = (t.system_attrs or {}).get("fixed_params") or {}
                plan_p = next(x for x in plan if x.get("role") == r).get("params") or {}
                # 只比「调度器真正入队的键」
                diff = []
                for k in sorted(set(fixed) | set(plan_p)):
                    if k not in allowed:
                        continue
                    a, b = fixed.get(k), plan_p.get(k)
                    if isinstance(a, float) or isinstance(b, float):
                        try:
                            if abs(float(a) - float(b)) > 1e-9:
                                diff.append("%s: study=%r plan=%r" % (k, a, b))
                        except Exception:
                            if a != b:
                                diff.append("%s: study=%r plan=%r" % (k, a, b))
                    elif a != b:
                        diff.append("%s: study=%r plan=%r" % (k, a, b))
                if diff:
                    problems.append("[已入队不一致] %s: %s" % (r, "; ".join(diff)))
                    print("  ❌ %-36s %s" % (r, "; ".join(diff)))
            print("  已入队臂 %d 条：%s" % (
                len(enq_roles),
                "全部逐位一致 ✅" if not any(p.startswith("[已入队不一致]") for p in problems)
                else "存在不一致 ❌"))
            not_enq = {x.get("role") for x in plan} - set(role_trial)
            print("  计划里尚未入队的臂 %d 条（会在调度器重启后按 plan 顺序入队）" % len(not_enq))
        else:
            print("\n（找不到 %s，跳过 study 一致性检查）" % db)

    print("\n" + "=" * 78)
    if problems:
        print("❌ 体检未通过，共 %d 个问题：" % len(problems))
        for p in problems:
            print("   - %s" % p)
        return 1
    print("✅ 体检全部通过：%d 条臂都在搜索空间内，与已入队臂逐位一致" % len(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
