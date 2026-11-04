#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""往**正在运行**的 study 里补投臂 / 剪掉低价值臂（2026-11-03，T3 前沿轮专用）。

为什么需要它
============
本轮的 30 条臂是**启动时一次性入队**的（入队顺序 = 执行顺序 = 价值顺序）。
但 30 条臂要跑 17.6 小时，**第一批结果在 12:30 左右就会出来** ⇒
到那时我们才知道「`epochs` 到底涨不涨 T3」，而此时队列里还排着 20+ 条按
「事前判断」排的臂。

⇒ 需要一个**不动正在跑的臂、不重启调度器**的调队列手段：
  · **补投**：直接往 study.db 里 `enqueue_trial` ⇒ 调度器下次 `ask()` 就会取走
    （Optuna 的 `ask()` 优先弹 WAITING 队列，按 trial number 升序 ⇒
    **新臂的 trial number 更大 ⇒ 排在队尾**，不能插到中间）。
  · **剪掉**：把指定编号的 **WAITING** 臂标成 PRUNED ⇒ 调度器不会再取它，
    且 `done_count` 立刻 +1 ⇒ 腾出的位置正好给新臂 ⇒ **总臂数不变**。

⚠️ **代价要说清楚**：
  · 补投的臂**只能排在队尾**，不能插队。要插队只能重启调度器（风险高，本轮不用）。
  · 剪掉的是**事前判断为低价值**的臂（默认是队尾的 ep 阶梯补点）⇒ 剪错了不可逆
    （Optuna 4.9 没有 `delete_trial` ⇒ 删不掉；标 PRUNED 后指纹仍在去重表里
    ⇒ **重新入队同参同种子会被跳过** ⇒ 想再跑它必须换 seed）。

用法（服务器，conda `uav`，项目根 /tmp/pycharm_project_10ae9e2e）：
```bash
# 只看队列现状（只读，安全）
python manu/training/enqueue_extra_arms.py --out-root /mnt/data/siping/runs/nogmc_t3 \
  --study-name mix_ratio_search_nogmc_t3 --status

# 剪掉最后 2 条低价值臂 + 补投 2 条新臂
python manu/training/enqueue_extra_arms.py --out-root /mnt/data/siping/runs/nogmc_t3 \
  --study-name mix_ratio_search_nogmc_t3 \
  --prune-last 2 \
  --plan-extra manu/training/nogmc_t3_extra_plan.json
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

TERMINAL = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL)


def _cfg_sig(d: dict, seed=None) -> tuple:
    """与调度器**逐位一致**的去重指纹（改一处就会重复入队 ⇒ 白烧 GPU 槽）。"""
    sig = []
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, bool):
            sig.append((k, bool(v)))
        elif isinstance(v, (int, float)):
            sig.append((k, round(float(v), 6)))
        else:
            sig.append((k, str(v)))
    sig.append(("__seed__", None if seed is None else int(seed)))
    return tuple(sig)


def load_study(out_root: Path, study_name: str):
    return optuna.load_study(study_name=study_name,
                              storage="sqlite:///" + str(out_root / "study.db"))


def show_status(st, plan_path: Path | None) -> None:
    trials = st.get_trials(deepcopy=False)
    print("=== study 现状 ===")
    cnt = {}
    for t in trials:
        cnt[t.state.name] = cnt.get(t.state.name, 0) + 1
    print("  状态分布：%s（合计 %d）" % (cnt, len(trials)))
    print("  %-6s %-30s %-9s %-8s %-7s %s" % ("trial", "role", "state", "mix", "lr", "ep"))
    for t in trials:
        role = (t.user_attrs or {}).get("role", "")
        p = t.params or {}
        if not p:
            fx = (t.system_attrs or {}).get("fixed_params") or {}
            p = fx
        print("  t%-5d %-30s %-9s %-8s %-7s %s" % (
            t.number, role[:30], t.state.name,
            str(p.get("mix_spec", ""))[:8], str(p.get("lr_scale", ""))[:7],
            p.get("epochs", "")))
    if plan_path and plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        have = {(t.user_attrs or {}).get("role") for t in trials}
        missing = [x.get("role") for x in plan if x.get("role") not in have]
        print("  计划里尚未入队的臂：%s" % (missing or "无"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--study-name", required=True)
    ap.add_argument("--plan", default="", help="原计划文件（只用于状态显示）")
    ap.add_argument("--plan-extra", default="", help="要补投的臂（JSON 数组，格式同原计划）")
    ap.add_argument("--prune-last", type=int, default=0,
                    help="剪掉队尾 N 条 **WAITING** 臂（标 PRUNED）")
    ap.add_argument("--prune-roles", default="", help="逗号分隔的 role，指定要剪的臂")
    ap.add_argument("--status", action="store_true", help="只打印现状，不做任何修改")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    st = load_study(out_root, args.study_name)

    if args.status or (not args.plan_extra and not args.prune_last and not args.prune_roles):
        show_status(st, Path(args.plan) if args.plan else None)
        return 0

    # ---- ① 剪 ----
    if args.prune_last or args.prune_roles:
        roles = {r.strip() for r in args.prune_roles.split(",") if r.strip()}
        waiting = [t for t in st.get_trials(deepcopy=False)
                   if t.state == optuna.trial.TrialState.WAITING]
        waiting.sort(key=lambda t: t.number)
        targets = []
        if args.prune_last:
            targets += waiting[-args.prune_last:] if args.prune_last else []
        if roles:
            targets += [t for t in waiting
                        if (t.user_attrs or {}).get("role", "") in roles]
        seen = set()
        for t in targets:
            if t.number in seen:
                continue
            seen.add(t.number)
            role = (t.user_attrs or {}).get("role", "")
            try:
                st.tell(t.number, state=optuna.trial.TrialState.PRUNED)
                print("[PRUNE] t%-4d role=%-28s ⇒ PRUNED（调度器不会再取它）" % (t.number, role))
            except Exception as exc:
                print("[PRUNE] t%-4d 失败：%s" % (t.number, exc))

    # ---- ② 补投 ----
    if args.plan_extra:
        extra = json.loads(Path(args.plan_extra).read_text(encoding="utf-8"))
        allowed = {"mix_spec", "lr_scale", "warmup_epochs", "weight_decay", "focal_beta",
                   "offset_weight", "mosaic", "scale", "translate", "clip_max_norm", "freeze_bn",
                   "init_from", "init_from_x", "epochs", "backbone_lr_scale",
                   "freeze_backbone_epochs"}
        existing = set()
        for t in st.get_trials(deepcopy=False):
            pr = dict(t.params or {})
            if not pr:
                pr = dict((getattr(t, "system_attrs", None) or {}).get("fixed_params") or {})
            if not pr:
                continue
            try:
                existing.add(_cfg_sig(pr, (t.user_attrs or {}).get("seed")))
            except Exception:
                continue
        n = 0
        for item in extra:
            p = {k: v for k, v in dict(item.get("params") or {}).items() if k in allowed}
            seed = item.get("seed")
            sig = _cfg_sig(p, seed)
            if sig in existing:
                print("[ENQ] 跳过（已存在同参同种子）role=%s" % item.get("role"))
                continue
            ua = {"role": str(item.get("role", "")), "enqueued_by": "enqueue_plan"}
            if seed is not None:
                ua["seed"] = int(seed)
            for attempt in range(5):
                try:
                    st.enqueue_trial(p, skip_if_exists=False, user_attrs=ua)
                    existing.add(sig)
                    n += 1
                    print("[ENQ] 入队 #%d role=%-28s mix=%s lr=%s ep=%s seed=%s" % (
                        n, ua["role"], p.get("mix_spec"), p.get("lr_scale"),
                        p.get("epochs"), seed))
                    break
                except Exception as exc:
                    # ⚠️ 与调度器并发写 sqlite 可能撞锁 ⇒ 重试（不重试就会丢这条臂）
                    print("[ENQ] 入队失败（第 %d 次）：%s ⇒ 重试" % (attempt + 1, exc))
                    time.sleep(3 + attempt * 2)
            else:
                print("[ENQ] ❌ role=%s 最终失败（数据库锁？）" % ua["role"])
        print("[ENQ] 共补投 %d 条（**排在队尾**，trial number 更大）" % n)

    show_status(st, Path(args.plan) if args.plan else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
