#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 **WAITING** 臂标成 PRUNED（Optuna 4.9 的 `study.tell()` 不允许 tell WAITING 臂）。

为什么需要它
------------
`enqueue_extra_arms.py` 的剪枝功能（`st.tell(t.number, state=PRUNED)`）在 Optuna 4.9
会抛 `ValueError: Cannot tell a WAITING trial.` ⇒ **剪不掉**。

而本轮（2026-11-06）**必须**剪掉一批臂：
· 队列里有 15 条「p2_fused 160 输出」臂，但实测证据（memory/hc32_deadmodel_truth_20261106.md §9）
  显示 **160 输出的 ΔT1 = −1.60 pp，是种子噪声带（−0.19~−0.33 pp）的 5 倍** ⇒ 结构性失败
  ⇒ 剪掉低价值的窄头 160 输出臂，把预算换成 late（320 输出）家族的多种子臂。

做法
----
直接走底层 storage 的 `set_trial_state_values`（Optuna 内部 API）⇒ 把 WAITING trial 的
state 改成 PRUNED ⇒ dispatcher 的 `study.ask()` 不会再取它（它只取 WAITING 队列）。

⚠️ **安全护栏**（本脚本自带，防误伤）：
  1. 只剪**显式列举**的 role（不做前缀匹配 —— 前缀 `S_p2_fused_hc32_relu_` 会把
     **late 家族**一起匹配上！实测踩过）；
  2. **拒绝剪任何 late 家族臂**（`late_upsample=True`）⇒ late 是唯一有希望的家族；
  3. **拒绝剪 RUNNING 臂**（在跑的臂不能动）；
  4. 剪之前打印清单、剪之后复核数量。

用法（服务器，conda uav）：
    python manu/training/prune_waiting_trials.py \
        --out-root /mnt/data/siping/runs/npu20ms \
        --study-name mix_ratio_search_npu20ms_v2 \
        --roles-file /mnt/data/siping/runs/npu20ms/prune_roles.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

optuna = None
TrialState = None


def _need():
    global optuna, TrialState
    if optuna is None:
        import optuna as _o
        optuna = _o
        TrialState = _o.trial.TrialState
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--study-name", required=True)
    ap.add_argument("--roles-file", required=True,
                    help="JSON：{\"prune_roles\": [role, ...]} 或 [role, ...]")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    o = _need()
    out_root = Path(args.out_root)
    db = out_root / "study.db"

    raw = json.loads(Path(args.roles_file).read_text(encoding="utf-8"))
    roles = set(raw["prune_roles"] if isinstance(raw, dict) else raw)

    st = o.load_study(study_name=args.study_name, storage=f"sqlite:///{db.resolve()}")
    storage = st._storage  # 底层 RDBStorage（load_study 返回的 Study 有它）

    targets = []
    for t in st.get_trials(deepcopy=False):
        role = (t.user_attrs or {}).get("role", "")
        if role not in roles:
            continue
        if t.state != o.trial.TrialState.WAITING:
            print(f"[SKIP] t{t.number} role={role} 状态是 {t.state.name}（只剪 WAITING）")
            continue
        # ★ 安全护栏：拒绝剪 late 家族
        fx = (t.system_attrs or {}).get("fixed_params") or {}
        if bool(fx.get("late_upsample")):
            print(f"[SKIP] t{t.number} role={role} 是 **late 家族** ⇒ 拒绝剪（唯一有希望的家族）")
            continue
        targets.append(t)

    if not targets:
        print("[PRUNE] 没有可剪的 WAITING 臂")
        return 0

    print(f"[PRUNE] 将剪 {len(targets)} 条 WAITING 臂：")
    for t in targets:
        print(f"   t{t.number:<4} role={(t.user_attrs or {}).get('role','')}")

    if not args.yes:
        ans = input("确认？输入 yes 继续：").strip().lower()
        if ans != "yes":
            print("[PRUNE] 已取消")
            return 1

    ok = 0
    for t in targets:
        try:
            # 底层 API：直接改 trial 的 state（Optuna 4.9 的 study.tell 不允许 WAITING）
            # ⚠️ 签名是 set_trial_state_values(trial_id, state, values=None)
            #    —— state 要**直接传枚举**（传元组会报 "'tuple' object has no attribute 'is_finished'"）。
            storage.set_trial_state_values(t._trial_id, TrialState.PRUNED)
            ok += 1
        except Exception as exc:
            print(f"[PRUNE] t{t.number} 失败：{exc}")

    # 复核
    st2 = o.load_study(study_name=args.study_name, storage=f"sqlite:///{db.resolve()}")
    from collections import Counter
    print(f"[PRUNE] 成功剪 {ok} 条；状态分布：{dict(Counter(t.state.name for t in st2.get_trials(deepcopy=False)))}")
    still = [(t.number, (t.user_attrs or {}).get("role", "")) for t in st2.get_trials(deepcopy=False)
             if (t.user_attrs or {}).get("role", "") in roles and t.state.name == "WAITING"]
    if still:
        print(f"[PRUNE] ⚠️ 仍有 {len(still)} 条目标臂处于 WAITING：{still[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
