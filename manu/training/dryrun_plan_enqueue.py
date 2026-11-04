#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调度器入队 + suggest 干跑验证（2026-11-02 新增）。

为什么需要它
------------
调度器重启后会在**运行期**做两件事：
① `--enqueue-plan` 把计划文件里的臂入队；
② 每次 `study.ask()` 之后调用 `suggest_params(trial)` —— 这一步会
   **校验固定参数是否落在搜索空间内**，越界就抛 ValueError ⇒ 臂当场崩。

这两步在**入队时完全看不出来**（`enqueue_trial` 不校验），
只有真跑到那条臂才会炸。本脚本在**study 的副本**上把这两步完整跑一遍，
从而在重启之前就知道：
  · 23 条已入队臂会不会被重复入队；
  · 18 条新臂能不能全部入队；
  · 每条臂 ask+suggest 会不会抛异常；
  · init_from_x（soup 权重）这条新路径能不能走通。

**绝不在真 study 上跑**（会污染正式结果）。

用法（服务器，conda `uav`，项目根 /tmp/pycharm_project_10ae9e2e）：
```bash
python manu/training/dryrun_plan_enqueue.py \
  --plan manu/training/nogmc_focus_plan.json \
  --out-root /mnt/data/siping/runs/nogmc_focus \
  --study-name mix_ratio_search_nogmc_focus
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import manu.training.optuna_mix_ratio_search as M  # noqa: E402


def _apply_overrides(M, o: dict) -> None:
    """把调度器同款的空间覆盖应用到模块常量（与 check_plan_vs_search_space.py 同语义）。

    ⚠️ 优先调用调度器自己的 `apply_space_overrides()`，保证干跑用的空间与真跑**逐位相同**。
    """
    if not o:
        return
    fn = getattr(M, "apply_space_overrides", None)
    if callable(fn):
        import types
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
        except Exception as exc:
            print("[DRYRUN] 调度器的 apply_space_overrides 不可用（%s）⇒ 用本地等价实现" % exc)
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


def _cfg_sig(d: dict, seed=None) -> tuple:
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


def backup_db(src: Path, dst: Path) -> None:
    """sqlite 在线备份（含 WAL）⇒ 拿到一致快照。"""
    s = sqlite3.connect(str(src))
    d = sqlite3.connect(str(dst))
    s.backup(d)
    d.close()
    s.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out-root", default="/mnt/data/siping/runs/nogmc_focus")
    ap.add_argument("--study-name", default="mix_ratio_search_nogmc_focus")
    ap.add_argument("--copy", default="/tmp/_nogmc_focus_dryrun.db")
    # ---- 空间覆盖（与调度器的 CLI 同名同义，2026-11-03 新 study 用）----
    ap.add_argument("--lr-scale-low", type=float, default=None)
    ap.add_argument("--lr-scale-high", type=float, default=None)
    ap.add_argument("--epoch-choices", default="")
    ap.add_argument("--mix-choices", default="")
    ap.add_argument("--init-choices", default="")
    ap.add_argument("--init-x-choices", default="")
    args = ap.parse_args()

    M.FOCUS_SPACE = True
    M.EXTENDED_SPACE = True
    overrides = dict(lr_scale_low=args.lr_scale_low, lr_scale_high=args.lr_scale_high,
                     epoch_choices=args.epoch_choices, mix_choices=args.mix_choices,
                     init_choices=args.init_choices, init_x_choices=args.init_x_choices)
    _apply_overrides(M, overrides)
    print("[DRYRUN] 生效的搜索空间覆盖：%s" % {k: v for k, v in overrides.items() if v not in (None, "")})

    src = Path(args.out_root) / "study.db"
    dst = Path(args.copy)
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(dst) + suffix)
        if p.exists():
            p.unlink()
    if src.exists():
        backup_db(src, dst)
        print("✅ 已备份 study 快照 → %s（真 study 未被触碰）" % dst)
    else:
        # ⚠️ **新 study 还没有 study.db**（2026-11-03）：这时要在副本里**新建**一个
        #    空 study，否则干跑无从下手（load_study 会抛「study 不存在」）。
        d = sqlite3.connect(str(dst))
        d.close()
        print("✅ 源 study.db 不存在（新 study）⇒ 在副本里新建空 study")
    try:
        st = optuna.load_study(storage="sqlite:///" + str(dst), study_name=args.study_name)
    except Exception:
        # ⚠️ **新 study 还没有 study.db**（2026-11-03）：副本里先建一个同名空 study，
        #    否则 load_study 抛「study 不存在」⇒ 干跑无从下手。
        st = optuna.create_study(study_name=args.study_name,
                                  storage="sqlite:///" + str(dst),
                                  direction="maximize")
        print("✅ 副本里新建空 study（%s）" % args.study_name)
    print("   快照里 %d 个 trial" % len(st.trials))

    # ---- 复刻调度器的 enqueue 逻辑 ----
    allowed = {"mix_spec", "lr_scale", "warmup_epochs", "weight_decay", "focal_beta",
               "offset_weight", "mosaic", "scale", "translate", "clip_max_norm", "freeze_bn"}
    allowed |= {"init_from", "init_from_x", "epochs", "backbone_lr_scale",
                "freeze_backbone_epochs"}

    existing_sigs = set()
    for t in st.get_trials(deepcopy=False):
        try:
            pr = dict((getattr(t, "system_attrs", None) or {}).get("fixed_params") or {})
            if not pr:
                pr = dict(t.params or {})
            seed = (getattr(t, "user_attrs", None) or {}).get("seed")
            existing_sigs.add(_cfg_sig(pr, seed))
        except Exception:
            continue

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    # ⚠️ 必须在 enqueue **之前**数 WAITING：入队后 WAITING 会 +n_enq，
    #    再加一次就会多 ask 一轮 ⇒ 凭空造 TPE trial（第一版踩过，造了 391 条）。
    pre_waiting = len([t for t in st.get_trials(deepcopy=False)
                        if t.state == optuna.trial.TrialState.WAITING])
    n_enq = n_skip = 0
    enq_roles = []
    for item in plan:
        raw = dict(item.get("params") or {})
        p = {k: v for k, v in raw.items() if k in allowed}
        sig = _cfg_sig(p, item.get("seed"))
        if sig in existing_sigs:
            n_skip += 1
            continue
        ua = {"role": str(item.get("role", "")), "enqueued_by": "enqueue_plan"}
        if item.get("seed") is not None:
            ua["seed"] = int(item["seed"])
        st.enqueue_trial(p, skip_if_exists=False, user_attrs=ua)
        existing_sigs.add(sig)
        n_enq += 1
        enq_roles.append(item.get("role"))
    print("\n[PLAN] 干跑：注入 %d 个臂（跳过已存在 %d 个）" % (n_enq, n_skip))
    print("       新入队：%s" % ", ".join(enq_roles))

    # ---- 复刻调度器的 ask + suggest_params ----
    # ⚠️ **必须按 trial 数封顶**：`study.ask()` 在 WAITING 用尽后会**新建 TPE trial**
    #    （不封顶 ⇒ 干跑时凭空造出几百条 TPE trial，纯浪费；只污染副本，
    #    但输出会被淹没）。封顶数 = 入队前的 WAITING 数 + 本次新入队数。
    n_to_ask = pre_waiting + n_enq
    print("\n=== 逐条 ask + suggest_params 干跑（验证不会抛 ValueError，共 %d 条）===" % n_to_ask)
    role_of = {(t.user_attrs or {}).get("role"): t for t in st.get_trials(deepcopy=False)
               if (t.user_attrs or {}).get("role")}
    n_ok = 0
    failures = []
    for _ in range(n_to_ask):
        try:
            t = st.ask()
        except Exception as exc:
            failures.append(("ask", type(exc).__name__, str(exc)[:200]))
            break
        if t is None:
            break
        role = role_of.get(t.number, "t%d" % t.number)
        try:
            params = M.suggest_params(t)
        except Exception as exc:
            failures.append((role, type(exc).__name__, str(exc)[:300]))
            st.tell(t, state=optuna.trial.TrialState.FAIL)
            continue
        init_key = str(params.get("init_from_x") or params.get("init_from") or "")
        # 核对 suggest 回来的值和固定参数一致
        fx = (t.system_attrs or {}).get("fixed_params") or {}
        for k, v in fx.items():
            if k in params and params[k] != v:
                failures.append((role, "MISMATCH", "%s: plan=%r suggest=%r" % (k, v, params[k])))
        st.tell(t, 0.0)
        n_ok += 1
        print("   ✅ %-36s t%-4d init=%-14s mix=%-8s lr=%-9s ep=%-3s"
              % (role, t.number, init_key, params.get("mix_spec"),
                 params.get("lr_scale"), params.get("epochs")))

    print("\n" + "=" * 78)
    print("干跑结果：成功 %d 条，失败 %d 条" % (n_ok, len(failures)))
    if failures:
        print("❌ 有失败：")
        for f in failures:
            print("   - %s | %s | %s" % f)
        return 1
    print("✅ 全部臂都能入队并通过 suggest_params ⇒ 调度器重启后不会崩")
    print("   （新臂 %d 条，已入队臂 %d 条跳过）" % (n_enq, n_skip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
