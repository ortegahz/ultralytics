#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在确定性臂全部跑完后**重启调度器**，让新增臂（soup-init 系列）入队。

为什么需要它
------------
调度器在**启动时**把 `--init-weights-map` 读进内存（``optuna_mix_ratio_search.py``
第 1134~1136 行）。新增的 ``soup_strict3`` / ``soup_high3_nobn`` 两个 init key
只有重启后才可见 ⇒ 想跑 soup-init 臂就必须重启一次调度器。

但重启会打断正在跑的 trial ⇒ 它们的指纹仍留在 study 里 ⇒ 重启后会被去重护栏
**跳过**（`existing_sigs` 收录所有状态的 trial）。所以要挑一个"没有臂在跑"的
时机重启。

本脚本的策略
------------
1. 轮询 study.db，等**原确定性臂**（``user_attrs.enqueued_by == "enqueue_plan"``
   且 role 不以 ``FOCUS_SOU`` 开头）**全部到达终态**；
2. 记录当时所有 RUNNING 的 trial 号（多半是 TPE 臂，丢弃无所谓）；
3. 给调度器发 SIGTERM（优雅：停止派新 trial），最多等 120s；不退再 SIGKILL；
4. 等看门狗把调度器拉起来（流水线每 60s 探测一次），最多等 300s；
5. 校验新的 plan 里每一条臂都在 study 里（按 role 查），并打印报告。

它**不删任何 trial**、**不改任何判据**，只做「择时重启 + 校验」。

用法（服务器，``conda activate uav``，根目录 ``/tmp/pycharm_project_10ae9e2e``）：

```bash
nohup python manu/training/restart_dispatcher_inject_arms.py \
  --out-root /mnt/data/siping/runs/nogmc_focus \
  --study-name mix_ratio_search_nogmc_focus \
  --plan manu/training/nogmc_focus_plan.json \
  > /mnt/data/siping/runs/nogmc_focus/logs/restart_inject.log 2>&1 &
```
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import optuna

TERMINAL = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL)


def _plan_roles(plan_path: Path) -> set[str]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return {str(x.get("role", "")) for x in plan if x.get("role")}


def _load(storage: str, study_name: str):
    return optuna.load_study(study_name=study_name, storage=storage)


def main() -> int:
    ap = argparse.ArgumentParser(description="择时重启调度器，让新增臂入队")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--study-name", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--max-wait-minutes", type=int, default=900)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    storage = f"sqlite:///{out_root / 'study.db'}"
    pid_file = out_root / "dispatcher.pid"

    # ---- 单实例锁 ----------------------------------------------------
    # ⚠️ **为什么必须加锁**（2026-11-02 15:31 实测踩到）：
    #    重启 watcher 时旧实例没被杀掉 ⇒ **两个 watcher 同时在跑**。
    #    两者都会在「已入队臂全终态」时触发重启 ⇒
    #    一个 SIGTERM 掉调度器、另一个等看门狗拉起新的、然后**第一个又去 SIGTERM 掉新的**
    #    ⇒ 调度器被反复重启，新增臂可能入队不全/被跳过。
    # ⇒ 用 PID 文件做单实例锁：已有活实例就拒绝启动。
    lock_file = out_root / "restart_inject.pid"
    if lock_file.exists():
        try:
            old_pid = int(lock_file.read_text(encoding="utf-8").strip())
            os.kill(old_pid, 0)
            print(f"[LOCK] 已存在运行中的实例 PID={old_pid} ⇒ 拒绝重复启动"
                  f"（两个 watcher 同时触发会互相把调度器重启两次）")
            return 3
        except (ProcessLookupError, ValueError, OSError):
            lock_file.unlink(missing_ok=True)
    lock_file.write_text(str(os.getpid()), encoding="utf-8")

    all_roles = _plan_roles(Path(args.plan))
    # 原确定性臂 = plan 里除 soup 系列以外的全部（soup 系列是本次要注入的）
    base_roles = {r for r in all_roles if not r.startswith("FOCUS_SOU")}
    injected_roles = {r for r in all_roles if r.startswith("FOCUS_SOU")}

    def say(msg: str) -> None:
        print(f"[{time.strftime('%F %T')}] {msg}", flush=True)

    say(f"目标：等**已入队**的 plan 臂全部到终态后重启调度器（plan 文件运行期会继续增补，"
        f"未入队的新臂不参与等待）")
    say(f"  study={args.study_name} | plan 共 {len(all_roles)} 条 | "
        f"等待集合 = 已入队的 plan 臂 | 重点注入 {sorted(injected_roles)}")

    deadline = time.time() + args.max_wait_minutes * 60
    while True:
        try:
            st = _load(storage, args.study_name)
            trials = st.get_trials(deepcopy=False)
        except Exception as exc:  # study.db 被写锁住等一会儿
            say(f"读 study 失败（忽略）：{exc}")
            time.sleep(args.poll_seconds)
            continue

        base = [t for t in trials
                if (t.user_attrs or {}).get("role") in base_roles
                and (t.user_attrs or {}).get("enqueued_by") == "enqueue_plan"]
        done = [t for t in base if t.state in TERMINAL]
        running = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]
        waiting = [t for t in trials if t.state == optuna.trial.TrialState.WAITING]
        say(f"已入队 plan 臂 {len(done)}/{len(base)} 终态 | RUNNING={len(running)} "
            f"WAITING={len(waiting)} | study 合计 {len(trials)}")
        # ⚠️ 判定的是「**已经入队**的 plan 臂」而不是「plan 文件里所有臂」。
        #    因为 plan 文件会在运行期继续增补（新增的臂还没入队 ⇒ 不该等它们，
        #    否则会死等到超时）。已入队的臂全部终态 ⇒ 就是安全的重启时机。
        if base and len(done) >= len(base):
            break
        if time.time() > deadline:
            say(f"⏰ 等待超时（{args.max_wait_minutes} 分钟），按现状继续")
            break
        time.sleep(args.poll_seconds)

    # ---- 重启前的最后一道保险：确保没有**非 plan 臂**在跑 ----
    #
    # ⚠️ **为什么必须等**（2026-11-02 15:15 实测读码发现）：
    #    调度器收到 SIGTERM 后走「收尾：等在跑的 trial 结束（或杀掉）」分支，
    #    那个分支**只 process.wait()，不收割**（不 study.tell()、不写 summary CSV）。
    #    ⇒ 重启时若还有臂在跑，那条臂的结果会被**永久丢弃**：
    #       · trial 永远停在 RUNNING（backfill 也补不回来，因为没有 user_attrs）
    #       · 它的 GPU 会被新调度器重新分配 ⇒ GPU 争抢 / OOM
    #
    #    上面那个等待循环已经覆盖了「在跑的 **plan 臂**」（它们都在 base 集合里），
    #    但 **TPE 臂**不在 ⇒ 最后一个 plan 臂跑完的瞬间，调度器可能刚好又派了一个 TPE 臂。
    #    ⇒ 这里再等一轮，直到没有非 plan 臂在跑。
    #
    # ⚠️ **等待时长从 60 min 缩到 8 min**（2026-11-02 18:3x 实测改）：
    #    等 TPE 臂等不满 —— TPE 臂一条要 92 min，而 60 min 的等待**必然超时**
    #    ⇒ 白等 1 小时 × 4 卡 = 4 GPU·h（≈ 2.6 条 8ep 臂的产出）。
    #    而这些 TPE 臂**本来就该丢**（watcher 的设计前提：plan 臂优先、TPE 臂可弃）。
    #    ⇒ 改成「等一小会儿让它们自己收尾 → 收不到就杀孤儿」，
    #       把 1 小时 GPU 时间还给 22 条**定点打靶**臂（它们才是本轮的价值所在）。
    wait_deadline = time.time() + 8 * 60
    while True:
        try:
            st = _load(storage, args.study_name)
            trials = st.get_trials(deepcopy=False)
        except Exception:
            time.sleep(args.poll_seconds)
            continue
        busy = [t for t in trials
                if t.state == optuna.trial.TrialState.RUNNING
                and (t.user_attrs or {}).get("role") not in base_roles]
        if not busy:
            say("✅ 没有非 plan 臂在跑 ⇒ 重启不会丢结果")
            break
        if time.time() > wait_deadline:
            say(f"⏰ 仍有 {len(busy)} 个非 plan（TPE）臂在跑，等超时 ⇒ 照常重启"
                f"（这些臂会被**杀孤儿**清理，结果丢弃 —— 它们本来就该让位给 plan 臂）")
            break
        say(f"等待 {len(busy)} 个非 plan 臂收尾（避免重启丢结果）："
            f"{[t.number for t in busy]}")
        time.sleep(args.poll_seconds)

    if args.dry_run:
        say("dry-run：到此为止，不重启")
        lock_file.unlink(missing_ok=True)
        return 0

    # ---- 重启调度器 --------------------------------------------------
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except Exception as exc:
        say(f"读不到 dispatcher.pid（{exc}）⇒ 不重启")
        return 1
    say(f"准备重启调度器 PID={pid}")

    try:
        os.kill(pid, signal.SIGTERM)
        say("已发 SIGTERM（优雅停止派新 trial）")
    except ProcessLookupError:
        say("调度器已不在（可能刚崩过），跳过 SIGTERM")
    except Exception as exc:
        say(f"SIGTERM 失败：{exc}")

    for _ in range(24):  # 最多 120s
        time.sleep(5)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            say("调度器已退出")
            break
    else:
        say("SIGTERM 后仍存活 ⇒ 发 SIGKILL")
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception as exc:
            say(f"SIGKILL 失败：{exc}")
        time.sleep(5)

    # ---- 杀**孤儿 worker**（调度器死了，但它派出去的训练子进程还活着）------
    #
    # ⚠️ **为什么必须杀**（2026-11-02 18:3x 实测读码发现）：
    #    worker 是调度器的 `subprocess.Popen` 子进程 ⇒ **调度器被 SIGKILL 时
    #    子进程不会被连带杀掉**（没有 setsid / 进程组管控）⇒ 它们变成孤儿，
    #    **继续占着 GPU 训练**：
    #      · 新调度器起来后把同一块卡再派一条臂 ⇒ **显存争抢 / OOM**
    #      · 孤儿臂的结果永远回不来（没人收割）⇒ 纯浪费 GPU 时间
    #    ⇒ workers.json 是调度器自己写的登记表（pid / gpu / trial_name）⇒ 照它杀。
    try:
        reg_path = out_root / "workers.json"
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
        workers = reg.get("workers") or {}
        killed = []
        for _key, w in workers.items():
            wpid = int(w.get("pid") or 0)
            if not wpid:
                continue
            try:
                os.kill(wpid, signal.SIGTERM)
                killed.append((w.get("trial_name"), wpid))
            except ProcessLookupError:
                continue
            except Exception as exc:
                say(f"  杀孤儿 worker pid={wpid} 失败：{exc}")
        if killed:
            say(f"已对 {len(killed)} 个孤儿 worker 发 SIGTERM：{killed}")
        time.sleep(5)
        # 还没死的补一刀 SIGKILL
        for _key, w in workers.items():
            wpid = int(w.get("pid") or 0)
            if not wpid:
                continue
            try:
                os.kill(wpid, 0)
            except ProcessLookupError:
                continue
            except Exception:
                continue
            try:
                os.kill(wpid, signal.SIGKILL)
                say(f"  孤儿 worker pid={wpid} 仍存活 ⇒ SIGKILL")
            except Exception:
                pass
    except Exception as exc:
        say(f"清理孤儿 worker 失败（忽略）：{exc}")

    # ---- 等看门狗把它拉起来 -----------------------------------------
    new_pid = None
    for _ in range(60):  # 最多 300s
        time.sleep(5)
        try:
            new_pid = int(pid_file.read_text(encoding="utf-8").strip())
        except Exception:
            continue
        if new_pid != pid:
            try:
                os.kill(new_pid, 0)
                say(f"看门狗已拉起新调度器 PID={new_pid}")
                break
            except ProcessLookupError:
                continue
    if new_pid is None or new_pid == pid:
        say("⚠️ 等待新调度器超时（看门狗可能没在跑）")
        lock_file.unlink(missing_ok=True)
        return 2

    time.sleep(45)  # 等它读完 plan 并入队
    try:
        st = _load(storage, args.study_name)
        trials = st.get_trials(deepcopy=False)
        have = {(t.user_attrs or {}).get("role") for t in trials}
        missing = sorted(all_roles - have)
        say(f"study 现有 {len(trials)} 个 trial；缺失 role：{missing or '无'}")
        for role in sorted(all_roles - base_roles):
            hit = [t for t in trials if (t.user_attrs or {}).get("role") == role]
            if hit:
                t = hit[0]
                say(f"  ✅ {role}: trial#{t.number} state={t.state.value} "
                    f"init={(t.params or {}).get('init_from')} "
                    f"ep={(t.params or {}).get('epochs')}")
            else:
                say(f"  ❌ {role}: 未入队")
        lock_file.unlink(missing_ok=True)
        return 0 if not missing else 3
    except Exception as exc:
        say(f"校验失败：{exc}")
        lock_file.unlink(missing_ok=True)
        return 4


if __name__ == "__main__":
    sys.exit(main())
