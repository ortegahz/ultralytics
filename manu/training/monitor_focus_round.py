#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定点打靶轮次**常驻监控**：把关键事件追加写进 events.log。

为什么需要它
------------
本轮要跑到 11-03 06:30，中间 8+ 小时没有人工干预 ⇒ 必须有**不依赖交互**的
过程留痕，否则只能靠"临时起意去查"。

它做什么（**只读**，不碰调度器/训练/GPU）：
1. 每 5 分钟读一次 study.db，统计各状态的 trial 数；
2. 每个 trial **首次到达终态**时，把它的 role / 三轨指标 / verdict 追加进 events.log；
3. 每 15 分钟写一次 status 快照（供本地轮询读）。

用法（服务器 ``conda activate uav``）：

```bash
nohup setsid python manu/training/monitor_focus_round.py \
  --out-root /mnt/data/siping/runs/nogmc_focus \
  --study-name mix_ratio_search_nogmc_focus \
  >> /mnt/data/siping/runs/nogmc_focus/logs/monitor.log 2>&1 < /dev/null &
```
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import optuna

TERMINAL = ("COMPLETE", "PRUNED", "FAIL")


def _f(v):
    try:
        return float(v)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="定点打靶常驻监控")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--study-name", required=True)
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--stop-at", default="", help="到该时刻（服务器本地时间）后退出，如 '2026-11-03 06:45'")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    storage = f"sqlite:///{out_root / 'study.db'}"
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    events = log_dir / "events.log"
    status = log_dir / "monitor_status.json"

    seen: set[int] = set()
    last_snap = 0.0
    print(f"[MON] 启动 | study={args.study_name} | events={events}", flush=True)

    while True:
        if args.stop_at:
            try:
                if time.strftime("%Y-%m-%d %H:%M") >= args.stop_at:
                    print("[MON] 到达停止时刻，退出", flush=True)
                    return 0
            except Exception:
                pass
        try:
            st = optuna.load_study(study_name=args.study_name, storage=storage)
            trials = st.get_trials(deepcopy=False)
            counts: dict[str, int] = {}
            for t in trials:
                counts[t.state.name] = counts.get(t.state.name, 0) + 1

            # ---- 新终态事件 ------------------------------------------
            for t in trials:
                if t.state.name not in TERMINAL or t.number in seen:
                    continue
                seen.add(t.number)
                ua = t.user_attrs or {}
                role = ua.get("role") or "-"
                # ⚠️ `FrozenTrial` **没有** `name` 属性（那是活 `Trial` 的）⇒ 用编号拼
                #    （调度器的 trial 命名规则就是 trial_0000 这种 4 位零填充）。
                tname = f"trial_{t.number:04d}"
                trf = out_root / "trials" / tname / "trial_result.json"
                m, verdict, util = {}, "", None
                try:
                    res = json.loads(trf.read_text(encoding="utf-8"))
                    mm = res.get("metrics") or {}
                    m = {
                        "t1": _f(mm.get("track1_f1")), "t1_fp": mm.get("track1_fp"),
                        "t2": _f(mm.get("track2_f1")), "t2_fp": mm.get("track2_fp"),
                        "t3": _f(mm.get("track3_f1")), "t3_fp": mm.get("track3_fp"),
                    }
                    verdict = res.get("gate_verdict") or ""
                    util = _f(res.get("utility"))
                except Exception:
                    verdict = "（无 trial_result.json）"
                line = (f"[{time.strftime('%F %T')}] {tname} [{t.state.name}] role={role} "
                        f"verdict={verdict} T1={m.get('t1')} T2={m.get('t2')}(FP {m.get('t2_fp')}) "
                        f"T3={m.get('t3')} U={util}")
                with events.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                print(line, flush=True)

            # ---- 快照 ------------------------------------------------
            if time.time() - last_snap > 900:
                last_snap = time.time()
                snap = {
                    "at": time.strftime("%F %T"),
                    "counts": counts,
                    "n_trials": len(trials),
                    "n_terminal_seen": len(seen),
                }
                status.write_text(json.dumps(snap, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
                print(f"[MON] {snap['at']} counts={counts}", flush=True)
        except Exception as exc:
            print(f"[MON] 轮询异常（忽略）：{exc}", flush=True)

        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
