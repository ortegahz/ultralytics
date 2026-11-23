#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Track-B 升轨守望（2026-11-18 16:45 新增）

每 10 分钟扫一次台账：
  · 找出**五项精度指标全 ≥ 当前基线且至少一项严格优**、**T2_fp ≤ 5**、
    **板卡 T4 ≤ 19.989 ms**、**同家族复现票 ≥ 2** 的臂；
  · 若其中 nano 严格优于当前基线 ⇒ **自动升轨**（Track-B 轨，严格轨旧基线留档可回退）；
  · 每轮把状态写 `$W/trackb_state.json`（晨报直接读它 ⇒ 用户 09:00 一眼看到结论）。

与严格轨守望（auto_promote_watch）**并存**：两条轨各自独立升轨，谁更严格谁先改基线；
Track-B 升轨时把严格轨旧基线存 `baseline_v2_strict_backup` ⇒ 随时可回退。

用法：
  $PY manu/pipeline/npu/trackb_promote_watch.py --watch --interval 600 --deadline "2026-11-19 08:30"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from trackb_core import (LIMIT_MS, T2_USER, base_of, collect_rows, family_votes,  # noqa: E402
                         load_ledger, qualifies)

REPO = "/tmp/pycharm_project_10ae9e2e"
RUNS = "/mnt/data/siping/runs/npu20ms/nano_search"
PY = "/home/huangzhe/anaconda3/envs/uav/bin/python"
LOG = os.path.join(RUNS, "trackb_promote_watch.log")


def log(msg: str) -> None:
    line = "[%s] %s" % (time.strftime("%F %T"), msg)
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def next_name(cur: str) -> str:
    m = re.match(r"Baseline_v2\.(\d+)", str(cur or ""))
    return "Baseline_v2.%d" % (int(m.group(1)) + 1 if m else 1)


def run_once() -> None:
    led = load_ledger()
    base_t = base_of(led)
    rows = collect_rows(led)
    cands = []
    for r in rows:
        ok, miss, win = qualifies(r, base_t, T2_USER, require_board=True)
        if ok:
            cands.append((r, win))
    cands.sort(key=lambda x: -(x[0].get("nano") or 0))
    # 状态文件（晨报读它）
    state = {
        "ts": time.strftime("%F %T"),
        "baseline": base_t,
        "n_trackb_qualified": len(cands),
        "candidates": [
            {"trial": r["trial"], "nano": r["nano"], "nano4": r.get("nano4"), "f1": r.get("f1"),
             "t1": r.get("t1"), "t2_fp": r.get("t2_fp"), "t3": r.get("t3"),
             "board_ms": r.get("board_ms"), "family": r.get("family"),
             "votes": family_votes(rows, r.get("family"), base_t, T2_USER),
             "win": win}
            for r, win in cands[:12]
        ],
    }
    try:
        with open(os.path.join(RUNS, "trackb_state.json"), "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass
    if not cands:
        log("[扫描] Track-B 合格候选 0 个（基线 %s）" % base_t["name"])
        return
    top, win = cands[0]
    log("[扫描] Track-B 合格 %d 个｜基线 %s nano=%.7f｜最优 trial=%d nano=%.7f "
        "nano4=%.7f F1=%.7f T1=%.7f T2fp=%s T3=%.7f 板卡=%s 票=%d"
        % (len(cands), base_t["name"], base_t["nano"], top["trial"], top["nano"],
           top.get("nano4") or 0, top.get("f1") or 0, top.get("t1") or 0, top.get("t2_fp"),
           top.get("t3") or 0, top.get("board_ms"),
           family_votes(rows, top.get("family"), base_t, T2_USER)))
    if not (top["nano"] > base_t["nano"]):
        log("[结论] 最优候选 nano 未严格优于基线 ⇒ 不动基线")
        return
    name = next_name(base_t.get("name"))
    log("[升轨] 试候选 trial=%d ⇒ %s" % (top["trial"], name))
    cmd = [PY, "manu/pipeline/npu/promote_trackb.py", "--trial", str(top["trial"]),
           "--name", name, "--t2-max", str(T2_USER), "--min-family", "2"]
    try:
        pr = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
    except Exception as exc:  # noqa: BLE001
        log("[升轨] 异常：%s" % exc)
        return
    log("[升轨] rc=%s\n%s\n%s" % (pr.returncode, (pr.stdout or "")[-1500:], (pr.stderr or "")[-400:]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--deadline", default="2026-11-19 08:30")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    if args.once or not args.watch:
        run_once()
        return
    end = time.mktime(time.strptime(args.deadline, "%Y-%m-%d %H:%M"))
    log("[启动] Track-B 升轨守望 interval=%ds deadline=%s" % (args.interval, args.deadline))
    while time.time() < end:
        try:
            run_once()
        except Exception as exc:  # noqa: BLE001
            log("[异常] %s" % exc)
        time.sleep(max(60, args.interval))


if __name__ == "__main__":
    main()
