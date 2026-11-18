#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SWA 候选 → 台账 **增量登记旁路**（2026-11-13 新增）。

═══ 为什么需要它 ═══
``fine_swa_sweep.py`` 在**每个候选评测完**后会调 ``write_report`` + ``register_synth_arms``，
所以理论上 SWA 候选会进台账。但有两个现实缺口：

  ① **进程代码快照**：``fine_swa_loop.sh`` 起的 python 进程在启动时就把源码读进内存
     ⇒ 之后 scp 到服务器的新代码（含 ``register_synth_arms``）对该进程无效
     ⇒ 只要这个长跑进程不退出，SWA 候选就**一直不进台账**。
  ② **单轮时长**：一轮 597 个候选按实测 ~1.5~2 min/个 ≈ 15~20 h
     ⇒ 即使新代码生效，也要等这一轮跑完。

缺口后果（致命）：
  · ``six_criteria_audit`` 按「台账 nanobench × 磁盘 gate_report」配对
    ⇒ 台账里没有该臂 ⇒ **六项全过的 SWA 候选被判成不存在**；
  · ``board_t4_watch`` 用同一口径 ⇒ **不会被自动测板卡 T4 时延**；
  · ``adaptive_runner`` 的抛光锚点选择也看不见 ⇒ 近失臂拿不到抛光机会。
  ⇒ 真出「beat」时，明早的审计会**静默漏掉**它。

═══ 本脚本做什么 ═══
轮询 ``fine_swa_report.json``（扫描器每评完一个候选就原子写一次 ⇒ 新条目会及时出现），
把**已经跑完官方三轨门禁**的 triad 候选补登记进台账。
登记口径**完全复用** ``fine_swa_sweep.register_synth_arms``（同一份代码 ⇒ 同一口径）。

用法（服务器）：
    python manu/pipeline/npu/swa_ledger_sync.py --interval 120
    python manu/pipeline/npu/swa_ledger_sync.py --once      # 单次同步
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

DEFAULT_REPORT = "/mnt/data/siping/runs/npu20ms/nano_search/fine_swa_report.json"
DEFAULT_LEDGER = "/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def _repo_root() -> str:
    cur = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(cur, "manu", "pipeline", "npu")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return "/tmp/pycharm_project_10ae9e2e"
        cur = parent


def sync_once(report_path: str, ledger_path: str) -> int:
    """同步一次；返回本次新登记的臂数。"""
    if not os.path.isfile(report_path):
        log(f"报告不存在（跳过）：{report_path}")
        return 0
    try:
        with open(report_path, encoding="utf-8") as f:
            rep = json.load(f)
    except Exception as exc:  # noqa: BLE001
        log(f"报告读取失败（跳过）：{exc}")
        return 0
    cands = [c for c in (rep.get("candidates") or []) if isinstance(c, dict) and c.get("key")]
    if not cands:
        return 0

    for p in (_repo_root(), "/tmp/pycharm_project_10ae9e2e"):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    try:
        from manu.pipeline.npu.fine_swa_sweep import register_synth_arms  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        log(f"导入 register_synth_arms 失败（跳过）：{exc}")
        return 0

    # 只挑「台账里还没有」的候选 ⇒ 平时零写盘，避免和调度器抢台账
    try:
        with open(ledger_path, encoding="utf-8") as f:
            led = json.load(f)
        arms = led.get("arms") or led
    except Exception as exc:  # noqa: BLE001
        log(f"台账读取失败（跳过）：{exc}")
        return 0

    fresh: dict[str, dict] = {}
    for c in cands:
        sid = c.get("synth_id")
        if sid is None:
            continue
        if str(int(sid)) in arms:
            continue
        g = c.get("gate") or {}
        if g.get("error") or g.get("t1") is None or g.get("t3") is None:
            continue          # 官方门禁还没跑完 ⇒ 先不登记
        if not c.get("triad"):
            continue          # triad 才可能过六项
        fresh[c["key"]] = c
    if not fresh:
        return 0
    n = register_synth_arms(fresh, ledger_path)
    if n:
        sids = sorted(int(c["synth_id"]) for c in fresh.values())
        log(f"补登记 {n} 个 SWA 候选进台账：synth_id={sids}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="SWA 候选 → 台账 增量登记旁路")
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--ledger", default=DEFAULT_LEDGER)
    ap.add_argument("--interval", type=int, default=120, help="轮询间隔（秒）")
    ap.add_argument("--deadline", default="2026-11-16 08:30", help="到期退出（本地时间）")
    ap.add_argument("--once", action="store_true", help="只同步一次就退出")
    args = ap.parse_args()

    log(f"启动：report={args.report} ledger={args.ledger} interval={args.interval}s")
    dl = None
    if args.deadline:
        try:
            dl = time.mktime(time.strptime(args.deadline, "%Y-%m-%d %H:%M"))
        except Exception:  # noqa: BLE001
            dl = None
    total = 0
    while True:
        try:
            total += sync_once(args.report, args.ledger)
        except Exception as exc:  # noqa: BLE001
            log(f"同步异常（继续）：{type(exc).__name__}: {exc}")
        if args.once:
            log(f"单次同步完成，共登记 {total} 个")
            return
        if dl is not None and time.time() > dl:
            log(f"到达截止 {args.deadline} ⇒ 退出（共登记 {total} 个）")
            return
        time.sleep(max(10, args.interval))


if __name__ == "__main__":
    main()
