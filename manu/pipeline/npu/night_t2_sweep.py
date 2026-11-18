#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2 虚警**分层扫描**（§196 追加4）——把「t2_fp 与 sw / cp 的关系」量化出来。

为什么必须单独做（实测依据）
--------------------------
· 官方门禁单臂 **9.5 min**（实测 1100275：23:05:33 → 23:15:00）⇒ 394 臂全门禁 ≈ 62 h，不可行；
· runner 只对「三轴全过」或「近失(min-margin ≥ -0.3σ)」的臂跑门禁 ⇒ **样本有偏**（只覆盖最好的臂）
  ⇒ 无法回答「哪些 (sw, cp) 组合能保 t2_fp ≤ 1」；
· 而这恰恰是今晚策略的核心：昨夜唯一三轴全过臂 1100275（sw4.2）**只差在 t2_fp=3**。

做法
----
**分层抽样**：按 (sw, cp_prob) 分格，每格取 nano 最高、边际最近的 1~2 个臂跑门禁
⇒ 用 ≈20 次门禁（单卡 ≈3 h）覆盖 sw∈{4.0,4.1,4.2} × cp∈{0,0.05,0.1,…} ⇒ 产出映射表。

与训练轴的关系
--------------
独立进程、**不重启训练轴**、只用 1 张卡（GPU3）；读台账选臂 → 跑门禁 → 回写台账 `official/six_ok`
（写前重新 load，尽量避开与训练 runner 的写冲突）→ 结果同时落到 `$W/t2_sweep_results.json`。

用法（服务器）：
    python -u manu/pipeline/npu/night_t2_sweep.py --gpu 3 --max-gates 20 --deadline "2026-11-18 07:00"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = "/tmp/pycharm_project_10ae9e2e"
W = "/mnt/data/siping/runs/npu20ms/nano_search"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "manu", "pipeline", "npu"))
os.chdir(REPO)

from manu.pipeline.npu import nano_search as NS  # noqa: E402

OUT = f"{W}/t2_sweep_results.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--max-gates", type=int, default=20)
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--deadline", default="2026-11-18 07:00")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要门禁的臂，不真跑")
    ap.add_argument("--min-margin", type=float, default=-2.0,
                    help="只对边际 ≥ 该值的臂做门禁（太远没意义）")
    args = ap.parse_args()
    dl = time.mktime(time.strptime(args.deadline, "%Y-%m-%d %H:%M"))

    base = NS.BASE_V2
    done_n = 0
    gated_cells: dict[tuple, int] = {}

    def scan():
        """扫描台账 ⇒ 按 (sw, cp) 分格挑候选（每格优先覆盖未测过的）。"""
        led = NS.load_ledger()
        arms = led.get("arms") or {}
        cells: dict[tuple, list] = {}
        for key, st in arms.items():
            picked = st.get("picked") or {}
            if not picked or "nano" not in picked:
                continue
            p = st.get("params") or {}
            sw = p.get("sw")
            cp = p.get("cp_prob") or 0.0
            marg = min((picked.get("nano", 0) - base["nano"]) / 0.0042,
                       (picked.get("nano4", 0) - base["nano4"]) / 0.0060,
                       (picked.get("f1", 0) - base["f1"]) / 0.0008)
            if marg < args.min_margin:
                continue
            if st.get("official"):
                continue          # 已经有门禁读数
            try:
                t = int(key)
            except Exception:  # noqa: BLE001
                continue
            cells.setdefault((sw, float(cp)), []).append((marg, t, st, picked))
        todo = []
        for cell, lst in cells.items():
            # 格内优先选「边际最近」的臂（最接近过线的候选）⇒ T2 读数对「能否过线」最有参考价值
            lst.sort(key=lambda x: (-x[0], -x[3].get("nano", 0.0)))
            # 覆盖优先：未测过的格先出，格内再按 nano 高者
            for item in lst[: args.per_cell]:
                todo.append((cell, item))
        todo.sort(key=lambda x: (gated_cells.get(x[0], 0), x[0][0] is None, x[0]))
        return todo, len(cells)

    while time.time() < dl and done_n < args.max_gates:
        todo, ncell = scan()
        if not todo:
            log(f"[T2扫描] 暂无未门禁候选 ⇒ 300 s 后重扫（已测格 {len(gated_cells)}）")
            time.sleep(300)
            continue
        log(f"[T2扫描] 本轮候选 {len(todo)}（格 {ncell}，已覆盖格 {len(gated_cells)}）"
            f"，累计已测 {done_n}/{args.max_gates}，GPU{args.gpu}")
        if args.dry_run:
            for cell, (marg, t, st, picked) in todo[: args.max_gates]:
                log(f"  · (sw={cell[0]}, cp={cell[1]}) trial {t} ep{picked.get('epoch')} "
                    f"nano={picked.get('nano'):.6f} nano4={picked.get('nano4'):.6f} "
                    f"F1={picked.get('f1'):.6f} m={marg:+.3f}σ")
            return
        for cell, (marg, t, st, picked) in todo[: max(1, args.max_gates - done_n)]:
            if time.time() > dl or done_n >= args.max_gates:
                break
            log(f"[T2扫描] ▶ 门禁 trial {t} ep{picked.get('epoch')}（sw={cell[0]}, cp={cell[1]}）"
                f" nano={picked.get('nano'):.6f} nano4={picked.get('nano4'):.6f} F1={picked.get('f1'):.6f} m={marg:+.3f}σ")
            try:
                off = NS.run_official_gate(t, picked.get("epoch"), args.gpu)
            except Exception as exc:  # noqa: BLE001
                log(f"[T2扫描] 门禁异常：{exc}")
                continue
            # 回写台账（写前重新 load，尽量不覆盖训练 runner 的更新）
            led2 = NS.load_ledger()
            st2 = NS.arm_state(led2, t)
            st2["official"] = off
            st2["six_ok"] = bool(
                off.get("t1") is not None and off.get("t3") is not None
                and (picked.get("nano", 0)) > base["nano"]
                and (picked.get("nano4", 0)) > base["nano4"]
                and (picked.get("f1", 0)) > base["f1"]
                and off.get("t1", 0) >= base["t1"]
                and off.get("t2_fp", 99) <= base["t2_fp"]
                and off.get("t3", 0) >= base["t3"])
            NS.save_ledger(led2)
            try:
                with open(OUT, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"ts": time.strftime("%F %T"), "trial": t,
                                         "ep": picked.get("epoch"), "sw": cell[0], "cp": cell[1],
                                         "nano": picked.get("nano"), "nano4": picked.get("nano4"),
                                         "f1": picked.get("f1"), "marg": marg,
                                         "t1": off.get("t1"), "t2_fp": off.get("t2_fp"),
                                         "t3": off.get("t3"), "six_ok": st2.get("six_ok")},
                                    ensure_ascii=False) + "\n")
            except Exception as exc:  # noqa: BLE001
                log(f"[T2扫描] 结果落盘失败：{exc}")
            log(f"[T2扫描] ✔ trial {t} ⇒ t1={off.get('t1')} t2_fp={off.get('t2_fp')} t3={off.get('t3')} "
                f"six_ok={st2.get('six_ok')}")
            done_n += 1
            gated_cells[cell] = gated_cells.get(cell, 0) + 1
    log(f"[T2扫描] 收工：完成 {done_n} 次门禁，覆盖格 {len(gated_cells)}")


if __name__ == "__main__":
    main()
