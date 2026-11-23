#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Track-B 轨公共口径（2026-11-18 16:30 新增）

**Track-B = 「交付红线内、但严格轨（六项全过）未过」的更强候选轨。**

为什么要开这一轨：
  * 严格轨要求 **T2_fp ≤ 基线(=1)** ⇒ 更强的臂往往只多几个虚警就被判不过；
  * **用户口径红线是 T2_fp ≤ 5**（trial_9863 即以 T2_fp=5 成为 Baseline_v2.0，
    相对 trial_0031 的 FP=10 是「腰斩」⇒ 用户当时明确认可）；
  * 任务书交付红线 T2_fp ≤ 20（官方门禁 ≤ 36）。
  ⇒ Track-B 判据：**nano / nano4 / f1 / t1 / t3 五项全部 ≥ 基线，且至少一项严格优**，
    且 **T2_fp ≤ 5**，且 **板卡 T4 ≤ 19.989 ms**（红线）。

取数口径（关键修复）：
  * **直接读台账**（`arms[tid].picked` / `.official` / `.board_ms`）；
  * **不用** `six_criteria_audit.collect()` —— 它以磁盘 `gate_report.json` 为准，
    合成汤臂（night_soup3 产出）的 nano/nano4/f1 取不到 ⇒ 旧版 Track-B 报告恒为「暂无」。
"""
from __future__ import annotations

import glob
import json
import os
import re

LEDGER = "/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json"
GATE_DIR = "/mnt/data/siping/runs/npu20ms_offcal/gate"
_NAME_RE = re.compile(r"^NANO_(\d+)_ep([0-9.]+)_official$")
_GATE_CACHE: dict = {}


def parse_gate_report(path: str):
    """gate_report.json → {t1,t1_fp,t2,t2_fp,t3,t3_fp,verdict}（与 six_criteria_audit.parse_report 同口径）。"""
    try:
        rep = json.load(open(path, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    tr = rep.get("tracks") or {}
    t1 = (tr.get("track1_uav") or {}).get("finetune_peak") or {}
    t2 = tr.get("track2_golden") or {}
    t3 = tr.get("track3_blind") or {}
    if not t1.get("f1") or not t3.get("f1"):
        return None
    return {
        "t1": float(t1.get("f1", 0)), "t1_fp": int(t1.get("fp", 0)),
        "t2": float(t2.get("f1", 0)), "t2_fp": int(t2.get("fp", 0)),
        "t3": float(t3.get("f1", 0)), "t3_fp": int(t3.get("fp", 0)),
        "verdict": rep.get("verdict"),
    }


def build_gate_map() -> dict:
    """一次扫描整个门禁目录 ⇒ {trial: official 读数}（避免每个臂都 glob 一次）。"""
    out = {}
    for d in glob.glob(f"{GATE_DIR}/NANO_*_official"):
        m = _NAME_RE.match(os.path.basename(d))
        if not m:
            continue
        rp = os.path.join(d, "gate_report.json")
        if not os.path.isfile(rp):
            continue
        g = parse_gate_report(rp)
        if g:
            out.setdefault(int(m.group(1)), g)
    return out


def gate_official(trial: int, ep: str):
    """**直读磁盘门禁报告**取官方三轨（台账可能缺 official 段 ⇒ 只读兜底，无写竞态）。"""
    key = (int(trial), str(ep))
    if key in _GATE_CACHE:
        return _GATE_CACHE[key]
    out = None
    gmap = _GATE_CACHE.get("__map__")
    if gmap is None:
        gmap = build_gate_map()
        _GATE_CACHE["__map__"] = gmap
    out = gmap.get(int(trial))
    _GATE_CACHE[key] = out
    return out
LIMIT_MS = 19.989        # 板卡 T4 硬红线（memory §「19.989 ms」）
T2_USER = 5              # 用户口径红线
T2_DELIVER = 20          # 任务书交付红线
AXES = ("nano", "nano4", "f1", "t1", "t3")


def load_ledger(path: str = LEDGER) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def base_of(led: dict) -> dict:
    b = led.get("baseline_v2") or {}
    return {
        "name": b.get("name"), "trial": b.get("trial"),
        "nano": float(b.get("nano") or 0.0),
        "nano4": float(b.get("nano4") or 0.0),
        "f1": float(b.get("f1") or 0.0),
        "t1": float(b.get("t1") or 0.0),
        "t3": float(b.get("t3") or 0.0),
        "t2_fp": int(b.get("t2_fp") or 0),
        "ms": b.get("ms_new_board"),
    }


def collect_rows(led: dict) -> list:
    """台账 → 每个臂的完整读数行（nano 系取 picked，官方三轨取 official，板卡取 board_ms）。"""
    rows = []
    for tid, st in (led.get("arms") or {}).items():
        pk = st.get("picked") or {}
        of = st.get("official") or {}
        r = {
            "trial": int(tid), "desc": st.get("desc", ""),
            "family": st.get("soup_family"), "weights": st.get("soup_weights"),
            "members": st.get("soup_members"),
            "nano": pk.get("nano"), "nano4": pk.get("nano4"), "f1": pk.get("f1"),
            "micro": pk.get("micro"), "small": pk.get("small"), "regular": pk.get("regular"),
            "t1": of.get("t1"), "t2_fp": of.get("t2_fp"), "t3": of.get("t3"),
            "t1_fp": of.get("t1_fp"), "t3_fp": of.get("t3_fp"),
            "board_ms": st.get("board_ms"), "board_pass": st.get("board_pass_t4"),
            "epoch": st.get("picked_epoch") or 5, "six_ok": st.get("six_ok"),
        }
        if r["t1"] is None or r["t3"] is None or r["t2_fp"] is None:
            # 🔴 台账 official 段可能缺失（老格式臂 / 多写者竞态覆盖）⇒ 直读门禁报告兜底
            g = gate_official(int(tid), r.get("epoch") or 5)
            if g:
                r["t1"] = g["t1"]; r["t1_fp"] = g["t1_fp"]
                r["t2_fp"] = g["t2_fp"]; r["t3"] = g["t3"]; r["t3_fp"] = g["t3_fp"]
        if r["nano"] is None or r["t1"] is None or r["t3"] is None or r["t2_fp"] is None:
            continue
        rows.append(r)
    return rows


def gaps(row: dict, base: dict) -> tuple:
    """(未过基线的轴, 严格胜出的轴) —— 值差为正数。"""
    miss, win = [], []
    for k in AXES:
        v, b = row.get(k), base.get(k)
        if v is None or b is None:
            continue
        if v < b:
            miss.append((k, float(b) - float(v)))
        elif v > b:
            win.append((k, float(v) - float(b)))
    return miss, win


def qualifies(row: dict, base: dict, t2_max: int = T2_USER, require_board: bool = False) -> tuple:
    """是否 Track-B 合格 ⇒ (ok, 未过的轴, 严格胜出的轴)。"""
    miss, win = gaps(row, base)
    if miss:
        return False, miss, win
    if not win:
        return False, miss, win
    if int(row.get("t2_fp") or 99) > int(t2_max):
        return False, [("t2_fp", float(int(row.get("t2_fp") or 99) - int(t2_max)))], win
    if require_board:
        bms = row.get("board_ms")
        # 🔴🔴 R4（reviewer）：本文件是**第三个消费者**，原先只按绝对口径判 ⇒
        #   板卡热漂移（±0.5 ms）下同图臂在 **Track-B 升轨链**仍被误杀
        #   （而 `board_t4_watch` 自己的注释写着「Track-B 桶是今晚唯一可能被升轨的一类」）
        #   ⇒ 两条升轨路径只修好一条 ⇒ **改动的收益可能为零**。
        #   修法：台账的 `board_pass_t4` 已是「绝对 OR 同会话相对」的**综合判定**
        #   （相对口径用同会话基线臂读数抵消漂移）⇒ 有它就以它为准；
        #   缺该字段（老臂）才退回绝对红线。本模块**不引入新依赖**（判据在 `t4_criterion.py`）。
        _bp = row.get("board_pass")
        if _bp is not None:
            if (not bool(_bp)) or bms is None:
                return False, [("ms", float(LIMIT_MS) + 1.0)], win
        elif bms is None or float(bms) > LIMIT_MS:
            return False, [("ms", float(LIMIT_MS) + 1.0)], win
    return True, miss, win


def family_votes(rows: list, family, base: dict, t2_max: int = T2_USER) -> int:
    """同家族（成员集 + 权重桶）里 Track-B 合格的臂数 ⇒ 种子复现票。"""
    if not family:
        return 0
    n = 0
    for r in rows:
        if r.get("family") != family:
            continue
        ok, _, _ = qualifies(r, base, t2_max)
        if ok:
            n += 1
    return n


def fmt_delta(v, b):
    return "%+.2e" % (float(v) - float(b))
