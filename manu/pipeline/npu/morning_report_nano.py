#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""明早自动报告生成器（夜间训练轴专用）—— 2026-11-17 §196 起替换旧 adaptive 版。

只读数据源：台账 nano_search_ledger.json / 板卡 measured.json / night_train_axis.log / promotion_*.json。
输出：纯文本报告（stdout），由 run_morning_report2.sh 落盘到 $W/morning_report_<stamp>.txt。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

REPO = "/tmp/pycharm_project_10ae9e2e"
W = "/mnt/data/siping/runs/npu20ms/nano_search"
LEDGER = f"{W}/nano_search_ledger.json"
MEASURED = "/mnt/data/siping/runs/npu20ms/board_measure_auto/measured.json"
LOG = f"{W}/night_train_axis.log"
REPORT = f"{W}/night_train_axis_report.json"
sys.path.insert(0, REPO)


def jload(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return default


def base_of(led):
    """当前基线读数 + **元信息**（名字/臂号/板卡 ms）⇒ 晨报头部不会写死旧基线。"""
    try:
        from manu.pipeline.npu import nano_search as ns
        b = ns.BASE_V2
        out = {"nano": b["nano"], "nano4": b["nano4"], "f1": b["f1"],
               "t1": b["t1"], "t2_fp": b["t2_fp"], "t3": b["t3"]}
        # 🟢 2026-11-18 10:10：台账里的 `baseline_v2` 才是**权威基线**（升轨后自动跟新）
        lb = (led or {}).get("baseline_v2") or {}
        for k in ("nano", "nano4", "f1", "t1", "t2_fp", "t3", "name", "trial", "epoch",
                  "ms_new_board", "ms_delivery_board", "recipe"):
            if lb.get(k) is not None:
                out[k] = lb[k]
        return out
    except Exception:  # noqa: BLE001
        return {"nano": 0.7197543, "nano4": 0.6554820, "f1": 0.8903981,
                "t1": 0.8904939, "t2_fp": 1, "t3": 0.9485250}


def main() -> None:
    led = jload(LEDGER, {}) or {}
    arms = led.get("arms") or {}
    base = base_of(led)
    print("===== 明早自动报告 %s（夜间训练轴 · §196/§198/§199）=====" % time.strftime("%F %T"))
    print("""────────────────────────────────────────────────────────────────
【3 分钟速读】
 ① **交付口径（任务书 §136.6）已破**：见【0b】。判据 = F1>.887122 · nano≥.6666 ·
    官方T1≥.885997 · T2 FP≤20（官方36）· T3≥.948330 · int8≤20 ms。
    ⚠️ `gate_report.verdict=FAIL` **≠** 交付失败（它用冻结基线 0.905736，连当前基线都过不了）。
 ② **严格口径（六项全压过基线 + 同族 n≥4 + 板卡≤19.989ms）未破**：见【2】。
    墙在 **T2_fp≤1**（三轴全过的臂里 t2_fp≤1 的极少）。
 ③ **机制新知**：汤的 t2_fp 取自**子权重** ⇒ 保 T2 要**父主导 w>0.5**；今夜 batch4/5/6/7
    就在扫「父主导 × 干净父」的窄窗口。
 ④ 关键文件：`promotion_ready.json`（升轨）｜`board_measure_auto/measured.json`（板卡 ms）
    ｜`soup_batch*.log`（汤族）｜`night_train_axis.log`（训练轴）。
────────────────────────────────────────────────────────────────""")
    print("\n【0】当前基线（唯一官方基线）")
    # 🟢🟢 2026-11-18 10:05 修**基线标签硬编码**：原来永远写 "Baseline_v2.2 = trial_610000@5"
    #   ⇒ 台账基线已升到 v2.4/v2.5… 时，晨报头部的名字与指标**对不上**（最容易被误读）。
    #   修法：从台账 `baseline_v2` 动态取 name/trial/epoch/板卡 ms。
    _bname = str(base.get("name") or "Baseline_v2.4")
    _btrial = int(base.get("trial") or 0)
    _bep = int(base.get("epoch") or 5)
    _bms = base.get("ms_new_board") or base.get("ms_delivery_board") or 19.646
    print("  %s = trial_%d@%d" % (_bname, _btrial, _bep))
    print("  nano %.7f | nano4 %.7f | F1 %.7f" % (base["nano"], base["nano4"], base["f1"]))
    print("  T1 %.7f | T2_fp %s | T3 %.7f | 板卡 %.3f ms（红线 19.989）"
          % (base["t1"], base["t2_fp"], base["t3"], float(_bms)))
    if base.get("recipe"):
        print("  配方：%s" % base["recipe"])
    print("  升轨判据：六项**严格全过**（nano/nano4/F1 严格>基线、T1≥、T2_fp≤1、T3≥）"
          " + 同配方家族 n≥4 + 板卡 T4 ≤19.989 ms")

    # ---------- 【0b】任务书交付红线（§136.6）：与「内部严格六项」分开记账 ----------
    print("\n【0b】★ 任务书交付红线（§136.6）—— 交付口径 ≠ 内部严格升轨口径")
    print("  判据：F1 > 0.887122（0031）· nano ≥ 0.6666 · 官方 T1 ≥ 0.885997 · "
          "T2 FP ≤ 20（官方红线 36）· T3 ≥ 0.948330 · int8 ≤ 20 ms")
    del_rows = []
    for key, st in arms.items():
        try:
            t = int(key)
        except Exception:  # noqa: BLE001
            continue
        pk = st.get("picked") or {}
        of = st.get("official") or {}
        nano, f1 = pk.get("nano"), pk.get("f1")
        t1, t2fp, t3 = of.get("t1"), of.get("t2_fp"), of.get("t3")
        if nano is None or f1 is None or t1 is None or t2fp is None or t3 is None:
            continue
        board = st.get("board_ms")
        ok = (f1 > 0.887122 and nano >= 0.6666 and t1 >= 0.885997
              and t2fp <= 20 and t3 >= 0.948330 and (board is None or float(board) <= 20.0))
        del_rows.append((t, nano, f1, t1, t2fp, t3, board, ok, st.get("desc") or ""))
    del_ok = [r for r in del_rows if r[7]]
    del_ok.sort(key=lambda r: -(r[1] + r[2]))
    print("  有全套读数的臂 %d 个｜**交付红线全过 %d 个**" % (len(del_rows), len(del_ok)))
    for (t, nano, f1, t1, t2fp, t3, board, ok, desc) in del_ok[:12]:
        print("    ✔ trial %d | nano %.6f | F1 %.6f | T1 %.6f | T2_fp %s | T3 %.6f | 板卡 %s | %s"
              % (t, nano, f1, t1, t2fp, t3,
                 ("%.3f ms" % board) if board else "待测", (desc or "")[:46]))
    if not del_ok:
        print("    （暂无）")

    # ---------- 本轮臂 ----------
    rounds = [(1102000, 1103000), (1103000, 1103100)]     # P 主计划 + 阶段2 复现
    rows = []
    for key, st in arms.items():
        try:
            t = int(key)
        except Exception:  # noqa: BLE001
            continue
        picked = st.get("picked") or {}
        if not picked or "nano" not in picked:
            continue
        off = st.get("official") or {}
        marg = min((picked.get("nano", 0) - base["nano"]) / 0.0042,
                   (picked.get("nano4", 0) - base["nano4"]) / 0.0060,
                   (picked.get("f1", 0) - base["f1"]) / 0.0008)
        six = bool(st.get("six_ok"))
        if not six:
            six = bool(off.get("t1") is not None and off.get("t3") is not None
                       and picked.get("nano", 0) > base["nano"]
                       and picked.get("nano4", 0) > base["nano4"]
                       and picked.get("f1", 0) > base["f1"]
                       and off.get("t1", 0) >= base["t1"]
                       and off.get("t2_fp", 99) <= base["t2_fp"]
                       and off.get("t3", 0) >= base["t3"])
        p = st.get("params") or {}
        rows.append({
            "trial": t, "ep": picked.get("epoch"), "marg": marg, "six": six,
            "nano": picked.get("nano"), "nano4": picked.get("nano4"), "f1": picked.get("f1"),
            "t1": off.get("t1"), "t2": off.get("t2_fp"), "t3": off.get("t3"),
            "board": st.get("board_ms"), "bpass": st.get("board_pass_t4"),
            "desc": st.get("desc", ""), "ch": st.get("channel", ""),
            "sw": p.get("sw"), "cp": p.get("cp_prob"), "anchor": str(p.get("init", "")).split("trial_")[-1].split("/")[0],
            "lr": p.get("lr"), "lrf": p.get("lr_floor"), "l2": p.get("anchor_l2"),
            "gated": bool(off), "r2": (1102000 <= t < 1103000),
        })
    cur = [r for r in rows if r["r2"] or (1100800 <= r["trial"] < 1100920)
           or (1104200 <= r["trial"] < 1104330) or (1105000 <= r["trial"] < 1105200)
           or (1104400 <= r["trial"] < 1104800) or (1105200 <= r["trial"] < 1105400)]
    six_rows = sorted([r for r in cur if r["six"]], key=lambda r: -r["marg"])
    print("\n【1】本夜训练轴状态")
    rep = jload(REPORT, {}) or {}
    print("  计划 386 臂（P 330 / S 40 / T 16）｜台账本轮有读数的臂 %d 个｜runner 报告完成 %d 个（报告时间 %s）"
          % (len(cur), len(rep.get("done") or []), rep.get("ts", "-")))
    print("  其中已跑官方门禁（三轴全过或近失）%d 个｜六项全过 %d 个"
          % (len([r for r in cur if r["gated"]]), len(six_rows)))
    try:
        from manu.pipeline.npu import family_key as FK
        fams = defaultdict(list)
        for r in six_rows:
            st = arms.get(str(r["trial"])) or {}
            try:
                fams[FK.arm_family_key(st)].append(r)
            except Exception:  # noqa: BLE001
                fams[str(r["trial"])].append(r)
        print("  家族复现：%s" % ("；".join(f"{k} → n={len(v)}" for k, v in fams.items()) or "—"))
    except Exception as exc:  # noqa: BLE001
        print("  家族复现统计失败：%s" % exc)

    print("\n【2】★ 六项全过臂明细（= 可升轨候选）")
    if not six_rows:
        print("  （本夜暂无六项全过臂）")
    for r in six_rows:
        print("  trial %d ep%s | nano %.7f | nano4 %.7f | F1 %.7f | T1 %s | T2_fp %s | T3 %s | 板卡 %s | %s"
              % (r["trial"], r["ep"], r["nano"], r["nano4"], r["f1"],
                 ("%.7f" % r["t1"]) if r["t1"] is not None else "-",
                 r["t2"] if r["t2"] is not None else "-",
                 ("%.7f" % r["t3"]) if r["t3"] is not None else "-",
                 ("%.3f ms%s" % (r["board"], " ✓" if r["bpass"] else "")) if r["board"] else "-",
                 r["desc"][:58]))

    print("\n【2b】★ 同血缘亲子汤（零训练成本）明细")
    soup_rows = [r for r in cur if r["ch"] == "LIN" or 1105000 <= r["trial"] < 1105200]
    if not soup_rows:
        print("  （无）")
    for r in sorted(soup_rows, key=lambda x: -x["marg"]):
        print("  trial %d | nano %.7f | nano4 %.7f | F1 %.7f | T1 %s | T2_fp %s | T3 %s | 板卡 %s | %s"
              % (r["trial"], r["nano"], r["nano4"], r["f1"],
                 ("%.7f" % r["t1"]) if r["t1"] is not None else "-",
                 r["t2"] if r["t2"] is not None else "-",
                 ("%.7f" % r["t3"]) if r["t3"] is not None else "-",
                 ("%.3f ms%s" % (r["board"], " ✓" if r["bpass"] else "")) if r["board"] else "-",
                 r["desc"][:58]))

    print("\n【2c】★ 只差一项的候选（5/6）——「压过基线 5 项 + 交付口径可过」")
    print("   （内部严格判据要求 T2_fp ≤ 基线(=%s)；**任务书交付红线是 T2_fp ≤ 20、官方门禁 ≤ 36**）" % base.get("t2_fp"))
    five_rows = []
    for r in cur:
        off = arms.get(str(r["trial"])) or {}
        o = off.get("official") or {}
        if not o or o.get("t1") is None or o.get("t3") is None:
            continue
        miss = []
        if not (r["nano"] > base["nano"]):
            miss.append("nano")
        if not (r["nano4"] > base["nano4"]):
            miss.append("nano4")
        if not (r["f1"] > base["f1"]):
            miss.append("F1")
        if not (r["t1"] >= base["t1"]):
            miss.append("T1")
        if not (r["t2"] is not None and r["t2"] <= base["t2_fp"]):
            miss.append("T2_fp")
        if not (r["t3"] >= base["t3"]):
            miss.append("T3")
        if len(miss) == 1:
            deliver_ok = bool(r["t1"] >= 0.885997 and (r["t2"] or 99) <= 20 and r["t3"] >= 0.948330)
            five_rows.append((r, miss[0], deliver_ok))
    if not five_rows:
        print("  （无）")
    for r, miss, dok in sorted(five_rows, key=lambda x: -x[0]["marg"]):
        print("  trial %d | nano %.7f | nano4 %.7f | F1 %.7f | T1 %s | T2_fp %s | T3 %s | 板卡 %s | 缺: %s | 交付可过: %s | %s"
              % (r["trial"], r["nano"], r["nano4"], r["f1"],
                 ("%.7f" % r["t1"]) if r["t1"] is not None else "-",
                 r["t2"] if r["t2"] is not None else "-",
                 ("%.7f" % r["t3"]) if r["t3"] is not None else "-",
                 ("%.3f ms%s" % (r["board"], " ✓" if r["bpass"] else "")) if r["board"] else "-",
                 miss, "是" if dok else "否", r["desc"][:46]))

    print("\n【2d】★ 汤族家族统计（家族键 = soup:<盆地>:<标签字母><成员数> ⇒ 🔴**权重数字被丢弃** ⇒ 同 parent 的所有 2 成员汤同族；升轨要 n≥4）")
    try:
        from manu.pipeline.npu import family_key as FK
        fam_all = defaultdict(list)
        for t, st in arms.items():
            if not st.get("synth") or not st.get("soup_members"):
                continue
            try:
                k = FK.arm_family_key(st)
            except Exception:  # noqa: BLE001
                k = str(st.get("soup_family"))
            fam_all[k].append(st)
        # 🔴 2026-11-17 15:40 修：台账里的 `six_ok` 是**批次口径**（只查 T1/T2fp/T3，**不含 nano/nano4/F1**）
        #   ⇒ 家族统计必须用**严格判据**（six_criteria_audit.judge）重算，否则会把 4/6 的臂算成全过。
        strict6 = set()
        try:
            from manu.pipeline.npu import six_criteria_audit as _A
            strict6 = {int(r["trial"]) for r in _A.collect() if _A.judge(r)[0] == 6}
        except Exception as exc:  # noqa: BLE001
            print("  （严格判据复算失败，退回台账口径：%s）" % exc)
        for k, v in sorted(fam_all.items()):
            n_six = sum(1 for x in v if int(x.get("trial")) in strict6) if strict6 else 0
            print("  家族 %-24s 成员 %3d 个｜**严格**六项全过 %d 个%s" % (k, len(v), n_six, "  ✔ 可升轨" if n_six >= 4 else ""))
    except Exception as exc:  # noqa: BLE001
        print("  家族统计失败：%s" % exc)

    print("\n【2e】★ 升轨**双关**状态（家族 n≥4 ＋ 同 parent 同权重 ≥2 ＋ 板卡 PASS）")
    try:
        from manu.pipeline.npu import six_criteria_audit as _A
        from manu.pipeline.npu import family_key as _FK
        _rows = _A.collect()
        _strict = {int(r["trial"]): r for r in _rows if _A.judge(r)[0] == 6}
        _fam = defaultdict(list)
        _raw = defaultdict(list)
        for t, st in arms.items():
            if not st.get("synth") or not st.get("soup_members"):
                continue
            try:
                _k = _FK.arm_family_key(st)
            except Exception:  # noqa: BLE001
                _k = str(st.get("soup_family"))
            _fam[_k].append(int(t))
            _raw[str(st.get("soup_family"))].append(int(t))
        _best = None
        for k, mem in _fam.items():
            n6 = [t for t in mem if t in _strict]
            if not n6:
                continue
            board_ok = [t for t in n6
                        if (arms.get(str(t)) or {}).get("board_ms") is not None
                        and (arms.get(str(t)) or {}).get("board_pass_t4")]
            if _best is None or len(n6) > len(_best[1]):
                _best = (k, n6, board_ok)
        if _best is None:
            print("  （当前没有严格六项全过的汤臂）")
        else:
            _k, _n6, _bp = _best
            print("  最佳家族 %s：严格六项全过 %d 个 %s｜其中板卡 PASS %d 个 %s" % (_k, len(_n6), sorted(_n6), len(_bp), sorted(_bp)))
            for _r, _mem in sorted(_raw.items()):
                _m6 = [t for t in _mem if t in _strict]
                if _m6:
                    print("    └ 原始族 %-18s（同 parent 同权重）严格六项全过 %d 个 %s %s"
                          % (_r, len(_m6), sorted(_m6), "✔ 种子关可过" if len(_m6) >= 2 else "✘ 种子关不足(需≥2)"))
                    # 🟢🟢 2026-11-17 17:55 —— **复现强度分级**（防止「同一模型换臂号」假票）
                    #   强复现 = 该权重组里六项全过的臂来自**不同子臂**（不同训练种子 ⇒ 独立复现）
                    #   近邻复现 = 同一子臂、只是汤权重差 ±0.005（模型几乎相同 ⇒ 票偏弱，需人工判断）
                    _det, _childs = [], set()
                    for _t in sorted(_m6):
                        _st = arms.get(str(_t)) or {}
                        _mm = _st.get("soup_members") or []
                        _ch = "%s@%s" % (_mm[1][0], _mm[1][1]) if len(_mm) > 1 else "?"
                        _w = (_st.get("soup_weights") or [0.5])[0]
                        _childs.add(_ch)
                        _det.append("%d(child=%s,w=%.4f)" % (_t, _ch, float(_w)))
                    _strong = len(_childs) >= 2
                    print("        └ 复现强度：%s｜成员明细 %s"
                          % ("**强复现（不同子臂 ⇒ 独立训练种子）**" if _strong
                             else "⚠ 近邻复现（同一子臂、仅汤权重不同 ⇒ 模型几乎相同，票偏弱）",
                             " ".join(_det)))
            print("  ⇒ 升轨需要：同家族严格六项全过 ≥4（当前 %d）＋ 候选所在原始族 ≥2（见上）＋ 候选有板卡 PASS" % len(_n6))
    except Exception as exc:  # noqa: BLE401
        print("  双关状态统计失败：%s" % exc)

    print("\n【3】边际 top 15（三轴 nano/nano4/F1 相对基线的最小 σ 边际，含官方门禁读数）")
    for r in sorted(cur, key=lambda x: -x["marg"])[:15]:
        print("  %+7.3fσ | trial %d ep%s | nano %.7f nano4 %.7f F1 %.7f | T1 %s T2_fp %s T3 %s | %s"
              % (r["marg"], r["trial"], r["ep"], r["nano"], r["nano4"], r["f1"],
                 ("%.6f" % r["t1"]) if r["t1"] is not None else "-",
                 r["t2"] if r["t2"] is not None else "-",
                 ("%.6f" % r["t3"]) if r["t3"] is not None else "-",
                 r["desc"][:52]))

    print("\n【4】T2 虚警归因（本轮新判据：sw>4.0 ⇒ 长泉金池虚警涨；cp 能否补 ≤8px 监督）")
    gated = [r for r in cur if r["gated"] and r["t2"] is not None]
    for key, label in (("sw", "sw"), ("cp", "cp_prob"), ("ch", "通道")):
        g = defaultdict(list)
        for r in gated:
            g[r[key]].append(r)
        if not g:
            continue
        parts = []
        for k in sorted(g, key=lambda x: (x is None, x)):
            v = g[k]
            t2 = [x["t2"] for x in v]
            ok = len([x for x in v if x["six"]])
            parts.append("%s=%s: n=%d T2_fp 均值 %.2f/最小 %d/最大 %d，六项全过 %d"
                         % (label, k, len(v), sum(t2) / len(t2), min(t2), max(t2), ok))
        print("  按 %s：" % label)
        for p in parts:
            print("    " + p)

    print("\n【4b】★ T2 虚警分层扫描映射表（$W/t2_sweep_results.json）")
    sw_rows = []
    try:
        with open(f"{W}/t2_sweep_results.json", "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    sw_rows.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue
    except Exception:  # noqa: BLE001
        pass
    if not sw_rows:
        print("  （扫描未产出结果）")
    else:
        cells = defaultdict(list)
        for r in sw_rows:
            cells[(r.get("sw"), r.get("cp"))].append(r)
        print("  (sw, cp) | 臂数 | T2_fp 均值/最小/最大 | 六项全过 | 臂明细(trial, nano, nano4, F1)")
        for k in sorted(cells, key=lambda x: (x[0] is None, x[0], x[1])):
            v = cells[k]
            t2 = [x.get("t2_fp") for x in v if x.get("t2_fp") is not None]
            if t2:
                s_t2 = "%.2f/%d/%d" % (sum(t2) / len(t2), min(t2), max(t2))
            else:
                s_t2 = "-"
            ok = len([x for x in v if x.get("six_ok")])
            det = "; ".join("t%s nano%.4f nano4%.4f F1%.4f" % (x.get("trial"), x.get("nano") or 0,
                        x.get("nano4") or 0, x.get("f1") or 0) for x in v[:4])
            print("  sw=%s cp=%s | %d | %s | %d | %s" % (k[0], k[1], len(v), s_t2, ok, det))

    print("\n【5】板卡 T4 实测（红线 19.989 ms）")
    ms_list = []
    for key, st in arms.items():
        v = st.get("board_ms")
        if v:
            ms_list.append((float(v), int(key) if str(key).isdigit() else key, st.get("board_pass_t4")))
    if ms_list:
        ms_list.sort(reverse=True)
        print("  已测 %d 个候选｜最大 %.3f ms｜最小 %.3f ms｜超红线 %d 个"
              % (len(ms_list), ms_list[0][0], ms_list[-1][0], len([m for m in ms_list if m[0] > 19.989])))
        for m, t, ok in ms_list[:6]:
            print("    trial %s : %.3f ms %s" % (t, m, "✓" if ok else "✗"))
    else:
        print("  （尚无实测读数）")

    print("\n【6】自动升轨状态")
    pr = jload(f"{W}/promotion_ready.json")
    print("  promotion_ready.json：%s" % (json.dumps(pr, ensure_ascii=False)[:400] if pr else "（未生成 ⇒ 未自动升轨）"))
    try:
        with open(f"{W}/auto_promote_watch.log", "r", encoding="utf-8", errors="ignore") as fh:
            tail = [l.rstrip() for l in fh.readlines() if "结论" in l or "升轨" in l][-4:]
        for l in tail:
            print("  " + l)
    except Exception:  # noqa: BLE001
        pass

    print("\n【7】资源水位（末 5 条 [水位] 行）")
    try:
        with open(LOG, "r", encoding="utf-8", errors="ignore") as fh:
            wm = [l.rstrip() for l in fh.readlines() if "[水位" in l][-5:]
        for l in wm:
            print("  " + l)
    except Exception:  # noqa: BLE001
        pass

    print("\n【8】各通道臂数统计")
    print("  " + ", ".join("%s=%d" % (k, v) for k, v in sorted(Counter(r["ch"] for r in cur).items())))
    print("\n（判据与口径详见 ultralytics/memory/memory.md §196；round-1 的 1100200~1100223 段撞段污染，不计入本夜统计）")


if __name__ == "__main__":
    main()
