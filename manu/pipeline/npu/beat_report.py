#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Beat-to-Pass 结论汇总：把台账里**打过 Baseline_v2.0** 的臂整理成一份可直读的 md。

用法（服务器）：
  /home/huangzhe/anaconda3/envs/uav/bin/python manu/pipeline/npu/beat_report.py
输出：
  /mnt/data/siping/runs/npu20ms/nano_search/beat_to_pass_report.md
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nano_search as ns  # noqa: E402

OUT = f"{ns.WORK}/beat_to_pass_report.md"


def fmt(val, spec: str = ".6f", dash: str = "-") -> str:
    """None 安全的格式化。"""
    if val is None:
        return dash
    try:
        return format(val, spec)
    except (TypeError, ValueError):
        return str(val)


def main() -> None:
    led = ns.load_ledger()
    base = dict(ns.BASE_V2)
    a98 = (led.get("arms") or {}).get(str(ns.BASE_V2["trial"]), {})
    pk98 = a98.get("picked") or {}
    of98 = a98.get("official") or {}
    for k in ("nano", "micro", "small", "regular", "f1", "nano4"):
        if pk98.get(k) is not None:
            base[k] = pk98[k]
    for k in ("t1", "t1_fp", "t2", "t2_fp", "t3", "t3_fp"):
        if of98.get(k) is not None:
            base[k] = of98[k]

    # 板端参照：直接读会话文件（台账里的 board_sessions 可能被搜索器覆盖成 null）
    ref_t31 = ns.board_reference_ms()

    lines = []
    lines.append("# Beat-to-Pass 结论汇总（参照 Baseline_v2.0 = trial_9863）\n")
    lines.append(f"> 基线：trial_9863 @ ep5 ｜ nano **{fmt(base['nano'])}** · nano@4px {fmt(base['nano4'])} · "
                 f"F1 {fmt(base['f1'])} · T1 {fmt(base['t1'])} · T2 FP {fmt(base['t2_fp'], 'd')} · T3 {fmt(base['t3'])}")
    lines.append(f"> 板端同会话参照：trial_0031 自身 = **{ref_t31} ms**（新板 192.168.0.211）\n")

    rows = []
    for k, st in sorted((led.get("arms") or {}).items()):
        b = st.get("picked") or {}
        g = st.get("official") or {}
        if not b:
            continue
        chk = {}
        if b.get("nano") is not None:
            chk["nano"] = b["nano"] >= base["nano"]
        if b.get("nano4") is not None:
            chk["nano4"] = b["nano4"] >= base["nano4"]
        if b.get("f1") is not None:
            chk["f1"] = b["f1"] >= base["f1"]
        if g.get("t1") is not None:
            chk["t1"] = g["t1"] >= base["t1"]
        if g.get("t2_fp") is not None:
            chk["t2fp"] = g["t2_fp"] <= base["t2_fp"]
        if g.get("t3") is not None:
            chk["t3"] = g["t3"] >= base["t3"]
        ms = st.get("board_ms")
        if ms is None and b.get("epoch") is not None:
            ms = ns.board_ms_for(int(k), int(b["epoch"]))   # 台账没存就去会话文件里找
        if ms is not None and ref_t31:
            chk["ms"] = ms <= ref_t31 * 1.02
        core = {kk: vv for kk, vv in chk.items() if kk != "ms"}
        # 🔴 六项必须**齐全**且全 True（缺项 = 官方三轨没测 ⇒ 不算打平）
        need = ("nano", "nano4", "f1", "t1", "t2fp", "t3")
        beat = all(k in core for k in need) and all(core[k] for k in need)
        rows.append((int(k), st.get("desc", ""), b.get("epoch"), b.get("nano"), b.get("nano4"),
                     b.get("f1"), b.get("micro"), g.get("t1"), g.get("t2_fp"), g.get("t3"), ms,
                     st.get("board_session"), beat, all(chk.values()) if chk else False,
                     st.get("is_baseline_v2", False)))

    lines.append("## 一、打平或超越 Baseline_v2.0 的臂\n")
    #   r[12] = 六项（nano/nano4/F1/T1/T2FP/T3）全平；r[13] = 再加板端时延；r[14] = 是否就是基线本体
    #   ⇒ 六项全平但**没测速**的臂也要列出来（标 PASS*），否则最重要的候选会从榜上消失
    winners = [r for r in rows if r[12] and not r[14]]
    if not winners:
        lines.append("（暂无 —— 无任何臂在 nano/nano@4px/F1/T1/T2FP/T3 六项上同时打平基线）\n")
    else:
        lines.append("| trial | 变更 | ep | nano | nano@4px | F1 | micro | T1 | T2FP | T3 | ms | 判定 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(winners, key=lambda x: -(x[3] or 0)):
            lines.append(f"| {r[0]} | {r[1][:30]} | {r[2]} | {fmt(r[3])} | {fmt(r[4])} | {fmt(r[5])} | "
                         f"{fmt(r[6], '.4f')} | {fmt(r[7])} | {fmt(r[8], 'd')} | {fmt(r[9])} | "
                         f"{fmt(r[10], '.3f')} | {'PASS' if r[13] else 'PASS*'} |")
        lines.append("")

    lines.append("## 二、全臂排序（按 nano）\n")
    lines.append("| trial | 变更 | ep | nano | nano@4px | F1 | micro | T1 | T2FP | T3 | ms | beat |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: -(x[3] or 0)):
        lines.append(f"| {r[0]} | {r[1][:30]} | {r[2]} | {fmt(r[3])} | {fmt(r[4])} | {fmt(r[5])} | "
                     f"{fmt(r[6], '.4f')} | {fmt(r[7])} | {fmt(r[8], 'd')} | {fmt(r[9])} | "
                     f"{fmt(r[10], '.3f')} | {'✅' if r[12] else '—'} |")
    lines.append("")
    lines.append(f"_生成时间 {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}；"
                 f"共 {len(rows)} 臂有读数；判据 = Baseline_v2.0(trial_9863) 六项全平，"
                 f"ms 项用「同会话不慢于 0031 自身 ×1.02」（绝对 20 ms 待新板重定标）_\n")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Beat-to-Pass 汇总已写出 {OUT}")
    print(f"     共 {len(rows)} 臂；打平基线的臂：{[r[0] for r in winners] or '无'}")


if __name__ == "__main__":
    main()
