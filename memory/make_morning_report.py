#!/usr/bin/env python3
"""host 侧晨报生成器（服务器晨报脚本的独立备份）：
   输入 ledger.json → 输出 markdown 晨报（基线 / 家族计数 / 六项臂 / 候选 / 板卡 / 升轨判定）。

用法: python3 make_morning_report.py <ledger.json> [-o report.md] [--top 15]
"""
from __future__ import annotations
import argparse, json, os, sys, importlib.util, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ha", os.path.join(HERE, "host_audit.py"))
ha = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ha)

BOARD_LIMIT = ha.BOARD_LIMIT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--promo", default=None, help="promotion_*.json 路径（可选）")
    a = ap.parse_args()

    ledger = json.load(open(a.ledger, encoding="utf-8"))
    arms = ledger
    if isinstance(ledger, dict):
        for k in ("arms", "trials", "entries"):
            if isinstance(ledger.get(k), (dict, list)):
                arms = ledger[k]
                break
    if isinstance(arms, list):
        arms = {str(x.get("trial", i)): x for i, x in enumerate(arms) if isinstance(x, dict)}
    th, src = ha.thresholds(ledger)

    fams, sig_seen = {}, {}
    for tid, arm in arms.items():
        if not isinstance(arm, dict):
            continue
        sig = ha.signature_of(arm, str(tid))
        if sig in sig_seen:
            continue
        sig_seen[sig] = str(tid)
        m = ha.arm_metrics(arm)
        ok, why = ha.six_pass(m, th)
        fams.setdefault(ha.family_of(arm, str(tid)), []).append({"tid": str(tid), "ok": ok, "why": why, "m": m})

    fam_stat = {f: {"total": len(l), "six_pass": sum(1 for x in l if x["ok"]),
                     "ids": [x["tid"] for x in l if x["ok"]]} for f, l in fams.items()}
    ready = {f: s for f, s in fam_stat.items() if s["six_pass"] >= 4}
    cands = []
    for fam, lst in fams.items():
        for x in lst:
            m = x["m"]
            if m.get("nano") is None or m.get("t1") is None:
                continue
            score = (m.get("nano4") or 0) + 3 * (m.get("t3") or 0) + 3 * (m.get("t1") or 0) + (m.get("f1") or 0)
            cands.append(dict(x, fam=fam, score=score))
    cands.sort(key=lambda c: (c["ok"], c["score"]), reverse=True)
    base = ledger.get("baseline_v2") if isinstance(ledger, dict) else None
    base_tid = base.get("trial") if isinstance(base, dict) else None

    L = []
    L.append(f"# YOLO nano 搜索晨报 — {datetime.datetime.now():%Y-%m-%d %H:%M}")
    L.append("")
    L.append(f"台账：`{a.ledger}`　臂数：{len(arms)}　去重后：{len(sig_seen)}")
    L.append("")
    L.append("## 1. 当前基线")
    if isinstance(base, dict):
        L.append(f"* **Baseline_v2 = trial_{base_tid}**"
                 f"（nano {base.get('nano')} / nano4 {base.get('nano4')} / F1 {base.get('f1')} / "
                 f"T1 {base.get('t1')} / T2fp {base.get('t2_fp')} / T3 {base.get('t3')}）")
    L.append(f"* 判据门槛（来源：{','.join(sorted(set(src.values()))) or 'fallback'}）："
             + "；".join(f"{k}{'>=' if k!='t2_fp' else '<='}{v:.6g}" for k, v in th.items()))
    L.append("")
    L.append("## 2. 升轨判据（同家族六项全过臂 ≥ 4）")
    if ready:
        for f, s in sorted(ready.items(), key=lambda kv: -kv[1]["six_pass"]):
            L.append(f"* ✅ **{f}**：六项全过 **{s['six_pass']}** 个（{', '.join(s['ids'][:10])}）⇒ 满足升轨条件")
    else:
        L.append("* ❌ 暂无家族达到 4 个六项全过臂。")
    L.append("")
    L.append("### 各家族计数")
    L.append("| 家族 | 总臂数 | 六项全过 |")
    L.append("| --- | --- | --- |")
    for f, s in sorted(fam_stat.items(), key=lambda kv: -kv[1]["six_pass"])[:15]:
        L.append(f"| `{f}` | {s['total']} | {s['six_pass']} |")
    L.append("")
    L.append(f"## 3. 六项全过臂（{sum(s['six_pass'] for s in fam_stat.values())} 个）")
    L.append("| trial | 家族 | nano | nano4 | F1 | T1 | T2fp | T3 | 板卡 ms |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for c in [c for c in cands if c["ok"]][:25]:
        m = c["m"]
        L.append(f"| {c['tid']} | `{c['fam']}` | {m.get('nano', float('nan')):.6f} | {m.get('nano4', float('nan')):.6f} | "
                 f"{m.get('f1', float('nan')):.6f} | {m.get('t1', float('nan')):.6f} | {m.get('t2_fp')} | "
                 f"{m.get('t3', float('nan')):.6f} | {m.get('board_ms') if m.get('board_ms') is not None else '—'} |")
    L.append("")
    L.append(f"## 4. 最接近的未过线候选 top {a.top}")
    L.append("| trial | 家族 | 6项 | nano | nano4 | T1 | T3 | T2fp | 板卡 ms | 未过原因 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for c in [c for c in cands if not c["ok"]][: a.top]:
        m = c["m"]
        L.append(f"| {c['tid']} | `{c['fam']}` | n | {m.get('nano', float('nan')):.6f} | {m.get('nano4', float('nan')):.6f} | "
                 f"{m.get('t1', float('nan')):.6f} | {m.get('t3', float('nan')):.6f} | {m.get('t2_fp')} | "
                 f"{m.get('board_ms') if m.get('board_ms') is not None else '—'} | {'; '.join(c['why'])[:60]} |")
    L.append("")
    L.append(f"> 板卡红线：T4 ≤ {BOARD_LIMIT} ms，只认 `board_measure_auto/measured.json`；板卡被别的任务占用时读数可能虚高。")
    if a.promo and os.path.isfile(a.promo):
        L.append("")
        L.append("## 5. 升轨记录")
        L.append("```json")
        L.append(open(a.promo, encoding="utf-8").read()[:2000])
        L.append("```")
    txt = "\n".join(L)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(txt)
        print(f"晨报写入 {a.out}（{len(txt.splitlines())} 行）")
    else:
        print(txt)


if __name__ == "__main__":
    main()
