#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""退化曲线**有效性复算器**（host 侧独立复算，2026-11-26）。

为什么需要它：`extract_knee` 只靠「Kneedle 与 PCHIP 双算法一致」判定 knee_reliable，
而在**跟住率从未塌掉**的平曲线上，两法必然落在同一箱 ⇒ 一致 ⇒ 误判 reliable=1。
本脚本不依赖任何项目模块，直接从 `degrade_detail.csv` + `knee_thresholds_v2.csv`
把「这条曲线到底有没有承载崩塌信息」算清楚，给每条曲线打**有效性等级**。

用法（服务器上）:
    export PYTHONPATH=/tmp/pylibs_cvcontrib
    /home/huangzhe/anaconda3/envs/uav/bin/python iqa_v2/recheck_curves.py

判据:
    A 级 (USABLE)     : 跟住率随退化**单调下降** 且 跨度 > MIN_SPAN(=0.05)
    B 级 (WEAK)       : 单调但跨度 <= MIN_SPAN  ⇒ 曲线近似水平，膝点位置不可靠
    C 级 (DEGENERATE) : 跟住率随退化**非单调** 或 跨度 < 1e-3 ⇒ 数值噪声
另: 指标侧单调性 —— 退化强度递增时指标中位必须单调下降，否则该族不承载该指标的信息。
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

MIN_SPAN = 0.05
TRACKERS = ("xcorr", "kcf_legacy", "csrt")
CORE_METRICS = ("local_scr", "target_local_contrast", "target_lap_var")
# 退化强度递增顺序（contrast 的 alpha 越小 = 退化越重）
LEVEL_ORDER = {
    "blur": ["0.6", "1.0", "1.5", "2.2", "3.0", "4.0"],
    "motion": ["3.0", "5.0", "7.0", "9.0", "11.0"],
    "contrast": ["0.8", "0.65", "0.5", "0.35", "0.25", "0.15"],
}
FAMILIES = ("blur", "motion", "contrast")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return None
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


def load_detail(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_knees(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def curve_validity(detail):
    """返回 {(tracker, family): dict(rates=..., span=..., mono=..., grade=...)}"""
    out = {}
    deg = [r for r in detail if r["condition_family"] != "clean"]
    for t in TRACKERS:
        for fam in FAMILIES:
            sub = [r for r in deg if r["condition_family"] == fam]
            bylev = defaultdict(list)
            for r in sub:
                v = _f(r[t + "_success"])
                if v is not None:
                    bylev[r["condition_level"]].append(v)
            levels = LEVEL_ORDER[fam]
            levels = [l for l in levels if l in bylev]
            rates = [sum(bylev[l]) / len(bylev[l]) for l in levels]
            span = (max(rates) - min(rates)) if rates else 0.0
            mono = all(rates[i + 1] <= rates[i] + 1e-9 for i in range(len(rates) - 1))
            if not mono:
                grade = "C-DEGENERATE(not-monotone)"
            elif span < 1e-3:
                grade = "C-DEGENERATE(flat)"
            elif span <= MIN_SPAN:
                grade = "B-WEAK(span<=%.2f)" % MIN_SPAN
            else:
                grade = "A-USABLE"
            out[(t, fam)] = {
                "rates": rates, "levels": levels, "span": span, "mono": mono, "grade": grade,
            }
    return out


def metric_monotonicity(detail):
    """返回 {(family, metric): (monotone, relspan, meds)}"""
    out = {}
    deg = [r for r in detail if r["condition_family"] != "clean"]
    for fam in FAMILIES:
        sub = [r for r in deg if r["condition_family"] == fam]
        for m in CORE_METRICS:
            bylev = defaultdict(list)
            for r in sub:
                v = _f(r[m])
                if v is not None:
                    bylev[r["condition_level"]].append(v)
            meds = [_median(bylev[l]) for l in LEVEL_ORDER[fam] if l in bylev]
            meds = [m for m in meds if m is not None]
            if not meds:
                continue
            mono = all(meds[i + 1] <= meds[i] * 1.001 for i in range(len(meds) - 1))
            relspan = (max(meds) - min(meds)) / max(abs(min(meds)), 1e-9)
            out[(fam, m)] = (mono, relspan, meds)
    return out



def blind_separation(root):
    """真实素材盲测的区分度（**一次性验证，不得据此调阈值**）。

    "落线下" = 低于红线 = 触发「物理退化、不考核算法」的豁免。
    对真实好片，落线下 = 错误豁免；对真实坏片，落线下 = 正确豁免。
    """
    path = os.path.join(root, "output", "blind_real_v2.csv")
    if not os.path.isfile(path):
        print("  找不到盲测明细:", path)
        return
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    print("  盲测集: %s" % ", ".join("%s=%d 帧" % (k, len(v)) for k, v in sorted(groups.items())))
    for m in CORE_METRICS:
        good = [r for r in groups.get("good", []) if _f(r[m]) is not None]
        bad = [r for r in groups.get("bad", []) if _f(r[m]) is not None]
        if not good or not bad:
            continue
        gv = sorted(_f(r[m]) for r in good)
        bv = sorted(_f(r[m]) for r in bad)
        print("  %-22s 真实好片 中位=%.4f (p05=%.4f)  真实坏片 中位=%.4f (p05=%.4f)"
              % (m, _median(gv), gv[max(0, len(gv) // 20)], _median(bv), bv[max(0, len(bv) // 20)]))
        # 在各候选红线上算区分间隔
        cands = []
        knee_path = os.path.join(root, "output", "knee_thresholds_v2.csv")
        if os.path.isfile(knee_path):
            for k in load_knees(knee_path):
                if k["metric"] == m and k["curve_set"] == "common_good" and k["stratum"] == "all":
                    v = _f(k["knee_x_final"])
                    if v is not None:
                        cands.append(("%s/%s" % (k["tracker"], k["degradation_family"]), v))
        for label, thr in sorted(set(cands), key=lambda x: x[1]):
            if thr != thr:  # NaN => 该组合没有可用膝点，跳过
                continue
            gb = sum(1 for r in good if _f(r[m]) < thr) / len(good)
            bb = sum(1 for r in bad if _f(r[m]) < thr) / len(bad)
            flag = "OK " if (bb - gb) >= 0.5 else "BAD"
            print("     %-28s thr=%-12.5f 好片落线下=%6.2f%%  坏片落线下=%6.2f%%  间隔=%6.2f pp  %s"
                  % (label, thr, gb * 100, bb * 100, (bb - gb) * 100, flag))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = args[0] if args else "iqa_v2"
    detail_path = os.environ.get("DEGRADE_DETAIL", "/tmp/iqa_v2_cache/degrade_detail.csv")
    knee_path = os.path.join(root, "output", "knee_thresholds_v2.csv")
    if not os.path.isfile(detail_path):
        print("找不到退化明细:", detail_path)
        return 2
    detail = load_detail(detail_path)
    knees = load_knees(knee_path)

    clean = [r for r in detail if r["condition_family"] == "clean"]
    pool = {(r["seq_id"], r["frame_id"]) for r in detail if r["condition_family"] != "clean"}
    pool_clean = [r for r in clean if (r["seq_id"], r["frame_id"]) in pool]

    print("=" * 78)
    print("§1  干净帧起点（纯净化底座，池内 clean 行）")
    print("=" * 78)
    for t in TRACKERS:
        vals = [_f(r[t + "_success"]) for r in pool_clean]
        vals = [v for v in vals if v is not None]
        print("  %-11s n=%-6d clean_success_rate=%.4f" % (t, len(vals), sum(vals) / len(vals)))
    print("  池大小=%d 窗口（clean 行 %d 条）" % (len(pool), len(pool_clean)))

    val = curve_validity(detail)
    print()
    print("=" * 78)
    print("§2  跟住率曲线有效性（退化强度递增方向的单调性 + 跨度）")
    print("=" * 78)
    for t in TRACKERS:
        print("  -- %s --" % t)
        for fam in FAMILIES:
            c = val[(t, fam)]
            print("     %-9s span=%.4f mono=%-5s %-26s rates=%s" % (
                fam, c["span"], c["mono"], c["grade"],
                " ".join("%s:%.3f" % (l, r) for l, r in zip(c["levels"], c["rates"]))))

    mm = metric_monotonicity(detail)
    print()
    print("=" * 78)
    print("§3  指标侧单调性（退化递增 ⇒ 指标中位必须单调下降）")
    print("=" * 78)
    for fam in FAMILIES:
        for m in CORE_METRICS:
            mono, relspan, meds = mm[(fam, m)]
            print("  %-9s %-22s %-8s relspan=%7.1f%%  %s" % (
                fam, m, "OK" if mono else "NOT-MONO", relspan * 100,
                " ".join("%.4g" % x for x in meds)))

    print()
    print("=" * 78)
    print("§4  三项核心指标的有效候选（curve_set=common_good / own_good）")
    print("=" * 78)
    hdr = "%-11s %-10s %-9s %-22s %12s %8s %8s %-12s %-26s"
    print(hdr % ("tracker", "curve_set", "family", "metric", "knee_x", "knee_y", "delta%",
                "knee_rel", "有效性"))
    for cs in ("common_good", "own_good"):
        for r in knees:
            if r["metric"] not in CORE_METRICS or r["curve_set"] != cs or r["stratum"] != "all":
                continue
            t, fam = r["tracker"], r["degradation_family"]
            c = val.get((t, fam), {})
            mono_m, relspan, _ = mm.get((fam, r["metric"]), (False, 0.0, []))
            kx = _f(r["knee_x_final"])
            kx_s = ("%.5f" % kx) if kx is not None else "nan"
            ky = _f(r["knee_y_final"])
            ky_s = ("%.4f" % ky) if ky is not None else "nan"
            dl = _f(r["delta_pct_of_span"])
            dl_s = ("%.3f" % dl) if dl is not None else "nan"
            why = []
            if c.get("grade", "").startswith("C"):
                why.append("曲线退化")
            if c.get("span", 0) <= MIN_SPAN:
                why.append("跨度小")
            if not mono_m:
                why.append("指标非单调")
            if r["knee_reliable"] != "1":
                why.append("双算法不一致")
            verdict = "USABLE" if not why else "REJECT(" + ",".join(why) + ")"
            print(hdr % (t, cs, fam, r["metric"], kx_s, ky_s, dl_s,
                         r["knee_reliable"], verdict))

    if "--blind" in sys.argv[1:]:
        print()
        print("=" * 78)
        print("§4b  真实素材盲测区分度（一次性验证，不得据此调阈值）")
        print("=" * 78)
        blind_separation(root)

    print()
    print("=" * 78)
    print("§5  结论")
    print("=" * 78)
    usable = [(t, fam) for t in TRACKERS for fam in FAMILIES
              if val[(t, fam)]["grade"].startswith("A")]
    print("  跟住率曲线达到 A 级(单调且跨度>%.2f)的组合: %s" % (MIN_SPAN, usable or "无"))
    print("  ⇒ 只有这些组合的膝点有资格进入约束线候选；其余必须标为不可用并在报告里明写。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
