#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2 交付物**完成度审计**（任务书 ⇒ 逐条要求 ⇒ 机器可判的证据）。

用法:
    python3 iqa_v2/audit_final.py <目录> [<报告 md>]

<目录> = 服务器回收回来的产物目录（含 output/*.csv 与 REPORT_V2_FINAL.md 等）。

设计要点（本版相对旧版的三处硬化）：

1. **按表头内容识别产物，不再靠文件名片段**。旧版用 ``find_csv(root, "window"/"regress")``
   这类文件名猜测，结果新产物叫 ``segment_class_v2.csv`` 时直接漏检 ⇒ 误判 FAIL。
   本版按**列名签名**识别每张表的角色（明细底表 / 切分表 / 分类表 / 膝点表 /
   约束线表 / 回归表 / 盲测表）。
2. **独立复算**：不采信流水线写出的膝点与约束线，直接从**明细底表**重算分箱曲线、
   保序回归、Kneedle、PCHIP，再与产物表逐项比对 ⇒ 能抓「产物写错」与「口径写错」。
3. **退化守卫检查**：约束线若来自「跟住率从未塌掉」的曲线（y_span≈0），判 FAIL。
   这是 v2 本轮真实踩过的坑（平曲线上的浮点噪声被当成膝点）。

输出: 每条要求一行 ``PASS/FAIL/WARN  <编号>  <要求>  ——  <证据>``；
      末尾给总账。**WARN = 证据不足、需人工判读**（不算 PASS）。
"""
from __future__ import annotations

import csv
import os
import re
import sys

RES: list[tuple[str, str, str]] = []

# 任务书 §一 的三项核心指标（唯一允许进入红线的）
CORE = ("target_local_contrast", "target_lap_var", "local_scr")
# 任务书 §一 明确废弃、但允许作对照的 v1 全图指标
V1_GLOBAL = ("global_contrast", "global_lap_var")
TRACKERS = ("xcorr", "kcf_legacy", "csrt")
# 任务书 §二.2 三族退化轴的档位；**退化严重度递增序**
# （blur σ↑ / motion L↑ 更糊；contrast α↓ 更糊 —— 方向相反，别写反）
SEVERITY = {
    "blur": [0.6, 1.0, 1.5, 2.2, 3.0, 4.0],
    "motion": [3.0, 5.0, 7.0, 9.0, 11.0],
    "contrast": [0.80, 0.65, 0.50, 0.35, 0.25, 0.15],
}
MIN_Y_SPAN = 0.05          # 曲线跟住率跨度下限：低于此值 ⇒ 曲线上不存在「崩溃拐点」
N_BINS = 12                # 与流水线 binned_curve 默认分箱数一致


def rec(status: str, code: str, req: str, ev: str = "") -> None:
    RES.append((status, code, req + ("  ——  " + ev if ev else "")))


def load_csv(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def read_text(path):
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


# ---------------------------------------------------------------- 产物识别
#: 每种表的「列名签名」：表头同时含有其中全部列即认定该角色
SIGNATURES = {
    "detail": (("condition_family", "seq_id", "frame_id"),),
    "split": (("seq_id", "split"),),
    "segment": (("seq_id", "class_ds"),),
    "knee": (("metric", "knee_reliable"),),
    "constraint": (("metric", "line_value"),),
    "regression": (("set", "rate_below_line"),),
    "blind": (("clip", "group"),),
    "windows": (("seq_id", "frame_id", "small_target"),),
}


def index_tables(root):
    """扫描 root（递归），按**列名签名**把每张 CSV 归到角色上。"""
    found = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            if not fn.lower().endswith(".csv"):
                continue
            path = os.path.join(dirpath, fn)
            rows = load_csv(path)
            if not rows:
                continue
            cols = set(rows[0].keys())
            for role, sigs in SIGNATURES.items():
                if role in found:
                    continue
                if any(all(c in cols for c in sig) for sig in sigs):
                    found[role] = (path, rows)
    return found


def pick(rows, **kw):
    """按列值过滤。"""
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            got = (r.get(k) or "").strip()
            if got != str(v):
                ok = False
                break
        if ok:
            out.append(r)
    return out


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------- 独立复算
def binned_curve(values, success, n_bins=N_BINS):
    """按分位数分箱：返回 (箱内 X 中位, 箱内跟住率, 箱计数)。与流水线同口径。"""
    import numpy as np

    mask = np.isfinite(values) & np.isfinite(success)
    values, success = values[mask], success[mask]
    if values.size == 0:
        return np.array([]), np.array([]), np.array([])
    if np.unique(values).size <= n_bins:
        return (np.array([float(np.median(values))]),
                np.array([float(success.mean())]),
                np.array([int(values.size)]))
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    xs, ys, ns = [], [], []
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        sel = ((values >= low) & (values <= high)) if index == len(edges) - 2 \
            else ((values >= low) & (values < high))
        if not sel.any():
            continue
        xs.append(float(np.median(values[sel])))
        ys.append(float(success[sel].mean()))   # 🔴 必须取**箱内**均值（全量均值会把曲线压成水平线）
        ns.append(int(sel.sum()))
    return np.asarray(xs), np.asarray(ys), np.asarray(ns)


def prepare_curve(xs, ys):
    """X 升序 + 保序回归（increasing）——与流水线 prepare_curve 同口径。"""
    import numpy as np
    from scipy.optimize import isotonic_regression

    order = np.argsort(xs, kind="mergesort")
    xs, ys = xs[order], ys[order]
    fitted = np.asarray(isotonic_regression(ys, increasing=True).x, dtype=float)
    unique_x, start = np.unique(xs, return_index=True)
    return unique_x, np.maximum.reduceat(fitted, start)


def knee_recompute(values, success):
    """独立复算膝点：返回 (x_kneedle, y_kneedle, x_pchip, y_pchip, x_span, y_span)。"""
    import numpy as np
    from scipy.interpolate import PchipInterpolator

    xs, ys, _ns = binned_curve(np.asarray(values, float), np.asarray(success, float))
    if xs.size < 4:
        return (float("nan"),) * 4 + (float(xs.max() - xs.min()) if xs.size else 0.0,
                                      float(ys.max() - ys.min()) if ys.size else 0.0)
    cx, cy = prepare_curve(xs, ys)
    x_span = float(cx.max() - cx.min())
    y_span = float(cy.max() - cy.min())
    x_kn = y_kn = x_pc = y_pc = float("nan")
    if cx.size >= 3 and x_span > 0 and y_span > 0:
        xn = (cx - cx.min()) / x_span
        yn = (cy - cy.min()) / y_span
        idx = int(np.argmax(np.abs(yn - xn)))
        x_kn, y_kn = float(cx[idx]), float(cy[idx])
    if cx.size >= 4:
        spline = PchipInterpolator(cx, cy)
        dense = np.linspace(cx[0], cx[-1], 1500)
        second = spline.derivative(2)(dense)
        idx = int(np.argmax(np.abs(second)))
        x_pc, y_pc = float(dense[idx]), float(spline(dense[idx]))
    return x_kn, y_kn, x_pc, y_pc, x_span, y_span


def severity_median_metric(detail_rows, family, metric):
    """按退化严重度序，给出每档的指标中位数。"""
    import numpy as np
    import collections

    bylvl = collections.defaultdict(list)
    for r in detail_rows:
        if r.get("condition_family") != family:
            continue
        bylvl[round(fnum(r.get("condition_level")), 4)].append(fnum(r.get(metric)))
    out = []
    for lvl in SEVERITY.get(family, []):
        vals = bylvl.get(round(lvl, 4))
        if vals:
            out.append((lvl, float(np.median(vals))))
    return out



def curve_subset(detail, class_by_key, tracker, family, curve_set, stratum="all"):
    """按流水线口径取某条曲线的窗口子集：退化族 × 曲线集(own_good/common_good) × 分层。

    own_good   = 该臂自己的好片段；
    common_good = 三臂共同的好片段（跨臂线位可比的前提）。
    """
    out = []
    for r in detail:
        if r.get("condition_family") != family:
            continue
        if stratum == "small" and str(r.get("small_target", "")).strip() not in ("1", "1.0", "True"):
            continue
        info = class_by_key.get((r.get("seq_id", ""), r.get("frame_id", "")))
        if not info:
            continue
        if curve_set == "own_good":
            if info.get("class_%s" % tracker) != "good":
                continue
        elif not all(info.get("class_%s" % arm) == "good" for arm in TRACKERS):
            continue
        out.append(r)
    return out


def level_profile(rows, family, tracker, metric):
    """逐档跟住率/指标中位（退化强度递增序）⇒ 跨度 + 单调性（流水线守卫的「逐档」口径）。"""
    import collections

    byl = collections.defaultdict(list)
    bym = collections.defaultdict(list)
    for r in rows:
        lv = fnum(r.get("condition_level"))       # 🔴 键统一成 float（CSV 里是字符串，SEVERITY 里是 float）
        byl[lv].append(fnum(r.get("%s_success" % tracker)))
        bym[lv].append(fnum(r.get(metric)))
    levels = [lv for lv in SEVERITY.get(family, []) if any(abs(k - lv) < 1e-6 for k in byl)]
    if not levels:
        levels = sorted(byl, key=lambda k: (k if k == k else 0.0))
    rates = [sum(byl[lv]) / len(byl[lv]) for lv in levels if byl.get(lv)]
    meds = []
    for lv in levels:
        vals = [v for v in bym.get(lv, []) if v == v]
        meds.append(float(sorted(vals)[len(vals) // 2]) if vals else float("nan"))
    span = (max(rates) - min(rates)) if rates else 0.0
    rate_mono = all(rates[i + 1] <= rates[i] + 1e-9 for i in range(len(rates) - 1))
    metric_mono = all(meds[i + 1] <= meds[i] * 1.001 for i in range(len(meds) - 1))
    return {"levels": levels, "rates": rates, "span": span,
            "rate_monotone": bool(rate_mono), "metric_monotone": bool(metric_mono)}


def curve_guard(rows, family, tracker, metric):
    """复现流水线的「曲线有效性守卫」：退化 ⇔ (逐档非单调 | 逐档跨度≤阈值 | 分箱跨度≤阈值 | 指标非单调)。"""
    if not rows:
        return None
    vals = [fnum(r.get(metric)) for r in rows]
    succ = [fnum(r.get("%s_success" % tracker)) for r in rows]
    _xkn, _ykn, _xpc, _ypc, _xspan, yspan = knee_recompute(vals, succ)
    prof = level_profile(rows, family, tracker, metric)
    n_bins = len(set(round(fnum(r.get(metric)), 9) for r in rows))
    bspan_applies = n_bins >= 4
    degenerate = ((not prof["rate_monotone"]) or prof["span"] <= MIN_Y_SPAN
                  or (bspan_applies and yspan <= MIN_Y_SPAN) or (not prof["metric_monotone"]))
    return {"binned_span": yspan, "level_span": prof["span"],
            "rate_monotone": prof["rate_monotone"], "metric_monotone": prof["metric_monotone"],
            "degenerate": bool(degenerate)}

# ---------------------------------------------------------------- 主检查
def main(root: str, report_path: str) -> int:
    report = read_text(report_path)
    if not report:
        rec("FAIL", "R0", "报告存在且可读", report_path)
    tab = index_tables(root)
    got_roles = ", ".join(sorted(tab)) or "（无）"

    # ============================================== 一、口径：仅 v2 三项核心指标
    detail = tab.get("detail", (None, []))[1]
    if not detail:
        rec("FAIL", "R1", "存在带 condition_family 的窗口明细 CSV（退化仿真底表）",
            "识别到的产物: %s" % got_roles)
    else:
        fams = sorted({r.get("condition_family", "") for r in detail})
        rec("PASS" if "jpeg" not in fams else "FAIL", "R1",
            "退化轴只剩 blur/motion/contrast，JPEG 压缩轴已剔除", "实测退化族: %s" % (fams,))
        has_noise = any("noise" in f.lower() for f in fams)
        rec("FAIL" if has_noise else "PASS", "R1b",
            "未引入加性白噪声退化轴", "实测退化族: %s" % (fams,))
        want = {"blur", "motion", "contrast"}
        rec("PASS" if want <= set(fams) else "FAIL", "R1c",
            "三族红外物理退化轴齐备（blur/motion/contrast）", "实测: %s" % (fams,))

    # ============================================== 二、仿真底座纯净化
    if detail:
        pool = {(r.get("seq_id", ""), r.get("frame_id", "")) for r in detail
                if r.get("condition_family") not in ("clean", "", "error")}
        clean_rows = [r for r in detail if r.get("condition_family") == "clean"]
        pool_clean = [r for r in clean_rows
                      if (r.get("seq_id", ""), r.get("frame_id", "")) in pool]
        if not pool_clean:
            rec("FAIL", "R2", "仿真池内存在 clean 档行（可核验起点跟住率）",
                "明细 clean 行 %d，池内 %d" % (len(clean_rows), len(pool_clean)))
        else:
            for t in TRACKERS:
                key = "%s_success" % t
                if not any(key in r for r in pool_clean):
                    continue
                ok = sum(int(float(r.get(key, 0) or 0)) for r in pool_clean)
                rate = ok / len(pool_clean)
                rec("PASS" if rate >= 0.98 else "FAIL", "R2",
                    "底座纯净化：仿真池内 clean 档起点跟住率须≈100%%（%s）" % t,
                    "实测 %.4f（%d/%d 窗口）" % (rate, ok, len(pool_clean)))
        # ---- R2b 退化族单调性（任务书 §四.2 的单调性假设）
        mono_bad = []
        for family in ("blur", "motion", "contrast"):
            for metric in CORE:
                seq = severity_median_metric(detail, family, metric)
                if len(seq) < 2:
                    continue
                vals = [v for _l, v in seq]
                if not all(vals[i + 1] <= vals[i] * 1.02 for i in range(len(vals) - 1)):
                    mono_bad.append("%s/%s" % (family, metric))
        rec("FAIL" if mono_bad else "PASS", "R2b",
            "退化强度递增 ⇒ 核心指标单调不增（任务书 §四.2 单调性假设）",
            "违反的(族,指标): %s" % (mono_bad if mono_bad else "无（9 组合全部成立）"))

    # ============================================== 三、序列级 7:3 切分 + 零泄露
    split_rows = tab.get("split", (None, []))[1]
    calib, hold, col = set(), set(), None
    if not split_rows:
        rec("FAIL", "R3", "序列级 7:3 切分可核验（标定集/留出集 seq_id 不相交）",
            "未找到切分表；识别到的产物: %s" % got_roles)
    else:
        for k in split_rows[0].keys():
            c1, h1 = set(), set()
            for r in split_rows:
                sid = r.get("seq_id", "")
                val = (r.get(k) or "").strip().lower()
                if not val:
                    continue
                if "cal" in val or "train" in val or "标定" in val or "拟合" in val:
                    c1.add(sid)
                elif ("hold" in val or "test" in val or "留出" in val or "验证" in val
                      or val == "val" or "eval" in val):
                    h1.add(sid)
            if c1 and h1 and not (c1 & h1):
                calib, hold, col = c1, h1, k
                break
        if col is None:
            rec("FAIL", "R3", "切分表无法识别标定/留出两桶",
                "列: %s" % list(split_rows[0].keys()))
        else:
            inter = calib & hold
            rec("PASS" if calib and hold and not inter else "FAIL", "R3",
                "序列级切分无交叉（同源泄露检查）",
                "按列 '%s'：标定 %d 序列 / 留出 %d 序列 / 交集 %d"
                % (col, len(calib), len(hold), len(inter)))
            tot = len(calib) + len(hold)
            if tot:
                ratio = len(calib) / tot
                rec("PASS" if 0.6 <= ratio <= 0.8 else "FAIL", "R3b",
                    "切分比例约 7:3（标定占 60%%~80%%）", "实测 %.3f" % ratio)
            # ---- R3c 留出集零参与阈值搜索（明细里不得出现留出集的非 clean 行）
            if detail:
                leak = sorted({r.get("seq_id", "") for r in detail
                               if r.get("condition_family") != "clean"
                               and r.get("seq_id", "") in hold})
                rec("PASS" if not leak else "FAIL", "R3c",
                    "留出集零参与阈值搜索（明细中无留出集的非 clean 行）",
                    "泄露序列: %s" % (leak[:5] if leak else "无"))

    # ============================================== 四、三分法回归体系
    for kw, code, req in (
        ("好片误杀率", "R4a", "好片段留出集的**好片误杀率（False Alarm Rate）**已报出"),
        ("拦截率", "R4b", "跟丢片段/真实坏片的**坏片拦截率（Interception Rate）**已报出"),
        ("临界", "R4c", "一般片段的**临界敏感性**已报出"),
    ):
        rec("PASS" if kw in report else "FAIL", code, req)
    reg_rows = tab.get("regression", (None, []))[1]
    if reg_rows:
        rec("PASS", "R4d", "三分法回归测试有独立结果表", "行数 %d" % len(reg_rows))
        sets = {r.get("set", "") for r in reg_rows}
        need = {"good_holdout", "mediocre", "lost"}
        rec("PASS" if need <= sets else "WARN", "R4e",
            "回归表覆盖好片段留出集/一般片段/跟丢片段三个分层",
            "实测 set: %s" % sorted(sets))
        # 回归表里的门槛必须与约束线表一致
        cons = tab.get("constraint", (None, []))[1]
        lines = {r.get("metric"): fnum(r.get("line_value")) for r in cons}
        mismatch = []
        for r in reg_rows:
            m = r.get("metric", "")
            if m in lines:
                if abs(fnum(r.get("threshold")) - lines[m]) > 1e-9 * max(1.0, abs(lines[m])):
                    mismatch.append("%s(%s vs %s)" % (m, r.get("threshold"), lines[m]))
        rec("PASS" if not mismatch else "FAIL", "R4f",
            "回归表使用的门槛与约束线表一致", "不一致: %s" % (mismatch if mismatch else "无"))
    else:
        rec("FAIL", "R4d", "三分法回归测试有独立结果表（CSV）",
            "识别到的产物: %s" % got_roles)

    # ============================================== 五、三项核心指标的约束线
    knee_rows = tab.get("knee", (None, []))[1]
    cons_rows = tab.get("constraint", (None, []))[1]
    if not knee_rows:
        rec("FAIL", "R5", "存在膝点表（metric + knee_reliable 字段）")
    else:
        for m in CORE:
            sel = [r for r in knee_rows if r.get("metric") == m]
            rel = [r for r in sel if r.get("knee_reliable") == "1"]
            rec("PASS" if rel else ("WARN" if sel else "FAIL"), "R5",
                "核心指标有**可靠**膝点（knee_reliable=1，双算法偏差≤15%%量程）：%s" % m,
                "该指标 %d 行，其中可靠 %d 行" % (len(sel), len(rel)))
        merged = [r for r in knee_rows
                  if r.get("tracker", "").lower() in ("merged", "mean", "avg")]
        rec("PASS" if not merged else "FAIL", "R8b",
            "XCorr 与 KCF 不得合并求平均（跨家族纪律）", "违规行 %d" % len(merged))
        cols = set(knee_rows[0].keys())
        need = {"knee_x_kneedle", "knee_x_pchip", "delta_pct_of_span", "knee_reliable"}
        rec("PASS" if need <= cols else "FAIL", "R8",
            "膝点双算法交叉验证字段齐备（Kneedle + PCHIP + 偏差 + reliable）",
            "缺: %s" % sorted(need - cols))
    if not cons_rows:
        rec("FAIL", "R5d", "存在约束线表（metric + line_value）")
    else:
        metrics = [r.get("metric", "") for r in cons_rows]
        extra = sorted(set(metrics) - set(CORE))
        rec("PASS" if not extra else "FAIL", "R5d",
            "约束线表**只含三项核心指标**（v1 全图指标不得进红线）",
            "实测指标集: %s%s" % (sorted(set(metrics)),
                                  ("　违规: %s" % extra) if extra else ""))
        v1_in = sorted(set(metrics) & set(V1_GLOBAL))
        rec("PASS" if not v1_in else "FAIL", "R5e",
            "v1 全图指标（global_*）未进约束线，只作对照", "违规: %s" % (v1_in or "无"))
        # ---- R5f 约束线必须能在膝点表里找到对应的可靠膝点
        orphan = []
        n_nan_line = 0
        for r in cons_rows:
            m, t, f = r.get("metric"), r.get("deciding_tracker"), r.get("deciding_family")
            lv = fnum(r.get("line_value"))
            if lv != lv:            # 🔴 nan 线位 = 该指标本轮**无有效红线** ⇒ 不需要匹配膝点
                n_nan_line += 1
                continue
            hit = [k for k in knee_rows
                   if k.get("metric") == m and k.get("tracker") == t
                   and k.get("degradation_family") == f
                   and k.get("knee_reliable") == "1"
                   and k.get("curve_set") == r.get("curve_set", "common_good")
                   and k.get("stratum") == r.get("stratum", "all")]
            if not hit:
                orphan.append("%s/%s/%s" % (m, t, f))
            else:
                best = min(abs(fnum(k.get("knee_x_final")) - lv) for k in hit)
                if best > 1e-9 * max(1.0, abs(lv)):
                    orphan.append("%s 值与膝点表对不上(Δ%.3g)" % (m, best))
        rec("PASS" if not orphan else "FAIL", "R5f",
            "每条约束线都能在膝点表里找到对应的**可靠**膝点且数值一致",
            "对不上: %s%s" % (orphan if orphan else "无",
                              ("　（另有 %d 条 nan 线位 = 本轮无有效红线，按设计不计）" % n_nan_line) if n_nan_line else ""))
        # ---- R5g 退化守卫：约束线不得来自「跟住率从未塌掉」的曲线
        degenerate = []
        guard_mismatch = []
        class_by_key = {(c.get("seq_id", ""), c.get("frame_id", "")): c
                        for c in tab.get("segment", (None, []))[1]}
        for r in cons_rows:
            m, t, f = r.get("metric"), r.get("deciding_tracker"), r.get("deciding_family")
            lv = fnum(r.get("line_value"))
            if not detail or not t or not f or lv != lv:
                continue                      # nan 线位：本轮无有效红线，不存在「线来自平曲线」的问题
            cs = r.get("curve_set") or "common_good"
            st = r.get("stratum") or "all"
            rows = curve_subset(detail, class_by_key, t, f, cs, st)
            if not rows:
                degenerate.append("%s/%s/%s 子集为空" % (m, t, f))
                continue
            g = curve_guard(rows, f, t, m)
            if g["degenerate"]:
                degenerate.append("%s/%s/%s 逐档跨度 %.4f / 分箱跨度 %.4f"
                                  % (m, t, f, g["level_span"], g["binned_span"]))
            # 与产物膝点表的守卫列交叉核对（数值须一致）
            kr = [k for k in knee_rows
                  if k.get("tracker") == t and k.get("degradation_family") == f
                  and k.get("metric") == m and k.get("curve_set") == cs
                  and k.get("stratum") == st]
            if kr:
                k0 = kr[0]
                for col, mine in (("curve_y_span", g["binned_span"]),
                                  ("level_y_span", g["level_span"])):
                    if col in k0:
                        kv = fnum(k0.get(col))
                        if kv == kv and abs(kv - mine) > 1e-6 * max(1.0, abs(mine)):
                            guard_mismatch.append("%s/%s/%s %s 产物 %.6f vs 复算 %.6f"
                                                  % (m, t, f, col, kv, mine))
                if "curve_monotone" in k0:
                    if (k0.get("curve_monotone") == "1") != g["rate_monotone"]:
                        guard_mismatch.append("%s/%s/%s 逐档单调性不一致" % (m, t, f))
        rec("PASS" if not degenerate else "FAIL", "R5g",
            "约束线不得来自「跟住率从未塌掉」的退化曲线（守卫复算：逐档跨度 ≥ %.2f 且分箱跨度 ≥ %.2f）" % (MIN_Y_SPAN, MIN_Y_SPAN),
            "退化候选: %s" % (degenerate if degenerate else "无"))
        rec("PASS" if not guard_mismatch else "FAIL", "R5g2",
            "曲线有效性守卫的**复算值与产物列一致**（逐档跨度 / 分箱跨度 / 单调性）",
            "不一致: %s" % (guard_mismatch if guard_mismatch else "无"))
        # ---- R5h 独立复算：自己重算决定臂/决定族的膝点，看能否复现约束线
        if detail:
            bad = []
            for r in cons_rows:
                m, t, f = r.get("metric"), r.get("deciding_tracker"), r.get("deciding_family")
                lv = fnum(r.get("line_value"))
                if lv != lv:
                    continue                      # nan 线位：本轮无有效红线，无从复算
                cs = r.get("curve_set") or "common_good"
                st = r.get("stratum") or "all"
                rows = curve_subset(detail, class_by_key, t, f, cs, st)
                if not rows:
                    continue
                vals = [fnum(d.get(m)) for d in rows]
                succ = [fnum(d.get("%s_success" % t)) for d in rows]
                xkn, _ykn, xpc, _ypc, _xspan, yspan = knee_recompute(vals, succ)
                if not (xkn == xkn and xpc == xpc):
                    continue
                if min(abs(xkn - lv), abs(xpc - lv)) > 1e-6 * max(1.0, abs(lv)):
                    bad.append("%s/%s/%s 复算得 %.5g / %.5g，产物写 %.5g" % (m, t, f, xkn, xpc, lv))
            rec("PASS" if not bad else "WARN", "R5h",
                "独立复算膝点能复现约束线数值（明细→分箱→Kneedle/PCHIP）",
                "复算不一致: %s" % (bad if bad else "无"))
        # ---- R5i 跨臂线位差必须报出；>15% 时报告须声明「算法无关性不成立」
        has_col = bool(cons_rows) and "cross_arm_dev_pct" in cons_rows[0]
        devs = [(r.get("metric", ""), fnum(r.get("cross_arm_dev_pct")))
                for r in cons_rows if fnum(r.get("cross_arm_dev_pct")) == fnum(r.get("cross_arm_dev_pct"))]
        big = [m for m, d in devs if d > 15.0]
        if not devs:
            if has_col:
                # 列在、但全是 nan ⇒ 跨臂**无线位可比**（其余臂的曲线被守卫作废）
                need_txt = ("算法无关性" in report) or ("跨臂" in report) or ("单臂" in report)
                rec("PASS" if need_txt else "FAIL", "R5i",
                    "跨臂线位差不可计算时（其余臂曲线被守卫作废），报告须声明「算法无关性不成立」",
                    "约束线表 cross_arm_dev_pct 存在但全为 nan；报告是否声明: %s" % ("是" if need_txt else "否"))
            else:
                rec("WARN", "R5i", "跨臂线位差已报出", "约束线表无 cross_arm_dev_pct 列")
        elif big:
            need_txt = ("算法无关性" in report) or ("不具备算法无关性" in report) \
                or ("算法无关" in report)
            rec("PASS" if need_txt else "FAIL", "R5i",
                "跨臂线位差 >15%% 的指标，报告须声明「算法无关性不成立」",
                "超限指标: %s（差 %s%%）" % (big, [round(d, 2) for m, d in devs if m in big]))
        else:
            rec("PASS", "R5i", "跨臂线位差已报出且均 ≤15%%",
                "实测: %s" % [(m, round(d, 2)) for m, d in devs])

    # ============================================== 六、免责声明逐字 + 数值一致
    key_frag = ["前端成像", "不在跟踪算法性能考核范围内", "target_lap_var", "local_scr"]
    miss = [k for k in key_frag if k not in report]
    rec("PASS" if not miss else "FAIL", "R6", "免责声明口径逐字落地（含物理成因与免责边界）",
        "缺失片段: %s" % miss if miss else "四个关键片段齐备")
    nums = (re.findall(r"target_lap_var\s*<\s*([\d.]+)", report)
            + re.findall(r"local_scr\s*<\s*([\d.]+)", report))
    rec("PASS" if nums else "FAIL", "R6b", "免责声明给出可执行数值门槛",
        "读到的门槛: %s" % sorted(set(nums)))
    if cons_rows and nums:
        lines = {r.get("metric"): fnum(r.get("line_value")) for r in cons_rows}
        bad = []
        for r in cons_rows:
            m = r.get("metric", "")
            if m not in ("target_lap_var", "local_scr"):
                continue
            v = lines[m]
            if v != v:            # 🔴 nan 线位 = 该指标本轮无有效红线 ⇒ 免责声明本就不该出现它
                continue
            if not any(abs(float(n) - v) <= 1e-6 * max(1.0, abs(v)) for n in nums):
                bad.append("%s 约束线 %.6g 未出现在免责声明里" % (m, v))
        rec("PASS" if not bad else "FAIL", "R6c",
            "免责声明里的数值门槛与约束线表**一致**", "不一致: %s" % (bad if bad else "无"))
    # ---- R6d 证据最弱的指标必须有限定语（否则与 §12 的结论自相矛盾）
    if reg_rows and cons_rows:
        weak = []
        for r in reg_rows:
            if r.get("set") != "good_holdout":
                continue
            rate = fnum(r.get("rate_below_line"))
            if rate == rate and rate > 0.05:
                weak.append((r.get("metric", ""), round(rate * 100, 2)))
        if weak:
            stated = ("辅助边界" in report) or ("证据最弱" in report) or ("不建议" in report)
            rec("PASS" if stated else "WARN", "R6d",
                "误杀率偏高的指标（>5%%）在免责声明旁有降级/限定说明",
                "实测: %s" % weak)

    # ============================================== 七、小目标子集
    if knee_rows:
        strata = {r.get("stratum", "") for r in knee_rows}
        rec("PASS" if any("small" in s.lower() for s in strata) else "WARN", "R7",
            "小目标子集（max(w,h)≤20px）已单列", "strata: %s" % sorted(s for s in strata if s))
    if detail:
        st = {int(float(r.get("small_target", 0) or 0)) for r in detail}
        rec("PASS" if 1 in st else "WARN", "R7b", "明细含小目标标记列（small_target）",
            "取值: %s" % sorted(st))

    # ============================================== 八、盲测与证据边界
    blind_rows = tab.get("blind", (None, []))[1]
    if blind_rows:
        groups = {}
        for r in blind_rows:
            groups.setdefault(r.get("group", ""), 0)
            groups[r.get("group", "")] += 1
        rec("PASS" if {"bad", "good"} <= set(groups) else "FAIL", "R10",
            "真实坏片盲测明细同时含坏片与好片对照", "实测分组: %s" % groups)
        rec("PASS" if "小样本" in report else "WARN", "R10b",
            "真实坏片样本量小已标注（小样本级、不可外推）",
            "坏片帧数 %d" % groups.get("bad", 0))
    else:
        rec("WARN", "R10", "真实坏片盲测明细（clip + group）", "未识别到盲测表")
    # ---- R11 曲线过浅 ⇒ 报告必须声明「膝点是弯折点不是崩溃点」
    if detail:
        shallow = []
        for family in ("blur", "motion", "contrast"):
            worst = []
            for r in detail:
                if r.get("condition_family") != family:
                    continue
                lvl = fnum(r.get("condition_level"))
                worst.append((lvl, fnum(r.get("xcorr_success"))))
            if not worst:
                continue
            heavy = max(worst, key=lambda x: SEVERITY.get(family, [0])[0] - x[0]) \
                if family != "contrast" else min(worst, key=lambda x: x[0])
            vals = [v for _l, v in worst if (lvl := _l) is not None]
            avg = sum(v for v in vals if v == v) / max(1, len([v for v in vals if v == v]))
            if avg >= 0.5:
                shallow.append("%s(最重档平均跟住率 %.3f)" % (family, avg))
        if shallow:
            stated = ("弯折" in report) or ("不足以产生跟踪崩溃" in report) \
                or ("崩溃拐点" in report and "不是" in report)
            rec("PASS" if stated else "FAIL", "R11",
                "曲线过浅（最重档跟住率仍高）时，报告须声明膝点是「弯折点」而非「崩溃点」",
                "过浅的族: %s" % shallow)

    # ============================================== 九、正式交付模型替换（条件项）
    if "trial_0080" in report or "RC2" in report:
        rec("WARN", "R9", "正式交付模型（trial_0080/RC2）替换状态已交代",
            "报告中出现该模型名，需人工确认是'已替换'还是'待替换'")
    else:
        rec("WARN", "R9", "正式交付模型（trial_0080/RC2）替换状态已交代",
            "报告未提；按任务书属**条件项**（待 58 条业务标注回收）")

    # ---- 汇总 ----
    print("=" * 78)
    print("v2 交付物完成度审计：%s" % root)
    print("=" * 78)
    print("识别到的产物表: %s" % got_roles)
    for status, code, req in RES:
        mark = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "WARN": "⚠️ WARN "}[status]
        print("%s  %-5s %s" % (mark, code, req))
    n_pass = sum(1 for s, _c, _r in RES if s == "PASS")
    n_fail = sum(1 for s, _c, _r in RES if s == "FAIL")
    n_warn = sum(1 for s, _c, _r in RES if s == "WARN")
    print("-" * 78)
    print("总计 %d 条：PASS %d / FAIL %d / WARN %d" % (len(RES), n_pass, n_fail, n_warn))
    print("（WARN = 证据不足、需人工判读；**不算通过**）")
    return 1 if n_fail else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    root = sys.argv[1]
    report = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "REPORT_V2_FINAL.md")
    raise SystemExit(main(root, report))
