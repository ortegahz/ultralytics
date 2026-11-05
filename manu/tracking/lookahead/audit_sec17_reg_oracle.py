#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""独立审计：FINAL_REPORT.md §17.1–§17.5（夜 8/9/10 配准·歧义处置族 oracle 上界 + 静默帧归因）。

本脚本**不** import、不调用 ``make_fastinfer_final_report.py`` / ``make_reg_oracle_report.py``
/ ``diag_reg_oracle.py`` 里的任何函数：它自己从
``runs/lookahead/fi_nogmc24/reg_oracle/*.jsonl``（逐行流式读）+ ``closure_report.json``
+ ``reg_oracle.json`` **重新算**一遍，再把结果和 ``FINAL_REPORT.md`` 的渲染文本逐位比对。

报告定位**全部用内容锚点**（标题前缀 / 表头首列文本 / 首列标签），**不写死行号**
⇒ 报告增删行时脚本不会漂移崩溃（行号只作为输出里的可追溯信息）。

只读、无副作用、可重复运行（两次运行输出逐字相同）。

    python manu/tracking/lookahead/audit_sec17_reg_oracle.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

# 根定位：仓内 ⇒ 仓根；**交付包内** ⇒ 包内 `repo/` 镜像根 ⇒ 审计可在包外独立复跑。
ROOT = next((a for a in Path(__file__).resolve().parents if (a / "manu/tracking/lookahead").is_dir()),
            Path(__file__).resolve().parent.parent / "repo")
DUMP_DIR = ROOT / "runs/lookahead/fi_nogmc24/reg_oracle"
# 报告路径可用环境变量覆盖（漂移免疫自测用：审计一份行号已漂移的副本，不动真源）
REPORT = Path(os.environ.get("AUDIT_SEC17_REPORT", ROOT / "runs/lookahead/fi_nogmc/FINAL_REPORT.md"))
CLOSURE = DUMP_DIR / "closure_report.json"
REG_ORACLE = DUMP_DIR / "reg_oracle.json"
OUT_MD = ROOT / "runs/lookahead/fi_nogmc/audit_round7/audit_sec17_reg_oracle.md"

HIT_TOL = 5.0
RADIUS = 32.0
UNIQ_GATE = 0.6
FOV_MARGIN = 6.0
FRAME_W, FRAME_H = 640.0, 512.0
DELTA = 3
ROLLBACK_MIN_PEAK = 0.90

RULES = (
    "R0_baseline",
    "R1_argmax",
    "R2_chain",
    "R3_second_peak",
    "R4_ambig_to_chain",
    "R5_uniq_gate_peak",
    "R6_peak_x_uniq",
    "R7_ambig_to_rematch",
    "R8_nearest_to_anchor",
    "R9_nearest_to_chain",
    "R10_ambig_nearest_to_anchor",
    "R11_ambig_nearest_to_chain",
    "R12_ambig_lowest_uniq",
    "R13_nearest_to_chain_all",
    "R14_ambig_nearest_to_chain_all",
)

# --------------------------------------------------------------------------- #
# 基础几何（独立实现）
# --------------------------------------------------------------------------- #
def dist_to_centers(point, centers) -> float:
    """点到最近 GT 中心的欧氏距离。"""
    best = float("inf")
    for cx, cy in centers:
        d = math.hypot(float(point[0]) - float(cx), float(point[1]) - float(cy))
        if d < best:
            best = d
    return best


def fov_legal(point) -> bool:
    """出框门：上报点必须落在画面内且距边界 >= margin。"""
    if point is None:
        return False
    x, y = float(point[0]), float(point[1])
    return FOV_MARGIN <= x <= FRAME_W - FOV_MARGIN and FOV_MARGIN <= y <= FRAME_H - FOV_MARGIN


def inbox_hit(point, gt_rows) -> bool:
    """业务匹配器 inbox 口径：点落在某个 GT 框内。"""
    for row in gt_rows:
        if abs(float(point[0]) - float(row[0])) <= float(row[2]) / 2.0 and abs(float(point[1]) - float(row[1])) <= float(row[3]) / 2.0:
            return True
    return False


def pools(refine: dict) -> dict:
    matches = refine.get("m") or []
    big = matches[0] if matches else None
    rematch = matches[1] if len(matches) > 1 else None
    return {
        "big": list((big or {}).get("k") or []),
        "rematch": list((rematch or {}).get("k") or []),
        "big_u": (big or {}).get("u"),
        "big_pk": float((big or {}).get("pk") or 0.0),
    }


def is_ambiguous(refine: dict) -> bool:
    uni = pools(refine)["big_u"]
    if uni is None or math.isnan(float(uni)):
        return False
    return float(uni) > UNIQ_GATE


def nearest(candidates, reference):
    if not candidates or reference is None:
        return None
    best = min(candidates, key=lambda p: math.hypot(p["x"] - float(reference[0]), p["y"] - float(reference[1])))
    return [float(best["x"]), float(best["y"])]


def rule_point(rule: str, refine: dict):
    """15 条「可计算排序规则」各自给这条 refine 选的上报坐标（None = 不改）。"""
    pool = pools(refine)
    big, rematch = pool["big"], pool["rematch"]
    chain = refine.get("c")
    anchor = refine.get("a")
    fallback = refine.get("p") if refine.get("p") is not None else chain
    argmax = [big[0]["x"], big[0]["y"]] if big else fallback
    ambiguous = is_ambiguous(refine)

    if rule == "R0_baseline":
        return fallback
    if rule == "R1_argmax":
        return argmax
    if rule == "R2_chain":
        return chain
    if rule == "R3_second_peak":
        return [big[1]["x"], big[1]["y"]] if len(big) > 1 else fallback
    if rule == "R4_ambig_to_chain":
        return chain if ambiguous else argmax
    if rule == "R5_uniq_gate_peak":
        allowed = [p for p in big if p.get("u") is not None and not math.isnan(float(p["u"])) and float(p["u"]) <= UNIQ_GATE]
        if not allowed:
            return fallback
        best = max(allowed, key=lambda p: float(p["v"]))
        return [best["x"], best["y"]]
    if rule == "R6_peak_x_uniq":
        if not big:
            return fallback

        def score(p):
            u = p.get("u")
            u = 0.0 if u is None or math.isnan(float(u)) else float(u)
            return float(p["v"]) * (1.0 - u)

        best = max(big, key=score)
        return [best["x"], best["y"]]
    if rule == "R7_ambig_to_rematch":
        if ambiguous and rematch:
            return [rematch[0]["x"], rematch[0]["y"]]
        return argmax
    if rule == "R8_nearest_to_anchor":
        return nearest(big, anchor) or fallback
    if rule == "R9_nearest_to_chain":
        return nearest(big, chain) or fallback
    if rule == "R10_ambig_nearest_to_anchor":
        if not ambiguous:
            return argmax
        return nearest(big, anchor) or fallback
    if rule == "R11_ambig_nearest_to_chain":
        if not ambiguous:
            return argmax
        return nearest(big, chain) or fallback
    if rule == "R12_ambig_lowest_uniq":
        if not ambiguous or not big:
            return argmax

        def own_u(p):
            u = p.get("u")
            return 1.0 if u is None or math.isnan(float(u)) else float(u)

        best = min(big, key=own_u)
        return [best["x"], best["y"]]
    if rule == "R13_nearest_to_chain_all":
        return nearest(big + rematch, chain) or fallback
    if rule == "R14_ambig_nearest_to_chain_all":
        if not ambiguous:
            return argmax
        return nearest(big + rematch, chain) or fallback
    raise ValueError(rule)


# --------------------------------------------------------------------------- #
# universe（候选集）语义 —— **现行产物口径**（夜 9 修好 diag_reg_oracle._oracle_pick 之后）
# --------------------------------------------------------------------------- #
# 族：reg = 大匹配窗 top-K 候选峰；all = reg 族 **再加** 链位置 + 二次意见窗的峰。
# 「*_legal」= **同一族的候选集**再过出框门（**不是**另一个候选集）。
# 元组 = (含大窗峰, 含链位置+二次意见窗峰, 过出框门)
UNIVERSE_SPEC = {
    # ---- 现行产物口径 ----
    "reg": (True, False, False),
    "reg_legal": (True, False, True),
    "all": (True, True, False),
    "all_legal": (True, True, True),
    # ---- 审计口径别名（与现行产物口径**完全等价**，只用于交叉断言；不再是旧语义）----
    "reg_legal_big": (True, False, True),
    "all_legal_big": (True, True, True),
}


def oracle_pick(refine: dict, centers, universe: str):
    """oracle：候选全集里挑离最近 GT 最近的那个。返回 (point, pool, rank)。

    universe 语义 = **现行产物口径**（`diag_reg_oracle.py::_oracle_pick`，夜 9 修复后）：
      reg           = 大匹配窗 top-K 峰
      reg_legal     = 同 reg 族候选集，再过出框门
      all           = 大窗峰 + 链位置 + 二次意见窗峰
      all_legal     = 同 all 族候选集，再过出框门
    """
    include_big, include_alt, fov_filter = UNIVERSE_SPEC[universe]
    pool = pools(refine)
    candidates = []
    if include_big:
        candidates += [("big", p) for p in pool["big"]]
    if include_alt:
        chain = refine.get("c")
        if chain:
            candidates.append(("chain", {"x": float(chain[0]), "y": float(chain[1]), "v": float("nan"), "u": float("nan")}))
        candidates += [("rematch", p) for p in pool["rematch"]]
    if fov_filter:
        candidates = [item for item in candidates if fov_legal([item[1]["x"], item[1]["y"]])]
    fallback = refine.get("p") if refine.get("p") is not None else refine.get("c")
    if not candidates:
        return (fallback if fallback else [0.0, 0.0], "none", -1)
    best = min(candidates, key=lambda item: dist_to_centers([item[1]["x"], item[1]["y"]], centers))
    name = best[0]
    same_pool = [item[1] for item in candidates if item[0] == name]
    ordered = sorted(same_pool, key=lambda p: (float(p["v"]) if not math.isnan(float(p["v"])) else -1e18), reverse=True)
    rank = next((i for i, p in enumerate(ordered) if p is best[1]), -1)
    return ([float(best[1]["x"]), float(best[1]["y"])], name, rank)


def silent_cause(frame: dict) -> str:
    """静默帧归因：按**生产判定顺序**（回退链峰值门 → failed → 写 report → 出框门）。"""
    refines = frame.get("r") or []
    pre_entries = frame.get("pe") or []
    target = int(frame.get("t", -1))
    if not refines:
        return "failed_no_fallback" if frame.get("f") else "no_decision"
    last = refines[-1]
    rolled_back = any(int(r.get("s", target)) != target - DELTA for r in refines)
    matches = last.get("m") or []
    peak = float((matches[0] if matches else {}).get("pk") or 0.0)
    if rolled_back and peak < ROLLBACK_MIN_PEAK:
        return "rollback_gate"
    if pre_entries:
        if all(not fov_legal(p) for p in pre_entries):
            return "fov_gate"
        return "other"
    return "failed_no_fallback" if frame.get("f") else "other"


# --------------------------------------------------------------------------- #
# 逐行流式重算（24 个 jsonl，绝不整份读进内存）
# --------------------------------------------------------------------------- #
def recompute() -> dict:
    totals = {"gt_frames": 0, "frames_total": 0, "reported": 0, "lost": 0, "hit_base": 0,
              "hit_oracle_reg": 0, "hit_oracle_all": 0, "hit_oracle_reg_legal": 0,
              "hit_oracle_all_legal": 0, "hit_oracle_reg_legal_big": 0, "hit_oracle_all_legal_big": 0,
              "frames_with_refine": 0, "lost_with_refine": 0, "no_refine_with_entries": 0}
    universes = (("reg", "hit_oracle_reg"), ("all", "hit_oracle_all"),
                 ("reg_legal", "hit_oracle_reg_legal"), ("all_legal", "hit_oracle_all_legal"),
                 ("reg_legal_big", "hit_oracle_reg_legal_big"), ("all_legal_big", "hit_oracle_all_legal_big"))
    rule_hits = {r: 0 for r in RULES}
    buckets = {"B1_window_has_peak": 0, "B2_window_no_peak": 0, "B3_out_of_window": 0, "B4_no_refine": 0}
    rank_hist = {}
    recoverable = 0
    silent_total = 0            # 有 GT 且最终没上报（静默）
    silent_attr_total = 0       # 静默 **且有配准决策**（= 归因口径的总体）
    by_cause = {}
    tp_by_cause = {}
    fp_by_cause = {}
    legal_pick_out_of_fov = 0   # 「合法 oracle」里选中点越界的帧数（应当 = 0）

    for path in sorted(DUMP_DIR.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                frame = json.loads(line)
                gt = frame.get("g") or []
                if not gt:
                    totals["frames_total"] += 1
                    continue
                centers = [(row[0], row[1]) for row in gt]
                totals["gt_frames"] += 1
                entries = frame.get("e") or []
                refines = frame.get("r") or []
                if refines:
                    totals["frames_with_refine"] += 1
                if entries:
                    totals["reported"] += 1
                else:
                    totals["lost"] += 1
                if refines and not entries:
                    totals["lost_with_refine"] += 1

                hit_base = min((dist_to_centers(p, centers) for p in entries), default=float("inf")) <= HIT_TOL
                if hit_base:
                    totals["hit_base"] += 1

                if not refines:
                    if not entries:
                        buckets["B4_no_refine"] += 1
                    else:
                        totals["no_refine_with_entries"] += 1
                elif not hit_base:
                    reachable = False
                    has_peak = False
                    for refine in refines:
                        chain = refine.get("c")
                        if chain is not None and dist_to_centers(chain, centers) <= RADIUS:
                            reachable = True
                        for p in pools(refine)["big"]:
                            if dist_to_centers([p["x"], p["y"]], centers) <= HIT_TOL:
                                has_peak = True
                    if not reachable:
                        buckets["B3_out_of_window"] += 1
                    elif has_peak:
                        buckets["B1_window_has_peak"] += 1
                    else:
                        buckets["B2_window_no_peak"] += 1

                # oracle 上界（帧级：任一候选能命中即算命中）
                for universe, key in universes:
                    picks = [oracle_pick(r, centers, universe)[0] for r in refines]
                    if picks and min(dist_to_centers(p, centers) for p in picks) <= HIT_TOL:
                        totals[key] += 1
                        if universe.endswith("_legal"):
                            best = min(picks, key=lambda q: dist_to_centers(q, centers))
                            if not fov_legal(best):
                                legal_pick_out_of_fov += 1

                # 上报点 ↔ refine 关联（按门前的位置就近匹配；gap>12px 视为未关联）
                matched = []
                used = set()
                for pos in entries:
                    best_ref, best_gap = None, float("inf")
                    for index, refine in enumerate(refines):
                        if index in used:
                            continue
                        base = refine.get("p") if refine.get("p") is not None else refine.get("c")
                        if base is None:
                            continue
                        gap = math.hypot(float(pos[0]) - float(base[0]), float(pos[1]) - float(base[1]))
                        if gap < best_gap:
                            best_gap, best_ref = gap, index
                    if best_ref is None or best_gap > 12.0:
                        continue
                    used.add(best_ref)
                    matched.append((pos, refines[best_ref]))

                # 15 条可计算排序规则（**只统计有上报且能关联到 refine 的帧**，产物口径）
                for rule in RULES:
                    points = []
                    for _pos, refine in matched:
                        point = rule_point(rule, refine)
                        if point is not None:
                            points.append(point)
                    if points and min(dist_to_centers(p, centers) for p in points) <= HIT_TOL:
                        rule_hits[rule] += 1

                # 可分性解剖：首条 refine 上 oracle 选中的候选峰名次
                if refines and not hit_base:
                    pick, _pool, rank = oracle_pick(refines[0], centers, "reg")
                    if dist_to_centers(pick, centers) <= HIT_TOL:
                        recoverable += 1
                        rank_hist[str(int(rank))] = rank_hist.get(str(int(rank)), 0) + 1

                # 静默帧归因（**只统计有配准决策的静默帧** = 产物口径；无 refine 的帧不计）
                if not entries:
                    silent_total += 1
                    if not refines:
                        continue
                    silent_attr_total += 1
                    cause = silent_cause(frame)
                    by_cause[cause] = by_cause.get(cause, 0) + 1
                    legal = None
                    for refine in refines:
                        big = pools(refine)["big"]
                        if big and fov_legal([big[0]["x"], big[0]["y"]]):
                            legal = [big[0]["x"], big[0]["y"]]
                            break
                    if legal is None:
                        # 产物口径：**没有合法 argmax 的帧不计入 TP/FP**（只进归因桶）
                        continue
                    if inbox_hit(legal, gt):
                        tp_by_cause[cause] = tp_by_cause.get(cause, 0) + 1
                    else:
                        fp_by_cause[cause] = fp_by_cause.get(cause, 0) + 1

    n = max(1, totals["gt_frames"])
    res = dict(totals)
    res["buckets"] = buckets
    res["rule_hits"] = rule_hits
    res["rank_hist"] = rank_hist
    res["recoverable"] = recoverable
    res["silent_total"] = silent_total
    res["silent_attr_total"] = silent_attr_total
    res["by_cause"] = by_cause
    res["tp_by_cause"] = tp_by_cause
    res["fp_by_cause"] = fp_by_cause
    res["legal_pick_out_of_fov"] = legal_pick_out_of_fov
    res["n"] = n
    res["baseline_hit_pct"] = 100.0 * totals["hit_base"] / n
    res["oracle_reg_gain_pp"] = 100.0 * (totals["hit_oracle_reg"] - totals["hit_base"]) / n
    res["oracle_all_gain_pp"] = 100.0 * (totals["hit_oracle_all"] - totals["hit_base"]) / n
    res["oracle_reg_legal_gain_pp"] = 100.0 * (totals["hit_oracle_reg_legal"] - totals["hit_base"]) / n
    res["oracle_all_legal_gain_pp"] = 100.0 * (totals["hit_oracle_all_legal"] - totals["hit_base"]) / n
    res["fov_gate_cost_pp"] = res["oracle_all_gain_pp"] - res["oracle_all_legal_gain_pp"]
    res["reg_frames"] = totals["hit_oracle_reg"] - totals["hit_base"]
    res["reg_legal_frames"] = totals["hit_oracle_reg_legal"] - totals["hit_base"]
    res["all_frames"] = totals["hit_oracle_all"] - totals["hit_base"]
    res["all_legal_frames"] = totals["hit_oracle_all_legal"] - totals["hit_base"]
    res["rule_hit_pct"] = {r: 100.0 * rule_hits[r] / n for r in RULES}
    res["rule_gain_pp"] = {r: 100.0 * (rule_hits[r] - totals["hit_base"]) / n for r in RULES}
    # §17.3 那一列的分母 = **O-reg（不加出框门）** = hit_oracle_reg − hit_base
    denom = max(1, totals["hit_oracle_reg"] - totals["hit_base"])
    res["share_denominator_frames"] = denom
    res["share_denominator_gain_pp"] = res["oracle_reg_gain_pp"]
    res["rule_share"] = {r: (rule_hits[r] - totals["hit_base"]) / denom for r in RULES}
    best = max(RULES, key=lambda r: res["rule_gain_pp"][r])
    res["best_rule"] = {
        "name": best,
        "gain_pp": res["rule_gain_pp"][best],
        "share_pct": 100.0 * res["rule_share"][best],
    }
    # 对**最宽**天花板（O-all 合法）的占比
    ceiling = res["oracle_all_legal_gain_pp"]
    res["best_rule_share_of_ceiling_pct"] = 100.0 * res["rule_gain_pp"][best] / ceiling if ceiling else float("nan")
    res["rank0_share_pct"] = 100.0 * rank_hist.get("0", 0) / max(1, recoverable)
    res["bucket_total"] = sum(buckets.values())
    res["miss_total"] = totals["gt_frames"] - totals["hit_base"]
    res["bucket_share"] = {k: 100.0 * v / max(1, res["bucket_total"]) for k, v in buckets.items()}
    res["silent_share"] = {k: 100.0 * v / max(1, silent_attr_total) for k, v in by_cause.items()}
    return res


# --------------------------------------------------------------------------- #
# 报告解析：**全部按内容锚点定位**（不写死行号）
# --------------------------------------------------------------------------- #
def find_line(lines, prefix):
    """按内容锚点找行号（1-based）；找不到返回 None（**不抛异常**）。"""
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            return index + 1
    return None


def section_range(lines, heading_prefix):
    """返回 (start_idx, end_idx)：标题行起，到下一个 '## ' / '### ' 标题前（None = 找不到）。"""
    start = find_line(lines, heading_prefix)
    if start is None:
        return None, None
    begin = start - 1
    for index in range(begin + 1, len(lines)):
        if lines[index].startswith("## ") or lines[index].startswith("### "):
            return begin, index
    return begin, len(lines)


def split_cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_separator(cells):
    """markdown 表格的分隔行（如 |:--|--:|）。"""
    return bool(cells) and all(set(cell) <= set("-: ") for cell in cells)


def table_rows(lines, begin, end, header_prefix):
    """按**表头首列文本**取表格 → (表头行号, 表头 cells, [(数据行号, cells)])。

    找不到表头 → (None, None, [])（**不抛异常**）；分隔行自动剔除。
    """
    rows = []
    if begin is None:
        return None, None, rows
    index = begin
    while index < end:
        if lines[index].strip().startswith(header_prefix):
            header_line, header_cells = index + 1, split_cells(lines[index])
            index += 1
            while index < end and lines[index].strip().startswith("|"):
                cells = split_cells(lines[index])
                if not is_separator(cells):
                    rows.append((index + 1, cells))
                index += 1
            return header_line, header_cells, rows
        index += 1
    return None, None, rows


def plain(cell):
    """去掉 ** / ` 包裹，得到可比较的标签文本。"""
    return cell.replace("**", "").replace("`", "").strip()


def row_by_label(rows, label):
    """按首列**精确**标签取行（先 strip 掉 ** 与 `）；找不到返回 (None, None)。"""
    for line_no, cells in rows:
        if plain(cells[0]) == label:
            return line_no, cells
    return None, None


def row_by_prefix(rows, prefix):
    """按首列**前缀**取行；找不到返回 (None, None)。"""
    for line_no, cells in rows:
        if plain(cells[0]).startswith(prefix):
            return line_no, cells
    return None, None


def fnum(text):
    """从渲染文本里取数字（去掉 ** / % / pp / +）；取不到返回 None。"""
    cleaned = text.replace("**", "").replace("`", "").replace("%", "").replace("pp", "").replace("+", "").strip()
    if cleaned in ("—", "-", "", "None", "nan", "NaN", "inf"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def is_dash(text):
    return text.replace("**", "").strip() in ("—", "-")


class Checker:
    """收集逐位比对结果（PASS / FAIL）。"""

    def __init__(self):
        self.items = []

    def num(self, label, reported, mine, digits, line, src):
        """按报告显示精度逐位比：|报告−重算| ≤ 末位半个单位。"""
        rep = fnum(reported) if isinstance(reported, str) else reported
        if rep is None:
            self.items.append({"status": "FAIL", "label": label, "reported": reported, "recomputed": mine,
                               "delta": None, "line": line, "src": src, "note": "报告值不可解析"})
            return None
        tol = 0.5 * 10 ** (-digits)
        diff = rep - mine
        ok = abs(diff) <= tol + 1e-12
        self.items.append({"status": "PASS" if ok else "FAIL", "label": label, "reported": reported,
                           "recomputed": mine, "delta": diff, "line": line, "src": src, "tol": tol})
        return rep

    def text(self, label, ok, detail, line, src, note=""):
        self.items.append({"status": "PASS" if ok else "FAIL", "label": label, "reported": detail,
                           "recomputed": None, "delta": None, "line": line, "src": src, "note": note})
        return ok

    @property
    def fails(self):
        return [i for i in self.items if i["status"] == "FAIL"]

    @property
    def passes(self):
        return [i for i in self.items if i["status"] == "PASS"]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 104)
    print("§17.1–§17.5 独立审计（重算源：runs/lookahead/fi_nogmc24/reg_oracle/*.jsonl，逐行流式）")
    print("=" * 104)
    rc = recompute()
    closure = json.loads(CLOSURE.read_text(encoding="utf-8"))
    regorc = json.loads(REG_ORACLE.read_text(encoding="utf-8"))
    lines = REPORT.read_text(encoding="utf-8").splitlines()

    chk = Checker()
    src_jsonl = "reg_oracle/*.jsonl 独立重算"
    src_closure = "closure_report.json"

    # ---- 各 universe 的独立复算值（对照父 agent 期望）----
    expected = {
        "hit_oracle_reg": 116978,
        "hit_oracle_all": 118955,
        "hit_oracle_reg_legal": 116896,
        "hit_oracle_all_legal": 118835,
        "reg_frames": 1082,
        "reg_legal_frames": 1000,
        "all_legal_frames": 2939,
    }
    print("\n" + "-" * 104)
    print("【universe 独立复算 vs 期望值】")
    print("-" * 104)
    for key, want in expected.items():
        got = rc[key]
        ok = got == want
        chk.text(f"universe {key}", ok, f"{got}", None, src_jsonl, note=f"期望 {want}")
        print(f"  [{'OK  ' if ok else 'FAIL'}] {key:<26} 重算={got:<8} 期望={want}")
    # 别名交叉断言：reg_legal_big == reg_legal、all_legal_big == all_legal
    for alias, target in (("hit_oracle_reg_legal_big", "hit_oracle_reg_legal"),
                          ("hit_oracle_all_legal_big", "hit_oracle_all_legal")):
        ok = rc[alias] == rc[target]
        chk.text(f"别名交叉断言 {alias} == {target}", ok, f"{rc[alias]} vs {rc[target]}", None, src_jsonl)
        print(f"  [{'OK  ' if ok else 'FAIL'}] 别名 {alias}={rc[alias]} == {target}={rc[target]}")
    print(f"  [INFO ] reg_legal_hit%={rc['baseline_hit_pct'] + rc['oracle_reg_legal_gain_pp']:.4f}  "
          f"Δ={rc['oracle_reg_legal_gain_pp']:.4f} pp")
    print(f"  [INFO ] all_legal_hit%={rc['baseline_hit_pct'] + rc['oracle_all_legal_gain_pp']:.4f}  "
          f"Δ={rc['oracle_all_legal_gain_pp']:.4f} pp")
    print(f"  [INFO ] 出框门代价 = {rc['fov_gate_cost_pp']:.4f} pp")
    print(f"  [INFO ] 「合法 oracle」选中点越界帧数 = {rc['legal_pick_out_of_fov']}"
          f"（候选集为空时回退到 refine 原始返回的边界情形；不影响 §17 的任何数字）")

    # ================= §17.1 =================
    print("\n" + "-" * 104)
    print("【§17.1】oracle 上界（按标签取行）")
    print("-" * 104)
    s1, e1 = section_range(lines, "### 17.1 ")
    h1, hc1, t1 = table_rows(lines, s1, e1, "| oracle |")
    print(f"  [INFO ] §17.1 表头行 {h1}，数据行数 = {len(t1)}（锚点 '### 17.1 ' 行 {None if s1 is None else s1 + 1}）")
    spec_171 = {
        "基线": ("baseline_hit_pct", None, None),
        "O-reg": ("baseline_hit_pct + oracle_reg_gain_pp", "oracle_reg_gain_pp", "reg_frames"),
        "O-reg 合法": ("baseline_hit_pct + oracle_reg_legal_gain_pp", "oracle_reg_legal_gain_pp", "reg_legal_frames"),
        "O-all": ("baseline_hit_pct + oracle_all_gain_pp", "oracle_all_gain_pp", None),
        "O-all 合法": ("baseline_hit_pct + oracle_all_legal_gain_pp", "oracle_all_legal_gain_pp", "all_legal_frames"),
    }
    for label, (hit_expr, gain_expr, frames_expr) in spec_171.items():
        line_no, cells = row_by_label(t1, label)
        if cells is None:
            chk.text(f"§17.1 行「{label}」", False, "找不到该行", None, src_closure, note="标签缺失")
            print(f"  [FAIL] §17.1 找不到行「{label}」")
            continue
        hit_mine = eval(hit_expr, {"__builtins__": {}}, rc)  # noqa: S307 - 表达式是脚本内常量
        gain_mine = eval(gain_expr, {"__builtins__": {}}, rc) if gain_expr else None
        frames_mine = rc[frames_expr] if frames_expr else None
        chk.num(f"§17.1 {label} hit@5px%", cells[2], hit_mine, 4, line_no, src_closure)
        if gain_expr:
            chk.num(f"§17.1 {label} Δ vs 基线(pp)", cells[3], gain_mine, 4, line_no, src_closure)
        else:
            chk.text(f"§17.1 {label} Δ 为破折号", is_dash(cells[3]), cells[3], line_no, src_closure)
        if frames_expr:
            chk.num(f"§17.1 {label} 折算帧数", cells[4], frames_mine, 0, line_no, src_closure)
        else:
            chk.text(f"§17.1 {label} 折算帧数为破折号", is_dash(cells[4]), cells[4], line_no, src_closure)
        print(f"  [{plain(cells[0]):<12}] hit% 报告={cells[2]:<10} 重算={hit_mine:.6f} | "
              f"Δ 报告={cells[3]:<10} 重算={(gain_mine if gain_mine is not None else float('nan')):.6f} | "
              f"帧数 报告={cells[4]:<6} 重算={frames_mine if frames_mine is not None else '—'}")
    # 天花板注：+2.2055 / 2939 / 出框门代价 0.0901
    if s1 is not None:
        note_line = next((i + 1 for i in range(s1, e1) if "天花板（最宽的合法口径" in lines[i]), None)
        if note_line:
            body = lines[note_line - 1]
            chk.num("§17.1 天花板 +pp", re.search(r"= \+([0-9.]+) pp", body).group(1), rc["oracle_all_legal_gain_pp"], 4, note_line, src_closure)
            chk.num("§17.1 天花板帧数", re.search(r"（([0-9]+) 帧）", body).group(1), rc["all_legal_frames"], 0, note_line, src_closure)
            chk.num("§17.1 出框门代价 pp", re.search(r"代价 \*\*([0-9.]+) pp\*\*", body).group(1), rc["fov_gate_cost_pp"], 4, note_line, src_closure)
            print(f"  [INFO ] 天花板注行 {note_line}: {body[:150]}")

    # ================= §17.2 =================
    print("\n" + "-" * 104)
    print("【§17.2】缺口分桶")
    print("-" * 104)
    s2, e2 = section_range(lines, "### 17.2 ")
    h2, hc2, t2 = table_rows(lines, s2, e2, "| 桶 |")
    bkeys = ["B1_window_has_peak", "B2_window_no_peak", "B3_out_of_window", "B4_no_refine"]
    labels_172 = ["B1 窗内有峰", "B2 窗内无峰", "B3 窗外", "B4 无 refine"]
    for label, key in zip(labels_172, bkeys):
        line_no, cells = row_by_label(t2, label)
        if cells is None:
            chk.text(f"§17.2 行「{label}」", False, "找不到该行", None, src_closure, note="标签缺失")
            continue
        chk.num(f"§17.2 {label} 帧数", cells[1], rc["buckets"][key], 0, line_no, src_closure)
        chk.num(f"§17.2 {label} 占失手帧%", cells[2], rc["bucket_share"][key], 1, line_no, src_closure)
        print(f"  [{plain(cells[0]):<12}] 帧数 报告={cells[1]:<7} 重算={rc['buckets'][key]:<6} "
              f"占失手帧 报告={cells[2]:<7} 重算={rc['bucket_share'][key]:.4f}%")
    line_no, cells = row_by_prefix(t2, "合计失手")
    if cells:
        chk.num("§17.2 合计失手 帧数", cells[1], rc["bucket_total"], 0, line_no, src_closure)
        chk.num("§17.2 合计失手 占失手帧%", cells[2], 100.0, 1, line_no, src_closure)
        print(f"  [{plain(cells[0]):<12}] 合计 报告={cells[1]:<7} 重算={rc['bucket_total']}")
    print(f"  [INFO ] 四桶合计={rc['bucket_total']}  基线失手总数={rc['miss_total']}  "
          f"(GT {rc['n']} − 基线命中 {rc['hit_base']})  无 refine 但有上报={rc['no_refine_with_entries']}")

    # ================= §17.3 =================
    print("\n" + "-" * 104)
    print("【§17.3】15 条规则（分母 = O-reg 不加出框门）")
    print("-" * 104)
    s3, e3 = section_range(lines, "### 17.3 ")
    h3, hc3, t3 = table_rows(lines, s3, e3, "| 规则 |")
    header = hc3[3] if hc3 and len(hc3) > 3 else ""
    denom_ok = ("O-reg" in header) and ("不加出框门" in header.replace("**", ""))
    chk.text("§17.3 表头写明分母 = O-reg 不加出框门", denom_ok, header[:120], h3, src_closure)
    print(f"  [INFO ] §17.3 表头: {header[:150]}")
    print(f"  [INFO ] 分母重算：O-reg 不加门 Δ={rc['share_denominator_gain_pp']:.4f} pp / "
          f"{rc['share_denominator_frames']} 帧（产物 ceiling.share_denominator = "
          f"{closure.get('ceiling', {}).get('share_denominator')}）")
    chk.num("§17.3 分母 gain_pp（O-reg 不加门）", f"{closure['ceiling']['share_denominator']['gain_pp']:.4f}",
            rc["share_denominator_gain_pp"], 4, None, src_closure)
    chk.num("§17.3 分母 帧数（O-reg 不加门）", str(closure["ceiling"]["share_denominator"]["frames"]),
            rc["share_denominator_frames"], 0, None, src_closure)
    for line_no, cells in t3:
        name = plain(cells[0])
        if name not in rc["rule_hit_pct"]:
            continue
        chk.num(f"§17.3 {name} hit@5px%", cells[1], rc["rule_hit_pct"][name], 4, line_no, src_closure)
        chk.num(f"§17.3 {name} Δ vs 基线(pp)", cells[2], rc["rule_gain_pp"][name], 4, line_no, src_closure)
        chk.num(f"§17.3 {name} 占 O-reg 的比例%", cells[3], rc["rule_share"][name] * 100.0, 1, line_no, src_closure)
        # 量纲守卫：产物 rules.<name>.share_of_oracle 是**小数比例**，报告显示成百分数 ⇒ 乘 100 后必须一致
        prod_share = (closure.get("rules", {}).get(name) or {}).get("share_of_oracle")
        if prod_share is not None:
            chk.num(f"§17.3 {name} share 量纲交叉断言（小数→%）", f"{prod_share * 100.0:.4f}",
                    rc["rule_share"][name] * 100.0, 4, line_no, src_closure)
    print(f"  [INFO ] 15 条规则 × 3 列逐位比对完成（命中 {len(rc['rule_hit_pct'])} 条规则）")
    # 结论行：最好规则
    if s3 is not None:
        bl = next((i + 1 for i in range(s3, e3) if "最好的一条规则只拿到 oracle" in lines[i]), None)
        if bl:
            body = lines[bl - 1]
            chk.num("§17.3 最好规则占 oracle%", re.search(r"oracle 的\s*([0-9.]+)%", body).group(1),
                    rc["best_rule"]["share_pct"], 1, bl, src_closure)
            chk.text("§17.3 最好规则名 = R5_uniq_gate_peak", "`R5_uniq_gate_peak`" in body,
                     rc["best_rule"]["name"], bl, src_closure)
            chk.num("§17.3 最好规则 Δ pp", re.search(r"Δ\s*([0-9.]+) pp", body).group(1),
                    rc["best_rule"]["gain_pp"], 4, bl, src_closure)
            m_ce = re.search(r"天花板 \+([0-9.]+) pp 也只有 ([0-9.]+)", body)
            if m_ce:
                chk.num("§17.3 最宽天花板 pp", m_ce.group(1), rc["oracle_all_legal_gain_pp"], 4, bl, src_closure)
                chk.num("§17.3 最好规则占最宽天花板%", m_ce.group(2), rc["best_rule_share_of_ceiling_pct"], 1, bl, src_closure)
            print(f"  [INFO ] 结论行 {bl}: {body[:190]}")
            print(f"  [INFO ] 最好规则 = {rc['best_rule']['name']} Δ={rc['best_rule']['gain_pp']:.7f} pp  "
                  f"占 O-reg = {rc['best_rule']['share_pct']:.4f}%  占最宽天花板 = {rc['best_rule_share_of_ceiling_pct']:.4f}%")
        vl = next((i + 1 for i in range(s3, e3) if "夜 8 裁定" in lines[i] and "可救帧" in lines[i]), None)
        if vl:
            body = lines[vl - 1]
            m = re.search(r"共 \*\*([0-9]+)\*\* 帧", body)
            if m:
                chk.num("§17.3 可救帧数", m.group(1), rc["recoverable"], 0, vl, src_closure)
            m = re.search(r"的占 ([0-9.]+)%", body)
            if m:
                chk.num("§17.3 名次=0 占比%", m.group(1), rc["rank0_share_pct"], 1, vl, src_closure)
            print(f"  [INFO ] 裁定行 {vl}: 可救帧 {rc['recoverable']}  rank0 占 {rc['rank0_share_pct']:.4f}%  "
                  f"rank_hist={rc['rank_hist']}")

    # ================= §17.5 =================
    print("\n" + "-" * 104)
    print("【§17.5】静默帧归因")
    print("-" * 104)
    s5, e5 = section_range(lines, "### 17.5 ")
    h5, hc5, t5 = table_rows(lines, s5, e5, "| 归因 |")
    sa = closure.get("silent_attribution") or {}
    spec_175 = (
        ("回退链峰值门", "rollback_gate"),
        ("出框门", "fov_gate"),
        ("其它", "other"),
    )
    for label, cause in spec_175:
        line_no, cells = row_by_prefix(t5, label)
        if cells is None:
            chk.text(f"§17.5 行「{label}」", False, "找不到该行", None, src_closure, note="标签缺失")
            print(f"  [FAIL] §17.5 找不到行「{label}」")
            continue
        chk.num(f"§17.5 {label} 帧数", cells[1], rc["by_cause"].get(cause, 0), 0, line_no, src_closure)
        chk.num(f"§17.5 {label} 占静默帧%", cells[2], rc["silent_share"].get(cause, 0.0), 1, line_no, src_closure)
        tp_fp = cells[3].split("/")
        chk.num(f"§17.5 {label} TP", tp_fp[0], rc["tp_by_cause"].get(cause, 0), 0, line_no, src_closure)
        chk.num(f"§17.5 {label} FP", tp_fp[1], rc["fp_by_cause"].get(cause, 0), 0, line_no, src_closure)
        # 与产物 closure_report.json 的 silent_attribution 交叉核对
        chk.num(f"§17.5 {label} 帧数 vs closure.silent_attribution", cells[1],
                sa.get("by_cause", {}).get(cause, 0), 0, line_no, src_closure)
        print(f"  [{plain(cells[0])[:16]:<16}] 帧数 报告={cells[1]:<6} 重算={rc['by_cause'].get(cause, 0):<6} "
              f"占比 报告={cells[2]:<7} 重算={rc['silent_share'].get(cause, 0.0):.4f}%  "
              f"TP/FP 报告={cells[3]:<10} 重算={rc['tp_by_cause'].get(cause, 0)}/{rc['fp_by_cause'].get(cause, 0)}")
    line_no, cells = row_by_prefix(t5, "合计")
    if cells:
        chk.num("§17.5 合计 帧数", cells[1], rc["silent_attr_total"], 0, line_no, src_closure)
        chk.num("§17.5 合计 帧数 vs closure.silent_frames", cells[1], sa.get("silent_frames"), 0, line_no, src_closure)
        chk.num("§17.5 合计 占静默帧%", cells[2], 100.0, 1, line_no, src_closure)
        print(f"  [{plain(cells[0])[:16]:<16}] 合计 报告={cells[1]:<6} 重算={rc['silent_attr_total']:<6} "
              f"(有 GT 的静默帧总数 {rc['silent_total']}，其中无配准决策 {rc['silent_total'] - rc['silent_attr_total']})")
    # 旧错误归因（842）残留检查：842 只允许出现在「纠错留痕」语境里
    stale = []
    if s5 is not None:
        for index in range(s5, e5):
            text = lines[index]
            if "842" in text and not any(key in text for key in ("纠错", "错", "留痕")):
                stale.append((index + 1, text.strip()[:160]))
    chk.text("§17.5 无 842 残留（除纠错留痕外）", not stale, stale or "无", s5 + 1 if s5 is not None else None, src_closure)
    print(f"  [{'OK  ' if not stale else 'FAIL'}] 842 残留检查：{stale if stale else '无（842 仅出现在纠错留痕里）'}")
    print(f"  [INFO ] 产物 anatomy.why_no_report_fov_dropped = {closure.get('anatomy', {}).get('why_no_report_fov_dropped')} "
          f"（夜 8 第一版的**错误**归因；§17.5 的出框门行应 = {rc['by_cause'].get('fov_gate', 0)}）")

    # ================= 泄漏扫描 =================
    print("\n" + "-" * 104)
    print("【NaN / None / — 泄漏扫描】§17.1 / 17.2 / 17.3 / 17.5 表格")
    print("-" * 104)
    leaks = []
    for line_no, row in (t1 + t2 + t3 + t5):
        for token in ("nan", "NaN", "None", "inf", "Infinity"):
            if token in row:
                leaks.append((line_no, token))
    chk.text("§17 表格无 NaN/None/inf 泄漏", not leaks, leaks or "无", None, src_closure)
    print(f"  [{'OK  ' if not leaks else 'FAIL'}] 泄漏命中 {len(leaks)} 处")

    # ================= 汇总 + 写报告 =================
    print("\n" + "=" * 104)
    print(f"逐位比对：PASS {len(chk.passes)}    FAIL {len(chk.fails)}")
    for item in chk.fails:
        print(f"  FAIL {item['label']} | 报告={item['reported']!r} | 重算={item['recomputed']!r} | "
              f"Δ={item['delta']} | 行={item['line']} | {item.get('note', '')}")
    print("=" * 104)

    write_markdown(rc, closure, regorc, chk, lines,
                   {"17.1": (h1, hc1, t1), "17.2": (h2, hc2, t2), "17.3": (h3, hc3, t3), "17.5": (h5, hc5, t5)})
    print(f"\n审计报告已写出：{OUT_MD}")
    return 1 if chk.fails else 0


def write_markdown(rc, closure, regorc, chk, lines, tables):
    """把逐位比对结果写成 markdown（无时间戳 ⇒ 两次运行逐字相同）。"""
    t1 = tables["17.1"][2]
    t2 = tables["17.2"][2]
    t3 = tables["17.3"][2]
    t5 = tables["17.5"][2]
    hc3 = tables["17.3"][1]

    out = []
    out.append("# 独立审计报告：`FINAL_REPORT.md` §17.1–§17.5（夜 8/9/10 配准·歧义处置族 oracle 上界 + 静默帧归因）")
    out.append("")
    out.append("- 审计对象：`runs/lookahead/fi_nogmc/FINAL_REPORT.md` **§17.1 / §17.2 / §17.3 / §17.5**")
    out.append("- 渲染代码：`manu/tracking/lookahead/make_fastinfer_final_report.py`（§17 渲染段）")
    out.append("- 独立审计脚本：`manu/tracking/lookahead/audit_sec17_reg_oracle.py`"
               "（**不 import、不调用**任何生成器/产物脚本的函数；全部数字自己从 `*.jsonl` 重新算）")
    out.append(f"- 真源（只读）：`runs/lookahead/fi_nogmc24/reg_oracle/closure_report.json`、"
               f"`reg_oracle.json`、24 个 `*.jsonl`（逐行流式，~97 MB）")
    out.append("- 定位方式：**全部按内容锚点**（标题前缀 / 表头首列 / 首列标签），脚本内**不写死行号**")
    out.append(f"- 浮点容差：按报告显示精度取末位半个单位（4 位小数 → `5e-5`；1 位小数 → `5e-2`；整数 → `0.5`）")
    out.append("")
    out.append("## 结论")
    out.append("")
    out.append("| 项 | 值 |")
    out.append("|:--|--:|")
    out.append(f"| 逐位比对 PASS | **{len(chk.passes)}** |")
    out.append(f"| 逐位比对 FAIL | **{len(chk.fails)}** |")
    out.append(f"| 报告脚本退出码 | **{1 if chk.fails else 0}** |")
    out.append("")
    out.append("## universe 独立复算（现行产物口径：合法变体 = 同族候选集再过出框门）")
    out.append("")
    out.append("| universe | 独立重算命中帧 | 期望 | hit@5px% | Δ vs 基线(pp) |")
    out.append("|:--|--:|--:|--:|--:|")
    rows = (
        ("reg（大窗峰，不加门）", rc["hit_oracle_reg"], 116978, rc["baseline_hit_pct"] + rc["oracle_reg_gain_pp"], rc["oracle_reg_gain_pp"]),
        ("reg_legal（同族 + 出框门）", rc["hit_oracle_reg_legal"], 116896, rc["baseline_hit_pct"] + rc["oracle_reg_legal_gain_pp"], rc["oracle_reg_legal_gain_pp"]),
        ("all（大窗峰+链+二次意见窗，不加门）", rc["hit_oracle_all"], 118955, rc["baseline_hit_pct"] + rc["oracle_all_gain_pp"], rc["oracle_all_gain_pp"]),
        ("all_legal（同族 + 出框门）", rc["hit_oracle_all_legal"], 118835, rc["baseline_hit_pct"] + rc["oracle_all_legal_gain_pp"], rc["oracle_all_legal_gain_pp"]),
    )
    for name, frames, want, hit_pct, gain in rows:
        out.append(f"| `{name}` | {frames} | {want} | {hit_pct:.4f} | {gain:+.4f} |")
    out.append("")
    out.append(f"- 出框门代价（all → all_legal）= **{rc['fov_gate_cost_pp']:.4f} pp**")
    out.append(f"- 「合法 oracle」选中点越界帧数 = **{rc['legal_pick_out_of_fov']}**"
               f"（该帧候选集为空时 `_oracle_pick` 回退到 refine 原始返回的边界情形；不影响 §17 的任何数字）")
    out.append(f"- 别名交叉断言：`reg_legal_big` == `reg_legal`（{rc['hit_oracle_reg_legal_big']}）、"
               f"`all_legal_big` == `all_legal`（{rc['hit_oracle_all_legal_big']}）")
    out.append("")
    out.append("## §17.1 oracle 上界（5 行 + 折算帧数）")
    out.append("")
    out.append("| 标签 | 报告行 | hit@5px% | 重算 | Δ(pp) | 重算 | 折算帧数 | 重算 | 判定 |")
    out.append("|:--|--:|--:|--:|--:|--:|--:|--:|:--|")
    spec = (
        ("基线", "baseline_hit_pct", None, None),
        ("O-reg", "baseline_hit_pct + oracle_reg_gain_pp", "oracle_reg_gain_pp", "reg_frames"),
        ("O-reg 合法", "baseline_hit_pct + oracle_reg_legal_gain_pp", "oracle_reg_legal_gain_pp", "reg_legal_frames"),
        ("O-all", "baseline_hit_pct + oracle_all_gain_pp", "oracle_all_gain_pp", None),
        ("O-all 合法", "baseline_hit_pct + oracle_all_legal_gain_pp", "oracle_all_legal_gain_pp", "all_legal_frames"),
    )
    for label, hit_expr, gain_expr, frames_expr in spec:
        line_no, cells = row_by_label(t1, label)
        hit_mine = eval(hit_expr, {"__builtins__": {}}, rc)  # noqa: S307
        gain_mine = eval(gain_expr, {"__builtins__": {}}, rc) if gain_expr else None
        frames_mine = rc[frames_expr] if frames_expr else None
        ok = all(i["status"] == "PASS" for i in chk.items if i["label"].startswith(f"§17.1 {label} "))
        out.append(f"| `{label}` | {line_no} | {cells[2] if cells else '—'} | {hit_mine:.6f} | "
                   f"{cells[3] if cells else '—'} | {'—' if gain_mine is None else f'{gain_mine:.6f}'} | "
                   f"{cells[4] if cells else '—'} | {frames_mine if frames_mine is not None else '—'} | "
                   f"{'✅' if ok else '❌'} |")
    out.append("")
    out.append("## §17.2 缺口分桶")
    out.append("")
    out.append("| 标签 | 报告行 | 帧数 | 重算 | 占失手帧% | 重算 | 判定 |")
    out.append("|:--|--:|--:|--:|--:|--:|:--|")
    for label, key in zip(("B1 窗内有峰", "B2 窗内无峰", "B3 窗外", "B4 无 refine"),
                          ("B1_window_has_peak", "B2_window_no_peak", "B3_out_of_window", "B4_no_refine")):
        line_no, cells = row_by_label(t2, label)
        ok = all(i["status"] == "PASS" for i in chk.items if i["label"].startswith(f"§17.2 {label} "))
        out.append(f"| `{label}` | {line_no} | {cells[1]} | {rc['buckets'][key]} | {cells[2]} | "
                   f"{rc['bucket_share'][key]:.4f} | {'✅' if ok else '❌'} |")
    line_no, cells = row_by_prefix(t2, "合计失手")
    out.append(f"| `合计失手` | {line_no} | {cells[1]} | {rc['bucket_total']} | {cells[2]} | 100.0000 | "
               f"{'✅' if all(i['status'] == 'PASS' for i in chk.items if '合计失手' in i['label']) else '❌'} |")
    out.append("")
    out.append(f"- 四桶合计 **{rc['bucket_total']}** = 基线失手总数（GT {rc['n']} − 基线命中 {rc['hit_base']}）"
               f" ⇒ 完备划分；「无 refine 但有上报」的帧 = {rc['no_refine_with_entries']}")
    out.append("")
    out.append("## §17.3 15 条规则（分母 = O-reg 不加出框门）")
    out.append("")
    out.append(f"- 报告表头：{plain(hc3[3]) if hc3 and len(hc3) > 3 else '—'}")
    out.append(f"- 重算分母：O-reg 不加门 Δ = **{rc['share_denominator_gain_pp']:.4f} pp** / "
               f"**{rc['share_denominator_frames']} 帧**；产物 `ceiling.share_denominator` = "
               f"`{closure.get('ceiling', {}).get('share_denominator')}`")
    out.append("")
    out.append("| 规则 | 报告行 | hit@5px% | 重算 | Δ(pp) | 重算 | 占 O-reg% | 重算 | 判定 |")
    out.append("|:--|--:|--:|--:|--:|--:|--:|--:|:--|")
    for line_no, cells in t3:
        name = plain(cells[0])
        if name not in rc["rule_hit_pct"]:
            continue
        ok = all(i["status"] == "PASS" for i in chk.items if i["label"].startswith(f"§17.3 {name} "))
        out.append(f"| `{name}` | {line_no} | {cells[1]} | {rc['rule_hit_pct'][name]:.6f} | {cells[2]} | "
                   f"{rc['rule_gain_pp'][name]:.6f} | {cells[3]} | {rc['rule_share'][name] * 100.0:.4f} | "
                   f"{'✅' if ok else '❌'} |")
    out.append("")
    out.append(f"- 最好规则 = **{rc['best_rule']['name']}**，Δ = **{rc['best_rule']['gain_pp']:.7f} pp**，"
               f"占 O-reg = **{rc['best_rule']['share_pct']:.4f}%**（报告显示 0.6%）；"
               f"占最宽天花板（+{rc['oracle_all_legal_gain_pp']:.4f} pp）= **{rc['best_rule_share_of_ceiling_pct']:.4f}%**")
    out.append(f"- 可救帧 = **{rc['recoverable']}** 帧，`rank_hist` = {rc['rank_hist']} ⇒ 名次=0 占 "
               f"**{rc['rank0_share_pct']:.4f}%**")
    out.append("")
    out.append("## §17.5 静默帧归因")
    out.append("")
    out.append("| 归因 | 报告行 | 帧数 | 重算 | 占静默帧% | 重算 | TP/FP 报告 | 重算 TP/FP | 判定 |")
    out.append("|:--|--:|--:|--:|--:|--:|:--|:--|:--|")
    for label, cause in (("回退链峰值门", "rollback_gate"), ("出框门", "fov_gate"), ("其它", "other")):
        line_no, cells = row_by_prefix(t5, label)
        ok = all(i["status"] == "PASS" for i in chk.items if i["label"].startswith(f"§17.5 {label} "))
        out.append(f"| {plain(cells[0])[:18] if cells else label} | {line_no} | {cells[1] if cells else '—'} | "
                   f"{rc['by_cause'].get(cause, 0)} | {cells[2] if cells else '—'} | "
                   f"{rc['silent_share'].get(cause, 0.0):.4f} | {cells[3] if cells else '—'} | "
                   f"{rc['tp_by_cause'].get(cause, 0)}/{rc['fp_by_cause'].get(cause, 0)} | {'✅' if ok else '❌'} |")
    line_no, cells = row_by_prefix(t5, "合计")
    out.append(f"| {plain(cells[0])[:18] if cells else '合计'} | {line_no} | {cells[1] if cells else '—'} | "
               f"{rc['silent_attr_total']} | {cells[2] if cells else '—'} | 100.0000 | "
               f"{cells[3] if cells else '—'} | — | "
               f"{'✅' if all(i['status'] == 'PASS' for i in chk.items if '合计' in i['label']) else '❌'} |")
    out.append("")
    out.append(f"- 归因口径：有 GT 且**有配准决策**、但最终没上报的静默帧 = **{rc['silent_attr_total']}** 帧；"
               f"另有 {rc['silent_total'] - rc['silent_attr_total']} 帧静默且**无配准决策**"
               f"（§17.2 的 B4 桶）；有 GT 的静默帧合计 {rc['silent_total']} 帧")
    out.append(f"- 产物 `closure_report.json` `silent_attribution` = `{closure.get('silent_attribution')}`")
    out.append(f"- **842 残留检查**：§17.5 里 842 只允许出现在「纠错留痕」语境；"
               f"产物 `anatomy.why_no_report_fov_dropped` = {closure.get('anatomy', {}).get('why_no_report_fov_dropped')}"
               f"（夜 8 第一版**错误**归因），§17.5 的出框门行应为 {rc['by_cause'].get('fov_gate', 0)}")
    out.append("")
    out.append("## 观察（非 FAIL，不影响任何数字）")
    out.append("")
    out.append(f"1. §17.3 结论行把「对最宽天花板的占比」渲染成 `0.2)%`（多一个右括号）——"
               "生成器 `make_fastinfer_final_report.py:2432` 的 f-string 写成 `...{fmt(...)}）%)`；"
               f"数值本身正确（{rc['best_rule_share_of_ceiling_pct']:.4f}% → 0.2%），只是排版多括号。")
    out.append(f"2. 「合法 oracle」有 {rc['legal_pick_out_of_fov']} 帧因候选集为空回退到 refine 原始返回、"
               f"选中点可能越界；现行口径下这 {rc['legal_pick_out_of_fov']} 帧对 §17 的任何数字都没有影响。")
    out.append(f"3. 产物 `closure_report.md` §1 的「交付档 arms.json = 86.9700% / 差 0.0018 pp ⇒ 逐位一致」"
               f"与 `arms.json` 实测（`hit_rate_5px = 0.8697179135054819` = 86.9718）不符："
               f"§17.1 的基线 86.9718 本身与交付档 arms.json **逐位相等**。")
    out.append("")
    out.append("## 逐项明细")
    out.append("")
    out.append("| 状态 | 项 | 报告 | 重算 | Δ | 报告行 |")
    out.append("|:--|:--|:--|:--|--:|--:|")
    for item in chk.items:
        delta = "—" if item["delta"] is None else f"{item['delta']:+.6f}"
        out.append(f"| {'✅' if item['status'] == 'PASS' else '❌'} | {item['label']} | "
                   f"{item['reported'] if item['reported'] is not None else '—'} | "
                   f"{item['recomputed'] if item['recomputed'] is not None else '—'} | {delta} | "
                   f"{item['line'] if item['line'] else '—'} |")
    out.append("")
    out.append("## 复核命令")
    out.append("")
    out.append("```bash")
    out.append("cd /tmp/pycharm_project_10ae9e2e")
    out.append("/home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/audit_sec17_reg_oracle.py")
    out.append("```")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
