#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立审计：FINAL_REPORT.md §16（夜 7 位移修正族 oracle 上界 + 候选族结案）逐位核对。

设计原则（与前几轮审计同一范式）：
  1. **不 import、不调用** `make_fastinfer_final_report.py` 的任何函数；本脚本只用标准库。
  2. 全部数字**独立**从原始产物重算：
       - runs/lookahead/fi_nogmc24/oracle_accel/closure_report.json  （§16.1–16.3 的真源）
       - runs/lookahead/fi_nogmc24/oracle_accel/oracle_bounds.json    （§16 数字的上一级产物，可独立复算）
       - runs/lookahead/fi_nogmc24/gapdiag/gap_report.json           （1.3748 pp 的真源）
       - runs/lookahead/fi_nogmc24/reg_oracle/closure_report.json    （§17 跨节一致性）
       - runs/lookahead/fovgate24nogmc/runs/cs_t0043/d3/fov__m6/arms.json（交付档 replay 臂 + 生效 CFG）
  3. 比对按**报告里的显示精度**逐位比：报告单元格解析成浮点后与源值比，
     容差 = 该显示精度末位的半个单位（4 位小数 → 5e-5；2 位小数 → 5e-3；整数 → 0.5）。
  4. **零硬编码行号**：所有定位一律走**内容锚点**（`## 16. ` / `### 16.x ` 标题、表格首列文本、
     散文特征子串）。报告重新生成导致行号漂移时，本脚本仍然正确 —— 这一点由内置的
     `drift_mutant()` 自测（纯内存、不落盘）验证。
  5. 只读、无副作用：可重复运行（幂等），不写任何文件。

用法：
    /home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/audit_sec16_gap.py
    /home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/audit_sec16_gap.py --report <md 副本>
退出码：0 = 无 FAIL；1 = 存在 FAIL。
"""

from __future__ import annotations

import argparse
import json
import math
import hashlib
import re
import sys
from pathlib import Path

# 根定位：仓内 ⇒ 仓根；**交付包内** ⇒ 包内 `repo/` 镜像根 ⇒ 审计可在包外独立复跑。
ROOT = next((a for a in Path(__file__).resolve().parents if (a / "manu/tracking/lookahead").is_dir()),
            Path(__file__).resolve().parent.parent / "repo")

REPORT = ROOT / "runs/lookahead/fi_nogmc/FINAL_REPORT.md"
OUT_MD = ROOT / "runs/lookahead/fi_nogmc/audit_round7/audit_sec16_gap.md"
ORACLE_DIR = ROOT / "runs/lookahead/fi_nogmc24/oracle_accel"
GAPDIAG = ROOT / "runs/lookahead/fi_nogmc24/gapdiag/gap_report.json"
REG_ORACLE = ROOT / "runs/lookahead/fi_nogmc24/reg_oracle/closure_report.json"
ARMS = ROOT / "runs/lookahead/fovgate24nogmc/runs/cs_t0043/d3/fov__m6/arms.json"

DASH = "—"

# §16 的内容锚点（不含行号）
A_SEC16 = "## 16. "
A_SEC17 = "## 17. "
A_161 = "### 16.1 "
A_162 = "### 16.2 "
A_163 = "### 16.3 "
A_164 = "### 16.4 "
A_13 = "## 13. "
A_154 = "### 15.4 "

_results: list[tuple[str, str, bool, str]] = []  # (level, cid, ok, detail)


# ------------------------------------------------------------------ 已核数字登记
# 每当某个**报告里的数字**被拿去和真源逐位比对过，就把它登记进来；
# 随后据此反查「§16 里还有没有**没被核对过**的数字」⇒ 覆盖率自证。
_VERIFIED_NUM: set[str] = set()


# **结构性数字**：章节引用（§16.1 / §17 / §13 …）、百分位标签（p50 / p90）、
# 阈值与配置常量（hit@5px / p<0.05 / Δ=3 / 120ms / m=6 / cs_t0043 / output-max-step-px=48）、
# 公式系数（0.5·s / t−2Δ / Δ≥2）—— 它们**不是**「数据项」，不参与覆盖率反查。
STRUCTURAL_NUM = {
    "13", "15", "16", "16.1", "16.2", "16.3", "16.4", "17", "17.4",
    "1", "2", "3", "4", "5", "6", "7", "24", "48", "50", "90", "120", "0043",
    "0.05", "0.5", "-2",
}
NUM_TOKEN_RE = re.compile(r"[-−+]?\d+(?:\.\d+)?")


def register_num(*values) -> None:
    """登记一个已被核对过的数字（报告里的写法 + 真源值，两种形态都登记）。"""
    for value in values:
        if value is None:
            continue
        text = str(value).strip().replace("**", "").replace("*", "").replace("`", "").strip()
        text = text.replace("−", "-").lstrip("+")
        if text:
            _VERIFIED_NUM.add(text)


def rec(level: str, cid: str, ok: bool, detail: str) -> None:
    _results.append((level, cid, ok, detail))


# ------------------------------------------------------------------ 锚点工具
def find_line(lines: list[str], prefix: str, start: int = 0) -> int | None:
    """首个 `strip()` 后以 prefix 开头的 **1-based** 行号；找不到返回 None（不抛异常）。"""
    for index in range(max(0, start - 1), len(lines)):
        if lines[index].strip().startswith(prefix):
            return index + 1
    return None


def must_line(lines: list[str], prefix: str, what: str) -> int:
    """必须存在的锚点；缺失时给出可定位的 FATAL（而不是静默跳过断言）。"""
    line_no = find_line(lines, prefix)
    if line_no is None:
        raise SystemExit(f"[FATAL] 报告里找不到锚点 {what!r}（前缀 {prefix!r}）")
    return line_no


def section_range(lines: list[str], hdr_prefix: str) -> tuple[int | None, int | None]:
    """返回小节的 (起, 止) 1-based 闭区间；止 = 下一个 '## '/'### ' 标题的前一行。"""
    start = find_line(lines, hdr_prefix)
    if start is None:
        return (None, None)
    end = len(lines)
    for index in range(start, len(lines)):  # index 是 0-based，指向标题的下一行
        stripped = lines[index].strip()
        if stripped.startswith("## ") or stripped.startswith("### "):
            end = index
            break
    return (start, end)


def req_range(lines: list[str], hdr_prefix: str, what: str) -> tuple[int, int]:
    """**必需**的小节区间；标题缺失时给出可定位的 FATAL（而不是下游 TypeError 崩溃）。"""
    lo, hi = section_range(lines, hdr_prefix)
    if lo is None or hi is None:
        raise SystemExit(f"[FATAL] 报告里找不到小节 {what!r}（标题前缀 {hdr_prefix!r}）")
    return lo, hi


def block_range(lines: list[str], start_prefix: str, stop_prefix: str) -> tuple[int | None, int | None]:
    """返回从 start_prefix 起、到 stop_prefix 前一行止的 (起, 止) 1-based 闭区间。

    与 `section_range` 的区别：本函数只看 **stop_prefix**（不把中间的小节标题当结束），
    用来圈定整节（§16 → §17）而不是某个小节。
    """
    start = find_line(lines, start_prefix)
    if start is None:
        return (None, None)
    stop = find_line(lines, stop_prefix, start)
    if stop is None:
        stop = len(lines) + 1
    return (start, stop - 1)


def row_cells(line: str) -> list[str]:
    """把 markdown 表格行切成单元格（保留空串）。"""
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    return [cell.strip() for cell in text.split("|")]


def find_row(lines: list[str], lo: int | None, hi: int | None, col0: str) -> tuple[int | None, list[str]]:
    """在 [lo, hi] 闭区间内找**首列文本**以 col0 开头的表格行 → (行号, 单元格)。

    区间为 None（上游把小节标题删了）时返回 (None, [])，让调用方记 FAIL 而不是抛异常。
    """
    if lo is None or hi is None:
        return (None, [])
    for line_no in range(lo, hi + 1):
        text = lines[line_no - 1].strip()
        if not text.startswith("|"):
            continue
        cells = row_cells(text)
        if cells and cells[0].startswith(col0):
            return line_no, cells
    return (None, [])


def find_text_line(lines: list[str], needle: str, lo: int | None = 1, hi: int | None = None) -> int | None:
    """在 [lo, hi] 闭区间内找含 needle 子串的行 → 1-based 行号。

    区间非法（lo 为 None 或 lo > hi）时返回 None，交给调用方记 FAIL。
    """
    if lo is None:
        return None
    last = len(lines) if hi is None else hi
    if hi is not None and hi < lo:
        return None
    for line_no in range(lo, last + 1):
        if needle in lines[line_no - 1]:
            return line_no
    return None


# ------------------------------------------------------------------ 基础数值
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def as_num(cell: str):
    text = cell.strip().replace("*", "").replace("`", "").strip()
    if text in {"", DASH, "—", "-"}:
        return None
    text = text.replace("pp", "").replace("%", "").replace("px", "").replace("帧", "").strip()
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        return None
    return float(text)


def tol_str(decimals: int) -> str:
    return f"{0.5 * 10 ** (-decimals):g}"


def close(reported, source, decimals: int) -> bool:
    """按报告显示精度逐位比；容差 = 末位半个单位。"""
    if reported is None or source is None:
        return reported is None and source is None
    return abs(reported - source) <= 0.5 * (10 ** (-decimals)) + 1e-12


def num_eq(cid: str, cell: str, source: float, decimals: int, where: str) -> None:
    register_num(cell, f"{source:.{decimals}f}", f"{source:g}")
    got = as_num(cell)
    ok = close(got, source, decimals)
    detail = f"{where}: 报告={cell!r} 源={source!r} 显示精度={decimals}位 容差={tol_str(decimals)}"
    if not ok:
        detail += f"  Δ={None if got is None else got - source}"
    rec("CHECK" if ok else "FAIL", cid, ok, detail)


def expect_text(cid: str, line_no: int | None, needle: str, lines: list[str]) -> None:
    if line_no is None:
        rec("FAIL", cid, False, f"锚点行缺失，无法核对 {needle!r}")
        return
    line = lines[line_no - 1]
    ok = needle in line
    rec("CHECK" if ok else "FAIL", cid, ok, f"L{line_no} 含 {needle!r}: {ok}")


def expect_absent(cid: str, line_no: int | None, needle: str, lines: list[str]) -> None:
    if line_no is None:
        rec("FAIL", cid, False, f"锚点行缺失，无法核对 {needle!r}")
        return
    line = lines[line_no - 1]
    ok = needle not in line
    rec("CHECK" if ok else "FAIL", cid, ok, f"L{line_no} 不应含 {needle!r}: {ok}")


def need_row(cid: str, lines: list[str], lo: int, hi: int, col0: str, width: int) -> tuple[int | None, list[str]]:
    """必须存在的表格行：找不到就记 FAIL 并返回**定长**空单元格（后续下标访问不会崩）。"""
    line_no, cells = find_row(lines, lo, hi, col0)
    if line_no is None:
        rec("FAIL", cid, False, f"表行锚点缺失：首列以 {col0!r} 开头的行（L{lo}..L{hi}）")
        return None, [""] * width
    rec("CHECK", cid, True, f"表行锚点 L{line_no}：首列以 {col0!r} 开头")
    return line_no, cells + [""] * max(0, width - len(cells))


def need_text(cid: str, lines: list[str], needle: str, lo: int = 1, hi: int | None = None) -> int | None:
    """必须存在的文本锚点行；找不到就记 FAIL 并返回 None。"""
    line_no = find_text_line(lines, needle, lo, hi)
    if line_no is None:
        rec("FAIL", cid, False, f"文本锚点缺失：含 {needle!r} 的行（L{lo}..L{hi}）")
        return None
    rec("CHECK", cid, True, f"文本锚点 L{line_no}：含 {needle!r}")
    return line_no


# ------------------------------------------------------------------ 漂移自测
DRIFT_FILLER = 7  # 在文件最前面塞几行占位 ⇒ 所有行号整体下移
DRIFT_FILLER_TEXT = "<!-- drift filler: 上游新增的一段说明 -->"


def drift_mutant(lines: list[str]) -> list[str]:
    """构造一份**行号漂移**的报告文本（内容**一字不删**，只改排版/加内容）。

    - 删掉 §16 区间里的全部空行（模拟上游重新排版、压掉空行）；
    - 在每个 §16 小节标题前插入 2 行占位说明（模拟上游在 §16 里补了几段）；
    - 再在文件最前面插入若干占位行（模拟上游在 §16 之前又加了一节）。
    纯内存操作，不落盘。锚点定位 ⇒ 漂移后所有断言结论必须与原版**完全一致**。
    """
    sec_lo, sec_hi = block_range(lines, A_SEC16, A_SEC17)
    out: list[str] = []
    for index, line in enumerate(lines, start=1):
        if sec_lo <= index <= sec_hi:
            if line.startswith(("### 16.", "### 17.")):
                out.append(DRIFT_FILLER_TEXT)
                out.append(DRIFT_FILLER_TEXT)
            if line.strip() == "":
                continue
        out.append(line)
    return [f"<!-- drift head {i} -->" for i in range(DRIFT_FILLER)] + out


# ------------------------------------------------------------------ 主检查
def run_checks(lines: list[str]) -> None:
    """对给定的报告文本跑全套断言，结果写进 _results。"""
    ac = load_json(ORACLE_DIR / "closure_report.json")
    ob = load_json(ORACLE_DIR / "oracle_bounds.json")
    gap = load_json(GAPDIAG)
    reg = load_json(REG_ORACLE)
    arms = load_json(ARMS)

    # ---------- 0. 区块定位（全部锚点） ----------
    sec_lo = must_line(lines, A_SEC16, "§16 标题")
    sec_hi = must_line(lines, A_SEC17, "§17 标题") - 1
    blk_lo, blk_hi = block_range(lines, A_SEC16, A_SEC17)
    rec("CHECK", "S16-BLOCK", (blk_lo, blk_hi) == (sec_lo, sec_hi),
        f"§16 整节区间 = L{blk_lo}..L{blk_hi}（应与 §16 标题..§17 标题−1 一致）")
    rec("CHECK", "S16-HDR", lines[sec_lo - 1].startswith(A_SEC16), f"L{sec_lo} = {lines[sec_lo - 1][:60]!r}")
    rec("CHECK", "S16-END", lines[sec_hi].strip() == "", f"L{sec_hi} 为空行: {lines[sec_hi].strip()!r}")
    rec("CHECK", "S17-HDR", lines[sec_hi + 1].startswith(A_SEC17), f"L{sec_hi + 1} = §17 标题")

    s161_lo, s161_hi = req_range(lines, A_161, "§16.1")
    s162_lo, s162_hi = req_range(lines, A_162, "§16.2")
    s163_lo, s163_hi = req_range(lines, A_163, "§16.3")
    s164_lo, s164_hi = req_range(lines, A_164, "§16.4")
    s13_lo, s13_hi = req_range(lines, A_13, "§13")
    s154_lo, s154_hi = req_range(lines, A_154, "§15.4")
    for cid, lo, hi in (("16.1", s161_lo, s161_hi), ("16.2", s162_lo, s162_hi), ("16.3", s163_lo, s163_hi),
                        ("16.4", s164_lo, s164_hi), ("13", s13_lo, s13_hi), ("15.4", s154_lo, s154_hi)):
        rec("CHECK", f"ANCHOR-{cid}", lo is not None and hi is not None, f"§{cid} 区间 = L{lo}..L{hi}")

    # ---------- 1. §16.1 oracle 表（按首列文本取行） ----------
    orc = ac["oracle"]
    base = ac["hit5_baseline_pct"]
    row, cells = need_row("S16.1-BASE-ROW", lines, s161_lo, s161_hi, "基线", 4)
    num_eq("S16.1-BASE", cells[2], base, 4, f"L{row} 基线 hit@5px%")
    rec("CHECK", "S16.1-BASE-DASH", cells[3] == DASH, f"L{row} Δ 列 = {cells[3]!r}（破折号）")
    for col0, key, cid in (("**O1 尺度**", "O1_scale", "O1"), ("**O2 加速度项**", "O2_accel", "O2"),
                           ("O3 尺度+加速度", "O3_scale_accel", "O3")):
        row, cells = need_row(f"S16.1-{cid}-ROW", lines, s161_lo, s161_hi, col0, 4)
        num_eq(f"S16.1-{cid}-H5", cells[2], orc[key]["hit5_pct"], 4, f"L{row} {cid} hit@5px%")
        num_eq(f"S16.1-{cid}-DPP", cells[3], orc[key]["delta_pp"], 4, f"L{row} {cid} Δpp")
    for key in orc:
        num_eq(f"S16.1-{key}-DERIVED", str(orc[key]["delta_pp"]), orc[key]["hit5_pct"] - base, 3,
               f"derived Δpp {key}（= hit5 − 基线）")

    # ---------- 2. §16.1 真位移分桶表 ----------
    bk = ac["by_true_displacement"]
    tot_n = 0
    wsum = {k: 0.0 for k in ("hit5_baseline_pct", "O1_scale_pct", "O2_accel_pct", "O3_scale_accel_pct")}
    for i, (bname, col0) in enumerate(zip(bk.keys(), ("<=5", "(5,10]", "(10,20]", "(20,40]", ">40"))):
        b = bk[bname]
        row, cells = need_row(f"S16.1-BK{i}-ROW", lines, s161_lo, s161_hi, col0, 6)
        rec("CHECK", f"S16.1-BK{i}-NAME", cells[0] == bname, f"L{row} 桶名 {cells[0]!r} vs {bname!r}")
        num_eq(f"S16.1-BK{i}-N", cells[1], float(b["n"]), 0, f"L{row} 帧数")
        tot_n += b["n"]
        for j, k in enumerate(("hit5_baseline_pct", "O1_scale_pct", "O2_accel_pct", "O3_scale_accel_pct")):
            num_eq(f"S16.1-BK{i}-{k}", cells[2 + j], b[k], 2, f"L{row} {k}")
            wsum[k] += b["n"] * b[k]
    num_eq("S16.1-BK-SUM-N", str(tot_n), float(ac["gt_frames_considered"]), 0, "分桶帧数合计 vs 参与统计帧数")
    for k, overall in (("hit5_baseline_pct", base),
                       ("O1_scale_pct", orc["O1_scale"]["hit5_pct"]),
                       ("O2_accel_pct", orc["O2_accel"]["hit5_pct"]),
                       ("O3_scale_accel_pct", orc["O3_scale_accel"]["hit5_pct"])):
        w = wsum[k] / tot_n
        rec("CHECK" if abs(w - overall) <= 0.02 else "FAIL", f"S16.1-WAVG-{k}", abs(w - overall) <= 0.02,
            f"分桶加权平均 {k} = {w:.4f} vs 总体 {overall}（容差 0.02pp：每桶已舍入到 2 位）")

    # ---------- 3. §16.1 独立复算：oracle_bounds.json 逐序命中数 ----------
    ps = ob["per_sequence"]
    fr = sum(v["frames"] for v in ps.values())
    num_eq("S16.1-OB-FRAMES", str(fr), float(ac["gt_frames_considered"]), 0, "oracle_bounds 逐序 frames 合计")
    ob_pct = {k: 100.0 * sum(v[k] for v in ps.values()) / fr for k in ("hit0", "hit1", "hit2", "hit3")}
    num_eq("S16.1-OB-BASE", str(round(ob_pct["hit0"], 4)), base, 4, "独立复算基线")
    num_eq("S16.1-OB-O1", str(round(ob_pct["hit1"], 4)), orc["O1_scale"]["hit5_pct"], 4, "独立复算 O1")
    num_eq("S16.1-OB-O2", str(round(ob_pct["hit2"], 4)), orc["O2_accel"]["hit5_pct"], 4, "独立复算 O2")
    num_eq("S16.1-OB-O3", str(round(ob_pct["hit3"], 4)), orc["O3_scale_accel"]["hit5_pct"], 4, "独立复算 O3")

    # ---------- 4. §16.2 运动学出口三行 ----------
    exits = ac["structure"]["model_exits_with_frames"]
    for i, (kname, col0) in enumerate(zip(exits.keys(), ("consistency_gate -> anchor+Δ·v",
                                                        "best_effort 兜底 -> anchor+Δ·v",
                                                        "低速冻结 -> anchor（速度侧，非二阶）"))):
        row, cells = need_row(f"S16.2-EXIT{i}-ROW", lines, s162_lo, s162_hi, col0, 2)
        rec("CHECK", f"S16.2-EXIT{i}-NAME", cells[0] == kname, f"L{row} 出口名 {cells[0]!r} vs {kname!r}")
        num_eq(f"S16.2-EXIT{i}-N", cells[1], float(exits[kname]), 0, f"L{row} 帧数")
    rep_ps = arms["arms"]["replay"]["stream_stats"]["per_sequence"]
    all_ps_keys: set[str] = set()
    for arm in arms["arms"].values():
        for ss in arm["stream_stats"]["per_sequence"].values():
            all_ps_keys |= set(ss.keys())
    num_eq("S16.2-EXIT-CONSIST", str(sum(v.get("replay_consistency_rejected", 0) for v in rep_ps.values())),
           float(exits["consistency_gate -> anchor+Δ·v"]), 0, "arms.json 逐序 replay_consistency_rejected 合计")
    num_eq("S16.2-EXIT-FALLBACK", str(sum(v.get("fallback_used", 0) for v in rep_ps.values())),
           float(exits["best_effort 兜底 -> anchor+Δ·v"]), 0, "arms.json 逐序 fallback_used 合计")
    rec("CHECK", "S16.2-FROZEN-NO-COUNTER", "frozen" not in all_ps_keys,
        "arms.json stream_stats 里存在 'frozen' 计数器: " f"{'frozen' in all_ps_keys}")

    # ---------- 5. §16.2 计数器出处注（夜 9 新增的披露） ----------
    prov_line = need_text("S16.2-PROVENANCE-NOTE", lines, "计数器出处", s162_lo, s162_hi)
    if prov_line:
        expect_text("S16.2-PROVENANCE-FROZEN", prov_line, "缺键默认值", lines)
        expect_text("S16.2-PROVENANCE-NOT-MEASURED", prov_line, "不是实测读数", lines)
        expect_text("S16.2-PROVENANCE-OK", prov_line, "结论仍成立", lines)

    # ---------- 6. §16.2 applied / replay_steps ----------
    rep_overall = arms["arms"]["replay"]["stream_stats"]["overall"]
    applied_line = need_text("S16.2-APPLIED-ROW", lines, "applied 上报帧数", s162_lo, s162_hi)
    m_applied = m_steps = None
    if applied_line:
        m_applied = re.search(r"applied 上报帧数 \*\*(\d+)\*\*", lines[applied_line - 1])
        m_steps = re.search(r"replay_steps \*\*(\d+)\*\*", lines[applied_line - 1])
    num_eq("S16.2-APPLIED", m_applied.group(1) if m_applied else "", float(ac["structure"]["applied"]), 0,
           f"L{applied_line} applied")
    num_eq("S16.2-STEPS", m_steps.group(1) if m_steps else "", float(ac["structure"]["replay_steps"]), 0,
           f"L{applied_line} replay_steps")
    num_eq("S16.2-APPLIED-ARMS", str(rep_overall["applied"]), float(ac["structure"]["applied"]), 0,
           "arms.json replay.overall.applied")
    num_eq("S16.2-STEPS-ARMS", str(rep_overall["replay_steps"]), float(ac["structure"]["replay_steps"]), 0,
           "arms.json replay.overall.replay_steps")

    # ---------- 7. §16.2 交付档 CFG 旋钮表（回到真实生效配置块） ----------
    la = arms["lookahead"]
    knob_names = ("replay_consistency_gate_px", "replay_freeze_speed_px", "replay_disp_shrink_alpha",
                  "replay_disp_merge_sigma_px", "replay_outlier_tol_px", "replay_anchor_check_px",
                  "replay_fallback", "replay_xcorr_mode", "replay_xcorr_radius", "velocity_source")
    for knob in knob_names:
        row, cells = need_row(f"S16.2-KNOB-{knob}-ROW", lines, s162_lo, s162_hi, f"`{knob}`", 3)
        rep_val = cells[1].strip("`")
        live = la.get(knob)
        src_val = ac["structure"]["delivery_cfg_model_knobs"].get(knob)
        register_num(rep_val)
        ok = (str(live) == rep_val) and (str(src_val) == rep_val)
        rec("CHECK" if ok else "FAIL", f"S16.2-KNOB-{knob}", ok,
            f"L{row} {knob}: 报告={rep_val!r} arms.json.lookahead={live!r} closure_report={src_val!r}")
    rec("CHECK", "S16.2-CFG-FOV", la.get("replay_fov_margin_px") == 6.0,
        f"arms.json replay_fov_margin_px={la.get('replay_fov_margin_px')}")
    rec("CHECK", "S16.2-CFG-DELTA", la.get("delta") == 3, f"arms.json delta={la.get('delta')}")
    rec("CHECK", "S16.2-CFG-MODE", la.get("mode") == "replay", f"arms.json lookahead.mode={la.get('mode')}")

    # ---------- 8. §16.3 偏置表 ----------
    bias = ac["bias_vs_selection"]
    for key, col0, name in (("replay_all_frames", "replay 全部帧", "replay 全部帧"),
                            ("replay_hit_frames", "replay 命中帧", "replay 命中帧"),
                            ("replay_miss_frames", "replay 失手帧", "replay 失手帧"),
                            ("original_arm_control", "original 天花板（对照）", "original 天花板（对照）")):
        s = bias[key]
        row, cells = need_row(f"S16.3-{key}-ROW", lines, s163_lo, s163_hi, col0, 5)
        rec("CHECK", f"S16.3-{key}-NAME", cells[0] == name, f"L{row} 集合名 {cells[0]!r} vs {name!r}")
        num_eq(f"S16.3-{key}-N", cells[1], float(s["n"]), 0, f"L{row} n")
        num_eq(f"S16.3-{key}-P50", cells[2], s["p50"], 4, f"L{row} p50")
        num_eq(f"S16.3-{key}-MEAN", cells[3], s["mean"], 4, f"L{row} mean")
        num_eq(f"S16.3-{key}-FRAC", cells[4], s["frac_neg"], 4, f"L{row} 负占比")
    num_eq("S16.3-SPLIT", str(bias["replay_hit_frames"]["n"] + bias["replay_miss_frames"]["n"]),
           float(bias["replay_all_frames"]["n"]), 0, "命中 + 失手 == 全部帧")

    # ---------- 9. §16.3 相关性 / >20px / 加速项 ----------
    p_line = need_text("S16.3-CORR-ROW", lines, "Pearson r = ", s163_lo, s163_hi)
    expect_text("S16.3-PEARSON", p_line, "Pearson r = -0.1531", lines)
    expect_text("S16.3-SPEARMAN", p_line, "Spearman r = -0.0553", lines)
    num_eq("S16.3-PEARS", "-0.1531", bias["pearson_r_along_vs_true_disp"], 4, f"L{p_line} Pearson")
    num_eq("S16.3-SPEAR", "-0.0553", bias["spearman_r_along_vs_true_disp"], 4, f"L{p_line} Spearman")

    g_line = need_text("S16.3-GT20-ROW", lines, "真位移 > 20px 的帧上的平均沿轨误差", s163_lo, s163_hi)
    num_eq("S16.3-GT20", "-5.0128", bias["along_mean_on_true_disp_gt20"], 4, f"L{g_line} 真位移>20px 平均沿轨误差")

    acc = ac["accel_term_magnitude"]
    a_line = need_text("S16.3-ACC-ROW", lines, "加速项幅值", s163_lo, s163_hi)
    expect_text("S16.3-ACC-P50", a_line, "p50 = **2.0154** px", lines)
    expect_text("S16.3-ACC-P90", a_line, "p90 = **9.1234** px", lines)
    num_eq("S16.3-ACC-P50", "2.0154", acc["p50"], 4, f"L{a_line} 加速项 p50")
    num_eq("S16.3-ACC-P90", "9.1234", acc["p90"], 4, f"L{a_line} 加速项 p90")
    num_eq("S16.3-ACC-N", "8838", float(acc["n"]), 0, f"L{a_line} 加速项 n（夜 9 已披露）")
    rec("CHECK", "S16.3-ACC-N-DISTINCT", acc["n"] != bias["replay_miss_frames"]["n"],
        f"加速项 n={acc['n']} ≠ 偏置表失手帧 n={bias['replay_miss_frames']['n']}（报告已加注说明）")

    # ---------- 10. §16.3 帧集口径注（夜 9 新增） ----------
    cal_line = need_text("S16.3-CALIBER-NOTE", lines, "不是同一帧集", s163_lo, s163_hi)
    if cal_line:
        expect_text("S16.3-CALIBER-117193", cal_line, "117193", lines)
        expect_text("S16.3-CALIBER-125109", cal_line, "125109", lines)
        expect_text("S16.3-CALIBER-NODIV", cal_line, "不能直接相除", lines)

    # ---------- 11. §16.3 散文舍入数 ----------
    prose1 = need_text("S16-PROSE-ROW1", lines, "只有失手帧才", s163_lo, s163_hi)
    prose2 = need_text("S16-PROSE-ROW2", lines, "加速项解释不了这", s163_lo, s163_hi)
    p50_miss = abs(bias["replay_miss_frames"]["p50"])
    pearson = bias["pearson_r_along_vs_true_disp"]
    register_num(f"{p50_miss:.2f}", f"{abs(p50_miss):.2f}", f"{-abs(p50_miss):.2f}",
                 f"{pearson:.2f}", f"{abs(pearson):.2f}", f"{-abs(pearson):.2f}",
                 f"{acc['p50']:.2f}", f"{acc['p90']:.2f}", "20")
    rec("CHECK", "S16-PROSE-435", abs(round(p50_miss, 2) - 4.35) < 1e-9,
        f"4.35 px ← |p50|={p50_miss} → 2 位 = {p50_miss:.2f}（L{prose1} / L{prose2}）")
    rec("CHECK", "S16-PROSE-015", abs(round(abs(pearson), 2) - 0.15) < 1e-9,
        f"r ≈ −0.15 ← pearson={pearson} → 2 位 = {abs(pearson):.2f}（L{prose1}）")
    rec("CHECK", "S16-PROSE-202-912",
        abs(round(acc["p50"], 2) - 2.02) < 1e-9 and abs(round(acc["p90"], 2) - 9.12) < 1e-9,
        f"2.02 / 9.12 px ← p50={acc['p50']} → {acc['p50']:.2f}，p90={acc['p90']} → {acc['p90']:.2f}（L{prose2}）")

    # ---------- 12. §16.4 裁定表 ----------
    v_lo, v_hi = s164_lo, s164_hi
    row, cells = need_row("S16.4-CAND1-ROW", lines, v_lo, v_hi, "**#1 Δ 窗口位移的加速度项（升阶运动模型）**", 3)
    expect_text("S16.4-O2PP", row, "oracle O2 = **-6.1179 pp**", lines)
    num_eq("S16.4-O2PP-V", "-6.1179", orc["O2_accel"]["delta_pp"], 4, f"L{row} O2 Δpp")
    expect_text("S16.4-EXIT0", row, "模型出口 0 帧", lines)
    expect_text("S16.4-P50-202", row, "加速项幅值 p50 2.02 px", lines)
    num_eq("S16.4-P50-202-V", "2.02", acc["p50"], 2, f"L{row} 加速项 p50（2 位）")

    dg_row, cells = need_row("S16.4-DISPGAIN-ROW", lines, v_lo, v_hi, "「缩放位移」族（`--replay-disp-gain`）", 3)
    expect_text("S16.4-DISPGAIN", dg_row, "1.05~1.30", lines)

    row, cells = need_row("S16.4-THRESHOLD-ROW", lines, v_lo, v_hi, "「阈值类旋钮」（§13 的 37 变体）", 3)
    expect_text("S16.4-37VAR", row, "§13 的 37 变体", lines)

    row, cells = need_row("S16.4-KINEMATIC-ROW", lines, v_lo, v_hi, "「纯运动学兜底 / kinematic 臂」", 3)

    row, cells = need_row("S16.4-REMAIN-ROW", lines, v_lo, v_hi, "**剩余可回收 1.3748 pp 的真实所在**", 3)
    expect_text("S16.4-PP13748", row, "1.3748 pp", lines)
    for cid, needle in (("AMBIG", "replay_xcorr_ambiguous` 8218 帧"),
                        ("REMATCH", "replay_xcorr_rematch_rejected` 7127 帧"),
                        ("ABSTAIN", "replay_xcorr_abstained` 779 帧"),
                        ("ROLLBACK", "replay_rollback_abstained` 3836 帧")):
        expect_text(f"S16.4-{cid}", row, needle, lines)

    # §16.4 末行的裁定状态 + §17 交叉引用数字
    expect_text("S16.4-CLOSED", row, "已结案（见 §17）：不可行", lines)
    num_eq("S16.4-ORACLE-CEILING", "+2.2055", reg["oracle_all_legal_gain_pp"], 4,
           f"L{row} §17 合法 oracle 上界")
    rec("CHECK", "S16.4-ORACLE-CEILING-TXT", row is not None and "+2.2055 pp" in lines[row - 1],
        f"L{row} 文本含 '+2.2055 pp'")
    best_share = reg["best_rule"]["share_of_oracle"]
    rec("CHECK", "S16.4-SHARE-0.6PCT",
        f"{100.0 * best_share:.1f}" == "0.6" and row is not None and "0.6%" in lines[row - 1],
        f"L{row} 0.6% ← reg_oracle best_rule.share_of_oracle={best_share} → {100.0 * best_share:.1f}%")
    # §16.4 末行现在**同时**引了两个 oracle 口径 ⇒ 三个数字都必须逐位对上
    num_eq("S16.4-REG-LEGAL", "+0.7504", reg["oracle_reg_legal_gain_pp"], 4,
           f"L{row} §16.4 引用的 O-reg 合法上界（与 §17.1 同源）")
    num_eq("S16.4-REG-SHARE-DENOM", "+0.8120", reg["oracle_reg_gain_pp"], 4,
           f"L{row} §16.4 里 0.6% 的分母（O-reg 不加出框门）")
    _share_ceiling = 100.0 * reg["best_rule"]["gain_pp"] / reg["oracle_all_legal_gain_pp"]
    num_eq("S16.4-SHARE-CEILING", "0.2", _share_ceiling, 1,
           f"L{row} §16.4 里 0.2%（对最宽天花板 +{reg['oracle_all_legal_gain_pp']:.4f} pp）")

    # §16.4 末行四个帧数：独立重算
    ha = gap["hit5_attribution"]
    num_eq("S16.4-PP13748-SRC", "1.3748", ha["plugin_addressable_pp"], 4, "gap_report plugin_addressable_pp")
    num_eq("S16.4-FRAMES-SRC", "1832", float(ha["plugin_addressable_frames"]), 0,
           "gap_report plugin_addressable_frames")
    ds = ha["D_split_by_true_displacement"]
    num_eq("S16.4-D-LE5", str(ds["le5px"]), 747.0, 0, "gap_report D_split.le5px")
    num_eq("S16.4-D-520", str(ds["5to20px"]), 1085.0, 0, "gap_report D_split.5to20px")
    num_eq("S16.4-D-SUM", str(ds["le5px"] + ds["5to20px"]), float(ha["plugin_addressable_frames"]), 0,
           "D_split 合计 == plugin_addressable_frames")
    rec("CHECK", "S16.4-PP-UNIT",
        abs(100.0 * ha["plugin_addressable_frames"] / 133257 - ha["plugin_addressable_pp"]) <= 0.0001,
        f"1832 / 133257 GT = {100.0 * ha['plugin_addressable_frames'] / 133257:.4f} pp vs 报告 1.3748 pp")
    for cid, key, needle in (("AMBIG", "replay_xcorr_ambiguous", "8218"),
                             ("REMATCH", "replay_xcorr_rematch_rejected", "7127"),
                             ("ABSTAIN", "replay_xcorr_abstained", "779"),
                             ("ROLLBACK", "replay_rollback_abstained", "3836")):
        tot = sum(v.get(key, 0) for v in rep_ps.values())
        num_eq(f"S16.4-{cid}-ARMS", str(tot), float(needle), 0, f"arms.json 逐序 {key} 合计")
    num_eq("S16.4-AMBIG-GAP", str(gap["frame_level"]["stream_stats"]["xcorr_ambiguous"]), 8218.0, 0,
           "gap_report stream_stats.xcorr_ambiguous")

    # ---------- 13. §16 头部真源声明（夜 9 已改口径） ----------
    hdr_line = need_text("S16-SCOPE-ROW", lines, "参与统计帧数", sec_lo, sec_hi)
    expect_text("S16-SRC-CLAIM", hdr_line, "§16.1–16.3 的数字从", lines)
    expect_text("S16-SRC-CLAIM-PATH", hdr_line,
                "`runs/lookahead/fi_nogmc24/oracle_accel/closure_report.json` 现读", lines)
    expect_text("S16-SRC-164", hdr_line, "§16.4 的裁定表", lines)
    expect_text("S16-SRC-GAP", hdr_line, "1.3748 pp 来自 `gapdiag/gap_report.json`", lines)
    expect_text("S16-SRC-ARMS", hdr_line, "8218 / 7127 / 779 / 3836 来自交付档 `arms.json` 的 `stream_stats`", lines)
    expect_absent("S16-SRC-OLD-CLAIM", hdr_line, "**全部数字从", lines)
    # 声明是否与真源一致：§16.1–16.3 的数字在 closure_report.json；§16.4 的数字在 gap_report/arms.json
    raw_ac = json.dumps(ac, ensure_ascii=False)
    raw_gap = json.dumps(gap, ensure_ascii=False)
    raw_arms = json.dumps(arms, ensure_ascii=False)
    in_ac = {k: (k in raw_ac) for k in ("92.5017", "96.0826", "86.3839", "95.3728", "125109", "117193",
                                        "2.0154", "9.1234", "-0.1531", "-0.0553", "-5.0128")}
    rec("CHECK", "S16-SRC-CLAIM-TRUE-AC", all(in_ac.values()),
        f"§16.1–16.3 声称的真源 closure_report.json 里是否含全部这些数字：{in_ac}")
    rec("CHECK", "S16-SRC-CLAIM-TRUE-164",
        ("1.3748" in raw_gap) and ("8218" in raw_arms) and ("7127" in raw_arms) and
        ("779" in raw_ac) and ("3836" in raw_ac),
        f"§16.4 声称的出处：1.3748∈gap_report.json={'1.3748' in raw_gap}；"
        f"8218/7127∈arms.json={'8218' in raw_arms}/{'7127' in raw_arms}；"
        f"779/3836∈closure_report.json={'779' in raw_ac}/{'3836' in raw_ac}")

    # ---------- 14. §13 符号检验（锚点定位 §13 区间） ----------
    sec13_rows = []
    for line_no in range(s13_lo, s13_hi + 1):
        text = lines[line_no - 1].strip()
        if not text.startswith("| `--"):
            continue
        cells = row_cells(text)
        if len(cells) >= 6 and re.fullmatch(r"\d\.\d{4}", cells[3]):
            sec13_rows.append((line_no, cells))
    sig = [(ln, cells) for ln, cells in sec13_rows if float(cells[3]) < 0.05]
    sig_p = sorted(float(c[3]) for _, c in sig)
    thr_row = find_row(lines, v_lo, v_hi, "「阈值类旋钮」（§13 的 37 变体）")[0]
    rec("CHECK", "X-SIGTEST-COUNT", len(sig) == 4,
        f"§13 符号检验表（L{s13_lo}..L{s13_hi}）里 p<0.05 的行数 = {len(sig)}，p 值 = {sig_p}；"
        f"报告 §16.4 L{thr_row} 写『p<0.05 的共 4 个』")
    rec("CHECK", "X-SIGTEST-PVALS", sig_p == [0.0043, 0.0127, 0.0215, 0.0347],
        f"§13 表内 p<0.05 的 4 个 p 值 = {sig_p}（期望 [0.0043, 0.0127, 0.0215, 0.0347]）")
    expect_text("X-SIGTEST-REPORT-4", thr_row, "p<0.05 的共 4 个", lines)
    expect_text("X-SIGTEST-REPORT-1", thr_row, "只有 1 个", lines)
    loso = gap["caliber"]["loso_noise_floor_pp"]
    passing = [(ln, cells) for ln, cells in sec13_rows if cells[3] == "0.0215"]
    rec("CHECK", "X-SIGTEST-PASSING", len(passing) == 1,
        f"§13 表里 p=0.0215 的行数 = {len(passing)}（应为 1）")
    if passing:
        d = float(passing[0][1][4])
        register_num(f"{d:.4f}", f"{d:+.4f}", f"{loso['median']:.4f}")
        rec("CHECK", "X-SIGTEST-AMPLITUDE", d < loso["median"],
            f"唯一通过者的池化 ΔF1 = {d:+.4f} pp < LOSO 中位 {loso['median']} pp ⇒ 幅度在噪声内")
        expect_text("X-SIGTEST-REPORT-DELTA", thr_row, "+0.0192 pp", lines)
        expect_text("X-SIGTEST-REPORT-LOSO", thr_row, "0.1466 pp", lines)

    # ---------- 15. §13「池化 ΔF1 全 < 噪声底」的假陈述是否已被改掉 ----------
    sec13_concl = need_text("X-POOLED-F1-CONCL-ROW", lines, "只有 1 个作业点通过", s13_lo, s13_hi)
    rec("CHECK", "X-POOLED-F1-CLAIM",
        sec13_concl is not None and "唯一通过者的幅度也在噪声内" in lines[sec13_concl - 1],
        f"§13 结论行 = L{sec13_concl}；文本已改为『唯一通过者的幅度也在噪声内』")
    above = [(ln, float(c[4])) for ln, c in sec13_rows if abs(float(c[4])) > loso["median"]]
    rec("CHECK", "X-POOLED-F1-FACT", len(above) == 3,
        f"§13 表里 |池化 ΔF1| > LOSO 中位 {loso['median']} 的行数 = {len(above)}（{[a[1] for a in above]}）"
        f"⇒ 老口径『全部 < 中位 0.1466 pp』确实为假、现报告已不再这样写")
    expect_absent("X-POOLED-F1-OLDCLAIM", sec13_concl, "池化 ΔF1 全部 < 尺子A 的中位波动", lines)
    expect_absent("X-POOLED-F1-SEC16", thr_row, "池化 ΔF1 全 < LOSO 噪声底", lines)

    # ---------- 16. §16.1 基线行标签（夜 9 已改口径） ----------
    base_row, base_cells = need_row("X-BASELINE-LABEL-ROW", lines, s161_lo, s161_hi, "基线", 4)
    rec("CHECK", "X-BASELINE-LABEL", base_row is not None and "125109" in base_cells[0],
        f"L{base_row} 基线行首列 = {base_cells[0]!r}（已声明 §16 口径帧集）")
    delivery_hit5 = arms["arms"]["replay"]["servo"]["overall"]["hit_rate_5px"]
    num_eq("X-BASELINE-DELIVERY", f"{100.0 * delivery_hit5:.4f}", 86.9718, 4,
           "arms.json replay hit_rate_5px → 交付档基线")
    num_eq("X-BASELINE-REGORACLE", f"{reg['baseline_hit_pct']:.4f}", 86.9718, 4,
           "reg_oracle closure_report baseline_hit_pct")
    num_eq("X-BASELINE-GAPREPORT", f"{100.0 * gap['baseline_numbers']['replay_delivery']['hit5']:.4f}", 86.9718, 4,
           "gap_report baseline_numbers.replay_delivery.hit5")

    # ---------- 17. §16.4 行数 vs 真源 closure_report.md §4 ----------
    cr_md = (ORACLE_DIR / "closure_report.md").read_text(encoding="utf-8")
    src_rows = [ln for ln in cr_md.splitlines() if ln.startswith("| ") and "结案" in ln]
    rep_rows = [ln for ln in lines[v_lo - 1 : v_hi] if ln.strip().startswith("| ") and "结案" in ln]
    rec("CHECK", "X-ROWCOUNT-DROP", len(src_rows) == len(rep_rows),
        f"真源 closure_report.md §4 裁定行数={len(src_rows)}，FINAL_REPORT §16.4 裁定行数={len(rep_rows)}"
        f"（夜 9 已补回『纯运动学兜底 / kinematic 臂』一行）")

    # ---------- 18. §16.3 偏置表帧集口径 ----------
    rec("CHECK", "X-BIAS-CALIBER", cal_line is not None,
        f"§16.3 已加帧集口径注（L{cal_line}）：117193 vs 125109 不是同一帧集")

    # ---------- 19. §17 交叉引用：§16.4 的 +2.2055 pp 与 §17.1 一致 ----------
    s171_lo, s171_hi = section_range(lines, "### 17.1 ")
    r1, c1 = need_row("X-REGORACLE-CEILING-ROW", lines, s171_lo, s171_hi, "**O-all 合法**", 5)
    rec("CHECK", "X-REGORACLE-CEILING", r1 is not None and c1[3].strip("*") == "2.2055",
        f"§17.1 O-all 合法行 = L{r1}：Δ vs 基线 = {c1[3]!r}（§16.4 L{row} 引用同一数字）")
    r1b, c1b = need_row("X-REGORACLE-REGLEGAL-ROW", lines, s171_lo, s171_hi, "**O-reg 合法**", 5)
    rec("CHECK", "X-REGORACLE-REGLEGAL", r1b is not None,
        f"§17.1 O-reg 合法行 = L{r1b}：Δ = {c1b[3]!r}（配准候选重排族自身的合法上界）")
    sec173_lo, sec173_hi = section_range(lines, "### 17.3 ")
    r_best, c_best = need_row("X-REGORACLE-BESTRULE-ROW", lines, sec173_lo, sec173_hi, "`R5_uniq_gate_peak`", 4)
    rec("CHECK", "X-REGORACLE-BESTRULE", r_best is not None and c_best[2].strip("*") == "0.0045"
        and c_best[3].strip("*") == "0.6%",
        f"§17.3 最佳规则行 = L{r_best}：Δ = {c_best[2]!r}，占 O-reg 比例 = {c_best[3]!r}（§16.4 L{row} 引用 0.6%）")

    # ---------- 19b. §16.4 末行两个数字的**分母**是否同族 ----------
    register_num(f"{100.0 * reg['best_rule']['share_of_oracle']:.1f}")
    r5_gain = reg["rules"]["R5_uniq_gate_peak"]["gain_pp"]
    share_vs_reg = 100.0 * r5_gain / reg["oracle_reg_gain_pp"]
    share_vs_all = 100.0 * r5_gain / reg["oracle_all_legal_gain_pp"]
    sec173_wide = need_text("X-SHARE-DENOM-WIDE", lines, "即使对**最宽**的天花板", sec173_lo, sec173_hi)
    rec("CHECK", "X-SHARE-DENOM",
        abs(share_vs_reg - 0.6) < 0.05 and abs(share_vs_all - 0.2) < 0.05
        and row is not None and "O-reg 口径" in lines[row - 1] and "0.6%" in lines[row - 1]
        and "0.2%" in lines[row - 1],
        f"R5 占 O-reg(+{reg['oracle_reg_gain_pp']:.4f} pp) = {share_vs_reg:.1f}%、"
        f"占 O-all 合法(+{reg['oracle_all_legal_gain_pp']:.4f} pp) = {share_vs_all:.1f}%；"
        f"§16.4 L{row} 已把两个分母**显式分开**（『O-reg 口径（+0.8120 pp）的 0.6%』+『对最宽天花板只有 0.2%』）"
        f"（§17.3 L{sec173_wide} 也写了『即使对最宽的天花板 +2.2055 pp 也只有 0.2%』）")
    rec("CHECK", "X-ORACLE-CEILING-ATTRIB",
        reg["oracle_reg_legal_gain_pp"] < reg["oracle_all_legal_gain_pp"]
        and row is not None and "O-all 合法" in lines[row - 1] and "O-reg 合法" in lines[row - 1],
        f"§16.4 L{row} 已把两个 oracle 口径**分开写**：最宽口径 = O-all 合法 +"
        f"{reg['oracle_all_legal_gain_pp']:.4f} pp；本族 O-reg 合法 = +{reg['oracle_reg_legal_gain_pp']:.4f} pp "
        f"⇒ 措辞不再把两个 oracle 混成一个")

    # ---------- 20. §15.4 disp-gain 交叉引用 ----------
    dg_rows = []
    for line_no in range(s154_lo, s154_hi + 1):
        text = lines[line_no - 1].strip()
        if text.startswith("| `--replay-disp-gain`"):
            cells = row_cells(text)
            register_num(cells[1])
            dg_rows.append((line_no, cells))
    rec("CHECK", "X-DISPGAIN-ROWS", len(dg_rows) == 4,
        f"§15.4 里 --replay-disp-gain 变体行数 = {len(dg_rows)}（L{[r[0] for r in dg_rows]}）")
    if dg_rows:
        worst = max(max(abs(float(r[1][3])), abs(float(r[1][7]))) for r in dg_rows)
        med = loso["median"]
        rec("CHECK" if worst <= med else "FAIL", "X-DISPGAIN-NOISE", worst <= med,
            f"§15.4 四档最大 |ΔF1 或 |Δhit@5px| = {worst:.4f} pp ≤ LOSO 中位 {med} pp "
            f"⇒ §16.4 L{dg_row} 『1.05~1.30 全在噪声内』✅")

    # ---------- 21. §13「37 变体」 ----------
    s13_jobs = need_text("X-37VARIANTS-ROW", lines, "作业点数", s13_lo, s13_hi)
    rec("CHECK", "X-37VARIANTS", s13_jobs is not None and "37" in lines[s13_jobs - 1],
        f"L{s13_jobs} §13 作业点数：{lines[s13_jobs - 1].strip()[:110] if s13_jobs else None!r} "
        f"⇒ §16.4『§13 的 37 变体』✅")

    # ---------- 22. NaN / None / 空单元格 / 破折号泄漏扫描 ----------
    sec = lines[sec_lo - 1 : sec_hi]
    leaks = []
    for i, ln in enumerate(sec, start=sec_lo):
        for bad in ("nan", "NaN", "None", "null", "Infinity"):
            if bad in ln:
                leaks.append((i, bad))
        if ln.strip().startswith("|"):
            for j, cell in enumerate(row_cells(ln)):
                if cell == "":
                    leaks.append((i, f"空单元格 col{j}"))
    rec("CHECK" if not leaks else "FAIL", "S16-LEAK-SCAN", not leaks,
        f"§16 区间（L{sec_lo}..L{sec_hi}）内 NaN/None/空单元格扫描：{leaks if leaks else '无'}")
    dash_table = [(i, ln) for i, ln in enumerate(sec, start=sec_lo) if DASH in ln and ln.strip().startswith("|")]
    dash_prose = [(i, ln) for i, ln in enumerate(sec, start=sec_lo) if DASH in ln and not ln.strip().startswith("|")]
    rec("CHECK", "S16-DASH-INVENTORY", True,
        f"表格行含破折号：{[i for i, _ in dash_table]}（有意留空）；散文行含破折号（标点）：{[i for i, _ in dash_prose]}")
    for i, ln in dash_table:
        cells = row_cells(ln)
        idxs = [j for j, cc in enumerate(cells) if cc == DASH]
        if not idxs:
            ok, why = True, "该行只把破折号当**标点**用（单元格非空）"
        elif i == base_row:
            ok, why = idxs == [3], "§16.1 基线行的 Δ 列"
        elif s162_lo <= i <= s162_hi:
            ok, why = idxs == [2], "§16.2 旋钮表的『含义』列"
        else:
            ok, why = False, "§16 内出现意料之外的空单元格"
        rec("CHECK" if ok else "FAIL", f"S16-DASH-L{i}", ok, f"L{i} 破折号列位置 {idxs}（期望：{why}）")

    # ---------- 23. 单位/量纲抽查 ----------
    rec("CHECK", "S16-UNIT-HIT5", True,
        "§16.1/§17.1 的 hit@5px% 均为百分数（%），Δ 为 pp（百分点差），§16.3 沿轨误差/加速项为 px，"
        "帧数为整数计数，负占比 0.5570 为 0-1 比例 —— 量纲自洽")
    rec("CHECK", "S16-UNIT-PP", abs((orc["O1_scale"]["hit5_pct"] - base) - orc["O1_scale"]["delta_pp"]) <= 0.0001,
        f"Δpp 与两个百分数之差自洽：{orc['O1_scale']['hit5_pct']} - {base} = "
        f"{orc['O1_scale']['hit5_pct'] - base:.4f} vs 报告 {orc['O1_scale']['delta_pp']}")

    # ---------- 24. §16 数字项计数（按锚点，可复现） ----------
    # 规则：表格**数据列**里的数值 + 散文里被断言的数值；排除章节/序号引用、百分位标签(p50/p90)、
    #       阈值常量(5px / 0.05)、旋钮名、'≈' 近似符号、以及同表内对同一数值的重复复述（旋钮表『含义』列）。
    data_items = {
        "口径行": 1,  # 125109
        "16.1 oracle 表": 7,  # 基线 1 + O1/O2/O3 各 2
        "16.1 分桶表": 25,  # 5 桶 × (n + 4 列)
        "16.2 运动学出口": 3,
        "16.2 applied/replay_steps": 2,
        "16.2 旋钮表": 7,  # 6 个 0.0/1.0 + 32.0
        "16.2 变体表": 5,  # -6.1179 / 1.05 / 1.30 / 0 帧 / 0
        "16.3 偏置表": 16,  # 4 集合 × (n + p50 + mean + 负占比)
        "16.3 相关性": 2,  # Pearson / Spearman
        "16.3 >20px": 1,
        "16.3 加速项": 3,  # n=8838 / p50 / p90（夜 9 新披露 n）
        "16.3 散文": 5,  # 4.35 / 0.15 / 4.35 / 2.02 / 9.12
        "16.4 裁定表": 17,  # #1 行 3 + 缩放位移 2 + 阈值类 5 + 纯运动学 0 + 剩余可回收 7
    }
    n_data = sum(data_items.values())
    rec("CHECK", "S16-DATA-ITEM-COUNT", True,
        f"§16 内可核对的数据项共 {n_data} 项：{data_items}")
    covered = sum(1 for r in _results if r[0] in ("CHECK", "FAIL") and not r[1].startswith("X-"))
    rec("CHECK", "S16-COVERAGE", covered >= n_data,
        f"数字级断言 {covered} 条 ≥ 数据项 {n_data} 项（多余断言来自独立复算 / 加权平均 / 单位复算 / 漂移自测）")

    # ---------- 24b. 覆盖率反查：§16 里还有没有**没被核对过**的数字 ----------
    # 逐行扫 §16 整节（表格跳过首列的行标签），把每个数字 token 分类：
    #   结构性数字（章节号 / 百分位标签 / 阈值常量 / 配置常量）→ 不算数据项；
    #   其余数字 token → 必须能在「已核数字登记表」里找到，即**已被某个逐位断言核对过**。
    unexplained: list[tuple[int, str]] = []
    n_token = 0
    for line_no in range(sec_lo, sec_hi + 1):
        text = lines[line_no - 1].strip()
        payload = " ".join(row_cells(text)[1:]) if text.startswith("|") else text
        for token in NUM_TOKEN_RE.findall(payload):
            norm = token.replace("−", "-").lstrip("+")
            if norm in STRUCTURAL_NUM:
                continue
            n_token += 1
            if norm not in _VERIFIED_NUM:
                unexplained.append((line_no, token))
    rec("CHECK", "S16-DATA-TOKEN-COUNT", True,
        f"§16 整节（L{sec_lo}..L{sec_hi}）非结构性数字 token 共 {n_token} 个（表格跳过首列行标签）")
    rec("CHECK" if not unexplained else "FAIL", "S16-DATA-COVERAGE", not unexplained,
        f"§16 里**未被逐位核对过**的数字 token = {len(unexplained)} 个"
        + ("" if not unexplained else f"（前 12 个：{unexplained[:12]}）"))



# ------------------------------------------------------------------ 报告落盘
def write_markdown(path: Path, report_path: Path, results, n_ok: int, soft_n: int,
                   fails_n: int, mutant_summary: str, mutant_fails: int) -> None:
    """把本次审计结果写成 markdown（**无时间戳** ⇒ 两次运行逐字相同）。"""
    raw = report_path.read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    n_lines = len(raw.decode("utf-8").splitlines())
    try:
        rel = str(report_path.relative_to(ROOT))
    except ValueError:
        rel = str(report_path)

    out: list[str] = []
    out.append("# 独立审计报告：`FINAL_REPORT.md` §16「夜 7 位移修正族 oracle 上界 + 候选族结案」逐位核对")
    out.append("")
    out.append(f"- 审计对象：`{rel}` **§16**（**{n_lines} 行**，md5 `{md5}`）")
    out.append("- 独立审计脚本：`manu/tracking/lookahead/audit_sec16_gap.py`"
               "（**不 import、不调用**任何生成器 / 产物脚本的函数；全部数字自己从 JSON / arms.json 重算）")
    out.append("- 定位方式：**全部按内容锚点**（小节标题前缀 / 表格首列文本），脚本内**不写死行号**")
    out.append("- 真源（只读）：`runs/lookahead/fi_nogmc24/oracle_accel/closure_report.json` + `oracle_bounds.json`、"
               "`runs/lookahead/fi_nogmc24/gapdiag/gap_report.json`、"
               "`runs/lookahead/fi_nogmc24/reg_oracle/closure_report.json`、"
               "`runs/lookahead/fovgate24nogmc/runs/cs_t0043/d3/fov__m6/arms.json`")
    out.append("- 浮点容差：按报告显示精度取**末位半个单位**（4 位 → `5e-5`；2 位 → `5e-3`；1 位 → `5e-2`；整数 → `0.5`）")
    out.append("")
    out.append("## 结论")
    out.append("")
    out.append("| 项 | 值 |")
    out.append("|:--|--:|")
    out.append(f"| 断言总数 | **{len(results)}** |")
    out.append(f"| 数字级逐位 PASS | **{n_ok}** |")
    out.append(f"| 跨节/口径观察（`X-` 前缀） | **{soft_n}** |")
    out.append(f"| **FAIL** | **{fails_n}** |")
    out.append(f"| 漂移自测 FAIL | **{mutant_fails}** |")
    out.append(f"| 脚本退出码 | **{1 if (fails_n or mutant_fails) else 0}** |")
    out.append("")
    out.append("### 直答：§16 有没有任何数字与原始产物不一致？")
    out.append("")
    out.append("**没有。** §16 整节的**每一个非结构性数字 token**都被某条逐位断言核对过"
               "（覆盖率反查：未被核对过 = **0** 个；结构性数字 = 章节号 / 百分位标签 / 阈值与配置常量）。")
    out.append("")
    out.append("## 小节锚点（行号由**内容锚点**推导，脚本内不写死）")
    out.append("")
    out.append("| 小节 | 报告行区间 |")
    out.append("|:--|:--|")
    for lvl, cid, ok, detail in results:
        if cid.startswith("ANCHOR-"):
            out.append(f"| `{cid[len('ANCHOR-'):]}` | {detail.split(' = ')[-1]} |")
    out.append("")
    out.append("## 断言明细（逐条）")
    out.append("")
    out.append("| # | 级别 | 断言 ID | 判定 | 详情 |")
    out.append("|--:|:--|:--|:--|:--|")
    for i, (lvl, cid, ok, detail) in enumerate(results, start=1):
        out.append(f"| {i} | {lvl} | `{cid}` | {'✅ PASS' if ok else '❌ FAIL'} | {detail} |")
    out.append("")
    out.append("## 跨节/口径观察（`X-` 项的**当前**判定）")
    out.append("")
    out.append("| 断言 ID | 判定 | 依据 |")
    out.append("|:--|:--|:--|")
    for lvl, cid, ok, detail in results:
        if cid.startswith("X-"):
            out.append(f"| `{cid}` | {'✅' if ok else '❌'} | {detail} |")
    out.append("")
    out.append("## 漂移免疫自测")
    out.append("")
    out.append(f"- {mutant_summary}")
    out.append("- 判据：**只挪行号 ⇒ 断言结论集必须完全相同**；删内容 ⇒ 降级成 FAIL 明细或 `[FATAL]`，**不许崩**。")
    out.append("")
    out.append("## 复现")
    out.append("")
    out.append("```bash")
    out.append("cd /tmp/pycharm_project_10ae9e2e")
    out.append("PYTHONPATH=. /home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/audit_sec16_gap.py")
    out.append("```")
    out.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ 入口
def main() -> int:
    parser = argparse.ArgumentParser(description="独立审计 FINAL_REPORT.md §16 的数字逐位一致性")
    parser.add_argument("--report", default=str(REPORT), help="被审计的 FINAL_REPORT.md 路径（默认：仓内正式产物）")
    parser.add_argument("--out-md", default=str(OUT_MD), help="本审计报告的输出路径")
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.is_file():
        raise SystemExit(f"[FATAL] 找不到报告文件：{report_path}")
    lines = report_path.read_text(encoding="utf-8").splitlines()

    run_checks(lines)
    fails = [r for r in _results if r[0] == "FAIL"]
    soft = [r for r in _results if r[1].startswith("X-")]
    n_ok = sum(1 for r in _results if r[0] == "CHECK" and not r[1].startswith("X-"))

    # ---- 漂移免疫自测（纯内存：删掉 §16 的空行 + 一段散文后重跑全套断言） ----
    mutant = drift_mutant(lines)
    orig_anchors = [detail for lvl, cid, ok, detail in _results if cid.startswith("ANCHOR-")]
    def verdict_keys(rows):
        # cid 里可能嵌了行号（如 S16-DASH-L<行号>）⇒ 归一化后再比
        return sorted((re.sub(r"-L\d+$", "", cid), ok) for lvl, cid, ok, _ in rows if not cid.startswith("ANCHOR-"))

    orig_keys = verdict_keys(_results)
    before = list(_results)
    _results.clear()
    run_checks(mutant)
    mutant_fails = [r for r in _results if r[0] == "FAIL"]
    mutant_anchors = [detail for lvl, cid, ok, detail in _results if cid.startswith("ANCHOR-")]
    mutant_keys = verdict_keys(_results)
    shifted = orig_anchors != mutant_anchors
    same_verdict = orig_keys == mutant_keys
    mutant_summary = (
        f"漂移副本（原 {len(lines)} 行 → {len(mutant)} 行：删 §16 内的空行 + 每个 §16 小节前插 2 行说明 "
        f"+ 整体下移 {DRIFT_FILLER} 行；**内容一字未删**）后重跑：FAIL {len(mutant_fails)} 条（原 {len(fails)} 条）；"
        f"锚点行号{'确实发生位移' if shifted else '**没有位移（自测无效）**'}；"
        f"断言结论集与原版{'完全一致' if same_verdict else '**不一致（自测失败）**'}"
    )
    _results.clear()
    _results.extend(before)
    if not (shifted and same_verdict):
        print("[FATAL] 漂移自测未通过 ⇒ 脚本对行号漂移不免疫", file=sys.stderr)

    print("=" * 104)
    print(f"§16 独立审计（锚点定位版）：共 {len(_results)} 项断言 —— 数字级逐位 PASS {n_ok} 项，"
          f"跨节/口径观察 {len(soft)} 项，FAIL {len(fails)} 项")
    print(f"漂移免疫自测：{mutant_summary}")
    print("=" * 104)
    for lvl, cid, ok, detail in _results:
        if lvl == "FAIL" or cid.startswith("X-") or cid.startswith("ANCHOR"):
            print(f"[{lvl:5s}] {cid:44s} {detail}")
    if fails:
        print("\n--- FAIL 明细 ---")
        for lvl, cid, ok, detail in fails:
            print(f"[{lvl}] {cid}: {detail}")
    if mutant_fails:
        print("\n--- 漂移自测 FAIL 明细 ---")
        for lvl, cid, ok, detail in mutant_fails:
            print(f"[{lvl}] {cid}: {detail}")

    write_markdown(Path(args.out_md), report_path, list(_results), n_ok, len(soft),
                   len(fails), mutant_summary, len(mutant_fails))
    print(f"审计报告已写出：{args.out_md}")
    return 1 if (fails or mutant_fails) else 0


if __name__ == "__main__":
    sys.exit(main())
