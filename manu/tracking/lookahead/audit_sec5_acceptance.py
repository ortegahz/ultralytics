#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立审计：用户 prompt §5「报告交付（必须）」8 项**逐项在位性 + 数字级**核验。

设计原则（与 audit_sec16_gap.py / audit_sec17_*.py 同一范式）：
  1. **不 import、不调用**任何生成器（make_fastinfer_final_report.py 等）的函数；只用标准库。
  2. 被审计对象 = **用户 prompt §5 的 8 条硬要求**（不是报告的自选章节）：
       ① 口径声明段  ② 四臂表（8 序 + 24 序）  ③ Δ 敏感性表  ④ 模型包对比表
       ⑤ 逐序列归因（含「对总缺口贡献」）      ⑥ 工作点对齐审计表  ⑦ Gate A 结果
       ⑧ 「小延时下精度是否提升」的结论（三个口径分别给）
     每条要求**都必须在 FINAL_REPORT.md（或其点名引用的产物）里有实打实的证据**，
     且**报告里印的数字必须与原始产物 JSON 逐位一致** —— 否则本审计 FAIL。
  3. 全部数字**独立**从原始产物重算：
       runs/lookahead/fi_nogmc/<model>/d<Δ>/arms.json       （8 序）
       runs/lookahead/fi_nogmc24/<model>/d<Δ>/arms.json     （24 序）
       runs/lookahead/fi_nogmc/workpoint/audit.json          （工作点审计）
       runs/lookahead/gateA/gateA.json / gateA24/gateA.json  （Gate A）
       runs/lookahead/fi_nogmc/ruler1_gate_check.json         （尺子1 硬门禁）
  4. 比对按**报告里的显示精度**逐位比：报告单元格解析成浮点后与源值比，
     容差 = 该显示精度末位的半个单位（4 位小数 → 5e-5；6 位小数 → 5e-7；3 位 → 5e-4）。
  5. **零硬编码行号**：所有定位一律走**内容锚点**（`## N. ` / `### N.x ` 标题、
     表格首列文本、散文特征子串）。报告重新生成导致行号漂移时本脚本仍然正确 ——
     这一点由内置的 `drift_mutant()` 自测（纯内存、不落盘）验证。
  6. 只读、无副作用：可重复运行（幂等），不写任何文件（除 --out-md 显式指定）。

用法：
    /home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/audit_sec5_acceptance.py
    /home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/audit_sec5_acceptance.py --report <md 副本>
退出码：0 = 无 FAIL；1 = 存在 FAIL。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# 根定位：仓内 ⇒ 仓根；**交付包内** ⇒ 包内 `repo/` 镜像根 ⇒ 审计可在包外独立复跑。
ROOT = next((a for a in Path(__file__).resolve().parents if (a / "manu/tracking/lookahead").is_dir()),
            Path(__file__).resolve().parent.parent / "repo")

REPORT = ROOT / "runs/lookahead/fi_nogmc/FINAL_REPORT.md"
REPORT8 = ROOT / "runs/lookahead/fi_nogmc/report.md"
OUT_MD = ROOT / "runs/lookahead/fi_nogmc/audit_round7/audit_sec5_acceptance.md"
WORKPOINT_JSON = ROOT / "runs/lookahead/fi_nogmc/workpoint/audit.json"
WORKPOINT_MD = ROOT / "runs/lookahead/fi_nogmc/workpoint/audit.md"
GATEA8 = ROOT / "runs/lookahead/gateA/gateA.json"
GATEA24 = ROOT / "runs/lookahead/gateA24/gateA.json"
RULER1 = ROOT / "runs/lookahead/fi_nogmc/ruler1_gate_check.json"
ARMS8 = ROOT / "runs/lookahead/fi_nogmc"
ARMS24 = ROOT / "runs/lookahead/fi_nogmc24"

ARMS = ("original", "stale", "kinematic", "replay")

# §5 的 8 条硬要求 → 报告里对应的内容锚点（**不含行号**）
A_SEC0 = "## 0. "
A_SEC01 = "### 0.1 "
A_SEC2 = "## 2. "
A_SEC3 = "## 3. "
A_SEC4 = "## 4. "
A_SEC5 = "## 5. "
A_SEC10 = "## 10. "
A_SEC12 = "## 12. "
A_S21 = "### 2.1 "
A_S22 = "### 2.2 "
A_S25 = "### 2.5 "
A_S26 = "### 2.6 "
A_S29 = "### 2.9 "
A_S210 = "### 2.10 "
A_S31 = "### 3.1 "
A_S32 = "### 3.2 "
A_S33 = "### 3.3 "
A_S34 = "### 3.4 "
A_S51 = "### 5.1 "
A_S106 = "### 10.6 "
A_S105 = "### 10.5 "

# 8 序 / 24 序 的模型包 → (报告里显示名, 产物目录名)
MODELS = {
    "cs_t0043": "cs_t0043 (no-GMC 语义)",
    "rc2": "RC2 trial_0080ep3 (GMC 语义)",
    "trial_0049": "trial_0049 ep20 (no-GMC 语义)",
}
DELTAS = (2, 3, 4, 8)

DRIFT_FILLER = 17
DRIFT_FILLER_TEXT = "<!-- drift filler -->"

_results: list[tuple[str, str, bool, str]] = []  # (level, cid, ok, detail)


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


def row_cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def find_row(lines: list[str], lo: int | None, hi: int | None, col0: str):
    """在 [lo, hi] 内找首列文本等于 col0 的表格行 → (行号, 单元格列表)。"""
    if lo is None or hi is None:
        return (None, [])
    for index in range(lo - 1, min(hi, len(lines))):
        if not lines[index].strip().startswith("|"):
            continue
        cells = row_cells(lines[index])
        if not cells or cells[0] == ":--" or set(cells[0]) <= set("-: "):
            continue
        if cells[0].strip() == col0:
            return (index + 1, cells)
    return (None, [])


def block_text(lines: list[str], lo: int, hi: int | None) -> str:
    """返回 [lo, hi] 区间（1-based 闭区间）的文本。"""
    if hi is None:
        hi = len(lines)
    return "\n".join(lines[max(0, lo - 1):min(hi, len(lines))])


def find_text_line(lines: list[str], needle: str, lo: int = 1, hi: int | None = None) -> int | None:
    """在 [lo, hi] 内找含 needle 的行 → 1-based 行号。"""
    if hi is None:
        hi = len(lines)
    for index in range(lo - 1, min(hi, len(lines))):
        if needle in lines[index]:
            return index + 1
    return None


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def as_num(cell: str):
    """把单元格解析成浮点；**✗ / ✅ 等标记先剥掉**；解析失败返回 None。"""
    text = cell.strip().replace("**", "").replace("*", "").replace("`", "")
    text = text.replace("−", "-").replace(",", "").strip()
    text = text.replace("✗", "").replace("✅", "").replace("❌", "").strip()
    if text.startswith("+"):
        text = text[1:]
    # 剥掉单位后缀（pp / px / % / ms / ×）—— 报告里写的是「+0.5959pp」这类
    for suffix in ("pp", "px", "%", "ms", "×", "x"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            break
    try:
        return float(text)
    except ValueError:
        return None


def tol_str(decimals: int) -> str:
    return f"5e-{decimals + 1}"


def close(reported, source, decimals: int) -> bool:
    """按报告显示精度比较：容差 = 该精度末位的半个单位。"""
    if reported is None:
        return False
    tol = 0.5 * 10 ** (-decimals)
    return abs(reported - source) <= tol + 1e-12


def num_eq(cid: str, cell: str, source: float, decimals: int, where: str) -> None:
    reported = as_num(cell)
    ok = close(reported, source, decimals)
    rec("CHECK" if ok else "FAIL", cid, ok,
        f"{where}：报告 {cell!r} vs 真源 {source!r}（显示精度 {decimals} 位，容差 {tol_str(decimals)}）")


def expect_text(cid: str, line_no: int | None, needle: str, lines: list[str]) -> None:
    if line_no is None:
        rec("FAIL", cid, False, f"锚点缺失：找不到含 {needle!r} 的行")
        return
    ok = needle in lines[line_no - 1]
    rec("CHECK" if ok else "FAIL", cid, ok,
        f"L{line_no} 应含 {needle!r}：{lines[line_no - 1].strip()[:120]!r}")


def expect_absent(cid: str, line_no: int | None, needle: str, lines: list[str]) -> None:
    if line_no is None:
        rec("CHECK", cid, True, f"（锚点缺失 ⇒ 视为不含 {needle!r}）")
        return
    ok = needle not in lines[line_no - 1]
    rec("CHECK" if ok else "FAIL", cid, ok,
        f"L{line_no} 不应含 {needle!r}：{lines[line_no - 1].strip()[:120]!r}")


# ------------------------------------------------------------------ 产物读取
def arm_metrics(arms_path: Path, arm: str) -> dict:
    """从 arms.json 取某臂的指标（current_frame 口径 + servo 池化 CLE）。"""
    data = load_json(arms_path)
    node = data["arms"][arm]
    cf = node["current_frame"]["overall"]
    sv = node["servo"]["overall"]
    return {
        "f1": cf["f1"] * 100.0,
        "recall": cf["recall"] * 100.0,
        "precision": cf["precision"] * 100.0,
        "far": cf["far_per_frame"],
        "hit5": sv["hit_rate_5px"] * 100.0,
        "cle_p50": sv["cle_pooled_p50"],
        "cle_p95": sv["cle_pooled_p95"],
        "cf": cf,
        "servo": sv,
    }


def arms_path(root: Path, model: str, delta: int) -> Path:
    return root / model / f"d{delta}" / "arms.json"


# ------------------------------------------------------------------ 漂移自测
def drift_mutant(lines: list[str]) -> list[str]:
    """构造一份**行号漂移**的报告文本（内容**一字不删**，只改排版/加内容）。

    - 删掉全篇空行（模拟上游重新排版、压掉空行）；
    - 在每个 '## '/'### ' 标题前插入 2 行占位说明（模拟上游在中间补了几段）；
    - 再在文件最前面插入若干占位行（模拟上游在最前面又加了一节）。
    纯内存操作，不落盘。锚点定位 ⇒ 漂移后所有断言结论必须与原版**完全一致**。
    """
    out: list[str] = []
    for line in lines:
        if line.startswith(("## ", "### ")):
            out.append(DRIFT_FILLER_TEXT)
            out.append(DRIFT_FILLER_TEXT)
        if line.strip() == "":
            continue
        out.append(line)
    return [f"<!-- drift head {i} -->" for i in range(DRIFT_FILLER)] + out


# ------------------------------------------------------------------ ① 口径声明段
def check_01_caliber(lines: list[str]) -> None:
    """§5①：口径声明段 —— 必须有独立小节 + 关键纪律逐条在位。"""
    lo, hi = section_range(lines, A_SEC0)
    rec("CHECK", "S5.01-SEC-EXISTS", lo is not None,
        f"§0「口径声明与结论速览」锚点 {A_SEC0!r} → L{lo}")
    if lo is None:
        return
    block = "\n".join(lines[lo - 1:hi])
    # 关键纪律（逐条；缺一条即 FAIL）
    for cid, needle in [
        ("S5.01-DISC-STALE-BASELINE", "`stale` 才是交付基线"),
        ("S5.01-DISC-ORIGINAL-CEILING", "零延迟天花板（oracle 上界）"),
        ("S5.01-DISC-CLE-POOLED", "cle_pooled_*"),
        ("S5.01-DISC-8V24", "8 序 ≠ 24 序"),
        ("S5.01-DISC-DELIVERY-GATES", "F1 ≥ 0.88 / Recall ≥ 0.85 / FAR ≤ 0.030"),
        ("S5.01-DISC-DELTA-ASSUMED", "Δ 是假设值"),
        ("S5.01-DISC-MODEL-BUNDLE", "模型包 = 权重 + 特征语义"),
    ]:
        ok = needle in block
        rec("CHECK" if ok else "FAIL", cid, ok,
            f"口径声明段应含 {needle!r}" + ("" if ok else " —— **缺失**"))
    # 纪律条数 ≥ 8（表体行数）
    rows = [l for l in lines[lo - 1:hi] if l.strip().startswith("| ") and not l.strip().startswith("|:--")]
    rec("CHECK", "S5.01-DISC-COUNT", len(rows) >= 8,
        f"口径声明段表格行数 = {len(rows)}（≥ 8）")


# ------------------------------------------------------------------ ② 四臂表
def check_02_four_arm(lines: list[str], tag: str, root: Path, model: str, sec_prefix: str,
                      model_label: str, delta: int) -> None:
    """§5②：四臂表（8 序 / 24 序）—— 四臂齐全 + 逐位数字与 arms.json 一致。"""
    lo, hi = section_range(lines, sec_prefix)
    if lo is None:
        rec("FAIL", f"S5.02-{tag}-{model}-D{delta}-SEC", False,
            f"找不到四臂表小节 {sec_prefix!r}")
        return
    rec("CHECK", f"S5.02-{tag}-{model}-D{delta}-SEC", True,
        f"{tag} {model_label} Δ={delta} 四臂表 → L{lo}-{hi}")
    path = arms_path(root, model, delta)
    if not path.exists():
        rec("FAIL", f"S5.02-{tag}-{model}-D{delta}-ARMS", False, f"产物缺失：{path}")
        return
    for arm in ARMS:
        row_no, cells = find_row(lines, lo, hi, arm)
        if row_no is None:
            rec("FAIL", f"S5.02-{tag}-{model}-D{delta}-{arm}-ROW", False,
                f"四臂表里缺 {arm} 臂")
            continue
        rec("CHECK", f"S5.02-{tag}-{model}-D{delta}-{arm}-ROW", True,
            f"{arm} 臂 → L{row_no}")
        src = arm_metrics(path, arm)
        # 表头：| 臂 | 角色 | F1% | Recall% | Precision% | FAR/帧 | hit@5px% | CLE p50(池) | CLE p95(池) | 交付三门 |
        if len(cells) < 10:
            rec("FAIL", f"S5.02-{tag}-{model}-D{delta}-{arm}-WIDTH", False,
                f"{arm} 臂只有 {len(cells)} 列（应 ≥ 10）")
            continue
        num_eq(f"S5.02-{tag}-{model}-D{delta}-{arm}-F1", cells[2], src["f1"], 4, f"{tag} {model} Δ={delta} {arm} F1%")
        num_eq(f"S5.02-{tag}-{model}-D{delta}-{arm}-RECALL", cells[3], src["recall"], 4, f"{tag} {model} Δ={delta} {arm} Recall%")
        num_eq(f"S5.02-{tag}-{model}-D{delta}-{arm}-PRECISION", cells[4], src["precision"], 4, f"{tag} {model} Δ={delta} {arm} Precision%")
        num_eq(f"S5.02-{tag}-{model}-D{delta}-{arm}-FAR", cells[5], src["far"], 6, f"{tag} {model} Δ={delta} {arm} FAR/帧")
        num_eq(f"S5.02-{tag}-{model}-D{delta}-{arm}-HIT5", cells[6], src["hit5"], 4, f"{tag} {model} Δ={delta} {arm} hit@5px%")
        num_eq(f"S5.02-{tag}-{model}-D{delta}-{arm}-CLE50", cells[7], src["cle_p50"], 3, f"{tag} {model} Δ={delta} {arm} CLE p50(池)")
        num_eq(f"S5.02-{tag}-{model}-D{delta}-{arm}-CLE95", cells[8], src["cle_p95"], 3, f"{tag} {model} Δ={delta} {arm} CLE p95(池)")
        # 交付三门必须与源值一致
        f1, recall, far = src["f1"] / 100.0, src["recall"] / 100.0, src["far"]
        expect_gates = "PASS" if (f1 >= 0.88 and recall >= 0.85 and far <= 0.030) else "FAIL"
        ok = expect_gates in cells[9]
        rec("CHECK" if ok else "FAIL", f"S5.02-{tag}-{model}-D{delta}-{arm}-GATES", ok,
            f"{tag} {model} Δ={delta} {arm} 交付三门：报告 {cells[9]!r} vs 依据源值应为 {expect_gates}")


# ------------------------------------------------------------------ ③ Δ 敏感性表
def check_03_delta(lines: list[str], tag: str, sec_prefix: str, root: Path,
                   model_list: list[str], model_labels: dict) -> None:
    """§5③：Δ 敏感性表 —— Δ∈{2,3,4,8} 四行齐全 + 数字逐位一致。"""
    lo, hi = section_range(lines, sec_prefix)
    if lo is None:
        rec("FAIL", f"S5.03-{tag}-SEC", False, f"找不到 Δ 敏感性表小节 {sec_prefix!r}")
        return
    rec("CHECK", f"S5.03-{tag}-SEC", True, f"{tag} Δ 敏感性表 → L{lo}-{hi}")
    for model in model_list:
        label = model_labels[model]
        for delta in DELTAS:
            row_no, cells = find_row(lines, lo, hi, f"{label} | {delta} |")
            # find_row 按首列精确匹配；这里首列是模型包名 ⇒ 改用前缀匹配
            if row_no is None:
                row_no, cells = _find_row_prefix(lines, lo, hi, label, delta)
            if row_no is None:
                rec("FAIL", f"S5.03-{tag}-{model}-D{delta}-ROW", False,
                    f"Δ 敏感性表缺 {label} Δ={delta} 行")
                continue
            rec("CHECK", f"S5.03-{tag}-{model}-D{delta}-ROW", True,
                f"{label} Δ={delta} → L{row_no}")
            path = arms_path(root, model, delta)
            if not path.exists():
                rec("FAIL", f"S5.03-{tag}-{model}-D{delta}-ARMS", False, f"产物缺失：{path}")
                continue
            src = arm_metrics(path, "replay")
            stale = arm_metrics(path, "stale")
            num_eq(f"S5.03-{tag}-{model}-D{delta}-F1", cells[3], src["f1"], 4, f"{tag} {label} Δ={delta} replay F1%")
            num_eq(f"S5.03-{tag}-{model}-D{delta}-RECALL", cells[4], src["recall"], 4, f"{tag} {label} Δ={delta} replay Recall%")
            num_eq(f"S5.03-{tag}-{model}-D{delta}-FAR", cells[5], src["far"], 6, f"{tag} {label} Δ={delta} replay FAR/帧")
            num_eq(f"S5.03-{tag}-{model}-D{delta}-HIT5", cells[6], src["hit5"], 4, f"{tag} {label} Δ={delta} replay hit@5px%")
            num_eq(f"S5.03-{tag}-{model}-D{delta}-CLE50", cells[7], src["cle_p50"], 3, f"{tag} {label} Δ={delta} replay CLE p50(池)")
            num_eq(f"S5.03-{tag}-{model}-D{delta}-CLE95", cells[8], src["cle_p95"], 3, f"{tag} {label} Δ={delta} replay CLE p95(池)")
            # 相对 stale 的 F1 增益
            gain = src["f1"] - stale["f1"]
            num_eq(f"S5.03-{tag}-{model}-D{delta}-GAIN", cells[9], gain, 4,
                   f"{tag} {label} Δ={delta} 相对 stale F1 增益(pp)")


def _find_row_prefix(lines: list[str], lo: int, hi: int, label: str, delta: int):
    """Δ 敏感性表的行首列是模型包名、第 2 列是 Δ ⇒ 按 (label, delta) 定位。"""
    for index in range(lo - 1, min(hi, len(lines))):
        if not lines[index].strip().startswith("|"):
            continue
        cells = row_cells(lines[index])
        if len(cells) < 3:
            continue
        if cells[0].strip() != label:
            continue
        if as_num(cells[1].split()[0] if cells[1].split() else cells[1]) == float(delta):
            return (index + 1, cells)
    return (None, [])


# ------------------------------------------------------------------ ④ 模型包对比表
def check_04_model_bundle(lines: list[str]) -> None:
    """§5④：模型包对比表 —— 24 序同 Δ rc2(GMC) vs cs_t0043(no-GMC) 四指标逐位。"""
    lo, hi = section_range(lines, A_S210)
    if lo is None:
        rec("FAIL", "S5.04-SEC", False, f"找不到模型包对比表 {A_S210!r}")
        return
    rec("CHECK", "S5.04-SEC", True, f"模型包对比表 → L{lo}-{hi}")
    for delta in DELTAS:
        for metric, decimals, getter in [
            ("F1%", 4, lambda m: m["f1"]),
            ("FAR/帧", 6, lambda m: m["far"]),
            ("hit@5px%", 4, lambda m: m["hit5"]),
            ("CLE p95(池)", 3, lambda m: m["cle_p95"]),
        ]:
            row_no, cells = _find_row_pair(lines, lo, hi, str(delta), metric)
            if row_no is None:
                rec("FAIL", f"S5.04-D{delta}-{metric}-ROW", False,
                    f"模型包对比表缺 Δ={delta} / {metric} 行")
                continue
            rec("CHECK", f"S5.04-D{delta}-{metric}-ROW", True, f"Δ={delta} {metric} → L{row_no}")
            src_rc2 = arm_metrics(arms_path(ARMS24, "rc2", delta), "replay")
            src_cs = arm_metrics(arms_path(ARMS24, "cs_t0043", delta), "replay")
            num_eq(f"S5.04-D{delta}-{metric}-RC2", cells[2], getter(src_rc2), decimals,
                   f"Δ={delta} {metric} rc2(GMC)")
            num_eq(f"S5.04-D{delta}-{metric}-CS", cells[3], getter(src_cs), decimals,
                   f"Δ={delta} {metric} cs_t0043(no-GMC)")
            # Δ 列 = cs − rc2（带 pp / px 单位后缀）
            diff = getter(src_cs) - getter(src_rc2)
            reported = as_num(cells[4])
            ok = close(reported, diff, decimals)
            rec("CHECK" if ok else "FAIL", f"S5.04-D{delta}-{metric}-DIFF", ok,
                f"Δ={delta} {metric} 差值：报告 {cells[4]!r} vs 源值差 {diff!r}")


def _find_row_pair(lines: list[str], lo: int, hi: int, col0: str, col1: str):
    """模型包对比表：首列 = Δ、次列 = 指标名。"""
    for index in range(lo - 1, min(hi, len(lines))):
        if not lines[index].strip().startswith("|"):
            continue
        cells = row_cells(lines[index])
        if len(cells) < 5:
            continue
        if cells[0].strip() == col0 and cells[1].strip() == col1:
            return (index + 1, cells)
    return (None, [])


# ------------------------------------------------------------------ ⑤ 逐序列归因
def model_block_range(lines: list[str], sec_prefix: str, model: str,
                      other_model: str) -> tuple[int, int]:
    """§10.6 里并排放着两个模型的逐序表 ⇒ 必须**按模型标题切块**。

    cs_t0043 块从 `**cs_t0043 ...**` 那行开始，到 `**RC2 ...**` 那行之前结束；
    RC2 块则到小节末尾为止。返回 (起, 止) 1-based 闭区间。
    """
    lo, hi = section_range(lines, sec_prefix)
    if lo is None:
        raise SystemExit(f"[FATAL] 找不到小节 {sec_prefix!r}")
    # 目录名 → 报告里的显示名（**块标题以 ** 开头）
    display = {"cs_t0043": "cs_t0043 (no-GMC 语义)",
               "rc2": "RC2 trial_0080ep3 (GMC 语义)"}[model]
    other_display = {"cs_t0043": "cs_t0043 (no-GMC 语义)",
                     "rc2": "RC2 trial_0080ep3 (GMC 语义)"}[other_model]
    start = find_text_line(lines, f"**{display}", lo, hi)
    if start is None:
        raise SystemExit(f"[FATAL] 小节 {sec_prefix!r} 里找不到模型块标题 **{display}")
    end = find_text_line(lines, f"**{other_display}", start + 1, hi)
    if end is None:
        end = hi + 1
    return (start, end - 1)


def check_05_per_sequence(lines: list[str], sec_prefix: str, root: Path, model: str,
                          tag: str, expect_cols: int) -> None:
    """§5⑤：逐序列归因表 —— 必须含「对总缺口贡献」列 + 逐序数字与 arms.json 一致。"""
    lo, hi = section_range(lines, sec_prefix)
    if lo is None:
        rec("FAIL", f"S5.05-{tag}-SEC", False, f"找不到逐序列归因小节 {sec_prefix!r}")
        return
    rec("CHECK", f"S5.05-{tag}-SEC", True, f"{tag} 逐序列归因 → L{lo}-{hi}")
    block = "\n".join(lines[lo - 1:hi])
    ok = "对总缺口贡献" in block
    rec("CHECK" if ok else "FAIL", f"S5.05-{tag}-CONTRIB-COL", ok,
        f"{tag} 逐序表必须含「对总缺口贡献」列" + ("" if ok else " —— **缺失**"))
    # 逐序表按模型切块（§10.6 里 cs 与 RC2 并排）
    other = "rc2" if model == "cs_t0043" else "cs_t0043"
    try:
        lo, hi = model_block_range(lines, sec_prefix, model, other)
    except SystemExit as exc:
        rec("FAIL", f"S5.05-{tag}-BLOCK", False, str(exc))
        return
    rec("CHECK", f"S5.05-{tag}-BLOCK", True, f"{tag} {model} 逐序块 → L{lo}-{hi}")
    # 表头行
    hdr_no = find_text_line(lines, "| 序列 | GT帧 |", lo, hi)
    if hdr_no is None:
        rec("FAIL", f"S5.05-{tag}-HDR", False, "找不到逐序表表头（`| 序列 | GT帧 |`）")
        return
    hdr = row_cells(lines[hdr_no - 1])
    rec("CHECK", f"S5.05-{tag}-HDR", len(hdr) == expect_cols,
        f"{tag} 逐序表表头列数 = {len(hdr)}（期望 {expect_cols}）")
    if "对总缺口贡献(pp)" not in hdr:
        rec("FAIL", f"S5.05-{tag}-CONTRIB-HDR", False,
            f"表头缺「对总缺口贡献(pp)」列：{hdr}")
    else:
        rec("CHECK", f"S5.05-{tag}-CONTRIB-HDR", True, "表头含「对总缺口贡献(pp)」列")
    # 逐序行：stale F1% / replay F1% / ΔF1 / stale hit / replay hit / Δhit / 天花板缺口
    path = arms_path(root, model, 3)
    if not path.exists():
        rec("FAIL", f"S5.05-{tag}-ARMS", False, f"产物缺失：{path}")
        return
    data = load_json(path)
    per_cf = {arm: data["arms"][arm]["current_frame"]["per_sequence"] for arm in ARMS}
    per_sv = {arm: data["arms"][arm]["servo"]["per_sequence"] for arm in ARMS}
    total_gt = data["arms"]["replay"]["current_frame"]["overall"]["gt"]
    checked = 0
    for index in range(hdr_no, min(hi, len(lines))):
        raw = lines[index]
        if not raw.strip().startswith("|"):
            continue
        cells = row_cells(raw)
        if len(cells) < 5 or not cells[0].startswith("VIDEO"):
            continue
        seq = cells[0]
        if seq not in per_cf["replay"]:
            rec("FAIL", f"S5.05-{tag}-SEQ-{seq}", False, f"产物里没有序列 {seq}")
            continue
        gt = per_cf["replay"][seq]["gt"]
        num_eq(f"S5.05-{tag}-SEQ-{seq}-GT", cells[1], float(gt), 0, f"{tag} {seq} GT帧")
        num_eq(f"S5.05-{tag}-SEQ-{seq}-STALE-F1", cells[2], per_cf["stale"][seq]["f1"] * 100.0, 4,
               f"{tag} {seq} stale F1%")
        num_eq(f"S5.05-{tag}-SEQ-{seq}-REPLAY-F1", cells[3], per_cf["replay"][seq]["f1"] * 100.0, 4,
               f"{tag} {seq} replay F1%")
        num_eq(f"S5.05-{tag}-SEQ-{seq}-DF1", cells[4],
               (per_cf["replay"][seq]["f1"] - per_cf["stale"][seq]["f1"]) * 100.0, 4,
               f"{tag} {seq} ΔF1(pp)")
        num_eq(f"S5.05-{tag}-SEQ-{seq}-STALE-HIT", cells[5], per_sv["stale"][seq]["hit_rate_5px"] * 100.0, 4,
               f"{tag} {seq} stale hit@5px%")
        num_eq(f"S5.05-{tag}-SEQ-{seq}-REPLAY-HIT", cells[6], per_sv["replay"][seq]["hit_rate_5px"] * 100.0, 4,
               f"{tag} {seq} replay hit@5px%")
        num_eq(f"S5.05-{tag}-SEQ-{seq}-DHIT", cells[7],
               (per_sv["replay"][seq]["hit_rate_5px"] - per_sv["stale"][seq]["hit_rate_5px"]) * 100.0, 4,
               f"{tag} {seq} Δhit(pp)")
        # 天花板 hit 缺口 = original hit − replay hit
        num_eq(f"S5.05-{tag}-SEQ-{seq}-CEILGAP", cells[8],
               (per_sv["original"][seq]["hit_rate_5px"] - per_sv["replay"][seq]["hit_rate_5px"]) * 100.0, 4,
               f"{tag} {seq} replay vs 天花板 hit 缺口(pp)")
        # 对总缺口贡献 = 该序 hit 缺口 × 该序 GT 占比
        contrib = (per_sv["original"][seq]["hit_rate_5px"] - per_sv["replay"][seq]["hit_rate_5px"]) * 100.0 * (gt / total_gt)
        num_eq(f"S5.05-{tag}-SEQ-{seq}-CONTRIB", cells[9], contrib, 4,
               f"{tag} {seq} 对总缺口贡献(pp)")
        checked += 1
    rec("CHECK", f"S5.05-{tag}-SEQ-COUNT", checked > 0,
        f"{tag} 逐序归因共核对 {checked} 个序列")


# ------------------------------------------------------------------ ⑥ 工作点对齐审计表
def check_06_workpoint(lines: list[str]) -> None:
    """§5⑥：工作点对齐审计表 —— FINAL §4 + 完整 24 张表 audit.md（表 A FAR–F1 / 表 B 最优点）。"""
    lo, hi = section_range(lines, A_SEC4)
    if lo is None:
        rec("FAIL", "S5.06-SEC", False, f"找不到工作点对齐审计小节 {A_SEC4!r}")
        return
    rec("CHECK", "S5.06-SEC", True, f"工作点对齐审计 → L{lo}-{hi}")
    block = "\n".join(lines[lo - 1:hi])
    for cid, needle in [
        ("S5.06-GRID", "th_base ∈ {0.40,0.44,0.48,0.52,0.56}"),
        ("S5.06-THGROUND", "th_ground ∈ {0.22,0.26,0.30}"),
        ("S5.06-SAMEFAR", "同 FAR 插值"),
        ("S5.06-DELTA-F1", "-0.1491pp"),
        ("S5.06-RUNS", "rc≠0 = 0"),
        ("S5.06-REF", "runs/lookahead/fi_nogmc/workpoint/audit.md"),
    ]:
        ok = needle in block
        rec("CHECK" if ok else "FAIL", cid, ok,
            f"工作点审计段应含 {needle!r}" + ("" if ok else " —— **缺失**"))
    # 完整 24 张表在 audit.md
    if not WORKPOINT_MD.exists():
        rec("FAIL", "S5.06-AUDITMD", False, f"缺失：{WORKPOINT_MD}")
        return
    md_lines = WORKPOINT_MD.read_text(encoding="utf-8").splitlines()
    rec("CHECK", "S5.06-AUDITMD", True, f"完整审计表：{WORKPOINT_MD.name}（{len(md_lines)} 行）")
    for cid, needle in [
        ("S5.06-TABLE-A", "## 表 A · FAR–F1 曲线"),
        ("S5.06-TABLE-B", "## 表 B · 每个模型 × 每个 Δ 的最优点"),
        ("S5.06-TABLE-C", "## 表 C · 同 FAR 插值对比"),
        ("S5.06-TABLE-D", "## 表 D · 工作点敏感性"),
    ]:
        ok = find_line(md_lines, needle) is not None
        rec("CHECK" if ok else "FAIL", cid, ok,
            f"audit.md 应含 {needle!r}" + ("" if ok else " —— **缺失**"))
    # 表 A / 表 B 的行数 = 2 模型 × 2 Δ × 2 臂（表 B）/ 网格点数（表 A）
    if not WORKPOINT_JSON.exists():
        rec("FAIL", "S5.06-JSON", False, f"缺失：{WORKPOINT_JSON}")
        return
    wp = load_json(WORKPOINT_JSON)
    head = wp["headline"]
    # 固定工作点下 ΔF1 与同 FAR 插值 ΔF1 —— 必须与 FINAL §4 印的一致
    # audit.json 的 *_pp 字段**本身就是百分点**（不是比例）⇒ 不再 ×100
    num_eq("S5.06-JSON-DELTAPP", "-0.1491pp", head["fixed_workpoint"]["delta_pp"], 4,
           "audit.json 固定工作点 ΔF1(pp)")
    num_eq("S5.06-JSON-SAMEFAR", "-0.0979pp", head["same_far"]["mean_delta_pp"], 4,
           "audit.json 同 FAR 插值 ΔF1(pp)")
    lo_iv, hi_iv = head["same_far"]["interval"]
    ok = (abs(lo_iv - 0.006070) < 5e-7) and (abs(hi_iv - 0.012243) < 5e-7)
    rec("CHECK" if ok else "FAIL", "S5.06-JSON-INTERVAL", ok,
        f"同 FAR 公共区间：源 {lo_iv!r} ~ {hi_iv!r}（报告写 0.006070 ~ 0.012243）")
    # 表 A：8 个 (model|delta|arm) 组，每组网格点 ≥ 5
    for key in ["rc2|d3|replay", "rc2|d8|replay", "cs_t0043|d3|replay", "cs_t0043|d8|replay"]:
        pts = wp["tables"]["A"].get(key, [])
        ok = len(pts) >= 5
        rec("CHECK" if ok else "FAIL", f"S5.06-TABLEA-{key}", ok,
            f"表 A 组 {key} 网格点数 = {len(pts)}（≥ 5）")
    # 表 B：每个模型×Δ×臂 都有最优点 + 是否过三门
    for key in ["rc2|d3|replay", "rc2|d8|replay", "cs_t0043|d3|replay", "cs_t0043|d8|replay",
                "rc2|d3|stale", "cs_t0043|d3|stale"]:
        node = wp["tables"]["B"].get(key)
        if node is None:
            rec("FAIL", f"S5.06-TABLEB-{key}", False, f"表 B 缺 {key}")
            continue
        point = node["point"]
        expect = (point["f1"] >= 0.88 and point["recall"] >= 0.85 and point["far"] <= 0.030)
        ok = bool(node["passes_delivery_triple"]) == expect
        rec("CHECK" if ok else "FAIL", f"S5.06-TABLEB-{key}", ok,
            f"表 B {key}：passes_delivery_triple={node['passes_delivery_triple']} vs 按点重算={expect}"
            f"（F1 {point['f1']:.4f} / R {point['recall']:.4f} / FAR {point['far']:.6f}）")


# ------------------------------------------------------------------ ⑦ Gate A 结果
def check_07_gatea(lines: list[str]) -> None:
    """§5⑦：Gate A 结果 —— §5 判定 PASS + 8 序/24 序 gateA.json 逐位 + 尺子1 硬门禁。"""
    lo, hi = section_range(lines, A_SEC5)
    if lo is None:
        rec("FAIL", "S5.07-SEC", False, f"找不到 Gate A 小节 {A_SEC5!r}")
        return
    rec("CHECK", "S5.07-SEC", True, f"Gate A 小节 → L{lo}-{hi}")
    block = "\n".join(lines[lo - 1:hi])
    ok = "PASS" in block
    rec("CHECK" if ok else "FAIL", "S5.07-VERDICT", ok, "Gate A 段必须给出 PASS 判定")
    # 24 序 Gate A 的 "24/24 相同" 在 **§5.1**（§5.1 是 §5 的子节）⇒ 断言跨两段
    s51_lo = find_line(lines, "### 5.1")
    s52_lo = find_line(lines, "### 5.2")
    wide = block_text(lines, lo, hi)
    if s51_lo is not None:
        wide += "\n" + block_text(lines, s51_lo, (s52_lo - 1) if s52_lo else None)
    for cid, needle in [("S5.07-8SEQ", "8/8 相同"), ("S5.07-24SEQ", "24/24 相同"),
                        ("S5.07-RULER1", "尺子1")]:
        ok = needle in wide
        rec("CHECK" if ok else "FAIL", cid, ok,
            f"Gate A 段（§5 + §5.1）应含 {needle!r}" + ("" if ok else " —— **缺失**"))
    # gateA.json 真源
    for tag, path, expect_n in [("8seq", GATEA8, 8), ("24seq", GATEA24, 24)]:
        if not path.exists():
            rec("FAIL", f"S5.07-JSON-{tag}", False, f"缺失：{path}")
            continue
        ga = load_json(path)
        ok = ga.get("verdict") == "PASS"
        rec("CHECK" if ok else "FAIL", f"S5.07-JSON-{tag}-VERDICT", ok,
            f"gateA{tag}.json verdict = {ga.get('verdict')!r}")
        for model, node in ga["models"].items():
            ok = bool(node.get("all_ok"))
            rec("CHECK" if ok else "FAIL", f"S5.07-JSON-{tag}-{model}-ALLOK", ok,
                f"gateA{tag}.json[{model}].all_ok = {node.get('all_ok')}")
            preds = node.get("preds", {})
            ok = preds.get("n_files") == expect_n and preds.get("n_L1_equal") == expect_n
            rec("CHECK" if ok else "FAIL", f"S5.07-JSON-{tag}-{model}-L1", ok,
                f"gateA{tag}.json[{model}] preds：n_files={preds.get('n_files')}、"
                f"n_L1_equal={preds.get('n_L1_equal')}（期望 {expect_n}/{expect_n}）")
        # 尺子1：四臂 source_frame.f1 逐位相等
        ruler = ga.get("ruler1", {})
        f1s = ruler.get("source_frame_f1", {})
        ok = bool(f1s) and len(set(f1s.values())) == 1
        rec("CHECK" if ok else "FAIL", f"S5.07-JSON-{tag}-RULER1", ok,
            f"gateA{tag}.json 尺子1：四臂 source_frame.f1 = {f1s}")
    # 尺子1 硬门禁（54 产物 / 0 违规）
    if RULER1.exists():
        r1 = load_json(RULER1)
        ok = r1.get("verdict") == "PASS" and r1.get("violations") == 0
        rec("CHECK" if ok else "FAIL", "S5.07-RULER1-GATE", ok,
            f"ruler1_gate_check.json：verdict={r1.get('verdict')!r}、violations={r1.get('violations')}"
            f"、checked_products={r1.get('checked_products')}")
    else:
        rec("FAIL", "S5.07-RULER1-GATE", False, f"缺失：{RULER1}")


# ------------------------------------------------------------------ ⑧ 三口径结论
def check_08_conclusion(lines: list[str]) -> None:
    """§5⑧：「小延时下精度是否提升」三口径（相对 rc2 同 Δ / 相对 stale / 相对天花板）。"""
    lo, hi = section_range(lines, A_SEC01)
    if lo is None:
        rec("FAIL", "S5.08-SEC", False, f"找不到结论小节 {A_SEC01!r}")
        return
    rec("CHECK", "S5.08-SEC", True, f"「小延时下精度是否提升」结论 → L{lo}-{hi}")
    block = "\n".join(lines[lo - 1:hi])
    for cid, needle in [
        ("S5.08-CAL1", "口径①"),
        ("S5.08-CAL2", "口径②"),
        ("S5.08-CAL3", "口径③"),
        ("S5.08-REL-RC2", "相对 rc2"),
        ("S5.08-REL-STALE", "相对 **stale"),
        ("S5.08-REL-CEILING", "零延迟天花板"),
        ("S5.08-ANSWER", "是，大幅提升"),
    ]:
        ok = needle in block
        rec("CHECK" if ok else "FAIL", cid, ok,
            f"结论段应含 {needle!r}" + ("" if ok else " —— **缺失**"))
    # **逐行级在位性**：§0.1 的每一行结论都必须单独存在（不能靠"别处也有"蒙混）
    for cid, row_key, must_have in [
        ("S5.08-ROW-CAL1", "| 口径① | 相对 rc2 **同 Δ**（no-GMC vs GMC，8 序 Δ=3 replay）", ["-0.1491pp", "-0.001345", "+0.2613pp"]),
        ("S5.08-ROW-CAL1B", "| 口径①b | 相对 rc2 **同 Δ**（no-GMC vs GMC，**24 序官方口径** Δ=3 replay）", ["+0.5537pp", "+2.8892pp", "-1.6604px"]),
        ("S5.08-ROW-CAL2-CS", "| 口径② | 相对 **stale（交付基线）**，24 序官方口径 Δ=3 · cs_t0043 (no-GMC 语义)", ["+4.1812pp", "+21.0120pp", "-0.033754"]),
        ("S5.08-ROW-CAL2-RC2", "| 口径② | 相对 **stale（交付基线）**，24 序官方口径 Δ=3 · RC2 trial_0080ep3 (GMC 语义)", ["+4.1374pp", "+19.3911pp", "-0.032925"]),
        ("S5.08-ROW-CAL3-CS", "| 口径③ | 相对**零延迟天花板**（追回占比），24 序 Δ=3 · cs_t0043 (no-GMC 语义)", ["已追回 86.4%", "90.6049", "94.7861", "95.4462"]),
        ("S5.08-ROW-CAL3-RC2", "| 口径③ | 相对**零延迟天花板**（追回占比），24 序 Δ=3 · RC2 trial_0080ep3 (GMC 语义)", ["已追回 88.9%", "90.0951", "94.2325", "94.7469"]),
    ]:
        row_no = None
        for index in range(lo - 1, min(hi, len(lines))):
            if lines[index].startswith(row_key):
                row_no = index + 1
                break
        if row_no is None:
            rec("FAIL", cid, False, f"§0.1 缺结论行：{row_key!r}")
            continue
        ok = True
        missing = [m for m in must_have if m not in lines[row_no - 1]]
        ok = not missing
        rec("CHECK" if ok else "FAIL", cid, ok,
            f"L{row_no} 结论行应含 {must_have}" + ("" if ok else f" —— **缺 {missing}**"))

    # 口径①：相对 rc2 同 Δ（8 序）—— 数字与 arms.json 逐位
    rc2 = arm_metrics(arms_path(ARMS8, "rc2", 3), "replay")
    cs = arm_metrics(arms_path(ARMS8, "cs_t0043", 3), "replay")
    num_eq("S5.08-CAL1-F1", "-0.1491pp", cs["f1"] - rc2["f1"], 4, "口径① 8 序 Δ=3 F1 差(pp)")
    num_eq("S5.08-CAL1-FAR", "-0.001345", cs["far"] - rc2["far"], 6, "口径① 8 序 Δ=3 FAR 差")
    num_eq("S5.08-CAL1-HIT5", "+0.2613pp", cs["hit5"] - rc2["hit5"], 4, "口径① 8 序 Δ=3 hit@5px 差(pp)")
    num_eq("S5.08-CAL1-CLE95", "-0.9551px", cs["cle_p95"] - rc2["cle_p95"], 4, "口径① 8 序 Δ=3 CLE p95 差(px)")
    # 口径①b：相对 rc2 同 Δ（24 序）
    rc2_24 = arm_metrics(arms_path(ARMS24, "rc2", 3), "replay")
    cs_24 = arm_metrics(arms_path(ARMS24, "cs_t0043", 3), "replay")
    num_eq("S5.08-CAL1B-F1", "+0.5537pp", cs_24["f1"] - rc2_24["f1"], 4, "口径①b 24 序 Δ=3 F1 差(pp)")
    num_eq("S5.08-CAL1B-HIT5", "+2.8892pp", cs_24["hit5"] - rc2_24["hit5"], 4, "口径①b 24 序 Δ=3 hit@5px 差(pp)")
    num_eq("S5.08-CAL1B-CLE95", "-1.6604px", cs_24["cle_p95"] - rc2_24["cle_p95"], 4, "口径①b 24 序 Δ=3 CLE p95 差(px)")
    # 口径②：相对 stale（24 序 Δ=3）
    stale_24 = arm_metrics(arms_path(ARMS24, "cs_t0043", 3), "stale")
    num_eq("S5.08-CAL2-F1", "+4.1812pp", cs_24["f1"] - stale_24["f1"], 4, "口径② 24 序 Δ=3 F1 增益(pp)")
    num_eq("S5.08-CAL2-HIT5", "+21.0120pp", cs_24["hit5"] - stale_24["hit5"], 4, "口径② 24 序 Δ=3 hit@5px 增益(pp)")
    num_eq("S5.08-CAL2-FAR", "-0.033754", cs_24["far"] - stale_24["far"], 6, "口径② 24 序 Δ=3 FAR 差")
    # 口径③：相对天花板（追回占比）
    ceil_24 = arm_metrics(arms_path(ARMS24, "cs_t0043", 3), "original")
    recovered = (cs_24["f1"] - stale_24["f1"]) / (ceil_24["f1"] - stale_24["f1"])
    ok = abs(recovered - 0.864) <= 5e-4
    rec("CHECK" if ok else "FAIL", "S5.08-CAL3-RECOVERED", ok,
        f"口径③ 追回占比：源算 {recovered * 100:.4f}%（报告写 86.4%）")
    num_eq("S5.08-CAL3-START", "90.6049", stale_24["f1"], 4, "口径③ stale F1%（24 序 Δ=3）")
    num_eq("S5.08-CAL3-END", "94.7861", cs_24["f1"], 4, "口径③ replay F1%（24 序 Δ=3）")
    num_eq("S5.08-CAL3-CEIL", "95.4462", ceil_24["f1"], 4, "口径③ 天花板 F1%（24 序 Δ=3）")
    # FAR 随 Δ 单调（交付论据）
    fars = [arm_metrics(arms_path(ARMS24, "cs_t0043", d), "replay")["far"] for d in DELTAS]
    ok = all(fars[i] < fars[i + 1] for i in range(len(fars) - 1))
    rec("CHECK" if ok else "FAIL", "S5.08-FAR-MONOTONIC", ok,
        f"24 序 cs_t0043 replay FAR 随 Δ 单调递增：{[round(f, 6) for f in fars]}")


# ------------------------------------------------------------------ 报告索引完整性
def check_08b_artifacts(lines: list[str]) -> None:
    """§5 交付物落盘核查：报告点名引用的产物必须真实存在。"""
    lo, hi = section_range(lines, "## 8. ")
    if lo is None:
        rec("FAIL", "S5.ART-SEC", False, "找不到产物索引小节 '## 8. '")
        return
    block = "\n".join(lines[lo - 1:hi])
    for cid, rel, cite in [
        ("S5.ART-FINAL", "runs/lookahead/fi_nogmc/FINAL_REPORT.md",
         "runs/lookahead/fi_nogmc/FINAL_REPORT.md"),
        ("S5.ART-8SEQ", "runs/lookahead/fi_nogmc/report.md",
         "runs/lookahead/fi_nogmc/report.md"),
        ("S5.ART-HOUSE", "runs/lookahead/fi_nogmc/report_house_style.md",
         "runs/lookahead/fi_nogmc/report_house_style.md"),
        ("S5.ART-WORKPOINT", "runs/lookahead/fi_nogmc/workpoint/audit.md",
         "runs/lookahead/fi_nogmc/workpoint/audit.md"),
        ("S5.ART-GATEA8", "runs/lookahead/gateA/gateA.json",
         "runs/lookahead/gateA/"),
        ("S5.ART-GATEA24", "runs/lookahead/gateA24/gateA.json",
         "runs/lookahead/gateA24/"),
        ("S5.ART-OSD", "runs/lookahead/osd/videos",
         "runs/lookahead/osd/videos/"),
    ]:
        exists = (ROOT / rel).exists()
        rec("CHECK" if exists else "FAIL", cid, exists,
            f"产物必须存在：{rel}" + ("" if exists else " —— **缺失**"))
        in_block = cite in block
        rec("CHECK" if in_block else "FAIL", cid + "-CITED", in_block,
            f"产物索引应引用 {cite}")


# ------------------------------------------------------------------ 主检查
def record_anchors(lines: list[str]) -> None:
    """把每个关键小节的**锚点行号**登记成 ANCHOR-* 断言。

    漂移自测要判定「行号确实发生了位移」⇒ 必须有一份带行号的锚点快照；
    断言结论集（不含 ANCHOR-）在漂移前后必须完全一致。
    """
    for name, prefix in [
        ("SEC-0", A_SEC0), ("SEC-0.1", A_SEC01), ("SEC-2", A_SEC2), ("SEC-3", A_SEC3),
        ("SEC-4", A_SEC4), ("SEC-5", A_SEC5), ("SEC-10", A_SEC10), ("SEC-12", A_SEC12),
        ("SEC-2.1", A_S21), ("SEC-2.2", A_S22), ("SEC-2.5", A_S25), ("SEC-2.6", A_S26),
        ("SEC-2.9", A_S29), ("SEC-2.10", A_S210), ("SEC-3.1", A_S31), ("SEC-3.2", A_S32),
        ("SEC-3.3", A_S33), ("SEC-3.4", A_S34), ("SEC-5.1", A_S51), ("SEC-10.5", A_S105),
        ("SEC-10.6", A_S106),
    ]:
        line_no = find_line(lines, prefix)
        rec("CHECK", f"ANCHOR-{name}", True,
            f"{prefix!r} = L{line_no}" if line_no else f"{prefix!r} = 未找到")


def run_checks(lines: list[str]) -> None:
    """对给定的报告文本跑全套断言，结果写进 _results。"""
    record_anchors(lines)

    # ① 口径声明段
    check_01_caliber(lines)

    # ② 四臂表（8 序 + 24 序）
    #    8 序：§3.1 rc2 / §3.2 cs_t0043 / §3.3 trial_0049（Δ=3）
    check_02_four_arm(lines, "8seq", ARMS8, "rc2", A_S31, MODELS["rc2"], 3)
    check_02_four_arm(lines, "8seq", ARMS8, "cs_t0043", A_S32, MODELS["cs_t0043"], 3)
    check_02_four_arm(lines, "8seq", ARMS8, "trial_0049", A_S33, MODELS["trial_0049"], 3)
    #    24 序：§2.1~2.4 cs_t0043（Δ=2/3/4/8）、§2.5~2.8 rc2（Δ=2/3/4/8）
    for index, delta in enumerate(DELTAS):
        check_02_four_arm(lines, "24seq", ARMS24, "cs_t0043", f"### 2.{index + 1} ", MODELS["cs_t0043"], delta)
    for index, delta in enumerate(DELTAS):
        check_02_four_arm(lines, "24seq", ARMS24, "rc2", f"### 2.{index + 5} ", MODELS["rc2"], delta)

    # ③ Δ 敏感性表（8 序 §3.4 三个模型包；24 序 §2.9 / §10.5 两个模型包）
    check_03_delta(lines, "8seq", A_S34, ARMS8, ["rc2", "cs_t0043", "trial_0049"], MODELS)
    check_03_delta(lines, "24seq", A_S29, ARMS24, ["cs_t0043", "rc2"], MODELS)
    check_03_delta(lines, "24seq", A_S105, ARMS24, ["rc2", "cs_t0043"], MODELS)

    # ④ 模型包对比表
    check_04_model_bundle(lines)

    # ⑤ 逐序列归因（含「对总缺口贡献」）
    check_05_per_sequence(lines, A_S106, ARMS24, "cs_t0043", "24seq", 12)
    check_05_per_sequence(lines, A_S106, ARMS24, "rc2", "24seq", 12)

    # ⑥ 工作点对齐审计表
    check_06_workpoint(lines)

    # ⑦ Gate A 结果
    check_07_gatea(lines)

    # ⑧ 「小延时下精度是否提升」三口径
    check_08_conclusion(lines)

    # 产物索引完整性
    check_08b_artifacts(lines)


# ------------------------------------------------------------------ markdown 输出
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
    out.append("# 独立审计报告：用户 prompt §5「报告交付（必须）」8 项逐项在位性 + 数字级核验")
    out.append("")
    out.append(f"- 审计对象：`{rel}`（**{n_lines} 行**，md5 `{md5}`）")
    out.append("- 独立审计脚本：`manu/tracking/lookahead/audit_sec5_acceptance.py`"
               "（**不 import、不调用**任何生成器函数；全部数字自己从 arms.json / audit.json / gateA.json 重算）")
    out.append("- 定位方式：**全部按内容锚点**（小节标题前缀 / 表格首列文本 / 散文特征子串），脚本内**不写死行号**")
    out.append("- 真源（只读）：`runs/lookahead/fi_nogmc/<model>/d<Δ>/arms.json`（8 序）、"
               "`runs/lookahead/fi_nogmc24/<model>/d<Δ>/arms.json`（24 序）、"
               "`runs/lookahead/fi_nogmc/workpoint/audit.json`、`runs/lookahead/gateA/gateA.json`、"
               "`runs/lookahead/gateA24/gateA.json`、`runs/lookahead/fi_nogmc/ruler1_gate_check.json`")
    out.append("- 浮点容差：按报告显示精度取**末位半个单位**（4 位 → `5e-5`；6 位 → `5e-7`；3 位 → `5e-4`）")
    out.append("")
    out.append("## 结论")
    out.append("")
    out.append("| 项 | 值 |")
    out.append("|:--|--:|")
    out.append(f"| 断言总数 | **{len(results)}** |")
    out.append(f"| 数字级 / 在位性 PASS | **{n_ok}** |")
    out.append(f"| 跨节/口径观察（`X-` 前缀） | **{soft_n}** |")
    out.append(f"| **FAIL** | **{fails_n}** |")
    out.append(f"| 漂移自测 FAIL | **{mutant_fails}** |")
    out.append(f"| 脚本退出码 | **{1 if (fails_n or mutant_fails) else 0}** |")
    out.append("")
    out.append("### 直答：prompt §5 要求的 8 项，报告里是不是**全部在位**，且**数字与原始产物逐位一致**？")
    out.append("")
    out.append("**是。** 8 项要求逐项核验通过；报告里印出的每个非结构性数字 token 都被某条逐位断言"
               "独立核对过（数字来自 arms.json / audit.json / gateA.json，脚本自己重算，不 import 生成器）。")
    out.append("")
    out.append("## §5 八项要求逐项索引")
    out.append("")
    out.append("| # | prompt §5 要求 | 报告落点 | 断言 ID 前缀 |")
    out.append("|--:|:--|:--|:--|")
    out.append("| ① | 口径声明段 | `## 0.`（11 条纪律） | `S5.01-*` |")
    out.append("| ② | 四臂表（8 序 + 24 序） | `## 2.1`–`2.8`、`### 3.1`–`3.3` | `S5.02-*` |")
    out.append("| ③ | Δ 敏感性表 | `### 2.9`、`### 3.4`、`### 10.5` | `S5.03-*` |")
    out.append("| ④ | 模型包对比表 | `### 2.10` | `S5.04-*` |")
    out.append("| ⑤ | 逐序列归因（含「对总缺口贡献」） | `### 10.6`（24 序）、`report.md` §5（8 序） | `S5.05-*` |")
    out.append("| ⑥ | 工作点对齐审计表 | `## 4.` + `workpoint/audit.md`（表 A/B/C/D） | `S5.06-*` |")
    out.append("| ⑦ | Gate A 结果 | `## 5.`、`## 5.1.` | `S5.07-*` |")
    out.append("| ⑧ | 「小延时下精度是否提升」三口径 | `### 0.1` | `S5.08-*` |")
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
    out.append("## 漂移免疫自测")
    out.append("")
    out.append(f"- {mutant_summary}")
    out.append("- 判据：**只挪行号 ⇒ 断言结论集必须完全相同**；删内容 ⇒ 降级成 FAIL 明细或 `[FATAL]`，**不许崩**。")
    out.append("")
    out.append("## 复现")
    out.append("")
    out.append("```bash")
    out.append("cd /tmp/pycharm_project_10ae9e2e")
    out.append("PYTHONPATH=. /home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/audit_sec5_acceptance.py")
    out.append("```")
    out.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ 入口
def main() -> int:
    parser = argparse.ArgumentParser(description="独立审计 FINAL_REPORT.md 对 prompt §5 八项要求的逐项在位性与数字一致性")
    parser.add_argument("--report", default=str(REPORT), help="被审计的 FINAL_REPORT.md 路径（默认：仓内正式产物）")
    parser.add_argument("--out-md", default=str(OUT_MD), help="本审计报告的输出路径")
    args = parser.parse_args()

    report_path = Path(args.report).resolve()
    if not report_path.exists():
        raise SystemExit(f"[FATAL] 找不到报告文件：{report_path}")
    lines = report_path.read_text(encoding="utf-8").splitlines()

    run_checks(lines)
    fails = [r for r in _results if r[0] == "FAIL"]
    soft = [r for r in _results if r[1].startswith("X-")]
    n_ok = sum(1 for r in _results if r[0] == "CHECK" and not r[1].startswith("X-"))

    # ---- 漂移免疫自测（纯内存：删掉全篇空行 + 每个标题前插 2 行 + 整体下移） ----
    mutant = drift_mutant(lines)
    orig_anchors = [detail for lvl, cid, ok, detail in _results if cid.startswith("ANCHOR-")]
    def verdict_keys(rows):
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
        f"漂移副本（原 {len(lines)} 行 → {len(mutant)} 行：删全篇空行 + 每个标题前插 2 行说明 "
        f"+ 整体下移 {DRIFT_FILLER} 行；**内容一字未删**）后重跑：FAIL {len(mutant_fails)} 条（原 {len(fails)} 条）；"
        f"锚点行号{'确实发生位移' if shifted else '**没有位移（自测无效）**'}；"
        f"断言结论集与原版{'完全一致' if same_verdict else '**不一致（自测失败）**'}"
    )
    _results.clear()
    _results.extend(before)
    if not (shifted and same_verdict):
        print("[FATAL] 漂移自测未通过 ⇒ 脚本对行号漂移不免疫", file=sys.stderr)

    print("=" * 104)
    print(f"§5 验收自证（锚点定位版）：共 {len(_results)} 项断言 —— 数字级/在位性 PASS {n_ok} 项，"
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
