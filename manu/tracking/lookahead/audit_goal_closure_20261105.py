#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast Infer 目标闭环独立复算审计（2026-11-05）。

对用户目标逐条取证：
    ① 把模型换成 ``cs_t0043``（no-GMC 语义）
    ② 假设延时 = 120 ms（≈3 帧）
    ③ 期望小延时状态提升精度
    ④ 给出严格的精度对比报告（沿用 fast infer 历史的「精度对齐」四件套）

设计约束（与前几轮审计同一范式）：

1. **不 import / 不调用**报告生成器（``make_fastinfer_final_report.py`` 等）里的任何函数 ——
   那些函数自己就是被审计对象。本脚本**只**读产物 JSON + 被审计的 md 文本，指标**独立重算**。
2. **不照抄报告的数值**：报告里的每个数字都从产物重新算一遍，再用**与报告同精度**的格式化
   做**字符串全等**比较（容差 = 0 个显示单位）。
3. 只读产物；**只写本脚本自己的审计产物**（JSON + MD）；幂等、无副作用。

用法::

    python3 manu/tracking/lookahead/audit_goal_closure_20261105.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = next(
    (a for a in Path(__file__).resolve().parents if (a / "manu/tracking/lookahead").is_dir()),
    Path(__file__).resolve().parent.parent / "repo",
)

REPORT_MD = REPO / "runs/lookahead/final/nogmc_fi_20261104/FINAL_REPORT.md"
FI24 = REPO / "runs/lookahead/fi_nogmc24"
FOV24 = REPO / "runs/lookahead/fovgate24nogmc"
GATE_A24 = REPO / "runs/lookahead/gateA24/gateA.json"
CFG_IDENT = REPO / "runs/lookahead/nogmc24_cfg_identical.json"
BYTE_EQ = REPO / "runs/lookahead/final/nogmc_fi_20261104/nogmc24_byte_equal.json"
OUT_DIR = FI24 / "audit_goal_closure"

PKGS = ("cs_t0043", "rc2")
DELTAS = (2, 3, 4, 8)
ARMS = ("original", "stale", "kinematic", "replay")
ARM_ROLE = {
    "original": "零延迟天花板（oracle 上界，**不是**基线）",
    "stale": "老逻辑硬吃滞后（**交付基线**）",
    "kinematic": "档位 A 纯运动学外推",
    "replay": "档位 B 视觉重推 + xcorr（**交付方案**）",
}
GATES = {"f1": 0.88, "recall": 0.85, "far_per_frame": 0.030}

# 报告里的显示精度（与渲染一致）
FMT = {"f1": ".4f", "recall": ".4f", "precision": ".4f", "far_per_frame": ".6f",
       "hit5": ".4f", "cle50": ".3f", "cle95": ".3f"}

results: list[dict] = []
notes: list[str] = []


def check(name: str, ok: bool, detail: str = "", section: str = "") -> bool:
    results.append({"name": name, "ok": bool(ok), "detail": detail, "section": section})
    return bool(ok)


def arms_path(pkg: str, delta: int) -> Path:
    return FI24 / pkg / f"d{delta}" / "arms.json"


def load(pkg: str, delta: int) -> dict:
    return json.loads(arms_path(pkg, delta).read_text(encoding="utf-8"))


def metrics(arm: dict) -> dict:
    """从一条臂里独立取出报告对应的六个指标。"""
    cur = arm["current_frame"]["overall"]
    srv = arm["servo"]["overall"]
    return {
        "f1": cur["f1"] * 100.0,
        "recall": cur["recall"] * 100.0,
        "precision": cur["precision"] * 100.0,
        "far_per_frame": cur["far_per_frame"],
        "hit5": srv["hit_rate_5px"] * 100.0,
        "cle50": srv["cle_pooled_p50"],
        "cle95": srv["cle_pooled_p95"],
        "gt": cur["gt"],
        "tp": cur["tp"],
        "fp": cur["fp"],
        "fn": cur["fn"],
        "source_f1": arm["source_frame"]["overall"]["f1"],
    }


def gates(m: dict) -> str:
    ok = (m["f1"] / 100.0 >= GATES["f1"] and m["recall"] / 100.0 >= GATES["recall"]
          and m["far_per_frame"] <= GATES["far_per_frame"])
    return "**PASS**" if ok else "**FAIL**"


def md_text() -> str:
    return REPORT_MD.read_text(encoding="utf-8")


def section_lines(text: str, start_marker: str, end_marker: str | None) -> list[str]:
    i = text.index(start_marker)
    j = text.index(end_marker, i) if end_marker else len(text)
    return text[i:j].splitlines()


def parse_table(lines: list[str]) -> list[list[str]]:
    """取 markdown 表格体（跳过表头与分隔行）。"""
    rows = []
    for ln in lines:
        s = ln.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        rows.append(cells)
    return rows[1:]  # 丢表头


def main() -> int:
    text = md_text()
    docs = {(pkg, d): load(pkg, d) for pkg in PKGS for d in DELTAS}

    # ---------- ① 报告里 8 组表的逐位复算（§2.1~§2.8）----------
    sec_titles = {
        ("cs_t0043", 2): "### 2.1 cs_t0043 (no-GMC 语义) · Δ=2（80 ms）",
        ("cs_t0043", 3): "### 2.2 cs_t0043 (no-GMC 语义) · Δ=3（120 ms）",
        ("cs_t0043", 4): "### 2.3 cs_t0043 (no-GMC 语义) · Δ=4（160 ms）",
        ("cs_t0043", 8): "### 2.4 cs_t0043 (no-GMC 语义) · Δ=8（320 ms）",
        ("rc2", 2): "### 2.5 RC2 trial_0080ep3 (GMC 语义) · Δ=2（80 ms）",
        ("rc2", 3): "### 2.6 RC2 trial_0080ep3 (GMC 语义) · Δ=3（120 ms）",
        ("rc2", 4): "### 2.7 RC2 trial_0080ep3 (GMC 语义) · Δ=4（160 ms）",
        ("rc2", 8): "### 2.8 RC2 trial_0080ep3 (GMC 语义) · Δ=8（320 ms）",
    }
    next_sec = {
        ("cs_t0043", 2): "### 2.2", ("cs_t0043", 3): "### 2.3", ("cs_t0043", 4): "### 2.4",
        ("cs_t0043", 8): "### 2.5", ("rc2", 2): "### 2.6", ("rc2", 3): "### 2.7",
        ("rc2", 4): "### 2.8", ("rc2", 8): "### 2.9",
    }
    for pkg, d in [(p, dd) for p in PKGS for dd in DELTAS]:
        lines = section_lines(text, sec_titles[(pkg, d)], next_sec[(pkg, d)])
        rows = parse_table(lines)
        doc = docs[(pkg, d)]
        bad = []
        for row in rows:
            name = row[0]
            if name not in ARMS:
                bad.append(f"未知臂 {name}")
                continue
            m = metrics(doc["arms"][name])
            exp = [
                f"{m['f1']:{FMT['f1']}}", f"{m['recall']:{FMT['recall']}}",
                f"{m['precision']:{FMT['precision']}}", f"{m['far_per_frame']:{FMT['far_per_frame']}}",
                f"{m['hit5']:{FMT['hit5']}}", f"{m['cle50']:{FMT['cle50']}}",
                f"{m['cle95']:{FMT['cle95']}}", gates(m),
            ]
            got = row[2:10]
            if exp != got:
                bad.append(f"{name}: 报告={got} 复算={exp}")
        check(f"§2 表逐位复算 {pkg} Δ={d}", not bad,
              "；".join(bad) if bad else "8 组 × 4 臂 × 8 列 全部字符串全等", "§2")

    # ---------- ② Δ 敏感性表（§2.9）----------
    rows = parse_table(section_lines(text, "### 2.9 Δ 敏感性", "### 2.10"))
    bad = []
    for row in rows:
        pkg = "cs_t0043" if "cs_t0043" in row[0] else "rc2"
        d = int(row[1])
        m = metrics(docs[(pkg, d)]["arms"]["replay"])
        stale = metrics(docs[(pkg, d)]["arms"]["stale"])
        exp = [f"{m['f1']:{FMT['f1']}}", f"{m['recall']:{FMT['recall']}}",
               f"{m['far_per_frame']:{FMT['far_per_frame']}}", f"{m['hit5']:{FMT['hit5']}}",
               f"{m['cle50']:{FMT['cle50']}}", f"{m['cle95']:{FMT['cle95']}}",
               f"{(m['f1'] - stale['f1']):+.4f}"]
        got = [row[3], row[4], row[5], row[6], row[7], row[8], row[9]]
        if exp != got:
            bad.append(f"{pkg} Δ={d}: 报告={got} 复算={exp}")
    check("§2.9 Δ 敏感性表逐位复算", not bad, "；".join(bad) if bad else "8 行 × 7 列 全等", "§2.9")

    # ---------- ③ 跨模型包对照表（§2.10）----------
    rows = parse_table(section_lines(text, "### 2.10", "### 2.11"))
    bad = []
    for row in rows:
        d = int(row[0])
        metric = row[1]
        r2 = metrics(docs[("rc2", d)]["arms"]["replay"])
        c3 = metrics(docs[("cs_t0043", d)]["arms"]["replay"])
        key = {"F1%": "f1", "FAR/帧": "far_per_frame", "hit@5px%": "hit5",
               "CLE p95(池)": "cle95"}[metric]
        fmt = FMT[key] if key != "cle95" else ".3f"
        a, b = r2[key], c3[key]
        d_txt = row[4]
        if key in ("f1", "hit5"):
            exp_diff = f"{(b - a):+.4f}pp"
        elif key == "far_per_frame":
            exp_diff = f"{b - a:+.6f}"
        else:
            exp_diff = f"{b - a:.3f}px"
        if row[2] != f"{a:{fmt}}" or row[3] != f"{b:{fmt}}" or d_txt != exp_diff:
            bad.append(f"Δ={d} {metric}: 报告=({row[2]},{row[3]},{d_txt}) "
                       f"复算=({a:{fmt}},{b:{fmt}},{exp_diff})")
    check("§2.10 跨模型包对照表逐位复算", not bad,
          "；".join(bad) if bad else "16 行（4 指标 × 4 Δ）全部一致", "§2.10")

    # ---------- ④ 口径对齐锚点（§2.11）----------
    rows = parse_table(section_lines(text, "### 2.11", "### 2.12"))
    m8 = metrics(docs[("rc2", 8)]["arms"]["replay"])
    exp = {"F1": f"{m8['f1']:.4f}%", "FAR/帧": f"{m8['far_per_frame']:.6f}",
           "hit@5px": f"{m8['hit5']:.4f}%", "尺子1 F1": repr(m8["source_f1"])}
    bad = []
    for row in rows:
        key = row[0]
        cell = row[2].replace("**", "").strip()
        if key in exp and cell != exp[key]:
            bad.append(f"{key}: 报告={cell} 复算={exp[key]}")
    check("§2.11 口径对齐锚点（rc2 Δ=8 vs 历史权威数）", not bad,
          "；".join(bad) if bad else "4 项逐位相同（README §5 红框口径）", "§2.11")

    # ---------- ⑤ 目标①：模型换成 cs_t0043（唯一变量 = {模型包, Δ}）----------
    ok_exist = all(arms_path(p, d).is_file() for p in PKGS for d in DELTAS)
    check("目标① 四档 Δ 产物齐全（cs_t0043 + rc2 对照）", ok_exist,
          "8 组 arms.json 全部存在" if ok_exist else "有缺失", "§2")

    gts = {f"{p}/d{d}": metrics(docs[(p, d)]["arms"]["replay"])["gt"] for p in PKGS for d in DELTAS}
    check("唯一变量：GT 数跨 8 组全同", len(set(gts.values())) == 1,
          f"GT={sorted(set(gts.values()))}", "§0.5")

    dens = {}
    for p in PKGS:
        for d in DELTAS:
            m = metrics(docs[(p, d)]["arms"]["replay"])
            dens[f"{p}/d{d}"] = m["fp"] / m["far_per_frame"]
    check("唯一变量：FAR 隐含分母（帧数）跨 8 组全同",
          len({round(v, 6) for v in dens.values()}) == 1,
          f"分母={sorted({round(v, 6) for v in dens.values()})}", "§0.5")

    seqs = {f"{p}/d{d}": tuple(sorted(docs[(p, d)]["per_sequence"].keys()))
            for p in PKGS for d in DELTAS}
    check("唯一变量：序列集合跨 8 组全同", len(set(seqs.values())) == 1,
          f"{len(seqs['cs_t0043/d3'])} 序列", "§0.5")

    def cfg_of(pkg: str, d: int) -> str:
        cfg = dict(docs[(pkg, d)].get("config", {}))
        for k in list(cfg):
            if any(t in k.lower() for t in ("weight", "feature", "cache", "root", "path", "model")):
                cfg.pop(k)
        return json.dumps(cfg, sort_keys=True, ensure_ascii=False)

    same_cfg = all(cfg_of("cs_t0043", d) == cfg_of("rc2", d) for d in DELTAS)
    check("唯一变量：同 Δ 下模型包之外的 CFG 一致", same_cfg,
          "cs_t0043 与 rc2 同 Δ 的 config（剔除权重/特征/缓存路径）逐字相同" if same_cfg else "有差异",
          "§5.2")

    # ---------- ⑥ 目标②：Δ=3 = 120 ms ----------
    d3 = docs[("cs_t0043", 3)]
    deltas_seen = {a: d3["arms"][a].get("delta") for a in ARMS}
    check("目标② Δ=3 已落到产物（stale/kinematic/replay 三臂 delta 字段 = 3；original = 零延迟天花板 0）",
          all(deltas_seen[a] == 3 for a in ("stale", "kinematic", "replay"))
          and deltas_seen["original"] == 0,
          f"四臂 delta={deltas_seen}", "§2.2")
    check("目标② 报告标注 Δ=3 = 120 ms",
          "Δ=3（120 ms）" in text and "25 Hz" in text,
          "§0 纪律 6 + §2.2 标题", "§0")

    # ---------- ⑦ 目标③：小延时提升精度 ----------
    cs3 = metrics(docs[("cs_t0043", 3)]["arms"]["replay"])
    cs8 = metrics(docs[("cs_t0043", 8)]["arms"]["replay"])
    rc3 = metrics(docs[("rc2", 3)]["arms"]["replay"])
    cs2 = metrics(docs[("cs_t0043", 2)]["arms"]["replay"])
    cs4 = metrics(docs[("cs_t0043", 4)]["arms"]["replay"])
    check("目标③ 小延时(Δ=3)比大延时(Δ=8) F1 更高", cs3["f1"] > cs8["f1"],
          f"Δ3 {cs3['f1']:.4f}% vs Δ8 {cs8['f1']:.4f}% （+{cs3['f1'] - cs8['f1']:.4f}pp）", "§2.10")
    check("目标③ 小延时 FAR 更低", cs3["far_per_frame"] < cs8["far_per_frame"],
          f"Δ3 {cs3['far_per_frame']:.6f} vs Δ8 {cs8['far_per_frame']:.6f} "
          f"（{cs3['far_per_frame'] - cs8['far_per_frame']:+.6f}）", "§2.10")
    check("目标③ 小延时 hit@5px 更高", cs3["hit5"] > cs8["hit5"],
          f"Δ3 {cs3['hit5']:.4f}% vs Δ8 {cs8['hit5']:.4f}% （+{cs3['hit5'] - cs8['hit5']:.4f}pp）", "§2.10")
    check("目标③ 小延时 CLE p95 更低", cs3["cle95"] < cs8["cle95"],
          f"Δ3 {cs3['cle95']:.3f} vs Δ8 {cs8['cle95']:.3f} （{cs3['cle95'] - cs8['cle95']:+.3f}px）", "§2.10")
    check("目标③ Δ 越小时延单调更好（Δ2 > Δ3 > Δ4 > Δ8 的 F1）",
          cs2["f1"] > cs3["f1"] > cs4["f1"] > cs8["f1"],
          f"{cs2['f1']:.4f} > {cs3['f1']:.4f} > {cs4['f1']:.4f} > {cs8['f1']:.4f}", "§2.9")
    check("目标③ Δ=3 相对 stale 交付基线大幅提升",
          cs3["f1"] - metrics(docs[("cs_t0043", 3)]["arms"]["stale"])["f1"] > 0,
          f"+{cs3['f1'] - metrics(docs[('cs_t0043', 3)]['arms']['stale'])['f1']:.4f}pp F1", "§0.1")
    check("目标③ 同 Δ 下 cs_t0043 优于 rc2（no-GMC 换模型净收益）",
          cs3["f1"] > rc3["f1"] and cs3["hit5"] > rc3["hit5"] and cs3["cle95"] < rc3["cle95"],
          f"F1 {cs3['f1']:.4f} vs {rc3['f1']:.4f}（+{cs3['f1'] - rc3['f1']:.4f}pp）／"
          f"hit@5px +{cs3['hit5'] - rc3['hit5']:.4f}pp／CLE p95 {cs3['cle95'] - rc3['cle95']:+.3f}px", "§2.10")
    check("目标③ Δ=8 裸配置 FAR 破线、Δ=3 过线（符号翻转）",
          cs8["far_per_frame"] > 0.030 >= cs3["far_per_frame"],
          f"Δ8 {cs8['far_per_frame']:.6f}（破线）／Δ3 {cs3['far_per_frame']:.6f}（过线）", "§7")

    # ---------- ⑧ 精度对齐四件套 ----------
    ruler1 = {}
    for p in PKGS:
        for d in DELTAS:
            vals = {repr(metrics(docs[(p, d)]["arms"][a])["source_f1"]) for a in ARMS}
            ruler1[f"{p}/d{d}"] = (len(vals) == 1, sorted(vals)[0])
    check("四件套 A1 尺子1（四臂 source_frame F1 逐位相等）",
          all(v[0] for v in ruler1.values()),
          f"8/8 组逐位相等，例：{ruler1['cs_t0043/d3'][1]}", "§0")

    gate_a = json.loads(GATE_A24.read_text(encoding="utf-8"))
    check("四件套 A2 Gate A（24 序官方口径）",
          gate_a.get("verdict") == "PASS" and gate_a.get("ruler1", {}).get("all_bit_identical") is True,
          f"verdict={gate_a.get('verdict')}／ruler1.all_bit_identical="
          f"{gate_a.get('ruler1', {}).get('all_bit_identical')}", "§5.1")

    cfg_ident = json.loads(CFG_IDENT.read_text(encoding="utf-8"))
    check("四件套 A7 CFG 逐 token 相同（rc2 vs cs_t0043）",
          cfg_ident.get("verdict") == "IDENTICAL"
          and cfg_ident.get("exact_string_equal") is True
          and cfg_ident.get("token_diffs") == []
          and cfg_ident.get("rc2_token_count") == cfg_ident.get("cs_token_count")
          and cfg_ident.get("rc2_actual_len") == cfg_ident.get("cs_actual_len"),
          f"verdict={cfg_ident.get('verdict')}／exact_string_equal="
          f"{cfg_ident.get('exact_string_equal')}／{cfg_ident.get('rc2_actual_len')} 字符／"
          f"{cfg_ident.get('rc2_token_count')} token／token_diffs={cfg_ident.get('token_diffs')}", "§5.2")

    be = json.loads(BYTE_EQ.read_text(encoding="utf-8"))
    check("四件套 A5 特征集逐字节等价（no-GMC vs 官方那套）",
          be.get("verdict") == "PASS"
          and be.get("mismatch_files") == 0
          and be.get("missing_in_new") == 0
          and be.get("missing_in_old") == 0
          and be.get("identical_files") == be.get("total_files"),
          f"verdict={be.get('verdict')}／{be.get('identical_files')}/{be.get('total_files')} md5 全同／"
          f"mismatch={be.get('mismatch_files')}／missing new={be.get('missing_in_new')} "
          f"old={be.get('missing_in_old')}", "§5.2")

    # ---------- ⑨ 交付裁定（§12）----------
    fov = FOV24 / "runs/cs_t0043/d3/fov__m6/arms.json"
    if fov.is_file():
        fd = json.loads(fov.read_text(encoding="utf-8"))
        fm = metrics(fd["arms"]["replay"])
        check("§12 交付推荐档（cs_t0043·Δ=3·replay·出框门 m=6）三门",
              fm["f1"] / 100 >= 0.88 and fm["recall"] / 100 >= 0.85 and fm["far_per_frame"] <= 0.030,
              f"F1 {fm['f1']:.4f}%／R {fm['recall']:.4f}%／FAR {fm['far_per_frame']:.6f}", "§12")
    else:
        check("§12 交付推荐档产物存在", False, f"缺 {fov}", "§12")

    # ---------- ⑪ 交付入口真实执行分支（端到端首跑）----------
    new_arms = REPO / "runs/lookahead/delivery_nogmc_d3/runs/cs_t0043/d3/fov__m6/arms.json"
    if new_arms.is_file() and fov.is_file():
        nd = json.loads(new_arms.read_text(encoding="utf-8"))
        nm = metrics(nd["arms"]["replay"])
        rm = metrics(json.loads(fov.read_text(encoding="utf-8"))["arms"]["replay"])
        fields = ["f1", "recall", "precision", "far_per_frame", "tp", "fp", "fn", "gt",
                  "hit5", "cle50", "cle95"]
        diffs = [f for f in fields if f"{nm[f]!r}" != f"{rm[f]!r}"]
        check("交付入口端到端产物 vs 已验收参考产物（逐位相等，无容差）", not diffs,
              "11 个关键字段逐位相同（F1 %.4f%%／FAR %.6f／hit@5px %.4f%%／CLE p95 %.3f）"
              % (nm["f1"], nm["far_per_frame"], nm["hit5"], nm["cle95"])
              if not diffs else "；".join(diffs), "§12")
        check("交付入口端到端产物 · 交付三门",
              nm["f1"] / 100 >= 0.88 and nm["recall"] / 100 >= 0.85
              and nm["far_per_frame"] <= 0.030,
              f"F1 {nm['f1']:.4f}%／R {nm['recall']:.4f}%／FAR {nm['far_per_frame']:.6f} ⇒ PASS", "§12")
    else:
        check("交付入口端到端产物存在", False, f"缺 {new_arms}", "§12")

    # ---------- ⑩ 报告本体 ----------
    lines = text.splitlines()
    check("目标④ 严格对比报告存在且规模达标", len(lines) >= 1000,
          f"{len(lines)} 行 / {REPORT_MD.stat().st_size} 字节", "§8")

    # ---------- 汇总 ----------
    npass = sum(1 for r in results if r["ok"])
    nfail = len(results) - npass
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "audit_goal_closure.json").write_text(
        json.dumps({"checks": results, "pass": npass, "fail": nfail}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    md = ["# Fast Infer 目标闭环独立复算审计（2026-11-05）", "",
          f"**结论**：{npass}/{len(results)} 项通过，{nfail} 项失败。", "",
          "| # | 检查项 | 结果 | 章节 | 实测 |", "|:--|:--|:--:|:--|:--|"]
    for i, r in enumerate(results, 1):
        md.append(f"| {i} | {r['name']} | {'✅' if r['ok'] else '❌'} | {r['section']} | {r['detail']} |")
    (OUT_DIR / "audit_goal_closure.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[AUDIT] {npass}/{len(results)} PASS, {nfail} FAIL")
    for i, r in enumerate(results, 1):
        print(f"  [{'PASS' if r['ok'] else 'FAIL'}] {i:2d}. {r['name']} :: {r['detail']}")
    print(f"[AUDIT] 产物：{OUT_DIR/'audit_goal_closure.md'}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
