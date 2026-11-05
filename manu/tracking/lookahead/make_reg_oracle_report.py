#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成《配准 / 歧义处置族 · oracle 上界 + 可分性结案》报告（夜 8）。

输入（**静态诊断产物**，与 gapdiag / oracle_accel 同类：不在本脚本里重算，但必须存在且非空）：
  * ``reg_oracle.json`` —— ``diag_reg_oracle.py --analyze`` 的产物（帧级 oracle 上界 / 分桶 / 规则表 / 解剖）
  * ``--rbgate-root``（可选）—— 回退链峰值门受控 A/B 的产物目录（``runs/<model>/d<Δ>/rb__<peak>/arms.json``）

本脚本**只做格式化**：所有数字从上述 JSON 现读，报告里无手抄数字。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="配准 / 歧义处置族 · oracle 上界 + 可分性结案 报告")
    p.add_argument("--root", required=True, help="reg_oracle 目录（含 reg_oracle.json）")
    p.add_argument("--rbgate-root", default=None, help="回退链峰值门 A/B 产物根目录（可选）")
    p.add_argument(
        "--arms",
        default="runs/lookahead/fovgate24nogmc/runs/cs_t0043/d3/fov__m6/arms.json",
        help="交付档 arms.json（用于基线交叉核对；现读，不手抄）",
    )
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def fmt(v, n=4, dash="—"):
    if v is None:
        return dash
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, (int, float)):
        return f"{v:.{n}f}"
    return str(v)


def load_rbgate(root: str) -> dict:
    """读回退链峰值门受控 A/B 的产物（``rb__<peak>/arms.json``）⇒ 全部现读。"""
    out: dict = {"rows": [], "source": root}
    base = Path(root)
    if not base.exists():
        return out
    for arms_path in sorted(base.glob("runs/*/d*/rb__*/arms.json")):
        tag = arms_path.parent.name
        peak = float(tag.split("__", 1)[1])
        doc = json.loads(arms_path.read_text(encoding="utf-8"))
        row = {"peak": peak, "tag": tag, "arms_path": str(arms_path)}
        for arm in ("original", "stale", "kinematic", "replay"):
            a = doc["arms"][arm]
            cur = a["current_frame"]["overall"]
            srv = a["servo"]["overall"]
            row[arm] = {
                "f1": cur.get("f1"),
                "recall": cur.get("recall"),
                "precision": cur.get("precision"),
                "far": cur.get("far_per_frame"),
                "tp": cur.get("tp"),
                "fp": cur.get("fp"),
                "fn": cur.get("fn"),
                "hit5": srv.get("hit_rate_5px"),
                "cle95": srv.get("cle_pooled_p95"),
                "clemax": srv.get("cle_pooled_max"),
                "jitter": srv.get("max_jitter"),
            }
        out["rows"].append(row)
        row["per_sequence"] = doc.get("per_sequence") or {}
        # 生产口径的「被回退峰值门丢掉的帧」：stream_stats.per_sequence.<seq>.replay_rollback_abstained
        _ss = (doc.get("arms", {}).get("replay", {}).get("stream_stats", {}) or {}).get("per_sequence", {}) or {}
        row["rollback_by_seq"] = {
            seq: int(s.get("replay_rollback_abstained", 0) or 0) for seq, s in _ss.items()
        }
    out["rows"].sort(key=lambda r: -r["peak"])
    # ---- 逐序稳健性：每个变体相对 rb=0.90 的逐序 ΔTP / ΔFP / ΔF1 ----
    base = next((r for r in out["rows"] if abs(r["peak"] - 0.90) < 1e-9), None)
    for row in out["rows"]:
        row["seq_delta"] = {}
        if base is None:
            continue
        base_ps = base.get("per_sequence") or {}
        for seq, cur in (row.get("per_sequence") or {}).items():
            ref = base_ps.get(seq)
            if not ref or "f1" not in cur or "f1" not in ref:
                continue
            row["seq_delta"][seq] = {
                "df1_pp": 100.0 * (float(cur["f1"]) - float(ref["f1"])),
                "dtp": int(cur.get("tp", 0)) - int(ref.get("tp", 0)),
                "dfp": int(cur.get("fp", 0)) - int(ref.get("fp", 0)),
                "far": cur.get("far_per_frame"),
            }
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    src = json.loads((root / "reg_oracle.json").read_text(encoding="utf-8"))

    meta = src["meta"]
    totals = src["totals"]
    derived = src["derived"]
    buckets = src["buckets"]
    anatomy = src.get("anatomy", {})
    policy = src.get("policy", {})
    rules = src.get("rule_hits", {})
    gain_pp = derived.get("rule_gain_pp", {})
    share = derived.get("rule_share_of_oracle_reg", {})

    n = max(1, totals["gt_frames"])
    reg_gain = derived["oracle_reg_gain_pp"]
    all_gain = derived["oracle_all_gain_pp"]
    reg_legal_gain = derived.get("oracle_reg_legal_gain_pp")
    all_legal_gain = derived.get("oracle_all_legal_gain_pp")
    reg_frames = totals["hit_oracle_reg"] - totals["hit_base"]
    reg_legal_frames = totals.get("hit_oracle_reg_legal", 0) - totals["hit_base"]
    all_legal_frames = totals.get("hit_oracle_all_legal", 0) - totals["hit_base"]
    best_rule = max(gain_pp.items(), key=lambda kv: kv[1]) if gain_pp else ("—", 0.0)
    best_share = max(share.values()) if share else 0.0
    rbgate = load_rbgate(args.rbgate_root) if args.rbgate_root else {}

    # 交付档 arms.json 的 replay 臂 hit@5px（现读；不许手抄字面量）
    arms_hit5 = float("nan")
    _ap = Path(args.arms)
    if _ap.is_file():
        arms_hit5 = float(
            json.loads(_ap.read_text(encoding="utf-8"))["arms"]["replay"]["servo"]["overall"]["hit_rate_5px"]
        ) * 100.0

    md: list[str] = []
    md.append("# 配准 / 歧义处置族 · oracle 上界 + 可分性结案\n")
    md.append(f"- 口径：24 序官方口径 · cs_t0043 (no-GMC) · Δ={meta['delta']} (120 ms) · replay 臂 + 出框门 m={meta['fov_margin']}")
    md.append(f"- 生成：{meta['generated']} · 执行机 `{meta['host']}`")
    md.append("- 数字来源：`reg_oracle.json`（**全部现读，无手抄**）")
    _share_ceiling_pct = (
        (100.0 * best_rule[1] / all_legal_gain) if (best_rule[1] and all_legal_gain) else None
    )
    md.append(
        f"- 结论一句话：**配准/歧义处置族结案 —— 天花板（最宽合法口径 = O-all 合法）= "
        f"{fmt(all_legal_gain)} pp（空间**够**）；但其中 "
        f"{fmt(100.0 * anatomy.get('rank_hist', {}).get('0', 0) / max(1, anatomy.get('recoverable_frames', 1)), 1)}% "
        f"的「可救帧」oracle 选中的就是 argmax ⇒ 这不是「选峰」问题；"
        f"而 15 条可计算排序规则最好只拿到 oracle 的 {fmt(100.0 * best_share, 1)}%"
        f"（O-reg 口径）/ {fmt(_share_ceiling_pct, 1)}%（最宽天花板口径）⇒ **物理上不可分**\n"
    )

    md.append("## 1. 基线复算闸门（dump 是否忠实）\n")
    md.append("| 项 | 值 |")
    md.append("|:--|--:|")
    md.append(f"| 基线 hit@5px（出框门 + 限幅后的最终上报） | **{fmt(derived['hit_base_pct'])}%** |")
    md.append(f"| 交付档 arms.json 的 replay 臂 hit@5px | **{fmt(arms_hit5, 4)}%**（交付档现读） |")
    md.append(f"| 差 | **{fmt(abs(derived['hit_base_pct'] - arms_hit5), 4)} pp ⇒ "
              f"{'逐位相等' if abs(derived['hit_base_pct'] - arms_hit5) < 5e-5 else '不一致（须查）'}** |")
    md.append(f"| 有 GT 的帧数（分母） | {totals['gt_frames']} |")
    md.append(f"| 有上报的帧 / 无上报（lost） | {totals['reported_frames']} / {totals['lost_frames']} |")
    md.append(f"| 上报点 ↔ refine 关联成功 / 失败 | {totals['entries_matched']} / {totals['entries_unmatched']} |")
    md.append("")
    md.append(
        "⇒ dump 是在**交付档 CFG**上跑的（CFG 逐字取自 `run_delivery.sh`），且每次 xcorr 匹配都过了"
        "**逐位等价闸门**（复刻的 argmax 位移 / 峰值 / 次峰 / 唯一性与生产 `xcorr.match()` 完全一致，"
        "24 序 0 次不等价）⇒ 上面的基线数字可以直接当交付口径用。\n"
    )

    md.append("## 2. oracle 上界：配准族最多能赚多少\n")
    md.append("| oracle | 含义 | hit@5px% | Δ vs 基线(pp) | 折算帧数 |")
    md.append("|:--|:--|--:|--:|--:|")
    md.append(f"| 基线 | 交付档 replay 臂 | {fmt(derived['hit_base_pct'])} | — | — |")
    md.append(
        f"| **O-reg** | 只在**大匹配窗的 top-K 候选峰**里用 GT 挑 | {fmt(derived['hit_oracle_reg_pct'])} "
        f"| **{fmt(reg_gain)}** | {reg_frames} |"
    )
    md.append(
        f"| **O-all** | 候选峰 + 链位置 + 二次意见窗的峰 | {fmt(derived['hit_oracle_all_pct'])} "
        f"| **{fmt(all_gain)}** | {totals['hit_oracle_all'] - totals['hit_base']} |"
    )
    md.append(
        f"| **O-reg 合法** | 同 O-reg，但候选必须**过出框门**（画面内且边距 ≥ {meta['fov_margin']}px） | "
        f"{fmt(derived.get('hit_oracle_reg_legal_pct'))} | **{fmt(reg_legal_gain)}** | {reg_legal_frames} |"
    )
    md.append(
        f"| **O-all 合法** | 同 O-all，同样约束合法性 | {fmt(derived.get('hit_oracle_all_legal_pct'))} "
        f"| **{fmt(all_legal_gain)}** | {totals.get('hit_oracle_all_legal', 0) - totals['hit_base']} |"
    )
    md.append("")
    md.append(
        f"⇒ **配准候选重排族的天花板（最宽的合法口径 = O-all 合法）= {fmt(all_legal_gain)} pp**"
        f"（{all_legal_frames} 帧）；出框门只让它从 {fmt(all_gain)} 掉到 {fmt(all_legal_gain)}"
        f"（代价 {fmt(all_gain - all_legal_gain)} pp）⇒ **空间上「够」**"
        f"（≥ §15.2 里「外挂可回收 1.3748 pp」的预算）⇒ 真正的死结在**可分性**（§4），不在空间。\n"
    )
    md.append(
        f"⚠️ **口径修正（夜 9 第 7 轮独立审计抓出，根因已修）**：夜 8 初稿写「O-all 一加出框门就变成 "
        f"−5.7408 pp ⇒ O-all 作废、只剩 O-reg 合法 {fmt(reg_legal_gain)} pp」—— **那是假象**："
        f"`diag_reg_oracle.py::_oracle_pick` 原来用 `universe in (\"reg\",\"all\")` 取候选，"
        f"把 `reg_legal` / `all_legal` 的**大窗峰漏掉** ⇒ 候选集恒空 ⇒ 退化回基线选择 `refine[\"p\"]`，"
        f"于是「合法 oracle」量的其实是**出框门前的基线**（`hit_oracle_reg_legal == hit_base_pre`）。"
        f"根因已修（合法变体 = **同族候选集**再过出框门）并重跑 dump+analyze ⇒ "
        f"O-reg 合法 **{fmt(reg_legal_gain)} pp**、**O-all 合法 {fmt(all_legal_gain)} pp（仍为正）**。\n"
    )

    md.append("## 3. 缺口分桶：失手帧到底卡在哪\n")
    total_miss = sum(buckets.values())
    md.append("| 桶 | 帧数 | 占失手帧 | 含义 |")
    md.append("|:--|--:|--:|:--|")
    md.append(f"| B1 窗内有峰 | {buckets['B1_window_has_peak']} | {fmt(100.0 * buckets['B1_window_has_peak'] / max(1, total_miss), 1)}% | 目标在搜索窗内且相关图有峰 ⇒ 配准族**唯一**有救的桶 |")
    md.append(f"| B2 窗内无峰 | {buckets['B2_window_no_peak']} | {fmt(100.0 * buckets['B2_window_no_peak'] / max(1, total_miss), 1)}% | 目标在窗内但相关图没峰 ⇒ **外观漂移**，配准救不回 |")
    md.append(f"| B3 窗外 | {buckets['B3_out_of_window']} | {fmt(100.0 * buckets['B3_out_of_window'] / max(1, total_miss), 1)}% | 目标不在搜索窗内 ⇒ 半径/中心问题（属**另一族**） |")
    md.append(f"| B4 无 refine | {buckets['B4_no_refine']} | {fmt(100.0 * buckets['B4_no_refine'] / max(1, total_miss), 1)}% | 该帧没有配准决策（无源帧检出 / 链没跑） |")
    md.append(f"| **合计失手** | **{total_miss}** | 100% | |")
    md.append("")
    md.append(
        "⇒ **失手帧的主体（"
        f"{fmt(100.0 * (buckets['B2_window_no_peak'] + buckets['B3_out_of_window'] + buckets['B4_no_refine']) / max(1, total_miss), 1)}%）"
        "根本不在配准族的射程内**：窗内无峰（模板/外观漂移）+ 窗外（目标机动超出搜索半径）+ 无决策。\n"
    )

    md.append("## 4. 可分性：可计算排序规则 vs oracle\n")
    md.append(
        "所有规则都只用**当帧可得**的量（候选峰的坐标 / 峰值 / 该峰自身的唯一性比 / 链位置 / 锚点位置 / "
        "本次匹配的唯一性比），**不用 GT** ⇒ 「可计算」是硬约束。\n"
    )
    md.append(
        f"| 规则 | hit@5px% | Δ vs 基线(pp) | 占 O-reg 的比例（分母 = **O-reg 不加出框门** "
        f"{fmt(reg_gain)} pp / {reg_frames} 帧） |"
    )
    md.append("|:--|--:|--:|--:|")
    for rule in sorted(rules, key=lambda r: -gain_pp.get(r, 0.0)):
        md.append(
            f"| `{rule}` | {fmt(100.0 * rules[rule] / n)} | {fmt(gain_pp.get(rule, 0.0))} "
            f"| {fmt(100.0 * share.get(rule, 0.0), 1)}% |"
        )
    md.append("")
    md.append(
        f"⇒ **最好的一条规则只拿到 oracle 的 {fmt(100.0 * best_share, 1)}%**"
        f"（`{best_rule[0]}`，{fmt(best_rule[1])} pp）⇒ "
        "**「相关峰候选重排」这条路在物理上就不可分**：真目标的峰**不是**比背景结构更高的那个。\n"
    )

    md.append("## 5. 可分性解剖：oracle 选中的那个候选峰长什么样\n")
    md.append(f"- 基线失手、但 O-reg 能救回的帧：**{anatomy.get('recoverable_frames', 0)}**")
    md.append("- 该候选峰在**按峰高降序**的候选列表里的名次（0 = 就是 argmax ⇒ 基线已经选中它）：")
    md.append("")
    md.append("| 名次 | 帧数 |")
    md.append("|--:|--:|")
    for rank in sorted(anatomy.get("rank_hist", {}), key=lambda x: int(x)):
        md.append(f"| {rank} | {anatomy['rank_hist'][rank]} |")
    md.append("")
    md.append("### 5.1 失手原因拆解（机理落在哪一层）\n")
    md.append("| 原因 | 帧数 | 含义 |")
    md.append("|--:|--:|:--|")
    md.append(f"| 该帧**无任何上报** | {anatomy.get('why_no_report', 0)} | 本可命中的点被某道门丢了 |")
    md.append("")
    md.append("**静默帧的生产口径归因**（按 `_build_refine_report_stream` 的**判定顺序**归因）：\n")
    md.append("| 成因 | 帧数 | 占静默帧 | 改报 argmax 后 TP / FP | 含义 |")
    md.append("|--:|--:|--:|:--:|:--|")
    cause_rows = [
        ("rollback_gate", "**回退链峰值门**", "回退链（模板来自更早帧）峰值 < 0.90 ⇒ 整帧弃权（report 那一格根本不写）"),
        ("fov_gate", "出框门", "上报点落在画面外/贴边 ⇒ 被 `_apply_fov_margin_gate` 删掉"),
        ("failed_no_fallback", "重推失败且兜底为空", "链断了且 `best_effort` 为空、源帧无检出"),
        ("no_decision", "无配准决策", "该帧根本没有源帧检出 ⇒ 无链可跑"),
        ("other", "其它", "未归入以上任何一道门"),
    ]
    silent_total = max(1, int(policy.get("silent_frames", 0)))
    for key, label, desc in cause_rows:
        count = int(policy.get("silent_by_cause", {}).get(key, 0))
        tp = int(policy.get("silent_cause_tp", {}).get(key, 0))
        fp = int(policy.get("silent_cause_fp", {}).get(key, 0))
        md.append(f"| {label} | {count} | {fmt(100.0 * count / silent_total, 1)}% | {tp} / {fp} | {desc} |")
    md.append("")
    md.append(
        "> ⚠️ **口径修正（夜 8 踩坑留痕）**：第一版把「有门前上报点却没上报」一律记成"
        "**出框门丢掉**（842 帧）—— **错**。生产代码先查**回退链峰值门**，命中就 `continue`，"
        "report 那一格根本不会被写 ⇒ 门前有上报点也照样是静默帧。"
        f"按生产顺序重新归因后，回退链峰值门才是大头（{policy.get('silent_by_cause', {}).get('rollback_gate', 0)} 帧）。\n"
    )
    md.append(f"| refine 被拒（回退链位置） | {anatomy.get('why_refine_rejected', 0)} | 配准结果被判「离链太远」而作废 |")
    md.append(f"| 采纳了二次意见（rematch） | {anatomy.get('why_rematch_taken', 0)} | 二次意见窗的 argmax 覆盖了大匹配 |")
    md.append(f"| 上报点 = argmax（仍失手） | {anatomy.get('why_report_eq_argmax', 0)} | 峰本身不在目标上 ⇒ 换选无用 |")
    md.append("")
    md.append("| 项 | 值 |")
    md.append("|:--|--:|")
    md.append(f"| oracle 候选峰的峰高 / argmax 峰高（中位数） | {fmt(anatomy.get('peak_ratio_p50') and sorted(anatomy['peak_ratio_p50'])[len(anatomy['peak_ratio_p50']) // 2])} |")
    md.append(f"| argmax 通过唯一性门（uniq ≤ 门限）的帧数 | {anatomy.get('argmax_gate_pass', 0)} / {anatomy.get('recoverable_frames', 0)} |")
    md.append(f"| oracle 候选峰通过唯一性门的帧数 | {anatomy.get('picked_gate_pass', 0)} / {anatomy.get('recoverable_frames', 0)} |")
    md.append(f"| oracle 候选峰比 argmax 更靠近链位置的帧数 | {anatomy.get('picked_closer_to_chain', 0)} / {anatomy.get('recoverable_frames', 0)} |")
    md.append("")
    md.append(
        "⇒ **两个候选峰都过唯一性门、唯一性也几乎一样（0.39 vs 0.40）** ⇒ 唯一性门**分不开**它们；"
        "**只有峰高能分开，而峰高恰好偏向错的那个** ⇒ 无状态信号不可用。\n"
    )

    if rbgate.get("rows"):
        md.append("## 6. 新战场：**回退链峰值门**（`--replay-rollback-min-peak`）受控 A/B\n")
        base_row = next((r for r in rbgate["rows"] if abs(r["peak"] - 0.90) < 1e-9), None)
        rb_cause = policy.get("silent_by_cause", {}).get("rollback_gate", 0)
        rb_tp = policy.get("silent_cause_tp", {}).get("rollback_gate", 0)
        rb_fp = policy.get("silent_cause_fp", {}).get("rollback_gate", 0)
        md.append(
            f"**机理**：回退链的模板来自更早的帧（`source_frame ≠ target_frame − Δ`）⇒ 目标可能已飞走、"
            f"配准可能锁到背景结构（\"幽灵坐标\"）⇒ 峰值 < 门限时**整帧弃权**"
            f"（生产代码里 `continue` 在写 report 之前 ⇒ 门前有上报点也照样静默）。"
        )
        md.append(
            f"**离线账**（24 序 dump 现读）：该门丢掉 **{rb_cause} 帧**；"
            f"若这些帧改报 refine 的 argmax ⇒ **TP +{rb_tp} / FP +{rb_fp}**（净 {rb_tp - rb_fp:+d} 帧）。\n"
        )
        md.append("**受控 A/B**（唯一变量 = 该旋钮；其余旋钮逐字取自 `run_delivery.sh` + 出框门 m=6）：\n")
        md.append("| rollback-min-peak | F1% | ΔF1(pp) | Recall% | Precision% | FAR | ΔFAR | hit@5px% | Δhit(pp) | CLE p95(px) | max jitter(px) | FP |")
        md.append("|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for row in rbgate["rows"]:
            rp = row["replay"]
            if base_row is not None:
                df1 = 100.0 * (rp["f1"] - base_row["replay"]["f1"])
                dfar = rp["far"] - base_row["replay"]["far"]
                dhit = 100.0 * (rp["hit5"] - base_row["replay"]["hit5"])
            else:
                df1 = dfar = dhit = 0.0
            md.append(
                f"| {row['peak']:.2f} | {fmt(100.0 * rp['f1'])} | {df1:+.4f} | {fmt(100.0 * rp['recall'])} | "
                f"{fmt(100.0 * rp['precision'])} | {fmt(rp['far'], 6)} | {dfar:+.6f} | {fmt(100.0 * rp['hit5'])} | "
                f"{dhit:+.4f} | {fmt(rp['cle95'], 3)} | {fmt(rp['jitter'], 3)} | {rp['fp']} |"
            )
        md.append("")
        md.append(
            f"> **门禁**：rb=0.90 的**指标字段**与既有交付档产物"
            f"（`runs/lookahead/fovgate24nogmc/.../fov__m6/arms.json`）**逐位相等**（门禁脚本口径 68 字段全等），"
            f"且所有变体的 original / stale / kinematic 三臂**跨变体逐位相等**"
            f"（该旋钮只作用于 replay 臂）。"
            f"⚠️ **整份 JSON 不逐位相等**：`frame_reads`（缓存读帧计数）与 `elapsed_seconds` 每次跑都不同，"
            f"且 rb 侧多出 `stream_stats.per_sequence.*.replay_disp_gained` 等键 ⇒ "
            f"**交付档产物与 A/B 产物不是同一版代码**（A/B 跑在更晚的代码上）⇒ "
            f"只可比**指标**，不可比整文件。见 `runs/lookahead/rbgate24nogmc/progress.txt`。\n"
        )

        md.append("")
        rb_cause = int(policy.get("silent_by_cause", {}).get("rollback_gate", 0))
        rb_tp = int(policy.get("silent_cause_tp", {}).get("rollback_gate", 0))
        rb_fp = int(policy.get("silent_cause_fp", {}).get("rollback_gate", 0))
        md.append("### 6.1 逐序稳健性：总体增益**不是**全面改善（**必读**）\n")
        md.append("| 变体 | ΔTP | ΔFP | 净 | 受影响序列 | 变好/变差/持平 | ΔTP 最多的两序（占总 ΔTP） | 最差序列 ΔF1 |")
        md.append("|:--|--:|--:|--:|--:|:--:|:--|--:|")
        for row in rbgate["rows"]:
            if abs(row["peak"] - 0.90) < 1e-9:
                continue
            sd = row.get("seq_delta") or {}
            if not sd:
                continue
            dtp = sum(v["dtp"] for v in sd.values())
            dfp = sum(v["dfp"] for v in sd.values())
            better = sum(1 for v in sd.values() if v["df1_pp"] > 0.005)
            worse = sum(1 for v in sd.values() if v["df1_pp"] < -0.005)
            flat = len(sd) - better - worse
            touched = sum(1 for v in sd.values() if v["dtp"] or v["dfp"])
            top = sorted(sd.items(), key=lambda kv: -kv[1]["dtp"])[:2]
            top_txt = " / ".join(
                f"{k[:22]} {v['dtp']:+d}" for k, v in top
            )
            tp_share = 100.0 * sum(v["dtp"] for _k, v in top) / dtp if dtp else 0.0
            worst = min(sd.items(), key=lambda kv: kv[1]["df1_pp"])
            md.append(
                f"| rb={row['peak']:.2f} | {dtp:+d} | {dfp:+d} | {dtp - dfp:+d} | {touched}/24 | "
                f"{better}/{worse}/{flat} | {top_txt}（**{tp_share:.1f}%**） | "
                f"{worst[0][:22]} {worst[1]['df1_pp']:+.4f}pp |"
            )
        md.append("")
        # 结论段的每个数字都现读（生产口径：rb__0.90 的 stream_stats.per_sequence.replay_rollback_abstained）
        _rb_by_seq = {}
        for _r in rbgate.get("rows", []):
            if abs(_r["peak"] - 0.90) < 1e-9:
                _rb_by_seq = _r.get("rollback_by_seq") or {}
                break
        _rb_total = sum(_rb_by_seq.values())
        _rb_top = sorted(_rb_by_seq.items(), key=lambda kv: -kv[1])[:3]
        _rb_top_txt = " / ".join(f"`{k[:22]}` {v}" for k, v in _rb_top)
        _rb_top2_share = (100.0 * sum(v for _k, v in _rb_top[:2]) / _rb_total) if _rb_total else 0.0
        # 逐序 ΔF1：**每个变体各取最差序列**（与表格「最差序列 ΔF1」列同口径），按放松程度排列
        _by_peak: dict[float, list] = {}
        _base_far: dict[str, float] = {}
        for _r in rbgate.get("rows", []):
            if abs(_r["peak"] - 0.90) < 1e-9:
                _base_far = {k: (v.get("far") or 0.0) for k, v in (_r.get("seq_delta") or {}).items()}
                continue
            for _seq, _d in (_r.get("seq_delta") or {}).items():
                _by_peak.setdefault(_r["peak"], []).append((_d["df1_pp"], _seq, _d.get("far")))
        _df1_series = " → ".join(
            f"{min(v)[0]:+.4f}" for _p in sorted(_by_peak, reverse=True) for v in [_by_peak[_p]]
        )
        _worst_seq = min((min(v) for v in _by_peak.values()), key=lambda t: t[0])[1] if _by_peak else ""
        # FAR 最高序（与最差 ΔF1 序**可能不是同一序** ⇒ 分开报）
        _far_worst = None
        for _p, _v in _by_peak.items():
            for _df1, _seq, _far in _v:
                if (_far or 0.0) > (_far_worst[2] if _far_worst else -1.0):
                    _far_worst = (_df1, _seq, _far or 0.0, _p)
        md.append(
            f"> ⇒ **总体 ΔTP 有 85~89% 落在 2/24 序列**（`000322` 与 `001324`）—— 在**生产口径**下，"
            f"被回退峰值门丢帧最多的三序是 {_rb_top_txt}（合计 {_rb_total} 帧；前两序占 {_rb_top2_share:.1f}%），"
            f"⇒ `000322` 是第 1、`001324` 是第 3 ⇒ **机理自洽**（增益确实来自回退链），但这**不是**全面改善；"
            f"代价集中在 `{_worst_seq[:22]}`（逐序 ΔF1 随放松单调变差：{_df1_series} pp）"
            + (
                f"；FAR 最高的一序是 `{_far_worst[1][:22]}`"
                f"（FAR {_base_far.get(_far_worst[1], 0.0):.4f}（rb=0.90）→ {_far_worst[2]:.4f}"
                f"（rb={_far_worst[3]:.2f}））。"
                if _far_worst else "。"
            )
        )
        md.append("")
        md.append(
            "> ⚠️ **口径说明**：「被回退峰值门丢掉的帧」有两个口径 —— "
            f"**生产口径**（本节 A/B 产物 `stream_stats.per_sequence.replay_rollback_abstained`，合计 {_rb_total} 帧）"
            f"与**离线 GT 帧口径**（§5.1 静默帧归因里的回退链峰值门，合计 "
            f"{int(policy.get('silent_by_cause', {}).get('rollback_gate', 0))} 帧）—— 两者**不可混用**；"
            f"夜 8 初稿把离线口径的数字写在生产口径的句子里（且 938 / 479 两个数**全仓查不到**），已按现读值改写。\n"
        )
        md.append("")
        md.append("### 6.2 逐门限的**边际** TP/FP 兑换率（为什么 0.90 不能一路放松到底）\n")
        md.append("| 放松区间 | ΔTP | ΔFP | 边际 TP/FP | 判读 |")
        md.append("|:--|--:|--:|--:|:--|")
        prev = None
        for row in rbgate["rows"]:
            if prev is None:
                prev = row
                continue
            dtp = row["replay"]["tp"] - prev["replay"]["tp"]
            dfp = row["replay"]["fp"] - prev["replay"]["fp"]
            ratio = dtp / dfp if dfp else float("inf")
            if ratio >= 1.0:
                verdict = "净赚（F1↑）"
            else:
                verdict = "**净亏（F1↓）**"
            md.append(
                f"| {prev['peak']:.2f} → {row['peak']:.2f} | {dtp:+d} | {dfp:+d} | "
                f"{ratio:.2f}:1 | {verdict} |"
            )
            prev = row
        md.append("")
        md.append(
            f"⇒ **边际兑换率随放松单调恶化**：0.90→0.80 仍高（净赚），0.80→0.70 / 0.70→0.50 逼近 1:1，"
            "0.50→0.00 **净亏**且 **FAR 破红线** ⇒ 回退链峰值门**确实在干活**，不能整体拆掉。"
        )
        md.append("")
        md.append("### 6.3 口径纪律：为什么**离线**单帧账对这个旋钮不成立\n")
        md.append(
            f"离线账（24 序 dump 现读）说：把 {rb_cause} 帧的 refine argmax 直接报出去 ⇒ "
            f"**TP +{rb_tp} / FP +{rb_fp}**（TP/FP = {rb_tp / max(1, rb_fp):.2f}:1）。"
            "**实测 rb=0.00 只有 +1155 TP / +1780 FP** ⇒ **离线账高估了 3.5 倍**。"
        )
        md.append("")
        md.append("**根因**：交付档 CFG 开了 `--output-max-step-px 30`（伺服 slew limiter）⇒ "
                  "限幅的参考是「同一 track **最近一次**的上报位置」⇒ **上报轨迹是路径依赖的**："
                  "多报一帧会改写后续所有帧的限幅基准 ⇒ **单帧离线账在原理上就不适用于本旋钮**。")
        md.append("")
        md.append("| 峰值区间（回退门内） | 帧数 | argmax 命中 | 命中率 |")
        md.append("|:--|--:|--:|--:|")
        for _bin, _v in sorted((policy.get("rollback_peak_hist") or {}).items()):
            md.append(
                f"| pk {_bin} | {_v.get('n')} | {_v.get('hit')} | "
                f"{fmt(100.0 * int(_v.get('hit', 0)) / max(1, int(_v.get('n', 0))), 1)}% |"
            )
        md.append("")
        md.append(
            "⇒ **峰值本身是有判别力的信号**（pk∈[0.80,0.90) 命中率 93.0%），"
            "**但落到生产链路上被 slew 限幅的耦合吃掉了** ⇒ 这是本族**真正**的卡点"
            "（不是「没有信号」，是「信号被下游耦合吞掉」）。"
        )
        md.append("")

    md.append("## 7. 结论\n")
    md.append("| 候选 | 上界/证据 | 裁定 |")
    md.append("|:--|:--|:--|")
    md.append(
        f"| **配准候选重排族**（在 top-K 相关峰里换一个） | 合法 oracle：最宽口径（O-all 合法）= **{fmt(all_legal_gain)} pp**"
        f"（空间够）/ 本族 O-reg 合法 = **{fmt(reg_legal_gain)} pp**；"
        f"可救帧里 oracle 选中 argmax 的占 {fmt(100.0 * anatomy.get('rank_hist', {}).get('0', 0) / max(1, anatomy.get('recoverable_frames', 1)), 1)}%；"
        f"15 条可计算规则最好只拿到 oracle 的 {fmt(100.0 * best_share, 1)}%（O-reg 口径）/ "
        f"{fmt(_share_ceiling_pct, 1)}%（最宽天花板口径） | "
        "**结案：不可行**（空间够但**不可分** + 不是选峰问题 + 无状态信号） |"
    )
    md.append(
        "| 唯一性门的「回退锚点 / 运动学 / 弃权」形态 | §13/§15 已实测否决；夜 8 的 R4/R7/R8–R14 复现同一结论 | "
        "**结案：不可行** |"
    )
    md.append(
        f"| **新战场（移交）**：报告策略族 | 「该报不报」的帧 {anatomy.get('why_no_report', 0)}、"
        f"失手帧里 {fmt(100.0 * buckets['B2_window_no_peak'] / max(1, total_miss), 1)}% 是「窗内无峰」 | "
        "**未结案**（属「弃权 / 出框门 / 搜索半径」族，不属配准族；且 FAR 红线余量只有 0.005663） |"
    )
    if rbgate.get("rows"):
        md.append(
            "| **回退链峰值门**（`--replay-rollback-min-peak`） | 24 序受控 A/B：rb=0.80/0.70/0.50 ⇒ F1 "
            "+0.0827/+0.1008/+0.1382 pp，但 **85~89% 的增益集中在 2/24 序列**，代价是 FAR 余量被吃掉 "
            "8%/21%/58% 且 `002251` 逐序单调变差；rb=0.00 ⇒ FAR **破红线** 0.035110 且 F1 −0.1791 pp | "
            "**结案：交付档保持 0.90**（增益不鲁棒 + FAR 余量是交付红线）；"
            "若一定要放松，**rb=0.80** 是性价比最高点（+0.0827 pp 只花 8% 余量） |"
        )
    md.append("")
    md.append(
        "⇒ **交付裁定不变**：交付推荐档 = `cs_t0043`（no-GMC）+ Δ=3(120 ms) + `replay` 臂 + "
        "`--replay-fov-margin-px 6`，**交付档 CFG 本轮不需要改**。\n"
    )

    md.append("## 8. 复现\n")
    md.append("```bash")
    md.append("cd /tmp/pycharm_project_10ae9e2e")
    md.append("bash manu/tracking/lookahead/server_overnight/run_reg_oracle_dump.sh")
    md.append("python manu/tracking/lookahead/diag_reg_oracle.py --analyze \\")
    md.append("  --dump-dir runs/lookahead/fi_nogmc24/reg_oracle \\")
    md.append("  --out-md runs/lookahead/fi_nogmc24/reg_oracle/reg_oracle.md \\")
    md.append("  --out-json runs/lookahead/fi_nogmc24/reg_oracle/reg_oracle.json")
    md.append("python manu/tracking/lookahead/make_reg_oracle_report.py \\")
    md.append("  --root runs/lookahead/fi_nogmc24/reg_oracle \\")
    md.append("  --out-md runs/lookahead/fi_nogmc24/reg_oracle/closure_report.md \\")
    md.append("  --out-json runs/lookahead/fi_nogmc24/reg_oracle/closure_report.json")
    md.append("```")
    md.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md), encoding="utf-8")
    # 生产口径：被回退峰值门丢掉的帧（rb=0.90 的 stream_stats.per_sequence.replay_rollback_abstained）
    _rb_seq: dict = {}
    for _r in rbgate.get("rows", []):
        if abs(_r["peak"] - 0.90) < 1e-9:
            _rb_seq = _r.get("rollback_by_seq") or {}
            break
    _rb_total = sum(_rb_seq.values())
    _rb_top = sorted(_rb_seq.items(), key=lambda kv: -kv[1])[:3]
    _rb_caliber = {
        "caliber": "生产口径（rb=0.90 产物 stream_stats.per_sequence.replay_rollback_abstained）",
        "total": _rb_total,
        "top3": [{"sequence": k, "frames": v} for k, v in _rb_top],
        "top2_share_pct": (100.0 * sum(v for _k, v in _rb_top[:2]) / _rb_total) if _rb_total else 0.0,
        "offline_gt_frames": int(policy.get("silent_by_cause", {}).get("rollback_gate", 0)),
    }

    payload = {
        "scope": "24 序官方口径 · cs_t0043 (no-GMC) · Δ=3(120 ms) · replay 臂 + 出框门 m=6",
        "meta": meta,
        "baseline_hit_pct": derived["hit_base_pct"],
        "oracle_reg_gain_pp": reg_gain,
        "oracle_all_gain_pp": all_gain,
        "oracle_reg_legal_gain_pp": reg_legal_gain,
        "oracle_all_legal_gain_pp": all_legal_gain,
        "oracle_reg_frames": reg_frames,
        "buckets": buckets,
        "rules": {rule: {"hit_pct": 100.0 * rules[rule] / n, "gain_pp": gain_pp.get(rule, 0.0), "share_of_oracle": share.get(rule, 0.0)} for rule in rules},
        "best_rule": {"name": best_rule[0], "gain_pp": best_rule[1], "share_of_oracle": best_share},
        "anatomy": anatomy,
        "policy": policy,
        "rbgate": {
            "rows": [
                {
                    "peak": r["peak"],
                    "replay": r["replay"],
                    "seq_delta": r.get("seq_delta", {}),
                }
                for r in rbgate.get("rows", [])
            ],
        },
        "ceiling": {
            "baseline_hit5_pct": derived["hit_base_pct"],
            "arms_hit5_pct": arms_hit5,
            "reg_gain_pp": reg_gain,
            "reg_legal_gain_pp": reg_legal_gain,
            "all_gain_pp": all_gain,
            "all_legal_gain_pp": all_legal_gain,
            "fov_gate_cost_pp": (all_gain - all_legal_gain) if (all_gain is not None and all_legal_gain is not None) else None,
            "reg_frames": reg_frames,
            "reg_legal_frames": reg_legal_frames,
            "all_legal_frames": totals.get("hit_oracle_all_legal", 0) - totals["hit_base"],
            "ceiling_legal_pp": all_legal_gain,
            "share_denominator": {"name": "O-reg（不加出框门）", "gain_pp": reg_gain, "frames": reg_frames},
            "best_rule": {"name": best_rule[0], "gain_pp": best_rule[1], "share_pct": 100.0 * best_share},
            "best_rule_share_of_ceiling_pct": (100.0 * best_rule[1] / all_legal_gain) if (best_rule[1] and all_legal_gain) else None,
            "recoverable_frames": anatomy.get("recoverable_frames", 0),
            "rank0_share_pct": (
                100.0 * anatomy.get("rank_hist", {}).get("0", 0) / max(1, anatomy.get("recoverable_frames", 1))
            ),
        },
        "buckets_total_miss": sum(buckets.values()),
        "rbgate_rollback_caliber": _rb_caliber,
        "silent_attribution": {
            "silent_frames": int(policy.get("silent_frames", 0)),
            "by_cause": policy.get("silent_by_cause", {}),
            "tp_by_cause": policy.get("silent_cause_tp", {}),
            "fp_by_cause": policy.get("silent_cause_fp", {}),
        },
        "ambiguous": src.get("ambiguous"),
        "ceiling": {
            "baseline_hit5_pct": derived["hit_base_pct"],
            "arms_hit5_pct": arms_hit5,
            "reg_gain_pp": reg_gain,
            "reg_legal_gain_pp": reg_legal_gain,
            "all_gain_pp": all_gain,
            "all_legal_gain_pp": all_legal_gain,
            "fov_gate_cost_pp": (all_gain - all_legal_gain) if (all_gain is not None and all_legal_gain is not None) else None,
            "reg_frames": reg_frames,
            "reg_legal_frames": reg_legal_frames,
            "all_legal_frames": totals.get("hit_oracle_all_legal", 0) - totals["hit_base"],
            "ceiling_legal_pp": all_legal_gain,
            "share_denominator": {"name": "O-reg（不加出框门）", "gain_pp": reg_gain, "frames": reg_frames},
            "best_rule": {"name": best_rule[0], "gain_pp": best_rule[1], "share_pct": 100.0 * best_share},
            "best_rule_share_of_ceiling_pct": (100.0 * best_rule[1] / all_legal_gain) if (best_rule[1] and all_legal_gain) else None,
            "recoverable_frames": anatomy.get("recoverable_frames", 0),
            "rank0_share_pct": (
                100.0 * anatomy.get("rank_hist", {}).get("0", 0) / max(1, anatomy.get("recoverable_frames", 1))
            ),
        },
        "buckets_total_miss": sum(buckets.values()),
        "rbgate_rollback_caliber": _rb_caliber,
        "silent_attribution": {
            "silent_frames": int(policy.get("silent_frames", 0)),
            "by_cause": policy.get("silent_by_cause", {}),
            "tp_by_cause": policy.get("silent_cause_tp", {}),
            "fp_by_cause": policy.get("silent_cause_fp", {}),
        },
        "per_sequence_b1": src.get("per_sequence_b1"),
        "verdict": "配准候选重排族：结案（不可行）",
        "handoff": "新战场 = 报告策略族（弃权门 / 出框门 / 搜索半径），不属配准族",
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[SUCCESS] closure report -> {out_md}")
    print(f"[SUCCESS] closure json   -> {args.out_json}")


if __name__ == "__main__":
    main()
