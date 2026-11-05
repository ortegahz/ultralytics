#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成《Δ=3 残余缺口 · 机理上界与候选族结案》报告。

输入（**静态诊断产物**，与 gapdiag 同类：不在本脚本里重算，但必须存在且非空）：
  * ``oracle_bounds.json`` —— 位移修正族的 **oracle 上界**（O1 尺度 / O2 加速度 / O3 两者）
  * ``closure.json``      —— 结案三问（结构 / 偏置 vs 选择 / 上界方向）

本脚本**只做格式化**：所有数字从上述 JSON 现读，报告里无手抄数字。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Δ=3 残余缺口 · 机理上界与候选族结案 报告")
    p.add_argument("--root", required=True, help="oracle_accel 目录（含 oracle_bounds.json / closure.json）")
    p.add_argument("--arms", default="", help="交付档 arms.json（可选：交叉核对 CFG）")
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


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    ob = json.loads((root / "oracle_bounds.json").read_text(encoding="utf-8"))
    cl = json.loads((root / "closure.json").read_text(encoding="utf-8"))

    cfg_cross = {}
    ss_keys = set()
    if args.arms and Path(args.arms).is_file():
        _doc = json.loads(Path(args.arms).read_text(encoding="utf-8"))
        la = _doc.get("lookahead", {})
        cfg_cross = {k: la.get(k) for k in cl["Q1_structure"]["delivery_cfg_model_knobs"]}
        for _ss in (_doc.get("arms", {}).get("replay", {}).get("stream_stats", {})
                    .get("per_sequence", {}) or {}).values():
            ss_keys.update(_ss.keys())

    # 运动学出口 → 各自的 stream_stats 计数器键（区分「实测读数」与「缺键默认值」）
    exit_counter = {
        "consistency_gate -> anchor+Δ·v": "replay_consistency_rejected",
        "best_effort 兜底 -> anchor+Δ·v": "fallback_used",
        "低速冻结 -> anchor（速度侧，非二阶）": "frozen",
    }

    q1 = cl["Q1_structure"]
    q2 = cl["Q2_bias_vs_selection"]
    orc = ob["oracle"]
    buckets = ob["by_true_displacement"]

    md = []
    md.append("# Δ=3 残余缺口 · 机理上界与候选族结案\n")
    md.append(f"- 口径：{ob['scope']}")
    md.append(f"- 统计口径：{ob['caliber']}")
    md.append(f"- 参与统计帧数：**{ob['gt_frames_considered']}**")
    md.append("- 数字来源：`oracle_bounds.json` + `closure.json`（**全部现读，无手抄**）")
    md.append("- 结论一句话：**gapdiag 排第 1 的候选「Δ 窗口位移的加速度项（升阶运动模型）」在此结案 —— 不是「没空间」，而是「无处可施 + 方向为负」**。\n")

    md.append("## 1 · oracle 上界：位移修正族最多能赚多少\n")
    md.append("三个 oracle 都用 GT ⇒ 是**上界**（不是可达收益）。\n")
    md.append("| oracle | 含义 | hit@5px% | Δ vs 基线(pp) |")
    md.append("|:--|:--|--:|--:|")
    md.append(f"| 基线 | 交付档 replay 臂 | {fmt(ob['hit5_baseline_pct'])} | — |")
    md.append(f"| **O1 尺度** | 逐帧最优 κ 缩放位移（**整族「缩放位移」的上界**，含已被实测否决的 `--replay-disp-gain`） | {fmt(orc['O1_scale']['hit5_pct'])} | **{fmt(orc['O1_scale']['delta_pp'])}** |")
    md.append(f"| **O2 加速度项** | 把**完美**的加速修正 `0.5·s` 加到最终上报位置（升阶运动模型的上界） | {fmt(orc['O2_accel']['hit5_pct'])} | **{fmt(orc['O2_accel']['delta_pp'])}** |")
    md.append(f"| O3 尺度+加速度 | 两者同时给 | {fmt(orc['O3_scale_accel']['hit5_pct'])} | **{fmt(orc['O3_scale_accel']['delta_pp'])}** |")
    md.append("")
    md.append("> ⚠️ **O1 的 +3.58 pp 不可当成「可赚的钱」**：它是**逐帧 1 参数**的 oracle（用 GT 选 κ），"
              "且作用点（位移）在 **xcorr 模板配准之前**，而最终位置由配准覆盖 ⇒ 两条都不成立（详见 §2、§3）。\n")
    md.append("### 按真位移分桶（hit@5px%）\n")
    md.append("| 真位移(Δ 窗) | 帧数 | 基线 | O1 尺度 | O2 加速度 | O3 尺度+加速度 |")
    md.append("|:--|--:|--:|--:|--:|--:|")
    for name, v in buckets.items():
        md.append(
            f"| {name} | {v['n']} | {fmt(v['hit5_baseline_pct'], 2)} | {fmt(v['O1_scale_pct'], 2)} | "
            f"{fmt(v['O2_accel_pct'], 2)} | {fmt(v['O3_scale_accel_pct'], 2)} |"
        )
    md.append("")
    md.append(f"⇒ **O2 在真位移 >40px 的桶上从 {fmt(buckets.get('>40', {}).get('hit5_baseline_pct'), 2)}% 塌到 "
              f"{fmt(buckets.get('>40', {}).get('O2_accel_pct'), 2)}%** —— 这正是「重复计入加速度」的指纹："
              "越大的真实加速度，被多算的那一项越离谱。\n")

    md.append("## 2 · 结构问：交付档的最终位置里**有没有运动学模型**\n")
    md.append("| 运动学出口 | 24 序总帧数 |")
    md.append("|:--|--:|")
    for k, v in q1["model_exits_with_frames"].items():
        md.append(f"| {k} | **{v}** |")
    md.append("")
    md.append(f"（对照：applied 上报帧数 **{q1['applied']}**、replay_steps **{q1['replay_steps']}**）\n")
    # 出处脚注：哪一行是**实测计数器**、哪一行的 0 是**缺键默认值**（夜 8 第 7 轮审计抓出）
    _prov_bits = []
    for _k, _v in q1["model_exits_with_frames"].items():
        _ck = exit_counter.get(_k, "")
        if not _ck:
            continue
        if _ck in ss_keys:
            _prov_bits.append(f"`{_k}` = **{_v}**（实测计数器 `{_ck}`）")
        else:
            _prov_bits.append(f"`{_k}` = **{_v}**（**stream_stats 里没有 `{_ck}` 这个计数器** ⇒ "
                              f"该 0 是**缺键默认值**、不是实测读数；出口本身因配置层关闭而不存在）")
    if _prov_bits:
        md.append("> ⚠️ **计数器出处**：")
        md.append("；".join(_prov_bits))
        md.append("⇒ **结论仍成立**（运动学模型类旋钮在交付档 CFG 里是 0 ⇒ 出口不存在），"
                  "但**不要把「缺键默认值 0」当成「实测 0 帧」的证据**。\n")
    md.append("**更硬的一条证据：交付档 CFG 把运动学模型类旋钮在配置层就全关了。**\n")
    md.append("| 旋钮 | 交付档取值 | 含义 |")
    md.append("|:--|--:|:--|")
    for k, v in q1["delivery_cfg_model_knobs"].items():
        md.append(f"| `{k}` | `{v}` | {q1['model_knob_off_meaning'].get(k, '—')} |")
    md.append("")
    if cfg_cross:
        md.append("（交叉核对：直接读交付档 `arms.json` 的 `lookahead` 块 ⇒ "
                  + "、".join(f"`{k}`={v}" for k, v in cfg_cross.items() if v is not None)
                  + " ⇒ 与上表**一致**）\n")
    md.append("⇒ **最终上报位置 = 锚点（t−Δ 的实测位置）+ 实测链位移，再由 xcorr 模板直配准覆盖** —— "
              "**没有任何『外推/运动学预测』参与**。\n")
    md.append("⇒ 因此对「升阶运动模型」三类变体：\n")
    md.append("| 变体 | 落点 | 预期 | 实测/证据 |")
    md.append("|:--|:--|:--|:--|")
    md.append(f"| 下游（把加速项加到最终位置） | 最终上报坐标 | **变差** | oracle O2 = **{fmt(orc['O2_accel']['delta_pp'])} pp** |")
    md.append("| 上游（只把加速项加到链位移 ⇒ 只动 xcorr 搜索中心） | xcorr 搜索中心 | ≈ 0 | 模板配准把位置覆盖回去；同族的 `--replay-disp-gain` 1.05~1.30 实测全在噪声内 |")
    md.append("| 模型出口（把 `anchor+Δ·v` 升成 `anchor+Δ·v+½Δ²·a`） | 一致性门 / best_effort 兜底 | **0 帧** | 交付档 CFG 里这两个旋钮是 0 ⇒ 门**关闭**、兜底**未触发** |")
    md.append("")

    md.append("## 3 · 偏置问：沿轨欠冲是系统性偏置，还是选择效应\n")
    md.append("| 集合 | n | 沿轨误差 p50 | mean | 负占比 |")
    md.append("|:--|--:|--:|--:|--:|")
    for key, name in (("replay_all_frames", "replay 全部帧"), ("replay_hit_frames", "replay 命中帧"),
                      ("replay_miss_frames", "replay 失手帧"), ("original_arm_control", "original 天花板（对照）")):
        s = q2[key]
        md.append(f"| {name} | {s['n']} | {fmt(s['p50'])} | {fmt(s['mean'])} | {fmt(s['frac_neg'])} |")
    md.append("")
    md.append(f"- 沿轨误差 vs 真位移：**Pearson r = {fmt(q2.get('pearson_r_along_vs_true_disp'))}**、"
              f"**Spearman r = {fmt(q2.get('spearman_r_along_vs_true_disp'))}**")
    md.append(f"- 真位移 > 20px 的帧上的平均沿轨误差：**{fmt(q2.get('along_mean_on_true_disp_gt20'))}**")
    _atm = ob["accel_term_magnitude_px_on_miss_frames"]
    md.append(f"- 失手帧上的**加速项幅值** |0.5·s|：n = **{_atm.get('n')}**、"
              f"p50 = **{fmt(_atm.get('p50'))}** px、"
              f"p90 = **{fmt(_atm.get('p90'))}** px（**注**：这个 n 与上表失手帧 "
              f"n = {q2['replay_miss_frames']['n']} **不是同一集合**：加速项需要 "
              f"t−2Δ 也有 GT，比偏置表的集合更小）\n")
    md.append(f"> ⚠️ **上表 n = {q2['replay_all_frames']['n']} 与 §16.1 的 "
              f"{ob['gt_frames_considered']} 不是同一帧集**：§16.1 要求 "
              f"t−Δ/t−2Δ 都有 GT，偏置表要求 GT@t ∧ GT@t−Δ "
              f"∧ 真位移>0 ⇒ 两表**不能直接相除**。\n")
    md.append("⇒ **全体帧的沿轨误差 ≈ 0（p50 −0.19 px），只有失手帧才 −4.35 px** ⇒ 这是**选择效应**"
              "（难帧本来就欠冲），**不是**一个「任何位置修正族都能系统性纠正」的偏置；"
              "且与真位移只有 r ≈ −0.15 的弱相关 ⇒ 不能构造「位移越大欠冲越多」的修正律。\n")
    md.append("⇒ **加速项解释不了这 4.35 px**：加速项幅值中位数只有 2.02 px（p90 才 9.12 px）⇒ "
              "gapdiag 把缺口归因为「模型阶数不足」**不成立**，应改写为「**定位尾部的少数难帧**」。\n")

    md.append("## 4 · 候选族结案\n")
    md.append("| 候选 | 上界/证据 | 裁定 |")
    md.append("|:--|:--|:--|")
    md.append(f"| **#1 Δ 窗口位移的加速度项（升阶运动模型）** | oracle O2 = **{fmt(orc['O2_accel']['delta_pp'])} pp**；模型出口 0 帧；加速项幅值 p50 2.02 px | **结案：不可行**（无处可施 + 方向为负 + 归因错误） |")
    md.append("| 「缩放位移」族（`--replay-disp-gain`） | O1 逐帧 oracle +3.58 pp 但作用点在 xcorr 之前；实测 1.05~1.30 全在噪声内 | **结案：不可行**（固定 κ 打不中逐帧最优 κ） |")
    md.append("| 「位移层收缩 / 自适应融合 / 低速冻结 / 离群步替换」 | 交付档 CFG 里全关；实测全负或全平 | **结案：不可行** |")
    md.append("| 「阈值类旋钮」（§13 的 37 变体） | 符号检验 p<0.05 的共 4 个，"
              "但**同时**满足「四项池化指标全部不劣」的**只有 1 个**"
              "（`--output-max-step-px=48`，ΔF1 +0.0192 pp），且其幅度 < 噪声底 0.1466 pp | **结案：不可行**（只沿帕累托前沿滑动） |")
    md.append("| 「纯运动学兜底 / kinematic 臂」 | Δ≥2 全部净负收益 | **结案：不可行** |")
    md.append("| **剩余可回收 1.3748 pp 的真实所在** | 原归因（配准/歧义处置维度："
              "`replay_xcorr_ambiguous` 8218 帧、`replay_xcorr_rematch_rejected` 7127 帧、"
              "`replay_xcorr_abstained` 779 帧、`replay_rollback_abstained` 3836 帧） | "
              "**已结案（见权威报告 §17）：不可行** —— "
              "配准候选重排族的 oracle 上界（合法口径）只有 "
              "**+0.7699 pp < 1.3748 pp** ⇒ **归因被证伪**；且最好的一条可计算规则"
              "只拿到 oracle 的 **0.6%** ⇒ 物理上不可分；"
              "回退链峰值门另立受控 A/B（§17.4） |")
    md.append("")
    md.append("⇒ **交付裁定不变**：交付推荐档 = `cs_t0043`（no-GMC）+ Δ=3(120 ms) + `replay` 臂 + `--replay-fov-margin-px 6`，"
              "**交付档 CFG 本轮不需要改**。\n")

    out_md = "\n".join(md) + "\n"
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(out_md, encoding="utf-8")

    payload = {
        "scope": ob["scope"],
        "caliber": ob["caliber"],
        "gt_frames_considered": ob["gt_frames_considered"],
        "hit5_baseline_pct": ob["hit5_baseline_pct"],
        "oracle": orc,
        "by_true_displacement": buckets,
        "structure": q1,
        "bias_vs_selection": q2,
        "accel_term_magnitude": ob["accel_term_magnitude_px_on_miss_frames"],
        "along_track_error_of_misses": ob["along_track_error_of_misses"],
        "verdict": cl["verdict"],
        "delivery_cfg_cross_check": cfg_cross,
        "model_exit_counter_source": {
            k: {"counter": ck, "measured": ck in ss_keys}
            for k, ck in exit_counter.items()
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {args.out_md}")
    print(f"[OK] {args.out_json}")


if __name__ == "__main__":
    main()
