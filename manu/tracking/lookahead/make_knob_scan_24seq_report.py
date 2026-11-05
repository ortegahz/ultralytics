#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""24 序官方口径 · Δ=3 专属受控单旋钮 A/B → 报告生成器（cs_t0043 / no-GMC）。

只读产物 JSON（``arms.json``），**所有数字直接读出、不重算**；差值 = 两个读出值相减。

产物布局（扫描器 ``server_overnight/run_knob_scan_d3.sh`` 写入）::

    <root>/runs/<model>/d<delta>/<tag>/arms.json   # 每个 tag = 一次运行
    <root>/manifest.tsv                             # tag|model|delta|flags|vals|rc|elapsed|path|ts

用法::

    python manu/tracking/lookahead/make_knob_scan_24seq_report.py \
        --root runs/lookahead/fi_nogmc24/knobscan_d3 \
        --ref-arms runs/lookahead/fovgate24nogmc/runs/cs_t0043/d3/fov__m6/arms.json \
        --out-md runs/lookahead/fi_nogmc24/knobscan_d3/report.md \
        --out-json runs/lookahead/fi_nogmc24/knobscan_d3/report.json
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from make_knob_scan_it2_report import loso_sensitivity, seq_f1

# 交付三门（业务域红线）
DELIVERY = {"f1_min": 0.88, "recall_min": 0.85, "far_max": 0.030}

DASH = "—"

KNOB_TITLE = {
    "--replay-rollback-min-peak": "回退峰值门 (`--replay-rollback-min-peak`)",
    "--replay-abstain-peak-below": "弃权峰值门 (`--replay-abstain-peak-below`)",
    "--output-max-step-px": "输出限幅 (`--output-max-step-px`, px)",
    "--replay-roi-selfpred-max-px": "ROI 前推上限 (`--replay-roi-selfpred-max-px`, px)",
    "--replay-roi-selfpred-window": "ROI 自预测窗长 (`--replay-roi-selfpred-window`, 帧)",
    "--replay-xcorr-radius": "相关搜索半径 (`--replay-xcorr-radius`, px)",
    "--replay-xcorr-th-k": "相关峰阈值 k (`--replay-xcorr-th-k`)",
    "--replay-xcorr-uniq-gate": "相关峰唯一性门 (`--replay-xcorr-uniq-gate`)",
    "--replay-abstain-disp-px": "弃权位移窗 (`--replay-abstain-disp-px`, px)",
    "--replay-rollback-max-frames": "回退最大帧数 (`--replay-rollback-max-frames`)",
    "--replay-max-hold": "回退保持帧数 (`--replay-max-hold`)",
}


def trow(cells: list[str]) -> str:
    """拼一行 Markdown 表格（统一 `| a | b |` 格式）。"""
    return "| " + " | ".join(str(c) for c in cells) + " |"


def fmt(value, scale: float = 1.0, digits: int = 4) -> str:
    if value is None:
        return DASH
    return f"{value * scale:.{digits}f}"


def fmt_signed(value, scale: float = 1.0, digits: int = 4) -> str:
    if value is None:
        return DASH
    return f"{value * scale:+.{digits}f}"


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def arm_row(artefact: dict, arm: str = "replay") -> dict:
    """单臂的读出指标（全部来自产物，不重算）。"""
    a = artefact["arms"][arm]
    cur = a["current_frame"]["overall"]
    srv = a["servo"]["overall"]
    return {
        "f1": cur["f1"], "recall": cur["recall"], "precision": cur["precision"],
        "far_per_frame": cur["far_per_frame"], "tp": cur["tp"], "fp": cur["fp"],
        "fn": cur["fn"], "det": cur["det"], "gt": cur["gt"],
        "hit5": srv.get("hit_rate_5px"),
        "cle50": srv.get("cle_pooled_p50"), "cle95": srv.get("cle_pooled_p95"),
        "cle99": srv.get("cle_pooled_p99"), "clemax": srv.get("cle_pooled_max"),
        "cle_samples": srv.get("cle_pooled_samples"),
        "source_f1": (a.get("source_frame") or {}).get("overall", {}).get("f1"),
        "source_far": (a.get("source_frame") or {}).get("overall", {}).get("far_per_frame"),
    }


def dominates(row: dict, base: dict) -> tuple[bool, list[str]]:
    """支配判定：四项交付关注指标（recall / hit5 / FAR / CLE p95）**全部**不劣于基线，
    且至少一项严格更优 ⇒ 该作业点把帕累托前沿向外推了（这才是真增益，不是阈值权衡）。"""
    checks = (
        ("recall", row["recall"] >= base["recall"], row["recall"] - base["recall"], 1.0),
        ("hit5", row["hit5"] >= base["hit5"], row["hit5"] - base["hit5"], 100.0),
        ("far", row["far_per_frame"] <= base["far_per_frame"],
         row["far_per_frame"] - base["far_per_frame"], 1.0),
        ("cle95", row["cle95"] <= base["cle95"], row["cle95"] - base["cle95"], 1.0),
    )
    strict = []
    for name, ok, delta, scale in checks:
        if not ok:
            return False, []
        if abs(delta) > 0:
            strict.append(f"{name} {delta * scale:+.4f}")
    return bool(strict), strict


def sign_test_p(n_up: int, n_down: int) -> float:
    """双侧符号检验 p 值（n_up / n_down 为同序配对里变好 / 变差的序列数）。"""
    n = n_up + n_down
    if n == 0:
        return 1.0
    k = min(n_up, n_down)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def paired_stats(base_art: dict, cand_art: dict, arm: str = "replay") -> dict:
    """同一集合上的**配对**逐序比较：变好/变差的序列数 + 符号检验 p 值。

    尺子A（LOSO 单序列波动）量的是「集合本身有多稳」，对**同集合配对比较**过于保守；
    尺子B（本函数）量的是「这个改动在这 24 条序列上是否**一致地**更好」⇒ 才是受控 A/B 该用的尺子。
    """
    base_seq = seq_f1(base_art, arm)
    cand_seq = seq_f1(cand_art, arm)
    n_up = n_down = n_same = 0
    deltas = []
    for seq in base_seq:
        d = (cand_seq.get(seq, 0.0) - base_seq[seq]) * 100.0
        deltas.append(d)
        if d > 1e-9:
            n_up += 1
        elif d < -1e-9:
            n_down += 1
        else:
            n_same += 1
    return {"n_up": n_up, "n_down": n_down, "n_same": n_same,
            "p": sign_test_p(n_up, n_down),
            "mean": sum(deltas) / len(deltas) if deltas else 0.0,
            "median": sorted(deltas)[len(deltas) // 2] if deltas else 0.0,
            "max_abs": max((abs(d) for d in deltas), default=0.0)}


def verdict(row: dict) -> str:
    ok = (row["f1"] >= DELIVERY["f1_min"] and row["recall"] >= DELIVERY["recall_min"]
          and row["far_per_frame"] <= DELIVERY["far_max"])
    return "**PASS**" if ok else "**FAIL**"


def gate_compare(new: dict, ref: dict, arms=("original", "stale", "kinematic", "replay")) -> list[dict]:
    """四臂逐位比对（严格相等，无容差）⇒ 扫描器是否引入额外变量。"""
    out = []
    for arm in arms:
        for key in ref[arm]:
            out.append({"arm": arm, "key": key,
                        "new": new[arm].get(key), "ref": ref[arm][key],
                        "ok": new[arm].get(key) == ref[arm][key]})
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="24 序 Δ=3 专属受控单旋钮 A/B 报告")
    p.add_argument("--root", default="runs/lookahead/fi_nogmc24/knobscan_d3", help="扫描产物根")
    p.add_argument("--ref-arms", default="", help="基线参考产物（默认 = 扫描产物里的 gate__baseline）")
    p.add_argument("--out-md", default="", help="Markdown 输出路径")
    p.add_argument("--out-json", default="", help="JSON 输出路径")
    p.add_argument("--target-delta", type=int, default=3, help="目标 Δ")
    p.add_argument("--model", default="cs_t0043", help="模型包")
    p.add_argument("--host", default="test-Super-Server", help="执行机名")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    manifest_path = root / "manifest.tsv"
    rows = []
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) < 9:
                continue
            rows.append({
                "tag": parts[0], "model": parts[1], "delta": parts[2], "flags": parts[3],
                "vals": parts[4], "rc": parts[5], "elapsed": parts[6], "path": parts[7],
                "ts": parts[8],
            })
    runs_dir = root / "runs" / args.model / f"d{args.target_delta}"
    # ⚠️ manifest 里**幂等跳过的作业也会留一行** ⇒ 必须按 tag 去重，否则作业点数虚高
    jobs = []
    seen_tags = set()
    for r in rows:
        if r["model"] != args.model or int(r["delta"]) != args.target_delta:
            continue
        if r["rc"] != "0":
            continue
        if r["tag"] in seen_tags:
            continue
        art = load_json(Path(r["path"]))
        if art is None:
            continue
        seen_tags.add(r["tag"])
        jobs.append({**r, "art": art, "row": arm_row(art)})
    gate_jobs = [j for j in jobs if j["tag"].startswith("gate__")]
    spec_jobs = [j for j in jobs if not j["tag"].startswith("gate__")]
    if not gate_jobs:
        raise SystemExit("[FATAL] 没有基线门禁产物 ⇒ 扫描器未跑 baseline 或跑挂了")
    base_job = gate_jobs[0]
    base = base_job["row"]
    base_art = base_job["art"]

    # ---- 噪声底噪：24 序 LOSO（与 8 序同法） --------------------------------
    loso = loso_sensitivity(base_art)
    swing_median_pp = loso["swing_median"] * 100.0
    swing_max_pp = loso["swing"] * 100.0

    # ---- 分组：旋钮 → 取值 → 作业 ------------------------------------------
    groups: dict[str, dict[str, dict]] = {}
    for j in spec_jobs:
        flag, val = j["flags"], j["vals"]
        groups.setdefault(flag, {})[val] = j

    # ---- 基线对照臂（oracle / stale / kinematic）-----------------------------
    oracle = arm_row(base_art, "original")
    stale = arm_row(base_art, "stale")
    kinematic = arm_row(base_art, "kinematic")

    md: list[str] = []
    md.append(f"# 24 序官方口径 · Δ={args.target_delta} 专属受控单旋钮 A/B —— {args.model}（no-GMC）")
    md.append("")
    md.append(f"- **执行机**：`{args.host}`；**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- **产物根**：`{root}`；**模型包**：{args.model}（no-GMC 特征语义）")
    md.append(f"- **参考基线产物**：`{args.ref_arms or base_job['path']}`")
    md.append("")

    # ================= 0. 口径声明 =================
    md.append("## 0. 口径声明（先读这一段）")
    md.append("")
    md.append("1. **本表是 24 序官方口径**（165,223 帧 / 133,257 GT / 24 条序列），"
              "**不是** 8 序 gate 集 ⇒ 可用于给交付档排序（8 序只覆盖 24 序的 20.39% GT / 17.55% 帧）。")
    md.append("2. **基线 = 交付档 CFG + `--replay-fov-margin-px 6`**"
              "（`run_delivery.sh` 的 `CFG=` 逐字取用 + 出框门 m=6）"
              "⇒ **基线就是 §12 交付表里的那一行**，本报告的所有 Δ 都是相对它。")
    md.append("3. **对比对象 = 同 Δ 同基线的 `replay` 臂**；`original` 臂是**零延迟天花板（oracle 上界）**，"
              "不是基线；`stale` 臂是「硬吃 Δ 滞后」的老逻辑对照。")
    md.append("4. **受控 A/B**：每个作业点**只替换一个旋钮**，其余一字不改；"
              "旋钮替换用 `sed` 全局替换且**断言该 flag 在 CFG 中只出现一次**。")
    md.append("5. **硬门禁**：baseline 复现产物必须与参考基线**四臂逐位一致**（严格相等，无容差）"
              "⇒ 不一致即说明扫描器引入了额外变量，整轮作废。")
    md.append("6. **噪声门槛**：用 24 序基线产物做 LOSO（留一序）⇒ "
              f"池化 F1 摆幅 **中位 {swing_median_pp:.4f} pp / 最大 {swing_max_pp:.4f} pp**。"
              f"判读：**ΔF1 < {swing_median_pp:.4f} pp ⇒ 噪声**；"
              f"**{swing_median_pp:.4f} ≤ ΔF1 < {swing_max_pp:.4f} pp ⇒ 疑似噪声（须看逐序归因）**；"
              f"**ΔF1 ≥ {swing_max_pp:.4f} pp ⇒ 超过最大单序列波动**。")
    md.append("7. **交付三门**：F1 ≥ 0.88 / Recall ≥ 0.85 / FAR ≤ 0.030。")
    md.append("8. **总体表 CLE 用 `cle_pooled_*`（池化）**；逐序语义见 `servo.per_sequence.<seq>.cle_p*`，"
              "两者口径不同，跨文档比较前必须标口径。")
    md.append("")

    # ================= 1. 基线门禁 =================
    md.append("## 1. 基线门禁：扫描器零额外变量（四臂逐位一致）")
    md.append("")
    ref_path = Path(args.ref_arms) if args.ref_arms else Path(base_job["path"])
    ref_art = load_json(ref_path)
    if ref_art is None:
        md.append(f"⚠️ 参考产物读不出：`{ref_path}`")
        gate_rows = []
    else:
        cmp_rows = gate_compare({a: arm_row(base_art, a) for a in
                                  ("original", "stale", "kinematic", "replay")},
                                 {a: arm_row(ref_art, a) for a in
                                  ("original", "stale", "kinematic", "replay")})
        gate_rows = cmp_rows
        n_bad = sum(1 for c in cmp_rows if not c["ok"])
        md.append(f"- 复现产物：`{base_job['path']}`")
        md.append(f"- 参考产物：`{ref_path}`")
        md.append(f"- 比对范围：**4 臂 × {len(cmp_rows) // 4} 字段 = {len(cmp_rows)} 项，严格相等无容差**")
        md.append(f"- 结果：**{len(cmp_rows) - n_bad}/{len(cmp_rows)} 一致，{n_bad} 处不一致** ⇒ "
                  + ("**PASS**（扫描器未引入额外变量）" if n_bad == 0 else "**FAIL**（扫描器引入额外变量，整轮作废）"))
        md.append("")
        if n_bad:
            md.append("| 臂 | 字段 | 新产物 | 参考 | 判定 |")
            md.append("|:--|:--|--:|--:|:--|")
            for c in cmp_rows:
                if not c["ok"]:
                    md.append(trow([c["arm"], c["key"], repr(c["new"]), repr(c["ref"]), "**DIFF**"]))
            md.append("")

    # ================= 2. 基线四臂对照 =================
    md.append(f"## 2. 基线四臂对照（Δ={args.target_delta}，基线 = 交付档 CFG + 出框门 m=6）")
    md.append("")
    md.append("| 臂 | F1% | R% | P% | FAR/帧 | TP | FP | hit@5px% | CLE p50(池) | CLE p95(池) | 交付三门 |")
    md.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
    for label, row in (("`original`（零延迟天花板 / oracle）", oracle),
                       ("`stale`（老逻辑硬吃 Δ 滞后）", stale),
                       ("`kinematic`（纯运动学前推）", kinematic),
                       ("**`replay`（基线 = 交付推荐配置）**", base)):
        md.append(trow([
            label, fmt(row["f1"], 100), fmt(row["recall"], 100), fmt(row["precision"], 100),
            fmt(row["far_per_frame"], 1, 6), str(row["tp"]), str(row["fp"]),
            fmt(row["hit5"], 100), fmt(row["cle50"], 1, 4), fmt(row["cle95"], 1, 4),
            verdict(row) if label.startswith("**") else DASH,
        ]))
    md.append("")
    md.append("**基线 `replay` 相对 oracle（零延迟天花板）的剩余缺口 = 本轮旋钮搜索的靶子：**")
    md.append("")
    md.append("| 指标 | 缺口 | 方向 |")
    md.append("|:--|--:|:--|")
    md.append(trow(["F1", fmt_signed(base["f1"] - oracle["f1"], 100) + " pp", "越大越好（缺口 = 负值）"]))
    md.append(trow(["FAR/帧", fmt_signed(base["far_per_frame"] - oracle["far_per_frame"], 1, 6),
                    "越小越好（缺口 = 正值）"]))
    md.append(trow(["hit@5px", fmt_signed(base["hit5"] - oracle["hit5"], 100) + " pp",
                    "越大越好（缺口 = 负值）"]))
    md.append(trow(["CLE p95(池)", fmt_signed(base["cle95"] - oracle["cle95"], 1, 4) + " px",
                    "越小越好（缺口 = 正值）"]))
    md.append(trow(["FP", f"{base['fp'] - oracle['fp']:+d}", "越小越好（缺口 = 正值）"]))
    md.append("")
    md.append(f"⇒ 24 序 LOSO 噪声底噪（基线产物算）：**中位 {swing_median_pp:.4f} pp / "
              f"最大 {swing_max_pp:.4f} pp**；最重单序 `{loso['worst_seq']}` 占 GT "
              f"**{loso['gt_share'][loso['worst_seq']] * 100:.2f}%**。")
    md.append("")

    # ---------- 2.1 残差分解：覆盖率 vs 定位精度 ----------
    md.append("### 2.1 残差分解：剩下的缺口是「覆盖率」还是「定位精度」？")
    md.append("")
    md.append("| 臂 | 上报帧数 applied | 未上报帧数 abstained | Recall% | hit@5px% | CLE p50(池) | CLE p95(池) | CLE p99(池) |")
    md.append("|:--|--:|--:|--:|--:|--:|--:|--:|")
    for label, arm in (("`original`（oracle 天花板）", "original"), ("`stale`", "stale"),
                       ("`kinematic`", "kinematic"), ("`replay`（基线）", "replay")):
        st = (base_art["arms"][arm].get("stream_stats") or {}).get("overall", {})
        r = arm_row(base_art, arm)
        md.append(trow([
            label, str(st.get("applied", DASH)), str(st.get("abstained", DASH)),
            fmt(r["recall"], 100), fmt(r["hit5"], 100),
            fmt(r["cle50"], 1, 4), fmt(r["cle95"], 1, 4), fmt(r["cle99"], 1, 4),
        ]))
    md.append("")
    _o_st = (base_art["arms"]["original"].get("stream_stats") or {}).get("overall", {})
    _r_st = (base_art["arms"]["replay"].get("stream_stats") or {}).get("overall", {})
    _cov_gap = (_o_st.get("applied") or 0) - (_r_st.get("applied") or 0)
    md.append(f"**读数 1 —— 覆盖率几乎不是问题**：oracle 上报 "
              f"**{_o_st.get('applied')}** 帧、replay 上报 **{_r_st.get('applied')}** 帧，"
              f"只差 **{_cov_gap} 帧**（占 oracle 上报帧数的 "
              f"{_cov_gap / max(_o_st.get('applied') or 1, 1) * 100:.2f}%）。")
    md.append("")
    md.append("**读数 2 —— 缺口在尾部，不在中位数**：CLE **p50** 只差 "
              f"{fmt_signed(base['cle50'] - oracle['cle50'], 1, 4)} px（几乎与 oracle 同级），"
              f"但 **p95 差 {fmt_signed(base['cle95'] - oracle['cle95'], 1, 4)} px**、"
              f"**p99 差 {fmt_signed(base['cle99'] - oracle['cle99'], 1, 4)} px**。")
    md.append("")
    md.append("**读数 3 —— 尾部对应的是「配准到背景结构」的少数帧**："
              f"replay 臂本轮 `replay_xcorr_ambiguous`（相关峰歧义）= "
              f"**{_r_st.get('replay_xcorr_ambiguous', DASH)}** 帧"
              f"（占上报帧 {(_r_st.get('replay_xcorr_ambiguous') or 0) / max(_r_st.get('applied') or 1, 1) * 100:.2f}%），"
              "这些帧正是「二次意见」机制的作用对象。")
    md.append("")
    md.append(f"⇒ **结论：Δ=3 下 replay 相对 oracle 的剩余缺口（hit@5px "
              f"{fmt_signed(base['hit5'] - oracle['hit5'], 100)} pp）不是「没上报」，"
              "而是「上报了但位置偏了、且偏得厉害的是少数帧」**。")
    md.append("⇒ **含义**：调**阈值类**旋钮只能在「多报/少报」之间滑动（前沿滑动），"
              "**压不掉这个尾部**；要继续压只能从「配准本身」入手（配准精度 / 歧义帧的处置策略）。")
    md.append("")

    # ================= 3. 逐旋钮结果 =================
    md.append("## 3. 逐旋钮受控 A/B 结果（每格 = 一个作业点，Δ 相对基线）")
    md.append("")
    for flag, vals in groups.items():
        title = KNOB_TITLE.get(flag, f"`{flag}`")
        md.append(f"### 3.{list(groups).index(flag) + 1} {title}")
        md.append("")
        md.append("| 取值 | F1% | ΔF1(pp) | R% | ΔR(pp) | FAR/帧 | ΔFAR | hit@5px% | Δhit5(pp) |"
                  " CLE p50(池) | Δp50 | CLE p95(池) | Δp95 | 三门 | 逐序变动序数 | 逐序最大ΔF1(pp) |")
        md.append("|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for val, job in vals.items():
            row = job["row"]
            base_seq = seq_f1(base_art)
            cur_seq = seq_f1(job["art"])
            deltas = {s: (cur_seq.get(s, 0.0) - base_seq.get(s, 0.0)) * 100.0 for s in base_seq}
            moved = sum(1 for v in deltas.values() if abs(v) > 1e-9)
            worst = max((abs(v) for v in deltas.values()), default=0.0)
            md.append(trow([
                val, fmt(row["f1"], 100), fmt_signed(row["f1"] - base["f1"], 100),
                fmt(row["recall"], 100), fmt_signed(row["recall"] - base["recall"], 100),
                fmt(row["far_per_frame"], 1, 6), fmt_signed(row["far_per_frame"] - base["far_per_frame"], 1, 6),
                fmt(row["hit5"], 100), fmt_signed(row["hit5"] - base["hit5"], 100),
                fmt(row["cle50"], 1, 4), fmt_signed(row["cle50"] - base["cle50"], 1, 4),
                fmt(row["cle95"], 1, 4), fmt_signed(row["cle95"] - base["cle95"], 1, 4),
                verdict(row), str(moved), f"{worst:.4f}",
            ]))
        md.append("")

    # ================= 4. 噪声判读 + 汇总 =================
    md.append("## 4. 噪声判读与每旋钮最优汇总")
    md.append("")
    md.append(f"> 判读门槛（24 序 LOSO）：**ΔF1 < {swing_median_pp:.4f} pp ⇒ 噪声**；"
              f"**{swing_median_pp:.4f} ≤ ΔF1 < {swing_max_pp:.4f} pp ⇒ 疑似噪声（看逐序归因）**；"
              f"**≥ {swing_max_pp:.4f} pp ⇒ 超过最大单序列波动**。")
    md.append("")
    md.append("| 旋钮 | 最优取值 | F1% | ΔF1(pp) | R% | ΔR(pp) | FAR/帧 | ΔFAR | hit@5px% | Δhit5(pp) |"
              " ΔCLE p95(池) | 三门 | 逐序变动序数 | 噪声判读 |")
    md.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|:--|")
    summary = []
    for flag, vals in groups.items():
        best = None
        for val, job in vals.items():
            row = job["row"]
            if best is None or row["f1"] > best["row"]["f1"]:
                best = {"val": val, "row": row, "job": job}
        if best is None:
            continue
        row = best["row"]
        base_seq = seq_f1(base_art)
        cur_seq = seq_f1(best["job"]["art"])
        moved = sum(1 for s in base_seq if abs(cur_seq.get(s, 0.0) - base_seq.get(s, 0.0)) > 1e-9)
        gain = row["f1"] - base["f1"]
        if gain * 100.0 < swing_median_pp:
            judge = f"噪声（< {swing_median_pp:.4f} pp）"
        elif gain * 100.0 < swing_max_pp:
            judge = f"疑似噪声（< {swing_max_pp:.4f} pp）"
        else:
            judge = "**超过最大单序列波动**"
        summary.append({"knob": flag, "val": best["val"], "row": row,
                        "d_f1": gain, "d_r": row["recall"] - base["recall"],
                        "d_far": row["far_per_frame"] - base["far_per_frame"],
                        "d_h5": row["hit5"] - base["hit5"],
                        "d_p95": row["cle95"] - base["cle95"], "moved": moved, "judge": judge})
        md.append(trow([
            KNOB_TITLE.get(flag, f"`{flag}`"), f"`{best['val']}`",
            fmt(row["f1"], 100), fmt_signed(gain, 100),
            fmt(row["recall"], 100), fmt_signed(row["recall"] - base["recall"], 100),
            fmt(row["far_per_frame"], 1, 6), fmt_signed(row["far_per_frame"] - base["far_per_frame"], 1, 6),
            fmt(row["hit5"], 100), fmt_signed(row["hit5"] - base["hit5"], 100),
            fmt_signed(row["cle95"] - base["cle95"], 1, 4), verdict(row), str(moved), judge,
        ]))
    md.append("")

    # ---------- 4.1 帕累托支配判定（真增益 vs 阈值权衡） ----------
    md.append("### 4.1 帕累托支配判定：真增益 vs 阈值权衡")
    md.append("")
    md.append("> **判据**：某作业点的 **Recall / hit@5px / FAR / CLE p95 四项全部不劣于基线**，"
              "且至少一项严格更优 ⇒ 它把帕累托前沿**向外推**了（**真增益**）。")
    md.append("> 否则它只是沿着既有前沿**滑动**（一涨一跌的阈值权衡）⇒ **不构成能力增益**。")
    md.append("> **符号约定**：本节所有 Δ 列一律 = **变体 − 基线**；"
              "Recall / hit@5px **越高越好**，FAR / CLE p95 **越低越好**（故它们的严格更优项显示为**负值**）。")
    md.append("")
    md.append("| 旋钮 | 取值 | ΔRecall(pp) | Δhit5(pp) | ΔFAR | ΔCLE p95(px) | 支配基线？ | 性质（严格更优项） |")
    md.append("|:--|--:|--:|--:|--:|--:|:--|:--|")
    n_dom = 0
    for flag, val, job in sorted(
        ((f, v, j) for f, vs in groups.items() for v, j in vs.items()),
        key=lambda t: -t[2]["row"]["f1"],
    ):
            row = job["row"]
            dom, strict = dominates(row, base)
            n_dom += int(dom)
            md.append(trow([
                KNOB_TITLE.get(flag, f"`{flag}`"), f"`{val}`",
                fmt_signed(row["recall"] - base["recall"], 100),
                fmt_signed(row["hit5"] - base["hit5"], 100),
                fmt_signed(row["far_per_frame"] - base["far_per_frame"], 1, 6),
                fmt_signed(row["cle95"] - base["cle95"], 1, 4),
                "**是**" if dom else "否",
                ("**支配（真增益）**：" + "；".join(strict)) if dom else "前沿滑动（阈值权衡）",
            ]))
    md.append("")
    _n_var = sum(len(v) for v in groups.values())
    md.append(f"⇒ **支配基线的作业点：{n_dom} / {_n_var}**"
              f"（分母 = **{_n_var} 个单旋钮变体**，**不含**基线复现 `gate__baseline`）。")
    if n_dom == 0:
        md.append("⇒ **没有任何单旋钮能把帕累托前沿向外推** ⇒ 本轮**全部差异都是阈值权衡**，"
                  "**不构成能力增益**，**不得据此改交付档**。")
    md.append("")

    # ---------- 4.2 配对符号检验（尺子B：同集合上的逐序一致性） ----------
    md.append("### 4.2 配对符号检验（尺子B：同一 24 序集合上的逐序一致性）")
    md.append("")
    md.append("> **为什么要有尺子B**：尺子A（LOSO 单序列波动）量的是「**集合本身**有多稳」，"
              "对**同一集合上的配对 A/B** 过于保守（两个配置都在同样 24 条序列上评过，"
              "配对差的不确定性远小于集合重采样的不确定性）。")
    md.append("> 尺子B 直接问「这个改动在这 24 条序列上是否**一致地**更好」："
              "逐序配对 ΔF1 的**变好 / 变差序列数** + **双侧符号检验 p 值**。")
    md.append("> **判定**：p < 0.05 且**四项池化指标全部不劣** ⇒ 该改动是**真增益**（可以进交付档）；"
              "否则只作机理线索。")
    md.append("")
    md.append("| 旋钮 | 取值 | 变好序数 | 变差序数 | 不变序数 | 符号检验 p | 逐序 ΔF1 中位数(pp) | 逐序最大\|ΔF1\|(pp) | 池化 ΔF1(pp) | 判定 |")
    md.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
    n_confirmed = 0
    confirmed = []
    for flag, val, job in sorted(
        ((f, v, j) for f, vs in groups.items() for v, j in vs.items()),
        key=lambda t: -t[2]["row"]["f1"],
    ):
        st = paired_stats(base_art, job["art"])
        row = job["row"]
        dom, _ = dominates(row, base)
        if st["p"] < 0.05 and dom:
            judge = "**真增益（p<0.05 且四项不劣）**"
            n_confirmed += 1
            confirmed.append((flag, val, st, row))
        elif st["p"] < 0.05:
            judge = "方向显著但**有指标变差**（阈值权衡）"
        elif dom:
            judge = "四项不劣但**逐序不一致**（不足以判定）"
        else:
            judge = "噪声"
        md.append(trow([
            KNOB_TITLE.get(flag, f"`{flag}`"), f"`{val}`",
            str(st["n_up"]), str(st["n_down"]), str(st["n_same"]),
            f"{st['p']:.4f}", f"{st['median']:+.4f}", f"{st['max_abs']:.4f}",
            fmt_signed(row["f1"] - base["f1"], 100), judge,
        ]))
    md.append("")
    _n_var_b = sum(len(v) for v in groups.values())
    md.append(f"⇒ **尺子B 判定为「真增益」的作业点：{n_confirmed} / {_n_var_b}**"
              f"（分母 = **{_n_var_b} 个单旋钮变体**，**不含**基线复现 `gate__baseline`）。")
    if confirmed:
        md.append("")
        for flag, val, st, row in confirmed:
            md.append(f"- **`{flag} = {val}`**：逐序变好 **{st['n_up']}/{st['n_up'] + st['n_down']}** 条，"
                      f"符号检验 p = **{st['p']:.4f}**；池化 ΔF1 {fmt_signed(row['f1'] - base['f1'], 100)} pp、"
                      f"ΔFAR {fmt_signed(row['far_per_frame'] - base['far_per_frame'], 1, 6)}、"
                      f"Δhit5 {fmt_signed(row['hit5'] - base['hit5'], 100)} pp、"
                      f"ΔCLE p95 {fmt_signed(row['cle95'] - base['cle95'], 1, 4)} px。")
    md.append("")

    # ================= 5. 逐序列归因（最优候选） =================
    md.append("## 5. 逐序列归因（基线 vs 最优候选）")
    md.append("")
    if not summary:
        md.append("（无作业点）")
    else:
        cand = max(summary, key=lambda s: s["d_f1"])
        cand_art = groups[cand["knob"]][cand["val"]]["art"]
        base_seq = seq_f1(base_art)
        cand_seq = seq_f1(cand_art)
        base_hit = {s: v["hit_rate_5px"] for s, v in base_art["arms"]["replay"]["servo"]["per_sequence"].items()}
        cand_hit = {s: v["hit_rate_5px"] for s, v in cand_art["arms"]["replay"]["servo"]["per_sequence"].items()}
        base_p95 = {s: v.get("cle_p95") for s, v in base_art["arms"]["replay"]["servo"]["per_sequence"].items()}
        cand_p95 = {s: v.get("cle_p95") for s, v in cand_art["arms"]["replay"]["servo"]["per_sequence"].items()}
        gt = {s: v["gt"] for s, v in base_art["arms"]["replay"]["current_frame"]["per_sequence"].items()}
        gt_total = sum(gt.values())
        md.append(f"候选：**{KNOB_TITLE.get(cand['knob'], cand['knob'])} = `{cand['val']}`**"
                  f"（ΔF1 {fmt_signed(cand['d_f1'], 100)} pp / ΔFAR {fmt_signed(cand['d_far'], 1, 6)} / "
                  f"Δhit5 {fmt_signed(cand['d_h5'], 100)} pp / ΔCLE p95 {fmt_signed(cand['d_p95'], 1, 4)} px）")
        md.append("")
        md.append("| 序列 | GT 占比% | ΔF1(pp) | Δhit5(pp) | ΔCLE p95(px) |")
        md.append("|:--|--:|--:|--:|--:|")
        seq_rows = []
        for s in sorted(base_seq):
            df1 = (cand_seq.get(s, 0.0) - base_seq.get(s, 0.0)) * 100.0
            dh5 = (cand_hit.get(s, 0.0) - base_hit.get(s, 0.0)) * 100.0
            b95, c95 = base_p95.get(s), cand_p95.get(s)
            d95 = (c95 - b95) if (b95 is not None and c95 is not None) else None
            seq_rows.append((s, gt.get(s, 0) / gt_total * 100.0, df1, dh5, d95))
        for s, share, df1, dh5, d95 in sorted(seq_rows, key=lambda r: -abs(r[2])):
            md.append(trow([s, f"{share:.2f}", f"{df1:+.4f}", f"{dh5:+.4f}",
                            f"{d95:+.4f}" if d95 is not None else DASH]))
        md.append("")
        n_moved = sum(1 for r in seq_rows if abs(r[2]) > 1e-9)
        md.append(f"⇒ 候选只作用在 **{n_moved}/{len(seq_rows)}** 条序列上；"
                  f"单序列最大 |ΔF1| = **{max((abs(r[2]) for r in seq_rows), default=0.0):.4f} pp**。")
        md.append("")

    # ================= 6. 结论 =================
    md.append("## 6. 结论")
    md.append("")
    if summary:
        beyond = [s for s in summary if s["d_f1"] * 100.0 >= swing_max_pp]
        suspected = [s for s in summary
                     if swing_median_pp <= s["d_f1"] * 100.0 < swing_max_pp]
        noise = [s for s in summary if s["d_f1"] * 100.0 < swing_median_pp]
        md.append(f"- **超过最大单序列波动（ΔF1 ≥ {swing_max_pp:.4f} pp）**："
                  + (("、".join(f"`{KNOB_TITLE.get(s['knob'], s['knob'])}={s['val']}`"
                                for s in beyond) if beyond else "**无**"))
                  + " ⇒ 这类改动才可以直接进交付档。")
        md.append(f"- **疑似噪声（{swing_median_pp:.4f} ≤ ΔF1 < {swing_max_pp:.4f} pp）**："
                  + (("、".join(f"`{KNOB_TITLE.get(s['knob'], s['knob'])}={s['val']}`"
                                for s in suspected) if suspected else "**无**"))
                  + " ⇒ 只作机理线索，不得据此改交付档。")
        md.append(f"- **噪声（ΔF1 < {swing_median_pp:.4f} pp）**：{len(noise)} 项。")
        md.append("")
        md.append(f"⇒ **交付档 CFG（`manu/tracking/lookahead/run_delivery.sh`）本轮**"
                  f"{'**不需要**改动（无超噪声旋钮）' if not beyond else '**可以**采纳上表中的超噪声旋钮'}。")
        md.append("")

    js = {
        "root": str(root), "model": args.model, "delta": args.target_delta,
        "host": args.host, "generated": datetime.now().isoformat(timespec="seconds"),
        "baseline": base, "oracle": oracle, "stale": stale, "kinematic": kinematic,
        "ref_arms": str(ref_path),
        "gate": {"total": len(gate_rows),
                 "bad": sum(1 for c in gate_rows if not c["ok"]),
                 "rows": gate_rows},
        "loso": {"full_f1": loso["full_f1"], "swing_pp": swing_max_pp,
                 "swing_median_pp": swing_median_pp, "worst_seq": loso["worst_seq"],
                 "max_gt_share": loso["max_gt_share"], "max_gt_seq": loso["max_gt_seq"],
                 "worst_gt_share": loso["gt_share"][loso["worst_seq"]]},
        "summary": summary,
        "paired": [
            {"knob": flag, "val": val, **paired_stats(base_art, job["art"]),
             "d_f1": job["row"]["f1"] - base["f1"]}
            for flag, vals in groups.items() for val, job in vals.items()
        ],
        "dominance": [
            {"knob": flag, "val": val,
             "dominates": dominates(job["row"], base)[0],
             "strict": dominates(job["row"], base)[1],
             "d_recall": job["row"]["recall"] - base["recall"],
             "d_hit5": job["row"]["hit5"] - base["hit5"],
             "d_far": job["row"]["far_per_frame"] - base["far_per_frame"],
             "d_cle95": job["row"]["cle95"] - base["cle95"]}
            for flag, vals in groups.items() for val, job in vals.items()
        ],
        "jobs": [{k: j[k] for k in ("tag", "flags", "vals", "rc", "elapsed", "ts", "path")}
                 for j in jobs],
        # 全部变体（不只每旋钮最优）⇒ 下游报告可做完整受控 A/B 表
        "variants": [
            {"knob": flag, "val": val, "tag": job["tag"], "flags": job["flags"],
             "vals": job["vals"], "rc": job["rc"], "elapsed": job["elapsed"], "ts": job["ts"],
             "path": job["path"], "row": job["row"],
             "d_f1": job["row"]["f1"] - base["f1"],
             "d_r": job["row"]["recall"] - base["recall"],
             "d_far": job["row"]["far_per_frame"] - base["far_per_frame"],
             "d_h5": job["row"]["hit5"] - base["hit5"],
             "d_p95": job["row"]["cle95"] - base["cle95"],
             "dominates": dominates(job["row"], base)[0]}
            for flag, vals in groups.items() for val, job in vals.items()
        ],
    }
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(js, ensure_ascii=False, indent=2))
        print(f"[WROTE] {args.out_json}")
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text("\n".join(md) + "\n")
        print(f"[WROTE] {args.out_md}")
    print(f"[SUMMARY] 作业点 {len(spec_jobs)} 个；门禁 "
          f"{len(gate_rows) - sum(1 for c in gate_rows if not c['ok'])}/{len(gate_rows)} 一致；"
          f"噪声底噪 中位 {swing_median_pp:.4f}pp / 最大 {swing_max_pp:.4f}pp")


if __name__ == "__main__":
    main()
