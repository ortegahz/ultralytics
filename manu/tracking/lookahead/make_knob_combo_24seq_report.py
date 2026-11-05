#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""24 序官方口径 · Δ=3 **多旋钮组合** A/B → 报告生成器（cs_t0043 / no-GMC）。

与 ``make_knob_scan_24seq_report.py``（**单旋钮**受控 A/B）是**两种不同的实验设计**：
那份生成器的口径声明写死「每个作业点**只替换一个旋钮**」；组合扫描**按构造就是多变量**
⇒ 沿用它会把方法学写成假陈述 ⇒ 本文件单独成文（工具函数复用，不重复实现）。

只读产物 JSON（``arms.json``），**所有数字直接读出、不重算**；差值 = 两个读出值相减。

产物布局（扫描器 ``server_overnight/run_knob_scan_d3.sh extra`` 写入）::

    <root>/runs/<model>/d<delta>/<tag>/arms.json   # 每个 tag = 一次运行
    <root>/manifest.tsv                             # tag|model|delta|flags|vals|rc|elapsed|path|ts

用法::

    python manu/tracking/lookahead/make_knob_combo_24seq_report.py \
        --root runs/lookahead/fi_nogmc24/knobscan_p3c \
        --ref-arms runs/lookahead/fovgate24nogmc/runs/cs_t0043/d3/fov__m6/arms.json \
        --single-report runs/lookahead/fi_nogmc24/knobscan_d3/report.json \
        --out-md runs/lookahead/fi_nogmc24/knobscan_p3c/report.md \
        --out-json runs/lookahead/fi_nogmc24/knobscan_p3c/report.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from make_knob_scan_it2_report import loso_sensitivity, seq_f1
from make_knob_scan_24seq_report import (
    DELIVERY,
    arm_row,
    dominates,
    fmt,
    fmt_signed,
    gate_compare,
    paired_stats,
    trow,
    verdict,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="24 序 多旋钮组合 A/B 报告（--target-delta 指定 Δ）")
    p.add_argument("--root", default="runs/lookahead/fi_nogmc24/knobscan_p3c", help="组合扫描产物根")
    p.add_argument("--ref-arms", default="runs/lookahead/fovgate24nogmc/runs/cs_t0043/d3/fov__m6/arms.json",
                   help="基线参考产物（交付档 CFG + 出框门 m=6）")
    p.add_argument("--single-report", default="runs/lookahead/fi_nogmc24/knobscan_d3/report.json",
                   help="单旋钮扫描报告（加性检验的单点参照）")
    p.add_argument("--out-md", default="", help="Markdown 输出路径")
    p.add_argument("--out-json", default="", help="JSON 输出路径")
    p.add_argument("--target-delta", type=int, default=3, help="目标 Δ")
    p.add_argument("--model", default="cs_t0043", help="模型包")
    p.add_argument("--host", default="test-Super-Server", help="执行机名")
    return p.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text())


def parse_manifest(path: Path) -> list[dict]:
    jobs = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        jobs.append({"tag": parts[0], "model": parts[1], "delta": parts[2], "flags": parts[3],
                     "vals": parts[4], "rc": int(parts[5]), "elapsed": int(parts[6]),
                     "path": parts[7], "ts": parts[8]})
    return jobs


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    manifest_path = root / "manifest.tsv"
    if not manifest_path.exists():
        raise SystemExit(f"[FATAL] 缺 manifest: {manifest_path}")
    jobs = parse_manifest(manifest_path)
    runs_dir = root / "runs" / args.model / f"d{args.target_delta}"
    ref_path = Path(args.ref_arms)

    def job_art(job: dict) -> dict | None:
        p = Path(job["path"])
        if not p.is_absolute():
            p = root / "runs" / args.model / f"d{args.target_delta}" / job["tag"] / "arms.json"
        if not p.exists():
            p = runs_dir / job["tag"] / "arms.json"
        return load_json(p) if p.exists() else None

    # ---- baseline：门禁产物（扫描器自带复现）与参考产物双源 ---------------------
    gate_jobs = [j for j in jobs if j["tag"] == "gate__baseline" and j["rc"] == 0]
    if not gate_jobs:
        raise SystemExit("[FATAL] 缺 gate__baseline 作业（扫描器零额外变量门禁未跑）")
    base_art = job_art(gate_jobs[0])
    ref_art = load_json(ref_path)
    base = arm_row(base_art)
    oracle = arm_row(base_art, "original")
    stale = arm_row(base_art, "stale")
    kinematic = arm_row(base_art, "kinematic")

    gate_rows = gate_compare({a: arm_row(base_art, a) for a in
                              ("original", "stale", "kinematic", "replay")},
                             {a: arm_row(ref_art, a) for a in
                              ("original", "stale", "kinematic", "replay")})
    gate_bad = sum(1 for c in gate_rows if not c["ok"])

    # ★ 噪声门槛必须与单旋钮报告**同一口径**：摆幅 = max(loo) − min(loo)（跨度），
    #   中位 = median|loo − full|（偏离中位数）⇒ 直接取 loso_sensitivity 的定义，不另立口径。
    loso = loso_sensitivity(base_art)
    swing_max = loso["swing"] * 100.0
    swing_median = loso["swing_median"] * 100.0

    # ---- 单旋钮参照（加性检验用） ---------------------------------------------
    single: dict[tuple[str, str], dict] = {}
    single_path = Path(args.single_report)
    if single_path.exists():
        for v in load_json(single_path).get("variants", []):
            single[(str(v["knob"]), str(v["val"]))] = v

    combos = []
    for job in jobs:
        if job["tag"] == "gate__baseline" or job["rc"] != 0:
            continue
        if job["model"] != args.model or int(job["delta"]) != args.target_delta:
            continue
        art = job_art(job)
        if art is None:
            continue
        row = arm_row(art)
        flags = job["flags"].split(",") if job["flags"] not in ("", "-") else []
        vals = job["vals"].split(",") if job["vals"] not in ("", "-") else []
        pairs = list(zip(flags, vals))
        dom, strict = dominates(row, base)
        ps_ = paired_stats(base_art, art)
        # 加性检验：组合 ΔF1 vs 各单点 ΔF1 之和
        singles = []
        for flag, val in pairs:
            ref = single.get((flag, val))
            singles.append({"knob": flag, "val": val,
                            "d_f1": (ref["d_f1"] * 100.0) if ref else None,
                            "found": ref is not None})
        sum_single = sum(s["d_f1"] for s in singles if s["d_f1"] is not None)
        have_all = all(s["found"] for s in singles) and bool(singles)
        d_f1_pp = (row["f1"] - base["f1"]) * 100.0
        combos.append({
            "tag": job["tag"], "flags": job["flags"], "vals": job["vals"], "rc": job["rc"],
            "elapsed": job["elapsed"], "ts": job["ts"], "row": row, "pairs": pairs,
            "d_f1_pp": d_f1_pp, "sum_single_pp": sum_single if have_all else None,
            "interaction_pp": (d_f1_pp - sum_single) if have_all else None,
            "dominates": dom, "strict": strict,
            "d_recall_pp": (row["recall"] - base["recall"]) * 100.0,
            "d_hit5_pp": (row["hit5"] - base["hit5"]) * 100.0,
            "d_far": row["far_per_frame"] - base["far_per_frame"],
            "d_cle95": row["cle95"] - base["cle95"],
            "paired": ps_, "singles": singles,
        })

    # =======================================================================
    md = []
    md.append(f"# 24 序官方口径 · Δ={args.target_delta} **多旋钮组合** A/B —— {args.model}（no-GMC）")
    md.append("")
    md.append(f"- **执行机**：`{args.host}`；**生成时间**：{datetime.now().isoformat(timespec='seconds')}")
    md.append(f"- **产物根**：`{root}`；**模型包**：{args.model}（no-GMC 特征语义）；**Δ = {args.target_delta}**")
    md.append(f"- **参考基线产物**：`{ref_path}`")
    md.append(f"- **单旋钮参照**（加性检验）：`{single_path}`")
    md.append("")
    md.append("## 0. 口径声明（先读这一段）")
    md.append("")
    md.append("1. **本表是 24 序官方口径**（24 条序列 / 133,257 GT），**不是** 8 序 gate 集 ⇒ 可用于给交付档排序。")
    md.append("2. **基线 = 交付档 CFG + `--replay-fov-margin-px 6`**（`run_delivery.sh` 的 `CFG=` 逐字取用 + 出框门 m=6）⇒ "
              "**基线就是交付表里的那一行**，本报告的所有 Δ 都是相对它。")
    md.append("3. **对比对象 = 同 Δ 同基线的 `replay` 臂**；`original` 臂是**零延迟天花板（oracle 上界）**，不是基线；"
              "`stale` 臂是「硬吃 Δ 滞后」的老逻辑对照。")
    md.append("4. ⚠️ **本表是「多旋钮组合」实验，与单旋钮受控 A/B 是两种不同设计**："
              "每个作业点**同时**替换 2~4 个旋钮 ⇒ **按构造就是多变量**，"
              "**不能**把它当作「单旋钮受控 A/B」来解读，也**不能**与单旋钮表并列引用。")
    md.append("5. **入选依据**：每个旋钮都**先在单旋钮扫描里各自**被验证过"
              "（`runs/lookahead/fi_nogmc24/knobscan_d3`），只有**帕累托占优**（recall/hit@5px/FAR/CLE p95 四项全不劣且至少一项严格更优）"
              "或**显著压 FAR** 的旋钮才进入组合 ⇒ 组合只做「已各自证明不劣的旋钮」的叠加。")
    md.append("6. **硬门禁**：baseline 复现产物必须与参考产物**四臂逐位一致**（严格相等，无容差）⇒ 不一致即整轮作废。")
    md.append(f"7. **噪声门槛（尺子A）**：用 24 序基线产物做 LOSO（留一序）⇒ 池化 F1 摆幅 **中位 {swing_median:.4f} pp / "
              f"最大 {swing_max:.4f} pp**。判读：**ΔF1 < {swing_median:.4f} pp ⇒ 噪声**；"
              f"**{swing_median:.4f} ≤ ΔF1 < {swing_max:.4f} pp ⇒ 疑似噪声（须看逐序归因）**；"
              f"**ΔF1 ≥ {swing_max:.4f} pp ⇒ 超过最大单序列波动**。")
    md.append("8. **尺子B（配对符号检验）**：同 24 序**配对**比较，变好序数 / 变差序数 / 双侧 p 值 ⇒ "
              "量的是「这个改动在这 24 条序列上是否**一致地**更好」。")
    md.append(f"9. **交付三门**：F1 ≥ {DELIVERY['f1_min']} / Recall ≥ {DELIVERY['recall_min']} / "
              f"FAR ≤ {DELIVERY['far_max']}。")
    md.append("10. **总体表 CLE 用 `cle_pooled_*`（池化）**；`servo.per_sequence.<seq>.cle_p*` 是逐序语义 ⇒ 两者不可混用。")
    md.append("11. **加性检验**：组合 ΔF1 − Σ(各单点 ΔF1) = **交互项**；0 ⇒ 纯可加（各旋钮互不干扰）；"
              "≠ 0 ⇒ 存在交互，**单旋钮表不能用来外推组合收益**。")
    md.append("")
    md.append("## 1. 基线门禁：扫描器零额外变量（四臂逐位一致）")
    md.append("")
    md.append(f"- 复现产物：`{runs_dir / 'gate__baseline' / 'arms.json'}`")
    md.append(f"- 参考产物：`{ref_path}`")
    md.append(f"- 比对范围：**4 臂 × 17 字段 = {len(gate_rows)} 项，严格相等无容差**")
    md.append(f"- 结果：**{len(gate_rows) - gate_bad}/{len(gate_rows)} 一致，{gate_bad} 处不一致** ⇒ "
              f"**{'PASS' if gate_bad == 0 else 'FAIL'}**（扫描器未引入额外变量）")
    md.append("")
    md.append(f"## 2. 基线四臂对照（Δ={args.target_delta}，基线 = 交付档 CFG + 出框门 m=6）")
    md.append("")
    md.append(trow(["臂", "F1%", "R%", "P%", "FAR/帧", "TP", "FP", "hit@5px%", "CLE p50(池)", "CLE p95(池)", "交付三门"]))
    md.append(trow(["--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", ":--"]))
    for name, row, role in (("`original`（零延迟天花板 / oracle）", oracle, ""),
                            ("`stale`（老逻辑硬吃 Δ 滞后）", stale, ""),
                            ("`kinematic`（纯运动学前推）", kinematic, ""),
                            ("**`replay`（基线 = 交付推荐配置）**", base, "")):
        md.append(trow([name, fmt(row["f1"], 100), fmt(row["recall"], 100), fmt(row["precision"], 100),
                        fmt(row["far_per_frame"], 1, 6), str(row["tp"]), str(row["fp"]),
                        fmt(row["hit5"], 100), fmt(row["cle50"], 1, 4), fmt(row["cle95"], 1, 4),
                        verdict(row) if "replay" in name else ""]))
    md.append("")
    md.append("**基线 `replay` 相对 oracle（零延迟天花板）的剩余缺口 = 本轮组合搜索的靶子：**")
    md.append("")
    md.append(trow(["指标", "缺口", "方向"]))
    md.append(trow(["--:", "--:", ":--"]))
    md.append(trow(["F1", f"{(base['f1'] - oracle['f1']) * 100:+.4f} pp", "越大越好"]))
    md.append(trow(["FAR/帧", f"{base['far_per_frame'] - oracle['far_per_frame']:+.6f}", "越小越好"]))
    md.append(trow(["hit@5px", f"{(base['hit5'] - oracle['hit5']) * 100:+.4f} pp", "越大越好"]))
    md.append(trow(["CLE p95(池)", f"{base['cle95'] - oracle['cle95']:+.4f} px", "越小越好"]))
    md.append("")
    md.append("## 3. 逐组合结果（每行 = 一个组合作业点，Δ 相对基线）")
    md.append("")
    md.append(trow(["组合", "旋钮数", "F1%", "ΔF1(pp)", "FAR/帧", "ΔFAR", "hit@5px%", "Δhit5(pp)",
                    "CLE p95(池)", "Δp95", "三门", "支配基线？", "配对 p", "变好/变差序"]))
    md.append(trow(["--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:"]))
    for c in sorted(combos, key=lambda x: -x["d_f1_pp"]):
        md.append(trow([f"`{c['tag']}`", str(len(c["pairs"])),
                        fmt(c["row"]["f1"], 100), f"{c['d_f1_pp']:+.4f}",
                        fmt(c["row"]["far_per_frame"], 1, 6), f"{c['d_far']:+.6f}",
                        fmt(c["row"]["hit5"], 100), f"{c['d_hit5_pp']:+.4f}",
                        fmt(c["row"]["cle95"], 1, 4), f"{c['d_cle95']:+.4f}",
                        verdict(c["row"]), "**是**" if c["dominates"] else "否",
                        f"{c['paired']['p']:.4f}", f"{c['paired']['n_up']}/{c['paired']['n_down']}"]))
    md.append("")
    md.append("**逐组合的旋钮明细：**")
    md.append("")
    md.append(trow(["组合", "旋钮串", "用时(s)"]))
    md.append(trow(["--:", ":--", "--:"]))
    for c in sorted(combos, key=lambda x: -x["d_f1_pp"]):
        md.append(trow([f"`{c['tag']}`", "`" + c["flags"].replace(",", " ` / ` ") + "`", str(c["elapsed"])]))
    md.append("")
    md.append("## 4. 加性检验：组合收益 vs 单点收益之和")
    md.append("")
    if single_path.exists():
        md.append(f"> 单点 ΔF1 取自单旋钮扫描 `{single_path}`（同一基线、同一 Δ、同一口径）；"
                  "**交互项 = 组合 ΔF1 − Σ 单点 ΔF1**。0 ⇒ 纯可加；≠ 0 ⇒ 存在交互。")
    else:
        md.append(f"> ⚠️ **本 Δ 没有单旋钮参照**（单旋钮扫描只覆盖 Δ=3；参照路径 `{single_path}` 不存在）"
                  "⇒ **加性检验不可算（§4 显示 —）**；"
                  "**切勿**把 Δ=3 的单点行混进本 Δ 的组合分析（那不是同一基线、也不是同一 Δ）。")
    md.append("")
    md.append(trow(["组合", "组合 ΔF1(pp)", "Σ 单点 ΔF1(pp)", "交互项(pp)", "各单点 ΔF1(pp)"]))
    md.append(trow(["--:", "--:", "--:", "--:", ":--"]))
    for c in sorted(combos, key=lambda x: -x["d_f1_pp"]):
        singles_txt = "、".join(f"`{s['knob']}={s['val']}` {s['d_f1']:+.4f}"
                                for s in c["singles"] if s["d_f1"] is not None) or "—"
        md.append(trow([f"`{c['tag']}`", f"{c['d_f1_pp']:+.4f}",
                        f"{c['sum_single_pp']:+.4f}" if c["sum_single_pp"] is not None else "—",
                        f"{c['interaction_pp']:+.4f}" if c["interaction_pp"] is not None else "—",
                        singles_txt]))
    md.append("")
    md.append("## 5. 帕累托支配判定 + 噪声判读")
    md.append("")
    md.append(f"> **判据**：Recall / hit@5px / FAR / CLE p95 **四项全部不劣于基线**，且至少一项严格更优 ⇒ 把帕累托前沿**向外推**（真增益）；"
              f"否则只是沿既有前沿**滑动**（阈值权衡）⇒ 不构成能力增益。")
    md.append(f"> **噪声门槛（尺子A）**：ΔF1 < {swing_median:.4f} pp ⇒ 噪声；≥ {swing_max:.4f} pp ⇒ 超过最大单序列波动。")
    md.append("")
    md.append(trow(["组合", "ΔR(pp)", "Δhit5(pp)", "ΔFAR", "ΔCLE p95(px)", "支配基线？", "性质", "噪声判读"]))
    md.append(trow(["--:", "--:", "--:", "--:", "--:", "--:", ":--", ":--"]))
    for c in sorted(combos, key=lambda x: -x["d_f1_pp"]):
        if c["d_f1_pp"] < swing_median:
            noise = f"噪声（< {swing_median:.4f} pp）"
        elif c["d_f1_pp"] < swing_max:
            noise = f"疑似噪声（< {swing_max:.4f} pp）"
        else:
            noise = f"**超单序列波动（≥ {swing_max:.4f} pp）**"
        md.append(trow([f"`{c['tag']}`", f"{c['d_recall_pp']:+.4f}", f"{c['d_hit5_pp']:+.4f}",
                        f"{c['d_far']:+.6f}", f"{c['d_cle95']:+.4f}",
                        "**是**" if c["dominates"] else "否",
                        ("支配（真增益）：" + "；".join(c["strict"])) if c["dominates"] else "前沿滑动（阈值权衡）",
                        noise]))
    md.append("")

    # ---- 结论 ----------------------------------------------------------------
    dom = [c for c in combos if c["dominates"]]
    beyond = [c for c in combos if c["d_f1_pp"] >= swing_max]
    beyond_mid = [c for c in combos if swing_median <= c["d_f1_pp"] < swing_max]
    sig = [c for c in combos if c["paired"]["p"] < 0.05]
    best = max(combos, key=lambda x: x["d_f1_pp"]) if combos else None
    md.append("## 6. 结论")
    md.append("")
    md.append(f"1. **作业点**：{len(combos)} 个组合（旋钮数 {sorted({len(c['pairs']) for c in combos})}），"
              f"baseline 门禁 **{len(gate_rows) - gate_bad}/{len(gate_rows)} 一致 ⇒ "
              f"{'PASS' if gate_bad == 0 else 'FAIL'}**。")
    if best:
        md.append(f"2. **最优组合 = `{best['tag']}`**：ΔF1 **{best['d_f1_pp']:+.4f} pp**、"
                  f"ΔFAR **{best['d_far']:+.6f}**、Δhit@5px **{best['d_hit5_pp']:+.4f} pp**、"
                  f"ΔCLE p95 **{best['d_cle95']:+.4f} px**；帕累托支配基线 = "
                  f"**{'是' if best['dominates'] else '否'}**；配对 p = **{best['paired']['p']:.4f}**。")
    md.append(f"3. **帕累托支配基线的组合数 = {len(dom)}/{len(combos)}**"
              + ("（" + "、".join(f"`{c['tag']}`" for c in dom) + "）" if dom else "") + "。")
    md.append(f"4. **噪声判读**：ΔF1 ≥ 尺子A 最大摆幅（{swing_max:.4f} pp）的组合 = **{len(beyond)}** 个"
              + ("（" + "、".join(f"`{c['tag']}`" for c in beyond) + "）" if beyond else "")
              + f"；疑似噪声区（{swing_median:.4f}~{swing_max:.4f} pp）= **{len(beyond_mid)}** 个"
              + ("（" + "、".join(f"`{c['tag']}`" for c in beyond_mid) + "）" if beyond_mid else "") + "。")
    md.append(f"5. **配对符号检验 p < 0.05 的组合 = {len(sig)}** 个"
              + ("（" + "、".join(f"`{c['tag']}`" for c in sig) + "）" if sig else "") + "。")
    md.append(f"6. **加性**：交互项见 §4；组合 ΔF1 与 Σ 单点 ΔF1 的关系由该表逐行给出。")
    md.append("")
    md.append("> **交付裁定**：本表是 **P3 探索**，不是交付口径 —— 交付档 CFG **未改动**；"
              f"本节结论只回答「把已各自验证不劣的旋钮叠加，能不能把 Δ={args.target_delta} 的剩余缺口再压一点」。")
    md.append("")

    if args.out_json:
        js = {
            "root": str(root), "model": args.model, "delta": args.target_delta,
            "host": args.host, "generated": datetime.now().isoformat(timespec="seconds"),
            "ref_arms": str(ref_path), "single_report": str(single_path),
            "baseline": base, "oracle": oracle, "stale": stale, "kinematic": kinematic,
            "gate": {"total": len(gate_rows), "bad": gate_bad, "rows": gate_rows},
            "loso": {"swing_median_pp": swing_median, "swing_max_pp": swing_max,
                     "worst_seq": loso.get("worst_seq"), "max_gt_share": loso.get("max_gt_share")},
            "combos": [{k: c[k] for k in ("tag", "flags", "vals", "rc", "elapsed", "ts",
                                          "d_f1_pp", "sum_single_pp", "interaction_pp",
                                          "dominates", "strict", "d_recall_pp", "d_hit5_pp",
                                          "d_far", "d_cle95", "singles")}
                       | {"row": c["row"], "paired": c["paired"]} for c in combos],
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(js, ensure_ascii=False, indent=2))
        print(f"[WROTE] {args.out_json}")
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text("\n".join(md) + "\n")
        print(f"[WROTE] {args.out_md}")


if __name__ == "__main__":
    main()
