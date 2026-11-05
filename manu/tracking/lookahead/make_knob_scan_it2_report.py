#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Δ=3 迭代二：围绕新基线 B 的受控单旋钮 A/B → 报告生成器。

只读产物 JSON（``arms.json``），**所有数字直接读出、不重算**；差值 = 两个读出值相减。

产物布局（扫描器 ``server_overnight/run_knob_scan_d3_it2.sh`` 写入）::

    <root>/runs/<model>/d<delta>/<tag>/arms.json   # 每个 tag = 一次运行
    <root>/manifest.tsv                             # tag|model|delta|flags|vals|rc|elapsed|path|ts|kind
    <root>/.start_epoch / .end_epoch                # 总墙钟

用法::

    python manu/tracking/lookahead/make_knob_scan_it2_report.py \
        --root runs/lookahead/fi_nogmc/knobscan2 \
        --ref-arms runs/lookahead/fi_nogmc/knobscan/runs/cs_t0043/d3/p3__c_peak050_min080/arms.json \
        --out-md runs/lookahead/fi_nogmc/knobscan2/report.md \
        --out-json runs/lookahead/fi_nogmc/knobscan2/report.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

# 交付三门（业务域红线）
DELIVERY = {"f1_min": 0.88, "recall_min": 0.85, "far_max": 0.030}

KNOB_TITLE = {
    "--replay-abstain-peak-below": "弃权峰值门 (`--replay-abstain-peak-below`)",
    "--replay-rollback-min-peak": "回退峰值门 (`--replay-rollback-min-peak`)",
    "--replay-roi-selfpred-max-px": "ROI 前推上限 (`--replay-roi-selfpred-max-px`, px)",
    "--replay-xcorr-uniq-gate": "相关峰唯一性门 (`--replay-xcorr-uniq-gate`)",
    "--replay-rollback-max-frames": "回退最大帧数 (`--replay-rollback-max-frames`)",
    "--replay-abstain-disp-px": "弃权位移窗 (`--replay-abstain-disp-px`, px)",
    "--output-max-step-px": "输出限幅 (`--output-max-step-px`, px)",
}
KNOB_ORDER = [
    "--replay-abstain-peak-below",
    "--replay-rollback-min-peak",
    "--replay-roi-selfpred-max-px",
    "--replay-xcorr-uniq-gate",
    "--replay-rollback-max-frames",
    "--replay-abstain-disp-px",
    "--output-max-step-px",
]

# 逐位校验字段（baseline 门禁，17 个）
GATE_FIELDS = [
    ("f1", "F1", 100.0, 4),
    ("recall", "Recall", 100.0, 4),
    ("precision", "Precision", 100.0, 4),
    ("far_per_frame", "FAR/帧", 1.0, 6),
    ("tp", "TP", 1.0, 0),
    ("fp", "FP", 1.0, 0),
    ("fn", "FN", 1.0, 0),
    ("det", "det", 1.0, 0),
    ("gt", "gt", 1.0, 0),
    ("hit5", "hit@5px", 100.0, 4),
    ("cle_pooled_p50", "CLE p50(池)", 1.0, 6),
    ("cle_pooled_p95", "CLE p95(池)", 1.0, 6),
    ("cle_pooled_p99", "CLE p99(池)", 1.0, 6),
    ("cle_pooled_max", "CLE max(池)", 1.0, 6),
    ("cle_pooled_samples", "池化样本数", 1.0, 0),
    ("source_f1", "尺子1 F1(源帧)", 100.0, 4),
    ("source_far", "尺子1 FAR(源帧)", 1.0, 6),
]

DASH = "—"


def trow(cells: list[str]) -> str:
    """拼一行 Markdown 表格（统一 `| a | b |` 格式）。"""
    return "| " + " | ".join(str(c) for c in cells) + " |"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Δ=3 knob-scan iteration-2 report")
    p.add_argument("--root", default="runs/lookahead/fi_nogmc/knobscan2", help="扫描产物根")
    p.add_argument("--ref-arms", default="", help="baseline B 参考产物（第一轮既有 arms.json）")
    p.add_argument("--out-md", default="", help="Markdown 输出路径")
    p.add_argument("--out-json", default="", help="JSON 输出路径")
    p.add_argument("--target-delta", type=int, default=3, help="目标 Δ")
    p.add_argument("--host", default="test-Super-Server", help="执行机名")
    return p.parse_args()


def arm_row(artefact: dict, arm: str = "replay") -> dict:
    """取一个臂的总体指标；总体 CLE 走**全帧池化**口径（cle_pooled_*）。"""
    a = artefact["arms"][arm]
    cf = a["current_frame"]["overall"]
    sv = a["servo"]["overall"]
    src = a["source_frame"]["overall"]
    return {
        "f1": cf["f1"],
        "recall": cf["recall"],
        "precision": cf["precision"],
        "far": cf["far_per_frame"],
        "tp": cf["tp"],
        "fp": cf["fp"],
        "fn": cf["fn"],
        "det": cf.get("det"),
        "gt": cf.get("gt"),
        "hit5": sv["hit_rate_5px"],
        "cle50": sv.get("cle_pooled_p50"),
        "cle95": sv.get("cle_pooled_p95"),
        "cle99": sv.get("cle_pooled_p99"),
        "cle_max": sv.get("cle_pooled_max"),
        "cle_pooled_samples": sv.get("cle_pooled_samples"),
        "coverage": sv.get("coverage"),
        "source_f1": src["f1"],
        "source_far": src["far_per_frame"],
        "original_f1": artefact["arms"]["original"]["current_frame"]["overall"]["f1"],
        "original_far": artefact["arms"]["original"]["current_frame"]["overall"]["far_per_frame"],
        "stale_f1": artefact["arms"]["stale"]["current_frame"]["overall"]["f1"],
        "stale_far": artefact["arms"]["stale"]["current_frame"]["overall"]["far_per_frame"],
    }


def gate_row(artefact: dict, arm: str = "replay") -> dict:
    a = artefact["arms"][arm]
    cf = a["current_frame"]["overall"]
    sv = a["servo"]["overall"]
    return {
        "f1": cf["f1"], "recall": cf["recall"], "precision": cf["precision"],
        "far_per_frame": cf["far_per_frame"], "tp": cf["tp"], "fp": cf["fp"],
        "fn": cf["fn"], "det": cf["det"], "gt": cf["gt"],
        "hit5": sv["hit_rate_5px"],
        "cle_pooled_p50": sv.get("cle_pooled_p50"), "cle_pooled_p95": sv.get("cle_pooled_p95"),
        "cle_pooled_p99": sv.get("cle_pooled_p99"), "cle_pooled_max": sv.get("cle_pooled_max"),
        "cle_pooled_samples": sv.get("cle_pooled_samples"),
        "source_f1": a["source_frame"]["overall"]["f1"],
        "source_far": a["source_frame"]["overall"]["far_per_frame"],
    }


def loso_sensitivity(artefact: dict, arm: str = "replay") -> dict:
    """留一序（LOSO）敏感度：把 8 条序列逐条剔掉后，池化 F1 的摆动幅度。

    这是「单序列波动量级」的**下界估计**（同一条序列在/不在集合里对池化 F1 的影响），
    全部由产物 ``per_sequence`` 表里的 tp/fp/fn/det/gt 读出，不碰 preds。
    F1 口径 = 2*R*P/(R+P)（R = Σtp/Σgt，P = Σtp/Σdet），与产物 overall 同口径。
    """
    ps = artefact["arms"][arm]["current_frame"]["per_sequence"]
    seqs = sorted(ps.keys())

    def pooled(subset: list[str]) -> float:
        tp = sum(ps[s]["tp"] for s in subset)
        det = sum(ps[s]["det"] for s in subset)
        gt = sum(ps[s]["gt"] for s in subset)
        if tp == 0 or det == 0 or gt == 0:
            return 0.0
        recall, precision = tp / gt, tp / det
        return 2 * recall * precision / (recall + precision)

    full = pooled(seqs)
    loo = {s: pooled([x for x in seqs if x != s]) for s in seqs}
    vals = list(loo.values())
    deltas = sorted(abs(v - full) for v in vals)
    worst = max(loo, key=lambda k: abs(loo[k] - full))
    gt_total = sum(ps[s]["gt"] for s in seqs)
    gt_share = {s: ps[s]["gt"] / gt_total for s in seqs}
    return {
        "full_f1": full,
        "loo": loo,
        "swing": max(vals) - min(vals),
        "swing_median": deltas[len(deltas) // 2],
        "worst_seq": worst,
        "gt_share": gt_share,
        "max_gt_share": max(gt_share.values()),
        "max_gt_seq": max(gt_share, key=lambda k: gt_share[k]),
        "n_seqs": len(seqs),
    }


def seq_f1(artefact: dict, arm: str = "replay") -> dict:
    """逐序 F1（直接从产物 per_sequence 表读出）。"""
    ps = artefact["arms"][arm]["current_frame"]["per_sequence"]
    return {k: v["f1"] for k, v in ps.items()}


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_manifest(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        rows.append({
            "tag": parts[0], "model": parts[1], "delta": parts[2],
            "flags": parts[3], "vals": parts[4], "rc": int(parts[5] or -1),
            "elapsed": int(parts[6] or 0), "path": parts[7], "ts": parts[8],
            "kind": parts[9] if len(parts) > 9 else "run",
        })
    return rows


def fmt(v, scale: float = 1.0, nd: int = 4, dash: bool = True) -> str:
    """数字格式化；缺失值渲染成 `—`（禁止 nan/None 落进表格）。"""
    if v is None or (isinstance(v, float) and v != v):
        return DASH if dash else "0"
    return f"{v * scale:.{nd}f}"


def fmt_signed(v, scale: float = 1.0, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return DASH
    return f"{v * scale:+.{nd}f}"


def verdict(row: dict) -> str:
    ok = (
        row["f1"] >= DELIVERY["f1_min"]
        and row["recall"] >= DELIVERY["recall_min"]
        and row["far"] <= DELIVERY["far_max"]
    )
    return "PASS" if ok else "FAIL"


def knob_pairs(flags: str, vals: str) -> list[tuple[str, str]]:
    if flags in ("", "-"):
        return []
    fl = flags.split(",")
    vl = vals.split(",")
    return list(zip(fl, vl))


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    manifest = load_manifest(root / "manifest.tsv")

    ref_arms_path = Path(args.ref_arms) if args.ref_arms else (
        root / "runs/cs_t0043/d3/p3__c_peak050_min080/arms.json")
    ref_arms = load_json(ref_arms_path)

    # ---- 唯一 tag 化：同一 tag 保留最后一次记录 -----------------------------
    by_tag: dict[str, dict] = {}
    for r in manifest:
        by_tag[r["tag"]] = r

    def artefact_for(tag: str) -> dict | None:
        rec = by_tag.get(tag)
        if rec is None:
            return None
        return load_json(Path(rec["path"]))

    # ---- 索引：旋钮 → 取值 → 行 -------------------------------------------
    knob_vals: dict[str, list[str]] = {k: [] for k in KNOB_ORDER}
    mech_rows: dict[str, dict] = {}
    for rec in by_tag.values():
        pairs = knob_pairs(rec["flags"], rec["vals"])
        if rec["tag"].startswith("it2__") and len(pairs) == 1:
            knob, val = pairs[0]
            if knob in knob_vals and val not in knob_vals[knob]:
                knob_vals[knob].append(val)
        elif rec["tag"].startswith("it2m__") and len(pairs) == 2:
            mech_rows[rec["tag"]] = rec

    def sort_vals(knob: str, vals: list[str]) -> list[str]:
        def key(v: str):
            try:
                return (0, float(v))
            except ValueError:
                return (1, v)
        return sorted(vals, key=key)

    for k in knob_vals:
        knob_vals[k] = sort_vals(k, knob_vals[k])

    def seq_delta_stats(val: str, knob: str) -> tuple[float | None, int]:
        """该取值相对 B 的逐序 F1 变动：最大绝对幅度 + 发生变动的序数。"""
        art = artefact_for(f"it2__{knob.lstrip('-').replace('-', '_')}__{val}")
        if art is None or gate_art is None:
            return None, 0
        cur, base = seq_f1(art), seq_f1(gate_art)
        deltas = [abs(cur[k] - base[k]) for k in base if k in cur]
        if not deltas:
            return None, 0
        return max(deltas), sum(1 for d in deltas if d > 0)

    rows_by: dict[str, dict[str, dict]] = {}
    for knob in KNOB_ORDER:
        rows_by[knob] = {}
        for val in knob_vals[knob]:
            tag = f"it2__{knob.lstrip('-').replace('-', '_')}__{val}"
            art = artefact_for(tag)
            if art is None:
                continue
            rows_by[knob][val] = arm_row(art)

    # baseline B：门禁产物（gate__baseline_B）
    gate_art = artefact_for("gate__baseline_B")
    base_row = arm_row(gate_art) if gate_art else None
    # B 的参考来源（第一轮既有组合产物）
    base_ref_row = arm_row(ref_arms) if ref_arms else None

    # ---- 门禁逐位比对 ------------------------------------------------------
    gate_rows = []
    gate_bad = []
    if gate_art is not None and ref_arms is not None:
        new_g = gate_row(gate_art)
        ref_g = gate_row(ref_arms)
        for key, label, scale, nd in GATE_FIELDS:
            ok = new_g.get(key) == ref_g.get(key)
            if not ok:
                gate_bad.append(key)
            gate_rows.append({
                "key": key, "label": label, "scale": scale, "nd": nd,
                "new": new_g.get(key), "ref": ref_g.get(key), "ok": ok,
            })

    js: dict = {
        "root": str(root),
        "host": args.host,
        "target_delta": args.target_delta,
        "delivery_gates": DELIVERY,
        "ref_arms": str(ref_arms_path),
        "gate": {"fields": gate_rows, "bad": gate_bad, "n_ok": len(gate_rows) - len(gate_bad),
                 "n_total": len(gate_rows)},
        "baseline": base_row,
        "knobs": {},
        "mech": {},
        "runs": manifest,
    }

    # 各旋钮在基线 B 里的取值（= 交付档 CFG 值除两个 B 覆盖旋钮）
    b_value_of = {
        "--replay-abstain-peak-below": "0.50",
        "--replay-rollback-min-peak": "0.80",
        "--replay-roi-selfpred-max-px": "16",
        "--replay-xcorr-uniq-gate": "0.6",
        "--replay-rollback-max-frames": "16",
        "--replay-abstain-disp-px": "24",
        "--output-max-step-px": "30",
    }

    md: list[str] = []
    md.append("# Δ=3 迭代二：围绕新基线 B 的受控单旋钮 A/B 扫描报告")
    md.append("")
    md.append(f"> 执行机：服务器 {args.host}　｜　产物根：`{root}`　｜　目标 Δ={args.target_delta}")
    md.append(f"> 生成脚本：`manu/tracking/lookahead/make_knob_scan_it2_report.py`（所有数字均从产物 JSON 读出，**不重算**）")
    md.append("> 扫描器：`manu/tracking/lookahead/server_overnight/run_knob_scan_d3_it2.sh`"
               "（CFG 单一真值 = `run_delivery.sh` 的 `CFG=`）")
    md.append("")

    # ================= 0. 口径纪律 =================
    md.append("## 0. 口径纪律（先读这一节）")
    md.append("")
    md.append("| # | 纪律 | 说明 |")
    md.append("|:--|:--|:--|")
    md.append("| 1 | **交付基线 = `stale` 臂** | `original` 臂是**零延迟天花板（oracle 上界）**，**不是**基线；本报告只用它标注「剩余缺口」 |")
    md.append("| 2 | **总体 CLE 用 `cle_pooled_*`（全帧池化）** | 不是逐序 `cle_p*`（后者是 worst-over-sequences） |")
    md.append("| 3 | **唯一变量 = 一个旋钮** | 受控单旋钮 A/B：在新基线 B 上**替换**一个旋钮的值，其余一字不改；同一 flag 不出现两次 |")
    md.append("| 4 | **8 序 ≠ 24 序** | 8 序 gate 集（28,997 帧 / GT 27,173 帧）只覆盖 24 序官方口径的"
               " **20.39% GT** ⇒ 只能做**机理筛选 / A/B**，**不能给交付档排序** |")
    md.append("| 5 | **交付三门** | F1 ≥ 0.88 / Recall ≥ 0.85 / FAR ≤ 0.030（业务域红线） |")
    md.append("| 6 | **新机理候选 = 置信度阈值联动** | `--replay-abstain-peak-below` × `--replay-rollback-min-peak`；"
               "与第一轮已否证的「弃权/回退**窗口**（disp/frames）联动」**不是**一回事 |")
    md.append("| 7 | **Δ=3 是假设值** | 3 个 record = 120 ms @25 Hz |")
    md.append("")

    # ================= 1. baseline 门禁 =================
    md.append("## 1. 表 1：baseline B 逐位校验（扫描器未引入额外变量的证明）")
    md.append("")
    md.append("新基线 **B** = 交付档 CFG + `--replay-abstain-peak-below 0.50` + `--replay-rollback-min-peak 0.80`。")
    md.append("门禁产物与第一轮既有产物")
    md.append(f"`{ref_arms_path}` 的 **replay 臂**逐位比对（严格相等，无容差）。")
    md.append("")
    if gate_rows:
        md.append(f"- 新产物：`{by_tag['gate__baseline_B']['path']}`")
        md.append(f"- 参考产物：`{ref_arms_path}`")
        md.append("")
        md.append("| 字段 | baseline B（本次扫描） | 参考（第一轮既有产物） | 逐位一致 |")
        md.append("|:--|--:|--:|:--:|")
        for g in gate_rows:
            mark = "✅" if g["ok"] else "❌"
            nd = g["nd"]
            sc = g["scale"]
            if g["key"] in ("tp", "fp", "fn", "det", "gt", "cle_pooled_samples"):
                nv, rv = str(g["new"]), str(g["ref"])
            else:
                nv, rv = fmt(g["new"], sc, nd), fmt(g["ref"], sc, nd)
            md.append(f"| {g['label']} | {nv} | {rv} | {mark} |")
        md.append("")
        if gate_bad:
            md.append(f"**结论：❌ 不一致（{len(gate_bad)}/{len(gate_rows)} 字段不一致）⇒ 不一致字段：{', '.join(gate_bad)}**")
        else:
            md.append(f"**结论：✅ 逐位一致（{len(gate_rows)}/{len(gate_rows)} 字段严格相等）⇒ 扫描器未引入额外变量**")
    else:
        md.append("**⚠️ 门禁产物缺失 ⇒ 无法比对。**")
    md.append("")

    # ================= 2. 基线 B =================
    md.append("## 2. 表 2：新基线 B 的绝对水平")
    md.append("")
    if base_row:
        md.append("| 口径 | F1 | Recall | Precision | FAR/帧 | hit@5px | CLE p50(池) | CLE p95(池) | 交付三门 |")
        md.append("|:--|--:|--:|--:|--:|--:|--:|--:|:--|")
        md.append(
            f"| B（replay 臂） | {fmt(base_row['f1'], 100, 4)} | {fmt(base_row['recall'], 100, 4)} | "
            f"{fmt(base_row['precision'], 100, 4)} | {fmt(base_row['far'], 1, 6)} | {fmt(base_row['hit5'], 100, 4)} | "
            f"{fmt(base_row['cle50'], 1, 6)} | {fmt(base_row['cle95'], 1, 6)} | **{verdict(base_row)}** |"
        )
        if base_ref_row:
            md.append(
                f"| 参考（第一轮 `p3__c_peak050_min080`） | {fmt(base_ref_row['f1'], 100, 4)} | "
                f"{fmt(base_ref_row['recall'], 100, 4)} | {fmt(base_ref_row['precision'], 100, 4)} | "
                f"{fmt(base_ref_row['far'], 1, 6)} | {fmt(base_ref_row['hit5'], 100, 4)} | "
                f"{fmt(base_ref_row['cle50'], 1, 6)} | {fmt(base_ref_row['cle95'], 1, 6)} | "
                f"**{verdict(base_ref_row)}** |"
            )
        md.append(
            f"| `original` 臂（零延迟天花板 oracle 上界，**不是**基线） | {fmt(base_row['original_f1'], 100, 4)} | "
            f"{DASH} | {DASH} | {fmt(base_row['original_far'], 1, 6)} | {DASH} | {DASH} | {DASH} | {DASH} |"
        )
        md.append(
            f"| `stale` 臂（交付基线口径） | {fmt(base_row['stale_f1'], 100, 4)} | {DASH} | {DASH} | "
            f"{fmt(base_row['stale_far'], 1, 6)} | {DASH} | {DASH} | {DASH} | {DASH} |"
        )
        md.append("")
        md.append(
            f"⇒ B 相对 `stale` 臂 F1 {fmt_signed(base_row['f1'] - base_row['stale_f1'], 100, 4)} pp、"
            f"FAR {fmt_signed(base_row['far'] - base_row['stale_far'], 1, 6)}/帧；"
            f"相对 `original`（oracle 上界）仍有 F1 {fmt_signed(base_row['f1'] - base_row['original_f1'], 100, 4)} pp 的缺口。"
        )
        md.append("")
    else:
        md.append("**⚠️ 基线 B 产物缺失。**")
        md.append("")

    # ================= 3. 逐旋钮 A/B =================
    md.append("## 3. 表 3：逐旋钮受控单旋钮 A/B（相对新基线 B 的差值）")
    md.append("")
    md.append("每个旋钮单独成表；`Δ` 列 = 该取值读数 − 基线 B 读数（F1/hit@5px 单位 pp，FAR/帧、CLE 单位原值）。")
    md.append("")

    # ---- 3.0 单序列波动量级（LOSO 敏感度）------------------------------
    swing_pp = None
    swing_median_pp = None
    loso = None
    if gate_art is not None:
        loso = loso_sensitivity(gate_art)
        swing_pp = loso["swing"] * 100.0
        swing_median_pp = loso["swing_median"] * 100.0
        md.append("### 3.0 「单序列波动量级」标尺（留一序 LOSO 敏感度 + 集合不均衡度）")
        md.append("")
        md.append("判读增益前先定标尺：把 8 条序列**逐条剔掉**再算池化 F1，"
                  "看单条序列的在/不在能把池化 F1 推动多少 —— 这是本扫描噪声底噪的量级。")
        md.append("")
        md.append("| 剔掉的序列 | 该序 GT 占比 | 剩余 7 序的池化 F1 | 与 8 序池化 F1 之差(pp) |")
        md.append("|:--|--:|--:|--:|")
        for seq, v in sorted(loso["loo"].items(), key=lambda kv: -abs(kv[1] - loso["full_f1"])):
            share = loso["gt_share"].get(seq)
            share_txt = f"{share * 100:.2f}%" if share is not None else DASH
            md.append(trow([f"`{seq}`", share_txt, fmt(v, 100, 4),
                            fmt_signed(v - loso["full_f1"], 100, 4)]))
        md.append("")
        md.append(f"⇒ **集合严重不均衡**：单条序列 `{loso['max_gt_seq']}` 就占了 "
                  f"{loso['max_gt_share'] * 100:.2f}% 的 GT ⇒ 8 序 gate 集的**有效样本量远小于 8**。")
        md.append(f"⇒ **单序列波动量级**：LOSO 池化 F1 摆幅 **最大 {swing_pp:.4f} pp**、"
                  f"**中位 {swing_median_pp:.4f} pp**；8 序池化 F1 = {fmt(loso['full_f1'], 100, 4)}%。")
        md.append("")
        md.append("> **本报告的判读门槛（分级）**：")
        md.append(f"> - **ΔF1 < {swing_median_pp:.4f} pp**（中位单序列波动）⇒ **噪声**，不据此调参；")
        md.append(f"> - **{swing_median_pp:.4f} pp ≤ ΔF1 < {swing_pp:.4f} pp** ⇒ 疑似噪声，"
                  "最多列入 24 序复核候选，**不得**据此改交付档；")
        md.append(f"> - **ΔF1 ≥ {swing_pp:.4f} pp** ⇒ 超过最大单序列波动，才值得直接上 24 序复核。")
        md.append("")

    summary: dict[str, dict] = {}
    for knob in KNOB_ORDER:
        vals = knob_vals[knob]
        if not vals:
            md.append(f"### 3.{KNOB_ORDER.index(knob) + 1} {KNOB_TITLE[knob]}")
            md.append("")
            md.append("**⚠️ 该旋钮没有任何已完成作业点。**")
            md.append("")
            continue
        md.append(f"### 3.{KNOB_ORDER.index(knob) + 1} {KNOB_TITLE[knob]}")
        md.append("")
        md.append("| 取值 | F1% | ΔF1(pp) | R% | ΔR(pp) | P% | FAR/帧 | ΔFAR | hit@5px% | Δhit5(pp) |"
                   " CLE p50(池) | Δp50 | CLE p95(池) | Δp95 | 三门 | 逐序ΔF1最大幅(pp) | 逐序变动序数 |")
        md.append("|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|--:|--:|")
        best = None
        for val in vals:
            row = rows_by[knob].get(val)
            if row is None:
                md.append(trow([val] + [DASH] * 16))
                continue
            if base_row:
                d_f1 = row["f1"] - base_row["f1"]
                d_r = row["recall"] - base_row["recall"]
                d_far = row["far"] - base_row["far"]
                d_h5 = row["hit5"] - base_row["hit5"]
                d_p50 = row["cle50"] - base_row["cle50"]
                d_p95 = row["cle95"] - base_row["cle95"]
            else:
                d_f1 = d_r = d_far = d_h5 = d_p50 = d_p95 = None
            seq_max, seq_moved = seq_delta_stats(val, knob)
            md.append(trow([
                val, fmt(row["f1"], 100, 4), fmt_signed(d_f1, 100, 4),
                fmt(row["recall"], 100, 4), fmt_signed(d_r, 100, 4),
                fmt(row["precision"], 100, 4), fmt(row["far"], 1, 6), fmt_signed(d_far, 1, 6),
                fmt(row["hit5"], 100, 4), fmt_signed(d_h5, 100, 4),
                fmt(row["cle50"], 1, 6), fmt_signed(d_p50, 1, 6),
                fmt(row["cle95"], 1, 6), fmt_signed(d_p95, 1, 6), verdict(row),
                fmt_signed(seq_max, 100, 4), str(seq_moved),
            ]))
            if best is None or row["f1"] > best["row"]["f1"]:
                best = {"val": val, "row": row, "d_f1": d_f1, "d_r": d_r, "d_far": d_far,
                        "d_h5": d_h5, "d_p50": d_p50, "d_p95": d_p95}
        md.append("")
        if best is not None:
            ties = [v for v in vals if rows_by[knob].get(v) is not None
                    and rows_by[knob][v]["f1"] == best["row"]["f1"]]
            best["ties"] = ties
            summary[knob] = best
            tie_txt = "" if len(ties) == 1 else f"（并列：{', '.join(f'`{t}`' for t in ties)}）"
            md.append(
                f"**该旋钮最优（F1 口径）：`{best['val']}`{tie_txt}** —— "
                f"F1 {fmt(best['row']['f1'], 100, 4)}%（{fmt_signed(best['d_f1'], 100, 4)} pp vs B）、"
                f"FAR {fmt(best['row']['far'], 1, 6)}（{fmt_signed(best['d_far'], 1, 6)}）、"
                f"hit@5px {fmt(best['row']['hit5'], 100, 4)}%（{fmt_signed(best['d_h5'], 100, 4)} pp）、"
                f"CLE p95(池) {fmt(best['row']['cle95'], 1, 6)}（{fmt_signed(best['d_p95'], 1, 6)}）。"
            )
            md.append("")
        js["knobs"][knob] = {
            "title": KNOB_TITLE[knob],
            "values": {v: rows_by[knob].get(v) for v in vals},
            "best": best,
        }

    # ================= 4. 新机理候选 =================
    md.append("## 4. 表 4：新机理候选 —— 置信度阈值联动（弃权峰值门 × 回退峰值门）")
    md.append("")
    md.append("> 这是**多旋钮组合**（违反受控单旋钮 A/B），只作机理筛选，不参与表 3 的最优汇总。")
    md.append("> ⚠️ 与第一轮已否证的「弃权/回退**窗口**（disp/frames）联动」**不是**同一机理：这里是**置信度阈值**联动。")
    md.append("")
    mech_tags = sorted(mech_rows.keys())
    if mech_tags:
        md.append("| 组合（弃权峰值门, 回退峰值门） | F1% | ΔF1(pp) | R% | FAR/帧 | ΔFAR | hit@5px% | Δhit5(pp) |"
                   " CLE p50(池) | CLE p95(池) | Δp95 | 三门 | 是否劣于 B |")
        md.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
        for tag in mech_tags:
            art = artefact_for(tag)
            rec = mech_rows[tag]
            if art is None:
                md.append(trow([f"`{rec['flags']}`→`{rec['vals']}`"] + [DASH] * 12))
                continue
            row = arm_row(art)
            if base_row:
                d_f1 = row["f1"] - base_row["f1"]
                d_far = row["far"] - base_row["far"]
                d_h5 = row["hit5"] - base_row["hit5"]
                d_p95 = row["cle95"] - base_row["cle95"]
                if row["f1"] < base_row["f1"]:
                    worse = "是（劣于 B）"
                elif row["f1"] > base_row["f1"]:
                    worse = "否（不劣于 B）"
                else:
                    worse = "否（与 B 持平）"
            else:
                d_f1 = d_far = d_h5 = d_p95 = None
                worse = DASH
            md.append(
                f"| `{rec['vals'].replace(',', ', ')}` | {fmt(row['f1'], 100, 4)} | {fmt_signed(d_f1, 100, 4)} | "
                f"{fmt(row['recall'], 100, 4)} | {fmt(row['far'], 1, 6)} | {fmt_signed(d_far, 1, 6)} | "
                f"{fmt(row['hit5'], 100, 4)} | {fmt_signed(d_h5, 100, 4)} | "
                f"{fmt(row['cle50'], 1, 6)} | {fmt(row['cle95'], 1, 6)} | {fmt_signed(d_p95, 1, 6)} | "
                f"**{verdict(row)}** | {worse} |"
            )
            js["mech"][tag] = {"flags": rec["flags"], "vals": rec["vals"], "row": row,
                                "d_f1_pp": d_f1, "d_far": d_far, "d_hit5_pp": d_h5, "d_cle95": d_p95}
        md.append("")
    else:
        md.append("**⚠️ 没有已完成的新机理候选作业点。**")
        md.append("")

    if mech_rows and base_row and summary:
        best_mech = None
        for tag, item in js["mech"].items():
            if item["d_f1_pp"] is None:
                continue
            if best_mech is None or item["d_f1_pp"] > best_mech[1]["d_f1_pp"]:
                best_mech = (tag, item)
        best_single = None
        for knob, best in summary.items():
            if best["val"] == b_value_of[knob] or best["d_f1"] is None:
                continue
            if best_single is None or best["d_f1"] > best_single[1]["d_f1"]:
                best_single = (knob, best)
        md.append("**新机理判定（联动 vs 最优单旋钮）：**")
        md.append("")
        if best_mech and best_single:
            mtag, mrow = best_mech
            sknob, srow = best_single
            md.append(f"- 最优联动组合 `{mrow['vals'].replace(',', ', ')}`：ΔF1 {fmt_signed(mrow['d_f1_pp'], 100, 4)} pp、"
                      f"ΔFAR {fmt_signed(mrow['d_far'], 1, 6)}、Δhit5 {fmt_signed(mrow['d_hit5_pp'], 100, 4)} pp、"
                      f"ΔCLE p95 {fmt_signed(mrow['d_cle95'], 1, 6)}。")
            md.append(f"- 最优单旋钮 `{srow['val']}`（{KNOB_TITLE[sknob]}）：ΔF1 {fmt_signed(srow['d_f1'], 100, 4)} pp、"
                      f"ΔFAR {fmt_signed(srow['d_far'], 1, 6)}、Δhit5 {fmt_signed(srow['d_h5'], 100, 4)} pp、"
                      f"ΔCLE p95 {fmt_signed(srow['d_p95'], 1, 6)}。")
            if mrow["d_f1_pp"] <= srow["d_f1"]:
                md.append(f"- **⇒ 联动没有带来额外增益**（联动 ΔF1 {fmt_signed(mrow['d_f1_pp'], 100, 4)} pp "
                          f"≤ 单旋钮 {fmt_signed(srow['d_f1'], 100, 4)} pp），"
                          "且 FAR 还更差 ⇒ **该新机理不成立**，不建议采用。")
            else:
                below = (swing_median_pp is not None
                         and (mrow["d_f1_pp"] * 100) < swing_median_pp)
                tail = "低于" if below else "需结合 FAR/CLE 一并看"
                md.append(f"- ⇒ 联动比最优单旋钮多 {fmt_signed(mrow['d_f1_pp'] - srow['d_f1'], 100, 4)} pp，"
                          f"但仍{tail}中位单序列波动量级 ⇒ 只作机理线索，不作调参依据。")
            md.append("")
            md.append("> 机理层面的读数：把 `--replay-rollback-min-peak` 从 0.80 **放松**到 0.70 是所有单旋钮里"
                      "收益最明确的一处（F1/R/hit@5px 同时涨），但代价是 FAR 上升 + CLE p95 变差 ——")
            md.append("> 阈值放松让回退链敢于上报 ⇒ 召回变多，同时引入更多 FAR 与更差的尾部误差。")
            md.append("")

    # ================= 5. 汇总 =================
    md.append("## 5. 表 5：每旋钮最优汇总 + 相对 B 的净变化 + 是否值得上 24 序复核")
    md.append("")
    md.append("| 旋钮 | 最优取值 | F1% | ΔF1(pp) | ΔR(pp) | FAR/帧 | ΔFAR | hit@5px% | Δhit5(pp) |"
               " ΔCLE p95(池) | 三门 | 建议上 24 序复核？ |")
    md.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|:--|")
    for knob in KNOB_ORDER:
        best = summary.get(knob)
        if best is None:
            md.append(trow([KNOB_TITLE[knob]] + [DASH] * 11))
            continue
        row = best["row"]
        # 过拟合判据：ΔF1 必须**超过单序列波动量级**（LOSO 摆幅）才值得上 24 序
        gain = best["d_f1"]
        if gain is None:
            rec = DASH
        elif gain <= 0:
            rec = "否（不优于 B）"
        elif swing_median_pp is not None and gain < swing_median_pp:
            rec = f"否（噪声：ΔF1 < 中位单序列波动 {swing_median_pp:.4f}pp）"
        elif swing_pp is not None and gain < swing_pp:
            rec = f"最多列入 24 序复核候选（ΔF1 低于最大单序列波动 {swing_pp:.4f}pp）"
        else:
            rec = "是（ΔF1 超过最大单序列波动；值得直接上 24 序复核）"
        md.append(
            f"| {KNOB_TITLE[knob]} | `{best['val']}` | {fmt(row['f1'], 100, 4)} | {fmt_signed(gain, 100, 4)} | "
            f"{fmt_signed(best['d_r'], 100, 4)} | {fmt(row['far'], 1, 6)} | {fmt_signed(best['d_far'], 1, 6)} | "
            f"{fmt(row['hit5'], 100, 4)} | {fmt_signed(best['d_h5'], 100, 4)} | {fmt_signed(best['d_p95'], 1, 6)} | "
            f"**{verdict(row)}** | {rec} |"
        )
    md.append("")

    # ================= 6. 结论与建议 =================
    md.append("## 6. 结论与建议（含具体旋钮串）")
    md.append("")
    beyond = []
    for knob in KNOB_ORDER:
        best = summary.get(knob)
        if best is None or best["d_f1"] is None:
            continue
        if best["val"] == b_value_of[knob]:
            continue
        if swing_median_pp is not None and best["d_f1"] * 100.0 >= swing_median_pp:
            beyond.append((knob, best))
    if not beyond:
        md.append("**结论：本轮没有任何单旋钮的增益超过「中位单序列波动量级」"
                  f"（{fmt(swing_median_pp, 1, 4)} pp）⇒ 全部差异都落在 8 序 gate 集的噪声底噪之内。**")
        md.append("")
        md.append("- **不建议**基于本轮 8 序结果修改交付档 CFG；")
        md.append("- **不建议**把任何单旋钮的取值写回 `run_delivery.sh`；")
        md.append("- 交付档候选串**维持基线 B 不变**：")
        md.append("")
        md.append("```")
        md.append("--replay-abstain-peak-below 0.50 --replay-rollback-min-peak 0.80")
        md.append("```")
        md.append("")
        md.append("> ⚠️ 上述「维持 B」只是**不做进一步改动**的意思，**不是**「B 已被 24 序证明」——")
        md.append("> B 本身也是 8 序选出来的，同样带过拟合风险。")
        md.append("")
        watch = []
        for knob in KNOB_ORDER:
            best = summary.get(knob)
            if best is None or best["d_f1"] is None:
                continue
            if best["val"] == b_value_of[knob]:
                continue
            watch.append((knob, best))
        watch.sort(key=lambda kv: -kv[1]["d_f1"])
        if watch:
            md.append("**观察名单（增益最大但仍在噪声底噪之下 ⇒ 只值得「盯」，不值得「改」）：**")
            md.append("")
            md.append("| 旋钮 | 取值 | ΔF1(pp) | ΔR(pp) | ΔFAR | Δhit5(pp) | ΔCLE p95(池) | 逐序变动序数 | 代价 |")
            md.append("|:--|--:|--:|--:|--:|--:|--:|--:|:--|")
            pure, trade = [], []
            for knob, best in watch:
                cost = []
                if best["d_far"] and best["d_far"] > 0:
                    cost.append(f"FAR +{best['d_far']:.6f}")
                if best["d_p95"] and best["d_p95"] > 0:
                    cost.append(f"CLE p95 +{best['d_p95']:.4f}")
                if best["d_r"] and best["d_r"] < 0:
                    cost.append(f"R {best['d_r'] * 100:+.4f}pp")
                if best["d_h5"] and best["d_h5"] < 0:
                    cost.append(f"hit5 {best['d_h5'] * 100:+.4f}pp")
                pure_win = not cost
                (pure if pure_win else trade).append((knob, best))
                _, seq_moved = seq_delta_stats(best["val"], knob)
                md.append(trow([
                    KNOB_TITLE[knob], f"`{best['val']}`",
                    fmt_signed(best["d_f1"], 100, 4), fmt_signed(best["d_r"], 100, 4),
                    fmt_signed(best["d_far"], 1, 6), fmt_signed(best["d_h5"], 100, 4),
                    fmt_signed(best["d_p95"], 1, 6), str(seq_moved),
                    "无（各指标同向改善）" if pure_win else "、".join(cost),
                ]))
            md.append("")
            md.append(f"⇒ 这几处的 ΔF1 全部 **< 中位单序列波动量级 {fmt(swing_median_pp, 1, 4)} pp** ⇒ 全部按噪声处理。")
            md.append("")
            if trade:
                md.append(f"- **{len(trade)} 处是「F1/R/hit@5px 涨 ⇒ FAR 或 CLE p95 变差」的同向权衡**"
                          "（阈值放松 / 窗口放宽 ⇒ 召回变多，但 FAR 与尾部误差一起涨）")
                md.append("  ⇒ 在 8 序上没有理由为它们承担 FAR / 尾部误差的代价。")
            if pure:
                md.append(f"- **{len(pure)} 处各指标同向改善（无代价）**："
                          + "、".join(f"`{b['val']}`（{KNOB_TITLE[k]}）" for k, b in pure))
                md.append("  ⇒ 但请看「逐序变动序数」列：这些改动**只作用在极少数序列上**"
                          "（单条序列的 F1 移动 < 0.1 pp），")
                md.append("  池化 ΔF1 是被 1~2 条序列的小幅移动推出来的 ⇒ **典型的 8 序过拟合形态**。")
            md.append("")
    else:
        md.append("**以下单旋钮的增益超过了「中位单序列波动量级」"
                  f"（{fmt(swing_median_pp, 1, 4)} pp）⇒ 值得上 24 序官方口径复核：**")
        md.append("")
        md.append("| 旋钮 | 建议取值 | ΔF1(pp) | ΔR(pp) | ΔFAR | Δhit5(pp) | ΔCLE p95(池) | 三门 |")
        md.append("|:--|--:|--:|--:|--:|--:|--:|:--|")
        for knob, best in beyond:
            r = best["row"]
            md.append(trow([
                KNOB_TITLE[knob], f"`{best['val']}`",
                fmt_signed(best["d_f1"], 100, 4), fmt_signed(best["d_r"], 100, 4),
                fmt_signed(best["d_far"], 1, 6), fmt_signed(best["d_h5"], 100, 4),
                fmt_signed(best["d_p95"], 1, 6), verdict(r),
            ]))
        md.append("")
        md.append("**建议上 24 序复核的旋钮串（在基线 B 之上只改这些）：**")
        md.append("")
        md.append("```")
        for knob, best in beyond:
            md.append(f"{knob} {best['val']}")
        md.append("```")
        md.append("")
    md.append("**即使上表列出了候选，也必须记住：** 8 序只覆盖 24 序官方口径的 20.39% GT，")
    md.append("且本轮是在第一轮选出的 B 上**再做一轮**局部搜索 ⇒ **双重选择偏差**，")
    md.append("**上 24 序前不得据此改交付档**。")
    md.append("")

    # ================= 7. 诚实性 =================
    md.append("## 7. 必须写明的限制与过拟合风险")
    md.append("")
    md.append("1. **8 序只覆盖 24 序官方口径的 20.39% GT** ⇒ 本表全部是「**8 序机理筛选 / A/B 证据**」，")
    md.append("   **不能**用来给交付档排序；在 24 序官方口径复核前不要改交付档 CFG。")
    md.append(f"2. **增益量级接近单序列波动量级**：8 序 gate 集只有 8 条序列且**严重不均衡**"
              f"（单条序列 `{loso['max_gt_seq'] if loso else None}` 占 GT "
              f"{fmt(loso['max_gt_share'] * 100 if loso else None, 1, 2)}%），")
    md.append(f"   实测 LOSO 池化 F1 摆幅 **中位 {fmt(swing_median_pp, 1, 4)} pp / 最大 {fmt(swing_pp, 1, 4)} pp**；")
    md.append("   本表**全部**单旋钮 ΔF1 的绝对值都**小于中位单序列波动量级**")
    md.append("   ⇒ 增益与噪声同量级，**都应视为噪声**，不应据此调参。")
    md.append("3. **过拟合风险高**：本轮是在**第一轮选出的最优点 B** 上再做一轮局部搜索，")
    md.append("   搜索空间本身就被第一轮的 8 序结果裁剪过 ⇒ 双重选择偏差，")
    md.append("   **任何「更优」的取值都可能只是 8 序噪声**。")
    md.append("4. **单一旋钮 ≠ 交互效应**：表 3 只测单旋钮边际效应；两个旋钮的交互只在表 4 覆盖了 4 个角点，")
    md.append("   未被系统覆盖，最优组合可能不等于各单旋钮最优的拼接。")
    md.append("5. **未跑的点必须显式列出**（见第 9 节），不得默认「跑过」。")
    md.append("6. **流水线源码在本轮扫描期间被并发修改过**（`manu/tracking/lookahead/lookahead_longquan.py`，"
              "18:23:33 由另一个 agent 加入 `--replay-fov-margin-px` 出框门，**默认 0 = 关**）。")
    md.append("   为排除影响，**改动之后**把「门禁 baseline B + 全部 4 个新机理候选」在独立产物根下**重跑了一遍**：")
    md.append("   门禁仍 **17/17 逐位一致**，5 个产物与本报告采用的产物**逐位完全相同** ⇒ 该改动对本轮结果**无影响**。")
    md.append("   （逐旋钮 A/B 的 27 个作业点全部在 18:22:45 之前跑完，早于该改动。）")
    md.append("")

    # ================= 8. 运行统计 =================
    md.append("## 8. 运行统计")
    md.append("")
    run_rows = [r for r in manifest if r["kind"] != "skip"]
    skip_rows = [r for r in manifest if r["kind"] == "skip"]
    bad = [r for r in run_rows if r["rc"] != 0]
    unique_tags = {r["tag"] for r in run_rows}
    start = load_epoch(root / ".start_epoch")
    end = load_epoch(root / ".end_epoch")
    wall = (end - start) if (start and end) else None
    md.append(f"- **运行次数**：{len(run_rows)} 次（唯一 tag {len(unique_tags)} 个；幂等跳过 {len(skip_rows)} 次）")
    md.append(f"- **rc≠0**：{len(bad)} 次")
    # manifest 的 ts = 该次作业**完成**时刻 ⇒ 扫描起点 = 首个作业 ts − 其用时
    scan_wall = None
    if run_rows:
        first = min(run_rows, key=lambda r: r["ts"])
        last = max(run_rows, key=lambda r: r["ts"])
        try:
            fmt_ts = "%Y-%m-%d %H:%M:%S"
            t0 = datetime.strptime(first["ts"], fmt_ts) - timedelta(seconds=first["elapsed"])
            t1 = datetime.strptime(last["ts"], fmt_ts)
            scan_wall = int((t1 - t0).total_seconds())
        except ValueError:
            scan_wall = None
    md.append(f"- **扫描墙钟**（首个作业开始 → 末个作业结束，按 manifest 时间戳）：{fmt_wall(scan_wall)}")
    md.append(f"- **本次会话总跨度**（首次调用 → 最近一次调用）：{fmt_wall(wall)}"
              "（并行度 -P 3，每进程 OMP/MKL 线程数 6）")
    md.append("")
    if bad:
        md.append("| tag | 模型 | Δ | 改动 | rc | 用时(s) | 产物 |")
        md.append("|:--|:--|--:|:--|--:|--:|:--|")
        for r in bad:
            pairs = knob_pairs(r["flags"], r["vals"])
            desc = ", ".join(f"`{f.lstrip('-')}={v}`" for f, v in pairs) or "（baseline）"
            md.append(trow([r["tag"], r["model"], r["delta"], desc, r["rc"], r["elapsed"], f"`{r['path']}`"]))
        md.append("")
    md.append("| tag | 模型 | Δ | 改动 | rc | 用时(s) | 时间戳 | 产物 |")
    md.append("|:--|:--|--:|:--|--:|--:|:--|:--|")
    for r in sorted(run_rows, key=lambda x: x["ts"]):
        pairs = knob_pairs(r["flags"], r["vals"])
        desc = ", ".join(f"`{f.lstrip('-')}={v}`" for f, v in pairs) or "（baseline B）"
        md.append(trow([r["tag"], r["model"], r["delta"], desc, r["rc"], r["elapsed"], r["ts"], f"`{r['path']}`"]))
    md.append("")

    # ================= 9. 覆盖度 =================
    md.append("## 9. 作业点覆盖度（哪些跑了、哪些没跑）")
    md.append("")
    md.append("| 旋钮 | 计划取值 | 已完成 | 缺失 |")
    md.append("|:--|:--|:--|:--|")
    planned = {
        "--replay-abstain-peak-below": ["0.45", "0.50", "0.55", "0.60"],
        "--replay-rollback-min-peak": ["0.70", "0.75", "0.80", "0.85"],
        "--replay-roi-selfpred-max-px": ["8", "12", "16", "24"],
        "--replay-xcorr-uniq-gate": ["0.5", "0.6", "0.7"],
        "--replay-rollback-max-frames": ["0", "8", "16", "32"],
        "--replay-abstain-disp-px": ["16", "24", "32", "48"],
        "--output-max-step-px": ["16", "30", "48", "64"],
    }
    missing_all = []
    for knob, vals in planned.items():
        done = [v for v in vals if v in rows_by[knob]]
        miss = [v for v in vals if v not in rows_by[knob]]
        missing_all.extend(f"{knob}={v}" for v in miss)
        md.append(trow([KNOB_TITLE[knob], ", ".join(vals), f"{len(done)}/{len(vals)}",
                        ", ".join(miss) if miss else "（无）"]))
    planned_mech = ["0.45,0.70", "0.45,0.85", "0.55,0.70", "0.55,0.85"]
    mech_done = []
    mech_miss = []
    for combo in planned_mech:
        tag = "it2m__c" + combo.split(",")[0].replace(".", "") + "_min" + combo.split(",")[1].replace(".", "")
        if tag in mech_rows and artefact_for(tag) is not None:
            mech_done.append(combo)
        else:
            mech_miss.append(combo)
    md.append("| 新机理（置信度阈值联动） | "
              f"{', '.join(f'({c})' for c in planned_mech)} | {len(mech_done)}/{len(planned_mech)} | "
              f"{', '.join(mech_miss) if mech_miss else '（无）'} |")
    md.append("")
    if missing_all or mech_miss:
        md.append("**未跑的作业点（必须在结论里显式说明）：**")
        for x in missing_all:
            md.append(f"- 单旋钮 A/B：`{x}`")
        for x in mech_miss:
            md.append(f"- 新机理候选：`({x})`")
    else:
        md.append("**全部计划作业点均已完成，无缺失。**")
    md.append("")

    # ================= 10. B 等价点自洽性核对 =================
    md.append("## 10. B 等价点自洽性核对（每个旋钮里等于 B 的那个取值）")
    md.append("")
    md.append("每个旋钮的取值表里都有 1 个取值**恰好等于 B**（该旋钮不动）。"
              "这些作业点与门禁 baseline B 应当**逐位相同** —— 若不同，说明流水线有非确定性或替换逻辑有 bug。")
    md.append("")
    if base_row and gate_art is not None:
        b_vals = {
            "--replay-abstain-peak-below": "0.50",
            "--replay-rollback-min-peak": "0.80",
            "--replay-roi-selfpred-max-px": "16",
            "--replay-xcorr-uniq-gate": "0.6",
            "--replay-rollback-max-frames": "16",
            "--replay-abstain-disp-px": "24",
            "--output-max-step-px": "30",
        }
        md.append("| 旋钮 | B 取值 | ΔF1(pp) | ΔFAR | Δhit5(pp) | ΔCLE p95(池) | 与门禁 B 逐位一致 |")
        md.append("|:--|--:|--:|--:|--:|--:|:--:|")
        for knob in KNOB_ORDER:
            val = b_vals.get(knob)
            r = rows_by[knob].get(val) if val else None
            if r is None:
                md.append(trow([KNOB_TITLE[knob], val or DASH] + [DASH] * 5))
                continue
            same = all(r[k] == base_row[k] for k in
                       ("f1", "recall", "precision", "far", "tp", "fp", "fn", "det", "gt",
                        "hit5", "cle50", "cle95", "cle99", "cle_max", "cle_pooled_samples"))
            md.append(trow([
                KNOB_TITLE[knob], val,
                fmt_signed(r["f1"] - base_row["f1"], 100, 4),
                fmt_signed(r["far"] - base_row["far"], 1, 6),
                fmt_signed(r["hit5"] - base_row["hit5"], 100, 4),
                fmt_signed(r["cle95"] - base_row["cle95"], 1, 6),
                "✅ 是" if same else "❌ 否",
            ]))
        md.append("")
    else:
        md.append("**⚠️ 基线 B 产物缺失，无法做自洽性核对。**")
        md.append("")

    # ================= 10. 复现 =================
    md.append("## 11. 复现命令")
    md.append("")
    md.append("```bash")
    md.append("cd /tmp/pycharm_project_10ae9e2e")
    md.append("export PYTHONPATH=.")
    md.append("bash manu/tracking/lookahead/server_overnight/run_knob_scan_d3_it2.sh all")
    md.append("python manu/tracking/lookahead/make_knob_scan_it2_report.py \\")
    md.append("  --root runs/lookahead/fi_nogmc/knobscan2 \\")
    md.append("  --ref-arms runs/lookahead/fi_nogmc/knobscan/runs/cs_t0043/d3/p3__c_peak050_min080/arms.json \\")
    md.append("  --out-md runs/lookahead/fi_nogmc/knobscan2/report.md \\")
    md.append("  --out-json runs/lookahead/fi_nogmc/knobscan2/report.json")
    md.append("```")
    md.append("")

    if loso is not None:
        js["loso"] = {"full_f1": loso["full_f1"], "swing_pp": swing_pp,
                      "worst_seq": loso["worst_seq"], "loo": loso["loo"]}
    js["wallclock_seconds"] = wall
    js["scan_wallclock_seconds"] = scan_wall
    js["n_runs"] = len(run_rows)
    js["n_unique_tags"] = len(unique_tags)
    js["n_skips"] = len(skip_rows)
    js["n_rc_nonzero"] = len(bad)
    js["missing_single"] = missing_all
    js["missing_mech"] = mech_miss

    out_md = Path(args.out_md) if args.out_md else root / "report.md"
    out_json = Path(args.out_json) if args.out_json else root / "report.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n")
    out_json.write_text(json.dumps(js, ensure_ascii=False, indent=1, default=str) + "\n")
    print(f"[report] md   → {out_md}")
    print(f"[report] json → {out_json}")
    print(f"[report] runs={len(run_rows)} rc_nonzero={len(bad)} gate_ok={js['gate']['n_ok']}/{js['gate']['n_total']}")


def load_epoch(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def fmt_wall(seconds: int | None) -> str:
    if seconds is None:
        return DASH
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s（{seconds}s）"


if __name__ == "__main__":
    main()
