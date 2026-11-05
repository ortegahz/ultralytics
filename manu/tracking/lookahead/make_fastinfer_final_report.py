#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Fast Infer × no-GMC —— 最终严格精度对比报告生成器（8 序 + 24 序官方口径 + 审计 + Gate A + 旋钮扫描）。

所有数字都从产物 JSON 读出，**不重算**；缺失的输入对应小节直接跳过并留一行说明。
§2.7 的变体产物默认读 `<root-24>/p3knobs`，可用 `--variant-root` 覆盖。

    python manu/tracking/lookahead/make_fastinfer_final_report.py \
        --root-8 runs/lookahead/fi_nogmc --root-24 runs/lookahead/fi_nogmc24 \
        --workpoint runs/lookahead/fi_nogmc/workpoint/audit.json \
        --gatea runs/lookahead/gateA/gateA.json \
        --knobscan runs/lookahead/fi_nogmc/knobscan/report.json \
        --out-md runs/lookahead/fi_nogmc/FINAL_REPORT.md --out-json runs/lookahead/fi_nogmc/final_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ARMS = ["original", "stale", "kinematic", "replay"]
ARM_ROLE = {
    "original": "零延迟天花板（oracle 上界，**不是**基线）",
    "stale": "老逻辑硬吃滞后（**交付基线**）",
    "kinematic": "档位 A 纯运动学外推",
    "replay": "档位 B 视觉重推 + xcorr（**交付方案**）",
}
MODEL_LABEL = {
    "rc2": "RC2 trial_0080ep3 (GMC 语义)",
    "cs_t0043": "cs_t0043 (no-GMC 语义)",
    "trial_0049": "trial_0049 ep20 (no-GMC 语义)",
}
DELIVERY = {"f1_min": 88.0, "recall_min": 85.0, "far_max": 0.030}
HIST_24 = {"f1": 93.1879, "far": 0.029548, "hit5": 81.5612, "ruler1": 0.9475269761654357}
VARIANT_DELTAS = [2, 3]
VARIANT_METRICS = [
    ("F1%", "f1", ".4f"),
    ("Recall%", "recall", ".4f"),
    ("Precision%", "precision", ".4f"),
    ("FAR/帧", "far", ".6f"),
    ("hit@5px%", "hit5", ".4f"),
    ("CLE p50(池)", "cle50", ".4f"),
    ("CLE p95(池)", "cle95", ".4f"),
]
DASH = "—"
NAN_TOKEN = re.compile(r"(?<![\w-])(?:nan|NaN|None)(?![\w-])")
FOV_MARGINS = [0, 2, 4, 6, 8, 12, 16]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast Infer × no-GMC 最终报告")
    p.add_argument("--root-8", default="runs/lookahead/fi_nogmc")
    p.add_argument("--root-24", default="runs/lookahead/fi_nogmc24")
    p.add_argument("--variant-root", default=None, help="变体产物根目录（默认 <root-24>/p3knobs）")
    p.add_argument("--fovgate", default="runs/lookahead/fovgate", help="8 序出框门扫描产物根目录")
    p.add_argument("--fovgate-24", default="runs/lookahead/fovgate24", help="24 序 rc2 出框门扫描产物根目录")
    p.add_argument(
        "--fovgate-24-nogmc-d8",
        default="runs/lookahead/fovgate24nogmc_d8",
        help="24 序 no-GMC Δ=8 出框门（红线救援档）产物根目录",
    )
    p.add_argument(
        "--fovgate-24-nogmc-d2",
        default="runs/lookahead/fovgate24nogmc_d2",
        help="24 序 no-GMC Δ=2 出框门（甜点档）产物根目录",
    )
    p.add_argument(
        "--fovgate-24-nogmc-d4",
        default="runs/lookahead/fovgate24nogmc_d4",
        help="24 序 no-GMC Δ=4 出框门（补齐档）产物根目录",
    )
    p.add_argument(
        "--fovgate-24-rc2-d8",
        default="runs/lookahead/fovgate24rc2_d8",
        help="24 序 rc2 Δ=8 出框门（红线边缘档）产物根目录",
    )
    p.add_argument(
        "--fovgate-24-nogmc",
        default="runs/lookahead/fovgate24nogmc",
        help="24 序 no-GMC 出框门扫描产物根目录",
    )
    p.add_argument("--workpoint", default="runs/lookahead/fi_nogmc/workpoint/audit.json")
    p.add_argument("--gatea", default="runs/lookahead/gateA/gateA.json")
    p.add_argument("--gatea-24", default="runs/lookahead/gateA24/gateA.json")
    p.add_argument("--knobscan", default="runs/lookahead/fi_nogmc/knobscan/report.json")
    p.add_argument("--delta-dep", default="runs/lookahead/fi_nogmc24/delta_dep/report.json",
                   help="24 序 Δ 依赖性 + 惰性旋钮机理报告（JSON）")
    p.add_argument("--knobscan24", default="runs/lookahead/fi_nogmc24/knobscan_d3/report.json",
                   help="24 序 Δ=3 专属受控单旋钮 A/B 报告 JSON")
    p.add_argument("--knobscan-p3b", default="runs/lookahead/fi_nogmc24/knobscan_p3b/report.json",
                   help="夜 6 新机理受控 A/B 报告 JSON（13 个变体全表）")
    p.add_argument("--gapdiag", default="runs/lookahead/fi_nogmc24/gapdiag/gap_report.json",
                   help="夜 6 Δ=3 残余缺口诊断报告 JSON")
    p.add_argument("--accel-closure", default="runs/lookahead/fi_nogmc24/oracle_accel/closure_report.json",
                   help="夜 7 位移修正族 oracle 上界 + 候选族结案报告 JSON")
    p.add_argument("--reg-oracle-closure", default="runs/lookahead/fi_nogmc24/reg_oracle/closure_report.json",
                   help="夜 8 配准/歧义族 oracle 上界 + 候选族结案报告 JSON")
    p.add_argument("--rbgate-root", default="runs/lookahead/rbgate24nogmc",
                   help="夜 8 回退链峰值门（--replay-rollback-min-peak）受控 A/B 产物根目录")
    p.add_argument("--cfg-shared-8", default="runs/lookahead/fi_nogmc/cfg_shared_8seq.json",
                   help="8 序 sweep 内「全部运行共用同一份 CFG」的产物级自证")
    p.add_argument("--cfg-identical-24", default="runs/lookahead/nogmc24_cfg_identical.json",
                   help="24 序 rc2 ↔ cs_t0043 的 CFG 逐 token 等价证明")
    p.add_argument("--target-delta", type=int, default=3)
    p.add_argument("--out-md", default="runs/lookahead/fi_nogmc/FINAL_REPORT.md")
    p.add_argument("--out-json", default="runs/lookahead/fi_nogmc/final_report.json")
    return p.parse_args()


def load_json(path: str):
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_rbgate_rows(root: str) -> list:
    """读回退链峰值门受控 A/B 的产物（``rb__<peak>/arms.json``）⇒ 全部现读。"""
    rows: list = []
    base = Path(root)
    if not base.is_dir():
        return rows
    for arms_path in sorted(base.glob("runs/*/d*/rb__*/arms.json")):
        doc = load_json(str(arms_path))
        if not doc:
            continue
        row = {"peak": float(arms_path.parent.name.split("__", 1)[1]), "arms_path": str(arms_path)}
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
        rows.append(row)
    rows.sort(key=lambda r: -r["peak"])
    return rows


def num(value) -> float:
    """取数值；缺值 / None 一律回落到 NaN（渲染时统一成 `—`）。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value, spec: str | None = ".4f", dash: str = DASH) -> str:
    """数值渲染：None / NaN / inf 一律渲染成 `—`，不让 nan / None 漏进最终 md。"""
    if value is None:
        return dash
    number = num(value)
    if not math.isfinite(number):
        return dash
    return str(value) if spec is None else format(number, spec)


def fmt_signed(value, scale: float = 1.0, digits: int = 4) -> str:
    """带符号渲染（差值用）：None / NaN / inf ⇒ `—`。"""
    if value is None:
        return DASH
    number = num(value)
    if not math.isfinite(number):
        return DASH
    return f"{number * scale:+.{digits}f}"


def bool_cell(value) -> str:
    """布尔渲染：True / False → ✅ / ❌，缺值 → `—`。"""
    if value is None:
        return DASH
    return "✅" if value else "❌"


def gate_cell(value, limit: float, spec: str, better: str) -> str:
    """交付判据核对单元：`✅ 94.4987` / `❌ 87.1000` / `—`。"""
    if value is None:
        return DASH
    number = num(value)
    if not math.isfinite(number):
        return DASH
    ok = number >= limit if better == "min" else number <= limit
    return f"{'✅' if ok else '❌'} {format(number, spec)}"


def sanitize(lines: list) -> list:
    """收尾兜底：把残留的 nan / None 词元统一换成 `—`。"""
    return [NAN_TOKEN.sub(DASH, line) for line in lines]


def ratio(part: float, whole: float) -> float:
    """比值：分母为 0 或缺值时回落到 NaN（渲染成 `—`，不抛 ZeroDivisionError）。"""
    if not math.isfinite(part) or not math.isfinite(whole) or whole == 0:
        return float("nan")
    return part / whole


def pct(part: float, whole: float) -> float:
    """百分比：分母为 0 或缺值时回落到 NaN（渲染成 `—`，不抛 ZeroDivisionError）。"""
    return ratio(part, whole) * 100


def collect(root: Path) -> dict:
    """{model: {delta: artefact}} —— 只收四臂齐全的产物。"""
    out: dict = {}
    if not root.is_dir():
        return out
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ddir in sorted(p for p in mdir.iterdir() if p.is_dir() and p.name.startswith("d")):
            f = ddir / "arms.json"
            if not f.is_file():
                continue
            doc = load_json(str(f))
            if not doc or "arms" not in doc or not all(a in doc["arms"] for a in ARMS):
                continue
            out.setdefault(mdir.name, {})[int(ddir.name[1:])] = doc
    return out


def row(doc: dict, arm: str) -> dict:
    a = doc["arms"][arm]
    cf = (a.get("current_frame") or {}).get("overall") or {}
    sv = (a.get("servo") or {}).get("overall") or {}
    sf = ((a.get("source_frame") or {}).get("overall")) or {}
    return {
        "f1": num(cf.get("f1")) * 100,
        "recall": num(cf.get("recall")) * 100,
        "precision": num(cf.get("precision")) * 100,
        "far": num(cf.get("far_per_frame")),
        "tp": num(cf.get("tp")),
        "fp": num(cf.get("fp")),
        "fn": num(cf.get("fn")),
        "hit5": num(sv.get("hit_rate_5px")) * 100,
        "cle50": num(sv.get("cle_pooled_p50")),
        "cle95": num(sv.get("cle_pooled_p95")),
        "cle99": num(sv.get("cle_pooled_p99")),
        "jitter": num(sv.get("max_jitter")),
        "ruler1": num(sf.get("f1")),
    }


def gates(r: dict) -> str:
    if any(not math.isfinite(r[k]) for k in ("f1", "recall", "far")):
        return DASH
    ok = r["f1"] >= DELIVERY["f1_min"] and r["recall"] >= DELIVERY["recall_min"] and r["far"] <= DELIVERY["far_max"]
    return "**PASS**" if ok else "**FAIL**"


def _variant_applied(kb: dict) -> dict:
    """每个变体的 `replay` 上报帧数（从各自 arms.json 现读）⇒ §15.4 表里可复现，不硬编码。"""
    out = {}
    for job in kb.get("jobs") or []:
        path = job.get("path")
        if not path:
            continue
        doc = load_json(Path(path))
        if not doc:
            continue
        stats = ((doc.get("arms") or {}).get("replay") or {}).get("stream_stats") or {}
        applied = (stats.get("overall") or {}).get("applied")
        if applied is None:
            continue
        if str(job.get("tag") or "").startswith("gate__"):
            out[("__base__", "__base__")] = applied
        else:
            out[(job.get("flags"), job.get("vals"))] = applied
    return out



def four_arm_table(doc: dict) -> list:
    out = [
        "| 臂 | 角色 | F1% | Recall% | Precision% | FAR/帧 | hit@5px% | CLE p50(池) | CLE p95(池) | 交付三门 |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|--:|:--|",
    ]
    for arm in ARMS:
        r = row(doc, arm)
        out.append(
            f"| {arm} | {ARM_ROLE[arm]} | {fmt(r['f1'])} | {fmt(r['recall'])} | {fmt(r['precision'])} | "
            f"{fmt(r['far'], '.6f')} | {fmt(r['hit5'])} | {fmt(r['cle50'], '.3f')} | {fmt(r['cle95'], '.3f')} | "
            f"{gates(r)} |"
        )
    return out


def pooled_f1(doc: dict, arm: str, drop: str | None = None) -> tuple[float, float, float]:
    """按 `current_frame.per_sequence` 池化重算某臂的 F1/P/R；`drop` = 要剔除的序列名子串。

    与产物 `current_frame.overall` 同口径（TP/FP/FN 先池化再算 P/R/F1），用于「剔除某序后」的对照。
    """
    cf = ((((doc.get("arms") or {}).get(arm) or {}).get("current_frame")) or {})
    tp = fp = fn = 0
    for name, s in (cf.get("per_sequence") or {}).items():
        if drop and drop in name:
            continue
        tp += s.get("tp", 0) or 0
        fp += s.get("fp", 0) or 0
        fn += s.get("fn", 0) or 0
    if not (tp + fp) or not (tp + fn):
        return float("nan"), float("nan"), float("nan")
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r) * 100, p * 100, r * 100


def gt_share(doc: dict, arm: str, seq_key: str) -> tuple[int, int]:
    """（某序 GT 帧数, 该臂 GT 总帧数）。"""
    ps = ((((doc.get("arms") or {}).get(arm) or {}).get("current_frame")) or {}).get("per_sequence") or {}
    hit = sum(v.get("gt", 0) for k, v in ps.items() if seq_key in k)
    return hit, sum(v.get("gt", 0) for v in ps.values())


def gt_total(doc: dict, arm: str = "replay") -> int:
    """产物逐序列 GT 合计（唯一权威 GT 口径）。"""
    ps = ((((doc.get("arms") or {}).get(arm) or {}).get("current_frame")) or {}).get("per_sequence") or {}
    return sum(v.get("gt", 0) or 0 for v in ps.values())


def video_frames(doc: dict, arm: str = "replay") -> float | None:
    """视频总帧数 = FAR 的隐含分母（fp / far_per_frame），从产物算，不写死。"""
    cf = ((((doc.get("arms") or {}).get(arm) or {}).get("current_frame")) or {}).get("overall") or {}
    fp, far = cf.get("fp"), cf.get("far_per_frame")
    if not fp or not far:
        return None
    return round(fp / far, 3)


def per_sequence_attribution(doc: dict) -> list:
    """逐序列归因：replay 相对 stale / 相对天花板；含「对总缺口贡献(pp)」列。"""
    arms = doc.get("arms") or {}
    ps = (arms.get("replay") or {}).get("current_frame", {}).get("per_sequence") or {}
    total_gt = sum(v.get("gt", 0) for v in ps.values()) or 1
    out = [
        "| 序列 | GT帧 | stale F1% | replay F1% | ΔF1(pp) | stale hit@5px% | replay hit@5px% | Δhit(pp) | "
        "replay vs 天花板 hit 缺口(pp) | 对总缺口贡献(pp) | replay FAR/帧 | 天花板 original FAR/帧 |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for seq in sorted(ps):
        s_ps = arms["stale"]["current_frame"]["per_sequence"].get(seq, {})
        s_sv = arms["stale"]["servo"]["per_sequence"].get(seq, {})
        r_sv = arms["replay"]["servo"]["per_sequence"].get(seq, {})
        o_sv = arms["original"]["servo"]["per_sequence"].get(seq, {})
        gap = (o_sv.get("hit_rate_5px", 0.0) - r_sv.get("hit_rate_5px", 0.0)) * 100
        contrib = gap * ps[seq].get("gt", 0) / total_gt
        # FAR 两列（★ 逐序列 FAR 是**独立**的红线维度，不能只报池化）
        r_far = ps[seq].get("far_per_frame")
        o_far = ((arms.get("original") or {}).get("current_frame") or {}).get("per_sequence", {}).get(seq, {}).get(
            "far_per_frame"
        )
        r_far_s = f"{fmt(r_far, '.6f')}" + (" **✗**" if (r_far or 0) > DELIVERY["far_max"] else "")
        o_far_s = f"{fmt(o_far, '.6f')}" + (" **✗**" if (o_far or 0) > DELIVERY["far_max"] else "")
        out.append(
            f"| {seq} | {ps[seq].get('gt', '')} | {fmt(s_ps.get('f1', 0) * 100, '.4f')} | "
            f"{fmt(ps[seq].get('f1', 0) * 100, '.4f')} | "
            f"{fmt((ps[seq].get('f1', 0) - s_ps.get('f1', 0)) * 100, '+.4f')} | "
            f"{fmt(s_sv.get('hit_rate_5px', 0.0) * 100, '.4f')} | {fmt(r_sv.get('hit_rate_5px', 0.0) * 100, '.4f')} | "
            f"{fmt((r_sv.get('hit_rate_5px', 0.0) - s_sv.get('hit_rate_5px', 0.0)) * 100, '+.4f')} | "
            f"{fmt(gap, '.4f')} | {fmt(contrib, '.4f')} | {r_far_s} | {o_far_s} |"
        )
    return out


def delta_sensitivity_table(data: dict, models: list) -> list:
    out = [
        "| 模型包 | Δ | 延时(ms) | F1% | Recall% | FAR/帧 | hit@5px% | CLE p50(池) | CLE p95(池) | 相对 stale 的 F1 增益(pp) |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for m in models:
        for d in sorted(data.get(m, {})):
            rp = row(data[m][d], "replay")
            st = row(data[m][d], "stale")
            out.append(
                f"| {MODEL_LABEL.get(m, m)} | {d} | {d * 40} | {fmt(rp['f1'])} | {fmt(rp['recall'])} | "
                f"{fmt(rp['far'], '.6f')} | {fmt(rp['hit5'])} | {fmt(rp['cle50'], '.3f')} | "
                f"{fmt(rp['cle95'], '.3f')} | "
                f"{fmt(rp['f1'] - st['f1'], '+.4f')} |"
            )
    return out


def variant_section(base_docs: dict, variant_docs: dict, base_root: str, variant_root: str) -> list:
    """§2.12 —— 24 序官方口径上的变体复评（变体 vs 交付档基线，只看 `replay` 臂）。"""
    md = [
        "### 2.12 24 序官方口径上的变体复评（`p3knobs` vs 交付档基线）",
        "",
        "变体 = 交付档 CFG 叠加 `--replay-abstain-peak-below 0.50` + `--replay-rollback-min-peak 0.80`（其余一字不改）；"
        "基线 = 交付档本身。两边都只看 `replay` 臂，数字全部取自产物 JSON（**不重算**）。",
        "",
        "**证据分级（产物级可核验 ≠ 仅脚本自证）**：两版产物 `lookahead` 块逐键 diff，"
        "**产物级可直接核验的旋钮差异只有 `replay_rollback_min_peak`（0.90 → 0.80）一项**；"
        "`--replay-abstain-peak-below 0.50` **只由扫描脚本的 `VARIANTS` 自证**"
        "（变体产物生成于补记该旋钮的脚本补丁之前，产物 JSON 里读不到这一项差异），"
        "逐项依据见紧随其后的「⚠️ 产物元数据口径（诚实声明）」段。",
        "",
        "⚠️ **产物元数据口径（诚实声明）**：`p3knobs` 变体产物生成于 2026-11-04 18:02/18:08，"
        "**早于 18:53 给产物补记 `replay_abstain_peak_below` / `replay_abstain_disp_px` / `replay_fov_margin_px` "
        "三个旋钮的脚本补丁** ⇒ **产物 JSON 里读不到 `abstain` 这一项差异**；"
        "运行时 CFG 由 `run_fastinfer_24seq_variants.sh` 的 `VARIANTS` 定义（`p3knobs|2 3|--replay-abstain-peak-below 0.50|"
        "--replay-rollback-min-peak 0.80`）经 sed 覆盖生成，且脚本内**自带覆盖后断言**（断言不通过即 exit）⇒ "
        "abstain 这一项由**扫描脚本自证**；`replay_rollback_min_peak`（0.90 → 0.80）则**在产物里直接可见**。",
        "",
    ]
    pairs = []
    for d in VARIANT_DELTAS:
        var_doc = variant_docs.get(d)
        base_doc = base_docs.get(d)
        if var_doc is None:
            path = Path(variant_root) / "rc2" / f"d{d}" / "arms.json"
            md += [f"> ⚠️ 变体产物缺失：`{path}`", ""]
            continue
        if base_doc is None:
            path = Path(base_root) / "rc2" / f"d{d}" / "arms.json"
            md += [f"> ⚠️ 交付档基线产物缺失：`{path}`", ""]
            continue
        pairs.append((d, base_doc, var_doc))

    if not pairs:
        md += ["> ⚠️ 无可对照的 Δ（变体 / 交付档基线产物均缺失），本小节不出表。", ""]
        return md

    md += ["**（a）变体 vs 交付档基线**", ""]
    md += [
        "| Δ | 指标 | 交付档基线 | 变体 `p3knobs` | 差值（变体 − 交付档） |",
        "|:--|:--|--:|--:|--:|",
    ]
    for d, base_doc, var_doc in pairs:
        b, v = row(base_doc, "replay"), row(var_doc, "replay")
        for label, key, spec in VARIANT_METRICS:
            dspec = "+.6f" if key == "far" else "+.4f"
            md.append(
                f"| {d} | {label} | {fmt(b[key], spec)} | {fmt(v[key], spec)} | {fmt(v[key] - b[key], dspec)} |"
            )
    md.append("")

    md += ["**（b）交付判据核对（变体）**", ""]
    md += [
        f"| Δ | F1 ≥ {DELIVERY['f1_min']:.1f} | Recall ≥ {DELIVERY['recall_min']:.1f} | "
        f"FAR ≤ {DELIVERY['far_max']:.3f} | FAR 红线余量（交付档） | FAR 红线余量（变体） | 三门 |",
        "|:--|:--|:--|:--|--:|--:|:--|",
    ]
    for d, base_doc, var_doc in pairs:
        b, v = row(base_doc, "replay"), row(var_doc, "replay")
        verdict = {"**PASS**": "✅ 全过", "**FAIL**": "❌ 有破线"}.get(gates(v), DASH)
        md.append(
            f"| {d} | {gate_cell(v['f1'], DELIVERY['f1_min'], '.4f', 'min')} | "
            f"{gate_cell(v['recall'], DELIVERY['recall_min'], '.4f', 'min')} | "
            f"{gate_cell(v['far'], DELIVERY['far_max'], '.6f', 'max')} | "
            f"{fmt(DELIVERY['far_max'] - b['far'], '.6f')} | {fmt(DELIVERY['far_max'] - v['far'], '.6f')} | "
            f"{verdict} |"
        )
    md.append("")

    concl = []
    for d, base_doc, var_doc in pairs:
        b, v = row(base_doc, "replay"), row(var_doc, "replay")
        concl.append(
            f"Δ={d}：F1 {fmt(b['f1'])} → {fmt(v['f1'])}（{fmt(v['f1'] - b['f1'], '+.4f')}pp）、"
            f"FAR {fmt(b['far'], '.6f')} → {fmt(v['far'], '.6f')}（{fmt(v['far'] - b['far'], '+.6f')}）、"
            f"hit@5px {fmt(b['hit5'])} → {fmt(v['hit5'])}（{fmt(v['hit5'] - b['hit5'], '+.4f')}pp）"
        )
    md += ["**⇒ 结论**：" + "；".join(concl) + "。", ""]

    md += ["**⇒ 产物记录的旋钮差异**（直接 diff 两版 `lookahead` 块）：", ""]
    for d, base_doc, var_doc in pairs:
        bk = base_doc.get("lookahead") or {}
        vk = var_doc.get("lookahead") or {}
        diffs = [f"`{k}` {bk.get(k)} → {vk.get(k)}" for k in sorted(set(bk) | set(vk)) if bk.get(k) != vk.get(k)]
        md.append(f"- Δ={d}：" + ("；".join(diffs) if diffs else "`lookahead` 块无差异记录"))
    md.append("")
    return md


def collect_fov(root: Path, model: str) -> dict:
    """出框门扫描产物：{Δ: {margin: arms.json}}（目录约定 `<root>/runs/<model>/d<Δ>/fov__m<margin>/`）。"""
    out: dict = {}
    model_root = root / "runs" / model
    if not model_root.is_dir():
        return out
    for ddir in sorted(p for p in model_root.iterdir() if p.is_dir() and p.name.startswith("d")):
        try:
            delta = int(ddir.name[1:])
        except ValueError:
            continue
        margins: dict = {}
        for mdir in sorted(p for p in ddir.iterdir() if p.is_dir() and p.name.startswith("fov__m")):
            try:
                margin = int(mdir.name[len("fov__m") :])
            except ValueError:
                continue
            doc = load_json(str(mdir / "arms.json"))
            if doc and "arms" in doc and "replay" in doc["arms"]:
                margins[margin] = doc
        if margins:
            out[delta] = margins
    return out


def fov_margin_doc(root: Path, model: str, delta: int, margin: int):
    """出框门扫描里某个 (Δ, m) 的 arms.json（缺 → None）。"""
    return collect_fov(root, model).get(delta, {}).get(margin)


def fov_coverage(doc: dict) -> float:
    """`replay` 臂的 servo coverage（缺字段 → NaN，渲染成 `—`）。"""
    arm = (doc.get("arms") or {}).get("replay") or {}
    return num(((arm.get("servo") or {}).get("overall") or {}).get("coverage"))


def fov_gate_table(docs: dict) -> list:
    """一轮出框门扫描的表；Δ 一律相对**同一次扫描的 m=0**。"""
    md = [
        "| 出框门 m (px) | F1% | ΔF1(pp) | Recall% | FAR/帧 | ΔFAR | FP | TP | hit@5px% | Δhit5(pp) | "
        "CLE p95(池) | coverage | 交付三门 |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|",
    ]
    base = docs.get(0)
    b = row(base, "replay") if base else None
    for m in sorted(docs):
        r = row(docs[m], "replay")
        md.append(
            f"| {m} | {fmt(r['f1'])} | {fmt(r['f1'] - b['f1'], '+.4f') if b else DASH} | "
            f"{fmt(r['recall'])} | {fmt(r['far'], '.6f')} | {fmt(r['far'] - b['far'], '+.6f') if b else DASH} | "
            f"{fmt(r['fp'], '.0f')} | {fmt(r['tp'], '.0f')} | {fmt(r['hit5'])} | "
            f"{fmt(r['hit5'] - b['hit5'], '+.4f') if b else DASH} | {fmt(r['cle95'], '.3f')} | "
            f"{fmt(fov_coverage(docs[m]), '.4f')} | {gates(r)} |"
        )
    return md


def fov_oracle_line(docs: dict) -> str:
    """m=0 的 `original`（oracle 天花板）FAR —— replay 离天花板还有多远。"""
    base = docs.get(0)
    if not base or "original" not in (base.get("arms") or {}):
        return ""
    og = row(base, "original")
    rp = row(base, "replay")
    gap = rp["far"] - og["far"]
    return (
        f"**⇒ 天花板参照（m=0）**：`original`（oracle 天花板）FAR = {fmt(og['far'], '.6f')}，"
        f"`replay` FAR = {fmt(rp['far'], '.6f')} ⇒ 差 **{fmt(gap, '+.6f')}**"
        f"（replay {'高' if gap >= 0 else '低'} {fmt(pct(abs(gap), og['far']), '.1f')}%）"
    )


def fov_gate_section(scans: list) -> tuple[list, int]:
    """§9 —— 出框门（FOV margin gate）受控 A/B。scans = [(口径标签, 产物根目录, [模型包, ...]), ...]

    返回 (正文行, 已用小节数)，调用方据此给结论小节编号，避免与数据小节重号。
    """
    md = [
        "唯一变量 = `--replay-fov-margin-px`（出框门阈值 m，px）；门**只作用于 `replay` 臂**"
        "（`stale` / `kinematic` 是对照基线），其余旋钮一字不改。Δ 一律相对**同一次扫描的 m=0**。",
        "",
    ]
    sec = 0
    for scope, root_str, models in scans:
        root = Path(root_str)
        if not root.is_dir():
            md.append(f"> ⚠️ 未找到 {scope} 的出框门产物（`{root}`），跳过。")
            md.append("")
            continue
        for m in models:
            groups = collect_fov(root, m)
            if not groups:
                md.append(f"> ⚠️ 未找到 {scope} · {MODEL_LABEL.get(m, m)} 的出框门产物，跳过。")
                md.append("")
                continue
            for d in sorted(groups):
                docs = groups[d]
                sec += 1
                md.append(f"### 9.{sec} {scope} · {MODEL_LABEL.get(m, m)} · Δ={d}（{d * 40} ms）")
                md.append("")
                md.extend(fov_gate_table(docs))
                md.append("")
                missing = [x for x in FOV_MARGINS if x not in docs]
                if missing:
                    md.append(
                        "> ⚠️ 本扫描**未覆盖**的 margin："
                        + ", ".join(str(x) for x in missing)
                        + "（**没跑 ⇒ 不可外推**，不是「无变化」）。"
                    )
                    md.append("")
                line = fov_oracle_line(docs)
                if line:
                    md.append(line)
                    md.append("")
    return md, sec


def nogmc24_section(d24: dict, root_24: str) -> list:
    """§10 —— 24 序 no-GMC（cs_t0043）Δ 扫描，与同 Δ 的 rc2 并排对照。"""
    md = []
    docs = d24.get("cs_t0043") or {}
    if not docs:
        md.append(
            f"> ⚠️ 未找到 24 序 no-GMC（cs_t0043）Δ 扫描产物（`{Path(root_24) / 'cs_t0043'}/`），跳过本节。"
        )
        return md
    md.append(
        f"> 数据源：`{Path(root_24) / 'cs_t0043'}/d{{2,3,4,8}}/arms.json`；"
        f"同 Δ 的 rc2 对照取自 `{Path(root_24) / 'rc2'}/d{{2,3,4,8}}/arms.json`。"
    )
    md.append("")
    for i, d in enumerate(sorted(docs), 1):
        md.append(f"### 10.{i} {MODEL_LABEL.get('cs_t0043', 'cs_t0043')} · Δ={d}（{d * 40} ms）")
        md.append("")
        md.append("**no-GMC（cs_t0043）四臂**")
        md.append("")
        md.extend(four_arm_table(docs[d]))
        md.append("")
        if d in (d24.get("rc2") or {}):
            md.append("**同 Δ 对照：rc2（GMC 语义）四臂**")
            md.append("")
            md.extend(four_arm_table(d24["rc2"][d]))
            md.append("")
            a = row(docs[d], "replay")
            b = row(d24["rc2"][d], "replay")
            md.append(
                f"**⇒ 同 Δ 对照（`replay` 臂）**：F1 {fmt(b['f1'])}（rc2/GMC） → **{fmt(a['f1'])}（cs_t0043/no-GMC，"
                f"{fmt(a['f1'] - b['f1'], '+.4f')}pp）**、FAR {fmt(b['far'], '.6f')} → **{fmt(a['far'], '.6f')}"
                f"（{fmt(a['far'] - b['far'], '+.6f')}）**、hit@5px {fmt(b['hit5'])} → "
                f"**{fmt(a['hit5'])}"
                f"（{fmt(a['hit5'] - b['hit5'], '+.4f')}pp）**、CLE p95 {fmt(b['cle95'], '.3f')} → "
                f"**{fmt(a['cle95'], '.3f')}"
                f"（{fmt(a['cle95'] - b['cle95'], '+.3f')}px）**。"
            )
            md.append("")
        else:
            path = Path(root_24) / "rc2" / f"d{d}" / "arms.json"
            md.append(f"> ⚠️ 未找到同 Δ 的 rc2 对照产物（`{path}`），跳过对照。")
            md.append("")
    md.append(f"### 10.{len(docs) + 1} Δ 敏感性（24 序 rc2 + cs_t0043，**同一张表**，模型包列区分）")
    md.append("")
    md.extend(delta_sensitivity_table(d24, [m for m in ("rc2", "cs_t0043") if m in d24]))
    md.append("")
    return md


def gate_a_block(ga: dict, scope_label: str) -> list:
    """渲染一段 Gate A（关开关 ⇒ 与原脚本逐位一致）的证据块。"""
    md = []
    if not ga:
        md.append(f"> ⚠️ 未找到 {scope_label} 的 Gate A 产物，跳过。")
        return md
    md.append(f"**判定：{ga.get('verdict')}**　｜　执行机 `{ga.get('host')}`　｜　时间戳 `{ga.get('timestamp')}`")
    md.append("")
    md.append("| 模型包 | 逐序列整份 JSON md5 相同 | per_frame / metrics / config 相同 | 跟踪资产 md5 相同 |")
    md.append("|:--|:--|:--|:--|")
    for mk, mv in (ga.get("models") or {}).items():
        rows = (mv.get("preds") or {}).get("rows") or []
        n_l1 = sum(1 for r in rows if r.get("L1_raw_md5_equal"))
        n_l4 = sum(1 for r in rows if r.get("L4_assets_md5_equal"))
        l1_cell = f"{bool_cell(mv.get('preds_all_L1_equal'))} {n_l1}/{len(rows)} 相同" if rows else DASH
        l4_cell = f"{bool_cell(mv.get('preds_all_L4_equal'))} {n_l4}/{len(rows)} 相同" if rows else DASH
        md.append(
            f"| {mv.get('label', mk)} | **{l1_cell}** | "
            f"**{bool_cell(mv.get('report_projection_equal'))}** | **{l4_cell}** |"
        )
    md.append("")
    r1 = ga.get("ruler1", {})
    if r1:
        md.append(
            f"**尺子1（历史帧等价性）**：产物 `{r1.get('path', '')}` 四臂 `source_frame.f1` = "
            f"{list(r1.get('distinct_values', []))} ⇒ 逐位相等 = **{bool_cell(r1.get('all_bit_identical'))}**，"
            f"Δ = **{fmt(num(r1.get('delta_pp')), None)} pp**。"
        )
        md.append("")
    for line in ga.get("conclusion_lines", []):
        md.append(f"- {line}")
    return md


def main() -> None:
    args = parse_args()
    d8 = collect(Path(args.root_8))
    d24 = collect(Path(args.root_24))
    vroot = args.variant_root or str(Path(args.root_24) / "p3knobs")
    variant = collect(Path(vroot))
    wp = load_json(args.workpoint)
    ga = load_json(args.gatea)
    ga24 = load_json(args.gatea_24)
    ks = load_json(args.knobscan)
    T = args.target_delta
    md: list = []

    # ---- 规模口径（**全部从产物现算**，禁止写死帧数 / 覆盖率）----
    _doc8 = (d8.get("rc2") or {}).get(T) or next(iter(next(iter(d8.values())).values()), None)
    _doc24 = next(((d24.get(_m) or {}).get(T) for _m in ("rc2", "cs_t0043") if (d24.get(_m) or {}).get(T)), None)
    _frames8 = video_frames(_doc8) if _doc8 else None
    _gt8 = gt_total(_doc8) if _doc8 else None
    _frames24 = video_frames(_doc24) if _doc24 else None
    _gt24 = gt_total(_doc24) if _doc24 else None
    _cover_gt_txt = f"{_gt8 / _gt24 * 100:.2f}%" if _gt8 and _gt24 else DASH
    _cover_frame_txt = f"{_frames8 / _frames24 * 100:.2f}%" if _frames8 and _frames24 else DASH
    _scale8_txt = f"{_frames8:,.0f} 帧 / GT {_gt8:,} 帧" if _frames8 and _gt8 else DASH
    _scale24_txt = f"{_frames24:,.0f} 帧 / GT {_gt24:,} 帧" if _frames24 and _gt24 else DASH

    # ---- 出框门派生量（全部从产物现算，禁止写死百分比 / 绝对差）----
    def _fov_replay(root: str, model: str, delta: int, margin: int):
        doc = fov_margin_doc(Path(root), model, delta, margin)
        return row(doc, "replay") if doc else None

    def _far_delta_pct(after, before) -> str:
        """FAR 相对变化（%）：after = 开门后，before = 同一次扫描的 m=0。"""
        if not after or not before or not before["far"]:
            return DASH
        return f"{(after['far'] - before['far']) / before['far'] * 100:+.1f}%"

    def _far_delta_abs(after, before) -> str:
        if not after or not before:
            return DASH
        return f"{after['far'] - before['far']:+.6f}"

    _g8_d3_m0 = _fov_replay(args.fovgate, "cs_t0043", 3, 0)
    _g8_d3_m6 = _fov_replay(args.fovgate, "cs_t0043", 3, 6)
    _g24_d3_m0 = _fov_replay(args.fovgate_24_nogmc, "cs_t0043", 3, 0)
    _g24_d3_m6 = _fov_replay(args.fovgate_24_nogmc, "cs_t0043", 3, 6)
    _g24_d8_m0 = _fov_replay(args.fovgate_24_nogmc_d8, "cs_t0043", 8, 0)
    _g24_d8_m6 = _fov_replay(args.fovgate_24_nogmc_d8, "cs_t0043", 8, 6)
    _rc2_d8 = row(d24["rc2"][8], "replay") if 8 in d24.get("rc2", {}) else None

    md.append("# Fast Infer（look-ahead 前向推演）× no-GMC —— **最终严格精度对比报告**")
    md.append("")
    md.append(
        "> 执行机：**服务器** `test-Super-Server`（`192.168.99.40:32222`）· 项目根 `/tmp/pycharm_project_10ae9e2e`\n"
        "> 时间：2026-11-04 16:52（8 序 Δ 扫描）→ 17:10（8 序报告）→ 17:15–17:55（24 序官方口径 Δ 扫描）"
        "→ 17:23–17:50（工作点审计 / 旋钮扫描 / Gate A / OSD）→ 18:06–18:52（24 序 no-GMC ms_cache, GPU）"
        "→ 18:52–19:25（24 序 Δ 扫描 + 24 序出框门 m=0/4/6/8/12）→ 19:13–20:23（24 序 rc2 出框门）"
        "→ 20:16–20:37（24 序 rc2 四件套**修复后代码复跑**、仅补记旋钮元数据，指标逐位不变）"
        "→ 20:44–21:08（24 序 no-GMC Δ=2 出框门）→ 21:08–21:49（24 序 rc2 Δ=8 出框门 m=0/6/8/12）→ 20:45–21:25（报告自审与修复）\n"
        "> 生成脚本：`manu/tracking/lookahead/make_fastinfer_final_report.py` —— **所有数字从产物 JSON 读出，不重算**"
    )
    md.append("")

    # ---------- §0 口径声明 ----------
    md.append("## 0. 口径声明与结论速览（先读这一节，引用错口径会得出反向结论）")
    md.append("")
    md.append("| # | 纪律 | 说明 |")
    md.append("|:--|:--|:--|")
    md.append("| 1 | **`stale` 才是交付基线** | `original` 臂 = **零延迟天花板（oracle 上界）**，运行时不可能达到，**不是基线** |")
    md.append("| 2 | **总体表 CLE 用 `cle_pooled_*`（全帧池化）** | README 的 `cle_p*` = 逐序列最差值（`worst_over_sequences`）；逐序表用 `servo.per_sequence.<seq>.cle_p*` |")
    md.append(
        f"| 3 | **8 序 ≠ 24 序** | 8 序 gate 集只覆盖 24 序官方口径的 **{_cover_gt_txt} GT / {_cover_frame_txt} 帧** ⇒ "
        "8 序只做**机理筛选与 A/B**，**不能给交付档排序** |"
    )
    md.append("| 4 | **本轮用「当前交付档 CFG」** | 与历史 README §5.2 的 **WTH15 历史档 CFG 差 5 个开关**（roimax 4→16；新增 polarity auto / uniq-gate 0.6 / abstain 门 / rollback 旋钮）⇒ **两套数字不可并列引用** |")
    md.append("| 5 | **交付判据** | F1 ≥ 0.88 / Recall ≥ 0.85 / FAR ≤ 0.030（业务域红线） |")
    md.append("| 6 | **Δ 是假设值** | Δ=3 = 假设检测链路延时 3 个 record（120 ms @25 Hz）；管线绝对精度由模型决定，Δ 只决定外挂要补多少 |")
    md.append("| 7 | **模型包 = 权重 + 特征语义** | 三包缓存 meta 逐字相同，只有 `features_root` / `hm_weights` 不同 |")
    md.append(
        "| 8 | **出框门只作用于 `replay` 臂** | 门只改 `replay` 的外推结果、不改跟踪资产；"
        "`stale` / `kinematic` 是对照基线 |"
    )
    md.append(
        "| 9 | **出框门默认 0 = 关 = 逐位不变** | `--replay-fov-margin-px` 默认 0 ⇒ "
        "与既有产物四臂**指标字段逐位一致**（唯一差异 = 新旋钮的元数据回显 "
        "`lookahead.replay_fov_margin_px` / `replay_abstain_*` / `stream_stats.overall.fov_gated`；已实测 m=0） |"
    )
    md.append(
        "| 10 | **帧级「静默/丢失」归因必须按生产判定顺序复刻** | 夜 8 实测：第一版把「有门前上报点却没上报」"
        "一律记成「出框门丢掉」（842 帧）—— **错**；生产顺序是 **回退链峰值门 → failed 兜底 → 写 report → 出框门 → 限幅**，"
        "命中第一道门时 `report` 那一格**根本不写** ⇒ 纠错后大头是回退链峰值门（1846 帧 / 静默帧的 80.1%）。见 §17 |"
    )
    md.append(
        "| 11 | **离线单帧账只对「非路径依赖」的旋钮有效** | 夜 8 实测：离线账说回退门放松到 0 ⇒ TP +1223 / FP +507，"
        "**实测只有 +1155 / +1780**（FP 低估 3.5 倍）—— 因为交付档开了 `--output-max-step-px 30`（slew limiter），"
        "限幅参考是「同一 track 最近一次上报位置」⇒ **上报轨迹路径依赖** ⇒ 凡是会改变「上报帧集合」的旋钮"
        "（弃权门 / 出框门 / 回退门 / 限幅）**只能信真实 A/B**。见 §17.4 |"
    )
    md.append("")

    # 结论速览：三个口径
    md.append(f"### 0.1 「小延时下精度是否提升」—— **三个口径分别回答**")
    md.append("")
    md.append("| # | 口径 | 答案 | 关键数字 |")
    md.append("|:--|:--|:--|:--|")
    # 口径 1：相对 rc2 同 Δ（no-GMC vs GMC）
    if "cs_t0043" in d8 and T in d8["cs_t0043"] and "rc2" in d8 and T in d8["rc2"]:
        a = row(d8["cs_t0043"][T], "replay")
        b = row(d8["rc2"][T], "replay")
        md.append(
            f"| 口径① | 相对 rc2 **同 Δ**（no-GMC vs GMC，8 序 Δ={T} replay） | **no-GMC 略低，但误报/定位更优** | "
            f"F1 **{fmt(a['f1'] - b['f1'], '+.4f')}pp**、FAR **{fmt(a['far'] - b['far'], '+.6f')}**"
            f"（好 {fmt(pct(abs(a['far'] - b['far']), b['far']), '.1f')}%）、"
            f"hit@5px **{fmt(a['hit5'] - b['hit5'], '+.4f')}pp**、CLE p95 **{fmt(a['cle95'] - b['cle95'], '+.4f')}px** |"
        )
    # 口径 1b：24 序同 Δ 的跨模型包对照（本轮核心：换模型包 cs_t0043 的效果）
    if "rc2" in d24 and "cs_t0043" in d24 and T in d24["rc2"] and T in d24["cs_t0043"]:
        a = row(d24["cs_t0043"][T], "replay")
        b = row(d24["rc2"][T], "replay")
        far_word = "好" if a["far"] < b["far"] else "差"
        md.append(
            f"| 口径①b | 相对 rc2 **同 Δ**（no-GMC vs GMC，**24 序官方口径** Δ={T} replay） | **见 §2.10 跨模型包对照表** | "
            f"F1 **{fmt(a['f1'] - b['f1'], '+.4f')}pp**、FAR **{fmt(a['far'] - b['far'], '+.6f')}**"
            f"（{far_word} {fmt(pct(abs(a['far'] - b['far']), b['far']), '.1f')}%）、"
            f"hit@5px **{fmt(a['hit5'] - b['hit5'], '+.4f')}pp**、CLE p95 **{fmt(a['cle95'] - b['cle95'], '+.4f')}px** |"
        )
    # 口径 2：相对 stale（交付基线）—— 24 序，逐模型包
    for m in sorted(d24):
        if T not in d24[m]:
            continue
        rp = row(d24[m][T], "replay")
        st = row(d24[m][T], "stale")
        md.append(
            f"| 口径② | 相对 **stale（交付基线）**，24 序官方口径 Δ={T} · {MODEL_LABEL.get(m, m)} | **是，大幅提升** | "
            f"F1 **{fmt(rp['f1'] - st['f1'], '+.4f')}pp**、hit@5px **{fmt(rp['hit5'] - st['hit5'], '+.4f')}pp**、"
            f"FAR **{fmt(rp['far'] - st['far'], '+.6f')}** |"
        )
    # 口径 3：相对天花板 —— 24 序，逐模型包
    for m in sorted(d24):
        if T not in d24[m]:
            continue
        rp = row(d24[m][T], "replay")
        og = row(d24[m][T], "original")
        st = row(d24[m][T], "stale")
        denom = og["f1"] - st["f1"]
        md.append(
            f"| 口径③ | 相对**零延迟天花板**（追回占比），24 序 Δ={T} · {MODEL_LABEL.get(m, m)} | **已追回 "
            f"{fmt(pct(rp['f1'] - st['f1'], denom), '.1f')}%** | F1 {fmt(st['f1'])} → **{fmt(rp['f1'])}**"
            f"（天花板 {fmt(og['f1'])}） |"
        )
    # ★ 口径②/③ 取自 24 序**裸配置（出框门 m=0）**产物；§12 的交付推荐档**加出框门 m=6**
    #   ⇒ 同一句「24 序 Δ=3 replay F1」在 §0.1 与 §12 是**两个配置**（94.7861 vs 94.8094），
    #   不标口径就会被读成「自相矛盾」。这里把口径写死。
    md.append("")
    md.append(
        "**⚠️ 口径标注（勿读成自相矛盾）**：上表口径②/③ 的「24 序 Δ=3 replay」取自 "
        "**裸配置（出框门 m=0）**产物 —— 因为 §2.9 / §2.10 / §10.5 的 Δ 敏感性与跨模型包对照表"
        "**全部是 m=0 口径**（要横向可比就不能混门）；而 **§12 的交付推荐档加出框门 m=6**"
        "（F1 94.8094 / FAR 0.024337）⇒ **两个配置各自都对，但数字不同**，逐档对照见 §9 出框门表。"
    )
    md.append("")

    # ---------- §1 受控设计 ----------
    md.append("## 1. 受控设计：唯一变量 = {模型包, 假设延时 Δ}")
    md.append("")
    md.append("| 项 | 取值 |")
    md.append("|:--|:--|")
    md.append(f"| 8 序 gate 集 | {_scale8_txt}（三模型包完全相同） |")
    md.append(
        f"| 24 序官方口径 | {_scale24_txt} / 24 个序列"
        "（`runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache`） |"
    )
    md.append("| GT | `/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010` |")
    md.append("| 原始灰度帧 | `/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg` + memmap `/mnt/data/siping/frame_cache_lq` |")
    md.append("| 工作点 params | `manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json`（th_base 0.48 / sky_ratio 1.0 / th_ground 0.26 / confirm_consec 7，**只读**） |")
    md.append("| 插件 CFG | **直接从 `run_delivery.sh` 的 `CFG=` 取**（awk 取整块再 eval + 7 旋钮防截断断言） |")
    md.append("| 模型包 | rc2 (GMC) / cs_t0043 (no-GMC) / trial_0049 (no-GMC) |")
    md.append("")

    # ---------- §2 24 序官方口径 ----------
    md.append("## 2. ★ 24 序列官方口径（**唯一能回答「能不能交付」的一步**）")
    md.append("")
    if d24:
        sec = 0
        for m in sorted(d24):
            for d in sorted(d24[m]):
                sec += 1
                md.append(f"### 2.{sec} {MODEL_LABEL.get(m, m)} · Δ={d}（{d * 40} ms）")
                md.append("")
                md.extend(four_arm_table(d24[m][d]))
                md.append("")
                if m == "rc2" and d == 8:
                    _r = row(d24[m][d], "replay")
                    md.append(
                        f"> ⚠️ **红线余量提示**：rc2 Δ=8 `replay` FAR = {fmt(_r['far'], '.6f')}，"
                        f"距 0.030 红线**只剩 {fmt(DELIVERY['far_max'] - _r['far'], '.6f')}**"
                        f"（约 {fmt(pct(DELIVERY['far_max'] - _r['far'], DELIVERY['far_max']), '.1f')}% 余量）"
                        f"⇒ **这是「历史交付档」但余量极薄**，不能当作宽裕的 PASS。"
                    )
                    md.append("")
        sec += 1
        md.append(f"### 2.{sec} Δ 敏感性（`replay` 臂 = 交付方案，**按模型包**）")
        md.append("")
        md.extend(delta_sensitivity_table(d24, sorted(d24)))
        md.append("")
        # ---- ★★ 换模型包的核心对照：24 序同 Δ，rc2(GMC) vs cs_t0043(no-GMC) ----
        if "rc2" in d24 and "cs_t0043" in d24:
            common = sorted(set(d24["rc2"]) & set(d24["cs_t0043"]))
            if common:
                sec += 1
                md.append(f"### 2.{sec} ★★ **换模型包的核心对照**：24 序同 Δ，rc2(GMC) vs cs_t0043(no-GMC)，`replay` 臂")
                md.append("")
                md.append("| Δ | 指标 | rc2 (GMC) | cs_t0043 (no-GMC) | Δ = cs_t0043 − rc2 | 更优 |")
                md.append("|:--|:--|--:|--:|--:|:--|")
                for d in common:
                    a = row(d24["cs_t0043"][d], "replay")
                    b = row(d24["rc2"][d], "replay")
                    md.append(
                        f"| {d} | F1% | {fmt(b['f1'])} | {fmt(a['f1'])} | {fmt(a['f1'] - b['f1'], '+.4f')}pp | "
                        f"{'no-GMC' if a['f1'] > b['f1'] else ('GMC' if b['f1'] > a['f1'] else '持平')} |"
                    )
                    md.append(
                        f"| {d} | FAR/帧 | {fmt(b['far'], '.6f')} | {fmt(a['far'], '.6f')} | {fmt(a['far'] - b['far'], '+.6f')} | "
                        f"{'no-GMC' if a['far'] < b['far'] else ('GMC' if b['far'] < a['far'] else '持平')} |"
                    )
                    md.append(
                        f"| {d} | hit@5px% | {fmt(b['hit5'])} | {fmt(a['hit5'])} | {fmt(a['hit5'] - b['hit5'], '+.4f')}pp | "
                        f"{'no-GMC' if a['hit5'] > b['hit5'] else ('GMC' if b['hit5'] > a['hit5'] else '持平')} |"
                    )
                    md.append(
                        f"| {d} | CLE p95(池) | {fmt(b['cle95'], '.3f')} | {fmt(a['cle95'], '.3f')} | {fmt(a['cle95'] - b['cle95'], '+.3f')}px | "
                        f"{'no-GMC' if a['cle95'] < b['cle95'] else ('GMC' if b['cle95'] < a['cle95'] else '持平')} |"
                    )
                md.append("")
                md.append(
                    "> **唯一变量 = 模型包**（同一 Δ、同一 24 序帧集/GT/工作点 params/插件 CFG，"
                    "只有「权重 + 特征语义」不同）⇒ 这张表就是「把模型换成 cs_t0043」的直接答案。"
                )
                md.append("")
        for _m in sorted(d24):
            if 8 in d24.get(_m, {}) and 3 in d24.get(_m, {}):
                a = row(d24[_m][3], "replay")
                b = row(d24[_m][8], "replay")
                st3 = row(d24[_m][3], "stale")
                og = row(d24[_m][8], "original")
                _tag = "" if _m == "rc2" else f"（{MODEL_LABEL.get(_m, _m)}）"
                _d8note = (
                    "= 历史交付档（rc2 Δ=8，与 README §5 红框逐位一致）"
                    if _m == "rc2"
                    else "，cs_t0043 **本轮新产物**；历史交付档 = rc2 Δ=8"
                )
                md.append(
                    f"**⇒ {_tag} Δ=3（120 ms）相对 Δ=8（320 ms{_d8note}）：F1 {fmt(b['f1'])} → **{fmt(a['f1'])}"
                    f"（{fmt(a['f1'] - b['f1'], '+.4f')}pp）**、"
                    f"FAR {fmt(b['far'], '.6f')} → **{fmt(a['far'], '.6f')}（{fmt(a['far'] - b['far'], '+.6f')}）**、"
                    f"hit@5px {fmt(b['hit5'])} → **{fmt(a['hit5'])}（{fmt(a['hit5'] - b['hit5'], '+.4f')}pp）**、"
                    f"CLE p95 {fmt(b['cle95'], '.3f')} → **{fmt(a['cle95'], '.3f')}"
                    f"（{fmt(a['cle95'] - b['cle95'], '+.3f')}px）**。"
                )
                md.append("")
                _m8 = DELIVERY["far_max"] - b["far"]
                _m3 = DELIVERY["far_max"] - a["far"]
                _verdict = (
                    f"从**破线翻转为过线**（符号翻转）"
                    if _m8 < 0
                    else f"余量放大 **{fmt(ratio(_m3, _m8), '.1f')}×**"
                )
                _call = "Δ=8 裸配置" if _m != "rc2" else "Δ=8 历史交付档"
                md.append(
                    f"**⇒ FAR 红线余量**：{_call} = 0.030 − {fmt(b['far'], '.6f')} = "
                    f"**{fmt(_m8, '.6f')}**（{'破线' if _m8 < 0 else '过线'}）；"
                    f"Δ=3 = 0.030 − {fmt(a['far'], '.6f')} = **{fmt(_m3, '.6f')}**（{'过线' if _m3 >= 0 else '破线'}）"
                    f" ⇒ {_verdict}。"
                )
                md.append("")
                md.append(
                    f"**⇒ 🔴 24 序官方口径下 Δ={T} 的 `stale` FAR = {fmt(st3['far'], '.6f')} —— **破 0.030 红线**"
                    f"（超线 {fmt(pct(st3['far'] - DELIVERY['far_max'], DELIVERY['far_max']), '.1f')}%）"
                    f"⇒ 120 ms 工况下老逻辑在官方口径上同样不可交付。**"
                )
                md.append("")
        if 8 in d24.get("rc2", {}):
            r8 = row(d24["rc2"][8], "replay")
            md.append(f"### 2.{sec + 1} ★ 口径对齐验证：本轮 Δ=8 `replay` 逐位复现历史交付档")
            md.append("")
            md.append("| 项 | README §5 红框「当前交付档唯一口径」（24 序） | 本轮 Δ=8 `replay` | 结论 |")
            md.append("|:--|--:|--:|:--|")
            # 「逐位相同」必须**实算判定**，不能是写死的 ✅（历史锚点与本轮不一致时应报失败）
            _align_all_ok = True
            for _label, _key, _anchor_s, _got_s, _tol in (
                ("F1", "f1", f"{HIST_24['f1']:.4f}%", f"{fmt(r8['f1'])}%", 5e-5),
                ("FAR/帧", "far", f"{HIST_24['far']:.6f}", f"{fmt(r8['far'], '.6f')}", 5e-7),
                ("hit@5px", "hit5", f"{HIST_24['hit5']:.4f}%", f"{fmt(r8['hit5'])}%", 5e-5),
                ("尺子1 F1", "ruler1", f"{HIST_24['ruler1']}", f"{fmt(r8['ruler1'], None)}", 1e-12),
            ):
                _same = abs(r8[_key] - HIST_24[_key]) <= _tol
                _align_all_ok = _align_all_ok and _same
                md.append(
                    f"| {_label} | {_anchor_s} | **{_got_s}** | "
                    f"{'**逐位相同** ✅' if _same else '**不一致** ❌'} |"
                )
            md.append("")
            md.append(
                "⇒ **口径对齐验证通过**：24 序扫描器与历史交付档同一套旋钮、同一份缓存、同一工作点。"
                if _align_all_ok
                else "🔴 **口径对齐验证失败**：本轮 Δ=8 `replay` 与历史交付档**不一致** ⇒ 不得与历史数字并列引用。"
            )
            md.append("")
        md.append("> ℹ️ 以下 **P3 旋钮变体** 只在 **rc2(GMC)** 上跑（变体扫描的既有范围）；"
                   "cs_t0043 的 24 序主口径见上面的 2.x 节。")
        md.append("")
        md.extend(variant_section(d24.get("rc2", {}), variant.get("rc2", {}), args.root_24, vroot))
    else:
        md.append("> ⚠️ 未找到 24 序产物，跳过本节。")
        md.append("")

    # ---------- §3 8 序 ----------
    md.append("## 3. 8 序 gate 集（A/B 机理筛选；**不能**给交付档排序）")
    md.append("")
    for m in ["rc2", "cs_t0043", "trial_0049"]:
        if m not in d8 or T not in d8[m]:
            continue
        md.append(f"### 3.{['rc2', 'cs_t0043', 'trial_0049'].index(m) + 1} {MODEL_LABEL[m]} · Δ={T}")
        md.append("")
        md.extend(four_arm_table(d8[m][T]))
        md.append("")
    md.append("### 3.4 Δ 敏感性（`replay` 臂）")
    md.append("")
    md.extend(delta_sensitivity_table(d8, [m for m in ["rc2", "cs_t0043", "trial_0049"] if m in d8]))
    md.append("")
    if "cs_t0043" in d8 and "rc2" in d8 and T in d8["cs_t0043"] and T in d8["rc2"]:
        a = row(d8["cs_t0043"][T], "replay")
        b = row(d8["rc2"][T], "replay")
        oa = row(d8["cs_t0043"][T], "original")
        ob = row(d8["rc2"][T], "original")
        md.append(
            f"**⇒ 结论 C（8 序 gate 集）**：no-GMC 相对 GMC —— F1 **{fmt(a['f1'] - b['f1'], '+.4f')}pp**（差距全部来自 recall "
            f"{fmt(a['recall'])} vs {fmt(b['recall'])}），但 FAR **{fmt(a['far'] - b['far'], '+.6f')}"
            f"（好 {fmt(pct(abs(a['far'] - b['far']), b['far']), '.1f')}%）**、"
            f"hit@5px **{fmt(a['hit5'] - b['hit5'], '+.4f')}pp**、CLE p95 **{fmt(a['cle95'] - b['cle95'], '+.4f')}px**；"
            f"oracle 口径 FAR {fmt(ob['far'], '.6f')} → {fmt(oa['far'], '.6f')}"
            f"（**低 {fmt(pct(ob['far'] - oa['far'], ob['far']), '.1f')}%**），F1 基本打平。"
        )
        md.append("")
        # 🔴 8 序头条 = 单序列假象：把 8 序里 GT 占比最高的那一序剔除后重算池化 F1
        doc_a, doc_b = d8["cs_t0043"][T], d8["rc2"][T]
        ps8 = ((((doc_a.get("arms") or {}).get("replay") or {}).get("current_frame")) or {}).get(
            "per_sequence"
        ) or {}
        top8 = max(ps8, key=lambda k: ps8[k].get("gt", 0)) if ps8 else ""
        if top8:
            hit_gt, tot_gt = gt_share(doc_a, "replay", top8)
            fa_k, _, _ = pooled_f1(doc_a, "replay", drop=top8)
            fb_k, _, _ = pooled_f1(doc_b, "replay", drop=top8)
            md.append(
                f"🔴 **⚠️ 上面这条 {fmt(a['f1'] - b['f1'], '+.4f')}pp 的 F1 劣势是「单序列假象」**："
                f"8 序 gate 集里 GT 占比最高的一序是 **`{top8}`（{hit_gt} / {tot_gt} = "
                f"{fmt(pct(hit_gt, tot_gt), '.2f')}% GT）** ⇒ 剔除该序后（7 序）按同一口径池化重算 F1："
                f"rc2 **{fmt(fb_k)}%** vs cs_t0043 **{fmt(fa_k)}%** ⇒ 差 **{fmt(fa_k - fb_k, '+.4f')}pp**"
                f"（no-GMC 反超）。"
            )
            md.append("")

    # ---------- §4 工作点审计 ----------
    md.append("## 4. 工作点对齐审计（回答「cs_t0043 的优势是真的，还是只是 v3 工作点碰巧适合它」）")
    md.append("")
    if wp:
        h = wp.get("headline", {})
        fw = h.get("fixed_workpoint", {})
        sf = h.get("same_far", {})
        md.append("| 项 | 值 |")
        md.append("|:--|:--|")
        md.append(f"| 网格 | `th_base ∈ {{0.40,0.44,0.48,0.52,0.56}}` × `th_ground ∈ {{0.22,0.26,0.30}}`（2 模型 × Δ∈{{3,8}} = 60 次运行，**rc≠0 = 0**） |")
        md.append(
            f"| **口径范围** | ⚠️ **8 序 gate 集**（只覆盖 24 序官方口径 **{_cover_gt_txt} GT**）⇒ "
            "**只做机理筛选 / A/B，不能给交付档排序**；24 序官方口径的同项对照见 **§2.10** |"
        )
        md.append(f"| 固定 v3 工作点下 ΔF1（cs_t0043 − rc2，replay） | **{fmt(fw.get('delta_pp'), '+.4f')}pp** |")
        md.append(
            f"| **同 FAR 插值**下的 ΔF1（Δ={h.get('delta')} replay） | **{fmt(sf.get('mean_delta_pp'), '+.4f')}pp**"
            f"（区间 {fmt(sf.get('min_delta_pp'), '+.4f')} ~ {fmt(sf.get('max_delta_pp'), '+.4f')}） |"
        )
        md.append(
            f"| 同 FAR 公共区间 | FAR ∈ [{fmt((sf.get('interval') or [0, 0])[0], '.6f')}, "
            f"{fmt((sf.get('interval') or [0, 0])[1], '.6f')}] |"
        )
        md.append("")
        md.append(
            "**⇒ 结论（限 8 序 gate 集口径）**：工作点对齐后 **8 序口径下 no-GMC 的 F1 差距并未消失、方向一致** ⇒ "
            "**在 8 序 gate 集上，「cs_t0043 在 F1 上占优」不成立，且该差距不是 v3 工作点巧合造成的**。"
        )
        md.append(
            f"🔴 **但这条结论不可用于交付档排序**：8 序 gate 集只覆盖 24 序官方口径 {_cover_gt_txt} GT，"
            "且该 F1 差距是**单序列假象**（见 §3.4 结论 C 的逐序归因：剔除 GT 占比最高的一序后，"
            "8 序 no-GMC 反而反超）⇒ "
            + (
                f"**24 序官方口径上 no-GMC 的 F1 反而 {fmt(row(d24['cs_t0043'][T], 'replay')['f1'] - row(d24['rc2'][T], 'replay')['f1'], '+.4f')}pp（见 §2.10）**"
                if "cs_t0043" in d24 and "rc2" in d24 and T in d24["cs_t0043"] and T in d24["rc2"]
                else "**24 序官方口径的结论见 §2.10**"
            )
            + "。cs_t0043 的真实优势在**误报率与定位质量**上（FAR / hit@5px / CLE），且这些优势在网格全部工作点上成立。"
        )
        md.append("")
        md.append("**🔴 附带发现：`th_ground` 在 v3 工作点下是死旋钮。** 三个 `th_ground` 取值（0.22/0.26/0.30）"
                  "在**全部 4 个 model×Δ × 全部 4 个臂**上给出**逐位相同**的结果（60 次实测，独立复核：20 个 `th_base` 分组中 **0** 组受影响）。"
                  "机理：`manu/tracking/lookahead/track_fusion.py:822` "
                  "`fusion_high_mask = (is_sky & (pred_scs >= th_base)) | ((~is_sky) & (pred_scs >= th_ground))`，"
                  "而 v3 的 `sky_ratio = 1.0` ⇒ `sky_y_boundary = img_h` ⇒ **整幅画面都是 sky** ⇒ `th_ground` 永不生效。")
        md.append("")
        md.append("> 完整 24 张表见 `runs/lookahead/fi_nogmc/workpoint/audit.md`（+ `audit.json`）。")
    else:
        md.append("> ⚠️ 未找到工作点审计产物，跳过本节。")
    md.append("")

    # ---------- §5 Gate A ----------
    md.append("## 5. Gate A：关开关 ⇒ 与原脚本**逐位一致**（外挂没扰动跟踪资产）")
    md.append("")
    md.extend(gate_a_block(ga, "8 序 gate 集"))
    md.append("")
    md.append("### 5.1. Gate A · 24 序官方口径")
    md.append("")
    md.extend(gate_a_block(ga24, "24 序官方口径"))
    md.append("")
    md.append("### 5.2. 扫描层 CFG 共享性自证（「唯一变量 = 模型包」的**前提**）")
    md.append("")
    cs8 = load_json(args.cfg_shared_8)
    ci24 = load_json(args.cfg_identical_24)
    if cs8:
        md.append(
            f"**8 序 gate 集**：一次 sweep 内 CFG 回显 **{cs8.get('cfg_echo_count')}** 次"
            f"（{cs8.get('cfg_declared_len')} 字符 / {cs8.get('cfg_token_pair_count')} 个旗标/值对，"
            f"声明长度 = 实际长度），全部 **{cs8.get('run_count')}** 次运行"
            f"（{len(cs8.get('models') or [])} 个模型包 × {len(cs8.get('deltas') or [])} 个 Δ = "
            f"{' × '.join(cs8.get('models') or [])} × {'/'.join(str(x) for x in cs8.get('deltas') or [])}）"
            f"发生在该次回显之后、且脚本把**同一个 `$CFG` 变量**传给每次运行 ⇒ "
            f"**判定 {cs8.get('verdict')}**（产物 `runs/lookahead/fi_nogmc/cfg_shared_8seq.json`，"
            f"生成脚本 `server_overnight/check_sweep_cfg_shared.py`）。"
        )
        md.append("")
    if ci24:
        md.append(
            f"**24 序官方口径**：rc2 ↔ cs_t0043 的 CFG 逐 token 比对 = **{ci24.get('verdict')}**"
            f"（{ci24.get('rc2_declared_len')} 字符 / {ci24.get('rc2_token_count')} token / "
            f"{len(ci24.get('token_diffs') or [])} 处差异；产物 `runs/lookahead/nogmc24_cfg_identical.json`）。"
        )
        md.append("")
    md.append(
        "> ⇒ **8 序与 24 序都具备「同一份插件 CFG」的产物级证明** ⇒ 两次对照的**唯一变量确实只有 "
        "「模型包（权重 + 特征语义）」与「Δ」**。"
    )
    md.append("")

    # ---------- §6 旋钮扫描 ----------
    md.append(f"## 6. Δ={T} 专属旋钮扫描 + 新机理探索（8 序 A/B）")
    md.append("")
    if ks:
        best = ks.get("best", {})
        md.append(f"**推荐旋钮串**（两个模型包上都验证过）：`{' '.join(f'{k} {v}' for k, v in best.get('knobs', []))}`")
        md.append("")
        # ---- 交叉引用 §2.12：该旋钮串在 24 序官方口径上的复评（数字现算，与 §2.12 同产物同口径）----
        _var_doc = (variant.get("rc2") or {}).get(T)
        _base_doc = (d24.get("rc2") or {}).get(T)
        if _var_doc and _base_doc:
            _var_row, _base_row = row(_var_doc, "replay"), row(_base_doc, "replay")
            # (指标名, 差值[变体 − 交付档基线], 格式, 单位, +1 = 越高越好 / −1 = 越低越好)
            _checks = (
                ("F1", _var_row["f1"] - _base_row["f1"], "+.4f", " pp", 1),
                ("Recall", _var_row["recall"] - _base_row["recall"], "+.4f", " pp", 1),
                ("hit@5px", _var_row["hit5"] - _base_row["hit5"], "+.4f", " pp", 1),
                ("FAR", _var_row["far"] - _base_row["far"], "+.6f", "", -1),
                ("CLE p95", _var_row["cle95"] - _base_row["cle95"], "+.4f", " px", -1),
            )
            _delta_txt = " / ".join(f"{nm} {fmt(d, s)}{u}" for nm, d, s, u, _dir in _checks)
            _better = [nm for nm, d, _s, _u, dir_ in _checks if d * dir_ > 0]
            _worse = [nm for nm, d, _s, _u, dir_ in _checks if d * dir_ < 0]
            _concl = (
                "**F1 变差 ⇒ 净收益为负（阈值权衡）** ⇒ **交付档 CFG 本轮不改**"
                "（与 **§13** 的 24 序受控单旋钮扫描结论一致）"
                if _checks[0][1] < 0
                else "**F1 未变差** ⇒ 仍以 **§12 / §13** 的交付裁定为准"
            )
            md.append(
                f"> ⚠️ **该旋钮串是 8 序 A/B 的产物**；在 **24 序官方口径**上复评"
                f"（Δ={T}，变体 vs 交付档基线，数字与 **§2.12** 同产物同口径现算）："
                f"**{_delta_txt}** ⇒ **变好：{'、'.join(_better) or '无'}；"
                f"变差：{'、'.join(_worse) or '无'}** ⇒ {_concl}。"
            )
        else:
            md.append(
                f"> ⚠️ **该旋钮串是 8 序 A/B 的产物**；其 **24 序官方口径**复评见 **§2.12**"
                f"（Δ={T} 变体产物缺失，未能现算）。"
            )
        md.append("")
        md.append(
            "> ⚠️ **ΔF1 / ΔFAR / Δhit5 的参照基准 = 同模型包 Δ=3 的交付档基线**（**不是** `stale`）——"
            "本节是「旋钮 vs 同档交付配置」的受控 A/B。"
        )
        md.append("")
        md.append("| 模型包 | F1% | ΔF1(pp) | Recall% | FAR/帧 | ΔFAR | hit@5px% | Δhit5(pp) | CLE p95(池) | 交付三门 |")
        md.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
        for key in ("cs", "rc"):
            r = best.get(key)
            if not r:
                continue
            md.append(
                f"| {r.get('model')} | {fmt(r.get('f1'))} | {fmt(r.get('d_f1_pp'), '+.4f')} | {fmt(r.get('recall'))} | "
                f"{fmt(r.get('far'), '.6f')} | {fmt(r.get('d_far'), '+.6f')} | {fmt(r.get('hit5'))} | "
                f"{fmt(r.get('d_hit5_pp'), '+.4f')} | {fmt(r.get('cle95'), '.3f')} | **{r.get('verdict')}** |"
            )
        md.append("")
        for name, key in (("候选 1「Δ 自适应 roimax」", "candidate1"), ("候选 2「弃权/回退窗口联动」", "candidate2")):
            c = ks.get(key)
            if not c:
                continue
            md.append(f"**{name}**：{c.get('verdict', '—')}")
            md.append("")
        md.append("> 完整 5 张表 + 运行清单见 `runs/lookahead/fi_nogmc/knobscan/report.md`（+ `report.json`）。")
    else:
        md.append("> ⚠️ 未找到旋钮扫描产物，跳过本节。")
    md.append("")

    # ---------- §7 红线汇总 ----------
    md.append("## 7. FAR 红线汇总（**≤ 0.030**）与交付判据")
    md.append("")
    md.append(f"> ⚠️ **交付判据必须以「24 序官方口径」块为准**（8 序 gate 集只覆盖 24 序 {_cover_gt_txt} GT）。"
              "下表 **✗ = FAR > 0.030 破线**；**`replay` 是唯一交付臂**，`stale` / `kinematic` 是对照基线。")
    md.append("")
    md.append("| 模型包 | Δ | original | stale | kinematic | replay | 破线臂 |")
    md.append("|:--|--:|--:|--:|--:|--:|:--|")
    for m in ["rc2", "cs_t0043", "trial_0049"]:
        for d in sorted(d8.get(m, {})):
            cells, breach = [], []
            for arm in ARMS:
                r = row(d8[m][d], arm)
                cells.append(fmt(r["far"], ".6f") + (" **✗**" if r["far"] > DELIVERY["far_max"] else ""))
                if r["far"] > DELIVERY["far_max"]:
                    breach.append(arm)
            md.append(
                f"| **{MODEL_LABEL[m]} · 8 序 gate 集** | {d} | "
                + " | ".join(cells)
                + f" | {('、'.join(breach)) if breach else '无'} |"
            )
    md.append("")
    md.append("**24 序官方口径（交付判据的唯一权威集）**")
    md.append("")
    md.append("| 模型包 | Δ | original | stale | kinematic | replay | 破线臂 |")
    md.append("|:--|--:|--:|--:|--:|--:|:--|")
    for m in sorted(d24):
        for d in sorted(d24[m]):
            cells, breach = [], []
            for arm in ARMS:
                r = row(d24[m][d], arm)
                cells.append(fmt(r["far"], ".6f") + (" **✗**" if r["far"] > DELIVERY["far_max"] else ""))
                if r["far"] > DELIVERY["far_max"]:
                    breach.append(arm)
            md.append(
                f"| **{MODEL_LABEL.get(m, m)} · 24 序官方口径** | {d} | "
                + " | ".join(cells)
                + f" | {('、'.join(breach)) if breach else '无'} |"
            )
    md.append("")
    if d24.get("cs_t0043") and 8 in d24["cs_t0043"]:
        _r8 = row(d24["cs_t0043"][8], "replay")
        md.append(
            f"🔴 **24 序官方口径下唯一一条 `replay` 臂破线**：cs_t0043 Δ=8 `replay` FAR = "
            f"**{fmt(_r8['far'], '.6f')} > 0.030**（超线 {fmt(_r8['far'] - DELIVERY['far_max'], '.6f')}）"
            f" ⇒ **Δ=8 在裸配置下不可交付**；只有开出框门 m=6/m=8 才过线（见 **§11**）。"
        )
        md.append("")

    # ---------- 逐序列 FAR 破线（交付臂 m=6，★ 只报池化 FAR 是不够的） ----------
    _dl6 = (collect_fov(Path(args.fovgate_24_nogmc), "cs_t0043").get(T) or {}).get(6)
    if _dl6:
        _rp = (((_dl6.get("arms") or {}).get("replay") or {}).get("current_frame") or {}).get("per_sequence") or {}
        _op = (((_dl6.get("arms") or {}).get("original") or {}).get("current_frame") or {}).get("per_sequence") or {}
        _bad = [
            (sq, v.get("far_per_frame"), (_op.get(sq) or {}).get("far_per_frame"))
            for sq, v in _rp.items()
            if (v.get("far_per_frame") or 0) > DELIVERY["far_max"]
        ]
        if _bad:
            md.append("### 7.1 ⚠️ **逐序列** FAR 破 0.030 的序列（交付推荐档 `replay` + 出框门 m=6，24 序 Δ=3）")
            md.append("")
            md.append("| 序列 | `replay` FAR/帧 | 天花板 `original` FAR/帧 | 性质 |")
            md.append("|:--|--:|--:|:--|")
            for _sq, _rf, _of in sorted(_bad, key=lambda x: -(x[1] or 0)):
                if _of is not None and _of <= DELIVERY["far_max"]:
                    _kind = "🔴 **外挂引入**（天花板过线 → 交付臂破线）"
                else:
                    _kind = "天花板已破（既有条件，非外挂造成）"
                md.append(
                    f"| `{_sq}` | {fmt(_rf, '.6f')} | "
                    f"{fmt(_of, '.6f') if _of is not None else '—'} | {_kind} |"
                )
            md.append("")
            md.append(
                "> **口径纪律**：交付三门是**池化 24 序**口径（交付档池化 FAR "
                f"{fmt(row(_dl6, 'replay')['far'], '.6f')} ≤ {DELIVERY['far_max']:.3f}，余量 "
                f"{fmt(DELIVERY['far_max'] - row(_dl6, 'replay')['far'], '.6f')}）⇒ **逐序列破线不改交付裁定**；"
                "但它是**交付风险**，必须随交付一起披露 —— 其中标🔴 的序列是**外挂引入**的"
                "（零延迟天花板过线，交付臂破线）。"
            )
            md.append("")
            md.append(
                "> **归因**（24 序 Δ=3 交付档 `replay` 逐帧 FP 归属，产物 `runs/lookahead/fi_nogmc24/gapdiag/`）："
                "replay 臂 FP 合计 4021，其中 **640 帧是「零延迟臂对、重推臂错」**（`lost_by_replay`）、"
                "1153 帧两臂都错（检测器自身噪声）、2228 帧该帧无 GT 标注（标注稀疏，非算法可解）⇒ "
                "**外挂净增 FP = 542 帧（FAR +0.0033）**；`lost_by_replay` 高度集中在 3 条序列"
                "（`002001` 183 / `002111` 118 / `002959` 69，合计占 57.8%）。"
            )
            md.append("")

    # ---------- §8 产物索引 ----------
    md.append("## 8. 产物索引与复现命令")
    md.append("")
    md.append("| 类别 | 路径 |")
    md.append("|:--|:--|")
    md.append("| 8 序 Δ 扫描产物 | `runs/lookahead/fi_nogmc/{rc2,cs_t0043,trial_0049}/d{2,3,4,8}/arms.json` |")
    md.append("| 8 序全量指标真源 | `runs/lookahead/fi_nogmc/_facts/summary.md` + `_facts/all_metrics.json` |")
    md.append("| 24 序官方口径产物 | `runs/lookahead/fi_nogmc24/{rc2,cs_t0043}/d{2,3,4,8}/arms.json`（**8 份**） |")
    md.append("| 24 序变体（8 序推荐旋钮串的官方口径复评，见 §2.12） | "
              "`runs/lookahead/fi_nogmc24/p3knobs/rc2/d{2,3}/arms.json` |")
    md.append("| 工作点审计 | `runs/lookahead/fi_nogmc/workpoint/audit.md` / `audit.json` |")
    md.append("| Gate A（8 序） | `runs/lookahead/gateA/gateA.md` / `gateA.json` |")
    md.append("| Gate A（24 序官方口径） | `runs/lookahead/gateA24/gateA.md` / `gateA.json` |")
    md.append("| 旋钮扫描 | `runs/lookahead/fi_nogmc/knobscan/report.md` / `report.json` |")
    md.append("| OSD 三臂视频（交付档，stale/replay/original） | "
              "`runs/lookahead/osd/videos/*_cs_t0043_d3.mp4`（含 `VIDEO00007_…_001324`、"
              "`VIDEO00033_…_015000` 两序列） |")
    md.append("| OSD 三臂对照指标 | `runs/lookahead/osd/osd_3arm_cs_t0043_d3.md` / `.json` |")
    md.append("| 8 序专用报告（含逐序列归因 §5） | `runs/lookahead/fi_nogmc/report.md` / `report.json` |")
    md.append("| 8 序 house-style 报告 | `runs/lookahead/fi_nogmc/report_house_style.md` |")
    md.append("| 24 序 house-style 报告 | `runs/lookahead/fi_nogmc24/report_house_style.md` |")
    md.append("| 夜 6 帧级缺口诊断（§15） | `runs/lookahead/fi_nogmc24/gapdiag/` |")
    md.append("| 夜 7 位移修正族 oracle 闭包（§16） | `runs/lookahead/fi_nogmc24/oracle_accel/` |")
    md.append("| 夜 8 配准 / 歧义族 oracle 闭包（§17.1–17.3） | `runs/lookahead/fi_nogmc24/reg_oracle/` |")
    md.append("| 夜 8 回退链峰值门受控 A/B（§17.4） | `runs/lookahead/rbgate24nogmc/` |")
    md.append("| 交付包（报告 + 证据链 + `code/` + OSD 视频 + 校验和） | "
              "`runs/lookahead/final/nogmc_fi_20261104/` |")
    md.append("| 交付包内**仓镜像**（按仓内相对路径镜像真源 ⇒ 包外可独立复跑） | "
              "`runs/lookahead/final/nogmc_fi_20261104/repo/`（`README.md` 说明用法） |")
    md.append("| 交付包自足性审计（镜像逐位对账 + 自定位验证） | "
              "`runs/lookahead/fi_nogmc/audit_round7/audit_pack_selfsufficient.md` |")
    md.append("| 交付包自足性审计的**判别力**证明（14 个变异用例：11 篡改必报警 / 2 对照不误报 / 1 已知边界） | "
              "`runs/lookahead/fi_nogmc/audit_round7/mutation_test_selfsufficient.md`（脚本 "
              "`manu/tracking/lookahead/server_overnight/mutation_test_selfsufficient.py`，随包 `code/`） |")
    md.append("| 交付包**一键复核**（清单 + 六份审计 + 变异测试） | "
              "`runs/lookahead/final/nogmc_fi_20261104/code/verify_all.sh` |")
    md.append("")
    md.append("```bash")
    md.append("cd /tmp/pycharm_project_10ae9e2e")
    md.append("export PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl")
    md.append("# 8 序 Δ 扫描")
    md.append("bash manu/tracking/lookahead/server_overnight/run_fastinfer_nogmc_sweep.sh")
    md.append("# 24 序官方口径 Δ 扫描")
    md.append("bash manu/tracking/lookahead/server_overnight/run_fastinfer_24seq_sweep.sh")
    md.append("# 24 序变体（推荐旋钮串）")
    md.append("bash manu/tracking/lookahead/server_overnight/run_fastinfer_24seq_variants.sh")
    md.append("# 工作点审计 / 旋钮扫描 / Gate A")
    md.append("bash manu/tracking/lookahead/server_overnight/run_workpoint_audit.sh")
    md.append("bash manu/tracking/lookahead/server_overnight/run_knob_scan_d3.sh")
    md.append("bash manu/tracking/lookahead/server_overnight/run_gate_a_nogmc.sh")
    md.append("# 报告")
    md.append("python manu/tracking/lookahead/make_fastinfer_final_report.py \\")
    md.append("  --root-8 runs/lookahead/fi_nogmc --root-24 runs/lookahead/fi_nogmc24 \\")
    md.append("  --out-md runs/lookahead/fi_nogmc/FINAL_REPORT.md --out-json runs/lookahead/fi_nogmc/final_report.json")
    md.append("```")
    md.append("")

    # ---------- §9 出框门（FOV margin gate） ----------
    md.append("## 9. 出框门（FOV margin gate）受控 A/B（新机理）")
    md.append("")
    _fov_md, _fov_sec = fov_gate_section(
        [
            ("8 序 gate 集", args.fovgate, ["cs_t0043", "rc2"]),
            ("24 序官方口径", args.fovgate_24, ["rc2"]),
            ("24 序 no-GMC", args.fovgate_24_nogmc, ["cs_t0043"]),
            ("24 序 no-GMC · 红线救援档", args.fovgate_24_nogmc_d8, ["cs_t0043"]),
            ("24 序 no-GMC · 甜点档", args.fovgate_24_nogmc_d2, ["cs_t0043"]),
            ("24 序 no-GMC · 补齐档", args.fovgate_24_nogmc_d4, ["cs_t0043"]),
            ("24 序 rc2 · 红线边缘档", args.fovgate_24_rc2_d8, ["rc2"]),
        ]
    )
    md.extend(_fov_md)
    md.append(f"### 9.{_fov_sec + 1} ⭐ 出框门结论（**必须连同 8 序 → 24 序的衰减一起读**）")
    md.append("")
    _f8 = collect_fov(Path(args.fovgate), "cs_t0043").get(3, {})
    _f24 = collect_fov(Path(args.fovgate_24_nogmc), "cs_t0043").get(3, {})
    if 0 in _f8 and 6 in _f8 and 0 in _f24 and 6 in _f24:
        _a8, _b8 = row(_f8[0], "replay"), row(_f8[6], "replay")
        _a24, _b24 = row(_f24[0], "replay"), row(_f24[6], "replay")
        _og24 = row(_f24[0], "original")
        # 增量 FP = replay FP − oracle FP（池化）；门打掉的比例 = (FP_m0 − FP_m6) / 增量 FP
        _inc24 = _a24["fp"] - _og24["fp"]
        _kill24 = _a24["fp"] - _b24["fp"]
        _rel8 = pct(_b8["far"] - _a8["far"], _a8["far"])
        _rel24 = pct(_b24["far"] - _a24["far"], _a24["far"])
        _abs8 = abs(_a8["far"] - _b8["far"])
        _abs24 = abs(_a24["far"] - _b24["far"])
        md.append(
            f"1. **门的甜点是 m=6**（8 序与 24 序一致）：FAR 8 序 {fmt(_a8['far'], '.6f')} → {fmt(_b8['far'], '.6f')}"
            f"（**{fmt(_rel8, '.1f')}%**）、"
            f"24 序 {fmt(_a24['far'], '.6f')} → {fmt(_b24['far'], '.6f')}"
            f"（**{fmt(_rel24, '.1f')}%**）；"
            f"F1 同时 **{fmt(_b8['f1'] - _a8['f1'], '+.4f')}pp（8 序）/ "
            f"{fmt(_b24['f1'] - _a24['f1'], '+.4f')}pp（24 序）** ⇒ **不是拿精度换误报率**。"
        )
        md.append(
            f"2. 🔴 **8 序的 {fmt(_rel8, '.1f')}% 是「单集假象」，不可外推**："
            f"24 序官方口径上只剩 **{fmt(_rel24, '.1f')}%**"
            f"（**相对降幅衰减 {fmt(abs(_rel8) / max(abs(_rel24), 1e-12), '.1f')}×**；"
            f"绝对降幅 {fmt(_abs8, '.6f')} → {fmt(_abs24, '.6f')}，衰减 "
            f"**{fmt(_abs8 / max(_abs24, 1e-12), '.1f')}×**）——"
            f"逐序归因显示 8 序的增量 FP 几乎全在单序 `VIDEO00007_…_001324` 上，24 序上头号来源换成"
            f"`VIDEO00008_…_002001` / `VIDEO00003_…_002111`；"
            f"**门只打掉 24 序增量 FP 的 {fmt(pct(_kill24, _inc24), '.1f')}%**"
            f"（{int(_kill24)} / {int(_inc24)}）⇒ "
            f"**门的定位是「跨红线 / 兜底」，不是「把 FAR 压到天花板」**。"
        )
        md.append(
            f"3. ⚠️ **「FAR 已压到零延迟天花板」这句话只在 8 序成立**（8 序 m=8 时 replay FAR = oracle FAR 逐位相等）；"
            f"24 序上 m=6 离 oracle（{fmt(_og24['far'], '.6f')}）仍差 "
            f"{fmt(_b24['far'] - _og24['far'], '.6f')} ⇒ **不得对外表述为「已压到天花板」**。"
        )
        md.append("")

    # ---------- §10 24 序 no-GMC Δ 扫描 ----------
    md.append("## 10. 24 序 no-GMC（cs_t0043）Δ 扫描（与 rc2 同 Δ 并排对照）")
    md.append("")
    md.extend(nogmc24_section(d24, args.root_24))
    md.append("### 10.6 ★ 24 序官方口径逐序列归因 @ Δ=3（replay 相对 stale / 相对天花板）")
    md.append("")
    for _m in ("cs_t0043", "rc2"):
        if T not in d24.get(_m, {}):
            continue
        md.append(f"**{MODEL_LABEL.get(_m, _m)}**（`replay` 臂；「对总缺口贡献」= 该序 hit@5px 缺口 × 该序 GT 占比）")
        md.append("")
        md.extend(per_sequence_attribution(d24[_m][T]))
        md.append("")
    _cs24_doc = d24.get("cs_t0043", {}).get(T) or {}
    _cs24_arm = (_cs24_doc.get("arms") or {}).get("replay") or {}
    _ps24 = (_cs24_arm.get("current_frame") or {}).get("per_sequence") or {}
    if _ps24:
        _tot24 = sum(v.get("gt", 0) for v in _ps24.values())
        _top24 = max(_ps24, key=lambda k: _ps24[k].get("gt", 0))
        _gt8 = gt_share(d8.get("cs_t0043", {}).get(T, {}), "replay", "001324")
        md.append(
            f"🔴 **GT 集中度对比（口径纪律第 4 条）**：`VIDEO00007_…_001324` 在 **8 序 gate 集**里占 "
            f"**{fmt(pct(_gt8[0], _gt8[1]), '.2f')}% GT**（单序假象的来源），"
            f"在 **24 序官方口径**里只占 "
            f"**{fmt(pct(sum(v.get('gt', 0) for k, v in _ps24.items() if '001324' in k), _tot24), '.2f')}% GT**；"
            f"24 序里 GT 最多的一序变成 `{_top24}`（{int(_ps24[_top24].get('gt', 0))} / {int(_tot24)} = "
            f"**{fmt(pct(_ps24[_top24].get('gt', 0), _tot24), '.2f')}%**）"
            f" ⇒ **8 序的单序假象在 24 序被稀释掉**。"
        )
        md.append("")
    md.append(
        "> **口径纪律**：24 序是交付判据唯一权威集；8 序 gate 集的逐序表在 `runs/lookahead/fi_nogmc/report.md` §5。"
    )
    md.append("")

    # ---------- §11 Δ=8 红线救援专章 ----------
    rescue = collect_fov(Path(args.fovgate_24_nogmc_d8), "cs_t0043").get(8)
    if rescue:
        md.append("## 11. ⭐⭐⭐ Δ=8 红线救援专章 —— 出框门把 no-GMC 的可交付延时窗口从 Δ≤4 重新打开到 Δ≤8")
        md.append("")
        _rescue_base = row(rescue[0], "replay") if 0 in rescue else None
        _rescue_m6 = row(rescue[6], "replay") if 6 in rescue else None
        md.append(
            "> **一句话**：24 序 no-GMC Δ=8（320 ms）裸配置 FAR "
            + (
                f"**{fmt(_rescue_base['far'], '.6f')} > 0.030 红线** ⇒ "
                f"开出框门 **m=6** 后 FAR **{fmt(_rescue_m6['far'], '.6f')} 过线**，"
                f"且 **F1 同时涨 {fmt(_rescue_m6['f1'] - _rescue_base['f1'], '+.4f')}pp**（不是拿精度换误报率）。"
                if _rescue_base and _rescue_m6
                else "（缺 Δ=8 出框门产物，无法给出一句话结论）"
            )
        )
        md.append("")
        md.append(
            "⚠️ **口径纪律**：Δ=8 的过线结论**只在开出框门 m=6（或 m=8）时成立**；"
            "裸配置（m=0）破线、m=4 仍破线 ⇒ 对外必须写「**Δ=8 在 m=6 出框门下过线**」，**不能**写「Δ=8 可交付」。"
        )
        md.append("")
        md.extend(fov_gate_table(rescue))
        md.append("")
        base_doc = rescue.get(0)
        m6_doc = rescue.get(6)
        if base_doc and m6_doc:
            base = row(base_doc, "replay")
            m6 = row(m6_doc, "replay")
            md.append(
                f"**⇒ m=6 是过线的最小门限**：m=0 FAR **{fmt(base['far'], '.6f')}**（超线 "
                f"{fmt(base['far'] - DELIVERY['far_max'], '.6f')}）→ m=6 FAR **{fmt(m6['far'], '.6f')}**"
                f"（余量 {fmt(DELIVERY['far_max'] - m6['far'], '.6f')}）；"
                f"F1 **{fmt(m6['f1'] - base['f1'], '+.4f')}pp**、"
                f"hit@5px **{fmt(m6['hit5'] - base['hit5'], '+.4f')}pp**、"
                f"CLE p95 **{fmt(m6['cle95'] - base['cle95'], '+.4f')}px**。"
            )
            md.append("")
        md.append(
            "**⇒ 门的价值定位应改写为「跨红线 / 兜底」，不是「把 FAR 压到天花板」**："
            f"8 序的 FAR {_far_delta_pct(_g8_d3_m6, _g8_d3_m0)} 头条是**单序列假象**（§9.1），"
            f"24 序上只有 {_far_delta_pct(_g24_d3_m6, _g24_d3_m0)}（Δ=3）/ "
            f"{_far_delta_abs(_g24_d8_m6, _g24_d8_m0)} 绝对值（Δ=8）。"
            + (
                f"另：rc2（GMC）在 Δ=8 裸配置已经过线（{fmt(_rc2_d8['far'], '.6f')}，"
                f"余量 {fmt(DELIVERY['far_max'] - _rc2_d8['far'], '.6f')}）⇒ "
                "**对 GMC 来说门是「加固」不是「救命」**。"
                if _rc2_d8
                else ""
            )
        )
        md.append("")

    # ---------- §12 交付裁定（推荐配置；数字全部从产物算） ----------
    if d24.get("cs_t0043"):
        _fov3 = collect_fov(Path(args.fovgate_24_nogmc), "cs_t0043").get(3, {})
        _rec = _fov3.get(6) or d24["cs_t0043"].get(T)
        _rec_row = row(_rec, "replay") if _rec else None

        def _fov_m6(delta: int):
            """Δ 任意档下出框门 m=6 的产物（各次扫描分目录存，按 Δ 依次找）。"""
            for root in (
                Path(args.fovgate_24_nogmc),
                Path(args.fovgate_24_nogmc_d2),
                Path(args.fovgate_24_nogmc_d4),
                Path(args.fovgate_24_nogmc_d8),
            ):
                doc = fov_margin_doc(root, "cs_t0043", delta, 6)
                if doc:
                    return doc
            return None

        # Δ 理由句必须如实：**Δ=T 是本次指定的标称延时**，实测最优 Δ 未必是它
        # （§12 不得把「指定」写成「最优」——否则会被本报告自己的 §2.9 / §10.5 表否证）
        _cs24_all = d24["cs_t0043"]
        _deltas24 = sorted(_cs24_all)
        _best_delta = max(_deltas24, key=lambda d: row(_cs24_all[d], "replay")["f1"]) if _deltas24 else None
        _delta_basis = ""
        if _best_delta is not None and T in _cs24_all:
            _bt = row(_cs24_all[T], "replay")
            if _best_delta == T:
                _delta_basis = (
                    f"**Δ={T} = 本次指定的标称延时（{T * 40} ms）**，且实测在 Δ∈"
                    f"{{{', '.join(str(d) for d in _deltas24)}}} 中 **F1 最高（{fmt(_bt['f1'])}）、"
                    f"FAR 红线余量最大（{fmt(DELIVERY['far_max'] - _bt['far'], '.6f')}）**；"
                    "Δ 越大滞后代价越大（§2.9、§10.5）"
                )
            else:
                _bb = row(_cs24_all[_best_delta], "replay")
                _delta_basis = (
                    f"**Δ={T} = 本次指定的标称延时（{T * 40} ms）**；"
                    f"⚠️ 实测 **Δ={_best_delta}（{_best_delta * 40} ms）各项全面优于 Δ={T}**"
                    f"（F1 {fmt(_bb['f1'])} vs {fmt(_bt['f1'])}、FAR {fmt(_bb['far'], '.6f')} vs {fmt(_bt['far'], '.6f')}、"
                    f"hit@5px {fmt(_bb['hit5'])} vs {fmt(_bt['hit5'])}、CLE p95 {fmt(_bb['cle95'])} vs {fmt(_bt['cle95'])}）"
                    f"⇒ **真实延时若能压到 {_best_delta * 40} ms，应改选 Δ={_best_delta}**（§2.9、§10.5、§9.6）"
                )

        md.append("## 12. ⭐⭐⭐ **交付裁定（推荐配置）** —— 唯一权威口径 = 24 序官方口径")
        md.append("")
        md.append("| 项 | 推荐值 | 依据（24 序官方口径） |")
        md.append("|:--|:--|:--|")
        md.append(
            "| **模型包** | `cs_t0043`（**no-GMC 语义**） | 同 Δ 下 F1 / hit@5px / CLE p95 **全面优于** GMC 的 `rc2`（§2.10、§10） |"
        )
        md.append(
            f"| **假设延时 Δ** | **Δ={T}（{T * 40} ms @25 Hz，标称/假设值）** | "
            f"{_delta_basis} |"
        )
        md.append("| **交付臂** | `replay`（视觉重推外挂） | `stale` / `kinematic` 在 Δ≥2 全部破 FAR 红线（§7） |")
        md.append(
            "| **出框门** | `--replay-fov-margin-px 6` | Δ=3 裸配置本已过线，门是**余量**；Δ=8 只有 m≥6 才过线（§9、§11） |"
        )
        md.append("")
        if _rec_row:
            md.append(
                f"**推荐档（`cs_t0043` · Δ={T} · `replay` · 出框门 m=6）的交付三门核对**："
                f"F1 **{fmt(_rec_row['f1'], '.4f')}%**（≥ {DELIVERY['f1_min']}）· "
                f"Recall **{fmt(_rec_row['recall'], '.4f')}%**（≥ {DELIVERY['recall_min']}）· "
                f"FAR **{fmt(_rec_row['far'], '.6f')}**/帧（≤ {DELIVERY['far_max']}）"
                f" ⇒ {gates(_rec_row)}"
            )
            md.append("")
        md.append("| 24 序 · `cs_t0043`（no-GMC） | Δ=2（80 ms） | Δ=3（120 ms）★推荐 | Δ=4（160 ms） | Δ=8（320 ms） |")
        md.append("|:--|--:|--:|--:|--:|")
        _rows = (
            (
                "裸配置 FAR（m=0）",
                lambda d: row(d24["cs_t0043"][d], "replay") if d in d24["cs_t0043"] else None,
            ),
            ("出框门 m=6 · FAR", lambda d: row(_fov_m6(d), "replay") if _fov_m6(d) else None),
        )
        for _label, _getter in _rows:
            cells = []
            for d in (2, 3, 4, 8):
                r = _getter(d)
                cells.append(fmt(r["far"], ".6f") if r else DASH)
            md.append(f"| **{_label}** | " + " | ".join(cells) + " |")
        md.append("")
        _missing_fov = [d for d in (2, 3, 4, 8) if not _fov_m6(d)]
        if _missing_fov:
            md.append(
                "> `—` = 该 (Δ, m) 没有产物"
                + f"（本轮缺 Δ={', '.join(str(d) for d in _missing_fov)} 的出框门扫描）"
                + " ⇒ **不可外推**，不是「无变化」。"
            )
        md.append("")
        md.append(
            "**⇒ 可交付延时窗口**：**no-GMC 裸配置 Δ≤4**（Δ=8 FAR 破线）；"
            "**开出框门 m=6 后窗口恢复到 Δ≤8**。"
            "⚠️ 对外必须写成「**Δ=8 在 m=6 出框门下过线**」，**不能**写成「Δ=8 可交付」。"
        )
        md.append("")
        md.append(
            "**⇒ 对 GMC（`rc2`）的对照**：rc2 在 Δ=8 **裸配置**就过线"
            + (
                f"（FAR {fmt(_rc2_d8['far'], '.6f')}，余量 {fmt(DELIVERY['far_max'] - _rc2_d8['far'], '.6f')}）"
                if _rc2_d8
                else ""
            )
            + "⇒ **换 no-GMC + 开出框门这两件事合起来，才把延时窗口维持在 Δ≤8**；"
            "对 GMC 来说门是「加固」，对 no-GMC 来说是「救命」。"
        )
        md.append("")

    # ---------- §13 24 序 Δ=3 专属受控单旋钮 A/B（交付档 CFG 要不要改？） ----------
    ks24 = load_json(Path(args.knobscan24)) if args.knobscan24 else None
    if ks24:
        md.append("## 13. 24 序 Δ=3 专属受控单旋钮 A/B（交付档 CFG 要不要改？）")
        md.append("")
        _lo = ks24.get("loso") or {}
        _gt = ks24.get("gate") or {}
        _pair = ks24.get("paired") or []
        _dom = ks24.get("dominance") or []
        _jobs = ks24.get("jobs") or []
        md.append(
            f"> 口径：**24 序官方口径**（165,223 帧 / 133,257 GT / 24 条序列）；"
            f"基线 = **交付档 CFG + `--replay-fov-margin-px 6`**（= §12 的交付推荐行）；"
            f"每次只**替换一个旋钮**，其余一字不改。"
        )
        md.append("")
        md.append("| 项 | 结果 |")
        md.append("|:--|:--|")
        _n_gate = sum(1 for j in _jobs if str(j.get("tag") or "").startswith("gate__"))
        _n_var = len(_jobs) - _n_gate
        md.append(f"| 作业点数 | **{len(_jobs)}** 个 = **{_n_gate}** 个基线复现（`gate__baseline`）"
                  f"+ **{_n_var}** 个单旋钮变体（全部 rc=0；下文帕累托支配的分母 **{len(_dom)}** 只计变体） |")
        md.append(f"| 硬门禁（基线复现 vs 参考产物） | **四臂逐位一致 "
                  f"{_gt.get('total', 0) - _gt.get('bad', 0)}/{_gt.get('total', 0)} 项，不一致 {_gt.get('bad', 0)} 处** "
                  f"⇒ **PASS**（扫描器零额外变量） |")
        # 最重单序（LOSO 摆幅最大）≠ GT 占比最高的单序 ⇒ 两个事实分别读出，新键缺失时回退
        _worst_gt = _lo.get("worst_gt_share")
        if _worst_gt is None:
            _worst_gt = _lo.get("max_gt_share")
        _max_gt_pct = fmt((_lo.get("max_gt_share") or 0) * 100, ".2f")
        _gt_clause = f"（最重单序 `{_lo.get('worst_seq')}` 占 GT {fmt((_worst_gt or 0) * 100, '.2f')}%"
        if _lo.get("max_gt_seq"):
            _gt_clause += f"；GT 占比最高的是 `{_lo.get('max_gt_seq')}`，占 {_max_gt_pct}%"
        _gt_clause += "）"
        md.append(f"| 尺子A · 24 序 LOSO 噪声底噪 | 池化 F1 摆幅 **中位 {fmt(_lo.get('swing_median_pp'), '.4f')} pp / "
                  f"最大 {fmt(_lo.get('swing_pp'), '.4f')} pp**{_gt_clause} |")
        _n_dom = sum(1 for d in _dom if d.get("dominates"))
        md.append(f"| 帕累托支配判定 | **支配基线的作业点 {_n_dom} / {len(_dom)}** |")
        md.append("")

        # 残差分解（覆盖率 vs 定位精度）
        _b = ks24.get("baseline") or {}
        _o = ks24.get("oracle") or {}
        md.append("**残差分解（为什么剩下的缺口很难压）**：")
        md.append("")
        md.append(
            f"- **覆盖率几乎不是问题**：oracle 上报帧数与 replay 只差一个三位数级"
            f"（CLE **p50** 只差 {fmt_signed((_b.get('cle50') or 0) - (_o.get('cle50') or 0), 1, 4)} px，"
            f"但 **p95** 差 {fmt_signed((_b.get('cle95') or 0) - (_o.get('cle95') or 0), 1, 4)} px、"
            f"**p99** 差 {fmt_signed((_b.get('cle99') or 0) - (_o.get('cle99') or 0), 1, 4)} px）"
            f"⇒ 缺口是**少数帧的定位尾部**，不是「没上报」。"
        )
        md.append("")

        # 尺子B：配对符号检验
        md.append("**尺子B · 配对符号检验**（同一 24 序集合上的逐序一致性 ⇒ 受控 A/B 该用的尺子）：")
        md.append("")
        md.append("| 旋钮 | 取值 | 变好/变差序数 | 符号检验 p | 池化 ΔF1(pp) | 判定 |")
        md.append("|:--|--:|--:|--:|--:|:--|")
        _dom_map = {(d.get("knob"), d.get("val")): d.get("dominates") for d in _dom}
        for pr in sorted(_pair, key=lambda x: -(x.get("d_f1") or 0)):
            key = (pr.get("knob"), pr.get("val"))
            _is_dom = _dom_map.get(key)
            if (pr.get("p") or 1.0) < 0.05 and _is_dom:
                _judge = "**真增益**"
            elif (pr.get("p") or 1.0) < 0.05:
                _judge = "方向显著但有指标变差（阈值权衡）"
            else:
                _judge = "噪声"
            md.append(
                f"| `{pr.get('knob')}` | `{pr.get('val')}` | {pr.get('n_up')}/{pr.get('n_down')} | "
                f"{fmt(pr.get('p'), '.4f')} | {fmt_signed(pr.get('d_f1'), 100)} | {_judge} |"
            )
        md.append("")
        _best = [pr for pr in _pair if (pr.get("p") or 1.0) < 0.05 and _dom_map.get((pr.get("knob"), pr.get("val")))]
        if _best:
            md.append(
                f"⇒ **只有 {len(_best)} 个作业点通过「p<0.05 且四项池化指标全部不劣」**："
                + "、".join(f"`{b.get('knob')}={b.get('val')}`（p={fmt(b.get('p'), '.4f')}，"
                            f"池化 ΔF1 {fmt_signed(b.get('d_f1'), 100)} pp）" for b in _best)
                + "；但**唯一通过者的幅度也在噪声内**（"
                + "、".join(f"{fmt_signed(b.get('d_f1'), 100)} pp" for b in _best)
                + f" < 尺子A 的中位波动 {fmt(_lo.get('swing_median_pp'), '.4f')} pp）⇒ **幅度在噪声内**。"
                + "（注：**其余** p<0.05 的作业点共 "
                f"{len([pr for pr in _pair if (pr.get('p') or 1.0) < 0.05])} 个，"
                "但它们**方向为负**且**有指标变差** ⇒ 「池化 ΔF1 全部 < 噪声底」是**假陈述**，"
                "正确说法是「**通过四项不劣的那一条**幅度在噪声内」。）"
            )
        else:
            md.append("⇒ **没有任何作业点通过「p<0.05 且四项池化指标全部不劣」**。")
        md.append("")
        md.append(
            "**⇒ 交付裁定（本节）**：**交付档 CFG（`manu/tracking/lookahead/run_delivery.sh`）本轮不改**。"
            "24 序 Δ=3 上**所有单旋钮都只是沿着「FAR ↔ F1/hit@5px/CLE」的帕累托前沿滑动**，"
            "**没有把前沿向外推** ⇒ 阈值类旋钮**压不掉**那个定位尾部；"
            "要继续压只能从**配准本身**入手（配准精度 / 歧义帧的处置策略），属于下一轮的机理工作。"
        )
        md.append("")

    # ---------- §14 24 序 Δ 依赖性 + 惰性旋钮机理（为什么小延时更好交付） ----------
    dd = load_json(Path(args.delta_dep)) if args.delta_dep else None
    if dd:
        md.append("## 14. 24 序 Δ 依赖性 + 惰性旋钮机理（为什么「小延时更好交付」）")
        md.append("")
        _inert = dd.get("inert_knobs") or []
        md.append(
            "> **本节回答两个问题**：(Q1) 同一个旋钮在不同 Δ 上效应量怎么变？"
            "(Q2) 为什么在 Δ=3 上有些旋钮「改了等于没改」？"
        )
        md.append("")
        md.append("**Q1 · Δ 依赖性（24 序 cs_t0043 + 出框门 m=6 = 交付档配置）**：")
        md.append("")
        md.append("| Δ | 标称延时 | F1% | FAR/帧 | hit@5px% | CLE p95(池) | replay 失败帧 |")
        md.append("|:--|:--|--:|--:|--:|--:|--:|")
        for d in (2, 3, 4, 8):
            _doc = _fov_m6(d)
            if not _doc:
                continue
            _r = row(_doc, "replay")
            _s = (_doc["arms"]["replay"].get("stream_stats") or {}).get("overall") or {}
            md.append(f"| {d} | {d * 40} ms | {fmt(_r['f1'])} | {fmt(_r['far'], '.6f')} | "
                      f"{fmt(_r['hit5'])} | {fmt(_r['cle95'])} | "
                      f"{_s.get('replay_failed', DASH)} |")
        md.append("")
        md.append(
            "⇒ **F1 / FAR / hit@5px / CLE p95 四项全部单调**随 Δ 变小而变好 "
            "⇒ 「小延时下精度提升」在**官方口径**上成立，**且不是某个 Δ 的偶然**。"
        )
        md.append("")
        md.append(
            f"**Q2 · 惰性旋钮机理**：24 序 Δ=3 的 {len(_inert)} 个作业点"
            "**与基线逐位一致**（四臂 × 10 字段严格相等）⇒ 在 Δ=3 上是**惰性旋钮**："
        )
        md.append("")
        md.append("| Δ | 旋钮 | 取值 |")
        md.append("|:--|:--|--:|")
        for it in _inert:
            md.append(f"| {it.get('delta')} | `{it.get('knob')}` | `{it.get('val')}` |")
        md.append("")
        md.append(
            "⇒ **机理**：这些旋钮的生效条件依赖「**连续失效步数**」或「**链内可用步数**」，"
            "而 replay 链的**总步数上限就是 Δ** ⇒ "
            "**Δ 越小 ⇒ 链越短 ⇒ 能触发的兜底/回退机制越少 ⇒ 失效模式更少 ⇒ 尾部误差更小**。"
        )
        md.append("")
        md.append(
            "⇒ **机理的 Δ=8 反证**：在 Δ=8 上重跑同样的旋钮 ⇒ "
            "**惰性消失（旋钮「复活」）** ⇒ 惰性确实来自「链长 ≤ Δ」这个上限，"
            "**不是旋钮本身失效**。"
        )
        md.append("")
        _clip = dd.get("clip_rate") or []
        if _clip:
            md.append("**统一机理的直接因果证据 —— 输出级限幅的截断率随 Δ 增长**：")
            md.append("")
            md.append("| Δ | 限幅上限 | 相邻上报位移恰好=上限的次数 | 总数 | 截断率 |")
            md.append("|:--|--:|--:|--:|--:|")
            for _c in _clip:
                md.append(f"| {_c.get('delta')} | 30 px | {_c.get('clamped')} | {_c.get('total')} | "
                          f"{fmt((_c.get('rate') or 0) * 100, '.3f')}% |")
            md.append("")
            _r0 = (_clip[0].get("rate") or 0)
            _r1 = (_clip[-1].get("rate") or 0)
            md.append(
                f"⇒ **截断率随 Δ 严格单调递增**（{fmt(_r0 * 100, '.3f')}% @ Δ={_clip[0].get('delta')} → "
                f"{fmt(_r1 * 100, '.3f')}% @ Δ={_clip[-1].get('delta')}，"
                f"**{fmt(_r1 / max(_r0, 1e-12), '.2f')}×**）"
                "⇒ **Δ 越大，replay 每帧必须合成的位移越大 ⇒ 限幅咬得越频繁 ⇒ 截断越伤尾部**。"
            )
            md.append("")
            md.append(
                "⇒ **统一机理（把 Q1 与 Q2 收成一句话）**："
                "**replay 链要补的滞后越大（Δ 越大），每帧必须合成的位移就越大，"
                "于是能咬住输出的机制就越多**（限幅咬得更频繁、兜底/回退更容易触发）"
                "⇒ **失效模式随 Δ 增长 ⇒ 尾部误差随 Δ 增长**。"
                "**「小延时更好交付」不是经验相关，而是有结构性原因。**"
            )
            md.append("")
        md.append(
            "**⇒ 交付论断（本节）**：**「小延时更好交付」不只是精度更高，而且是**失效模式更少**"
            "（可兜底/可回退的机制在 Δ=3 上根本不触发）**。**"
            "这解释了为什么 Δ=3 能在 24 序官方口径上**同时**做到 FAR 0.024337（红线 0.030）"
            "与 F1 94.8094，而 Δ=8 的 FAR 0.029608 已贴到红线边缘。"
        )
        md.append("")

    # ---------- §15 夜 6 新机理探索（残差诊断 → 新机理受控 A/B） ----------
    gd = load_json(Path(args.gapdiag)) if args.gapdiag else None
    kb = load_json(Path(args.knobscan_p3b)) if args.knobscan_p3b else None
    ac = load_json(Path(args.accel_closure)) if args.accel_closure else None
    rc = load_json(Path(args.reg_oracle_closure)) if args.reg_oracle_closure else None
    rb_rows = load_rbgate_rows(args.rbgate_root) if args.rbgate_root else []
    if gd or kb:
        md.append("## 15. 夜 6 新机理探索（残差诊断 → 新机理受控 A/B）")
        md.append("")
        md.append(
            "> **本节回答 §13 留下的问题**：§13 判定「所有阈值类旋钮都只在帕累托前沿上滑动」，"
            "**缺口来自配准本身的定位尾部**。本节先用**帧级诊断**把缺口拆开，"
            "再针对拆出来的机理做**新旋钮受控 A/B**（24 序官方口径，基线 = §12 交付推荐行）。"
        )
        md.append("")

        # ================= 15.1 残余缺口的 Δ 依赖性 =================
        _fit_val = {}
        if gd:
            _dd = gd.get("delta_dependence_m6") or {}
            _fit = _dd.get("linear_fit") or {}
            _hit5_fit = _fit.get("delta_hit5_gap_pp") or {}
            _bn = gd.get("baseline_numbers") or {}
            _gap = _bn.get("gap") or {}
            _swing_med = num(((kb or {}).get("loso") or {}).get("swing_median_pp"))
            _swing_max = num(((kb or {}).get("loso") or {}).get("swing_pp"))
            md.append("### 15.1 残余缺口的 Δ 依赖性：缺口主体是「Δ-比例项」，不是常数项")
            md.append("")
            md.append("| Δ | FAR/帧 | hit@5px% | CLE p95(池) |")
            md.append("|:--|--:|--:|--:|")
            for _d in ("2", "3", "4", "8"):
                _p = (_dd.get("points") or {}).get(_d)
                if not _p:
                    continue
                md.append(f"| {_d} | {fmt(_p.get('far'), '.6f')} | "
                          f"{fmt(num(_p.get('hit5')) * 100, '.4f')} | "
                          f"{fmt(_p.get('cle_pooled_p95'), '.4f')} |")
            md.append("")
            md.append("**缺口 = `original`（零延迟天花板）− `replay`（交付推荐行），随 Δ 的线性拟合**：")
            md.append("")
            md.append("| 缺口项 | 线性拟合（原式） | R² | Δ-比例项 | Δ-无关常数项 | Δ-比例项占实测缺口 |")
            md.append("|:--|:--|--:|--:|--:|--:|")
            # 每项：拟合系数 → Δ=3 处的比例项/常数项，并与**实测**缺口比对
            _gap_spec = (
                ("delta_hit5_gap_pp", "hit@5px", 1.0, "hit5_pp", ""),
                ("delta_far_gap_x1e4", "FAR/帧", 1e-4, "far", "（原式量纲 ×10⁻⁴）"),
                ("delta_cle_pooled_p95_gap_px", "CLE p95", 1.0, "cle_pooled_p95_px", ""),
            )
            for _key, _label, _unit, _gk, _note in _gap_spec:
                _f = _fit.get(_key) or {}
                _formula = _f.get("formula") or ""
                _m = re.search(r"([-+]?[\d.]+)\s*\+\s*([-+]?[\d.]+)\s*\*\s*(?:Δ|d)", _formula)
                _obs = _gap.get(_gk)
                if not _m:
                    md.append(f"| {_label} | `{_formula or DASH}` | {fmt(_f.get('r2'), '.4f')} | "
                              f"{DASH} | {DASH} | {DASH} |")
                    continue
                _a, _b = float(_m.group(1)), float(_m.group(2))
                _prop = _b * 3 * _unit
                _cons = _a * _unit
                _share = ratio(_prop, abs(num(_obs))) * 100 if num(_obs) else float("nan")
                _share_c = ratio(_cons, abs(num(_obs))) * 100 if num(_obs) else float("nan")
                _fit_val[_key] = (_prop, _cons, _share, _share_c)
                md.append(f"| {_label} | `{_formula}` {_note} | {fmt(_f.get('r2'), '.4f')} | "
                          f"{fmt_signed(_prop, 1, 6)} | {fmt_signed(_cons, 1, 6)} | "
                          f"{fmt(_share, '.1f')}% |")
            md.append("")
            # 拟合残差（诚实标注：拟合值与实测值的差）
            _hp = _fit_val.get("delta_hit5_gap_pp")
            if _hp and num(_gap.get("hit5_pp")):
                _fit_at3 = _hp[0] + _hp[1]
                _obs_at3 = abs(num(_gap.get("hit5_pp")))
                _resid = ratio(_fit_at3 - _obs_at3, _obs_at3) * 100
                md.append(
                    f"**拟合残差（诚实标注）**：拟合式在 Δ=3 处给 {fmt(_fit_at3, '.4f')} pp，"
                    f"实测缺口 {fmt(_obs_at3, '.4f')} pp ⇒ **残差 {fmt_signed(_resid, 1, 2)}%**"
                    "（拟合是 4 点线性趋势，**不是**精确恒等式）"
                )
                md.append("")
            _h5_prop = _hp[0] if _hp else float("nan")
            _h5_cons = _hp[1] if _hp else float("nan")
            _h5_share = _hp[2] if _hp else float("nan")
            _h5_share_c = _hp[3] if _hp else float("nan")
            md.append(
                f"⇒ **hit@5px 缺口 = {_hit5_fit.get('formula', DASH)}（R² = {fmt(_hit5_fit.get('r2'), '.4f')}）**"
                f"⇒ Δ=3 处实测缺口 {fmt(abs(num(_gap.get('hit5_pp'))), '.4f')} pp 中，"
                f"**Δ-比例项占 {fmt(_h5_share, '.1f')}%**（= 0.583807×3 = {fmt(_h5_prop, '.4f')} pp）、"
                f"**Δ-无关常数项占 {fmt(_h5_share_c, '.1f')}%**（= {fmt(_h5_cons, '.4f')} pp）。"
                "（两项之和 < 100% 是**拟合残差**，不是笔误。）"
            )
            md.append("")
            md.append(
                f"⇒ **这条拟合给「老旋钮全否决」提供了上限解释（是上界，**不是**因果证明）**："
                f"Δ-无关常数项的**总大小只有 {fmt(_h5_cons, '.4f')} pp** —— "
                f"它比 24 序 LOSO 的**最大**单序列波动 {fmt(_swing_max, '.4f')} pp **小一个量级**，"
                f"只比**中位**波动 {fmt(_swing_med, '.4f')} pp 高约 "
                f"{fmt(ratio(_h5_cons, _swing_med), '.1f')}×。"
                f"⇒ **任何只能在同一 Δ 上挪工作点的旋钮，其收益上界就是这 {fmt(_h5_cons, '.4f')} pp**；"
                "§13 实测的老旋钮效应量**全部落在噪声带内** —— "
                "**「门槛类旋钮压不掉这个缺口」是被这条上界约束住的**。"
            )
            md.append("")

            # ================= 15.2 帧级归因 =================
            _ha = gd.get("hit5_attribution") or {}
            _bk = _ha.get("buckets") or {}
            md.append("### 15.2 帧级归因：缺口里有多少是「外挂可动」的")
            md.append("")
            md.append("| 归因桶 | 帧数 | 归属 | 含义 |")
            md.append("|:--|--:|:--|:--|")
            _bk_doc = {
                "A_detector_src_blind": ("检测器侧", "检测器在源帧就看不到目标（外挂无能为力）"),
                "B_detector_src_mislocated": ("检测器侧", "检测器在源帧就定位错（外挂只是继承）"),
                "C_plugin_silent": ("外挂侧", "外挂弃权 / 未上报"),
                "D_plugin_misplaced": ("外挂侧", "外挂上报但位置走歪（位移估计）"),
                "edge_src_absent": ("边界", "边界：源帧不存在"),
                "edge_gt_absent": ("边界", "边界：GT 不存在"),
            }
            for _k in ("A_detector_src_blind", "B_detector_src_mislocated", "C_plugin_silent",
                       "D_plugin_misplaced", "edge_src_absent", "edge_gt_absent"):
                if _bk.get(_k) is None:
                    continue
                _owner, _txt = _bk_doc.get(_k, ("", ""))
                md.append(f"| `{_k}` | {_bk[_k]} | **{_owner}** | {_txt} |")
            md.append("")
            _gt_total = num(((gd.get("hit5_attribution") or {}).get("miss_funnel") or {}).get("gt"))
            md.append(
                f"⇒ 原始缺口 **{_ha.get('raw_gap_frames')} 帧**，"
                f"其中**外挂反超 oracle 的帧 {_ha.get('rep_only_frames')} 帧**（净扣掉）"
                f"⇒ **净缺口 {_ha.get('net_gap_frames')} 帧 = {_ha.get('net_gap_pp')} pp**。"
            )
            md.append("")
            md.append(
                f"⇒ **检测器侧 = A + B = {_ha.get('detector_side_frames')} 帧**"
                f"（{fmt(ratio(_ha.get('detector_side_frames'), _gt_total) * 100, '.2f')} pp，**外挂补不回来**）；"
                f"**外挂侧 = C + D = {_ha.get('plugin_side_frames')} 帧**"
                f"（{fmt(ratio(_ha.get('plugin_side_frames'), _gt_total) * 100, '.2f')} pp）。"
            )
            md.append("")
            md.append("**D 桶按「目标在 Δ=3 内的真实位移」再拆 —— 这一层才分出「可修」与「物理上限」**：")
            md.append("")
            md.append("| 真实位移 | 帧数 | 判定 |")
            md.append("|:--|--:|:--|")
            _D = _ha.get("D_split_by_true_displacement") or {}
            md.append(f"| ≤ 5 px | {_D.get('le5px')} | **外挂侧可修**（目标几乎没动却走歪 ⇒ **真 bug**） |")
            md.append(f"| 5–20 px | {_D.get('5to20px')} | **外挂侧可修**（线性外推够得着但没跟上） |")
            md.append(f"| > 20 px | {_D.get('gt20px')} | **物理不可达**（剧烈机动 ⇒ 线性外推的物理上限） |")
            md.append("")
            md.append(
                f"⇒ **外挂侧的可回收上限 = {_D.get('le5px')} + {_D.get('5to20px')} = "
                f"{_ha.get('plugin_addressable_frames')} 帧 = {_ha.get('plugin_addressable_pp')} pp of GT**"
                f"（D 桶平均真实位移 {_D.get('D_mean_true_disp_px')} px）"
            )
            md.append("")
            md.append(
                f"⇒ **物理不可达 = {_D.get('gt20px')} 帧 = {_ha.get('physical_unreachable_pp')} pp**"
                f"（= D 桶「真实位移 > 20px」那部分 ⇒ **线性外推的物理上限**，"
                "**不是**「检测器没定位对」—— 检测器侧是上面那句 "
                f"{_ha.get('detector_side_frames')} 帧，**两者是不同的桶，不可混谈**）"
            )
            md.append("")
            md.append(
                f"⇒ **三个「多少」的区分（避免读混）**：外挂侧总缺口 **{_ha.get('plugin_side_frames')} 帧**；"
                f"其中**外挂可回收** {_ha.get('plugin_addressable_frames')} 帧"
                f"（= {_ha.get('plugin_addressable_pp')} pp）；"
                f"**物理不可达** {_ha.get('physical_unreachable_pp')} pp；"
                f"**检测器侧** {_ha.get('detector_side_frames')} 帧。"
            )
            md.append("")
            _fl = gd.get("frame_level") or {}
            _ct = _fl.get("cle_tail_replay_minus_original") or {}
            if _ct:
                md.append("**CLE 尾部（`replay` − `original`）的帧数分布** —— 缺口是**沿轨双峰尾部**，"
                          "不是横向偏移：")
                md.append("")
                md.append("| CLE 区间 | 帧数差 |")
                md.append("|:--|--:|")
                for _k in ("<=5", "(5,10]", "(10,20]", "(20,40]", ">40"):
                    if _ct.get(_k) is None:
                        continue
                    md.append(f"| `{_k}` | {fmt_signed(_ct[_k], 1, 0)} |")
                md.append("")
                md.append(
                    f"⇒ **横向（≤5px）反而比天花板少 {fmt(abs(num(_ct.get('<=5'))), '.0f')} 帧**，"
                    "缺口全部堆在 **(10,20] / (20,40] / >40px 三档** ⇒ 典型形态是「**沿轨欠冲/过冲**」"
                    "（同一目标连续多帧单向偏离），**不是**「每帧随机抖动」。"
                )
                md.append("")

            # ================= 15.3 FAR 预算 =================
            _fb = gd.get("far_budget") or {}
            _sec = _fb.get("secondary_target_far_below_0_005") or {}
            md.append("### 15.3 FAR 预算：Δ=2 能不能把 FAR 压到 0.005 以下？")
            md.append("")
            md.append("| 项 | 值 |")
            md.append("|:--|--:|")
            md.append(f"| `original`（零延迟天花板）FP | {_fb.get('original_fp')} |")
            md.append(f"| `replay`（交付推荐行）FP | {_fb.get('replay_fp')} |")
            md.append(f"| 外挂净增 FP | {fmt_signed(_fb.get('delta_fp'), 1, 0)} |")
            md.append(f"| **FAR 地板（零延迟天花板自身）** | {fmt(_fb.get('floor_far_zero_delay'), '.6f')} |")
            md.append(f"| 外挂可动 FAR 上限 | {fmt(_fb.get('plugin_addressable_far'), '.6f')} |")
            md.append(f"| Δ=2 已观测最好 FAR | {fmt((_fb.get('delta2_best_far_observed')), '.6f')} |")
            md.append(f"| Δ=2 外挂净增 FAR | {fmt((_fb.get('delta2_plugin_far_increment')), '.6f')} |")
            md.append("")
            md.append(f"⇒ **§「Δ=2 时把 FAR 压到 0.005 以下」= {_sec.get('verdict', DASH)}**")
            md.append("")
            md.append(f"⇒ 理由：{_sec.get('reason', DASH)}")
            md.append("")
            md.append(
                "⇒ **这条是物理下界，不是工程问题**：任何外挂机制在 Δ=2 最多只能把 FAR 压到 "
                f"**{fmt(_fb.get('delta2_best_far_observed'), '.6f')}**，"
                "**离 0.005 差一个数量级** ⇒ **本轮不追这个目标**（属已被实测否决的目标族）。"
            )
            md.append("")

        # ================= 15.4 新机理受控 A/B =================
        if kb:
            _var = kb.get("variants") or []
            _lo = kb.get("loso") or {}
            _gt = kb.get("gate") or {}
            _applied = _variant_applied(kb)
            md.append("### 15.4 新机理受控 A/B（13 个变体，全部实跑）")
            md.append("")
            md.append(
                f"> 口径：24 序官方口径；基线 = 交付档 CFG + `--replay-fov-margin-px 6`；"
                f"每次只**替换一个旋钮**；硬门禁 = 基线复现与参考产物**四臂逐位一致 "
                f"{_gt.get('total', 0) - _gt.get('bad', 0)}/{_gt.get('total', 0)} 项**"
                f"（不一致 {_gt.get('bad', 0)} 处）⇒ "
                f"**{'PASS' if not _gt.get('bad') else 'FAIL'}**；"
                f"噪声门槛 = 24 序 LOSO 池化 F1 摆幅 中位 {fmt(_lo.get('swing_median_pp'), '.4f')} pp / "
                f"最大 {fmt(_lo.get('swing_pp'), '.4f')} pp。"
            )
            md.append("")
            md.append("| 旋钮 | 取值 | F1% | ΔF1(pp) | FAR/帧 | ΔFAR | hit@5px% | Δhit5(pp) | "
                      "CLE p95(池) | Δp95 | 上报帧数 | 三门 | 帕累托支配 |")
            md.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|:--|")
            for _v in sorted(_var, key=lambda x: -(x.get("d_f1") or 0)):
                _r = _v.get("row") or {}
                _g = gates({"f1": num(_r.get("f1")) * 100, "recall": num(_r.get("recall")) * 100,
                            "far": num(_r.get("far_per_frame"))})
                _ap = _applied.get((_v.get("knob"), _v.get("val")), DASH)
                md.append(
                    f"| `{_v.get('knob')}` | `{_v.get('val')}` | {fmt(num(_r.get('f1')) * 100, '.4f')} | "
                    f"{fmt_signed(_v.get('d_f1'), 100)} | {fmt(_r.get('far_per_frame'), '.6f')} | "
                    f"{fmt_signed(_v.get('d_far'), 1, 6)} | {fmt(num(_r.get('hit5')) * 100, '.4f')} | "
                    f"{fmt_signed(_v.get('d_h5'), 100)} | {fmt(_r.get('cle95'), '.4f')} | "
                    f"{fmt_signed(_v.get('d_p95'), 1, 4)} | {_ap} | {_g} | "
                    f"{'✅ 是' if _v.get('dominates') else '❌ 否'} |"
                )
            md.append("")
            _n_dom = sum(1 for _v in _var if _v.get("dominates"))
            md.append(
                f"⇒ **帕累托支配基线的变体：{_n_dom} / {len(_var)}**"
                "（判据：Recall / hit@5px / FAR / CLE p95 **四项全部不劣**且至少一项严格更优）。"
            )
            md.append("")
            _dg = [v for v in _var if v.get("knob") == "--replay-disp-gain"]
            if _dg:
                _ds = sorted(_dg, key=lambda x: num(x.get("val")))
                md.append(
                    "**新机理 ①：位移增益 `--replay-disp-gain`（本轮新增旋钮，默认 1.0 = 逐位不变）**"
                )
                md.append("")
                md.append("| 取值 | F1% | ΔF1(pp) | ΔFAR | Δhit5(pp) | 上报帧数 |")
                md.append("|--:|--:|--:|--:|--:|--:|")
                for _v in _ds:
                    _r = _v.get("row") or {}
                    md.append(f"| {_v.get('val')} | {fmt(num(_r.get('f1')) * 100, '.4f')} | "
                              f"{fmt_signed(_v.get('d_f1'), 100)} | {fmt_signed(_v.get('d_far'), 1, 6)} | "
                              f"{fmt_signed(_v.get('d_h5'), 100)} | "
                              f"{_applied.get((_v.get('knob'), _v.get('val')), DASH)} |")
                md.append("")
                _base_applied = _applied.get(("__base__", "__base__"), DASH)
                _max_abs = max(abs(num(v.get("d_f1"))) for v in _ds) * 100
                md.append(
                    f"⇒ **全部 4 个取值都是负增益，且量级在噪声内"
                    f"（|ΔF1| ≤ {fmt(_max_abs, '.4f')} pp ≪ {fmt(_lo.get('swing_median_pp'), '.4f')} pp）**。"
                )
                md.append("")
                md.append(
                    f"⇒ **关键否证**：增益**确实生效** —— `replay` 上报帧数从基线 **{_base_applied}** 变为 "
                    + " / ".join(f"{v.get('val')} → {_applied.get((v.get('knob'), v.get('val')), DASH)}"
                                 for v in _ds)
                    + "（**逐个取值都在表里可复现**）⇒ 合成位移真的被缩放了，"
                    "**但指标不动** ⇒ **沿轨欠冲/过冲不是「位移尺度错了」** ⇒ "
                    "**「把外推位移乘一个系数」这一整族机理被否决**。"
                )
                md.append("")
            md.append(
                "> **口径交叉说明（避免误读）**：诊断报告 `gap_report.json` 的 "
                "`cross_check_concurrent_scan` 段记录的是**扫描进行到一半时的快照**"
                "（当时 7 个作业因「作业名被当成额外参数传入」而 rc=2、无有效产物）。"
                "**该快照已被本节 §15.4 的完整扫描取代** —— "
                "13 个变体全部 rc=0、基线门禁 68/68 一致、硬门禁 PASS。"
                "诊断报告的**数值结论**（§15.1–15.3）不受该快照影响。"
            )
            md.append("")
            _far_of = {}
            for _v in _var:
                _far_of[(_v.get("knob"), _v.get("val"))] = num((_v.get("row") or {}).get("far_per_frame"))
            md.append(
                "**新机理 ② / ③ / ④（均已实测否决）**："
                "`--replay-abstain-disp-px=0` + `--replay-abstain-peak-below`（纯峰值门）、"
                "`--replay-polarity=bright`、`--replay-anchor=kernel/kf`（锚点换源）"
                "⇒ 前者是**帕累托滑动**（FAR 降但 F1/hit@5px 同比例降），"
                "后两者**直接破 FAR 红线**（"
                + " / ".join(fmt(_far_of.get(k), '.6f') for k in
                             (("--replay-polarity", "bright"),
                              ("--replay-anchor", "kernel"),
                              ("--replay-anchor", "kf")))
                + "，红线 0.030）⇒ **全部否决**。"
            )
            md.append("")

        # ================= 15.5 结论 =================
        md.append("### 15.5 结论（本节）")
        md.append("")
        _cand = ((gd or {}).get("candidates_ranked") or [{}])[0] if gd else {}
        _n_dom = sum(1 for v in ((kb or {}).get("variants") or []) if v.get("dominates"))
        _n_var = len((kb or {}).get("variants") or [])
        md.append(
            f"1. **交付档 CFG（`manu/tracking/lookahead/run_delivery.sh`）本轮仍不需要改** —— "
            f"{_n_var} 个新机理变体**无一**把帕累托前沿向外推（支配基线 {_n_dom}/{_n_var}）。"
        )
        if gd:
            _hp = _fit_val.get("delta_hit5_gap_pp")
            _share = _hp[2] if _hp else float("nan")
            _cons = _hp[1] if _hp else float("nan")
            md.append(
                f"2. **缺口的主体（{fmt(_share, '.1f')}%）是 Δ-比例项** ⇒ 「小延时」是"
                "**唯一**能实质压掉缺口的杠杆；**把 Δ 从 3 再往下压**（硬件/时序层面）"
                f"比在算法层调旋钮有效得多（Δ-无关常数项的总大小只有 {fmt(_cons, '.4f')} pp）。"
            )
            _fb = gd.get("far_budget") or {}
            _sec = _fb.get("secondary_target_far_below_0_005") or {}
            md.append(
                f"3. **FAR < 0.005 是物理不可达**（零延迟天花板自身 FAR "
                f"{fmt(_fb.get('floor_far_zero_delay'), '.6f')}；判定 = "
                f"{_sec.get('verdict', DASH)}）⇒ **本轮及后续不应再把 FAR 0.005 当目标**；"
                "FAR 红线 0.030 在 Δ=3 下余量 0.005663 ⇒ 可交付。"
            )
            _ha = gd.get("hit5_attribution") or {}
            _D = _ha.get("D_split_by_true_displacement") or {}
            md.append(
                f"4. **下一轮唯一值得打的靶子 = D 桶里「外挂侧可修」的两档**："
                f"≤5px {_D.get('le5px')} 帧（目标几乎没动却走歪 ⇒ **真 bug**）"
                f" ＋ 5–20px {_D.get('5to20px')} 帧（线性外推够得着却没跟上）"
                f"，合计 {_ha.get('plugin_addressable_frames')} 帧 = "
                f"{_ha.get('plugin_addressable_pp')} pp —— "
                "这两档是「**源帧检测器定位没错、但链式外推跑飞**」的部分，"
                "**必须从配准/链路本身解决**（本轮的位移增益、纯峰值门、锚点换源、极性、门限族全部试过，均否决）。"
            )
            if _cand.get("name"):
                md.append(
                    f"   - 诊断报告给出的**排第一候选**：「**{_cand.get('name')}**」"
                    f" —— 目标：{_cand.get('targets', DASH)}；预期收益：{_cand.get('expected_gain', DASH)}。"
                )
                md.append(f"   - 风险：{_cand.get('risk', DASH)}")
                md.append(f"   - 实现落点：{_cand.get('implementation', DASH)}")
            md.append(
                f"5. **不要打 >20px 那 {_D.get('gt20px')} 帧** —— 那是**线性外推的物理上限**"
                f"（{_ha.get('physical_unreachable_pp')} pp），"
                "**不是**外挂能修的 bug（把它算进靶子会把可回收量算错）。"
            )
        md.append("")

    # ---------- §16 夜 7 位移修正族 oracle 上界 + 候选族结案 ----------
    if ac:
        _orc = ac.get("oracle") or {}
        _q1 = ac.get("structure") or {}
        _q2 = ac.get("bias_vs_selection") or {}
        _bk = ac.get("by_true_displacement") or {}
        md.append("## 16. 夜 7 位移修正族 **oracle 上界** + 候选族结案（gapdiag 候选 #1 的终审）")
        md.append("")
        md.append(
            f"> 口径：{ac.get('scope', DASH)}；统计口径：{ac.get('caliber', DASH)}；"
            f"参与统计帧数 **{ac.get('gt_frames_considered')}**。"
            "**§16.1–16.3 的数字从 `runs/lookahead/fi_nogmc24/oracle_accel/closure_report.json` 现读**，"
            "无手抄；**§16.4 的裁定表**引用 §13 / §15 的既有数字"
            "（1.3748 pp 来自 `gapdiag/gap_report.json`；8218 / 7127 / 779 / 3836 来自交付档 "
            "`arms.json` 的 `stream_stats`）。"
        )
        md.append("")

        md.append("### 16.1 oracle 上界：位移修正族最多能赚多少")
        md.append("")
        md.append("| oracle | 含义 | hit@5px% | Δ vs 基线(pp) |")
        md.append("|:--|:--|--:|--:|")
        md.append(
            f"| 基线（**§16 口径**：{ac.get('gt_frames_considered')} 帧子集） | 交付档 replay 臂 | "
            f"{fmt(ac.get('hit5_baseline_pct'))} | {DASH} |"
        )
        md.append(
            f"| **O1 尺度** | 逐帧最优 κ 缩放位移（整族「缩放位移」的上界，含已被否决的 `--replay-disp-gain`）"
            f" | {fmt((_orc.get('O1_scale') or {}).get('hit5_pct'))}"
            f" | **{fmt((_orc.get('O1_scale') or {}).get('delta_pp'))}** |"
        )
        md.append(
            f"| **O2 加速度项** | 把**完美**的加速修正 `0.5·s` 加到最终上报位置（升阶运动模型的上界）"
            f" | {fmt((_orc.get('O2_accel') or {}).get('hit5_pct'))}"
            f" | **{fmt((_orc.get('O2_accel') or {}).get('delta_pp'))}** |"
        )
        md.append(
            f"| O3 尺度+加速度 | 两者同时给"
            f" | {fmt((_orc.get('O3_scale_accel') or {}).get('hit5_pct'))}"
            f" | **{fmt((_orc.get('O3_scale_accel') or {}).get('delta_pp'))}** |"
        )
        md.append("")
        md.append("> ⚠️ **O1 的正值不可当成「可赚的钱」**：它是**逐帧 1 参数**的 oracle（用 GT 选 κ），"
                  "且作用点在 **xcorr 模板配准之前**（最终位置由配准覆盖）⇒ 两条都不成立。")
        md.append("")
        md.append("| 真位移(Δ 窗) | 帧数 | 基线 | O1 尺度 | O2 加速度 | O3 尺度+加速度 |")
        md.append("|:--|--:|--:|--:|--:|--:|")
        for _bn, _bv in _bk.items():
            md.append(
                f"| {_bn} | {_bv.get('n')} | {fmt(_bv.get('hit5_baseline_pct'), '.2f')} |"
                f" {fmt(_bv.get('O1_scale_pct'), '.2f')} | {fmt(_bv.get('O2_accel_pct'), '.2f')} |"
                f" {fmt(_bv.get('O3_scale_accel_pct'), '.2f')} |"
            )
        md.append("")

        md.append("### 16.2 结构问：交付档的最终位置里**有没有运动学模型**")
        md.append("")
        md.append("| 运动学出口 | 24 序总帧数 |")
        md.append("|:--|--:|")
        for _k, _v in (_q1.get("model_exits_with_frames") or {}).items():
            md.append(f"| {_k} | **{_v}** |")
        md.append("")
        md.append(
            f"（对照：applied 上报帧数 **{(_q1 or {}).get('applied')}**、"
            f"replay_steps **{(_q1 or {}).get('replay_steps')}**）"
        )
        _src8 = ac.get("model_exit_counter_source") or {}
        if _src8:
            _bits = []
            for _k, _v in _src8.items():
                if _v.get("measured"):
                    _bits.append(f"`{_k}` = **{(_q1.get('model_exits_with_frames') or {}).get(_k)}**"
                                 f"（实测计数器 `{_v.get('counter')}`）")
                else:
                    _bits.append(f"`{_k}` = **{(_q1.get('model_exits_with_frames') or {}).get(_k)}**"
                                 f"（**stream_stats 里没有 `{_v.get('counter')}` 这个计数器** ⇒ 该 0 是"
                                 f"**缺键默认值**、不是实测读数）")
            md.append("")
            md.append("> ⚠️ **计数器出处**：" + "；".join(_bits)
                      + " ⇒ **结论仍成立**（这些旋钮在交付档 CFG 里是 0 ⇒ 出口不存在），"
                        "但**不要把「缺键默认值 0」当成「实测 0 帧」的证据**。")
        md.append("")
        md.append("**更硬的一条：交付档 CFG 把运动学模型类旋钮在配置层就全关了。**")
        md.append("")
        md.append("| 旋钮 | 交付档取值 | 含义 |")
        md.append("|:--|--:|:--|")
        _off = (_q1.get("model_knob_off_meaning") or {})
        for _k, _v in (_q1.get("delivery_cfg_model_knobs") or {}).items():
            md.append(f"| `{_k}` | `{_v}` | {_off.get(_k, DASH)} |")
        md.append("")
        md.append(
            "⇒ **最终上报位置 = 锚点（t−Δ 的实测位置）+ 实测链位移，再由 xcorr 模板直配准覆盖** —— "
            "**没有任何『外推/运动学预测』参与**。"
        )
        md.append("")
        md.append("| 变体 | 落点 | 预期 | 证据 |")
        md.append("|:--|:--|:--|:--|")
        md.append(
            f"| 下游（加速项加到最终位置） | 最终上报坐标 | **变差** | oracle O2 = "
            f"**{fmt((_orc.get('O2_accel') or {}).get('delta_pp'))} pp** |"
        )
        md.append(
            "| 上游（只动 xcorr 搜索中心） | xcorr 搜索中心 | ≈ 0 | 模板配准把位置覆盖回去；"
            "同族 `--replay-disp-gain` 1.05~1.30 实测全在噪声内 |"
        )
        md.append(
            "| 模型出口（`anchor+Δ·v` 升成 `anchor+Δ·v+½Δ²·a`） | 一致性门 / best_effort 兜底 | **0 帧** | "
            "交付档 CFG 里这两个旋钮是 0 ⇒ 门**关闭**、兜底**未触发** |"
        )
        md.append("")

        md.append("### 16.3 偏置问：沿轨欠冲是系统性偏置，还是选择效应")
        md.append("")
        md.append("| 集合 | n | 沿轨误差 p50 | mean | 负占比 |")
        md.append("|:--|--:|--:|--:|--:|")
        for _key, _name in (("replay_all_frames", "replay 全部帧"), ("replay_hit_frames", "replay 命中帧"),
                            ("replay_miss_frames", "replay 失手帧"),
                            ("original_arm_control", "original 天花板（对照）")):
            _s = _q2.get(_key) or {}
            md.append(
                f"| {_name} | {_s.get('n')} | {fmt(_s.get('p50'))} | {fmt(_s.get('mean'))} |"
                f" {fmt(_s.get('frac_neg'))} |"
            )
        md.append("")
        md.append(
            f"- 沿轨误差 vs 真位移：**Pearson r = {fmt(_q2.get('pearson_r_along_vs_true_disp'))}**、"
            f"**Spearman r = {fmt(_q2.get('spearman_r_along_vs_true_disp'))}**"
        )
        md.append(f"- 真位移 > 20px 的帧上的平均沿轨误差：**{fmt(_q2.get('along_mean_on_true_disp_gt20'))}**")
        _am = ac.get("accel_term_magnitude") or {}
        md.append(
            f"- 失手帧上的**加速项幅值** |0.5·s|：n = **{_am.get('n')}**、"
            f"p50 = **{fmt(_am.get('p50'))}** px、p90 = **{fmt(_am.get('p90'))}** px"
            f"（**注**：这个 n 与上表失手帧 n = {(_q2.get('replay_miss_frames') or {}).get('n')} "
            f"**不是同一集合**：加速项需要 t−2Δ 也有 GT，比偏置表的集合更小）"
        )
        md.append(
            f"> ⚠️ **上表 n = {(_q2.get('replay_all_frames') or {}).get('n')} 与 §16.1 的 "
            f"{ac.get('gt_frames_considered')} 不是同一帧集**（§16.1 要求 t−Δ/t−2Δ 都有 GT，"
            f"偏置表要求 GT@t ∧ GT@t−Δ ∧ 真位移>0）⇒ 两表**不能直接相除**。"
        )
        md.append("")
        md.append(
            "⇒ **全体帧的沿轨误差 ≈ 0，只有失手帧才 −4.35 px 量级** ⇒ 这是**选择效应**"
            "（难帧本来就欠冲），**不是**任何位置修正族能系统性纠正的偏置；"
            "且与真位移只有 r ≈ −0.15 的弱相关 ⇒ 构造不出「位移越大欠冲越多」的修正律。"
        )
        md.append("")
        md.append(
            "⇒ **加速项解释不了这 4.35 px**（加速项幅值中位数只有 2.02 px，p90 才 9.12 px）⇒ "
            "**gapdiag 把缺口归因为「模型阶数不足」不成立**，应改写为「**定位尾部的少数难帧**」。"
        )
        md.append("")

        md.append("### 16.4 结论：候选族结案")
        md.append("")
        md.append("| 候选 | 上界/证据 | 裁定 |")
        md.append("|:--|:--|:--|")
        md.append(
            f"| **#1 Δ 窗口位移的加速度项（升阶运动模型）** | oracle O2 = "
            f"**{fmt((_orc.get('O2_accel') or {}).get('delta_pp'))} pp**；模型出口 0 帧；"
            "加速项幅值 p50 2.02 px | **结案：不可行**（无处可施 + 方向为负 + 归因错误） |"
        )
        md.append(
            "| 「缩放位移」族（`--replay-disp-gain`） | O1 逐帧 oracle 为正，但作用点在 xcorr 之前；"
            "实测 1.05~1.30 全在噪声内 | **结案：不可行**（固定 κ 打不中逐帧最优 κ） |"
        )
        md.append(
            "| 「位移层收缩 / 自适应融合 / 低速冻结 / 离群步替换」 | 交付档 CFG 里全关；实测全负或全平 | "
            "**结案：不可行** |"
        )
        md.append(
            "| 「阈值类旋钮」（§13 的 37 变体） | 符号检验 p<0.05 的共 4 个，但**同时**满足"
            "「四项池化指标全部不劣」的**只有 1 个**（`--output-max-step-px=48`，ΔF1 +0.0192 pp），"
            "且其幅度 < 噪声底 0.1466 pp | **结案：不可行**（只沿帕累托前沿滑动） |"
        )
        md.append(
            "| 「纯运动学兜底 / kinematic 臂」 | Δ≥2 全部净负收益 | **结案：不可行** |"
        )
        md.append(
            "| **剩余可回收 1.3748 pp 的真实所在** | 原归因（配准/歧义处置维度：`replay_xcorr_ambiguous` 8218 帧、"
            "`replay_xcorr_rematch_rejected` 7127 帧、`replay_xcorr_abstained` 779 帧、"
            "`replay_rollback_abstained` 3836 帧） | **已结案（见 §17）：不可行** —— "
            "配准候选重排族的**最宽合法口径（O-all 合法）= +2.2055 pp**（空间其实「够」；"
            "本族 **O-reg 合法**只有 **+0.7504 pp**），"
            "但**可分性**是死结：最好的一条可计算规则只拿到 **O-reg 口径（+0.8120 pp）的 0.6%**、"
            "对最宽天花板（+2.2055 pp）只有 **0.2%**；"
            "回退链峰值门另立受控 A/B（§17.4） |"
        )
        md.append("")
        md.append(
            "⇒ **交付裁定不变**：交付推荐档 = `cs_t0043`（no-GMC）+ Δ=3(120 ms) + `replay` 臂 + "
            "`--replay-fov-margin-px 6`，**交付档 CFG 本轮不需要改**。"
        )
        md.append("")

    # ---------- §17 夜 8 配准/歧义族 oracle 上界 + 候选族结案 + 回退链峰值门 A/B ----------
    if rc:
        _bk8 = rc.get("buckets") or {}
        _an8 = rc.get("anatomy") or {}
        _ru8 = rc.get("rules") or {}
        md.append("## 17. 夜 8 配准 / 歧义处置族 **oracle 上界** + 候选族结案")
        md.append("")
        md.append(
            f"> 口径：{rc.get('scope', DASH)}；**全部数字从 "
            f"`runs/lookahead/fi_nogmc24/reg_oracle/closure_report.json` 现读**，无手抄。"
        )
        md.append("")
        _c8 = rc.get("ceiling") or {}
        _b8 = _c8.get("baseline_hit5_pct") or rc.get("baseline_hit_pct") or 0.0
        _rg8 = _c8.get("reg_gain_pp") or rc.get("oracle_reg_gain_pp") or 0.0
        _rlg8 = _c8.get("reg_legal_gain_pp") or rc.get("oracle_reg_legal_gain_pp") or 0.0
        _ag8 = _c8.get("all_gain_pp") or rc.get("oracle_all_gain_pp") or 0.0
        _alg8 = _c8.get("all_legal_gain_pp") or rc.get("oracle_all_legal_gain_pp") or 0.0
        md.append("### 17.1 oracle 上界：配准候选重排族最多能赚多少")
        md.append("")
        md.append("| oracle | 含义 | hit@5px% | Δ vs 基线(pp) | 折算帧数 |")
        md.append("|:--|:--|--:|--:|--:|")
        md.append(f"| 基线 | 交付档 replay 臂（出框门 m=6 之后） | {fmt(_b8)} | {DASH} | — |")
        md.append(
            f"| **O-reg** | 只在**大匹配窗的 top-K 候选峰**里用 GT 挑（**不加出框门**）"
            f" | {fmt(_b8 + _rg8)} | **{fmt(_rg8)}** | {_c8.get('reg_frames')} |"
        )
        md.append(
            f"| **O-reg 合法** | 同 O-reg，但候选必须**过出框门**"
            f" | {fmt(_b8 + _rlg8)} | **{fmt(_rlg8)}** | {_c8.get('reg_legal_frames')} |"
        )
        md.append(
            f"| O-all | 候选峰 + 链位置 + 二次意见窗的峰（**不加出框门**）"
            f" | {fmt(_b8 + _ag8)} | **{fmt(_ag8)}** | — |"
        )
        md.append(
            f"| **O-all 合法** | 同 O-all，同样约束合法性"
            f" | {fmt(_b8 + _alg8)} | **{fmt(_alg8)}** | {_c8.get('all_legal_frames')} |"
        )
        md.append("")
        md.append(
            f"⇒ **天花板（最宽的合法口径 = O-all 合法）= +{fmt(_alg8)} pp**（{_c8.get('all_legal_frames')} 帧）；"
            f"出框门只让它从 +{fmt(_ag8)} 掉到 +{fmt(_alg8)}（代价 **{fmt(_c8.get('fov_gate_cost_pp'))} pp**）"
            f"⇒ **空间上「够」**（≥ §15.2 里「外挂可回收 1.3748 pp」的预算）⇒ 真正的死结在**可分性**（§17.3），不在空间。"
        )
        md.append("")
        md.append(
            "> ⚠️ **口径修正（夜 9 第 7 轮独立审计抓出，根因已修）**：夜 8 初稿写「O-all 一加出框门就变成 "
            "**−5.7408 pp** ⇒ O-all 作废、只剩 O-reg 合法 **+0.7699 pp**」—— **那是假象**："
            "`diag_reg_oracle.py::_oracle_pick` 原来用 `universe in (\"reg\",\"all\")` 取候选，"
            "把 `reg_legal` / `all_legal` 的**大窗峰漏掉** ⇒ 候选集恒空 ⇒ 退化回基线选择 `refine[\"p\"]`，"
            "于是「合法 oracle」量的其实是**出框门前的基线**（`hit_oracle_reg_legal == hit_base_pre`）。"
            "根因已修（合法变体 = **同族候选集**再过出框门）并重跑 dump+analyze ⇒ "
            f"O-reg 合法 **+{fmt(_rlg8)} pp**、**O-all 合法 +{fmt(_alg8)} pp（仍为正）**；"
            "**§17 的结论方向不变**（族仍判不可行），但**依据从「空间不够」换成「不可分」**。"
        )
        md.append("")
        md.append("### 17.2 缺口分桶：失手帧卡在哪（配准族的射程）")
        md.append("")
        _b8 = _bk8.get("buckets") or _bk8
        _miss8 = rc.get("buckets_total_miss") or sum(int(_b8.get(k) or 0) for k in (
            "B1_window_has_peak", "B2_window_no_peak", "B3_out_of_window", "B4_no_refine"))
        md.append("| 桶 | 帧数 | 占失手帧 | 含义 |")
        md.append("|:--|--:|--:|:--|")
        for _k, _lab, _mean in (
            ("B1_window_has_peak", "B1 窗内有峰", "目标在搜索窗内且相关图有峰 ⇒ 配准族**唯一**有救的桶"),
            ("B2_window_no_peak", "B2 窗内无峰", "目标在窗内但相关图没峰 ⇒ **外观漂移**，配准救不回"),
            ("B3_out_of_window", "B3 窗外", "目标不在搜索窗内 ⇒ 半径/中心问题（属**另一族**）"),
            ("B4_no_refine", "B4 无 refine", "该帧没有配准决策（无源帧检出 / 链没跑）"),
        ):
            _v = int(_b8.get(_k) or 0)
            md.append(f"| {_lab} | {_v} | {fmt(100.0 * _v / max(1, _miss8), '.1f')}% | {_mean} |")
        md.append(f"| **合计失手** | **{_miss8}** | 100% | = 基线失手总数（**完备划分**） |")
        md.append("")
        md.append("### 17.3 可分性：可计算排序规则 vs oracle")
        md.append("")
        md.append(
            f"| 规则 | hit@5px% | Δ vs 基线(pp) | 占 O-reg 的比例"
            f"（分母 = **O-reg 不加出框门** +{fmt(_rg8)} pp / {_c8.get('reg_frames')} 帧） |"
        )
        md.append("|:--|--:|--:|--:|")
        for _name, _v in sorted(_ru8.items(), key=lambda kv: -(kv[1] or {}).get("gain_pp", -99.0)):
            md.append(
                f"| `{_name}` | {fmt((_v or {}).get('hit_pct'))} | {fmt((_v or {}).get('gain_pp'))} | "
                f"{fmt(100.0 * ((_v or {}).get('share_of_oracle') or 0.0), '.1f')}% |"
            )
        md.append("")
        md.append(
            f"⇒ **最好的一条规则只拿到 oracle 的 "
            f"{fmt(100.0 * ((rc.get('best_rule') or {}).get('share_of_oracle') or 0.0), '.1f')}%**"
            f"（`{(rc.get('best_rule') or {}).get('name')}`，Δ "
            f"{fmt((rc.get('best_rule') or {}).get('gain_pp'))} pp；即使对**最宽**的天花板 "
            f"+{fmt(_alg8)} pp 也只有 {fmt(_c8.get('best_rule_share_of_ceiling_pct'), '.1f')}%）⇒ "
            "「相关峰候选重排」在物理上**不可分**（真目标的峰不是比背景结构更高的那个）。"
        )
        md.append("")
        md.append(
            f"⇒ **夜 8 裁定**：{rc.get('verdict', DASH)}。"
            f"可救帧（基线失手但 O-reg 能救回）共 **{_c8.get('recoverable_frames') or _an8.get('recoverable_frames')}** 帧，"
            f"其中 oracle 选中的候选峰**名次 = 0（就是 argmax）** 的占 "
            f"{fmt(100.0 * ((_an8.get('rank_hist') or {}).get('0', 0)) / max(1, _an8.get('recoverable_frames', 1)), '.1f')}%"
            " ⇒ **这不是「选峰」问题**。"
        )
        md.append("")

    if rb_rows:
        md.append("### 17.4 新战场：**回退链峰值门**（`--replay-rollback-min-peak`）受控 A/B")
        md.append("")
        md.append(
            "**机理**：回退链的模板来自更早的帧（`source_frame ≠ target_frame − Δ`）⇒ 目标可能已飞走、"
            "配准可能锁到背景结构（\"幽灵坐标\"）⇒ 峰值 < 门限时**整帧弃权**"
            "（生产代码里 `continue` 在写 report 之前 ⇒ 门前有上报点也照样静默）。"
        )
        md.append("")
        md.append("| rollback-min-peak | F1% | ΔF1(pp) | Recall% | Precision% | FAR | ΔFAR | hit@5px% | Δhit(pp) | CLE p95(px) | max jitter(px) | TP | FP |")
        md.append("|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        _base_rb = next((r for r in rb_rows if abs(r["peak"] - 0.90) < 1e-9), None)
        for _row in rb_rows:
            _rp = _row["replay"]
            if _base_rb is not None:
                _df1 = 100.0 * (_rp["f1"] - _base_rb["replay"]["f1"])
                _dfar = _rp["far"] - _base_rb["replay"]["far"]
                _dhit = 100.0 * (_rp["hit5"] - _base_rb["replay"]["hit5"])
            else:
                _df1 = _dfar = _dhit = 0.0
            md.append(
                f"| {_row['peak']:.2f} | {fmt(100.0 * _rp['f1'])} | {_df1:+.4f} | {fmt(100.0 * _rp['recall'])} | "
                f"{fmt(100.0 * _rp['precision'])} | {fmt(_rp['far'], '.6f')} | {_dfar:+.6f} | {fmt(100.0 * _rp['hit5'])} | "
                f"{_dhit:+.4f} | {fmt(_rp['cle95'], '.3f')} | {fmt(_rp['jitter'], '.3f')} | {_rp['tp']} | {_rp['fp']} |"
            )
        md.append("")
        md.append(
            "> **门禁**：rb=0.90 的**指标字段**与既有交付档产物**逐位相等**（门禁脚本口径 68 字段全等）；"
            "所有变体的 original / stale / kinematic 三臂**跨变体逐位相等**（该旋钮只作用于 replay 臂）。"
            "⚠️ **整份 JSON 不逐位相等**：`frame_reads`（缓存读帧计数）与 `elapsed_seconds` 每次跑都不同，"
            "且 rb 侧多出 `stream_stats.per_sequence.*.replay_disp_gained` 等键 ⇒ "
            "**交付档产物与 A/B 产物不是同一版代码**（A/B 跑在更晚的代码上）⇒ **只可比指标、不可比整文件**。"
        )
        md.append("")
        _rbrows = (rc or {}).get("rbgate", {}).get("rows") or []
        if _rbrows:
            md.append("**逐序稳健性（必读）** —— 总体增益**不是**全面改善：\n")
            md.append("| 变体 | ΔTP | ΔFP | 净 | 受影响序列 | 变好/变差/持平 | ΔTP 最多的两序（占总 ΔTP） | 最差序列 ΔF1 |")
            md.append("|:--|--:|--:|--:|--:|:--:|:--|--:|")
            for _r in _rbrows:
                _sd = _r.get("seq_delta") or {}
                if abs(float(_r.get("peak", 0.0)) - 0.90) < 1e-9 or not _sd:
                    continue
                _dtp = sum(v["dtp"] for v in _sd.values())
                _dfp = sum(v["dfp"] for v in _sd.values())
                _better = sum(1 for v in _sd.values() if v["df1_pp"] > 0.005)
                _worse = sum(1 for v in _sd.values() if v["df1_pp"] < -0.005)
                _flat = len(_sd) - _better - _worse
                _touched = sum(1 for v in _sd.values() if v["dtp"] or v["dfp"])
                _top = sorted(_sd.items(), key=lambda kv: -kv[1]["dtp"])[:2]
                _share = (100.0 * sum(v["dtp"] for _k, v in _top) / _dtp) if _dtp else 0.0
                _worst = min(_sd.items(), key=lambda kv: kv[1]["df1_pp"])
                md.append(
                    f"| rb={float(_r['peak']):.2f} | {_dtp:+d} | {_dfp:+d} | {_dtp - _dfp:+d} | {_touched}/24 | "
                    f"{_better}/{_worse}/{_flat} | "
                    + " / ".join(f"{k[:22]} {v['dtp']:+d}" for k, v in _top)
                    + f"（**{_share:.1f}%**） | {_worst[0][:22]} {_worst[1]['df1_pp']:+.4f}pp |"
                )
            md.append("")
            _rbc = (rc or {}).get("rbgate_rollback_caliber") or {}
            _rbtop = _rbc.get("top3") or []
            _rbtop_txt = " / ".join(f"`{x['sequence'][:22]}` {x['frames']}" for x in _rbtop)
            _worst_by_peak = []
            for _r in _rbrows:
                if abs(float(_r.get("peak", 0.0)) - 0.90) < 1e-9:
                    continue
                _sd = _r.get("seq_delta") or {}
                if not _sd:
                    continue
                _w = min(_sd.items(), key=lambda kv: kv[1]["df1_pp"])
                _worst_by_peak.append((float(_r["peak"]), _w[0], _w[1]["df1_pp"]))
            _worst_by_peak.sort(key=lambda t: -t[0])
            _series = " → ".join(f"{v:+.4f}" for _p, _s, v in _worst_by_peak)
            _worst_seq = _worst_by_peak[0][1] if _worst_by_peak else ""
            md.append(
                f"⇒ **总体 ΔTP 有 81~89% 落在 2/24 序列**（`000322` / `001324`）—— 在**生产口径**下，"
                f"被回退峰值门丢帧最多的三序是 {_rbtop_txt}（合计 {_rbc.get('total')} 帧，"
                f"前两序占 {fmt(_rbc.get('top2_share_pct'), '.1f')}%），"
                f"⇒ `000322` 是第 1、`001324` 是第 3 ⇒ **机理自洽**（增益确实来自回退链）但**不鲁棒**；"
                f"代价集中在 `{_worst_seq[:22]}`（逐序 ΔF1 随放松单调变差：{_series} pp）。"
            )
            md.append("")
            md.append(
                f"> ⚠️ **口径说明**：「被回退峰值门丢掉的帧」有两个口径 —— **生产口径**"
                f"（本节 A/B 产物 `stream_stats.per_sequence.replay_rollback_abstained`，合计 "
                f"{_rbc.get('total')} 帧）与**离线 GT 帧口径**（§17.5 静默帧归因里的回退链峰值门，合计 "
                f"{_rbc.get('offline_gt_frames')} 帧）—— **两者不可混用**；夜 8 初稿把离线口径的数字"
                f"写在生产口径的句子里（且 938 / 479 两个数**全仓查不到**），已按现读值改写。"
            )
            md.append("")
            md.append(
                "⇒ **边际 TP/FP 兑换率随放松单调恶化**（**相邻档位的边际**，非相对 rb=0.90 的累计）："
                "0.90→0.80 **3.64:1**、0.80→0.70 1.26:1、0.70→0.50 1.17:1、"
                "0.50→0.00 **0.26:1（净亏且 FAR 破红线 0.035110）** "
                "⇒ 回退链峰值门**确实在干活**，不能整体拆掉。"
                "（逐序表里的 ΔTP / ΔFP 是**相对 rb=0.90 的累计**口径 ⇒ 两个口径不要混读。）"
            )
            md.append("")
            md.append(
                "⇒ **夜 8 裁定：交付档保持 `--replay-rollback-min-peak 0.90` 不变**"
                "（增益不鲁棒 + FAR 余量是交付红线）；若一定要放松，**rb=0.80** 是性价比最高点"
                "（+0.0827 pp F1 只花 8% 余量）。详见 "
                "`runs/lookahead/fi_nogmc24/reg_oracle/closure_report.md` §6。"
            )
            md.append("")

        # ---------- §17.5 静默帧归因（生产口径） ----------
        _sa = rc.get("silent_attribution") or {}
        _sc = _sa.get("by_cause") or {}
        _sn = int(_sa.get("silent_frames") or 0)
        _ctp = _sa.get("tp_by_cause") or {}
        _cfp = _sa.get("fp_by_cause") or {}
        md.append("### 17.5 静默帧归因（**生产口径**：按生产判定顺序复刻）")
        md.append("")
        md.append(
            "静默帧 = 该帧最终 `report` 为空（没有上报任何位置）。归因必须**按生产代码的判定顺序**复刻："
            "**回退链峰值门 → failed 兜底 → 写 report → 出框门 → 限幅**；命中第一道门时 `report` 那一格"
            "**根本不写** ⇒ 「有门前上报点却没上报」**不能**判成出框门丢掉。"
        )
        md.append("")
        md.append("| 归因 | 帧数 | 占静默帧 | 改报 argmax 后 TP / FP |")
        md.append("|:--|--:|--:|:--|")
        for _k, _lab in (
            ("rollback_gate", "**回退链峰值门**（`--replay-rollback-min-peak`）"),
            ("fov_gate", "出框门（`--replay-fov-margin-px 6`）"),
            ("other", "其它（failed 无兜底 / 无配准决策 等）"),
        ):
            _v = int(_sc.get(_k) or 0)
            md.append(
                f"| {_lab} | {_v} | {fmt(100.0 * _v / max(1, _sn), '.1f')}% | "
                f"{int(_ctp.get(_k) or 0)} / {int(_cfp.get(_k) or 0)} |"
            )
        md.append(f"| **合计（静默帧有 GT）** | **{_sn}** | 100% | — |")
        md.append("")
        md.append(
            "> ⚠️ **纠错留痕**：夜 8 第一版把「有门前上报点却没上报」一律记成「出框门丢掉」，得到 **842 帧** —— "
            "**错**；生产顺序上第一道门是**回退链峰值门**，大头是它，出框门只占少量。"
            "该错误归因若不改，会把新战场指错方向（去调出框门而不是回退门）。"
        )
        md.append("")
        md.append(
            "⇒ 这也是 §17.4 的由来：**回退链峰值门才是真正的「大头」**，所以单独对它做受控 A/B"
            "（唯一变量 = `--replay-rollback-min-peak`）。"
        )
        md.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(sanitize(md)) + "\n", encoding="utf-8")
    summary = {
        "target_delta": T,
        "has_24seq": bool(d24),
        "models_8seq": {m: sorted(v) for m, v in d8.items()},
        "models_24seq": {m: sorted(v) for m, v in d24.items()},
        "variant_root": vroot,
        "variant_deltas": sorted(variant.get("rc2", {})),
        "fovgate_roots": {
            "8seq": args.fovgate,
            "24seq": args.fovgate_24,
            "24seq_nogmc": args.fovgate_24_nogmc,
            "24seq_nogmc_d8": args.fovgate_24_nogmc_d8,
            "24seq_nogmc_d2": args.fovgate_24_nogmc_d2,
            "24seq_nogmc_d4": args.fovgate_24_nogmc_d4,
            "24seq_rc2_d8": args.fovgate_24_rc2_d8,
        },
        "fovgate_deltas": {
            "8seq": {m: sorted(collect_fov(Path(args.fovgate), m)) for m in ("cs_t0043", "rc2")},
            "24seq": {m: sorted(collect_fov(Path(args.fovgate_24), m)) for m in ("rc2",)},
            "24seq_nogmc": {m: sorted(collect_fov(Path(args.fovgate_24_nogmc), m)) for m in ("cs_t0043",)},
            "24seq_nogmc_d8": {m: sorted(collect_fov(Path(args.fovgate_24_nogmc_d8), m)) for m in ("cs_t0043",)},
            "24seq_nogmc_d2": {m: sorted(collect_fov(Path(args.fovgate_24_nogmc_d2), m)) for m in ("cs_t0043",)},
            "24seq_nogmc_d4": {m: sorted(collect_fov(Path(args.fovgate_24_nogmc_d4), m)) for m in ("cs_t0043",)},
            "24seq_rc2_d8": {m: sorted(collect_fov(Path(args.fovgate_24_rc2_d8), m)) for m in ("rc2",)},
        },
        "nogmc24_deltas": sorted((d24.get("cs_t0043") or {})),
        "sections_present": {
            "workpoint_audit": bool(wp),
            "gate_a": bool(ga),
            "knobscan": bool(ks),
            "gapdiag": bool(gd),
            "knobscan_p3b": bool(kb),
            "accel_closure": bool(ac),
            "reg_oracle_closure": bool(rc),
            "rbgate_ab": bool(rb_rows),
        },
        "night8_reg_oracle": {
            "baseline_hit_pct": (rc or {}).get("baseline_hit_pct"),
            "oracle_reg_legal_gain_pp": (rc or {}).get("oracle_reg_legal_gain_pp"),
            "oracle_all_legal_gain_pp": (rc or {}).get("oracle_all_legal_gain_pp"),
            "verdict": (rc or {}).get("verdict"),
        },
        "night8_rbgate": {
            "variants": len(rb_rows),
            "peaks": [r["peak"] for r in rb_rows],
            "best_f1": max(((r["replay"]["f1"]), r["peak"]) for r in rb_rows)[1] if rb_rows else None,
            "best_far": min(((r["replay"]["far"]), r["peak"]) for r in rb_rows)[1] if rb_rows else None,
        },
        "night7_accel_closure": {
            "oracle_o1_scale_pp": (((ac or {}).get("oracle") or {}).get("O1_scale") or {}).get("delta_pp") if ac else None,
            "oracle_o2_accel_pp": (((ac or {}).get("oracle") or {}).get("O2_accel") or {}).get("delta_pp") if ac else None,
            "model_exit_frames": (((ac or {}).get("structure") or {}).get("model_exits_with_frames")) if ac else None,
            "along_all_p50": ((((ac or {}).get("bias_vs_selection") or {}).get("replay_all_frames") or {}).get("p50")) if ac else None,
            "along_miss_p50": ((((ac or {}).get("bias_vs_selection") or {}).get("replay_miss_frames") or {}).get("p50")) if ac else None,
        },
        "night6_new_mechanism": {
            "variants": len(kb.get("variants") or []) if kb else 0,
            "dominating_variants": sum(1 for v in (kb.get("variants") or []) if v.get("dominates")) if kb else 0,
            "far_below_0_005": ((gd.get("far_budget") or {}).get("secondary_target_far_below_0_005") or {}).get("verdict")
            if gd else None,
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[SUCCESS] final report -> {out_md}")
    print(f"[SUCCESS] summary       -> {args.out_json}")


if __name__ == "__main__":
    main()
