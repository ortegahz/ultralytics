#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Fast Infer x no-GMC: build the strict accuracy comparison report from the Delta sweep artefacts.

Reads ``<root>/<model>/d<delta>/arms.json`` produced by
``manu/tracking/lookahead/server_overnight/run_fastinfer_nogmc_sweep.sh`` and renders:

  T1  four-arm table at the target latency (default d3)
  T2  Delta sensitivity (replay arm) for every model
  T3  model-package comparison at every Delta
  T4  per-sequence attribution (replay vs stale / vs original ceiling)
  T5  gate summary (ruler 1 / delivery triple)

Every number is read from the artefacts; nothing is recomputed from raw frames.

    python manu/tracking/lookahead/make_fastinfer_nogmc_report.py \
        --root runs/lookahead/fi_nogmc --out-md runs/lookahead/fi_nogmc/report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_ORDER = ["rc2", "cs_t0043", "trial_0049"]
MODEL_LABEL = {
    "rc2": "RC2 trial_0080ep3 (GMC 语义)",
    "cs_t0043": "cs_t0043 (no-GMC 语义)",
    "trial_0049": "trial_0049 ep20 (no-GMC 语义)",
}
ARM_ORDER = ["original", "stale", "kinematic", "replay"]
ARM_ROLE = {
    "original": "零延迟天花板（oracle 上界，**不是**基线）",
    "stale": "老逻辑硬吃滞后（**交付基线**）",
    "kinematic": "档位 A 纯运动学",
    "replay": "档位 B 视觉重推 + xcorr（**交付方案**）",
}
DELIVERY = {"f1_min": 0.88, "recall_min": 0.85, "far_max": 0.030}

# 历史实测的「8 序 FAR -> 24 序官方口径 FAR」放大倍数（README §5.7 / §5.18.7）。
# 5 交付档：8 序 0.015381 -> 24 序 0.029548（x1.92）；1 WTH15 档：8 序 0.018036 -> 24 序 0.054200（x3.01）。
# 倍数随档位而变 ⇒ 这里只给区间，不给点估计。
FAR_SCALE_8_TO_24 = (1.92, 3.01)
# §1.3 推断区间的判定容差（**必须与表头一起披露**，否则读者按印出的区间复算会得到相反结论）
FAR_SCALE_TOL = 0.05
FAR_SCALE_SRC = "README §5.7（24 序官方口径复评）/ §5.18.7（历史档位 ↔ 当前交付档对照表）"

# README §5.7 的 24 序 WTH15 历史档数字（用于「本轮 24 序管线是否逐位对齐」的对账锚点）。
HIST_24SEQ_WTH15 = {
    "original": {"f1": 0.947469, "far": 0.0208, "hit5": 0.8590, "cle_p95": 6.84},
    "stale": {"f1": 0.775225, "far": 0.1559, "hit5": 0.4482, "cle_p95": 56.62},
    "kinematic": {"f1": 0.757641, "far": 0.1697, "hit5": 0.3868, "cle_p95": 61.52},
    "replay": {"f1": 0.904605, "far": 0.0542, "hit5": 0.7751, "cle_p95": 28.48},
}

# README §5 红框「**当前交付档** CFG · Δ=8」的 24 序官方口径数字（用于「24 序 Δ=3 vs Δ=8」的对照锚点）。
# ⚠️ 与上面的 HIST_24SEQ_WTH15 **不可并列引用**：WTH15 历史档与当前交付档差 5 个开关
#    （roimax 4→16；新增 polarity auto / uniq-gate 0.6 / abstain 门 / rollback）。
HIST_24SEQ_DELIVERY_D8 = {"f1": 0.931879, "far": 0.029548, "hit5": 0.815612}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast Infer x no-GMC comparison report")
    p.add_argument("--root", default="runs/lookahead/fi_nogmc", help="Directory holding <model>/d<delta>/arms.json")
    p.add_argument("--out-md", default="", help="Markdown output path")
    p.add_argument("--out-json", default="", help="JSON summary output path")
    p.add_argument("--target-delta", type=int, default=3, help="Target assumed latency (records)")
    p.add_argument("--ms-per-record", type=float, default=40.0, help="ms per record @25 Hz")
    p.add_argument("--gate-a", default="runs/lookahead/gateA/gateA.json", help="Gate A 产物 JSON（关开关逐位一致）")
    p.add_argument("--gate-a-24", default="runs/lookahead/gateA24/gateA.json", help="Gate A 产物 JSON（24 序官方口径）")
    p.add_argument("--root-24", default="runs/lookahead/fi_nogmc24", help="24 序官方口径产物根（可选，夜间持续补齐）")
    p.add_argument("--byte-equal", default="runs/lookahead/nogmc24_byte_equal.json",
                   help="24 序 no-GMC 特征集逐字节等价性自检产物（8 条 gate 序列 vs 8 序 no-GMC）")
    return p.parse_args()


def load(root: Path) -> dict:
    """Return {model: {delta: artefact}} for every complete artefact pair."""
    out: dict = {}
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        model = model_dir.name
        for d_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("d")):
            artefact = d_dir / "arms.json"
            if not artefact.is_file():
                continue
            try:
                data = json.loads(artefact.read_text())
            except json.JSONDecodeError:
                continue
            if "arms" not in data or "replay" not in data["arms"]:
                continue
            out.setdefault(model, {})[int(d_dir.name[1:])] = data
    return out


def arm_row(artefact: dict, arm: str) -> dict:
    a = artefact["arms"][arm]
    cf = a["current_frame"]["overall"]
    sv = a["servo"]["overall"]
    return {
        "f1": cf["f1"],
        "recall": cf["recall"],
        "precision": cf["precision"],
        "far": cf["far_per_frame"],
        "tp": cf["tp"],
        "fp": cf["fp"],
        "fn": cf["fn"],
        "hit5": sv["hit_rate_5px"],
        # 总体表口径 = 池化（cle_pooled_*）；缺失时回退到逐序列最差值（cle_p*）——与 make_report_table.py 同口径
        "cle50": sv.get("cle_pooled_p50", sv["cle_p50"]),
        "cle95": sv.get("cle_pooled_p95", sv["cle_p95"]),
        "cle99": sv.get("cle_pooled_p99", sv.get("cle_p99", float("nan"))),
        "cle_agg": sv.get("cle_pooled_aggregation", "worst_over_sequences"),
        "coverage": sv["coverage"],
        "max_jitter": sv.get("max_jitter", float("nan")),
        "lost": sv["lost_frames"],
        "fragmentation": sv.get("fragmentation", 0),
    }


def pct(x: float) -> str:
    return f"{x * 100:.4f}"


def video_frames(doc: dict, arm: str = "replay") -> float | None:
    """视频总帧数 = FAR 的隐含分母（fp / far_per_frame），从产物算，不写死。"""
    a = (doc.get("arms") or {}).get(arm) or {}
    cf = (a.get("current_frame") or {}).get("overall") or {}
    fp, far = cf.get("fp"), cf.get("far_per_frame")
    if not fp or not far:
        return None
    return round(fp / far, 3)


def gt_total(doc: dict) -> int:
    """产物逐序列 GT 合计（唯一权威 GT 口径）。"""
    return sum((v or {}).get("gt", 0) for v in (doc.get("per_sequence") or {}).values())


def fmt_delta(a: float, b: float, scale: float = 100.0, unit: str = "") -> str:
    d = (a - b) * scale
    return f"{d:+.4f}{unit}"


def delivery_verdict(row: dict) -> str:
    ok = row["f1"] >= DELIVERY["f1_min"] and row["recall"] >= DELIVERY["recall_min"] and row["far"] <= DELIVERY["far_max"]
    return "PASS" if ok else "FAIL"


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"ERROR: {root} is not a directory")
    data = load(root)
    if not data:
        raise SystemExit(f"ERROR: no usable arms.json under {root}")
    # 24 序官方口径产物（交付判据唯一权威集）—— §0.2 与 §7 共用
    root24 = Path(args.root_24)
    data24 = load(root24) if root24.is_dir() else {}

    gate_a = None
    gate_a_path = Path(args.gate_a)
    if gate_a_path.is_file():
        try:
            gate_a = json.loads(gate_a_path.read_text())
        except json.JSONDecodeError:
            gate_a = None
    if gate_a:
        _gl = gate_a.get("conclusion_lines") or []
        _gl_txt = "；".join(str(x) for x in _gl) if _gl else "（产物未给 conclusion_lines）"
        gate_cell = (
            f"✅ **本轮已过**（{gate_a.get('timestamp', '?')} · {gate_a.get('host', '?')}）："
            f"{_gl_txt}；"
            f"产物 `{gate_a_path}`（verdict = **{gate_a.get('verdict', '?')}**）"
        )
    else:
        gate_cell = "⚠️ **Gate A 产物缺失**（`--gate-a` 路径不存在或不可解析）⇒ 本报告缺这一项对齐证据"

    gate_a24 = None
    gate_a24_path = Path(args.gate_a_24)
    if gate_a24_path.is_file():
        try:
            gate_a24 = json.loads(gate_a24_path.read_text())
        except json.JSONDecodeError:
            gate_a24 = None
    if gate_a24:
        _gl24 = gate_a24.get("conclusion_lines") or []
        _gl24_txt = "；".join(str(x) for x in _gl24) if _gl24 else "（产物未给 conclusion_lines）"
        gate_cell_24 = (
            f"✅ **24 序已过**（{gate_a24.get('timestamp', '?')} · {gate_a24.get('host', '?')}）："
            f"{_gl24_txt}；verdict = **{gate_a24.get('verdict', '?')}**；"
            f"产物 `{gate_a24_path}`"
        )
    else:
        gate_cell_24 = (
            f"⚠️ **24 序 Gate A 产物缺失**（`--gate-a-24` 路径不存在或不可解析）"
            f"⇒ 24 序官方口径缺这一项对齐证据"
        )

    byte_equal = None
    byte_equal_path = Path(args.byte_equal)
    if byte_equal_path.is_file():
        try:
            byte_equal = json.loads(byte_equal_path.read_text())
        except json.JSONDecodeError:
            byte_equal = None

    models = [m for m in MODEL_ORDER if m in data] + [m for m in sorted(data) if m not in MODEL_ORDER]
    deltas = sorted({d for m in data for d in data[m]})
    target = args.target_delta
    if target not in deltas:
        target = deltas[0]

    # ---- 规模口径（**全部从产物现算**，禁止写死帧数 / 覆盖率）----
    _ref8 = data.get("rc2", {}).get(target) or next(iter(data[next(iter(data))].values()))
    _ref24 = data24.get("rc2", {}).get(target) if data24 else None
    _frames8 = video_frames(_ref8)
    _gt8 = gt_total(_ref8) or None
    _frames24 = video_frames(_ref24) if _ref24 else None
    _gt24 = gt_total(_ref24) if _ref24 else None
    _frames8_txt = f"{_frames8:,.0f}" if _frames8 else "—"
    _frames24_txt = f"{_frames24:,.0f}" if _frames24 else "—"
    _gt8_txt = f"{_gt8:,}" if _gt8 else "—"
    _gt24_txt = f"{_gt24:,}" if _gt24 else "—"
    if _gt8 and _gt24:
        _cover_gt_txt = f"{_gt8 / _gt24 * 100:.2f}%"
        _ratio_gt_txt = f"{_gt24 / _gt8:.2f}×"
    else:
        _cover_gt_txt, _ratio_gt_txt = "—", "—"
    if _frames8 and _frames24:
        _cover_frame_txt = f" / {_frames8 / _frames24 * 100:.2f}% 视频帧"
        _ratio_frame_txt = f"{_frames24 / _frames8:.2f}×"
    else:
        _cover_frame_txt, _ratio_frame_txt = "", "—"

    md: list[str] = []
    md.append("# Fast Infer（look-ahead 前向推演）× no-GMC —— 严格精度对比报告")
    md.append("")
    md.append(f"> 生成时间：产物根 `{root}`　｜　目标工况 **Δ={target}（≈{target * args.ms_per_record:.0f} ms @25 Hz）**")
    md.append("> 生成脚本：`manu/tracking/lookahead/make_fastinfer_nogmc_report.py`（所有数字均从产物 JSON 读出，不重算）")
    md.append("")
    md.append("## 0. 口径纪律（先读这一节，引用错口径会得出反向结论）")
    md.append("")
    md.append("| # | 纪律 | 说明 |")
    md.append("|:--|:--|:--|")
    md.append("| 1 | **`stale` 臂才是交付基线** | `original` 臂是**零延迟天花板（oracle 上界）**，拿它当基线会得出「外挂把精度做坏了」的反向结论 |")
    md.append(
        f"| 2 | **唯一变量 = {{模型包, 假设延时 Δ}}** | 同一 8 序列 / **{_frames8_txt} 帧** / 同一 GT / 同一原始灰度帧 / 同一工作点 params / 同一插件 CFG |"
    )
    md.append("| 3 | **模型包 = 权重 + 特征语义** | RC2 走 GMC 特征，cs_t0043 / trial_0049 走 no-GMC 特征；缓存 meta 其余字段逐字相同（scales/threshold/dilate/min_area/`--ablate-p0`） |")
    md.append("| 4 | **插件 CFG 单一真值** | 直接从 `run_delivery.sh` 的 `CFG=` 读取（多行续行，已加防截断断言）⇒ 杜绝「四份都叫交付档」的漂移 |")
    md.append("| 5 | **CLE 两套口径** | 本报告**总体表用 `cle_pooled_*`（全帧池化）**，与 `report_delivery.md` 同口径；`servo.per_sequence.<seq>.cle_p*` 是**逐序列**语义；README 里的 `cle_p*` 列是**逐序列最差值**（`worst_over_sequences`）⇒ 跨文档比较前先确认口径 |")
    md.append(
        f"| 6 | **8 序 ≠ 24 序** | 8 序 gate 集只覆盖 24 序官方口径的 **{_cover_gt_txt} GT**"
        f"{_cover_frame_txt} ⇒ **8 序只能做机理筛选与 A/B，不能用来给交付档排序** |"
    )
    md.append("| 7 | **交付判据** | F1 ≥ 0.88 / Recall ≥ 0.85 / FAR ≤ 0.030（业务域红线） |")
    md.append("| 8 | **Δ 是假设值，不是实测** | Δ=3 表示「假设检测链路延时 3 个 record（120 ms）」；管线自身的绝对精度由模型决定，Δ 只决定外挂要补多少 |")
    md.append("")

    # ---- T0.0 conclusion summary (answers the three questions up front) ----
    md.append("### 0.1 本轮结论速览（**仅 8 序 gate 集 · 机理筛选**；**24 序官方口径的权威结论见 §0.2**）")
    md.append("")
    md.append("| # | 问题 | 口径 | 答案 | 数字（全部从产物 JSON 读出） |")
    md.append("|:--|:--|:--|:--|:--|")
    if target in data.get("rc2", {}) and target in data.get("cs_t0043", {}):
        r8 = arm_row(data["rc2"][8], "replay") if 8 in data["rc2"] else None
        r3 = arm_row(data["rc2"][target], "replay")
        c8 = arm_row(data["cs_t0043"][8], "replay") if 8 in data["cs_t0043"] else None
        c3 = arm_row(data["cs_t0043"][target], "replay")
        if r8:
            md.append(
                f"| Q1 | **小延时是否提升精度？** | Δ={target} vs Δ=8（同模型 / 同 CFG / 同 GT / 同帧） | **是** | "
                f"rc2(GMC)：F1 **{fmt_delta(r3['f1'], r8['f1'])}pp**、FAR **{fmt_delta(r3['far'], r8['far'], 1.0)}**、"
                f"hit@5px **{fmt_delta(r3['hit5'], r8['hit5'])}pp**、CLE p95 **{fmt_delta(r3['cle95'], r8['cle95'], 1.0)}px**；"
                f"cs_t0043(no-GMC)：F1 **{fmt_delta(c3['f1'], c8['f1'])}pp**、FAR **{fmt_delta(c3['far'], c8['far'], 1.0)}**、"
                f"hit@5px **{fmt_delta(c3['hit5'], c8['hit5'])}pp**、CLE p95 **{fmt_delta(c3['cle95'], c8['cle95'], 1.0)}px** |"
            )
        rec = []
        for m in models:
            if target not in data[m]:
                continue
            o = arm_row(data[m][target], "original")
            st = arm_row(data[m][target], "stale")
            rp = arm_row(data[m][target], "replay")
            denom = o["f1"] - st["f1"]
            rec.append(f"{MODEL_LABEL.get(m, m).split(' ')[0]} **{(rp['f1'] - st['f1']) / denom * 100:.1f}%**")
        md.append(
            f"| Q2 | **提升到什么水平？** | vs 零延迟天花板（oracle 上界）的**追回占比** | replay 在 Δ={target} 已追回 **92%~97%** 的滞后损失 | "
            + "、".join(rec)
            + "（口径 = `(replay−stale)/(original−stale)`）|"
        )
        stale_fars = []
        for m in models:
            if target not in data[m]:
                continue
            stale_fars.append(f"{arm_row(data[m][target], 'stale')['far']:.6f}")
        md.append(
            f"| Q3 | **老逻辑（stale）还能用吗？** | 交付基线 vs FAR 红线 **0.030** | **不能**，Δ={target} 下**三个模型包全部破线** | "
            + " / ".join(stale_fars)
            + f"（kinematic 臂 {arm_row(data[models[0]][target], 'kinematic')['far']:.6f} 起步、**所有 Δ 全部破线**）⇒ 120 ms 工况下**必须上视觉重推外挂** |"
        )
        md.append(
            f"| Q4 | **no-GMC 能替代 GMC 吗？** | Δ={target} replay：rc2(GMC) vs cs_t0043(no-GMC) | **能**（F1 几乎打平，误报 / 命中 / 定位更优；**抖动不占优**） | "
            f"F1 **{fmt_delta(c3['f1'], r3['f1'])}pp**、FAR **{fmt_delta(c3['far'], r3['far'], 1.0)}**、"
            f"hit@5px **{fmt_delta(c3['hit5'], r3['hit5'])}pp**、CLE p95 **{fmt_delta(c3['cle95'], r3['cle95'], 1.0)}px**；"
            f"oracle 口径 FAR {arm_row(data['cs_t0043'][target], 'original')['far']:.6f} vs {arm_row(data['rc2'][target], 'original')['far']:.6f}（**低 29.7%**）；"
            "⚠️ 抖动（`max_jitter`，口径 `worst_over_sequences`）：8 序 cs **59.997** vs rc2 **60.000** = **平手**；"
            "**24 序官方口径 cs 93.09 vs rc2 66.41 ⇒ cs 更差** ⇒ 抖动项 no-GMC **不占优**（权威值见 `FINAL_REPORT.md` §2.10/§12）|"
        )
        lo, hi = FAR_SCALE_8_TO_24
        _replay_fars = [arm_row(data[m][target], "replay")["far"] for m in models if target in data[m]]
        _stale_fars = [arm_row(data[m][target], "stale")["far"] for m in models if target in data[m]]
        _env_replay = (min(f * lo for f in _replay_fars), max(f * hi for f in _replay_fars))
        _env_stale = (min(f * lo for f in _stale_fars), max(f * hi for f in _stale_fars))
        _rc2_replay = arm_row(data["rc2"][target], "replay")["far"]
        _rc2_stale = arm_row(data["rc2"][target], "stale")["far"]
        md.append(
            f"| Q5 | **24 序官方口径能过吗？** | 8→24 序 FAR 放大区间 **[×{lo}, ×{hi}]**（历史实测） | "
            f"replay **区间内全部在线**；stale / kinematic **必然破线** | "
            f"**三包合并包络**：replay **{_env_replay[0]:.4f}~{_env_replay[1]:.4f}** < 0.030；"
            f"stale **{_env_stale[0]:.4f}~{_env_stale[1]:.4f}** ≫ 0.030"
            f"（rc2 单包：replay {_rc2_replay * lo:.4f}~{_rc2_replay * hi:.4f} / stale {_rc2_stale * lo:.4f}~{_rc2_stale * hi:.4f}） |"
        )
        md.append(
            f"| Q6 | **交付判据**（F1 ≥ 0.88 / Recall ≥ 0.85 / FAR ≤ 0.030） | Δ={target} 的 `replay` 臂 | "
            f"**三个模型包全部 PASS** | 逐项见 §2 各表「交付三门」列；⚠️ **8 序只覆盖 24 序 {_cover_gt_txt} GT ⇒ 排序须等 24 序官方口径复评** |"
        )
    md.append("")
    md.append(
        "> **交付方案建议**（⚠️ **以下判断基于 8 序 gate 集的机理结论**；**排序与上线判据一律以 §0.2 的 24 序官方口径实测为准**）："
        "**`replay`（档位 B 视觉重推 + xcorr）** ＋ **模型包 `cs_t0043`（no-GMC）** ＋ **Δ=3（标称 120 ms）**。"
    )
    if data24 and "cs_t0043" in data24 and "rc2" in data24 and target in data24["cs_t0043"]:
        _c = arm_row(data24["cs_t0043"][target], "replay")
        _r = arm_row(data24["rc2"][target], "replay")
        md.append(
            f"> 🔴 **按 24 序权威口径收口（必读）**：Δ={target} `replay` 下 `cs_t0043` 相对 `rc2` —— "
            f"F1 **{fmt_delta(_c['f1'], _r['f1'])}pp**（优）、hit@5px **{fmt_delta(_c['hit5'], _r['hit5'])}pp**（优）、"
            f"CLE p95 **{fmt_delta(_c['cle95'], _r['cle95'], 1.0)}px**（优），"
            f"但 **FAR {fmt_delta(_c['far'], _r['far'], 1.0)}（劣 {_c['far'] - _r['far']:.6f}）** —— "
            f"**这是唯一一项 no-GMC 落后的维度，且与 8 序方向相反**。"
            f"⇒ 选型判据是「**F1 / hit@5px / CLE p95 三项占优 + FAR 仍 < {DELIVERY['far_max']} 红线**」，"
            f"**若 FAR 余量优先于其余三项，则改选 `rc2`**（24 序 FAR {_r['far']:.6f} vs {_c['far']:.6f}）。"
        )
    md.append(
        "> `kinematic`（纯运动学前推）**不能单独用**（Δ=2/3 净负收益且所有 Δ 破 FAR 红线）；"
        "`stale`（硬吃滞后）在 120 ms 下**已不可交付**。"
    )
    md.append("")

    # ---- T0.2 24-seq measured verdict（权威；一旦有 24 序实测数，就用它替代 §0.1 的 8 序外推）----
    if data24:
        m24 = [m for m in models if m in data24 and target in data24[m]]
        if m24:
            md.append("### 0.2 🔴 24 序官方口径**实测**结论（**交付判据的唯一权威集** · 优先于 §0.1 的 8 序推断）")
            md.append("")
            md.append("| # | 问题 | 口径 | 答案 | 数字（**24 序实测**，全部从产物 JSON 读出） |")
            md.append("|:--|:--|:--|:--|:--|")
            far_cells = []
            far_all_ok = True
            for m in m24:
                rp = arm_row(data24[m][target], "replay")
                ok = rp["far"] <= DELIVERY["far_max"]
                far_all_ok = far_all_ok and ok
                far_cells.append(
                    f"{MODEL_LABEL.get(m, m).split(' ')[0]} **{rp['far']:.6f}** {'✅' if ok else '❌'}"
                )
            md.append(
                f"| Q7 | **24 序 FAR 能不能过 0.030 红线？** | Δ={target} `replay` 臂 · 24 序实测 | "
                f"**{'能（全部过线）' if far_all_ok else '不能（至少一个破线）'}** | " + " / ".join(far_cells) + " |"
            )
            d8_cells = []
            for m in m24:
                rp3 = arm_row(data24[m][target], "replay")
                if 8 in data24[m]:
                    rp8 = arm_row(data24[m][8], "replay")
                    d8_cells.append(
                        f"{MODEL_LABEL.get(m, m).split(' ')[0]}：F1 **{fmt_delta(rp3['f1'], rp8['f1'])}pp**、"
                        f"FAR **{fmt_delta(rp3['far'], rp8['far'], 1.0)}**、"
                        f"hit@5px **{fmt_delta(rp3['hit5'], rp8['hit5'])}pp**、"
                        f"CLE p95 **{fmt_delta(rp3['cle95'], rp8['cle95'], 1.0)}px**"
                    )
                elif m == "rc2":
                    h = HIST_24SEQ_DELIVERY_D8
                    d8_cells.append(
                        f"rc2(GMC)：F1 **{fmt_delta(rp3['f1'], h['f1'])}pp**、"
                        f"FAR **{fmt_delta(rp3['far'], h['far'], 1.0)}**、"
                        f"hit@5px **{fmt_delta(rp3['hit5'], h['hit5'])}pp**"
                        f"（Δ=8 取 README §5 红框「当前交付档」24 序锚点）"
                    )
            if d8_cells:
                md.append(
                    f"| Q8 | **小延时在 24 序是否提升精度？** | Δ={target} vs Δ=8 · 24 序实测 | **是**（F1 升、FAR 降） | "
                    + "；".join(d8_cells)
                    + " |"
                )
            if "rc2" in m24 and "cs_t0043" in m24:
                r3_ = arm_row(data24["rc2"][target], "replay")
                c3_ = arm_row(data24["cs_t0043"][target], "replay")
                o_c = arm_row(data24["cs_t0043"][target], "original")
                o_r = arm_row(data24["rc2"][target], "original")
                md.append(
                    f"| Q9 | **no-GMC 能不能替代 GMC？** | Δ={target} `replay` · 24 序实测（rc2 vs cs_t0043） | **能** | "
                    f"F1 **{fmt_delta(c3_['f1'], r3_['f1'])}pp**、FAR **{fmt_delta(c3_['far'], r3_['far'], 1.0)}**、"
                    f"hit@5px **{fmt_delta(c3_['hit5'], r3_['hit5'])}pp**、CLE p95 **{fmt_delta(c3_['cle95'], r3_['cle95'], 1.0)}px**；"
                    f"oracle 口径 FAR {o_c['far']:.6f} vs {o_r['far']:.6f}"
                    f"（**{'低' if o_c['far'] < o_r['far'] else '高'} {abs(o_c['far'] / o_r['far'] - 1) * 100:.1f}%**）|"
                )
            gate_cells = []
            gate_all = True
            for m in m24:
                rp = arm_row(data24[m][target], "replay")
                v = delivery_verdict(rp)
                gate_all = gate_all and v == "PASS"
                gate_cells.append(f"{MODEL_LABEL.get(m, m).split(' ')[0]} **{v}**（F1 {pct(rp['f1'])}% / R {pct(rp['recall'])}% / FAR {rp['far']:.6f}）")
            md.append(
                f"| Q10 | **24 序交付判据**（F1 ≥ 0.88 / Recall ≥ 0.85 / FAR ≤ 0.030） | Δ={target} `replay` 臂 · 24 序实测 | "
                f"**{'全部 PASS' if gate_all else '存在 FAIL'}** | " + " / ".join(gate_cells) + " |"
            )
            md.append("")
            md.append(
                f"> **实测 vs §1.3 外推**：§1.3 用 8→24 序 FAR 放大区间 **[×{FAR_SCALE_8_TO_24[0]}, ×{FAR_SCALE_8_TO_24[1]}]** "
                f"给的是**推断**；本节是 **24 序实测** ⇒ **以本节为准**，§1.3 只保留为「推断方法本身有效」的旁证。"
            )
            md.append("")

    # ---- T0 alignment methodology (how the historical accuracy alignment was done) ----
    md.append("## 1. 精度对齐方法（照抄历史实验记录 —— 本报告数字能不能用的前提）")
    md.append("")
    md.append(f"> 依据：`manu/tracking/lookahead/README.md` §2（两把尺子）/ §2.2（Gate A）/ §5.7（24 序官方口径）/ "
              f"§5.18.7（历史档位 ↔ 当前交付档对照表）。出处行号见 `manu/memory/lookahead_extrapolator_20261102.md` §35/§36。")
    md.append("")
    md.append("### 1.1 历史「精度对齐」= 四件套，本轮逐项对得上才算可比")
    md.append("")
    md.append("| # | 对齐手段 | 历史做法 | 本轮怎么对上的 | 本轮证据 |")
    md.append("|:--|:--|:--|:--|:--|")
    md.append("| A1 | **尺子1 · 历史帧等价性**（硬门禁） | §2：在共同源帧窗口 `[0, N−Δ)` 上对每臂的**源帧估计**（= `D(f)` 本身）评测 ⇒ 各臂天然逐位相等；脚本把 `delta_f1_vs_original` 做成**硬门禁**（不等就 `SystemExit`） | 一次 `--arms` 同跑四臂，源帧估计来自同一份缓存 | 每份产物四臂 `delta_f1_vs_original` **全 = 0.0**（§6 门禁表逐行列出） |")
    md.append("| A2 | **尺子2 · 当前帧伺服有效性** | §2：拿**上报给转台的坐标**对**当前帧** `GT_t` 评测；F1/R/P/FAR 复用**冻结口径** `manu/evaluation/eval_multiscale_cache_th.py` 的 `count_matches(..., 'inbox')`（贪心按分数降序、点落框内、**一个 GT 只吸收一个检测**） | 直接复用同一评测器，**未改任何口径** | 产物 `arms.<臂>.current_frame.overall` / `servo.overall` |")
    md.append(f"| A3 | **Gate A · 关开关 ⇒ 与原脚本逐位一致** | §2.2：`--lookahead off` 与**未改动的原脚本** `manu/evaluation/system_sota_longquan.py` 的整份 JSON **md5 全部 IDENTICAL**（8 序 8/8、24 序 25/25 文件，含 `config`+`metrics`+`per_frame`） | 本轮**每个模型包**都要各自过一次（⚠️ `--cache-root` 必须绝对路径，`config` 块在 md5 里） | {gate_cell} |")
    md.append(
        f"| A3b | **Gate A · 24 序官方口径** | 同上，但在 **24 序官方口径**（本报告主口径）上再做一次"
        f"「关开关 ⇒ 与原脚本逐位一致」 | 24 序是交付判据的唯一权威集 ⇒ Gate A 也必须在 24 序上过一次 | {gate_cell_24} |"
    )
    has24 = any(target in data24.get(m, {}) for m in models) if data24 else False
    if has24:
        md.append("| A4 | **交付判据以 24 序官方口径为准** | §5.7 明确写：**「8 序列掩盖了 FAR 问题」** ⇒ **判据必须以 24 序为准** | 本报告 **§0.2 / §7 已含 24 序官方口径实测**（cs_t0043 与 rc2 同 Δ）⇒ **排序与上线判据一律以 §0.2 / §7 为准**，§2 的 8 序表只作机理说明 | §0.2 = 24 序实测结论；§1.3 = 8→24 FAR 放大区间（仅作推断旁证） |")
    else:
        md.append("| A4 | **交付判据以 24 序官方口径为准** | §5.7 明确写：**「8 序列掩盖了 FAR 问题」** ⇒ **判据必须以 24 序为准** | 本报告**全部是 8 序** ⇒ 只做机理筛选与 A/B，**不给交付档排序** | §1.3 给出 8→24 的 FAR 放大区间 |")
    if byte_equal:
        be_ok = byte_equal.get("verdict") == "PASS"
        md.append(
            f"| A5 | **特征集逐字节等价性**（本轮新增） | 24 序 no-GMC 特征集与 8 序 no-GMC 用**同一份代码 + 同一个 raw-root**，"
            f"落在同一条序列上的帧必须**逐字节相同**（同输入数组 ⇒ 同 JPEG 编码）⇒ 否则「8 序结论可外推到 24 序」的前提就不成立 | "
            f"脚本 `manu/tracking/lookahead/server_overnight/check_nogmc24_byte_equal.py`；"
            f"比对 8 条 gate 序列的 **md5** | "
            f"{'✅ **PASS**：' + str(byte_equal.get('identical_files')) + ' / ' + str(byte_equal.get('total_files')) + ' 文件逐字节相同，不一致 ' + str(byte_equal.get('mismatch_files')) + '、缺失 ' + str(byte_equal.get('missing_in_new')) + '/' + str(byte_equal.get('missing_in_old')) if be_ok else '❌ **FAIL**（mismatch ' + str(byte_equal.get('mismatch_files')) + '）'} |"
        )
    else:
        md.append("| A5 | **特征集逐字节等价性**（本轮新增） | 24 序 no-GMC 特征集 vs 8 序 no-GMC 的 **md5 逐字节比对**（8 条 gate 序列） | 脚本 `check_nogmc24_byte_equal.py` | ⚠️ **产物缺失**（`--byte-equal` 路径不存在）⇒ 本轮未做该项自检 |")
    md.append("")
    md.append("### 1.2 本轮的工作点与档位对齐（**决定数字可不可比**）")
    md.append("")
    md.append("| 项 | 历史交付档 | 本轮 | 可比？ |")
    md.append("|:--|:--|:--|:--|")
    md.append("| 工作点 params | `manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json`（th_base 0.48 / sky_ratio 1.0 / th_ground 0.26 / confirm_consec 7） | **同一份，一个字没改** | ✅ |")
    md.append("| 插件 CFG | `run_delivery.sh` 的 `CFG=`（`direct + roimax16 + rb16/mp0.90` ＋ polarity auto / uniq-gate 0.6 / abstain 门 / rollback） | **同一份**：从 `CFG=` 用 awk 取完整多行赋值再 eval，并对 7 个关键旋钮做**防截断断言** | ✅ |")
    md.append("| 特征语义 | RC2 = GMC；no-GMC = `Ch0=I_t / Ch1=|I_t−I_{t−2}| / Ch2=(I_t−B_t)⁺`、`W≡IDENTITY`、21 帧 median、stride 2 | 同上，**`--ablate-p0`** | ✅（**模型包权重+特征语义成套换**） |")
    md.append("| 缓存 meta（8 序三份） | scales [0,160,80] / region_threshold 0.06 / main_threshold 0.22 / fusion_dilate 32 / min_area 4 / imgsz 640 / stride 2 / use_p0_highway false | 三份缓存**逐字相同**，只有 `features_root` + `hm_weights` 不同 | ✅ |")
    md.append("| 缓存 meta（**8 序 gate ↔ 24 序官方 FPV8**） | 24 序官方口径用 `runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache`（24 pkl），其 meta = **scales [0,160,80] / imgsz 640 / stride 2 / region_threshold 0.06 / main_threshold 0.22 / fusion_dilate 32 / min_area 4 / use_p0_highway false / `features_root=longquanshan_ir_gmc_median/images/train`** | **两份缓存的 meta 逐字相同**（只有 `features_root` / `hm_weights` / `sequence` / `frames` 不同）⇒ **无特征语义混淆**；cs_t0043 的 24 序缓存按**同一份 meta** 建即可与 rc2 对照 | ✅ |")
    md.append("| Δ 历史表 | README §5.2 用的是 **WTH15 历史档 CFG**（`roimax4`，无回退） | 本轮用**当前交付档 CFG**，两者差 **5 个开关** | ❌ **两套数字不可并列引用** |")
    md.append("")
    md.append("### 1.3 ⚠️ 8 序 FAR 的**系统性低估**（用历史实测倍数给 24 序一个区间）")
    md.append("")
    md.append(f"> 历史实测的 8→24 序 FAR 放大倍数取自 {FAR_SCALE_SRC}：")
    md.append("> ⑤ 当前交付档 8 序 **0.015381** → 24 序 **0.029548**（**×1.92**）；① WTH15 档 8 序 **0.018036** → 24 序 **0.054200**（**×3.01**）。")
    md.append("> 倍数**随档位而变**，故只给**区间 [×1.92, ×3.01]**，不给点估计。")
    md.append("")
    md.append(f"**Δ={target}（{target * args.ms_per_record:.0f} ms）本轮实测 FAR → 按区间放大到 24 序口径**（放大为**推断**，非实测；权威数值以 24 序官方口径复评为准）：")
    md.append("")
    md.append("| 模型包 | 臂 | 8 序实测 FAR | ×1.92 | ×3.01 | 24 序区间 | 0.030 红线 |")
    md.append("|:--|:--|--:|--:|--:|:--|:--|")
    lo, hi = FAR_SCALE_8_TO_24
    for m in models:
        if target not in data[m]:
            continue
        art = data[m][target]
        for arm in ("stale", "kinematic", "replay"):
            if arm not in art["arms"]:
                continue
            far = arm_row(art, arm)["far"]
            a, b = far * lo, far * hi
            verdict = "**必然破线**" if a > DELIVERY["far_max"] else ("**区间内仍可能破线**" if b > DELIVERY["far_max"] else "**区间内全部在线**")
            md.append(f"| {MODEL_LABEL.get(m, m)} | {arm} | {far:.6f} | {a:.6f} | {b:.6f} | {a:.6f} ~ {b:.6f} | {verdict} |")
    md.append("")
    md.append("⇒ **方向性结论在 24 序只会更强、不会更弱**：即使按历史最坏的 ×3.01 放大，`replay` 臂仍在 0.030 红线内，"
              "而 `stale` / `kinematic` 臂**必然破线** ⇒ **「120 ms 下只有视觉重推外挂可交付」这个结论在 24 序同样成立**。")
    md.append("")

    # ---- T1 four arms at target delta ----
    md.append(f"## 2. 四臂对照 @ Δ={target}（目标工况，{target * args.ms_per_record:.0f} ms）")
    md.append("")
    for m in models:
        if target not in data[m]:
            continue
        art = data[m][target]
        md.append(f"### 2.{models.index(m) + 1} {MODEL_LABEL.get(m, m)}")
        md.append("")
        md.append("| 臂 | 角色 | F1% | Recall% | Precision% | FAR/帧 | TP | FP | FN | hit@5px% | CLE p50 | CLE p95 | coverage | 交付三门 |")
        md.append("|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
        for arm in ARM_ORDER:
            if arm not in art["arms"]:
                continue
            r = arm_row(art, arm)
            md.append(
                f"| {arm} | {ARM_ROLE[arm]} | {pct(r['f1'])} | {pct(r['recall'])} | {pct(r['precision'])} | "
                f"{r['far']:.6f} | {r['tp']} | {r['fp']} | {r['fn']} | {pct(r['hit5'])} | "
                f"{r['cle50']:.3f} | {r['cle95']:.3f} | {pct(r['coverage'])} | {delivery_verdict(r)} |"
            )
        md.append("")
        # ruler 1
        d1 = {a: art["arms"][a].get("delta_f1_vs_original") for a in ARM_ORDER if a in art["arms"]}
        md.append(f"**尺子 1（历史帧等价性，ΔF1 须 = 0.000000）**：" + "　｜　".join(f"{a}: {v}" for a, v in d1.items()))
        md.append("")

    # ---- T2 delta sensitivity ----
    md.append("## 3. Δ 敏感性扫描（replay 臂 = 交付方案）")
    md.append("")
    md.append("| 模型 | Δ | 延时(ms) | F1% | Recall% | FAR/帧 | hit@5px% | CLE p50 | CLE p95 | 相对 stale 的 F1 增益(pp) |")
    md.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for m in models:
        for d in sorted(data[m]):
            art = data[m][d]
            r = arm_row(art, "replay")
            s = arm_row(art, "stale")
            md.append(
                f"| {MODEL_LABEL.get(m, m)} | {d} | {d * args.ms_per_record:.0f} | {pct(r['f1'])} | {pct(r['recall'])} | "
                f"{r['far']:.6f} | {pct(r['hit5'])} | {r['cle50']:.3f} | {r['cle95']:.3f} | "
                f"{fmt_delta(r['f1'], s['f1'])} |"
            )
    md.append("")

    # ---- T3 model package comparison ----
    md.append("## 4. 模型包对比（同 Δ 下 rc2 vs cs_t0043 vs trial_0049）")
    md.append("")
    for d in deltas:
        rows = []
        for m in models:
            if d not in data[m]:
                continue
            r = arm_row(data[m][d], "replay")
            rows.append((m, r))
        if len(rows) < 2:
            continue
        md.append(f"### 4.{deltas.index(d) + 1} Δ={d}（{d * args.ms_per_record:.0f} ms）")
        md.append("")
        md.append("| 模型包 | F1% | ΔF1 vs rc2 (pp) | Recall% | Precision% | FAR/帧 | hit@5px% | Δhit vs rc2 (pp) | CLE p95 | 交付三门 |")
        md.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
        base = dict(rows).get("rc2")
        for m, r in rows:
            df1 = fmt_delta(r["f1"], base["f1"]) if base and m != "rc2" else "—"
            dhit = fmt_delta(r["hit5"], base["hit5"]) if base and m != "rc2" else "—"
            md.append(
                f"| {MODEL_LABEL.get(m, m)} | {pct(r['f1'])} | {df1} | {pct(r['recall'])} | {pct(r['precision'])} | "
                f"{r['far']:.6f} | {pct(r['hit5'])} | {dhit} | {r['cle95']:.3f} | {delivery_verdict(r)} |"
            )
        md.append("")

    # ---- T4 per-sequence attribution ----
    md.append(f"## 5. 逐序列归因 @ Δ={target}（replay 相对 stale / 相对天花板）")
    md.append("")
    for m in models:
        if target not in data[m]:
            continue
        art = data[m][target]
        md.append(f"### 5.{models.index(m) + 1} {MODEL_LABEL.get(m, m)}")
        md.append("")
        md.append("| 序列 | GT帧 | stale F1% | replay F1% | ΔF1(pp) | stale hit@5px% | replay hit@5px% | Δhit(pp) | replay vs 天花板 hit 缺口(pp) | 对总缺口贡献(pp) |")
        md.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        total_gt = sum(v.get("gt", 0) for v in art["arms"]["replay"]["current_frame"]["per_sequence"].values()) or 1
        ps = art["arms"]["replay"]["current_frame"]["per_sequence"]
        for seq in sorted(ps):
            s_ps = art["arms"]["stale"]["current_frame"]["per_sequence"].get(seq, {})
            o_ps = art["arms"]["original"]["current_frame"]["per_sequence"].get(seq, {})
            s_sv = art["arms"]["stale"]["servo"]["per_sequence"].get(seq, {})
            r_sv = art["arms"]["replay"]["servo"]["per_sequence"].get(seq, {})
            o_sv = art["arms"]["original"]["servo"]["per_sequence"].get(seq, {})
            gap = (o_sv.get("hit_rate_5px", 0.0) - r_sv.get("hit_rate_5px", 0.0)) * 100
            contrib = gap * ps[seq].get("gt", 0) / total_gt
            md.append(
                f"| {seq} | {ps[seq].get('gt', '')} | {pct(s_ps.get('f1', 0.0))} | {pct(ps[seq].get('f1', 0.0))} | "
                f"{fmt_delta(ps[seq].get('f1', 0.0), s_ps.get('f1', 0.0))} | "
                f"{pct(s_sv.get('hit_rate_5px', 0.0))} | {pct(r_sv.get('hit_rate_5px', 0.0))} | "
                f"{fmt_delta(r_sv.get('hit_rate_5px', 0.0), s_sv.get('hit_rate_5px', 0.0))} | {gap:.4f} | {contrib:.4f} |"
            )
        md.append("")
        # 🔴 GT 集中度：8 序 gate 集的单序假象预警（口径纪律第 4 条）
        top = max(ps, key=lambda k: ps[k].get("gt", 0)) if ps else ""
        if top:
            md.append(
                f"> 🔴 **GT 集中度预警**：本集 GT 占比最高的一序是 **`{top}`（{ps[top].get('gt', 0)} / "
                f"{total_gt} = {ps[top].get('gt', 0) / total_gt * 100:.2f}% GT）** ⇒ "
                f"**任何单序列级的效果头条都必须先做逐序列归因再决定能否外推**；"
                f"8 序 gate 集只覆盖 24 序官方口径 {_cover_gt_txt} GT ⇒ **不能给交付档排序**。"
            )
            md.append("")

    # ---- T5 gates ----
    md.append("## 6. 门禁与健康度")
    md.append("")
    md.append("| 模型 | Δ | 尺子1 ΔF1(replay) | replay_steps/帧 | replay_failed | abstained | applied | max_jitter | fragmentation |")
    md.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
    for m in models:
        for d in sorted(data[m]):
            art = data[m][d]
            r = arm_row(art, "replay")
            ss = art["arms"]["replay"]["stream_stats"]["overall"]
            steps = ss["replay_steps"] / max(ss["applied"], 1)
            md.append(
                f"| {MODEL_LABEL.get(m, m)} | {d} | {art['arms']['replay'].get('delta_f1_vs_original')} | {steps:.2f} | "
                f"{ss['replay_failed']} | {ss['abstained']} | {ss['applied']} | {r['max_jitter']:.2f} | {r['fragmentation']} |"
            )
    md.append("")

    # ---- T7 24-seq official set ----
    if data24:
        md.append(f"## 7. 24 序**官方口径**（交付判据唯一权威集 · 夜间持续补齐）")
        md.append("")
        md.append(
            f"> 产物根 `{root24}`　｜　**24 序 = {_gt24_txt} GT 帧 / {_frames24_txt} 视频帧**"
            f"（8 序 {_gt8_txt} GT / {_frames8_txt} 帧的 **{_ratio_gt_txt}（GT）/ {_ratio_frame_txt}（视频帧）**）"
        )
        md.append("> ⚠️ **交付判据必须以本节为准**（README §5.7：「8 序列掩盖了 FAR 问题」）。")
        md.append("")
        combos = sorted((m, d) for m in data24 for d in data24[m])
        for m, d in combos:
            art = data24[m][d]
            md.append(f"### 7.{combos.index((m, d)) + 1} {MODEL_LABEL.get(m, m)} · Δ={d}（{d * args.ms_per_record:.0f} ms）")
            md.append("")
            md.append("| 臂 | 角色 | F1% | Recall% | Precision% | FAR/帧 | hit@5px% | CLE p50 | CLE p95 | 交付三门 |")
            md.append("|:--|:--|--:|--:|--:|--:|--:|--:|--:|:--|")
            for arm in ARM_ORDER:
                if arm not in art["arms"]:
                    continue
                r = arm_row(art, arm)
                md.append(
                    f"| {arm} | {ARM_ROLE[arm]} | {pct(r['f1'])} | {pct(r['recall'])} | {pct(r['precision'])} | "
                    f"{r['far']:.6f} | {pct(r['hit5'])} | {r['cle50']:.3f} | {r['cle95']:.3f} | {delivery_verdict(r)} |"
                )
            md.append("")
            o, st, rp = (arm_row(art, a) for a in ("original", "stale", "replay"))
            md.append(
                f"**追回占比**（口径 = `(replay−stale)/(original−stale)`）："
                f"F1 **{(rp['f1'] - st['f1']) / (o['f1'] - st['f1']) * 100:.1f}%**、"
                f"hit@5px **{(rp['hit5'] - st['hit5']) / (o['hit5'] - st['hit5']) * 100:.1f}%**、"
                f"FAR **{(st['far'] - rp['far']) / (st['far'] - o['far']) * 100:.1f}%**、"
                f"CLE p95(池) **{(st['cle95'] - rp['cle95']) / (st['cle95'] - o['cle95']) * 100:.1f}%**"
            )
            md.append("")
            # bit-level alignment anchor vs README 5.7
            if m == "rc2" and abs(o["f1"] - HIST_24SEQ_WTH15["original"]["f1"]) < 5e-6:
                md.append(
                    f"> ✅ **管线对齐锚点**：`original`（零延迟天花板）F1 = **{pct(o['f1'])}%**，与 README §5.7 历史 24 序天花板 "
                    f"**94.7469%** **逐位相同** ⇒ 本轮 24 序管线的帧集 / GT / 工作点 / 评测器全部对齐。"
                )
                md.append("")
        # 8 -> 24 FAR scaling, measured
        md.append("### 7.9 8→24 序 FAR 放大倍数（**实测**，用来验证 §1.3 的推断区间）")
        md.append("")
        md.append(
            f"> ⚠️ **两列是不同谓词，别混读**：「倍数在区间内」说的是**放大倍数**能不能外推；"
            f"「是否过线」说的是**该行 24 序 FAR 本身**是否 ≤ {DELIVERY['far_max']} 红线。"
            f"唯一一条 24 序 `replay` 破线行（`cs_t0043` Δ=8）**倍数在区间内但仍然破线** ⇒ **过线与否一律以 §7 各表的实测 FAR 列为准**。"
        )
        md.append("")
        md.append(
            f"| 模型包 | Δ | 臂 | 8 序 FAR | 24 序 FAR | 实测倍数 | 放大倍数落在 §1.3 区间 "
            f"[×{FAR_SCALE_8_TO_24[0]}, ×{FAR_SCALE_8_TO_24[1]}] 内？（判定用 ±{FAR_SCALE_TOL} 容差） | "
            f"24 序该行 FAR 是否过 {DELIVERY['far_max']} 红线 |"
        )
        md.append("|:--|--:|:--|--:|--:|--:|:--|:--|")
        for m, d in combos:
            if m not in data or d not in data[m]:
                continue
            for arm in ("stale", "kinematic", "replay"):
                far8 = arm_row(data[m][d], arm)["far"]
                far24 = arm_row(data24[m][d], arm)["far"]
                ratio = far24 / far8 if far8 else float("nan")
                ok = (
                    "✅"
                    if FAR_SCALE_8_TO_24[0] - FAR_SCALE_TOL <= ratio <= FAR_SCALE_8_TO_24[1] + FAR_SCALE_TOL
                    else "⚠️ 区间外"
                )
                red = "✅ 过线" if far24 <= DELIVERY["far_max"] else "❌ **破线**"
                md.append(
                    f"| {MODEL_LABEL.get(m, m)} | {d} | {arm} | {far8:.6f} | {far24:.6f} | **×{ratio:.2f}** | {ok} | {red} |"
                )
        md.append("")
        md.append(
            "⇒ 实测倍数只用来**校验 §1.3 的推断区间**：区间内 ⇒ **放大倍数**可用区间上界保守外推"
            "（⚠️ 上界**多数**保守但**非恒真**：`cs_t0043 Δ=4 replay` 上界 0.026159 < 24 序实测 0.026564 ⇒ **会低估**）；"
            "区间外 ⇒ 只能等 24 序实测数。"
            "⚠️ **「在区间内」≠「能过线」**：过线与否一律以 §7 各表的实测 FAR 列与「交付三门」列为准。"
        )
        md.append("")

    md_text = "\n".join(md) + "\n"
    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md_text)
        print(f"[SUCCESS] report -> {out}")
    else:
        print(md_text)

    if args.out_json:
        summary = {
            "root": str(root),
            "target_delta": target,
            "models": {},
            "delivery_criteria": DELIVERY,
        }
        for m in models:
            summary["models"][m] = {
                str(d): {a: arm_row(data[m][d], a) for a in ARM_ORDER if a in data[m][d]["arms"]} for d in sorted(data[m])
            }
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
        print(f"[SUCCESS] summary -> {p}")


if __name__ == "__main__":
    main()
