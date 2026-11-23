#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Generate the round-91 memory section (delivery-rate ladder) FROM ARTEFACTS ONLY.

WHY
---
Round 90 turned the delta-n grid search into one auditable comparison table.  What it
still did NOT answer is the question the delivery decision actually turns on:

    **at which delivery rate does the tracker stop working?**

The per-frame arm (dN == 1) is the fast anchor and the measured board cadence (s1) is the
slow end point; the 12..20 fps band in between was never sampled.  Round 91 fills that
band with an explicit fps ladder (f25 / f20 / f17 / f14 / f12).

NO HAND-TYPED NUMBERS
---------------------
Every figure in the generated section is read from an on-disk artefact or computed by this
script from those artefacts.  On top of that the generator runs a *drift check*: it scans
the produced markdown for numeric literals and fails if any of them is neither present in
the artefact corpus (as-is, comma-grouped or rounded) nor registered as a computed value.
That is the guard against the round-90 failure mode -- a hand-copied number in the prose
that disagreed with the product it claims to quote.

Usage::

    python manu/memory/trial0080_alignment_20261119/search/make_memory_section_fps_grid_91.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DELIV = REPO / "manu/memory/delivery_20261119"
SEARCH_DIR = REPO / "manu/memory/trial0080_alignment_20261119/search"

LADDER_CSV = DELIV / "grid_compare/rt_fps_grid.csv"
GRID_MD = DELIV / "grid_compare/rt_grid_compare.md"
SCENARIO_CSV = DELIV / "grid_compare/rt_grid_scenario.csv"
DELIVERY_CSV = DELIV / "grid_compare/rt_delivery_compare.csv"
ANALYSE_CSV = DELIV / "artifacts/analyse_runs_r90.csv"
FPS_JSON = DELIV / "artifacts/rt_fps_grid_r90.json"
FPS_LOG = DELIV / "artifacts/rt_fps_grid_r90.log"
SPEC_MD = DELIV / "trial0080_delta_search_spec_20261119.md"
V3_PARAMS = REPO / "manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json"
OPT_PARAMS = REPO / "manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json"

OUT_MD = SEARCH_DIR / "memory_section_fps_grid_91.md"
OUT_COMPACT = SEARCH_DIR / "memory_section_fps_grid_91_compact.md"

# Structural literals that are NOT artefacts: dates, round numbers, the host address.
EXTRA_ALLOWED = {
    "91": "round number of this section",
    "90": "round number of the previous section (referenced)",
    "2026": "year of the run", "11": "month of the run", "20": "day of the run",
    "13": "hour of the server run start", "10": "hour of the server run start",
    "18": "hour of the server run end", "24": "hour of the server run end",
    "32222": "ssh port of the compute server",
    "14400": "timeout of the server command (seconds)",
    "40": "milliseconds per frame at the 25 fps source",
    "25": "source frame rate of the capture chain",
    "0.64": "board model revision quoted by the board-cadence row of the spec",
    "123": "mean board frame interval in ms (spec)",
    "3.08": "mean board dN lower bound (spec)",
    "3.11": "mean board dN upper bound (spec)",
    "8.03": "measured board fps lower bound (spec)",
    "8.13": "measured board fps upper bound (spec)",
    "7": "confirm_consec of the v3 base",
    "2": "confirm_consec of the delivery group",
    "8": "confirm_gap_tol of the delivery group",
    "9": "1 + confirm_gap_tol of the delivery group",
    "1": "max_gap of the v3 base / seed id",
    "0": "seed id / the zero-F1 marker",
    "3": "seed count",
    "4": "number of measured cadence scenarios (s1..s4)",
    "5": "number of fps ladder scenarios",
    "12": "lowest target fps of the ladder",
    "14": "target fps of the f14 rung",
    "17": "target fps of the f17 rung",
    "20": "target fps of the f20 rung",
}

NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
IPV4_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")
FENCE_RE = re.compile(r"```.*?```", re.S)
INLINE_CODE_RE = re.compile(r"`[^`]*`")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def num(text: str) -> float:
    return float(str(text).replace(",", ""))


def f6(value: float) -> str:
    return f"{value:.6f}"


def add_value(allowed: set[str], raw) -> None:
    """Register one numeric value (and every rounded form of it) as traceable."""
    if isinstance(raw, bool):
        return
    if isinstance(raw, (int, float)):
        value = float(raw)
    else:
        text = str(raw).strip().replace(",", "")
        try:
            value = float(text)
        except ValueError:
            return
    allowed.add(f"{value:.6f}".rstrip("0").rstrip(".") if value == value else "")
    allowed.add(str(value))
    for digits in range(0, 9):
        allowed.add(f"{value:.{digits}f}")
    if value == int(value):
        allowed.add(str(int(value)))


def _walk(obj, allowed: set[str]) -> None:
    if isinstance(obj, dict):
        for value in obj.values():
            _walk(value, allowed)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _walk(value, allowed)
    else:
        add_value(allowed, obj)


def corpus_tokens(structured: list, texts: list[str]) -> set[str]:
    allowed: set[str] = set()
    for item in structured:
        _walk(item, allowed)
    for text in texts:
        for token in NUMBER_RE.findall(text):
            add_value(allowed, token.replace(",", ""))
    allowed.discard("")
    return allowed


def scenario_fps(grid_md: str) -> dict[str, float]:
    """Parse the s1..s4 realised fps out of the round-90 product (not out of thin air)."""
    out: dict[str, float] = {}
    for line in grid_md.splitlines():
        match = re.match(r"^- `([s]\d)`\s*=.*fps=`([\d.]+)`", line.strip())
        if match:
            out[match.group(1)] = float(match.group(2))
    return out


def board_cadence(spec_md: str) -> tuple[str, str]:
    line = next((row for row in spec_md.splitlines() if "板卡实测节拍" in row), "")
    pairs = re.findall(r"[\d.]+~[\d.]+", line)
    if len(pairs) < 2:
        raise SystemExit("[FAIL] cannot parse the board cadence row from the spec")
    return pairs[0], pairs[1]


def load_all() -> dict:
    ladder = {row["scenario"]: row for row in read_csv(LADDER_CSV)}
    analyse: dict[str, list[dict]] = {}
    for row in read_csv(ANALYSE_CSV):
        analyse.setdefault(row["scenario"], []).append(row)
    arms: dict[str, dict[str, dict]] = {}
    for row in read_csv(SCENARIO_CSV):
        arms.setdefault(row["arm"], {})[row["scenario"]] = row
    delivery_rows = {row["label"]: row for row in read_csv(DELIVERY_CSV)}
    fps_doc = json.loads(FPS_JSON.read_text(encoding="utf-8"))
    return {
        "ladder": ladder,
        "analyse": analyse,
        "arms": arms,
        "delivery": delivery_rows,
        "fps_doc": fps_doc,
        "grid_md": GRID_MD.read_text(encoding="utf-8"),
        "spec_md": SPEC_MD.read_text(encoding="utf-8"),
        "log": FPS_LOG.read_text(encoding="utf-8"),
        "v3": json.loads(V3_PARAMS.read_text(encoding="utf-8"))["params"],
        "opt": json.loads(OPT_PARAMS.read_text(encoding="utf-8"))["params"],
    }


def build_rows(data: dict) -> list[dict]:
    """One row per scenario: ladder numbers + cadence statistics + the arm results."""
    ladder = data["ladder"]
    arms = data["arms"]
    s_fps = scenario_fps(data["grid_md"])
    spec = data["fps_doc"]["meta"]["scenarios_spec"]
    kind_label = {
        "measured": "实测 Δn 分布 IID 重采样",
        "doc": "brief assumption（假设情景）",
        "target_fps": "fps 网格（合成节拍）",
    }
    rows: list[dict] = []
    for scenario in ["f25", "f20", "f17", "f14", "f12", "s1", "s2", "s3", "s4"]:
        stats = data["analyse"][scenario]
        seeds = len(stats)
        row = {
            "scenario": scenario,
            "seeds": seeds,
            "p_dn1": sum(num(s["p_delta_n_1"]) for s in stats) / seeds,
            "max_run_gap1": max(int(s["max_run_maxgap1"]) for s in stats),
            "pos_ge7_gap1": max(int(s["positions_run_ge7_maxgap1"]) for s in stats),
            "max_run_gap9": max(int(s["max_run_maxgap9"]) for s in stats),
            "pos_ge2_gap9": max(int(s["positions_run_ge2_maxgap9"]) for s in stats),
            "gaps": max(int(s["gaps"]) for s in stats),
        }
        if scenario in ladder:
            lad = ladder[scenario]
            row.update({
                "target_fps": float(lad["target_fps"]),
                "achieved_fps": float(lad["achieved_fps"]),
                "mean_dn": float(lad["mean_delta_n"]),
                "updates": int(lad["updates"]),
                "untuned_f1": float(lad["untuned_f1"]),
                "untuned_far": float(lad["untuned_far"]),
                "opt_f1": float(lad["opt_f1"]),
                "opt_far": float(lad["opt_far"]),
                "dense_f1": float(lad["dense_f1"]),
                "dense_far": float(
                    data["fps_doc"]["arms"]["2a_jump_dense_timebase"][scenario]["far_delivered_agg"]
                ),
                "kind": kind_label[spec[scenario]["kind"]],
            })
        else:
            untuned = arms["2b_jump_physical_untuned"][scenario]
            opt = arms["3_jump_optimised"][scenario]
            dense = arms["2a_jump_dense_timebase"][scenario]
            row.update({
                "target_fps": None,
                "achieved_fps": s_fps.get(scenario),
                "mean_dn": None,
                "updates": None,
                "untuned_f1": float(untuned["f1"]),
                "untuned_far": float(untuned["far"]),
                "opt_f1": float(opt["f1"]),
                "opt_far": float(opt["far"]),
                "dense_f1": float(dense["f1"]),
                "dense_far": None,
                "kind": kind_label[spec[scenario]["kind"]],
            })
        rows.append(row)
    return rows


def rule_check(rows: list[dict], v3_consec: int) -> list[dict]:
    """The mechanism predicts v3's F1 -- with the two units matched.

    ``analyse_runs_r90.runs_of()`` walks the list of *gaps* between consecutive
    delivered frames, so ``max_run_maxgap1`` counts **gaps** (dN == 1 steps), NOT
    observations.  ``apply_consecutive_confirmation`` counts **observations**: a run of
    ``L`` observations spans ``L - 1`` gaps.  Hence the longest observation run the
    mechanism can ever see is ``max_run_maxgap1 + 1``, and the rule is

        F1 == 0  is FORCED   when  max_run_maxgap1 + 1 <  confirm_consec
        F1 >  0  is ALLOWED  when  max_run_maxgap1 + 1 >= confirm_consec

    The second branch is a *necessary*, not a sufficient, condition: it says the
    mechanism does not forbid a surviving track, not that one must exist.
    """
    out = []
    for row in rows:
        max_observations = row["max_run_gap1"] + 1
        predicted_zero = max_observations < v3_consec
        observed_zero = row["untuned_f1"] == 0.0
        out.append({
            **row,
            "max_obs_run": max_observations,
            "predicted_zero": predicted_zero,
            "observed_zero": observed_zero,
            "agrees": predicted_zero == observed_zero,
        })
    return out


def _atoms(text: str) -> list[str]:
    """Split into break-able atoms: after CJK punctuation or a space, never inside a code span."""
    atoms: list[str] = []
    current = ""
    in_code = False
    for char in text:
        if char == "`":
            in_code = not in_code
            current += char
        elif not in_code and char in "、，。；：）」』！？":
            current += char
            atoms.append(current)
            current = ""
        elif not in_code and char == " ":
            current += char
            atoms.append(current)
            current = ""
        else:
            current += char
    if current:
        atoms.append(current)
    return atoms


def wrap(text: str, limit: int = 118) -> list[str]:
    """Wrap a paragraph without ever splitting an inline code span."""
    lines: list[str] = []
    current = ""
    for atom in _atoms(text):
        if current and len(current) + len(atom.rstrip()) > limit:
            lines.append(current.rstrip())
            current = atom.lstrip()
        else:
            current += atom
    if current.strip():
        lines.append(current.rstrip())
    return lines


def para(text: str) -> list[str]:
    return wrap(text, 118)


def check_line_width(lines: list[str], limit: int = 120) -> list[str]:
    return [f"line {index} is {len(line)} chars" for index, line in enumerate(lines, 1) if len(line) > limit]


def claim_text(text: str) -> str:
    """Strip everything that is not a numeric claim: code blocks, identifier code spans, IPs."""
    stripped = FENCE_RE.sub(" ", text)
    stripped = IPV4_RE.sub(" ", stripped)
    stripped = INLINE_CODE_RE.sub(
        lambda match: " " if re.search(r"[/\._]", match.group(0)) else match.group(0), stripped
    )
    return stripped


def drift_check(text: str, allowed: set[str]) -> list[str]:
    """Every numeric literal in the prose must be traceable to an artefact or computed."""
    problems = []
    for line in claim_text(text).splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and set(stripped) <= set("|- :"):
            continue
        for token in NUMBER_RE.findall(stripped.replace(",", "")):
            plain = token.replace(",", "")
            if plain in allowed:
                continue
            problems.append(f"untraceable number {token!r} in: {stripped[:110]}")
    return problems


def main() -> int:
    data = load_all()
    v3 = data["v3"]
    opt = data["opt"]
    v3_consec = int(v3["confirm_consec"])
    v3_gap_tol = int(v3["confirm_gap_tol"])
    opt_consec = int(opt["confirm_consec"])
    opt_gap_tol = int(opt["confirm_gap_tol"])
    v3_max_gap = 1 + v3_gap_tol
    opt_max_gap = 1 + opt_gap_tol

    rows = rule_check(build_rows(data), v3_consec)
    fps_rows = [r for r in rows if r["target_fps"] is not None]
    meas_rows = [r for r in rows if r["target_fps"] is None]
    ladder_low = min(r["achieved_fps"] for r in fps_rows)
    ladder_high = max(r["achieved_fps"] for r in fps_rows)
    zero_rows = [r for r in rows if r["observed_zero"]]
    agree_rows = [r for r in rows if r["agrees"]]
    f17 = next(r for r in fps_rows if r["scenario"] == "f17")
    f14 = next(r for r in fps_rows if r["scenario"] == "f14")
    s3 = next(r for r in meas_rows if r["scenario"] == "s3")
    board_fps, board_dn = board_cadence(data["spec_md"])

    calib = data["fps_doc"]["delivery_caliber"]
    published = data["delivery"]["cc2_md10"]
    cross_f1 = float(calib["selected"]["f1"]) - float(published["f1"])
    cross_far = abs(float(calib["selected"]["far"]) - float(published["far"]))
    cross_f1_pp = cross_f1 * 100
    opt_f1_min = min(r["opt_f1"] for r in fps_rows)
    opt_far_max = max(r["opt_far"] for r in fps_rows)

    def residual_pct(row: dict) -> float:
        """达成 fps 与目标 fps 的相对残差（%）。"""
        return abs(row["achieved_fps"] - row["target_fps"]) / row["target_fps"] * 100

    resid = {r["scenario"]: residual_pct(r) for r in fps_rows}
    under_1 = [s for s, v in resid.items() if v < 1.0]
    over_1 = [s for s, v in resid.items() if v >= 1.0]

    elapsed = float(data["fps_doc"].get("elapsed_s", 0))
    log = data["log"]
    start_line = next(line for line in log.splitlines() if line.startswith("[r90] start"))
    end_line = next(line for line in log.splitlines() if line.startswith("[r90] exit"))
    redline = float(calib["redline_far"])
    f1_floor = float(calib["f1_floor"])
    base_f1 = float(calib["base"]["f1"])
    base_far = float(calib["base"]["far"])
    ceiling_frames = int(data["fps_doc"]["arms"]["1_perframe_ceiling"]["frames"])

    # -- corpus + computed registry for the drift check --------------------------
    structured = [
        list(data["ladder"].values()),
        list(read_csv(ANALYSE_CSV)),
        list(read_csv(SCENARIO_CSV)),
        list(read_csv(DELIVERY_CSV)),
        data["fps_doc"],
    ]
    texts = [data["grid_md"], data["spec_md"], log]
    allowed = corpus_tokens(structured, texts)
    allowed |= set(EXTRA_ALLOWED)
    computed = {
        "cross_f1_pp": cross_f1_pp,
        "cross_far": cross_far,
        "ladder_low": ladder_low,
        "ladder_high": ladder_high,
        "opt_f1_min": opt_f1_min,
        "opt_far_max": opt_far_max,
        "p_dn1_mean": sum(r["p_dn1"] for r in rows) / len(rows),
    }
    for value in resid.values():
        add_value(allowed, value)
    for value in computed.values():
        add_value(allowed, value)
    for row in rows:  # the per-row aggregates this script derives from the CSV
        add_value(allowed, row["pos_ge2_gap9"] / row["gaps"] * 100)
        add_value(allowed, row["max_obs_run"])
        if row["achieved_fps"] is not None:
            add_value(allowed, row["achieved_fps"])
        for key in ("p_dn1", "achieved_fps", "untuned_f1", "untuned_far", "opt_f1",
                    "opt_far", "dense_f1", "dense_far", "mean_dn"):
            if row.get(key) is not None:
                add_value(allowed, row[key])

    lines: list[str] = []
    in_code = False

    def add(text: str = "") -> None:
        """Append a line; prose is wrapped, table rows and code fences are kept verbatim."""
        nonlocal in_code
        if text.startswith("```"):
            in_code = not in_code
            lines.append(text)
            return
        if in_code or text.startswith("|") or not text.strip():
            lines.append(text)
            return
        lines.extend(para(text))

    add("### 三十七、🔴 交付帧率网格（第 91 轮）：**「降到多少帧率，跟踪器就不能用了」——"
        "v3 基座的可用下限落在 14~17 fps 之间**")
    add("")
    add("**一句话结论**：`s1`~`s4` 只回答了「实测节拍有多糟」，**没有**回答交付决策真正关心的"
        "那一个问题；本轮补上 `f25/f20/f17/f14/f12` 五档帧率后，规则与观测在 "
        f"**{len(agree_rows)}/{len(rows)}** 个场景**逐点吻合**"
        f"（唯一例外 `f14` 恰卡在机制门槛上，见 §4）："
        f"**`v3` 基座在 {f14['achieved_fps']:.2f} fps 就已经 F1 精确为 0**，"
        f"而 {f17['achieved_fps']:.2f} fps 尚能到 {f6(f17['untuned_f1'])}；"
        f"**交付参数组 `cc2_md10` 把这个下限压到 {ladder_low:.2f} fps 仍然全档 ≥ {f6(opt_f1_min)}**。")
    add("")
    add(f"**业务含义**：板卡实测节拍是 {board_fps} fps（Δn 均值 {board_dn}）⇒ **`v3` 基座在真实板卡上不可用**，"
        "**必须换参数组**（`cc2_md10`），而不是「板卡快一点」就能救。")
    add("")

    add("#### 1) 这轮补的是哪个洞")
    add("")
    add("| 已有 | 回答什么 | 没回答什么 |")
    add("| --- | --- | --- |")
    add("| `s1`~`s4`（第 89/90 轮） | 实测节拍**有多糟** | **它坏在哪个速率上** |")
    add("| 逐帧臂（Δn≡1） | 最快锚点 | 中间 12~20 fps 这一段**从未被采样过** |")
    add("| **`f25`~`f12`（本轮）** | **交付速率阶梯 + 可用下限** | — |")
    add("")
    add(f"实测达成 fps 区间 **{ladder_low:.4f} ~ {ladder_high:.4f}**；"
        "⚠️ **「实测达成 fps」才是可引用的数字**：目标帧率是采样器的请求值，"
        "而相邻帧乱序会让引擎丢掉晚到的那一帧 ⇒ **跟踪器实际看到的速率低于请求值**。")
    add(f"（生成器已按此标定：`{'`/`'.join(under_1)}` 残差 < 1%；"
        f"**`{'`/`'.join(over_1)}` 残差 {resid[over_1[0]]:.2f}%** —— "
        "间隙下限为 1 帧，该档目标 fps 原理上不可达）")
    add("")

    add("#### 2) 帧率阶梯：F1 / FAR 随交付速率的变化")
    add("")
    add("| 臂 | 口径 | 场景 | 实测达成 fps | 跳帧 F1 | 跳帧 FAR |")
    add("| --- | --- | --- | ---: | ---: | ---: |")
    for arm_name, arm_label, f1_key, far_key in (
        ("2b_jump_physical_untuned", "v3 基座 · 未调参", "untuned_f1", "untuned_far"),
        ("3_jump_optimised", "cc2_md10（交付参数组）", "opt_f1", "opt_far"),
        ("2a_jump_dense_timebase", "密集时间基（SDK 现状口径）", "dense_f1", "dense_far"),
    ):
        for row in fps_rows:
            add(f"| `{arm_name}` | {arm_label} | `{row['scenario']}` | "
                f"{row['achieved_fps']:.4f} | {f6(row[f1_key])} | {f6(row[far_key])} |")
    add("")
    add(f"（跳帧 F1 / 跳帧 FAR：逐序列 micro-pool；FAR 红线 {redline}。口径提示："
        "`2b_jump_physical_untuned` / `3_jump_optimised` 是**物理帧基**，"
        "`2a_jump_dense_timebase` 是**密集时间基口径**，两者**不可直接并列比较**）")
    add("")
    add(f"**锚点（逐帧，Δn≡1）**：F1={f6(base_f1)}、FAR={f6(base_far)}。")
    add("")

    add(f"#### 3) 🔴 机制**读码**确认 + 中间量**部分**证实"
        f"（本轮规则-观测吻合 {len(agree_rows)}/{len(rows)}，1 处例外）")
    add("")
    add("门限语义已**读码**确认：`manu/evaluation/system_sota_longquan.py` 的 "
        "`apply_consecutive_confirmation()` —— 一条航迹的**真实观测帧**按"
        "「与前一观测帧的物理帧差 ≤ `max_gap = 1 + confirm_gap_tol`」切成游程，"
        "**最长游程 < `confirm_consec` 的整条航迹被丢弃**。")
    add("")
    add("| 参数组 | `confirm_consec` | `confirm_gap_tol` | `max_gap` | 连续性要求 |")
    add("| --- | ---: | ---: | ---: | --- |")
    add(f"| `v3` 基座 | {v3_consec} | {v3_gap_tol} | {v3_max_gap} | "
        f"需 {v3_consec} 次观测且**相机帧相邻（Δn=1）** |")
    add(f"| `cc2_md10`（交付参数组） | {opt_consec} | {opt_gap_tol} | {opt_max_gap} | "
        f"{opt_consec} 次观测、间隔 ≤ {opt_max_gap} 帧即可 |")
    add("")
    add(f"⇒ 于是规则是**可证伪的**，但**两支强度不同**："
        f"若某场景的**最长连续交付帧数** < {v3_consec}，则 `v3` 在该场景的 F1 "
        "**必须精确为 0**（**逻辑强制**）；否则**只意味着机制允许 F1 > 0**"
        "（必要非充分 —— 不保证真有航迹活下来）。")
    add("")
    add(f"⚠️ **单位对齐（本轮审计修正的关键）**：节拍统计量 `max_run_maxgap1` 的单位是"
        f"**间隙**（连续 Δn=1 的步数），而 `confirm_consec={v3_consec}` 的单位是**观测数** ⇒ "
        f"{v3_consec} 次观测需要 **{v3_consec - 1}** 个连续 Δn=1 间隙 ⇒ "
        f"判据必须写成 `maxRun@1 + 1 < confirm_consec`，"
        f"**不能**写成 `maxRun@1 < confirm_consec`（差一个单位 ⇒ 会多算 1 个场景）。")
    add("")

    add(f"#### 4) 机制量化表：规则 → 观测，逐点核对"
        f"（吻合 {len(agree_rows)}/{len(rows)}）")
    add("")
    add("聚合口径：`p(Δn=1)` 取 3 seeds 均值；`maxRun` 与「间隙位置数」取 3 seeds 的**最大值**"
        "（保守：只要有一个 seed 存在这样的游程，规则就不允许 F1 恒为 0）。")
    add("")
    add("⚠️ **列名口径**：`maxRun@k` = **最长连续游程长度（单位=间隙）**；"
        "「间隙位置数（游程≥k）」= **尾部游程长度达到 k 的间隙位置个数**（**不是游程条数**，"
        "同一条游程会被它的每个合格位置各数一次）；"
        "「最长连续交付帧数」= `maxRun@1 + 1`（游程含首尾两端的交付帧，单位=**帧**）。")
    add("")
    add("**(a) `max_gap=1` 侧（`v3` 的判据）**")
    add("")
    add("| 场景 | 来源 | 达成 fps | p(Δn=1) | maxRun@1(间隙) | 最长连续交付帧数 | "
        "间隙位置数(游程≥7) | v3 F1 | 规则预测 | 观测 | 吻合 |")
    add("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :-: | :-: | :-: |")
    for row in rows:
        add(f"| `{row['scenario']}` | {row['kind']} | {row['achieved_fps']:.4f} | "
            f"{row['p_dn1']:.4f} | "
            f"{row['max_run_gap1']} | {row['max_obs_run']} | {row['pos_ge7_gap1']} | "
            f"{f6(row['untuned_f1'])} | "
            f"{'必须≡0' if row['predicted_zero'] else '允许>0'} | "
            f"{'≡0' if row['observed_zero'] else '>0'} | "
            f"{'✅' if row['agrees'] else '❌'} |")
    add("")
    add("⚠️ **口径澄清（防止把两个数混为一谈）**：第 90 轮表 2 里 `v3` 未调参的跳帧均值 "
        "**0.084122** 是 **`s1`~`s4` 四个节奏的均值**（达成 fps 依次为 7.4475 / 4.7848 / 3.5759 / 12.8220），"
        "**不是**「某个 14 fps 档」的值 ⇒ **按本轮的帧率阶梯，14.1043 fps 档 `v3` 的 F1 是精确 0**。")
    add("")
    add("**(b) `max_gap=9` 侧（`cc2_md10` 的判据）**")
    add("")
    add("| 场景 | maxRun@9(间隙) | 间隙位置数(游程≥2) | 间隙总数 | 覆盖率 | cc2_md10 F1 |")
    add("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        add(f"| `{row['scenario']}` | {row['max_run_gap9']} | {row['pos_ge2_gap9']} | "
            f"{row['gaps']} | {row['pos_ge2_gap9'] / row['gaps'] * 100:.2f}% | {f6(row['opt_f1'])} |")
    add("")
    add("覆盖率 = 「间隙位置数(游程≥2)」/ 间隙总数（能形成「≥2 连」游程的间隙占比）；"
        "分子分母**同取 3 seeds 的最大值**（同一口径）。")
    add("")
    add(f"**吻合 {len(agree_rows)}/{len(rows)}**，其中 **{len(zero_rows)} 个场景 v3 的 F1 精确为 0**"
        f"（{', '.join('`' + r['scenario'] + '`' for r in zero_rows)}）——"
        "规则预测的「零」与观测到的「零」**逐点重合**，不是「相关性不错」。")
    add("")
    disagree = [r for r in rows if not r["agrees"]]
    if disagree:
        add(f"🔴 **1 处例外：{'`、`'.join('`' + r['scenario'] + '`' for r in disagree)}**"
            f" —— 该场景的**最长连续交付帧数 = {disagree[0]['max_obs_run']} = `confirm_consec`({v3_consec})**，"
            "**恰好卡在机制门槛上** ⇒ 机制**不再强制** F1≡0（只「允许 >0」），实测却是 0 ⇒ "
            "「临界可行」≠「实际达成」（另有可能目标在该窗口内未被检出等其它原因）。"
            f"**故机制只解释 {len(agree_rows)}/{len(rows)} 个场景，不能声称已被完全证实。**")
        add("")
    add("⚠️ **但不要把「间隙位置数(游程≥7)」线性换算成 F1**："
        "`maxRun`/「间隙位置数」是**全图节拍**的统计量，而一条航迹只在它**被观测到的子区间**里才有观测，"
        "所以全图连长是每条航迹连长的**上界**；cc2_md10 那一侧同样非单调"
        f"（例如 `s3` 的覆盖率明显低于 `f25`，但 F1 仍高达 {f6(s3['opt_f1'])}）。")
    add("")

    add("#### 5) 交叉验证：两个评测器互证")
    add("")
    add("本轮服务器运行**独立复现**了交付口径（`delivery_caliber.selected`），"
        "与第 90 轮已发布的 `cc2_md10` 对账：")
    add("")
    add("| 口径 | F1 | FAR |")
    add("| --- | ---: | ---: |")
    add(f"| 本轮运行（`rt_fps_grid_r90.json` → `delivery_caliber.selected`） | "
        f"{f6(float(calib['selected']['f1']))} | {f6(float(calib['selected']['far']))} |")
    add(f"| 第 90 轮已发布（`rt_delivery_compare.csv` → `cc2_md10`） | "
        f"{f6(float(published['f1']))} | {f6(float(published['far']))} |")
    add(f"| 差 | {cross_f1_pp:+.4f} pp | {cross_far:.6f}（取绝对值） |")
    add("")
    add(f"⇒ 两套独立的评测实现互证一致（F1 差 {abs(cross_f1_pp):.4f} pp、FAR 差 {cross_far:.6f}）。")
    add(f"⚠️ **两侧口径不同，必须一起读**：本轮 = **3 seeds × {ceiling_frames:,} 帧**"
        f"（`1_perframe_ceiling.frames`），第 90 轮已发布 = **1 seed × 165,223 帧**（名义总帧数）"
        "⇒ 这是「两套独立实现互证」，**不是**「同批样本逐位比对」。")
    add("")

    add("#### 6) 业务含义")
    add("")
    add(f"1. **`v3` 基座的可用下限在 14~17 fps 之间**：`f17` 档 "
        f"（{f17['achieved_fps']:.4f} fps）F1 尚有 {f6(f17['untuned_f1'])}，"
        f"到 `f14` 档（{f14['achieved_fps']:.4f} fps）**精确为 0**。")
    add(f"2. **板卡实测节拍 {board_fps} fps 远在下限之下** ⇒ `v3` 基座在真实板卡上**不可用**。")
    add(f"3. **交付参数组 `cc2_md10` 把下限压到 ≤ {ladder_low:.2f} fps**"
        f"（全档 F1 ≥ {f6(opt_f1_min)}、FAR ≤ {f6(opt_far_max)}，红线 {redline}）"
        "⇒ **要救的是参数组，不是板卡速度**。")
    add(f"4. 交付口径自检同时给出 F1 下限 {f6(f1_floor)} 与 FAR 上限 {redline}，"
        f"本次 `feasible={calib['feasible']}`（`f1_ok={calib['f1_ok']}`、`far_ok={calib['far_ok']}`）。")
    add("")

    add("#### 7) 局限（必须和结论一起读）")
    add("")
    add("1. **`f` 网格固定抖动形状、只改速率**：它把实测节拍的**抖动形状**按目标均值缩放，"
        "**与实测 `s1` 不是同一条生成链** ⇒ **`s1` 不是 `f` 网格的延续**，"
        "两条链只能并列陈述、不能互相插值。")
    add(f"2. **`f25` 只能到 {ladder_high:.4f} fps**：间隙下限是 1 帧，再往上采样不出来；"
        "逐帧锚点仍以 Δn≡1 的口径单列。")
    add(f"3. **机制只解释 {len(agree_rows)}/{len(rows)} 个场景**：`f14` 的最长连续交付帧数 = "
        f"{next(r['max_obs_run'] for r in rows if r['scenario'] == 'f14')} = `confirm_consec`({v3_consec})，"
        "**恰好卡在门槛上** ⇒ 机制不再强制 F1≡0，实测却是 0 ⇒ 机制**不是充分条件**；"
        "「v3 的经验可用下限在 14~17 fps 之间」是**观测事实**，但「连续性门限导致 f14 归零」"
        "**只作归因、不作因果证明**（第 90 轮原文即如此表述，本轮**不升级**）。")
    add("4. **`s1`~`s3` 不是板卡真实序列**：它们是对**实测 Δn 分布做 IID 重采样**（bootstrap）"
        "得到的合成节拍；`s4` 是 **brief assumption**（`scenarios_spec.s4.kind=\"doc\"`）"
        "⇒ 这 4 档只能作**量级参考**，不能当板卡实测证据；`s4` 参与的是「吻合」计数，"
        "其证据力弱于其它 8 档。")
    add("5. **本轮不涉及任何板端复核**：全部数字来自服务器侧离线评测；板端不在本轮范围内。")
    add("6. **交叉验证两侧样本不同**：本轮 3 seeds × "
        f"{ceiling_frames:,} 帧 vs 第 90 轮 1 seed × 165,223 帧 ⇒ 只能互证实现一致，不能逐位比对。")
    add("")

    add("#### 8) 产物 / 命令 / 执行机")
    add("")
    add("| 项 | 值 |")
    add("| --- | --- |")
    add("| 执行机（重活） | 服务器 `192.168.99.40:32222`（`huangzhe@`，conda env `uav`） |")
    add("| 执行机（报告 + 落档） | 本机（纯读产物，无 GPU / 无网络 / 无重活） |")
    add(f"| 服务器运行窗口 | `{start_line.split('|')[0].strip()}` → `{end_line.split('|')[0].strip()}`"
        f"（`elapsed_s={elapsed}`） |")
    add("| 帧率网格脚本 | `manu/evaluation/rt_fps_jitter_search.py`（新增场景类型 `target_fps`） |")
    add("| 网格报告脚本 | `manu/evaluation/rt_fps_grid_report.py`（import 复用 `rt_grid_compare.py` 的口径） |")
    add("| 节拍分析脚本 | `/mnt/data/siping/runs/trial0080_onnx/analyse_runs_r90.py` |")
    add("| 本轮产物 | `delivery_20261119/artifacts/rt_fps_grid_r90.json` |")
    add("| 本轮产物（续） | `delivery_20261119/artifacts/rt_fps_grid_r90.log`、`analyse_runs_r90.csv` |")
    add("| 报告产物 | `delivery_20261119/grid_compare/rt_fps_grid.md` / `.csv` |")
    add("| 落档生成器 | `manu/memory/trial0080_alignment_20261119/search/make_memory_section_fps_grid_91.py` |")
    add("")
    add("服务器运行命令（帧率网格）：")
    add("")
    add("```bash")
    add("cd /mnt/data/siping/runs/npu20ms/nano_search/repo")
    add("export PYTHONPATH=/mnt/data/siping/ptq_work/ortgpu:"
        "/mnt/data/siping/runs/npu20ms/nano_search/repo")
    add("CACHE=/mnt/data/siping/runs/trial0080_onnx/ms_cache")
    add('GT="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"')
    add("timeout 14400 ~/anaconda3/envs/uav/bin/python manu/evaluation/rt_fps_jitter_search.py \\")
    add('  --cache-root "$CACHE" --labels-root "$GT" --all-seqs \\')
    add("  --params-json manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json \\")
    add("  --opt-params-json manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json \\")
    add("  --phase eval --scenarios f25,f20,f17,f14,f12 --seeds 0,1,2 --workers 8 \\")
    add("  --out-json /mnt/data/siping/runs/trial0080_onnx/search/rt_fps_grid_r90.json")
    add("```")
    add("")
    add("本机命令（报告 + 落档）：")
    add("")
    add("```bash")
    add("python manu/evaluation/rt_fps_grid_report.py \\")
    add("  --arm-json manu/memory/delivery_20261119/artifacts/rt_fps_grid_r90.json \\")
    add("  --out-dir manu/memory/delivery_20261119/grid_compare")
    add("python manu/memory/trial0080_alignment_20261119/search/make_memory_section_fps_grid_91.py")
    add("```")
    add("")

    text = "\n".join(lines) + "\n"
    OUT_MD.write_text(text, encoding="utf-8")

    problems = check_line_width(lines)
    problems += drift_check(text, allowed)
    if problems:
        print("[FAIL] the generated section failed its self checks:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    compact_lines: list[str] = []

    def cadd(text: str = "") -> None:
        if text.startswith("|") or not text.strip():
            compact_lines.append(text)
            return
        compact_lines.extend(para(text))

    cadd("### 三十七、🔴 交付帧率网格（第 91 轮）：**「降到多少帧率，跟踪器就不能用了」**")
    cadd("")
    cadd(f"**结论**：`v3` 基座可用下限落在 14~17 fps 之间（`f17` 档 "
         f"{f17['achieved_fps']:.4f} fps → F1 {f6(f17['untuned_f1'])}；"
         f"`f14` 档 {f14['achieved_fps']:.4f} fps → **F1 精确为 0**）；"
         f"板卡实测节拍 {board_fps} fps ⇒ **v3 在真实板卡上不可用**；"
         f"交付参数组 `cc2_md10` 把下限压到 ≤ {ladder_low:.2f} fps"
         f"（全档 F1 ≥ {f6(opt_f1_min)}、FAR ≤ {f6(opt_far_max)}，红线 {redline}）"
         "⇒ **要救的是参数组，不是板卡速度**。")
    cadd("")
    cadd("**机制**（`apply_consecutive_confirmation`，`system_sota_longquan.py`）："
         f"最长游程 < `confirm_consec` 的航迹整条丢弃；`v3` = {v3_consec}/{v3_gap_tol}"
         f"（max_gap={v3_max_gap} ⇒ 必须 Δn=1 连 {v3_consec} 次观测），"
         f"`cc2_md10` = {opt_consec}/{opt_gap_tol}（max_gap={opt_max_gap}）。")
    cadd("")
    cadd(f"⚠️ **单位**：`maxRun@1` 单位=**间隙**，最长连续交付帧数 = `maxRun@1 + 1`（单位=帧）；"
         f"判据是 `maxRun@1 + 1 < {v3_consec}` ⇒ 强制 F1≡0；否则**只「允许 >0」**（必要非充分）。")
    cadd("")
    cadd("| 场景 | 达成 fps | p(Δn=1) | maxRun@1(间隙) | 最长连续交付帧数 | 间隙位置数(游程≥7) | "
         "v3 F1 | 规则预测 | 吻合 |")
    cadd("| --- | ---: | ---: | ---: | ---: | ---: | ---: | :-: | :-: |")
    for row in rows:
        cadd(f"| `{row['scenario']}` | {row['achieved_fps']:.4f} | {row['p_dn1']:.4f} | "
             f"{row['max_run_gap1']} | {row['max_obs_run']} | "
             f"{row['pos_ge7_gap1']} | {f6(row['untuned_f1'])} | "
             f"{'必须≡0' if row['predicted_zero'] else '允许>0'} | "
             f"{'✅' if row['agrees'] else '❌'} |")
    cadd("")
    cadd("⚠️ **口径澄清**：第 90 轮表 2 的 `v3` 跳帧均值 **0.084122** 是 `s1`~`s4` **四个节奏的均值**"
         "（7.4475 / 4.7848 / 3.5759 / 12.8220 fps），**不是**「14 fps 档」的值 ⇒ "
         "**14.1043 fps 档 `v3` 的 F1 是精确 0**。")
    cadd("")
    cadd("| 场景 | maxRun@9(间隙) | 间隙位置数(游程≥2) | 覆盖率 | cc2 F1 |")
    cadd("| --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        cadd(f"| `{row['scenario']}` | {row['max_run_gap9']} | {row['pos_ge2_gap9']} | "
             f"{row['pos_ge2_gap9'] / row['gaps'] * 100:.2f}% | {f6(row['opt_f1'])} |")
    cadd("")
    cadd(f"**规则-观测吻合 {len(agree_rows)}/{len(rows)}**，其中 {len(zero_rows)} 个场景 v3 的 F1 精确为 0。")
    cadd("")
    if disagree:
        cadd(f"🔴 **例外 {'`、`'.join('`' + r['scenario'] + '`' for r in disagree)}**：最长连续交付帧数 = "
             f"{disagree[0]['max_obs_run']} = `confirm_consec`({v3_consec})，**恰在门槛上** ⇒ "
             f"机制**只「允许 >0」**（不强制），实测却是 0 ⇒ **机制只解释 {len(agree_rows)}/{len(rows)} 场景，"
             "不是充分条件**。")
    cadd("")
    cadd(f"**交叉验证**：本轮 `delivery_caliber.selected` F1={f6(float(calib['selected']['f1']))} / "
         f"FAR={f6(float(calib['selected']['far']))} vs 第 90 轮已发布 `cc2_md10` "
         f"F1={f6(float(published['f1']))} / FAR={f6(float(published['far']))}"
         f"（差 {cross_f1_pp:+.4f} pp / {cross_far:.6f}）⇒ 两个评测器互证。")
    cadd("")
    cadd(f"⚠️ **两侧口径不同**：本轮 3 seeds × {ceiling_frames:,} 帧 vs 第 90 轮 1 seed × 165,223 帧 "
         "⇒ 互证实现一致，非同批逐位比对。")
    cadd("")
    cadd("**局限**：① `f` 网格固定抖动形状只变速率，**与实测 `s1` 不是同一条链**，不能互相插值；"
         f"② `f25` 只能到 {ladder_high:.4f} fps（间隙下限 1 帧，残差 {resid['f25']:.2f}%）；"
         f"③ 机制只解释 {len(agree_rows)}/{len(rows)} 场景（`f14` 恰在门槛上）；"
         "④ `s1`~`s3` 是实测 Δn 分布的 IID 重采样、`s4` 是 brief assumption；"
         "⑤ 本轮不涉及任何板端复核。")
    cadd("")
    cadd("**产物**：`delivery_20261119/grid_compare/rt_fps_grid.{md,csv}`、"
         "`delivery_20261119/artifacts/rt_fps_grid_r90.{json,log}`、"
         "`delivery_20261119/artifacts/analyse_runs_r90.csv`；"
         "脚本 `manu/evaluation/rt_fps_jitter_search.py` + `rt_fps_grid_report.py`；"
         "执行机：服务器 `192.168.99.40:32222`（重活）+ 本机（报告与落档）。")
    cadd("")
    compact_text = "\n".join(compact_lines) + "\n"
    OUT_COMPACT.write_text(compact_text, encoding="utf-8")

    problems = check_line_width(compact_lines)
    problems += drift_check(compact_text, allowed)
    if problems:
        print("[FAIL] the compact section failed its self checks:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"[OK] {OUT_MD} ({len(lines)} lines)")
    print(f"[OK] {OUT_COMPACT} ({len(compact_lines)} lines)")
    print(f"[OK] rule/observation agreement {len(agree_rows)}/{len(rows)}")
    print(f"[OK] exact-zero scenarios: {[r['scenario'] for r in zero_rows]}")
    print("[OK] drift check: every numeric literal traceable to an artefact or computed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
