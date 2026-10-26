#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
36_mosaic_vs_f32_report.py -- turn the per-arm results of 35_mosaic_board_eval.py into
an auditable comparison table (RK3588 board f16 mosaic  vs  PC f32 multi-scale).

What it produces
----------------
    <out>/mosaic_vs_f32_report.md     the human-readable tables
    <out>/mosaic_vs_f32_report.json   the same content, machine-readable

    A) accuracy       -- one row per arm, F1/R/P/TP/FP/FN/GT + the deltas vs pc-all
    B) decomposition  -- the chain pc-all -> pc-native -> board-single ->
                         board-mosaic-native -> board-mosaic, one row per step, with
                         the MIDDLE QUANTITIES of each step (a bare delta-F1 is not
                         evidence; the TP/FP/FN movement is)
    C) timing         -- the 8 board timing segments (total + ms/frame) + wall clock +
                         fps, and the side-by-side rknn_run of board-single vs
                         board-mosaic (the "mosaic NPU time == single-scale NPU time" claim)
    D) per-tag counts -- the tag 0/1/2 detection count of every arm
    E) the contract   -- what the scorer is, what is NOT the same between the two
                         sides, and why a delta-F1 may not be attributed to quantisation
    F) provenance     -- every number's source: script + artefact + reproduce command

The arms that were never run are printed as ``n/a`` WITH THE REASON -- this report
never invents a number for a missing arm.

Usage
-----
    python manu/pipeline/npu/ptq/36_mosaic_vs_f32_report.py --out <dir>
    python manu/pipeline/npu/ptq/36_mosaic_vs_f32_report.py \
        --result <dir>/pc-all/result.json --result <dir>/board-mosaic/result.json --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The arms, in the order of the decomposition chain.  The chain is also the table order:
# every consecutive pair is ONE isolated change.
CHAIN = ("pc-all", "pc-native", "board-single", "board-mosaic-native", "board-mosaic")
ARM_LABEL = {
    "pc-all": "pc-all (PC f32, 3 尺度)",
    "pc-native": "pc-native (PC f32, 仅 native)",
    "board-single": "board-single (板端 f16, 单尺度画布)",
    "board-mosaic-native": "board-mosaic-native (板端 f16, 拼图/仅 native tile)",
    "board-mosaic": "board-mosaic (板端 f16, 拼图 3 tile)",
}
STEP_EXPLANATION = {
    ("pc-all", "pc-native"):
        "去掉 160/80 两个缩略图尺度 ⇒ **多尺度贡献**。此段内 PC 与 PC 同为 f32、同为同一份 "
        "特征、同一份打分器，差异只来自尺度数量。",
    ("pc-native", "board-single"):
        "PC native(letterbox) → 板端 f16 单尺度画布 ⇒ **f16 + 板端量化**。但这一段仍含"
        "两个未拆开的次级效应：反投影顺序不同（先裁 320 格再缩放 vs 先 320→640 放大再裁，"
        "≈0.25 输入像素的相位差）与板端算子实现的数值差 ⇒ 严格说这是「f16 + 板端实现」的总和，"
        "不是纯量化。",
    ("board-single", "board-mosaic-native"):
        "单尺度画布 → 拼图画布（但只取 native tile）⇒ **拼图画布对 native 检测的扰动**。"
        "两臂的模型、权重、输入像素完全相同，唯一变化是 native tile 在画布里的位置与邻域。",
    ("board-mosaic-native", "board-mosaic"):
        "拼图只取 native tile → 拼图 3 tile ⇒ **拼图里的多尺度贡献**。与 pc-all→pc-native "
        "是同一件事在板端的对照。",
}

CONTRACT_LINES = [
    "**打分口径** = `manu/evaluation/system_sota_longquan.py --no-pipeline`，"
    "`dedup_radius=768`、`th_base=0.3`、`tol=8.0`、`match_mode=inbox`、只用 `tag 0/1/2`"
    "（刻意排除 tag=255 融合行）。",
    "**`th_base=0.3` / `dedup_radius=768` 来自 gate 产物自己的 `config.params`**"
    "（`runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/track3_blind.json`），"
    "**不是** `system_sota_longquan.py` 的 `DEFAULT_PARAMS`（那里是 `th_base=0.22` 且 "
    "`dedup_radius=32.0`）—— 四个组合的实测见下方「口径标定矩阵」，"
    "只有 `(768, 0.30)` 这一格等于冻结基准 `F1=0.897694`；本报告全部数字都用这一格。",
    "**PC 侧是逐尺度 3 次前向，板端是单次拼图前向 ⇒ 两者不是同一个计算图。**"
    "因此 `board-mosaic` 与 `pc-all` 的 ΔF1 同时包含「拼图打包」与「f16/板端量化」两部分，"
    "**不能把 ΔF1 全部归因给量化**；要归因必须看表 B 的逐段分解。",
    "**画布几何差异（已知的口径断裂，不是 bug）**：PC 的 native 走 letterbox（640×512 → 640×640，"
    "上下各 64 行 pad=114），拼图里 native 直接贴 (0,0) 无 pad；PC 的缩略图画布背景是模型序"
    "（residual=0, diff=0, I_t=mean），拼图统一背景 114。",
    "**板端反投影是「先裁 320 格再缩放」，PC 是「先 320→640 放大再裁」** ⇒ 亚像素相位差"
    "（≈0.25 输入像素），**非逐比特等价**。",
    "**板端 TSV 的 `scores` 转成 float16 后再进 pkl，与 PC 侧精度一致**"
    "（PC 侧缓存本身就是 float16）。该链路已用「PC 缓存 → 合成 TSV → 回读打分」验证可复现冻结基准。",
    "**跨口径禁止越级对比**：`manu/pipeline/npu/ptq/ptq_common.py` 的 `match_distance`"
    "（8 px 贪心匹配）与本报告用的 `count_matches(..., 'inbox')` 是**两个不同的打分器**；"
    "任何用前者得到的数字**不可与本表并列**。",
    "**特征溯源已闭环**：板端 BMP 与 gate 缓存所用的 official JPG 特征**逐像素一致**"
    "（`manu/pipeline/npu/ptq/06_verify_reference_anchor.py` 的 `pixel_proof()`，"
    "4000 抽样对 mismatches=0 / max_abs_diff=0，因为 BMP 是 `cv2.imwrite(cv2.imread(jpg))` 的无损重编码）"
    "⇒ 板端与 PC 不是两个不同的特征源。",
    "**gate/epoch3 与 lq_eval 的特征语义不同，不可混用**"
    "（`gate/epoch3/DO_NOT_USE_FOR_LQ_EVAL.md`：gate 用 official 逐 lag 拟合特征、8 序列；"
    "lq_eval 用 FPV 仿射连乘特征、24 序列，逐像素差 26~70/255）。本表只做 gate 口径的 T3 单序列对照。",
    "**表 C 的耗时列只在三个板端 arm 的 `canvas-mode` / `input-mode` / `cpu-affinity` 一致时才可比**"
    "（见 C3）；`rknn_run` 是 NPU 段，`decode`/`rga_*`/`write` 是 CPU/IO 段。",
]


# The口径 calibration matrix: the SAME cache + the SAME GT, scored with the four
# (dedup_radius, th_base) combinations that are in play.  Only the last row is the
# frozen baseline -- this is what makes "which parameters reproduce 0.897694" a
# measurement instead of an assertion.
# Measured with:
#   python manu/evaluation/system_sota_longquan.py \
#       --cache-root <gate>/epoch3/ms_cache --labels-root <Final_Labels...> \
#       --seq VIDEO00007_19700101_001324 --no-pipeline \
#       --params-json <{"th_base":X,"dedup_radius":Y,...}>
CALIBRATION_MATRIX = [
    {"dedup_radius": 32.0, "th_base": 0.22, "tp": 9310, "fp": 2104, "fn": 1225,
     "det": 11414, "f1": 0.848330, "gt": 10535,
     "note": "`system_sota_longquan.py` 的 DEFAULT_PARAMS（脚本默认值）"},
    {"dedup_radius": 32.0, "th_base": 0.30, "tp": 9032, "fp": 1121, "fn": 1503,
     "det": 10153, "f1": 0.873163, "gt": 10535,
     "note": "只对齐了 th_base"},
    {"dedup_radius": 768.0, "th_base": 0.22, "tp": 9205, "fp": 750, "fn": 1330,
     "det": 9955, "f1": 0.898487, "gt": 10535,
     "note": "只对齐了 dedup_radius"},
    {"dedup_radius": 768.0, "th_base": 0.30, "tp": 8972, "fp": 482, "fn": 1563,
     "det": 9454, "f1": 0.897694, "gt": 10535,
     "note": "**gate 产物的 `config.params` —— 本报告的口径**"},
]


def build_calibration_table() -> str:
    """The口径 calibration matrix -- which parameter pair reproduces the frozen baseline."""
    rows = []
    for row in CALIBRATION_MATRIX:
        rows.append([
            f"{row['dedup_radius']:.0f}", f"{row['th_base']:.2f}",
            fmt_int(row["tp"]), fmt_int(row["fp"]), fmt_int(row["fn"]), fmt_int(row["gt"]),
            fmt_int(row["det"]), fmt_float(row["f1"]), row["note"],
        ])
    return md_table(
        ["dedup_radius", "th_base", "TP", "FP", "FN", "GT", "det", "F1", "说明"], rows)


# --------------------------------------------------------------------------------------
def load_results(args) -> tuple[dict, dict]:
    """Return (results_by_arm, provenance_by_arm)."""
    results: dict = {}
    provenance: dict = {}
    sources: list[Path] = []
    if args.result:
        sources = [Path(item) for item in args.result]
    else:
        root = Path(args.result_dir)
        sources = [root / arm / "result.json" for arm in CHAIN]
    for path in sources:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        arm = payload.get("arm")
        if not arm:
            continue
        results[arm] = payload
        provenance[arm] = {"result_json": str(path.resolve())}
    return results, provenance


def fmt_float(value, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_int(value) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):d}"


def fmt_signed(value, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"


def fmt_signed_int(value) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):+d}"


def delta(metrics: dict, key: str, base: dict | None, sign: bool = False):
    """The delta of one metric against the baseline arm."""
    if not metrics or not base:
        return None
    if key not in metrics or key not in base:
        return None
    value = float(metrics[key]) - float(base[key])
    return value


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def reproduce_command(arm: str, result: dict) -> str:
    """The command that reproduces this arm."""
    script = "manu/pipeline/npu/ptq/35_mosaic_board_eval.py"
    config = result.get("config", {})
    parts = [
        "python", script, "--arm", arm,
        "--cache-root", str(config.get("cache_root", "")),
        "--labels-root", str(config.get("labels_root", "")),
        "--out", "<out>",
    ]
    if result.get("start"):
        parts += ["--start", str(result["start"])]
    if result.get("limit"):
        parts += ["--limit", str(result["limit"])]
    gate_json = result.get("gate_json")
    if gate_json:
        parts += ["--gate-json", gate_json]
    else:
        params_json = result.get("params_json")
        if params_json:
            parts += ["--params-json", params_json]
    return " ".join(parts)


# --------------------------------------------------------------------------------------
def build_accuracy_table(results: dict) -> tuple[str, list[dict]]:
    base = results.get("pc-all")
    base_metrics = base.get("metrics") if base else None
    rows = []
    structured = []
    for arm in CHAIN:
        result = results.get(arm)
        if not result:
            rows.append([ARM_LABEL[arm], "n/a", "n/a", "n/a", "n/a", "n/a", "n/a",
                         "n/a", "n/a", "n/a", "n/a"])
            structured.append({"arm": arm, "available": False,
                               "reason": "该 arm 尚未运行（没有 result.json）"})
            continue
        metrics = result["metrics"]
        d_f1 = delta(metrics, "f1", base_metrics)
        d_fp = delta(metrics, "fp", base_metrics)
        d_fn = delta(metrics, "fn", base_metrics)
        rows.append([
            ARM_LABEL[arm],
            fmt_float(metrics.get("f1")),
            f"{metrics.get('recall', 0) * 100:.2f}%" if metrics.get("recall") is not None else "n/a",
            f"{metrics.get('precision', 0) * 100:.2f}%" if metrics.get("precision") is not None else "n/a",
            fmt_int(metrics.get("tp")), fmt_int(metrics.get("fp")), fmt_int(metrics.get("fn")),
            fmt_int(metrics.get("gt")),
            fmt_signed(d_f1), fmt_signed_int(d_fp), fmt_signed_int(d_fn),
        ])
        structured.append({
            "arm": arm, "available": True, "metrics": metrics,
            "delta_f1_vs_pc_all": d_f1, "delta_fp_vs_pc_all": d_fp,
            "delta_fn_vs_pc_all": d_fn, "frames": result.get("frames"),
            "per_tag_counts": result.get("per_tag_counts", {}),
        })
    table = md_table(
        ["arm", "F1", "Recall", "Precision", "TP", "FP", "FN", "GT",
         "ΔF1 vs pc-all", "ΔFP", "ΔFN"], rows)
    return table, structured


def build_decomposition_table(results: dict) -> tuple[str, list[dict]]:
    rows = []
    structured = []
    for before, after in zip(CHAIN, CHAIN[1:]):
        left, right = results.get(before), results.get(after)
        if not left or not right:
            missing = before if not left else after
            rows.append([f"{before} → {after}", "n/a", "n/a", "n/a", "n/a", "n/a",
                         "n/a", "n/a", f"**{missing} 未运行**"])
            structured.append({"step": f"{before}->{after}", "available": False,
                               "reason": f"{missing} 未运行"})
            continue
        lm, rm = left["metrics"], right["metrics"]
        d_f1 = float(rm["f1"]) - float(lm["f1"])
        d_tp = int(rm["tp"]) - int(lm["tp"])
        d_fp = int(rm["fp"]) - int(lm["fp"])
        d_fn = int(rm["fn"]) - int(lm["fn"])
        rows.append([
            f"{before} → {after}",
            fmt_signed(d_f1), fmt_signed_int(d_tp), fmt_signed_int(d_fp), fmt_signed_int(d_fn),
            fmt_int(lm["tp"]), fmt_int(rm["tp"]),
            fmt_int(lm["fp"]), fmt_int(rm["fp"]),
            STEP_EXPLANATION[(before, after)],
        ])
        structured.append({
            "step": f"{before}->{after}", "available": True,
            "delta_f1": d_f1, "delta_tp": d_tp, "delta_fp": d_fp, "delta_fn": d_fn,
            "from_metrics": lm, "to_metrics": rm,
            "explanation": STEP_EXPLANATION[(before, after)],
        })
    table = md_table(
        ["段", "ΔF1", "ΔTP", "ΔFP", "ΔFN", "TP(before)", "TP(after)",
         "FP(before)", "FP(after)", "这一段隔离的是什么"], rows)
    return table, structured


def build_timing_table(results: dict) -> tuple[str, list[dict], str]:
    """The board timing table + the rknn_run side-by-side."""
    segments = ("decode", "rga_import", "rga_improcess", "fp16_or_upload",
                "rknn_run", "outputs_get", "backproject_cc", "write")
    rows = []
    structured = []
    for arm in ("board-mosaic", "board-mosaic-native", "board-single"):
        result = results.get(arm)
        if not result:
            rows.append([ARM_LABEL[arm]] + ["n/a"] * (len(segments) * 2 + 3) +
                        [f"**{arm} 未运行**"])
            structured.append({"arm": arm, "available": False,
                               "reason": f"{arm} 未运行"})
            continue
        timing = result.get("timing", {})
        seg = timing.get("segments_ms", {})
        per_frame = timing.get("ms_per_frame", {})
        frames = timing.get("frames") or result.get("frames") or 0
        row = [ARM_LABEL[arm]]
        for segment in segments:
            row.append(fmt_float(seg.get(segment), 3))
        for segment in segments:
            row.append(fmt_float(per_frame.get(segment), 4))
        row.append(fmt_float(timing.get("wall_ms"), 3))
        row.append(fmt_float(timing.get("fps"), 3))
        row.append(fmt_int(frames))
        rows.append(row)
        structured.append({
            "arm": arm, "available": True, "segments_ms": seg,
            "ms_per_frame": per_frame, "wall_ms": timing.get("wall_ms"),
            "fps": timing.get("fps"), "frames": frames,
            "raw_timing": timing.get("raw", ""),
        })

    header = ["arm"]
    header += [f"{s} (ms)" for s in segments]
    header += [f"{s} (ms/frame)" for s in segments]
    header += ["墙钟 (ms)", "fps", "帧数"]
    table = md_table(header, rows)

    # the mosaic-vs-single NPU claim
    claim_rows = []
    for arm in ("board-mosaic", "board-single"):
        result = results.get(arm)
        if not result:
            claim_rows.append([ARM_LABEL[arm], "n/a", "n/a", "n/a", f"**{arm} 未运行**"])
            continue
        timing = result.get("timing", {})
        seg = timing.get("segments_ms", {})
        per_frame = timing.get("ms_per_frame", {})
        frames = timing.get("frames") or result.get("frames") or 0
        claim_rows.append([
            ARM_LABEL[arm],
            fmt_float(seg.get("rknn_run"), 3),
            fmt_float(per_frame.get("rknn_run"), 4),
            fmt_int(frames),
            "",
        ])
    if len(claim_rows) == 2 and all(row[1] != "n/a" for row in claim_rows):
        mosaic = float(claim_rows[0][1])
        single = float(claim_rows[1][1])
        ratio = mosaic / single if single else float("nan")
        verdict = (f"拼图单次前向的 rknn_run = {mosaic:.3f} ms，"
                   f"单尺度单次前向 = {single:.3f} ms，比值 {ratio:.3f}"
                   f" ⇒ {'**成立**（拼图与单尺度 NPU 时间基本相同）' if abs(ratio - 1) < 0.05 else '**不成立**'}")
        claim_rows[0][4] = verdict
        claim_rows[1][4] = "（对照行）"
    else:
        claim_rows[0][4] = "两臂都未运行或缺少 rknn_run ⇒ 该论断无法判定"
    claim_table = md_table(
        ["arm", "rknn_run 合计 (ms)", "rknn_run (ms/frame)", "帧数", "判定"], claim_rows)
    return table, structured, claim_table


def build_tag_table(results: dict) -> tuple[str, list[dict]]:
    rows = []
    structured = []
    for arm in CHAIN:
        result = results.get(arm)
        if not result:
            rows.append([ARM_LABEL[arm], "n/a", "n/a", "n/a", "n/a", f"**{arm} 未运行**"])
            structured.append({"arm": arm, "available": False})
            continue
        counts = result.get("per_tag_counts", {})
        rows.append([
            ARM_LABEL[arm],
            fmt_int(counts.get("0")), fmt_int(counts.get("1")), fmt_int(counts.get("2")),
            fmt_int(counts.get("255")),
            "tag 255 融合行**不参与**打分（`SCALE_TAGS=(0,1,2)`）",
        ])
        structured.append({"arm": arm, "available": True, "per_tag_counts": counts})
    table = md_table(
        ["arm", "tag 0 (native 640×512)", "tag 1 (thumb160)", "tag 2 (thumb80)",
         "tag 255 (融合行)", "备注"], rows)
    return table, structured


def build_run_param_table(results: dict) -> tuple[str, list[dict]]:
    """The board run parameters -- they decide whether the timing columns are comparable."""
    rows = []
    structured = []
    for arm in ("board-mosaic", "board-mosaic-native", "board-single"):
        result = results.get(arm)
        if not result:
            rows.append([ARM_LABEL[arm], "n/a", "n/a", "n/a", "n/a", "n/a",
                         f"**{arm} 未运行**"])
            structured.append({"arm": arm, "available": False})
            continue
        board = result.get("board", {})
        timing = result.get("timing", {})
        command = board.get("run_command", "")
        affinity = ""
        tokens = command.split()
        for position, token in enumerate(tokens):
            if token == "--cpu-affinity" and position + 1 < len(tokens):
                affinity = tokens[position + 1]
            elif token.startswith("--cpu-affinity="):
                affinity = token.split("=", 1)[-1]
            elif token == "taskset" and position + 2 < len(tokens):
                affinity = f"taskset {tokens[position + 2]}"
        rows.append([
            ARM_LABEL[arm],
            str(board.get("canvas_mode", "n/a")),
            str(board.get("tiles", "n/a")),
            str(board.get("input_mode", "n/a") or "n/a"),
            affinity or "n/a",
            str(board.get("model_md5", "n/a"))[:12],
        ])
        structured.append({
            "arm": arm, "available": True, "canvas_mode": board.get("canvas_mode"),
            "tiles": board.get("tiles"), "input_mode": board.get("input_mode"),
            "model_md5": board.get("model_md5"), "run_command": command,
            "board_summary": board.get("board_summary", ""),
        })
    table = md_table(
        ["arm", "canvas-mode", "tiles", "input-mode", "cpu-affinity", "model md5(前12位)"], rows)
    return table, structured


# --------------------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the auditable mosaic-vs-f32 comparison report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--result-dir", default="",
                        help="directory holding <arm>/result.json (default: the --out dir)")
    parser.add_argument("--result", action="append", default=[],
                        help="an explicit result.json (repeatable)")
    parser.add_argument("--out", required=True, help="the report output directory")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir = args.result_dir or str(out_dir)
    args.result_dir = result_dir

    results, provenance = load_results(args)
    if not results:
        print(f"[36][FATAL] no arm result.json found under {result_dir}; "
              f"run 35_mosaic_board_eval.py first", file=sys.stderr)
        return 2
    missing = [arm for arm in CHAIN if arm not in results]
    if missing:
        print(f"[36][warn] arms without a result (reported as n/a): {missing}")

    acc_table, acc_struct = build_accuracy_table(results)
    dec_table, dec_struct = build_decomposition_table(results)
    time_table, time_struct, claim_table = build_timing_table(results)
    param_table, param_struct = build_run_param_table(results)
    tag_table, tag_struct = build_tag_table(results)

    # provenance
    prov_rows = []
    for arm in CHAIN:
        result = results.get(arm)
        if not result:
            prov_rows.append([ARM_LABEL[arm], "n/a", "n/a", "n/a"])
            continue
        artifacts = result.get("artifacts", {})
        prov_rows.append([
            ARM_LABEL[arm],
            f"manu/pipeline/npu/ptq/35_mosaic_board_eval.py --arm {arm}",
            provenance[arm]["result_json"],
            reproduce_command(arm, result),
        ])
    prov_table = md_table(["arm", "产出脚本", "结果文件", "复现命令"], prov_rows)

    params = {}
    for result in results.values():
        if result.get("params"):
            params = result["params"]
            break

    generated = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    lines = []
    lines.append("# RK3588 板端 f16 拼图推理 vs PC 侧 f32 多尺度 —— 可审计对照表")
    lines.append("")
    lines.append(f"- 生成时间：`{generated}`")
    lines.append(f"- 生成脚本：`manu/pipeline/npu/ptq/36_mosaic_vs_f32_report.py`")
    lines.append(f"- 序列：`{next(iter(results.values())).get('sequence', 'n/a')}`（T3 blind hold-out）")
    lines.append(f"- 打分器：`manu/evaluation/system_sota_longquan.py --no-pipeline`")
    lines.append(f"- 参数：`th_base={params.get('th_base', 'n/a')}`、"
                 f"`dedup_radius={params.get('dedup_radius', 'n/a')}`、"
                 f"`tol=8.0`、`match_mode=inbox`、`SCALE_TAGS=(0,1,2)`")
    lines.append(f"- 冻结基准（T3, official val ms_cache）：`F1=0.897694` / "
                 f"`TP=8972` / `FP=482` / `FN=1563` / `GT=10535` / `det=9454`")
    lines.append("")

    lines.append("## 表 A —— 精度对照")
    lines.append("")
    lines.append("`pc-all` 是基准行（Δ 列全为 0）。")
    lines.append("")
    lines.append(acc_table)
    lines.append("")
    lines.append("### 口径标定矩阵 —— 哪一组参数才等于冻结基准")
    lines.append("")
    lines.append("同一份 ms_cache、同一份 GT、同一套打分器，只改 `dedup_radius` / `th_base`：")
    lines.append("")
    lines.append(build_calibration_table())
    lines.append("")

    lines.append("## 表 B —— 逐段分解（每段只改一个东西）")
    lines.append("")
    lines.append("否定性结论必须量中间量：下表给出每段的 ΔTP / ΔFP / ΔFN，"
                 "而不只是一个 ΔF1 —— FP 涨了还是 FN 涨了，决定了这是「多出来的检测」还是"
                 "「漏掉的目标」。")
    lines.append("")
    lines.append(dec_table)
    lines.append("")

    lines.append("## 表 C —— 板端耗时")
    lines.append("")
    lines.append("### C1 —— 8 段计时")
    lines.append("")
    lines.append(time_table)
    lines.append("")
    lines.append("### C2 —— 拼图单次前向 vs 单尺度单次前向（`rknn_run` 段）")
    lines.append("")
    lines.append(claim_table)
    lines.append("")
    lines.append("### C3 —— 板端运行参数（决定 C1/C2 的耗时列是否可比）")
    lines.append("")
    lines.append(param_table)
    lines.append("")

    lines.append("## 表 D —— 每个 tag 的检测数")
    lines.append("")
    lines.append(tag_table)
    lines.append("")

    lines.append("## 口径声明")
    lines.append("")
    for index, line in enumerate(CONTRACT_LINES, 1):
        lines.append(f"{index}. {line}")
    lines.append("")

    lines.append("## 溯源")
    lines.append("")
    lines.append(prov_table)
    lines.append("")
    lines.append("> 每一个数字都来自上表的「结果文件」，该文件由「产出脚本」生成，"
                 "「复现命令」是重建它的最小命令。表 B/表 C/表 D 里的数字没有第二个来源。")
    lines.append("")

    md_path = out_dir / "mosaic_vs_f32_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "generated_at": generated,
        "generator": "manu/pipeline/npu/ptq/36_mosaic_vs_f32_report.py",
        "result_dir": str(Path(result_dir).resolve()),
        "sequence": next(iter(results.values())).get("sequence"),
        "params": params,
        "frozen_baseline": {
            "f1": 0.897694, "tp": 8972, "fp": 482, "fn": 1563, "gt": 10535, "det": 9454,
        },
        "calibration_matrix": CALIBRATION_MATRIX,
        "arms_present": sorted(results),
        "arms_missing": missing,
        "table_a_accuracy": acc_struct,
        "table_b_decomposition": dec_struct,
        "table_c_timing": time_struct,
        "table_c_run_params": param_struct,
        "table_d_tag_counts": tag_struct,
        "contract": CONTRACT_LINES,
        "provenance": provenance,
        "markdown": str(md_path.resolve()),
    }
    json_path = out_dir / "mosaic_vs_f32_report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[36] arms present : {sorted(results)}")
    print(f"[36] arms missing : {missing}")
    print(f"[36] report -> {md_path.resolve()}")
    print(f"[36] json   -> {json_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
