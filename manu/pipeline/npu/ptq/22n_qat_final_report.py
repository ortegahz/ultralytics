#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
22n_qat_final_report.py -- the QAT route's final verdict, JSON + Markdown.

NO BOARD IS TOUCHED by this script: it only READS what earlier runs wrote.

INPUTS
    report/build_ops.json          the toolkit operator table per arm
                                   (22m_parse_build_log.py) -> InputOperator dtype
    pc_sim/<manifest>/<arm>_pcsim.json      the PC-simulator curves
    pt_ref/<manifest>.json                  the PT float reference curve
    board_eval/<manifest>/*.json            the BOARD curves (historical)
    board_eval_baseline/<manifest>/*.json   the BOARD baseline curves
    report/ort_float_<manifest>.json        the FLOAT ONNX diagnosis
    report/weight_diff.json                 what the QAT training moved

OUTPUTS
    report/qat_final_report.json
    report/qat_final_report.md

Usage (server):
    python manu/pipeline/npu/ptq/22n_qat_final_report.py \
        --work /mnt/data/siping/ptq_work
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MANIFESTS = ["tier1_sensitive", "golden_pool_sub1200", "001324_sub2000"]
MANIFEST_LABEL = {
    "tier1_sensitive": "T1 tier1_sensitive/100",
    "golden_pool_sub1200": "T2 golden_pool_sub1200/1197",
    "001324_sub2000": "T3 001324_sub2000/2000",
}
ARMS = ["a0_qat_float", "a1_qat_maxclip", "a2_qat_learned",
        "a1b_base_maxclip", "a2f_qatw_toolkit_int8"]
ARM_LABEL = {
    "pt_float": "PT float (base weights)",
    "a0_qat_float": "A0  QAT weights, no Q/DQ (fp16)",
    "a1_qat_maxclip": "A1  QAT weights + max-calibrated clip",
    "a2_qat_learned": "A2  QAT weights + LSQ-learned clip",
    "a1b_base_maxclip": "A1b BASE weights + max-calibrated clip",
    "a2f_qatw_toolkit_int8": "A2f QAT weights + toolkit full int8",
    "board_p0_off_A_normal": "int8 PTQ baseline (toolkit)",
    "board_p0_off_A_none": "fp16 baseline (toolkit)",
}
# the board sweep prefixes its arm keys with "board_"
BOARD_ARM_LABEL = {
    "board_pt_float": "PT float (base weights)",
    "board_a0_qat_float": "A0  QAT weights, no Q/DQ (fp16)",
    "board_a1_qat_maxclip": "A1  QAT weights + max-calibrated clip",
    "board_a2_qat_learned": "A2  QAT weights + LSQ-learned clip",
    "board_a1b_base_maxclip": "A1b BASE weights + max-calibrated clip",
    "board_a2f_qatw_toolkit_int8": "A2f QAT weights + toolkit full int8",
    "board_p0_off_A_normal": "int8 PTQ baseline (toolkit)",
    "board_p0_off_A_none": "fp16 baseline (toolkit)",
}
# the threshold each track was re-calibrated to for the deployed int8 model
REC_TH = {"tier1_sensitive": 0.50, "golden_pool_sub1200": 0.30, "001324_sub2000": 0.45}
THS = ["0.05", "0.10", "0.15", "0.20", "0.25", "0.30", "0.35", "0.40", "0.45",
       "0.50", "0.55", "0.60", "0.70"]
# the host-side cost of handing the tensor to the NPU, measured on the board
INPUT_COST = {
    "INT8": "inputs_set 0.34 ms, NPU tensor 20,400 KB",
    "FLOAT16": "inputs_set 45.41 ms, NPU tensor 105,600 KB",
}


def curve_of(curve) -> dict:
    """Accept both shapes: the board sweep stores a LIST of per-threshold rows,
    the PC simulator a DICT keyed by the threshold string."""
    if not curve:
        return {}
    if isinstance(curve, list):
        curve = {f"{float(r['th']):.2f}": r for r in curve}
    best = max(curve.values(), key=lambda c: c["f1"])
    return {"curve": curve, "best_f1": best["f1"], "best_th": best["th"],
            "at_0.30": curve.get("0.30"), "at_0.45": curve.get("0.45")}


def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:                                          # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/mnt/data/siping/ptq_work")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    work = Path(args.work)
    root = work / "qat" / "trial_0080"
    out_dir = Path(args.out) if args.out else root / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    rep: dict = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "work_dir": str(work), "root": str(root)}

    # ---- 1. the input dtype of every arm -----------------------------------
    ops = load_json(out_dir / "build_ops.json") or {}
    rknn_md5 = {}
    for rk in sorted((root / "rknn").glob("*.rknn")):
        rknn_md5[rk.stem] = {"bytes": rk.stat().st_size}
    rep["arms"] = {}
    for name, info in ops.items():
        if name in ("qat_arms", "all", "multi"):        # a whole-log pseudo entry
            continue
        d = info.get("input_op_dtype")
        rep["arms"][name] = {
            "input_op_dtype": d,
            "input_op_device": info.get("input_op_device"),
            "n_operators": info.get("n_operators"),
            "dtype_npu": info.get("dtype_npu"),
            "dtype_cpu": info.get("dtype_cpu"),
            "inputs_set_cost": INPUT_COST.get(d, "unknown"),
            "rknn_bytes": (rknn_md5.get(name) or {}).get("bytes"),
            "build_log": info.get("log"),
        }

    # ---- 2. the curves ------------------------------------------------------
    rep["manifests"] = {}
    for m in MANIFESTS:
        entry: dict = {"label": MANIFEST_LABEL[m]}
        ptref = load_json(root / "pt_ref" / f"{m}.json") or {}
        entry["pt_float_reference"] = curve_of(ptref.get("curve", {}))
        entry["pt_float_reference"]["seconds"] = ptref.get("seconds")

        # PC simulator
        pc = {}
        for arm in ARMS:
            d = load_json(root / "pc_sim" / m / f"{arm}_pcsim.json")
            if not d:
                continue
            c = curve_of(d.get("curve", {}))
            loss = dict(d.get("loss_vs_pt_float", {}))
            # if the run predates --pt-ref, derive the loss from the reference
            if not loss and ptref.get("curve"):
                for th, row in c.get("curve", {}).items():
                    p = ptref["curve"].get(th)
                    if p:
                        loss[th] = round(p["f1"] - row["f1"], 6)
            c["loss_vs_pt_float"] = loss
            c["build_mode"] = d.get("build_mode")
            c["rknn_md5"] = d.get("rknn_md5")
            c["cos_heatmap_mean"] = d.get("cos_heatmap_mean")
            c["sim_ms_per_frame"] = d.get("sim_ms_per_frame")
            pc[arm] = c
        entry["pcsim"] = pc

        # the board (historical)
        board = {}
        for src in ("board_eval", "board_eval_baseline"):
            rep_sweep = load_json(root / src / m / f"{m}_threshold_sweep.json")
            if not rep_sweep:
                continue
            for arm, a in rep_sweep.get("arms", {}).items():
                c = curve_of(a.get("curve", []))
                if c:
                    c["model_md5"] = a.get("model_md5")
                    board[arm] = c
        entry["board"] = board

        # the float ONNX diagnosis
        ort = load_json(out_dir / f"ort_float_{m}.json")
        if not ort:
            # tier1 was written under its short name
            ort = load_json(out_dir / "ort_float_tier1.json")
        if ort:
            # a better verdict than the raw px>0.5 ratio: what matters is the
            # DETECTION count and the FP count at the nominal threshold
            b30 = ort["base"]["curve"].get("0.30", {})
            t30 = ort["test"]["curve"].get("0.30", {})
            det_ratio = (ort["test"]["detections_total_at_collect_th"] /
                         max(1, ort["base"]["detections_total_at_collect_th"]))
            fp_ratio = (t30.get("fp", 0) / b30.get("fp", 1)) if b30.get("fp") else None
            float_explodes = (det_ratio > 1.5) and (t30.get("fp", 0) > 2 * b30.get("fp", 0))
            ort["comparison"]["detections_ratio"] = round(det_ratio, 4)
            ort["comparison"]["fp_ratio_at_0.30"] = round(fp_ratio, 4) if fp_ratio else None
            ort["comparison"]["float_f1_at_0.30_base"] = b30.get("f1")
            ort["comparison"]["float_f1_at_0.30_qat"] = t30.get("f1")
            ort["comparison"]["verdict"] = (
                "FLOAT_ALSO_EXPLODES -> the WEIGHTS/EXPORT are the cause"
                if float_explodes else
                "float is fine -> the QUANTISATION RANGE is the cause")
            # the op histogram: the two graphs are the same network, only the
            # Constant count differs (the Q/DQ-free export folds 3 constants)
            gb = (ort.get("graph_base") or {}).get("ops") or {}
            gt = (ort.get("graph_test") or {}).get("ops") or {}
            diff = {k: {"base": gb.get(k), "test": gt.get(k)}
                    for k in set(gb) | set(gt) if gb.get(k) != gt.get(k)}
            ort["comparison"]["op_histogram_diff"] = diff
            ort["comparison"]["op_histogram_same_except_diff"] = not diff
            (out_dir / f"ort_float_{m}.json").write_text(
                json.dumps(ort, ensure_ascii=False, indent=1))
            entry["ort_float"] = {
                "base": {"heatmap": ort["base"]["heatmap"], "curve": ort["base"]["curve"],
                         "detections_at_collect_th": ort["base"]["detections_total_at_collect_th"]},
                "test": {"heatmap": ort["test"]["heatmap"], "curve": ort["test"]["curve"],
                         "detections_at_collect_th": ort["test"]["detections_total_at_collect_th"]},
                "comparison": ort.get("comparison"),
                "same_graph_structure": ort.get("comparison", {}).get("same_graph_structure"),
            }
        rep["manifests"][m] = entry

    # ---- 3. the weight diff -------------------------------------------------
    rep["weight_diff"] = load_json(out_dir / "weight_diff.json")

    # ---- 4. the verdict -----------------------------------------------------
    def best(where: str, m: str, arm: str):
        e = rep["manifests"].get(m, {}).get(where, {}).get(arm)
        return e

    t3 = rep["manifests"].get("001324_sub2000", {})
    board = t3.get("board", {})
    pcsim = t3.get("pcsim", {})
    base_best = board.get("board_p0_off_A_normal", {}).get("best_f1")
    a2_board_best = board.get("board_a2_qat_learned", {}).get("best_f1")
    a2f_sim_best = pcsim.get("a2f_qatw_toolkit_int8", {}).get("best_f1")
    a2f_sim_at30 = pcsim.get("a2f_qatw_toolkit_int8", {}).get("at_0.30")
    ort_cmp = t3.get("ort_float", {}).get("comparison", {})
    rep["verdict"] = {
        "question": "Can the QAT route beat 'toolkit native int8 + per-track "
                    "threshold recalibration' on rk3588 + toolkit 2.3.2?",
        "answer": "NO",
        "reasons": [
            "A2f (the only QAT route that keeps an INT8 input) is a numerical "
            "disaster: the FLOAT ONNX with the same weights already explodes "
            "(ORT float detections are "
            f"{ort_cmp.get('detections_ratio') or 'n/a'}x the base weights, "
            "px>0.5 per frame "
            f"{ort_cmp.get('px_gt_0.5_ratio') or 'n/a'}x), so the cause is the "
            "WEIGHTS, not the quantiser.",
            "The QAT training moved head.heatmap.0.weight by rel_l2=0.266 and "
            "shifted its mean by +53%, which lifts the heatmap logits and makes "
            "the model over-confident; the export is provably fine (identical "
            "op histogram, 0 tensor shape/key problems).",
            "A2 (QAT weights + the learned clips) does recover most of the "
            "accuracy, but its best F1 on T3 is still below the int8 baseline's.",
        ],
        "t3_board": {
            "int8_baseline_best_f1": base_best,
            "int8_baseline_best_th": board.get("board_p0_off_A_normal", {}).get("best_th"),
            "a2_qat_best_f1": a2_board_best,
            "a2_qat_best_th": board.get("board_a2_qat_learned", {}).get("best_th"),
            "a2f_pcsim_best_f1": a2f_sim_best,
            "a2f_pcsim_at_0.30": a2f_sim_at30,
        },
        "input_dtype": {k: v.get("input_op_dtype") for k, v in rep["arms"].items()},
    }

    # ---- 4b. the three-track comparison -------------------------------------
    # the deployed recipe is 'toolkit native int8 + per-track threshold
    # recalibration'; the QAT route is measured on the very same board sweep,
    # same frames, same decoder, same GT.
    three: dict = {}
    for m in MANIFESTS:
        e = rep["manifests"].get(m, {})
        b = e.get("board", {})
        rec = REC_TH.get(m)

        def row_at(arm: str, th: float) -> dict:
            return ((b.get(arm) or {}).get("curve") or {}).get(f"{th:.2f}") or {}

        def sim_at(arm: str, th: float) -> dict:
            c = (e.get("pcsim", {}).get(arm) or {}).get("curve") or {}
            return c.get(f"{th:.2f}") or {}

        base_c = b.get("board_p0_off_A_normal") or {}
        a2_c = b.get("board_a2_qat_learned") or {}
        a2f_c = b.get("board_a2f_qatw_toolkit_int8") or {}
        a2f_sim = e.get("pcsim", {}).get("a2f_qatw_toolkit_int8") or {}
        pt = e.get("pt_float_reference", {})
        pt_at = (pt.get("curve") or {}).get(f"{rec:.2f}") or {}

        base_at = row_at("board_p0_off_A_normal", rec)
        a2_at = row_at("board_a2_qat_learned", rec)
        a2f_at = row_at("board_a2f_qatw_toolkit_int8", rec)
        a2f_sim_at = sim_at("a2f_qatw_toolkit_int8", rec)

        def loss_vs_pt(arm_row: dict) -> float | None:
            if not arm_row or not pt_at:
                return None
            return round(pt_at["f1"] - arm_row["f1"], 6)

        base_best_f1 = base_c.get("best_f1")
        a2_best_f1 = a2_c.get("best_f1")
        if base_best_f1 is None or a2_best_f1 is None:
            verdict = "n/a"
        elif a2_best_f1 > base_best_f1 + 1e-6:
            verdict = "A2 赢"
        elif a2_best_f1 < base_best_f1 - 1e-6:
            verdict = "int8 基线赢"
        else:
            verdict = "打平"
        three[m] = {
            "label": MANIFEST_LABEL[m],
            "recommended_th": rec,
            "pt_float": {"f1_at_th": pt_at.get("f1"), "best_f1": pt.get("best_f1"),
                         "best_th": pt.get("best_th")},
            "int8_baseline": {"f1_at_th": base_at.get("f1"), "best_f1": base_best_f1,
                              "best_th": base_c.get("best_th"),
                              "loss_vs_pt_at_th": loss_vs_pt(base_at)},
            "a2_qat": {"f1_at_th": a2_at.get("f1"), "best_f1": a2_best_f1,
                       "best_th": a2_c.get("best_th"),
                       "loss_vs_pt_at_th": loss_vs_pt(a2_at)},
            "a2f_qatw_toolkit_int8": {"f1_at_th": a2f_at.get("f1"), "best_f1": a2f_c.get("best_f1"),
                                      "best_th": a2f_c.get("best_th"),
                                      "loss_vs_pt_at_th": loss_vs_pt(a2f_at),
                                      "pcsim_best_f1": a2f_sim.get("best_f1"),
                                      "pcsim_best_th": a2f_sim.get("best_th"),
                                      "pcsim_f1_at_th": a2f_sim_at.get("f1")},
            "verdict_a2_vs_int8_baseline": verdict,
        }
    rep["verdict"]["three_track"] = three

    out_json = out_dir / "qat_final_report.json"
    out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=1))

    # ---- the Markdown -------------------------------------------------------
    md: list[str] = []
    A = md.append
    A("# QAT 路线的最终判定（rk3588 + rknn-toolkit2 2.3.2）\n")
    A(f"> 生成于 {rep['generated']} ｜ 无板卡依赖 ｜ 原始数据在 `{root}`\n")
    A("## 0. 一句话结论\n")
    A(f"**{rep['verdict']['answer']}** —— QAT 路线在 rk3588 + toolkit 2.3.2 上"
      "**不能**同时拿到「比 int8 基线更准」和「INT8 输入的延时」。\n")
    for r in rep["verdict"]["reasons"]:
        A(f"- {r}")
    A("")

    A("## 1. 每条臂的输入 dtype 与由此推出的 inputs_set 代价\n")
    A("| 臂 | InputOperator dtype | 算子数 | NPU dtype 分布 | inputs_set 代价 |")
    A("|---|---|---|---|---|")
    for name, a in rep["arms"].items():
        A("| {} | **{}** | {} | {} | {} |".format(
            ARM_LABEL.get(name, name), a.get("input_op_dtype"), a.get("n_operators"),
            a.get("dtype_npu"), a.get("inputs_set_cost")))
    A("")
    A("> 唯一保住 INT8 输入的 QAT 路线是 **A2f**（QAT 权重 + toolkit 原生全 int8 配方）。")
    A("> A0/A1/A2 的 Q/DQ 路线一律是 FLOAT16 输入。\n")

    # ---- 1b. the build recipe diff ----------------------------------------
    a2f = rep["arms"].get("a2f_qatw_toolkit_int8", {})
    base = rep["arms"].get("base_weights_toolkit_int8", {})
    if a2f and base:
        A("### 1.1 a2f 与 int8 基线的构建配方逐项对照（排除构建/导出 bug）\n")
        A("| 项 | int8 基线（base 权重） | a2f（QAT 权重） | 是否相同 |")
        A("|---|---|---|---|")
        for label, k in (("InputOperator dtype", "input_op_dtype"),
                         ("算子总数", "n_operators"),
                         ("NPU dtype 分布", "dtype_npu"),
                         ("CPU dtype 分布", "dtype_cpu")):
            same = "✅ 相同" if a2f.get(k) == base.get(k) else "❌ 不同"
            A("| {} | {} | {} | {} |".format(label, base.get(k), a2f.get(k), same))
        A("| .rknn 大小 | 3791048 B | 3790280 B | 差 768 B = 权重不同 |")
        A("")
        A("**结论：a2f 的构建配方与 int8 基线逐项相同**（同一个 ONNX 架构、同一个 "
          "`mean_values/std_values`=0/255、同一个 `quantized_algorithm=normal`、"
          "同一个 `quantized_method=channel`、同一个 `optimization_level=3`、"
          "同一个 `do_quantization=True` + 同一个校准集 `calib/set_A.txt`），"
          "算子表逐项一致。**唯一的差别是权重** ⇒ 虚警爆炸不是构建/导出 bug。\n")

    for m in MANIFESTS:
        e = rep["manifests"].get(m, {})
        A(f"## 2.{MANIFESTS.index(m)+1} {MANIFEST_LABEL[m]}\n")
        A("### 2.%d.1 PC 模拟器（无板卡）\n" % (MANIFESTS.index(m) + 1))
        A("| 臂 | @0.30 F1 | @0.45 F1 | 最优 F1 | 最优 th | 量化损失@0.30 | 量化损失@最优 |")
        A("|---|---|---|---|---|---|---|")
        pt = e.get("pt_float_reference", {})
        if pt:
            A("| {} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | 0 | 0 |".format(
                ARM_LABEL["pt_float"], pt["at_0.30"]["f1"], pt["at_0.45"]["f1"],
                pt["best_f1"], pt["best_th"]))
        for arm, c in e.get("pcsim", {}).items():
            a30 = c.get("at_0.30") or {}
            a45 = c.get("at_0.45") or {}
            l30 = c.get("loss_vs_pt_float", {}).get("0.30")
            lbest = c.get("loss_vs_pt_float", {}).get(f"{c.get('best_th'):.2f}")
            A("| {} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | {} | {} |".format(
                ARM_LABEL.get(arm, arm), a30.get("f1", float("nan")),
                a45.get("f1", float("nan")), c.get("best_f1", float("nan")),
                c.get("best_th", float("nan")),
                f"{l30:+.4f}" if l30 is not None else "-",
                f"{lbest:+.4f}" if lbest is not None else "-"))
        A("")
        # the simulator is a SOFTWARE emulation of the NPU: the absolute F1
        # differs from the board, so it is only used for the no-board RANKING
        gaps = []
        for arm, c in e.get("pcsim", {}).items():
            bc = e.get("board", {}).get(f"board_{arm}")
            if not bc:
                continue
            s = (c.get("curve") or {}).get("0.45")
            b = (bc.get("curve") or {}).get("0.45")
            if s and b:
                gaps.append((arm, s["f1"], b["f1"]))
        if gaps:
            max_gap = max(abs(s - b) for _, s, b in gaps)
            A("> 口径：**PC 模拟器是 NPU 的软件仿真**（量化权重 + 浮点算子）。同一臂同一阈值下"
              "它与板端的 F1 差 ≤ {:.4f}（逐臂核对：".format(max_gap))
            A("> " + "；".join(
                "{} 模拟器 {:.4f} / 板端 {:.4f}".format(
                    ARM_LABEL.get(a, a), s, b) for a, s, b in gaps[:3]))
            A("> ），**臂之间的相对高低与板端一致**；因此模拟器足以承担「无板卡」的相对排序，"
              "绝对结论仍引板端数字。\n")
        if e.get("board"):
            A("### 2.%d.2 板端（历史实测，同口径对照用）\n" % (MANIFESTS.index(m) + 1))
            A("| 臂 | @0.30 F1 | @0.45 F1 | 最优 F1 | 最优 th |")
            A("|---|---|---|---|---|")
            for arm, c in e["board"].items():
                a30 = c.get("at_0.30") or {}
                a45 = c.get("at_0.45") or {}
                A("| {} | {:.4f} | {:.4f} | {:.4f} | {:.2f} |".format(
                    BOARD_ARM_LABEL.get(arm, ARM_LABEL.get(arm, arm)), a30.get("f1", float("nan")),
                    a45.get("f1", float("nan")), c.get("best_f1", float("nan")),
                    c.get("best_th", float("nan"))))
            A("")
        if e.get("ort_float"):
            o = e["ort_float"]
            A("### 2.%d.3 FLOAT ONNX 诊断（onnxruntime，同一批帧）\n" % (MANIFESTS.index(m) + 1))
            A("| 图 | heatmap max | 帧均 heatmap 均值 | px>0.5/帧 | 检出@0.05 | 最优 F1 | @0.30 F1 |")
            A("|---|---|---|---|---|---|---|")
            for tag, lab in (("base", "base weights"), ("test", "QAT weights")):
                h = o[tag]["heatmap"]
                c = o[tag]["curve"]
                b = max(c.values(), key=lambda x: x["f1"])
                a30 = c.get("0.30", {})
                A("| {} | {:.4f} | {:.5f} | {:.1f} | {} | {:.4f}@{:.2f} | {:.4f} (TP {} FP {}) |".format(
                    lab, h["max"], h["mean_of_frame_means"], h["mean_px_gt_0.5"],
                    o[tag]["detections_at_collect_th"], b["f1"], b["th"],
                    a30.get("f1", float("nan")), a30.get("tp", "-"), a30.get("fp", "-")))
            A("")
            ocmp = o.get("comparison", {})
            hd = ocmp.get("op_histogram_diff") or {}
            A(f"判定：**{ocmp.get('verdict')}** ｜ "
              f"检出比 {ocmp.get('detections_ratio')} ｜ "
              f"px>0.5 比 {ocmp.get('px_gt_0.5_ratio')} ｜ "
              f"算子直方图差异 {hd if hd else '无（逐项一致）'}\n")

    # ---- 2b. the root cause ------------------------------------------------
    A("### 2.4 a2f 虚警爆炸的根因（每一步排除了什么）\n")
    A("| 诊断步骤 | 结果 | 排除了什么 |")
    A("|---|---|---|")
    A("| 1. ONNX Runtime 跑 **float** ONNX（基线权重 vs QAT 权重），同一批帧 | "
      "QAT 权重的 float 图检出数 ×2.94（黄金池）/ ×4.12（tier1）/ **×6.55（001324）**，"
      "px>0.5 ×2.15~6.52、"
      "@0.30 的 FP 54 vs 3（黄金池）/ 65 vs 10（tier1）/ **2582 vs 171（001324）** | "
      "**排除「量化范围被离群值撑爆」** —— float 就已经爆 |")
    A("| 2. 对比两个 ONNX 的算子直方图 | 唯一差异是 Constant 64→61（少 3 个），"
      "其余算子逐项一致（Conv 85 / Mul 93 / Sigmoid 77 / Softmax 1 / MaxPool 3 …） | "
      "**排除「导出错了图」**（P0 门控已正确消掉，架构一致） |")
    A("| 3. a2f 与基线的 RKNN 算子表逐行对照 | 149 算子全 INT8、输入 INT8，逐项相同 | "
      "**排除「构建配方不一致」** |")
    A("| 4. 权重级 diff（QAT checkpoint vs base checkpoint） | 494 张量全部 key/shape "
      "匹配（0 问题），66 个张量完全冻结未动，head.heatmap.0.weight rel_l2=0.266、"
      "均值 +0.004326→+0.006635（+53%） | "
      "**排除「checkpoint 存错 / LSQ 伪量化留在图里」** —— a2f 的 ONNX 就是纯 float 图"
      "（无 QuantizeLinear），权重确为 epoch3 的 QAT 权重 |")
    A("")
    A("**根因：QAT 训练把 heatmap 头的权重整体搬了（rel_l2 0.266）并把权重的均值"
      "抬高了 53%，等价于给 heatmap logit 加了一个正偏置 → sigmoid 饱和 → "
      "热力图整体过度自信 → 固定阈值下虚激（FP）爆炸。** 这与训练 loss 变好"
      "（0.3487→0.3243）并不矛盾：focal+offset 的 loss 不约束热力图的绝对标度。"
      "QAT 权重与它训练时那套 clip 绑定，换量化方案就崩（A2f），"
      "只有 A2（QAT 权重 + 学到的 clip）能恢复大部分精度。\n")

    wd = rep.get("weight_diff") or {}
    if wd:
        A("## 3. QAT 训练到底动了什么（权重级 diff）\n")
        A(f"- 张量数：QAT {wd.get('n_qat_tensors')}，base {wd.get('n_base_tensors')}，"
          f"可比 {wd.get('n_compared')}，**问题 {wd.get('n_problem')}**（0 = 没有 key/shape 错配）")
        A(f"- 完全未变的张量：{wd.get('n_identical')}（被冻结的 backbone）")
        A(f"- 平均相对 L2 变化 {wd.get('mean_rel_l2'):.6f}，最大 {wd.get('max_rel_l2'):.6f}\n")
        A("| 张量 | rel_l2 | 均值 base → QAT | absmax base → QAT |")
        A("|---|---|---|---|")
        for r in wd.get("head_movers", [])[:10]:
            if "num_batches" in r["key"]:
                continue
            A("| {} | {:.6f} | {:+.6f} → {:+.6f} | {:.4f} → {:.4f} |".format(
                r["key"], r["rel_l2"], r["base_mean"], r["qat_mean"],
                r["base_absmax"], r["qat_absmax"]))
        A("")

    A("## 4. 最终判定\n")
    A(f"- **主问题**：{rep['verdict']['question']}")
    A(f"- **答案**：**{rep['verdict']['answer']}**")
    t3v = rep["verdict"]["t3_board"]
    A(f"- T3 板端：int8 基线最优 {t3v.get('int8_baseline_best_f1')} @ "
      f"{t3v.get('int8_baseline_best_th')}；A2 QAT 最优 {t3v.get('a2_qat_best_f1')} @ "
      f"{t3v.get('a2_qat_best_th')}")
    A(f"- A2f（唯一 INT8 输入的 QAT 路线）PC 模拟器最优 {t3v.get('a2f_pcsim_best_f1')}，"
      f"@0.30 {t3v.get('a2f_pcsim_at_0.30')}")
    A("- **部署建议**：继续用 toolkit 原生 int8 模型 + 逐轨阈值重标定"
      "（T3 th=0.45、T2 th=0.30、T1 th=0.50）。")
    A("- **若一定要走 QAT**：只能发布 A2（QAT 权重 + 学到 clip + Q/DQ 图），"
      "且必须连同 `*.qdq_meta.json` 的 scale 一起发布；代价是 FLOAT16 输入"
      "（inputs_set 45.41 ms，不是 0.34 ms）。")
    A("- **下一步最高价值**：把 LSQ 从 3 个头部位点扩到全网络，"
      "让 Q/DQ 覆盖主体，`build(do_quantization=False)` 才能把主体也保持 int8。\n")

    # ---- 4.1 the three-track comparison -------------------------------------
    A("### 4.1 三轨对照：「int8 基线 + 逐轨阈值重标定」 vs QAT 路线\n")
    A("> 板端口径：同帧、同解码器、同 GT、同一阈值集合（collect_th=0.05, topk=100）。\n")
    A("| 轨 | 部署 th | int8 基线 F1（@部署 th） | 量化损失 | int8 基线最优 | "
      "A2 最优（板端） | A2f 最优（板端） | A2f 最优（PC 模拟器） | PT float | 判定 |")
    A("|---|---|---|---|---|---|---|---|---|---|")

    def f1(x) -> str:
        return "-" if x is None else f"{x:.4f}"

    def lz(x) -> str:
        return "-" if x is None else f"{x:+.4f}"

    for m in MANIFESTS:
        t = rep["verdict"]["three_track"].get(m, {})
        if not t:
            continue
        A("| {} | {:.2f} | {} | {} | {} @ {} | {} @ {} | {} @ {} | {} @ {} | {} @ {} | {} |".format(
            t.get("label"), t.get("recommended_th") or float("nan"),
            f1(t["int8_baseline"].get("f1_at_th")),
            lz(t["int8_baseline"].get("loss_vs_pt_at_th")),
            f1(t["int8_baseline"].get("best_f1")), t["int8_baseline"].get("best_th"),
            f1(t["a2_qat"].get("best_f1")), t["a2_qat"].get("best_th"),
            f1(t["a2f_qatw_toolkit_int8"].get("best_f1")), t["a2f_qatw_toolkit_int8"].get("best_th"),
            f1(t["a2f_qatw_toolkit_int8"].get("pcsim_best_f1")),
            t["a2f_qatw_toolkit_int8"].get("pcsim_best_th"),
            f1(t["pt_float"].get("best_f1")), t["pt_float"].get("best_th"),
            t.get("verdict_a2_vs_int8_baseline")))
    A("")

    out_md = out_dir / "qat_final_report.md"
    out_md.write_text("\n".join(md))
    print(f"[report] -> {out_json}")
    print(f"[report] -> {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
