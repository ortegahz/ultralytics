#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""配准 / 歧义处置族：**oracle 上界 + 可分性分析**（夜 8 收口用）。

夜 7 结案了「位移修正族」（升阶运动模型：无处可施 + 方向为负 + 归因错误），
剩下**唯一未结案**的战场是配准 / 歧义处置维度（``replay_xcorr_ambiguous`` 8218 帧 /
``replay_xcorr_rematch_rejected`` 7127 帧 / ``replay_xcorr_abstained`` 779 帧 /
``replay_rollback_abstained`` 3836 帧），缺口预算 1.3748 pp。
本脚本沿用夜 7 的「**先上界后实现**」方法论：

阶段 A（``--dump``）
    在**交付档 CFG** 上逐序列跑 replay 臂，用 monkey-patch 插桩记录每次 xcorr
    匹配的**相关图 top-K 候选峰**。生产 ``xcorr.match()`` 只返回 ``matchTemplate``
    的全局 argmax ⇒ 候选峰根本拿不到 ⇒ 配准族的上界必须自己算。
    **生产代码零改动**；每次匹配跑一道**逐位等价闸门**（复刻的 argmax 位移 / 峰值 /
    次峰 / 唯一性必须与生产 ``match()`` 完全一致），任何一次不等价 ⇒ 立刻中止。

阶段 B（``--analyze``）
    帧级算 oracle 上界、缺口分桶、以及「可计算排序规则」的可分性。

口径：24 序官方口径 · cs_t0043 (no-GMC) · Δ=3(120 ms) · replay 臂 + 出框门 m=6。
hit@5px 口径**照抄** ``extrapolator.servo_metrics``：**帧级**，存在某个上报点到
最近 GT 中心的欧氏距离 ≤ 5px 即算命中，分母 = 有 GT 的帧数。
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.tracking.lookahead import extrapolator  # noqa: E402
from manu.tracking.lookahead import xcorr as xcorr_mod  # noqa: E402
from manu.tracking.lookahead.lookahead_longquan import (  # noqa: E402
    build_frame_source,
    build_lookahead_config,
    build_records,
    load_params,
    run_system_sota,
)

SCRIPT_REL = "manu/tracking/lookahead/diag_reg_oracle.py"
DEFAULT_CACHE = "/mnt/data/siping/runs/nogmc_t3/swa/ms_cache_24seq"
DEFAULT_LABELS = "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
DEFAULT_FRAMES = "/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg"
DEFAULT_FRAMES_CACHE = "/mnt/data/siping/frame_cache_lq"
DEFAULT_PARAMS = "manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json"
DEFAULT_DUMP_DIR = PROJECT_ROOT / "runs/lookahead/fi_nogmc24/reg_oracle"
DEFAULT_OUT_JSON = PROJECT_ROOT / "runs/lookahead/fi_nogmc24/reg_oracle/reg_oracle.json"
DEFAULT_OUT_MD = PROJECT_ROOT / "runs/lookahead/fi_nogmc24/reg_oracle/reg_oracle.md"
DELIVERY_SH = PROJECT_ROOT / "manu/tracking/lookahead/run_delivery.sh"
IMG_H = 512
HIT_TOL = 5.0
TOP_K = 8


# --------------------------------------------------------------------------- #
# 交付档 CFG：唯一真值 = run_delivery.sh 的 CFG= 赋值块（多行续行，单行 grep 会截断）
# --------------------------------------------------------------------------- #
def read_delivery_cfg() -> str:
    """从 ``run_delivery.sh`` 取完整 ``CFG=`` 赋值块并 eval（口径与扫描脚本一致）。"""
    text = DELIVERY_SH.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith('CFG="'):
            start = index
            break
    if start is None:
        raise SystemExit(f"[FATAL] {DELIVERY_SH} 里找不到 CFG= 赋值")
    block = [lines[start]]
    for line in lines[start + 1 :]:
        block.append(line)
        if line.rstrip().endswith('"'):
            break
    namespace: dict = {}
    exec("\n".join(block), namespace)  # noqa: S102 - 只 eval 本仓交付脚本里的字面量赋值
    cfg = namespace.get("CFG", "")
    if not cfg:
        raise SystemExit("[FATAL] CFG 解析为空")
    for must in (
        "--replay-roi-selfpred-max-px 16",
        "--replay-rollback-min-peak 0.90",
        "--replay-polarity auto",
        "--replay-xcorr-uniq-gate 0.6",
        "--replay-mode perframe",
        "--replay-abstain-peak-below 0.30",
        "--replay-rollback-max-frames 16",
    ):
        if f" {must} " not in f" {cfg} ":
            raise SystemExit(f"[FATAL] CFG 缺少必需旋钮 '{must}' ⇒ 解析被截断，拒绝跑")
    return cfg


# --------------------------------------------------------------------------- #
# 阶段 A：插桩（生产代码零改动）
# --------------------------------------------------------------------------- #
def replica_match(dst_img, template, anchor, center, radius, tsize, origin, polarity):
    """复刻 ``xcorr.match()`` 的口径，**额外**返回相关图的 top-K 候选峰。

    复刻的每一行都照抄生产实现（``xcorr.py:108-186``），只多了 NMS；
    等价性由调用方逐次核对（argmax 位移 / 峰值 / 次峰 / 唯一性）。

    返回 ``None``（生产口径：窗取不到 / 模板比窗大）或
    ``{"disp", "peak", "second", "uniq", "peaks"}``；``peaks`` 每项是
    ``{"x", "y", "v", "u"}``：**绝对上报坐标** + 峰值 + 该峰自身的局部唯一性。
    """
    import cv2

    if int(polarity) < 0:
        template = -template
    size = 2 * int(radius) + int(tsize)
    search, (ox, oy) = xcorr_mod.window(dst_img, float(center[0]), float(center[1]), size)
    if search is None:
        return None
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return None
    res = cv2.matchTemplate(search.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
    _, max_value, _, max_loc = cv2.minMaxLoc(res)
    px, py = max_loc
    dx = dy = 0.0
    if 0 < px < res.shape[1] - 1:
        a = float(res[py, px - 1])
        b = float(res[py, px])
        c = float(res[py, px + 1])
        den = a - 2 * b + c
        if abs(den) > 1e-9:
            dx = 0.5 * (a - c) / den
    if 0 < py < res.shape[0] - 1:
        a = float(res[py - 1, px])
        b = float(res[py, px])
        c = float(res[py + 1, px])
        den = a - 2 * b + c
        if abs(den) > 1e-9:
            dy = 0.5 * (a - c) / den
    if origin is None:
        origin = (
            float(np.round(float(anchor[0]))) - int(tsize) // 2,
            float(np.round(float(anchor[1]))) - int(tsize) // 2,
        )

    def _abs(qx: float, qy: float) -> tuple[float, float]:
        return (
            float(anchor[0]) + (ox + qx - origin[0]),
            float(anchor[1]) + (oy + qy - origin[1]),
        )

    matched_x, matched_y = _abs(px + dx, py + dy)
    displacement = np.asarray(
        [matched_x - float(anchor[0]), matched_y - float(anchor[1])], dtype=np.float64
    )
    mask = np.ones_like(res, dtype=bool)
    half = max(1, int(tsize))
    mask[
        max(0, py - half) : min(res.shape[0], py + half + 1),
        max(0, px - half) : min(res.shape[1], px + half + 1),
    ] = False
    second = float(res[mask].max()) if mask.any() else 0.0
    uniq = second / float(max_value) if max_value > 1e-6 else float("nan")

    # ---- top-K NMS（排除框半径 = tsize，与生产 uniqueness 的排除框同口径）----
    work = res.copy()
    peaks: list[dict] = []
    for _ in range(int(TOP_K)):
        idx = int(np.argmax(work))
        qy, qx = divmod(idx, work.shape[1])
        value = float(work[qy, qx])
        if not np.isfinite(value):
            break
        ex = ey = 0.0
        if 0 < qx < res.shape[1] - 1:
            a = float(res[qy, qx - 1])
            b = float(res[qy, qx])
            c = float(res[qy, qx + 1])
            den = a - 2 * b + c
            if abs(den) > 1e-9:
                ex = 0.5 * (a - c) / den
        if 0 < qy < res.shape[0] - 1:
            a = float(res[qy - 1, qx])
            b = float(res[qy, qx])
            c = float(res[qy + 1, qx])
            den = a - 2 * b + c
            if abs(den) > 1e-9:
                ey = 0.5 * (a - c) / den
        mx, my = _abs(qx + ex, qy + ey)
        own = np.ones_like(res, dtype=bool)
        own[
            max(0, qy - half) : min(res.shape[0], qy + half + 1),
            max(0, qx - half) : min(res.shape[1], qx + half + 1),
        ] = False
        own_second = float(res[own].max()) if own.any() else 0.0
        own_uniq = own_second / value if value > 1e-6 else float("nan")
        peaks.append({"x": float(mx), "y": float(my), "v": value, "u": float(own_uniq)})
        work[
            max(0, qy - half) : min(res.shape[0], qy + half + 1),
            max(0, qx - half) : min(res.shape[1], qx + half + 1),
        ] = -np.inf
    return {
        "disp": displacement,
        "peak": float(max_value),
        "second": second,
        "uniq": float(uniq),
        "peaks": peaks,
    }


class Recorder:
    """一次插桩运行的全套状态（进程内，不落盘）。"""

    def __init__(self) -> None:
        self.gate_calls = 0
        self.gate_mismatch = 0
        self.mismatch_examples: list[str] = []
        self.refine_stack: list[dict] = []
        self.refine_log: list[dict] = []
        self.frame_refines: list[tuple[int, list[dict]]] = []


def install(rec: Recorder):
    """装上三块 monkey-patch（生产代码零改动）。"""
    original_match = extrapolator._xcorr_match
    original_refine = extrapolator._xcorr_refine
    original_replay = extrapolator._replay_detections

    def patched_match(
        dst_img,
        template,
        anchor,
        center,
        radius=48,
        tsize=24,
        origin=None,
        polarity=1,
        diagnostics=None,
    ):
        out = original_match(
            dst_img,
            template,
            anchor,
            center,
            radius=radius,
            tsize=tsize,
            origin=origin,
            polarity=polarity,
            diagnostics=diagnostics,
        )
        replica = replica_match(dst_img, template, anchor, center, radius, tsize, origin, polarity)
        rec.gate_calls += 1
        why = ""
        if (out is None) != (replica is None):
            why = "None 口径不一致"
        elif out is not None:
            if not np.allclose(replica["disp"], out[0], atol=1e-12, rtol=0.0):
                why = f"argmax 位移不一致 {replica['disp']} vs {out[0]}"
            elif float(replica["peak"]) != float(out[1]):
                why = f"峰值不一致 {replica['peak']} vs {out[1]}"
            else:
                prod_uniq = (diagnostics or {}).get("uniqueness")
                if prod_uniq is not None:
                    mine = replica["uniq"]
                    if math.isnan(float(prod_uniq)):
                        if not math.isnan(mine):
                            why = "唯一性 NaN 口径不一致"
                    elif math.isnan(mine) or abs(mine - float(prod_uniq)) > 1e-12:
                        why = f"唯一性不一致 {mine} vs {prod_uniq}"
                if not why and len(replica["peaks"]) >= 2:
                    if abs(replica["peaks"][1]["v"] - replica["second"]) > 1e-12:
                        why = "候选 2 峰值 != 生产 second"
        if why:
            rec.gate_mismatch += 1
            if len(rec.mismatch_examples) < 10:
                rec.mismatch_examples.append(why)
        call = {
            "c": [float(center[0]), float(center[1])],
            "pk": float(out[1]) if out is not None and out[1] is not None else 0.0,
            "u": (diagnostics or {}).get("uniqueness"),
            "d": None if out is None or out[0] is None else [float(out[0][0]), float(out[0][1])],
            "k": replica["peaks"] if replica is not None else [],
        }
        if rec.refine_stack:
            rec.refine_stack[-1]["m"].append(call)
        return out

    def patched_refine(frame_source, src_frame, dst_frame, anchor, chain_pos, cfg, result, polarity=1):
        ctx = {
            "s": int(src_frame),
            "d": int(dst_frame),
            "a": [float(anchor[0]), float(anchor[1])],
            "c": [float(chain_pos[0]), float(chain_pos[1])],
            "m": [],
        }
        rec.refine_stack.append(ctx)
        try:
            position = original_refine(
                frame_source, src_frame, dst_frame, anchor, chain_pos, cfg, result, polarity
            )
        finally:
            rec.refine_stack.pop()
        ctx["p"] = None if position is None else [float(position[0]), float(position[1])]
        rec.refine_log.append(ctx)
        return position

    def patched_replay(
        source_dets, frame, target_frame, cfg_, kf_states, tracker, frame_source_, dets_by_frame=None
    ):
        first = len(rec.refine_log)
        outcome = original_replay(
            source_dets, frame, target_frame, cfg_, kf_states, tracker, frame_source_, dets_by_frame
        )
        rec.frame_refines.append(
            (
                int(target_frame),
                rec.refine_log[first:],
                {
                    "failed": bool(outcome.get("failed")),
                    "reason": str(outcome.get("reason", "")),
                    "steps": int(outcome.get("steps", 0)),
                    "held": int(outcome.get("held", 0)),
                    "pe": [
                        [float(det["pos"][0]), float(det["pos"][1])]
                        for det in outcome.get("entries", [])
                    ],
                },
            )
        )
        return outcome

    extrapolator._xcorr_match = patched_match
    extrapolator._xcorr_refine = patched_refine
    extrapolator._replay_detections = patched_replay
    return original_match, original_refine, original_replay


def restore(original_match, original_refine, original_replay) -> None:
    extrapolator._xcorr_match = original_match
    extrapolator._xcorr_refine = original_refine
    extrapolator._replay_detections = original_replay


def build_argv(args: argparse.Namespace, cfg: str, sequences: list[str]) -> list[str]:
    """把交付档 CFG 交给**生产 CLI 解析器**（杜绝诊断脚本自己复刻参数）。"""
    argv = [
        "lookahead_longquan.py",
        "--cache-root",
        args.cache_root,
        "--labels-root",
        args.labels_root,
        "--frames-root",
        args.frames_root,
        "--params-json",
        args.params_json,
        "--lookahead",
        "replay",
        "--arms",
        "original,stale,kinematic,replay",
        "--delta",
        str(args.delta),
        "--replay-fov-margin-px",
        str(args.fov_margin),
    ]
    argv += cfg.split()
    if sequences:
        argv += ["--seqs", ",".join(sequences)]
    else:
        argv += ["--all-seqs"]
    return argv


def dump_sequence(args, cfg, parsed, sequence: str, rec: Recorder, out_path: Path) -> dict:
    """跑一遍交付档 replay 臂，把该序列的配准候选峰落盘。"""
    cache_root = Path(args.cache_root)
    labels_root = Path(args.labels_root)
    params = load_params(parsed.params_json)
    records = build_records(
        cache_root,
        labels_root,
        sequence,
        float(params["dedup_radius"]),
        0.0,
        int(params["persist_min_hits"]),
        int(params["persist_window"]),
        float(params["persist_tol"]),
    )
    dets, system_res = run_system_sota(records, params, img_h=IMG_H, capture_kf_states=True)
    frame_source = build_frame_source(
        parsed.frames_root,
        parsed.frames_cache or None,
        sequence,
        [int(record["frame_index"]) for record in records],
        int(parsed.delta),
    )
    if frame_source is None:
        raise SystemExit(f"[DUMP] {sequence} 缺帧")

    original_match, original_refine, original_replay = install(rec)
    try:
        report, _stats = extrapolator.build_report_stream(
            dets, len(records), cfg, kf_states=system_res.get("kf_states"), frame_source=frame_source
        )
    finally:
        restore(original_match, original_refine, original_replay)

    truth = {
        index: np.asarray(record["gt_bboxes"], dtype=np.float32).reshape(-1, 4)
        for index, record in enumerate(records)
    }
    refines_by_frame = {int(t): rs for t, rs, _meta in rec.frame_refines}
    meta_by_frame = {int(t): meta for t, _rs, meta in rec.frame_refines}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for index in range(len(records)):
            boxes = truth.get(index)
            gt = []
            if boxes is not None and np.asarray(boxes).size:
                for row in np.asarray(boxes, dtype=np.float64).reshape(-1, 4):
                    gt.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
            entries = [
                [float(det["pos"][0]), float(det["pos"][1])] for det in (report[index] if index < len(report) else [])
            ]
            handle.write(
                json.dumps(
                    {
                        "t": int(index),
                        "e": entries,
                        "g": gt,
                        "r": refines_by_frame.get(int(index), []),
                        "pe": meta_by_frame.get(int(index), {}).get("pe", []),
                        "f": meta_by_frame.get(int(index), {}).get("failed", False),
                        "rs": meta_by_frame.get(int(index), {}).get("reason", ""),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    return {
        "sequence": sequence,
        "frames": len(records),
        "gt_frames": sum(1 for i in range(len(records)) if truth.get(i) is not None and np.asarray(truth[i]).size),
        "reported_frames": sum(1 for i in range(len(records)) if report[i]),
        "refines": len(rec.refine_log),
    }


# --------------------------------------------------------------------------- #
# 阶段 B：分析
# --------------------------------------------------------------------------- #
def fmt(v, n=4, dash="—"):
    if v is None:
        return dash
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, (int, float)):
        return f"{v:.{n}f}"
    return str(v)


def _p50(values) -> float:
    arr = [v for v in (values or []) if v is not None and not math.isnan(float(v))]
    if not arr:
        return float("nan")
    return float(np.percentile(np.asarray(arr, dtype=np.float64), 50))


def _dist(point, centers) -> float:
    best = float("inf")
    for cx, cy in centers:
        d = math.hypot(float(point[0]) - float(cx), float(point[1]) - float(cy))
        if d < best:
            best = d
    return best


def _pools(refine: dict) -> dict:
    """候选池：big = 大匹配窗的 top-K 峰；rematch = 二次意见窗的 top-K 峰。"""
    matches = refine.get("m") or []
    big = matches[0] if matches else None
    rematch = matches[1] if len(matches) > 1 else None
    return {
        "big": list((big or {}).get("k") or []),
        "rematch": list((rematch or {}).get("k") or []),
        "big_u": (big or {}).get("u"),
        "big_pk": float((big or {}).get("pk") or 0.0),
    }


def _is_ambiguous(refine: dict, uniq_gate: float) -> bool:
    uni = _pools(refine)["big_u"]
    if uni is None or math.isnan(float(uni)):
        return False
    return float(uni) > uniq_gate


def _nearest(candidates, reference) -> list | None:
    if not candidates or reference is None:
        return None
    best = min(candidates, key=lambda p: math.hypot(p["x"] - float(reference[0]), p["y"] - float(reference[1])))
    return [float(best["x"]), float(best["y"])]


def _rule_point(rule: str, refine: dict, uniq_gate: float) -> list | None:
    """按一条「可计算排序规则」给这个 refine 选一个上报坐标（None = 沿用原位置）。

    所有规则都只用**当帧可得**的量（候选峰的坐标 / 峰值 / 该峰自身的唯一性比 /
    链位置 / 锚点位置 / 本次匹配的唯一性比），**不用 GT** ⇒ 「可计算」是硬约束。
    """
    pools = _pools(refine)
    big = pools["big"]
    rematch = pools["rematch"]
    chain = refine.get("c")
    anchor = refine.get("a")
    fallback = refine.get("p") if refine.get("p") is not None else chain
    argmax = [big[0]["x"], big[0]["y"]] if big else fallback
    ambiguous = _is_ambiguous(refine, uniq_gate)

    if rule == "R0_baseline":
        return fallback
    if rule == "R1_argmax":
        return argmax
    if rule == "R2_chain":
        return chain
    if rule == "R3_second_peak":
        return [big[1]["x"], big[1]["y"]] if len(big) > 1 else fallback
    if rule == "R4_ambig_to_chain":
        return chain if ambiguous else argmax
    if rule == "R5_uniq_gate_peak":
        allowed = [p for p in big if p.get("u") is not None and not math.isnan(float(p["u"])) and float(p["u"]) <= uniq_gate]
        if not allowed:
            return fallback
        best = max(allowed, key=lambda p: float(p["v"]))
        return [best["x"], best["y"]]
    if rule == "R6_peak_x_uniq":
        if not big:
            return fallback

        def score(p):
            u = p.get("u")
            u = 0.0 if u is None or math.isnan(float(u)) else float(u)
            return float(p["v"]) * (1.0 - u)

        best = max(big, key=score)
        return [best["x"], best["y"]]
    if rule == "R7_ambig_to_rematch":
        if ambiguous and rematch:
            return [rematch[0]["x"], rematch[0]["y"]]
        return argmax
    # ---- 夜 8 新增：用「链 / 锚点」给候选峰**排序**（不是直接用链位置）----
    if rule == "R8_nearest_to_anchor":
        return _nearest(big, anchor) or fallback
    if rule == "R9_nearest_to_chain":
        return _nearest(big, chain) or fallback
    if rule == "R10_ambig_nearest_to_anchor":
        if not ambiguous:
            return argmax
        return _nearest(big, anchor) or fallback
    if rule == "R11_ambig_nearest_to_chain":
        if not ambiguous:
            return argmax
        return _nearest(big, chain) or fallback
    if rule == "R12_ambig_lowest_uniq":
        if not ambiguous or not big:
            return argmax
        def own_u(p):
            u = p.get("u")
            return 1.0 if u is None or math.isnan(float(u)) else float(u)

        best = min(big, key=own_u)
        return [best["x"], best["y"]]
    if rule == "R13_nearest_to_chain_all":
        return _nearest(big + rematch, chain) or fallback
    if rule == "R14_ambig_nearest_to_chain_all":
        if not ambiguous:
            return argmax
        return _nearest(big + rematch, chain) or fallback
    raise ValueError(f"unknown rule {rule}")


def _inbox_hit(point, gt_rows) -> bool:
    """业务匹配器口径（``count_matches`` mode="inbox"）：点落在某个 GT 框内即算命中。"""
    if not gt_rows:
        return False
    for row in gt_rows:
        if abs(float(point[0]) - float(row[0])) <= float(row[2]) / 2.0 and abs(float(point[1]) - float(row[1])) <= float(row[3]) / 2.0:
            return True
    return False


def _fov_legal(point, margin: float = 6.0, width: float = 640.0, height: float = 512.0) -> bool:
    """出框门（``_apply_fov_margin_gate``）的判定：上报点必须落在画面内且距边界 ≥ margin。"""
    if point is None:
        return False
    x, y = float(point[0]), float(point[1])
    return margin <= x <= width - margin and margin <= y <= height - margin


PK_BINS = ((0.00, 0.25), (0.25, 0.40), (0.40, 0.50), (0.50, 0.60),
           (0.60, 0.70), (0.70, 0.80), (0.80, 0.90))


def _pk_bin(peak: float) -> str:
    for lo, hi in PK_BINS:
        if lo <= peak < hi:
            return f"[{lo:.2f},{hi:.2f})"
    return f"[{PK_BINS[-1][1]:.2f},1.00)"


def _silent_cause(
    frame: dict,
    delta: int,
    rollback_min_peak: float = 0.90,
    margin: float = 6.0,
) -> str:
    """静默帧（最终 report 为空）的**生产口径**归因。

    必须按生产代码的**判定顺序**归因，否则会把「回退链峰值门丢的帧」误记成
    「出框门丢的帧」（夜 8 踩过这个坑）：

    1. ``_build_refine_report_stream`` 先查**回退链峰值门**
       （``rolled_back and peak < replay_rollback_min_peak``）⇒ 直接 ``continue``，
       **report 那一格根本不会被写** ⇒ 即使 outcome 自带 entries 也一样是静默帧；
    2. 再看 ``outcome["failed"]`` ⇒ 走 ``replay_fallback``（holdv）⇒ 通常**有**上报；
    3. 正常路径 ``report[target] = outcome["entries"]`` ⇒ 之后才过**出框门**与限幅。

    Returns
    -------
    str
        ``rollback_gate`` / ``fov_gate`` / ``no_decision`` / ``failed_no_fallback`` / ``other``
    """
    refines = frame.get("r") or []
    pre_entries = frame.get("pe") or []
    target = int(frame.get("t", -1))
    if not refines:
        return "failed_no_fallback" if frame.get("f") else "no_decision"
    last = refines[-1]
    rolled_back = any(int(r.get("s", target)) != target - delta for r in refines)
    matches = last.get("m") or []
    peak = float((matches[0] if matches else {}).get("pk") or 0.0)
    if rolled_back and peak < float(rollback_min_peak):
        return "rollback_gate"
    if pre_entries:
        if all(not _fov_legal(p, margin) for p in pre_entries):
            return "fov_gate"
        return "other"
    return "failed_no_fallback" if frame.get("f") else "other"


def _oracle_pick(refine: dict, centers, universe: str):
    """在候选全集里挑**离最近 GT 最近**的那个（oracle：用 GT 挑 ⇒ 上界）。

    返回 ``(point, pool, rank, peak)``：``rank`` = 该候选在**按峰高降序**的候选列表里的
    名次（0 = 就是 argmax ⇒ 基线已经选中它；≥1 ⇒ 必须换排序才能选中）。
    """
    pools = _pools(refine)
    candidates: list[tuple[str, dict]] = []
    # 候选集按**族**取：「reg」族 = 大匹配窗 top-K 候选峰；「all」族 = 再加链位置 + 二次意见窗的峰。
    # 「*_legal」**只是同一候选集再过出框门**，不是另一个候选集
    # ⇒ 必须先按族取候选、再统一过滤（夜 9 第 7 轮独立审计抓出：原来的
    #    `universe in ("reg", "all")` 把 "reg_legal"/"all_legal" 的大窗峰**漏掉**，
    #    候选集恒空 ⇒ 退化成 refine["p"]（基线选择），
    #    于是「合法 oracle」其实量的是**出框门前的基线**，不是 oracle）。
    _is_reg = universe.startswith("reg")
    _is_all = universe.startswith("all")
    if _is_reg or _is_all:
        candidates += [("big", p) for p in pools["big"]]
    if _is_all:
        chain = refine.get("c")
        if chain:
            candidates.append(("chain", {"x": float(chain[0]), "y": float(chain[1]), "v": float("nan"), "u": float("nan")}))
        candidates += [("rematch", p) for p in pools["rematch"]]
    if universe.endswith("_legal"):
        candidates = [
            item
            for item in candidates
            if _fov_legal([item[1]["x"], item[1]["y"]])
        ]
    fallback = refine.get("p") if refine.get("p") is not None else refine.get("c")
    if not candidates:
        return (fallback if fallback else [0.0, 0.0], "none", -1, float("nan"))
    best = min(candidates, key=lambda item: _dist([item[1]["x"], item[1]["y"]], centers))
    # 名次按**同一池内按峰高降序**排；跨池时用各自池内的名次
    pool = best[0]
    same_pool = [item[1] for item in candidates if item[0] == pool]
    same_pool_sorted = sorted(same_pool, key=lambda p: (float(p["v"]) if not math.isnan(float(p["v"])) else -1e18), reverse=True)
    rank = next((i for i, p in enumerate(same_pool_sorted) if p is best[1]), -1)
    return ([float(best[1]["x"]), float(best[1]["y"])], pool, rank, float(best[1]["v"]))


RULES = (
    "R0_baseline",
    "R1_argmax",
    "R2_chain",
    "R3_second_peak",
    "R4_ambig_to_chain",
    "R5_uniq_gate_peak",
    "R6_peak_x_uniq",
    "R7_ambig_to_rematch",
    "R8_nearest_to_anchor",
    "R9_nearest_to_chain",
    "R10_ambig_nearest_to_anchor",
    "R11_ambig_nearest_to_chain",
    "R12_ambig_lowest_uniq",
    "R13_nearest_to_chain_all",
    "R14_ambig_nearest_to_chain_all",
)

RULE_CN = {
    "R0_baseline": "基线（refine 返回 / 链位置）",
    "R1_argmax": "取 argmax 峰（= 生产的默认选择）",
    "R2_chain": "直接用链位置（不信配准）",
    "R3_second_peak": "盲目取窗内次峰",
    "R4_ambig_to_chain": "歧义时回退链位置",
    "R5_uniq_gate_peak": "唯一性门内取峰高最高",
    "R6_peak_x_uniq": "按 峰高×(1−唯一性) 排序",
    "R7_ambig_to_rematch": "歧义时用二次意见窗的 argmax",
    "R8_nearest_to_anchor": "候选峰里离锚点(t−Δ)最近的",
    "R9_nearest_to_chain": "候选峰里离链位置最近的",
    "R10_ambig_nearest_to_anchor": "歧义时取离锚点最近的候选峰",
    "R11_ambig_nearest_to_chain": "歧义时取离链位置最近的候选峰",
    "R12_ambig_lowest_uniq": "歧义时取自身唯一性最低的候选峰",
    "R13_nearest_to_chain_all": "候选峰(含二次意见窗)里离链位置最近的",
    "R14_ambig_nearest_to_chain_all": "歧义时取离链位置最近的候选峰(含二次意见窗)",
}


def analyze(args) -> dict:
    dump_dir = Path(args.dump_dir)
    files = sorted(p for p in dump_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"[FATAL] {dump_dir} 里没有 dump（*.jsonl）")
    uniq_gate = float(args.uniq_gate)
    radius = float(args.radius)

    totals = {
        "gt_frames": 0,
        "reported_frames": 0,
        "lost_frames": 0,
        "hit_base": 0,
        "hit_base_pre": 0,
        "hit_oracle_reg": 0,
        "hit_oracle_all": 0,
        "hit_oracle_reg_legal": 0,
        "hit_oracle_all_legal": 0,
        "frames_with_refine": 0,
        "refines": 0,
        "entries": 0,
        "entries_matched": 0,
        "entries_unmatched": 0,
        "silent_with_refine": 0,
        "lost_with_refine": 0,
    }
    rule_hits = {rule: 0 for rule in RULES}
    buckets = {"B1_window_has_peak": 0, "B2_window_no_peak": 0, "B3_out_of_window": 0, "B4_no_refine": 0}
    per_sequence: dict[str, dict] = {}
    ambiguous_frames = 0
    ambiguous_recoverable = 0
    anatomy = {
        "recoverable_frames": 0,
        "why_no_report": 0,
        "why_no_report_fov_dropped": 0,
        "why_no_report_abstained": 0,
        "why_refine_rejected": 0,
        "why_rematch_taken": 0,
        "why_report_eq_argmax": 0,
        "no_report_with_refine": 0,
        "no_report_abstain_gate": 0,
        "no_report_fov_band": 0,
        "no_report_neither": 0,
        "rank_hist": {},
        "pool_hist": {},
        "peak_ratio_p50": [],
        "argmax_uniq_p50": [],
        "picked_uniq_p50": [],
        "argmax_gate_pass": 0,
        "picked_gate_pass": 0,
        "picked_closer_to_chain": 0,
        "argmax_closer_to_chain": 0,
    }
    per_sequence_b1: dict[str, int] = {}
    policy = {
        "frames_total": 0,
        "silent_frames": 0,
        "silent_with_legal_argmax": 0,
        "silent_argmax_tp": 0,
        "silent_argmax_fp": 0,
        "reported_frames": 0,
        "reported_hit": 0,
        "reported_miss": 0,
        "legal_argmax_of_silent_but_hit": 0,
        "silent_peak_rejected": 0,
        "silent_dist_rejected": 0,
        "silent_chain_illegal": 0,
        "silent_chain_legal": 0,
        "silent_dist_rejected_chain_illegal": 0,
        "silent_dist_rejected_chain_legal": 0,
        "silent_dist_rejected_tp": 0,
        "silent_dist_rejected_fp": 0,
        "silent_peak_rejected_tp": 0,
        "silent_peak_rejected_fp": 0,
        "dist_reject_gap_p50": [],
        "silent_by_cause": {},
        "silent_cause_tp": {},
        "silent_cause_fp": {},
        "rollback_peak_hist": {},
    }

    for path in files:
        sequence = path.stem
        seq = {
            "gt_frames": 0,
            "hit_base": 0,
            "hit_oracle_reg": 0,
            "hit_oracle_all": 0,
            "hit_oracle_reg_legal": 0,
            "hit_oracle_all_legal": 0,
            "refines": 0,
        }
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            frame = json.loads(line)
            gt = frame.get("g") or []
            if not gt:
                policy["frames_total"] += 1
                continue
            centers = [(row[0], row[1]) for row in gt]
            totals["gt_frames"] += 1
            seq["gt_frames"] += 1
            entries = frame.get("e") or []
            pre_entries = frame.get("pe") or []
            refines = frame.get("r") or []
            totals["refines"] += len(refines)
            seq["refines"] += len(refines)
            if entries:
                totals["reported_frames"] += 1
            else:
                totals["lost_frames"] += 1
            if refines:
                totals["frames_with_refine"] += 1
                if not entries:
                    totals["lost_with_refine"] = totals.get("lost_with_refine", 0) + 1

            # ---- 基线（交付口径：出框门 + 限幅后的最终上报）----
            d_base = min((_dist(p, centers) for p in entries), default=float("inf"))
            hit_base = d_base <= HIT_TOL
            if hit_base:
                totals["hit_base"] += 1
                seq["hit_base"] += 1

            # ---- 门前的基线（refine 的原始返回 / 链位置）----
            pre_points = [
                (r["p"] if r.get("p") is not None else r.get("c")) for r in refines if r.get("p") or r.get("c")
            ]
            if pre_points and min(_dist(p, centers) for p in pre_points) <= HIT_TOL:
                totals["hit_base_pre"] += 1

            if not refines:
                if not entries:
                    buckets["B4_no_refine"] += 1
                continue

            # ---- 上报点 ↔ refine 关联（按门前的位置就近匹配）----
            matched: list[tuple[list, dict]] = []
            used = set()
            for pos in entries:
                best_ref = None
                best_gap = float("inf")
                for index, refine in enumerate(refines):
                    if index in used:
                        continue
                    base = refine.get("p") if refine.get("p") is not None else refine.get("c")
                    if base is None:
                        continue
                    gap = math.hypot(float(pos[0]) - float(base[0]), float(pos[1]) - float(base[1]))
                    if gap < best_gap:
                        best_gap = gap
                        best_ref = index
                if best_ref is None or best_gap > 12.0:
                    totals["entries_unmatched"] += 1
                    continue
                used.add(best_ref)
                matched.append((pos, refines[best_ref]))
            totals["entries"] += len(entries)
            totals["entries_matched"] += len(matched)

            # ---- 上报策略账（离线）：「无上报」的帧若改报 refine 的 argmax ----
            if not entries:
                policy["silent_frames"] += 1
                cause = _silent_cause(
                    frame,
                    int(args.delta),
                    rollback_min_peak=float(args.rollback_min_peak),
                    margin=float(args.fov_margin),
                )
                policy["silent_by_cause"][cause] = policy["silent_by_cause"].get(cause, 0) + 1
                if cause == "rollback_gate" and refines:
                    _matches = refines[-1].get("m") or []
                    _peak = float((_matches[0] if _matches else {}).get("pk") or 0.0)
                    _bin = policy["rollback_peak_hist"].setdefault(_pk_bin(_peak), {"n": 0, "hit": 0})
                    _bin["n"] += 1
                legal = None
                for refine in refines:
                    big = ((refine.get("m") or [{}])[0] or {}).get("k") or []
                    if big and _fov_legal([big[0]["x"], big[0]["y"]]):
                        legal = [big[0]["x"], big[0]["y"]]
                        break
                if legal is not None:
                    policy["silent_with_legal_argmax"] += 1
                    hit = _inbox_hit(legal, gt)
                    if hit:
                        policy["silent_argmax_tp"] += 1
                        policy["legal_argmax_of_silent_but_hit"] += 1
                        policy["silent_cause_tp"][cause] = policy["silent_cause_tp"].get(cause, 0) + 1
                    else:
                        policy["silent_argmax_fp"] += 1
                        policy["silent_cause_fp"][cause] = policy["silent_cause_fp"].get(cause, 0) + 1
                    if cause == "rollback_gate" and refines:
                        _bin["hit"] += 1 if hit else 0
                    # ---- 落在哪个否决分支？----
                    refine = refines[0]
                    big = ((refine.get("m") or [{}])[0] or {})
                    peak = float(big.get("pk") or 0.0)
                    chain = refine.get("c")
                    chain_legal = _fov_legal(chain) if chain else False
                    policy["silent_chain_legal" if chain_legal else "silent_chain_illegal"] += 1
                    if peak < 0.25:
                        policy["silent_peak_rejected"] += 1
                        policy["silent_peak_rejected_tp" if hit else "silent_peak_rejected_fp"] += 1
                    else:
                        policy["silent_dist_rejected"] += 1
                        policy["silent_dist_rejected_tp" if hit else "silent_dist_rejected_fp"] += 1
                        if chain_legal:
                            policy["silent_dist_rejected_chain_legal"] += 1
                        else:
                            policy["silent_dist_rejected_chain_illegal"] += 1
                        if chain is not None:
                            gap = math.hypot(legal[0] - float(chain[0]), legal[1] - float(chain[1]))
                            policy["dist_reject_gap_p50"].append(gap)

            # ---- oracle 上界（帧级：任一上报点能命中即算命中）----
            for universe, key in (
                ("reg", "hit_oracle_reg"),
                ("all", "hit_oracle_all"),
                ("reg_legal", "hit_oracle_reg_legal"),
                ("all_legal", "hit_oracle_all_legal"),
            ):
                points = []
                for _pos, refine in matched:
                    points.append(_oracle_pick(refine, centers, universe)[0])
                for index, refine in enumerate(refines):
                    if index not in used:
                        points.append(_oracle_pick(refine, centers, universe)[0])
                if points and min(_dist(p, centers) for p in points) <= HIT_TOL:
                    totals[key] += 1
                    seq[key] += 1

            # ---- 可计算排序规则（离线筛选；不含门/限幅的二次效应）----
            for rule in RULES:
                points = []
                for _pos, refine in matched:
                    point = _rule_point(rule, refine, uniq_gate)
                    if point is not None:
                        points.append(point)
                if points and min(_dist(p, centers) for p in points) <= HIT_TOL:
                    rule_hits[rule] += 1

            # ---- 缺口分桶（对基线失手帧）----
            if not hit_base:
                reachable = False
                has_peak = False
                for refine in refines:
                    chain = refine.get("c")
                    if chain is None:
                        continue
                    if _dist(chain, centers) <= radius:
                        reachable = True
                    for p in ((refine.get("m") or [{}])[0].get("k") or []):
                        if _dist([p["x"], p["y"]], centers) <= HIT_TOL:
                            has_peak = True
                if not reachable:
                    buckets["B3_out_of_window"] += 1
                elif has_peak:
                    buckets["B1_window_has_peak"] += 1
                    per_sequence_b1[sequence] = per_sequence_b1.get(sequence, 0) + 1
                else:
                    buckets["B2_window_no_peak"] += 1

                # ---- 可分性解剖：oracle 选中的那个候选峰长什么样 ----
                pick, pool, rank, pick_peak = _oracle_pick(refines[0], centers, "reg")
                if _dist(pick, centers) <= HIT_TOL:
                    pools = _pools(refines[0])
                    big = pools["big"]
                    anatomy["recoverable_frames"] += 1
                    # ---- 失手原因拆解（决定机理落在哪一层）----
                    argmax_pos = [big[0]["x"], big[0]["y"]] if big else None
                    refined = refines[0].get("p")
                    if not entries:
                        anatomy["why_no_report"] += 1
                        if pre_entries:
                            anatomy["why_no_report_fov_dropped"] = anatomy.get("why_no_report_fov_dropped", 0) + 1
                        else:
                            anatomy["why_no_report_abstained"] = anatomy.get("why_no_report_abstained", 0) + 1
                        peak = float(pools["big_pk"])
                        anchor = refines[0].get("a")
                        disp = (
                            math.hypot(argmax_pos[0] - float(anchor[0]), argmax_pos[1] - float(anchor[1]))
                            if argmax_pos and anchor
                            else 0.0
                        )
                        abstain = peak < float(args.abstain_peak_below) and disp > float(args.abstain_disp_px)
                        in_fov = (
                            argmax_pos is not None
                            and 6.0 <= argmax_pos[0] <= 640.0 - 6.0
                            and 6.0 <= argmax_pos[1] <= 512.0 - 6.0
                        )
                        anatomy["no_report_abstain_gate"] += 1 if abstain else 0
                        anatomy["no_report_fov_band"] += 0 if in_fov else 1
                        if not abstain and in_fov:
                            anatomy["no_report_neither"] += 1
                    elif refined is None:
                        anatomy["why_refine_rejected"] += 1
                    elif argmax_pos is not None and math.hypot(
                        float(refined[0]) - argmax_pos[0], float(refined[1]) - argmax_pos[1]
                    ) > 1.0:
                        # refine 返回的不是大匹配的 argmax ⇒ 采纳了二次意见
                        anatomy["why_rematch_taken"] += 1
                    else:
                        anatomy["why_report_eq_argmax"] += 1
                    anatomy["rank_hist"][str(int(rank))] = anatomy["rank_hist"].get(str(int(rank)), 0) + 1
                    anatomy["pool_hist"][pool] = anatomy["pool_hist"].get(pool, 0) + 1
                    if big:
                        argmax_peak = float(big[0]["v"])
                        if argmax_peak > 1e-6 and not math.isnan(pick_peak):
                            anatomy["peak_ratio_p50"].append(pick_peak / argmax_peak)
                        argmax_u = pools["big_u"]
                        if argmax_u is not None and not math.isnan(float(argmax_u)):
                            anatomy["argmax_uniq_p50"].append(float(argmax_u))
                        picked = next((p for p in big if p["x"] == pick[0] and p["y"] == pick[1]), None)
                        if picked is not None:
                            u = picked.get("u")
                            if u is not None and not math.isnan(float(u)):
                                anatomy["picked_uniq_p50"].append(float(u))
                                if float(u) <= uniq_gate:
                                    anatomy["picked_gate_pass"] += 1
                        if argmax_u is not None and not math.isnan(float(argmax_u)) and float(argmax_u) <= uniq_gate:
                            anatomy["argmax_gate_pass"] += 1
                        chain = refines[0].get("c")
                        if chain is not None:
                            d_arg = math.hypot(big[0]["x"] - float(chain[0]), big[0]["y"] - float(chain[1]))
                            d_pick = math.hypot(pick[0] - float(chain[0]), pick[1] - float(chain[1]))
                            if d_pick < d_arg:
                                anatomy["picked_closer_to_chain"] += 1
                            else:
                                anatomy["argmax_closer_to_chain"] += 1

            # ---- 歧义帧（uniq > 门限）里可被救回的帧数 ----
            for refine in refines:
                matches = refine.get("m") or []
                big = matches[0] if matches else None
                uni = (big or {}).get("u")
                if uni is None or math.isnan(float(uni)) or float(uni) <= uniq_gate:
                    continue
                ambiguous_frames += 1
                best = _oracle_pick(refine, centers, "all")[0]
                if _dist(best, centers) <= HIT_TOL and not hit_base:
                    ambiguous_recoverable += 1
                break
        per_sequence[sequence] = seq

    n = max(1, totals["gt_frames"])
    result = {
        "meta": {
            "script": SCRIPT_REL,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "delta": args.delta,
            "fov_margin": args.fov_margin,
            "uniq_gate": uniq_gate,
            "radius_px": radius,
            "hit_tol_px": HIT_TOL,
            "top_k": TOP_K,
        },
        "totals": totals,
        "buckets": buckets,
        "rule_hits": rule_hits,
        "ambiguous": {"frames": ambiguous_frames, "recoverable": ambiguous_recoverable},
        "anatomy": anatomy,
        "policy": policy,
        "per_sequence_b1": per_sequence_b1,
        "per_sequence": per_sequence,
    }
    result["derived"] = {
        "hit_base_pct": 100.0 * totals["hit_base"] / n,
        "hit_oracle_reg_pct": 100.0 * totals["hit_oracle_reg"] / n,
        "hit_oracle_all_pct": 100.0 * totals["hit_oracle_all"] / n,
        "hit_oracle_reg_legal_pct": 100.0 * totals["hit_oracle_reg_legal"] / n,
        "hit_oracle_all_legal_pct": 100.0 * totals["hit_oracle_all_legal"] / n,
        "oracle_reg_gain_pp": 100.0 * (totals["hit_oracle_reg"] - totals["hit_base"]) / n,
        "oracle_all_gain_pp": 100.0 * (totals["hit_oracle_all"] - totals["hit_base"]) / n,
        "oracle_reg_legal_gain_pp": 100.0 * (totals["hit_oracle_reg_legal"] - totals["hit_base"]) / n,
        "oracle_all_legal_gain_pp": 100.0 * (totals["hit_oracle_all_legal"] - totals["hit_base"]) / n,
        "rule_gain_pp": {
            rule: 100.0 * (rule_hits[rule] - totals["hit_base"]) / n for rule in RULES
        },
        "policy_account_biz": {
            "base_tp": args.base_tp,
            "base_fp": args.base_fp,
            "base_fn": args.base_fn,
            "base_f1": args.base_f1,
            "base_far": args.base_far,
            "total_frames": args.total_frames,
            "tp_gain": policy["silent_argmax_tp"],
            "fp_gain": policy["silent_argmax_fp"],
            "new_tp": args.base_tp + policy["silent_argmax_tp"],
            "new_fp": args.base_fp + policy["silent_argmax_fp"],
            "new_fn": args.base_fn - policy["silent_argmax_tp"],
            "fp_budget_to_redline": int(args.red_line * args.total_frames) - args.base_fp,
        },
        "policy_account": {
            "silent_frames": policy["silent_frames"],
            "silent_with_legal_argmax": policy["silent_with_legal_argmax"],
            "tp_gain": policy["silent_argmax_tp"],
            "fp_gain": policy["silent_argmax_fp"],
            "far_delta_pp": 100.0 * policy["silent_argmax_fp"] / max(1, policy["frames_total"]),
            "far_margin_frames": int(0.030 * policy["frames_total"]),
        },
        "rule_share_of_oracle_reg": {
            rule: (
                (rule_hits[rule] - totals["hit_base"]) / max(1, totals["hit_oracle_reg"] - totals["hit_base"])
            )
            for rule in RULES
        },
    }
    return result


def write_report(result: dict, out_md: Path, out_json: Path, args) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    totals = result["totals"]
    derived = result["derived"]
    meta = result["meta"]
    n = max(1, totals["gt_frames"])
    md: list[str] = []
    md.append("# 配准 / 歧义处置族：oracle 上界 + 可分性（夜 8）")
    md.append("")
    md.append(
        f"- 生成：{meta['generated']} · {meta['host']} · Δ={meta['delta']} · "
        f"出框门 m={meta['fov_margin']} · uniq 门 {meta['uniq_gate']} · 半径 {meta['radius_px']}px"
    )
    md.append(f"- 有 GT 的帧数：**{totals['gt_frames']}**（分母口径 = 有 GT 的帧）")
    md.append("")
    md.append("## 1. 基线复算闸门")
    md.append("")
    md.append("| 项 | 值 |")
    md.append("|:--|--:|")
    md.append(f"| 基线 hit@5px（出框门+限幅后，最终上报） | **{derived['hit_base_pct']:.4f}%** |")
    md.append(f"| 基线 hit@5px（门前：refine 返回 / 链位置） | {100.0 * totals['hit_base_pre'] / n:.4f}% |")
    md.append(f"| 有上报的帧 | {totals['reported_frames']} |")
    md.append(f"| 有 GT 但无上报（lost） | {totals['lost_frames']} |")
    md.append(f"| 上报点 ↔ refine 关联成功 / 失败 | {totals['entries_matched']} / {totals['entries_unmatched']} |")
    md.append("")
    md.append("## 2. oracle 上界")
    md.append("")
    md.append("| oracle | 含义 | hit@5px% | Δ vs 基线(pp) |")
    md.append("|:--|:--|--:|--:|")
    md.append(
        f"| 基线 | 交付档 replay 臂 | {derived['hit_base_pct']:.4f} | — |"
    )
    md.append(
        f"| **O-reg** | 只在**大匹配窗的 top-K 候选峰**里用 GT 挑 | {derived['hit_oracle_reg_pct']:.4f} "
        f"| **{derived['oracle_reg_gain_pp']:+.4f}** |"
    )
    md.append(
        f"| **O-all** | 候选峰 + 链位置 + 二次意见窗的峰 | {derived['hit_oracle_all_pct']:.4f} "
        f"| **{derived['oracle_all_gain_pp']:+.4f}** |"
    )
    md.append(
        f"| **O-reg 合法** | 同 O-reg，但候选必须**过出框门**（画面内且边距 ≥ {meta['fov_margin']}px） | "
        f"{derived['hit_oracle_reg_legal_pct']:.4f} | **{derived['oracle_reg_legal_gain_pp']:+.4f}** |"
    )
    md.append(
        f"| **O-all 合法** | 同 O-all，同样约束合法性 | {derived['hit_oracle_all_legal_pct']:.4f} "
        f"| **{derived['oracle_all_legal_gain_pp']:+.4f}** |"
    )
    md.append("")
    md.append("## 3. 基线失手帧的缺口分桶")
    md.append("")
    md.append("| 桶 | 帧数 | 含义 |")
    md.append("|:--|--:|:--|")
    md.append(f"| B1 窗内有峰 | {result['buckets']['B1_window_has_peak']} | 目标在搜索窗内且相关图有峰 ⇒ **可重排救回** |")
    md.append(f"| B2 窗内无峰 | {result['buckets']['B2_window_no_peak']} | 目标在窗内但相关图没峰 ⇒ 外观漂移，配准救不回 |")
    md.append(f"| B3 窗外 | {result['buckets']['B3_out_of_window']} | 目标不在搜索窗内 ⇒ 半径/中心问题（属另一族） |")
    md.append(f"| B4 无 refine | {result['buckets']['B4_no_refine']} | 该帧没有配准决策（链失败/无检测） |")
    md.append("")
    md.append("## 4. 可计算排序规则的可分性（离线筛选）")
    md.append("")
    md.append(
        "所有规则都只用**当帧可得**的量（候选峰的坐标 / 峰值 / 该峰自身的唯一性比 / "
        "链位置 / 锚点位置 / 本次匹配的唯一性比），**不用 GT** ⇒ 「可计算」是硬约束。"
    )
    md.append("")
    md.append("| 规则 | 含义 | hit@5px% | Δ vs 基线(pp) | 占 O-reg 的比例 |")
    md.append("|:--|:--|--:|--:|--:|")
    gain_reg = derived["oracle_reg_gain_pp"]
    for rule in RULES:
        pct = 100.0 * result["rule_hits"][rule] / n
        gain = derived["rule_gain_pp"][rule]
        ratio = (gain / gain_reg) if gain_reg > 1e-9 else float("nan")
        md.append(
            f"| `{rule}` | {RULE_CN.get(rule, '')} | {pct:.4f} | {gain:+.4f} | "
            f"{'—' if math.isnan(ratio) else f'{ratio * 100:.1f}%'} |"
        )
    md.append("")
    anat = result.get("anatomy", {})
    if anat.get("recoverable_frames"):
        md.append("## 5. 可分性解剖：oracle 选中的那个候选峰长什么样")
        md.append("")
        md.append(
            f"- 基线失手、但 O-reg 能救回的帧：**{anat['recoverable_frames']}**"
        )
        md.append(
            "- 该候选峰在**按峰高降序**的候选列表里的名次（0 = 就是 argmax ⇒ 基线已经选中它）："
        )
        md.append("")
        md.append("| 名次 | 帧数 |")
        md.append("|--:|--:|")
        for rank in sorted(anat.get("rank_hist", {}), key=lambda x: int(x)):
            md.append(f"| {rank} | {anat['rank_hist'][rank]} |")
        md.append("")
        md.append("| 项 | 值 |")
        md.append("|:--|--:|")
        md.append(f"| oracle 候选峰的峰高 / argmax 峰高（中位数） | {_p50(anat.get('peak_ratio_p50')):.4f} |")
        md.append(f"| argmax 的唯一性比（中位数） | {_p50(anat.get('argmax_uniq_p50')):.4f} |")
        md.append(f"| oracle 候选峰的唯一性比（中位数） | {_p50(anat.get('picked_uniq_p50')):.4f} |")
        md.append(
            f"| argmax 通过唯一性门（uniq ≤ 门限）的帧数 | {anat.get('argmax_gate_pass', 0)} / "
            f"{anat['recoverable_frames']} |"
        )
        md.append(
            f"| oracle 候选峰通过唯一性门的帧数 | {anat.get('picked_gate_pass', 0)} / "
            f"{anat['recoverable_frames']} |"
        )
        md.append(
            f"| oracle 候选峰比 argmax 更靠近链位置的帧数 | {anat.get('picked_closer_to_chain', 0)} / "
            f"{anat['recoverable_frames']} |"
        )
        md.append("")
    md.append("### 5.1 失手原因拆解（机理落在哪一层）")
    md.append("")
    md.append("| 原因 | 帧数 | 含义 |")
    md.append("|--:|--:|:--|")
    md.append(f"| 该帧**无任何上报** | {anat.get('why_no_report', 0)} | 弃权门 / 出框门 把本可命中的点丢了 |")
    md.append(f"| refine 被拒（回退链位置） | {anat.get('why_refine_rejected', 0)} | 配准结果被判「离链太远」而作废 |")
    md.append(f"| 采纳了二次意见（rematch） | {anat.get('why_rematch_taken', 0)} | 二次意见窗的 argmax 覆盖了大匹配 |")
    md.append(f"| 上报点 = argmax（仍失手） | {anat.get('why_report_eq_argmax', 0)} | 峰本身不在目标上 ⇒ 换选无用 |")
    md.append("")
    md.append(f"- 有 GT 但**无上报**且**有配准决策**的帧：**{totals.get('lost_with_refine', 0)}**")
    md.append(f"- 其中 argmax 会被**弃权门**（peak<{args.abstain_peak_below} 且 离锚点>{args.abstain_disp_px}px）拦下的：{anat.get('no_report_abstain_gate', 0)}")
    md.append(f"- 其中 argmax 落在**出框门**带外（640×512 画面、边距 {args.fov_margin}px）的：{anat.get('no_report_fov_band', 0)}")
    md.append(f"- 两者都不是（即：别的问题，比如链失败）：{anat.get('no_report_neither', 0)}")
    md.append("")
    md.append("## 6. 逐序列 B1（窗内有峰 ⇒ 配准族唯一有救的桶）")
    md.append("")
    md.append("| 序列 | B1 帧数 |")
    md.append("|:--|--:|")
    for sequence in sorted(result.get("per_sequence_b1", {}), key=lambda s: -result["per_sequence_b1"][s]):
        md.append(f"| {sequence} | {result['per_sequence_b1'][sequence]} |")
    md.append("")
    md.append("## 7. 上报策略账（离线）：「无上报」的帧若改报 refine 的 argmax")
    md.append("")
    pol = result.get("policy", {})
    acct = derived.get("policy_account", {})
    md.append("| 项 | 值 |")
    md.append("|:--|--:|")
    md.append(f"| 参与统计的帧（总数） | {pol.get('frames_total', 0)} |")
    md.append(f"| 其中**无任何上报**的帧 | {pol.get('silent_frames', 0)} |")
    md.append(f"| 其中 refine 的 argmax **合法**（过出框门）的帧 | {pol.get('silent_with_legal_argmax', 0)} |")
    md.append(f"| └ 改报后会变成 **TP**（命中 GT）的帧 | **{acct.get('tp_gain', 0)}** |")
    md.append(f"| └ 改报后会变成 **FP**（没命中）的帧 | **{acct.get('fp_gain', 0)}** |")
    biz = derived.get("policy_account_biz", {})
    new_tp = biz.get("new_tp", 0)
    new_fp = biz.get("new_fp", 0)
    new_fn = biz.get("new_fn", 0)
    prec = new_tp / max(1, new_tp + new_fp)
    rec = new_tp / max(1, new_tp + new_fn)
    f1_new = 2 * prec * rec / max(1e-12, prec + rec)
    far_new = new_fp / max(1, biz.get("total_frames", 1))
    md.append("")
    md.append("**否决分支拆解（这 2068 帧是被哪条规则吞掉的）：**")
    md.append("")
    md.append("| 分支 | 帧数 | 其中 TP | 其中 FP |")
    md.append("|:--|--:|--:|--:|")
    md.append(f"| refine 被**峰值门**否决（peak < 0.25） | {pol.get('silent_peak_rejected', 0)} | {pol.get('silent_peak_rejected_tp', 0)} | {pol.get('silent_peak_rejected_fp', 0)} |")
    md.append(f"| refine 被**离链太远**否决（|匹配−链| > 64px） | {pol.get('silent_dist_rejected', 0)} | {pol.get('silent_dist_rejected_tp', 0)} | {pol.get('silent_dist_rejected_fp', 0)} |")
    md.append(f"| └ 其中链位置**出框**（链已跑飞 ⇒ 否决参考失效） | {pol.get('silent_dist_rejected_chain_illegal', 0)} | — | — |")
    md.append(f"| └ 其中链位置**合法** | {pol.get('silent_dist_rejected_chain_legal', 0)} | — | — |")
    md.append(f"| 静默帧里链位置出框的总数 | {pol.get('silent_chain_illegal', 0)} | — | — |")
    md.append("")
    md.append("**按业务口径（`count_matches` mode=inbox，阈值 0）重算这笔账：**")
    md.append("")
    md.append("| 项 | 基线（交付档） | 改报后 | Δ |")
    md.append("|:--|--:|--:|--:|")
    md.append(f"| TP | {biz.get('base_tp')} | {new_tp} | **{biz.get('tp_gain', 0):+}** |")
    md.append(f"| FP | {biz.get('base_fp')} | {new_fp} | **{biz.get('fp_gain', 0):+}** |")
    md.append(f"| FN | {biz.get('base_fn')} | {new_fn} | **{-biz.get('tp_gain', 0):+}** |")
    md.append(f"| F1 | {fmt(biz.get('base_f1'), 6)} | {fmt(f1_new, 6)} | **{fmt((f1_new - biz.get('base_f1', 0)) * 100, 4)} pp** |")
    md.append(f"| FAR（FP/帧） | {fmt(biz.get('base_far'), 6)} | {fmt(far_new, 6)} | **{fmt((far_new - biz.get('base_far', 0)) * 100, 4)} pp** |")
    md.append(f"| **FAR 红线 0.030 的余量** | **{biz.get('fp_budget_to_redline', 0)} 帧 FP** | — | — |")
    md.append("")
    verdict_far = "**破线**" if far_new > 0.030 else "未破线"
    md.append(
        f"⇒ 若把「链跑飞」帧的配准 argmax 报出来：F1 {fmt((f1_new - biz.get('base_f1', 0)) * 100, 4)} pp、"
        f"FAR {fmt((far_new - biz.get('base_far', 0)) * 100, 4)} pp ⇒ FAR {verdict_far}"
        f"（红线 0.030，余量只有 {biz.get('fp_budget_to_redline', 0)} 帧 FP）。\n"
    )
    md.append("")
    md.append("## 8. 歧义帧（uniq > 门限）")
    md.append("")
    md.append(f"- 歧义 refine 数：**{result['ambiguous']['frames']}**")
    md.append(f"- 其中基线失手、但候选全集里有命中点：**{result['ambiguous']['recoverable']}**")
    md.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Registration / ambiguity family oracle upper bound.")
    parser.add_argument("--dump", action="store_true", help="run the instrumented replay arm and dump the candidates")
    parser.add_argument("--analyze", action="store_true", help="score the oracle bounds and the ranking rules")
    parser.add_argument("--cache-root", default=DEFAULT_CACHE)
    parser.add_argument("--labels-root", default=DEFAULT_LABELS)
    parser.add_argument("--frames-root", default=DEFAULT_FRAMES)
    parser.add_argument("--frames-cache", default=DEFAULT_FRAMES_CACHE)
    parser.add_argument("--params-json", default=DEFAULT_PARAMS)
    parser.add_argument("--dump-dir", type=Path, default=DEFAULT_DUMP_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--delta", type=int, default=3)
    parser.add_argument("--fov-margin", type=float, default=6.0)
    parser.add_argument("--uniq-gate", type=float, default=0.6)
    parser.add_argument("--radius", type=float, default=32.0)
    parser.add_argument("--abstain-peak-below", type=float, default=0.30)
    parser.add_argument("--abstain-disp-px", type=float, default=24.0)
    parser.add_argument("--rollback-min-peak", type=float, default=0.90)
    parser.add_argument("--base-tp", type=int, default=123730)
    parser.add_argument("--base-fp", type=int, default=4021)
    parser.add_argument("--base-fn", type=int, default=9527)
    parser.add_argument("--base-f1", type=float, default=0.9480935450254399)
    parser.add_argument("--base-far", type=float, default=0.024336805408447976)
    parser.add_argument("--total-frames", type=int, default=165210)
    parser.add_argument("--red-line", type=float, default=0.030)
    parser.add_argument("--seq", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dump:
        cfg_text = read_delivery_cfg()
        print(f"[CFG] {len(cfg_text)} 字符 · {cfg_text}", flush=True)
        import sys as _sys

        from manu.tracking.lookahead import lookahead_longquan

        argv = build_argv(args, cfg_text, args.seq or [])
        backup = _sys.argv
        _sys.argv = argv
        try:
            parsed = lookahead_longquan.parse_args()
        finally:
            _sys.argv = backup
        cfg = build_lookahead_config(parsed, "replay")
        sequences = args.seq or sorted(p.stem for p in Path(args.cache_root).glob("*.pkl"))
        out_dir = Path(args.dump_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "host": socket.gethostname(),
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cache_root": args.cache_root,
            "delta": args.delta,
            "fov_margin": args.fov_margin,
            "sequences": [],
        }
        total_calls = 0
        total_mismatch = 0
        for sequence in sequences:
            rec = Recorder()
            t0 = time.time()
            info = dump_sequence(args, cfg, parsed, sequence, rec, out_dir / f"{sequence}.jsonl")
            elapsed = time.time() - t0
            total_calls += rec.gate_calls
            total_mismatch += rec.gate_mismatch
            info.update(
                {
                    "elapsed_s": round(elapsed, 1),
                    "gate_calls": rec.gate_calls,
                    "gate_mismatch": rec.gate_mismatch,
                    "mismatch_examples": rec.mismatch_examples,
                }
            )
            summary["sequences"].append(info)
            print(
                f"[DUMP] {sequence} frames={info['frames']} refines={info['refines']} "
                f"gate={rec.gate_calls} mismatch={rec.gate_mismatch} {elapsed:.1f}s",
                flush=True,
            )
            if rec.gate_mismatch:
                print(f"[FATAL] {sequence} 等价闸门失败：{rec.mismatch_examples}", flush=True)
                summary["gate_mismatch_examples"] = rec.mismatch_examples
                (out_dir / "dump_summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                raise SystemExit(2)
        summary["gate_calls_total"] = total_calls
        summary["gate_mismatch_total"] = total_mismatch
        (out_dir / "dump_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[DONE] {len(sequences)} 序列 · 等价闸门 {total_calls} 次调用 / {total_mismatch} 次不等价 "
            f"⇒ 产物在 {out_dir}",
            flush=True,
        )
        return
    if args.analyze:
        result = analyze(args)
        write_report(result, Path(args.out_md), Path(args.out_json), args)
        derived = result["derived"]
        print(
            f"[结论摘要] 基线 hit@5px {derived['hit_base_pct']:.4f}% ⇒ O-reg "
            f"{derived['hit_oracle_reg_pct']:.4f}% ({derived['oracle_reg_gain_pp']:+.4f}pp) / O-all "
            f"{derived['hit_oracle_all_pct']:.4f}% ({derived['oracle_all_gain_pp']:+.4f}pp) ⇒ "
            f"产物 {args.out_md} + {args.out_json}",
            flush=True,
        )
        return
    raise SystemExit("必须给 --dump 或 --analyze")


if __name__ == "__main__":
    main()
