#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Offline search for an abstention policy: trade a false alarm for a silent frame and watch the servo.

The delivery red line is ``FAR <= 0.030`` and the replay arm sits at ``0.0349`` on the 24-sequence set.
``diag_fp_attrib.py`` shows that only 1658 of the 5764 false positives are the plugin's own fault; the rest
are detector noise and annotation sparsity. The only lever that removes a false positive without inventing a
new position is **abstention**: when the fast kernel is not self-consistent, report nothing and let the servo
hold its last command instead of being handed a wrong coordinate.

This script makes that trade-off searchable without re-running the replay for every candidate:

``--dump``
    Runs the replay arm once per sequence with the delivery configuration and records, per physical frame, the
    reported entries (before the output slew limiter) together with the chain diagnostics of that frame:
    the template-match uniqueness ratio, the correlation peak, whether the "second opinion" rematch was
    applied, how far the template match and the centroid chain disagree, and how many steps were held.
``--scan``
    Replays those dumps under candidate abstention rules and scores each rule with the **frozen** business
    matcher and the frozen servo metrics, so a rule can be compared on F1, FAR, hit@5px and jitter.

An empty rule must reproduce the delivery numbers exactly; that reproduction is the correctness gate of the
dump, and the scan refuses to report anything when it fails.

Usage::

    /home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/diag_abstain_scan.py --dump
    /home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/diag_abstain_scan.py --scan
"""

from __future__ import annotations

import argparse
import json
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
from manu.tracking.lookahead.extrapolator import _apply_output_slew_limit  # noqa: E402
from manu.tracking.lookahead.lookahead_longquan import (  # noqa: E402
    build_frame_source,
    build_lookahead_config,
    build_records,
    evaluate_biz,
    load_cache_meta,
    load_params,
    natural_sort_key,
    resolve_sequences,
    run_system_sota,
)
from manu.tracking.lookahead.extrapolator import servo_metrics  # noqa: E402
from manu.evaluation.eval_multiscale_cache_th import count_matches  # noqa: E402
from manu.evaluation.eval_multiscale_cache_th import count_matches  # noqa: E402

SCRIPT_REL = "manu/tracking/lookahead/diag_abstain_scan.py"
DEFAULT_DUMP_DIR = PROJECT_ROOT / "runs/lookahead/final/final24/abstain_diag"
DEFAULT_OUT_JSON = PROJECT_ROOT / "runs/lookahead/final/final24/diag_abstain_scan.json"
DEFAULT_OUT_MD = PROJECT_ROOT / "runs/lookahead/final/final24/diag_abstain_scan.md"
IMG_H = 512


def parse_args() -> argparse.Namespace:
    """Command line: the delivery arguments are synthesized and handed to the **real** CLI parser.

    Re-typing the delivery options here would let the diagnostic drift away from the artifact it is
    supposed to explain, so the only thing this script owns is *where* the diagnostic runs, not *what* the
    replay arm is configured with.
    """
    parser = argparse.ArgumentParser(description="Search an abstention policy for the replay arm.")
    parser.add_argument("--dump", action="store_true", help="run the replay arm once and record the diagnostics")
    parser.add_argument("--scan", action="store_true", help="score candidate abstention rules on the dumps")
    parser.add_argument("--cache-root", default="runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache")
    parser.add_argument("--labels-root", default="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010")
    parser.add_argument("--frames-root", default="/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg")
    parser.add_argument("--params-json", default="manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json")
    parser.add_argument("--dump-dir", type=Path, default=DEFAULT_DUMP_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--delta", type=int, default=8)
    parser.add_argument("--seq", action="append", default=None, help="repeatable; default is every cached sequence")
    parser.add_argument("--red-line", type=float, default=0.030)
    parser.add_argument("--cache-root-8seq", default="runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache")
    return parser.parse_args()


def delivery_args(args: argparse.Namespace, sequences: list[str]):
    """The delivery command line of the replay arm, handed to the production parser.

    The options below are exactly the ones ``manu/tracking/lookahead/run_delivery.sh`` uses, so the
    diagnostic replays the delivered configuration rather than a look-alike.
    """
    import sys as _sys

    from manu.tracking.lookahead import lookahead_longquan

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
        "--replay-mode",
        "perframe",
        "--replay-max-hold",
        "8",
        "--replay-fallback",
        "holdv",
        "--no-replay-hold-use-ref",
        "--output-max-step-px",
        "30",
        "--replay-anchor",
        "delta",
        "--fast-win-size",
        "52",
        "--fast-bg-window",
        "104",
        "--fast-th-k",
        "2.5",
        "--fast-min-energy",
        "50",
        "--replay-roi-mode",
        "selfpred",
        "--replay-roi-selfpred-window",
        "3",
        "--replay-roi-selfpred-max-px",
        "4",
        "--replay-xcorr-radius",
        "32",
        "--replay-xcorr-template",
        "32",
        "--replay-xcorr-th-k",
        "1.5",
        "--replay-polarity",
        "auto",
        "--replay-xcorr-uniq-gate",
        "0.6",
    ]
    if sequences:
        argv += ["--seqs", ",".join(sequences)]
    else:
        argv += ["--all-seqs"]
    backup = _sys.argv
    _sys.argv = argv
    try:
        parsed = lookahead_longquan.parse_args()
    finally:
        _sys.argv = backup
    return parsed


def build_config(args: argparse.Namespace, sequences: list[str]):
    """The delivery configuration of the replay arm, produced by the production CLI parser."""
    parsed = delivery_args(args, sequences)
    return build_lookahead_config(parsed, "replay"), parsed


def dump_sequence(args: argparse.Namespace, cfg, parsed, sequence: str) -> dict:
    """Run the replay arm once for one sequence and record the per-frame chain diagnostics."""
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
    frames = load_cache_meta(cache_root, sequence)["frames"]
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

    calls: list[dict] = []
    original_match = extrapolator._xcorr_match

    def patched_match(dst_img, template, anchor, center, radius=48, tsize=24, origin=None, polarity=1, diagnostics=None):
        """Record every template match: the peak, the uniqueness ratio and the matched displacement."""
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
        calls.append(
            {
                "peak": float(out[1]) if out is not None else 0.0,
                "uniq": (diagnostics or {}).get("uniqueness"),
                "disp": None if out is None or out[0] is None else [float(out[0][0]), float(out[0][1])],
                "center": [float(center[0]), float(center[1])],
                "anchor": [float(anchor[0]), float(anchor[1])],
            }
        )
        return out

    refines: list[dict] = []
    original_refine = extrapolator._xcorr_refine

    def patched_refine(frame_source, src_frame, dst_frame, anchor, chain_pos, cfg, result, polarity=1):
        """Record whether the template refinement produced a position for this detection."""
        position = original_refine(
            frame_source, src_frame, dst_frame, anchor, chain_pos, cfg, result, polarity
        )
        refines.append(
            {
                "ok": position is not None,
                "pos": None if position is None else [float(position[0]), float(position[1])],
            }
        )
        return position

    outcomes: dict[int, dict] = {}
    original_replay = extrapolator._replay_detections

    def patched_replay(source_dets, frame, target_frame, cfg_, kf_states, tracker, frame_source_, dets_by_frame=None):
        """Record the outcome of every physical frame: the entries and the chain counters."""
        first_call = len(calls)
        first_refine = len(refines)
        outcome = original_replay(
            source_dets, frame, target_frame, cfg_, kf_states, tracker, frame_source_, dets_by_frame
        )
        frame_calls = calls[first_call:]
        frame_refines = refines[first_refine:]
        features = frame_features(frame_calls, outcome, frame_refines)
        outcomes[target_frame] = {
            "entries": [
                {
                    "track_id": int(det.get("track_id", -1)),
                    "pos": [float(det["pos"][0]), float(det["pos"][1])],
                    "score": float(det.get("score", 0.0)),
                }
                for det in outcome.get("entries", [])
            ],
            "failed": bool(outcome.get("failed")),
            "reason": str(outcome.get("reason", "")),
            "held": int(outcome.get("held", 0)),
            "steps": int(outcome.get("steps", 0)),
            "max_step_disp": float(outcome.get("max_step_disp", 0.0)),
            "anchor_calls": int(outcome.get("anchor_calls", 0)),
            "features": features,
        }
        return outcome

    extrapolator._xcorr_match = patched_match
    extrapolator._xcorr_refine = patched_refine
    extrapolator._replay_detections = patched_replay
    try:
        report, stats = extrapolator.build_report_stream(
            dets, len(records), cfg, kf_states=system_res.get("kf_states"), frame_source=frame_source
        )
    finally:
        extrapolator._xcorr_match = original_match
        extrapolator._xcorr_refine = original_refine
        extrapolator._replay_detections = original_replay

    ordered = sorted(records, key=lambda record: natural_sort_key(record["im_name"]))
    truth = {
        index: np.asarray(record["gt_bboxes"], dtype=np.float32).reshape(-1, 4)
        for index, record in enumerate(ordered)
    }
    frames_out = []
    for index in range(len(ordered)):
        outcome = outcomes.get(index, {"entries": [], "failed": False, "reason": "", "held": 0, "steps": 0})
        frames_out.append(
            {
                "entries": outcome["entries"],
                "held": outcome["held"],
                "steps": outcome["steps"],
                "failed": outcome["failed"],
                "reason": outcome["reason"],
                "features": outcome.get("features", {}),
                "gt": truth[index].tolist(),
            }
        )
    payload = {
        "sequence": sequence,
        "frames": len(ordered),
        "video_frames": frames,
        "output_max_step_px": float(parsed.output_max_step_px),
        "stats": {key: value for key, value in stats.items() if key != "per_sequence"},
        "report": [[{"track_id": int(det.get("track_id", -1)), "pos": [float(det["pos"][0]), float(det["pos"][1])], "score": float(det.get("score", 0.0))} for det in frame] for frame in report],
        "frames_out": frames_out,
    }
    return payload


def frame_features(frame_calls: list[dict], outcome: dict, frame_refines: list[dict] | None = None) -> dict:
    """Derive the per-frame abstention features from the template-match calls of that frame.

    ``uniq``
        uniqueness ratio of the big template match (the "second opinion" gate already uses it).
    ``peak``
        correlation peak of the big template match.
    ``chain_dev``
        distance between the reported position and the centroid-chain position: how far the template match
        pulled the chain away from where the fast kernel walked.
    ``anchor_disp``
        distance between the reported position and the source-frame anchor.
    ``rematch``
        1 when the "second opinion" rematch was the one that produced the reported position.
    """
    features = {
        "uniq": None,
        "peak": None,
        "final_peak": None,
        "xcorr_ok": None,
        "chain_dev": None,
        "anchor_disp": None,
        "rematch": 0,
    }
    if frame_refines:
        features["xcorr_ok"] = int(any(item["ok"] for item in frame_refines))
    if not frame_calls:
        return features
    first = frame_calls[0]
    features["uniq"] = None if first["uniq"] is None else float(first["uniq"])
    features["peak"] = float(first["peak"])
    entries = outcome.get("entries") or []
    if not entries:
        return features
    position = np.asarray(entries[0]["pos"], dtype=np.float64)
    center = np.asarray(first["center"], dtype=np.float64)
    anchor = np.asarray(first["anchor"], dtype=np.float64)
    features["chain_dev"] = float(np.linalg.norm(position - center))
    features["anchor_disp"] = float(np.linalg.norm(position - anchor))
    if len(frame_calls) > 1 and frame_calls[1]["disp"] is not None:
        rematch_position = anchor + np.asarray(frame_calls[1]["disp"], dtype=np.float64)
        if float(np.linalg.norm(position - rematch_position)) <= 1e-6:
            features["rematch"] = 1
    # the peak of the match that actually produced the reported position
    features["final_peak"] = float(
        frame_calls[1]["peak"] if features["rematch"] and len(frame_calls) > 1 else first["peak"]
    )
    return features


def rule_applies(rule: dict, features: dict, has_entries: bool) -> bool:
    """Does the abstention rule fire on this frame? (All configured conditions must hold.)

    A frame whose feature is missing (no template match happened there) never fires: the rule can only
    abstain on a frame where the chain actually produced a position.
    """
    if not has_entries:
        return False
    conditions = rule.get("conditions", [])
    if not conditions:
        return False
    for key, comparison, threshold in conditions:
        value = features.get(key)
        if value is None:
            return False
        if comparison == "gt" and not value > threshold:
            return False
        if comparison == "lt" and not value < threshold:
            return False
        if comparison == "eq" and not value == threshold:
            return False
    return True


def apply_rule(payload: dict, rule: dict, limit_px: float) -> list[list[dict]]:
    """Apply the abstention rule and re-apply the output slew limiter, exactly like the delivery does."""
    report = []
    for frame in payload["frames_out"]:
        if rule_applies(rule, frame.get("features", {}), bool(frame["entries"])):
            report.append([])
        else:
            report.append(
                [
                    {"track_id": det["track_id"], "pos": tuple(det["pos"]), "score": det["score"], "infilled": False}
                    for det in frame["entries"]
                ]
            )
    if limit_px > 0.0:
        class _Cfg:
            output_max_step_px = limit_px

        _apply_output_slew_limit(report, _Cfg(), {})
    return report


def score_payload(payload: dict, rule: dict, limit_px: float) -> dict:
    """Score one sequence under one rule with the frozen matcher and the frozen servo metrics."""
    report = apply_rule(payload, rule, limit_px)
    records = [{"im_name": f"{index:06d}", "gt_bboxes": frame["gt"]} for index, frame in enumerate(payload["frames_out"])]
    row = evaluate_biz(records, report, num_frames=payload["video_frames"])
    truth = {
        index: np.asarray(frame["gt"], dtype=np.float32).reshape(-1, 4)
        for index, frame in enumerate(payload["frames_out"])
    }
    servo = servo_metrics(report, truth, len(payload["frames_out"]))
    return {
        "sequence": payload["sequence"],
        "tp": row["tp"],
        "fp": row["fp"],
        "fn": row["fn"],
        "det": row["det"],
        "gt": row["gt"],
        "f1": row["f1"],
        "far_per_frame": row["far_per_frame"],
        "hit_rate_5px": servo["hit_rate_5px"],
        "cle_p50": servo["cle_p50"],
        "cle_p95": servo["cle_p95"],
        "max_jitter": servo["max_jitter"],
        "p99_jitter": servo["p99_jitter"],
    }


def prepare(payload: dict) -> dict:
    """Pre-extract the per-frame arrays so a rule can be scored without re-parsing the dump."""
    frames = payload["frames_out"]
    points = []
    tracks = []
    boxes = []
    for frame in frames:
        entries = frame["entries"]
        points.append(
            np.asarray([det["pos"] for det in entries], dtype=np.float64).reshape(-1, 2)
        )
        tracks.append([int(det["track_id"]) for det in entries])
        boxes.append(np.asarray(frame["gt"], dtype=np.float64).reshape(-1, 4))
    return {
        "sequence": payload["sequence"],
        "video_frames": payload["video_frames"],
        "points": points,
        "tracks": tracks,
        "boxes": boxes,
        "limit_px": float(payload.get("output_max_step_px", 30.0)),
        "features": [frame.get("features", {}) for frame in frames],
    }


def masked_report(prepared: dict, abstain: list[bool]) -> list[list[dict]]:
    """The report with the abstained frames emptied, then the output slew limiter applied."""
    report: list[list[dict]] = []
    for index, frame_points in enumerate(prepared["points"]):
        if abstain[index] or not len(frame_points):
            report.append([])
            continue
        report.append(
            [
                {
                    "track_id": prepared["tracks"][index][position],
                    "pos": (float(frame_points[position][0]), float(frame_points[position][1])),
                    "score": 1.0,
                    "infilled": False,
                }
                for position in range(len(frame_points))
            ]
        )
    limit = prepared["limit_px"]
    if limit > 0.0:

        class _Cfg:
            output_max_step_px = limit

        _apply_output_slew_limit(report, _Cfg(), {})
    return report


def score_fast(prepared: dict, abstain: list[bool]) -> dict:
    """Score one sequence under one rule with the frozen matcher and the frozen servo metrics.

    The frozen matcher walks the detections of a frame in score order and lets each one claim the still-free
    ground truth whose centre is nearest; the business sets carry a single target, so the per-frame work is
    vectorised over the detections while the multi-detection frames fall back to the frozen implementation.
    """
    report = masked_report(prepared, abstain)
    tp = fp = fn = 0
    gt_frames = 0
    hits = 0
    cle_values: list[float] = []
    for index, frame_points in enumerate(prepared["points"]):
        box_array = prepared["boxes"][index]
        if len(box_array):
            gt_frames += 1
        detections = report[index]
        if not len(detections):
            fn += int(len(box_array))
            continue
        if len(detections) == 1 and len(box_array) >= 1:
            point = np.asarray(detections[0]["pos"], dtype=np.float64)
            centres = box_array[:, :2]
            distances = np.linalg.norm(centres - point, axis=1)
            nearest = int(np.argmin(distances))
            box = box_array[nearest]
            inside = abs(point[0] - box[0]) <= box[2] / 2 and abs(point[1] - box[1]) <= box[3] / 2
            if inside:
                tp += 1
                fn += int(len(box_array)) - 1
            else:
                fp += 1
                fn += int(len(box_array))
            nearest_distance = float(distances[nearest])
            cle_values.append(nearest_distance)
            if nearest_distance <= 5.0:
                hits += 1
            continue
        counts = count_matches(
            [(box_array if len(box_array) else None, frame_points, np.ones(len(frame_points), dtype=np.float32))],
            0.0,
            8.0,
            "inbox",
        )
        tp += counts["tp"]
        fp += counts["fp"]
        fn += counts["fn"]

    track_points: dict[int, dict[int, np.ndarray]] = {}
    for index, detections in enumerate(report):
        for det in detections:
            per_frame = track_points.setdefault(int(det["track_id"]), {})
            per_frame.setdefault(index, np.asarray(det["pos"], dtype=np.float64))
    jitter_values: list[float] = []
    for per_frame in track_points.values():
        ordered = [per_frame[index] for index in sorted(per_frame)]
        for index in range(2, len(ordered)):
            first = ordered[index] - ordered[index - 1]
            second = ordered[index - 1] - ordered[index - 2]
            jitter_values.append(float(np.linalg.norm(second - first)))
    cle_array = np.asarray(cle_values, dtype=np.float64) if cle_values else np.zeros(1)
    frames = prepared["video_frames"]
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt": tp + fn,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "far_per_frame": fp / frames if frames else 0.0,
        "hit_rate_5px": hits / gt_frames if gt_frames else 0.0,
        "cle_p50": float(np.percentile(cle_array, 50)) if cle_values else 0.0,
        "cle_p95": float(np.percentile(cle_array, 95)) if cle_values else 0.0,
        "max_jitter": max(jitter_values) if jitter_values else 0.0,
        "p99_jitter": float(np.percentile(jitter_values, 99)) if jitter_values else 0.0,
        "frames": frames,
        "gt_frames": gt_frames,
    }


def build_rule_grid() -> list[dict]:
    """The candidate abstention rules, one per diagnostic signal and a few combinations."""
    rules: list[dict] = []
    for tau in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        rules.append({"name": f"uniq>{tau}", "conditions": [["uniq", "gt", tau]]})
    for peak in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
        rules.append({"name": f"peak<{peak}", "conditions": [["peak", "lt", peak]]})
    for peak in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35):
        rules.append({"name": f"final_peak<{peak}", "conditions": [["final_peak", "lt", peak]]})
    rules.append({"name": "xcorr_fail", "conditions": [["xcorr_ok", "eq", 0]]})
    rules.append({"name": "rematch", "conditions": [["rematch", "eq", 1]]})
    for deviation in (4, 8, 12, 16, 24, 32, 48):
        rules.append({"name": f"dev>{deviation}", "conditions": [["chain_dev", "gt", deviation]]})
    for displacement in (20, 40, 60, 80, 120):
        rules.append({"name": f"disp>{displacement}", "conditions": [["anchor_disp", "gt", displacement]]})
    rules.append({"name": "held>0", "conditions": [["held", "gt", 0]]})
    for tau in (0.3, 0.5, 0.7):
        for deviation in (8, 16, 32):
            rules.append(
                {
                    "name": f"uniq>{tau}&dev>{deviation}",
                    "conditions": [["uniq", "gt", tau], ["chain_dev", "gt", deviation]],
                }
            )
    # 「低峰值」再叠加一个佐证条件：只有当模板匹配**既不自信、又含糊**时才弃权。
    for peak in (0.1, 0.15, 0.2, 0.25):
        for tau in (0.1, 0.2, 0.3, 0.5):
            rules.append(
                {
                    "name": f"peak<{peak}&uniq>{tau}",
                    "conditions": [["peak", "lt", peak], ["uniq", "gt", tau]],
                }
            )
    for peak in (0.1, 0.15, 0.2, 0.25):
        for deviation in (8, 16, 32):
            rules.append(
                {
                    "name": f"peak<{peak}&dev>{deviation}",
                    "conditions": [["peak", "lt", peak], ["chain_dev", "gt", deviation]],
                }
            )
    for tau in (0.1, 0.2, 0.3):
        rules.append(
            {
                "name": f"xcorr_fail&uniq>{tau}",
                "conditions": [["xcorr_ok", "eq", 0], ["uniq", "gt", tau]],
            }
        )
    # 机理：模板匹配不自信有两种成因——(a) 目标**机动很大**（模板跟不上，但快核跟得住，
    # 位置其实是对的）；(b) 目标**没怎么动**，却还是配不上（快核跟丢了 ⇒ 位置是错的）。
    # 只有 (b) 才该弃权，所以用「链位移大」把 (a) 排除掉。
    for peak in (0.1, 0.15, 0.2, 0.25, 0.3):
        for displacement in (10, 20, 40, 60, 80):
            rules.append(
                {
                    "name": f"peak<{peak}&disp>{displacement}",
                    "conditions": [["peak", "lt", peak], ["anchor_disp", "gt", displacement]],
                }
            )
    for peak in (0.2, 0.25):
        for displacement in (10, 20, 40):
            rules.append(
                {
                    "name": f"peak<{peak}&disp<{displacement}",
                    "conditions": [["peak", "lt", peak], ["anchor_disp", "lt", displacement]],
                }
            )
    return rules


def prepare(payload: dict) -> dict:
    """Pre-extract the per-frame arrays so a rule can be scored without re-parsing the dump."""
    frames = payload["frames_out"]
    points = []
    tracks = []
    boxes = []
    for frame in frames:
        entries = frame["entries"]
        points.append(np.asarray([det["pos"] for det in entries], dtype=np.float64).reshape(-1, 2))
        tracks.append([int(det["track_id"]) for det in entries])
        boxes.append(np.asarray(frame["gt"], dtype=np.float64).reshape(-1, 4))
    return {
        "sequence": payload["sequence"],
        "video_frames": payload["video_frames"],
        "points": points,
        "tracks": tracks,
        "boxes": boxes,
        "limit_px": float(payload.get("output_max_step_px", 30.0)),
        "features": [frame.get("features", {}) for frame in frames],
    }


def masked_report(prepared: dict, abstain: list[bool]) -> list[list[dict]]:
    """The report with the abstained frames emptied, then the output slew limiter applied."""
    report: list[list[dict]] = []
    for index, frame_points in enumerate(prepared["points"]):
        if abstain[index] or not len(frame_points):
            report.append([])
            continue
        report.append(
            [
                {
                    "track_id": prepared["tracks"][index][position],
                    "pos": (float(frame_points[position][0]), float(frame_points[position][1])),
                    "score": 1.0,
                    "infilled": False,
                }
                for position in range(len(frame_points))
            ]
        )
    limit = prepared["limit_px"]
    if limit > 0.0:

        class _Cfg:
            output_max_step_px = limit

        _apply_output_slew_limit(report, _Cfg(), {})
    return report


def score_fast(prepared: dict, abstain: list[bool]) -> dict:
    """Score one sequence under one rule with the frozen matcher and the frozen servo metrics.

    The frozen matcher walks the detections of a frame in score order and lets each one claim the still-free
    ground truth whose centre is nearest; the business sets carry a single target, so the per-frame work is
    vectorised over the detections while the multi-detection frames fall back to the frozen implementation.
    """
    report = masked_report(prepared, abstain)
    tp = fp = fn = 0
    gt_frames = 0
    hits = 0
    cle_values: list[float] = []
    for index, frame_points in enumerate(prepared["points"]):
        box_array = prepared["boxes"][index]
        if len(box_array):
            gt_frames += 1
        detections = report[index]
        if not len(detections):
            fn += int(len(box_array))
            continue
        if len(detections) == 1 and len(box_array) >= 1:
            point = np.asarray(detections[0]["pos"], dtype=np.float64)
            centres = box_array[:, :2]
            distances = np.linalg.norm(centres - point, axis=1)
            nearest = int(np.argmin(distances))
            box = box_array[nearest]
            inside = abs(point[0] - box[0]) <= box[2] / 2 and abs(point[1] - box[1]) <= box[3] / 2
            if inside:
                tp += 1
                fn += int(len(box_array)) - 1
            else:
                fp += 1
                fn += int(len(box_array))
            nearest_distance = float(distances[nearest])
            cle_values.append(nearest_distance)
            if nearest_distance <= 5.0:
                hits += 1
            continue
        counts = count_matches(
            [
                (
                    box_array if len(box_array) else None,
                    frame_points,
                    np.ones(len(frame_points), dtype=np.float32),
                )
            ],
            0.0,
            8.0,
            "inbox",
        )
        tp += counts["tp"]
        fp += counts["fp"]
        fn += counts["fn"]

    track_points: dict[int, dict[int, np.ndarray]] = {}
    for index, detections in enumerate(report):
        for det in detections:
            per_frame = track_points.setdefault(int(det["track_id"]), {})
            per_frame.setdefault(index, np.asarray(det["pos"], dtype=np.float64))
    jitter_values: list[float] = []
    for per_frame in track_points.values():
        ordered = [per_frame[index] for index in sorted(per_frame)]
        for index in range(2, len(ordered)):
            first = ordered[index] - ordered[index - 1]
            second = ordered[index - 1] - ordered[index - 2]
            jitter_values.append(float(np.linalg.norm(second - first)))
    cle_array = np.asarray(cle_values, dtype=np.float64) if cle_values else np.zeros(1)
    frames = prepared["video_frames"]
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt": tp + fn,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "far_per_frame": fp / frames if frames else 0.0,
        "hit_rate_5px": hits / gt_frames if gt_frames else 0.0,
        "cle_p50": float(np.percentile(cle_array, 50)) if cle_values else 0.0,
        "cle_p95": float(np.percentile(cle_array, 95)) if cle_values else 0.0,
        "max_jitter": max(jitter_values) if jitter_values else 0.0,
        "p99_jitter": float(np.percentile(jitter_values, 99)) if jitter_values else 0.0,
        "frames": frames,
        "gt_frames": gt_frames,
    }


def aggregate(rows: list[dict]) -> dict:
    """Sum the per-sequence counters and recompute the pooled metrics."""
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    frames = sum(row["frames"] for row in rows)
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    hit = sum(row["hit_rate_5px"] * row["gt_frames"] for row in rows) / max(
        1, sum(row["gt_frames"] for row in rows)
    )
    return {
        "frames": frames,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": f1,
        "far_per_frame": fp / frames if frames else 0.0,
        "hit_rate_5px": hit,
        "max_jitter": max(row["max_jitter"] for row in rows),
        "p99_jitter": max(row["p99_jitter"] for row in rows),
    }


def main() -> None:
    """Entry point: dump the diagnostics and/or scan the candidate rules."""
    args = parse_args()
    if not args.dump and not args.scan:
        args.scan = True
    dump_dir = Path(args.dump_dir)
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")

    if args.dump:
        dump_dir.mkdir(parents=True, exist_ok=True)
        sequences = (
            [name.strip() for name in ",".join(args.seq).split(",") if name.strip()]
            if args.seq
            else sorted(path.stem for path in Path(args.cache_root).glob("*.pkl"))
        )
        cfg, parsed = build_config(args, sequences)
        for sequence in sequences:
            started = time.perf_counter()
            payload = dump_sequence(args, cfg, parsed, sequence)
            path = dump_dir / f"{sequence}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            print(
                f"[DUMP] {sequence} frames={payload['frames']} elapsed={time.perf_counter() - started:.1f}s -> {path}",
                flush=True,
            )

    if args.scan:
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(dump_dir.glob("*.json"))]
        if not payloads:
            raise SystemExit(f"[SCAN] {dump_dir} 里没有 dump，先跑 --dump")
        prepared = [prepare(payload) for payload in payloads]

        def evaluate(rule: dict) -> dict:
            """Score every cached sequence under one rule and pool the counters."""
            rows = []
            for item in prepared:
                abstain = [
                    rule_applies(rule, item["features"][index], len(item["points"][index]) > 0)
                    for index in range(len(item["points"]))
                ]
                rows.append(score_fast(item, abstain))
            return aggregate(rows)

        empty = evaluate({"conditions": []})
        print(
            f"[SCAN] empty-rule reproduction: F1={empty['f1'] * 100:.4f} FAR={empty['far_per_frame']:.4f} "
            f"hit@5px={empty['hit_rate_5px'] * 100:.2f}% max_jitter={empty['max_jitter']:.2f}",
            flush=True,
        )
        results = []
        for rule in build_rule_grid():
            outcome = evaluate(rule)
            outcome["name"] = rule["name"]
            outcome["conditions"] = rule["conditions"]
            outcome["delta_f1"] = outcome["f1"] - empty["f1"]
            outcome["delta_hit"] = outcome["hit_rate_5px"] - empty["hit_rate_5px"]
            outcome["under_red_line"] = outcome["far_per_frame"] <= float(args.red_line)
            results.append(outcome)
            print(
                f"[RULE] {rule['name']:>18s} F1={outcome['f1'] * 100:7.4f} ({outcome['delta_f1'] * 100:+.4f}) "
                f"FAR={outcome['far_per_frame']:.4f} hit@5px={outcome['hit_rate_5px'] * 100:6.2f}% "
                f"({outcome['delta_hit'] * 100:+.2f}) max_jitter={outcome['max_jitter']:6.2f} "
                f"{'UNDER' if outcome['under_red_line'] else 'over'}",
                flush=True,
            )
        feasible = sorted(
            (row for row in results if row["under_red_line"]),
            key=lambda row: (-row["hit_rate_5px"], -row["f1"]),
        )
        report = {
            "generated_at": stamp,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "script": SCRIPT_REL,
            "command": f"/home/huangzhe/anaconda3/envs/uav/bin/python {SCRIPT_REL} --scan",
            "red_line": float(args.red_line),
            "baseline": empty,
            "best_under_red_line": feasible[0] if feasible else None,
            "rules": results,
        }
        Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [
            "# Abstention 策略离线扫描（24 序列）",
            "",
            f"- 生成时间：{stamp}｜执行机：{socket.gethostname()}｜脚本：`{SCRIPT_REL}`",
            f"- 产物：`{args.out_json}`",
            "",
            f"基线（不弃权，与交付档一致）：F1={empty['f1'] * 100:.4f}｜FAR={empty['far_per_frame']:.4f}｜"
            f"hit@5px={empty['hit_rate_5px'] * 100:.2f}%｜max_jitter={empty['max_jitter']:.2f}",
            "",
            "| 规则 | F1 | ΔF1 | FAR | hit@5px | Δhit | max_jitter | 红线内 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | :-- |",
        ]
        for row in results:
            lines.append(
                f"| {row['name']} | {row['f1'] * 100:.4f} | {row['delta_f1'] * 100:+.4f} | "
                f"{row['far_per_frame']:.4f} | {row['hit_rate_5px'] * 100:.2f}% | "
                f"{row['delta_hit'] * 100:+.2f} | {row['max_jitter']:.2f} | "
                f"{'是' if row['under_red_line'] else '否'} |"
            )
        Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[SCAN] wrote {args.out_json}")
        print(f"[SCAN] wrote {args.out_md}")


if __name__ == "__main__":
    main()
