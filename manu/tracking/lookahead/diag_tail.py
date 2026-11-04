#!/usr/bin/env python3
"""重推臂误差尾巴归因诊断（**只读诊断工具，不参与交付管线**）。

回答一个问题：replay 臂的 CLE 尾巴（p95 ≈ 35 px、p99 ≈ 209 px）到底来自哪里？

候选归因只有两个：

* **兜底帧（fallback）**：重推失败 → 走 kinematic / hold / holdv 兜底；
* **走偏帧（walk-off）**：重推"成功"了，但快核链在 8 步里走到了别的亮斑上，
  于是融合出来的坐标整体偏掉（表现为单帧大跳变）。

做法：包一层 :func:`_replay_detections`，把每帧的 ``failed``/``reason`` 记下来，
再和逐帧 CLE / 抖动对齐，分桶统计。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.eval_multiscale_cache_th import load_ground_truth  # noqa: E402
from manu.tracking.lookahead import extrapolator  # noqa: E402
from manu.tracking.lookahead.lookahead_longquan import (  # noqa: E402
    SequenceFrameSource,
    build_records,
    load_cache_meta,
    load_params,
    run_system_sota,
)
from manu.tracking.lookahead.extrapolator import LookaheadConfig, build_report_stream  # noqa: E402


class _TracingTracker:
    """包一层快核，把每次调用的 (ROI 中心, 输出, 是否有效) 记下来，用于取证链条内部行为。"""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[dict] = []

    def step(self, image, center_xy):
        result = self.inner.step(image, center_xy)
        self.calls.append(
            {
                "center": [float(center_xy[0]), float(center_xy[1])],
                "xy": [float(result.xy[0]), float(result.xy[1])],
                "valid": bool(result.valid),
            }
        )
        return result


def _pct(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if len(values) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--params-json", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--delta", type=int, default=8)
    ap.add_argument("--replay-anchor", default="delta")
    ap.add_argument("--replay-fallback", default="holdv")
    ap.add_argument("--replay-max-hold", type=int, default=8)
    ap.add_argument("--replay-anchor-check-px", type=float, default=0.0)
    ap.add_argument("--replay-pred-radius-px", type=float, default=12.0)
    ap.add_argument("--replay-step-rel-tol-px", type=float, default=0.0)
    ap.add_argument("--replay-outlier-tol-px", type=float, default=0.0)
    ap.add_argument("--trace-frames", default="", help="逗号分隔的 record 下标，打印重推链条轨迹")
    ap.add_argument("--mode", default="replay", help="build_report_stream 的档位：replay / kinematic / stale")
    ap.add_argument("--fast-win-size", type=int, default=48)
    # ↓ 口径对齐（2026-11-03）：诊断工具必须能复现**交付口径**的 hold/限幅设置，
    #   否则归因结论会建立在与线上不同的配置上（曾漏掉 replay_hold_use_ref，
    #   诊断用 hold_use_ref=True、交付用 False ⇒ 结论不可信）。
    ap.add_argument("--replay-hold-use-ref", dest="replay_hold_use_ref", action="store_true", default=False)
    ap.add_argument("--no-replay-hold-use-ref", dest="replay_hold_use_ref", action="store_false")
    ap.add_argument("--replay-hold-robust", dest="replay_hold_robust", action="store_true", default=False)
    ap.add_argument("--no-replay-hold-robust", dest="replay_hold_robust", action="store_false")
    ap.add_argument("--output-max-step-px", type=float, default=0.0)
    ap.add_argument("--fast-bg-window", type=int, default=96)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    labels_root = Path(args.labels_root)
    params = load_params(args.params_json)

    records = build_records(
        cache_root,
        labels_root,
        args.seq,
        float(params["dedup_radius"]),
        0.0,
        int(params["persist_min_hits"]),
        int(params["persist_window"]),
        float(params["persist_tol"]),
    )
    frames = load_cache_meta(cache_root, args.seq)["frames"]
    dets, system_res = run_system_sota(records, params, img_h=512, capture_kf_states=True)
    kf_states = system_res.get("kf_states") if system_res else None

    frames_dir = Path(args.frames_root) / args.seq
    frame_source = SequenceFrameSource(
        frames_dir, [int(r["frame_index"]) for r in records], int(args.delta)
    )

    # ---- 包一层 _replay_detections，记录每帧的成败与原因 ----
    outcomes: dict[int, dict] = {}
    original_replay = extrapolator._replay_detections

    def recording_replay(source_dets, frame, target_frame, cfg, kf, tracker, fs, dets_by_frame=None):
        start = len(tracker.calls)
        outcome = original_replay(source_dets, frame, target_frame, cfg, kf, tracker, fs, dets_by_frame)
        outcome["trace"] = tracker.calls[start:]
        outcomes[int(target_frame)] = {
            "failed": bool(outcome["failed"]),
            "reason": str(outcome["reason"]),
            "held": int(outcome["held"]),
            "steps": int(outcome["steps"]),
            "trace": outcome.get("trace", []),
        }
        return outcome

    extrapolator._replay_detections = recording_replay
    real_tracker_cls = extrapolator.LocalPatchTracker
    tracer_box = {}

    def traced_tracker_cls(*args, **kwargs):
        tracer_box["tracker"] = _TracingTracker(real_tracker_cls(*args, **kwargs))
        return tracer_box["tracker"]

    extrapolator.LocalPatchTracker = traced_tracker_cls

    import dataclasses as _dc

    _allowed = {f.name for f in _dc.fields(LookaheadConfig)}
    _wanted = dict(
        delta=int(args.delta),
        mode=str(args.mode),
        velocity_source="kf",
        velocity_window=3,
        position_source="output",
        win_size=int(args.fast_win_size),
        bg_window=int(args.fast_bg_window),
        replay_anchor=str(args.replay_anchor),
        replay_fallback=str(args.replay_fallback),
        replay_max_hold=int(args.replay_max_hold),
        replay_anchor_check_px=float(args.replay_anchor_check_px),
        replay_pred_radius_px=float(args.replay_pred_radius_px),
        replay_step_rel_tol_px=float(args.replay_step_rel_tol_px),
        replay_outlier_tol_px=float(args.replay_outlier_tol_px),
        replay_hold_use_ref=bool(args.replay_hold_use_ref),
        replay_hold_robust=bool(args.replay_hold_robust),
        output_max_step_px=float(args.output_max_step_px),
        require_valid=True,
    )
    cfg = LookaheadConfig(**{k: v for k, v in _wanted.items() if k in _allowed})
    report, stats = build_report_stream(
        dets,
        int(frames),
        cfg,
        kf_states=kf_states,
        frame_source=frame_source,
    )
    extrapolator._replay_detections = original_replay
    extrapolator.LocalPatchTracker = real_tracker_cls

    # ---- 逐帧 CLE / 抖动 ----
    meta = load_cache_meta(cache_root, args.seq)
    truth = load_ground_truth(labels_root, args.seq, meta["native_size"][0], meta["native_size"][1])
    record_to_frame = [int(r["frame_index"]) for r in records]

    rows = []
    prev_pos = None
    prev_prev_pos = None
    for index in range(len(report)):
        dets_here = report[index]
        if not dets_here:
            prev_pos = prev_prev_pos = None
            continue
        pos = np.asarray(dets_here[0]["pos"], dtype=np.float64).reshape(2)
        frame_index = record_to_frame[index] if index < len(record_to_frame) else index
        boxes = truth.get(frame_index)
        cle = float("nan")
        if boxes is not None and len(boxes):
            centers = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)[:, :2]
            cle = float(np.min(np.linalg.norm(centers - pos[None, :], axis=1)))
        jitter = float("nan")
        if prev_pos is not None and prev_prev_pos is not None:
            jitter = float(np.linalg.norm((pos - prev_pos) - (prev_pos - prev_prev_pos)))
        outcome = outcomes.get(index, {"failed": False, "reason": "", "held": 0, "steps": 0})
        rows.append(
            {
                "record": index,
                "frame_index": frame_index,
                "pos": [float(pos[0]), float(pos[1])],
                "cle": cle,
                "jitter": jitter,
                "failed": outcome["failed"],
                "reason": outcome["reason"],
                "held": outcome["held"],
            }
        )
        prev_prev_pos = prev_pos
        prev_pos = pos

    def bucket(name, subset):
        cles = [r["cle"] for r in subset if not np.isnan(r["cle"])]
        jitters = [r["jitter"] for r in subset if not np.isnan(r["jitter"])]
        return {
            "bucket": name,
            "frames": len(subset),
            "cle_p50": _pct(cles, 50),
            "cle_p90": _pct(cles, 90),
            "cle_p95": _pct(cles, 95),
            "cle_p99": _pct(cles, 99),
            "cle_max": max(cles) if cles else float("nan"),
            "cle_gt_10": int(sum(1 for c in cles if c > 10.0)),
            "cle_gt_30": int(sum(1 for c in cles if c > 30.0)),
            "jitter_p50": _pct(jitters, 50),
            "jitter_p99": _pct(jitters, 99),
            "jitter_max": max(jitters) if jitters else float("nan"),
        }

    ok_rows = [r for r in rows if not r["failed"]]
    bad_rows = [r for r in rows if r["failed"]]
    walk_rows = [r for r in ok_rows if not np.isnan(r["jitter"]) and r["jitter"] > 10.0]
    smooth_rows = [r for r in ok_rows if r not in walk_rows]

    summary = {
        "sequence": args.seq,
        "config": {
            "delta": int(args.delta),
            "replay_anchor": str(args.replay_anchor),
            "replay_fallback": str(args.replay_fallback),
            "replay_max_hold": int(args.replay_max_hold),
            "replay_anchor_check_px": float(args.replay_anchor_check_px),
            "replay_pred_radius_px": float(args.replay_pred_radius_px),
            "replay_step_rel_tol_px": float(args.replay_step_rel_tol_px),
            "replay_outlier_tol_px": float(args.replay_outlier_tol_px),
            "fast_win_size": int(args.fast_win_size),
            "fast_bg_window": int(args.fast_bg_window),
        },
        "stats": stats,
        "buckets": [
            bucket("all", rows),
            bucket("replay_ok", ok_rows),
            bucket("replay_ok_smooth(jitter<=10px)", smooth_rows),
            bucket("replay_ok_walkoff(jitter>10px)", walk_rows),
            bucket("fallback", bad_rows),
        ],
        "reason_hist": {},
        "worst": sorted(
            [r for r in rows if not np.isnan(r["cle"])],
            key=lambda r: -r["cle"],
        )[:15],
    }
    for row in bad_rows:
        key = row["reason"].split(":")[0]
        summary["reason_hist"][key] = summary["reason_hist"].get(key, 0) + 1

    print(json.dumps(summary["buckets"], ensure_ascii=False, indent=1))
    print("reason_hist:", json.dumps(summary["reason_hist"], ensure_ascii=False))
    print("stats:", json.dumps(stats, ensure_ascii=False))
    if args.trace_frames:
        wanted = {int(x) for x in args.trace_frames.split(",") if x.strip()}
        for target in sorted(wanted):
            outcome = outcomes.get(target)
            row = next((r for r in rows if r["record"] == target), None)
            print(f"--- target record {target} ---")
            if row:
                print("  report:", row["pos"], "cle=", row["cle"], "failed=", row["failed"], row["reason"])
            if outcome and outcome.get("trace"):
                for i, call in enumerate(outcome["trace"]):
                    d = call["xy"][0] - call["center"][0]
                    dy = call["xy"][1] - call["center"][1]
                    print(
                        f"   step{i} center=({call['center'][0]:.1f},{call['center'][1]:.1f})"
                        f" -> ({call['xy'][0]:.1f},{call['xy'][1]:.1f})"
                        f" valid={call['valid']}"
                    )

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=1))
        print("wrote", args.out_json)


if __name__ == "__main__":
    main()
