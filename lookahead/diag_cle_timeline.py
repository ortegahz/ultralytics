#!/usr/bin/env python3
"""逐帧 CLE 时间线诊断：把每个对照臂的**当前帧**误差按帧摊开。

尺子 2 汇总出来的 ``cle_p95`` 只告诉你"有 5% 的帧很糟"，不告诉你**哪一段**糟。
转台伺服关心的是**连续性**：一次 366 px 的跳变比 100 帧各差 3 px 致命得多。
本工具把每个臂的逐帧误差序列拉出来，并标出：

* 误差 > ``--bad-px``（默认 20 px）的帧：脱靶帧；
* 相邻帧位移突变量 > ``--jitter-px``（默认 10 px）的帧：抖动帧（转台会抽搐的地方）；
* 连续脱靶段（fragmentation 的证据）。

用法::

    python manu/tracking/lookahead/diag_cle_timeline.py \
        --cache-root <ms_cache> --labels-root <labels> --seqs <SEQ[,SEQ...]> \
        --params-json <params.json> --frames-root <frames_ir_jpg> \
        --delta 8 --arms original,stale,replay --replay-mode rolling \
        --out-json runs/lookahead/diag/cle_timeline.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.tracking.lookahead import lookahead_longquan as lq  # noqa: E402
from manu.tracking.lookahead.extrapolator import build_report_stream  # noqa: E402
from manu.tracking.lookahead.track_fusion import natural_sort_key  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="逐帧 CLE 时间线诊断")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--seqs", default="")
    parser.add_argument("--all-seqs", action="store_true")
    parser.add_argument("--params-json", default="")
    parser.add_argument("--frames-root", default="")
    parser.add_argument("--frames-cache", default="", help="本地 memmap 帧缓存目录（优先）")
    parser.add_argument("--delta", type=int, default=8)
    parser.add_argument("--arms", default="original,stale,replay")
    parser.add_argument("--replay-mode", default="rolling", choices=["perframe", "rolling"])
    parser.add_argument("--rolling-recenter", type=int, default=1)
    parser.add_argument("--rolling-hold-mode", default="last_step")
    parser.add_argument("--rolling-max-step-px", type=float, default=12.0)
    parser.add_argument("--rolling-max-offset-px", type=float, default=40.0)
    parser.add_argument("--rolling-reacquire-px", type=float, default=30.0)
    parser.add_argument("--replay-fallback", default="holdc")
    parser.add_argument("--replay-max-hold", type=int, default=8)
    parser.add_argument("--velocity-source", default="diff1")
    parser.add_argument("--velocity-window", type=int, default=3)
    parser.add_argument("--fast-win-size", type=int, default=48)
    parser.add_argument("--fast-bg-window", type=int, default=96)
    parser.add_argument("--fast-mode", default="centroid")
    parser.add_argument("--fast-th-k", type=float, default=3.0)
    parser.add_argument("--bad-px", type=float, default=20.0)
    parser.add_argument("--jitter-px", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def _cle_series(report, truth):
    """逐帧 CLE：每帧取**离最近 GT 中心最近**的那个上报点。"""
    series = []
    for frame_index, detections in enumerate(report):
        boxes = truth.get(frame_index)
        if boxes is None:
            continue
        box_array = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        if not box_array.size:
            continue
        centres = box_array[:, :2]
        if not detections:
            series.append((frame_index, float("inf"), None, None))
            continue
        positions = np.asarray([d["pos"] for d in detections], dtype=np.float64).reshape(-1, 2)
        distances = np.linalg.norm(positions[:, None, :] - centres[None, :, :], axis=2)
        flat = int(np.argmin(distances))
        row, col = divmod(flat, distances.shape[1])
        series.append(
            (
                frame_index,
                float(distances[row, col]),
                positions[row].tolist(),
                centres[col].tolist(),
            )
        )
    return series


def _segments(flags):
    """把布尔序列切成连续段 [(起, 止, 长度)]。"""
    segments = []
    start = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            segments.append((start, index - 1, index - start))
            start = None
    if start is not None:
        segments.append((start, len(flags) - 1, len(flags) - start))
    return segments


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    cache_root = Path(args.cache_root)
    labels_root = Path(args.labels_root)
    sequences = lq.resolve_sequences(cache_root, "", args.all_seqs, args.seqs)
    params = lq.load_params(args.params_json)
    arms = [name.strip() for name in args.arms.split(",") if name.strip()]

    report_rows: dict[str, dict] = {}
    for sequence in sequences:
        records = lq.build_records(
            cache_root,
            labels_root,
            sequence,
            float(params["dedup_radius"]),
            0.0,
            int(params["persist_min_hits"]),
            int(params["persist_window"]),
            float(params["persist_tol"]),
        )
        dets, _ = lq.run_system_sota(
            records, params, img_h=lq.IMG_H, capture_kf_states=True
        )
        ordered = sorted(records, key=lambda record: natural_sort_key(record["im_name"]))
        truth = {
            index: np.asarray(record["gt_bboxes"], dtype=np.float32).reshape(-1, 4)
            for index, record in enumerate(ordered)
        }
        frame_source = lq.build_frame_source(
            args.frames_root,
            args.frames_cache,
            sequence,
            [int(record["frame_index"]) for record in records],
            int(args.delta),
        )
        print(f"[SEQ] {sequence} frames={len(ordered)} gt={sum(1 for v in truth.values() if v.size)}", flush=True)

        for arm in arms:
            if arm == "original":
                stream = dets
                stats = {}
            else:
                cfg = lq.LookaheadConfig(
                    delta=int(args.delta),
                    mode="replay" if arm == "replay" else arm,
                    replay_mode=str(args.replay_mode),
                    rolling_recenter=bool(int(args.rolling_recenter)),
                    rolling_hold_mode=str(args.rolling_hold_mode),
                    rolling_max_step_px=float(args.rolling_max_step_px),
                    rolling_max_offset_px=float(args.rolling_max_offset_px),
                    rolling_reacquire_px=float(args.rolling_reacquire_px),
                    velocity_source=str(args.velocity_source),
                    velocity_window=int(args.velocity_window),
                    replay_fallback=str(args.replay_fallback),
                    replay_max_hold=int(args.replay_max_hold),
                    win_size=int(args.fast_win_size),
                    bg_window=int(args.fast_bg_window),
                    fast_mode=str(args.fast_mode),
                    th_k=float(args.fast_th_k),
                )
                stream, stats = build_report_stream(
                    dets,
                    len(ordered),
                    cfg,
                    kf_states=None,
                    frame_source=frame_source,
                )
            series = _cle_series(stream, truth)
            finite = [item[1] for item in series if np.isfinite(item[1])]
            values = np.asarray(finite, dtype=np.float64)
            bad_flags = [bool(np.isfinite(item[1]) and item[1] > args.bad_px) for item in series]
            positions = [item[2] for item in series]
            jitter_flags = [False] * len(series)
            jitters = []
            prev = None
            for index, position in enumerate(positions):
                if position is None:
                    prev = None
                    continue
                if prev is not None:
                    delta = np.linalg.norm(np.asarray(position) - np.asarray(prev))
                    jitters.append(delta)
                    jitter_flags[index] = delta > args.jitter_px
                prev = position
            worst = sorted(
                (item for item in series if np.isfinite(item[1])),
                key=lambda item: item[1],
                reverse=True,
            )[: args.top]
            row = {
                "frames": len(ordered),
                "gt_frames": len(series),
                "cle_p50": float(np.percentile(values, 50)) if values.size else 0.0,
                "cle_p95": float(np.percentile(values, 95)) if values.size else 0.0,
                "cle_max": float(values.max()) if values.size else 0.0,
                "bad_frames": int(sum(bad_flags)),
                "bad_segments": _segments(bad_flags),
                "max_step": float(max(jitters)) if jitters else 0.0,
                "jitter_frames": int(sum(jitter_flags)),
                "worst": [
                    {"frame": f, "cle": round(c, 3), "pos": p, "gt": g} for f, c, p, g in worst
                ],
                "stream_stats": {k: v for k, v in stats.items() if isinstance(v, (int, float))},
            }
            report_rows.setdefault(arm, {})[sequence] = row
            print(
                f"  [{arm:8s}] p50={row['cle_p50']:7.3f} p95={row['cle_p95']:8.3f} "
                f"max={row['cle_max']:9.3f} bad={row['bad_frames']:4d} "
                f"segments={len(row['bad_segments'])} maxstep={row['max_step']:7.3f} "
                f"jitter_frames={row['jitter_frames']}",
                flush=True,
            )

    payload = {
        "config": {
            "cache_root": str(cache_root),
            "labels_root": str(labels_root),
            "delta": int(args.delta),
            "arms": arms,
            "replay_mode": str(args.replay_mode),
            "bad_px": float(args.bad_px),
            "jitter_px": float(args.jitter_px),
        },
        "per_sequence": report_rows,
    }
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SUCCESS] -> {out}")
    print(f"[SUCCESS] elapsed {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
