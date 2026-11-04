#!/usr/bin/env python3
"""rolling 档位的参数扫描（**流水线复用**：每条序列只跑一次管线，其余配置复用同一份检测）。

为什么要它：rolling 的参数（单步位移门 / 锚点残差门 / 重捕获半径 / 速度来源 / ROI 边长）
互相耦合，**单序列调出来的参数很容易过拟合**。本工具在多条序列上同时评估一组候选
配置，逐序列给 CLE 统计，再按"逐序列最差"汇总（与对照表口径一致），避免被某一条
序列的特例带偏。

用法::

    python manu/tracking/lookahead/sweep_rolling.py \
        --cache-root <ms_cache> --labels-root <labels> --frames-root <frames> \
        --params-json <params.json> --all-seqs \
        --grid-json <候选配置 json> --out-json runs/lookahead/sweep.json
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
from manu.tracking.lookahead.extrapolator import LookaheadConfig, build_report_stream  # noqa: E402
from manu.tracking.lookahead.track_fusion import natural_sort_key  # noqa: E402

BASE = dict(
    delta=8,
    mode="replay",
    replay_mode="rolling",
    rolling_recenter=True,
    rolling_hold_mode="vel",
    rolling_vel_alpha=0.5,
    rolling_max_step_px=25.0,
    rolling_max_offset_px=40.0,
    rolling_reacquire_px=80.0,
    velocity_source="diffN",
    velocity_window=5,
    replay_fallback="holdc",
    replay_max_hold=8,
    win_size=96,
    bg_window=192,
    fast_mode="centroid",
    th_k=3.0,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rolling 参数扫描")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--frames-root", default="")
    parser.add_argument("--frames-cache", default="", help="本地 memmap 帧缓存目录（优先）")
    parser.add_argument("--params-json", default="")
    parser.add_argument("--seqs", default="")
    parser.add_argument("--all-seqs", action="store_true")
    parser.add_argument("--grid-json", default="", help="候选配置列表（dict 列表），不给则用内置默认网格")
    parser.add_argument("--bad-px", type=float, default=20.0)
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def _cle_series(report, truth):
    """逐帧 CLE（离最近 GT 中心最近的那个上报点），与 servo_metrics 同口径。"""
    out = []
    for frame_index, detections in enumerate(report):
        boxes = truth.get(frame_index)
        if boxes is None:
            continue
        array = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        if not array.size:
            continue
        centres = array[:, :2]
        if not detections:
            out.append((frame_index, float("inf")))
            continue
        positions = np.asarray([d["pos"] for d in detections], dtype=np.float64).reshape(-1, 2)
        distances = np.linalg.norm(positions[:, None, :] - centres[None, :, :], axis=2)
        out.append((frame_index, float(distances.min())))
    return out


def _stats(series, bad_px):
    values = np.asarray([e for _, e in series if np.isfinite(e)], dtype=np.float64)
    if not values.size:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0, "bad": 0, "n": 0}
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "bad": int(sum(1 for _, e in series if e > bad_px)),
        "n": int(values.size),
    }


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    cache_root = Path(args.cache_root)
    labels_root = Path(args.labels_root)
    frames_root = Path(args.frames_root)
    sequences = lq.resolve_sequences(cache_root, "", args.all_seqs, args.seqs)
    params = lq.load_params(args.params_json)

    grid = json.loads(Path(args.grid_json).read_text(encoding="utf-8")) if args.grid_json else None
    if grid:
        for spec in grid:
            spec.setdefault("overrides", {})
    if grid is None:
        grid = [
            {"name": "tuned", "overrides": {}},
            {"name": "tuned_win48", "overrides": {"win_size": 48, "bg_window": 96}},
            {"name": "tuned_gate12", "overrides": {"rolling_max_step_px": 12.0}},
            {"name": "tuned_reacq30", "overrides": {"rolling_reacquire_px": 30.0}},
            {"name": "tuned_diff1", "overrides": {"velocity_source": "diff1"}},
            {"name": "tuned_diffN3", "overrides": {"velocity_window": 3}},
            {"name": "tuned_holdlast", "overrides": {"rolling_hold_mode": "last_step"}},
            {"name": "tuned_gate40", "overrides": {"rolling_max_step_px": 40.0}},
            {"name": "tuned_alpha0.8", "overrides": {"rolling_vel_alpha": 0.8}},
            {"name": "tuned_alpha0.3", "overrides": {"rolling_vel_alpha": 0.3}},
            {"name": "old_base", "overrides": {
                "rolling_hold_mode": "last_step", "rolling_vel_alpha": 0.5,
                "rolling_max_step_px": 12.0, "rolling_reacquire_px": 30.0,
                "velocity_source": "diff1", "velocity_window": 3,
                "win_size": 48, "bg_window": 96}},
        ]

    results: dict[str, dict[str, dict]] = {}
    for sequence in sequences:
        records = lq.build_records(
            cache_root, labels_root, sequence,
            float(params["dedup_radius"]), 0.0,
            int(params["persist_min_hits"]), int(params["persist_window"]),
            float(params["persist_tol"]),
        )
        dets, kf_states = lq.run_system_sota(
            records, params, img_h=lq.IMG_H, capture_kf_states=True
        )
        ordered = sorted(records, key=lambda record: natural_sort_key(record["im_name"]))
        truth = {
            index: np.asarray(record["gt_bboxes"], dtype=np.float32).reshape(-1, 4)
            for index, record in enumerate(ordered)
        }
        frame_source = lq.build_frame_source(
            frames_root,
            args.frames_cache,
            sequence,
            [int(record["frame_index"]) for record in ordered],
            int(BASE["delta"]),
        )
        if frame_source is None:
            print(f"[SKIP] {sequence} 无帧源", flush=True)
            continue
        print(f"[SEQ] {sequence} frames={len(ordered)}", flush=True)
        for spec in grid:
            settings = dict(BASE)
            settings.update(spec.get("overrides", {}))
            settings["delta"] = int(args.delta) if hasattr(args, "delta") else settings["delta"]
            config = LookaheadConfig(**settings)
            report, stats = build_report_stream(
                dets, len(ordered), config, kf_states=kf_states, frame_source=frame_source
            )
            row = _stats(_cle_series(report, truth), args.bad_px)
            row["reacq"] = int(stats.get("rolling_reacquired", 0))
            row["held"] = int(stats.get("replay_held_steps", 0))
            row["rej"] = int(stats.get("rolling_step_rejected", 0))
            row["maxstep"] = round(float(stats.get("replay_max_step_disp", 0.0)), 1)
            results.setdefault(spec["name"], {})[sequence] = row
            print(
                f"  [{spec['name']:18s}] p50={row['p50']:7.2f} p95={row['p95']:8.2f} "
                f"max={row['max']:9.2f} bad={row['bad']:4d}/{row['n']:5d} "
                f"reacq={row['reacq']:4d} held={row['held']:4d} rej={row['rej']:3d} "
                f"maxstep={row['maxstep']:8.1f}",
                flush=True,
            )

    print("\n=== 汇总（逐序列最差 / 均值） ===")
    summary = {}
    for name, per_seq in results.items():
        p95s = [r["p95"] for r in per_seq.values()]
        p50s = [r["p50"] for r in per_seq.values()]
        maxs = [r["max"] for r in per_seq.values()]
        bads = [r["bad"] for r in per_seq.values()]
        ns = [r["n"] for r in per_seq.values()]
        summary[name] = {
            "worst_p50": max(p50s), "mean_p50": float(np.mean(p50s)),
            "worst_p95": max(p95s), "mean_p95": float(np.mean(p95s)),
            "worst_max": max(maxs),
            "bad_total": int(sum(bads)), "frames_total": int(sum(ns)),
            "bad_rate": float(sum(bads) / max(1, sum(ns))),
            "max_reacq": max(r["reacq"] for r in per_seq.values()),
            "max_step_disp": max(r["maxstep"] for r in per_seq.values()),
        }
        s = summary[name]
        print(
            f"{name:20s} worst_p50={s['worst_p50']:6.2f} worst_p95={s['worst_p95']:8.2f} "
            f"worst_max={s['worst_max']:9.2f} bad={s['bad_total']:5d}/{s['frames_total']:6d} "
            f"({s['bad_rate']*100:5.2f}%) max_reacq={s['max_reacq']:4d} "
            f"max_step={s['max_step_disp']:8.1f}"
        )

    payload = {"grid": grid, "per_sequence": results, "summary": summary}
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SUCCESS] -> {out}")
    print(f"[SUCCESS] elapsed {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
