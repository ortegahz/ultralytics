#!/usr/bin/env python3
"""外挂档位/参数扫描（**流水线复用** + **完整伺服指标**）。

与 ``sweep_rolling.py`` 的三处差别（都是它踩过的坑）：

1. **KF 状态的取法**：``lq.run_system_sota`` 的第二个返回值是**整个** ``res``，
   KF 状态在 ``res["kf_states"]`` 里。把它直接当 ``kf_states`` 传给外挂，
   ``_estimate_velocity(source="kf")`` 会全程 ``available=False``，
   于是「一致性门」永远不触发、「kf 速度」退化成 0 —— 扫描结论会**静默失真**。
2. **算伺服指标**：直接调 :func:`extrapolator.servo_metrics`，
   这样 ``max_jitter``（转台抽搐）也能参与选型，而不只是 CLE 分位。
3. **算业务域指标**：F1/P/R/FAR 用 ``evaluate_biz``（inbox 判据），与交付表同口径。

聚合口径与 CLI 的 ``aggregate_servo`` 对齐：计数类跨序列相加，分位数/抖动取
**逐序列最差**（跨序列拼起来算二阶差分会在序列边界造出假抖动）。

用法::

    python manu/tracking/lookahead/sweep_arms.py \
        --cache-root <ms_cache> --labels-root <labels> --frames-cache <cache> \
        --params-json <params.json> --all-seqs \
        --grid-json <候选配置 json> --out-json runs/lookahead/sweep/arms.json
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
from manu.tracking.lookahead.extrapolator import (  # noqa: E402
    LookaheadConfig,
    build_report_stream,
    servo_metrics,
)

# 各候选配置的公共基线（与 CLI 的 ``build_lookahead_config`` 字段一一对应）。
BASE = dict(
    delta=8,
    mode="replay",
    position_source="output",
    replay_mode="perframe",
    replay_anchor="delta",
    replay_fallback="holdv",
    replay_hold_use_ref=False,
    replay_max_hold=8,
    win_size=48,
    bg_window=96,
    fast_mode="centroid",
    th_k=3.0,
)

DEFAULT_GRID = [
    {"name": "perframe_gate0", "overrides": {}},
    {"name": "perframe_gate24", "overrides": {"replay_consistency_gate_px": 24.0}},
    {"name": "perframe_gate48", "overrides": {"replay_consistency_gate_px": 48.0}},
    {"name": "perframe_gate96", "overrides": {"replay_consistency_gate_px": 96.0}},
    {"name": "refine_g24_w48", "overrides": {"replay_mode": "refine", "refine_gate_px": 24.0}},
    {"name": "refine_g40_w48", "overrides": {"replay_mode": "refine", "refine_gate_px": 40.0}},
    {
        "name": "refine_g40_w96",
        "overrides": {"replay_mode": "refine", "refine_gate_px": 40.0, "win_size": 96, "bg_window": 192},
    },
]

PERCENTILE_KEYS = (
    "cle_p50",
    "cle_p90",
    "cle_p95",
    "cle_p99",
    "cle_max",
    "cle_mean",
    "max_jitter",
    "p99_jitter",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="外挂档位扫描（含伺服指标）")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--frames-root", default="")
    parser.add_argument("--frames-cache", default="", help="本地 memmap 帧缓存目录（优先）")
    parser.add_argument("--params-json", default="")
    parser.add_argument("--seqs", default="")
    parser.add_argument("--all-seqs", action="store_true")
    parser.add_argument("--grid-json", default="", help="候选配置列表（dict 列表），不给则用内置默认网格")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-preds-dir", default="", help="可选：把每个候选的上报流落盘（JSON）")
    return parser.parse_args()


def _metrics_from_counts(tp: int, fp: int, fn: int, frames: int) -> dict:
    """由跨序列累加的计数算 F1/P/R/FAR（口径同 ``eval_multiscale_cache_th.metrics``）。"""
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "far_per_frame": fp / frames if frames else 0.0,
    }


def _aggregate(per_seq: dict[str, dict]) -> dict:
    """跨序列汇总：计数相加，分位数/抖动取逐序列最差。"""
    rows = list(per_seq.values())
    if not rows:
        return {}
    tp = sum(int(r["biz"]["tp"]) for r in rows)
    fp = sum(int(r["biz"]["fp"]) for r in rows)
    fn = sum(int(r["biz"]["fn"]) for r in rows)
    frames = sum(int(r["frames"]) for r in rows)
    gt_frames = sum(int(r["servo"]["gt_frames"]) for r in rows)
    inbox_hits = sum(int(r["servo"]["inbox_hits"]) for r in rows)
    weighted_5px = sum(float(r["servo"]["hit_rate_5px"]) * int(r["servo"]["gt_frames"]) for r in rows)
    out = _metrics_from_counts(tp, fp, fn, frames)
    out.update(
        {
            "frames": frames,
            "gt_frames": gt_frames,
            "inbox_hits": inbox_hits,
            "hit_rate_inbox": inbox_hits / gt_frames if gt_frames else 0.0,
            "hit_rate_5px": weighted_5px / gt_frames if gt_frames else 0.0,
            "lost_frames": sum(int(r["servo"]["lost_frames"]) for r in rows),
            "fragmentation": sum(int(r["servo"]["fragmentation"]) for r in rows),
            "coverage": (
                sum(int(r["servo"]["frames_with_report"]) for r in rows) / frames if frames else 0.0
            ),
        }
    )
    for key in PERCENTILE_KEYS:
        out[key] = max(float(r["servo"].get(key, 0.0)) for r in rows)
    out["percentile_aggregation"] = "worst_over_sequences"
    return out


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    cache_root = Path(args.cache_root)
    labels_root = Path(args.labels_root)
    frames_root = Path(args.frames_root)
    sequences = lq.resolve_sequences(cache_root, "", args.all_seqs, args.seqs)
    params = lq.load_params(args.params_json)

    grid = json.loads(Path(args.grid_json).read_text(encoding="utf-8")) if args.grid_json else DEFAULT_GRID
    for spec in grid:
        spec.setdefault("overrides", {})

    results: dict[str, dict[str, dict]] = {}
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
        dets, system_res = lq.run_system_sota(records, params, img_h=lq.IMG_H, capture_kf_states=True)
        # 关键：run_system_sota 返回的第二个值是**整个 res**，KF 状态在 res["kf_states"]。
        kf_states = (system_res or {}).get("kf_states")
        ordered = sorted(records, key=lambda record: lq.natural_sort_key(record["im_name"]))
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
        print(
            f"[SEQ] {sequence} frames={len(ordered)} kf_frames={len(kf_states or {})} "
            f"kf_tracks={sum(len(v) for v in (kf_states or {}).values())}",
            flush=True,
        )
        for spec in grid:
            settings = dict(BASE)
            settings.update(spec.get("overrides", {}))
            config = LookaheadConfig(**settings)
            report, stats = build_report_stream(
                dets, len(ordered), config, kf_states=kf_states, frame_source=frame_source
            )
            biz = lq.evaluate_biz(ordered, report, num_frames=len(ordered))
            servo = servo_metrics(report, truth, len(ordered))
            row = {
                "biz": biz,
                "servo": servo,
                "frames": len(ordered),
                "stats": {k: v for k, v in stats.items() if isinstance(v, (int, float, str, bool))},
            }
            results.setdefault(spec["name"], {})[sequence] = row
            print(
                f"  [{spec['name']:20s}] F1={biz['f1']*100:7.4f}% P={biz['precision']*100:7.4f}% "
                f"R={biz['recall']*100:7.4f}% FAR={biz['far_per_frame']:.4f} "
                f"hit5={servo['hit_rate_5px']*100:6.2f}% "
                f"p50={servo['cle_p50']:6.2f} p95={servo['cle_p95']:8.2f} "
                f"jit={servo['max_jitter']:8.1f} p99j={servo['p99_jitter']:7.1f} "
                f"frag={servo['fragmentation']:3d} lost={servo['lost_frames']:5d} "
                f"rej={int(stats.get('replay_consistency_rejected', 0)):5d}",
                flush=True,
            )

    summary = {name: _aggregate(per_seq) for name, per_seq in results.items()}
    print("\n=== 汇总（计数跨序列相加 / 分位数取逐序列最差） ===")
    for name, row in summary.items():
        print(
            f"{name:22s} F1={row['f1']*100:7.4f}% P={row['precision']*100:7.4f}% "
            f"R={row['recall']*100:7.4f}% FAR={row['far_per_frame']:.4f} "
            f"hit5={row['hit_rate_5px']*100:6.2f}% "
            f"p50={row['cle_p50']:6.2f} p95={row['cle_p95']:8.2f} "
            f"jit={row['max_jitter']:8.1f} p99j={row['p99_jitter']:7.1f} "
            f"frag={row['fragmentation']:3d} lost={row['lost_frames']:5d}",
            flush=True,
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
