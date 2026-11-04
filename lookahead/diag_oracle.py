#!/usr/bin/env python3
"""Oracle 消融：把「320 ms 延迟的物理代价」与「快核链的能力缺口」**分开量化**。

背景
----
交付候选是「臂 3 · per-frame 退回重算」：物理时刻 t 只拿得到 D(t-Δ)（Δ=8 @25Hz
= 320 ms），于是从 t-Δ 起链、用局部快核重推 Δ 步到 t，再上报。
它相对「无滞后天花板」（同一时刻直接用 D(t)）差 2.16 pp。

这 2.16 pp 到底是**实现缺陷**（快核链不可靠）还是**延迟的物理代价**
（8 帧后的位置本就不可知）？只靠现有臂分不出来——它们同时含了「锚点误差」
与「位移估计误差」。本工具用 oracle 逐层剥开：

==========  ==========================================  ==========================================
臂           上报坐标                                      回答的问题
==========  ==========================================  ==========================================
truth        GT(t)                                         标尺自检（期望 F1 ≈ 100%）
truth_delay  GT(t-Δ)                                       完美检测器 + 320ms 延迟、无补偿 ⇒ **物理地板**
truth_cv     GT(t-Δ) + v̂·Δ（v̂ 由 GT 历史最小二乘）           完美检测器 + **完美匀速模型** ⇒ 运动模型天花板
fast_oracle  GT(t-Δ) 锚点 + **真实快核链**位移                快核链本身的能力上限（锚点误差被消掉）
replay_real  真实 D(t-Δ) 锚点 + 真实快核链位移               现交付候选
==========  ==========================================  ==========================================

读法：

* ``truth_delay`` 很低 ⇒ 延迟本身就很致命，任何因果系统都要吃亏；
* ``truth_cv`` 明显高于 ``truth_delay`` ⇒ 匀速模型能补回一部分延迟，
  但它离 ``truth`` 还有多远，就是**运动模型能力的天花板**；
* ``fast_oracle`` 明显低于 ``truth_cv`` ⇒ **快核链比完美匀速模型还差**，
  差额就是本实现真正该攻的工程缺口；
* ``replay_real`` 与 ``fast_oracle`` 的差额 = 锚点误差（D(t-Δ) 相对 GT(t-Δ)）的贡献。

用法::

    python manu/tracking/lookahead/diag_oracle.py \
        --cache-root <ms_cache> --labels-root <labels> --frames-root <frames> \
        --params-json <params.json> --all-seqs --delta 8 \
        --out-json runs/lookahead/diag/oracle.json
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
from manu.tracking.lookahead import extrapolator  # noqa: E402
from manu.tracking.lookahead.extrapolator import (  # noqa: E402
    LookaheadConfig,
    build_report_stream,
    servo_metrics,
)
from manu.tracking.lookahead.track_fusion import natural_sort_key  # noqa: E402


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="oracle 消融诊断")
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--frames-cache", default="")
    ap.add_argument("--params-json", required=True)
    ap.add_argument("--seq", default="")
    ap.add_argument("--seqs", default="")
    ap.add_argument("--all-seqs", action="store_true")
    ap.add_argument("--delta", type=int, default=8)
    ap.add_argument("--fast-win-size", type=int, default=48)
    ap.add_argument("--fast-bg-window", type=int, default=96)
    ap.add_argument("--cv-window", type=int, default=5, help="匀速模型的拟合窗（record 数）")
    ap.add_argument("--out-json", default="")
    return ap.parse_args()


def _gt_centers(ordered: list[dict]) -> dict[int, np.ndarray]:
    """GT 中心（cx, cy）⇒ record index → 中心点。多目标时取第一个。"""
    centers: dict[int, np.ndarray] = {}
    for index, record in enumerate(ordered):
        boxes = np.asarray(record["gt_bboxes"], dtype=np.float32).reshape(-1, 4)
        if len(boxes):
            centers[index] = boxes[0, :2].astype(np.float64).copy()
    return centers


def _fit_velocity(centers: dict[int, np.ndarray], frame: int, window: int) -> np.ndarray | None:
    """用 GT 历史点最小二乘拟合 (frame-window, frame] 区间的速度（px/record）。"""
    xs, ys = [], []
    for f in range(frame - window + 1, frame + 1):
        point = centers.get(f)
        if point is None:
            continue
        xs.append(float(f))
        ys.append((float(point[0]), float(point[1])))
    if len(xs) < 2:
        return None
    xs_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    xs_c = xs_arr - xs_arr.mean()
    denom = float(np.sum(xs_c * xs_c))
    if denom <= 0.0:
        return None
    slope = (xs_c[:, None] * (y_arr - y_arr.mean(axis=0))).sum(axis=0) / denom
    return slope


def _det(point: np.ndarray, score: float = 1.0) -> dict:
    return {"pos": (float(point[0]), float(point[1])), "score": float(score)}


def _build_truth_stream(centers: dict[int, np.ndarray], num_frames: int, shift: int) -> list[list[dict]]:
    """pos = GT(t-shift)；没有 GT 的帧留空。"""
    stream: list[list[dict]] = []
    for index in range(num_frames):
        point = centers.get(index - shift)
        stream.append([_det(point)] if point is not None else [])
    return stream


def _build_truth_cv_stream(
    centers: dict[int, np.ndarray], num_frames: int, delta: int, window: int
) -> list[list[dict]]:
    """pos = GT(t-Δ) + v̂·Δ，v̂ 由 GT 历史拟合（= 完美匀速模型）。"""
    stream: list[list[dict]] = []
    for index in range(num_frames):
        source = index - delta
        anchor = centers.get(source)
        if anchor is None:
            stream.append([])
            continue
        velocity = _fit_velocity(centers, source, window)
        if velocity is None:
            stream.append([])
            continue
        stream.append([_det(anchor + velocity * float(delta))])
    return stream


def _build_fast_oracle_stream(
    dets: list[list[dict]],
    num_frames: int,
    cfg: LookaheadConfig,
    centers: dict[int, np.ndarray],
    frame_source,
) -> tuple[list[list[dict]], dict]:
    """**真实重推链**，但锚点换成 GT(t-Δ) ⇒ 消掉锚点误差，只留快核链的能力。

    实现方式：临时把 ``extrapolator._base_position`` 换成「按源帧查 GT 中心」的版本，
    链、门、hold、限幅等逻辑**一行不改**（复用生产代码路径，避免诊断与实现漂移）。
    """
    original = extrapolator._base_position

    def patched(det: dict, frame: int, cfg_unused, kf_states) -> np.ndarray:
        point = centers.get(int(frame))
        if point is None:
            return original(det, frame, cfg_unused, kf_states)
        return np.asarray(point, dtype=np.float64).reshape(2).copy()

    extrapolator._base_position = patched
    try:
        stream, stats = build_report_stream(
            dets, num_frames, cfg, kf_states=None, frame_source=frame_source
        )
    finally:
        extrapolator._base_position = original
    return stream, stats


def _metrics(ordered: list[dict], stream: list[list[dict]], truth: dict, frames: int) -> dict:
    biz = lq.evaluate_biz(ordered, stream, num_frames=frames)
    servo = servo_metrics(stream, truth, len(ordered))
    return {
        "f1": biz["f1"],
        "precision": biz["precision"],
        "recall": biz["recall"],
        "far_per_frame": biz["far_per_frame"],
        "tp": biz["tp"],
        "fp": biz["fp"],
        "fn": biz["fn"],
        "hit_rate_5px": servo.get("hit_rate_5px"),
        "cle_p50": servo.get("cle_p50"),
        "cle_p95": servo.get("cle_p95"),
        "max_jitter": servo.get("max_jitter"),
    }


def main() -> None:
    args = _parse_args()
    cache_root = Path(args.cache_root)
    labels_root = Path(args.labels_root)
    sequences = lq.resolve_sequences(cache_root, args.seq, args.all_seqs, args.seqs)
    params = lq.load_params(args.params_json)
    delta = int(args.delta)

    cfg = LookaheadConfig(
        delta=delta,
        mode="replay",
        win_size=int(args.fast_win_size),
        bg_window=int(args.fast_bg_window),
        replay_anchor="delta",
        replay_mode="perframe",
        replay_max_hold=8,
        replay_fallback="holdv",
        replay_source_anchor_fallback=True,
        require_valid=True,
        output_max_step_px=30.0,
    )

    rows: dict[str, dict[str, dict]] = {}
    started = time.perf_counter()
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
        frames = lq.load_cache_meta(cache_root, sequence)["frames"]
        dets, _ = lq.run_system_sota(records, params, img_h=lq.IMG_H, capture_kf_states=False)

        ordered = sorted(records, key=lambda record: natural_sort_key(record["im_name"]))
        truth = {
            index: np.asarray(record["gt_bboxes"], dtype=np.float32).reshape(-1, 4)
            for index, record in enumerate(ordered)
        }
        centers = _gt_centers(ordered)
        num_frames = len(ordered)

        frame_source = lq.build_frame_source(
            args.frames_root,
            args.frames_cache,
            sequence,
            [int(record["frame_index"]) for record in records],
            delta,
        )
        streams = {
            "truth": _build_truth_stream(centers, num_frames, 0),
            "truth_delay": _build_truth_stream(centers, num_frames, delta),
            "truth_cv": _build_truth_cv_stream(centers, num_frames, delta, int(args.cv_window)),
        }
        if frame_source is not None:
            streams["replay_real"], _ = build_report_stream(
                dets, num_frames, cfg, kf_states=None, frame_source=frame_source
            )
            streams["fast_oracle"], _ = _build_fast_oracle_stream(
                dets, num_frames, cfg, centers, frame_source
            )

        for name, stream in streams.items():
            rows.setdefault(name, {})[sequence] = _metrics(ordered, stream, truth, frames)

        summary = {name: f"{rows[name][sequence]['f1'] * 100:.2f}" for name in streams}
        print(f"[SEQ] {sequence} " + " ".join(f"{k}={v}" for k, v in summary.items()), flush=True)

    print(f"\n[OVERALL] elapsed={time.perf_counter() - started:.1f}s")
    print(f"{'arm':<14}{'F1':>10}{'P':>9}{'R':>9}{'FAR/帧':>10}{'hit@5px':>10}{'CLE p50':>10}{'CLE p95':>10}")
    for name, per_seq in rows.items():
        tp = sum(r["tp"] for r in per_seq.values())
        fp = sum(r["fp"] for r in per_seq.values())
        fn = sum(r["fn"] for r in per_seq.values())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        weighted = lambda key: (
            sum(float(r[key]) * int(r["tp"]) for r in per_seq.values()) / max(1, tp)
        )
        print(
            f"{name:<14}{f1 * 100:>9.3f}%{precision * 100:>8.2f}%{recall * 100:>8.2f}%"
            f"{weighted('far_per_frame'):>10.4f}{weighted('hit_rate_5px') * 100:>9.2f}%"
            f"{weighted('cle_p50'):>10.2f}{weighted('cle_p95'):>10.2f}"
        )

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"delta": delta, "per_sequence": rows}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
