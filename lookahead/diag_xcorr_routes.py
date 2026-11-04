#!/usr/bin/env python3
"""复现 §23.11 的三条「离群帧修复路线」并落盘（负结果，必须可复现）。

三条路线（都只改**搜索策略**，不改管线语义）：
① 换搜索中心：把 xcorr 的搜索中心从「质心链位置」换成「运动学预测 anchor + Δ·v」；
② 大半径重试：r32 峰值低 ⇒ 用 r64 / r96 / r128 再搜一次；
③ 纯峰值门：peak < τ ⇒ 放弃 xcorr、保留链位置。

口径：帧集 = 有 GT 且交付臂有上报的帧（= 官方 CLE 样本集）；命中 = 上报点到最近 GT 中心 ≤ 5 px。

用法::

    python manu/tracking/lookahead/diag_xcorr_routes.py \
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
        --frames-root "/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg" \
        --arms-json runs/lookahead/final/arms_final_WTH15.json \
        --preds original=runs/lookahead/final/preds_off \
               chain=runs/lookahead/final/preds_SPC4 \
               delivery=runs/lookahead/final/preds_final_WTH15 \
        --seq VIDEO00007_19700101_001324 \
        --out-json runs/lookahead/final/diag_001324_xcorr_routes.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.eval_multiscale_cache_th import load_ground_truth  # noqa: E402
from manu.tracking.lookahead import xcorr  # noqa: E402


def load_preds_dir(directory: Path, seq: str) -> dict[int, np.ndarray]:
    doc = json.loads((directory / f"{seq}.json").read_text(encoding="utf-8"))
    out: dict[int, np.ndarray] = {}
    for key, dets in doc["per_frame"].items():
        if not dets:
            continue
        best = max(dets, key=lambda d: float(d.get("score", 0.0)))
        out[int(key)] = np.asarray(best["pos"], dtype=np.float64).reshape(2)
    return out


def summarize(points: dict[int, np.ndarray], truth: dict[int, np.ndarray], frames: list[int]) -> dict:
    hits = 0
    cle: list[float] = []
    for frame in frames:
        point = points.get(frame)
        if point is None:
            continue
        centres = np.asarray(truth[frame], dtype=np.float64).reshape(-1, 4)[:, :2]
        distance = float(np.linalg.norm(point[None, :] - centres[None, :, :], axis=2).min())
        cle.append(distance)
        if distance <= 5.0:
            hits += 1
    array = np.asarray(cle, dtype=np.float64)
    return {
        "n": len(cle),
        "hits_5px": hits,
        "hit_rate_5px": hits / len(cle) if cle else 0.0,
        "cle_p50": float(np.percentile(array, 50)) if cle else 0.0,
        "cle_p95": float(np.percentile(array, 95)) if cle else 0.0,
        "cle_max": float(array.max()) if cle else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--arms-json", required=True)
    parser.add_argument("--preds", nargs="+", required=True)
    parser.add_argument("--seq", default="VIDEO00007_19700101_001324")
    parser.add_argument("--delta", type=int, default=8)
    parser.add_argument("--velocity-window", type=int, default=3, help="速度估计窗（帧）")
    parser.add_argument("--tsize", type=int, default=32)
    parser.add_argument("--bg-size", type=int, default=96)
    parser.add_argument("--th-k", type=float, default=1.5)
    parser.add_argument("--radius", type=int, default=32)
    parser.add_argument("--min-peak", type=float, default=0.25)
    parser.add_argument("--retry-radii", default="64,96,128")
    parser.add_argument("--peak-gates", default="0.4,0.5")
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    arms_doc = json.loads(Path(args.arms_json).read_text(encoding="utf-8"))
    width = int(arms_doc["config"]["img_w"])
    height = int(arms_doc["config"]["img_h"])
    truth = load_ground_truth(Path(args.labels_root), args.seq, width, height)

    preds: dict[str, dict[int, np.ndarray]] = {}
    for item in args.preds:
        label, _, path = item.partition("=")
        preds[label] = load_preds_dir(Path(path), args.seq)

    original = preds["original"]
    chain = preds["chain"]
    delivery = preds["delivery"]

    frames_dir = Path(args.frames_root) / args.seq
    delta = int(args.delta)

    frame_set = sorted(frame for frame in truth if frame in delivery)
    retry_radii = [int(value) for value in args.retry_radii.split(",") if value]
    peak_gates = [float(value) for value in args.peak_gates.split(",") if value]

    variants: dict[str, dict[int, np.ndarray]] = {
        "chain_baseline": {},
        "delivery_real_pipeline": {},
        "xcorr_chain_center": {},
        "xcorr_kinematic_center": {},
        "xcorr_two_stage": {},
        "xcorr_retry_cascade": {},
    }
    for radius in retry_radii:
        variants[f"xcorr_retry_r{radius}"] = {}
    for gate in peak_gates:
        variants[f"xcorr_peak_gate_{gate}"] = {}

    peaks_log: list[float] = []
    skipped = 0
    started = time.time()
    for count, frame in enumerate(frame_set):
        source_frame = frame - delta
        if source_frame < 0:
            continue
        anchor = original.get(source_frame)
        chain_pos = chain.get(frame)
        if anchor is None or chain_pos is None:
            skipped += 1
            continue
        src_image = cv2.imread(str(frames_dir / f"frame_{source_frame:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        dst_image = cv2.imread(str(frames_dir / f"frame_{frame:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if src_image is None or dst_image is None:
            skipped += 1
            continue
        template, origin = xcorr.make_template(
            src_image, anchor, tsize=args.tsize, bg_size=args.bg_size, th_k=args.th_k
        )
        if template is None:
            skipped += 1
            continue

        # 运动学预测中心：anchor + Δ·v，v 由**原臂**在源帧附近的输出位置差分估计
        # （真实可得：原臂就是管线的卡尔曼输出；KF 内部速度未落盘，用差分近似）。
        prev_anchor = original.get(source_frame - int(args.velocity_window))
        if prev_anchor is not None:
            velocity = (anchor - prev_anchor) / float(args.velocity_window)
        else:
            velocity = np.zeros(2, dtype=np.float64)
        kinematic_center = anchor + delta * velocity

        displacement, peak = xcorr.match(
            dst_image, template, anchor, chain_pos, radius=args.radius, tsize=args.tsize, origin=origin
        )
        matched_chain = anchor + displacement if displacement is not None else None
        peaks_log.append(peak)

        displacement_kin, peak_kin = xcorr.match(
            dst_image, template, anchor, kinematic_center, radius=args.radius, tsize=args.tsize, origin=origin
        )
        matched_kin = anchor + displacement_kin if displacement_kin is not None else None

        variants["chain_baseline"][frame] = chain_pos
        variants["delivery_real_pipeline"][frame] = delivery[frame]
        if matched_chain is not None:
            variants["xcorr_chain_center"][frame] = matched_chain
        if matched_kin is not None:
            variants["xcorr_kinematic_center"][frame] = matched_kin
        # 两段式：峰值低才换运动学中心
        two_stage = matched_chain
        if peak < float(args.min_peak) and matched_kin is not None:
            two_stage = matched_kin
        if two_stage is not None:
            variants["xcorr_two_stage"][frame] = two_stage

        # ② 大半径重试：单半径版本各存一份；再存一份「级联」版本
        #    （r32 峰值低 ⇒ 依次换更大半径，取**第一个峰值达标**的结果）。
        best_by_radius: dict[int, tuple[float, np.ndarray]] = {
            int(args.radius): (peak, matched_chain if matched_chain is not None else chain_pos)
        }
        for radius in retry_radii:
            retry_disp, retry_pk = xcorr.match(
                dst_image, template, anchor, chain_pos, radius=radius, tsize=args.tsize, origin=origin
            )
            if retry_disp is not None:
                best_by_radius[radius] = (retry_pk, anchor + retry_disp)
        cascade = None
        for radius in [int(args.radius)] + retry_radii:
            if radius not in best_by_radius:
                continue
            retry_pk, retry_point = best_by_radius[radius]
            cascade = retry_point
            if retry_pk >= float(args.min_peak):
                break
        if cascade is not None:
            variants["xcorr_retry_cascade"][frame] = cascade
        for radius, (retry_pk, point) in best_by_radius.items():
            if radius != int(args.radius):
                variants[f"xcorr_retry_r{radius}"][frame] = point

        # ③ 纯峰值门
        for gate in peak_gates:
            variants[f"xcorr_peak_gate_{gate}"][frame] = (
                matched_chain if (matched_chain is not None and peak >= gate) else chain_pos
            )

        if count % 1000 == 0:
            print(f"[ROUTES] {count}/{len(frame_set)} elapsed={time.time()-started:.1f}s", flush=True)

    report = {
        "seq": args.seq,
        "delta": delta,
        "frame_set": {
            "size": len(frame_set),
            "definition": "有 GT 且交付臂有上报（= 官方 CLE 样本集）",
            "skipped": skipped,
        },
        "params": {
            "tsize": args.tsize,
            "bg_size": args.bg_size,
            "th_k": args.th_k,
            "radius": args.radius,
            "min_peak": args.min_peak,
            "retry_radii": retry_radii,
            "peak_gates": peak_gates,
            "velocity_window": args.velocity_window,
        },
        "peak_stats": {
            "n": len(peaks_log),
            "p50": float(np.percentile(peaks_log, 50)) if peaks_log else 0.0,
            "p05": float(np.percentile(peaks_log, 5)) if peaks_log else 0.0,
            "frac_below_min_peak": (
                sum(1 for value in peaks_log if value < float(args.min_peak)) / len(peaks_log)
                if peaks_log
                else 0.0
            ),
        },
        "variants": {},
    }
    for name, points in variants.items():
        report["variants"][name] = summarize(points, truth, frame_set)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[ROUTES] peak p05={report['peak_stats']['p05']:.3f} p50={report['peak_stats']['p50']:.3f} "
          f"frac<min_peak={report['peak_stats']['frac_below_min_peak']*100:.2f}%")
    for name, row in report["variants"].items():
        print(f"[ROUTES] {name:>26}: hit@5px={row['hit_rate_5px']*100:6.2f}% "
              f"(n={row['n']}) p50={row['cle_p50']:.2f} p95={row['cle_p95']:.2f}")
    print(f"[ROUTES] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
