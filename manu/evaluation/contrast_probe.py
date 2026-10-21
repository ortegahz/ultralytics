#!/usr/bin/env python3
"""目标对比度探针:目标在原始 IR 帧里到底比背景暗还是亮、对比度有多大。

这一步回答"极性翻转有没有物理前提":翻转能把"暗目标"变成"亮目标",但前提是
**目标确实比周围暗**。如果目标在原始帧里和背景一样亮/一样暗(无对比度),那
255-I 也救不了 —— 没有任何图像处理能凭空造出信号。

对每帧测:
* ``box_mean``  : GT 框内灰度均值
* ``ring_mean`` : 紧邻框外一圈(1~2 倍框尺寸)的背景均值
* ``contrast``  : box_mean - ring_mean(**有符号**:负 = 目标比背景暗)
* ``hot_frac``  : 框内比 ring_mean + 3σ 更亮的像素占比(热目标证据)
* ``cold_frac`` : 框内比 ring_mean - 3σ 更暗的像素占比(冷目标证据)

用法:
    python manu/evaluation/contrast_probe.py \
        --frames-root "/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg" \
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
        --seq VIDEO00007_19700101_001324 --start 677 --end 876 --sample 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.eval_multiscale_cache_th import load_ground_truth  # noqa: E402


def measure(gray: np.ndarray, boxes: np.ndarray) -> list[dict]:
    """对每帧的每个 GT 框测量目标-背景对比度。"""
    rows = []
    height, width = gray.shape[:2]
    for cx, cy, w, h in boxes:
        x0 = max(0, int(round(cx - w * 0.5)))
        x1 = min(width, int(round(cx + w * 0.5)) + 1)
        y0 = max(0, int(round(cy - h * 0.5)))
        y1 = min(height, int(round(cy + h * 0.5)) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        patch = gray[y0:y1, x0:x1].astype(np.float32)

        # 背景环:框外扩到 2 倍尺寸,再挖掉框本身
        pad_x = int(round(w * 0.5))
        pad_y = int(round(h * 0.5))
        rx0, rx1 = max(0, x0 - pad_x), min(width, x1 + pad_x)
        ry0, ry1 = max(0, y0 - pad_y), min(height, y1 + pad_y)
        ring = gray[ry0:ry1, rx0:rx1].astype(np.float32).copy()
        ring[y0 - ry0 : y1 - ry0, x0 - rx0 : x1 - rx0] = np.nan
        ring_values = ring[~np.isnan(ring)]
        if ring_values.size < 20:
            continue
        ring_mean = float(ring_values.mean())
        ring_std = float(ring_values.std())
        box_mean = float(patch.mean())
        hot = float((patch > ring_mean + 3 * ring_std).mean())
        cold = float((patch < ring_mean - 3 * ring_std).mean())
        rows.append(
            {
                "box_mean": box_mean,
                "ring_mean": ring_mean,
                "ring_std": ring_std,
                "contrast": box_mean - ring_mean,
                "box_min": float(patch.min()),
                "box_max": float(patch.max()),
                "hot_frac": hot,
                "cold_frac": cold,
                "w": float(w),
                "h": float(h),
            }
        )
    return rows


def summarize(rows: list[dict], label: str) -> dict:
    if not rows:
        return {"label": label, "n": 0}
    contrast = np.asarray([r["contrast"] for r in rows])
    hot = np.asarray([r["hot_frac"] for r in rows])
    cold = np.asarray([r["cold_frac"] for r in rows])
    sizes = np.asarray([max(r["w"], r["h"]) for r in rows])
    return {
        "label": label,
        "n": len(rows),
        "contrast_median": float(np.median(contrast)),
        "contrast_mean": float(contrast.mean()),
        "contrast_p10": float(np.percentile(contrast, 10)),
        "contrast_p90": float(np.percentile(contrast, 90)),
        "darker_fraction": float((contrast < 0).mean()),
        "brighter_fraction": float((contrast > 0).mean()),
        "hot_frac_median": float(np.median(hot)),
        "cold_frac_median": float(np.median(cold)),
        "cold_frac_p90": float(np.percentile(cold, 90)),
        "target_size_median_px": float(np.median(sizes)),
        "contrast_over_noise_median": float(np.median(np.abs(contrast) / np.maximum([r["ring_std"] for r in rows], 1e-6))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--seq", action="append", required=True, help="Repeatable")
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--sample", type=int, default=200, help="Max frames to measure per sequence")
    parser.add_argument("--out-json", default="runs/logs/contrast_probe.json")
    args = parser.parse_args()

    frames_root = Path(args.frames_root)
    labels_root = Path(args.labels_root)
    report = {}
    for sequence in args.seq:
        truth = load_ground_truth(labels_root, sequence, 640, 512)
        keys = sorted(truth)
        if args.start >= 0 and args.end >= 0:
            keys = [k for k in keys if args.start <= k <= args.end]
        if len(keys) > args.sample:
            step = max(1, len(keys) // args.sample)
            keys = keys[::step][: args.sample]
        rows = []
        for frame_index in keys:
            gray = cv2.imread(str(frames_root / sequence / f"frame_{frame_index:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            rows.extend(measure(gray, np.asarray(truth[frame_index], np.float32).reshape(-1, 4)))
        summary = summarize(rows, sequence)
        report[sequence] = summary
        print(
            f"{sequence}: n={summary['n']:>4} | contrast 中位 {summary['contrast_median']:+.2f} "
            f"(p10 {summary['contrast_p10']:+.2f} / p90 {summary['contrast_p90']:+.2f}) | "
            f"比背景暗 {summary['darker_fraction'] * 100:.1f}% / 比背景亮 {summary['brighter_fraction'] * 100:.1f}% | "
            f"冷像素占比中位 {summary['cold_frac_median'] * 100:.1f}% (p90 {summary['cold_frac_p90'] * 100:.1f}%) | "
            f"热像素占比中位 {summary['hot_frac_median'] * 100:.1f}% | 目标边长中位 {summary['target_size_median_px']:.0f}px | "
            f"|对比|/噪声 中位 {summary['contrast_over_noise_median']:.2f}",
            flush=True,
        )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[OUT] {args.out_json}")


if __name__ == "__main__":
    main()
