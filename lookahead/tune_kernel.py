#!/usr/bin/env python3
"""快核（局部质心）估计器的独立 A/B 工具。

**为什么要有这个工具**：外挂的 replay 臂最终上报的是**快核的绝对位置**，
所以整条链的精度上限由"快核这一步到底测得准不准"决定。链上其它环节
（锚点、兜底、抖动）会掩盖这个事实，所以把估计器单独拎出来测：

* **单步误差**：以**管线自己在该帧的检测位置**为起点跑 1 步，量它与**下一帧
  管线位置**的差 ⇒ 纯估计器精度（管线位置本身误差 ~2 px，可当参考真值用）；
* **多步漂移**：同样起点连跑 Δ 步，量终点与 **Δ 帧后管线位置**的差
  ⇒ 链式累积效应（"质心收缩 / 误差自我放大"）。

**为什么不用 GT 框中心当锚点**：龙泉山的 GT 有**合并标注**（001324 最宽 573 px），
框中心落在空天里，锚点本身就是错的 ⇒ 测出来的"无效率 95%"是标注形态造成的假象。
GT 只用来算命中率先验，不参与定位。

**口径**：帧取服务器 SSHFS 只读挂载下的 `frames_ir_jpg`；管线位置取
``runs/lookahead/ref/preds_original_v3/<SEQ>.json``。所有数字落在 JSON 产物里
（铁律九：文档数字必须来自产物 JSON）。

用法::

    PYTHONPATH=. python manu/tracking/lookahead/tune_kernel.py \
        --labels-root /home/manu/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010 \
        --frames-root /home/manu/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg \
        --preds-dir runs/lookahead/ref/preds_original_v3 \
        --seqs VIDEO00007_19700101_001324 \
        --out-json runs/lookahead/tune/kernel_ab.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manu.tracking.async_dual_rate.fast_tracker import LocalPatchTracker  # noqa: E402

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast-kernel estimator A/B (single-step + multi-step drift)")
    parser.add_argument("--labels-root", required=True, help="Flat Final_Labels directory")
    parser.add_argument("--frames-root", required=True, help="Root of <sequence>/frame_XXXXXX.jpg")
    parser.add_argument("--preds-dir", required=True, help="Directory of <sequence>.json pipeline predictions")
    parser.add_argument("--seqs", default="", help="Comma separated sequences")
    parser.add_argument("--delta", type=int, default=8, help="Drift chain length (records)")
    parser.add_argument("--win-sizes", default="24,32,48,64", help="Comma separated ROI edge lengths")
    parser.add_argument("--bg-window", type=int, default=0, help="Background window edge (0 = 2x win_size)")
    parser.add_argument("--modes", default="centroid", help="Comma separated extraction modes")
    parser.add_argument("--th-ks", default="3.0", help="Comma separated threshold multipliers")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride (speed knob)")
    parser.add_argument("--out-json", default="", help="Output JSON path")
    return parser.parse_args()


def load_ground_truth(labels_root: Path, sequence: str, frames: set[int]) -> dict[int, np.ndarray]:
    """Read Final_Labels boxes as (N,4) cx,cy,w,h in native pixels."""
    truth: dict[int, np.ndarray] = {}
    prefix = f"{sequence}__"
    with os.scandir(labels_root) as entries:
        for entry in entries:
            name = entry.name
            if not name.startswith(prefix) or not name.endswith(".txt"):
                continue
            digits = name[len(prefix) : -4]
            if not digits.isdigit() or int(digits) not in frames or entry.stat().st_size == 0:
                continue
            rows, seen = [], set()
            for line in Path(entry.path).read_text(encoding="utf-8").splitlines():
                values = line.split()
                if len(values) < 5:
                    continue
                row = tuple(float(value) for value in values[1:5])
                if row in seen:
                    continue
                seen.add(row)
                rows.append(row)
            if rows:
                truth[int(digits)] = np.asarray(rows, np.float32)
    return truth


def frame_paths(frames_root: Path, sequence: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for path in (frames_root / sequence).iterdir():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        digits = path.stem.rpartition("_")[2]
        if digits.isdigit():
            out[int(digits)] = path
    return out


def load_positions(preds_dir: Path, sequence: str) -> dict[int, np.ndarray]:
    """管线每帧的主目标位置（取分数最高的那个检测）。"""
    data = json.loads((preds_dir / f"{sequence}.json").read_text(encoding="utf-8"))
    out: dict[int, np.ndarray] = {}
    for key, rows in (data.get("per_frame") or {}).items():
        if not rows:
            continue
        best = max(rows, key=lambda item: float(item.get("score", 0.0)))
        position = best.get("pos")
        if position and len(position) >= 2:
            out[int(key)] = np.asarray(position[:2], dtype=np.float64)
    return out


def summarise(errors: list[float]) -> dict:
    if not errors:
        return {"n": 0}
    array = np.asarray(errors, dtype=np.float64)
    return {
        "n": int(array.size),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def main() -> None:
    args = parse_args()
    labels_root = Path(args.labels_root)
    frames_root = Path(args.frames_root)
    preds_dir = Path(args.preds_dir)
    sequences = [item for item in args.seqs.split(",") if item]
    win_sizes = [int(value) for value in args.win_sizes.split(",") if value]
    modes = [value for value in args.modes.split(",") if value]
    th_ks = [float(value) for value in args.th_ks.split(",") if value]
    delta = int(args.delta)

    report: dict = {"delta": delta, "sequences": sequences, "configs": []}
    for win_size in win_sizes:
        bg_window = int(args.bg_window) if args.bg_window else win_size * 2
        for mode in modes:
            for th_k in th_ks:
                tracker = LocalPatchTracker(
                    win_size=win_size,
                    th_k=th_k,
                    min_energy=1.0,
                    min_contrast=5.0,
                    mode=mode,
                    peak_window=9,
                    bg_window=max(bg_window, win_size),
                )
                single: list[float] = []
                drift: list[float] = []
                invalid_single = 0
                invalid_drift = 0
                total = 0
                for sequence in sequences:
                    available = frame_paths(frames_root, sequence)
                    truth = load_ground_truth(labels_root, sequence, set(available))
                    positions = load_positions(preds_dir, sequence)
                    frames = sorted(f for f in positions if f in available)
                    for frame in frames:
                        if frame % max(1, int(args.stride)):
                            continue
                        anchor = positions[frame]
                        nxt = frame + 1
                        if nxt not in available or nxt not in positions:
                            continue
                        image = cv2.imread(str(available[nxt]), cv2.IMREAD_GRAYSCALE)
                        if image is None:
                            continue
                        total += 1
                        result = tracker.step(image, anchor)
                        if result.valid:
                            single.append(float(np.linalg.norm(np.asarray(result.xy) - positions[nxt])))
                        else:
                            invalid_single += 1
                        # 多步漂移
                        position = anchor.copy()
                        ok = True
                        for step in range(1, delta + 1):
                            target = frame + step
                            if target not in available:
                                break
                            step_image = cv2.imread(str(available[target]), cv2.IMREAD_GRAYSCALE)
                            if step_image is None:
                                break
                            step_result = tracker.step(step_image, position)
                            if not step_result.valid:
                                ok = False
                                break
                            position = np.asarray(step_result.xy, dtype=np.float64)
                        if not ok:
                            invalid_drift += 1
                            continue
                        end = frame + delta
                        if end in positions:
                            drift.append(float(np.linalg.norm(position - positions[end])))
                entry = {
                    "win_size": win_size,
                    "bg_window": max(bg_window, win_size),
                    "mode": mode,
                    "th_k": th_k,
                    "total_frames": total,
                    "single_step_px": summarise(single),
                    "single_invalid_rate": (invalid_single / total) if total else None,
                    "drift_delta_px": summarise(drift),
                    "drift_invalid_rate": (invalid_drift / total) if total else None,
                }
                report["configs"].append(entry)
                print(
                    f"win={win_size:3d} bg={max(bg_window, win_size):3d} {mode:<13} th_k={th_k:<4} "
                    f"单步 p50={entry['single_step_px'].get('p50', float('nan')):6.2f} "
                    f"p95={entry['single_step_px'].get('p95', float('nan')):6.2f} "
                    f"无效率={100 * (entry['single_invalid_rate'] or 0):5.1f}% | "
                    f"Δ={delta} 漂移 p50={entry['drift_delta_px'].get('p50', float('nan')):7.2f} "
                    f"p95={entry['drift_delta_px'].get('p95', float('nan')):7.2f} "
                    f"失效={100 * (entry['drift_invalid_rate'] or 0):5.1f}%",
                    flush=True,
                )

    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[WROTE] {path}")


if __name__ == "__main__":
    main()
