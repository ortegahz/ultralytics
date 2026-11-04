#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**快核能否穿越「检测器空窗」**：NPU 漏检 ≠ 目标在红外图上不可见。

检测器空窗 = 管线在该帧**没有任何输出**（NPU 没检出，且没有轨迹可以续）。
但快核（局部质心）比 NPU 灵敏得多：只要目标在 52×52 的 ROI 里还有亮度对比，
快核就能跟住。于是「源帧无检测 ⇒ 该帧无输出」这个硬约束，可能是**可以绕开**的。

本脚本对每段检测器空窗 (a, b)：
  1. 用**空窗前最后一帧**的管线输出作种子；
  2. 用快核逐帧跟 (a → b)，ROI 中心走「上一帧位置 + 链自身速度」（selfpred 口径，
     与交付档 ``--replay-roi-mode selfpred`` 一致）；
  3. 逐帧算到 GT 中心的距离，统计 hit@5px。

用法::

    python manu/tracking/lookahead/diag_gap_coast.py \
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
        --frames-root "/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg" \
        --frames-cache /mnt/data/siping/frame_cache_lq \
        --original runs/lookahead/final/preds_off_24seq \
        --seqs VIDEO00007_19700101_001324 --out-json ...json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.tracking.async_dual_rate.fast_tracker import LocalPatchTracker  # noqa: E402

IMG_W, IMG_H = 640, 512


def load_gt(labels_root: Path, seq: str) -> dict[int, list[tuple[float, float]]]:
    out: dict[int, list[tuple[float, float]]] = {}
    for f in sorted(labels_root.glob(f"{seq}__*.txt")):
        idx = int(f.stem.split("__")[-1])
        pts = []
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                pts.append((float(parts[1]) * IMG_W, float(parts[2]) * IMG_H))
        if pts:
            out[idx] = pts
    return out


def load_per_frame(preds_dir: Path, seq: str) -> dict[int, list[tuple[float, float]]]:
    path = preds_dir / f"{seq}.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8")).get("per_frame", {})
    return {
        int(k): [(float(d["pos"][0]), float(d["pos"][1])) for d in (dets or [])]
        for k, dets in payload.items()
    }


class Frames:
    def __init__(self, root: Path, cache: Path | None, seq: str):
        self.dir = root / seq
        self.array = None
        self.first = 0
        if cache is not None:
            npy, meta = cache / f"{seq}.npy", cache / f"{seq}.json"
            if npy.is_file() and meta.is_file():
                info = json.loads(meta.read_text(encoding="utf-8"))
                if info.get("contiguous"):
                    self.array = np.load(npy, mmap_mode="r")
                    self.first = int(info.get("first_frame", 0))

    def get(self, frame: int):
        if self.array is not None:
            position = int(frame) - self.first
            if position < 0 or position >= self.array.shape[0]:
                return None
            return np.array(self.array[position], dtype=np.uint8)
        path = self.dir / f"{frame:06d}.jpg"
        if not path.is_file():
            return None
        import cv2

        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--frames-cache", default="")
    ap.add_argument("--original", required=True)
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--min-gap", type=int, default=10)
    ap.add_argument("--fast-win-size", type=int, default=52)
    ap.add_argument("--fast-bg-window", type=int, default=104)
    ap.add_argument("--fast-th-k", type=float, default=2.5)
    ap.add_argument("--fast-min-energy", type=float, default=50.0)
    ap.add_argument("--selfpred-window", type=int, default=3)
    ap.add_argument("--selfpred-max-px", type=float, default=16.0)
    ap.add_argument("--hit-tol-px", type=float, default=5.0)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    labels_root = Path(args.labels_root)
    frames_root = Path(args.frames_root)
    cache = Path(args.frames_cache) if args.frames_cache else None
    orig_dir = Path(args.original)
    tracker = LocalPatchTracker(
        win_size=args.fast_win_size,
        th_k=args.fast_th_k,
        min_energy=args.fast_min_energy,
        min_contrast=0.0,
        mode="centroid",
        bg_window=args.fast_bg_window,
    )

    per_seq = {}
    for seq in [s.strip() for s in args.seqs.split(",") if s.strip()]:
        gt = load_gt(labels_root, seq)
        orig = load_per_frame(orig_dir, seq)
        frames = Frames(frames_root, cache, seq)
        empty = sorted(f for f, dets in orig.items() if not dets)
        gaps: list[tuple[int, int]] = []
        if empty:
            start = prev = empty[0]
            for f in empty[1:]:
                if f == prev + 1:
                    prev = f
                else:
                    gaps.append((start, prev))
                    start = prev = f
            gaps.append((start, prev))
        gaps = [(a, b) for a, b in gaps if b - a + 1 >= args.min_gap]

        n_frames = n_hit = n_valid = 0
        rows = []
        for a, b in gaps:
            seed_frame = a - 1
            while seed_frame >= 0 and not orig.get(seed_frame):
                seed_frame -= 1
            if seed_frame < 0 or not orig.get(seed_frame):
                continue
            position = np.asarray(orig[seed_frame][0], dtype=np.float64)
            steps: list[np.ndarray] = []
            hit = 0
            counted = 0
            for frame in range(a, b + 1):
                image = frames.get(frame)
                if image is None:
                    break
                if steps:
                    recent = steps[-args.selfpred_window :] if args.selfpred_window > 0 else steps
                    velocity = np.median(np.asarray(recent, dtype=np.float64), axis=0)
                    norm = float(np.linalg.norm(velocity))
                    if norm > args.selfpred_max_px > 0:
                        velocity = velocity * (args.selfpred_max_px / norm)
                    center = position + velocity
                else:
                    center = position
                result = tracker.step(image, center)
                if not result.valid:
                    steps.append(np.zeros(2))
                    continue
                new_position = np.asarray(result.xy, dtype=np.float64).reshape(2)
                steps.append(new_position - position)
                position = new_position
                n_valid += 1
                if frame in gt:
                    counted += 1
                    err = min(
                        math.hypot(position[0] - gx, position[1] - gy)
                        for gx, gy in gt[frame]
                    )
                    if err <= args.hit_tol_px:
                        hit += 1
            n_frames += counted
            n_hit += hit
            rows.append(
                {
                    "gap": [a, b],
                    "length": b - a + 1,
                    "frames_with_gt": counted,
                    "hit": hit,
                    "hit_rate": hit / counted if counted else 0.0,
                }
            )
        per_seq[seq] = {
            "gaps": len(gaps),
            "gap_frames": sum(r["length"] for r in rows),
            "frames_with_gt": n_frames,
            "hit": n_hit,
            "hit_rate": n_hit / n_frames if n_frames else 0.0,
            "rows": rows,
        }
        print(f"=== {seq}：{len(gaps)} 段空窗，共 {per_seq[seq]['gap_frames']} 帧 ===")
        print(
            f"    其中有 GT 的 {n_frames} 帧里，快核命中 {n_hit} 帧"
            f"（{per_seq[seq]['hit_rate'] * 100:.2f}%）"
        )
        for r in sorted(rows, key=lambda r: -r["length"])[:12]:
            print(
                f"    空窗 {r['gap'][0]}-{r['gap'][1]}（{r['length']:4d} 帧）"
                f" 有GT {r['frames_with_gt']:4d}  命中 {r['hit']:4d}  ({r['hit_rate'] * 100:5.1f}%)"
            )
    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(per_seq, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[SAVED] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
