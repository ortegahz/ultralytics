#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**位移估计器大比试**：在同一批真实帧上横向比较各种配准策略的位移误差。

§32 的证据链把问题逼到了这一步：
  * 锚点误差 p50 = 1.50 px / p99 = 4.54 px ⇒ 锚点几乎完美；
  * ``oracle_disp``（位移完美）hit@5px = 87.80% vs 实测 85.22% ⇒ **剩余 2.58 pp
    全在位移估计**；
  * 但 74.48% 的失手帧真实位移 ≤ 搜索半径(32px) ⇒ **不是"搜不到"，是"配不准"**。

所以真正该问的不是「半径够不够」，而是「**哪种配准方式能把位移估得最准**」。
本脚本在**同一批帧**（同一源帧、同一目标帧、同一真值位移）上比较：

  * ``direct``    ：一次跨 Δ 帧的大匹配（当前交付档的做法）；
  * ``chain``     ：逐帧配准，模板每步刷新；
  * ``strideN``   ：每 N 帧配一次、模板每段刷新（**分段配准**，本次新增候选）；
  * ``direct_c``  ：大匹配 + 链式精修（``chain_direct``）。

每个估计器报三个量：
  * 位移误差 p50 / p90 / p95 / max（对**真值位移**）；
  * 由此推出的**上报位置**误差 p50 / p95（对**当前帧真值中心**）；
  * 由此推出的 hit@5px（上报位置到当前帧 GT 中心 ≤ 5 px）。

⚠️ 口径：真值位移 = GT(t) − GT(t−Δ) 的中心差；上报位置 = 源帧锚点 + 估计位移。
锚点取**交付档管线在源帧的输出**（``preds_off``），与真实系统一致。

用法::

    python manu/tracking/lookahead/bench_xcorr_estimators.py \
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
        --frames-root "/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg" \
        --frames-cache /mnt/data/siping/frame_cache_lq \
        --original runs/lookahead/final/preds_off \
        --seqs VIDEO00007_19700101_001324 --delta 8 --max-frames 1500 \
        --out-json runs/lookahead/final/final8/bench_xcorr_estimators.json
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

from manu.evaluation.eval_multiscale_cache_th import load_ground_truth  # noqa: E402
from manu.tracking.lookahead.xcorr import chain_step, make_template, match  # noqa: E402

IMG_W, IMG_H = 640, 512


def load_per_frame(preds_dir: Path, seq: str) -> dict[int, list[tuple[float, float]]]:
    path = preds_dir / f"{seq}.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8")).get("per_frame", {})
    return {
        int(key): [(float(d["pos"][0]), float(d["pos"][1])) for d in (dets or [])]
        for key, dets in payload.items()
    }


class Frames:
    """按视频帧号取灰度帧；优先本地 memmap 缓存，退回 JPEG 目录。"""

    def __init__(self, root: Path, cache: Path | None, seq: str):
        self.dir = root / seq
        self.array = None
        self.first = 0
        if cache is not None:
            npy = cache / f"{seq}.npy"
            meta = cache / f"{seq}.json"
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

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return image


def centre(boxes) -> np.ndarray:
    return np.asarray(boxes, dtype=np.float64)[:, :2]


def estimate_direct(src, dst, anchor, tsize, bg_size, th_k, radius, polarity, center=None):
    """一次跨 Δ 帧的大匹配。

    ``center`` = 搜索中心。**必须与管线一致**：管线用「重推链的末位置」当搜索中心
    （链已经把目标跟到附近，大匹配只在链附近 ±radius 里精修）；
    用 anchor 当搜索中心会**大幅低估** direct（实测 001324：74.2% vs 管线 77.2%）。
    """
    search_center = anchor if center is None else center
    template, origin = make_template(src, anchor, tsize=tsize, bg_size=bg_size, th_k=th_k, polarity=polarity)
    if template is None:
        return None
    displacement, peak = match(
        dst, template, anchor, search_center, radius=radius, tsize=tsize, origin=origin, polarity=polarity
    )
    if displacement is None:
        return None
    return np.asarray(anchor, dtype=np.float64).reshape(2) + displacement, peak


def estimate_chain(src_frames, dst_frames, anchor, tsize, bg_size, th_k, radius, polarity, stride,
                   on_fail="hold", global_src=None, global_dst=None):
    """逐 ``stride`` 帧配一次、模板每段刷新；返回 (末位置, 最小峰值)。

    ``on_fail``：
      * ``hold``    —— 配准失败就沿用最后一步位移（管线当前行为）；
      * ``global``  —— 配准失败时改用**全局大匹配**（源帧模板直接配到当前帧）兜底。
    """
    position = np.asarray(anchor, dtype=np.float64).reshape(2).copy()
    worst = 1.0
    last_step = np.zeros(2, dtype=np.float64)
    for index in range(len(src_frames)):
        step = chain_step(
            dst_frames[index],
            src_frames[index],
            position,
            position,
            radius=radius,
            tsize=tsize,
            bg_size=bg_size,
            th_k=th_k,
            min_peak=0.0,
            polarity=polarity,
        )
        if step[0] is None:
            if on_fail == "global" and global_src is not None and global_dst is not None:
                fallback = estimate_direct(
                    global_src, global_dst, anchor, tsize, bg_size, th_k, radius, polarity
                )
                if fallback is not None:
                    position = fallback[0]
                    last_step = position - np.asarray(anchor, dtype=np.float64).reshape(2)
                    continue
            position = position + last_step
            continue
        previous = position
        position = np.asarray(step[0], dtype=np.float64).reshape(2)
        last_step = position - previous
        worst = min(worst, float(step[1]))
    return position, worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--frames-cache", default="")
    ap.add_argument("--original", required=True)
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--delta", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=1500)
    ap.add_argument("--tsize", type=int, default=32)
    ap.add_argument("--bg-size", type=int, default=96)
    ap.add_argument("--th-k", type=float, default=1.5)
    ap.add_argument("--radius", type=int, default=32)
    ap.add_argument("--strides", default="1,2,4")
    ap.add_argument("--hit-tol-px", type=float, default=5.0)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    labels_root = Path(args.labels_root)
    frames_root = Path(args.frames_root)
    cache = Path(args.frames_cache) if args.frames_cache else None
    orig_dir = Path(args.original)
    delta = int(args.delta)
    strides = [int(s) for s in args.strides.split(",") if s.strip()]

    per_seq: dict[str, dict] = {}

    for seq in args.seqs.split(","):
        seq = seq.strip()
        truth = load_ground_truth(labels_root, seq, IMG_W, IMG_H)
        orig = load_per_frame(orig_dir, seq)
        frames = Frames(frames_root, cache, seq)
        # 收集可用的 (源帧, 目标帧, 锚点, 真值位移, 当前帧真值中心)
        samples = []
        for target in sorted(truth):
            source = target - delta
            if source < 0 or source not in truth or not orig.get(source):
                continue
            src_boxes = truth.get(source)
            if src_boxes is None or not len(src_boxes):
                continue
            anchor_pts = orig[source]
            src_centres = centre(src_boxes)
            dst_centres = centre(truth[target])
            distances = np.linalg.norm(
                np.asarray(anchor_pts, dtype=np.float64)[:, None, :] - src_centres[None], axis=2
            )
            best = np.unravel_index(int(np.argmin(distances)), distances.shape)
            anchor = np.asarray(anchor_pts[best[0]], dtype=np.float64)
            true_disp = dst_centres[int(best[1])] - src_centres[int(best[1])]
            samples.append((target, source, anchor, true_disp, dst_centres))
            if len(samples) >= args.max_frames:
                break
        if not samples:
            continue

        strategies = ["direct"] + [f"stride{s}" for s in strides] + [f"stride{s}g" for s in strides]
        acc: dict[str, list] = {name: {"disp": [], "pos": [], "hit": 0, "n": 0, "fail": 0} for name in strategies}

        for target, source, anchor, true_disp, dst_centres in samples:
            src_img = frames.get(source)
            dst_img = frames.get(target)
            if src_img is None or dst_img is None:
                continue
            for name in strategies:
                if name == "direct":
                    estimate = estimate_direct(
                        src_img, dst_img, anchor, args.tsize, args.bg_size, args.th_k, args.radius, 1
                    )
                    position = None if estimate is None else estimate[0]
                else:
                    label = name[:-1] if name.endswith("g") else name
                    stride = int(label.replace("stride", ""))
                    on_fail = "global" if name.endswith("g") else "hold"
                    src_frames, dst_frames = [], []
                    cursor = source
                    ok = True
                    while cursor + stride <= target:
                        a = frames.get(cursor)
                        b = frames.get(cursor + stride)
                        if a is None or b is None:
                            ok = False
                            break
                        src_frames.append(a)
                        dst_frames.append(b)
                        cursor += stride
                    if not ok or not src_frames:
                        position = None
                    else:
                        position, _ = estimate_chain(
                            src_frames, dst_frames, anchor, args.tsize, args.bg_size, args.th_k,
                            args.radius, 1, stride, on_fail=on_fail,
                            global_src=src_img, global_dst=dst_img,
                        )
                slot = acc[name]
                slot["n"] += 1
                if position is None:
                    slot["fail"] += 1
                    continue
                estimated_disp = position - anchor
                slot["disp"].append(float(np.linalg.norm(estimated_disp - true_disp)))
                reported = anchor + estimated_disp
                error = min(float(np.linalg.norm(reported - c)) for c in dst_centres)
                slot["pos"].append(error)
                if error <= args.hit_tol_px:
                    slot["hit"] += 1

        results: dict[str, dict] = {}
        for name in strategies:
            slot = acc[name]
            disp = np.asarray(slot["disp"], dtype=np.float64)
            pos = np.asarray(slot["pos"], dtype=np.float64)
            results[name] = {
                "samples": slot["n"],
                "failed": slot["fail"],
                "disp_p50": float(np.percentile(disp, 50)) if disp.size else float("nan"),
                "disp_p90": float(np.percentile(disp, 90)) if disp.size else float("nan"),
                "disp_p95": float(np.percentile(disp, 95)) if disp.size else float("nan"),
                "disp_max": float(disp.max()) if disp.size else float("nan"),
                "pos_p50": float(np.percentile(pos, 50)) if pos.size else float("nan"),
                "pos_p95": float(np.percentile(pos, 95)) if pos.size else float("nan"),
                "hit_rate_5px": slot["hit"] / slot["n"] * 100.0 if slot["n"] else 0.0,
            }
        per_seq[seq] = results
        print(f"=== {seq}（{len(samples)} 个样本）===")
        print(
            f"{'策略':12s} {'n':>6s} {'失败':>5s} {'位移p50':>8s} {'位移p90':>8s} {'位移p95':>8s} "
            f"{'位移max':>9s} {'位置p50':>8s} {'位置p95':>8s} {'hit@5':>8s}"
        )
        for name in strategies:
            row = results[name]
            print(
                f"{name:12s} {row['samples']:6d} {row['failed']:5d} {row['disp_p50']:8.3f} "
                f"{row['disp_p90']:8.3f} {row['disp_p95']:8.3f} {row['disp_max']:9.2f} "
                f"{row['pos_p50']:8.3f} {row['pos_p95']:8.3f} {row['hit_rate_5px']:7.3f}%"
            )

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"per_sequence": per_seq, "config": vars(args)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[SUCCESS] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
