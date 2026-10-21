#!/usr/bin/env python3
"""极性翻转(255 - I)前向探针:在"目标落入复杂地表背景"工况下,零训练能否救回暗色目标。

设计依据(先算清物理,再跑实验)
--------------------------------
本管线的特征是三通道,磁盘顺序 ``[I_t, GMC motion, median residual]``:

* ``motion   = absdiff(I_t, warp(I_{t-1}))`` —— **对极性不变**,翻转后逐位相同;
* ``residual = clip(I_t - B_t, 0, 255)``  —— ``B_t`` 是中值背景,单边截断。

把原始帧翻转 ``I' = 255 - I`` 之后,中值背景同样翻转(``B' = 255 - B``),于是

* ``motion'  = motion``                              (不变)
* ``residual' = clip((255-I) - (255-B), 0, 255)
                = clip(B - I, 0, 255)                (**极性反转**)

⇒ **暗机身(比背景暗)在翻转后 ``residual'`` 是正脉冲**,而这正是模型唯一会响应的方向;
同时通道 ``[residual', motion, 255-I]`` 的统计分布与训练分布完全不同(模型从没见过"暗目标"),
**能不能响必须实测**,这正是本探针要量的东西。

三个臂 + 一个融合方案
----------------------
============  ==============================================================
臂            含义
============  ==============================================================
normal        原始帧。与生产缓存同一套模型/尺度/门限/膨胀,逐位同口径。
invert        **物理翻转**:从翻转后的原始帧重算全部特征(主实验臂)。
tensor_flip   字面版 ``255 - input``,直接作用在 3 通道模型张量上(对照臂,物理不自洽)。
fusion        ``P = max(P_normal, P_invert)``(用户提的"走向 A"方案),在热力图像素级做。
============  ==============================================================

用法
----
    python manu/evaluation/invert_pass_probe.py \
        --seq VIDEO00007_19700101_001324 \
        --frames-root "/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg" \
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
        --cache-root runs/ms_cache_nop0 \
        --auto-segment 400 --out-dir runs/logs/invert_probe_001324
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.data.build_full_median_dataset import FastGMCEstimator  # noqa: E402
from manu.data.multiscale_heatmap_cache import (  # noqa: E402
    FUSED_TAG,
    compute_features,
    detect_components,
    heatmap_to_native,
    run_heatmaps,
    scale_inputs,
)
from manu.data.preannotate_hm_bbox import load_hm_model, resolve_weights  # noqa: E402
from manu.evaluation.eval_multiscale_cache_th import (  # noqa: E402
    load_ground_truth,
    merge_scales,
)

ARMS = ("normal", "invert", "tensor_flip", "abs_residual")


# ==============================================================================
# 1. 工具:逐帧 inbox 匹配(与 eval_multiscale_cache_th.count_matches 同规则)
# ==============================================================================


def point_in_box(box: np.ndarray, point: np.ndarray) -> bool:
    """纯 point-in-bbox:检出中心必须落在 GT 框内。"""
    half_w = box[2] * 0.5
    half_h = box[3] * 0.5
    return bool(abs(point[0] - box[0]) <= half_w and abs(point[1] - box[1]) <= half_h)


def score_detections(boxes: np.ndarray, detections: list[tuple]) -> tuple[int, int, list[int]]:
    """greedy 按分数降序,一个 GT 只吸收一个检出。返回 (tp, fp, 每个检出的判定)。"""
    ranked = sorted(range(len(detections)), key=lambda i: -detections[i][5])
    taken = np.zeros(len(boxes), dtype=bool)
    verdicts = [0] * len(detections)  # 0 = 未判定
    tp = 0
    for index in ranked:
        point = np.asarray(detections[index][:2], dtype=np.float32)
        if len(boxes):
            free = np.nonzero(~taken)[0]
            if len(free):
                distance = ((boxes[free, :2] - point) ** 2).sum(axis=1)
                nearest = int(free[int(np.argmin(distance))])
                if point_in_box(boxes[nearest], point):
                    taken[nearest] = True
                    tp += 1
                    verdicts[index] = 1
                    continue
        verdicts[index] = 2
    return tp, len(detections) - tp, verdicts


def dedup_by_radius(detections: list[tuple], radius: float) -> list[tuple]:
    """按生产口径做跨尺度去重:r 像素内只留分数最高的一个。"""
    if radius <= 0:
        return list(detections)
    kept: list[tuple] = []
    for item in sorted(detections, key=lambda d: -d[5]):
        x, y = float(item[0]), float(item[1])
        if any((x - float(k[0])) ** 2 + (y - float(k[1])) ** 2 <= radius * radius for k in kept):
            continue
        kept.append(item)
    return kept


# ==============================================================================
# 2. 片段选取:从单帧评估里抓"连续漏检"最密集的窗口
# ==============================================================================


def single_frame_hit_map(
    cache_root: Path, sequence: str, labels_root: Path, radius: float, threshold: float, floor: float = 0.0
) -> dict[int, bool]:
    """用现成缓存复现冻结单帧基线(radius=768 / th=0.22 / inbox),返回每帧是否命中 GT。

    口径与 runs/logs/eval_nop0_inbox_ext2.json 完全一致:tag='all'(三个尺度) + 跨尺度去重。
    """
    cache = pickle.loads((cache_root / f"{sequence}.pkl").read_bytes())
    meta = cache["meta"]
    width, height = meta["native_size"]
    truth = load_ground_truth(labels_root, sequence, width, height)

    per_scale: dict[int, dict[int, tuple]] = {}
    offset = cache["offset"]
    for position, frame_index in enumerate(cache["frame_index"]):
        start, end = offset[position], offset[position + 1]
        tags = cache["tag"][start:end]
        scores = cache["scores"][start:end]
        points = cache["points"][start:end]
        for tag in np.unique(tags):
            mask = tags == tag
            if int(tag) == FUSED_TAG:
                continue
            keep = mask & (scores.astype(np.float32) >= floor)
            if not keep.any():
                continue
            bucket = per_scale.setdefault(int(tag), {})
            bucket[int(frame_index)] = (
                np.asarray(scores[keep], dtype=np.float32),
                np.asarray(points[keep], dtype=np.float32),
            )
    merged = merge_scales(per_scale, radius)

    hits: dict[int, bool] = {}
    for frame_index, boxes in truth.items():
        scores, points = merged.get(frame_index, (np.zeros(0, np.float32), np.zeros((0, 2), np.float32)))
        keep = scores >= threshold
        if not keep.any():
            hits[frame_index] = False
            continue
        ranked = sorted(
            zip(scores[keep].tolist(), points[keep].tolist()), key=lambda item: -item[0]
        )
        taken = np.zeros(len(boxes), dtype=bool)
        hit = False
        for _, (x, y) in ranked:
            free = np.nonzero(~taken)[0]
            if not len(free):
                break
            distance = ((boxes[free, :2] - np.asarray([x, y], np.float32)) ** 2).sum(axis=1)
            nearest = int(free[int(np.argmin(distance))])
            if point_in_box(boxes[nearest], np.asarray([x, y], np.float32)):
                taken[nearest] = True
                hit = True
                break
        hits[frame_index] = hit
    return hits


def pick_densest_miss_window(hits: dict[int, bool], length: int) -> tuple[int, int, int]:
    """返回 (起始帧, 结束帧, 窗口内漏检帧数):漏检最密集的连续窗口。"""
    frames = sorted(hits)
    if not frames:
        raise RuntimeError("no frames with ground truth")
    best = (frames[0], frames[min(length, len(frames)) - 1], -1)
    for start in range(0, max(1, len(frames) - length + 1)):
        window = frames[start : start + length]
        misses = sum(1 for f in window if not hits.get(f, False))
        if misses > best[2]:
            best = (window[0], window[-1], misses)
    return best


# ==============================================================================
# 3. 特征构造:两个物理臂
# ==============================================================================


def compute_features_ex(
    current: np.ndarray,
    history: list[np.ndarray],
    median_window: int,
    temporal_stride: int,
    gmc: FastGMCEstimator,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """重算一帧的三通道特征,并额外返回**未截断**的有符号残差 ``I_t - B_t``。

    ``multiscale_heatmap_cache.compute_features`` 把残差截断在 0(``clip(I-B, 0, 255)``),
    负半轴的信息被丢掉了 —— 而"目标比背景暗"恰好在负半轴。要做双向绝对值残差,
    必须自己保留符号,所以这里复刻它的逻辑但多返回一项。

    返回 ``(feature, signed_residual, history)``。
    """
    lag = history[max(0, len(history) - temporal_stride)] if history else current
    motion = cv2.absdiff(current, gmc.warp(lag, gmc.compute_affine(lag, current)))
    aligned = []
    for step in range(1, median_window + 1):
        history_frame = (
            history[max(0, len(history) - step * temporal_stride)] if history else current
        )
        aligned.append(gmc.warp(history_frame, gmc.compute_affine(history_frame, current)))
    background = np.median(np.stack(aligned, axis=0), axis=0).astype(np.float32)
    signed = current.astype(np.float32) - background
    residual = np.clip(signed, 0, 255).astype(np.uint8)
    feature = np.stack([current, motion, residual], axis=2)
    history = history + [current]
    return feature, signed, history


def build_arm_features(
    frames_root: Path,
    sequence: str,
    start: int,
    end: int,
    invert: bool,
    median_window: int,
    temporal_stride: int,
    history_window: int,
    abs_residual: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """从原始帧重算三通道特征。

    ``invert=True`` 时先把原始帧取反再重算 —— 这才是物理翻转:
    ``motion' = motion``、``residual' = clip(B - I, 0, 255)``。

    ``abs_residual=True`` 时不改原始帧,只把残差通道换成**双向绝对值** ``|I - B|``
    (需要保留符号,所以走 :func:`compute_features_ex`)。这样暗目标在通道 0 上也是
    正脉冲,而通道 1/2 仍是正常极性 —— 模型看到的是"训练时见过的正残差团块"。
    """
    gmc = FastGMCEstimator(downscale=2)
    history: list[np.ndarray] = []
    features: list[np.ndarray] = []
    args = argparse.Namespace(median_window=median_window, temporal_stride=temporal_stride)
    for frame_index in range(start, end + 1):
        path = frames_root / sequence / f"frame_{frame_index:06d}.jpg"
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"unreadable frame {path}")
        if invert:
            gray = (255 - gray).astype(np.uint8)
        if abs_residual:
            feature, signed, history = compute_features_ex(
                gray, history, median_window, temporal_stride, gmc
            )
            feature = np.stack(
                [gray, feature[:, :, 1], np.clip(np.abs(signed), 0, 255).astype(np.uint8)], axis=2
            )
        else:
            chunk, history = compute_features([gray], history, args, gmc)
            feature = chunk[0]
        history = history[-history_window:]
        features.append(feature)
    return features, history


# ==============================================================================
# 4. 前向:把特征变成原生分辨率热力图 + 融合
# ==============================================================================


def run_arm(
    model,
    features: list[np.ndarray],
    device: torch.device,
    args: argparse.Namespace,
    tensor_flip: bool = False,
) -> list[np.ndarray]:
    """对一臂的特征跑多尺度前向,返回每帧的原生分辨率融合热力图。"""
    all_scales = [0] + args.scales
    height, width = features[0].shape[:2]
    fused_maps: list[np.ndarray] = []
    for start in range(0, len(features), args.chunk_size):
        chunk = features[start : start + args.chunk_size]
        canvases = np.zeros((len(chunk) * len(all_scales), 3, args.imgsz, args.imgsz), dtype=np.uint8)
        mappings = []
        for index, feature in enumerate(chunk):
            if tensor_flip:
                feature = (255 - feature.astype(np.int16)).astype(np.uint8)
            scale_canvases, chunk_mapping = scale_inputs(feature, args.scales, args.imgsz, height, width)
            canvases[index * len(all_scales) : (index + 1) * len(all_scales)] = np.stack(scale_canvases)
            mappings.append(chunk_mapping)
        heatmaps = run_heatmaps(model, canvases, device, args.batch_size)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for index in range(len(chunk)):
            span = slice(index * len(all_scales), (index + 1) * len(all_scales))
            native = [
                heatmap_to_native(heatmaps[span][scale_id], *mappings[index][scale_id], args.imgsz, width, height)
                for scale_id in range(len(all_scales))
            ]
            fused_maps.append(np.maximum.reduce(native))
    return fused_maps


# ==============================================================================
# 5. 指标
# ==============================================================================


def heatmap_stats(heatmap: np.ndarray, boxes: np.ndarray, threshold: float) -> dict:
    """GT 内 / GT 外的热力图统计:峰值、超过门限的像素数。"""
    mask = np.zeros(heatmap.shape, dtype=bool)
    for cx, cy, w, h in boxes:
        x0 = max(0, int(round(cx - w * 0.5)))
        x1 = min(heatmap.shape[1], int(round(cx + w * 0.5)) + 1)
        y0 = max(0, int(round(cy - h * 0.5)))
        y1 = min(heatmap.shape[0], int(round(cy + h * 0.5)) + 1)
        mask[y0:y1, x0:x1] = True
    inside = heatmap[mask]
    outside = heatmap[~mask]
    return {
        "hm_in_gt_max": float(inside.max()) if inside.size else 0.0,
        "hm_out_gt_max": float(outside.max()) if outside.size else 0.0,
        "fp_pixels": int((outside >= threshold).sum()),
        "gt_pixels": int(inside.size),
    }


def arm_metrics(
    heatmaps: list[np.ndarray],
    frames: list[int],
    truth: dict[int, np.ndarray],
    args: argparse.Namespace,
) -> list[dict]:
    """逐帧把一臂的融合热力图转成生产口径的检出,并打分。"""
    rows = []
    for heatmap, frame_index in zip(heatmaps, frames):
        boxes = truth.get(frame_index)
        boxes = (
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            if boxes is not None
            else np.zeros((0, 4), np.float32)
        )
        components = detect_components(
            heatmap, args.main_threshold, args.fusion_dilate, args.min_area, args.max_detections
        )
        detections = dedup_by_radius(components, args.dedup_radius)
        tp, fp, verdicts = score_detections(boxes, detections)
        stats = heatmap_stats(heatmap, boxes, args.main_threshold)
        rows.append(
            {
                "frame": frame_index,
                "n_gt": len(boxes),
                "tp": tp,
                "fp": fp,
                "n_det": len(detections),
                "verdicts": verdicts,
                **stats,
            }
        )
    return rows


def summarize(rows: list[dict], label: str) -> dict:
    gt_frames = [r for r in rows if r["n_gt"] > 0]
    if not gt_frames:
        return {"arm": label, "frames": len(rows)}
    tp = sum(r["tp"] for r in gt_frames)
    fp = sum(r["fp"] for r in gt_frames)
    gt = sum(r["n_gt"] for r in gt_frames)
    hit_frames = sum(1 for r in gt_frames if r["tp"] > 0)
    inside = [r["hm_in_gt_max"] for r in gt_frames]
    outside = [r["hm_out_gt_max"] for r in gt_frames]
    fp_pixels = sum(r["fp_pixels"] for r in gt_frames)
    return {
        "arm": label,
        "frames": len(rows),
        "gt_frames": len(gt_frames),
        "tp": tp,
        "fp": fp,
        "gt": gt,
        "recall": tp / gt if gt else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "f1": 2 * tp / (2 * tp + fp + gt) if (2 * tp + fp + gt) else 0.0,
        "hit_frame_rate": hit_frames / len(gt_frames),
        "hm_in_gt_median": float(np.median(inside)),
        "hm_in_gt_mean": float(np.mean(inside)),
        "hm_out_gt_median": float(np.median(outside)),
        "hm_out_gt_max": float(np.max(outside)),
        "fp_pixels_total": fp_pixels,
        "fp_pixels_per_frame": fp_pixels / len(gt_frames),
    }


# ==============================================================================
# 6. 可视化
# ==============================================================================


def overlay(gray: np.ndarray, heatmap: np.ndarray, boxes: np.ndarray, detections: list[tuple],
            verdicts: list[int], title: str) -> np.ndarray:
    """左:原图 + GT(绿)+ 检出(红=命中/橙=误检);右:热力图叠加。"""
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for cx, cy, w, h in boxes:
        x0, y0 = int(cx - w * 0.5), int(cy - h * 0.5)
        x1, y1 = int(cx + w * 0.5), int(cy + h * 0.5)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 1)
    for index, det in enumerate(detections):
        color = (0, 0, 255) if verdicts[index] == 1 else (0, 165, 255)
        cv2.circle(canvas, (int(det[0]), int(det[1])), 4, color, 1)
    colour = cv2.applyColorMap((np.clip(heatmap, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(canvas, 0.55, colour, 0.45, 0)
    cv2.putText(blended, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.hstack([canvas, blended])


# ==============================================================================
# 7. 主流程
# ==============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="255-I inversion probe on the complex-ground sequence")
    parser.add_argument("--seq", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--cache-root", default="runs/ms_cache_nop0")
    parser.add_argument("--out-dir", default="runs/logs/invert_probe")
    parser.add_argument("--hm-weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    parser.add_argument("--ablate-p0", action="store_true", help="Drop the P0 residual branch (matches the nop0 cache)")
    parser.add_argument("--device", default="0")
    # 生产口径
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--scales", default="160,80")
    parser.add_argument("--median-window", type=int, default=21)
    parser.add_argument("--temporal-stride", type=int, default=2)
    parser.add_argument("--region-threshold", type=float, default=0.06)
    parser.add_argument("--main-threshold", type=float, default=0.22)
    parser.add_argument("--fusion-dilate", type=int, default=32)
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--dedup-radius", type=float, default=768.0)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    # 片段
    parser.add_argument("--auto-segment", type=int, default=400, help="Length of the FN-dense window to probe")
    parser.add_argument("--start", type=int, default=-1, help="Explicit first frame (overrides --auto-segment)")
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=120, help="Frames of history before the segment")
    parser.add_argument("--arms", default="normal,invert,tensor_flip,abs_residual")
    parser.add_argument("--save-vis", type=int, default=6, help="How many visualisations to dump")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.scales = sorted({int(item) for item in args.scales.split(",") if item.strip() and int(item) > 0}, reverse=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_root = Path(args.frames_root)
    labels_root = Path(args.labels_root)
    cache_root = Path(args.cache_root)

    # ---- 1. 片段选取 ----
    if args.start >= 0 and args.end >= 0:
        seg_start, seg_end = args.start, args.end
        miss_note = "manual segment"
        hits = None
    else:
        print("[SEGMENT] 用冻结单帧基线(r=768/th=0.22/inbox)统计每帧是否命中,找漏检最密集窗口...", flush=True)
        hits = single_frame_hit_map(
            cache_root, args.seq, labels_root, args.dedup_radius, args.main_threshold, args.region_threshold
        )
        seg_start, seg_end, misses = pick_densest_miss_window(hits, args.auto_segment)
        total = sum(1 for v in hits.values() if not v)
        gt_in_window = sum(1 for f in hits if seg_start <= f <= seg_end)
        miss_note = (
            f"auto window: 帧 {seg_start}..{seg_end}({seg_end - seg_start + 1} 帧索引,其中 {gt_in_window} 帧有 GT)"
            f" | 漏检 {misses}/{gt_in_window} 个带 GT 的帧 | 全序列漏检 {total}/{len(hits)} 帧"
        )
    print(f"[SEGMENT] {miss_note}", flush=True)

    # ---- 2. 装载模型 + GT ----
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model, _ = load_hm_model(
        resolve_weights(args.hm_weights), device, use_p0_highway=not args.ablate_p0
    )
    model.eval()
    truth = load_ground_truth(labels_root, args.seq, 640, 512)
    frames = list(range(seg_start, seg_end + 1))
    history_window = args.median_window * args.temporal_stride + 2

    # ---- 3. 特征(两臂共用一次原始帧读取) ----
    warm_start = max(0, seg_start - args.warmup)
    print(f"[FEAT] 从帧 {warm_start} 开始重算特征(预热 {seg_start - warm_start} 帧)", flush=True)
    t0 = time.perf_counter()
    feats_normal, _ = build_arm_features(
        frames_root, args.seq, warm_start, seg_end, False,
        args.median_window, args.temporal_stride, history_window,
    )
    feats_invert, _ = build_arm_features(
        frames_root, args.seq, warm_start, seg_end, True,
        args.median_window, args.temporal_stride, history_window,
    )
    feats_abs: list[np.ndarray] = []
    if "abs_residual" in args.arms:
        feats_abs, _ = build_arm_features(
            frames_root, args.seq, warm_start, seg_end, False,
            args.median_window, args.temporal_stride, history_window, abs_residual=True,
        )
    print(f"[FEAT] 完成,耗时 {time.perf_counter() - t0:.1f}s", flush=True)

    # ---- 4. 前向 ----
    arm_heatmaps: dict[str, list[np.ndarray]] = {}
    for arm in ARMS:
        if arm not in args.arms.split(","):
            continue
        t0 = time.perf_counter()
        if arm == "normal":
            arm_heatmaps[arm] = run_arm(model, feats_normal, device, args)
        elif arm == "invert":
            arm_heatmaps[arm] = run_arm(model, feats_invert, device, args)
        elif arm == "abs_residual":
            arm_heatmaps[arm] = run_arm(model, feats_abs, device, args)
        else:
            arm_heatmaps[arm] = run_arm(model, feats_normal, device, args, tensor_flip=True)
        print(f"[FWD] {arm} 完成,耗时 {time.perf_counter() - t0:.1f}s", flush=True)

    # ---- 5. 指标 ----
    segment_feats = feats_normal[seg_start - warm_start :]
    arm_rows: dict[str, list[dict]] = {}
    for arm, maps in arm_heatmaps.items():
        arm_rows[arm] = arm_metrics(maps, frames, truth, args)
    if "normal" in arm_heatmaps and "invert" in arm_heatmaps:
        fused_maps = [
            np.maximum(a, b) for a, b in zip(arm_heatmaps["normal"], arm_heatmaps["invert"])
        ]
        arm_heatmaps["fusion"] = fused_maps
        arm_rows["fusion"] = arm_metrics(fused_maps, frames, truth, args)
    if "normal" in arm_heatmaps and "abs_residual" in arm_heatmaps:
        fused_abs = [
            np.maximum(a, b) for a, b in zip(arm_heatmaps["normal"], arm_heatmaps["abs_residual"])
        ]
        arm_heatmaps["fusion_abs"] = fused_abs
        arm_rows["fusion_abs"] = arm_metrics(fused_abs, frames, truth, args)

    summary = {arm: summarize(rows, arm) for arm, rows in arm_rows.items()}

    # 双流互补:normal 漏而 invert 中 / 反之
    if "normal" in arm_rows and "invert" in arm_rows:
        normal_hit = {r["frame"]: r["tp"] > 0 for r in arm_rows["normal"] if r["n_gt"] > 0}
        invert_hit = {r["frame"]: r["tp"] > 0 for r in arm_rows["invert"] if r["n_gt"] > 0}
        only_invert = [f for f in normal_hit if not normal_hit[f] and invert_hit.get(f)]
        only_normal = [f for f in normal_hit if normal_hit[f] and not invert_hit.get(f, False)]
        both = [f for f in normal_hit if normal_hit[f] and invert_hit.get(f)]
        neither = [f for f in normal_hit if not normal_hit[f] and not invert_hit.get(f, False)]
        summary["_overlap"] = {
            "both_hit": len(both),
            "only_normal_hit": len(only_normal),
            "only_invert_hit": len(only_invert),
            "neither_hit": len(neither),
            "only_invert_frames": only_invert[:40],
        }

    # 自洽校验:normal 臂(从原始帧重算特征)应与冻结单帧基线(缓存里的特征 JPG)基本一致。
    # 不完全一致是预期的 —— 缓存用的是预存特征 JPG(JPEG 有损),这里用的是原始帧重算。
    if hits is not None and "normal" in arm_rows:
        agree = 0
        compared = 0
        for row in arm_rows["normal"]:
            if row["n_gt"] == 0 or row["frame"] not in hits:
                continue
            compared += 1
            agree += int((row["tp"] > 0) == hits[row["frame"]])
        summary["_selfcheck"] = {
            "normal_vs_frozen_single_frame_agreement": agree / compared if compared else 0.0,
            "compared_frames": compared,
            "note": "normal 臂从原始帧重算特征,冻结基线用预存特征 JPG(JPEG 有损),存在轻微差异",
        }

    # ---- 6. 输出 ----
    report = {
        "sequence": args.seq,
        "segment": {"start": seg_start, "end": seg_end, "note": miss_note},
        "config": {
            "hm_weights": args.hm_weights,
            "use_p0_highway": not args.ablate_p0,
            "scales": [0] + args.scales,
            "imgsz": args.imgsz,
            "median_window": args.median_window,
            "temporal_stride": args.temporal_stride,
            "main_threshold": args.main_threshold,
            "fusion_dilate": args.fusion_dilate,
            "min_area": args.min_area,
            "dedup_radius": args.dedup_radius,
            "arms": args.arms,
        },
        "summary": summary,
        "per_frame": {arm: [{k: v for k, v in r.items() if k != "verdicts"} for r in rows]
                      for arm, rows in arm_rows.items()},
    }
    report_path = out_dir / f"{args.seq}_invert_probe.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        f"# 255-I 极性翻转前向探针 —— {args.seq}",
        "",
        f"片段:{seg_start}..{seg_end}({len(frames)} 帧)|{miss_note}",
        "",
        "| 臂 | TP | FP | GT | Recall% | Precision% | F1% | 命中帧率% | GT内峰值中位 | GT外峰值中位 | GT外FP像素/帧 |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("normal", "invert", "tensor_flip", "abs_residual", "fusion", "fusion_abs"):
        if arm not in summary:
            continue
        s = summary[arm]
        lines.append(
            f"| {arm} | {s['tp']} | {s['fp']} | {s['gt']} | {s['recall'] * 100:.2f} | {s['precision'] * 100:.2f} | "
            f"{s['f1'] * 100:.2f} | {s['hit_frame_rate'] * 100:.2f} | {s['hm_in_gt_median']:.3f} | "
            f"{s['hm_out_gt_median']:.3f} | {s['fp_pixels_per_frame']:.0f} |"
        )
    if "_overlap" in summary:
        o = summary["_overlap"]
        lines += [
            "",
            f"双流互补:两臂都中 {o['both_hit']} 帧 | 只有 normal 中 {o['only_normal_hit']} 帧 | "
            f"**只有 invert 中 {o['only_invert_hit']} 帧** | 都不中 {o['neither_hit']} 帧",
        ]
    if "_selfcheck" in summary:
        c = summary["_selfcheck"]
        lines += [
            "",
            f"自洽校验:normal 臂 vs 冻结单帧基线 一致率 = {c['normal_vs_frozen_single_frame_agreement'] * 100:.2f}%"
            f"(比对 {c['compared_frames']} 帧;{c['note']})",
        ]
    (out_dir / f"{args.seq}_invert_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)

    # ---- 7. 可视化 ----
    if args.save_vis > 0 and arm_heatmaps:
        vis_dir = out_dir / "vis"
        vis_dir.mkdir(exist_ok=True)
        gt_frames = [f for f in frames if f in truth]
        picks = gt_frames[:: max(1, len(gt_frames) // args.save_vis)][: args.save_vis]
        for frame_index in picks:
            position = frame_index - seg_start
            gray = cv2.imread(str(frames_root / args.seq / f"frame_{frame_index:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
            boxes = np.asarray(truth[frame_index], dtype=np.float32).reshape(-1, 4)
            panels = []
            for arm in ("normal", "invert", "tensor_flip", "abs_residual", "fusion", "fusion_abs"):
                if arm not in arm_heatmaps:
                    continue
                row = next(r for r in arm_rows[arm] if r["frame"] == frame_index)
                components = detect_components(
                    arm_heatmaps[arm][position], args.main_threshold, args.fusion_dilate, args.min_area, args.max_detections
                )
                detections = dedup_by_radius(components, args.dedup_radius)
                tp, fp, verdicts = score_detections(boxes, detections)
                panels.append(overlay(gray, arm_heatmaps[arm][position], boxes, detections, verdicts, f"{arm}"))
            cv2.imwrite(str(vis_dir / f"frame_{frame_index:06d}.png"), np.hstack(panels))
        print(f"[VIS] {len(picks)} 张对比图 -> {vis_dir}", flush=True)

    print(f"\n[OUT] {report_path}", flush=True)


if __name__ == "__main__":
    main()
