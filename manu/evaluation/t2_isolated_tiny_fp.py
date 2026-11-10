#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""T2「孤立微小假峰 FP」监控台 —— 龙泉山黄金池（官方三轨门禁的 T2 轨）。

这是**纯新增**的板端交付监控台，不修改任何已有文件。它提供三个子命令：

* ``build``   —— 封装 ``manu/data/multiscale_heatmap_cache.py``，为某个候选模型在 **no-GMC** 龙泉山
  特征目录上重建多尺度热图缓存。
* ``report``  —— 读缓存 + Final_Labels GT，输出 T2 的 TP/FP/FN/F1，以及新增的「孤立微小假峰 FP」
  细分（按物理尺寸分档 + 按孤立度分档）。
* ``compare`` —— 并排比较两个及以上缓存（候选 vs 参照）。

口径红线
--------
* **no-GMC 与 GMC 严禁混用**：本工具只认 no-GMC 特征目录（路径含 ``nogmc``）；缓存 ``meta`` 里记录的
  ``features_root`` 也会被复查，命中 GMC 目录直接拒绝运行（除非显式 ``--allow-non-nogmc``）。
* **匹配口径 = inbox**（预测中心落在 GT 框内），与 ``eval_multiscale_cache_th.py`` 完全一致。
* **阈值选择直接复用** ``eval_multiscale_cache_th.py`` 的扫阈值逻辑：同一张阈值网格、同一套
  ``count_matches``、同一个 best 选取规则（先 F1 后 Precision，取首个最大值），因此本工具报出的
  TP/FP/FN/F1 与 ``eval_multiscale_cache_th.py`` **逐位相同**。

「孤立微小假峰」的定义
----------------------
* **物理尺寸**：FP 所在连通域在 **native 640x512 像素**下的 bbox 最大边。
  注意融合行（tag=255）在建缓存时被 ``--fusion-dilate 32`` 膨胀过 33x33 核，它的 bbox 最大边恒 ≥33px，
  根本分不出「1~8px 的微小假峰」；所以尺寸默认归因到**未膨胀的原生尺度行（tag=0）**，再退化到粗尺度行
  （tag=1/2），归因不上的记 ``unknown``（绝不静默丢弃）。
  粗尺度（160/80）的连通域在 native 网格上是量化过的（scale 80 一格 ≈ 8×6.4 物理像素），因此由粗尺度
  归因出来的尺寸是**上界**；报告按「档位 × 归因来源」分列，避免把量化误差当成真实尺寸。
* **孤立度**：该 FP 到**同帧**其它 FP 的最近距离 > 8px 即记为孤立（单点帧天然孤立）。
* **主指标**：``孤立微小假峰 FP`` = 尺寸 ≤8px（nano + micro）且孤立的 FP 数。
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.eval_multiscale_cache_th import (  # noqa: E402
    COUNT_KEYS,
    _hits,
    count_matches,
    load_ground_truth,
    load_predictions,
    merge_scales,
    merge_scales_with_attrs,
    metrics,
    resolve_sources,
)

# --------------------------------------------------------------------------------------
# 常量：官方三轨门禁里的 T2 黄金池 / T3 blind 序列（与 manu/training/preflight_check.py 同源）
# --------------------------------------------------------------------------------------
NOGMC_FEATURES_ROOT = "/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc/images/val"
LABELS_ROOT = "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
T2_GOLDEN_SEQUENCES = (
    "VIDEO00009_19700101_001531",
    "VIDEO00029_19700101_014053",
    "VIDEO00033_19700101_015000",
    "VIDEO00034_19700101_015052",
    "VIDEO00006_19700101_000703",
    "VIDEO00008_19700101_001307",
    "VIDEO00007_19700101_000818",
)
T3_BLIND_SEQUENCES = ("VIDEO00007_19700101_001324",)

# FP 尺寸分档（按物理像素最大边，native 640x512）
SIZE_BUCKETS = ("nano", "micro", "small", "regular")
SIZE_BUCKET_EDGES = {"nano": (0.0, 4.0), "micro": (4.0, 8.0), "small": (8.0, 16.0), "regular": (16.0, float("inf"))}
TINY_MAX_EDGE = 8.0  # 「微小」= 最大边 ≤ 8px（nano + micro）


# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------
def assert_nogmc(path_text: str, allow: bool, what: str) -> None:
    """no-GMC 红线：路径里必须带 ``nogmc`` 标记，否则拒绝使用。

    官方口径规定 no-GMC 与 GMC 严禁混用，而 ``longquanshan_ft_gmc_median_nogmc`` 本身含有
    ``gmc_median`` 子串，所以这里只认 ``nogmc`` 标记，不做子串排除。
    """
    lowered = path_text.lower()
    if "nogmc" in lowered:
        return
    message = f"[红线] {what} 不带 nogmc 标记，按红线拒绝使用: {path_text}"
    if allow:
        print(f"[WARN] {message}（已用 --allow-non-nogmc 越过，仅限排查）", flush=True)
        return
    raise SystemExit(message + "\n        确实要跑 GMC 口径请显式加 --allow-non-nogmc。")


def bucket_of(edge: float | None) -> str:
    """按物理最大边（native px）把一个 FP 归到尺寸档。"""
    if edge is None:
        return "unknown"
    if edge <= 4.0:
        return "nano"
    if edge <= 8.0:
        return "micro"
    if edge <= 16.0:
        return "small"
    return "regular"


def nearest_other_distance(centres: np.ndarray) -> np.ndarray:
    """每个点到**其它**点的最近距离（自身排除）。单点返回 inf。"""
    count = len(centres)
    if count == 0:
        return np.zeros((0,), dtype=np.float64)
    if count == 1:
        return np.full((1,), np.inf, dtype=np.float64)
    diff = centres[:, None, :] - centres[None, :, :]
    distance = np.sqrt((diff * diff).sum(axis=2))
    np.fill_diagonal(distance, np.inf)
    return distance.min(axis=1)


def nearest_other_distance_chunked(centres: np.ndarray, block: int = 512) -> np.ndarray:
    """分块版最近邻距离，避免大序列一次性开 NxN 矩阵。"""
    count = len(centres)
    if count == 0:
        return np.zeros((0,), dtype=np.float64)
    if count == 1:
        return np.full((1,), np.inf, dtype=np.float64)
    result = np.full((count,), np.inf, dtype=np.float64)
    for start in range(0, count, block):
        stop = min(start + block, count)
        diff = centres[start:stop, None, :] - centres[None, :, :]
        distance = np.sqrt((diff * diff).sum(axis=2))
        for local in range(stop - start):
            distance[local, start + local] = np.inf
        result[start:stop] = distance.min(axis=1)
    return result


# --------------------------------------------------------------------------------------
# 缓存读取（行号回查 + 尺寸归因用的逐帧表）
# --------------------------------------------------------------------------------------
def build_frame_tables(cache: dict) -> dict[int, dict[str, np.ndarray]]:
    """按帧整理缓存行：points / sizes / scores / tag，供尺寸归因与行号回查。"""
    tables: dict[int, dict[str, np.ndarray]] = {}
    offset = cache["offset"]
    for position, frame_index in enumerate(cache["frame_index"]):
        start, end = offset[position], offset[position + 1]
        if end <= start:
            continue
        tables[int(frame_index)] = {
            "points": cache["points"][start:end],
            "sizes": cache["sizes"][start:end],
            "scores": cache["scores"][start:end].astype(np.float32),
            "tag": cache["tag"][start:end],
        }
    return tables


def load_predictions_with_row_index(cache_root: Path, sequence: str, tags: list[int], floor: float) -> dict:
    """镜像官方 ``load_predictions``，但额外把每行在帧内的**原始行号**放进 tag 槽。

    行号用来在合并之后回查该候选在缓存里的 ``sizes``（融合行被膨胀过，原生尺度行没有）。
    本函数的结果会与官方 ``load_predictions`` 逐行比对，不一致直接报错。
    """
    cache = pickle.loads((cache_root / f"{sequence}.pkl").read_bytes())
    by_tag: dict[int, dict[int, tuple]] = {int(tag): {} for tag in tags}
    offset = cache["offset"]
    for position, frame_index in enumerate(cache["frame_index"]):
        start, end = offset[position], offset[position + 1]
        if end <= start:
            continue
        points = cache["points"][start:end]
        scores = cache["scores"][start:end]
        sums = cache["sums"][start:end]
        area = cache["area"][start:end]
        row_tag = cache["tag"][start:end]
        rows = np.isin(row_tag, tags)
        if not rows.any():
            continue
        selected = np.nonzero(rows & (scores.astype(np.float32) >= floor))[0]
        if not len(selected):
            continue
        order = selected[np.argsort(-scores[selected], kind="stable")]
        for tag in np.unique(row_tag[order]):
            subset = order[row_tag[order] == tag]
            by_tag[int(tag)][int(frame_index)] = (
                scores[subset].astype(np.float32),
                points[subset],
                area[subset].astype(np.int32),
                sums[subset].astype(np.float32),
                subset.astype(np.int16),  # ← 行号（占用 tag 槽；本行数远小于 int16 上限）
            )
    return by_tag


def check_loader_matches_official(cache_root: Path, sequence: str, tags: list[int], floor: float, by_tag: dict) -> None:
    """校验自建 loader 与官方 ``load_predictions`` 选行、排序完全一致。"""
    _, reference = load_predictions(cache_root, sequence, tags, floor)
    for tag, frames in reference.items():
        mine = by_tag.get(tag, {})
        if set(frames) != set(mine):
            raise RuntimeError(f"{sequence}: tag {tag} 的帧集合与官方口径不一致")
        for frame_index, (scores, points) in frames.items():
            my_scores, my_points = mine[frame_index][0], mine[frame_index][1]
            if not np.array_equal(my_scores, scores) or not np.array_equal(my_points, points):
                raise RuntimeError(f"{sequence}: tag {tag} frame {frame_index} 的候选与官方口径不一致")


def merge_with_attrs(by_tag: dict, tag_mode: str, fused_tag: int, scale_tags: list[int], radius: float) -> dict:
    """用官方 ``_merge_scale_rows`` 合并，并逐帧校验与 ``merge_scales`` 的分数/坐标完全一致。"""
    sources = dict(enumerate(resolve_sources(by_tag, tag_mode, fused_tag, scale_tags)))
    merged = merge_scales_with_attrs(sources, radius)
    plain = merge_scales(sources, radius)
    if set(merged) != set(plain):
        raise RuntimeError("合并结果的帧集合与官方口径不一致")
    for frame_index, payload in merged.items():
        reference_scores, reference_points = plain[frame_index]
        if not np.array_equal(payload[0], reference_scores) or not np.array_equal(payload[1], reference_points):
            raise RuntimeError(f"合并结果与官方口径不一致: frame {frame_index}")
    return merged


def build_frames_with_index(truth: dict, predictions: dict) -> list[tuple]:
    """官方 ``build_frames`` 的带帧号版本（帧集合与顺序一致，判定结果与顺序无关）。"""
    merged = []
    for frame_index in set(truth) | set(predictions):
        centres = truth.get(frame_index)
        found = predictions.get(frame_index)
        merged.append((int(frame_index), centres, found[1] if found else None, found[0] if found else None))
    return merged


def extract_frame_matches(
    boxes: np.ndarray | None, points: np.ndarray, scores: np.ndarray, threshold: float, tol: float, mode: str
) -> tuple[list[int], list[int]]:
    """复刻官方 ``count_matches`` 的逐帧贪心匹配，只为了拿回「哪些检出是 FP」这个身份信息。

    返回 ``(tp_rows, fp_rows)``，下标是**过阈值之后**的候选序号。匹配规则与官方逐行一致：
    分数降序贪心、GT 最近邻、``inbox`` = 预测中心落在 GT 框内、每个 GT 只吸收一个检出。
    """
    tol_squared = tol * tol
    detected_mask = scores >= threshold
    detected = int(detected_mask.sum())
    truth_count = 0 if boxes is None else len(boxes)
    if detected == 0:
        return [], []
    filtered_points = points[detected_mask]
    taken = np.zeros(truth_count, dtype=bool)
    tp_rows: list[int] = []
    fp_rows: list[int] = []
    for position in range(detected):
        if truth_count == 0:
            fp_rows.append(position)
            continue
        free = np.nonzero(~taken)[0]
        if not len(free):
            fp_rows.append(position)
            continue
        distance = ((boxes[free, :2] - filtered_points[position]) ** 2).sum(axis=1)
        nearest = int(np.argmin(distance))
        box = free[nearest]
        if _hits(boxes[box], filtered_points[position], mode, tol_squared):
            taken[box] = True
            tp_rows.append(position)
        else:
            fp_rows.append(position)
    return tp_rows, fp_rows


def attribute_size(
    frame_table: dict | None,
    x: float,
    y: float,
    radius: float,
    native_tag: int,
    coarse_tags: list[int],
) -> tuple[float | None, int | None, float | None]:
    """给一个 FP 归因「物理最大边」。

    在同一帧的所有**未膨胀**尺度行里找证据：中心落在 ``radius`` 内、分数最高的那一行
    （同分取更近的，再同分取更细的尺度）。融合行是被 33x33 核膨胀过的，它的连通域形心相对真实峰值
    最多能偏出约 16px，所以归因半径默认取 32px（膨胀半径的一半）。

    返回 ``(最大边, 归因来源 tag, 归因距离)``；归因不到时三项都是 None。
    """
    if frame_table is None:
        return None, None, None
    best: tuple | None = None
    for order, tag in enumerate([native_tag, *coarse_tags]):
        rows = frame_table["tag"] == tag
        if not rows.any():
            continue
        points = frame_table["points"][rows]
        edges = np.maximum(frame_table["sizes"][rows][:, 0], frame_table["sizes"][rows][:, 1]).astype(np.float32)
        scores = frame_table["scores"][rows]
        distance = np.hypot(points[:, 0] - x, points[:, 1] - y)
        for local in np.nonzero(distance <= radius)[0]:
            key = (-float(scores[local]), float(distance[local]), order)
            if best is None or key < best[:3]:
                best = (*key, int(local), tag, float(edges[local]))
    if best is None:
        return None, None, None
    return best[5], best[4], best[1]


# --------------------------------------------------------------------------------------
# 核心统计
# --------------------------------------------------------------------------------------
def analyse(cache_root: Path, labels_root: Path, sequences: list[str], options: dict) -> dict:
    """跑完 T2 判定 + FP 细分，返回可 JSON 化的结果字典。"""
    tag_mode = options["tag"]
    match_mode = options["match_mode"]
    tol = options["tol"]
    th_min = options["th_min"]
    thresholds = np.round(np.arange(th_min, options["th_max"] + 1e-9, options["th_step"]), 6)
    radii = options["radii"]
    size_source = options["size_source"]
    attr_radius = options["size_attr_radius"]
    isolation_radius = options["isolation_radius"]
    isolation_scope = options["isolation_scope"]

    started = time.perf_counter()
    loaded: dict[str, tuple] = {}
    cache_tags: dict[str, list[int]] = {}
    meta_sample = None
    for sequence in sequences:
        cache = pickle.loads((cache_root / f"{sequence}.pkl").read_bytes())
        meta = cache["meta"]
        features_root = str(meta.get("features_root", "") or "")
        if features_root:
            assert_nogmc(features_root, options["allow_non_nogmc"], f"{sequence} 缓存记录的 features_root")
        if meta_sample is None:
            meta_sample = meta
        tags = sorted(int(value) for value in np.unique(cache["tag"]))
        cache_tags[sequence] = tags
        selected = tags if tag_mode == "all" else [int(tag_mode)] if int(tag_mode) in tags else tags
        by_tag = load_predictions_with_row_index(cache_root, sequence, selected, th_min)
        check_loader_matches_official(cache_root, sequence, selected, th_min, by_tag)
        truth = load_ground_truth(labels_root, sequence, meta["native_size"][0], meta["native_size"][1])
        tables = build_frame_tables(cache)
        loaded[sequence] = (by_tag, truth, meta, tables)

    fused_tag = int(meta_sample.get("fused_tag", 255))
    # scale_tags 是官方 resolve_sources 用的（只含被选中的 tag），保持原样以保证口径一致
    scale_tags = [tag for tag in range(len(meta_sample["scales"])) if tag in loaded[sequences[0]][0]]
    # 尺寸归因要用「缓存里真实存在的全部尺度行」：官方那条 scale_tags 在 tag=255 模式下是空的，
    # 拿它做归因会把粗尺度（160/80）证据整批丢掉，FP 会被误判成 unknown
    per_scale_tags = [tag for tag in range(len(meta_sample["scales"])) if tag in cache_tags[sequences[0]]]
    native_tag = 0
    for index, scale in enumerate(meta_sample["scales"]):
        if int(scale) == 0:
            native_tag = index
            break
    coarse_tags = [tag for tag in per_scale_tags if tag != native_tag and tag != fused_tag]

    # ---- 扫阈值（与官方 eval_multiscale_cache_th.py 同网格、同选取规则） ----
    sweep: list[dict] = []
    for radius in radii:
        merged_frames: dict[str, list[tuple]] = {}
        for sequence, (by_tag, truth, _meta, _tables) in loaded.items():
            merged = merge_with_attrs(by_tag, tag_mode, fused_tag, scale_tags, radius)
            merged_frames[sequence] = build_frames_with_index(truth, merged)
        counter_rows = {
            sequence: [(centres, points, scores) for _index, centres, points, scores in frames]
            for sequence, frames in merged_frames.items()
        }
        for threshold in thresholds:
            totals = dict.fromkeys(COUNT_KEYS, 0)
            frame_total = 0
            for sequence, rows in counter_rows.items():
                for key, value in count_matches(rows, float(threshold), tol, match_mode).items():
                    totals[key] += value
                frame_total += int(loaded[sequence][2]["frames"])
            row = metrics(totals, frame_total)
            row["th"] = float(threshold)
            row["merge_radius"] = float(radius)
            sweep.append(row)

    best = max(sweep, key=lambda row: (row["f1"], row["precision"]))
    active = best
    if options["th"] is not None:
        wanted = round(float(options["th"]), 6)
        candidates = [row for row in sweep if row["th"] == wanted and row["merge_radius"] == best["merge_radius"]]
        if not candidates:
            raise SystemExit(f"--th {options['th']} 不在官方阈值网格上（{th_min}~{options['th_max']} 步长 {options['th_step']}）")
        active = candidates[0]

    # ---- FP 细分 ----
    per_sequence: list[dict] = []
    size_totals = {name: 0 for name in (*SIZE_BUCKETS, "unknown")}
    isolation_totals = {"isolated": 0, "clustered": 0}
    attribution_totals = {"native": 0, "coarse": 0, "fused": 0, "unknown": 0}
    attribution_by_bucket = {
        name: {"native": 0, "coarse": 0, "fused": 0, "unknown": 0} for name in (*SIZE_BUCKETS, "unknown")
    }
    attr_distances: list[float] = []
    nearest_distances: list[float] = []
    solo_in_frame = 0
    tiny_isolated_total = 0
    for sequence, (by_tag, truth, meta, tables) in loaded.items():
        merged = merge_with_attrs(by_tag, tag_mode, fused_tag, scale_tags, active["merge_radius"])
        frames = build_frames_with_index(truth, merged)
        entries: list[dict] = []
        frame_rows: dict[int, list[int]] = {}
        for frame_index, centres, points, scores in frames:
            if points is None or scores is None:
                continue
            _tp_rows, fp_rows = extract_frame_matches(centres, points, scores, active["th"], tol, match_mode)
            if not fp_rows:
                continue
            detected_mask = scores >= active["th"]
            filtered_points = points[detected_mask]
            filtered_scores = scores[detected_mask]
            frame_table = tables.get(frame_index)
            for position in fp_rows:
                x = float(filtered_points[position][0])
                y = float(filtered_points[position][1])
                score = float(filtered_scores[position])
                if size_source == "fused":
                    edge = fused_edge_of(frame_table, merged[frame_index], position)
                    source_tag = fused_tag if edge is not None else None
                    attr_distance = 0.0
                else:
                    edge, source_tag, attr_distance = attribute_size(
                        frame_table, x, y, attr_radius, native_tag, coarse_tags
                    )
                entries.append(
                    {
                        "frame": frame_index,
                        "x": x,
                        "y": y,
                        "score": score,
                        "edge": edge,
                        "bucket": bucket_of(edge),
                        "source_tag": source_tag,
                        "attr_distance": attr_distance,
                    }
                )
                frame_rows.setdefault(frame_index, []).append(len(entries) - 1)

        if isolation_scope == "sequence" and entries:
            centres = np.asarray([(item["x"], item["y"]) for item in entries], dtype=np.float64)
            nearest = nearest_other_distance_chunked(centres)
            for item, value in zip(entries, nearest):
                item["isolated"] = bool(value > isolation_radius)
                item["nearest"] = None if not np.isfinite(value) else float(value)
                item["alone_in_frame"] = False
        else:
            for frame_index, indices in frame_rows.items():
                centres = np.asarray([(entries[i]["x"], entries[i]["y"]) for i in indices], dtype=np.float64)
                nearest = nearest_other_distance(centres)
                for local, index in enumerate(indices):
                    value = float(nearest[local])
                    entries[index]["isolated"] = bool(value > isolation_radius)
                    entries[index]["nearest"] = None if not np.isfinite(value) else value
                    entries[index]["alone_in_frame"] = len(indices) == 1

        counts = count_matches(
            [(centres, points, scores) for _index, centres, points, scores in frames], active["th"], tol, match_mode
        )
        row = metrics(counts, int(meta["frames"]))
        row["sequence"] = sequence
        row["frames"] = int(meta["frames"])
        row["fp_size_buckets"] = {name: 0 for name in (*SIZE_BUCKETS, "unknown")}
        row["isolated"] = 0
        row["clustered"] = 0
        row["isolated_tiny"] = 0
        row["size_attribution"] = {"native": 0, "coarse": 0, "fused": 0, "unknown": 0}
        for item in entries:
            row["fp_size_buckets"][item["bucket"]] += 1
            size_totals[item["bucket"]] += 1
            if item["isolated"]:
                row["isolated"] += 1
                isolation_totals["isolated"] += 1
            else:
                row["clustered"] += 1
                isolation_totals["clustered"] += 1
            if item["source_tag"] is None:
                row["size_attribution"]["unknown"] += 1
                attribution_totals["unknown"] += 1
                attribution_by_bucket[item["bucket"]]["unknown"] += 1
            elif item["source_tag"] == fused_tag:
                row["size_attribution"]["fused"] += 1
                attribution_totals["fused"] += 1
                attribution_by_bucket[item["bucket"]]["fused"] += 1
            elif item["source_tag"] == native_tag:
                row["size_attribution"]["native"] += 1
                attribution_totals["native"] += 1
                attribution_by_bucket[item["bucket"]]["native"] += 1
            else:
                row["size_attribution"]["coarse"] += 1
                attribution_totals["coarse"] += 1
                attribution_by_bucket[item["bucket"]]["coarse"] += 1
            if item["attr_distance"] is not None:
                attr_distances.append(float(item["attr_distance"]))
            if item["nearest"] is not None:
                nearest_distances.append(float(item["nearest"]))
            if item.get("alone_in_frame"):
                solo_in_frame += 1
            if item["edge"] is not None and item["edge"] <= TINY_MAX_EDGE and item["isolated"]:
                row["isolated_tiny"] += 1
                tiny_isolated_total += 1
        row["fp_extracted"] = len(entries)
        if len(entries) != counts["fp"]:
            raise RuntimeError(
                f"{sequence}: FP 归因出的个数 {len(entries)} 与官方口径的 FP {counts['fp']} 不一致，报告作废"
            )
        per_sequence.append(row)

    fp_total = int(active["fp"])
    ratio = tiny_isolated_total / fp_total if fp_total else 0.0
    distance_summary = {}
    if attr_distances:
        values = np.asarray(attr_distances, dtype=np.float64)
        distance_summary = {
            "count": int(len(values)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "max": float(values.max()),
        }
    nearest_summary = {}
    if nearest_distances:
        values = np.asarray(nearest_distances, dtype=np.float64)
        nearest_summary = {
            "count": int(len(values)),
            "min": float(values.min()),
            "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "max": float(values.max()),
        }
    return {
        "tool": "t2_isolated_tiny_fp",
        "config": {
            "cache_root": str(cache_root),
            "labels_root": str(labels_root),
            "sequences": list(sequences),
            "tag": tag_mode,
            "match_mode": match_mode,
            "tol_px": tol,
            "merge_radius": float(active["merge_radius"]),
            "sweep_merge_radii": [float(item) for item in radii],
            "th_min": th_min,
            "th_max": options["th_max"],
            "th_step": options["th_step"],
            "threshold": float(active["th"]),
            "threshold_source": "sweep_best" if options["th"] is None else "explicit",
            "size_source": size_source,
            "size_attr_radius": attr_radius,
            "isolation_radius": isolation_radius,
            "isolation_scope": isolation_scope,
            "native_tag": native_tag,
            "fused_tag": fused_tag,
        },
        "best": active,
        "sweep_best": best,
        "fp_size_buckets": size_totals,
        "fp_isolation": isolation_totals,
        "size_attribution": attribution_totals,
        "size_attribution_by_bucket": attribution_by_bucket,
        "size_attribution_distance": distance_summary,
        "fp_nearest_distance": nearest_summary,
        "fp_solo_in_frame": int(solo_in_frame),
        "main_metric": {
            "isolated_tiny_fp": int(tiny_isolated_total),
            "fp_total": fp_total,
            "ratio": ratio,
            "definition": "尺寸 ≤ 8px（nano+micro）且到同帧其它 FP 的最近距离 > 隔离半径",
        },
        "per_sequence": per_sequence,
        "sweep": sweep,
        "elapsed_seconds": time.perf_counter() - started,
    }


def fused_edge_of(frame_table: dict | None, payload: tuple, position: int) -> float | None:
    """融合行（tag=255）自身的 bbox 最大边：它被膨胀过，恒 ≥33px。

    ``payload`` 是 ``merge_scales_with_attrs`` 的结果，它的 tag 槽存着该候选在缓存帧内的**原始行号**，
    因此可以直接回查 ``sizes``。``position`` 是该候选在过阈值之后的序号，与 payload 的行序一致。
    """
    if frame_table is None:
        return None
    row_index = int(payload[4][position])
    sizes = frame_table["sizes"][row_index]
    return float(max(sizes[0], sizes[1]))


# --------------------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------------------
def print_report(result: dict, title: str = "") -> None:
    """打印 T2 总量 + FP 细分表。"""
    config = result["config"]
    best = result["best"]
    main = result["main_metric"]
    total_frames = sum(int(row["frames"]) for row in result["per_sequence"])
    if title:
        print(f"\n===== {title} =====", flush=True)
    print(
        f"[T2] 缓存={config['cache_root']} | GT={config['labels_root']} | 序列={len(config['sequences'])} 条 | "
        f"行={config['tag']} | 匹配={config['match_mode']} | merge_radius={config['merge_radius']:.1f}px | "
        f"阈值={config['threshold']:.2f}（{config['threshold_source']}）",
        flush=True,
    )
    print(
        f"{'序列':30s} {'帧':>7} {'GT':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'FN-abs':>7} {'FN-cmp':>7} "
        f"{'Recall':>8} {'Prec':>8} {'F1':>9} {'FAR/帧':>9}",
        flush=True,
    )
    for row in result["per_sequence"]:
        print(
            f"{row['sequence']:30s} {row['frames']:7d} {row['gt']:6d} {row['tp']:6d} {row['fp']:6d} {row['fn']:6d} "
            f"{row['fn_absent']:7d} {row['fn_compete']:7d} {row['recall'] * 100:7.2f}% {row['precision'] * 100:7.2f}% "
            f"{row['f1'] * 100:8.3f}% {row['far_per_frame']:9.4f}",
            flush=True,
        )
    print(
        f"{'合计':30s} {total_frames:7d} {best['gt']:6d} {best['tp']:6d} {best['fp']:6d} {best['fn']:6d} "
        f"{best['fn_absent']:7d} {best['fn_compete']:7d} {best['recall'] * 100:7.2f}% {best['precision'] * 100:7.2f}% "
        f"{best['f1'] * 100:8.3f}% {best['far_per_frame']:9.4f}",
        flush=True,
    )

    buckets = result["fp_size_buckets"]
    fp_total = max(1, int(best["fp"]))
    print(f"\n[FP 尺寸分档] 物理最大边（native 640x512 物理像素，来源={config['size_source']} 行）", flush=True)
    for name in (*SIZE_BUCKETS, "unknown"):
        low, high = SIZE_BUCKET_EDGES.get(name, (0.0, 0.0))
        if name == "unknown":
            span = "-"
        elif name == "nano":
            span = f"≤{high:g}px"
        elif name == "regular":
            span = f">{low:g}px"
        else:
            span = f"{low:g}~{high:g}px"
        print(f"  {name:8s} {span:>10s} {buckets[name]:8d}  {buckets[name] / fp_total * 100:6.2f}%", flush=True)

    isolation = result["fp_isolation"]
    print(f"\n[FP 孤立度分档] 最近距离 > {config['isolation_radius']:.0f}px 记为孤立（范围={config['isolation_scope']}）", flush=True)
    for name, label in (("isolated", "孤立"), ("clustered", "非孤立")):
        print(f"  {label:6s} {isolation[name]:8d}  {isolation[name] / fp_total * 100:6.2f}%", flush=True)
    nearest = result["fp_nearest_distance"]
    solo = result["fp_solo_in_frame"]
    if nearest:
        print(
            f"  FP 到最近其它 FP 的距离(native px) n={nearest['count']} min={nearest['min']:.1f} "
            f"p10={nearest['p10']:.1f} p50={nearest['p50']:.1f} p90={nearest['p90']:.1f} max={nearest['max']:.1f}",
            flush=True,
        )
    if solo:
        print(
            f"  [提示] 帧内只有它一个 FP 的 FP 有 {solo} 个（最近其它 FP 距离=∞，任何孤立半径下都算孤立）",
            flush=True,
        )
    if config["isolation_radius"] <= config["merge_radius"] and config["isolation_scope"] == "frame":
        print(
            f"  [提示] 孤立半径({config['isolation_radius']:.0f}px) ≤ 去重半径({config['merge_radius']:.0f}px)："
            "候选池里任意两点本来就 > 去重半径，孤立判定会被平凡满足（通常 100% 孤立）。\n"
            "         想看真实聚集结构，把 --isolation-radius 调到 16/24/32px，或改用 --isolation-scope sequence。",
            flush=True,
        )

    attribution = result["size_attribution"]
    print(
        f"\n[尺寸归因来源] native={attribution['native']}（原生尺度 tag=0）| coarse={attribution['coarse']}（粗尺度回退）"
        f" | fused={attribution['fused']}（融合行自身 bbox）| unknown={attribution['unknown']}（归因不到，单列不丢弃）",
        flush=True,
    )
    by_bucket = result["size_attribution_by_bucket"]
    distance_summary = result["size_attribution_distance"]
    print(
        "  分档 × 归因来源  "
        + "  ".join(
            f"{name}[native={by_bucket[name]['native']} coarse={by_bucket[name]['coarse']} "
            f"fused={by_bucket[name]['fused']} unk={by_bucket[name]['unknown']}]"
            for name in (*SIZE_BUCKETS, "unknown")
        ),
        flush=True,
    )
    if distance_summary:
        print(
            f"  归因距离(native px) n={distance_summary['count']} p50={distance_summary['p50']:.1f} "
            f"p90={distance_summary['p90']:.1f} max={distance_summary['max']:.1f}",
            flush=True,
        )
    print(
        f"\n[主指标] 孤立微小假峰 FP = {main['isolated_tiny_fp']} | 占全部 FP（{main['fp_total']}）的 "
        f"{main['ratio'] * 100:.2f}%",
        flush=True,
    )

    print(
        f"\n[按序列细分] {'序列':30s} {'FP':>6} {'nano':>6} {'micro':>6} {'small':>6} {'reg':>6} {'未知':>6} "
        f"{'孤立':>6} {'非孤立':>7} {'孤立微小':>9}",
        flush=True,
    )
    for row in result["per_sequence"]:
        buckets = row["fp_size_buckets"]
        print(
            f"{row['sequence']:30s} {row['fp']:6d} {buckets['nano']:6d} {buckets['micro']:6d} "
            f"{buckets['small']:6d} {buckets['regular']:6d} {buckets['unknown']:6d} {row['isolated']:6d} "
            f"{row['clustered']:7d} {row['isolated_tiny']:9d}",
            flush=True,
        )


def print_compare(results: list[tuple[str, dict]]) -> None:
    """并排比较多个缓存的 T2 指标与孤立微小假峰 FP。"""
    print(
        f"\n{'模型':26s} {'序列':>4} {'帧':>7} {'TP':>6} {'FP':>6} {'FN':>6} {'F1':>9} {'Prec':>8} "
        f"{'孤立微小FP':>10} {'占FP':>7} {'nano':>6} {'micro':>6} {'small':>6} {'reg':>6} {'未知':>6}",
        flush=True,
    )
    for name, result in results:
        best = result["best"]
        main = result["main_metric"]
        buckets = result["fp_size_buckets"]
        frames = sum(row["frames"] for row in result["per_sequence"])
        print(
            f"{name:26s} {len(result['config']['sequences']):4d} {frames:7d} {best['tp']:6d} {best['fp']:6d} "
            f"{best['fn']:6d} {best['f1'] * 100:8.3f}% {best['precision'] * 100:7.2f}% {main['isolated_tiny_fp']:10d} "
            f"{main['ratio'] * 100:6.2f}% {buckets['nano']:6d} {buckets['micro']:6d} {buckets['small']:6d} "
            f"{buckets['regular']:6d} {buckets['unknown']:6d}",
            flush=True,
        )


# --------------------------------------------------------------------------------------
# 子命令
# --------------------------------------------------------------------------------------
def resolve_sequences(cache_root: Path, requested: str, track: str) -> list[str]:
    """挑出本次要统计的序列。

    ``track`` 语义来自官方三轨门禁：``t2`` = 龙泉山黄金池（7 条），``t3`` = blind 难例（1 条），
    ``all`` = 缓存目录里的全部序列。``--sequences`` 显式指定时优先。
    """
    available = sorted(path.stem for path in cache_root.glob("*.pkl"))
    if not available:
        raise FileNotFoundError(f"缓存目录里没有 .pkl: {cache_root}")
    wanted = {item.strip() for item in requested.split(",") if item.strip()}
    if wanted:
        unknown = wanted - set(available)
        if unknown:
            raise KeyError(f"缓存目录里没有这些序列: {sorted(unknown)}")
        selected = [name for name in available if name in wanted]
    elif track == "t2":
        selected = [name for name in available if name in T2_GOLDEN_SEQUENCES]
    elif track == "t3":
        selected = [name for name in available if name in T3_BLIND_SEQUENCES]
    else:
        selected = available
    if not selected:
        raise KeyError(f"按 --track {track} / --sequences 过滤后没有序列可用（缓存里有 {len(available)} 条）")
    return selected


def analysis_options(args: argparse.Namespace) -> dict:
    """把命令行参数整理成 ``analyse`` 用的选项字典。"""
    radii = (
        sorted({float(item) for item in args.sweep_merge_radius.split(",") if item.strip()})
        if args.sweep_merge_radius
        else [float(args.merge_radius)]
    )
    return {
        "tag": args.tag,
        "match_mode": args.match_mode,
        "tol": args.tol,
        "th_min": args.th_min,
        "th_max": args.th_max,
        "th_step": args.th_step,
        "th": args.th,
        "radii": radii,
        "size_source": args.size_source,
        "size_attr_radius": args.size_attr_radius,
        "isolation_radius": args.isolation_radius,
        "isolation_scope": args.isolation_scope,
        "allow_non_nogmc": args.allow_non_nogmc,
    }


def cmd_build(args: argparse.Namespace) -> int:
    """封装缓存构建器，重建 no-GMC 龙泉山缓存。"""
    features_root = Path(args.features_root)
    assert_nogmc(str(features_root), args.allow_non_nogmc, "--features-root")
    if not features_root.is_dir():
        raise NotADirectoryError(features_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    script = PROJECT_ROOT / "manu" / "data" / "multiscale_heatmap_cache.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    command = [
        sys.executable,
        str(script),
        "--features-root",
        str(features_root),
        "--cache-root",
        str(out_dir),
        "--hm-weights",
        str(args.weights),
        "--scales",
        args.scales,
        "--region-threshold",
        str(args.region_threshold),
        "--main-threshold",
        str(args.main_threshold),
        "--fusion-dilate",
        str(args.fusion_dilate),
        "--min-area",
        str(args.min_area),
        "--device",
        str(args.device),
        "--chunk-size",
        str(args.chunk_size),
        "--batch-size",
        str(args.batch_size),
        "--cpu-threads",
        str(args.cpu_threads),
        "--overwrite",
    ]
    if not args.keep_p0:
        command.append("--ablate-p0")  # 板端参考模型：训练后丢掉 P0 残差分支
    if args.sequences:
        command += ["--sequences", args.sequences]
    if args.shard:
        command += ["--shard", args.shard]
    if args.limit_per_sequence:
        command += ["--limit-per-sequence", str(args.limit_per_sequence)]
    print(f"[BUILD] 特征根目录={features_root}（no-GMC）", flush=True)
    print("[BUILD] " + " ".join(command), flush=True)
    return subprocess.call(command, cwd=str(PROJECT_ROOT))


def cmd_report(args: argparse.Namespace) -> int:
    """产出单个缓存的 T2 FP 细分表。"""
    cache_root = Path(args.cache_root)
    labels_root = Path(args.labels_root)
    if not cache_root.is_dir():
        raise NotADirectoryError(cache_root)
    if not labels_root.is_dir():
        raise NotADirectoryError(labels_root)
    sequences = resolve_sequences(cache_root, args.sequences, args.track)
    options = analysis_options(args)
    result = analyse(cache_root, labels_root, sequences, options)
    print_report(result, title=f"T2 孤立微小假峰 FP 监控台 | {cache_root}")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[SUCCESS] JSON -> {out_path.resolve()}", flush=True)
    print(f"[SUCCESS] 用时 {result['elapsed_seconds']:.1f}s", flush=True)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """并排比较多个缓存。"""
    labels_root = Path(args.labels_root)
    if not labels_root.is_dir():
        raise NotADirectoryError(labels_root)
    models: list[tuple[str, Path]] = []
    for item in args.model:
        name, _, root = item.partition("=")
        if not root:
            raise SystemExit(f"--model 需要 name=cache_root 形式，收到: {item}")
        models.append((name, Path(root)))
    for index, root in enumerate(args.cache_root):
        path = Path(root)
        models.append((path.name or f"model{index}", path))
    if len(models) < 2:
        raise SystemExit("compare 至少需要两个缓存（--model name=cache_root 或多个 --cache-root）")
    options = analysis_options(args)
    results: list[tuple[str, dict]] = []
    for name, root in models:
        if not root.is_dir():
            raise NotADirectoryError(root)
        sequences = resolve_sequences(root, args.sequences, args.track)
        results.append((name, analyse(root, labels_root, sequences, options)))
    print_compare(results)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tool": "t2_isolated_tiny_fp",
            "command": "compare",
            "models": [{"name": name, "result": result} for name, result in results],
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[SUCCESS] JSON -> {out_path.resolve()}", flush=True)
    return 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    """给 report / compare 挂上共用的口径参数。"""
    parser.add_argument("--labels-root", default=LABELS_ROOT, help="Final_Labels 标注根目录")
    parser.add_argument("--sequences", default="", help="逗号分隔的序列名；空=按 --track 过滤")
    parser.add_argument("--track", default="t2", choices=["t2", "t3", "all"], help="t2=黄金池 / t3=blind / all=缓存里全部")
    parser.add_argument("--tag", default="255", help="候选行：255=融合行（权威）/ all=全部尺度 / 单个尺度号")
    parser.add_argument("--match-mode", default="inbox", choices=["dist", "inbox"], help="匹配口径，默认 inbox（T2 权威）")
    parser.add_argument("--tol", type=float, default=8.0, help="匹配容差（native px，仅 dist 模式生效）")
    parser.add_argument("--merge-radius", type=float, default=8.0, help="跨尺度去重半径（native px）")
    parser.add_argument("--sweep-merge-radius", default="", help="逗号分隔的半径列表，替代 --merge-radius")
    parser.add_argument("--th-min", type=float, default=0.0)
    parser.add_argument("--th-max", type=float, default=0.90)
    parser.add_argument("--th-step", type=float, default=0.01)
    parser.add_argument("--th", type=float, default=None, help="钉死阈值（默认用扫阈值的 F1 最优点）")
    parser.add_argument("--size-source", default="native", choices=["native", "fused"], help="FP 物理尺寸归因来源")
    parser.add_argument(
        "--size-attr-radius",
        type=float,
        default=32.0,
        help="尺寸归因半径（native px）；融合行被 33x33 核膨胀过，形心最多偏出约 16px",
    )
    parser.add_argument("--isolation-radius", type=float, default=8.0, help="孤立判定半径（native px）")
    parser.add_argument("--isolation-scope", default="frame", choices=["frame", "sequence"], help="孤立度统计范围")
    parser.add_argument("--allow-non-nogmc", action="store_true", help="越过 no-GMC 红线（仅限排查）")
    parser.add_argument("--out", default="", help="JSON 输出路径")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T2「孤立微小假峰 FP」监控台（龙泉山黄金池，no-GMC 口径）")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="为候选模型重建 no-GMC 龙泉山缓存")
    build.add_argument("--weights", required=True, help="热图检测器权重（.pt）")
    build.add_argument("--out-dir", required=True, help="缓存输出目录（产出 <out>/<sequence>.pkl）")
    build.add_argument("--features-root", default=NOGMC_FEATURES_ROOT, help="no-GMC 特征根目录")
    build.add_argument("--device", default="0", help="GPU 序号")
    build.add_argument("--sequences", default="", help="逗号分隔的序列名；空=全部")
    build.add_argument("--shard", default="", help="分片 i/n")
    build.add_argument("--scales", default="160,80", help="下采样尺度")
    build.add_argument("--region-threshold", type=float, default=0.06)
    build.add_argument("--main-threshold", type=float, default=0.22)
    build.add_argument("--fusion-dilate", type=int, default=32)
    build.add_argument("--min-area", type=int, default=4)
    build.add_argument("--chunk-size", type=int, default=64)
    build.add_argument("--batch-size", type=int, default=8)
    build.add_argument("--cpu-threads", type=int, default=1)
    build.add_argument("--limit-per-sequence", type=int, default=0)
    build.add_argument("--keep-p0", action="store_true", help="保留 P0 残差分支（默认按板端口径 --ablate-p0）")
    build.add_argument("--allow-non-nogmc", action="store_true", help="越过 no-GMC 红线（仅限排查）")
    build.set_defaults(func=cmd_build)

    report = subparsers.add_parser("report", help="单个缓存的 T2 FP 细分表")
    report.add_argument("--cache-root", required=True, help="缓存目录（<sequence>.pkl）")
    add_analysis_arguments(report)
    report.set_defaults(func=cmd_report)

    compare = subparsers.add_parser("compare", help="多个缓存并排比较")
    compare.add_argument("--cache-root", action="append", default=[], help="缓存目录，可重复")
    compare.add_argument("--model", action="append", default=[], help="name=cache_root，可重复")
    add_analysis_arguments(compare)
    compare.set_defaults(func=cmd_compare)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
