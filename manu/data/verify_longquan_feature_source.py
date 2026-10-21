#!/usr/bin/env python
"""判定既有龙泉山特征缓存的 GMC 语义（官方「逐 lag 直接拟合」 vs FPV「相邻帧仿射连乘」）。

背景
----
``runs/ms_cache_nop0`` 的 meta 记录 ``source="features"``，即 2026-10-20 冻结的三轨单帧
baseline 是在 ``longquanshan_ir_gmc_median`` 这份既有特征缓存上跑出来的。要让新构建的
微调数据集能与 baseline 逐位对账，新特征必须与这份缓存**同语义、同编码**。

本脚本对同��帧同时计算两套特征并与缓存实际像素比对：

* A = 官方语义：``H = compute_affine(im_h, im_curr)``（逐 lag 直接拟合）
* B = FPV 语义：``H = compose_affine([step1(k) for k in range(h_idx, curr)])``（仿射连乘）

判定口径：重新用 cv2.imwrite 编码后与缓存文件**逐字节**比较（最强证据），其次比较解码后
逐像素最大绝对差。

用法::

    python manu/data/verify_longquan_feature_source.py \\
        --raw-root "/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg" \\
        --features-root "/mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median/images/train" \\
        --sequence VIDEO00029_19700101_014053 \\
        --frames 100,300,500 --out-json runs/logs/lq_feature_source.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.data.build_full_median_dataset import FastGMCEstimator, natural_key  # noqa: E402


def compose_affine(h_list: list[np.ndarray]) -> np.ndarray:
    """Compose 2x3 affine matrices: H_total = H_n @ ... @ H_1（与 FPV 构建器同语义）。

    注意 FPV 原实现为 ``M = H3 @ M`` 的左乘累积，等价于 H_n @ ... @ H_1；
    这里保持完全一致的左乘顺序，**不可**改成右乘。
    """
    if not h_list:
        return np.eye(2, 3, dtype=np.float32)
    total = np.eye(3, dtype=np.float32)
    for h in h_list:
        h3 = np.eye(3, dtype=np.float32)
        h3[:2, :] = h
        total = h3 @ total
    return total[:2, :].astype(np.float32)


class FrameReader:
    """按需读取原始帧，保留一个有限窗口以避免长序列爆内存。"""

    def __init__(self, seq_dir: Path, keep: int = 96) -> None:
        self.frames = [f for f in seq_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        self.frames.sort(key=natural_key)
        self.keep = keep
        self.cache: dict[int, np.ndarray] = {}
        self.index_map: dict[int, int] = {}
        for position, frame in enumerate(self.frames):
            match = None
            digits = ""
            for ch in reversed(frame.stem):
                if ch.isdigit():
                    digits = ch + digits
                elif digits:
                    break
            match = int(digits) if digits else position
            self.index_map.setdefault(match, position)

    def get(self, list_idx: int) -> np.ndarray | None:
        if list_idx in self.cache:
            return self.cache[list_idx]
        if list_idx < 0 or list_idx >= len(self.frames):
            return None
        image = cv2.imread(str(self.frames[list_idx]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        self.cache[list_idx] = image
        if len(self.cache) > self.keep:
            for key in sorted(self.cache)[: len(self.cache) - self.keep]:
                del self.cache[key]
        return image

    def list_index_of(self, frame_idx: int) -> int:
        return self.index_map.get(frame_idx, min(frame_idx, len(self.frames) - 1))


def features_official(reader: FrameReader, curr_list_idx: int, window: int, stride_step: int, estimator) -> np.ndarray | None:
    """官方语义：逐 lag 直接拟合（build_full_median_dataset.py 一致）。"""
    im_curr = reader.get(curr_list_idx)
    if im_curr is None:
        return None
    idx_prev2 = max(0, curr_list_idx - 2)
    im_prev2 = reader.get(idx_prev2)
    if im_prev2 is None:
        return None
    h2 = estimator.compute_affine(im_prev2, im_curr)
    diff2 = cv2.absdiff(im_curr, estimator.warp(im_prev2, h2))

    history = []
    for step in range(1, window + 1):
        h_idx = max(0, curr_list_idx - step * stride_step)
        im_h = reader.get(h_idx)
        if im_h is None:
            continue
        h_h = estimator.compute_affine(im_h, im_curr)
        history.append(estimator.warp(im_h, h_h))

    if len(history) >= 3:
        stack = np.stack(history, axis=0)
        median_bg = np.median(stack, axis=0).astype(np.float32)
        res_median = np.clip(im_curr.astype(np.float32) - median_bg, 0, 255).astype(np.uint8)
    else:
        res_median = diff2
    return np.stack([im_curr, diff2, res_median], axis=-1)


def features_fpv(reader: FrameReader, curr_list_idx: int, window: int, stride_step: int, estimator) -> np.ndarray | None:
    """FPV 语义：相邻帧仿射连乘（build_fpv_ir_gmc_median.py 一致）。"""
    im_curr = reader.get(curr_list_idx)
    if im_curr is None:
        return None
    step1: dict[int, np.ndarray] = {}

    def get_step1(i: int) -> np.ndarray | None:
        if i not in step1:
            im_a = reader.get(i)
            im_b = reader.get(i + 1)
            if im_a is None or im_b is None:
                return None
            step1[i] = estimator.compute_affine(im_a, im_b)
        return step1[i]

    if curr_list_idx >= 2:
        h_a = get_step1(curr_list_idx - 2)
        h_b = get_step1(curr_list_idx - 1)
        if h_a is None or h_b is None:
            return None
        h_prev2 = compose_affine([h_a, h_b])
        im_prev2 = reader.get(curr_list_idx - 2)
        if im_prev2 is None:
            return None
        diff2 = cv2.absdiff(im_curr, estimator.warp(im_prev2, h_prev2))
    elif curr_list_idx == 1:
        im_prev1 = reader.get(0)
        if im_prev1 is None:
            return None
        h_prev1 = get_step1(0)
        diff2 = cv2.absdiff(im_curr, estimator.warp(im_prev1, h_prev1)) if h_prev1 is not None else im_curr
    else:
        diff2 = np.zeros_like(im_curr)

    history = []
    for step in range(1, window + 1):
        h_idx = curr_list_idx - step * stride_step
        if h_idx < 0:
            break
        im_h = reader.get(h_idx)
        if im_h is None:
            continue
        chain = []
        ok = True
        for k in range(h_idx, curr_list_idx):
            h_k = get_step1(k)
            if h_k is None:
                ok = False
                break
            chain.append(h_k)
        if not ok:
            continue
        history.append(estimator.warp(im_h, compose_affine(chain)))

    if len(history) >= 3:
        stack = np.stack(history, axis=0)
        median_bg = np.median(stack, axis=0).astype(np.float32)
        res_median = np.clip(im_curr.astype(np.float32) - median_bg, 0, 255).astype(np.uint8)
    else:
        res_median = diff2
    return np.stack([im_curr, diff2, res_median], axis=-1)


def encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def compare(label: str, candidate: np.ndarray, cached: np.ndarray, cached_bytes: bytes) -> dict:
    cand_bytes = encode(candidate)
    diff = np.abs(candidate.astype(np.int16) - cached.astype(np.int16))
    return {
        "arm": label,
        "byte_identical": cand_bytes == cached_bytes,
        "sha256_candidate": hashlib.sha256(cand_bytes).hexdigest()[:16],
        "sha256_cached": hashlib.sha256(cached_bytes).hexdigest()[:16],
        "decoded_max_abs_diff": int(diff.max()),
        "decoded_mean_abs_diff": round(float(diff.mean()), 6),
        "decoded_exact_ratio": round(float((diff == 0).mean()), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="判定既有龙泉山特征缓存的 GMC 语义")
    parser.add_argument("--raw-root", required=True, help="frames_ir_jpg 根目录")
    parser.add_argument("--features-root", required=True, help="既有特征缓存目录（扁平布局）")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frames", required=True, help="逗号分隔的帧号")
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--stride-step", type=int, default=2)
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    features_root = Path(args.features_root)
    seq_dir = raw_root / args.sequence
    if not seq_dir.is_dir():
        print(f"[FAIL] 原始序列目录不存在: {seq_dir}")
        return 2

    reader = FrameReader(seq_dir)
    estimator = FastGMCEstimator(downscale=args.downscale)
    frame_ids = [int(x) for x in args.frames.split(",") if x.strip()]

    samples = []
    for frame_id in frame_ids:
        cached_path = features_root / f"{args.sequence}__frame_{frame_id:06d}.jpg"
        if not cached_path.is_file():
            print(f"[SKIP] 缓存缺失: {cached_path.name}")
            continue
        cached_bytes = cached_path.read_bytes()
        cached = cv2.imdecode(np.frombuffer(cached_bytes, np.uint8), cv2.IMREAD_COLOR)
        if cached is None:
            print(f"[SKIP] 缓存解码失败: {cached_path.name}")
            continue

        curr_list_idx = reader.list_index_of(frame_id)
        official = features_official(reader, curr_list_idx, args.window, args.stride_step, estimator)
        fpv = features_fpv(reader, curr_list_idx, args.window, args.stride_step, estimator)
        if official is None or fpv is None:
            print(f"[SKIP] 特征计算失败: frame {frame_id}")
            continue

        entry = {
            "frame_index": frame_id,
            "cached_shape": list(cached.shape),
            "official": compare("official_direct_fit", official, cached, cached_bytes),
            "fpv": compare("fpv_chained_affine", fpv, cached, cached_bytes),
        }
        entry["semantic_gap_official_vs_fpv_max_abs_diff"] = int(
            np.abs(official.astype(np.int16) - fpv.astype(np.int16)).max()
        )
        samples.append(entry)
        print(
            f"[frame {frame_id:>6}] official byte_identical={entry['official']['byte_identical']} "
            f"maxdiff={entry['official']['decoded_max_abs_diff']:>3} | "
            f"fpv byte_identical={entry['fpv']['byte_identical']} "
            f"maxdiff={entry['fpv']['decoded_max_abs_diff']:>3} | "
            f"A-vs-B gap={entry['semantic_gap_official_vs_fpv_max_abs_diff']}"
        )

    if not samples:
        print("[FAIL] 无有效样本")
        return 2

    official_hits = sum(1 for s in samples if s["official"]["byte_identical"])
    fpv_hits = sum(1 for s in samples if s["fpv"]["byte_identical"])
    total = len(samples)
    # 边界帧（序列首帧）两条实现会退化到同一结果，允许它归属任意一方，
    # 因此判定门槛为「非边界帧中至少 90% 逐字节命中且显著优于另一方」。
    boundary = [s for s in samples if s["frame_index"] <= 1]
    core = [s for s in samples if s["frame_index"] > 1]
    core_total = len(core) or total
    official_core = sum(1 for s in core if s["official"]["byte_identical"])
    fpv_core = sum(1 for s in core if s["fpv"]["byte_identical"])

    if official_core == core_total and official_core > fpv_core:
        verdict = "OFFICIAL_DIRECT_FIT"
    elif fpv_core == core_total and fpv_core > official_core:
        verdict = "FPV_CHAINED_AFFINE"
    elif official_core >= 0.9 * core_total and official_core > fpv_core:
        verdict = "OFFICIAL_DIRECT_FIT"
    elif fpv_core >= 0.9 * core_total and fpv_core > official_core:
        verdict = "FPV_CHAINED_AFFINE"
    else:
        verdict = "INDETERMINATE"

    report = {
        "sequence": args.sequence,
        "raw_root": str(raw_root),
        "features_root": str(features_root),
        "window": args.window,
        "stride_step": args.stride_step,
        "downscale": args.downscale,
        "verdict": verdict,
        "official_byte_identical": official_hits,
        "fpv_byte_identical": fpv_hits,
        "core_samples": core_total,
        "official_core_byte_identical": official_core,
        "fpv_core_byte_identical": fpv_core,
        "samples": samples,
        "conclusion": {
            "OFFICIAL_DIRECT_FIT": "既有缓存由官方逐 lag 直接拟合语义产出，可直接复刻并与 baseline 逐位对账",
            "FPV_CHAINED_AFFINE": "既有缓存由 FPV 仿射连乘语义产出，与官方构建器不一致，需重建后再对账",
            "INDETERMINATE": "两套语义均未命中，需人工排查（可能是其它编码参数或已修复的构建器版本）",
        }[verdict],
    }

    print("\n" + "=" * 78)
    print(f"判定结果: {verdict}")
    print(f"  官方语义逐字节命中: {official_hits}/{total}（非边界帧 {official_core}/{core_total}）")
    print(f"  FPV  语义逐字节命中: {fpv_hits}/{total}（非边界帧 {fpv_core}/{core_total}）")
    print(f"  结论: {report['conclusion']}")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"  报告已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
