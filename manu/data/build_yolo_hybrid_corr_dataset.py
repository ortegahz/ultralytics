#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Generate YOLO26-format hybrid correlation dataset:
    [I_t, |I_t - I_{t-lag_diff}|, I_{t-lag_corr}]
Default lags:
    lag_diff = 2  (short-term transient motion, preserves 0.8898 F1 baseline performance)
    lag_corr = 8  (mid/long temporal reference frame for feature-level local correlation)

Directly mirrors reference YOLO dataset (e.g. /mnt/data/siping/datasets/manu/uav):
1. Reads existing filenames directly from {ref_dataset}/images/{train,val}.
2. Fetches I_t, I_{t-2}, I_{t-8} from the raw sequence folder with clamp fallback.
3. Channel 0: I_t (Current gray)
   Channel 1: |I_t - I_{t-2}| (Short-term absolute frame difference)
   Channel 2: I_{t-8} (Long-term reference gray frame)
4. Copies labels directly from {ref_dataset}/labels/{train,val}.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import sys

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build hybrid correlation dataset [I_t, |I_t - I_{t-2}|, I_{t-8}]"
    )
    parser.add_argument(
        "--ref-dataset",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav",
        help="Path to current reference YOLO dataset",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/anti-uav",
        help="Root directory containing raw sequence folders",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_hybrid_corr",
        help="Target directory to write the new hybrid dataset",
    )
    parser.add_argument("--lag-diff", type=int, default=2, help="Short difference lag (default: 2)")
    parser.add_argument("--lag-corr", type=int, default=8, help="Correlation reference lag (default: 8)")
    return parser.parse_args()


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def parse_seq_and_frame(im_name: str) -> tuple[str, int]:
    stem = Path(im_name).stem
    if "___" in stem:
        parts = stem.split("___")
        seq = parts[0]
        match = re.search(r"(\d+)$", parts[1])
        frame_idx = int(match.group(1)) if match else 0
        return seq, frame_idx

    if "__" in stem:
        parts = stem.split("__")
        seq = parts[0]
        match = re.search(r"(\d+)$", parts[1])
        frame_idx = int(match.group(1)) if match else 0
        return seq, frame_idx

    match = re.search(r"^(.*?)(?:[_-]+)?(\d+)$", stem)
    if match:
        seq = match.group(1).rstrip("_-")
        frame_idx = int(match.group(2))
        return seq, frame_idx

    raise ValueError(f"Cannot parse sequence name and frame number from image name: {im_name}")


def find_sequence_folder(raw_root: Path, seq_name: str, cache: dict[str, Path]) -> Path | None:
    if seq_name in cache:
        return cache[seq_name]

    # 1. Direct child
    cand = raw_root / seq_name
    if cand.is_dir():
        cache[seq_name] = cand
        return cand

    # 2. Inside Data/train, train, or val
    for sub in [raw_root / "Data" / "train" / seq_name, raw_root / "train" / seq_name, raw_root / "val" / seq_name]:
        if sub.is_dir():
            cache[seq_name] = sub
            return sub

    # 3. Recursive search
    for p in raw_root.rglob(seq_name):
        if p.is_dir():
            cache[seq_name] = p
            return p

    return None


def process_split_from_ref(
    ref_dir: Path,
    split_name: str,
    raw_root: Path,
    out_dir: Path,
    lag_diff: int = 2,
    lag_corr: int = 8,
):
    print(f"\n==================== Processing [{split_name}] ====================")
    ref_img_dir = ref_dir / "images" / split_name
    ref_lbl_dir = ref_dir / "labels" / split_name

    if not ref_img_dir.is_dir():
        raise FileNotFoundError(f"Reference image directory not found: {ref_img_dir}")

    out_img_dir = out_dir / "images" / split_name
    out_lbl_dir = out_dir / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    ref_images = [p for p in ref_img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    ref_images.sort(key=natural_key)
    total_imgs = len(ref_images)
    print(f"Found {total_imgs} images in reference split [{split_name}].")

    seq_dir_cache: dict[str, Path] = {}
    seq_frames_cache: dict[Path, list[Path]] = {}
    seq_name_to_idx: dict[Path, dict[int, int]] = {}

    stats = {"total": total_imgs, "success": 0, "missing_seq": 0, "read_fail": 0}

    for i, ref_img_p in enumerate(ref_images):
        im_name = ref_img_p.name
        seq_name, frame_idx = parse_seq_and_frame(im_name)

        seq_dir = find_sequence_folder(raw_root, seq_name, seq_dir_cache)
        if seq_dir is None:
            if stats["missing_seq"] < 5:
                print(f"[WARN] 未找到序列目录: {seq_name}")
            stats["missing_seq"] += 1
            continue

        if seq_dir not in seq_frames_cache:
            frames = [f for f in seq_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES]
            frames.sort(key=natural_key)
            seq_frames_cache[seq_dir] = frames

            idx_map = {}
            for list_i, f in enumerate(frames):
                match = re.search(r"(\d+)$", f.stem)
                if match:
                    idx_map[int(match.group(1))] = list_i
                else:
                    idx_map[list_i] = list_i
            seq_name_to_idx[seq_dir] = idx_map

        frames_list = seq_frames_cache[seq_dir]
        curr_list_idx = seq_name_to_idx[seq_dir].get(frame_idx)
        if curr_list_idx is None:
            curr_list_idx = min(frame_idx, len(frames_list) - 1)

        # Clamp 回退机制
        diff_list_idx = max(0, curr_list_idx - lag_diff)
        corr_list_idx = max(0, curr_list_idx - lag_corr)

        p_curr = frames_list[curr_list_idx]
        p_diff = frames_list[diff_list_idx]
        p_corr = frames_list[corr_list_idx]

        im_curr = cv2.imread(str(p_curr), cv2.IMREAD_GRAYSCALE)
        im_diff = cv2.imread(str(p_diff), cv2.IMREAD_GRAYSCALE)
        im_corr = cv2.imread(str(p_corr), cv2.IMREAD_GRAYSCALE)

        if im_curr is None or im_diff is None or im_corr is None:
            stats["read_fail"] += 1
            continue

        # 核心：计算短时绝对差分
        diff_transient = cv2.absdiff(im_curr, im_diff)

        # 拼接 3 通道: [I_t, |I_t - I_{t-2}|, I_{t-8}]
        merged = np.stack([im_curr, diff_transient, im_corr], axis=-1)

        out_img_p = out_img_dir / im_name
        cv2.imwrite(str(out_img_p), merged)

        ref_lbl_p = ref_lbl_dir / f"{ref_img_p.stem}.txt"
        out_lbl_p = out_lbl_dir / f"{ref_img_p.stem}.txt"
        if ref_lbl_p.exists():
            shutil.copy(ref_lbl_p, out_lbl_p)
        else:
            out_lbl_p.write_text("", encoding="utf-8")

        stats["success"] += 1

        if (i + 1) % 3000 == 0 or (i + 1) == total_imgs:
            print(f"[{split_name}] Processed {i + 1}/{total_imgs} images...")

    print(f"[{split_name}] Complete: {stats}")


def write_data_yaml(out_dir: Path):
    yaml_content = f"""# Ultralytics UAV Dataset: Hybrid Correlation Mode [I_t, |I_t - I_{{t-2}}|, I_{{t-8}}]
path: {out_dir.resolve()}
train: images/train
val: images/val

names:
  0: uav
"""
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"[INFO] 成功写入 dataset 配置: {yaml_path}")


def main():
    args = parse_args()
    ref_dir = Path(args.ref_dataset)
    raw_root = Path(args.raw_root)
    out_dir = Path(args.output)

    # 路径 fallback 检查
    if not ref_dir.exists():
        candidates = [
            Path("/mnt/data/siping/datasets/manu/uav"),
            Path("/home/manu/mnt/datasets/manu/uav"),
            Path("/media/manu/1TB-Volume/data/uav"),
        ]
        for c in candidates:
            if c.exists():
                ref_dir = c
                break

    if not raw_root.exists():
        candidates = [
            Path("/mnt/data/siping/datasets/manu/anti-uav"),
            Path("/home/manu/mnt/datasets/manu/anti-uav"),
            Path("/media/manu/1TB-Volume/data/anti-uav"),
        ]
        for c in candidates:
            if c.exists():
                raw_root = c
                break

    print(f"[INFO] Reference dataset : {ref_dir}")
    print(f"[INFO] Raw videos root   : {raw_root}")
    print(f"[INFO] Output dataset dir: {out_dir}")
    print(f"[INFO] Lag settings      : diff_lag={args.lag_diff}, corr_lag={args.lag_corr}")

    process_split_from_ref(ref_dir, "train", raw_root, out_dir, args.lag_diff, args.lag_corr)
    process_split_from_ref(ref_dir, "val", raw_root, out_dir, args.lag_diff, args.lag_corr)
    write_data_yaml(out_dir)
    print("\n[SUCCESS] 全部数据集生成完成！")


if __name__ == "__main__":
    main()
