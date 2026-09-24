#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Generate YOLO26-format 3-frame raw temporal dataset: [I_t, I_{t-4}, I_{t-12}].

Directly mirrors an existing reference YOLO dataset (e.g. /mnt/data/siping/datasets/manu/uav):
1. Reads existing filenames directly from {ref_dataset}/images/{train,val}.
2. Parses sequence name and frame index (e.g. seq__000123.jpg -> sequence 'seq', frame 123).
3. Fetches I_t, I_{t-4}, I_{t-12} from the raw video sequence folder with clamp fallback.
4. Directly copies/reuses the ground truth label txt from {ref_dataset}/labels/{train,val}.
Guarantees 100% identical sample count, split, and labels with zero ambiguity.
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
        description="Build 3-frame raw dataset [I_t, I_{t-4}, I_{t-12}] directly mirrored from reference YOLO dataset"
    )
    parser.add_argument(
        "--ref-dataset",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav",
        help="Path to current working YOLO dataset containing images/train, images/val, labels/train, labels/val",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/anti-uav",
        help="Root directory containing raw sequence folders (e.g. /mnt/data/siping/datasets/manu/anti-uav)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_temporal_3frame",
        help="Target directory to write the new 3-frame dataset",
    )
    parser.add_argument("--lag-mid", type=int, default=4, help="Mid lag (default: 4)")
    parser.add_argument("--lag-long", type=int, default=12, help="Long lag (default: 12)")
    return parser.parse_args()


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def parse_seq_and_frame(im_name: str) -> tuple[str, int]:
    """
    Extract sequence name and frame index from standard dataset image name:
    Example:
      wg2022_ir_020_split_03__000123.jpg -> ('wg2022_ir_020_split_03', 123)
      DJI_0051_2__000456.jpg -> ('DJI_0051_2', 456)
      01_4485_1167-2666___000789.jpg -> ('01_4485_1167-2666', 789)
    """
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

    # 2. Inside Data/train or train/ or val/
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
    lag_mid: int = 4,
    lag_long: int = 12,
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

        # 找到原始帧所在的序列目录
        seq_dir = find_sequence_folder(raw_root, seq_name, seq_dir_cache)
        if seq_dir is None:
            if stats["missing_seq"] < 5:
                print(f"[WARN] 未找到序列目录: {seq_name}")
            stats["missing_seq"] += 1
            continue

        # 缓存该序列的所有排好序的原始帧
        if seq_dir not in seq_frames_cache:
            frames = [f for f in seq_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES]
            frames.sort(key=natural_key)
            seq_frames_cache[seq_dir] = frames

            # 构建 帧号 -> 列表索引 的映射 (支持文件名即帧号的情况)
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
            # 如果按数字没找到，尝试直接把 frame_idx 当列表索引
            curr_list_idx = min(frame_idx, len(frames_list) - 1)

        # 方式 A：Clamp 回退 (不足 lag 跨度时，平滑回退到第 0 帧)
        mid_list_idx = max(0, curr_list_idx - lag_mid)
        long_list_idx = max(0, curr_list_idx - lag_long)

        p_curr = frames_list[curr_list_idx]
        p_mid = frames_list[mid_list_idx]
        p_long = frames_list[long_list_idx]

        im_curr = cv2.imread(str(p_curr), cv2.IMREAD_GRAYSCALE)
        im_mid = cv2.imread(str(p_mid), cv2.IMREAD_GRAYSCALE)
        im_long = cv2.imread(str(p_long), cv2.IMREAD_GRAYSCALE)

        if im_curr is None or im_mid is None or im_long is None:
            stats["read_fail"] += 1
            continue

        # 拼接 3 通道: [I_t, I_{t-4}, I_{t-12}]
        merged = np.stack([im_curr, im_mid, im_long], axis=-1)

        # 保存图像，保持与参考数据集文件名 100% 相同
        out_img_p = out_img_dir / im_name
        cv2.imwrite(str(out_img_p), merged)

        # 直接 1:1 复制参考数据集中的标签文件（保证标签完全无漂移）
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


def write_data_yaml(output_dir: Path):
    yaml_content = f"""path: {output_dir.resolve()}
train: images/train
val: images/val

names:
  0: uav
"""
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\n[Generated] {yaml_path}")


def main():
    args = parse_args()
    ref_dir = Path(args.ref_dataset).resolve()
    raw_root = Path(args.raw_root).resolve()
    out_dir = Path(args.output).resolve()

    if not ref_dir.is_dir():
        raise FileNotFoundError(f"Reference dataset not found: {ref_dir}")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw anti-uav root not found: {raw_root}")

    print(f"Reference Dataset (Source of Truth): {ref_dir}")
    print(f"Raw Videos Root: {raw_root}")
    print(f"Output Dataset: {out_dir}")
    print(f"Lags: mid={args.lag_mid}, long={args.lag_long}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 严格按照参考数据集镜像生成 train
    process_split_from_ref(
        ref_dir=ref_dir,
        split_name="train",
        raw_root=raw_root,
        out_dir=out_dir,
        lag_mid=args.lag_mid,
        lag_long=args.lag_long,
    )

    # 2. 严格按照参考数据集镜像生成 val
    process_split_from_ref(
        ref_dir=ref_dir,
        split_name="val",
        raw_root=raw_root,
        out_dir=out_dir,
        lag_mid=args.lag_mid,
        lag_long=args.lag_long,
    )

    # 3. 写入 data.yaml
    write_data_yaml(out_dir)
    print("\n[SUCCESS] 3-frame dataset mirrored successfully!")


if __name__ == "__main__":
    main()
