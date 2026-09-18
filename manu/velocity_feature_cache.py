#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build compact GMC-aligned temporal residual caches for the official YOLO splits."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def natural_key(path: Path | str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", Path(path).stem)]


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Frame number missing: {path}")
    return int(match.group(1))


def estimate_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    height, width = target.shape
    source_small = cv2.resize(source, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    target_small = cv2.resize(target, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    points = cv2.goodFeaturesToTrack(source_small, maxCorners=600, qualityLevel=0.01, minDistance=4, blockSize=3)
    if points is None or len(points) < 6:
        return np.eye(2, 3, dtype=np.float32)
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(source_small, target_small, points, None, winSize=(15, 15), maxLevel=2)
    if tracked is None:
        return np.eye(2, 3, dtype=np.float32)
    valid = status.ravel() == 1
    if valid.sum() < 6:
        return np.eye(2, 3, dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(points[valid], tracked[valid], method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None:
        return np.eye(2, 3, dtype=np.float32)
    matrix = matrix.astype(np.float32)
    matrix[:, 2] *= 2.0
    return matrix


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    pad_width, pad_height = size - new_width, size - new_height
    left, right = pad_width // 2, pad_width - pad_width // 2
    top, bottom = pad_height // 2, pad_height - pad_height // 2
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)


def aligned_residual(history: np.ndarray, current: np.ndarray) -> np.ndarray:
    matrix = estimate_affine(history, current)
    warped = cv2.warpAffine(history, matrix, (current.shape[1], current.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return np.abs(current.astype(np.float32) - warped.astype(np.float32)) / 255.0


def build_sequence(sequence_dir: Path, sample_stems: set[str], output_dir: Path, window: int, size: int) -> int:
    frames = sorted((path for path in sequence_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)
    if not frames:
        return 0
    by_number = {frame_number(path): (index, path) for index, path in enumerate(frames)}
    saved = 0
    frame_cache: dict[int, np.ndarray] = {}
    targets = [(index, current_path) for index, current_path in enumerate(frames) if f"{sequence_dir.name}__{frame_number(current_path):06d}" in sample_stems]
    for target_index, (index, current_path) in enumerate(targets, 1):
        stem = f"{sequence_dir.name}__{frame_number(current_path):06d}"
        if target_index == 1 or target_index % 100 == 0 or target_index == len(targets):
            print(f"  {sequence_dir.name}: {target_index}/{len(targets)}", flush=True)
        destination = output_dir / f"{stem}.npy"
        if destination.exists():
            saved += 1
            continue
        current = cv2.imread(str(current_path), cv2.IMREAD_GRAYSCALE)
        if current is None:
            continue
        residuals = []
        for lag in range(1, window + 1):
            history_index = max(0, index - lag)
            if history_index not in frame_cache:
                frame_cache[history_index] = cv2.imread(str(frames[history_index]), cv2.IMREAD_GRAYSCALE)
            history = frame_cache[history_index]
            residual = aligned_residual(history, current) if history is not None else np.zeros_like(current, dtype=np.float32)
            residuals.append(letterbox(residual, size))
        np.save(destination, np.stack(residuals).astype(np.float16))
        saved += 1
        old_index = index - window - 8
        frame_cache.pop(old_index, None)
    return saved


def build_split(raw_root: Path, image_dir: Path, output_dir: Path, window: int, size: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    official = sorted((path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)
    groups: dict[str, set[str]] = {}
    for path in official:
        sequence, separator, _ = path.stem.rpartition("__")
        if not separator:
            raise ValueError(f"Invalid official image stem: {path.stem}")
        groups.setdefault(sequence, set()).add(path.stem)
    sequence_dirs = {path.name: path for path in raw_root.iterdir() if path.is_dir()}
    missing = sorted(set(groups) - set(sequence_dirs))
    if missing:
        raise FileNotFoundError(f"Missing raw sequences ({len(missing)}): {missing[:8]}")
    total = 0
    sequences = sorted(groups.items(), key=lambda item: natural_key(item[0]))
    print(f"[INFO] {image_dir.name}: {len(official)} official samples across {len(sequences)} sequences", flush=True)
    for sequence_index, (sequence, stems) in enumerate(sequences, 1):
        print(f"[{image_dir.name}] {sequence_index}/{len(sequences)} {sequence}: starting {len(stems)} frames", flush=True)
        count = build_sequence(sequence_dirs[sequence], stems, output_dir, window, size)
        total += count
        print(f"[{image_dir.name}] {sequence_index}/{len(sequences)} {sequence}: +{count}, total={total}/{len(official)}", flush=True)
    if total != len(official):
        raise RuntimeError(f"Cache count mismatch for {image_dir.name}: built={total}, official={len(official)}")
    print(f"[SUCCESS] {image_dir.name}: {total} samples, cache shape=[{window}, {size}, {size}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-train-root", required=True)
    parser.add_argument("--raw-val-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--cache-size", type=int, default=160)
    args = parser.parse_args()
    dataset = Path(args.dataset)
    output = Path(args.output)
    build_split(Path(args.raw_train_root), dataset / "images" / "train", output / "train", args.window, args.cache_size)
    build_split(Path(args.raw_val_root), dataset / "images" / "val", output / "val", args.window, args.cache_size)


if __name__ == "__main__":
    main()
