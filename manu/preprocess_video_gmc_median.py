#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build Trial 0474 GMC and temporal-median features from extracted frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.build_full_median_dataset import FastGMCEstimator


def main():
    parser = argparse.ArgumentParser(description="Preprocess extracted video frames with GMC and median background")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--stride-step", type=int, default=2)
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.window < 1 or args.stride_step < 1:
        raise ValueError("window and stride-step must be positive")

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No frame_*.png files found in {frames_dir}")

    estimator = FastGMCEstimator(downscale=args.downscale)
    gray_frames = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in frame_paths]
    if any(frame is None for frame in gray_frames):
        raise RuntimeError("At least one extracted frame cannot be decoded")

    metadata_path = frames_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    progress = tqdm(total=len(gray_frames), desc="Building GMC+median features", unit="frame", dynamic_ncols=True)
    for index, current in enumerate(gray_frames):
        output_path = output_dir / f"feature_{index:06d}.npy"
        if output_path.exists() and not args.overwrite:
            progress.update(1)
            continue
        previous_index = max(0, index - args.stride_step)
        previous = gray_frames[previous_index]
        transform = estimator.compute_affine(previous, current)
        aligned_previous = estimator.warp(previous, transform)
        diff = cv2.absdiff(current, aligned_previous)

        history = []
        for step in range(1, args.window + 1):
            history_index = max(0, index - step * args.stride_step)
            history_frame = gray_frames[history_index]
            history_transform = estimator.compute_affine(history_frame, current)
            history.append(estimator.warp(history_frame, history_transform))
        if len(history) >= 5:
            background = np.median(np.stack(history, axis=0), axis=0).astype(np.float32)
            residual = np.clip(current.astype(np.float32) - background, 0, 255).astype(np.uint8)
        else:
            residual = diff
        feature = np.stack([residual, diff, current], axis=-1).astype(np.uint8)
        np.save(output_path, feature, allow_pickle=False)
        progress.update(1)
    progress.close()
    metadata.update({"feature_count": len(gray_frames), "window": args.window, "stride_step": args.stride_step, "downscale": args.downscale})
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SUCCESS] Features saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
