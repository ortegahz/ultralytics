#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Official YOLO dataset wrapper for aligned temporal velocity features."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset


def natural_key(path: Path | str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", Path(path).stem)]


class OfficialVelocityDataset(Dataset):
    def __init__(self, yolo_dataset, cache_dir: str | Path, feature_channels: int | None = None):
        self.yolo_dataset = yolo_dataset
        self.cache_dir = Path(cache_dir)
        self.feature_channels = feature_channels
        files = sorted(self.cache_dir.glob("*.npy"), key=natural_key)
        self.paths = {path.stem: path for path in files}
        missing = [Path(item["im_file"]).stem for item in yolo_dataset.labels if Path(item["im_file"]).stem not in self.paths]
        if missing:
            raise FileNotFoundError(f"Missing temporal cache entries: {missing[:5]} ({len(missing)} total)")
        self.collate_fn = collate_velocity_batch
        print(f"[OfficialVelocityDataset] samples={len(yolo_dataset)} cache={len(self.paths)} dir={self.cache_dir}")

    def __len__(self) -> int:
        return len(self.yolo_dataset)

    def __getitem__(self, index: int) -> dict:
        item = self.yolo_dataset[index]
        stem = Path(item["im_file"]).stem
        path = self.paths[stem]
        feature = np.load(path, mmap_mode="r")
        if feature.ndim != 3:
            raise ValueError(f"Expected [K, H, W] cache, got {feature.shape} at {path}")
        if self.feature_channels is not None:
            feature = feature[: self.feature_channels]
        velocity_seq = torch.from_numpy(np.asarray(feature, dtype=np.float32).copy())
        velocity_seq = torch.nn.functional.interpolate(velocity_seq.unsqueeze(1), size=(320, 320), mode="bilinear", align_corners=False).squeeze(1)
        return {"img": item["img"], "velocity_seq": velocity_seq, "bboxes": item.get("bboxes", torch.zeros((0, 4))), "batch_idx": item.get("batch_idx", torch.zeros((0,), dtype=torch.long)), "im_file": item["im_file"]}


def collate_velocity_batch(batch: list[dict]) -> dict:
    imgs = torch.stack([item["img"] for item in batch])
    sequences = torch.stack([item["velocity_seq"] for item in batch])
    boxes = []
    batch_idx = []
    for index, item in enumerate(batch):
        for box in item["bboxes"]:
            boxes.append(box)
            batch_idx.append(index)
    if boxes:
        bboxes = torch.stack(boxes)
        batch_indices = torch.tensor(batch_idx, dtype=torch.long)
    else:
        bboxes = torch.zeros((0, 4), dtype=torch.float32)
        batch_indices = torch.zeros((0,), dtype=torch.long)
    return {"img": imgs, "velocity_seq": sequences, "bboxes": bboxes, "batch_idx": batch_indices}
