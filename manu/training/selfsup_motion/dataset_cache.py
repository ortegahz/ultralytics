#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Memmap dataset for five-frame grayscale windows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MotionMemmapDataset(Dataset):
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        meta = json.loads((self.cache_dir / "meta.json").read_text(encoding="utf-8"))
        self.shape = tuple(meta["shape"])
        self.data = None

    def _open(self):
        if self.data is None:
            self.data = np.memmap(self.cache_dir / "frames.bin", mode="r", dtype=np.uint8, shape=self.shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        self._open()
        frames = np.asarray(self.data[index], dtype=np.float32) / 255.0
        return torch.from_numpy(frames.copy())


def worker_init_fn(_worker_id):
    torch.set_num_threads(1)
