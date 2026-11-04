#!/usr/bin/env python3
"""把 sshfs 上的灰度帧一次性解码成本地 memmap 缓存（uint8，按绝对帧号索引）。

为什么需要：龙泉山帧目录挂在 **sshfs（网络文件系统）** 上，实测读+解码
**11.3 ms/帧**，而同样的文件放本地盘只要 **2.82 ms/帧**。评测里每条臂、每组参数
都要把 29000 帧过一遍，重复走网络光 I/O 就要 4 分钟/趟；本地 memmap 后
单帧访问降到 ~0.1 ms，整个参数扫描快一个量级。

产物：``<out>/<SEQ>.npy``（形状 ``(N, H, W)`` uint8，索引 = 绝对帧号）
与 ``<out>/<SEQ>.json``（记录帧号列表与尺寸，供校验）。

用法::

    python manu/tracking/lookahead/build_frame_cache.py \
        --frames-root /home/manu/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg \
        --out runs/lookahead/frame_cache --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

cv2.setNumThreads(1)


def _decode_chunk(task):
    """解码一段帧，返回 (起始帧号, uint8 数组)。"""
    directory, start, indices, height, width = task
    out = np.zeros((len(indices), height, width), dtype=np.uint8)
    for row, index in enumerate(indices):
        path = Path(directory) / f"frame_{index:06d}.jpg"
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        out[row] = image
    return start, out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建本地灰度帧 memmap 缓存")
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seqs", default="")
    parser.add_argument("--all-seqs", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    frames_root = Path(args.frames_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.seqs:
        sequences = [name.strip() for name in args.seqs.split(",") if name.strip()]
    elif args.all_seqs:
        sequences = sorted(
            path.name for path in frames_root.iterdir() if path.is_dir()
        )
    else:
        sequences = sorted(
            path.name for path in frames_root.iterdir()
            if path.is_dir() and (path / "metadata.json").exists()
        )

    for sequence in sequences:
        directory = frames_root / sequence
        indices = sorted(
            int(path.stem.split("_")[-1])
            for path in directory.glob("frame_*.jpg")
        )
        if not indices:
            print(f"[SKIP] {sequence} 没有帧", flush=True)
            continue
        npy_path = out_root / f"{sequence}.npy"
        meta_path = out_root / f"{sequence}.json"
        if npy_path.exists() and meta_path.exists() and not args.overwrite:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("count") == len(indices):
                print(f"[SKIP] {sequence} 已缓存 ({len(indices)} 帧)", flush=True)
                continue
        probe = cv2.imread(
            str(directory / f"frame_{indices[0]:06d}.jpg"), cv2.IMREAD_GRAYSCALE
        )
        if probe is None:
            raise RuntimeError(f"无法解码 {sequence}")
        height, width = probe.shape
        array = np.lib.format.open_memmap(
            npy_path, mode="w+", dtype=np.uint8, shape=(len(indices), height, width)
        )
        tasks = []
        for start in range(0, len(indices), args.chunk):
            tasks.append(
                (str(directory), start, indices[start:start + args.chunk], height, width)
            )
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for start, block in pool.map(_decode_chunk, tasks):
                array[start:start + block.shape[0]] = block
        array.flush()
        del array
        meta_path.write_text(
            json.dumps(
                {
                    "sequence": sequence,
                    "count": len(indices),
                    "height": height,
                    "width": width,
                    "first_frame": indices[0],
                    "last_frame": indices[-1],
                    "contiguous": indices == list(range(indices[0], indices[-1] + 1)),
                    "source": str(directory),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[OK] {sequence}: {len(indices)} 帧 {width}x{height} "
            f"-> {npy_path} ({npy_path.stat().st_size/1e9:.2f} GB)",
            flush=True,
        )
    print(f"[SUCCESS] elapsed {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
