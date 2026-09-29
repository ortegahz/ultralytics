#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build grouped 5-frame uint8 memmap caches for self-supervised motion training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def find_sequences(root: Path):
    candidates = []
    for directory in [root, *sorted((p for p in root.rglob("*") if p.is_dir()), key=natural_key)]:
        images = sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES), key=natural_key)
        if len(images) >= 5:
            candidates.append((directory.name, images))
    seen = set()
    return [(name, images) for name, images in candidates if not (str(images[0].parent) in seen or seen.add(str(images[0].parent)))]


def source_records(root: Path, source: str, stride: int):
    records = []
    for sequence, frames in find_sequences(root):
        for center in range(2, len(frames) - 2, stride):
            records.append({"frames": [str(p) for p in frames[center - 2 : center + 3]], "source": source, "sequence": sequence})
    return records


def select_records(records, quota, seed):
    if not records or quota <= 0:
        return []
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(records), size=min(quota, len(records)), replace=False)
    return [records[int(index)] for index in indices]


def split_by_sequence(records, val_fraction, seed):
    val = []
    train = []
    for record in records:
        digest = hashlib.blake2b(f"{seed}:{record['source']}:{record['sequence']}".encode(), digest_size=8).digest()
        (val if int.from_bytes(digest, "little") / 2**64 < val_fraction else train).append(record)
    if not train or not val:
        raise RuntimeError("Sequence split produced an empty train or val set; add more sequences.")
    return train, val


def write_cache(records, output: Path, height: int, width: int):
    output.mkdir(parents=True, exist_ok=True)
    bin_path = output / "frames.bin"
    array = np.memmap(bin_path, mode="w+", dtype=np.uint8, shape=(len(records), 5, height, width))
    manifest = []
    for index, record in enumerate(records):
        for offset, filename in enumerate(record["frames"]):
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Unable to read {filename}")
            if image.shape != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            array[index, offset] = image
        manifest.append({"source": record["source"], "sequence": record["sequence"], "frames": record["frames"]})
    array.flush()
    (output / "records.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in manifest) + "\n", encoding="utf-8")
    (output / "meta.json").write_text(json.dumps({"shape": [len(records), 5, height, width], "dtype": "uint8", "records": len(records)}, indent=2), encoding="utf-8")
    print(f"[DONE] {output}: {len(records)} samples, {bin_path.stat().st_size / 1024**3:.2f} GiB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anti-uav-root", required=True)
    parser.add_argument("--fpv-root", required=True)
    parser.add_argument("--longquanshan-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    if args.stride < 1 or args.max_samples < 0:
        raise ValueError("stride must be positive and max-samples must be non-negative")
    roots = [(Path(args.anti_uav_root), "anti-uav"), (Path(args.fpv_root), "fpv_data"), (Path(args.longquanshan_root), "longquanshan")]
    all_records = {}
    for root, source in roots:
        if not root.is_dir():
            raise NotADirectoryError(root)
        all_records[source] = source_records(root, source, args.stride)
        print(f"[SCAN] {source}: {len(all_records[source])} candidate windows")
    weights = {"anti-uav": 0.2, "fpv_data": 0.2, "longquanshan": 0.6}
    total = args.max_samples or sum(len(v) for v in all_records.values())
    records = []
    for source, source_weight in weights.items():
        records.extend(select_records(all_records[source], round(total * source_weight), args.seed + len(records)))
    if len(records) < 10:
        raise RuntimeError("Too few temporal windows")
    train, val = split_by_sequence(records, args.val_fraction, args.seed)
    output = Path(args.output_root)
    write_cache(train, output / "train_cache", args.height, args.width)
    write_cache(val, output / "val_cache", args.height, args.width)
    (output / "dataset_meta.json").write_text(json.dumps({"total": len(records), "train": len(train), "val": len(val), "stride": args.stride, "sources": weights}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
