#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build anti-uav train and official validation memmap caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np
from tqdm import tqdm

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_key(path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", Path(path).name)]


def write_cache(records, output, height, width):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    shape = (len(records), 5, height, width)
    data = np.memmap(output / "frames.bin", mode="w+", dtype=np.uint8, shape=shape)
    with (output / "records.jsonl").open("w", encoding="utf-8") as file:
        for index, record in enumerate(tqdm(records, desc=f"Writing {output.name}", unit="sample")):
            for offset, filename in enumerate(record["frames"]):
                image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise RuntimeError(f"Unable to read {filename}")
                if image.ndim == 3:
                    image = image[:, :, 0]
                if image.shape != (height, width):
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                data[index, offset] = image
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    data.flush()
    (output / "meta.json").write_text(json.dumps({"shape": list(shape), "dtype": "uint8", "records": len(records)}, indent=2), encoding="utf-8")
    print(f"[CACHE] {output}: {len(records)} samples", flush=True)


def raw_train_records(root, stride, max_samples):
    records = []
    root = Path(root)
    sequence_dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=natural_key)
    for sequence_dir in tqdm(sequence_dirs, desc="Indexing anti-uav train", unit="sequence"): 
        frames = sorted((p for p in sequence_dir.iterdir() if p.suffix.lower() in SUFFIXES), key=natural_key)
        for center in range(2, len(frames) - 2, stride):
            records.append({"frames": [str(p) for p in frames[center - 2 : center + 3]], "sequence": sequence_dir.name, "source": "anti-uav-train"})
            if max_samples and len(records) >= max_samples:
                return records
    return records


def val_records(root, labels_root, stride, max_samples):
    root, labels_root = Path(root), Path(labels_root)
    grouped = {}
    for path in root.glob("*.jpg"):
        sequence = path.stem.split("__", 1)[0]
        grouped.setdefault(sequence, []).append(path)
    records = []
    for sequence, files in tqdm(sorted(grouped.items()), desc="Indexing official val", unit="sequence"): 
        files.sort(key=natural_key)
        for center in range(2, len(files) - 2, stride):
            center_path = files[center]
            center_image = cv2.imread(str(center_path), cv2.IMREAD_UNCHANGED)
            if center_image is None:
                continue
            native_height, native_width = center_image.shape[:2]
            label_path = labels_root / f"{center_path.stem}.txt"
            gt = []
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) >= 5:
                        gt.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
            records.append({"frames": [str(p) for p in files[center - 2 : center + 3]], "sequence": sequence, "source": "official-val", "gt": gt, "native_width": native_width, "native_height": native_height})
            if max_samples and len(records) >= max_samples:
                return records
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anti-uav-train", required=True)
    parser.add_argument("--official-val-images", required=True)
    parser.add_argument("--official-val-labels", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-stride", type=int, default=12)
    parser.add_argument("--val-stride", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    train = raw_train_records(args.anti_uav_train, args.train_stride, args.max_train_samples)
    val = val_records(args.official_val_images, args.official_val_labels, args.val_stride, args.max_val_samples)
    if not train or not val:
        raise RuntimeError(f"Empty cache: train={len(train)}, val={len(val)}")
    output = Path(args.output_root)
    write_cache(train, output / "train_cache", args.height, args.width)
    write_cache(val, output / "val_cache", args.height, args.width)
    (output / "dataset_meta.json").write_text(json.dumps({"train": len(train), "val": len(val), "train_stride": args.train_stride, "val_stride": args.val_stride}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
