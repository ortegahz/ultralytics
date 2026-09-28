#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build an unlabeled temporal triplet index from infrared JPG/PNG frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def collect_sequences(root: Path):
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    nested = {}
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), key=natural_key):
        images = sorted((path for path in directory.iterdir() if path.suffix.lower() in image_suffixes), key=natural_key)
        if len(images) >= 3:
            nested[directory.name] = images
    if nested:
        return nested
    flat = {}
    for path in sorted((item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in image_suffixes), key=natural_key):
        stem = path.stem
        sequence = stem.split("__", 1)[0] if "__" in stem else root.name
        flat.setdefault(sequence, []).append(path)
    return {key: sorted(value, key=natural_key) for key, value in flat.items() if len(value) >= 3}


def main():
    parser = argparse.ArgumentParser(description="Build unlabeled temporal triplets")
    parser.add_argument("--root", action="append", required=True, help="Raw frame root; repeat for multiple datasets")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-per-sequence", type=int, default=0)
    args = parser.parse_args()
    if args.lag < 1 or args.stride < 1:
        raise ValueError("lag and stride must be positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    sequence_count = 0
    for root_value in args.root:
        root = Path(root_value).expanduser()
        if not root.is_dir():
            print(f"[WARN] Missing root: {root}")
            continue
        sequences = collect_sequences(root)
        for sequence, frames in sequences.items():
            sequence_count += 1
            limit = len(frames) - 2 * args.lag * args.stride
            indices = range(args.lag * args.stride, max(args.lag * args.stride, len(frames) - args.lag * args.stride), args.stride)
            count = 0
            for center in indices:
                if center - args.lag * args.stride < 0 or center + args.lag * args.stride >= len(frames):
                    continue
                records.append({"prev": str(frames[center - args.lag * args.stride].resolve()), "center": str(frames[center].resolve()), "next": str(frames[center + args.lag * args.stride].resolve()), "sequence": sequence})
                count += 1
                if args.max_per_sequence and count >= args.max_per_sequence:
                    break
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {"roots": args.root, "records": len(records), "sequences": sequence_count, "lag": args.lag, "stride": args.stride}
    output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SUCCESS] {len(records)} triplets from {sequence_count} sequences -> {output}")


if __name__ == "__main__":
    main()
