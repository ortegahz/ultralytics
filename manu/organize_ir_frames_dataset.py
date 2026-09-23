#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Organize extracted infrared frame sequences for GMC+Median preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Organize infrared frame folders into an Ultralytics split layout")
    parser.add_argument("--source-root", required=True, help="Directory containing one subdirectory per video sequence")
    parser.add_argument("--output-root", required=True, help="Output dataset root containing the selected split")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--copy", action="store_true", help="Copy images instead of creating symbolic links")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output sequence directories")
    return parser.parse_args()


def sequence_images(sequence_dir: Path) -> list[Path]:
    return sorted(
        (path for path in sequence_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name,
    )


def main():
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)

    split_root = output_root / args.split
    split_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "split": args.split,
        "mode": "copy" if args.copy else "symlink",
        "sequences": [],
    }

    sequence_dirs = sorted((path for path in source_root.iterdir() if path.is_dir()), key=lambda path: path.name)
    if not sequence_dirs:
        raise FileNotFoundError(f"No sequence directories found under {source_root}")

    for sequence_dir in sequence_dirs:
        images = sequence_images(sequence_dir)
        if not images:
            print(f"[WARN] Skipping empty sequence: {sequence_dir.name}")
            continue

        output_sequence = split_root / sequence_dir.name
        output_images = output_sequence / "ir" / "images"
        if output_sequence.exists() or output_sequence.is_symlink():
            if not args.overwrite:
                raise FileExistsError(f"Output sequence already exists: {output_sequence}")
            shutil.rmtree(output_sequence)
        output_images.mkdir(parents=True, exist_ok=True)

        for image_path in images:
            destination = output_images / image_path.name
            if args.copy:
                shutil.copy2(image_path, destination)
            else:
                destination.symlink_to(image_path)

        manifest["sequences"].append(
            {
                "name": sequence_dir.name,
                "source": str(sequence_dir),
                "frames": len(images),
                "first_frame": images[0].name,
                "last_frame": images[-1].name,
            }
        )
        print(f"[DONE] {sequence_dir.name}: {len(images)} frames")

    manifest["sequence_count"] = len(manifest["sequences"])
    manifest["frame_count"] = sum(item["frames"] for item in manifest["sequences"])
    (output_root / f"organize_{args.split}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] Organized {manifest['frame_count']} frames in {manifest['sequence_count']} sequences")
    print(f"[INFO] Dataset split root: {split_root}")


if __name__ == "__main__":
    main()
