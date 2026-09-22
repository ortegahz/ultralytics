#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Extract native-resolution grayscale frames from every video under a root directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import cv2
from tqdm import tqdm

VIDEO_SUFFIXES = {".h264", ".264", ".mp4", ".avi", ".mov", ".mkv"}


def safe_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def extract_video(
    video_path: Path,
    output_dir: Path,
    overwrite: bool,
    image_format: str,
    jpg_quality: int,
    restore_native: bool,
    crop_left: int,
    crop_top: int,
    crop_width: int,
    crop_height: int,
    native_width: int,
    native_height: int,
) -> dict:
    video_output = output_dir / safe_name(video_path)
    video_output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {"video": str(video_path.resolve()), "status": "open_failed", "frame_count": 0}

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    declared_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    width = source_width
    height = source_height
    started = time.perf_counter()
    records = []
    index = 0
    progress = tqdm(total=declared_count or None, desc=video_path.name, unit="frame", dynamic_ncols=True)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if restore_native:
                frame_height, frame_width = frame.shape[:2]
                x2 = crop_left + crop_width
                y2 = crop_top + crop_height
                if crop_left < 0 or crop_top < 0 or x2 > frame_width or y2 > frame_height:
                    raise ValueError(f"Native crop {(crop_left, crop_top, x2, y2)} exceeds frame size {(frame_width, frame_height)}")
                frame = cv2.resize(frame[crop_top:y2, crop_left:x2], (native_width, native_height), interpolation=cv2.INTER_AREA)
            if width == 0 or height == 0 or restore_native:
                height, width = frame.shape[:2]
            frame_name = f"frame_{index:06d}.{image_format}"
            frame_path = video_output / frame_name
            if overwrite or not frame_path.exists():
                params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality] if image_format == "jpg" else []
                if not cv2.imwrite(str(frame_path), frame, params):
                    raise RuntimeError(f"Cannot write frame: {frame_path}")
            records.append(frame_name)
            index += 1
            progress.update(1)
            progress.set_postfix(fps=f"{index / max(time.perf_counter() - started, 1e-6):.1f}")
    finally:
        progress.close()
        capture.release()

    metadata = {
        "source_video": str(video_path.resolve()),
        "source_name": video_path.name,
        "source_size": [source_width, source_height],
        "restored_native": restore_native,
        "restore_crop": [crop_left, crop_top, crop_width, crop_height] if restore_native else None,
        "fps": fps,
        "declared_frame_count": declared_count,
        "decoded_frame_count": index,
        "width": width,
        "height": height,
        "pixel_format": "grayscale_uint8",
        "lossless": image_format == "png",
        "image_format": image_format,
        "jpg_quality": jpg_quality if image_format == "jpg" else None,
        "frame_pattern": f"frame_%06d.{image_format}",
        "frames": records,
        "decode_seconds": time.perf_counter() - started,
        "status": "ok",
    }
    (video_output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] {video_path.name}: {index} frames -> {video_output}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Batch-extract native grayscale frames from videos")
    parser.add_argument("--input-root", required=True, help="Root directory containing videos")
    parser.add_argument("--output-root", required=True, help="Root directory for extracted frames")
    parser.add_argument("--pattern", default="", help="Optional suffix or glob, for example .h264 or '*.h264'")
    parser.add_argument("--recursive", action="store_true", help="Scan nested directories")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--format", choices=["png", "jpg"], default="png")
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality from 0 to 100")
    parser.add_argument("--restore-native", action="store_true", help="Crop letterbox and restore frames to native 640x512")
    parser.add_argument("--crop-left", type=int, default=320)
    parser.add_argument("--crop-top", type=int, default=28)
    parser.add_argument("--crop-width", type=int, default=1280)
    parser.add_argument("--crop-height", type=int, default=1024)
    parser.add_argument("--native-width", type=int, default=640)
    parser.add_argument("--native-height", type=int, default=512)
    args = parser.parse_args()
    if not 1 <= args.jpg_quality <= 100:
        raise ValueError("--jpg-quality must be between 1 and 100")

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.pattern:
        if args.pattern.startswith("."):
            videos = [p for p in (input_root.rglob("*") if args.recursive else input_root.glob("*")) if p.suffix.lower() == args.pattern.lower()]
        else:
            videos = sorted(input_root.rglob(args.pattern) if args.recursive else input_root.glob(args.pattern))
    else:
        candidates = input_root.rglob("*") if args.recursive else input_root.glob("*")
        videos = [p for p in candidates if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
    videos = sorted(videos)
    if not videos:
        raise FileNotFoundError(f"No videos found under {input_root}")

    print(f"[INFO] Found {len(videos)} videos under {input_root.resolve()}", flush=True)
    results = []
    for number, video_path in enumerate(videos, 1):
        print(f"[START] {number}/{len(videos)} {video_path.name}", flush=True)
        try:
            results.append(
                extract_video(
                    video_path,
                    output_root,
                    args.overwrite,
                    args.format,
                    args.jpg_quality,
                    args.restore_native,
                    args.crop_left,
                    args.crop_top,
                    args.crop_width,
                    args.crop_height,
                    args.native_width,
                    args.native_height,
                )
            )
        except Exception as error:
            results.append({"source_video": str(video_path.resolve()), "source_name": video_path.name, "status": "failed", "error": str(error)})
            print(f"[FAIL] {video_path.name}: {error}", flush=True)

    manifest = {
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "video_count": len(videos),
        "completed_count": sum(item.get("status") == "ok" for item in results),
        "failed_count": sum(item.get("status") != "ok" for item in results),
        "videos": results,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SUMMARY] Total={len(videos)} Done={manifest['completed_count']} Failed={manifest['failed_count']}", flush=True)
    if manifest["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
