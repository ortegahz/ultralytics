#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Probe black letterbox borders and content geometry of infrared video frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def border_run(mask: np.ndarray) -> tuple[int, int]:
    leading = 0
    for value in mask:
        if not value:
            break
        leading += 1
    trailing = 0
    for value in mask[::-1]:
        if not value:
            break
        trailing += 1
    return leading, trailing


def analyze_image(image_path: Path, black_thresh: int) -> dict:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {"image": str(image_path), "status": "read_failed"}
    height, width = gray.shape[:2]
    row_max = gray.max(axis=1)
    col_max = gray.max(axis=0)
    top, bottom = border_run(row_max <= black_thresh)
    left, right = border_run(col_max <= black_thresh)

    content = gray[top : height - bottom, left : width - right]
    content_h, content_w = content.shape[:2]
    scale_w = content_w / 640.0 if content_w else 0.0
    scale_h = content_h / 512.0 if content_h else 0.0
    scale_480 = content_h / 480.0 if content_h else 0.0

    inner_top = inner_bottom = inner_left = inner_right = 0
    if content_w > 0 and content_h > 0:
        inner_row_max = content.max(axis=1)
        inner_col_max = content.max(axis=0)
        inner_top, inner_bottom = border_run(inner_row_max <= black_thresh)
        inner_left, inner_right = border_run(inner_col_max <= black_thresh)

    return {
        "image": str(image_path.resolve()),
        "status": "ok",
        "size": [width, height],
        "border": {"top": top, "bottom": bottom, "left": left, "right": right},
        "content_size": [content_w, content_h],
        "content_aspect": round(content_w / content_h, 6) if content_h else None,
        "inner_border": {"top": inner_top, "bottom": inner_bottom, "left": inner_left, "right": inner_right},
        "estimated_scale_vs_640x512": [round(scale_w, 6), round(scale_h, 6)],
        "estimated_scale_vs_640x480": round(scale_480, 6),
        "border_pixel_max": int(max(row_max[:top].max() if top else 0, row_max[height - bottom :].max() if bottom else 0, col_max[:left].max() if left else 0, col_max[width - right :].max() if right else 0)),
        "content_stats": {"min": int(content.min()), "max": int(content.max()), "mean": round(float(content.mean()), 3)},
    }


def sample_paths(frames_dir: Path, sample: int) -> list[Path]:
    frames = sorted(frames_dir.glob("frame_*.jpg")) + sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        return []
    if len(frames) <= sample:
        return frames
    indices = np.linspace(0, len(frames) - 1, sample).round().astype(int)
    return [frames[int(index)] for index in sorted(set(indices.tolist()))]


def summarize(results: list[dict]) -> dict:
    valid = [item for item in results if item.get("status") == "ok"]
    if not valid:
        return {"frames": len(results), "valid": 0}
    borders = np.array([[item["border"][key] for key in ("top", "bottom", "left", "right")] for item in valid])
    content = np.array([item["content_size"] for item in valid])
    return {
        "frames": len(results),
        "valid": len(valid),
        "border_min": borders.min(axis=0).tolist(),
        "border_max": borders.max(axis=0).tolist(),
        "content_min": content.min(axis=0).tolist(),
        "content_max": content.max(axis=0).tolist(),
        "content_mode": content[0].tolist(),
        "stable": bool((borders.min(axis=0) == borders.max(axis=0)).all()),
    }


def main():
    parser = argparse.ArgumentParser(description="Probe letterbox borders of extracted frames")
    parser.add_argument("--image", type=str, default="", help="Single image to analyze")
    parser.add_argument("--frames-dir", type=str, default="", help="Directory containing extracted frames")
    parser.add_argument("--scan-root", type=str, default="", help="Root containing per-video frame directories")
    parser.add_argument("--sample", type=int, default=5, help="Frames sampled per directory")
    parser.add_argument("--black-thresh", type=int, default=16, help="Max pixel value considered black")
    parser.add_argument("--json-out", type=str, default="", help="Optional JSON report path")
    args = parser.parse_args()

    report: dict = {"black_thresh": args.black_thresh, "results": [], "summary": {}}
    if args.image:
        item = analyze_image(Path(args.image), args.black_thresh)
        report["results"].append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2))
    elif args.frames_dir:
        frames_dir = Path(args.frames_dir)
        results = [analyze_image(path, args.black_thresh) for path in sample_paths(frames_dir, args.sample)]
        report["results"] = results
        report["summary"] = summarize(results)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    elif args.scan_root:
        scan_root = Path(args.scan_root)
        for video_dir in sorted(path for path in scan_root.iterdir() if path.is_dir()):
            results = [analyze_image(path, args.black_thresh) for path in sample_paths(video_dir, args.sample)]
            summary = summarize(results)
            report["results"].append({"video": video_dir.name, "summary": summary, "samples": results})
            print(
                f"{video_dir.name:<34} size={results[0]['size'] if results else '-'} "
                f"content={summary.get('content_mode')} border[min]={summary.get('border_min')} "
                f"border[max]={summary.get('border_max')} stable={summary.get('stable')}"
            )
    else:
        raise SystemExit("Provide --image, --frames-dir or --scan-root")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Report written to {args.json_out}")


if __name__ == "__main__":
    main()
