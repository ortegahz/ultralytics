#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Compose aligned before-and-after diagnostic videos into a labeled comparison video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose a before/after diagnostic comparison video")
    parser.add_argument("--before", required=True, help="Baseline diagnostic MP4")
    parser.add_argument("--after", required=True, help="Candidate/improved diagnostic MP4")
    parser.add_argument("--output", required=True, help="Output side-by-side MP4")
    parser.add_argument("--before-title", default="优化前（基线）")
    parser.add_argument("--after-title", default="优化后（候选方案）")
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--fps", type=float, default=0.0, help="Output FPS; 0 uses the before-video FPS")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--resize-height", type=int, default=640)
    return parser.parse_args()


def open_capture(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return capture


def fit_frame(frame: np.ndarray, height: int) -> np.ndarray:
    if frame is None:
        raise RuntimeError("Video returned an empty frame")
    scale = height / frame.shape[0]
    width = max(1, round(frame.shape[1] * scale))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def pad_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    if frame.shape[1] == width:
        return frame
    canvas = np.zeros((frame.shape[0], width, 3), dtype=np.uint8)
    offset = (width - frame.shape[1]) // 2
    canvas[:, offset : offset + frame.shape[1]] = frame
    return canvas


def add_header(frame: np.ndarray, title: str, color: tuple[int, int, int], subtitle: str) -> None:
    header_height = 54
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], header_height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
    cv2.rectangle(frame, (0, 0), (7, header_height), color, -1)
    cv2.putText(frame, title, (18, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(frame, subtitle, (18, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (220, 220, 220), 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    before_path = Path(args.before).expanduser()
    after_path = Path(args.after).expanduser()
    output_path = Path(args.output).expanduser()
    if not before_path.exists():
        raise FileNotFoundError(before_path)
    if not after_path.exists():
        raise FileNotFoundError(after_path)

    before = open_capture(before_path)
    after = open_capture(after_path)
    before_fps = before.get(cv2.CAP_PROP_FPS)
    output_fps = args.fps if args.fps > 0 else before_fps
    if output_fps <= 0:
        output_fps = 25.0
    before_count = int(before.get(cv2.CAP_PROP_FRAME_COUNT))
    after_count = int(after.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_limit = min(before_count, after_count) if before_count > 0 and after_count > 0 else 10**18
    if args.max_frames > 0:
        frame_limit = min(frame_limit, args.max_frames)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    processed = 0
    try:
        while processed < frame_limit:
            ok_before, before_frame = before.read()
            ok_after, after_frame = after.read()
            if not ok_before or not ok_after:
                break
            before_frame = fit_frame(before_frame, args.resize_height)
            after_frame = fit_frame(after_frame, args.resize_height)
            panel_width = max(before_frame.shape[1], after_frame.shape[1])
            before_frame = pad_to_width(before_frame, panel_width)
            after_frame = pad_to_width(after_frame, panel_width)
            add_header(before_frame, args.before_title, (0, 90, 255), args.subtitle)
            add_header(after_frame, args.after_title, (0, 210, 80), args.subtitle)
            divider = np.full((args.resize_height, 4, 3), (230, 230, 230), dtype=np.uint8)
            canvas = np.hstack((before_frame, divider, after_frame))
            cv2.putText(
                canvas,
                f"FRAME {processed:06d}",
                (canvas.shape[1] - 180, canvas.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    output_fps,
                    (canvas.shape[1], canvas.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open output video: {output_path}")
            writer.write(canvas)
            processed += 1
    finally:
        before.release()
        after.release()
        if writer is not None:
            writer.release()
    if processed == 0:
        raise RuntimeError("No aligned frames were written")
    print(f"[SUCCESS] Saved: {output_path.resolve()} | frames={processed} | fps={output_fps:.2f}")


if __name__ == "__main__":
    main()
