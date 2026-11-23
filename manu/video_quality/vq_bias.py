#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Measure the scale bias that the canonical-640 normalisation introduces.

The two groups are recorded at different native resolutions, so every metric that depends on the
pixel scale (noise sigma, SNR, the gradient energy ratio rho and the Laplacian sharpness) carries a
systematic offset between the groups.  This script quantifies that offset per clip: it computes the
metric twice on the very same frames - once on the cropped frame at its native resolution and once
after the resize to the canonical analysis width - and reports the ratio.

    python manu/video_quality/vq_bias.py --clip "bad IR=/path/clip.mp4" --clip "good IR=/path/clip.h264" \
        --samples 120 --out manu/video_quality/report/resample_bias.txt

Interpretation: a ratio above 1 means the canonical width *raises* the metric.  When the bad group is
the one that gets resized, the reported group gap is therefore compressed (the bias is conservative
for "the bad clip is worse" claims) - it never hides a gap, it can only shrink one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vq_metrics import (  # noqa: E402
    CANON_WIDTH,
    canonical,
    detect_content_box,
    noise_sigma_immerkaer,
    open_frames,
    probe_video,
)

cv2.setNumThreads(1)

SCALE_METRICS = ("noise_sigma", "snr_db", "grad_energy_ratio", "lap_var")


def frame_metrics(frame: np.ndarray) -> dict[str, float]:
    """The four scale-sensitive metrics, computed exactly like ``vq_metrics.analyze_video`` does."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sigma = noise_sigma_immerkaer(gray)
    mean_val = float(gray.mean()) + 1e-6
    grad_x = float(cv2.absdiff(gray[:, 1:], gray[:, :-1]).sum())
    grad_y = float(cv2.absdiff(gray[1:, :], gray[:-1, :]).sum())
    return {
        "noise_sigma": float(sigma),
        "snr_db": float(20.0 * np.log10(mean_val / (sigma + 1e-6))),
        "grad_energy_ratio": (grad_x + grad_y) / (float(gray.sum()) + 1e-6),
        "lap_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def measure(path: Path, samples: int) -> tuple[tuple[int, int], dict[str, tuple[float, float, float]]]:
    """Return the crop size and, per metric, (native median, canonical-640 median, ratio)."""
    info = probe_video(path)
    width, height = info["width"], info["height"]
    crop = None
    native: dict[str, list[float]] = {name: [] for name in SCALE_METRICS}
    scaled: dict[str, list[float]] = {name: [] for name in SCALE_METRICS}
    seen = 0
    for frame in open_frames(path, width, height):
        if crop is None:
            crop = detect_content_box(frame)
        top, bottom, left, right = crop
        content = frame[top : height - bottom, left : width - right]
        if content.size == 0:
            content = frame
        for name, value in frame_metrics(content).items():
            native[name].append(value)
        for name, value in frame_metrics(canonical(content, CANON_WIDTH)).items():
            scaled[name].append(value)
        seen += 1
        if seen >= samples:
            break
    if crop is None:
        raise SystemExit(f"no frames decoded from {path}")
    top, bottom, left, right = crop
    out = {}
    for name in SCALE_METRICS:
        native_median = float(np.median(native[name]))
        scaled_median = float(np.median(scaled[name]))
        ratio = scaled_median / native_median if native_median else float("nan")
        out[name] = (native_median, scaled_median, ratio)
    return (height - bottom - top, width - right - left), out


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the canonical-640 scale bias per clip")
    parser.add_argument("--clip", action="append", default=[], help="LABEL=PATH (repeatable)")
    parser.add_argument("--samples", type=int, default=120, help="frames sampled per clip")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    lines = []
    for spec in args.clip:
        label, _, path = spec.partition("=")
        clip_path = Path(path)
        if not clip_path.exists():
            raise SystemExit(f"missing clip: {clip_path}")
        (crop_h, crop_w), values = measure(clip_path, args.samples)
        lines.append(f"--- {label}  {clip_path.name}  (crop -> ({crop_h}, {crop_w}))")
        for name in SCALE_METRICS:
            native_median, scaled_median, ratio = values[name]
            lines.append(
                f"    {name:<18s} native={native_median:>12.5f}  canonical640={scaled_median:>12.5f}  ratio={ratio:>7.4f}"
            )
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
