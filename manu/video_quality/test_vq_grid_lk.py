#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Synthetic self-test for the LK flow maths of ``vq_grid_lk.py``.

"Still trackable" has to be a property of the picture, not of the implementation, so this
test needs no video at all: it builds a textured grey frame, shifts it by a *known* integer
translation and checks that the LK grid test recovers exactly that translation and closes
the forward/backward loop.

Two controls are asserted:
  * positive - a frame and its translated copy: the median cell displacement must equal the
    known shift, the forward-backward error must vanish and most cells must be trackable;
  * negative - two *independent* noise frames (no true correspondence anywhere): not a
    single cell may be reported as trackable, otherwise "trackable" would mean nothing.

Run it with ``python manu/video_quality/test_vq_grid_lk.py`` (exit code 0 on success) or
through ``python manu/video_quality/vq_grid_lk.py --selftest``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vq_grid_lk import lk_pair  # noqa: E402

FRAME_HEIGHT = 360
FRAME_WIDTH = 640
SHIFT_X = 7
SHIFT_Y = -4
FRAME_SEED = 11
UNRELATED_SEED = 12
UNRELATED_SEEDS = 10


def synthetic_frame(seed: int, height: int = FRAME_HEIGHT, width: int = FRAME_WIDTH) -> np.ndarray:
    """Grey frame with enough high-frequency detail for Lucas-Kanade to lock on."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
    blurred = cv2.GaussianBlur(noise, (5, 5), 0.0)
    spread = float(np.ptp(blurred))
    if spread <= 0.0:
        raise RuntimeError("degenerate synthetic frame")
    return ((blurred - float(blurred.min())) / spread * 255.0).astype(np.uint8)


def shifted_frame(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Copy of ``frame`` whose content is translated by the integer vector ``(dx, dy)``."""
    height, width = frame.shape
    matrix = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
    return cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def measure() -> dict:
    """Run the LK grid test on the translated pair and on the unrelated pair."""
    first = synthetic_frame(seed=FRAME_SEED)
    translated = shifted_frame(first, SHIFT_X, SHIFT_Y)
    unrelated = synthetic_frame(seed=UNRELATED_SEED)
    return {
        "expected_disp": float(np.hypot(SHIFT_X, SHIFT_Y)),
        "translated": lk_pair(first, translated),
        "unrelated": lk_pair(first, unrelated),
    }


def check_translated_pair(data: dict) -> None:
    """The known translation must be recovered and the forward/backward loop must close."""
    result = data["translated"]
    assert result is not None, "the grid did not fit into the synthetic frame"
    expected = data["expected_disp"]
    assert abs(result["lk_disp_p50"] - expected) <= 1.0, (
        f"lk_disp_p50 {result['lk_disp_p50']:.3f} px is not the known shift {expected:.3f} px (+-1 px)")
    assert result["lk_fb_err_p50"] <= 0.5, (
        f"lk_fb_err_p50 {result['lk_fb_err_p50']:.3f} px is not ~0 px (+-0.5 px)")
    assert result["lk_trackable"] >= 0.8, (
        f"lk_trackable {result['lk_trackable']:.3f} is too low for a rigid translation")
    assert result["lk_trackable_fb2"] >= result["lk_trackable"], "the relaxed threshold must not be stricter"
    assert result["lk_diverged"] == 0, f"{result['lk_diverged']} cells diverged from the initial guess"


def check_unrelated_pair(data: dict) -> None:
    """Two independent noise frames have no correspondence: nothing may be trackable."""
    result = data["unrelated"]
    assert result is not None, "the grid did not fit into the synthetic frame"
    assert result["lk_trackable"] <= 0.05, (
        f"lk_trackable {result['lk_trackable']:.3f} on an unrelated pair - the metric reports a match "
        "where there is none")
    assert result["lk_trackable_fb2"] <= 0.05, (
        f"lk_trackable_fb2 {result['lk_trackable_fb2']:.3f} on an unrelated pair - the relaxed threshold "
        "reports a match where there is none")
    assert result["lk_fb_err_p50"] > 1.0, (
        f"lk_fb_err_p50 {result['lk_fb_err_p50']:.3f} px is too small for an unrelated pair")


def check_unrelated_pairs(data: dict) -> None:
    """Across many unrelated pairs the metric must stay at zero on average."""
    first = synthetic_frame(seed=FRAME_SEED)
    ratios = []
    for offset in range(UNRELATED_SEEDS):
        result = lk_pair(first, synthetic_frame(seed=UNRELATED_SEED + offset))
        assert result is not None, "the grid did not fit into the synthetic frame"
        ratios.append(result["lk_trackable"])
    median = float(np.median(ratios))
    mean = float(np.mean(ratios))
    assert median <= 0.05, f"median lk_trackable {median:.3f} over {UNRELATED_SEEDS} unrelated pairs is not ~0"
    assert mean <= 0.05, f"mean lk_trackable {mean:.3f} over {UNRELATED_SEEDS} unrelated pairs is not ~0"


def run() -> bool:
    """Run every control, print the measured numbers, return True when all of them hold."""
    data = measure()
    translated, unrelated = data["translated"], data["unrelated"]
    print(f"frame                 : {FRAME_WIDTH}x{FRAME_HEIGHT} textured noise, seed {FRAME_SEED}")
    print(f"known shift           : ({SHIFT_X:+d}, {SHIFT_Y:+d}) px  ->  |d| = {data['expected_disp']:.3f} px")
    print(f"translated pair       : disp_p50={translated['lk_disp_p50']:8.3f} px  "
          f"fb_err_p50={translated['lk_fb_err_p50']:7.3f} px  trackable={translated['lk_trackable']:.3f}  "
          f"trackable_fb2={translated['lk_trackable_fb2']:.3f}  diverged={translated['lk_diverged']}")
    print(f"unrelated noise pair  : disp_p50={unrelated['lk_disp_p50']:8.3f} px  "
          f"fb_err_p50={unrelated['lk_fb_err_p50']:7.3f} px  trackable={unrelated['lk_trackable']:.3f}  "
          f"trackable_fb2={unrelated['lk_trackable_fb2']:.3f}  diverged={unrelated['lk_diverged']}")

    failures = []
    for name, check in (("translated pair", lambda: check_translated_pair(data)),
                        ("unrelated pair", lambda: check_unrelated_pair(data)),
                        ("unrelated pairs", lambda: check_unrelated_pairs(data))):
        try:
            check()
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        return False
    print("PASS all LK flow self-tests")
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
