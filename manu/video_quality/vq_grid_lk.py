#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Control experiment: the same grid, but with Lucas-Kanade optical flow instead of
template matching.

The task document allows either implementation ("前向-后向双向模板匹配（或 LK 光流）").
Running both on the same clips answers the question a reviewer will ask first: *is the
"this clip is still trackable" result a property of the picture, or an artefact of the
template-matching implementation?*

A grid cell counts as trackable when
  * the forward Lucas-Kanade flow returns a valid status for the click point,
  * the backward flow (from the predicted position in frame t+gap back to frame t) is valid too,
  * the forward-backward error is <= 1 px (the classic LK consistency criterion),
  * the flow did not run away from its initial guess (a diverging LK is a failure, not a
    match) and the implied motion stays inside the physically plausible range.

``cv2.calcOpticalFlowPyrLK`` returns the **absolute** position of every point in the second
image, not the per-point displacement.  The forward pass therefore yields the position in
frame t+gap directly, the backward pass yields the position in frame t, and the motion of a
cell is the distance between its two absolute positions.

The frame preparation is identical to ``vq_grid.py`` (black-bar crop, resize to width 640), so
the numbers are directly comparable with the template-matching run.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vq_grid import GRID_COLS, GRID_ROWS, PATCH_SIZE, SEARCH_R, grid_centers  # noqa: E402
from vq_metrics import canonical, detect_content_box, open_frames, probe_video  # noqa: E402

FB_THRESH = 1.0
FB_THRESH_RELAXED = 2.0
MAX_FLOW_FROM_GUESS = 30.0
MAX_DISP = 80.0
GAP_FRAMES = 5
PAIRS_PER_WINDOW = 2
MIN_STD = 3.0

LK_PARAMS = dict(winSize=(21, 21), maxLevel=3, minEigThreshold=1e-4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def global_shift(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """Global translation between two frames from phase correlation (initial guess for LK).

    Lucas-Kanade without an initial guess diverges on the shaky, low-texture frames of this
    task (the measured displacements were hundreds of pixels), so the global shift is measured
    first and LK is only allowed to refine it locally.
    """
    width = first.shape[1]
    small_width = 320
    small_height = max(1, int(round(first.shape[0] * small_width / width)))
    small_first = cv2.resize(first, (small_width, small_height), interpolation=cv2.INTER_AREA)
    small_second = cv2.resize(second, (small_width, small_height), interpolation=cv2.INTER_AREA)
    (dx, dy), _ = cv2.phaseCorrelate(small_first.astype(np.float32), small_second.astype(np.float32))
    scale = width / small_width
    return float(dx * scale), float(dy * scale)


def lk_pair(gray_t: np.ndarray, gray_tk: np.ndarray) -> dict | None:
    """Forward/backward Lucas-Kanade closure test over the whole grid."""
    height, width = gray_t.shape
    centers = grid_centers(height, width, GRID_ROWS, GRID_COLS, PATCH_SIZE, SEARCH_R)
    if centers is None:
        return None
    ys, xs = centers
    half = PATCH_SIZE // 2

    points = np.array([[cx, cy] for cy in ys for cx in xs], dtype=np.float32).reshape(-1, 2)
    std = np.zeros(len(points), dtype=float)
    flat = np.zeros(len(points), dtype=bool)
    for index, (cx, cy) in enumerate(points):
        patch = gray_t[int(cy) - half: int(cy) + half, int(cx) - half: int(cx) + half]
        if patch.size != PATCH_SIZE * PATCH_SIZE:
            continue
        std[index] = float(np.std(patch))
        flat[index] = std[index] < MIN_STD

    shift_x, shift_y = global_shift(gray_t, gray_tk)
    guess = points + np.array([shift_x, shift_y], dtype=np.float32)
    # OpenCV returns the absolute positions in nextImg (not a delta) and overwrites nextPts.
    forward, status_fw, _ = cv2.calcOpticalFlowPyrLK(gray_t, gray_tk, points.reshape(-1, 1, 2),
                                                     guess.reshape(-1, 1, 2).copy(),
                                                     flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **LK_PARAMS)
    forward = forward.reshape(-1, 2)
    status_fw = status_fw.reshape(-1)

    backward, status_bw, _ = cv2.calcOpticalFlowPyrLK(gray_tk, gray_t, forward.reshape(-1, 1, 2).copy(),
                                                      points.reshape(-1, 1, 2).copy(),
                                                      flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **LK_PARAMS)
    backward = backward.reshape(-1, 2)
    status_bw = status_bw.reshape(-1)

    error = np.linalg.norm(backward - points, axis=1)
    displacement = np.linalg.norm(forward - points, axis=1)
    drift = np.maximum(np.linalg.norm(forward - guess, axis=1),
                       np.linalg.norm(backward - points, axis=1))
    diverged = drift > MAX_FLOW_FROM_GUESS

    usable = (status_fw == 1) & (status_bw == 1) & (~flat) & (~diverged) & (displacement <= MAX_DISP)
    trackable = usable & (error <= FB_THRESH)
    trackable_fb2 = usable & (error <= FB_THRESH_RELAXED)
    return {
        "lk_trackable": float(trackable.mean()),
        "lk_trackable_fb2": float(trackable_fb2.mean()),
        "lk_valid_fwd": float((status_fw == 1).mean()),
        "lk_valid_bwd": float((status_bw == 1).mean()),
        "lk_fb_err_p50": float(np.median(error)),
        "lk_disp_p50": float(np.median(displacement)),
        "lk_diverged": int(diverged.sum()),
        "lk_flat_cells": int(flat.sum()),
    }


def analyse(path: Path, meta: dict, gap: int = GAP_FRAMES, pairs_per_window: int = PAIRS_PER_WINDOW,
            max_windows: int = 0) -> tuple[dict, list[dict]]:
    info = probe_video(path)
    width, height = info["width"], info["height"]
    fps = float(meta.get("fps_used") or info["fps"] or 25.0)
    window = max(1, int(round(fps * float(meta.get("stride", 1)))))

    crop = None
    ring: dict[int, np.ndarray] = {}
    index = 0
    rows_out: list[dict] = []
    for frame in open_frames(path, width, height):
        index += 1
        if crop is None:
            crop = detect_content_box(frame)
        top, bottom, left, right = crop
        content = frame[top:height - bottom, left:width - right]
        if content.size == 0:
            content = frame
        ring[index - 1] = canonical(content)
        window_index = (index - 1) // window
        for pair_index in range(pairs_per_window):
            base = window_index * window + pair_index * max(1, window // pairs_per_window)
            first, second = base, base + gap
            if index - 1 == second and first in ring and second in ring:
                result = lk_pair(ring[first], ring[second])
                if result is not None:
                    result.update({"file": meta.get("file", path.name), "group": meta.get("group", "?"),
                                   "kind": meta.get("kind", "?"), "modality": meta.get("modality", "?"),
                                   "window": window_index, "pair": pair_index,
                                   "frame_first": first, "frame_second": second})
                    rows_out.append(result)
        keep_from = (index - 1) - gap
        for key in [k for k in ring if k < keep_from]:
            del ring[key]
        if max_windows and index >= max_windows * window:
            break

    def agg(key: str, fn) -> float:
        values = [row[key] for row in rows_out if row[key] == row[key]]
        return float(fn(values)) if values else float("nan")

    summary = {
        "lk_trackable_p50": agg("lk_trackable", np.median),
        "lk_trackable_mean": agg("lk_trackable", np.mean),
        "lk_trackable_p50_fb2": agg("lk_trackable_fb2", np.median),
        "lk_trackable_mean_fb2": agg("lk_trackable_fb2", np.mean),
        "lk_valid_fwd_p50": agg("lk_valid_fwd", np.median),
        "lk_fb_err_p50": agg("lk_fb_err_p50", np.median),
        "lk_disp_p50": agg("lk_disp_p50", np.median),
        "lk_diverged_p50": agg("lk_diverged", np.median),
        "lk_flat_cells_p50": agg("lk_flat_cells", np.median),
    }
    return {"file": meta.get("file", path.name), "group": meta.get("group", "?"),
            "kind": meta.get("kind", "?"), "modality": meta.get("modality", "?"),
            "evaluations": len(rows_out), "summary": summary}, rows_out


def run_selftest() -> bool:
    """Run the synthetic flow self-test of ``test_vq_grid_lk.py`` and report the result."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_vq_grid_lk

    return bool(test_vq_grid_lk.run())


def main() -> None:
    parser = argparse.ArgumentParser(description="LK optical flow control experiment")
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--meta-dir", type=str, default="")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--rows-csv", type=str, default="")
    parser.add_argument("--gap", type=int, default=GAP_FRAMES)
    parser.add_argument("--pairs-per-window", type=int, default=PAIRS_PER_WINDOW)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--selftest", action="store_true", help="run the synthetic flow self-test and exit")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if not (args.video and args.meta_dir and args.out):
        parser.error("--video, --meta-dir and --out are required unless --selftest is given")

    video = Path(args.video)
    meta_path = Path(args.meta_dir) / f"{video.stem}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    result, rows = analyse(video, meta, gap=args.gap, pairs_per_window=args.pairs_per_window,
                           max_windows=args.max_windows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.rows_csv and rows:
        csv_path = Path(args.rows_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"file": result["file"], "evaluations": result["evaluations"],
                      **result["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
