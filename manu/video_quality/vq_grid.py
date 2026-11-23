#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Spatial trackability by uniform-grid "simulated operator clicks".

WHY
---
The quality metrics of ``vq_metrics.py`` are whole-frame numbers.  The real operator does not
track "the target" - they click *anywhere* on the picture and expect the tracker to lock on.
So the question that matters for admission is: **if the operator clicks at an arbitrary place,
how often does the tracker actually stay locked?**

This module answers that with no annotation and no detection: it lays a uniform grid over the
valid picture (6 rows x 8 columns = 48 cells of 32x32 px on the canonical 640-wide frame),
simulates 48 operator clicks and runs a forward/backward template-matching closure test on
every cell:

    forward   t -> t+gap : NCC of the 32x32 template inside the search window must be >= 0.65
    backward  t+gap -> t : the template found at t+gap must match back inside the original
                           search window
    closure   || click_centre - back_matched_centre || <= 3 px

A cell counts as *trackable* only when all three conditions hold.  The headline number is

    Trackable Grid Ratio (TGR) = trackable cells / 48 clicks

The frames are prepared exactly like the quality-metric pipeline (black-bar crop, then resize to
the canonical width of 640), so the grid cells cover the same fraction of the scene in every clip
and the numbers are comparable with ``comparison.csv`` / ``windows.csv``.

TWO MATCHING DOMAINS (v2)
-------------------------
The plain (raw grey) closure test is **not sharp enough on its own**: a grossly defocused picture
still has a consistent low-frequency intensity pattern, so ``TM_CCOEFF_NORMED`` still finds it
and the backward pass still returns to the click point.

The second domain therefore repeats the whole closure test on a **band-pass (DoG) filtered**
picture.  A DoG (difference of two Gaussians) keeps the scale band a tracker actually locks on
and removes the smooth shading that survives defocus, so a blurred cell loses its correlation
peak instead of matching a smeared blob.  On top of that the forward response is checked for
**peak uniqueness** (ratio of the second-best peak to the best one, with a disc around the best
peak masked out), which is what the raw test cannot do: it rejects self-similar / mosaic
artefacts that would match "consistently" but lock onto the wrong structure.

MEASURED, NOT ASSUMED (full-batch run over the 25 clips of this task, 26,464 window evaluations)
-----------------------------------------------------------------------------------------------
The DoG domain alone is **not** the decisive filter - measured on the worst bad clip
``120BT-04-可见光.mp4`` (a heavily defocused close-up of reeds) the raw TGR is **0.833** and the
DoG TGR is still **0.771**.  What actually rejects that clip is the **texture floor**: only
**29.2 %** of its 48 cells carry enough Laplacian energy to host a feature at all, so the
combined "textured AND trackable" ratio (UGR) collapses to **0.125** - and two of the four
rejected clips sit at **UGR = 0.000**.

The two failure modes are therefore complementary and both are needed:
  * a clip that is **sharp but not trackable** (``130CT-02红外_IRT``: DET = 0.917 but DoG TGR =
    0.354, UGR = 0.333) is caught by the *closure* test;
  * a clip that is **smooth everywhere** (``120BT-04-可见光``: DoG TGR = 0.771 but DET = 0.292) is
    caught by the *texture floor*;
and the headline number reported to the business is **UGR**, because "clickable in principle" and
"lockable in practice" have to hold *in the same cell*.

Both domains are stored **per cell** (``<clip>.npz``), so the decision thresholds can be chosen
from the data afterwards without re-decoding the videos.

Besides TGR the module records what the strict ratio cannot express:
  * ``flat_cells``      - cells with no texture at all (sky, water): clicking there can never
                          lock, so they are counted as failures but reported separately;
  * ``tgr_textured``    - TGR restricted to the cells that do have texture;
  * ``tgr_best_block``  - best contiguous 2x2 block, i.e. "is there *somewhere* a textured,
                          trackable region?" - this separates "one textured corner" from "the
                          whole picture is mush";
  * ``lambda_min_*``    - Shi-Tomasi minimum eigenvalue of the structure tensor per cell
                          (static lockability: is there a corner to track at all?).
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
from vq_metrics import (  # noqa: E402
    canonical,
    detect_content_box,
    open_frames,
    probe_video,
)

GRID_ROWS = 6
GRID_COLS = 8
PATCH_SIZE = 32
SEARCH_R = 32
NCC_THRESH = 0.65
CLOSURE_THRESH = 3.0
FLAT_STD_THRESH = 3.0
GAP_FRAMES = 5
PAIRS_PER_WINDOW = 2

# Band-pass (difference of Gaussians) used for the blur-sensitive matching domain.
DOG_SIGMA_LOW = 1.5
DOG_SIGMA_HIGH = 4.0
# Radius (px) of the disc around the best peak that is ignored when looking for the runner-up.
PEAK_EXCLUDE_R = 6

# Default decision thresholds for the two domains.
RAW_NCC_THRESH = 0.65
RAW_CLOSURE_THRESH = 3.0
DOG_NCC_THRESH = 0.65
DOG_CLOSURE_THRESH = 3.0
DOG_UNIQ_THRESH = 0.90

CELL_METRICS = (
    "ncc_raw",
    "peak2_raw",
    "closure_raw",
    "ncc_dog",
    "peak2_dog",
    "closure_dog",
    "lam",
    "std",
    "lap_var_cell",
    "edge_cell",
)


class LazyArrays:
    """Array view over a ``.npz`` that decompresses each array exactly once.

    ``np.load`` on an ``.npz`` returns a lazy ``NpzFile``: every ``archive[key]`` call
    inflates that array again.  The per-window analyses read the same handful of arrays
    once per evaluated second, so the naive access pattern inflates a multi-megabyte
    array thousands of times and turns a seconds-long analysis into a tens-of-minutes one.
    """

    def __init__(self, npz_path: Path) -> None:
        self._archive = np.load(npz_path)
        self._cache: dict[str, np.ndarray] = {}

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = self._archive[key]
        return self._cache[key]


def shitiomasi_lambda_min(patch: np.ndarray) -> float:
    """Minimum eigenvalue of the structure tensor of one cell (Shi-Tomasi trackability)."""
    if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2:
        return 0.0
    image = patch.astype(np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3) * 0.125
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3) * 0.125
    sxx = float((gx * gx).mean())
    syy = float((gy * gy).mean())
    sxy = float((gx * gy).mean())
    half_trace = 0.5 * (sxx + syy)
    diff = 0.5 * (sxx - syy)
    return float(half_trace - np.sqrt(diff * diff + sxy * sxy))


def dog_filter(image: np.ndarray, sigma_low: float = DOG_SIGMA_LOW, sigma_high: float = DOG_SIGMA_HIGH) -> np.ndarray:
    """Band-pass (difference of Gaussians) version of a grey frame.

    A defocused picture keeps its low frequencies but loses the band a tracker can lock on, so
    matching inside the DoG picture reacts to blur while matching inside the raw picture does not.
    """
    src = image.astype(np.float32)
    low = cv2.GaussianBlur(src, (0, 0), sigma_low)
    high = cv2.GaussianBlur(src, (0, 0), sigma_high)
    return low - high


def _peak_ratio(response: np.ndarray, exclude: int = PEAK_EXCLUDE_R) -> float:
    """Ratio of the runner-up peak to the best peak (1.0 = completely ambiguous response)."""
    _, max_val, _, max_loc = cv2.minMaxLoc(response)
    if not np.isfinite(max_val) or max_val <= 0:
        return float("nan")
    mask = np.ones(response.shape, dtype=bool)
    y, x = int(max_loc[1]), int(max_loc[0])
    mask[max(0, y - exclude): min(response.shape[0], y + exclude + 1),
         max(0, x - exclude): min(response.shape[1], x + exclude + 1)] = False
    if not mask.any():
        return float("nan")
    return float(response[mask].max() / max_val)


def grid_centers(height: int, width: int, rows: int, cols: int, patch: int, search_r: int):
    """Uniform grid of simulated click centres inside the valid picture."""
    half = patch // 2
    margin = half + search_r + 2
    if height <= 2 * margin or width <= 2 * margin:
        return None
    ys = np.linspace(margin, height - margin, rows).astype(int)
    xs = np.linspace(margin, width - margin, cols).astype(int)
    return ys, xs


def evaluate_pair(
    gray_t: np.ndarray,
    gray_tk: np.ndarray,
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
    patch: int = PATCH_SIZE,
    search_r: int = SEARCH_R,
    raw_ncc_thresh: float = RAW_NCC_THRESH,
    raw_closure_thresh: float = RAW_CLOSURE_THRESH,
    dog_ncc_thresh: float = DOG_NCC_THRESH,
    dog_closure_thresh: float = DOG_CLOSURE_THRESH,
) -> dict | None:
    """One forward/backward closure test over the whole grid, in both matching domains.

    Returns ``None`` when the picture is too small for the requested grid.  Every per-cell value
    is returned so the decision thresholds can be re-derived offline from the stored arrays.
    """
    height, width = gray_t.shape
    centers = grid_centers(height, width, rows, cols, patch, search_r)
    if centers is None:
        return None
    ys, xs = centers
    half = patch // 2

    dog_t = dog_filter(gray_t)
    dog_tk = dog_filter(gray_tk)

    cells = {
        name: np.full((rows, cols), np.nan, dtype=float)
        for name in CELL_METRICS
    }
    flat = np.zeros((rows, cols), dtype=bool)

    for row_index, cy in enumerate(ys):
        for col_index, cx in enumerate(xs):
            template = gray_t[cy - half: cy + half, cx - half: cx + half]
            if template.size != patch * patch:
                continue
            cells["std"][row_index, col_index] = float(np.std(template))
            cells["lam"][row_index, col_index] = shitiomasi_lambda_min(template)
            # per-cell fine detail: the same two statistics the whole-frame red lines use
            # (Laplacian variance and Canny edge density), so the grid is spatially resolved
            # version of the existing "sharpness / edge density" red lines.
            cells["lap_var_cell"][row_index, col_index] = float(cv2.Laplacian(template, cv2.CV_64F).var())
            cells["edge_cell"][row_index, col_index] = float((cv2.Canny(template, 60, 160) > 0).mean())
            if float(np.std(template)) < FLAT_STD_THRESH:
                flat[row_index, col_index] = True
                continue

            x1, y1 = cx - search_r, cy - search_r
            x2, y2 = cx + search_r, cy + search_r

            # --- raw (grey) domain: exactly the test described in the task document ---
            ncc_raw, peak2_raw, closure_raw = _domain_closure(
                gray_t, gray_tk, cx, cy, half, x1, y1, x2, y2, raw_ncc_thresh, raw_closure_thresh
            )
            cells["ncc_raw"][row_index, col_index] = ncc_raw
            cells["peak2_raw"][row_index, col_index] = peak2_raw
            cells["closure_raw"][row_index, col_index] = closure_raw

            # --- band-pass (DoG) domain: the blur-sensitive variant ---
            ncc_dog, peak2_dog, closure_dog = _domain_closure(
                dog_t, dog_tk, cx, cy, half, x1, y1, x2, y2, dog_ncc_thresh, dog_closure_thresh
            )
            cells["ncc_dog"][row_index, col_index] = ncc_dog
            cells["peak2_dog"][row_index, col_index] = peak2_dog
            cells["closure_dog"][row_index, col_index] = closure_dog

    return {"cells": cells, "flat": flat}


def _domain_closure(
    source: np.ndarray,
    target: np.ndarray,
    click_x: int,
    click_y: int,
    half: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    ncc_thresh: float,
    closure_thresh: float,
) -> tuple[float, float, float]:
    """Forward/backward closure test of one cell inside one matching domain.

    ``source`` is the frame the click template is taken from, ``target`` the frame the template is
    searched in.  Returns ``(ncc, peak_ratio, closure_error)``; ``ncc`` is NaN when the forward
    match failed, ``closure_error`` is NaN when the backward match did not return.
    """
    height, width = source.shape
    template = source[click_y - half: click_y + half, click_x - half: click_x + half]
    if template.size != half * 2 * half * 2:
        return float("nan"), float("nan"), float("nan")
    roi_target = target[y1:y2, x1:x2]
    if roi_target.shape[0] < 2 * half or roi_target.shape[1] < 2 * half:
        return float("nan"), float("nan"), float("nan")

    response = cv2.matchTemplate(roi_target, template, cv2.TM_CCOEFF_NORMED)
    _, max_ncc, _, location = cv2.minMaxLoc(response)
    peak_ratio = _peak_ratio(response)
    if not np.isfinite(max_ncc) or max_ncc < ncc_thresh:
        return float(max_ncc), peak_ratio, float("nan")

    pred_x = x1 + location[0] + half
    pred_y = y1 + location[1] + half
    if pred_x - half < 0 or pred_x + half >= width or pred_y - half < 0 or pred_y + half >= height:
        return float(max_ncc), peak_ratio, float("nan")

    template_next = target[pred_y - half: pred_y + half, pred_x - half: pred_x + half]
    roi_source = source[y1:y2, x1:x2]
    if template_next.size != half * 2 * half * 2 or roi_source.shape[0] < 2 * half or roi_source.shape[1] < 2 * half:
        return float(max_ncc), peak_ratio, float("nan")
    back_response = cv2.matchTemplate(roi_source, template_next, cv2.TM_CCOEFF_NORMED)
    _, _, _, back_location = cv2.minMaxLoc(back_response)
    back_x = x1 + back_location[0] + half
    back_y = y1 + back_location[1] + half
    error = float(np.hypot(click_x - back_x, click_y - back_y))
    return float(max_ncc), peak_ratio, error


def decide(cells: dict, flat: np.ndarray, domain: str, ncc_thresh: float, closure_thresh: float,
           uniq_thresh: float | None) -> np.ndarray:
    """Boolean mask of the trackable cells for one matching domain."""
    ncc = cells[f"ncc_{domain}"]
    closure = cells[f"closure_{domain}"]
    ok = (~flat) & (ncc >= ncc_thresh) & (closure <= closure_thresh)
    if uniq_thresh is not None:
        peak2 = cells[f"peak2_{domain}"]
        ok = ok & (peak2 <= uniq_thresh)
    return ok


def summarise(cells: dict, flat: np.ndarray) -> dict:
    """Scalar summary of one grid evaluation (both domains)."""
    rows, cols = flat.shape
    total = rows * cols
    textured = int((~flat).sum())

    def _block(mask: np.ndarray) -> int:
        best = 0
        for r in range(rows - 1):
            for c in range(cols - 1):
                best = max(best, int(mask[r:r + 2, c:c + 2].sum()))
        return best

    out: dict = {}
    for domain, uniq in (("raw", None), ("dog", DOG_UNIQ_THRESH)):
        mask = decide(cells, flat, domain,
                      RAW_NCC_THRESH if domain == "raw" else DOG_NCC_THRESH,
                      RAW_CLOSURE_THRESH if domain == "raw" else DOG_CLOSURE_THRESH,
                      uniq)
        count = int(mask.sum())
        out[f"tgr_{domain}"] = count / total
        out[f"tgr_{domain}_textured"] = (count / textured) if textured else float("nan")
        out[f"tgr_{domain}_best_block"] = _block(mask) / 4.0
        out[f"tgr_{domain}_best_row"] = max((float(mask[i].mean()) for i in range(rows)), default=0.0)
        out[f"tgr_{domain}_best_col"] = max((float(mask[:, j].mean()) for j in range(cols)), default=0.0)
        out[f"tgr_{domain}_top_half"] = float(mask[: rows // 2].mean())
        out[f"tgr_{domain}_bottom_half"] = float(mask[rows // 2:].mean())
        out[f"cells_{domain}_map"] = mask.astype(float).tolist()

    out["flat_cells"] = int(flat.sum())
    out["textured_cells"] = textured
    out["total_cells"] = total
    out["ncc_raw_p50"] = float(np.nanmedian(cells["ncc_raw"])) if np.isfinite(cells["ncc_raw"]).any() else float("nan")
    out["ncc_dog_p50"] = float(np.nanmedian(cells["ncc_dog"])) if np.isfinite(cells["ncc_dog"]).any() else float("nan")
    out["peak2_dog_p50"] = (
        float(np.nanmedian(cells["peak2_dog"])) if np.isfinite(cells["peak2_dog"]).any() else float("nan")
    )
    out["closure_raw_p50"] = (
        float(np.nanmedian(cells["closure_raw"])) if np.isfinite(cells["closure_raw"]).any() else float("nan")
    )
    out["closure_dog_p50"] = (
        float(np.nanmedian(cells["closure_dog"])) if np.isfinite(cells["closure_dog"]).any() else float("nan")
    )
    out["lambda_min_p50"] = float(np.nanmedian(cells["lam"]))
    out["lambda_min_p90"] = float(np.nanpercentile(cells["lam"], 90))
    out["std_p50"] = float(np.nanmedian(cells["std"]))
    out["lap_var_cell_p50"] = float(np.nanmedian(cells["lap_var_cell"]))
    out["lap_var_cell_p90"] = float(np.nanpercentile(cells["lap_var_cell"], 90))
    out["edge_cell_p50"] = float(np.nanmedian(cells["edge_cell"]))
    out["flat_map"] = flat.astype(float).tolist()
    return out


def analyse_clip(
    path: Path,
    meta: dict,
    gap: int = GAP_FRAMES,
    pairs_per_window: int = PAIRS_PER_WINDOW,
    max_windows: int = 0,
) -> tuple[dict, list[dict]]:
    """Decode one clip and evaluate the grid on every sampled frame pair."""
    info = probe_video(path)
    width, height = info["width"], info["height"]
    fps = float(meta.get("fps_used") or info["fps"] or 25.0)
    window = max(1, int(round(fps * float(meta.get("stride", 1)))))

    crop: tuple[int, int, int, int] | None = None
    ring: dict[int, np.ndarray] = {}
    index = 0
    processed = 0
    rows_out: list[dict] = []
    cell_stack: dict[str, list[np.ndarray]] = {name: [] for name in CELL_METRICS}
    flat_stack: list[np.ndarray] = []
    window_stack: list[int] = []
    pair_stack: list[int] = []
    first_stack: list[int] = []
    second_stack: list[int] = []

    for frame in open_frames(path, width, height):
        index += 1
        if crop is None:
            crop = detect_content_box(frame)
        top, bottom, left, right = crop
        content = frame[top:height - bottom, left:width - right]
        if content.size == 0:
            content = frame
        canon = canonical(content)
        ring[index - 1] = canon

        window_index = (index - 1) // window
        # Evaluate each pair EXACTLY once, when its second frame has just been decoded.
        for pair_index in range(pairs_per_window):
            base = window_index * window + pair_index * max(1, window // pairs_per_window)
            first = base
            second = base + gap
            if index - 1 == second and first in ring and second in ring:
                result = evaluate_pair(ring[first], ring[second])
                if result is not None:
                    cells, flat = result["cells"], result["flat"]
                    row = summarise(cells, flat)
                    row.update({
                        "file": meta.get("file", path.name),
                        "group": meta.get("group", "?"),
                        "kind": meta.get("kind", "?"),
                        "modality": meta.get("modality", "?"),
                        "window": window_index,
                        "pair": pair_index,
                        "frame_first": first,
                        "frame_second": second,
                    })
                    rows_out.append(row)
                    for name in CELL_METRICS:
                        cell_stack[name].append(cells[name])
                    flat_stack.append(flat.astype(float))
                    window_stack.append(window_index)
                    pair_stack.append(pair_index)
                    first_stack.append(first)
                    second_stack.append(second)

        keep_from = (index - 1) - gap
        for key in [k for k in ring if k < keep_from]:
            del ring[key]

        processed = index
        if max_windows and index >= max_windows * window:
            break

    def agg(values: list[float], fn) -> float:
        values = [v for v in values if v == v]
        return float(fn(values)) if values else float("nan")

    def agg_of(key: str, fn) -> float:
        return agg([row[key] for row in rows_out], fn)

    summary = {
        "tgr_raw_p50": agg_of("tgr_raw", np.median),
        "tgr_raw_mean": agg_of("tgr_raw", np.mean),
        "tgr_raw_p05": agg_of("tgr_raw", lambda v: np.percentile(v, 5)),
        "tgr_dog_p50": agg_of("tgr_dog", np.median),
        "tgr_dog_mean": agg_of("tgr_dog", np.mean),
        "tgr_dog_p05": agg_of("tgr_dog", lambda v: np.percentile(v, 5)),
        "tgr_dog_p95": agg_of("tgr_dog", lambda v: np.percentile(v, 95)),
        "tgr_dog_textured_p50": agg_of("tgr_dog_textured", np.median),
        "tgr_dog_best_block_p50": agg_of("tgr_dog_best_block", np.median),
        "tgr_dog_best_block_max": agg_of("tgr_dog_best_block", np.max),
        "tgr_dog_best_row_p50": agg_of("tgr_dog_best_row", np.median),
        "flat_cells_p50": agg_of("flat_cells", np.median),
        "ncc_raw_p50": agg_of("ncc_raw_p50", np.median),
        "ncc_dog_p50": agg_of("ncc_dog_p50", np.median),
        "peak2_dog_p50": agg_of("peak2_dog_p50", np.median),
        "closure_raw_p50": agg_of("closure_raw_p50", np.median),
        "closure_dog_p50": agg_of("closure_dog_p50", np.median),
        "lambda_min_p50": agg_of("lambda_min_p50", np.median),
        "lap_var_cell_p50": agg_of("lap_var_cell_p50", np.median),
        "edge_cell_p50": agg_of("edge_cell_p50", np.median),
        "localisable_ratio": agg_of("tgr_dog_best_block", lambda v: float(np.mean(np.array(v) >= 0.5))),
    }

    arrays = {
        "flat": (np.asarray(flat_stack, dtype=np.float32) if flat_stack
                 else np.zeros((0, GRID_ROWS, GRID_COLS), np.float32)),
        "window": np.asarray(window_stack, dtype=np.int32),
        "pair": np.asarray(pair_stack, dtype=np.int32),
        "frame_first": np.asarray(first_stack, dtype=np.int32),
        "frame_second": np.asarray(second_stack, dtype=np.int32),
    }
    for name in CELL_METRICS:
        arrays[name] = (np.asarray(cell_stack[name], dtype=np.float32) if cell_stack
                        else np.zeros((0, GRID_ROWS, GRID_COLS), np.float32))

    result = {
        "file": meta.get("file", path.name),
        "group": meta.get("group", "?"),
        "kind": meta.get("kind", "?"),
        "modality": meta.get("modality", "?"),
        "frames_decoded": processed,
        "fps_used": fps,
        "window_frames": window,
        "grid_rows": GRID_ROWS,
        "grid_cols": GRID_COLS,
        "patch_size": PATCH_SIZE,
        "search_r": SEARCH_R,
        "gap_frames": gap,
        "pairs_per_window": pairs_per_window,
        "dog_sigma_low": DOG_SIGMA_LOW,
        "dog_sigma_high": DOG_SIGMA_HIGH,
        "thresholds": {
            "raw_ncc": RAW_NCC_THRESH,
            "raw_closure": RAW_CLOSURE_THRESH,
            "dog_ncc": DOG_NCC_THRESH,
            "dog_closure": DOG_CLOSURE_THRESH,
            "dog_uniq": DOG_UNIQ_THRESH,
        },
        "evaluations": len(rows_out),
        "summary": summary,
    }
    return result, rows_out, arrays


def load_meta(meta_dir: Path, name: str) -> dict:
    path = meta_dir / f"{Path(name).stem}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Uniform-grid spatial trackability per clip")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--meta-dir", type=str, required=True, help="Directory with the per-clip metric JSONs")
    parser.add_argument("--out", type=str, required=True, help="Output JSON path")
    parser.add_argument("--npz", type=str, default="", help="Optional per-cell NPZ path")
    parser.add_argument("--rows-csv", type=str, default="", help="Optional CSV path for the per-pair rows")
    parser.add_argument("--gap", type=int, default=GAP_FRAMES)
    parser.add_argument("--pairs-per-window", type=int, default=PAIRS_PER_WINDOW)
    parser.add_argument("--max-windows", type=int, default=0)
    args = parser.parse_args()

    video = Path(args.video)
    meta = load_meta(Path(args.meta_dir), video.name)
    result, rows, arrays = analyse_clip(
        video, meta, gap=args.gap, pairs_per_window=args.pairs_per_window, max_windows=args.max_windows
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.npz:
        npz_path = Path(args.npz)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(npz_path), **arrays)
    if args.rows_csv:
        csv_path = Path(args.rows_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [k for k in rows[0] if k not in ("cells_raw_map", "cells_dog_map", "flat_map")] if rows else []
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"file": result["file"], "evaluations": result["evaluations"],
                      "tgr_raw_p50": result["summary"]["tgr_raw_p50"],
                      "tgr_dog_p50": result["summary"]["tgr_dog_p50"],
                      "tgr_dog_best_block_p50": result["summary"]["tgr_dog_best_block_p50"],
                      "flat_cells_p50": result["summary"]["flat_cells_p50"]}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
