#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Connected-component attributes (``area`` / ``sums``) of the heatmap peak candidates.

What this module is
-------------------
A **post-processing** helper for the single-frame heatmap detector.  Given the raw heatmap and
the sub-pixel offset head it returns the *same* peak list as
:func:`manu.evaluation.heatmap_evaluate.extract_peaks` -- the ``points`` / ``scores`` arrays are
literally the arrays that function returned, so they are bit-identical by construction -- plus
two attributes of the connected component the peak lives in:

* ``area``  -- pixel count of that component (8-connectivity) on the model output grid,
* ``sums``  -- the integral of the heatmap over that component (the "blob energy").

What this module is not
-----------------------
* It is **not** a second peak extractor: the peak set is owned by ``extract_peaks``; this module
  only *adds* attributes to it.  ``extract_peaks`` itself is never modified.
* It is **not** a detector, not a training script and not an inference path -- it never touches a
  GPU and never rebuilds a cache.
* It is **not** the Longquanshan decision layer: the T3 pipeline has its own multi-scale
  candidate pool and its own gate (``manu/evaluation/candidate_gate.py``).  This module is the
  single-scale T1 counterpart of the *attribute computation* only.

How to use it
-------------
::

    from manu.evaluation.peak_attributes import extract_peaks_with_area_sums

    peaks = extract_peaks_with_area_sums(heatmap, offset, stride=2, conf_thresh=0.02,
                                         top_k=100, component_thresh=0.06)
    peaks[0]["points"], peaks[0]["scores"], peaks[0]["area"], peaks[0]["sums"]

Semantics of a peak without a component
---------------------------------------
The components are computed on the **binarised** heatmap (``heatmap >= component_thresh``).  A
peak whose own score is below ``component_thresh`` therefore belongs to **no** component and gets
``area = 0`` / ``sums = 0``.  Such a peak is rejected by every gate with ``area_min >= 1``.

Relationship between ``component_thresh`` and ``conf_thresh``
------------------------------------------------------------
``conf_thresh`` decides **whether a peak is cached at all** (``heatmap >= conf_thresh``);
``component_thresh`` decides **which pixels form the blob the peak is measured against**.  The
two are independent knobs:

* ``component_thresh <= conf_thresh``  -> every cached peak owns a component, so ``area_min >= 1``
  is a no-op and only the energy gate can discriminate.
* ``component_thresh > conf_thresh``  -> peaks with a score in ``[conf_thresh, component_thresh)``
  get ``area = 0`` / ``sums = 0`` and are removed by ``area_min >= 1``, which effectively raises
  the harvest threshold.  That is a *threshold change in disguise*, not a discriminative gate,
  and is reported as such.

The default ``component_thresh = 0.06`` mirrors the per-scale ``region_threshold`` of the
Longquanshan multi-scale cache (``manu/data/multiscale_heatmap_cache.py``).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from manu.evaluation.heatmap_evaluate import extract_peaks

# The default binarisation level of the connected-component analysis (see the module docstring).
DEFAULT_COMPONENT_THRESH = 0.06


def peak_grid_indices(
    heatmap: torch.Tensor,
    conf_thresh: float = 0.20,
    top_k: int = 100,
    kernel_size: int = 3,
    native_scale: int = 1,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Re-derive the grid indices of the peaks that ``extract_peaks`` selects.

    The selection is the **same op sequence** as in ``extract_peaks`` (3x3 max-pool NMS, the same
    threshold, the same ``torch.topk``), so the returned indices are the ones the reference
    implementation used, in the same order.  They are needed because the connected component of a
    peak can only be found through its grid cell.

    Args:
        heatmap: (B, 1, H, W) heatmap tensor in [0, 1].
        conf_thresh: peak confidence threshold (lower => higher recall).
        top_k: max number of detections per image.
        kernel_size: max pool kernel size.
        native_scale: late_upsample 的上采样倍数 L（>1 时在原生网格上抽峰，
            见 ``extract_peaks`` 的说明）。

    Returns:
        One ``(y_indices, x_indices, scores)`` triple per image; the arrays are int64 / int64 /
        float32 and are aligned one to one with ``extract_peaks``'s output.
    """
    if native_scale and native_scale > 1:
        heatmap = heatmap[:, :, ::native_scale, ::native_scale]
    pad = (kernel_size - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel_size=kernel_size, stride=1, padding=pad)
    keep = (hmax == heatmap) & (heatmap >= conf_thresh)

    results = []
    for b in range(heatmap.shape[0]):
        mask_b = keep[b, 0]
        scores = heatmap[b, 0][mask_b]
        if scores.numel() == 0:
            results.append((
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
            ))
            continue
        if scores.numel() > top_k:
            scores, topk_inds = torch.topk(scores, top_k)
            nz = torch.nonzero(mask_b, as_tuple=False)
            y_coords = nz[:, 0][topk_inds]
            x_coords = nz[:, 1][topk_inds]
        else:
            nz = torch.nonzero(mask_b, as_tuple=False)
            y_coords = nz[:, 0]
            x_coords = nz[:, 1]
        results.append((
            y_coords.detach().cpu().numpy().astype(np.int64),
            x_coords.detach().cpu().numpy().astype(np.int64),
            scores.detach().cpu().numpy().astype(np.float32),
        ))
    return results


def component_tables(heatmap: np.ndarray, component_thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Connected components of one heatmap frame and their per-label area / energy.

    The mask is ``heatmap >= component_thresh`` (uint8) and the connectivity is 8, i.e. the same
    convention as ``manu/data/multiscale_heatmap_cache.py::detect_components``.

    Args:
        heatmap: (H, W) float32 heatmap of one frame.
        component_thresh: binarisation level of the component analysis.

    Returns:
        ``(labels, area_per_label, sums_per_label)``: the int32 label image (0 = background), the
        pixel count of every label and the heatmap integral of every label.  The integrals are
        accumulated in float64 and are therefore free of the pairwise-summation rounding of a
        float32 ``ndarray.sum()``.
    """
    mask = (heatmap >= component_thresh).astype(np.uint8)
    count, labels = cv2.connectedComponentsWithStats(mask, connectivity=8)[:2]
    flat = labels.reshape(-1)
    area = np.bincount(flat, minlength=count).astype(np.int64)
    sums = np.bincount(flat, weights=heatmap.reshape(-1).astype(np.float64), minlength=count)
    return labels, area, sums


def extract_peaks_with_area_sums(
    heatmap: torch.Tensor,
    offset: torch.Tensor,
    stride: int = 4,
    conf_thresh: float = 0.20,
    top_k: int = 100,
    kernel_size: int = 3,
    component_thresh: float = DEFAULT_COMPONENT_THRESH,
    verify: bool = True,
    native_scale: int = 1,
    offset_scale: float = 1.0,
    offset_bias: float = 0.0,
    offset_scale_xy: tuple[float, float] | None = None,
    offset_bias_xy: tuple[float, float] | None = None,
) -> list[dict[str, np.ndarray]]:
    """Extract the peaks of ``extract_peaks`` and attach their component attributes.

    The returned ``points`` and ``scores`` are **the very arrays** produced by
    :func:`manu.evaluation.heatmap_evaluate.extract_peaks` (bit-identical by construction), and
    ``area`` / ``sums`` are the attributes of the connected component each peak belongs to.

    Args:
        heatmap: (B, 1, H, W) heatmap tensor in [0, 1].
        offset: (B, 2, H, W) sub-pixel offsets (dx, dy).
        stride: downsample stride of the heatmap grid.
        conf_thresh: peak confidence threshold.
        top_k: max number of detections per image.
        kernel_size: max pool kernel size of the NMS.
        component_thresh: binarisation level of the connected-component analysis.
        verify: check that the re-derived peak indices reproduce the reference scores exactly and
            raise if they do not (the guard against a silent misalignment of the two paths).
        native_scale: late_upsample 的上采样倍数 L（>1 时在原生网格上抽峰并用 stride*L 解码，
            见 :func:`manu.evaluation.heatmap_evaluate.extract_peaks`）。
        offset_scale / offset_bias: offset 仿射校准（默认空操作，见 extract_peaks 的说明）。

    Returns:
        One dict per image with ``points`` (N, 2) float32, ``scores`` (N,) float32,
        ``area`` (N,) int32 and ``sums`` (N,) float32.
    """
    reference = extract_peaks(
        heatmap=heatmap,
        offset=offset,
        stride=stride,
        conf_thresh=conf_thresh,
        top_k=top_k,
        kernel_size=kernel_size,
        native_scale=native_scale,
        offset_scale=offset_scale,
        offset_bias=offset_bias,
        offset_scale_xy=offset_scale_xy,
        offset_bias_xy=offset_bias_xy,
    )
    indices = peak_grid_indices(
        heatmap,
        conf_thresh=conf_thresh,
        top_k=top_k,
        kernel_size=kernel_size,
        native_scale=native_scale,
    )

    results = []
    for b, ref in enumerate(reference):
        y_idx, x_idx, scores = indices[b]
        n_ref = len(ref["scores"])
        if len(y_idx) != n_ref:
            raise RuntimeError(
                f"[PEAK-ATTR] peak count mismatch on image {b}: reference={n_ref} indices={len(y_idx)}"
            )
        if verify and not np.array_equal(scores, ref["scores"]):
            raise RuntimeError(
                f"[PEAK-ATTR] peak scores differ from extract_peaks on image {b}; "
                f"the index path and the reference path are out of sync"
            )
        area = np.zeros(n_ref, dtype=np.int32)
        sums = np.zeros(n_ref, dtype=np.float32)
        if n_ref:
            heat_np = heatmap[b, 0].detach().cpu().numpy()
            labels, area_lut, sums_lut = component_tables(heat_np, component_thresh)
            label_of_peak = labels[y_idx, x_idx]
            inside = label_of_peak > 0
            area[inside] = area_lut[label_of_peak[inside]].astype(np.int32)
            sums[inside] = sums_lut[label_of_peak[inside]].astype(np.float32)
        results.append({
            "points": ref["points"],
            "scores": ref["scores"],
            "area": area,
            "sums": sums,
        })
    return results
