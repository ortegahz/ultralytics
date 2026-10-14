#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Shared host-side utilities for the Trial-0474 heatmap detector on RK3588 NPU.

Everything in this module is deliberately *one* definition of each rule, used by
three consumers that must agree bit-for-bit:

  1. hm_reference.py   -- CPU/PyTorch reference detections
  2. rknn_heatmap_smoke.cpp -- the board binary must reproduce `letterbox()` and
     `extract_peaks()` exactly, so both are mirrored 1:1 there
  3. hm_osd.py         -- rendering

Contract of Trial 0474 (verified against the checkpoint, not guessed):
  weights      runs/optuna_p0_nas/trial_0474/weights/best.pt
  metrics      f1 0.9064 / recall 86.19% / precision 95.57% / tp 21643 / fp 1004
               (matches the frozen single-frame baseline in memory)
  stride       2          -> 640x640 input yields 320x320 heatmap
  imgsz        640
  scale        'n'        -> 523/523 state-dict tensors match exactly ('s' matches 107)
  p0_kwargs    read from the checkpoint, never hard-coded here
  input        (1, 3, 640, 640) float32, RGB, value = uint8/255
               channel 0 = I_t (raw IR grayscale)
               channel 1 = |I_t - W(I_{t-2})|        GMC short difference
               channel 2 = (I_t - B_t)^+             21-frame temporal median residual
  outputs      heatmap (1, 320, 320) sigmoid clamped to [1e-5, 1-1e-5]
               offset  (2, 320, 320) raw, sub-pixel, in stride-cell units

The dataset that produced the calibration/eval tensors is
`uav_gmc_median`: images are already-built 3-channel FEATURE JPEGs (not raw IR),
640x512 (a few 512x512), and the labels live in the 640x640 letterbox space.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMGSZ = 640
PAD_VALUE = 114  # ultralytics LetterBox constant


# ---------------------------------------------------------------------------
# Preprocessing -- mirrors ultralytics/data/augment.py::LetterBox(val mode)
# ---------------------------------------------------------------------------
def letterbox(img: np.ndarray, size: int = IMGSZ, pad_value: int = PAD_VALUE) -> np.ndarray:
    """Letterbox an HxWx3 uint8 RGB image into size x size, padding with 114.

    Reproduces what the ultralytics val path ACTUALLY does, which is not the
    same as `LetterBox(scaleup=False)` alone:

        ultralytics/data/base.py::load_image(index, rect_mode=True)  (the default)
        -> `rect_resize`: the LONG side is scaled to `imgsz` FIRST
        -> then LetterBox(scaleup=False) runs on that already-resized image

    For a 640x512 source the long side is already 640, so the ratio is 1.0 and
    LetterBox pads 64 px top and bottom -- which is why the obvious
    implementation looks right. For a SQUARE 512x512 source the long side is
    512, so the loader first UPSCALES by 1.25 to 640x640 and LetterBox then adds
    no padding at all. Measured against the real dataloader on
    `25_1__000003.png` .. `000007.png`: `cv2.resize(src, 640, 640)` reproduces the
    loader tensor with max abs diff 0.000e+00, while capping the ratio at 1.0 and
    padding differs by up to 0.447 per pixel. 4682 of the 31613 val frames (14.8%)
    are square, so getting this wrong silently corrupts every one of them.

    Do not "fix" this by re-adding a `min(r, 1.0)` cap.
    """
    import cv2

    h, w = img.shape[:2]
    r = min(size / h, size / w)  # NOTE: no cap -- the loader upscales too
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (size - new_unpad[0]) / 2
    dh = (size - new_unpad[1]) / 2
    if new_unpad != (w, h):
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3)
    return img


def load_feature_jpg(path: str | Path) -> np.ndarray:
    """Load a pre-built feature JPEG as HxWx3 uint8 RGB."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_input_tensor(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 RGB -> (1, 3, 640, 640) float32 with values in [0, 1]."""
    a = img_rgb_uint8.astype(np.float32) / 255.0
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None])


def letterbox_to_input(path: str | Path, size: int = IMGSZ) -> np.ndarray:
    return to_input_tensor(letterbox(load_feature_jpg(path), size))


# ---------------------------------------------------------------------------
# Letterbox geometry -- needed to map detections back onto the source image
# ---------------------------------------------------------------------------
def letterbox_geometry(src_hw: tuple[int, int], size: int = IMGSZ) -> dict:
    """Reproduce LetterBox geometry so 640x640 detections can be un-padded.

    Returns the affine mapping (scale, pad_left, pad_top) from source-image
    pixel coordinates to the 640x640 canvas, plus the inverse.
    """
    h, w = src_hw
    r = min(size / h, size / w)  # same rule as letterbox(): no upscale cap
    dw = (size - int(round(w * r))) / 2
    dh = (size - int(round(h * r))) / 2
    left = int(round(dw - 0.1))
    top = int(round(dh - 0.1))
    return {"scale": r, "pad_left": left, "pad_top": top, "src_hw": (h, w), "size": size}


def canvas_to_src(points_xy: np.ndarray, geom: dict) -> np.ndarray:
    """Map (N, 2) [x, y] points from the letterbox canvas back to source pixels."""
    p = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2).copy()
    p[:, 0] = (p[:, 0] - geom["pad_left"]) / geom["scale"]
    p[:, 1] = (p[:, 1] - geom["pad_top"]) / geom["scale"]
    return p


# ---------------------------------------------------------------------------
# Labels -- GOTCHA, verified against the real ultralytics dataloader
# ---------------------------------------------------------------------------
# The raw YOLO label files are normalized by the SOURCE image dimensions, NOT
# by the 640x640 letterbox canvas. Most sources are 640x512, but 512x512 also
# occurs (150 of 600 sampled val frames), so the two axes must both be read from
# the per-frame source shape -- hard-coding 640 works for x on a 640-wide source
# and is silently wrong for y on anything shorter.
#
# The ultralytics val dataloader proves it: for 01_4485_1167-2666__000003 the
# raw y0 is 0.71582, and the dataloader's canvas-normalized y0*640 is 430.50,
# which is exactly 0.71582*512 + 64 (= 366.50 + 64). So the pipeline is
#     source_norm * src_hw * scale + pad
# and reading the raw file with a fixed `size=640` on both axes silently
# produces plausible-looking numbers that never match the detections. Measured
# cost of getting it wrong: recall 0.196 instead of 0.911 on a 200-frame sample.
#
# (manu/inference/cache_trial0474_inferences.py multiplies by imgsz on both
# axes too, but it reads `batch["bboxes"]` -- already letterbox-transformed by
# the dataloader -- so there the 640 is correct. Two label sources, two rules;
# they must not be mixed.)
def gt_from_label(label_path: str | Path, geom: dict, size: int = IMGSZ) -> tuple[np.ndarray, np.ndarray]:
    """Return (points_xy (N,2), boxes_cxcywh (N,4)) in letterbox-canvas pixels.

    `geom` must come from letterbox_geometry() for the SAME image.
    """
    h, w = geom["src_hw"]
    scale, pl, pt = geom["scale"], geom["pad_left"], geom["pad_top"]
    pts, boxes = [], []
    p = Path(label_path)
    if not p.is_file():
        return np.zeros((0, 2), np.float32), np.zeros((0, 4), np.float32)
    for line in p.read_text().splitlines():
        f = line.split()
        if len(f) < 5:
            continue
        # raw label -> source pixels -> canvas pixels
        cx = float(f[1]) * w * scale + pl
        cy = float(f[2]) * h * scale + pt
        bw = float(f[3]) * w * scale
        bh = float(f[4]) * h * scale
        boxes.append([cx, cy, bw, bh])
        pts.append([cx, cy])
    return (np.asarray(pts, np.float32).reshape(-1, 2),
            np.asarray(boxes, np.float32).reshape(-1, 4))


def split_seq_frame(filename: str) -> tuple[str, int]:
    """`01_4485_1167-2666__000123.jpg` -> ("01_4485_1167-2666", 123)."""
    stem = Path(filename).stem
    if "__" in stem:
        seq, idx = stem.rsplit("__", 1)
        return seq, int(idx)
    return stem, -1


# ---------------------------------------------------------------------------
# Peak extraction -- numpy port of manu/evaluation/heatmap_evaluate.py
# ---------------------------------------------------------------------------
def _maxpool3x3(x: np.ndarray) -> np.ndarray:
    """3x3 max-pool, matching F.max_pool2d(k=3, s=1, p=1).

    PyTorch pads max-pooling with -inf, not 0; for a heatmap clamped to >= 1e-5
    the two agree, but -inf is what the op actually does so it is what we do.
    """
    p = np.pad(x.astype(np.float32), ((1, 1), (1, 1)), mode="constant", constant_values=-np.inf)
    h, w = x.shape
    out = p[0:h, 0:w]
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out = np.maximum(out, p[dy:dy + h, dx:dx + w])
    return out


def extract_peaks_np(
    heatmap: np.ndarray,
    offset: np.ndarray,
    stride: int = 2,
    conf_thresh: float = 0.22,
    top_k: int = 100,
) -> dict[str, np.ndarray]:
    """Port of extract_peaks() for a single (1,H,W) heatmap and (2,H,W) offset.

    Returns {'points': (N,2) float32 [x, y] in canvas pixels,
             'scores': (N,) float32}.

    Ordering matches the torch version exactly: candidate cells are enumerated in
    row-major order (torch.nonzero order), then top-k'd by score. Equal scores
    keep row-major order because torch.topk on CPU is stable for ties in
    practice -- and any residual tie-break difference shows up immediately in the
    host-vs-NPU comparison, which is exactly what that comparison is for.
    """
    hm = np.asarray(heatmap, np.float32).reshape(heatmap.shape[-2], heatmap.shape[-1])
    off = np.asarray(offset, np.float32).reshape(2, *hm.shape)
    hmax = _maxpool3x3(hm)
    mask = (hmax == hm) & (hm >= conf_thresh)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return {"points": np.zeros((0, 2), np.float32), "scores": np.zeros((0,), np.float32)}
    scores = hm[ys, xs]
    if scores.size > top_k:
        keep = np.argsort(-scores, kind="stable")[:top_k]
        keep = keep[np.argsort(keep, kind="stable")]  # restore row-major order
        ys, xs, scores = ys[keep], xs[keep], scores[keep]
    dx = off[0, ys, xs]
    dy = off[1, ys, xs]
    pts = np.stack([(xs.astype(np.float32) + dx) * stride,
                    (ys.astype(np.float32) + dy) * stride], axis=1)
    return {"points": pts.astype(np.float32), "scores": scores.astype(np.float32)}


def match_distance(dets: np.ndarray, gts: np.ndarray, dist_thr: float = 8.0):
    """Greedy nearest-GT matching at <= dist_thr. Returns (n_tp, n_fp, n_fn)."""
    if len(gts) == 0:
        return 0, len(dets), 0
    if len(dets) == 0:
        return 0, 0, len(gts)
    used = np.zeros(len(gts), bool)
    tp = 0
    order = np.argsort(-dets[:, 2])
    for i in order:
        d = np.hypot(gts[:, 0] - dets[i, 0], gts[:, 1] - dets[i, 1])
        j = int(d.argmin())
        if d[j] <= dist_thr and not used[j]:
            used[j] = True
            tp += 1
    return tp, len(dets) - tp, int((~used).sum())
