#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Host-side reference for the stock YOLO26n ONNX: same letterbox, same decode,
same NMS as the board-side C++ (rknn_yolo26n_smoke.cpp).

Why this exists
---------------
The board tool decodes an int8 tensor by hand: it dequantizes with the
runtime's zp/scale, assumes the 4 box channels are (cx, cy, w, h) scaled by
the input size, and runs a per-class NMS. Every one of those assumptions is a
place where a silent mistake looks like "the model just is not confident on
infrared frames". This script recomputes the same numbers from the ONNX with
onnxruntime, so the two can be compared box-for-box.

The letterbox here is written to match the C++ byte-for-byte in intent:
r = min(size/sh, size/sw), NO cap at 1.0, pad 114, half-pixel centres.

Usage:
    python yolo26n_host_ref.py --onnx yolo26n.onnx --image bus.jpg --th 0.25
    python yolo26n_host_ref.py ... --compare board.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

SIZE = 640
NUM_CLASSES = 80
# 8400 = 6400 (P3/8) + 1600 (P4/16) + 400 (P5/32), P3 first.
NUM_ANCHORS = 80 * 80 + 40 * 40 + 20 * 20


def letterbox(img: np.ndarray, size: int = SIZE, pad: int = 114) -> tuple[np.ndarray, float, int, int]:
    """Letterbox to a square canvas. Returns (canvas HWC uint8, r, left, top)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    left = int(round(((size - nw) / 2) - 0.1))
    top = int(round(((size - nh) / 2) - 0.1))
    canvas = np.full((size, size, 3), pad, dtype=np.uint8)
    resized = Image.fromarray(img).resize((nw, nh), Image.BILINEAR)
    canvas[top : top + nh, left : left + nw] = np.asarray(resized)
    return canvas, r, left, top


def decode(out: np.ndarray, th: float, iou_thr: float) -> list[tuple[int, float, list[float]]]:
    """out: (84, 8400) float. Box channels are (cx, cy, w, h) in canvas pixels."""
    boxes = out[:4] * SIZE  # (4, 8400)
    scores = out[4:]
    best_cls = scores.argmax(axis=0)
    best_score = scores.max(axis=0)
    keep = best_score >= th
    cands = []
    for i in np.nonzero(keep)[0]:
        cx, cy, w, h = boxes[:, i]
        cands.append(
            [float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2), int(best_cls[i]), float(best_score[i])]
        )
    # per-class NMS
    kept: list[list[float]] = []
    for c in range(NUM_CLASSES):
        cls = [d for d in cands if d[4] == c]
        cls.sort(key=lambda d: -d[5])
        ck: list[list[float]] = []
        for d in cls:
            if all(_iou(d, k) <= iou_thr for k in ck):
                ck.append(d)
        kept.extend(ck)
    kept.sort(key=lambda d: -d[5])
    return [(int(d[4]), float(d[5]), [d[0], d[1], d[2], d[3]]) for d in kept]


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    w, h = ix2 - ix1, iy2 - iy1
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--th", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--compare", default="", help="board TSV to compare against")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    img = np.asarray(Image.open(args.image).convert("RGB"))
    canvas, r, left, top = letterbox(img)
    print(f"[image ] {args.image} {img.shape[1]}x{img.shape[0]} -> 640x640 r={r:.6f} pad l={left} t={top}")

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    x = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    out = sess.run(None, {inp.name: x})[0]
    out = out[0]  # (84, 8400)
    print(f"[onnx  ] output shape={out.shape}")

    dets = decode(out, args.th, args.iou)
    print(f"[decode] {len(dets)} detections (th={args.th})")
    for cls, score, box in dets[: args.top]:
        print(f"        cls={cls:<3} score={score:.4f} box=[{box[0]:7.2f} {box[1]:7.2f} {box[2]:7.2f} {box[3]:7.2f}]")

    if args.compare:
        print(f"\n[compare] board TSV: {args.compare}")
        board = []
        for line in Path(args.compare).read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            f = line.split("\t")
            board.append((int(f[1]), float(f[2]), [float(f[3]), float(f[4]), float(f[5]), float(f[6])]))
        print(f"        board: {len(board)} detections, host: {len(dets)} detections")
        worst = 0.0
        for i in range(min(len(board), len(dets))):
            bc, bs, bb = board[i]
            hc, hs, hb = dets[i]
            d = max(abs(bb[k] - hb[k]) for k in range(4))
            worst = max(worst, d)
            flag = "OK " if bc == hc and abs(bs - hs) < 0.02 and d < 5.0 else "DIFF"
            print(f"        {flag} board[{i}] cls={bc} s={bs:.4f} {['%.2f' % v for v in bb]}")
            print(f"             host[{i}] cls={hc} s={hs:.4f} {['%.2f' % v for v in hb]}")
        print(f"        worst box delta = {worst:.3f} px")


if __name__ == "__main__":
    main()
