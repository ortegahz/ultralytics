#!/usr/bin/env python3
"""
hm_make_video.py -- encode a frame range of an Anti-UAV sequence into a
LOSSLESS video, so the ``--video`` demo path can be reconciled against GT
without the codec becoming a confounder.

Why lossless matters here
-------------------------
The accuracy comparison is "board int8 NPU" vs "CPU authority arm". If the
video were H.264/MP4V, the decoded pixels would differ from the source frames
(max|diff| measured at 23 for mp4v, 18 for MJPG on real IR frames), and any
detection difference could be blamed on the codec. FFV1 in AVI round-trips this
data at **max|diff| = 0**, verified by ``--verify`` — so both arms provably see
the same pixels and the only remaining difference is the quantisation.

Frame numbering
---------------
The raw sequence is ``000001.png .. 000538.png``; the dataset ids are the same
numbers, and GT exists from id 3 onward (ids 1-2 are the cold-start frames with
no emitted feature). Decoded video frame ``i`` (0-based) therefore corresponds to
sequence frame ``--start + i``, which is what ``--id-base`` must be set to when
driving the demo.

Usage:
    python hm_make_video.py --sequence 34_1 --start 3 --frames 300 --out /tmp/c.avi
    python hm_make_video.py --verify /tmp/c.avi          # re-check losslessness
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_RAW = "/home/manu/mnt/data/siping/datasets/manu/anti-uav/train"


def natural_key(p: str):
    return int("".join(c for c in Path(p).stem if c.isdigit()) or 0)


def read_sequence_frames(raw_root: str, seq: str, start: int, count: int):
    d = Path(raw_root) / seq
    if not d.is_dir():
        raise SystemExit(f"[ERROR] no such sequence: {d}")
    fs = sorted((p for p in os.listdir(d)
                 if p.lower().endswith((".png", ".jpg", ".bmp", ".tif"))), key=natural_key)
    by_num = {natural_key(p): p for p in fs}
    want = [n for n in range(start, start + count) if n in by_num]
    if not want:
        raise SystemExit(f"[ERROR] no frames at {start}..{start + count - 1} in {d}")
    return [(n, cv2.imread(str(d / by_num[n]), cv2.IMREAD_GRAYSCALE)) for n in want]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=DEFAULT_RAW)
    ap.add_argument("--sequence")
    ap.add_argument("--start", type=int, default=3,
                    help="first sequence frame number to encode (3 = first id with GT)")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--fourcc", default="FFV1")
    ap.add_argument("--verify", default="",
                    help="verify this video round-trips losslessly instead of encoding")
    args = ap.parse_args()

    if args.verify:
        cap = cv2.VideoCapture(args.verify)
        got = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            got.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        cap.release()
        ref = read_sequence_frames(args.raw_root, args.sequence, args.start, len(got))
        diffs = [int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())
                 for (_, a), b in zip(ref, got)]
        print(f"[verify] {args.verify}: {len(got)} frames decoded, "
              f"{len(ref)} reference frames")
        print(f"[verify] max|diff| = {max(diffs) if diffs else 'n/a'}"
              f"   {'LOSSLESS ✅' if diffs and max(diffs) == 0 else 'LOSSY ❌'}")
        sys.exit(0 if diffs and max(diffs) == 0 else 1)

    if not args.sequence:
        raise SystemExit("[ERROR] --sequence is required when encoding")

    frames = read_sequence_frames(args.raw_root, args.sequence, args.start, args.frames)
    if any(im is None for _, im in frames):
        raise SystemExit("[ERROR] an unreadable frame")
    h, w = frames[0][1].shape[:2]
    print(f"[encode] {args.sequence} frames {frames[0][0]}..{frames[-1][0]} "
          f"({len(frames)} frames, {w}x{h}) -> {args.out}")

    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*args.fourcc),
                         args.fps, (w, h), False)
    if not vw.isOpened():
        raise SystemExit(f"[ERROR] cannot open writer: {args.fourcc} {args.out}")
    for _, im in frames:
        vw.write(im)
    vw.release()
    print(f"[encode] wrote {os.path.getsize(args.out) / 1e6:.2f} MB, codec {args.fourcc}")

    # verify immediately: a silent codec fallback would poison every later number
    cap = cv2.VideoCapture(args.out)
    got = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        got.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(got) != len(frames):
        raise SystemExit(f"[FATAL] decoded {len(got)} of {len(frames)} frames")
    md = max(int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())
             for (_, a), b in zip(frames, got))
    print(f"[verify] round-trip max|diff| = {md}")
    if md != 0:
        print(f"[FATAL] codec {args.fourcc} is NOT lossless on this data; "
              f"an accuracy alignment through it would be confounded.", file=sys.stderr)
        sys.exit(1)
    print(f"[ok] lossless ✅  id_base = {args.start} "
          f"(decoded frame i <-> sequence id {args.start} + i)")


if __name__ == "__main__":
    main()
