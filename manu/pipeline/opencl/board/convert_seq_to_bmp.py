#!/usr/bin/env python3
"""
convert_seq_to_bmp.py -- turn a JPEG sequence into BMP frames on NFS.

Why BMP and not JPEG
--------------------
The board has no usable OpenCV image codec: apt only offers 4.2.0, the cross
sysroot has no zlib/libjpeg/libpng headers at all, and the board's own
libjpeg-turbo is 1.5.2 (.so.62) / 2.0.3 (.so.8) while x86 has 2.1.5. A lossy
decoder difference at the *input* would propagate through the whole pipeline and
make every downstream comparison meaningless.

BMP sidesteps the whole question: OpenCV's BMP codec is built in (an
unconditional file(GLOB ... grfmt*.cpp)) with no external dependency, and the
format is uncompressed, so both machines read byte-identical pixels. This is the
same "ship the decoded pixels" guarantee the raw .gray route gave, but in a
container the pipeline's own imread() can consume -- so gmc_stream_ocl.cpp runs
unmodified.

Grayscale conversion happens here, on x86, with the same OpenCV the reference
runs. Doing it here rather than on the board is what makes the two sides agree:
the board is never asked to reproduce cvtColor.

Frames are written 8-bit single-channel, which keeps each 640x512 frame near
321 KB -- about the same as the raw route -- so the NFS footprint is unchanged.
"""

import argparse
import os
import sys


def die(msg, code=1):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True,
                    help="directory holding 000001.jpg, 000002.jpg, ...")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()

    try:
        import cv2
        import numpy as np
    except ImportError as e:
        die(f"need OpenCV and numpy on the x86 host: {e}\n"
            f"try /home/manu/anaconda3/bin/python")

    names = sorted(f for f in os.listdir(args.seq_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
    if not names:
        die(f"no images in {args.seq_dir}")
    take = names[:args.frames] if args.frames > 0 else names

    print(f"=== convert_seq_to_bmp ===")
    print(f"OpenCV      : {cv2.__version__}")
    print(f"source      : {args.seq_dir}  ({len(names)} images, taking {len(take)})")
    print(f"destination : {args.out_dir}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    ref = None
    written = 0
    for i, nm in enumerate(take):
        src = os.path.join(args.seq_dir, nm)
        bgr = cv2.imread(src, cv2.IMREAD_COLOR)
        if bgr is None:
            die(f"cannot read {src}")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if ref is None:
            ref = gray.shape
        elif gray.shape != ref:
            die(f"{nm} is {gray.shape}, expected {ref} -- geometry must be constant")
        dst = os.path.join(args.out_dir, f"{i:06d}.bmp")
        if not cv2.imwrite(dst, gray):
            die(f"cannot write {dst}")
        written += 1
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  {dst}  {gray.shape[1]}x{gray.shape[0]}")

    # Prove the round trip is lossless on this host before anyone trusts it.
    probe = os.path.join(args.out_dir, "000000.bmp")
    back = cv2.imread(probe, cv2.IMREAD_UNCHANGED)
    orig = cv2.cvtColor(cv2.imread(os.path.join(args.seq_dir, take[0]), cv2.IMREAD_COLOR),
                       cv2.COLOR_BGR2GRAY)
    same = back is not None and np.array_equal(back, orig)
    total = sum(os.path.getsize(os.path.join(args.out_dir, f))
                for f in os.listdir(args.out_dir) if f.endswith(".bmp"))

    print(f"\n[ok] {written} BMP frames, {ref[1]}x{ref[0]} 8UC1, {total/1e6:.1f} MB total")
    print(f"round-trip lossless on this host: {same}")
    if not same:
        die("BMP round trip is NOT lossless -- do not ship these frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
