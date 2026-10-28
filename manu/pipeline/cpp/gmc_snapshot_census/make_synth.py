#!/usr/bin/env python3
"""Synthetic 640x512 8-bit grayscale BMP sequence for the GMC read-set census.

A fixed random texture is panned by (2,1) px per frame, which gives LK real
corners to track while keeping the motion deterministic.  Pure stdlib: the
machine has no numpy/cv2.
"""
import os
import random
import struct
import sys

W, H = 640, 512
TW, TH = W + 64, H + 64
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "seq")

rng = random.Random(20261028)
tex = bytearray(rng.randbytes(TW * TH))
# low-frequency structure so the corners are not pure white noise
for y in range(0, TH, 16):
    for x in range(0, TW, 16):
        v = (x * 7 + y * 13) % 256
        for dy in range(16):
            row = (y + dy) * TW + x
            for dx in range(16):
                tex[row + dx] = (tex[row + dx] + v) // 2

os.makedirs(OUT, exist_ok=True)

# 8-bit BMP, bottom-up, with a 256-entry grey palette.
pix_off = 54 + 1024
row_bytes = W  # already a multiple of 4
file_size = pix_off + row_bytes * H
hdr = b"BM" + struct.pack("<IHHI", file_size, 0, 0, pix_off)
hdr += struct.pack("<IiiHHIIiiII", 40, W, H, 1, 8, 0, row_bytes * H, 2835, 2835, 256, 0)
pal = b"".join(struct.pack("<BBBB", i, i, i, 0) for i in range(256))

for k in range(N):
    ox, oy = (k * 2) % 64, (k * 1) % 64
    body = bytearray()
    for y in range(H - 1, -1, -1):  # BMP rows are bottom-up
        sy = y + oy
        base = sy * TW + ox
        body += tex[base:base + W]
    with open(os.path.join(OUT, "f%04d.bmp" % k), "wb") as f:
        f.write(hdr + pal + bytes(body))
print("wrote %d frames to %s" % (N, OUT))
