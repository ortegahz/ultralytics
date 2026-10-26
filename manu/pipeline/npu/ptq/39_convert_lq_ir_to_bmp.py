#!/usr/bin/env python3
# ===========================================================================
# convert_lq_ir_to_bmp.py -- stage the Longquan raw infrared sequence as
# 8-bit grayscale BMPs, so the BOARD can read it.
#
# WHY BMP AND NOT THE ORIGINAL JPG
# ---------------------------------
# The board's cross-built OpenCV is configured WITH_JPEG=OFF / WITH_PNG=OFF
# (manu/pipeline/opencl/board/build_opencv_arm64.sh:107-110), so cv::imread
# on the board can only read BMP.  That is also why every existing board-side
# pipeline reads BMP features.
#
# WHY THE PIXELS ARE THE RIGHT ONES
# ----------------------------------
# The frozen official feature dataset was built by
# manu/data/build_full_median_dataset.py with
#     cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# on exactly these JPGs.  This script calls the SAME function on the SAME
# files and writes the result as an 8-bit grayscale BMP, so the grayscale
# values on the board are bit-identical to the ones the official builder
# saw -- no decoder difference can leak into the comparison.
#
# Usage:
#   python convert_lq_ir_to_bmp.py --src <frames_ir_jpg> --dst <out_dir> \
#       --seq VIDEO00007_19700101_001324
# ===========================================================================
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2


def convert_one(job):
    src, dst = job
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (src, "READ_FAIL")
    if not cv2.imwrite(dst, img):
        return (src, "WRITE_FAIL")
    return (src, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory holding frame_%%06d.jpg")
    ap.add_argument("--dst", required=True, help="output directory")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for p in sorted(src_dir.glob("frame_*.jpg")):
        jobs.append((str(p), str(dst_dir / (p.stem + ".bmp"))))
    print(f"[convert] {len(jobs)} frames -> {dst_dir}", flush=True)

    bad = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for src, err in ex.map(convert_one, jobs, chunksize=64):
            done += 1
            if err:
                bad.append((src, err))
            if done % 2000 == 0:
                print(f"[convert] {done}/{len(jobs)}", flush=True)
    print(f"[convert] done {done}, failures {len(bad)}", flush=True)
    for s, e in bad[:20]:
        print(f"[convert] FAIL {e} {s}", flush=True)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
