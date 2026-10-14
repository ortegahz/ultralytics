#!/usr/bin/env python3
"""Stage raw Anti-UAV frames as 8-bit gray BMP for the board's GMC stage.

WHY THIS EXISTS
---------------
The board's OpenCV build only carries the built-in BMP codec, so the board cannot
read the archived .jpg / .png frames directly. This converts them -- and nothing
else. It is a FORMAT CONVERSION, not a feature regeneration: every feature is
still computed on the board by gmc_stream_ocl.

LOSSY-NESS IS NOT A LOSS
------------------------
gmc_stream_ocl reads its input with cv::IMREAD_GRAYSCALE (gmc_stream_ocl.cpp:2458),
i.e. the grayscale conversion happens INSIDE GMC. So writing a decoded 8-bit gray
BMP means the board sees exactly the same gray matrix the CPU/frozen pipeline saw
when it read the same .jpg/.png. PNG sources are bit-exact through this; JPEG
sources inherit the JPEG's own loss, identically on both arms.

NAMING / ID MAP
---------------
Board outputs are named by PUSH INDEX (0-based, in sorted read order), while the
frozen dataset and the labels are keyed by the frame id in the filename
(000001, 000002, ...). Raw frames are numbered from 1, so:

    push_index = raw_id - 1          (PUSH_OFFSET = 1)

That is an ASSUMPTION inherited from the single-sequence run; hm_val_verify.py
re-measures it empirically per sequence by correlating the board feature against
several candidate frozen ids, so a wrong offset cannot pass silently.

All raw frames are staged, not only the ids that have a frozen counterpart: the
frozen set drops the first two and the last frame of every sequence, but GMC
itself should still see the true sequence start rather than an artificial cut.

Usage
-----
    hm_stage_val_bmp.py --frames-list L.txt --raw-root RAW --out-root NFS/demo
    hm_stage_val_bmp.py ... --sequence 34_1          # one sequence only
    hm_stage_val_bmp.py ... --dry-run
"""
import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

_NUM = re.compile(r"(\d+)")


def seq_of(path):
    """<root>/<SEQ>__<id><ext>  ->  <SEQ>"""
    return os.path.basename(path).split("__", 1)[0]


def raw_id_of(name):
    """000123.png -> 123 ; returns None when the name carries no number."""
    m = _NUM.search(name)
    return int(m.group(1)) if m else None


def find_raw(raw_root, seq, rid):
    """Locate the raw frame <raw_root>/<seq>/<anything-with-rid>.<ext>."""
    d = os.path.join(raw_root, seq)
    want = f"{rid:06d}"
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        p = os.path.join(d, want + ext)
        if os.path.isfile(p):
            return p
    # fall back to a scan when the zero-padding differs
    try:
        for f in os.listdir(d):
            if raw_id_of(f) == rid and "." in f:
                p = os.path.join(d, f)
                if os.path.isfile(p):
                    return p
    except OSError:
        pass
    return None


def stage_one(job):
    """Convert one raw frame to gray BMP. Returns (push, status, detail)."""
    src, dst = job
    push = os.path.basename(dst)
    if os.path.isfile(dst) and os.path.getsize(dst) > 1078:
        return (push, "skip", os.path.getsize(dst))
    im = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return (push, "unreadable", src)
    if not cv2.imwrite(dst, im):
        return (push, "writefail", dst)
    return (push, "new", im.shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-list", required=True,
                    help="txt, one frozen-feature path per line; defines seq + id set")
    ap.add_argument("--raw-root", required=True, help="anti-uav/train root")
    ap.add_argument("--out-root", required=True,
                    help="NFS demo root; BMPs land in <out-root>/<seq>/raw")
    ap.add_argument("--sequence", action="append", default=[],
                    help="restrict to these sequences (repeatable)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # group the frozen ids per sequence
    ids = {}
    with open(args.frames_list) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            b = os.path.basename(line)
            if "__" not in b:
                continue
            s, rest = b.split("__", 1)
            rid = raw_id_of(rest)
            if rid is not None:
                ids.setdefault(s, set()).add(rid)

    want = set(args.sequence) or set(ids)
    seqs = sorted(want & set(ids))
    missing = sorted(want - set(ids))
    if missing:
        print(f"[ERROR] 这些序列不在帧清单里: {missing}", file=sys.stderr)
        return 2

    print(f"[plan] {len(seqs)} 序列, 清单内共 {sum(len(ids[s]) for s in seqs)} 帧")

    total_new = total_skip = 0
    problems = []
    for s in seqs:
        d = os.path.join(args.raw_root, s)
        if not os.path.isdir(d):
            problems.append(f"{s}: raw dir missing {d}")
            continue
        # Stage EVERY raw frame of the sequence, not just the frozen-id subset:
        # GMC must see the true sequence start. push = raw_id - 1.
        raws = {}
        for f in os.listdir(d):
            if "." not in f:
                continue          # e.g. IR_label.json sidecars carry no frame number
            if raw_id_of(f) is None:
                continue
            rid = raw_id_of(f)
            if os.path.isfile(os.path.join(d, f)):
                raws.setdefault(rid, os.path.join(d, f))
        if not raws:
            problems.append(f"{s}: no raw frames under {d}")
            continue

        # GMC resolves its input through find_sequence_folder(), which REQUIRES a
        # directory at <raw-root>/<seq> -- a flat pile of BMPs directly in <raw-root>
        # dies with "[FATAL] sequence folder not found for <seq> under raw" and the
        # exit code is easy to lose behind a shell pipe. So: <out>/<seq>/raw/<seq>/.
        # This matches run_hm_demo.sh's RAWDIR="$NFS_DIR/raw/$SEQ".
        outdir = os.path.join(args.out_root, s, "raw", s)
        jobs = []
        for rid in sorted(raws):
            jobs.append((raws[rid], os.path.join(outdir, f"{rid:06d}.bmp")))

        # every frozen id must have a raw frame, or the whole sequence is unusable
        absent = sorted(ids[s] - set(raws))
        if absent:
            problems.append(f"{s}: 冻结 id {absent[:5]}{'...' if len(absent) > 5 else ''} "
                            f"({len(absent)} 个) 在 raw 里找不到")

        if args.dry_run:
            print(f"[dry] {s:24s} raw={len(raws):5d} -> {outdir}")
            continue
        os.makedirs(outdir, exist_ok=True)

        new = skip = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for push, st, det in ex.map(stage_one, jobs, chunksize=64):
                if st == "new":
                    new += 1
                elif st == "skip":
                    skip += 1
                else:
                    problems.append(f"{s}/{push}: {st} ({det})")
        total_new += new
        total_skip += skip
        print(f"[ok] {s:24s} raw={len(raws):5d} new={new:5d} skip={skip:5d} "
              f"frozen_ids={min(ids[s])}..{max(ids[s])}")

    if args.dry_run:
        return 0

    print(f"[done] new={total_new} skip={total_skip}")
    if problems:
        print(f"[PROBLEMS] {len(problems)} 条:", file=sys.stderr)
        for p in problems[:40]:
            print("  " + p, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())