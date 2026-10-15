#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Losslessly transcode a YOLO feature-image directory from JPEG to PNG.

Why this exists
---------------
The Trial 0474 baseline is defined on `uav_gmc_median/images/val/*.jpg`. A 3-channel
JPEG stores chroma at 4:2:0, so whenever Ch1/Ch2 change the decoded Ch0 changes with
them (`rules.md` §三·8). That makes the on-disk pixels a poor reference when comparing
against the embedded pipeline, which keeps features as raw uint8 buffers with no JPEG
in the loop.

This script does NOT rebuild features from raw frames -- it re-encodes the exact array
returned by the JPEG decoder into PNG. So PNG pixel == JPG decode pixel, byte for byte;
the only thing that changes is the storage layer. That isolation is the whole point, and
it is enforced by the per-file gate below rather than assumed.

Usage
-----
    python manu/data/jpg_features_to_png.py \
        --src-images /mnt/data/siping/datasets/manu/uav_gmc_median/images/val \
        --dst-root  /mnt/data/siping/datasets/manu/uav_gmc_median_png \
        --split val --workers 16
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

from ultralytics.utils import colorstr

PNG_COMPRESSION = 6
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def count_images(directory: Path) -> int:
    """Count files the converter is expected to pick up, ignoring sidecar metadata.

    Anti-UAV sequence dirs carry an `IR_label.json` next to the frames; counting every
    entry would make the completeness gate fire on legitimate non-image files.
    """
    return sum(1 for e in os.scandir(directory) if e.is_file() and Path(e.name).suffix.lower() in IMAGE_SUFFIXES)


def parse_args():
    p = argparse.ArgumentParser(description="Losslessly transcode YOLO feature images JPEG -> PNG")
    p.add_argument("--src-images", type=str, default="", help="Flat mode: directory holding the source images")
    p.add_argument(
        "--dst-root",
        type=str,
        required=True,
        help="Dataset root to create: images/<split>, labels/<split> (symlink) and data.yaml",
    )
    p.add_argument(
        "--src-labels",
        type=str,
        default="",
        help="Labels dir to symlink in (default: <src-images>/../../labels/<split>)",
    )
    p.add_argument("--src-yaml", type=str, default="", help="data.yaml template (default: sibling of src-images)")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument(
        "--seq-root",
        type=str,
        default="",
        help="Sequence mode: each subdirectory of this root is one sequence, laid out as <dst>/<seq>/images/",
    )
    p.add_argument("--seq-list", type=str, default="", help="File with one sequence name per line (sequence mode)")
    return p.parse_args()


def convert_one(job: tuple[str, str]) -> tuple[str, int, str]:
    """Transcode a single file and prove the PNG decodes back to the identical array.

    Sources may already be PNG: the official Anti-UAV val split is mixed (27,957 .jpg
    + 3,656 .png). Those are hardlinked instead of re-encoded -- they are already
    lossless and a link is pixel-identical by construction, which avoids running the
    encoder over another ~0.9 GB for no gain.
    """
    src, dst = job
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if img is None:
        return Path(src).name, 0, "unreadable source"
    if src.lower().endswith(".png"):
        dst_path = Path(dst)
        if dst_path.exists() or dst_path.is_symlink():
            dst_path.unlink()
        try:
            os.link(src, dst)
        except OSError:
            os.symlink(Path(src).resolve(), dst)
    elif not cv2.imwrite(dst, img, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]):
        return Path(src).name, 0, "write failed"
    back = cv2.imread(dst, cv2.IMREAD_UNCHANGED)
    if back is None or back.shape != img.shape or back.dtype != img.dtype:
        got = "None" if back is None else f"{back.shape}/{back.dtype}"
        return Path(src).name, 0, f"roundtrip shape/dtype mismatch {got} vs {img.shape}/{img.dtype}"
    if not np.array_equal(back, img):
        return Path(src).name, 0, "roundtrip pixel mismatch"
    return Path(src).name, 1, ""


def run_sequence_mode(args, dst_root: Path):
    """Transcode every sequence dir under --seq-root into <dst>/<seq>/images/*.png.

    Used to stage raw IR frames losslessly before recomputing GMC+median features. Frame
    selection is deliberately NOT applied here: the feature builder needs the leading
    frames for its history window, so the caller subsets afterwards.
    """
    seq_root = Path(args.seq_root)
    if not seq_root.is_dir():
        sys.exit(f"[ERROR] seq-root not found: {seq_root}")

    wanted = None
    if args.seq_list:
        wanted = {ln.strip() for ln in Path(args.seq_list).read_text(encoding="utf-8").splitlines() if ln.strip()}

    seqs = sorted(d for d in seq_root.iterdir() if d.is_dir())
    if wanted is not None:
        missing = wanted - {d.name for d in seqs}
        if missing:
            sys.exit(f"[ERROR] seq-list names absent from {seq_root}: {sorted(missing)[:5]}")
        seqs = [d for d in seqs if d.name in wanted]

    jobs = []
    for sd in seqs:
        out_dir = dst_root / sd.name / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(list(sd.glob("*.jpg")) + list(sd.glob("*.png")))
        n_entries = count_images(sd)
        if len(files) != n_entries:
            sys.exit(f"[ERROR] {sd.name}: matched {len(files)} of {n_entries} files; refusing a partial sequence")
        jobs += [(str(f), str(out_dir / (f.stem + ".png"))) for f in files]

    print(f"[INFO] sequence mode: {len(seqs)} sequences, {len(jobs)} frames -> {dst_root}/<seq>/images/")
    run_jobs(jobs, args.workers, dst_root)


def run_jobs(jobs, workers: int, dst_root: Path):
    t0 = time.time()
    ok = 0
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(convert_one, j) for j in jobs]
        done = 0
        for fut in as_completed(futs):
            name, good, err = fut.result()
            ok += good
            if err:
                failures.append(f"{name}: {err}")
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(jobs)} ({done / (time.time() - t0):.0f} img/s)", flush=True)
    print(f"[INFO] transcoded {ok}/{len(jobs)} in {time.time() - t0:.1f}s")
    if failures:
        print(colorstr("red", f"[ERROR] {len(failures)} files failed the lossless roundtrip gate:"))
        for f in failures[:20]:
            print(f"    {f}")
        sys.exit(1)
    print(colorstr("green", f"[SUCCESS] lossless on all {ok} files under {dst_root}"))


def main():
    args = parse_args()
    dst_root = Path(args.dst_root)

    if args.seq_root:
        run_sequence_mode(args, dst_root)
        return
    if not args.src_images:
        sys.exit("[ERROR] need --src-images (flat mode) or --seq-root (sequence mode)")

    src_images = Path(args.src_images)
    if not src_images.is_dir():
        sys.exit(f"[ERROR] source images dir not found: {src_images}")

    srcs = sorted(list(src_images.glob("*.jpg")) + list(src_images.glob("*.png")))
    if not srcs:
        sys.exit(f"[ERROR] no images under {src_images}")
    # GATE: never let a glob quietly cover a subset. An earlier revision globbed only *.jpg
    # and reported SUCCESS on 27,957 of 31,613 files because 4 sequences ship as PNG --
    # a "lossless" dataset built from a partial match is worse than no dataset.
    n_entries = count_images(src_images)
    if len(srcs) != n_entries:
        sys.exit(f"[ERROR] matched {len(srcs)} images but {src_images} holds {n_entries} files; refusing a partial set")
    n_png_src = sum(1 for s in srcs if s.suffix.lower() == ".png")
    print(f"[INFO] {len(srcs)} images ({len(srcs) - n_png_src} jpg to re-encode, {n_png_src} png to hardlink)")

    dst_images = dst_root / "images" / args.split
    dst_images.mkdir(parents=True, exist_ok=True)

    jobs = [(str(s), str(dst_images / (s.stem + ".png"))) for s in srcs]

    # GATE: a "lossless" claim that has failures is not a lossless claim. Fail loudly and
    # leave the partial output in place for inspection rather than deleting evidence.
    run_jobs(jobs, args.workers, dst_root)

    # Labels: symlink so the dataset stays a valid YOLO tree without duplicating GT.
    src_labels = Path(args.src_labels) if args.src_labels else src_images.parent.parent / "labels" / args.split
    dst_labels = dst_root / "labels" / args.split
    dst_labels.parent.mkdir(parents=True, exist_ok=True)
    if dst_labels.is_symlink() or dst_labels.exists():
        dst_labels.unlink() if dst_labels.is_symlink() else shutil.rmtree(dst_labels)
    dst_labels.symlink_to(src_labels)
    print(f"[INFO] labels -> {src_labels}")

    src_yaml = Path(args.src_yaml) if args.src_yaml else src_images.parent.parent / "data.yaml"
    dst_yaml = dst_root / "data.yaml"
    text = src_yaml.read_text(encoding="utf-8")
    dst_yaml.write_text(text.replace(str(src_yaml.parent), str(dst_root)), encoding="utf-8")
    print(f"[INFO] data.yaml -> {dst_yaml}")

    size_gb = sum(f.stat().st_size for f in dst_images.glob("*.png")) / 1e9
    print(colorstr("green", f"[SUCCESS] PNG dataset at {dst_root} ({size_gb:.2f} GB, lossless verified)"))


if __name__ == "__main__":
    main()