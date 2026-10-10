#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Merge annotation patches into a flat label root and report exactly what changed.

The vendor ships patches as one nested ``<sequence>_correct_labels/`` directory per sequence, and each
patch only covers the frames it actually touched. ``render_gt_osd_video.py`` instead indexes a *flat*
directory with ``iterdir()`` and drives the render from the label indices it finds there, so pointing
it at the patch root reads zero labels, and pointing it at a per-sequence patch silently drops every
frame the patch did not mention. This script therefore materialises a complete flat root: the base
labels as the floor, the patch files overwritten on top, plus a per-sequence diff summary.

The output is review material, not a new dataset: it is deliberately rebuilt from scratch on every run
so a stale directory can never be mistaken for a fresh merge.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge label patches into a flat label root for GT OSD review")
    parser.add_argument("--base", required=True, help="Flat base label directory holding {sequence}__{index:06d}.txt")
    parser.add_argument("--patch-root", required=True, help="Patch root; may nest one directory per sequence")
    parser.add_argument("--out", required=True, help="Merged flat label directory to create (must not already exist)")
    parser.add_argument(
        "--frames-root", default="", help="Frame root; when given, every label must address a real frame"
    )
    parser.add_argument("--link", action="store_true", help="Hardlink untouched base labels instead of copying them")
    return parser.parse_args()


def parse_label_name(path: Path) -> tuple[str, int] | None:
    """Split ``{sequence}__{index:06d}.txt`` into its sequence and frame index, or None for classes.txt."""
    sequence, separator, digits = path.stem.rpartition("__")
    if not separator or not digits.isdigit():
        return None
    return sequence, int(digits)


def link_or_copy(source: Path, target: Path, use_link: bool) -> None:
    """Hardlink when asked and the filesystem allows it, otherwise fall back to a plain copy."""
    if use_link:
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def frame_indices(sequence_dir: Path) -> set[int]:
    """Map trailing frame number to an index, so nothing relies on lexicographic filename order."""
    indices = set()
    for path in sequence_dir.iterdir():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        digits = path.stem.rpartition("_")[2]
        if digits.isdigit():
            indices.add(int(digits))
    return indices


def contiguous(indexes: list[int]) -> str:
    """Compress sorted frame indices into readable ``start-end`` spans for the report."""
    spans = []
    for index in sorted(indexes):
        if spans and index == spans[-1][1] + 1:
            spans[-1][1] = index
        else:
            spans.append([index, index])
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in spans)


def main() -> None:
    args = parse_args()
    base, patch_root, out = Path(args.base), Path(args.patch_root), Path(args.out)

    for required, role in ((base, "--base"), (patch_root, "--patch-root")):
        if not required.is_dir():
            raise FileNotFoundError(f"{role} is not a directory: {required}")

    base_labels = sorted(p for p in base.iterdir() if p.suffix.lower() == ".txt" and parse_label_name(p))
    if not base_labels:
        raise RuntimeError(f"no {{sequence}}__{{index:06d}}.txt found under {base}; refusing to build an empty merge")

    patch_labels = sorted(p for p in patch_root.rglob("*.txt") if parse_label_name(p))
    if not patch_labels:
        raise RuntimeError(f"no patch label found under {patch_root} (rglob *.txt, {{sequence}}__{{index:06d}}.txt)")

    if out.exists():
        if any(out.iterdir()):
            raise RuntimeError(f"{out} already exists and is not empty; delete it first so no stale merge is reviewed")
        out.rmdir()

    # Floor: the complete base label set, so sequences the patch never touched keep every frame.
    out.mkdir(parents=True)
    for path in base_labels:
        link_or_copy(path, out / path.name, args.link)
    for path in (base / "classes.txt", patch_root / "classes.txt"):
        if path.is_file() and not (out / "classes.txt").is_file():
            link_or_copy(path, out / "classes.txt", args.link)

    # Overlay: every patch file, nested or not, wins over the base file of the same name.
    stats: dict[str, dict[str, list[int]]] = {}
    for path in patch_labels:
        parsed = parse_label_name(path)
        assert parsed is not None  # guaranteed by the filter above
        sequence, index = parsed
        target = out / path.name
        before = target.read_text(encoding="utf-8").strip() if target.is_file() else None
        after = path.read_text(encoding="utf-8").strip()
        if before is not None:
            target.unlink()
        shutil.copy2(path, target)
        bucket = stats.setdefault(sequence, {"filled": [], "adjusted": [], "cleared": [], "added": []})
        if before is None:
            bucket["added"].append(index)
        elif before != after:
            bucket["filled" if not before else ("cleared" if not after else "adjusted")].append(index)

    print(f"[MERGE] base={len(base_labels)} labels -> {out} (patched overlays={len(patch_labels)})")
    print(f"{'sequence':<42} {'empty>filled':>13} {'adjusted':>9} {'empty>removed':>13} {'new':>5}")
    for sequence, bucket in sorted(stats.items()):
        print(
            f"{sequence:<42} {len(bucket['filled']):>13} {len(bucket['adjusted']):>9} "
            f"{len(bucket['cleared']):>13} {len(bucket['added']):>5}"
        )
        for kind in ("filled", "cleared", "adjusted", "added"):
            if bucket[kind]:
                print(f"    {kind:<14} {contiguous(bucket[kind])}")

    # The render walks label indices, so an index with no frame would drop frames silently downstream.
    if args.frames_root:
        frames_root = Path(args.frames_root)
        indexed: dict[str, list[int]] = {}
        for path in out.iterdir():
            parsed = parse_label_name(path)
            if parsed is not None:
                indexed.setdefault(parsed[0], []).append(parsed[1])
        for sequence, indices in sorted(indexed.items()):
            sequence_dir = frames_root / sequence
            if not sequence_dir.is_dir():
                raise RuntimeError(f"no frame directory for sequence {sequence}: {sequence_dir}")
            available = frame_indices(sequence_dir)
            missing = sorted(set(indices) - available)
            if missing:
                raise RuntimeError(
                    f"{sequence}: {len(missing)}/{len(indices)} labels address no frame "
                    f"(first={missing[0]} last={missing[-1]}); label index does not match the frame directory"
                )
            print(f"[FRAMES] {sequence}: labels={len(indices)} frames={len(available)} -> all labels resolve")

    total = sum(1 for p in out.iterdir() if parse_label_name(p))
    print(f"[SUCCESS] merged flat labels={total} in {out}")


if __name__ == "__main__":
    main()
