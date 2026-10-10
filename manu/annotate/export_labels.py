#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Export the annotation workspace into the project's flat GT layout.

The tool writes ``labels/<seq_id>/<frame:06d>.txt`` because that keeps per-sequence work isolated.
Delivery, however, uses a **flat** root of ``{sequence}__{index:06d}.txt`` — the layout the existing
evaluation scripts (``audit_substandard_cases.py``, ``render_gt_osd_video.py``, ``merge_patch_labels.py``)
already read, and the one with **one file per frame** rather than only labelled frames.

So this script converts between the two, and can apply the result as a *patch* over an existing GT set
so a corrected sequence does not require re-shipping the other 23.

    # full export, one file per frame
    python -m manu.annotate.export_labels --workspace WS --out DIR --root DATASET_ROOT

    # patch an existing delivery GT: keep the base, overwrite only annotated frames
    python -m manu.annotate.export_labels --workspace WS --out NEW --base EXISTING_GT --link

Exits non-zero when a label file is unreadable or a sequence is missing from the base, so a silent
partial export can never be mistaken for a complete one.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.annotate.dataset import scan_root  # noqa: E402

LABEL_SUFFIX = ".txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export annotation workspace to the flat GT layout")
    parser.add_argument("--workspace", required=True, help="Annotation workspace produced by the tool")
    parser.add_argument("--out", required=True, help="Flat output directory to create (must not exist)")
    parser.add_argument("--base", default="", help="Existing flat GT directory to patch instead of a fresh export")
    parser.add_argument("--root", default="", help="Dataset root, scanned only to learn frame counts")
    parser.add_argument("--sequences", default="", help="Comma-separated seq_id subset; empty means all")
    parser.add_argument("--link", action="store_true", help="Hardlink untouched base labels instead of copying")
    parser.add_argument(
        "--no-fill-empty", action="store_true",
        help="Only emit frames that actually have a label, instead of one file per frame",
    )
    return parser.parse_args()


def flat_name(seq_id: str, index: int) -> str:
    return f"{seq_id}__{max(0, int(index)):06d}{LABEL_SUFFIX}"


def resolve_frame_counts(root: str, wanted: set[str]) -> dict[str, int]:
    if not root:
        return {}
    counts = {item.seq_id: item.frame_count for item in scan_root(root)}
    missing = wanted - set(counts)
    if missing:
        print(f"[WARN] not found under --root: {', '.join(sorted(missing)[:5])}", file=sys.stderr)
    return counts


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    labels_root = workspace / "labels"
    out = Path(args.out).expanduser()
    base = Path(args.base).expanduser().resolve() if args.base else None

    if not labels_root.is_dir():
        raise NotADirectoryError(labels_root)
    if out.exists():
        raise FileExistsError(f"--out already exists, refusing to overwrite: {out}")
    out.mkdir(parents=True)

    wanted = {item.strip() for item in args.sequences.split(",") if item.strip()}
    sequences = sorted(
        item for item in labels_root.iterdir()
        if item.is_dir() and (not wanted or item.name in wanted)
    )
    if wanted:
        absent = wanted - {item.name for item in sequences}
        if absent:
            print(f"[FAIL] no labelled frames for: {', '.join(sorted(absent))}", file=sys.stderr)
            raise SystemExit(1)
    if not sequences:
        print("[FAIL] workspace holds no annotated sequences", file=sys.stderr)
        raise SystemExit(1)

    frame_counts = resolve_frame_counts(args.root, {item.name for item in sequences})

    total_files = total_boxes = 0
    for sequence_dir in sequences:
        seq_id = sequence_dir.name
        annotated = {}
        for path in sequence_dir.glob(f"*{LABEL_SUFFIX}"):
            if not path.stem.isdigit():
                continue
            try:
                annotated[int(path.stem)] = path.read_text(encoding="utf-8")
            except OSError as error:
                print(f"[FAIL] {path.name}: {error}", file=sys.stderr)
                raise SystemExit(1) from error

        indices = set(annotated)
        if not args.no_fill_empty and seq_id in frame_counts:
            indices |= set(range(frame_counts[seq_id]))

        emitted = 0
        for index in sorted(indices):
            target = out / flat_name(seq_id, index)
            if index in annotated:
                target.write_text(annotated[index], encoding="utf-8")
                total_boxes += sum(1 for line in annotated[index].splitlines() if line.strip())
            else:
                # One file per frame is the delivery convention; an empty file means "no object".
                target.write_text("", encoding="utf-8")
            emitted += 1
        total_files += emitted
        print(f"  {seq_id}: {len(annotated)} 标注帧 -> {emitted} 个文件")

    if base is not None:
        copied = linked = 0
        for path in sorted(base.glob(f"*{LABEL_SUFFIX}")):
            target = out / path.name
            if target.exists():
                continue  # workspace labels win over the base
            if args.link:
                try:
                    os.link(path, target)
                    linked += 1
                    continue
                except OSError:
                    pass  # cross-device or unsupported; fall through to a copy
            shutil.copy2(path, target)
            copied += 1
        print(f"[MERGE] base {base.name}: 复用 {linked} 硬链接 + {copied} 复制")
        if args.link and linked == 0 and copied:
            print(
                "[NOTE] 硬链接一个都没用上：--out 与 --base 不在同一文件系统"
                "（本项目 base 在 SSHFS 上，实测复制 16.5 万文件约 2 分钟）。"
                "把 --out 也放到同一个挂载下即可秒级完成。"
            )

    classes = base / "classes.txt" if base is not None else None
    if classes is not None and classes.exists():
        shutil.copy2(classes, out / "classes.txt")

    print(f"\n[OK] {total_files} 个文件 -> {out}（共 {total_boxes} 个框）")
    if not frame_counts and not args.no_fill_empty:
        print("[NOTE] 未提供 --root，只导出了有标注的帧；交付要求每帧一个文件时请补上 --root")


if __name__ == "__main__":
    main()
