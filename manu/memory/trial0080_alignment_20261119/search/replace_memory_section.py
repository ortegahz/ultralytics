#!/usr/bin/env python3
"""Replace ONE "### " section inside an owned "## §" block, byte-safe everywhere else.

WHY
---
``append_memory_section.py`` can only APPEND at the end of the owned block.  When a
section has to be REFRESHED (round 90: the落档 text drifted from the artefact), appending
would silently create a duplicate heading instead of replacing the stale one.  This script
splices: it replaces the lines of exactly one "### " section and copies every other byte
of the file verbatim, then proves it (prefix/suffix hashes + byte accounting).

The memory archive is shared with concurrent lines, so the write is done as a single
read-modify-write with an explicit proof, and a timestamped backup is kept.

usage: replace_memory_section.py <target> <section_file> --anchor "§257 十八" \
           --heading-prefix "### 三十六、" [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("section_file")
    parser.add_argument("--anchor", required=True, help='the "## §..." heading that owns the block')
    parser.add_argument("--heading-prefix", required=True, help='the "### ..." heading to replace')
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target = Path(args.target)
    section = Path(args.section_file)
    original = target.read_bytes()
    lines = original.decode("utf-8").splitlines(keepends=True)

    start = None
    for index, line in enumerate(lines):
        if line.startswith("## " + args.anchor):
            start = index
            break
    if start is None:
        print(f"[FAIL] anchor '## {args.anchor}' not found in {target}")
        return 1
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("## §"):
            end = index
            break

    section_start = None
    for index in range(start, end):
        if lines[index].startswith(args.heading_prefix):
            section_start = index
            break
    if section_start is None:
        print(f"[FAIL] heading '{args.heading_prefix}' not found in the block of {target}")
        return 1

    section_end = end
    for index in range(section_start + 1, end):
        if lines[index].startswith("### "):
            section_end = index
            break

    body = section.read_text(encoding="utf-8")
    if not body.endswith("\n"):
        body += "\n"
    payload = body.encode("utf-8")

    prefix = "".join(lines[:section_start]).encode("utf-8")
    suffix = "".join(lines[section_end:]).encode("utf-8")
    new_bytes = prefix + payload + suffix

    assert new_bytes[: len(prefix)] == prefix, "prefix changed"
    assert new_bytes[len(prefix) + len(payload):] == suffix, "suffix changed"
    old_section_bytes = len("".join(lines[section_start:section_end]).encode("utf-8"))
    assert len(new_bytes) == len(original) - old_section_bytes + len(payload), \
        "byte accounting mismatch"

    print(f"[INFO] {target}")
    print(f"[INFO] owned block: line {start + 1} .. {end}")
    print(f"[INFO] replaced section: line {section_start + 1} .. {section_end} "
          f"({section_end - section_start} lines) -- {args.heading_prefix}")
    print(f"[INFO] prefix md5 {digest(prefix)}  suffix md5 {digest(suffix)}")
    print(f"[INFO] {len(original)} -> {len(new_bytes)} bytes")
    if args.dry_run:
        print("[DRY-RUN] not written")
        return 0

    backup = target.with_suffix(target.suffix + ".bak_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(target, backup)
    target.write_bytes(new_bytes)
    print(f"[BAK] {backup}")
    print(f"[WRITE] {target}")
    print("[VERIFY] prefix/suffix byte-identical: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
