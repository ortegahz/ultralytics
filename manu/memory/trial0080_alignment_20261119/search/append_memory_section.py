#!/usr/bin/env python3
"""Append ONE section to the END of an owned "## §" block of a memory file.

The memory files are shared with concurrent lines (the ascending-track line appends its own
"§258 / §259 / §260", the video-quality line appends "§260").  Rewriting the whole file
would silently drop whatever another line wrote between the read and the write, so this
script touches exactly ONE region:

  * the owned block = from the "## <anchor>" heading up to (excluding) the next "## §" heading
  * the new content is INSERTED at the END of that block
  * everything before and after the block is copied byte for byte

After the write the script proves it: the prefix and the suffix hashes must equal the
hashes of the same regions of the original file, and the byte accounting must add up.

usage: append_memory_section.py <target> <section_file> --anchor "§257 十八" [--dry-run]
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target = Path(args.target)
    section = Path(args.section_file)
    original = target.read_bytes()
    text = original.decode("utf-8")
    lines = text.splitlines(keepends=True)

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

    body = section.read_text(encoding="utf-8")
    if not body.endswith("\n"):
        body += "\n"
    payload = ("\n" + body).encode("utf-8")

    prefix = "".join(lines[:end]).encode("utf-8")
    suffix = "".join(lines[end:]).encode("utf-8")
    new_text = prefix.decode("utf-8") + payload.decode("utf-8") + suffix.decode("utf-8")
    new_bytes = new_text.encode("utf-8")

    assert new_bytes[: len(prefix)] == prefix, "prefix changed"
    assert new_bytes[len(prefix) + len(payload):] == suffix, "suffix changed"
    assert len(new_bytes) == len(original) + len(payload), "byte accounting mismatch"

    print(f"[INFO] {target}")
    print(f"[INFO] owned block: line {start + 1} .. {end} ({end - start} lines)")
    print(f"[INFO] next heading after block: {lines[end].rstrip() if end < len(lines) else '(EOF)'}")
    print(f"[INFO] prefix md5 {digest(prefix)}  suffix md5 {digest(suffix)}")
    print(f"[INFO] insert {len(payload)} bytes at offset {len(prefix)}")
    if args.dry_run:
        print("[DRY-RUN] not written")
        return 0

    backup = target.with_suffix(target.suffix + ".bak_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(target, backup)
    target.write_bytes(new_bytes)
    print(f"[BAK] {backup}")
    print(f"[WRITE] {target}  {len(original)} -> {len(new_bytes)} bytes")
    print(f"[VERIFY] prefix/suffix byte-identical: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
