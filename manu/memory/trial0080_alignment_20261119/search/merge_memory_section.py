#!/usr/bin/env python3
"""Merge ONE memory section into a memory file without touching any other byte.

The memory archive is shared with a concurrent line (ascending-track / video-quality) that
appends its own "## §258 / §259 / §260" sections.  Copying this server's whole file onto the
host would silently drop whatever the host has that this copy does not.  So the only block we
ever rewrite is the one this line owns ("## §257 十八"):

  * absent            -> insert right before the next "## §" heading after the "## §257" anchor
  * present, identical-> no-op
  * present, different-> replaced in place (this line owns that block)

Everything outside the block is copied byte-for-byte.  The script reports byte counts and the
neighbouring "## §" headings so the caller can prove the other line's sections survived.

usage: merge_memory_section.py <target> <section_file> [--heading "§257 十八"] [--dry-run]
"""
import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path


def section_blocks(lines):
    """Return [(heading, start, end)] for every "## §" heading; end is exclusive."""
    starts = [i for i, line in enumerate(lines) if line.startswith("## §")]
    blocks = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        blocks.append((lines[start], start, end))
    return blocks


def find_block(blocks, heading):
    for head, start, end in blocks:
        if head.startswith("## " + heading):
            return head, start, end
    return None


def merge(original, body, heading):
    """Return (merged_text, action)."""
    lines = original.splitlines(keepends=True)
    blocks = section_blocks(lines)
    hit = find_block(blocks, heading)

    if hit is not None:
        _, start, end = hit
        old_block = "".join(lines[start:end])
        current = old_block.rstrip("\n")
        if current == body.rstrip("\n"):
            return original, "skip"
        # Keep the block's original trailing blank-line convention so the separator
        # before the next "## §" heading survives the replacement untouched.
        tail = old_block[len(old_block.rstrip("\n")):] or "\n"
        return "".join(lines[:start]) + body.rstrip("\n") + tail + "".join(lines[end:]), "replace"

    anchor = None
    for head, start, _ in blocks:
        if head.startswith("## §257") and not head.startswith("## §257 十八"):
            anchor = start
            break
    if anchor is None:
        return original.rstrip("\n") + "\n\n" + body, "append"

    following = [s for h, s, _ in blocks if s > anchor]
    at = following[0] if following else len(lines)
    while at - 1 > anchor and lines[at - 1].strip() == "":
        at -= 1
    return "".join(lines[:at]) + body.rstrip("\n") + "\n\n" + "".join(lines[at:]), "insert"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("section_file")
    ap.add_argument("--heading", default="§257 十八")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    target = Path(args.target)
    body = Path(args.section_file).read_text(encoding="utf-8")
    if not body.endswith("\n"):
        body += "\n"
    original = target.read_text(encoding="utf-8")

    merged, action = merge(original, body, args.heading)
    changed = merged != original
    print(f"[{action.upper()}] {target}  bytes {len(original.encode())} -> {len(merged.encode())}")
    print(f"[SECTION] md5(body)={hashlib.md5(body.encode()).hexdigest()}")
    after = [h for h, _, _ in section_blocks(merged.splitlines(keepends=True))]
    keep = [h for h in after if h.startswith(("## §258", "## §259", "## §260"))]
    print(f"[KEEP] concurrent-line sections still present: {len(keep)}")
    for h in keep:
        print(f"       {h[:90]}")
    if not changed:
        print("[NO-OP] file already up to date")
        return 0
    if args.dry_run:
        print("[DRY-RUN] not written")
        return 0
    # Timestamp the backup: a fixed name gets overwritten by the next merge, so the
    # audit baseline ("what did this file look like before this merge?") would be lost.
    # Keep only the two most recent copies per target so they stay bounded.
    backup = target.with_suffix(target.suffix + f".bak_pre_push_{time.strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(target, backup)
    for old in sorted(target.parent.glob(target.name + ".bak_pre_push_*"))[:-2]:
        old.unlink()
    target.write_text(merged, encoding="utf-8")
    print(f"[OK] backup -> {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
