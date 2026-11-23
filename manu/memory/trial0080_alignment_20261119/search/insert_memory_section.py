#!/usr/bin/env python3
"""Insert a new memory section WITHOUT rewriting the file.

The memory archive is shared with another (ascending-track) line that appends its
own sections concurrently.  A whole-file overwrite would silently drop those.
This inserts the new text immediately before the next "## §" heading that follows
the anchor section (ANY "## §" heading -- the inserted block is itself titled
"## §257 十八", so the next heading must not be searched for the anchor again), leaving every other byte untouched, and it is idempotent.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1])
anchor = sys.argv[2]          # e.g. "§257"
text_file = Path(sys.argv[3])
compact = sys.argv[4] if len(sys.argv) > 4 else ""

body = text_file.read_text(encoding="utf-8")
if not body.endswith("\n"):
    body += "\n"

original = path.read_text(encoding="utf-8")
if body.strip() in original:
    print(f"[SKIP] {path} already contains the section")
    raise SystemExit(0)
lines = original.splitlines(keepends=True)

# Locate the anchor section's heading and the next top-level "## §" heading.
anchor_index = None
next_index = len(lines)
for index, line in enumerate(lines):
    if not line.startswith("## §"):
        continue
    if anchor_index is None:
        if anchor in line.split("｜")[0].split("·")[0]:
            anchor_index = index
        continue
    next_index = index
    break
if anchor_index is None:
    raise SystemExit(f"[FAIL] anchor section {anchor} not found in {path}")

insert_at = next_index
# Back up over the blank lines that separate the sections.
while insert_at - 1 > anchor_index and lines[insert_at - 1].strip() == "":
    insert_at -= 1

merged = "".join(lines[:insert_at]) + body + "".join(lines[insert_at:])
if merged == original:
    print(f"[SKIP] {path} already up to date")
    raise SystemExit(0)

backup = path.with_suffix(path.suffix + ".bak_20261119_1820")
backup.write_text(original, encoding="utf-8")
path.write_text(merged, encoding="utf-8")
print(f"[OK] {path}: inserted {len(body.splitlines())} lines at line {insert_at + 1}")
print(f"[OK] backup -> {backup}")

if compact:
    compact_path = Path(compact)
    compact_original = compact_path.read_text(encoding="utf-8")
    if body.strip() not in compact_original:
        compact_backup = compact_path.with_suffix(compact_path.suffix + ".bak_20261119_1820")
        compact_backup.write_text(compact_original, encoding="utf-8")
        # Same anchor rule as the full file: insert right after the anchor section,
        # before the next "## §" heading -- never append past another line's sections.
        compact_lines = compact_original.splitlines(keepends=True)
        c_anchor = None
        c_next = len(compact_lines)
        for index, line in enumerate(compact_lines):
            if not line.startswith("## §"):
                continue
            if c_anchor is None:
                if anchor in line.split("\uff5c")[0].split("\u00b7")[0]:
                    c_anchor = index
                continue
            c_next = index
            break
        if c_anchor is None:
            compact_path.write_text(compact_original.rstrip("\n") + "\n\n" + body, encoding="utf-8")
            print(f"[OK] {compact_path}: appended (anchor {anchor} not found)")
        else:
            c_at = c_next
            while c_at - 1 > c_anchor and compact_lines[c_at - 1].strip() == "":
                c_at -= 1
            compact_path.write_text(
                "".join(compact_lines[:c_at]) + body + "".join(compact_lines[c_at:]),
                encoding="utf-8",
            )
            print(f"[OK] {compact_path}: inserted {len(body.splitlines())} lines at line {c_at + 1}")
        print(f"[OK] backup -> {compact_backup}")
    else:
        print(f"[SKIP] {compact_path} already contains the section")
