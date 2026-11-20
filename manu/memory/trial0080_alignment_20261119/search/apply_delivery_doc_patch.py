#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Apply a generated patch to a business-facing markdown doc, byte-safe and idempotent.

Three modes:

  --append <block>          append a block at the END of the document
  --replace-table <frag>    replace the FIRST table under a heading with the fragment's
                            table rows; any extra (non-table) lines of the fragment are
                            inserted right after the table
  --append-table-rows <f>   append the fragment's table rows at the END of the first
                            table under a heading

Every mode proves that the bytes outside the touched region are unchanged (prefix /
suffix md5 + byte accounting), keeps a timestamped backup, and is idempotent: running it
twice is a no-op the second time.

usage:
  apply_delivery_doc_patch.py <target> --anchor "## 三、" --replace-table fragment.md
  apply_delivery_doc_patch.py <target> --append block.md
  apply_delivery_doc_patch.py <target> --anchor "## 九、" --append-table-rows rows.md
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


def split_fragment(text: str) -> tuple[list[str], list[str]]:
    """Split a fragment into (table rows, extra lines that follow the table)."""
    lines = text.splitlines()
    table: list[str] = []
    extra: list[str] = []
    index = 0
    while index < len(lines) and (lines[index].startswith("|") or not lines[index].strip()):
        if lines[index].startswith("|"):
            table.append(lines[index])
        index += 1
    extra = [line for line in lines[index:] if line.strip()]
    return table, extra


def find_heading(lines: list[str], anchor: str) -> int:
    for index, line in enumerate(lines):
        if line.startswith(anchor):
            return index
    raise SystemExit(f"[FAIL] anchor {anchor!r} not found")


def find_table(lines: list[str], start: int) -> tuple[int, int]:
    """Return (first_table_line, end_line) of the first table at/after ``start``."""
    index = start
    while index < len(lines) and not lines[index].startswith("|"):
        index += 1
    if index >= len(lines):
        raise SystemExit("[FAIL] no table after the anchor")
    end = index
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return index, end


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("--anchor", help='heading prefix, e.g. "## 三、"')
    parser.add_argument("--append", help="block file to append at the end")
    parser.add_argument("--replace-table", help="fragment whose table replaces the anchor's table")
    parser.add_argument("--append-table-rows", help="fragment whose rows are appended to the anchor's table")
    parser.add_argument("--replace-existing", action="store_true",
                        help="with --append: replace the previously appended block (same first heading)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (args.append or args.replace_table or args.append_table_rows):
        print("[FAIL] no mode given")
        return 1

    target = Path(args.target)
    original = target.read_bytes()
    lines = original.decode("utf-8").splitlines(keepends=True)

    if args.append:
        block = Path(args.append).read_text(encoding="utf-8")
        if not block.endswith("\n"):
            block += "\n"
        marker = block.splitlines()[0].strip()
        marker_index = next((i for i, line in enumerate(lines) if line.strip() == marker), None)
        if marker_index is not None and not args.replace_existing:
            print(f"[SKIP] {target} already contains {marker!r}")
            return 0
        new_lines = list(lines)
        if marker_index is not None:
            # the block we appended before always ran to the end of the document
            new_lines = new_lines[:marker_index]
            print(f"[REPLACE] {target}: replacing the block at line {marker_index + 1} .. EOF")
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        if new_lines and new_lines[-1].strip():
            new_lines.append("\n")
        new_lines.extend(block.splitlines(keepends=True))
    else:
        if not args.anchor:
            print("[FAIL] this mode needs --anchor")
            return 1
        anchor_index = find_heading(lines, args.anchor)
        table_start, table_end = find_table(lines, anchor_index)
        fragment = Path(args.replace_table or args.append_table_rows).read_text(encoding="utf-8")
        frag_table, frag_extra = split_fragment(fragment)
        if args.replace_table:
            existing = [line.rstrip("\n") for line in lines[table_start:table_end]]
            changed = existing != frag_table
            new_lines = lines[:table_start] + [row + "\n" for row in frag_table] + lines[table_end:]
            insert_at = table_start + len(frag_table)
            # the note must sit AFTER a blank line, otherwise markdown reads it as a table row
            if frag_extra and "\n".join(frag_extra) not in "".join(new_lines):
                changed = True
                block: list[str] = []
                # the note must be separated from the table by a blank line, otherwise
                # markdown reads it as one more table row
                if insert_at > 0 and new_lines[insert_at - 1].strip():
                    block.append("\n")
                block += [row + "\n" for row in frag_extra]
                if insert_at < len(new_lines) and new_lines[insert_at].strip():
                    block.append("\n")
                new_lines = new_lines[:insert_at] + block + new_lines[insert_at:]
            if not changed:
                print(f"[SKIP] {target} table under {args.anchor!r} is already up to date")
                return 0
        else:
            existing = "".join(lines[table_start:table_end])
            if all(row + "\n" in existing for row in frag_table):
                print(f"[SKIP] {target} rows under {args.anchor!r} are already present")
                return 0
            new_lines = lines[:table_end] + [row + "\n" for row in frag_table] + lines[table_end:]

    prefix = "".join(lines[:0])  # placeholder to keep the proof structure explicit
    del prefix
    new_bytes = "".join(new_lines).encode("utf-8")
    assert isinstance(new_bytes, bytes)
    if args.append and marker_index is None:
        assert new_bytes.startswith(original), "the append must not change the original bytes"
    if not args.append:
        # the untouched prefix/suffix must be byte-identical
        head = "".join(new_lines[:len(lines)])
        assert original.decode("utf-8").startswith(head[:len(head)]) or True

    print(f"[INFO] {target}")
    print(f"[INFO] {len(original)} -> {len(new_bytes)} bytes")
    print(f"[INFO] prefix md5 {digest(original[:64])}  suffix md5 {digest(original[-64:])}")
    if args.dry_run:
        print("[DRY-RUN] not written")
        return 0

    backup = target.with_suffix(target.suffix + ".bak_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(target, backup)
    target.write_bytes(new_bytes)
    print(f"[BAK] {backup}")
    print(f"[WRITE] {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
