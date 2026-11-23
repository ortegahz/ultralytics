#!/usr/bin/env python3
"""Regenerate artifacts/memory_md5_manifest.txt from what is on disk right now.

Two different things are recorded, because two different things can change:

  * whole-file md5 of the 6 memory copies -- an INFORMATIONAL snapshot.  The
    concurrent line (ascending-track / video-quality) also appends its own
    sections to these files, so a mismatch here does NOT mean our delivery broke.
  * md5 of the "## §257 十八" block inside each copy -- the only thing this line
    owns.  Nobody else writes it, so this is the GATE: if it drifts, our delivery
    really did change.

usage: python3 regen_manifest.py [--repo <repo root>]
"""
import argparse
import datetime
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEL = HERE.parent
REPO = DEL.parents[2]
FILES = ["memory.md", "memory_compact.md", "memory/memory.md", "memory/memory_compact.md",
         "manu/memory/memory.md", "manu/memory/memory_compact.md"]
HEADING = "## §257 十八"


def md5_bytes(b):
    return hashlib.md5(b).hexdigest()


def md5_text(t):
    return md5_bytes(t.encode("utf-8"))


def section_block(text):
    """Return the '## §257 十八' block (normalised: rstrip + newline), or None."""
    lines = text.splitlines(keepends=True)
    starts = [i for i, l in enumerate(lines) if l.startswith("## §")]
    for n, s in enumerate(starts):
        if lines[s].startswith(HEADING):
            end = starts[n + 1] if n + 1 < len(starts) else len(lines)
            return "".join(lines[s:end]).rstrip("\n") + "\n"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO))
    args = ap.parse_args()
    repo = Path(args.repo)

    full = (HERE / "memory_section_257_18.md")
    compact = (HERE / "memory_section_257_18_compact.md")
    if not full.exists() or not compact.exists():
        raise SystemExit("[FAIL] 缺两份正文（memory_section_257_18.md / _compact.md）")
    full_md5 = md5_text(full.read_text(encoding="utf-8"))
    compact_md5 = md5_text(compact.read_text(encoding="utf-8"))

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# memory md5 manifest -- generated {ts} (python3 artifacts/regen_manifest.py)",
        "#",
        "# whole-file md5 of the 6 memory copies -- INFORMATIONAL: the concurrent line",
        "# appends its own sections to these files, so a difference here is expected",
        "# and does not mean our delivery broke.",
        "# usage: md5sum -c <(grep -v '^#' this_file)  (from the repo root)",
        "# NOTE: regenerate after ANY memory edit; HOST_BACKFILL.sh / HOST_PUSH.sh verify against this file.",
    ]
    for rel in FILES:
        p = repo / rel
        lines.append(f"{md5_bytes(p.read_bytes()) if p.exists() else 'MISSING':32}  {rel}")

    lines += [
        "#",
        "# --- §257 十八 block md5 inside each copy -- THE GATE (this line owns it;",
        "#     the concurrent line's §258/§259/§260 never touch it) ---",
    ]
    for rel in FILES:
        p = repo / rel
        blk = section_block(p.read_text(encoding="utf-8")) if p.exists() else None
        want = full_md5 if rel.endswith("memory.md") and "compact" not in rel else compact_md5
        got = md5_text(blk) if blk else "MISSING"
        lines.append(f"# sec  {rel:32} = {got}  ({'ok' if got == want else 'DRIFT'})")

    lines += [
        "# --- §257 十八 section bodies (the two canonical sources of that section) ---",
        f"# full    md5(body) = {full_md5}   memory_section_257_18.md",
        f"# compact md5(body) = {compact_md5}   memory_section_257_18_compact.md",
    ]
    out = HERE / "memory_md5_manifest.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
