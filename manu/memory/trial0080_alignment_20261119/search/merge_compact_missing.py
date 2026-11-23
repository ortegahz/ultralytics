"""Surgically append missing canonical sections into a mirror compact file.

Only appends; never rewrites existing content. Verifies byte-identity of the
untouched prefix before/after.
"""
import sys


def section_span(txt, heading_prefix):
    """Return (start, end) of the `## ` section whose title starts with heading_prefix."""
    import re

    start = None
    for m in re.finditer(r"^## ", txt, re.M):
        if txt[m.start() :].startswith("## " + heading_prefix):
            start = m.start()
            break
    if start is None:
        return None
    nxt = None
    for m in re.finditer(r"^## ", txt, re.M):
        if m.start() > start:
            nxt = m.start()
            break
    return start, (len(txt) if nxt is None else nxt)


def main(canon_path, target_path, prefixes):
    canon = open(canon_path, encoding="utf-8").read()
    tgt = open(target_path, encoding="utf-8").read()
    blocks = []
    for p in prefixes:
        span = section_span(canon, p)
        if span is None:
            print(f"  [SKIP] canonical has no section {p!r}")
            continue
        blocks.append(canon[span[0] : span[1]].rstrip("\n"))
    if not blocks:
        print("  [FAIL] nothing to append")
        return 1
    # target: insert at end of file (these trailing sections are the newest)
    missing = [p for p in prefixes if section_span(tgt, p) is None]
    if not missing:
        print("  [OK] target already has all requested sections; nothing to do")
        return 0
    new = tgt.rstrip("\n") + "\n\n" + "\n\n".join(blocks) + "\n"
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(new)
    # verify: prefix unchanged
    after = open(target_path, encoding="utf-8").read()
    assert after.startswith(tgt.rstrip("\n")), "prefix changed!"
    for p in missing:
        assert section_span(after, p) is not None, f"{p} still missing"
    print(f"  [OK] appended {len(missing)} section(s) {missing}; +{len(after) - len(tgt)} chars")
    return 0


if __name__ == "__main__":
    canon, target = sys.argv[1], sys.argv[2]
    prefixes = sys.argv[3:]
    print(f"merge: canonical={canon} target={target}")
    sys.exit(main(canon, target, prefixes))
