"""Block-level merge: insert missing `## ` sections from a canonical file into a
mirror file, at the position dictated by canonical order.

Guarantees:
  * every existing byte of the target is preserved verbatim (existing blocks are
    copied byte-for-byte, preamble copied byte-for-byte);
  * only ADDITIONS are made -- the diff contains no '-' lines;
  * the result is idempotent (re-running adds nothing).
"""
import re
import sys

SPLIT = re.compile(r"^(## )", re.M)


def split_blocks(txt):
    """-> (preamble, [(key, block_text), ...])."""
    starts = [m.start() for m in re.finditer(r"^## ", txt, re.M)]
    if not starts:
        return txt, []
    pre = txt[: starts[0]]
    blocks = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(txt)
        blocks.append((txt[s:e].split("\n", 1)[0], txt[s:e]))
    return pre, blocks


def key(title):
    """Normalised section key: strip '## ', leading section number, punctuation."""
    t = title[3:].strip()
    t = re.sub(r"^[§\s]*", "", t)
    t = re.sub(r"^[〇一二三四五六七八九十百\d\.\s]*[·、\.\)]?", "", t)
    t = re.sub(r"^\d+", "", t).strip()
    return t[:40]


def main(canon_path, target_path):
    canon = open(canon_path, encoding="utf-8").read()
    tgt = open(target_path, encoding="utf-8").read()

    cpre, cblocks = split_blocks(canon)
    tpre, tblocks = split_blocks(tgt)

    ckeys = [key(t) for t, _ in cblocks]
    tkeys = [key(t) for t, _ in tblocks]
    tset = set(tkeys)

    added = []
    merged = list(tblocks)
    mkeys = list(tkeys)
    for i, (ctitle, cblock) in enumerate(cblocks):
        ck = ckeys[i]
        if ck in tset:
            continue
        # insert before the first existing block that comes AFTER it in canonical order
        insert_at = len(merged)
        for j in range(i + 1, len(cblocks)):
            nk = ckeys[j]
            if nk in tset:
                insert_at = mkeys.index(nk)
                break
        merged.insert(insert_at, (ctitle, cblock))
        mkeys.insert(insert_at, ck)
        added.append(ctitle)

    if not added:
        print("  [OK] nothing missing; target already consistent")
        return 0

    out = tpre + "".join(b for _, b in merged)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(out)

    # ---- verification -------------------------------------------------
    after = open(target_path, encoding="utf-8").read()
    apre, ablocks = split_blocks(after)
    assert after.startswith(tpre), "preamble changed"
    amap = {key(t): b for t, b in ablocks}
    for t, b in tblocks:  # every original block preserved byte-for-byte
        k = key(t)
        assert amap.get(k) == b, f"original block altered: {t[:50]}"
    assert len(after) >= len(tgt)
    print(f"  [OK] added {len(added)} block(s) to {target_path}:")
    for t in added:
        print(f"        + {t[:88]}")
    print(f"        chars {len(tgt)} -> {len(after)} (+{len(after) - len(tgt)})")
    return 0


if __name__ == "__main__":
    print(f"merge: canonical={sys.argv[1]} target={sys.argv[2]}")
    sys.exit(main(sys.argv[1], sys.argv[2]))
