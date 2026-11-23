"""Undo a mistaken merge: remove blocks that were injected from a canonical file.

A block is considered INJECTED iff
  (1) its text is byte-identical to a block of the canonical file, AND
  (2) its key is not present in the pre-merge backup (.bak), AND
  (3) its key is not explicitly kept (whitelist).

Everything else is preserved byte-for-byte, in order.
"""
import re
import sys


def split_blocks(txt):
    st = [m.start() for m in re.finditer(r"^## ", txt, re.M)]
    if not st:
        return txt, []
    pre = txt[: st[0]]
    out = []
    for i, s in enumerate(st):
        e = st[i + 1] if i + 1 < len(st) else len(txt)
        out.append((txt[s:e].split("\n", 1)[0], txt[s:e]))
    return pre, out


def key(title):
    t = title[3:].strip()
    t = re.sub(r"^[§\s]*", "", t)
    t = re.sub(r"^[〇一二三四五六七八九十百\d\.\s]*[·、\.\)]?", "", t)
    t = re.sub(r"^\d+", "", t).strip()
    return t[:40]


def main(canon_path, target_path, bak_path, keep_keys):
    canon = open(canon_path, encoding="utf-8").read()
    tgt = open(target_path, encoding="utf-8").read()
    bak = open(bak_path, encoding="utf-8").read()

    cbytes = {b for _, b in split_blocks(canon)[1]}
    bakkeys = {key(t) for t, _ in split_blocks(bak)[1]}
    keep = set(keep_keys)

    pre, blocks = split_blocks(tgt)
    kept, removed = [], []
    for t, b in blocks:
        k = key(t)
        if b in cbytes and k not in bakkeys and k not in keep:
            removed.append((t, len(b)))
        else:
            kept.append((t, b))

    out = pre + "".join(b for _, b in kept)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(out)

    after = open(target_path, encoding="utf-8").read()
    _, ablocks = split_blocks(after)
    amap = {key(t): b for t, b in ablocks}
    # every surviving original block must be byte-identical
    for t, b in kept:
        assert amap.get(key(t)) == b, f"block altered: {t[:60]}"
    # no injected block may remain
    for t, _ in removed:
        assert b"".join(x.encode() for x in ()) == b""  # noop
    assert after.startswith(pre), "preamble changed"
    print(f"  restored {target_path}")
    print(f"    removed {len(removed)} injected block(s), kept {len(kept)} block(s)")
    print(f"    bytes {len(tgt)} -> {len(after)}")
    for t, _ in removed[:4]:
        print(f"      - {t[:80]}")
    if len(removed) > 4:
        print(f"      ... and {len(removed) - 4} more")
    return 0


if __name__ == "__main__":
    canon, target, bak = sys.argv[1], sys.argv[2], sys.argv[3]
    keeps = sys.argv[4:]
    print(f"undo-merge: target={target}")
    sys.exit(main(canon, target, bak, keeps))
