#!/usr/bin/env python3
"""
cmp_e2e_outputs.py -- byte-exact comparison of two end-to-end demo outputs.

Compares, on one sequence:
  * the three-channel feature BMPs (stage A output)
  * the inferred heatmap + offset .bin tensors (stage B output)

FRAME MAPPING IS NOT 1:1, and getting this wrong produces a false "identical".
The two demos name the same physical frame differently:

  reference (shell demo) : gmc writes feat/<seq>_<PUSH>.bmp, then a host-side
                           rename step symlinks it to <seq>__<PUSH+1>.bmp
                           (PUSH_OFFSET=1, measured). The NPU stage therefore
                           sees ids 3..537 for pushes 2..536.
  e2e_demo (single proc) : the NPU stage is handed the feature paths straight
                           from stage A, so it sees pushes 0..536 with no rename.

So reference id N is the same frame as e2e push N-1. Feeding e2e pushes 0 and 1
to the reference would compare different frames and could report a difference
that does not exist (or hide one that does).

The whole prefix that only one side has is reported separately instead of being
quietly padded -- a file that exists on one side only is a finding, not noise.
"""

import argparse
import hashlib
import sys
from pathlib import Path

PUSH_OFFSET = 1


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def first_diff(a: Path, b: Path) -> str:
    """Byte offset of the first difference, or 'same-size-different-bytes'."""
    if a.stat().st_size != b.stat().st_size:
        return f"size {a.stat().st_size} vs {b.stat().st_size}"
    with a.open("rb") as fa, b.open("rb") as fb:
        off = 0
        while True:
            ca, cb = fa.read(1 << 20), fb.read(1 << 20)
            if not ca and not cb:
                return "identical content, same size (should not reach here)"
            if ca != cb:
                for i, (x, y) in enumerate(zip(ca, cb)):
                    if x != y:
                        return f"offset {off + i}: {x} vs {y}"
            off += len(ca)


def compare_features(ref_dir: Path, e2e_dir: Path, seq: str, limit: int):
    ref = sorted(ref_dir.glob(f"{seq}_*.bmp"))
    e2e = sorted(e2e_dir.glob(f"{seq}_*.bmp"))
    if limit:
        ref, e2e = ref[:limit], e2e[:limit]
    names_ref = {p.name for p in ref}
    names_e2e = {p.name for p in e2e}

    only_ref = sorted(names_ref - names_e2e)
    only_e2e = sorted(names_e2e - names_ref)
    common = sorted(names_ref & names_e2e)

    print(f"[feature] ref={len(ref)} e2e={len(e2e)} common={len(common)} "
          f"only_ref={len(only_ref)} only_e2e={len(only_e2e)}")
    if only_ref:
        print(f"  [note] ref-only (first 3): {only_ref[:3]}"
              f"{' ...' if len(only_ref) > 3 else ''}")
    if only_e2e:
        print(f"  [note] e2e-only (first 3): {only_e2e[:3]}"
              f"{' ...' if len(only_e2e) > 3 else ''}")

    bad = []
    for n in common:
        if md5(ref_dir / n) != md5(e2e_dir / n):
            bad.append(n)
    print(f"[feature] byte-identical: {len(common) - len(bad)}/{len(common)}")
    if bad:
        print(f"  [FAIL] {len(bad)} mismatched, e.g. {bad[:3]}")
        n = bad[0]
        print(f"         {n}: {first_diff(ref_dir / n, e2e_dir / n)}")
    return len(common), len(bad)


def compare_heatmaps(ref_dir: Path, e2e_dir: Path, seq: str, ref_tag: str, e2e_tag: str):
    # reference: <ref_tag>_<seq>__<id>_{hm,off}.bin   id = push + PUSH_OFFSET
    # e2e     : <e2e_tag>_<seq>_<push>_{hm,off}.bin
    pairs, missing_ref, missing_e2e = [], [], []
    for rp in sorted(ref_dir.glob(f"{ref_tag}_{seq}__*_hm.bin")):
        fid = int(rp.name.split("__")[1].split("_")[0])
        push = fid - PUSH_OFFSET
        # Compare like kind with like kind: the reference frame contributes BOTH
        # a _hm.bin and an _off.bin, and the offset tensor is twice the size of
        # the heatmap one (2 channels vs 1). Pairing the reference heatmap
        # against the e2e offset tensor reports a size mismatch on every frame
        # and looks like a wholesale difference when nothing is wrong.
        for kind in ("hm", "off"):
            rk = ref_dir / f"{ref_tag}_{seq}__{fid:06d}_{kind}.bin"
            ep = e2e_dir / f"{e2e_tag}_{seq}_{push:06d}_{kind}.bin"
            if not rk.exists():
                missing_ref.append(rk.name)
            elif not ep.exists():
                missing_e2e.append(ep.name)
            else:
                pairs.append((rk, ep, kind, fid, push))
    for ep in sorted(e2e_dir.glob(f"{e2e_tag}_{seq}_*_hm.bin")):
        push = int(ep.name.split("_")[-2])
        fid = push + PUSH_OFFSET
        if not (ref_dir / f"{ref_tag}_{seq}__{fid:06d}_hm.bin").exists():
            missing_ref.append(ep.name)

    print(f"[heatmap] compared pairs (hm+off): {len(pairs)}")
    if missing_ref:
        print(f"  [note] e2e frames with no reference counterpart (first 3): {missing_ref[:3]}"
              f"{' ...' if len(missing_ref) > 3 else ''}")
    if missing_e2e:
        print(f"  [note] reference frames with no e2e counterpart (first 3): {missing_e2e[:3]}"
              f"{' ...' if len(missing_e2e) > 3 else ''}")

    bad = []
    for rp, ep, kind, fid, push in pairs:
        if md5(rp) != md5(ep):
            bad.append((rp, ep, kind, fid))
    print(f"[heatmap] byte-identical: {len(pairs) - len(bad)}/{len(pairs)}")
    if bad:
        rp, ep, kind, fid = bad[0]
        print(f"  [FAIL] {len(bad)} mismatched, e.g. {rp.name} vs {ep.name}")
        print(f"         {kind}: {first_diff(rp, ep)}")
    return len(pairs), len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-root", required=True, help="reference out-root (shell demo)")
    ap.add_argument("--e2e-root", required=True, help="e2e_demo out-root")
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--ref-tag", default="frame")
    ap.add_argument("--e2e-tag", default=None, help="default = sequence")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    ref_root, e2e_root = Path(a.ref_root), Path(a.e2e_root)
    e2e_tag = a.e2e_tag or a.sequence

    print("=" * 78)
    print(f"  byte-exact comparison  sequence={a.sequence}")
    print(f"  reference : {ref_root}")
    print(f"  e2e_demo  : {e2e_root}")
    print(f"  push_offset={PUSH_OFFSET}  (reference id N  ==  e2e push N-1)")
    print("=" * 78)

    nf, bf = compare_features(ref_root / "feat", e2e_root / "feat", a.sequence, a.limit)
    print()
    nh, bh = compare_heatmaps(ref_root / "hm", e2e_root / "hm", a.sequence, a.ref_tag, e2e_tag)
    print()

    print("=" * 78)
    if bf == 0 and bh == 0 and nf > 0 and nh > 0:
        print(f"  VERDICT: IDENTICAL  features {nf}/{nf}, heatmap+offset tensors {nh}/{nh}")
        print("=" * 78)
        return 0
    print(f"  VERDICT: DIFFERENT  feature mismatches={bf}, tensor mismatches={bh}")
    print("=" * 78)
    return 1


if __name__ == "__main__":
    sys.exit(main())
