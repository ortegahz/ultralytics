#!/usr/bin/env python3
"""Build the board-side feature symlink map for every sequence (host side).

Board outputs are named by PUSH INDEX (`<SEQ>_<push:06d>.bmp`, 0-based), while the
frozen dataset and the labels are keyed by FRAME ID (`<SEQ>__<id:06d>.<ext>`,
numbered from 1). The link therefore carries a constant offset:

    push = id - PUSH_OFFSET           (PUSH_OFFSET = 1)

Two things this script refuses to get wrong:

1. **The symlink target must be RELATIVE.** The board mounts the same tree at a
   different prefix, so an absolute host path resolves to nothing over there and
   the board dies with a bare "stbi_load failed" -- no error summary, zero rows.

2. **Expected frame counts come from the frames list, not from whatever happens
   to be on disk.** A short TSV must be caught here, not silently accepted.

Usage:
    hm_val_prepare_links.py --frames-list L.txt --board-root /mnt/manu/npu/out/demo \
                            --host-root  /run/media/manu/SSD960-LOCAL/nfs/npu/out/demo
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-list", required=True)
    ap.add_argument("--board-root", required=True,
                    help="board-side prefix, e.g. /mnt/manu/npu/out/demo")
    ap.add_argument("--host-root", required=True,
                    help="host-side prefix of the SAME tree")
    ap.add_argument("--push-offset", type=int, default=1)
    args = ap.parse_args()

    ids = {}
    with open(args.frames_list) as f:
        for line in f:
            line = line.strip()
            if not line or "__" not in os.path.basename(line):
                continue
            seq, rest = os.path.basename(line).split("__", 1)
            stem = rest.rsplit(".", 1)[0]
            try:
                ids.setdefault(seq, []).append(int(stem))
            except ValueError:
                continue

    off = args.push_offset
    problems = []
    total = 0
    for seq in sorted(ids):
        want = sorted(ids[seq])
        named_host = os.path.join(args.host_root, seq, "feat_named")
        os.makedirs(named_host, exist_ok=True)

        # stale links from a previous, differently-offset run would poison the
        # board run, so rebuild the whole map rather than merging into it
        for f in os.listdir(named_host):
            os.unlink(os.path.join(named_host, f))

        for fid in want:
            push = fid - off
            if push < 0:
                problems.append(f"{seq} id={fid} -> push={push} 越界")
                continue
            link = os.path.join(named_host, f"{seq}__{fid:06d}.bmp")
            os.symlink(f"../feat/{seq}_{push:06d}.bmp", link)

        with open(os.path.join(named_host, "list_board.txt"), "w") as fh:
            for fid in want:
                fh.write(f"{args.board_root}/{seq}/feat_named/{seq}__{fid:06d}.bmp\n")

        # the raw staging must reach at least the highest push index we link to
        rawdir = os.path.join(args.host_root, seq, "raw", seq)
        n_raw = len([x for x in os.listdir(rawdir) if x.endswith(".bmp")]) \
            if os.path.isdir(rawdir) else 0
        need = max(want) - off + 1
        flag = "" if n_raw >= need else f"  <== RAW 不足(raw={n_raw} < 需 {need})"
        if n_raw < need:
            problems.append(f"{seq}: raw={n_raw} < 需要 {need}")
        print(f"[link] {seq:24s} frames={len(want):5d} ids={want[0]}..{want[-1]} "
              f"raw={n_raw:5d}{flag}")
        total += len(want)

    print(f"[done] {len(ids)} 序列, 共 {total} 帧链接")
    if problems:
        print(f"[PROBLEMS] {len(problems)} 条:", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())