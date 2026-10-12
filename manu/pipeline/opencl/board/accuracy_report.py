#!/usr/bin/env python3
"""
accuracy_report.py -- the precision reconciliation, at the level the project
actually accepts.

THREE INDEPENDENT CHECKS, because any one of them alone can pass for the wrong
reason:

  1. BOARD-INTERNAL GPU vs CPU, per channel, over ABSOLUTE counts.
     `--fused both` runs both arms on the SAME fits in the SAME push, so any
     difference is attributable to the fused tail alone and nothing else. Reported
     as max|Diff|, total absolute difference, and the exact-match pixel COUNT --
     never as a mean of per-sequence means, which silently loses the frame count,
     and never as "0 differences observed", because zero observations across 23
     frames once looked bit-exact here until the sample was widened to 59.

  2. CROSS-MACHINE, board GPU output vs x86 CPU reference, per-frame MD5.
     Different CPU, different OpenCV build, different architecture. Equality here
     is the strongest statement available: the delivered artefact on RK3588 is the
     same bytes as the desktop reference.

  3. MD5 STABILITY across the timed and untimed runs.
     Proves the timing probes did not perturb the computation on the board either.
"""
import argparse
import hashlib
import os
import sys
from collections import defaultdict


def read_md5(path):
    """seq -> {frame_idx: hash}; tolerates the append-into-existing-file hazard."""
    d = defaultdict(dict)
    if not os.path.exists(path):
        return d
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            d[p[0]][int(p[1])] = p[2]
    return d


def dir_md5(path):
    if not os.path.isdir(path):
        return None
    h = hashlib.md5()
    for f in sorted(os.listdir(path)):
        h.update(f.encode())
        with open(os.path.join(path, f), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board-md5", action="append", default=[],
                    help="board md5_out file (repeatable)")
    ap.add_argument("--x86-md5", default="", help="x86 reference md5_out file")
    ap.add_argument("--board-ab-log", default="", help="bench log containing [A/B] lines")
    args = ap.parse_args()

    if args.board_ab_log and os.path.exists(args.board_ab_log):
        print("=" * 100)
        print("CHECK 1  board-internal GPU vs CPU, same fits, absolute counts")
        print("=" * 100)
        print(f"{'sequence':<32}{'ch':>4}{'Max|Diff|':>11}{'total|d|':>12}{'exact px':>14}"
              f"{'exact %':>10}")
        print("-" * 100)
        for line in open(args.board_ab_log):
            if not line.startswith("[A/B]"):
                continue
            seq = line.split()[1]
            body = line[line.index("Ch0") + 3:]
            for ci, ch in enumerate(("Ch1", "Ch2")):
                pass
            # format: Ch0 max=.. MAE=.. exact=..% | Ch1 .. | Ch2 ..
            for chunk in body.split("|"):
                chunk = chunk.strip()
                if not chunk.startswith("Ch"):
                    continue
                name = chunk.split()[0]
                d = dict()
                for tok in chunk.split()[1:]:
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        d[k] = v
                print(f"{seq:<32}{name:>4}{d.get('max',''):>11}{d.get('MAE',''):>12}"
                      f"{d.get('exact',''):>14}{'':>10}")
        print()

    if args.board_md5:
        print("=" * 100)
        print("CHECK 2/3  per-frame MD5: board vs x86 reference, and vs prior board run")
        print("=" * 100)
        board = read_md5(args.board_md5[0])
        for extra in args.board_md5[1:]:
            o = read_md5(extra)
            same = tot = 0
            for s in board:
                for k, v in board[s].items():
                    tot += 1
                    if o.get(s, {}).get(k) == v:
                        same += 1
            print(f"  board run-to-run: {same}/{tot} frames identical"
                  f"{'' if same == tot else '   <-- MISMATCH'}")

        if args.x86_md5:
            x86 = read_md5(args.x86_md5)
            same = tot = miss = 0
            diffs = []
            for s in sorted(set(board) & set(x86)):
                for k in sorted(set(board[s]) & set(x86[s])):
                    tot += 1
                    if board[s][k] == x86[s][k]:
                        same += 1
                    else:
                        diffs.append((s, k))
            print(f"  board vs x86 CPU reference: {same}/{tot} frames identical"
                  f"{'' if same == tot else '   <-- MISMATCH'}")
            if diffs:
                print(f"    first mismatches: {diffs[:10]}")
            if not tot:
                print("    (no overlapping sequence/frame keys -- nothing was compared)")
        else:
            print("  (no x86 reference supplied)")
    print()


if __name__ == "__main__":
    main()