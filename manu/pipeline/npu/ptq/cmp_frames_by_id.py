#!/usr/bin/env python3
"""
cmp_frames_by_id.py -- compare two GMC feature streams BY PHYSICAL FRAME ID.

WHY THIS EXISTS
    A live-rate run (25 fps) drops frames, so the two arms do NOT produce the
    same number of rows and the same row order.  Comparing row i with row i
    compares different frames -- and a shifted comparison silently reports
    differences that are not differences at all.  The frame id is column 2 of
    feature_md5.txt, so every comparison here is done on the frame id.

WHAT IT REPORTS
    1. Per pair of arms: the common frame ids, how many are bit-identical,
       how many differ, and where the differing ones are (runs).
    2. For each arm, the HISTORY COMPLETENESS of every frame it produced:
       the GMC history of frame f is the set of physical frames
       {f - lag} for lag in {step, 2*step, ..., window*step}.  A frame is
       COMPLETE when every one of those ids is also in the arm's own frame
       set, and it has a HOLE when at least one is missing.
    3. The same split for the reference arm, so that "the md5 differs" can be
       attributed to a hole rather than to the change under test.

USAGE
    python3 cmp_frames_by_id.py \
        --arm  ref=/path/ref/feature_md5.txt \
        --arm  new=/path/new/feature_md5.txt \
        --lag-step 2 --window 21 \
        --out /tmp/cmp.md

    --arm name=path      repeatable; the FIRST arm is the reference.
    --lag-step N         the GMC stride (default 2).
    --window N           the number of lags (default 21).
    --out PATH           write the markdown report here (stdout as well).
"""
import argparse
import os
import sys


def load(path):
    """frame id -> md5.  Column 2 is the frame id, column 3 the hash."""
    d = {}
    if not os.path.exists(path):
        return None
    with open(path, "r", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                try:
                    fid = int(parts[1])
                except ValueError:
                    continue
                d[fid] = parts[2]
    return d


def runs_of(sorted_ids):
    """Contiguous runs of a sorted id list -> [(lo, hi), ...]."""
    out = []
    for x in sorted_ids:
        if out and x == out[-1][1] + 1:
            out[-1][1] = x
        else:
            out.append([x, x])
    return [(a, b) for a, b in out]


def history_lags(lag_step, window):
    return [lag_step * (i + 1) for i in range(window)]


def completeness(frames, lag_step, window):
    """frame id -> number of MISSING history frames (0 == complete)."""
    lags = history_lags(lag_step, window)
    have = set(frames)
    missing = {}
    for f in frames:
        miss = sum(1 for lag in lags if (f - lag) not in have)
        missing[f] = miss
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    metavar="name=path", help="repeatable; first one is the reference")
    ap.add_argument("--lag-step", type=int, default=2)
    ap.add_argument("--window", type=int, default=21)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    arms = []
    for spec in args.arm:
        if "=" not in spec:
            print("[ERROR] --arm must be name=path", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        data = load(path)
        if data is None:
            print("[ERROR] missing feature_md5.txt: %s" % path, file=sys.stderr)
            return 2
        arms.append((name, path, data))

    lines = []
    def emit(s=""):
        lines.append(s)
        print(s)

    emit("### GMC feature streams compared BY FRAME ID")
    emit()
    emit("- lag-step = %d, window = %d  =>  history = {f-%d .. f-%d} on a %d grid"
         % (args.lag_step, args.window, args.lag_step,
            args.lag_step * args.window, args.lag_step))
    emit()
    for name, path, data in arms:
        ids = sorted(data)
        miss = completeness(ids, args.lag_step, args.window)
        complete = sum(1 for f in ids if miss[f] == 0)
        holes = len(ids) - complete
        holes_with = sum(1 for f in ids if miss[f] > 0)
        emit("- **%s**  frames=%d  complete-history=%d  with-hole=%d  "
             "(%s)" % (name, len(ids), complete, holes_with, path))
    emit()

    ref_name, ref_path, ref = arms[0]
    for name, path, data in arms[1:]:
        common = sorted(set(ref) & set(data))
        same = [f for f in common if ref[f] == data[f]]
        diff = [f for f in common if ref[f] != data[f]]
        emit("#### %s  vs  %s" % (name, ref_name))
        emit()
        emit("| | frames |")
        emit("|:--|--:|")
        emit("| %s (reference) | %d |" % (ref_name, len(ref)))
        emit("| %s | %d |" % (name, len(data)))
        emit("| common frame ids | %d |" % len(common))
        emit("| **bit-identical** | **%d** |" % len(same))
        emit("| different | %d |" % len(diff))
        only_ref = sorted(set(ref) - set(data))
        only_new = sorted(set(data) - set(ref))
        emit("| only in %s | %d |" % (ref_name, len(only_ref)))
        emit("| only in %s | %d |" % (name, len(only_new)))
        emit()

        if only_ref:
            rr = runs_of(only_ref)
            emit("- only in %s: %d frames, runs (first 12): %s"
                 % (ref_name, len(only_ref), rr[:12]))
        if only_new:
            rr = runs_of(only_new)
            emit("- only in %s: %d frames, runs (first 12): %s"
                 % (name, len(only_new), rr[:12]))
        emit()

        # Split the comparison by history completeness, so that a difference
        # can be attributed to a HOLE rather than to the change under test.
        ref_miss = completeness(sorted(ref), args.lag_step, args.window)
        new_miss = completeness(sorted(data), args.lag_step, args.window)
        buckets = {
            "both complete": [], "both holed": [],
            "ref complete / new holed": [], "ref holed / new complete": [],
        }
        for f in common:
            rc = ref_miss[f] == 0
            nc = new_miss[f] == 0
            key = ("both complete" if rc and nc else
                   "both holed" if not rc and not nc else
                   "ref complete / new holed" if rc else
                   "ref holed / new complete")
            buckets[key].append(f)
        emit("| history state | common frames | md5 identical | md5 different |")
        emit("|:--|--:|--:|--:|")
        for key in ("both complete", "both holed",
                    "ref complete / new holed", "ref holed / new complete"):
            ids = buckets[key]
            if not ids:
                emit("| %s | 0 | 0 | 0 |" % key)
                continue
            n_same = sum(1 for f in ids if ref[f] == data[f])
            emit("| %s | %d | %d | %d |"
                 % (key, len(ids), n_same, len(ids) - n_same))
        emit()
        if diff:
            rr = runs_of(diff)
            emit("- differing frame ids: %d, first = %d, runs (first 12): %s"
                 % (len(diff), diff[0], rr[:12]))
            holed = [f for f in diff
                     if ref_miss[f] > 0 or new_miss[f] > 0]
            emit("- of the differing frames, %d have a hole in at least one arm"
                 % len(holed))
            emit()

    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print("[ok] wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
