#!/usr/bin/env python3
"""Compare detections from two RKNN runs box-by-box.

Input: two stdout captures from rknn_yolov5_demo, whose per-detection lines look
like    person @ (114 235 212 527)  0.819099
(the demo prints left, top, right, bottom and the score).

Reports matched pairs with per-corner deltas, so "same result" is backed by
numbers rather than by eyeballing two OSD images.
"""
import re
import sys

LINE = re.compile(r"^\s*(\S.*?)\s*@\s*\((-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\)\s+([0-9.]+)\s*$")


def parse(path):
    dets = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        m = LINE.match(ln)
        if m:
            name = m.group(1).strip()
            dets.append(dict(name=name, l=int(m.group(2)), t=int(m.group(3)),
                             r=int(m.group(4)), b=int(m.group(5)),
                             score=float(m.group(6))))
    return dets


def iou(a, b):
    x1 = max(a["l"], b["l"]); y1 = max(a["t"], b["t"])
    x2 = min(a["r"], b["r"]); y2 = min(a["b"], b["b"])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    ua = (a["r"] - a["l"]) * (a["b"] - a["t"]) + (b["r"] - b["l"]) * (b["b"] - b["t"]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    pa, pb, thresh = sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    A, B = parse(pa), parse(pb)
    print("A = %s  (%d detections)" % (pa, len(A)))
    print("B = %s  (%d detections)" % (pb, len(B)))
    print("IoU match threshold = %.2f\n" % thresh)

    used = set()
    print("%-11s %-11s %7s  %-28s %-28s %s" % ("A.class", "B.class", "IoU", "A (l,t,r,b)", "B (l,t,r,b)", "score A/B  maxcorner_delta"))
    print("-" * 118)
    for a in A:
        best, bi = 0.0, -1
        for j, b in enumerate(B):
            if j in used:
                continue
            v = iou(a, b)
            if v > best:
                best, bi = v, j
        if bi < 0 or best < thresh:
            print("%-11s %-11s %7.3f  %-28s %-28s %s" % (
                a["name"][:11], "-- unmatched --", best,
                "(%d,%d,%d,%d)" % (a["l"], a["t"], a["r"], a["b"]), "-",
                "%.6f" % a["score"]))
            continue
        used.add(bi)
        b = B[bi]
        d = max(abs(a["l"] - b["l"]), abs(a["t"] - b["t"]), abs(a["r"] - b["r"]), abs(a["b"] - b["b"]))
        print("%-11s %-11s %7.3f  %-28s %-28s %.6f/%.6f  maxdelta=%d px" % (
            a["name"][:11], b["name"][:11], best,
            "(%d,%d,%d,%d)" % (a["l"], a["t"], a["r"], a["b"]),
            "(%d,%d,%d,%d)" % (b["l"], b["t"], b["r"], b["b"]),
            a["score"], b["score"], d))

    un = [B[j] for j in range(len(B)) if j not in used]
    if un:
        print("\nB-only (extra in new model): %d" % len(un))
        for b in un[:15]:
            print("   %-11s (%d,%d,%d,%d) %.6f" % (b["name"][:11], b["l"], b["t"], b["r"], b["b"], b["score"]))

    matched = len(used)
    print("\nsummary: matched %d/%d of A" % (matched, len(A)))
    cls_ok = 0
    for a in A:
        best, bi = 0.0, -1
        for j, b in enumerate(B):
            v = iou(a, b)
            if v > best:
                best, bi = v, j
        if bi >= 0 and best >= thresh and B[bi]["name"] == a["name"]:
            cls_ok += 1
    print("        same class on matched: %d/%d" % (cls_ok, len(A)))


if __name__ == "__main__":
    main()