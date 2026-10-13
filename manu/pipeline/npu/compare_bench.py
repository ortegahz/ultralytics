#!/usr/bin/env python3
"""Aggregate per-image detection logs from several .rknn runs and diff them.

The board wrote, for each (model, image), a text file of the official demo's
lines:    person @ (114 235 212 527)  0.819099

This walks a bench directory laid out as

    <bench>/<model>/<image>.txt

matches boxes across models per image by IoU, and reports agreement statistics
so "the re-quantized model gives the same results" rests on numbers over a
held-out set rather than on a single picture.
"""
import os
import re
import sys

LINE = re.compile(r"^\s*(\S.*?)\s*@\s*\((-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\)\s+([0-9.]+)\s*$")


def parse(path):
    out = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        ln = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", ln).replace("\r", "")
        m = LINE.match(ln)
        if m:
            out.append(dict(name=m.group(1).strip(), l=int(m.group(2)), t=int(m.group(3)),
                            r=int(m.group(4)), b=int(m.group(5)), score=float(m.group(6))))
    return out


def iou(a, b):
    x1, y1 = max(a["l"], b["l"]), max(a["t"], b["t"])
    x2, y2 = min(a["r"], b["r"]), min(a["b"], b["b"])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    ua = (a["r"] - a["l"]) * (a["b"] - a["t"]) + (b["r"] - b["l"]) * (b["b"] - b["t"]) - inter
    return inter / ua if ua > 0 else 0.0


def load(bench, model):
    d = os.path.join(bench, model)
    if not os.path.isdir(d):
        sys.exit("no such model dir: %s" % d)
    res = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".txt"):
            res[f[:-4]] = parse(os.path.join(d, f))
    return res


def main():
    bench = sys.argv[1]
    ref = sys.argv[2]
    others = sys.argv[3:]
    thr = float(os.environ.get("IOU_THR", "0.5"))

    models = [ref] + others
    data = {m: load(bench, m) for m in models}
    imgs = sorted(set().union(*[set(v) for v in data.values()]))
    print("bench dir : %s" % bench)
    print("images    : %d" % len(imgs))
    print("reference : %s" % ref)
    print("IoU thr   : %.2f\n" % thr)

    hdr = "%-30s %8s" % ("image", ref[:8])
    for o in others:
        hdr += " | %-28s" % ("vs " + o[:26])
    print(hdr)
    print("-" * len(hdr))
    tot = {m: 0 for m in others}
    tot_match = {m: 0 for m in others}
    ious = {m: [] for m in others}
    clsmatch = {m: 0 for m in others}

    for im in imgs:
        A = data[ref].get(im, [])
        line = "%-30s %8d" % (im[:30], len(A))
        for o in others:
            B = data[o].get(im, [])
            used, mt, cls, local_ious = set(), 0, 0, []
            for a in A:
                best, bi = 0.0, -1
                for j, b in enumerate(B):
                    if j in used:
                        continue
                    v = iou(a, b)
                    if v > best:
                        best, bi = v, j
                if bi >= 0 and best >= thr:
                    used.add(bi)
                    mt += 1
                    local_ious.append(best)
                    if B[bi]["name"] == a["name"]:
                        cls += 1
            tot_match[o] += mt
            clsmatch[o] += cls
            ious[o].extend(local_ious)
            mi = sum(local_ious) / len(local_ious) if local_ious else 0.0
            line += " | %3d/%-3d IoU %.3f cls %d/%d" % (mt, len(A), mi, cls, mt)
        print(line)

    print("\n=== aggregate (reference detections = %d) ===" % sum(len(data[ref].get(i, [])) for i in imgs))
    for o in others:
        n_ref = sum(len(data[ref].get(i, [])) for i in imgs)
        n_o = sum(len(data[o].get(i, [])) for i in imgs)
        m = tot_match[o]
        mi = sum(ious[o]) / len(ious[o]) if ious[o] else 0.0
        print("  vs %-30s dets=%3d  matched=%3d/%3d (%.1f%%)  meanIoU=%.3f  same-class=%3d/%3d (%.1f%%)"
              % (o, n_o, m, n_ref, 100.0 * m / n_ref if n_ref else 0, mi, clsmatch[o], n_ref,
                 100.0 * clsmatch[o] / n_ref if n_ref else 0))


if __name__ == "__main__":
    main()