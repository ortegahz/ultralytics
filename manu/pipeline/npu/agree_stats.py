#!/usr/bin/env python3
"""Aggregate box-level detection agreement over a large held-out set.

Input tree (one file per image, as written by manu/pipeline/npu/batch_eval.sh):

    <root>/<model>/<image>.txt
        #0 stop sign score=0.827 box=(l=203.6 t=90.3 r=437.4 b=288.6) w=233.8 h=198.3 class_id=11

Boxes are matched between the reference model and each candidate per image by
greedy highest-IoU assignment, then agreement is reported both overall and
stratified by confidence and box size -- because a single headline percentage
hides the fact that disagreements cluster entirely at the decision boundary.

Also reports a one-sided summary: how many detections each model produces in
total, so "matches the reference" cannot be confused with "finds the same
number of things".
"""
import os
import re
import sys

LINE = re.compile(
    r"^#?\d*\s*(?P<name>.+?)\s+score=(?P<score>[0-9.]+)\s+"
    r"box=\(l=\s*(?P<l>-?[0-9.]+)\s+t=\s*(?P<t>-?[0-9.]+)\s+"
    r"r=\s*(?P<r>-?[0-9.]+)\s+b=\s*(?P<b>-?[0-9.]+)\)")


def parse(path):
    out = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        ln = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", ln).replace("\r", "").strip()
        m = LINE.match(ln)
        if m:
            out.append(dict(name=m.group("name").strip(), score=float(m.group("score")),
                            l=float(m.group("l")), t=float(m.group("t")),
                            r=float(m.group("r")), b=float(m.group("b"))))
    return out


def iou(a, b):
    x1, y1 = max(a["l"], b["l"]), max(a["t"], b["t"])
    x2, y2 = min(a["r"], b["r"]), min(a["b"], b["b"])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a["r"] - a["l"]) * (a["b"] - a["t"]) + (b["r"] - b["l"]) * (b["b"] - b["t"]) - inter
    return inter / ua if ua > 0 else 0.0


def load(root, model):
    d = os.path.join(root, model)
    if not os.path.isdir(d):
        sys.exit("missing model dir: %s" % d)
    res = {}
    for f in os.listdir(d):
        if f.endswith(".txt"):
            res[f[:-4]] = parse(os.path.join(d, f))
    return res


def match(A, B, thr):
    """Greedy highest-IoU one-to-one matching; returns list of (a, b, iou)."""
    pairs, used = [], set()
    for i, a in enumerate(A):
        best, bj = 0.0, -1
        for j, b in enumerate(B):
            if j in used:
                continue
            v = iou(a, b)
            if v > best:
                best, bj = v, j
        if bj >= 0 and best >= thr:
            used.add(bj)
            pairs.append((a, B[bj], best))
    return pairs, used


def wilson(k, n, z=1.96):
    """Wilson score interval -- honest CI for a proportion, unlike normal approx."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - s) / d, (c + s) / d)


def main():
    root, ref = sys.argv[1], sys.argv[2]
    others = sys.argv[3:]
    thr = float(os.environ.get("IOU_THR", "0.5"))

    data = {m: load(root, m) for m in [ref] + others}
    imgs = sorted(set().union(*[set(v) for v in data.values()]))

    print("root=%s" % root)
    print("images=%d   reference=%s   IoU thr=%.2f" % (len(imgs), ref, thr))
    for m in data:
        print("   %-32s %5d detections" % (m, sum(len(v) for v in data[m].values())))
    print()

    for o in others:
        n_ref = n_o = 0
        matched = cls_ok = 0
        ious = []
        rec = []          # (score, maxside, same_class, iou)
        miss_small = miss_big = 0
        extra = 0
        for im in imgs:
            A, B = data[ref].get(im, []), data[o].get(im, [])
            n_ref += len(A)
            n_o += len(B)
            pairs, used = match(A, B, thr)
            for a, b, v in pairs:
                matched += 1
                ious.append(v)
                same = (a["name"] == b["name"])
                cls_ok += same
                ms = max(a["r"] - a["l"], a["b"] - a["t"])
                rec.append((a["score"], ms, same, v))
            for i, a in enumerate(A):
                if not any(a is pa for pa, _, _ in pairs):
                    if max(a["r"] - a["l"], a["b"] - a["t"]) < 50:
                        miss_small += 1
                    else:
                        miss_big += 1
            extra += len(B) - len(used)

        def rate(k, n):
            return "%.1f%%" % (100.0 * k / n) if n else "  n/a"

        lo, hi = wilson(matched, n_ref)
        mious = sum(ious) / len(ious) if ious else 0.0
        print("=" * 78)
        print("vs %s" % o)
        print("  detections: reference=%d  candidate=%d  (candidate finds %+d, %.1f%%)" % (
            n_ref, n_o, n_o - n_ref, 100.0 * n_o / n_ref if n_ref else 0))
        print("  reference detections matched at IoU>=%.2f : %d/%d = %s   95%% CI [%s, %s]" % (
            thr, matched, n_ref, rate(matched, n_ref), "%.1f%%" % (100 * lo), "%.1f%%" % (100 * hi)))
        print("  mean IoU of matched boxes : %.3f" % mious)
        print("  same class among matched   : %d/%d = %s" % (cls_ok, matched, rate(cls_ok, matched)))
        print("  reference boxes MISSED     : %d total  (<50px: %d, >=50px: %d)" % (
            n_ref - matched, miss_small, miss_big))
        print("  candidate-only boxes       : %d" % extra)
        print()
        print("  agreement stratified by the REFERENCE detection's own score:")
        for lo_s, hi_s in [(0.25, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.01)]:
            sel = [x for x in rec if lo_s <= x[0] < hi_s]
            if not sel:
                continue
            k = sum(1 for x in sel if x[2])
            print("     score [%.2f,%.2f)  %5d boxes  same-class %s" % (lo_s, hi_s, len(sel), rate(k, len(sel))))
        print("  agreement stratified by the REFERENCE box's longest side:")
        for lo_s, hi_s in [(0, 25), (25, 50), (50, 150), (150, 10000)]:
            sel = [x for x in rec if lo_s <= x[1] < hi_s]
            if not sel:
                continue
            k = sum(1 for x in sel if x[2])
            print("     max-side [%3d,%5d)px  %5d boxes  same-class %s" % (lo_s, hi_s, len(sel), rate(k, len(sel))))
        print()


if __name__ == "__main__":
    main()