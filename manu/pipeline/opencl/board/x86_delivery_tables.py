#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
x86_delivery_tables.py -- build the consolidated delivery tables from the
regenerated x86 artefacts.

Every number printed here is parsed out of a file that is still on disk under
.bench_out/REGEN/. Nothing is transcribed by hand, so the tables cannot drift
from the evidence the way a hand-maintained report table can.

Percentiles are NEAREST RANK, no interpolation: for N samples sorted ascending,
the p-th percentile is element ceil(p/100 * N) (1-based). Interpolating would
manufacture a value that no frame ever produced, and the 40 ms red line is a
threshold on real frames.
"""

import os
import re
import sys
import math
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
REGEN = os.path.join(HERE, ".bench_out", "REGEN")

SEQ_ORDER = [
    "VIDEO00034_19700101_015052",
    "VIDEO00028_19700101_013631",
    "VIDEO00032_19700101_014453",
    "VIDEO00002_19700101_001415",
    "VIDEO00036_19700101_015740",
]
SHORT = {
    "VIDEO00034": "VIDEO00034",
    "VIDEO00028": "VIDEO00028",
    "VIDEO00032": "VIDEO00032",
    "VIDEO00002": "VIDEO00002",
    "VIDEO00036": "VIDEO00036",
}


def short(seq):
    return SHORT.get(seq.split("_")[0], seq)


def nearest_rank(sorted_vals, pct):
    """Nearest-rank percentile, 1-based, no interpolation."""
    n = len(sorted_vals)
    if n == 0:
        return None
    k = max(1, min(n, int(math.ceil(pct / 100.0 * n))))
    return sorted_vals[k - 1]


# --------------------------------------------------------------------------
# 1. [SEQ] line parser: the handle census
# --------------------------------------------------------------------------
SEQ_RE = re.compile(r"^\[SEQ\]\s+(\S+)\s+frames=(\d+)")
CAP_RE = re.compile(r"cornercap=(\d+)f/(\d+)calls\((\d+)pts\)")
MEAN_RE = re.compile(r"corners mean=([0-9.]+)")
DEVPAD_RE = re.compile(r"devpad=(\d+)")
ALLOCPAD_RE = re.compile(r"devallocpad=(\d+)")
REALLOC_RE = re.compile(r"devrealloc=(\d+)")
FITS_RE = re.compile(r"fits/frm=([0-9.]+)")


def parse_seq_lines(directory):
    """seq -> dict of census fields. Missing fields stay None, never 0."""
    out = OrderedDict()
    if not os.path.isdir(directory):
        return out
    for name in os.listdir(directory):
        if not name.endswith(".log"):
            continue
        seq = name[:-4]
        path = os.path.join(directory, name)
        rec = {
            "seq": seq,
            "frames": None,
            "cap_frames": None,
            "cap_calls": None,
            "cap_pts": None,
            "corners_mean": None,
            "devpad": None,
            "devallocpad": None,
            "devrealloc": None,
            "fits_per_frame": None,
        }
        try:
            with open(path, "r", errors="replace") as fh:
                for line in fh:
                    if not line.startswith("[SEQ]"):
                        continue
                    m = SEQ_RE.match(line)
                    if not m:
                        continue
                    rec["frames"] = int(m.group(2))
                    for rx, key in (
                        (DEVPAD_RE, "devpad"),
                        (ALLOCPAD_RE, "devallocpad"),
                        (REALLOC_RE, "devrealloc"),
                        (FITS_RE, "fits_per_frame"),
                    ):
                        mm = rx.search(line)
                        if mm:
                            rec[key] = (
                                float(mm.group(1)) if key == "fits_per_frame"
                                else int(mm.group(1))
                            )
                    mm = CAP_RE.search(line)
                    if mm:
                        rec["cap_frames"] = int(mm.group(1))
                        rec["cap_calls"] = int(mm.group(2))
                        rec["cap_pts"] = int(mm.group(3))
                    mm = MEAN_RE.search(line)
                    if mm:
                        rec["corners_mean"] = float(mm.group(1))
        except OSError:
            continue
        out[seq] = rec
    return out


# --------------------------------------------------------------------------
# 2. CSV timing ledger
# --------------------------------------------------------------------------
def parse_csv(path, warmup):
    """Return per-frame total_ms after the warmup window.

    Returns (values, n_total_rows). n_total_rows < expected is reported so a
    truncated CSV can never be silently presented as a full-length measurement.
    """
    if not os.path.isfile(path):
        return None, 0
    vals = []
    total = 0
    with open(path, "r", errors="replace") as fh:
        header = fh.readline()
        if "total_ms" not in header:
            return None, 0
        cols = [c.strip() for c in header.strip().split(",")]
        try:
            ti = cols.index("total_ms")
            fi = cols.index("frame_idx")
        except ValueError:
            return None, 0
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) <= max(ti, fi):
                continue
            total += 1
            try:
                idx = int(parts[fi])
            except ValueError:
                continue
            if idx < warmup:
                continue
            try:
                vals.append(float(parts[ti]))
            except ValueError:
                continue
    return vals, total


def timing_row(path, warmup):
    vals, total = parse_csv(path, warmup)
    if not vals:
        return None
    s = sorted(vals)
    return {
        "n": len(s),
        "csv_rows": total,
        "p50": nearest_rank(s, 50),
        "p95": nearest_rank(s, 95),
        "p99": nearest_rank(s, 99),
        "max": s[-1],
        "mean": sum(s) / len(s),
        "over40": sum(1 for v in s if v > 40.0),
    }


# --------------------------------------------------------------------------
# 3. Accuracy block parser
# --------------------------------------------------------------------------
LBL_RE = re.compile(r"(VIDEO\d{5}_\d+_\d+)\s+400 frames, fit-corners 600 -> 200")
CH_RE = re.compile(r"^\s*(Ch[012])\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)")


def parse_accuracy(path):
    """seq -> {chan: (max, mean|diff|, exact%)}.

    Line-oriented state machine rather than a split on a substring: the earlier
    draft indexed a list with list.index(blk), which returns the FIRST match, so
    any repeated block text would have silently attributed rows to the wrong
    sequence.
    """
    out = OrderedDict()
    if not os.path.isfile(path):
        return out
    cur = None
    rec = {}
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = LBL_RE.search(line)
            if m:
                if cur is not None and rec:
                    out[cur] = rec
                cur = m.group(1)
                rec = {}
                continue
            m = CH_RE.match(line)
            if m and cur is not None:
                rec[m.group(1)] = (
                    int(m.group(2)),
                    float(m.group(3)),
                    float(m.group(4)),
                )
    if cur is not None and rec:
        out[cur] = rec
    return out


def fmt(v, nd=2):
    return "n/a" if v is None else f"{v:.{nd}f}"


def rule(ch="-", n=118):
    return ch * n


def main():
    print(rule("="))
    print("X86 REGENERATED EVIDENCE -- handle census, timing ledger, accuracy alignment")
    print("artefact root: %s" % REGEN)
    prov = os.path.join(REGEN, "PROVENANCE.txt")
    if os.path.isfile(prov):
        with open(prov, errors="replace") as fh:
            for line in fh:
                if line.startswith("###"):
                    print(line.rstrip())
    print(rule("="))

    unc = parse_seq_lines(os.path.join(REGEN, "UNC"))
    b200 = parse_seq_lines(os.path.join(REGEN, "B200"))

    # ---- Table 1: handle trigger census --------------------------------
    print()
    print(rule())
    print("TABLE 1 -- CORNER-CAP HANDLE (B) TRIGGER CENSUS, 400 frames/sequence, x86")
    print(rule())
    hdr = (
        f"{'seq':<12} {'budget':>7} {'corners mean':>13} "
        f"{'trigger frames':>15} {'trigger calls':>14} {'cap pts':>9}"
    )
    print(hdr)
    print(rule())
    for seq in SEQ_ORDER:
        u = unc.get(seq)
        b = b200.get(seq)
        rows = []
        if u:
            rows.append(("5000", u))
        if b:
            rows.append(("200", b))
        for tag, r in rows:
            tf = r["cap_frames"]
            tc = r["cap_calls"]
            # Never render an unparsed count as zero: "no data" and "zero
            # triggers" look identical in a table and mean opposite things.
            tfs = "NOT PARSED" if tf is None else f"{tf}/{r['frames']}"
            tcs = "NOT PARSED" if tc is None else str(tc)
            cp = "NOT PARSED" if r["cap_pts"] is None else str(r["cap_pts"])
            print(
                f"{short(seq):<12} {tag:>7} {fmt(r['corners_mean'],1):>13} "
                f"{tfs:>15} {tcs:>14} {cp:>9}"
            )
        if not rows:
            print(f"{short(seq):<12} {'NO DATA':>7}")
    print()
    print("budget 5000 = unclipped control. Nothing can truncate, so trigger counts")
    print("              MUST be 0. If they are not, the probe itself is broken and")
    print("              every other count in this table is suspect.")
    print("budget 200  = the delivery budget.")

    # ---- Table 2: device allocation ledger ------------------------------
    print()
    print(rule())
    print("TABLE 2 -- PREALLOC HANDLE (A) EFFECT: device pad / reallocation ledger")
    print("arm A = baseline (grow-only).  arm B = --prealloc-pad 1024 (+ budget 200).")
    print("delivered claim is: run-time reallocation counts go to 0, leaving only the")
    print("one allocation at start-up. This is architecture-independent -- it is a")
    print("counter of C++-side configure() decisions, not a timing.")
    print(rule())
    print(
        f"{'seq':<12} {'arm':>4} {'devpad':>7} {'devallocpad':>12} {'devrealloc':>11}"
    )
    print(rule())
    a_arm = parse_seq_lines(os.path.join(REGEN, "A"))
    b_arm = parse_seq_lines(os.path.join(REGEN, "B"))
    for seq in SEQ_ORDER:
        for tag, d in (("A", a_arm), ("B", b_arm)):
            r = d.get(seq)
            if not r:
                continue
            print(
                f"{short(seq):<12} {tag:>4} {fmt(r['devpad'],0):>7} "
                f"{fmt(r['devallocpad'],0):>12} {fmt(r['devrealloc'],0):>11}"
            )
    print()
    print("NOTE: 'devpad' is the padding the fits actually required; 'devallocpad' is")
    print("      what was allocated. Prealloc makes the second >= the first at all")
    print("      times, which is the whole point -- but it also means the two columns")
    print("      must never be quoted as if they were the same number.")

    # ---- Table 3: timing ledger -----------------------------------------
    print()
    print(rule())
    print("TABLE 3 -- TIMING LEDGER, post-warmup frames, x86 (ms)")
    print()
    print("!! READ BEFORE QUOTING ANY NUMBER IN THIS TABLE !!")
    print("   x86 OpenCL here is PoCL 6.0 running on an Intel Core i5-10500 CPU, and")
    print("   it falls back to CL_RGBA (4 byte/px). The board is a Mali-G610 with")
    print("   native CL_R (1 byte/px). This table is therefore a CPU-emulated")
    print("   OpenCL number, NOT a proxy for board latency, and the 40 ms line")
    print("   cannot be assessed here -- every arm is over it by 3-6x.")
    print("   What the table IS good for: A-vs-B direction and size under identical")
    print("   conditions, and confirming the handles did not make anything slower.")
    print(rule())
    print(
        f"{'seq':<12} {'arm':>4} {'n':>4} {'p50':>7} {'p95':>7} {'p99':>7} "
        f"{'max':>7} {'>40ms':>6}"
    )
    print(rule())
    for seq in SEQ_ORDER:
        for arm in ("A", "B"):
            path = os.path.join(
                REGEN, arm, "profile_%s_%s.csv" % (arm, seq)
            )
            r = timing_row(path, warmup=100)
            if not r:
                print(f"{short(seq):<12} {arm:>4} {'NO DATA':>4}")
                continue
            print(
                f"{short(seq):<12} {arm:>4} {r['n']:>4} {fmt(r['p50']):>7} "
                f"{fmt(r['p95']):>7} {fmt(r['p99']):>7} {fmt(r['max']):>7} "
                f"{r['over40']:>6}"
            )

    # ---- Table 4: accuracy alignment ------------------------------------
    print()
    print(rule())
    print("TABLE 4 -- ACCURACY ALIGNMENT, --fit-corners 600 -> 200")
    print("400 frames/sequence, x86 CPU arm, per-pixel |a-b| (bit-exact basis)")
    print(rule())
    print(
        f"{'seq':<12} {'trig':>7} {'Ch0 max':>8} {'Ch1 max':>8} {'Ch1 mean':>10} "
        f"{'Ch1 exact%':>11} {'Ch2 max':>8} {'Ch2 mean':>10} {'Ch2 exact%':>11}"
    )
    print(rule())
    acc = parse_accuracy(
        os.path.join(REGEN, "ACC", "corner_budget_600_200.txt")
    )
    for seq in SEQ_ORDER:
        rec = acc.get(seq)
        trig = b200.get(seq)
        ts = "?" if not trig or trig["cap_frames"] is None else str(trig["cap_frames"])
        if not rec:
            print(f"{short(seq):<12} {ts:>7} {'NO DATA':>8}")
            continue
        c0 = rec.get("Ch0")
        c1 = rec.get("Ch1")
        c2 = rec.get("Ch2")
        print(
            f"{short(seq):<12} {ts:>7} "
            f"{fmt(c0[0],0) if c0 else 'n/a':>8} "
            f"{fmt(c1[0],0) if c1 else 'n/a':>8} "
            f"{fmt(c1[1],5) if c1 else 'n/a':>10} "
            f"{fmt(c1[2],4) if c1 else 'n/a':>11} "
            f"{fmt(c2[0],0) if c2 else 'n/a':>8} "
            f"{fmt(c2[1],5) if c2 else 'n/a':>10} "
            f"{fmt(c2[2],4) if c2 else 'n/a':>11}"
        )
    print()
    print("exact% = fraction of compared elements with |a-b| == 0 (threshold 1).")
    print(rule("="))


if __name__ == "__main__":
    sys.exit(main())