#!/usr/bin/env python3
"""guarantee_report.py -- the three delivery tables, recomputed on the host.

Nothing here trusts the board's own statistics. Percentiles are recomputed from
the raw per-frame CSVs with the nearest-rank rule (no interpolation), so every
P99 printed below is a frame that actually happened.

Deliberately all-numeric. The predecessor of this file was a shell/awk pipeline
whose max was taken with string comparison -- "51" < "6" -- so every two-digit
deviation was reported as if it were single-digit, silently. If you extend this,
keep the +0 coercions.
"""
import csv
import glob
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WARMUP = 100
RED_LINE = 40.0
STAGES = ["io", "prep", "fit", "img_up", "mats_up", "kernel", "readback"]


def pct(values, q):
    """Nearest-rank percentile: no interpolation, by design."""
    s = sorted(values)
    k = max(1, math.ceil(q / 100.0 * len(s)))
    return s[k - 1]


def load_seq(path, warmup):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return rows[warmup:]


def timing_table(d, warmup, expect_total=400):
    print("=" * 118)
    print("TABLE 1  per-frame end-to-end timing  (recomputed from raw CSV, "
          f"warmup {warmup} dropped, nearest-rank percentiles)")
    print("=" * 118)
    print(f"{'sequence':<30}{'frames':>7}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}"
          f"{'fps@p50':>9}  {'40 ms':<6}{'completeness'}")
    print("-" * 118)
    worst = 0.0
    fails = []
    seen = 0
    partial = []
    n_csv = 0
    n_short = 0
    for path in sorted(glob.glob(os.path.join(d, "profile_*.csv"))):
        n_csv += 1
        name = os.path.basename(path)
        # Sequence ids contain underscores, so splitting on "_" truncates them.
        # Strip the extension and the profile prefix instead.
        seq = re.sub(r"^profile_[A-Z]_", "", name)
        if seq.endswith(".csv"):
            seq = seq[:-4]
        with open(path) as fh:
            total_rows = sum(1 for _ in csv.DictReader(fh))
        rows = load_seq(path, warmup)
        if not rows:
            # Two very different situations land here and must not be conflated:
            # a CSV shorter than the warm-up window (so there is nothing to
            # measure), and a CSV that exists but failed to parse. Saying "no CSV
            # found" when one is sitting right there is simply a false statement.
            n_short += 1
            print(f"{seq:<30}  (CSV has {total_rows} frames, fewer than the "
                  f"{warmup}-frame warm-up -- nothing to measure)")
            partial.append((seq, total_rows))
            continue
        seen += 1
        # A run killed part-way (SIGKILL / OOM) leaves a SHORT csv. Its
        # percentiles are computed from whatever survived and would otherwise be
        # printed exactly like a complete run's -- a PASS built on half the data.
        # Silence here would be the single most dangerous output this file can
        # produce, so short runs are marked, not merely implied.
        complete = total_rows >= expect_total
        if not complete:
            partial.append((seq, total_rows))
        v = [float(r["total_ms"]) for r in rows]
        p50, p95, p99, mx = pct(v, 50), pct(v, 95), pct(v, 99), max(v)
        verdict = "PASS" if (p99 <= RED_LINE and mx <= RED_LINE) else "FAIL"
        if verdict == "FAIL":
            fails.append((seq, p99, mx))
        worst = max(worst, mx)
        print(f"{seq:<30}{len(v):>7}{p50:>9.2f}{p95:>9.2f}{p99:>9.2f}{mx:>9.2f}"
              f"{1000.0 / p50:>9.1f}  {verdict:<6}"
              f"{'complete' if complete else f'PARTIAL ({total_rows}/{expect_total})'}")
    print("-" * 118)
    if partial:
        print("  INCOMPLETE RUNS -- the numbers above are computed from surviving frames")
        print("  only and must not be read as a full-sequence result:")
        for seq, n in partial:
            print(f"    {seq}: {n}/{expect_total} frames")
    if n_csv == 0:
        # "No breaches" is NOT the same claim as "no breaches found". Printing a
        # PASS here because the glob matched nothing is exactly the
        # zero-observations-does-not-mean-absence error this project has been
        # bitten by twice already.
        print(f"  NO DATA: no profile CSV found under {d}. "
              "Nothing was measured, so nothing is certified.")
    elif seen == 0:
        print(f"  {n_csv} CSV(s) found but none had a usable steady-state window "
              f"(every one is shorter than the {warmup}-frame warm-up).")
        print("  Nothing was measured, so nothing is certified.")
    elif fails:
        print(f"  BREACH: {len(fails)} sequence(s) over {RED_LINE:.0f} ms -> "
              + ", ".join(f"{s} (p99 {a:.2f}, max {b:.2f})" for s, a, b in fails))
    else:
        print(f"  ALL {seen} SEQUENCES WITHIN {RED_LINE:.0f} ms at p99 AND max. "
              f"Worst single frame in the run: {worst:.2f} ms.")
    print()

    print("  stage breakdown (p50 / p99, ms)")
    print(f"  {'sequence':<30}" + "".join(f"{s:>16}" for s in STAGES))
    for path in sorted(glob.glob(os.path.join(d, "profile_*.csv"))):
        name = os.path.basename(path)
        # Sequence ids contain underscores, so splitting on "_" truncates them.
        # Strip the extension and the profile prefix instead.
        seq = re.sub(r"^profile_[A-Z]_", "", name)
        if seq.endswith(".csv"):
            seq = seq[:-4]
        # warmup MUST be passed here. It was omitted, and load_seq() has no
        # default for it, so the stage breakdown raised TypeError -- which
        # aborted main() before TABLE 2 (trigger census) and TABLE 3 (accuracy
        # ledger) ever ran. Under guarantee_run.sh's `set -e` the whole campaign
        # then exited non-zero AFTER deploying and running all three arms, so
        # the two most important delivery tables were lost to a missing argument
        # in a cosmetic sub-table.
        rows = load_seq(path, warmup)
        if not rows:
            continue
        cells = ""
        for s in STAGES:
            v = [float(r[s + "_ms"]) for r in rows]
            cells += f"{pct(v, 50):>7.2f}/{pct(v, 99):<8.2f}"
        print(f"  {seq:<30}{cells}")
    print()


def ledger_table(path, title):
    print("=" * 118)
    print(title)
    print("=" * 118)
    if not os.path.exists(path):
        print(f"  (missing: {path})")
        print()
        return
    text = open(path, errors="replace").read()
    print("  -- special-handle census --")
    print(f"  {'sequence':<30}{'cornercap':>22}{'fitgate':>10}{'chain':>8}"
          f"{'deadline':>10}{'corners':>14}")
    print("  " + "-" * 94)
    total = {"corner_frames": 0, "corner_pts": 0, "fitgate": 0, "chain": 0, "dl": 0}
    matched = 0
    for m in re.finditer(
            r"\[SEQ\]\s+(\S+).*?\| \[HANDLE\] cornercap=(\d+)f/(\d+)calls\((\d+)pts\) "
            r"fitgate=(\d+) chain=(\d+) fitdeadline=(\d+) \| corners mean=([\d.]+) "
            r"over (\d+) dets \(budget (\d+)\)", text):
        matched += 1
        seq, cf, cp, pts, fg, ch, dl, cm, dets, budget = m.groups()
        cf, cp, pts, fg, ch, dl = int(cf), int(cp), int(pts), int(fg), int(ch), int(dl)
        total["corner_frames"] += cf
        total["corner_pts"] += pts
        total["fitgate"] += fg
        total["chain"] += ch
        total["dl"] += dl
        print(f"  {seq:<30}{cf:>10}f {pts:>6}pts{fg:>10}{ch:>8}{dl:>10}"
              f"{float(cm):>9.1f}/{budget:<4}")
    print("  " + "-" * 94)
    if matched == 0:
        # A regex that silently matches nothing prints a table of zeros, which
        # reads as "the handles never fired". That is indistinguishable from the
        # real finding and is the most dangerous possible failure for this file:
        # it would certify a guarantee it never observed.
        print("  NO CENSUS ROWS PARSED. The log predates the [HANDLE] block, or the")
        print("  binary on the board is older than this report. Zero is NOT the")
        print("  reported trigger count here -- the count is unknown.")
    else:
        print(f"  {'TOTAL':<30}{total['corner_frames']:>10}f {total['corner_pts']:>6}pts"
              f"{total['fitgate']:>10}{total['chain']:>8}{total['dl']:>10}")
    print()

    print("  -- GPU vs CPU, same fits, absolute per-pixel counts --")
    print(f"  {'sequence':<30}{'ch':>4}{'Max|Diff|':>12}{'MAE':>14}{'exact px':>16}")
    print("  " + "-" * 76)
    ab_rows = 0
    # The three channels share one [A/B] line, separated by '|'. Anchoring every
    # channel on its own [A/B] would match Ch0 and nothing else, which reads as
    # "only Ch0 differs" rather than "the parser stopped".
    for line in text.splitlines():
        if "[A/B]" not in line:
            continue
        seq = line.split("[A/B]", 1)[1].split()[0]
        for m in re.finditer(r"Ch(\d) max=(\d+) MAE=([\d.]+) exact=([\d.]+)%", line):
            ch, mx, mae, ex = m.groups()
            ab_rows += 1
            print(f"  {seq:<30}{ch:>4}{int(mx):>12}{float(mae):>14.5f}{float(ex):>15.4f}%")
    if ab_rows == 0:
        print("  NO [A/B] ROWS PARSED -- the accuracy ledger is unknown, not zero.")
    print()


def _series(d, warmup):
    """Map sequence -> steady-state per-frame totals, or {} if the arm is absent.

    Reports WHY a directory yields nothing. A hardcoded warmup that does not
    match the run's --profile-warmup empties every list here, and the caller
    then says "arm ABSENT" about a directory full of CSVs -- which is exactly
    what a 100-frame run analysed with WARMUP=100 used to do.
    """
    out = {}
    n_csv = 0
    for path in sorted(glob.glob(os.path.join(d, "profile_*.csv"))):
        n_csv += 1
        name = os.path.basename(path)
        seq = re.sub(r"^profile_[A-Z]_", "", name)
        if seq.endswith(".csv"):
            seq = seq[:-4]
        with open(path) as fh:
            rows = list(csv.DictReader(fh))[warmup:]
        if rows:
            out[seq] = [float(r["total_ms"]) for r in rows]
    return out, n_csv


def attribution_table(c_dir, d_dir, warmup):
    """Which handle bought which millisecond.

    Arm C is pre-allocation ALONE (corner budget frozen at 600); arm D is the
    delivery configuration (pre-allocation plus the corner budget on
    VIDEO00036 only). The difference is therefore handle B's timing effect on
    exactly one sequence, and ~0 everywhere else -- which is the check that the
    per-sequence flag routing actually worked. A single combined run could not
    tell these apart at all.
    """
    print("=" * 118)
    print("TABLE 0  handle attribution:  arm C (prealloc only)  vs  arm D (delivery config)")
    print("=" * 118)
    C, n_c = _series(c_dir, warmup)
    D, n_d = _series(d_dir, warmup)
    if not C or not D:
        print(f"  NEEDS BOTH ARMS (warmup {warmup}):")
        for tag, series, n in (("arm C", C, n_c), ("arm D", D, n_d)):
            if series:
                print(f"    {tag}: {len(series)} sequence(s)")
            elif n:
                print(f"    {tag}: {n} CSV(s) present but all shorter than the "
                      f"{warmup}-frame warm-up -- data EXISTS, no steady window")
            else:
                print(f"    {tag}: no CSV found in {os.path.basename(c_dir if tag.endswith('C') else d_dir)}")
        print("  No attribution is computed from one arm alone.")
        print()
        return
    print(f"  {'sequence':<30}{'C p50':>9}{'D p50':>9}{'dp50%':>9}"
          f"{'C p99':>9}{'D p99':>9}{'dp99%':>9}{'C max':>9}{'D max':>9}")
    print("  " + "-" * 92)
    for seq in sorted(set(C) & set(D)):
        c, dd = C[seq], D[seq]
        print(f"  {seq:<30}{pct(c,50):>9.2f}{pct(dd,50):>9.2f}"
              f"{(pct(dd,50)-pct(c,50))/pct(c,50)*100:>8.1f}%"
              f"{pct(c,99):>9.2f}{pct(dd,99):>9.2f}"
              f"{(pct(dd,99)-pct(c,99))/max(pct(c,99),1e-9)*100:>8.1f}%"
              f"{max(c):>9.2f}{max(dd):>9.2f}")
    print("  " + "-" * 92)
    print("  Only VIDEO00036 differs by design (it is the only sequence with a corner")
    print("  budget). Large deltas on any OTHER sequence mean BENCH_EXTRA_MAP did not")
    print("  route per-sequence flags correctly -- the run is misconfigured, not faster.")
    print()


def _is_synthetic(d):
    """True if this directory holds make_fake_boards.py output.

    The dry-run harness used to write its synthetic frames straight into
    .bench_out/G{a} -- the exact paths the real board results land in. A fetch
    that fails leaves whatever is already on disk in place, so those frames
    would have been reported as RK3588 measurements. A sentinel file makes the
    distinction impossible to miss, and this function refuses to read it.
    """
    return os.path.exists(os.path.join(d, "SYNTHETIC_DO_NOT_REPORT.txt"))


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("c_dir", nargs="?", default=os.path.join(HERE, ".bench_out", "GC"))
    ap.add_argument("d_dir", nargs="?", default=os.path.join(HERE, ".bench_out", "GD"))
    ap.add_argument("e_dir", nargs="?", default=os.path.join(HERE, ".bench_out", "GE"))
    # Must match the --profile-warmup the run was given. bench_run.sh records it
    # as "### environment ... warmup=N"; mismatching it silently empties every
    # steady-state series and the tool then reports data that exists as ABSENT.
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    a = ap.parse_args()

    bad = [p for p in (a.c_dir, a.d_dir, a.e_dir) if _is_synthetic(p)]
    if bad:
        print("REFUSING TO REPORT.")
        for p in bad:
            print(f"  {p} contains SYNTHETIC_DO_NOT_REPORT.txt -- these frames were")
            print("  produced by make_fake_boards.py and never ran on the board.")
        print("  Point the report at the real .bench_out/G{a} directories.")
        sys.exit(2)

    attribution_table(a.c_dir, a.d_dir, a.warmup)
    timing_table(a.d_dir, a.warmup)
    for name, title in (
            (os.path.join(a.d_dir, "bench_D.log"),
             "TABLE 2  special-handle trigger census (delivery config, arm D)"),
            (os.path.join(a.e_dir, "bench_E.log"),
             "TABLE 3  accuracy ledger (arm E, --fused both, same fits)")):
        ledger_table(name, title)


if __name__ == "__main__":
    main()
