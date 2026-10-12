#!/usr/bin/env bash
# redeploy_verify.sh -- close the md5 gap left by a comment-only source edit.
#
# The situation: the board is running 9901bac8, which is what produced every
# reported number. Then a comment in the source was corrected, and the rebuild
# gave ddacad5b. Comments cannot change semantics, and the zero-regression run
# confirms they did not -- but 30 bytes of .text DID move (branch displacements),
# so "the deployed binary is byte-identical to the current source" is no longer
# literally true, and leaving the board on a stale md5 is exactly the kind of
# loose end that makes a later run unreproducible.
#
# This deploys the current-source binary and re-runs the delivery arm, so the
# reported numbers come from a binary that provably matches the source tree.
#
# Usage: RK3588_PASSWORD=... bash redeploy_verify.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
BIN=out/gmc_stream_ocl_arm64
REMOTE=/userdata/prof

[ -n "${RK3588_PASSWORD:-}" ] || { echo "RK3588_PASSWORD is not set" >&2; exit 2; }

LOCAL_MD5=$(md5sum "$BIN" | cut -d' ' -f1)
echo "local binary md5: $LOCAL_MD5"
python3 deploy_bench.py --bench-dir .bench_bmp --binary "$BIN" \
        --kernel-dir ../kernels --remote-root "$REMOTE" --skip-data >/dev/null 2>&1
REMOTE_MD5=$(python3 board_run.py --exec "md5sum $REMOTE/gmc_stream_ocl | cut -d' ' -f1" \
             | tr -d '\r' | grep -Eo '^[0-9a-f]{32}$' | head -1)
echo "remote binary md5: $REMOTE_MD5"
[ "$LOCAL_MD5" = "$REMOTE_MD5" ] || { echo "MD5 MISMATCH -- stop" >&2; exit 3; }

echo "== re-run delivery arm as I =="
python3 board_run.py --exec \
  "cd $REMOTE && BENCH_EXTRA='--prealloc-pad 1024 --pin-pad 1' \
     BENCH_EXTRA_MAP='VIDEO00036=--fit-corners 200' \
     BENCH_FUSED=gpu sh bench_run.sh I $REMOTE/out/I 100 400" || true

python3 fetch_results.py --remote "$REMOTE/out/I" --local ".bench_out/GI"
echo
echo "== arm I (current-source binary) vs arm F (binary that produced the report) =="
python3 board_noise_floor.py 2>/dev/null || true
python3 - "$REMOTE" <<'PY'
import csv, glob, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
W = 100
def load(d, a):
    out = {}
    for p in glob.glob(os.path.join(HERE, ".bench_out", d, f"profile_{a}_*.csv")):
        rows = list(csv.DictReader(open(p)))[W:]
        if not rows: continue
        v = sorted(float(r["total_ms"]) for r in rows)
        nr = lambda q: v[max(1, math.ceil(q/100*len(v)))-1]
        out[os.path.basename(p).split("_")[2]] = (nr(50), nr(99), v[-1],
                                                    sum(1 for x in v if x > 40.0))
    return out
F = load("GF", "F"); I = load("GI", "I")
if not I:
    print("  arm I: NO DATA"); sys.exit(0)
print(f"{'seq':<12}{'F p50':>8}{'I p50':>8}{'F p99':>8}{'I p99':>8}{'F max':>8}{'I max':>8}{'F br':>6}{'I br':>6}")
for s in ["VIDEO00034","VIDEO00028","VIDEO00032","VIDEO00002","VIDEO00036"]:
    if s not in F or s not in I:
        print(f"{s:<12}  (missing)"); continue
    f, i = F[s], I[s]
    print(f"{s:<12}{f[0]:>8.2f}{i[0]:>8.2f}{f[1]:>8.2f}{i[1]:>8.2f}{f[2]:>8.2f}{i[2]:>8.2f}{f[3]:>6}{i[3]:>6}")
tf = sum(F[s][3] for s in F); ti = sum(I[s][3] for s in I)
print(f"{'TOTAL breaches':<12}{'':>34}{tf:>6}{ti:>6}")
PY