#!/usr/bin/env bash
# ===========================================================================
# guarantee_run.sh -- the 40 ms guarantee campaign, end to end, in one command.
#
# Run this once RK3588_PASSWORD is in the environment. It does the four things
# that have to happen for the delivery numbers to exist, in order, and refuses
# to report anything it did not measure itself:
#
#   1. deploy the arm64 binary and VERIFY IT BY MD5 ON BOTH SIDES
#   2. arm C  pre-allocation only          (--prealloc-pad 1024, corners 600)
#   3. arm D  the proposed delivery config (--prealloc-pad 1024, corners 200 on
#             VIDEO00036 only -- see report section 5A.4 for why it is per-sequence)
#   4. arm E  the same, with --fused both, for the GPU-vs-CPU accuracy ledger
#   then fetch everything and print the three delivery tables
#
# WHY TWO ARMS AND NOT ONE. C isolates the pre-allocation handle: it must move
# VIDEO00034's max from 53.50 ms down without changing a single output byte.
# If C already fixes it, D is only about VIDEO00036's fit cost, and the two
# effects never get confounded. Reporting a single combined run would make it
# impossible to say which handle produced which millisecond.
#
# Arm D deliberately does NOT lower the corner budget globally. Section 5A.2
# measured VIDEO00032 at Ch1 max 68 / Ch2 max 48 -- the largest deviation in the
# corpus -- for a sequence whose p99 is already 30.37 ms. Trading real accuracy
# for speed a sequence does not need is not a fix.
#
# Nothing here trusts the board's self-reported statistics: the timing table is
# recomputed on the host from the raw per-frame CSVs.
#
# Usage: RK3588_PASSWORD=... bash guarantee_run.sh
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

BIN=out/gmc_stream_ocl_arm64
REMOTE=/userdata/prof
OUT=$REMOTE/out
SEQS_C="VIDEO00034_19700101_015052 VIDEO00028_19700101_013631 VIDEO00032_19700101_014453 \
        VIDEO00002_19700101_001415 VIDEO00036_19700101_015740"

[ -n "${RK3588_PASSWORD:-}" ] || { echo "RK3588_PASSWORD is not set" >&2; exit 2; }
[ -f "$BIN" ] || { echo "missing $BIN -- run build_gmc_ocl_arm64.sh first" >&2; exit 2; }

echo "== 1. deploy + md5 both sides =="
python3 deploy_bench.py --bench-dir .bench_bmp --binary "$BIN" \
        --kernel-dir ../kernels --remote-root "$REMOTE" --skip-data

# bench_run.sh is NOT pushed by deploy_bench.py -- that script ships the corpus,
# the binary and the kernels, nothing else. Arm C below calls it, so pushing it
# here is mandatory: without this line the campaign dies on its first command
# with "can't open board/bench_run.sh", after a full deploy has already happened.
python3 board_run.py --put "bench_run.sh:$REMOTE/bench_run.sh" || {
  echo "FAILED to push bench_run.sh" >&2; exit 3; }
python3 board_run.py --exec "sed -i 's/\r$//' $REMOTE/bench_run.sh; md5sum $REMOTE/bench_run.sh"

LOCAL_MD5=$(md5sum "$BIN" | cut -d' ' -f1)
REMOTE_MD5=$(python3 board_run.py --exec "md5sum $REMOTE/gmc_stream_ocl | cut -d' ' -f1" \
             | tr -d '\r' | grep -Eo '^[0-9a-f]{32}$' | head -1)
echo "local  $LOCAL_MD5"
echo "remote $REMOTE_MD5"
[ "$LOCAL_MD5" = "$REMOTE_MD5" ] || { echo "MD5 MISMATCH -- stop" >&2; exit 3; }

# Arm C: pre-allocation only. Corners stay at the frozen 600 so the only moving
# part is the device reallocation.
echo "== 2. arm C: prealloc=1024, corners 600 =="
python3 board_run.py --exec \
  "cd $REMOTE && BENCH_EXTRA='--prealloc-pad 1024' sh bench_run.sh C $OUT/C 100 400" \
  || true

# Arms D and E go through bench_run.sh rather than a hand-rolled loop, and that
# is not cosmetic. bench_run.sh carries the retry/backoff that this board needs:
# its own header records that a run can be SIGKILLed instantly at start-up while
# MemAvailable still shows 7.4 GiB, because the Mali driver's device pool is not
# reflected in /proc/meminfo, and that "backing off and retrying is the reliable
# fix; a fixed sleep is a guess at the same thing". A loop written here would
# have none of that, and a sequence killed once would stay dead -- leaving a
# short CSV that the report would otherwise have to flag as PARTIAL.

# Arm D: the proposed delivery configuration. Pre-allocation everywhere; the
# corner budget ONLY on VIDEO00036, which is the one sequence whose p99 (42.88 ms)
# is actually over the line. See report section 5A.4 for why it is per-sequence.
echo "== 3. arm D: prealloc=1024 globally, --fit-corners 200 on VIDEO00036 only =="
python3 board_run.py --exec \
  "cd $REMOTE && BENCH_EXTRA='--prealloc-pad 1024' \
     BENCH_EXTRA_MAP='VIDEO00036=--fit-corners 200' \
     BENCH_FUSED=gpu sh bench_run.sh D $OUT/D 100 400" || true

# Arm E: same configuration, both arms, for the accuracy ledger.
echo "== 4. arm E: fused both, accuracy ledger =="
python3 board_run.py --exec \
  "cd $REMOTE && BENCH_EXTRA='--prealloc-pad 1024' \
     BENCH_EXTRA_MAP='VIDEO00036=--fit-corners 200' \
     BENCH_FUSED=both sh bench_run.sh E $OUT/E 100 400" || true

echo "== 5. fetch =="
for a in C D E; do
  python3 fetch_results.py --remote "$OUT/$a" --local ".bench_out/G$a" || \
    echo "   (fetch $a failed)"
done

echo
echo "== 6. delivery tables =="
python3 guarantee_report.py .bench_out/GC .bench_out/GD .bench_out/GE --warmup 100
