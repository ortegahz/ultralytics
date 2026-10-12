#!/usr/bin/env bash
# ===========================================================================
# pin_pad_board.sh -- board validation of --pin-pad, plus the noise floor that
# was still missing.
#
# Three arms, all 5 sequences x 400 frames:
#
#   F  delivery config + --pin-pad 1          the fix. Isolates pad pinning from
#                                            everything else arm D already has.
#   G  same config, --fused both              accuracy ledger for the fix.
#   H  arm D config REPEATED on the new binary
#                                            Two jobs at once:
#                                              1. --pin-pad defaults OFF, so H
#                                                 must reproduce D. That is the
#                                                 zero-regression check, run on
#                                                 the BOARD rather than on x86.
#                                              2. D vs H is the same config twice,
#                                                 which is the run-to-run noise
#                                                 floor that report section 7
#                                                 has been listing as "not done".
#
# Usage: RK3588_PASSWORD=... bash pin_pad_board.sh
# ===========================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
BIN=out/gmc_stream_ocl_arm64
REMOTE=/userdata/prof
OUT=$REMOTE/out

[ -n "${RK3588_PASSWORD:-}" ] || { echo "RK3588_PASSWORD is not set" >&2; exit 2; }
[ -f "$BIN" ] || { echo "missing $BIN" >&2; exit 2; }

echo "== 1. deploy + md5 both sides =="
python3 deploy_bench.py --bench-dir .bench_bmp --binary "$BIN" \
        --kernel-dir ../kernels --remote-root "$REMOTE" --skip-data
python3 board_run.py --put "bench_run.sh:$REMOTE/bench_run.sh"
python3 board_run.py --exec "sed -i 's/\r$//' $REMOTE/bench_run.sh"

LOCAL_MD5=$(md5sum "$BIN" | cut -d' ' -f1)
REMOTE_MD5=$(python3 board_run.py --exec "md5sum $REMOTE/gmc_stream_ocl | cut -d' ' -f1" \
             | tr -d '\r' | grep -Eo '^[0-9a-f]{32}$' | head -1)
echo "local  $LOCAL_MD5"
echo "remote $REMOTE_MD5"
[ "$LOCAL_MD5" = "$REMOTE_MD5" ] || { echo "MD5 MISMATCH -- stop" >&2; exit 3; }

DELIVERY_MAP="VIDEO00036=--fit-corners 200"

run_arm() {
  arm=$1; fused=$2; extra=$3
  echo "== arm $arm : fused=$fused extra=$extra =="
  python3 board_run.py --exec \
    "cd $REMOTE && BENCH_EXTRA='$extra' BENCH_EXTRA_MAP='$DELIVERY_MAP' \
       BENCH_FUSED=$fused sh bench_run.sh $arm $OUT/$arm 100 400" || true
}

run_arm F gpu "--prealloc-pad 1024 --pin-pad 1"
run_arm G both "--prealloc-pad 1024 --pin-pad 1"
run_arm H gpu "--prealloc-pad 1024"

echo "== fetch =="
for a in F G H; do
  python3 fetch_results.py --remote "$OUT/$a" --local ".bench_out/G$a" || \
    echo "   (fetch $a failed)"
done

echo
echo "== arm F (pin-pad) timing + census =="
python3 guarantee_report.py .bench_out/GH .bench_out/GF .bench_out/GG --warmup 100