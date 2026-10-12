#!/usr/bin/env bash
# ===========================================================================
# contended_board.sh -- does the 40 ms guarantee survive system interference?
#
# bench_run.sh's experiment B pins a spin loop to cpu0-3 (the little A55 cores)
# and leaves the pipeline alone on cpu4-7 (the A76 cores it is pinned to). The
# question is not "is it slower" but "does p99/max break 40 ms".
#
# Why this is the right next experiment and not just another data point:
#
#   The original brief asked for a runtime fuse "若遭遇未知的系统抢占". Handle C
#   (--fit-deadline-ms) is that fuse, and it is currently OFF by default because
#   a wall-clock trigger makes the output depend on machine load. That decision
#   has been argued from first principles this whole time. This run settles it
#   with a measurement: if the delivery config holds under load, the fuse stays
#   off and the bit-reproducibility contract is intact; if it breaks, the fuse
#   has a real job to do and the cost of turning it on stops being hypothetical.
#
# bench_run.sh triggers the load on the arm letter being exactly "B", so the
# arm directory is named to keep the artefact path distinct from the old
# gate-version experiment-B data.
#
# Usage: RK3588_PASSWORD=... bash contended_board.sh
# ===========================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
BIN=out/gmc_stream_ocl_arm64
REMOTE=/userdata/prof

[ -n "${RK3588_PASSWORD:-}" ] || { echo "RK3588_PASSWORD is not set" >&2; exit 2; }

LOCAL_MD5=$(md5sum "$BIN" | cut -d' ' -f1)
REMOTE_MD5=$(python3 board_run.py --exec "md5sum $REMOTE/gmc_stream_ocl | cut -d' ' -f1" \
             | tr -d '\r' | grep -Eo '^[0-9a-f]{32}$' | head -1)
echo "local  $LOCAL_MD5"
echo "remote $REMOTE_MD5"
[ "$LOCAL_MD5" = "$REMOTE_MD5" ] || { echo "MD5 MISMATCH -- stop" >&2; exit 3; }

echo "== delivery config + background load on cpu0-3 (experiment B) =="
python3 board_run.py --exec \
  "cd $REMOTE && BENCH_EXTRA='--prealloc-pad 1024 --pin-pad 1' \
     BENCH_EXTRA_MAP='VIDEO00036=--fit-corners 200' \
     BENCH_FUSED=gpu sh bench_run.sh B $REMOTE/out/JB 100 400" || true

echo "== fetch =="
python3 fetch_results.py --remote "$REMOTE/out/JB" --local ".bench_out/GJB"

echo
echo "== contended timing (arm JB) =="
python3 guarantee_report.py .bench_out/GJB .bench_out/GJB .bench_out/GJB --warmup 100