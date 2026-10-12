#!/usr/bin/env bash
# simd_corner_portability.sh -- does the corner census survive an architecture change?
#
# The verifier's standing objection was: the trigger counts in this report are x86
# numbers, and nobody has shown they would hold on the board's NEON. The leading
# suspect was IPP (now excluded on both sides -- see report section 7). What is
# left is the SIMD path: same algorithm, different vector width, different
# floating-point rounding.
#
# OpenCV lets us toggle exactly that without changing the binary:
# OPENCV_CPU_DISABLE=<features> forces the scalar path. If corner DETECTION is
# insensitive to the toggle, the census is SIMD-insensitive and should transfer.
#
# MEASUREMENT TRAP, and why the budget here is 5000 and not 200. With the
# delivery budget of 200 the reported mean pins at exactly 200.0 on every frame,
# because the value is censored by truncation -- the true count is >=201 and is
# simply not observable. Comparing two censored series at their ceiling proves
# nothing. The budget must exceed the real corner count for the comparison to
# mean anything, and the real counts here are ~1250 (see the output).
#
# What this CANNOT settle: NEON itself. Disabling SIMD on x86 is evidence that
# the count is insensitive to vector width in general, not proof that aarch64
# produces the same number. The board remains the only authority.
set -u
HERE=/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/opencl/board
BIN=${BIN:-/tmp/gmc_stream_ocl_cap}
B=$HERE/.bench_bmp
K=$HERE/../kernels
BUDGET=${BUDGET:-5000}      # must exceed the true corner count or the data is censored
FRAMES=${FRAMES:-150}

# run <on|off> <logfile> [extra args...]
# NOTE the shift: an earlier version used "$@" unshifted, so "on" and the logfile
# path were handed to the binary as sequence arguments. Both runs then died on
# "unknown argument", both logs stayed empty, and the comparison printed YES --
# a broken run rendered identical to a passing one. shift 2 fixes the argument
# list; the non-empty check below stops that failure mode from ever returning.
run() {
  _mode=$1; _log=$2
  shift 2
  if [ "$_mode" = off ]; then
    OPENCV_CPU_DISABLE=AVX2,AVX,AVX_512_SKX,FMA3,SSSE3,SSE4_2,SSE4_1,SSE3 \
      "$BIN" --raw-root "$B" --limit "$FRAMES" --fused cpu --threads 4 \
             --kernel-dir "$K" --fit-corners "$BUDGET" "$@" > "$_log" 2>&1
  else
    "$BIN" --raw-root "$B" --limit "$FRAMES" --fused cpu --threads 4 \
           --kernel-dir "$K" --fit-corners "$BUDGET" "$@" > "$_log" 2>&1
  fi
  if ! grep -q '\[SEQ\]' "$_log"; then
    echo "FATAL: the SIMD=$_mode run produced no [SEQ] line -- it did not run." >&2
    echo "       last 3 lines of $_log:" >&2
    tail -3 "$_log" >&2
    exit 3
  fi
}

SEQS="$@"
for mode in on off; do
  args=""
  for s in $SEQS; do args="$args --sequence $s"; done
  # shellcheck disable=SC2086
  run "$mode" "/tmp/simd_$mode.log" $args
done

printf '%-30s %-22s %-22s %s\n' sequence "SIMD on" "SIMD off" same?
for s in $SEQS; do
  o=$(grep "SEQ\] $s" /tmp/simd_on.log | grep -o 'corners mean=[0-9.]* over [0-9]* dets')
  f=$(grep "SEQ\] $s" /tmp/simd_off.log | grep -o 'corners mean=[0-9.]* over [0-9]* dets')
  [ "$o" = "$f" ] && verdict="YES" || verdict="*** NO ***"
  printf '%-30s %-22s %-22s %s\n' "$s" "${o#corners mean=}" "${f#corners mean=}" "$verdict"
done
echo
echo "output md5 (SIMD on  vs off):"
diff <(grep -o 'md5=[0-9a-f]*' /tmp/simd_on.log) <(grep -o 'md5=[0-9a-f]*' /tmp/simd_off.log) \
  > /dev/null && echo "  identical" || echo "  DIFFERENT -- expected; only per-pixel counting is valid cross-arch (report 6.3)"
