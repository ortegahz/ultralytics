#!/usr/bin/env bash
# v32_budget_scan.sh -- where does VIDEO00032 start to pay for a corner budget?
#
# Why this sequence needs its own scan: at budget 200 it truncates on 306 of 400
# frames (mean 217.6 corners, so it sits right on the cliff) and the cost is NOT
# small -- Ch1 max 68, Ch2 max 48 over the full 400 frames. Meanwhile its board
# p99 is 30.37 ms, i.e. it already passes the 40 ms line without any budget at
# all. Spending real accuracy on a sequence that does not need the speed is the
# wrong trade, so the budget that suits VIDEO00036 must not be imposed on it.
set -u
HERE=/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/opencl/board
BIN=/tmp/gmc_stream_ocl_cap
B=$HERE/.bench_bmp
K=$HERE/../kernels
S=${1:-VIDEO00032_19700101_014453}
TRASH=/home/manu/.minimax/bin/mavis-trash
W=/tmp/scan_$S
mkdir -p "$W/ref"
$BIN --raw-root "$B" --sequence "$S" --limit 400 --fused cpu --threads 4 \
     --kernel-dir "$K" --fit-corners 600 --dump-dir "$W/ref" > /dev/null 2>&1

for c in 400 300 250 200; do
  d="$W/b$c"
  mkdir -p "$d"
  $BIN --raw-root "$B" --sequence "$S" --limit 400 --fused cpu --threads 4 \
       --kernel-dir "$K" --fit-corners "$c" --dump-dir "$d" > /dev/null 2>&1
  cap=$($BIN --raw-root "$B" --sequence "$S" --limit 400 --fused cpu --threads 4 \
             --kernel-dir "$K" --fit-corners "$c" 2>&1 \
             | sed -n 's/.*cornercap=\([0-9]*\)f.*/\1/p' | head -1)
  bash "$HERE/npy_diff_summary.sh" "$W/ref" "$d" \
       "$S  400 frames, fit-corners 600 -> $c   (cornercap=${cap}f/400f)"
  echo
  "$TRASH" -- "$d" > /dev/null 2>&1
done
"$TRASH" -- "$W" > /dev/null 2>&1
