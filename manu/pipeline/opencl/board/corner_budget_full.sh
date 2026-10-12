#!/usr/bin/env bash
# corner_budget_full.sh -- full 400-frame accuracy cost of --fit-corners 200,
# with the numeric-max aggregator. Supersedes the 40-frame table, which was both
# under-sampled (it missed the frames where VIDEO00032 actually truncates) and
# numerically wrong (lexicographic max).
set -u
HERE=/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/opencl/board
BIN=/tmp/gmc_stream_ocl_cap
B=$HERE/.bench_bmp
K=$HERE/../kernels
TRASH=/home/manu/.minimax/bin/mavis-trash

for S in "$@"; do
  W=/tmp/full_$S
  mkdir -p "$W/ref" "$W/b200"
  $BIN --raw-root "$B" --sequence "$S" --limit 400 --fused cpu --threads 4 \
       --kernel-dir "$K" --fit-corners 600 --dump-dir "$W/ref" > /dev/null 2>&1
  $BIN --raw-root "$B" --sequence "$S" --limit 400 --fused cpu --threads 4 \
       --kernel-dir "$K" --fit-corners 200 --dump-dir "$W/b200" > /dev/null 2>&1
  bash "$HERE/npy_diff_summary.sh" "$W/ref" "$W/b200" \
       "$S  400 frames, fit-corners 600 -> 200, x86 CPU arm"
  echo
  "$TRASH" -- "$W" > /dev/null 2>&1
done
