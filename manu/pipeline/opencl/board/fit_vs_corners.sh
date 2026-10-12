#!/usr/bin/env bash
# fit_vs_corners.sh -- MEASURED fit cost against the corner budget.
#
# Why this replaces the earlier cross-sequence regression: that one had to compare
# five DIFFERENT sequences, so per-sequence LK difficulty was mixed into the
# corner count and the fit only reached R^2 = 0.706. Here the sequence is fixed
# and only the budget moves, so the confound is gone by construction.
#
# Caveat that does not go away: this runs on x86, the delivery target is an A76.
# Absolute milliseconds are NOT transferable. The RATIO between budgets is the
# portable part, and even that is an estimate until measured on the board.
#
# --fused cpu is deliberate: the warp and median stages then run on the CPU and
# their cost swamps the number, so only the fit= field is read. The fit itself is
# CPU work on both boards and is the only stage the budget touches.
set -u
HERE=/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/opencl/board
BIN=/tmp/gmc_stream_ocl_cap
B=$HERE/.bench_bmp
K=$HERE/../kernels
FRAMES=${FRAMES:-100}
OUT=/tmp/fit_vs_corners.tsv
: > "$OUT"

for s in "$@"; do
  for c in 600 400 300 250 200 150 100; do
    $BIN --raw-root "$B" --sequence "$s" --limit "$FRAMES" --fused cpu --threads 4 \
         --kernel-dir "$K" --fit-corners "$c" --timing > /tmp/fit_vs_corners.log 2>&1
    fit=$(sed -n 's/.*\[SEQTIMING\].*fit=\([0-9.]*\).*/\1/p' /tmp/fit_vs_corners.log | head -1)
    cc=$(sed -n 's/.*\[HANDLE\] cornercap=\([0-9]*\)f\/\([0-9]*\)calls(\([0-9]*\)pts\).*/\1 \2 \3/p' \
         /tmp/fit_vs_corners.log | head -1)
    printf '%s\t%s\t%s\t%s\n' "$s" "$c" "${fit:-NA}" "${cc:-NA}" >> "$OUT"
    printf '  %-30s budget=%-4s fit=%-9s cap=%s\n' "$s" "$c" "${fit:-NA}" "${cc:-NA}"
  done
done
echo "wrote $OUT"
