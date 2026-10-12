#!/usr/bin/env bash
# corner_budget_accuracy.sh -- how much output does the corner budget change?
#
# The pre-allocation and the fit gate are output-neutral: they either do nothing
# or they reject a transform that was already garbage. The corner budget is NOT
# output-neutral -- it silently drops points from a fit that was otherwise fine.
# So it is the one handle whose cost has to be measured rather than argued about.
#
# Reference = the shipped default (--fit-corners 600). Both arms run --fused cpu,
# because this question is about the transform estimation itself and the CPU arm
# is the one that must stay bit-comparable with gmc_stream.cpp.
set -u
HERE=/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/opencl/board
BIN=/tmp/gmc_stream_ocl_cap
B=$HERE/.bench_bmp
K=$HERE/../kernels
WORK=/tmp/corner_acc
FRAMES=40

SEQS="VIDEO00034_19700101_015052 VIDEO00028_19700101_013631 VIDEO00032_19700101_014453 \
      VIDEO00002_19700101_001415 VIDEO00036_19700101_015740"

mkdir -p "$WORK"
for s in $SEQS; do
  for cfg in ref 300 200; do
    d=$WORK/$cfg/$s
    [ -d "$d" ] && continue
    mkdir -p "$d"
    n=$cfg
    [ "$cfg" = ref ] && n=600
    $BIN --raw-root "$B" --sequence "$s" --limit $FRAMES --fused cpu --threads 4 \
         --kernel-dir "$K" --fit-corners $n --dump-dir "$d" > "$WORK/$cfg-$s.log" 2>&1 \
      || echo "RUN FAILED $cfg $s"
  done
done
echo "dumps ready"
