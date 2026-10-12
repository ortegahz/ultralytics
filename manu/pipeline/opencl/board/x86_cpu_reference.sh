#!/usr/bin/env bash
# x86 CPU reference: the SAME BMP corpus the board consumed, --fused cpu.
#
# This is the cross-machine anchor for the accuracy reconciliation. The board
# runs the Mali fused tail; this runs the pure-OpenCV CPU path that gmc_stream.cpp
# was validated against. Same pixels in (BMP is byte-identical by construction),
# same frozen parameters, two machines. If the per-frame MD5s match, the board's
# delivered channels are the same bytes as the desktop reference -- which no
# in-process A/B can establish on its own.
set -uo pipefail

BMP=/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/opencl/board/.bench_bmp
K=/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/opencl/kernels
BIN=/tmp/gmc_stream_ocl_profiled
OUT=/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/opencl/board/.bench_out
LIMIT=${1:-400}

mkdir -p "$OUT"
: > "$OUT/md5_x86_cpu.txt"

SEQS="VIDEO00034_19700101_015052 \
      VIDEO00028_19700101_013631 \
      VIDEO00032_19700101_014453 \
      VIDEO00002_19700101_001415 \
      VIDEO00036_19700101_015740"
ARGS=""
for s in $SEQS; do ARGS="$ARGS --sequence $s"; done

# --fused cpu: no OpenCL needed, so this also runs on a host without a GPU ICD.
"$BIN" --raw-root "$BMP" $ARGS --limit "$LIMIT" --fused cpu \
       --pyr-cache --threads 4 --affinity none --hw-linear 0 \
       --kernel-dir "$K" --md5-out "$OUT/md5_x86_cpu.txt" --timing \
       > "$OUT/x86_cpu.log" 2>&1
echo "rc=$? lines=$(wc -l < "$OUT/md5_x86_cpu.txt")"
grep -E "\[SEQ\]|\[MD5-SUM\]" "$OUT/x86_cpu.log" | tail -8