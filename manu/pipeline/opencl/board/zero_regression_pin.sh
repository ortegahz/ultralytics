#!/usr/bin/env bash
# ===========================================================================
# zero_regression_pin.sh -- is --pin-pad output-neutral?
#
# TWO comparisons, because they answer different questions:
#
#   (a) OLD binary, default flags        vs NEW binary, default flags
#       Proves the source change itself is inert. If this fails, the edit broke
#       something before pin-pad is even switched on.
#
#   (b) OLD binary, default flags        vs NEW binary, --prealloc-pad 1024 --pin-pad 1
#       Proves pin-pad changes no output byte. THIS is the one that matters:
#       pin-pad is only defensible if it is free in accuracy, because what it
#       buys is 2.5 ms of p50 against a 43 ms breach.
#
# BOTH ARMS ARE COMPARED, always. The CPU arm's per-frame MD5 is blind by
# construction -- it never touches the device images -- so a wrong-offset or
# stale-slot bug is invisible there and shows up only as a degraded GPU-vs-CPU
# A/B score. That exact bug happened once already in this project (46.55% ->
# 12.86% with the CPU arm completely unmoved), so a CPU-only check is not a
# check.
#
# Usage: bash zero_regression_pin.sh [OLD_BIN] [NEW_BIN]
# ===========================================================================
set -u
HERE=/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/opencl/board
OLD=${1:-/tmp/gmc_stream_ocl_cap}
NEW=${2:-/tmp/gmc_pin}
B=$HERE/.bench_bmp
K=$HERE/../kernels
FRAMES=50
SEQS="VIDEO00034_19700101_015052 VIDEO00036_19700101_015740"

echo "OLD $OLD  md5=$(md5sum "$OLD" | cut -d' ' -f1)"
echo "NEW $NEW  md5=$(md5sum "$NEW" | cut -d' ' -f1)"
echo

for SEQ in $SEQS; do
  W=/tmp/zr_$SEQ
  rm -rf "$W"; mkdir -p "$W"

  echo "=== $SEQ : (a) NEW default vs OLD ==="
  "$OLD" --raw-root "$B" --sequence "$SEQ" --limit $FRAMES --fused both --threads 4 \
         --kernel-dir "$K" --md5-out "$W/old.md5" > "$W/old.log" 2>&1
  "$NEW" --raw-root "$B" --sequence "$SEQ" --limit $FRAMES --fused both --threads 4 \
         --kernel-dir "$K" --md5-out "$W/new.md5" > "$W/new.log" 2>&1

  echo "=== $SEQ : (b) NEW --pin-pad vs OLD ==="
  "$NEW" --raw-root "$B" --sequence "$SEQ" --limit $FRAMES --fused both --threads 4 \
         --kernel-dir "$K" --prealloc-pad 1024 --pin-pad 1 \
         --md5-out "$W/pin.md5" > "$W/pin.log" 2>&1

  for arm in new pin; do
    same=$(cmp -s <(sort "$W/old.md5") <(sort "$W/$arm.md5") && echo YES || echo NO)
    n=$(wc -l < "$W/old.md5")
    echo "  CPU arm per-frame MD5  OLD vs $arm : $same  ($n frames)"
  done

  echo "  GPU arm A/B (absolute per-pixel, six numbers must be identical):"
  for arm in old new pin; do
    printf '    %-4s ' "$arm"
    grep -o 'Ch[0-2] max=[0-9]* MAE=[0-9.]* exact=[0-9.]*%' "$W/$arm.log" | tr '\n' ' '
    echo
  done
  echo
done

rm -rf /tmp/zr_VIDEO00034_19700101_015052 /tmp/zr_VIDEO00036_19700101_015740
echo "If any line above says NO, or any arm's six numbers differ from OLD,"
echo "do NOT deploy: pin-pad is not output-neutral on this input."