#!/bin/sh
# mem_probe_multi.sh -- is the memory per-sequence or per-run?
#
# A single sequence peaks at ~207 MiB and stays flat, yet the 5-sequence run was
# OOM-killed at ~7.4 GiB. The structure that survives a sequence boundary is the
# FusedGpu object (created once, shared by every sequence) -- so the question is
# whether something inside IT accumulates. Running N sequences in ONE process and
# watching RSS between them answers it directly: flat => per-sequence leak in the
# pipeline object; climbing per sequence => leak in the shared GPU object.
#
# Usage: sh mem_probe_multi.sh N SEQ1 SEQ2 ...
set -u
N=$1; shift
ARGS=""
for s in "$@"; do ARGS="$ARGS --sequence $s"; done

OUT=/userdata/prof/out/mem
mkdir -p $OUT
LOG=$OUT/multi_${N}.log

: > $OUT/rss_multi_${N}.log
/userdata/prof/gmc_stream_ocl \
  --raw-root /userdata/prof/bmp $ARGS --limit 400 --fused gpu \
  --pyr-cache --threads 4 --affinity a76 --hw-linear 0 \
  --kernel-dir /userdata/prof/kernels --timing > $LOG 2>&1 &
PID=$!

PEAK=0
i=0
while kill -0 $PID 2>/dev/null; do
  rss=$(awk '/^VmRSS:/{print $2}' /proc/$PID/status 2>/dev/null)
  [ -n "$rss" ] && echo "t=${i}s rss_kb=$rss rss_MiB=$((rss/1024))" >> $OUT/rss_multi_${N}.log
  [ -n "$rss" ] && [ "$rss" -gt "$PEAK" ] && PEAK=$rss
  i=$((i+1))
  sleep 2
done
wait $PID
RC=$?
echo "rc=$RC samples=$((i/2)) peak_MiB=$((PEAK/1024))"
echo "--- RSS trace (2s samples) ---"
awk 'NR%3==1' $OUT/rss_multi_${N}.log
echo "--- per-sequence completion ---"
grep -E "\[SEQ\]|\[CACHE\]" $LOG