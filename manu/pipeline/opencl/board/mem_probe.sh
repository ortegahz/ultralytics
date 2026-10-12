#!/bin/sh
# mem_probe.sh -- where does the board pipeline's memory go, and does it GROW?
#
# WHY MEASURE RATHER THAN REASON
#   The first full run was SIGKILLed by the global OOM killer with gmc_stream_ocl
#   at ~7.4 GiB RSS on an 8 GiB board, and mean per-frame total time climbing from
#   16 ms at frame 50 to 52 ms at frame 100. Both symptoms point at unbounded
#   growth, but WHICH structure grows is not guessable -- there are four plausible
#   ones (pyramid cache ring, CL image slots, the resize of each frame, or the
#   transform map). This samples RSS per frame group so the growth is attributed
#   by measurement instead of by intuition, which is the only reliable way given
#   the project's standing lesson about optimising the wrong stage.
#
# Usage: sh mem_probe.sh <seq> <limit>
set -u
SEQ=$1
LIMIT=$2
OUT=/userdata/prof/out/mem
mkdir -p $OUT

# /usr/bin/time is NOT installed on this board (verified: exit 127), so peak RSS
# comes from the sampler below instead. Sampling is coarse but the growth here is
# over minutes, so 1 Hz is ample to attribute it.
/userdata/prof/gmc_stream_ocl \
  --raw-root /userdata/prof/bmp --sequence "$SEQ" --limit "$LIMIT" --fused gpu \
  --pyr-cache --threads 4 --affinity a76 --hw-linear 0 \
  --kernel-dir /userdata/prof/kernels --timing \
  > $OUT/mem_${SEQ}_${LIMIT}.log 2>&1 &
PID=$!

PEAK=0
echo "sampling RSS of $PID"
i=0
while kill -0 $PID 2>/dev/null; do
  if [ -r /proc/$PID/status ]; then
    rss=$(awk '/^VmRSS:/{print $2}' /proc/$PID/status 2>/dev/null)
    vsz=$(awk '/^VmSize:/{print $2}' /proc/$PID/status 2>/dev/null)
    echo "sample $i rss_kb=$rss vsz_kb=$vsz" >> $OUT/rss_${SEQ}_${LIMIT}.log
    [ -n "$rss" ] && [ "$rss" -gt "$PEAK" ] && PEAK=$rss
  fi
  i=$((i+1))
  sleep 1
done
wait $PID
echo "exit=$? samples=$i peak_rss_kb=$PEAK peak_rss_MiB=$((PEAK/1024))"
echo "--- first/last RSS samples ---"
head -3 $OUT/rss_${SEQ}_${LIMIT}.log
tail -3 $OUT/rss_${SEQ}_${LIMIT}.log
echo "--- cache telemetry ---"
grep -E "\[CACHE\]|\[SEQ\]|\[TIMING\]" $OUT/mem_${SEQ}_${LIMIT}.log | tail -12