#!/bin/sh
# mem_where.sh -- WHICH mapping grows. Theory has already been wrong once here:
# the GPU image set is only ~21 MiB total, so it cannot explain a 2.6 GiB
# per-sequence step. /proc/PID/smaps names the culprit instead of guessing:
# a driver-side leak shows up as a huge anonymous/file mapping owned by the Mali
# libraries, a heap leak as [heap], a thread-arena effect as many 64 MiB regions.
set -u
SEQ1=$1; SEQ2=$2
OUT=/userdata/prof/out/mem
mkdir -p $OUT

/userdata/prof/gmc_stream_ocl \
  --raw-root /userdata/prof/bmp --sequence "$SEQ1" --sequence "$SEQ2" --limit 400 \
  --fused gpu --pyr-cache --threads 4 --affinity a76 --hw-linear 0 \
  --kernel-dir /userdata/prof/kernels --timing > $OUT/where.log 2>&1 &
PID=$!

i=0
while kill -0 $PID 2>/dev/null; do
  RSS=$(awk '/^VmRSS:/{print $2}' /proc/$PID/status 2>/dev/null)
  if [ -n "$RSS" ] && [ "$RSS" -gt 400000 ]; then
    echo "=== RSS=$((RSS/1024)) MiB at t=${i}s : top mappings ===" >> $OUT/where_smaps.log
    awk '/^[0-9a-f]+-[0-9a-f]+ /{addr=$1; perm=$2; name=$6}
         /^Rss:/{if ($2 > 4096) printf "%10d kB  %s %s\n", $2, perm, (name==""?"[anon]":name)}' \
        /proc/$PID/smaps 2>/dev/null | sort -rn | head -12 >> $OUT/where_smaps.log
  fi
  i=$((i+1))
  sleep 2
done
wait $PID
echo "rc=$?"
echo "--- top mappings captured while RSS > 400 MiB ---"
cat $OUT/where_smaps.log 2>/dev/null | head -60