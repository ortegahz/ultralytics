#!/bin/sh
# mem_where2.sh -- aggregate smaps by owner. The previous probe filtered to
# mappings > 4 MiB and so reported ~200 MiB against a 5 GiB RSS: the memory was
# there, just split across thousands of small mappings the filter discarded.
# This one counts and sums EVERY mapping, grouped by owner, which is the shape a
# driver that allocates per-object (rather than per-buffer) produces.
set -u
OUT=/userdata/prof/out/mem
mkdir -p $OUT

/userdata/prof/gmc_stream_ocl \
  --raw-root /userdata/prof/bmp \
  --sequence VIDEO00034_19700101_015052 --sequence VIDEO00028_19700101_013631 \
  --limit 400 --fused gpu --pyr-cache --threads 4 --affinity a76 --hw-linear 0 \
  --kernel-dir /userdata/prof/kernels --timing > $OUT/where2.log 2>&1 &
PID=$!

i=0
while kill -0 $PID 2>/dev/null; do
  RSS=$(awk '/^VmRSS:/{print $2}' /proc/$PID/status 2>/dev/null)
  if [ -n "$RSS" ] && [ "$RSS" -gt 1500000 ]; then
    echo "=== RSS=$((RSS/1024)) MiB t=${i}s ===" >> $OUT/where2_smaps.log
    awk '/^[0-9a-f]+-[0-9a-f]+ /{
           n=split($0,a," "); name=""; perm=a[2];
           for (j=6; j<=NF; j++) if ($j != "") name=name " " $j;
           if (name == "") name = "[anon]"; cur=name;
           cnt[cur]++; perm2[cur]=perm }
         /^Size:/{sz[cur]=$2}
         /^Rss:/{rss[cur]+=$2; total+=$2}
         END{
           printf "  total_rss_from_smaps = %d kB\n", total;
           n=0;
           for (k in rss) if (rss[k] > 1024) {
             printf "  %10d kB  n=%-6d vsz=%-10d kB  %s %s\n", rss[k], cnt[k], sz[k], perm2[k], k;
             n++; if (n >= 8) break;
           }
         }' /proc/$PID/smaps 2>/dev/null >> $OUT/where2_smaps.log
    echo "--- mapping count / size histogram ---" >> $OUT/where2_smaps.log
    awk '/^Size:/{s=$2; if (s < 1024) b["<1MB"]++; else if (s<4096) b["1-4MB"]++; else b[">=4MB"]++}
         END{for (k in b) printf "  %-8s : %d\n", k, b[k]}' /proc/$PID/smaps 2>/dev/null >> $OUT/where2_smaps.log
  fi
  i=$((i+1))
  sleep 3
done
wait $PID
echo "rc=$?"
tail -40 $OUT/where2_smaps.log