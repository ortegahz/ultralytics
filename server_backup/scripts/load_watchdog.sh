#!/bin/bash
# 负载/内存看门狗：MemAvailable<8GB ⇒ 杀全部策展；load1>48 且在跑臂>3 ⇒ 杀 heatmap/cache 派生
while true; do
  sleep 60
  L=$(awk '{print $1}' /proc/loadavg)
  M=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
  if [ "${M:-0}" -lt 8 ]; then
    pkill -9 -f "curated_[l]oop.sh"; pkill -9 -f "basin_soup_[t]argeted.py"
    pkill -9 -f "multiscale_heatmap_[c]ache.py"; pkill -9 -f "cache_trial0474_[i]nferences.py"
    pkill -9 -f "dual_constraint_[g]ate.py"
    echo "$(date '+%F %T') MEM_CRIT ${M}GB -> 杀全部策展通道" >> /tmp/load_watchdog.log
    sleep 900; continue
  fi
  if awk "BEGIN{exit !($L > 48)}"; then
    N=$(pgrep -f "[b]asin_soup_targeted.py" | wc -l)
    if [ "$N" -gt 3 ]; then
      pkill -9 -f "multiscale_heatmap_[c]ache.py"; pkill -9 -f "cache_trial0474_[i]nferences.py"
      echo "$(date '+%F %T') LOAD ${L} MEM ${M}GB -> 杀掉 heatmap/cache 派生" >> /tmp/load_watchdog.log
    fi
  fi
done
