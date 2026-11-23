#!/usr/bin/env bash
# 常驻内存看门狗：每 60s 采样 free/load，按 AGENTS.md §〇之四 分级刹车
#   MemAvailable >= 30% RAM : 正常
#   MemAvailable <  20% RAM : 杀最新 worker（收缩并发）
#   MemAvailable <  10% RAM : 停掉整个搜索
LOG=/mnt/data/siping/runs/trial0080_onnx/search/watchdog.log
TOTAL=$(free -m | awk '/^Mem:/{print $2}')
echo "[watchdog] start $(date +%H:%M:%S) total=${TOTAL}MB" >> "$LOG"
while true; do
  sleep 60
  AVAIL=$(free -m | awk '/^Mem:/{print $7}')
  PCT=$(awk -v a="$AVAIL" -v t="$TOTAL" 'BEGIN{printf "%.1f", a*100/t}')
  LOAD=$(cut -d' ' -f1 /proc/loadavg)
  echo "[watchdog] $(date +%H:%M:%S) avail=${AVAIL}MB (${PCT}%) load1=${LOAD}" >> "$LOG"
  LOW=$(awk -v p="$PCT" 'BEGIN{print (p<10)?1:0}')
  MID=$(awk -v p="$PCT" 'BEGIN{print (p<20 && p>=10)?1:0}')
  if [ "$LOW" = "1" ]; then
    echo "[watchdog] !!! avail ${PCT}% < 10% -> STOP search" >> "$LOG"
    pkill -f "rt_fps_jitter_search.py" >> "$LOG" 2>&1
    pkill -f "watchdog_mem.sh" >> "$LOG" 2>&1
    exit 0
  fi
  if [ "$MID" = "1" ]; then
    NEWEST=$(pgrep -f "rt_fps_jitter_search.py" | sort -n | tail -1)
    if [ -n "$NEWEST" ]; then
      echo "[watchdog] avail ${PCT}% < 20% -> kill newest worker pid=$NEWEST" >> "$LOG"
      kill -9 "$NEWEST" >> "$LOG" 2>&1
    fi
  fi
done
