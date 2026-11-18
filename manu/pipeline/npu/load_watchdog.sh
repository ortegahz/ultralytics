#!/bin/bash
# 负载/内存看门狗（2026-11-16 §195 版）：分级刹车，覆盖「夜间训练轴」
#   🔴 §194 铁律：MemAvailable < 总内存 30% ⇒ 禁启新任务；<20% ⇒ 杀掉最新任务；<10% ⇒ 全部停机
#      ⇒ 阈值：<38G 不再启新臂（由 runner 闸门负责）｜<25G 杀**最新**训练臂｜<12G 全部停机
set -u
TOTAL_KB=$(awk '/MemTotal/{printf "%d", $2}' /proc/meminfo)
L30=$(( TOTAL_KB * 30 / 100 / 1048576 ))
L20=$(( TOTAL_KB * 20 / 100 / 1048576 ))
L10=$(( TOTAL_KB * 10 / 100 / 1048576 ))
L90=$(( TOTAL_KB * 90 / 100 / 1048576 ))
LOG=/tmp/load_watchdog.log
while true; do
  sleep 60
  L=$(awk '{print $1}' /proc/loadavg)
  M=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
  NPROC=$(ps -eLf | wc -l)
  # ---- 内存分级刹车 ----
  if [ "${M:-0}" -lt "${L10:-12}" ]; then
    pkill -9 -f "[o]ptuna_mix_ratio_search" 2>/dev/null
    pkill -9 -f "[c]ache_trial0474_inferences.py" 2>/dev/null
    echo "$(date '+%F %T') MEM_CRIT ${M}GB(<${L10}GB) -> 全部停机" >> "$LOG"
    sleep 300; continue
  fi
  if [ "${M:-0}" -lt "${L20:-25}" ]; then
    YOUNG=$(ps -eo etimes,args | grep "[o]ptuna_mix_ratio_search" | sort -n | head -1 | grep -o "trial-number [0-9]*" | cut -d" " -f2)
    if [ -n "${YOUNG:-}" ]; then
      pkill -9 -f "trial-number ${YOUNG} " 2>/dev/null
      echo "$(date '+%F %T') MEM_LOW ${M}GB(<${L20}GB) -> 杀最新训练臂 trial=${YOUNG}" >> "$LOG"
    fi
    pkill -9 -f "[c]ache_trial0474_inferences.py" 2>/dev/null
    sleep 120; continue
  fi
  # ---- 负载刹车（load1 > 0.9×nproc 时先杀评测派生）----
  NCPU=$(nproc --all)
  if awk "BEGIN{exit !($L > $NCPU * 0.9)}"; then
    pkill -9 -f "[c]ache_trial0474_inferences.py" 2>/dev/null
    echo "$(date '+%F %T') LOAD ${L} > 0.9*${NCPU} -> 杀评测派生" >> "$LOG"
  fi
  echo "$(date '+%F %T') 水位 mem_avail=${M}GB(30%线${L30}GB) load=${L}/${NCPU} threads=${NPROC}" >> "$LOG"
done
