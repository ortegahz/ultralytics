#!/bin/bash
# 重启周末轮看门狗（独立脚本，避免长命令行/自匹配问题）
set -uo pipefail
PAT="npu20ms_watchd"
PAT="${PAT}og.sh"

# 1) 停掉旧看门狗（用 PID 文件/精确匹配，不匹配自身命令行）
for p in $(ps -eo pid,args | grep -F "npu20ms_watch" | grep -v grep | awk '{print $1}'); do
  [ "$p" = "$$" ] && continue
  kill -TERM "$p" 2>/dev/null
done
sleep 2

cd /tmp/pycharm_project_10ae9e2e || exit 1
setsid nohup env \
  OUT_ROOT=/mnt/data/siping/runs/npu20ms \
  STUDY_NAME=mix_ratio_search_npu20ms_v2 \
  SPEED_TABLE=/tmp/pycharm_project_10ae9e2e/manu/pipeline/npu/struct_speed_table.json \
  LIMIT_MS=21.0 T3_DROP_MAX_PP=1.0 T1_DROP_MAX_PP=0.5 T2_FP_MAX=10 \
  bash manu/training/npu20ms_watchdog.sh > /dev/null 2>&1 &
sleep 8

echo "=== 看门狗进程 ==="
ps -eo pid,args | grep -F "npu20ms_watch" | grep -v grep | head -3
echo "=== watchdog.log ==="
tail -4 /mnt/data/siping/runs/npu20ms/logs/watchdog.log 2>/dev/null
echo "=== dispatcher ==="
cat /mnt/data/siping/runs/npu20ms/dispatcher.pid 2>/dev/null; echo
ps -eo pid,args | grep -F "optuna_mix_ratio_search.py --output-root" | grep -v grep | head -1 | cut -c1-90
