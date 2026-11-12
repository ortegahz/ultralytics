#!/bin/bash
# 资源采样器：每 5 min 记一行「可用内存 / swap 已用 / 四卡 util / 在训臂数」
# 用法（服务器）：setsid nohup bash manu/pipeline/npu/sample_resources.sh >/dev/null 2>&1 </dev/null &
set -u
W=/mnt/data/siping/runs/npu20ms/nano_search
LOG=$W/mem_samples.log
while true; do
  avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  swapu=$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{print t-f}' /proc/meminfo)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' ',' )
  arms=$(ps -eo args | grep -o "trial-number [0-9]*" | sort -u | grep -c "trial-number")
  echo "$(date '+%F %T') avail_MB=$((avail/1024)) swap_used_MB=$((swapu/1024)) gpu=[$gpu] arms=$arms" >> "$LOG"
  sleep 300
done
