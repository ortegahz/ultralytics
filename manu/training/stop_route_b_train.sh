#!/usr/bin/env bash
# 停掉路线 B 训练进程（按命令行精确匹配，不误伤 SSH 自身）
set -uo pipefail
pids=$(ps -ef | grep 'mixed_ft_train[.]py' | tr -s ' ' | cut -d' ' -f2)
if [ -z "$pids" ]; then
  echo "no training process found"
else
  echo "killing: $pids"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  sleep 4
  left=$(ps -ef | grep -c 'mixed_ft_train[.]py' || true)
  echo "remaining: $left"
fi
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
