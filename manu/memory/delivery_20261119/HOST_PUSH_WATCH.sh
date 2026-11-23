#!/usr/bin/env bash
# 服务器侧守望：等 host 上电后**自动**把「§257 十八」推送过去（免人工干预）。
#
# host 2026-11-19 18:30 关机，服务器无法在它关机期间写它的文件。
# 本脚本轮询 host 的可达性，一恢复就调 HOST_PUSH.sh 推送；成功即退出，失败则记录原因继续等。
#
# 用法（服务器侧）:
#   bash manu/memory/delivery_20261119/HOST_PUSH_WATCH.sh &
#   WATCH_INTERVAL=60 WATCH_DEADLINE_MIN=720 bash .../HOST_PUSH_WATCH.sh
#
# 日志: /mnt/data/siping/runs/trial0080_onnx/search/host_push_watch.log
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL="${WATCH_INTERVAL:-60}"
DEADLINE_MIN="${WATCH_DEADLINE_MIN:-720}"
LOG="${WATCH_LOG:-/mnt/data/siping/runs/trial0080_onnx/search/host_push_watch.log}"
PUSH="$HERE/HOST_PUSH.sh"
HOST_ADDR="${HOST_ADDR:-192.168.0.115}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "守望启动：每 ${INTERVAL}s 探一次 ${HOST_ADDR}，最长 ${DEADLINE_MIN} 分钟；日志 ${LOG}"
START=$(date +%s)
while :; do
  NOW=$(date +%s)
  ELAPSED_MIN=$(( (NOW - START) / 60 ))
  if [ "$ELAPSED_MIN" -ge "$DEADLINE_MIN" ]; then
    log "[DEADLINE] 超过 ${DEADLINE_MIN} 分钟仍未完成，退出（host 若已上电可手动跑 HOST_PUSH.sh）"
    exit 2
  fi
  if ping -c1 -W2 "$HOST_ADDR" >/dev/null 2>&1; then
    log "[UP] ${HOST_ADDR} 可达，开始推送"
    if bash "$PUSH" >> "$LOG" 2>&1; then
      log "[SUCCESS] 推送完成并通过主机侧复核；本守望退出"
      exit 0
    fi
    log "[RETRY] 推送未通过（详见日志上文），${INTERVAL}s 后重试"
  else
    log "[WAIT] ${HOST_ADDR} 无响应（host 仍关机？）"
  fi
  sleep "$INTERVAL"
done
