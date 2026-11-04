#!/usr/bin/env bash
# ============================================================
# host 资源守卫（铁律十三：重活上服务器 + host 资源自监控）
#
#   check   一次性体检：打印 load / 内存 / 本 Agent 的重活，OK 或 BLOCK
#   watch   持续守卫（建议用户在【自己终端】跑，30 s 采样；越线自动收口本 Agent 的重活）
#   stop    停掉 watch
#
# 阈值：LOAD_MAX=6（12 核）/ MEM_MIN_GB=3（available）
# 越线动作：对本 Agent 的重活 renice +9，再 30 s 仍越线则 TERM
# ============================================================
set -uo pipefail

LOAD_MAX="${LOAD_MAX:-6}"
MEM_MIN_GB="${MEM_MIN_GB:-3}"
SAMPLE_SEC="${SAMPLE_SEC:-30}"
LOG_DIR="${LOG_DIR:-/tmp/host_guard}"
LOG_FILE="$LOG_DIR/guard.log"
PID_FILE="$LOG_DIR/watch.pid"

# 本 Agent 拉起的重活（只收这些，绝不碰用户的其它进程）
HEAVY_PATTERNS=(
  "lookahead_longquan.py"
  "sweep_arms.py"
  "sweep_rolling.py"
  "tune_kernel.py"
  "build_frame_cache.py"
  "diag_cle_timeline.py"
  "diag_tail.py"
  "preannotate_hm_bbox.py"
)

mkdir -p "$LOG_DIR"

load1() { awk '{print $1}' /proc/loadavg; }
mem_avail_gb() { awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo; }

heavy_pids() {
  # 只看 comm 以 python 开头的进程，并对 cmdline 做【字面量】匹配
  # （不能用 pgrep -f：沙箱启动器会把整条命令行塞进 cmdline ⇒ 自匹配）
  local pid args pat
  ps -eo pid=,comm= | awk '$2 ~ /^python/ {print $1}' | while read -r pid; do
    args=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || continue
    [ -z "$args" ] && continue
    for pat in "${HEAVY_PATTERNS[@]}"; do
      case "$args" in *"$pat"*) echo "$pid"; break ;; esac
    done
  done
}

heavy_cmd() {
  local pid
  for pid in $(heavy_pids); do
    printf '  pid=%s  %s\n' "$pid" "$(ps -o args= -p "$pid" 2>/dev/null | cut -c1-110)"
  done
}

report() {
  local l m pids
  l=$(load1); m=$(mem_avail_gb)
  pids=$(heavy_pids | tr '\n' ' ')
  echo "load(1m)=${l}  mem_available=${m}GB  (阈值 load<${LOAD_MAX} / mem>${MEM_MIN_GB}GB)"
  if [ -n "${pids// /}" ]; then
    echo "本 Agent 的重活: $pids"
    heavy_cmd
  else
    echo "本 Agent 的重活: 无"
  fi
  if awk "BEGIN{exit !($l > $LOAD_MAX || $m < $MEM_MIN_GB)}"; then
    echo "状态: **BLOCK**（资源越线 ⇒ 禁止在本机起新重活，请迁到服务器）"
    return 1
  fi
  echo "状态: OK"
  return 0
}

case "${1:-check}" in
  check) report ;;
  watch)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "watch 已在跑 (pid=$(cat "$PID_FILE"))"; exit 0
    fi
    echo "启动 watch（每 ${SAMPLE_SEC}s 采样，日志 $LOG_FILE），Ctrl-C 退出"
    (
      while true; do
        ts=$(date '+%F %T')
        l=$(load1); m=$(mem_avail_gb)
        pids=$(heavy_pids | tr '\n' ' ')
        over=0
        awk "BEGIN{exit !($l > $LOAD_MAX || $m < $MEM_MIN_GB)}" && over=1
        {
          echo "$ts load=$l avail_mem=${m}GB heavy=[${pids}] over=$over"
          if [ "$over" = "1" ] && [ -n "${pids// /}" ]; then
            for pid in $pids; do
              renice -n 15 -p "$pid" >/dev/null 2>&1 && echo "$ts  renice15 pid=$pid"
            done
          fi
        } >> "$LOG_FILE"
        sleep "$SAMPLE_SEC"
      done
    ) &
    echo $! > "$PID_FILE"
    echo "watch pid=$(cat "$PID_FILE")"
    ;;
  stop)
    if [ -f "$PID_FILE" ]; then kill "$(cat "$PID_FILE")" 2>/dev/null && echo "已停"; rm -f "$PID_FILE"; else echo "未在跑"; fi
    ;;
  *) echo "用法: $0 {check|watch|stop}"; exit 2 ;;
esac
