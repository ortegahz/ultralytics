#!/bin/bash
# 板卡稳温 soak 一键脚本：起负载 → 起监控（带自动停负载）→ 收尾
#
#   board_thermal_soak.sh [duration_s] [npu_procs] [cpu_stress] [gpu] [out_prefix] [label]
#
# 监控一旦「连续 N 个采样点」确认降频（cooling device 动作 / 频率低于上限 /
# PVTM 钳住上限 / dmesg 温控事件），会立刻执行 stop-cmd 把压测负载停掉。
set -u

DURATION="${1:-3600}"
NPU_PROCS="${2:-2}"
CPU_STRESS="${3:-1}"
GPU="${4:-0}"
OUT="${5:-/tmp/thermal_run}"
LABEL="${6:-soak}"

BTS_DIR="${BTS_DIR:-/root/bts}"
LOAD="${LOAD:-$BTS_DIR/board_thermal_load.sh}"
MON="${MON:-$BTS_DIR/board_thermal_monitor.py}"
STOP_CMD="${STOP_CMD:-$BTS_DIR/bts_stop.sh}"

cat > "$STOP_CMD" <<STOPEOF
#!/bin/sh
$LOAD stop > /tmp/bts_stop.log 2>&1
STOPEOF
chmod +x "$STOP_CMD"

echo "### soak 开始 $(date '+%F %T')  duration=${DURATION}s npu=$NPU_PROCS cpu=$CPU_STRESS gpu=$GPU out=$OUT label=$LABEL"
"$LOAD" start "$DURATION" "$NPU_PROCS" "$CPU_STRESS" "$GPU"
echo "### 负载已起，监控开始"
python3 "$MON" --duration "$DURATION" --interval 2 --print-interval 60 \
    --out "$OUT" --label "$LABEL" --stop-cmd "$STOP_CMD" --throttle-confirm 3
echo "### 监控结束，清理负载"
"$LOAD" stop
echo "### soak 结束 $(date '+%F %T')"
