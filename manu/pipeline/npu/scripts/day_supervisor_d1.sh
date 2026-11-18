#!/bin/bash
# 🟢 白天波 D1 通道自愈守望（2026-11-18 09:35）
#   每 5 分钟检查 4 条通道 + 4 个守望进程，掉线即拉起（幂等：号段/配方固定 ⇒ 复用同臂号）。
#   铁律：拉起前查 MemAvailable（<40GB ⇒ 等 120s）；每个通道重启上限 30 次。
set -u
REPO=/tmp/pycharm_project_10ae9e2e
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search
LOG=$W/day_supervisor.log
END="2026-11-19 08:45"
cd $REPO || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
log(){ echo "[$(date "+%F %T")] $*" >> $LOG; }
# 判活：PAT 是 PID 文件路径 ⇒ 读 PID 后 kill -0；PAT 是正则 ⇒ 走 pgrep（已锚定，不会自匹配）
alive(){
  local pat="$1" pid
  case "$pat" in
    /*) [ -f "$pat" ] || return 1; pid=$(cat "$pat" 2>/dev/null); [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null ;;
    *)   pgrep -f "$pat" >/dev/null 2>&1 ;;
  esac
}
exec 9>"$W/.lock_day_supervisor"
flock -n 9 || { log "[DUP] 已有守望实例在跑 ⇒ 本实例退出"; exit 0; }
pkill -f "synth_gpu2.sh" 2>/dev/null
pkill -f "synth_gpu3.sh" 2>/dev/null
log "[启动] 守望实例 pid=$$ （flock 互斥已生效）"
declare -A PAT REST
# 🟢🟢 2026-11-18 10:10 **判活口径全面改为 PID 文件 / 锚定正则**
#   坑：`pgrep -f "<脚本名>"` 会匹配**任何提到该名字的 shell**（运维 ssh 命令里也含这些名字）
#        ⇒ 守望误判「已在跑」⇒ 通道永远不重启（11-18 10:03 真实踩到：synth2/synth3/gateC 都没起来）。
PAT[gateA]="$W/.pid_gateA"
PAT[gateB]="$W/.pid_gateB"
# 🟢 2026-11-18 10:25 新增门禁通道 C（GPU0，高权重交叉段）
PAT[gateC]="$W/.pid_gateC"
PAT[synth2]="$W/.pid_synth2"
PAT[synth3]="$W/.pid_synth3"
PAT[boardw]="^/home/huangzhe/anaconda3/envs/uav/bin/python -u manu/pipeline/npu/board_t4_watch.py"
PAT[promow]="^/home/huangzhe/anaconda3/envs/uav/bin/python -u manu/pipeline/npu/auto_promote_watch.py"
PAT[guard]="baseline_guard.py"
PAT[mreport]="^bash .*run_morning_report4.sh"
for k in "${!PAT[@]}"; do REST[$k]=0; done
while :; do
  now=$(date +%s); end=$(date -d "$END" +%s)
  [ "$now" -lt "$end" ] || { log "[截止] $END ⇒ 守望退出"; break; }
  avail=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
  for k in gateA gateB gateC synth2 synth3 boardw promow guard mreport; do
    if alive "${PAT[$k]}"; then continue; fi
    [ "${REST[$k]}" -lt 30 ] || { log "[跳过] $k 重启超限"; continue; }
    if [ "${avail:-0}" -lt 40 ]; then log "[闸门] avail=${avail}GB < 40GB ⇒ 暂不重启 $k"; continue; fi
    REST[$k]=$(( ${REST[$k]} + 1 ))
    log "[重启] $k（第 ${REST[$k]} 次）avail=${avail}GB"
    case $k in
      gateA) setsid nohup bash $W/scripts/gate_loop_d1.sh >> $W/day_gateA.log 2>&1 < /dev/null 9>&- & ;;
      gateB) setsid nohup bash $W/scripts/gate_loop_d1b.sh >> $W/day_gateB.log 2>&1 < /dev/null 9>&- & ;;
      gateC) setsid nohup bash $W/scripts/gate_loop_d1c.sh >> $W/day_gateC.log 2>&1 < /dev/null 9>&- & ;;
      synth2) setsid nohup bash $W/scripts/synth_gpu2_d2.sh >> $W/day_synth2.log 2>&1 < /dev/null 9>&- & ;;
      synth3) setsid nohup bash $W/scripts/synth_gpu3_d2.sh >> $W/day_synth3.log 2>&1 < /dev/null 9>&- & ;;
      boardw) setsid nohup $PY -u manu/pipeline/npu/board_t4_watch.py --watch --interval 600 \
               --deadline "2026-11-19 09:00" >> $W/board_t4_watch.log 2>&1 < /dev/null 9>&- & ;;
      promow) setsid nohup $PY -u manu/pipeline/npu/auto_promote_watch.py --watch --interval 600 \
               --deadline "2026-11-19 08:30" >> $W/auto_promote_watch.log 2>&1 < /dev/null 9>&- & ;;
      guard) setsid nohup $PY -u /tmp/baseline_guard.py >> $W/baseline_guard.log 2>&1 < /dev/null 9>&- & ;;
      mreport) setsid nohup bash manu/pipeline/npu/run_morning_report4.sh "2026-11-19 08:40" \
               >> $W/morning_report_runner.log 2>&1 < /dev/null 9>&- & ;;
    esac
    sleep 30
  done
  # ---- sb3 家族标签统一（粗粒度 ⇒ 关A/关B 能攒票）----
  $PY -u manu/pipeline/npu/scripts/retag_sb3.py >> $W/retag_sb3.log 2>&1 9>&-
  sleep 300 9>&-
done
