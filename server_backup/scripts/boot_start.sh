#!/bin/bash
# 服务器开机自启：看门狗 + 守望链路 + 3 条策展通道（幂等）
set -u
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
REPO=/tmp/pycharm_project_10ae9e2e
W=/mnt/data/siping/runs/npu20ms/nano_search
S=$W/scripts
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p "$W"
echo "[$(date '+%F %T')] boot_start 执行"
pgrep -f "[l]oad_watchdog.sh" >/dev/null || { setsid nohup bash "$S/load_watchdog.sh" >> /tmp/load_watchdog.log 2>&1 < /dev/null & }
cd "$REPO" 2>/dev/null || { echo "REPO_MISSING"; exit 1; }
pgrep -f "[a]uto_promote_watch.py" >/dev/null || { setsid nohup $PY -u manu/pipeline/npu/auto_promote_watch.py --watch --interval 600 --deadline "2026-11-17 09:00" >> "$W/auto_promote_watch.log" 2>&1 < /dev/null & }
pgrep -f "[b]oard_t4_watch.py"   >/dev/null || { setsid nohup $PY -u manu/pipeline/npu/board_t4_watch.py --watch --interval 600 --deadline "2026-11-17 09:00" >> "$W/board_t4_watch.log" 2>&1 < /dev/null & }
pgrep -f "[s]oup_sync_loop.sh"   >/dev/null || { setsid nohup bash "$S/soup_sync_loop.sh" >> "$W/soup_sync.log" 2>&1 < /dev/null & }
pgrep -f "[r]un_morning_report_v2.sh" >/dev/null || { setsid nohup bash manu/pipeline/npu/run_morning_report_v2.sh "2026-11-17 09:10" >> "$W/morning_report_runner.log" 2>&1 < /dev/null & }
pgrep -f "[r]efresh_sets.sh"    >/dev/null || { setsid nohup bash "$S/refresh_sets.sh" >> /tmp/refresh_sets.log 2>&1 < /dev/null & }
cd "$W"
L="$S/curated_loop.sh"; [ -f "$L" ] || L="$W/curated_loop.sh"
if [ -f "$L" ]; then
  if [ "$(pgrep -f '[c]urated_loop.sh' | wc -l)" -ge 3 ]; then
    echo "已有通道在跑 ⇒ boot_start 不重复拉起"
  else
    setsid nohup nice -n 10 bash "$L" 0 860000 0 c0 >> "$W/curated_c0.log" 2>&1 < /dev/null &
    setsid nohup nice -n 10 bash "$L" 0 870000 1 c1 >> "$W/curated_c1.log" 2>&1 < /dev/null &
    setsid nohup nice -n 10 bash "$L" 1 880000 2 c2 >> "$W/curated_c2.log" 2>&1 < /dev/null &
  fi
fi
echo "[$(date '+%F %T')] boot_start 完成"
# 延迟 15 分钟起 1 个训练家族（仅 GPU2，且要求可用内存 >= 60G ⇒ 保守启动）
( sleep 900
  M=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
  if [ "${M:-0}" -ge 60 ] && ! pgrep -f "[r]un_adaptive_night.sh" >/dev/null; then
    cd "$REPO" || exit 0
    sed -i 's/--gpus 0,1,2,3/--gpus 2/' manu/pipeline/npu/run_adaptive_night.sh 2>/dev/null
    setsid nohup nice -n 15 bash manu/pipeline/npu/run_adaptive_night.sh "2026-11-17 09:00" 6 18 22 >> "$W/adaptive.log" 2>&1 < /dev/null &
    echo "[$(date '+%F %T')] MemAvailable=${M}GB => 起 GPU2 训练家族"
  else
    echo "[$(date '+%F %T')] MemAvailable=${M}GB 或已有训练家族 => 暂不起 adaptive"
  fi ) >> /tmp/boot_start.log 2>&1 &
