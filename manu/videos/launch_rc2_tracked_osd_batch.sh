#!/bin/bash
# detached 起活：纵横AI RC2 + 龙泉山时空轨迹后处理，全量 58 条（th=0.20）
set -u
cd /tmp/pycharm_project_10ae9e2e || exit 2

LOG=/mnt/data/siping/runs/zongheng_rc2_tracked_th020_batch.log
setsid nohup bash manu/videos/run_rc2_tracked_osd_batch.sh > "$LOG" 2>&1 < /dev/null &
sleep 30
echo "[LAUNCH] batch log=$LOG"
tail -14 "$LOG"
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== renderers ==="
pgrep -cf "^python -u manu/videos/render_rc2_tracked_osd_video.py" || true
