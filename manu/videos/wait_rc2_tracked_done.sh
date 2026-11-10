#!/bin/bash
# 等待「纵横AI RC2 时空轨迹档」全链路完成：渲染 → 服务器收尾 → host 收集提交。
# 以 setsid nohup 脱离运行，避免会话结束时被回收。
set -uo pipefail
LOCAL=/media/manu/1TB-Volume/workspace/ultralytics
SERVER="huangzhe@192.168.99.40"
PORT=32222
OUT=/mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_tracked_th020_20261110
FINAL_LOG=/mnt/data/siping/runs/zongheng_rc2_tracked_finalize.log
STATUS=/tmp/tracked_done.status

echo "START $(date '+%F %T')" > "$STATUS"

for i in $(seq 1 480); do
  n_mp4=$(ssh -p "$PORT" -o BatchMode=yes -o ConnectTimeout=10 "$SERVER" \
    "ls -1 '$OUT'/*.mp4 2>/dev/null | wc -l" 2>/dev/null || echo 0)
  fin=$(ssh -p "$PORT" -o BatchMode=yes -o ConnectTimeout=10 "$SERVER" \
    "grep -c '\[FINALIZE\] done' '$FINAL_LOG' 2>/dev/null || echo 0" 2>/dev/null || echo 0)
  col=$(grep -c '\[COLLECT\] done' /tmp/collect_tracked.log 2>/dev/null || echo 0)
  echo "PROGRESS $(date '+%H:%M:%S') mp4=${n_mp4}/58 finalize=${fin} collect=${col}" >> "$STATUS"
  if [ "${fin:-0}" -ge 1 ] && [ "${col:-0}" -ge 1 ]; then
    echo "DONE $(date '+%F %T')" >> "$STATUS"
    exit 0
  fi
  sleep 45
done
echo "TIMEOUT $(date '+%F %T')" >> "$STATUS"
exit 1
