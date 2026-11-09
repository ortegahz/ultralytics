#!/bin/bash
# Clear the partial RC2 OSD output from the aborted run and relaunch, detached, one shard per GPU.
set -u
cd /tmp/pycharm_project_10ae9e2e || exit 2

OUT=/mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_20261109
LOG=/mnt/data/siping/runs/zongheng_rc2_osd_20261109.log

# The aborted run left a partial set. Remove it: a half-rendered OSD set is worse than none,
# because a reviewer cannot tell a missing video from an unrendered one.
if [ -d "$OUT" ]; then
  N=$(ls -1 "$OUT"/*.mp4 2>/dev/null | wc -l)
  echo "[CLEAN] removing partial output: $OUT ($N mp4 from the aborted run)"
  rm -rf "$OUT"
fi
mkdir -p "$OUT"
rm -f "$LOG"

echo "[LAUNCH] $(date '+%F %T') | GPUs 0-3 | batch 8 | th 0.44 | r 768 | out=$OUT"
setsid nohup bash manu/videos/run_rc2_zongheng_osd.sh > "$LOG" 2>&1 < /dev/null &
sleep 25
echo "[LAUNCH] log=$LOG"
tail -14 "$LOG"
echo "=== GPU state ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== renderer procs ==="
pgrep -fc "render_rc2_center_osd_video.py --input-root" || true
