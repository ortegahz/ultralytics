#!/bin/bash
# 纵横AI RC2 OSD —— detached 启动（可指定工作点与输出目录）
#
# 用法：
#   bash manu/videos/launch_rc2_zongheng_osd.sh                      # 默认 th=0.44 → ..._osd_rc2_20261109
#   TH=0.20 OUT=/mnt/.../纵横AI数据采集_osd_rc2_th020 \
#     SPLIT_LOCAL=manu/data/zongheng_ft_split_th020.json \
#     REVIEW_LOCAL=manu/docs/dataset/zongheng-tk-th020.md \
#     PRELABEL_OUT=/mnt/.../纵横AI数据采集_preannot_rc2_th020 \
#     bash manu/videos/launch_rc2_zongheng_osd.sh
#
# 不同工作点必须写进**不同目录**，否则会覆盖上一档的产物。
set -u
cd /tmp/pycharm_project_10ae9e2e || exit 2

DATA_ROOT=/mnt/data/siping/datasets/manu
TH="${TH:-0.44}"
OUT="${OUT:-$DATA_ROOT/纵横AI数据采集_osd_rc2_20261109}"
SHARDS="${SHARDS:-4}"
LOG="${LOG:-/mnt/data/siping/runs/zongheng_rc2_osd_$(date +%Y%m%d_%H%M%S).log}"
SPLIT_LOCAL="${SPLIT_LOCAL:-manu/data/zongheng_ft_split.json}"
REVIEW_LOCAL="${REVIEW_LOCAL:-manu/docs/dataset/zongheng-tk.md}"
PRELABEL_OUT="${PRELABEL_OUT:-$DATA_ROOT/纵横AI数据采集_preannot_rc2}"
FINAL_LOG="${FINAL_LOG:-/mnt/data/siping/runs/zongheng_rc2_osd_finalize_$(date +%Y%m%d_%H%M%S).log}"

# Refuse to touch an existing output directory (never overwrite a delivered OSD set).
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "[FAIL] output dir exists and is not empty: $OUT"
  echo "       每个工作点用各自的新目录（改 OUT=...）"
  exit 1
fi

echo "[LAUNCH] $(date '+%F %T') | th=$TH | shards=$SHARDS | out=$OUT"
setsid nohup bash manu/videos/run_rc2_zongheng_osd.sh > "$LOG" 2>&1 < /dev/null &
sleep 20
echo "[LAUNCH] render log=$LOG"
tail -12 "$LOG"

# 收尾（划分 + 复核底稿 + 预标注候选）在渲染结束后自动执行
OUT="$OUT" LOG="$LOG" SPLIT_LOCAL="$SPLIT_LOCAL" REVIEW_LOCAL="$REVIEW_LOCAL" \
PRELABEL_OUT="$PRELABEL_OUT" \
  setsid nohup bash manu/videos/finalize_rc2_zongheng_osd.sh > "$FINAL_LOG" 2>&1 < /dev/null &
sleep 5
echo "[LAUNCH] finalize log=$FINAL_LOG"
echo "=== GPU state ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== renderer procs ==="
pgrep -fc "render_rc2_center_osd_video.py --input-root" || true
