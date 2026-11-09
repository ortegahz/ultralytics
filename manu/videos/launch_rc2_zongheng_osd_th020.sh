#!/bin/bash
# 启动「纵横AI RC2 OSD」的**敏感档**（th=0.20）重跑 —— 独立目录，不覆盖 th=0.44 那一档。
#
# 为什么单独一档：
#   th=0.44 是业务域**交付**工作点（每帧最多一个点、F1 94.069%），用于"看最终结论"；
#   th=0.20 是**召回优先**的敏感档，用于人工评估时**把候选都捞出来**看（多帧带点、
#   弱目标也能出现），两者服务的目的不同 ⇒ 产物必须分目录存放，互不覆盖。
set -u
cd /tmp/pycharm_project_10ae9e2e || exit 2

DATA_ROOT=/mnt/data/siping/datasets/manu
OUT="$DATA_ROOT/纵横AI数据采集_osd_rc2_th020_20261109"
LOG=/mnt/data/siping/runs/zongheng_rc2_osd_th020_20261109.log
FINAL_LOG=/mnt/data/siping/runs/zongheng_rc2_osd_th020_finalize.log

# GPU 1 上还有别的训练臂（trial 9072）⇒ 只用 0/2/3 三张空闲卡，分片数也相应降为 3。
export OUT LOG TH=0.20 RADIUS=768 SHARDS=3 DEVICES="0,2,3" BATCH=8
export SPLIT_LOCAL=manu/data/zongheng_ft_split_th020.json
export REVIEW_LOCAL=manu/docs/dataset/zongheng-tk-th020.md
export PRELABEL_OUT="$DATA_ROOT/纵横AI数据采集_preannot_rc2_th020"

if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "[FAIL] output dir exists and is not empty: $OUT"
  exit 1
fi

echo "[LAUNCH] $(date '+%F %T') | th=0.20 | r=768 | shards=3 (gpu 0,2,3) | out=$OUT"
setsid nohup bash manu/videos/run_rc2_zongheng_osd.sh > "$LOG" 2>&1 < /dev/null &
sleep 25
echo "[LAUNCH] render log=$LOG"
tail -14 "$LOG"

OUT="$OUT" LOG="$LOG" SPLIT_LOCAL="$SPLIT_LOCAL" REVIEW_LOCAL="$REVIEW_LOCAL" \
PRELABEL_OUT="$PRELABEL_OUT" \
  setsid nohup bash manu/videos/finalize_rc2_zongheng_osd.sh > "$FINAL_LOG" 2>&1 < /dev/null &
sleep 5
echo "[LAUNCH] finalize log=$FINAL_LOG"
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== renderers ==="
pgrep -fc "render_rc2_center_osd_video.py --input-root" || true
