#!/bin/bash
# 纵横AI RC2 OSD —— th=0.20 敏感档：**改用 4 卡**重跑（GPU 1 的训练臂只剩 5 GB，够跑本渲染器）
#
# 为什么重跑而不是把第 4 片挂上去：
#   3 分片与 4 分片的**视频归属不同**（index % N），中途加片会让同一条视频被两个分片同时渲染。
#   ⇒ 干净做法：停掉现有 3 片 → 删掉**未完成**的 MP4（半截文件）→ 用 4 片重跑。
#   已完成的视频（MP4 + 每帧记录都在）由 --skip-existing 直接跳过，不重算。
set -u
cd /tmp/pycharm_project_10ae9e2e || exit 2

DATA_ROOT=/mnt/data/siping/datasets/manu
OUT="$DATA_ROOT/纵横AI数据采集_osd_rc2_th020_20261109"
LOG=/mnt/data/siping/runs/zongheng_rc2_osd_th020_20261109.log
FINAL_LOG=/mnt/data/siping/runs/zongheng_rc2_osd_th020_finalize.log

echo "[REBALANCE] $(date '+%F %T') stopping the 3-shard run"
pkill -f "render_rc2_center_osd_video.py --input-root" || true
sleep 4
pkill -f "finalize_rc2_zongheng_osd.sh" || true
sleep 2
echo "[REBALANCE] renderers left: $(pgrep -cf 'render_rc2_center_osd_video.py --input-root' || true)"

# 删掉**没有**每帧记录配套的 MP4 ⇒ 它们是半截文件，必须重渲。
echo "[REBALANCE] dropping half-written videos (no per-frame record => incomplete)"
for m in "$OUT"/*_osd.mp4; do
  [ -f "$m" ] || continue
  b=$(basename "$m" _osd.mp4)
  if [ ! -f "$OUT/frames/$b.jsonl" ]; then
    echo "  - $(basename "$m")"
    rm -f "$m"
  fi
done

echo "[REBALANCE] keeping: mp4=$(ls -1 "$OUT"/*_osd.mp4 2>/dev/null | wc -l) records=$(ls -1 "$OUT/frames"/*.jsonl 2>/dev/null | wc -l)"

export OUT LOG TH=0.20 RADIUS=768 SHARDS=4 DEVICES="0,1,2,3" BATCH=8 RESUME=1
export SPLIT_LOCAL=manu/data/zongheng_ft_split_th020.json
export REVIEW_LOCAL=manu/docs/dataset/zongheng-tk-th020.md
export PRELABEL_OUT="$DATA_ROOT/纵横AI数据采集_preannot_rc2_th020"

echo "[REBALANCE] relaunching with 4 shards on gpu 0,1,2,3"
setsid nohup bash manu/videos/run_rc2_zongheng_osd.sh > "$LOG" 2>&1 < /dev/null &
sleep 25
tail -12 "$LOG"

OUT="$OUT" LOG="$LOG" SPLIT_LOCAL="$SPLIT_LOCAL" REVIEW_LOCAL="$REVIEW_LOCAL" \
PRELABEL_OUT="$PRELABEL_OUT" \
  setsid nohup bash manu/videos/finalize_rc2_zongheng_osd.sh > "$FINAL_LOG" 2>&1 < /dev/null &
sleep 5
echo "[REBALANCE] finalize log=$FINAL_LOG"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "renderers: $(pgrep -cf 'render_rc2_center_osd_video.py --input-root' || true)"
