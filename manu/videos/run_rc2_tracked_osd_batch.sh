#!/bin/bash
# 纵横AI数据采集 —— RC2 + **龙泉山时空轨迹后处理** 的全量 OSD 渲染（th=0.20，独立目录）
#
# 与前两档的差别：
#   th=0.44 交付档   单帧检测，无跨帧状态
#   th=0.20 敏感档   单帧检测，无跨帧状态
#   **本档**         单帧检测 + 龙泉山业务域时空轨迹后处理
#                    （双向跟踪器 + 时序平滑/插补 + 连续帧确认），画的是**最终系统检出**
#
# 输出目录与前两档完全分开，三档互不覆盖。
set -uo pipefail
cd /tmp/pycharm_project_10ae9e2e || exit 2
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
export PYTHONPATH=.

DATA_ROOT=/mnt/data/siping/datasets/manu
SRC="$DATA_ROOT/纵横AI数据采集_mp4"
OUT="$DATA_ROOT/纵横AI数据采集_osd_rc2_tracked_th020_20261110"
LOG=/mnt/data/siping/runs/zongheng_rc2_tracked_th020.log
SHARDS="${SHARDS:-4}"
DEVICES="${DEVICES:-0,1,2,3}"
TH="${TH:-0.20}"

if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "[FAIL] output dir exists and is not empty: $OUT"
  exit 1
fi
mkdir -p "$OUT/logs"

echo "[LAUNCH] $(date '+%F %T') | th=$TH | shards=$SHARDS gpus=$DEVICES | out=$OUT"
IFS=',' read -r -a DEV_ARR <<< "$DEVICES"
PIDS=()
for ((i = 0; i < SHARDS; i++)); do
  DEV="${DEV_ARR[$i]:-$i}"
  python -u manu/videos/render_rc2_tracked_osd_video.py \
    --input-root "$SRC" \
    --output-dir "$OUT" \
    --shard "$i/$SHARDS" \
    --hm-weights runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt \
    --threshold "$TH" \
    --th-ground "$TH" \
    --merge-radius 768 \
    --batch-size 8 \
    --cpu-threads 2 \
    --device "$DEV" \
    --out-json "$OUT/summary_shard$i.json" \
    --out-preds-dir "$OUT/preds_track" \
    --out-frames-dir "$OUT/frames" \
    --out-cache-dir "$OUT/ms_cache" \
    > "$OUT/logs/shard$i.log" 2>&1 &
  PIDS+=($!)
  echo " shard $i/$SHARDS -> pid ${PIDS[-1]} (gpu $DEV)"
done

RC=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || RC=1
done
echo "[LAUNCH] render rc=$RC"

echo "=== 产物核验 ==="
echo " mp4        = $(ls -1 "$OUT"/*.mp4 2>/dev/null | wc -l) / 58"
echo " frames     = $(ls -1 "$OUT/frames"/*.jsonl 2>/dev/null | wc -l) / 58"
echo " preds      = $(ls -1 "$OUT/preds_track"/*.json 2>/dev/null | wc -l) / 58"
echo " cache      = $(ls -1 "$OUT/ms_cache"/*.pkl 2>/dev/null | wc -l) / 58"
for i in 0 1 2 3; do
  grep -h "^\[SUCCESS\]" "$OUT/logs/shard$i.log" 2>/dev/null | tail -1
done
du -sh "$OUT"
echo "[LAUNCH] done $(date '+%F %T')"
