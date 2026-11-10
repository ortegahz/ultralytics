#!/bin/bash
# Smoke test: tracked (龙泉山 spatio-temporal) OSD renderer on one 纵横AI video.
set -u
cd /tmp/pycharm_project_10ae9e2e || exit 2
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
export PYTHONPATH=.

DATA_ROOT=/mnt/data/siping/datasets/manu
SRC="$DATA_ROOT/纵横AI数据采集_mp4"
OUT="$DATA_ROOT/纵横AI数据采集_osd_rc2_tracked_smoke"

VIDEO=$(ls -S -r "$SRC"/*.mp4 | head -1)
echo "[SMOKE] video=$(basename "$VIDEO")"

python manu/videos/render_rc2_tracked_osd_video.py \
  --input "$VIDEO" \
  --output "$OUT/$(basename "${VIDEO%.mp4}")_osd.mp4" \
  --threshold 0.20 \
  --limit 150 \
  --batch-size 4 \
  --device 0 \
  --cpu-threads 2 \
  --out-json "$OUT/summary.json" \
  --out-preds-dir "$OUT/preds_track" \
  --out-cache-dir "$OUT/ms_cache"
echo "[SMOKE] rc=$?"
echo "=== outputs ==="
ls -la "$OUT" 2>/dev/null
echo "=== preds ==="
ls -la "$OUT/preds_track" 2>/dev/null
echo "=== cache ==="
ls -la "$OUT/ms_cache" 2>/dev/null
