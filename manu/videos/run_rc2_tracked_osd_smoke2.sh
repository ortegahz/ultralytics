#!/bin/bash
# Smoke test on a video with a MOVING target (the first pick was a static pole -> correctly culled
# by the rigid-static pruner, which is the 龙泉山 behaviour working as designed).
set -u
cd /tmp/pycharm_project_10ae9e2e || exit 2
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
export PYTHONPATH=.

DATA_ROOT=/mnt/data/siping/datasets/manu
SRC="$DATA_ROOT/纵横AI数据采集_mp4"
OUT="$DATA_ROOT/纵横AI数据采集_osd_rc2_tracked_smoke"

VIDEO="$SRC/VIDEO00087_19700101_003101.mp4"
echo "[SMOKE2] video=$(basename "$VIDEO")"

python manu/videos/render_rc2_tracked_osd_video.py \
  --input "$VIDEO" \
  --output "$OUT/$(basename "${VIDEO%.mp4}")_osd.mp4" \
  --threshold 0.20 \
  --limit 400 \
  --batch-size 4 \
  --device 0 \
  --cpu-threads 2 \
  --out-json "$OUT/summary_moving.json" \
  --out-preds-dir "$OUT/preds_track" \
  --out-cache-dir "$OUT/ms_cache"
echo "[SMOKE2] rc=$?"

echo "=== stage trace ==="
python manu/evaluation/diag_zongheng_track_stages.py "$OUT/ms_cache/$(basename "${VIDEO%.mp4}").pkl" 2>&1 | tail -20
