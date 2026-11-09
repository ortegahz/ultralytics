#!/bin/bash
# Smoke test for the RC2 centre-only OSD renderer on 纵横AI数据采集.
set -u
cd /tmp/pycharm_project_10ae9e2e || exit 2
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
export PYTHONPATH=.

DATA_ROOT=/mnt/data/siping/datasets/manu
SRC="$DATA_ROOT/纵横AI数据采集_mp4"
OUT="$DATA_ROOT/纵横AI数据采集_osd_rc2_smoke"

# Pick the shortest video so the smoke test is quick and cheap.
VIDEO=$(ls -S -r "$SRC"/*.mp4 | head -1)
echo "[SMOKE] video=$(basename "$VIDEO")"

python manu/videos/render_rc2_center_osd_video.py \
  --input "$VIDEO" \
  --output "$OUT/$(basename "${VIDEO%.mp4}")_osd.mp4" \
  --limit 150 \
  --batch-size 4 \
  --device 0 \
  --cpu-threads 2 \
  --out-json "$OUT/summary.json"
echo "[SMOKE] rc=$?"
