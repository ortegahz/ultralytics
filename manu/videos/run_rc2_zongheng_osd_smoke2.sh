#!/bin/bash
# Smoke test #2: verify the per-frame JSONL export and the shard launcher plumbing.
set -u
cd /tmp/pycharm_project_10ae9e2e || exit 2
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
export PYTHONPATH=.

DATA_ROOT=/mnt/data/siping/datasets/manu
SRC="$DATA_ROOT/纵横AI数据采集_mp4"
OUT="$DATA_ROOT/纵横AI数据采集_osd_rc2_smoke2"

# Two videos, one per GPU, so both the JSONL export and the multi-shard plumbing are exercised.
VIDEO=$(ls -S -r "$SRC"/*.mp4 | head -1)
echo "[SMOKE2] video=$(basename "$VIDEO")"

python manu/videos/render_rc2_center_osd_video.py \
  --input "$VIDEO" \
  --output "$OUT/$(basename "${VIDEO%.mp4}")_osd.mp4" \
  --limit 60 \
  --batch-size 4 \
  --device 0 \
  --cpu-threads 2 \
  --out-json "$OUT/summary.json" \
  --out-frames-jsonl "$OUT/frames.jsonl" 2>&1 | tail -6
echo "[SMOKE2] rc=$?"

echo "=== JSONL head ==="
head -3 "$OUT/frames.jsonl"
echo "=== JSONL lines ==="
wc -l "$OUT/frames.jsonl"
echo "=== detection-bearing lines ==="
grep -c '"detections": \[{' "$OUT/frames.jsonl" || true
