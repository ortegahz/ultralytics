#!/bin/bash
# Generate the stale-arm per-frame predictions for the OSD sequences.
set -u
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
GT="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
FRAMES="/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg"
CACHE=runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache
PARAMS=manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json
OUT=runs/lookahead/osd
for SEQ in "$@"; do
  echo "=== [$(date '+%F %T')] stale arm $SEQ ==="
  $PY manu/tracking/lookahead/lookahead_longquan.py \
    --cache-root "$CACHE" \
    --labels-root "$GT" \
    --frames-root "$FRAMES" \
    --params-json "$PARAMS" \
    --seqs "$SEQ" --lookahead stale --arms stale \
    --out-json "$OUT/arms_stale_${SEQ}.json" \
    --out-preds-dir "$OUT/preds_stale_${SEQ}" \
    > "$OUT/logs/stale_${SEQ}.log" 2>&1
  echo "[rc=$?] $SEQ  $(date '+%F %T')"
  tail -3 "$OUT/logs/stale_${SEQ}.log"
done
echo "STALE_ARMS_DONE"
