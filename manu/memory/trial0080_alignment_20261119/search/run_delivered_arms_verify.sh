#!/usr/bin/env bash
# 独立复核：直接用【交付参数档】跑三臂（逐帧天花板 / 跳帧未调 / 跳帧优化）
# run A: --params-json = v3 基座 -> arm1=v3逐帧天花板, arm2a=v3密集基, arm2b=v3跳帧未调(物理基), arm3=交付参数跳帧优化, arm5=交付参数逐帧
# run B: --params-json = 交付参数 -> arm1=交付逐帧天花板, arm2a=交付密集基, arm2b=交付物理基(应与 run A 的 arm3 一致)
set -uo pipefail
cd /mnt/data/siping/runs/npu20ms/nano_search/repo || exit 2
export PYTHONPATH=/mnt/data/siping/ptq_work/ortgpu:/mnt/data/siping/runs/npu20ms/nano_search/repo
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
CACHE=/mnt/data/siping/runs/trial0080_onnx/ms_cache
GT="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
OUT=/mnt/data/siping/runs/trial0080_onnx/search
PY=~/anaconda3/envs/uav/bin/python
V3=manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json
DELIV=manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json

echo "[start] $(date +%F %T) run A: base=v3, opt=delivered"
$PY -u manu/evaluation/rt_fps_jitter_search.py \
  --cache-root "$CACHE" --labels-root "$GT" \
  --params-json "$V3" --all-seqs --phase eval \
  --scenarios s1,s2,s3,s4 --seeds 0,1,2 --workers 4 --max-frames 4000 \
  --opt-params-json "$DELIV" \
  --out-json "$OUT/rt_eval_delivered_armA.json" \
  --out-md "$OUT/rt_eval_delivered_armA.md" \
  2>&1 | tee "$OUT/eval_delivered_armA.log"
echo "[runA rc=${PIPESTATUS[0]}] $(date +%F %T)"

echo
echo "[start] $(date +%F %T) run B: base=delivered, opt=delivered"
$PY -u manu/evaluation/rt_fps_jitter_search.py \
  --cache-root "$CACHE" --labels-root "$GT" \
  --params-json "$DELIV" --all-seqs --phase eval \
  --scenarios s1,s2,s3,s4 --seeds 0,1,2 --workers 4 --max-frames 4000 \
  --opt-params-json "$DELIV" \
  --out-json "$OUT/rt_eval_delivered_armB.json" \
  --out-md "$OUT/rt_eval_delivered_armB.md" \
  2>&1 | tee "$OUT/eval_delivered_armB.log"
echo "[runB rc=${PIPESTATUS[0]}] $(date +%F %T)"
echo "[done] $(date -F %T)"
