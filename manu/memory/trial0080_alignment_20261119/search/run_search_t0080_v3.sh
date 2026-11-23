#!/usr/bin/env bash
# trial_0080 重建缓存 + v3 基座上的 Δn（跳帧）超参重搜
# 目标函数已加「逐帧口径不回退」硬约束（per-frame FAR <= 0.03 且 per-frame F1 >= base）
set -uo pipefail
cd /mnt/data/siping/runs/npu20ms/nano_search/repo || exit 2
export PYTHONPATH=/mnt/data/siping/ptq_work/ortgpu:/mnt/data/siping/runs/npu20ms/nano_search/repo
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

CACHE=/mnt/data/siping/runs/trial0080_onnx/ms_cache
GT="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
OUT=/mnt/data/siping/runs/trial0080_onnx/search
mkdir -p "$OUT"

echo "[start] $(date +%H:%M:%S)  缓存=$CACHE"
echo "[start] 基座 = v3（th_base 0.48 / th_ground 0.26 / sky_ratio 1.0 / confirm_consec 7 / confirm_gap_tol 0）"
echo "[start] 约束 = 逐帧 FAR<=0.03 且 逐帧 F1 >= base-0.001"

~/anaconda3/envs/uav/bin/python -u manu/evaluation/rt_fps_jitter_search.py \
  --cache-root "$CACHE" --labels-root "$GT" \
  --params-json manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json \
  --all-seqs --phase search \
  --scenarios s1,s2,s3,s4 --seeds 0,1,2 --workers 4 --max-frames 4000 --sweeps 2 \
  --space-json "$OUT/space_t0080_v3.json" \
  --far-redline 0.03 --perframe-f1-tol 0.001 \
  --lambda-far 2.0 --mu-far 25.0 --nu-f1 25.0 \
  --out-json "$OUT/rt_search_t0080_v3.json" \
  --out-md "$OUT/rt_search_t0080_v3.md" \
  --opt-params-json "$OUT/opt_params_t0080_v3.json" \
  --params-out "$OUT/opt_params_t0080_v3_flat.json" \
  2>&1 | tee "$OUT/search.log"

echo
echo "=== 复核：用最优参数跑三臂报告（per-frame / jump-untuned / jump-optimised）==="
~/anaconda3/envs/uav/bin/python -u manu/evaluation/rt_fps_jitter_search.py \
  --cache-root "$CACHE" --labels-root "$GT" \
  --params-json "$OUT/opt_params_t0080_v3_flat.json" \
  --all-seqs --phase eval \
  --scenarios s1,s2,s3,s4 --seeds 0,1,2 --workers 4 --max-frames 4000 \
  --out-json "$OUT/rt_eval_t0080_v3_opt.json" \
  --out-md "$OUT/rt_eval_t0080_v3_opt.md" \
  2>&1 | tee "$OUT/eval_opt.log"

echo "[done] $(date +%H:%M:%S)"
