#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# 22d_build_qat_arms.sh -- build the .rknn of every exported QDQ arm.
#
# The build runs in the conda env `rknn` (rknn-toolkit2 2.3.2), NOT in `uav`:
# the toolkit needs torch 2.4.0+cpu and has no CUDA build.
#
# HARD CONSTRAINTS (both measured, see 17_probe_qdq_import.py):
#   * rknn.build(do_quantization=False) -- the Q/DQ carries the calibration;
#     re-quantising on top of it fails with rc=3 / ret=-1.
#   * NO mean_values / std_values -- the /255 lives inside the graph.
#
# Run (server):
#   source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate rknn
#   cd /tmp/pycharm_project_10ae9e2e
#   bash manu/pipeline/npu/ptq/22d_build_qat_arms.sh
set -uo pipefail

source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate rknn
cd /tmp/pycharm_project_10ae9e2e

W=/mnt/data/siping/ptq_work/qat/trial_0080
OUT=$W/rknn
LOG=/mnt/data/siping/ptq_work/logs/qat_build_arms.log

mkdir -p "$OUT" "$(dirname "$LOG")"
: > "$LOG"

for name in a2_qat_learned a1_qat_maxclip a0_qat_float a1b_base_maxclip; do
  for ms in 1 0; do
    if [ "$ms" = "1" ]; then suffix=""; else suffix="_nomeanstd"; fi
    echo "" | tee -a "$LOG"
    echo "=================== BUILD ${name}${suffix} (mean_std=$ms) ===================" | tee -a "$LOG"
    python -u manu/pipeline/npu/ptq/22c_build_qdq_rknn.py \
      --onnx "$W/onnx/${name}.onnx" \
      --out  "$OUT/${name}${suffix}.rknn" \
      --mean-std "$ms" 2>&1 | tee -a "$LOG"
    echo "[done] ${name}${suffix} rc=${PIPESTATUS[0]}" | tee -a "$LOG"
  done
done

echo "" | tee -a "$LOG"
echo "=== md5 of the built RKNN ===" | tee -a "$LOG"
md5sum "$OUT"/*.rknn | tee -a "$LOG"
echo "[ALL DONE] $LOG"
