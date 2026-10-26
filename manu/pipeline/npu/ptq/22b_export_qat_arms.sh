#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# 22b_export_qat_arms.sh -- export the A0 / A1 / A2 / A1b QDQ ONNX arms.
#
# The four arms form a ladder in which exactly ONE factor changes at a time:
#
#   arm   weights            clips                 graph
#   ----  -----------------  --------------------  --------------------------
#   A0    QAT re-trained     none (no Q/DQ)        float   -> fp16 control
#   A1    QAT re-trained     max-calibrated        int8 head
#   A2    QAT re-trained     LSQ-learned           int8 head   (main candidate)
#   A1b   BASE (untouched)   max-calibrated        int8 head   (PTQ-style control)
#
# A0 vs A1  isolates the clip/quantisation effect
# A1 vs A2  isolates "learned clip" vs "calibrated clip"
# A1b vs A1 isolates the QAT weight update under identical clips
#
# Run (server, conda env `uav`):
#   source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
#   cd /tmp/pycharm_project_10ae9e2e
#   bash manu/pipeline/npu/ptq/22b_export_qat_arms.sh
set -uo pipefail

source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd /tmp/pycharm_project_10ae9e2e

W=/mnt/data/siping/ptq_work/qat/trial_0080
CKPT=$W/qat_epoch3.pt
OUT=$W/onnx
LOG=/mnt/data/siping/ptq_work/logs/qat_export_arms.log

mkdir -p "$OUT" "$(dirname "$LOG")"

run_arm () {
  local name=$1 mode=$2 src=$3
  echo "" | tee -a "$LOG"
  echo "=================== ARM $name (mode=$mode weights=$src) ===================" | tee -a "$LOG"
  python -u manu/pipeline/npu/ptq/22_export_qdq_onnx.py \
    --ckpt "$CKPT" \
    --out  "$OUT/${name}.onnx" \
    --mode "$mode" --weights-source "$src" 2>&1 | tee -a "$LOG"
  echo "[done] $name rc=${PIPESTATUS[0]}" | tee -a "$LOG"
}

: > "$LOG"
run_arm a2_qat_learned  qat   qat
run_arm a1_qat_maxclip  ptq   qat
run_arm a0_qat_float    float qat
run_arm a1b_base_maxclip ptq  base

echo "" | tee -a "$LOG"
echo "=== md5 of the exported ONNX ===" | tee -a "$LOG"
md5sum "$OUT"/*.onnx | tee -a "$LOG"
echo "[ALL DONE] $LOG"
