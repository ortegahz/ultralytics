#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# 22j_board_sweep_ptq_baseline.sh -- the SAME board sweep for the PTQ baseline
# arms, so the comparison against the QAT arms is controlled.
#
# WHY THIS EXISTS
# ---------------
# The baseline numbers in the tier2 reports were measured at ONE threshold
# (0.30) with the 04 driver, while the QAT arms are measured with the 19
# sweep (collect at 0.05, re-score offline).  Re-running the baseline arms
# through the SAME sweep, on the SAME manifests, with the SAME collector,
# makes the comparison exact at every threshold -- and it is the only way to
# get the baseline's curve at th=0.45 on the golden pool.
#
# ARMS
#   p0_off_A_normal  the int8 PTQ model (fully quantised by the toolkit)
#   p0_off_A_none    the fp16 model (the float control)
#
# Run (server, conda env `uav`):
#   source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
#   cd /tmp/pycharm_project_10ae9e2e
#   setsid nohup bash manu/pipeline/npu/ptq/22j_board_sweep_ptq_baseline.sh \
#       > /mnt/data/siping/ptq_work/logs/qat_baseline_sweep.out 2>&1 < /dev/null &
set -uo pipefail

source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd /tmp/pycharm_project_10ae9e2e

W=/mnt/data/siping/ptq_work
OUT=$W/qat/trial_0080/board_eval_baseline
mkdir -p "$OUT"

# The board runs ONE driver at a time (at most two RKNN contexts; a third
# one soft-resets the NPU).  Wait for any other board sweep to finish
# before taking the board.
echo "### waiting for the board to be free ###"
for _ in $(seq 1 240); do
  n=$(ps -eo args | grep -c "[1]9_board_threshold_sweep")
  [ "$n" -eq 0 ] && break
  echo "  another board sweep is running ($n), waiting..."
  sleep 15
done
echo "### board is free ###"

MODELS=""
# the CONTROLLED arm: the QAT re-trained weights under the toolkit's OWN full
# int8 recipe -- the same config as the p0_off_A_normal baseline, so the ONLY
# difference between the two is the weights.  That is the apples-to-apples
# "did the QAT training help?" comparison.
A2F=$W/qat/trial_0080/rknn/a2f_qatw_toolkit_int8.rknn
if [ -f "$A2F" ]; then
  MODELS="$MODELS $A2F"
  echo "### a2f_qatw_toolkit_int8 -> $A2F  md5=$(md5sum "$A2F" | cut -d" " -f1)"
else
  echo "### a2f_qatw_toolkit_int8 MISSING -- build it first (see 22c_build_qdq_rknn.py)"
fi

for name in p0_off_A_normal p0_off_A_none; do
  found=""
  for d in "$W/screen_n" "$W/screen" "$W/final_models" "$W/models"; do
    if [ -f "$d/$name.rknn" ]; then found="$d/$name.rknn"; break; fi
  done
  if [ -z "$found" ]; then
    echo "[FATAL] no .rknn found for $name"; exit 1
  fi
  MODELS="$MODELS $found"
  echo "### $name -> $found  md5=$(md5sum "$found" | cut -d" " -f1)"
done

# re-stage onto the board so both sides agree byte-for-byte
for m in $MODELS; do
  scp -o StrictHostKeyChecking=no "$m" ptq-board:/mnt/manu/ptq/models/ >/dev/null || exit 1
  echo "  staged $(basename "$m") md5=$(md5sum "$m" | cut -d" " -f1)"
done

run_sweep () {
  local man=$1 tag=$2
  echo ""
  echo "############################################################"
  echo "### BASELINE SWEEP $tag   manifest=$man"
  echo "############################################################"
  python -u manu/pipeline/npu/ptq/19_board_threshold_sweep.py \
    --manifest "$man" \
    --models $MODELS \
    --collect-th 0.05 \
    --thresholds 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70 \
    --topk 100 \
    --variant p0_off \
    --pt-device cuda:2 \
    --board-timeout 900 \
    --out "$OUT/$tag" 2>&1
  echo "[sweep] $tag rc=$?"
}

run_sweep "$W/fast/tier1_sensitive.json"      tier1_sensitive
run_sweep "$W/fast/001324_sub2000.json"       001324_sub2000
run_sweep "$W/fast/golden_pool_sub1200.json"  golden_pool_sub1200

echo ""
echo "=== md5 of the evaluation reports ==="
md5sum "$OUT"/*/*.json "$OUT"/*/dets_*.tsv 2>&1
echo "[ALL DONE]"
