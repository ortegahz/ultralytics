#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# 22g_board_eval_qat_arms.sh -- the board evaluation of the four QAT arms.
#
# ONE board pass per (model, manifest): the board's detections are collected at
# a LOW collection threshold (0.05) and re-scored OFFLINE at every threshold of
# interest (0.30 and 0.45 in particular), so the two thresholds that the
# delivery compares cost ONE board pass, not two.
#
# The board is driven ONE process at a time (at most two RKNN contexts, a third
# one soft-resets the NPU), so the manifests run SEQUENTIALLY here.
#
# The models are the mean/std builds (the ones the smoke test proved alive);
# the _nomeanstd variants are broken on the board (the NPU output does not
# depend on the input) and are NOT evaluated.
#
# Run (server):
#   source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
#   cd /tmp/pycharm_project_10ae9e2e
#   setsid nohup bash manu/pipeline/npu/ptq/22g_board_eval_qat_arms.sh \
#       > /mnt/data/siping/ptq_work/logs/qat_board_eval.out 2>&1 < /dev/null &
set -uo pipefail

source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd /tmp/pycharm_project_10ae9e2e

W=/mnt/data/siping/ptq_work
RK=$W/qat/trial_0080/rknn
LOGDIR=$W/logs
OUT=$W/qat/trial_0080/board_eval
mkdir -p "$OUT"

MODELS="$RK/a2_qat_learned.rknn $RK/a1_qat_maxclip.rknn $RK/a0_qat_float.rknn $RK/a1b_base_maxclip.rknn"

# Re-stage the models onto the board BEFORE the run.  The driver aborts with
# "model md5 MISMATCH" when the server's copy and the board's copy differ, and
# the toolkit build is NOT bit-reproducible, so the two sides have to be
# re-synced from one source of truth before every evaluation.
echo "### staging the models onto the board ###"
for m in $MODELS; do
  scp -o StrictHostKeyChecking=no "$m" ptq-board:/mnt/manu/ptq/models/ >/dev/null || exit 1
  echo "  staged $(basename "$m")  md5=$(md5sum "$m" | cut -d" " -f1)"
done

run_sweep () {
  local man=$1 tag=$2
  echo ""
  echo "############################################################"
  echo "### SWEEP $tag   manifest=$man"
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

run_sweep "$W/fast/tier1_sensitive.json"    tier1_sensitive
run_sweep "$W/fast/001324_sub2000.json"     001324_sub2000
run_sweep "$W/fast/golden_pool_sub1200.json" golden_pool_sub1200

echo ""
echo "=== md5 of the evaluation reports ==="
md5sum "$OUT"/*/*.json "$OUT"/*/dets_*.tsv 2>&1
echo "[ALL DONE]"
