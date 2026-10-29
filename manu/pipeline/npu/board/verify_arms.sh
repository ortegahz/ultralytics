#!/usr/bin/env bash
# ===========================================================================
# verify_arms.sh -- run the subset accuracy check on the server for each arm.
#
# THE REFERENCE TSV MUST COME FROM THE SAME RUN AS THE REFERENCE MD5 FILE.
# The stale board_tsv_svc/P1b_step2_nodrop_orig copy is NOT the same run: it
# makes the check fail FOR THE REFERENCE ARM ITSELF (22 frames differ), which
# is a false alarm, not a defect of the arm under test.
#
# The server SSH is flaky (it times out from time to time), so every access
# retries.
#
# Usage: verify_arms.sh <ARM> [<ARM> ...]
# ===========================================================================
set -uo pipefail

SERVER=huangzhe@192.168.99.40
SPORT=32222
REPO=/tmp/pycharm_project_10ae9e2e
WORK=/mnt/data/siping/ptq_work
REF_MD5=${WORK}/psearch/results/A_gate_step2_nodrop/feature_md5.txt
REF_TSV=${WORK}/psearch/results/A_gate_step2_nodrop/VIDEO00007_19700101_001324_mosaic.tsv
LABELS="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
PARAMS=${REPO}/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/track3_blind.json

ssh_try() {
  ssh -p ${SPORT} -o ConnectTimeout=30 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
      "${SERVER}" "$@" 2>&1 | grep -v "WARNING: connection\|post-quantum\|store now\|may need to be upgraded"
}

for ARM in "$@"; do
  echo "==================== ${ARM} ===================="
  for i in $(seq 1 10); do
    out=$(ssh_try "source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav && cd ${REPO} && python manu/pipeline/npu/ptq/load_and_align_subset.py \
      --arm-md5 ${WORK}/psearch/results/${ARM}/feature_md5.txt \
      --arm-tsv ${WORK}/psearch/results/${ARM}/VIDEO00007_19700101_001324_mosaic.tsv \
      --ref-md5 ${REF_MD5} \
      --ref-tsv ${REF_TSV} \
      --seq VIDEO00007_19700101_001324 \
      --labels-root '${LABELS}' \
      --params-json ${PARAMS} \
      --gap-log ${WORK}/psearch/logs/${ARM}.run.log \
      --work-dir /tmp/surv_work_${ARM} \
      --out-json ${WORK}/surviving_frames/${ARM}.json \
      --out-md ${WORK}/surviving_frames/${ARM}.md \
      --manifest-out ${WORK}/surviving_frames/${ARM}_processed_frames.csv; echo SCRIPT_EXIT=\$?")
    if echo "${out}" | grep -qE "VERDICT|MD5\]|SCRIPT_EXIT"; then
      echo "${out}"
      break
    fi
    echo "[verify] server not answering (attempt ${i}); retrying in 15s"
    sleep 15
  done
done
