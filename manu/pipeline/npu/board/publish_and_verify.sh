#!/usr/bin/env bash
# ===========================================================================
# publish_and_verify.sh -- publish the v10 arms to the server and run the
# subset accuracy check on them.
#
# WHY A SCRIPT: the server SSH (192.168.99.40:32222) is FLAKY -- it times out
# from time to time -- so every access has to retry.  The board's /mnt/srv2 is
# an sshfs mount of the SAME directory, and it survived the outage, so the
# publish goes through the board (it is also where the files are).
#
# Usage: publish_and_verify.sh <ARM> [<ARM> ...]
# ===========================================================================
set -uo pipefail

SERVER=huangzhe@192.168.99.40
SPORT=32222
REPO=/tmp/pycharm_project_10ae9e2e
WORK=/mnt/data/siping/ptq_work

# The reference arm.  NOTE: the ref TSV must come from the SAME run as the
# ref md5 file -- the stale board_tsv_svc/P1b_step2_nodrop_orig copy is NOT
# the same run and makes the check fail for the reference arm itself.
REF_MD5=${WORK}/psearch/results/A_gate_step2_nodrop/feature_md5.txt
REF_TSV=${WORK}/psearch/results/A_gate_step2_nodrop/VIDEO00007_19700101_001324_mosaic.tsv
LABELS="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
PARAMS=${REPO}/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/track3_blind.json

ssh_try() { ssh -p ${SPORT} -o ConnectTimeout=25 -o BatchMode=yes "${SERVER}" "$@" 2>/dev/null; }

for ARM in "$@"; do
  echo "==================== ${ARM} ===================="

  # ---- 1. publish from the board through the sshfs mount ------------------
  ssh -p 22 -o ConnectTimeout=10 root@192.168.0.64 "
    mkdir -p /mnt/srv2/psearch/results/${ARM} /mnt/srv2/psearch/logs
    cp -f /root/psearch/results/${ARM}/VIDEO00007_19700101_001324_mosaic.tsv \
          /root/psearch/results/${ARM}/feature_md5.txt \
          /root/psearch/results/${ARM}/timing.txt \
          /root/psearch/results/${ARM}/processed_frames.csv \
          /mnt/srv2/psearch/results/${ARM}/ 2>/dev/null
    cp -f /root/psearch/results/${ARM}.run.log /mnt/srv2/psearch/logs/${ARM}.run.log 2>/dev/null
    [ -f /root/psearch/results/${ARM}/npu_load.txt ] && \
      cp -f /root/psearch/results/${ARM}/npu_load.txt /mnt/srv2/psearch/results/${ARM}/ 2>/dev/null
    ls -l /mnt/srv2/psearch/results/${ARM}/ | tail -6
  " 2>&1 | grep -v "WARNING\|post-quantum\|store now\|upgraded"

  # ---- 2. verify the publish from the SERVER side (the authoritative check)
  for i in 1 2 3 4 5 6; do
    out=$(ssh_try "ls -l ${WORK}/psearch/results/${ARM}/ 2>/dev/null | tail -6")
    if echo "${out}" | grep -q feature_md5.txt; then
      echo "${out}"
      break
    fi
    echo "[publish] server not answering yet (attempt ${i}); retrying"
    sleep 15
  done

  # ---- 3. the subset accuracy check ---------------------------------------
  for i in 1 2 3 4 5 6 7 8; do
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
      --manifest-out ${WORK}/surviving_frames/${ARM}_processed_frames.csv; echo EXIT=\$?")
    if echo "${out}" | grep -qE "VERDICT|MD5\]"; then
      echo "${out}"
      break
    fi
    echo "[verify] server not answering yet (attempt ${i}); retrying"
    sleep 15
  done
done
