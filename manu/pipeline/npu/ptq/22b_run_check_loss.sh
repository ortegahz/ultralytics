#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# 22b_run_check_loss.sh -- the loader / fake-quant loss A/B check, DETACHED.
#
# Runs on cuda:1 so it never shares a card with the training run on cuda:0
# (project OOM rule: one card, one job).
#
#   setsid nohup /mnt/data/siping/ptq_work/qat/22b_run_check_loss.sh < /dev/null > /dev/null 2>&1 &
#
set -uo pipefail

source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd /tmp/pycharm_project_10ae9e2e

LOG_DIR=/mnt/data/siping/ptq_work/logs
mkdir -p "$LOG_DIR"

exec python -u manu/pipeline/npu/ptq/21b_check_loss.py \
  --n-batches 4 --batch 32 --gpu 1 \
  >> "$LOG_DIR/qat_check_loss.log" 2>&1
