#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# 22_run_qat_train.sh -- launch the RK3588 QAT fine-tune of trial_0080, DETACHED.
#
# WHY A .sh AND NOT AN INLINE COMMAND
# -----------------------------------
# A remote `cmd && cmd &` chain is killed when the SSH session drops; `setsid
# nohup <script>` survives it.  The training therefore has to be started through
# this file (project rule), not through an inline background chain.
#
# The script is idempotent-ish: the trainer itself is resumable -- it reads
# progress.json from the output directory and continues from the last COMPLETED
# epoch, so re-running this launcher after a crash picks up where it stopped.
#
# Start (on the server):
#   setsid nohup /mnt/data/siping/ptq_work/qat/22_run_qat_train.sh < /dev/null > /dev/null 2>&1 &
#
set -uo pipefail

source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd /tmp/pycharm_project_10ae9e2e

LOG_DIR=/mnt/data/siping/ptq_work/logs
OUT_DIR=/mnt/data/siping/ptq_work/qat/trial_0080
mkdir -p "$LOG_DIR" "$OUT_DIR"

# GPU 2, not GPU 0: at 2026-10-22 18:21 GPU 0 already carried another job
# (PIDs 1344908 / 1345159, 3344 MiB each) and this run OOM'd there
# ("23.55 GiB total, 770.56 MiB free").  GPU 2 is idle (18 MiB) and the QAT
# peak is ~14 GiB, so it fits.  GPU 1 is kept free for the evaluation /
# simulator jobs (one card, one job).
TRAIN_GPU=${TRAIN_GPU:-2}

exec python -u manu/pipeline/npu/ptq/21_train_qat.py \
  --weights runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt \
  --out-dir "$OUT_DIR" \
  --epochs 3 \
  --batch 32 \
  --imgsz 640 \
  --workers 8 \
  --gpu "$TRAIN_GPU" \
  --uav-per-lq 2.5 \
  --lr-scale 1.0 \
  --scale-lr 1e-4 \
  --calib-batches 16 \
  --log-every 20 \
  --resume \
  >> "$LOG_DIR/qat_train_trial_0080.log" 2>&1
