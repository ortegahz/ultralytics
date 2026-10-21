#!/usr/bin/env bash
# 启动路线 B 混训（5 epoch，每 epoch 自动三轨门禁 + 熔断），脱离 SSH 会话运行。
set -uo pipefail
cd /tmp/pycharm_project_10ae9e2e || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
export PYTHONPATH=.
RUN_DIR="${RUN_DIR:-runs/mixed_ft_route_b}"
mkdir -p "$RUN_DIR/logs"
LOG="$RUN_DIR/logs/train_stdout.log"
nohup python manu/training/mixed_ft_train.py \
  --epochs "${EPOCHS:-5}" \
  --batch "${BATCH:-32}" \
  --gate-gpus "${GATE_GPUS:-4}" \
  --fuse-mode "${FUSE_MODE:-stop}" \
  --out-dir "$RUN_DIR" > "$LOG" 2>&1 &
echo "TRAIN_PID=$!  LOG=$LOG  FUSE_MODE=${FUSE_MODE:-stop}"
sleep 3
