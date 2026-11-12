#!/bin/bash
# 自适应闭环夜间调度（2026-11-11 夜 → 2026-11-12 09:00，服务器自主跑）
#   目标：击败 Baseline_v2.0 = trial_9863（nano 严格线 0.708648 / 稳健线 0.700969）
#   v3 特性：① 训练臂一律从 base init(cs_t0043.pt) 起训 ② epochs=5（与 epochs=20 的 ep5 逐位等价）
#             ③ 配方级（多 seed 均值）判优 ④ 官方三轨门禁只跑候选臂
# 用法（服务器）：setsid nohup bash manu/pipeline/npu/run_adaptive_night.sh > /dev/null 2>&1 &
set -u
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=.
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search
DEADLINE="${1:-2026-11-12 09:00}"
START_ROUND="${4:-1}"
echo "[$(date "+%F %T")] === ADAPTIVE v3 START | deadline=${DEADLINE} ===" >> $W/adaptive.log
$PY -u manu/pipeline/npu/adaptive_runner.py \
    --deadline "${DEADLINE}" --gpus 0,1,2,3 \
    --n-seed "${2:-6}" --n-mut "${3:-18}" --swa-top-n 6 --start-round "${4:-1}" \
    >> $W/adaptive.log 2>&1
echo "[$(date "+%F %T")] === ADAPTIVE v3 DONE ===" >> $W/adaptive.log
