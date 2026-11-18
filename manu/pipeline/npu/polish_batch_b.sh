#!/bin/bash
# 抛光旁路**第二批**（2026-11-13 15:35 启动）
#   起因：今天 `trial_999821`（= 12679@3 + lr0.2/λ1e-3/fb1/ep1 抛光）六项全过
#        + 板卡 T4 19.618 ms ⇒ **L2 锚定抛光轴从「证伪」变成「打穿」** ⇒ 立刻加码。
#   锚点：999821@1（新 beat）+ 6 个 5/6 近失臂（12519/12694/12709/12520/12053）
#   配置：小 lr 家族加密（lr ∈ {0.05,0.1,0.15,0.2,0.3} × ep ∈ {1,2}），
#         赢家 (0.2,1e-3,1,1) 排第一；lr≥0.6 不在本批（步子大会踢出可行域）。
#   纪律：与调度器共用台账 ⇒ 自动参与六项审计 / 板卡 T4 守望 / 抛光锚点选择；
#         臂号段 999831+（保留段，调度器会跳过）。
set -u
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=.
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search
# 等第一批（999821~999826）跑完，避免两个抛光进程抢同一批 GPU
while pgrep -f "polish_nearmiss[.]py" > /dev/null; do sleep 60; done
echo "[$(date "+%F %T")] === polish_batch_b START ===" >> $W/polish_batch_b.log
nice -n 19 $PY -u manu/pipeline/npu/polish_nearmiss.py \
  --anchors "999821:1,12679:3,96835:5,12519:3,12694:5,12709:4,12520:4,12053:4" \
  --configs "0.2:0.001:1:1,0.1:0.001:1:1,0.15:0.001:1:1,0.3:0.001:1:1,0.05:0.001:1:1,0.2:0.001:1:2,0.1:0.001:1:2,0.15:0.001:1:2,0.3:0.001:1:2,0.2:0.0005:1:1,0.2:0.001:0:1,0.2:0:1:1" \
  --trial-base 999831 --gpus 0,1 --deadline "2026-11-16 07:00" \
  --report $W/polish_batch_b.json \
  >> $W/polish_batch_b.log 2>&1
echo "[$(date "+%F %T")] === polish_batch_b DONE rc=$? ===" >> $W/polish_batch_b.log
