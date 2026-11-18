#!/bin/bash
# 抛光**级联**主循环（2026-11-13）
#   起因：`trial_999821` 打穿六项 + 板卡 T4 ⇒ 抛光轴是当前唯一有效轴；
#         但固定锚点表跑完就没有新组合了 ⇒ 做成**自动加深的级联**：
#           每一轮重新从台账挑「离六项全过最近」的 K 个锚点（滑窗 ⇒ 换新臂），
#           再跑一遍配置切片 ⇒ 抛光产物自己又成为新锚点 ⇒ 天然支持深度 2/3 级联。
#   口径：锚点排序与调度器 `adaptive_runner._near_miss_score` 一致（同一套 BASE + 归一化）。
#   并行：GPU 2,3（GPU 0,1 给第二批抛光旁路；调度器 4 卡满载）⇒ 绝不杀板上/服务器他人进程。
#   臂号：1000000+ 段（保留段，调度器跳过 ⇒ 不会撞臂号）。
#   截止：到点停止（逐臂检查，不会半途丢权重）。
set -u
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=.
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search
LOG=$W/polish_cascade.log
DL=$(date -d "2026-11-16 07:00" +%s)
K=8                       # 每轮锚点数
CFG_SLICE=4               # 每轮配置切片长度
# 配置表刻意与「第二批」和调度器 WARM_CONFIGS **不重叠**（warm_key 去重是跨进程共享的）
# ⇒ 级联专门开拓新区域：λ 更强锚定（2e-3/3e-3）、λ 更弱（5e-4）、ep=3、lr=0.25
ALL_CFGS="0.2:0.002:1:1,0.1:0.002:1:1,0.15:0.002:1:1,0.05:0.002:1:1,0.3:0.002:1:1,0.2:0.003:1:1,0.1:0.0005:1:1,0.15:0.0005:1:1,0.05:0.0005:1:1,0.2:0.0005:1:2,0.1:0.0005:1:2,0.15:0.0005:1:2,0.2:0.001:1:3,0.1:0.001:1:3,0.15:0.001:1:3,0.25:0.001:1:1"
N_CFG=$(echo "$ALL_CFGS" | tr ',' '\n' | wc -l)
R=0
echo "[$(date "+%F %T")] === polish_cascade START (gpus=2,3) ===" >> $LOG
while true; do
  NOW=$(date +%s)
  if [ "$NOW" -gt "$DL" ]; then
    echo "[$(date "+%F %T")] polish_cascade 到达截止 ⇒ 退出" >> $LOG
    break
  fi
  # 配置切片（循环推进）
  START=$(( (R * CFG_SLICE) % N_CFG ))
  SLICE=""
  for i in $(seq 0 $((CFG_SLICE - 1))); do
    IDX=$(( (START + i) % N_CFG ))
    C=$(echo "$ALL_CFGS" | cut -d, -f$((IDX + 1)))
    SLICE="${SLICE:+$SLICE,}$C"
  done
  # 锚点（滑窗 ⇒ 每轮换一批臂；抛光产物会自动进入下一轮）
  ANCHORS=$($PY manu/pipeline/npu/polish_cascade_anchors.py --top-k $K --offset $((R * K)) 2>>$LOG)
  if [ -z "$ANCHORS" ]; then
    echo "[$(date "+%F %T")] [r$R] 没有可用锚点 ⇒ 等 10 min 再试" >> $LOG
    sleep 600
    R=$((R + 1))
    continue
  fi
  echo "[$(date "+%F %T")] [r$R] 锚点=$ANCHORS ｜ 配置=$SLICE" >> $LOG
  BASE=$(( 1000000 + R * 300 ))
  $PY -u manu/pipeline/npu/polish_nearmiss.py \
      --anchors "$ANCHORS" --configs "$SLICE" \
      --trial-base $BASE --gpus 2,3 --deadline "2026-11-16 06:45" \
      --report $W/polish_cascade.json >> $LOG 2>&1
  echo "[$(date "+%F %T")] [r$R] 本轮结束 rc=$? base=$BASE" >> $LOG
  R=$((R + 1))
  sleep 30
done
