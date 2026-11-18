#!/bin/bash
# 臂内细粒度 SWA 扫描（**零训练**）常驻循环 v2 —— 2026-11-13
#
# v1 的问题：--limit 20 固定
#   · 计划按「最佳单 epoch 的 nano」降序排 ⇒ 前 20 个臂几乎恒定
#   · R14+ 不断产出的新近失臂排在后面 ⇒ 永远扫不到
#   · 首轮 597 候选跑完后 --resume 会空转 ⇒ GPU2 白白闲置 ~40 h
# v2 的做法：--limit 递增 20 → 40 → 80 → 160（封顶）
#   ⇒ 每轮把扫描面扩一档 ⇒ 新臂一定能被扫到，且不会一次性挤爆 GPU
#
# 其它约定沿用 v1：
#   · --resume  ⇒ 跳过报告里已完成的候选（可中断续跑）
#   · --triad-first ⇒ 先扫「已有 triad epoch」的近失臂（真·升轨候选所在）
#   · --gate ⇒ triad 候选自动补跑官方三轨门禁
#   · nice 15 ⇒ 不抢自适应搜索的训练槽
set -u
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=.
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search
DL=$(date -d "2026-11-16 08:00" +%s)
LIM=20
echo "[$(date "+%F %T")] === fine_swa_loop v2 START (limit=$LIM) ===" >> $W/fine_swa_sweep.log
while true; do
  NOW=$(date +%s)
  if [ "$NOW" -gt "$DL" ]; then
    echo "[$(date "+%F %T")] fine_swa_loop v2 到达截止 ⇒ 退出" >> $W/fine_swa_sweep.log
    break
  fi
  # 切换脚本时旧的 sweep 进程可能仍在跑 ⇒ 等它结束，避免两个 sweep 抢 GPU2
  if pgrep -f "fine_swa_sweep[.]py" > /dev/null; then
    echo "[$(date "+%F %T")] 等待在跑的 sweep 结束（避免抢 GPU2）…" >> $W/fine_swa_sweep.log
    while pgrep -f "fine_swa_sweep[.]py" > /dev/null; do sleep 60; done
  fi
  nice -n 15 $PY -u manu/pipeline/npu/fine_swa_sweep.py \
      --triad-first --limit $LIM --resume --gate --gpu 2 \
      >> $W/fine_swa_sweep.log 2>&1
  RC=$?
  echo "[$(date "+%F %T")] fine_swa_loop v2 一轮结束 rc=$RC limit=$LIM" >> $W/fine_swa_sweep.log
  LIM=$((LIM * 2))
  if [ "$LIM" -gt 160 ]; then LIM=160; fi
  if [ "$RC" -ne 0 ]; then sleep 120; fi
  sleep 30
done
