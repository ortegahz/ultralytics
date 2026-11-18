#!/bin/bash
# 🟢 策展汤通道（显式集合 + 集合均值过线预测）｜每轮 12 组（= 官方门禁预算）
GPU=$1; REGION=$2; SHARD=$3; TAG=$4
W=/mnt/data/siping/runs/npu20ms/nano_search
LOG=$W/curated_${TAG}.log
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=.
DL=$(date -d "2026-11-17 09:00" +%s)
OFF=0; FAIL=0
while true; do
  NOW=$(date +%s)
  if [ "$NOW" -ge "$DL" ]; then echo "[$(date "+%F %T")] [$TAG] 到 deadline ⇒ 退出" >> $LOG; break; fi
  echo "[$(date "+%F %T")] [$TAG] 第 $((OFF/12+1)) 轮 offset=$OFF shard=$SHARD base=$REGION" >> $LOG
  T0=$(date +%s)
  $PY -u manu/pipeline/npu/basin_soup_targeted.py --gpu "$GPU" --synth-base "$REGION" \
      --curated-sets $W/curated/k5_s${SHARD}.json --curated-offset "$OFF" --curated-limit 12 \
      --gate-top 0 --max-eval 12 --report $W/basin_cur_${TAG}.json >> $LOG 2>&1
  RC=$?; T1=$(date +%s)
  OFF=$((OFF+12))
  if [ $((T1-T0)) -lt 150 ]; then
    FAIL=$((FAIL+1))
    echo "[$(date "+%F %T")] [$TAG] 快速退出 rc=$RC ⇒ 空转 $FAIL/40（等刷新器追加新集合）" >> $LOG
    if [ "$FAIL" -ge 40 ]; then echo "[$(date "+%F %T")] [$TAG] 连续失败过多 ⇒ 退出" >> $LOG; break; fi
    sleep 60
  else FAIL=0; fi
  sleep 10
done
