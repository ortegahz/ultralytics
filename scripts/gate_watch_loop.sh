#!/bin/bash
# 🔴 门禁**优先**守望循环（2026-11-17 16:45 新增）
#   问题：各汤通道是「先合成完再统一门禁」⇒ 票（严格六项全过）要等几小时才落地，
#         而升轨要的是**票数**（家族 n≥4 + 同权重 ≥2）⇒ 票的**产生速度**才是瓶颈。
#   做法：独立循环每 2 min 重扫台账 ⇒ 把「三轴全过且还没门禁」的臂**立刻**送官方门禁。
#   安全：① 复用 `run_official_gate` 的报告复用机制（报告已存在 ⇒ 不重算）
#        ② 门禁目录已存在（正在打）⇒ 跳过 ⇒ 天然避免与其它通道抢同一个臂
#        ③ MemAvailable < 44GB ⇒ 不开跑（铁律）
set -u
REPO=/tmp/pycharm_project_10ae9e2e
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search
LOG=$W/gate_watch.log
END="2026-11-18 07:00"
GPU=${1:-0}
cd $REPO || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
log(){ echo "[$(date '+%F %T')] $*" >> $LOG; }
while :; do
  now=$(date +%s); end=$(date -d "$END" +%s)
  [ "$now" -lt "$end" ] || { log "[截止] $END ⇒ 退出"; break; }
  avail=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
  if [ "${avail:-0}" -lt 44 ]; then log "[闸门] MemAvailable=${avail}GB < 44GB ⇒ 等 120s"; sleep 120; continue; fi
  log "[门禁轮] gpu=$GPU avail=${avail}GB 开始扫描"
  timeout 40000 $PY -u manu/pipeline/npu/night_soup_batch.py --only-gate --gpu "$GPU" \
      --gate-family-quota 25 --max-gates 60 --mem-gate-gb 44 \
      --sid-base 1190000 --sid-span 1000 --deadline "$END" >> $W/gate_watch_run.log 2>&1
  log "[门禁轮] 本轮结束"
  sleep 120
done
