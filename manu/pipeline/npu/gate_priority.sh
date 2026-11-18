#!/bin/bash
# 定向门禁通道：把「岭线 w56 组」放在门禁队列最前面
#   理由：1220011-1220014 是关A(n>=4)/关B(w56格第2票) 的最短路径，
#         但它们 margin 最低(+0.006σ) ⇒ 在 margin 降序队列里排最后 ⇒ 被拖到后半夜。
#   保护：MemAvailable < 40GB 不开新门禁；门禁目录已存在 ⇒ 自动跳过（不与主循环抢臂）。
set -u
REPO=/tmp/pycharm_project_10ae9e2e
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search
LOG=$W/gate_priority.log
END="2026-11-18 07:00"
cd $REPO || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
log(){ echo "[$(date "+%F %T")] $*" >> $LOG; }
SIDS="1220011,1220012,1220013,1220014,1220010,1220008,1220003,1220015,1220009,1220005,1220006,1220007,1220004"
while :; do
  now=$(date +%s); end=$(date -d "$END" +%s)
  [ "$now" -lt "$end" ] || { log "[截止] $END 退出"; break; }
  avail=$(awk "/MemAvailable/{printf \"%d\", \$2/1024}" /proc/meminfo)
  if [ "${avail:-0}" -lt 40 ]; then log "[闸门] avail=${avail}GB < 40GB => 等 120s"; sleep 120; continue; fi
  log "[轮] 定向门禁开始 avail=${avail}GB"
  timeout 14000 $PY -u manu/pipeline/npu/night_soup_batch.py --only-gate --gate-sids "$SIDS" \
      --gpu 1 --gate-family-quota 25 --max-gates 60 --mem-gate-gb 40 \
      --sid-base 1190000 --sid-span 1000 --deadline "$END" >> $W/gate_priority_run.log 2>&1
  log "[轮] 定向门禁结束"
  sleep 120
done
