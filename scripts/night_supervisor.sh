#!/bin/bash
# 🔴 夜间通道自愈守望 v3（2026-11-17 17:15）：**同盆地岭线精扫 + 种子复现波**
#   通道表（臂号段独占，sid-base/sid-span ⇒ 绝不撞号）：
#     1220000 GPU1 冠军配对 × 细权重岭线扫描（0.5325~0.5675）
#     1210000 GPU3 同盆地三兄弟 × 种子权重
#     1180000 GPU2 gapfill（P 系 × w56~w64 backlog）
#   另监督「种子复现波」(GPU0, 臂号 1300001-4)：掉线就重拉（幂等：权重已存在则跳过训练）
#   作废段（实测零票，不再拉起）：1160000 / 1111000 / 1117000 / 1120000 / 1200000
set -u
REPO=/tmp/pycharm_project_10ae9e2e
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search
LOG=$W/supervisor.log
END="2026-11-18 07:00"
cd $REPO || exit 1
log(){ echo "[$(date '+%F %T')] $*" >> $LOG; }

declare -A KIDS GPU WTS MAXG QUOTA MEMG
SEEDK="1100296:2,1100275:2,1100277:1,1102000:2,1102004:2,1102087:1,1102006:2,1102088:2,1102012:2,1102092:2"
SIBK="1102087:2,1102088:1,1102092:1"
FINE="0.5325,0.535,0.5375,0.54,0.5425,0.545,0.5475,0.5525,0.555,0.5575,0.5625,0.565,0.5675"
GOOD="1100275:1,1100277:2,1102006:1,1102088:1,1100296:1,1102087:2,1104407:1,1100287:1,1102000:1,1102012:1,1102092:1,1104446:1,1104411:1,1102004:1,1104445:1,1104443:2"

KIDS[1220000]="1100296:1"; GPU[1220000]=1; WTS[1220000]="$FINE"; MAXG[1220000]=40; QUOTA[1220000]=8; MEMG[1220000]=44
KIDS[1210000]="$SIBK";  GPU[1210000]=3; WTS[1210000]="0.545,0.555,0.565"; MAXG[1210000]=30; QUOTA[1210000]=6; MEMG[1210000]=44
KIDS[1180000]="$GOOD";  GPU[1180000]=2; WTS[1180000]="0.56,0.57,0.58,0.59,0.61,0.62,0.63,0.64"; MAXG[1180000]=60; QUOTA[1180000]=12; MEMG[1180000]=44

declare -A RESTARTS
for k in "${!GPU[@]}"; do RESTARTS[$k]=0; done
RP_RESTARTS=0

while :; do
  now=$(date +%s); end=$(date -d "$END" +%s)
  [ "$now" -lt "$end" ] || { log "[截止] $END ⇒ 守望退出"; break; }
  for k in "${!GPU[@]}"; do
    if pgrep -f "night_soup_batch.py .*--sid-base $k " >/dev/null 2>&1; then continue; fi
    [ "${RESTARTS[$k]}" -lt 2 ] || { log "[跳过] 臂号段 $k 已重启 2 次 ⇒ 不再拉起"; continue; }
    avail=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
    if [ "${avail:-0}" -lt 40 ]; then log "[闸门] MemAvailable=${avail}GB < 40GB ⇒ 暂不重启 $k"; continue; fi
    RESTARTS[$k]=$(( ${RESTARTS[$k]} + 1 ))
    log "[重启] 通道 sid-base=$k 卡 GPU${GPU[$k]}（第 ${RESTARTS[$k]} 次）MemAvailable=${avail}GB"
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
    setsid nohup $PY -u manu/pipeline/npu/night_soup_batch.py --gpu ${GPU[$k]} \
      --parents 640001:5 --children-arms "${KIDS[$k]}" --weights "${WTS[$k]}" \
      --sid-base $k --sid-span 1000 --max-gates ${MAXG[$k]} --gate-family-quota ${QUOTA[$k]} \
      --mem-gate-gb ${MEMG[$k]} --deadline "$END" >> $W/soup_sup_${k}.log 2>&1 < /dev/null &
    sleep 60
  done
  # ---- 种子复现波（GPU0）自愈 ----
  if ! pgrep -f "seed_replicate.py" >/dev/null 2>&1; then
    if [ "$RP_RESTARTS" -lt 3 ]; then
      avail=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
      if [ "${avail:-0}" -ge 42 ]; then
        RP_RESTARTS=$(( RP_RESTARTS + 1 ))
        log "[重启] 种子复现波（第 ${RP_RESTARTS} 次）GPU0 MemAvailable=${avail}GB"
        export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
        GPU=0 DEADLINE="$END" setsid nohup $PY -u /tmp/seed_replicate.py >> $W/seed_replicate_run.log 2>&1 < /dev/null &
        sleep 60
      else
        log "[闸门] MemAvailable=${avail}GB < 42GB ⇒ 种子复现波暂不重启"
      fi
    else
      log "[跳过] 种子复现波已重启 3 次"
    fi
  fi
  sleep 600
done
