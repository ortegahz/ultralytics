#!/bin/bash
# 🟢 门禁通道 C（GPU0）：按「家族能过几项」排队（gate_prio.py）⇒ 不再把机时花在死路上
#   坑①：默认按 bench margin 排序 ⇒ H1 主攻臂（边际仅 +0.08σ）被高边际死路臂挤掉。
#   坑②：通道之间抢同一个臂 ⇒ gate_prio.py 用 mkdir 原子占号（claim）自动错开。
set -u
REPO=/tmp/pycharm_project_10ae9e2e; PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search; END="${GATE_END:-2026-11-30 09:00}"
cd $REPO || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
log(){ echo "[$(date "+%F %T")] $*" >> $W/gate_loop_C.log; }
PIDF=$W/.pid_gateC
exec 9>"$W/.lock_gateC"
flock -n 9 || { log "[闸门] 已有实例在跑 ⇒ 退出"; exit 0; }
echo $$ > "$PIDF"
cleanup_gate(){ [ "$(cat "$PIDF" 2>/dev/null)" = "$$" ] && rm -f "$PIDF"; }
trap cleanup_gate EXIT
trap 'cleanup_gate; exit 143' INT TERM
while :; do
  now=$(date +%s); end=$(date -d "$END" +%s)
  [ "$now" -lt "$end" ] || { log "[截止] 退出"; break; }
  avail=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
  if [ "${avail:-0}" -lt 40960 ]; then log "[闸门] avail=${avail}MB < 40960MB ⇒ 等 120s"; sleep 120; continue; fi
  for _p in $(ps -eo pid,ppid,args | grep "multiscale_heatmap_cache.py" | grep -v grep | awk '$2==1{print $1}'); do
    kill -9 "$_p" 2>/dev/null && echo "[清理] 孤儿缓存构建 pid=$_p" >> "$W/gate_orphan_clean.log"
  done
  SIDS=$($PY $W/scripts/gate_prio.py C 8)
  log "[轮] 门禁C 开始 avail=${avail}MB sids=${SIDS}"
  if [ -n "$SIDS" ]; then
# 🔴🔴 2026-11-23 12:5x timeout 3600 → 5400：单轮 8 臂 × ~9.5min ≈ 76min > 60min
#   ⇒ 原设置每轮被砍在第 6 臂 ⇒ 门禁吞吐白丢 25%。现与 CLAIM_TTL(100min) 配套。
    timeout 5400 $PY -u manu/pipeline/npu/night_soup_batch.py --only-gate --gate-sids "$SIDS" \
      --gpu 0 --gate-family-quota 40 --max-gates 8 --mem-gate-gb 40 \
      --deadline "$END" >> $W/day_gateC.log 2>&1 9>&-
  else
    log "[轮] 门禁C 无候选 ⇒ 睡 60s"
  fi
  log "[轮] 门禁C 结束"
  for _p in $(ps -eo pid,ppid,args | grep "multiscale_heatmap_cache.py" | grep -v grep | awk '$2==1{print $1}'); do
    kill -9 "$_p" 2>/dev/null && echo "[清理] 孤儿缓存构建 pid=$_p" >> "$W/gate_orphan_clean.log"
  done
  sleep 60 9>&-
done
