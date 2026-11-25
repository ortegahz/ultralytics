#!/bin/bash
# 🟢🟢 T1 收口扫网 · **第 4 条门禁通道**（2026-11-23 新增）
#
# ★ 定位：`gate_loop_d1.sh`(A) / `_d1b.sh`(B) / `_d1c.sh`(C) 之外的**一条额外通道**，
#   跑在**专攻通道（原 gate_priority13）腾出来的 GPU2** 上。
#   ⚠️ **门禁通道总数保持 4 条（3 永久 + 1 额外）** —— 不新增第 5 条 ⇒ 不违反「门禁通道铁律 ≤3(+1)」，
#      也不触碰当年「5 通道 × 4 分片 × 4.5GB = 68GB 把内存打爆」的事故。
#
# ★ 为什么值得开：机时瓶颈在**门禁**（~9 min/臂）而不是合成（~70 s/臂）。
#   实测吞吐：合成 51 臂/小时/通道，门禁 6.5 臂/小时/通道 ⇒ **门禁是 8 倍瓶颈**。
#   ⇒ 扫网新臂的消化速度决定了搜索速度 ⇒ 多一条门禁通道 = 搜索速度 +33%。
#
# ★ 取臂口径：直接复用 `gate_prio.py`（含 2026-11-23 的**分辨率地板过滤器**）
#   ⇒ 队列里不会有 maxΔ<0.03 的废料臂；队首永远是「严格三轴全过且非基线克隆」的臂
#   （含本轮 T1 收口扫网新产出的 FAM5+供体 6 成员臂）。
#
# ★ 资源纪律：flock 互斥 + PID 文件 + **MemAvailable 闸门（<45GB 停、<60GB 不启动）**
#   + 截止时间 + **按 PID 判活**（禁 pkill -f）。
set -u
REPO=/tmp/pycharm_project_10ae9e2e; PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search; END="${T1CLOSEGATE_END:-2026-11-30 09:00}"
GPU="${T1CLOSEGATE_GPU:-2}"
CHAN="${T1CLOSEGATE_CHAN:-D}"
MAXG="${T1CLOSEGATE_MAXG:-8}"
cd "$REPO" || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
LOG=$W/t1close_gate_gpu${GPU}.log
log(){ echo "[$(date "+%F %T")] $*" >> "$LOG"; }
exec 9>"$W/.lock_t1close_gate_gpu${GPU}"
flock -n 9 || { log "🔴 已有实例在跑 ⇒ 退出"; exit 0; }
echo $$ > "$W/.pid_t1close_gate_gpu${GPU}"
cleanup(){ [ "$(cat "$W/.pid_t1close_gate_gpu${GPU}" 2>/dev/null)" = "$$" ] && rm -f "$W/.pid_t1close_gate_gpu${GPU}"; }
trap cleanup EXIT
trap 'cleanup; exit 143' INT TERM
log "🔒 启动 t1close 门禁通道 gpu=$GPU chan=$CHAN maxgates=$MAXG END=$END pid=$$"
while :; do
  now=$(date +%s); end=$(date -d "$END" +%s)
  [ "$now" -lt "$end" ] || { log "[截止] $END ⇒ 退出"; break; }
  avail=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
  if [ "${avail:-0}" -lt 45000 ]; then log "[闸门] avail=${avail}MB < 45000MB ⇒ 等 120s"; sleep 120; continue; fi
  for _p in $(ps -eo pid,ppid,args | grep "multiscale_heatmap_cache.py" | grep -v grep | awk '$2==1{print $1}'); do
    kill -9 "$_p" 2>/dev/null && log "[清理] 孤儿缓存构建 pid=$_p"
  done
  SIDS=$($PY $W/scripts/gate_prio.py "$CHAN" "$MAXG" 2>>"$LOG")
  log "[轮] 门禁$CHAN 开始 avail=${avail}MB sids=${SIDS}"
  if [ -n "$SIDS" ]; then
# 🔴🔴 2026-11-23 12:5x timeout 3600 → 5400：单轮 8 臂 × ~9.5min ≈ 76min > 60min
#   ⇒ 原设置每轮被砍在第 6 臂 ⇒ 门禁吞吐白丢 25%。现与 CLAIM_TTL(100min) 配套。
    timeout 5400 $PY -u manu/pipeline/npu/night_soup_batch.py --only-gate --gate-sids "$SIDS" \
      --gpu "$GPU" --gate-family-quota 40 --max-gates "$MAXG" --mem-gate-gb 45 \
      --deadline "$END" >> $W/day_gateD.log 2>&1 9>&-
  else
    log "[轮] 门禁$CHAN 无候选 ⇒ 睡 60s"
  fi
  log "[轮] 门禁$CHAN 结束"
  sleep 60 9>&-
done
log "t1close 门禁通道结束"
