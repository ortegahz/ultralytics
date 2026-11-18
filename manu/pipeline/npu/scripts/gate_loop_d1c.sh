#!/bin/bash
# 🟢 门禁通道 C（GPU0，10:25 新增）：**专攻「T2_fp=1 × T3 保线」的交叉窗口**
#   依据（11-18 晨门禁实测）：
#     · 2 路汤 640001@5+1100277@2：w=.78 ⇒ T2fp=2 / T3 ✓ ；w=.88 ⇒ T2fp=1 ✔ / T3 ✘（.948374 < .9485822）
#     ⇒ 「T2fp 2→1」与「T3 掉线」两个转变点在 w∈[.78,.88] ⇒ **该窗口的臂必须最快拿到官方门禁**。
#   做法：只打 bs640001 的 **w7*/w8*/w9*** 高权重段（与通道 A 的 w55/w56 段错开 ⇒ 不抢同一批臂），
#        每轮 8 个臂，10 分钟一轮；MemAvailable<40GB 拒绝启动（铁律）。
set -u
REPO=/tmp/pycharm_project_10ae9e2e; PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search; END="2026-11-19 08:00"
cd $REPO || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
exec 9>"$W/.lock_gateC"
flock -n 9 || { echo "[$(date "+%F %T")] [门禁C] 已有实例在跑 ⇒ 退出" >> $W/gate_loop_d1c.log; exit 0; }
log(){ echo "[$(date "+%F %T")] $*" >> $W/gate_loop_d1c.log; }
# 🟢🟢 2026-11-18 10:10 **PID 文件判活**（替代守望里的 `pgrep -f`）
#   坑：`pgrep -f "<脚本名>"` 会**匹配到任何提到该名字的 shell**（例如运维 ssh 命令本身）
#        ⇒ 守望误判「已在跑」⇒ 通道永远不重启。⇒ 改为脚本自己登记 PID + kill -0 验活。
PIDF=$W/.pid_gateC
echo $$ > "$PIDF"
trap 'rm -f "$PIDF"' EXIT

while :; do
  now=$(date +%s); end=$(date -d "$END" +%s)
  [ "$now" -lt "$end" ] || { log "[截止] 退出"; break; }
  avail=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
  if [ "${avail:-0}" -lt 40 ]; then log "[闸门] avail=${avail}GB < 40GB ⇒ 等 120s"; sleep 120; continue; fi
  # 🟢🟢 2026-11-18 10:35 **孤儿缓存构建清理**（防内存爆炸，根因修复）
  #   龙泉山热图缓存用**候选自身权重**（--hm-weights args.weights）⇒ 每臂必须重建 ⇒ 单个构建进程 RSS ≈ 5GB。
  #   坑：门禁被 `timeout` 杀掉/被人工杀掉时，它的构建子进程**变孤儿继续跑**（父进程没了仍占 5GB）
  #      ⇒ 一夜累积数十个 ⇒ 内存爆掉（11-18 晨实测 12 个孤儿 = 62GB）。
  #   修法：每轮前后清一次「父进程已死(PPID=1)」的构建进程；活门禁的构建进程父进程仍在 ⇒ 不受影响。
  for _p in $(ps -eo pid,ppid,args | grep "multiscale_heatmap_cache.py" | grep -v grep | awk '$2==1{print $1}'); do
    kill -9 "$_p" 2>/dev/null && echo "[清理] 孤儿缓存构建 pid=$_p" >> "$W/gate_orphan_clean.log"
  done

  log "[轮] 门禁C 开始 avail=${avail}GB（高权重交叉段）"
  timeout 3600 $PY -u manu/pipeline/npu/night_soup_batch.py --only-gate \
    --gate-family-prefix "bs640001:w7,bs640001:w8,bs640001:w9" \
    --gpu 0 --gate-family-quota 40 --max-gates 8 --mem-gate-gb 40 \
    --sid-base 1590000 --sid-span 1000 --deadline "$END" >> $W/day_gateC.log 2>&1
  log "[轮] 门禁C 结束"

  # 🟢🟢 2026-11-18 10:35 **孤儿缓存构建清理**（防内存爆炸，根因修复）
  #   龙泉山热图缓存用**候选自身权重**（--hm-weights args.weights）⇒ 每臂必须重建 ⇒ 单个构建进程 RSS ≈ 5GB。
  #   坑：门禁被 `timeout` 杀掉/被人工杀掉时，它的构建子进程**变孤儿继续跑**（父进程没了仍占 5GB）
  #      ⇒ 一夜累积数十个 ⇒ 内存爆掉（11-18 晨实测 12 个孤儿 = 62GB）。
  #   修法：每轮前后清一次「父进程已死(PPID=1)」的构建进程；活门禁的构建进程父进程仍在 ⇒ 不受影响。
  for _p in $(ps -eo pid,ppid,args | grep "multiscale_heatmap_cache.py" | grep -v grep | awk '$2==1{print $1}'); do
    kill -9 "$_p" 2>/dev/null && echo "[清理] 孤儿缓存构建 pid=$_p" >> "$W/gate_orphan_clean.log"
  done
  sleep 60
done
