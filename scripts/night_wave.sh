#!/bin/bash
# 夜间接力 v5（2026-11-17 15:15）—— **单家族集中打击波**
#
# 🔴 口径复核（15:15 读代码确认的，两条都必须满足）：
#   (a) `auto_promote_watch`：同**家族键** `soup:<parent>:w<成员数>`（**权重不进键**）里
#       **≥4 个严格六项全过** + 其中有板卡 T4 PASS。
#   (b) `promote_candidate` 的**种子复现关**：候选臂自己的**原始 soup_family**
#       （= 同 parent **同权重**，如 `bs640001:w55`）里 **≥2 个严格六项全过**。
# ⇒ **只有「同 parent + 同权重」才算一份可升轨的证据** ⇒ 全部火力压到
#   **parent=640001:5、w=0.55**（`bs640001:w55` 已有 1 票 1106580；`soup:bs640001:w2` 已有 1 票）。
#
# 用法：night_wave.sh <gpu> <sid_base> <children> <weights>
#   每个通道用**互不重叠**的子池与权重 ⇒ 绝不重复造同一个模型（否则家族 n 会被同款模型灌水）。
W=/mnt/data/siping/runs/npu20ms/nano_search
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
REPO=/tmp/pycharm_project_10ae9e2e
cd "$REPO" || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4

GPU=$1; SIDBASE=$2; CHILDREN=$3; WEIGHTS=$4; MEMG=${5:-44}; MAXG=${6:-36}
avail_gb() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
wait_gpu() {
  while pgrep -f "night_soup_batch.py --gpu $GPU" > /dev/null 2>&1; do
    echo "[$(date '+%F %T')] 等 gpu$GPU 空闲… avail=$(avail_gb)GB"
    sleep 60
  done
}
echo "[$(date '+%F %T')] ▶ wave gpu=$GPU sid=$SIDBASE w=$WEIGHTS memgate=${MEMG}GB children=$CHILDREN"
wait_gpu
while true; do
  a=$(avail_gb)
  if [ "${a:-0}" -ge "$MEMG" ]; then break; fi
  echo "[$(date '+%F %T')] 内存闸门 avail=${a}GB < ${MEMG}GB ⇒ 等 120s"
  sleep 120
done
"$PY" -u manu/pipeline/npu/night_soup_batch.py --gpu "$GPU" --parents "640001:5" \
    --children-arms "$CHILDREN" --weights "$WEIGHTS" --sid-base "$SIDBASE" \
    --max-gates "$MAXG" --gate-family-quota 14 --mem-gate-gb "$MEMG" \
    --deadline "2026-11-18 07:00" >> "$W/soup_wave_gpu${GPU}.log" 2>&1
echo "[$(date '+%F %T')] ■ wave gpu=$GPU 结束"
