#!/bin/bash
# 🟢🟢 2026-11-20 09:55 **tipdose 专用通道**（针尖供体 1102033@1 亚断崖剂量 ε≤0.008 掺入基线族）
#
# ★ 为什么必须独立成通道（而不是等 w13 波次）：
#   ① w13 波次排在循环**最前**，但每轮要等 w12→w11→…→d5 全部跑完才回到 w13
#      ⇒ 实测昨夜 tipdose 臂**一臂未出**（w13 round 8 卡在队列里）。
#   ② 本通道用 `flock` + 常驻 wrapper ⇒ 进程**不是孤儿**（PPID≠1）
#      ⇒ 不会被 `day_supervisor_d1.sh` 的「孤儿 soup3 收割」误杀
#      （昨夜独立通道就是这么死的：PPID=1 ⇒ 被收割）。
#
# 铁律：起跑前查 MemAvailable（<60GB 让位给门禁）；OMP/MKL/OPENBLAS/NUMEXPR/NUMBA 全钉 4。
set -u
REPO=/tmp/pycharm_project_10ae9e2e; PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search; END="2026-11-23 08:30"
cd $REPO || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
log(){ echo "[$(date "+%F %T")] $*" >> $W/tipdose_loop.log; }
exec 9>"$W/.lock_tipdose"
flock -n 9 || { log "[闸门] 已有实例 ⇒ 退出"; exit 0; }
echo $$ > "$W/.pid_tipdose"
trap 'rm -f "$W/.pid_tipdose"; exit 143' INT TERM
RND=0
while :; do
  now=$(date +%s); end=$(date -d "$END" +%s)
  [ "$now" -lt "$end" ] || { log "[截止] $END ⇒ 退出"; break; }
  avail=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
  if [ "${avail:-0}" -lt 61440 ]; then log "[让位] avail=${avail}MB < 60GB ⇒ 等 300s"; sleep 300; continue; fi
  REC=$($PY $W/scripts/gen_tipdose.py $RND)
  if [ -z "$REC" ]; then log "[空配方] 等 300s"; sleep 300; continue; fi
  n=$(echo "$REC" | tr ';' '\n' | grep -c .)
  log "[轮] tipdose 开始 avail=${avail}MB 臂数=$n round=$RND"
  timeout 14400 $PY -u manu/pipeline/npu/night_soup3.py --gpu 1 --sid-base 2030000 --sid-span 4000 \
    --max-gates 0 --mem-gate-gb 40 --deadline "$END" --recipes "$REC" >> $W/soup3_tipdose.log 2>&1 9>&-
  log "[轮] tipdose 结束"
  RND=$((RND + 1))
  echo $RND > "$W/.tipdose_round"
  sleep 60
done
