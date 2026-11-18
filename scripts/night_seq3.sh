#!/bin/bash
# 夜间接力 v3（2026-11-17 14:55）—— **冠军细权重扫**（GPU3；训练轴已停，腾出的卡）
# 目的：把更多彩票打进**唯一有严格六项全过臂的家族** `soup:bs640001:w2`（成员 = 1106580）。
# 手法：父 640001@5 × 子「近失冠军汤」× **细一档的权重**（0.52/0.53/0.56/0.58/0.62）
#   ⇒ 权重数字不影响家族键（label 字母恒为 'w'）⇒ 全部计入同一个家族 n。
# 铁律：MemAvailable ≥ 48GB 才启新臂（比主批更严，给其它 3 个批次留余量）；OMP/MKL/…=4。
W=/mnt/data/siping/runs/npu20ms/nano_search
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
REPO=/tmp/pycharm_project_10ae9e2e
cd "$REPO" || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4

avail_gb() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }

run_batch() {
  local gpu=$1 sidbase=$2 parents=$3 children=$4 weights=$5 quota=$6 maxg=$7 memg=$8 tag=$9
  echo "[$(date '+%F %T')] ▶ $tag gpu=$gpu sid=$sidbase w=$weights memgate=${memg}GB"
  while true; do
    a=$(avail_gb)
    if [ "${a:-0}" -ge "$memg" ]; then break; fi
    echo "[$(date '+%F %T')] 内存闸门 avail=${a}GB < ${memg}GB ⇒ 等 120s"
    sleep 120
  done
  "$PY" -u manu/pipeline/npu/night_soup_batch.py --gpu "$gpu" --parents "$parents" \
      --children-arms "$children" --weights "$weights" --sid-base "$sidbase" \
      --max-gates "$maxg" --gate-family-quota "$quota" --mem-gate-gb "$memg" \
      --deadline "2026-11-18 07:00" >> "$W/soup_batch_seq3.log" 2>&1
  echo "[$(date '+%F %T')] ■ $tag 结束 sid=$sidbase"
}

sleep 30
# batchD：冠军 × 细权重（25 臂，全进 soup:bs640001:w2）
run_batch 3 1117000 "640001:5" \
  "1106580:5,1106512:5,1106562:5,1112006:5,1100296:1" \
  "0.52,0.53,0.56,0.58,0.62" 12 24 48 "batchD-champion-finew"
sleep 30
# batchE：老干净盆地 × 冠军父（跨盆地混合，20 臂，同样进 soup:bs640001:w2）
run_batch 3 1118000 "640001:5" \
  "12679:5,69914:5,69902:5,69906:5,99002:5,99100:5,98903:5,99000:5,98901:5,99101:5" \
  "0.55,0.6" 12 20 48 "batchE-oldclean-mix"
echo "[$(date '+%F %T')] night_seq3 全部结束"
