#!/bin/bash
# 夜间接力 v2（2026-11-17 15:55 立项）—— **冠军递归汤**轴（零训练成本）
#
# 🔴 核心口径发现（今晚 15:40 实测代码读出来的，不是猜的）：
#    `family_key._soup_family_key()` 的正则 `^(bs\d+):([A-Za-z]+)(\d*)$`
#    只取 label 的**字母**部分 + **成员数** ⇒ **权重数字被丢弃**！
#    ⇒ 同一个 parent 的**所有 2 成员汤**（w=0.55/0.60/0.65/0.70/0.75…）**同属一个家族**
#      `soup:bs640001:w2`（1106580 就在里面，已 1 个六项全过）
#    ⇒ 升轨关「同族 n≥4」= **同 parent 再买 3 张彩票**（而不是每个权重各买 4 张）。
#
# 🎯 打法：把**已接近六项全过且 t2_fp=1** 的汤臂当**子**，跟**干净父**做 2 成员汤
#    （子臂本身 nano/nano4/F1 都在基线之上 ⇒ 汤的三轴全过率大幅提高）。
#    近失子臂池：1106580(6/6) 1106512(5/6 t1-1.4e-5) 1106562(5/6 f1-1.4e-5) 1112006(4/6 T1/T2/T3全过)
#
# 资源铁律：每批启动前查 MemAvailable（≥40GB 才启）；批间错峰 30s；
#          OMP/MKL/OPENBLAS/NUMEXPR/NUMBA=4；单批 deadline 07:00；通道并发 ≤ 3。
W=/mnt/data/siping/runs/npu20ms/nano_search
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
REPO=/tmp/pycharm_project_10ae9e2e
cd "$REPO" || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4

avail_gb() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
load1()   { cut -d' ' -f1 /proc/loadavg; }

wait_gpu() {   # 等某个 GPU 上已有的汤批次退出（避免同卡双跑）
  local g=$1
  while pgrep -f "night_soup_batch.py --gpu $g" > /dev/null 2>&1; do
    echo "[$(date '+%F %T')] 等 gpu$g 上的旧批次退出… avail=$(avail_gb)GB load=$(load1)"
    sleep 60
  done
  echo "[$(date '+%F %T')] gpu$g 空闲 ⇒ 准备接棒"
}

run_batch() {  # gpu sidbase parents children weights quota maxg tag
  local gpu=$1 sidbase=$2 parents=$3 children=$4 weights=$5 quota=$6 maxg=$7 tag=$8
  echo "[$(date '+%F %T')] ▶ $tag 启动 gpu=$gpu sid=$sidbase w=$weights"
  while true; do
    a=$(avail_gb); l=$(load1)
    if [ "${a:-0}" -ge 40 ]; then break; fi
    echo "[$(date '+%F %T')] 内存闸门 avail=${a}GB < 40GB ⇒ 等 120s"
    sleep 120
  done
  "$PY" -u manu/pipeline/npu/night_soup_batch.py --gpu "$gpu" --parents "$parents" \
      --children-arms "$children" --weights "$weights" --sid-base "$sidbase" \
      --max-gates "$maxg" --gate-family-quota "$quota" --mem-gate-gb 40 \
      --deadline "2026-11-18 07:00" >> "$W/soup_batch_seq2.log" 2>&1
  echo "[$(date '+%F %T')] ■ $tag 结束 sid=$sidbase"
}

# 近失/全过子臂池（全部 t2_fp=1 ⇒ 汤的 T2 有机会守住 ≤1）
CHAMP="1106580:5,1106512:5,1106562:5,1112006:5,1100296:1,98902:5,98913:5,98905:5,600004:5,300001:5,620002:5"
WIDS="0.55,0.6,0.65,0.7,0.75"     # 全部映射到 label 字母 'w' ⇒ 同家族 soup:bs<parent>:w2

sleep 60

# 批次A：父 640001（家族 soup:bs640001:w2 已有 1106580 六项全过 ⇒ 再要 3 个）
( wait_gpu 0; run_batch 0 1114000 "640001:5" "$CHAMP" "$WIDS" 12 44 "batchA-champion-640001" ) &
sleep 30
# 批次B：父 610000（家族 soup:bs610000:w2 已有 1106562 六项全过）
( wait_gpu 1; run_batch 1 1115000 "610000:5" "$CHAMP" "$WIDS" 12 44 "batchB-champion-610000" ) &
sleep 30
# 批次C：父 69901/98905/600004（69901 家族有 1106512 近失；另两个开新家族）
( wait_gpu 2; run_batch 2 1116000 "69901:5,98905:5,600004:5" \
      "1106580:5,1106512:5,1106562:5,1112006:5,1100296:1,98902:5,98913:5" "0.55,0.6,0.65,0.7" 12 40 "batchC-champion-more" ) &
sleep 30
wait
echo "[$(date '+%F %T')] 批次 A/B/C 全部结束 ⇒ 进入收尾门禁收割"
# 收尾：把所有还没打门禁的「三轴全过」汤臂按家族配额打满（12/族），争取把某个家族顶到 n≥4
"$PY" -u manu/pipeline/npu/night_soup_batch.py --gpu 0 --only-gate --gate-family-quota 12 \
    --max-gates 40 --mem-gate-gb 38 --deadline "2026-11-18 08:00" >> "$W/soup_batch_seq2.log" 2>&1
echo "[$(date '+%F %T')] 收尾门禁收割结束"
