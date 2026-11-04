#!/usr/bin/env bash
# ============================================================
# ★ 服务器端「24 序列 FPV8 语义多尺度热图缓存」重建（GPU 分片）
#
#   用法（服务器上）：
#     cd /tmp/pycharm_project_10ae9e2e
#     nohup bash manu/tracking/lookahead/server_overnight/rebuild_fpv8_24seq_server.sh \
#          > runs/lookahead/final/fpv8_24seq.log 2>&1 &
#
#   ★ 为什么必须用 GPU（2026-11-03 修正）：
#     官方 24 序列缓存（runs/optuna_mix_ratio/trials/trial_0080/lq_eval/ms_cache，
#     2026-10-22 建的、产出冻结基线 F1 94.747% 的那份）就是用 **GPU 分片**建的：
#       run_trial0080_longquan_eval.sh §2 → --device "$i" --shard "$i/$NGPU"
#     ⇒ 用 CPU 重建会（a）慢 13~16 倍（实测 CPU 72~100 s/chunk vs GPU 6.0 s/chunk），
#       （b）**数值口径对不上官方基线**（FP32 前后处理顺序不同 → 阈值边界可能翻像素）。
#     本脚本因此完全对齐官方口径：--device GPU + --shard 分片 + 同超参。
#
#   并行度：**每卡 1 个 shard**（共 4 shard × 4 线程 = 16 核）。
#   ⚠️ 每卡 2 个 shard 会 OOM：训练已占 15.7/24.5 GB，余 8.4 GB，两进程同卡
#      在 batch=8 时峰值越界（实测 s2/s6 报 "CUDA error: out of memory"）。
#   口径与官方一致：--shard i/NGPU，--device i（官方 run_trial0080_longquan_eval.sh §2 同款）。
# ============================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
PY="${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}"
FEAT="${FEAT:-/mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median/images/train}"
W="${W:-$REPO/runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt}"
OUT="${OUT:-$REPO/runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache}"
LOGDIR="${LOGDIR:-$REPO/runs/lookahead/fpv8_24}"
NGPU="${NGPU:-4}"
SHARDS_PER_GPU="${SHARDS_PER_GPU:-1}"
THREADS="${THREADS:-4}"

export PYTHONPATH="$REPO"
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

cd "$REPO" || exit 2
mkdir -p "$OUT" "$LOGDIR"

NSHARD=$((NGPU * SHARDS_PER_GPU))
echo "[$(date '+%F %T')] shard 数=$NSHARD（每卡 $SHARDS_PER_GPU 个，每 shard $THREADS 线程）"

run(){  # run <shard号>
  local i="$1"
  local dev=$(( i % NGPU ))
  local bs=8
  # OOM 自动降 batch 重试（训练抢显存时可能一次性不够）
  while :; do
    "$PY" manu/data/multiscale_heatmap_cache.py \
        --features-root "$FEAT" --cache-root "$OUT" --shard "$i/$NSHARD" \
        --scales 160,80 --region-threshold 0.06 --main-threshold 0.22 \
        --fusion-dilate 32 --min-area 4 --device "$dev" --chunk-size 64 \
        --batch-size "$bs" --cpu-threads "$THREADS" \
        --hm-weights "$W" --ablate-p0 --overwrite > "$LOGDIR/s$i.log" 2>&1
    local rc=$?
    if [ "$rc" -eq 0 ]; then break; fi
    if grep -q "out of memory" "$LOGDIR/s$i.log" && [ "$bs" -gt 1 ]; then
      bs=$(( bs / 2 ))
      echo "[$(date '+%F %T')] shard$i OOM → 降 batch 到 $bs 重试" >> "$LOGDIR/progress.txt"
      sleep 20
      continue
    fi
    break
  done
  echo "[$(date '+%F %T')] DONE shard$i(dev$dev) rc=$rc 累计 pkl=$(ls "$OUT"/*.pkl 2>/dev/null | wc -l)/24" >> "$LOGDIR/progress.txt"
}

for ((i=0; i<NSHARD; i++)); do run "$i" & done
wait
echo "[$(date '+%F %T')] ALL DONE: $(ls "$OUT"/*.pkl 2>/dev/null | wc -l)/24 pkl" | tee -a "$LOGDIR/progress.txt"
