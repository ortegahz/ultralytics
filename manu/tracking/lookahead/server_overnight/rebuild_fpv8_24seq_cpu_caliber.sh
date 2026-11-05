#!/usr/bin/env bash
# ============================================================
# ★ 24 序 **CPU 口径** ms_cache 构建（GPU 不可用时的同口径对照缓存）
#
#   用途：本服务器（test-Super-Server）**无 GPU**，而官方 24 序 RC2 缓存是 GPU 建的
#         （runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache，2026-11-03）。
#         CPU 与 GPU 的 FP32 前后处理顺序不同 ⇒ 检出数会差个位数（实测 4977 vs 4974）
#         ⇒ **CPU 缓存不能直接与 GPU 缓存拼在一起比**。
#         本脚本给**两个模型包各建一份 CPU 缓存**，于是
#             cs_t0043(CPU 缓存) vs rc2(CPU 缓存)
#         是**同口径内部一致的相对 A/B**（✔ 可比），
#         但**绝对数字不等于 GPU 交付档数字**（✘ 不可当交付档）。
#
#   ★ 与官方 GPU 构建**逐字相同的超参**（只有 --device 与输出目录不同）：
#         --scales 160,80 --region-threshold 0.06 --main-threshold 0.22
#         --fusion-dilate 32 --min-area 4 --chunk-size 64 --batch-size 8 --ablate-p0
#     ⚠️ batch-size / chunk-size **不许改**（会改 FP32 分组顺序 ⇒ 像素翻转）。
#     ⚠️ `--shard` 是**序列级**切分（index % count）⇒ shard 数不影响单序列产物内容，
#        只影响并行度 ⇒ 可以放心用 24 路并行。
#
#   用法：
#     cd /tmp/pycharm_project_10ae9e2e
#     FEAT=/mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median_nogmc/images/train \
#     W=/mnt/data/siping/runs/nogmc_t3/swa/cs_t0043.pt \
#     OUT=/mnt/data/siping/runs/nogmc_t3/swa/ms_cache_24seq_cpu_nogmc \
#     TAG=nogmc NSHARD=24 THREADS=2 \
#     bash manu/tracking/lookahead/server_overnight/rebuild_fpv8_24seq_cpu_caliber.sh
# ============================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
PY="${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}"
FEAT="${FEAT:?必须指定 FEAT=<数据集>/images/train}"
W="${W:?必须指定 W=<hm_weights .pt>}"
OUT="${OUT:?必须指定 OUT=<缓存输出目录>}"
TAG="${TAG:-cache}"
NSHARD="${NSHARD:-24}"
THREADS="${THREADS:-2}"
LOGDIR="${LOGDIR:-$REPO/runs/lookahead/fpv8_24_cpu/$TAG}"

export PYTHONPATH="$REPO"
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

cd "$REPO" || exit 2
mkdir -p "$OUT" "$LOGDIR"

log(){ echo "[$(date '+%F %T')] [$TAG] $*" | tee -a "$LOGDIR/progress.txt"; }

log "FEAT=$FEAT"
log "W=$W"
log "OUT=$OUT"
log "NSHARD=$NSHARD THREADS=$THREADS device=cpu"

pids=()
for ((i=0; i<NSHARD; i++)); do
  (
    "$PY" manu/data/multiscale_heatmap_cache.py \
        --features-root "$FEAT" --cache-root "$OUT" --shard "$i/$NSHARD" \
        --scales 160,80 --region-threshold 0.06 --main-threshold 0.22 \
        --fusion-dilate 32 --min-area 4 --device cpu --chunk-size 64 \
        --batch-size 8 --cpu-threads "$THREADS" \
        --hm-weights "$W" --ablate-p0 --overwrite > "$LOGDIR/s$i.log" 2>&1
    echo "$?" > "$LOGDIR/s$i.rc"
  ) &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

npkl=$(ls "$OUT"/*.pkl 2>/dev/null | wc -l)
log "ALL DONE: ${npkl}/24 pkl  fail=$fail"
if [ "$npkl" -ne 24 ]; then
  log "ERROR: 只出了 ${npkl}/24 个 pkl —— 必须查清缺哪几条再重跑"
  exit 1
fi
exit $fail
