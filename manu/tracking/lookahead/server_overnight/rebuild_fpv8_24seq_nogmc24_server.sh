#!/usr/bin/env bash
# ============================================================
# ★ 24 序 **no-GMC** 的 FPV8 语义多尺度热图缓存（GPU 分片）
#
#   用途：给「Fast Infer × no-GMC」在**24 序官方口径**上建 cs_t0043 的缓存，
#         用来回答「换 no-GMC 模型包后，小延时（Δ=3 / 120 ms）能不能交付」。
#
#   ★ 口径**逐字对齐官方 rc2 那份 24 序缓存**
#     （runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache，2026-11-03 重建、
#      产出冻结基线 F1 94.747% 的那份）—— 官方口径是
#       run_trial0080_longquan_eval.sh §2 → --device i --shard i/NGPU
#     同超参：--scales 160,80 --region-threshold 0.06 --main-threshold 0.22
#             --fusion-dilate 32 --min-area 4 --chunk-size 64 --batch-size 8 --ablate-p0
#
#   ★ **必须 GPU**：CPU 慢 13~16×，且 FP32 前后处理顺序不同 ⇒ 数值口径对不上官方基线。
#   ★ **每卡 1 shard**、batch 固定 8：batch 变化会改 FP32 分组顺序 ⇒ 可能翻像素；
#     所以这里**不自动降 batch**，OOM 就报错退出（宁可失败也不要口径漂移）。
#
#   与官方脚本的唯一差别：**显式 GPU 列表**（GPUS="1 2 3"），
#     避免占用 GPU 0（上面有别人的进程 honpho，跑了 1 天）。
#
#   用法（服务器）：
#     cd /tmp/pycharm_project_10ae9e2e
#     FEAT=/mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median_nogmc/images/train \
#     W=/mnt/data/siping/runs/nogmc_t3/swa/cs_t0043.pt \
#     OUT=/mnt/data/siping/runs/nogmc_t3/swa/ms_cache_24seq \
#     GPUS="1 2 3" NSHARD=3 \
#     nohup bash manu/tracking/lookahead/server_overnight/rebuild_fpv8_24seq_nogmc24_server.sh \
#       > runs/lookahead/fpv8_24_nogmc/launch.log 2>&1 &
# ============================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
PY="${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}"
FEAT="${FEAT:?必须指定 FEAT=<no-GMC 24序数据集>/images/train}"
W="${W:?必须指定 W=<cs_t0043.pt>}"
OUT="${OUT:?必须指定 OUT=<缓存输出目录>}"
LOGDIR="${LOGDIR:-$REPO/runs/lookahead/fpv8_24_nogmc}"
GPUS="${GPUS:-1 2 3}"
NSHARD="${NSHARD:-3}"
THREADS="${THREADS:-4}"

export PYTHONPATH="$REPO"
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

cd "$REPO" || exit 2
mkdir -p "$OUT" "$LOGDIR"

read -r -a GPU_ARR <<< "$GPUS"
if [ "${#GPU_ARR[@]}" -ne "$NSHARD" ]; then
  echo "ERROR: GPUS 数量(${#GPU_ARR[@]}) 必须等于 NSHARD($NSHARD)" | tee -a "$LOGDIR/progress.txt"
  exit 2
fi

echo "[$(date '+%F %T')] shard 数=$NSHARD  GPU 映射: $(for i in "${!GPU_ARR[@]}"; do echo -n "s$i->gpu${GPU_ARR[$i]} "; done)" \
  | tee -a "$LOGDIR/progress.txt"
echo "[$(date '+%F %T')] FEAT=$FEAT" | tee -a "$LOGDIR/progress.txt"
echo "[$(date '+%F %T')] W=$W" | tee -a "$LOGDIR/progress.txt"
echo "[$(date '+%F %T')] OUT=$OUT" | tee -a "$LOGDIR/progress.txt"

run(){  # run <shard号>
  local i="$1"
  local dev="${GPU_ARR[$i]}"
  "$PY" manu/data/multiscale_heatmap_cache.py \
      --features-root "$FEAT" --cache-root "$OUT" --shard "$i/$NSHARD" \
      --scales 160,80 --region-threshold 0.06 --main-threshold 0.22 \
      --fusion-dilate 32 --min-area 4 --device "$dev" --chunk-size 64 \
      --batch-size 8 --cpu-threads "$THREADS" \
      --hm-weights "$W" --ablate-p0 --overwrite > "$LOGDIR/s$i.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] DONE shard$i(dev$dev) rc=$rc 累计 pkl=$(ls "$OUT"/*.pkl 2>/dev/null | wc -l)/24" \
    >> "$LOGDIR/progress.txt"
  return $rc
}

pids=()
for ((i=0; i<NSHARD; i++)); do
  run "$i" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done

npkl=$(ls "$OUT"/*.pkl 2>/dev/null | wc -l)
echo "[$(date '+%F %T')] ALL DONE: ${npkl}/24 pkl  fail=$fail" | tee -a "$LOGDIR/progress.txt"
if [ "$npkl" -ne 24 ]; then
  echo "ERROR: 只出了 ${npkl}/24 个 pkl —— 必须查清缺哪几条再重跑（别盲目 --overwrite 全量）" | tee -a "$LOGDIR/progress.txt"
  exit 1
fi
exit $fail
