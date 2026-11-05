#!/usr/bin/env bash
# ============================================================
# ★ 夜 8：配准 / 歧义处置族 —— 插桩 dump（候选峰全集）
#
#   目的：给「配准/歧义处置维度」算 **oracle 上界**。生产 `xcorr.match()` 只返回
#     `matchTemplate` 的全局 argmax ⇒ 候选峰拿不到 ⇒ 上界必须自己算。
#     本脚本跑交付档 replay 臂，用 monkey-patch 记录每次匹配的 top-K 候选峰，
#     **生产代码零改动**，并对每次匹配做**逐位等价闸门**（argmax 位移/峰值/次峰/唯一性）。
#
#   ★ 唯一 CFG 真值：`manu/tracking/lookahead/run_delivery.sh` 的 `CFG=` 赋值块
#     （多行反斜杠续行，单行 grep 会截断 ⇒ 本脚本用 awk 取完整块再 eval，
#      并对 7 个关键旋钮做防截断断言，与 run_fastinfer_nogmc_sweep.sh 逐字一致）。
#   ★ 出框门 m=6：交付推荐档（`--replay-fov-margin-px 6`）单独加，与扫描脚本同口径。
#
#   用法（服务器）：
#     cd /tmp/pycharm_project_10ae9e2e
#     bash manu/tracking/lookahead/server_overnight/run_reg_oracle_dump.sh
#     env NSHARD=1 bash .../run_reg_oracle_dump.sh     # 单分片（调试）
# ============================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
PY="${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}"
CACHE="${CACHE:-/mnt/data/siping/runs/nogmc_t3/swa/ms_cache_24seq}"
GT="${GT:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
FRAMES="${FRAMES:-/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg}"
FC="${FRAMES_CACHE:-/mnt/data/siping/frame_cache_lq}"
OUT="${OUT:-$REPO/runs/lookahead/fi_nogmc24/reg_oracle}"
DELIVERY_SH="$REPO/manu/tracking/lookahead/run_delivery.sh"
NSHARD="${NSHARD:-4}"
THREADS="${THREADS:-4}"
DELTA="${DELTA:-3}"
FOV="${FOV:-6}"

export PYTHONPATH="$REPO"
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

cd "$REPO" || exit 2
mkdir -p "$OUT"

# ---- 唯一 CFG 真值：直接从交付脚本读取，杜绝配置漂断 ----------------------
if [ ! -f "$DELIVERY_SH" ]; then echo "[FATAL] 找不到 $DELIVERY_SH"; exit 3; fi
CFG_ASSIGN="$(awk '/^CFG="/ {f=1} f {print} f && /"[[:space:]]*$/ {exit}' "$DELIVERY_SH")"
if [ -z "$CFG_ASSIGN" ]; then echo "[FATAL] 无法从 $DELIVERY_SH 解析 CFG"; exit 3; fi
CFG=""
eval "$CFG_ASSIGN"
if [ -z "$CFG" ]; then echo "[FATAL] CFG 解析为空"; exit 3; fi
for _must in "--replay-roi-selfpred-max-px 16" "--replay-rollback-min-peak 0.90" \
             "--replay-polarity auto" "--replay-xcorr-uniq-gate 0.6" "--replay-mode perframe" \
             "--replay-abstain-peak-below 0.30" "--replay-rollback-max-frames 16"; do
  case " $CFG " in
    *" $_must "*) ;;
    *) echo "[FATAL] CFG 缺少必需旋钮 '$_must' ⇒ 解析被截断，拒绝跑"; exit 3 ;;
  esac
done
echo "[$(date '+%F %T')] CFG（源自 run_delivery.sh，${#CFG} 字符）: $CFG"

NP=$(ls "$CACHE"/*.pkl 2>/dev/null | wc -l)
echo "[$(date '+%F %T')] 24 序缓存 $CACHE pkl=$NP（期望 24）· 分片=$NSHARD · Δ=$DELTA · 出框门 m=$FOV"

SEQS=()
for f in "$CACHE"/*.pkl; do SEQS+=("$(basename "$f" .pkl)"); done

pids=()
for shard in $(seq 0 $((NSHARD - 1))); do
  ARGS=()
  for i in "${!SEQS[@]}"; do
    if [ $((i % NSHARD)) -eq "$shard" ]; then ARGS+=(--seq "${SEQS[$i]}"); fi
  done
  [ ${#ARGS[@]} -eq 0 ] && continue
  "$PY" manu/tracking/lookahead/diag_reg_oracle.py --dump "${ARGS[@]}" \
    --cache-root "$CACHE" --labels-root "$GT" --frames-root "$FRAMES" --frames-cache "$FC" \
    --delta "$DELTA" --fov-margin "$FOV" --dump-dir "$OUT" \
    > "$OUT/shard${shard}.log" 2>&1 &
  pids+=($!)
  echo "[$(date '+%F %T')] shard $shard pid=${pids[-1]} 序列数=${#ARGS[@]}"
done

rc_all=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc_all=1
done
echo "[$(date '+%F %T')] ALL DONE rc_all=$rc_all"

# ---- 硬门禁：等价闸门必须 0 次不等价，且 24 序 dump 齐全 -----------------
MISMATCH=$(grep -h "gate_mismatch" "$OUT"/shard*.log 2>/dev/null | grep -o "mismatch=[0-9]*" | cut -d= -f2 | paste -sd+ | bc 2>/dev/null || echo 0)
[ "${MISMATCH:-0}" -ne 0 ] && { echo "[FATAL] 等价闸门失败（mismatch=$MISMATCH）⇒ dump 不可信"; rc_all=1; }
NDUMP=$(ls "$OUT"/*.jsonl 2>/dev/null | wc -l)
if [ "$NDUMP" -ne "$NP" ]; then
  echo "[FATAL] dump 份数 $NDUMP ≠ 缓存序列数 $NP"; rc_all=1
fi
echo "[$(date '+%F %T')] 门禁：等价闸门 mismatch=$MISMATCH · dump 份数=$NDUMP/$NP"
find "$OUT" -name "*.jsonl" | sort
exit $rc_all
