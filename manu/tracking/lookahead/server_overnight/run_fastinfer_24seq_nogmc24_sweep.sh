#!/usr/bin/env bash
# ============================================================
# ★ Fast Infer **24 序官方口径** × **no-GMC 模型包（cs_t0043）** 的 Δ 扫描
#
#   与 `run_fastinfer_24seq_sweep.sh`（rc2 / GMC）**逐字同口径**，唯一差别：
#     --cache-root 指向 **24 序 no-GMC 特征集**建出来的缓存（特征语义 = no-GMC），
#     产物落在 $OUT/cs_t0043/d<Δ>/（rc2 落在 $OUT/rc2/d<Δ>/）。
#     GT / 帧根 / 帧缓存 / 工作点 params / 插件 CFG 全部一致 ⇒ 唯一变量 = {模型包, Δ}。
#
#   用法（服务器）：
#     cd /tmp/pycharm_project_10ae9e2e
#     bash manu/tracking/lookahead/server_overnight/run_fastinfer_24seq_nogmc24_sweep.sh
#     env DELTAS="3" bash .../run_fastinfer_24seq_nogmc24_sweep.sh
# ============================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
PY="${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}"
GT="${GT:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
FRAMES="${FRAMES:-/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg}"
FC="${FRAMES_CACHE:-/mnt/data/siping/frame_cache_lq}"
PARAMS="${PARAMS:-manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json}"
OUT="${OUT:-$REPO/runs/lookahead/fi_nogmc24}"
DELIVERY_SH="$REPO/manu/tracking/lookahead/run_delivery.sh"
THREADS="${THREADS:-12}"
DELTAS="${DELTAS:-3 2 4 8}"
CACHE="${CACHE:-/mnt/data/siping/runs/nogmc_t3/swa/ms_cache_24seq}"
MODEL="${MODEL:-cs_t0043}"

export PYTHONPATH="$REPO"
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

cd "$REPO" || exit 2
mkdir -p "$OUT"

# ---- 唯一 CFG 真值：直接从 run_delivery.sh 取（多行续行 ⇒ awk 取整块再 eval）----
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

FRAME_ARGS=""
if [ -d "$FC" ]; then
  FRAME_ARGS="--frames-cache $FC"
  echo "[$(date '+%F %T')] 帧缓存 memmap: $FC"
else
  echo "[$(date '+%F %T')] ⚠️ 帧缓存不存在 $FC ⇒ 走原始帧（慢）"
fi

NPKL=$(ls "$CACHE"/*.pkl 2>/dev/null | wc -l)
echo "[$(date '+%F %T')] 24 序 no-GMC 缓存 $CACHE pkl=$NPKL（期望 24）"
if [ "$NPKL" -ne 24 ]; then echo "[FATAL] 24 序缓存 pkl 数=$NPKL ≠ 24"; exit 3; fi

for d in $DELTAS; do
  O="$OUT/$MODEL/d$d"
  if [ -f "$O/arms.json" ]; then
    echo "[$(date '+%F %T')] SKIP $MODEL Δ=$d（已有产物）"; continue
  fi
  mkdir -p "$O"
  echo "[$(date '+%F %T')] ===== RUN 24seq $MODEL Δ=$d ====="
  t0=$(date +%s)
  "$PY" manu/tracking/lookahead/lookahead_longquan.py \
    --cache-root "$CACHE" --labels-root "$GT" --frames-root "$FRAMES" $FRAME_ARGS \
    --params-json "$PARAMS" --all-seqs \
    --lookahead replay --arms original,stale,kinematic,replay \
    --delta "$d" $CFG \
    --out-json "$O/arms.json" --out-preds-dir "$O/preds" \
    > "$O/run.log" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "[$(date '+%F %T')] DONE 24seq $MODEL Δ=$d rc=$rc 用时 $((t1-t0))s" | tee -a "$OUT/progress.txt"
  if [ $rc -ne 0 ]; then tail -25 "$O/run.log"; fi
done

echo "[$(date '+%F %T')] ALL DONE. 产物在 $OUT"
find "$OUT" -name arms.json | sort
