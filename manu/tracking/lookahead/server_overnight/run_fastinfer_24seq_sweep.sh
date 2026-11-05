#!/usr/bin/env bash
# ============================================================
# ★ Fast Infer 24 序列官方口径 Δ 扫描（RC2 / GMC 语义）
#
#   目的：把 8 序 gate 集上的 Δ 结论（Δ=3 优于 Δ=8）推到**24 序官方口径**上，
#         这是唯一能回答「能不能交付」的一步（8 序只覆盖 24 序 20.39% GT）。
#
#   ★ 唯一变量 = 假设延时 Δ；缓存 / GT / 帧 / 工作点 params / 插件 CFG 全部与
#     历史交付档 run_delivery.sh 逐字一致（CFG 直接从 run_delivery.sh 的 CFG= 取，
#     不重打一遍 —— 避免「四份都叫交付档」的漂移）。
#
#   为什么只有 rc2：24 序的 no-GMC 特征集 + ms_cache 需要 GPU 重建，
#     而本 agent 沙箱**没有 /dev/nvidia*、torch.cuda.is_available()==False**
#     ⇒ 24 序 no-GMC 缓存无法在今夜构建（见 P0b 可行性探测产物）。
#
#   用法（服务器）：
#     cd /tmp/pycharm_project_10ae9e2e
#     bash manu/tracking/lookahead/server_overnight/run_fastinfer_24seq_sweep.sh
#     env DELTAS="3" bash .../run_fastinfer_24seq_sweep.sh    # 单点复跑
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
DELTAS="${DELTAS:-2 3 4 8}"
CACHE="${CACHE:-$REPO/runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache}"

export PYTHONPATH="$REPO"
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

cd "$REPO" || exit 2
mkdir -p "$OUT"

# ---- 唯一 CFG 真值：直接从交付脚本读取（多行续行，awk 取整块再 eval）--------
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
echo "[$(date '+%F %T')] 24 序缓存 $CACHE pkl=$NPKL（期望 24）"
if [ "$NPKL" -ne 24 ]; then echo "[FATAL] 24 序缓存 pkl 数=$NPKL ≠ 24"; exit 3; fi

for d in $DELTAS; do
  O="$OUT/rc2/d$d"
  if [ -f "$O/arms.json" ]; then
    echo "[$(date '+%F %T')] SKIP Δ=$d（已有产物）"; continue
  fi
  mkdir -p "$O"
  echo "[$(date '+%F %T')] ===== RUN 24seq Δ=$d ====="
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
  echo "[$(date '+%F %T')] DONE 24seq Δ=$d rc=$rc 用时 $((t1-t0))s" | tee -a "$OUT/progress.txt"
  if [ $rc -ne 0 ]; then tail -25 "$O/run.log"; fi
done

echo "[$(date '+%F %T')] ALL DONE. 产物在 $OUT"
find "$OUT" -name arms.json | sort
