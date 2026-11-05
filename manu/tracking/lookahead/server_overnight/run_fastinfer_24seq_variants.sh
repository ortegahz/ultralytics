#!/usr/bin/env bash
# ============================================================
# ★ 24 序列官方口径 · 变体扫描器
#
#   用途：把 8 序 A/B 找到的旋钮候选**搬到 24 序官方口径**复评（8 序只覆盖
#         24 序 20.39% GT ⇒ 结论只能做机理筛选，不能给交付档排序）。
#
#   变体定义在下面的 VARIANTS 数组里，格式：<name>|<Δ列表>|<额外 CFG 覆盖(空格分隔，可空)>
#   额外 CFG 会用 sed 在 run_delivery.sh 的 CFG 上做**按键替换**（不动源文件）。
#
#   用法：
#     cd /tmp/pycharm_project_10ae9e2e
#     bash manu/tracking/lookahead/server_overnight/run_fastinfer_24seq_variants.sh
#     env ONLY="p3knobs" bash .../run_fastinfer_24seq_variants.sh
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
CACHE="${CACHE:-$REPO/runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache}"

export PYTHONPATH="$REPO"
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"

cd "$REPO" || exit 2
mkdir -p "$OUT"

# ---- 唯一 CFG 真值：从交付脚本读取（多行续行，awk 取整块再 eval）------------
CFG_ASSIGN="$(awk '/^CFG="/ {f=1} f {print} f && /"[[:space:]]*$/ {exit}' "$DELIVERY_SH")"
[ -n "$CFG_ASSIGN" ] || { echo "[FATAL] 无法解析 CFG"; exit 3; }
# eval 的是 `CFG="..."` 这条赋值本身 ⇒ 赋值目标是 CFG，随后再拷进 BASE_CFG
CFG=""; eval "$CFG_ASSIGN"
BASE_CFG="$CFG"
[ -n "$BASE_CFG" ] || { echo "[FATAL] CFG 解析为空"; exit 3; }
for _must in "--replay-roi-selfpred-max-px 16" "--replay-rollback-min-peak 0.90" \
             "--replay-polarity auto" "--replay-xcorr-uniq-gate 0.6" "--replay-mode perframe" \
             "--replay-abstain-peak-below 0.30" "--replay-rollback-max-frames 16"; do
  case " $BASE_CFG " in
    *" $_must "*) ;;
    *) echo "[FATAL] CFG 缺少必需旋钮 '$_must'"; exit 3 ;;
  esac
done

FRAME_ARGS=""
[ -d "$FC" ] && FRAME_ARGS="--frames-cache $FC"

# name | Δ列表 | 额外 CFG（sed 覆盖表达式）
VARIANTS=(
  "p3knobs|2 3|--replay-abstain-peak-below 0.50|--replay-rollback-min-peak 0.80"
)

ONLY="${ONLY:-}"
for v in "${VARIANTS[@]}"; do
  IFS='|' read -r NAME DELTAS_EXTRA OVER1 OVER2 <<< "$v"
  [ -n "$ONLY" ] && [ "$ONLY" != "$NAME" ] && continue
  CFG="$BASE_CFG"
  for ov in "$OVER1" "$OVER2"; do
    [ -n "$ov" ] || continue
    KEY="${ov%% *}"; VAL="${ov##* }"
    CFG="$(printf '%s' "$CFG" | sed -E "s|${KEY}[[:space:]]+[^[:space:]]+|${KEY} ${VAL}|g")"
  done
  case " $CFG " in
    *" --replay-abstain-peak-below $OVER2 "*) : ;;
  esac
  echo "[$(date '+%F %T')] ===== 变体 $NAME  Δ=$DELTAS_EXTRA ====="
  echo "[$(date '+%F %T')] 覆盖后 CFG: $CFG"
  for d in $DELTAS_EXTRA; do
    O="$OUT/$NAME/rc2/d$d"
    if [ -f "$O/arms.json" ]; then echo "[$(date '+%F %T')] SKIP $NAME Δ=$d"; continue; fi
    mkdir -p "$O"
    t0=$(date +%s)
    "$PY" manu/tracking/lookahead/lookahead_longquan.py \
      --cache-root "$CACHE" --labels-root "$GT" --frames-root "$FRAMES" $FRAME_ARGS \
      --params-json "$PARAMS" --all-seqs \
      --lookahead replay --arms original,stale,kinematic,replay \
      --delta "$d" $CFG \
      --out-json "$O/arms.json" --out-preds-dir "$O/preds" \
      > "$O/run.log" 2>&1
    rc=$?; t1=$(date +%s)
    echo "[$(date '+%F %T')] DONE $NAME Δ=$d rc=$rc 用时 $((t1-t0))s" | tee -a "$OUT/variants_progress.txt"
    [ $rc -ne 0 ] && tail -20 "$O/run.log"
  done
done
echo "[$(date '+%F %T')] VARIANTS DONE"
find "$OUT" -name arms.json | sort
