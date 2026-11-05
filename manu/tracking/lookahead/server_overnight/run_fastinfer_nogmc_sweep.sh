#!/usr/bin/env bash
# ============================================================
# ★ Fast Infer（look-ahead 前向推演）× no-GMC：模型包对比 × 假设延时 Δ 扫描
#
#   目的：
#     (1) 把 look-ahead 前向推演所用的模型，从 RC2（trial_0080 ep3, GMC 特征语义）
#         换成 no-GMC 交付基线 cs_t0043（no-GMC 特征语义）；
#     (2) 把「假设延时」从 Δ=8 record（320 ms @25 Hz）降到 Δ=3 record（120 ms）；
#     (3) 验证「小延时 + no-GMC 模型」能否在低延时下提升精度，并给出严格对比报告。
#
#   ★ 严格受控 A/B（唯一变量 = {模型包, 假设延时 Δ}）：
#     - 同一 8 序列 gate 集 / 同一 28,997 帧 / 同一 GT（Final_Labels_merged_20261010）
#     - 同一原始灰度帧根（快核 fast core 与 xcorr 模板匹配的输入，两模型完全一致）
#     - 同一工作点 params json（th_base 0.48 / sky_ratio 1.0 / th_ground 0.26 / confirm_consec 7）
#     - 同一插件 CFG：**直接从 run_delivery.sh 读取**（杜绝「四份都叫交付档」式的配置漂移）
#     - 缓存构建参数逐字相同：scales 0,160,80 / region-threshold 0.06 / main-threshold 0.22
#       / fusion-dilate 32 / min-area 4 / --ablate-p0（已逐位核对两个缓存的 meta）
#
#   用法（服务器）：
#     cd /tmp/pycharm_project_10ae9e2e
#     bash manu/tracking/lookahead/server_overnight/run_fastinfer_nogmc_sweep.sh
#     env MODELS="cs_t0043" DELTAS="3" bash .../run_fastinfer_nogmc_sweep.sh   # 单点复跑
# ============================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
PY="${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}"
GT="${GT:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
FRAMES="${FRAMES:-/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg}"
FC="${FRAMES_CACHE:-/mnt/data/siping/frame_cache_lq}"
PARAMS="${PARAMS:-manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json}"
OUT="${OUT:-$REPO/runs/lookahead/fi_nogmc}"
DELIVERY_SH="$REPO/manu/tracking/lookahead/run_delivery.sh"
THREADS="${THREADS:-8}"
DELTAS="${DELTAS:-2 3 4 8}"

export PYTHONPATH="$REPO"
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

cd "$REPO" || exit 2
mkdir -p "$OUT"

# ---- 唯一 CFG 真值：直接从交付脚本读取，杜绝配置漂移 ----------------------
# ⚠️ run_delivery.sh 的 CFG 是**多行反斜杠续行**的赋值，单行 grep 会把它截断
#    （本轮真踩到：只取到第一行 ⇒ 插件跑的是半套配置）。
#    ⇒ 用 awk 取完整赋值块（直到行尾是 " 的那行），再交给 bash 自己 eval。
if [ ! -f "$DELIVERY_SH" ]; then echo "[FATAL] 找不到 $DELIVERY_SH"; exit 3; fi
CFG_ASSIGN="$(awk '/^CFG="/ {f=1} f {print} f && /"[[:space:]]*$/ {exit}' "$DELIVERY_SH")"
if [ -z "$CFG_ASSIGN" ]; then echo "[FATAL] 无法从 $DELIVERY_SH 解析 CFG"; exit 3; fi
CFG=""
eval "$CFG_ASSIGN"
if [ -z "$CFG" ]; then echo "[FATAL] CFG 解析为空"; exit 3; fi
# 防截断断言：交付档的关键旋钮必须全部在位（少一个就说明解析被截断 ⇒ 拒绝跑）
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
  echo "[$(date '+%F %T')] ⚠️ 帧缓存不存在 $FC ⇒ 走原始帧（慢 11.3 ms/帧）"
fi

# ---- 模型包（缓存 + 标签）------------------------------------------------
#   每个模型包的缓存 meta 已逐位核对：scales/threshold/dilate/min_area/use_p0_highway 全同，
#   只有 features_root（特征语义）与 hm_weights（权重）不同 —— 这正是要测的变量。
declare -A CACHE
declare -A LABEL
CACHE[rc2]="$REPO/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache"
LABEL[rc2]="RC2 trial_0080ep3 / GMC 语义"
CACHE[cs_t0043]="/mnt/data/siping/runs/nogmc_t3/swa/gate/cs_t0043/ms_cache"
LABEL[cs_t0043]="cs_t0043 / no-GMC 语义"
CACHE[trial_0049]="/mnt/data/siping/runs/nogmc_t3/trials/trial_0049/gate/epoch20/ms_cache"
LABEL[trial_0049]="trial_0049 ep20 / no-GMC 语义"

MODELS="${MODELS:-rc2 cs_t0043 trial_0049}"

echo "[$(date '+%F %T')] ===== 计划 ====="
for m in $MODELS; do
  n=$(ls "${CACHE[$m]}"/*.pkl 2>/dev/null | wc -l)
  echo "  模型 $m (${LABEL[$m]})  缓存 pkl=$n  Δ=$DELTAS"
  if [ "$n" -ne 8 ]; then echo "[WARN] $m 缓存 pkl 数=$n（≠8）⇒ 跳过该模型"; fi
done

for m in $MODELS; do
  n=$(ls "${CACHE[$m]}"/*.pkl 2>/dev/null | wc -l)
  [ "$n" -eq 8 ] || continue
  for d in $DELTAS; do
    O="$OUT/$m/d$d"
    if [ -f "$O/arms.json" ]; then
      echo "[$(date '+%F %T')] SKIP $m Δ=$d（已有产物）"; continue
    fi
    mkdir -p "$O"
    echo "[$(date '+%F %T')] ===== RUN 模型=$m Δ=$d ====="
    t0=$(date +%s)
    "$PY" manu/tracking/lookahead/lookahead_longquan.py \
      --cache-root "${CACHE[$m]}" --labels-root "$GT" --frames-root "$FRAMES" $FRAME_ARGS \
      --params-json "$PARAMS" --all-seqs \
      --lookahead replay --arms original,stale,kinematic,replay \
      --delta "$d" $CFG \
      --out-json "$O/arms.json" --out-preds-dir "$O/preds" \
      > "$O/run.log" 2>&1
    rc=$?
    t1=$(date +%s)
    echo "[$(date '+%F %T')] DONE $m Δ=$d rc=$rc 用时 $((t1-t0))s" | tee -a "$OUT/progress.txt"
    if [ $rc -ne 0 ]; then tail -20 "$O/run.log"; fi
  done
done

echo "[$(date '+%F %T')] ALL DONE. 产物在 $OUT"
find "$OUT" -name arms.json | sort
