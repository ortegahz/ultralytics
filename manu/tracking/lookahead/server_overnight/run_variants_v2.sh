#!/usr/bin/env bash
# ============================================================
# ★ 服务器端「变体矩阵」执行器（纯服务器侧脚本，不依赖 host 网络）
#
#   用法（服务器上）：
#     cd /tmp/pycharm_project_10ae9e2e
#     nohup bash manu/tracking/lookahead/server_overnight/run_variants_on_server.sh \
#          > runs/lookahead/final/variants_matrix.log 2>&1 &
#
#   环境变量：
#     VARIANTS  变体名列表（逗号分隔），默认见下
#     THREADS   每变体线程数（默认 4）
#     DELTA     滞后帧数（默认 8）
# ============================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
PY="${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}"
CACHE="${CACHE:-$REPO/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache}"
LABELS="${LABELS:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
FRAMES="${FRAMES:-/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg}"
OUT="${OUT:-$REPO/runs/lookahead/final}"
PARAMS="$REPO/manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json"

DELTA="${DELTA:-8}"
THREADS="${THREADS:-4}"
VARIANTS="${VARIANTS:-delta_s30,pred_r24_s30,pred_r40_s30,w32_sg20,refine_s30,gateA}"

export PYTHONPATH="$REPO"
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

cd "$REPO" || exit 2
mkdir -p "$OUT"

# replay 档位的公共口径：与 host 端 eval_all_arms.sh 逐参数一致
COMMON=(
  --cache-root "$CACHE" --labels-root "$LABELS" --all-seqs
  --params-json "$PARAMS" --frames-root "$FRAMES"
  --lookahead replay --delta "$DELTA" --arms original,stale,kinematic,replay
  --fast-win-size 48 --fast-bg-window 96
)
# 注意：不能用 "${PERFRAME[@]/--replay-anchor delta/--replay-anchor pred}" 做替换 ——
# bash 的数组模式替换**不跨元素匹配**（"--replay-anchor" 与 "delta" 是两个元素），
# 实测会让 pred 变体静默退回 delta（结果与 delta 逐位相同）。故锚点单独传参。
PERFRAME=(--replay-mode perframe --replay-max-hold 8 --replay-fallback holdv
          --no-replay-hold-use-ref --output-max-step-px 30)

run_one() {
  local tag="$1"; shift
  local j="$OUT/arms_$tag.json"
  local log="$OUT/arms_$tag.log"
  echo "### [RUN ] $tag  $(date '+%H:%M:%S')  :: $*"
  local t0=$SECONDS
  "$PY" manu/tracking/lookahead/lookahead_longquan.py \
      "${COMMON[@]}" "$@" \
      --out-json "$j" --out-preds-dir "$OUT/preds_$tag" > "$log" 2>&1
  local rc=$?
  echo "### [DONE] $tag rc=$rc  ($((SECONDS-t0))s)"
  grep -E '^\[OVERALL\]|^\[ARM\]|^\[GATE\]' "$log" || tail -5 "$log"
}

for v in $(echo "$VARIANTS" | tr ',' ' '); do
  case "$v" in
    delta_s30)    run_one delta_s30    "${PERFRAME[@]}" --replay-anchor delta ;;
    pred_r12_s30) run_one pred_r12_s30 "${PERFRAME[@]}" --replay-anchor pred --replay-pred-radius-px 12 ;;
    pred_r24_s30) run_one pred_r24_s30 "${PERFRAME[@]}" --replay-anchor pred --replay-pred-radius-px 24 ;;
    pred_r40_s30) run_one pred_r40_s30 "${PERFRAME[@]}" --replay-anchor pred --replay-pred-radius-px 40 ;;
    w32_sg20)     run_one w32_sg20     "${PERFRAME[@]}" --replay-anchor delta --fast-win-size 32 --replay-step-gate-px 20 ;;
    refine_s30)   run_one refine_s30   --replay-mode refine --refine-gate-px 24 --output-max-step-px 30 ;;
    # ---- hold 口径三档（治「单步假跳变被盲推 8 倍」）----
    holdref_s30)   run_one holdref_s30   "${PERFRAME[@]}" --replay-anchor delta --replay-hold-use-ref ;;
    holdrob_s30)   run_one holdrob_s30   "${PERFRAME[@]}" --replay-anchor delta --replay-hold-use-ref --replay-hold-robust ;;
    rel8_s30)      run_one rel8_s30      "${PERFRAME[@]}" --replay-anchor delta --replay-step-rel-tol-px 8 ;;
    holdref_rel8)  run_one holdref_rel8  "${PERFRAME[@]}" --replay-anchor delta --replay-hold-use-ref --replay-step-rel-tol-px 8 ;;
    holdrob_rel8)  run_one holdrob_rel8  "${PERFRAME[@]}" --replay-anchor delta --replay-hold-use-ref --replay-hold-robust --replay-step-rel-tol-px 8 ;;
    gateA)        run_one gateA        --lookahead off --arms original ;;
    *) echo "### [SKIP] 未知变体 $v" ;;
  esac
done
echo "--- 变体矩阵结束 $(date '+%H:%M:%S') ---"
