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
    # ---- hit@5px 两把药（2026-11-03）----
    shrink085)  run_one shrink085  "${PERFRAME[@]}" --replay-anchor delta --replay-obs-shrink-alpha 0.85 ;;
    shrink070)  run_one shrink070  "${PERFRAME[@]}" --replay-anchor delta --replay-obs-shrink-alpha 0.70 ;;
    shrink055)  run_one shrink055  "${PERFRAME[@]}" --replay-anchor delta --replay-obs-shrink-alpha 0.55 ;;
    freeze08)   run_one freeze08   "${PERFRAME[@]}" --replay-anchor delta --replay-freeze-speed-px 0.8 ;;
    freeze15)   run_one freeze15   "${PERFRAME[@]}" --replay-anchor delta --replay-freeze-speed-px 1.5 ;;
    shr070fz08) run_one shr070fz08 "${PERFRAME[@]}" --replay-anchor delta --replay-obs-shrink-alpha 0.70 --replay-freeze-speed-px 0.8 ;;
    shr070fz15) run_one shr070fz15 "${PERFRAME[@]}" --replay-anchor delta --replay-obs-shrink-alpha 0.70 --replay-freeze-speed-px 1.5 ;;
    gateA)        run_one gateA        --lookahead off --arms original ;;
    # ---- 位移层 shrinkage（③）：只收缩最终位移，链内不动 ⇒ 悬停/机动都不吃亏 ----
    dispshr090)  run_one dispshr090 "${PERFRAME[@]}" --replay-anchor delta --replay-disp-shrink-alpha 0.90 ;;
    dispshr075)  run_one dispshr075 "${PERFRAME[@]}" --replay-anchor delta --replay-disp-shrink-alpha 0.75 ;;
    dispshr050)  run_one dispshr050 "${PERFRAME[@]}" --replay-anchor delta --replay-disp-shrink-alpha 0.50 ;;
    # ---- 位移自适应融合（④）：实测 vs 参考矛盾时退向参考（治 walk-off 100+px）----
    merge010)    run_one merge010   "${PERFRAME[@]}" --replay-anchor delta --replay-disp-merge-sigma-px 10 ;;
    merge020)    run_one merge020   "${PERFRAME[@]}" --replay-anchor delta --replay-disp-merge-sigma-px 20 ;;
    merge040)    run_one merge040   "${PERFRAME[@]}" --replay-anchor delta --replay-disp-merge-sigma-px 40 ;;
    # ---- 快核 peak_centroid：先取局部极大再在邻域求质心（避开远处杂波拽偏质心）----
    pk9)         run_one pk9        "${PERFRAME[@]}" --replay-anchor delta --fast-mode peak_centroid ;;
    pk5)         run_one pk5        "${PERFRAME[@]}" --replay-anchor delta --fast-mode peak_centroid --fast-peak-window 5 ;;
    pk15)        run_one pk15       "${PERFRAME[@]}" --replay-anchor delta --fast-mode peak_centroid --fast-peak-window 15 ;;
    # ---- peak_centroid × 位移层收缩：两个治"质心/位移偏置"的手段叠加 ----
    pk9_ds075)   run_one pk9_ds075  "${PERFRAME[@]}" --replay-anchor delta --fast-mode peak_centroid --replay-disp-shrink-alpha 0.75 ;;
    pk9_m020)    run_one pk9_m020   "${PERFRAME[@]}" --replay-anchor delta --fast-mode peak_centroid --replay-disp-merge-sigma-px 20 ;;
    # ---- peak_centroid × win32（已知 win32 压 F1 但压 p95）----
    pk9_w32)     run_one pk9_w32    "${PERFRAME[@]}" --replay-anchor delta --fast-mode peak_centroid --fast-win-size 32 ;;
    pk9_w32_sg20) run_one pk9_w32_sg20 "${PERFRAME[@]}" --replay-anchor delta --fast-mode peak_centroid --fast-win-size 32 --replay-step-gate-px 20 ;;
    # ---- hold 口径下的位移层收缩（holdref 是当前第二好的臂）----
    holdref_ds075) run_one holdref_ds075 "${PERFRAME[@]}" --replay-anchor delta --replay-hold-use-ref --replay-disp-shrink-alpha 0.75 ;;
    *) echo "### [SKIP] 未知变体 $v" ;;
  esac
done
echo "--- 变体矩阵结束 $(date '+%H:%M:%S') ---"
