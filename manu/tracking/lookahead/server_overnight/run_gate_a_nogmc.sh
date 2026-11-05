#!/usr/bin/env bash
# ============================================================
# ★ P2 Gate A —— 「关开关 ⇒ 逐位一致」硬门禁
#
#   目的：证明外挂 look-ahead **没有扰动** no-GMC 新模型（cs_t0043）的跟踪资产：
#         `--lookahead off` 时，插件版的输出与**未改动的原脚本**
#         （manu/evaluation/system_sota_longquan.py，只读、禁改）在同一缓存上
#         **整份 JSON 逐位一致（md5 相同）**。
#
#   两边各跑一对：
#     A = 原脚本          manu/evaluation/system_sota_longquan.py
#     B = 插件（开关关）  manu/tracking/lookahead/lookahead_longquan.py --lookahead off --arms original
#
#   用法（服务器）：
#     cd /tmp/pycharm_project_10ae9e2e
#     bash manu/tracking/lookahead/server_overnight/run_gate_a_nogmc.sh
#     env MODELS="cs_t0043" bash .../run_gate_a_nogmc.sh     # 只跑主模型
#
#   ★ 幂等：已有的产物跳过（被杀后重跑不浪费）。
#   ★ --cache-root **必须绝对路径**：config 块会进 md5，相对路径 ⇒ md5 天然不同。
# ============================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
PY="${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}"
GT="${GT:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
FRAMES="${FRAMES:-/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg}"
FC="${FRAMES_CACHE:-/mnt/data/siping/frame_cache_lq}"
PARAMS="${PARAMS:-manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json}"
OUT="${OUT:-$REPO/runs/lookahead/gateA}"
THREADS="${THREADS:-6}"

export PYTHONPATH="$REPO"
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

cd "$REPO" || exit 2
mkdir -p "$OUT"

# ---- 模型包（缓存根一律绝对路径）---------------------------------------
declare -A CACHE
declare -A LABEL
CACHE[cs_t0043]="/mnt/data/siping/runs/nogmc_t3/swa/gate/cs_t0043/ms_cache"
LABEL[cs_t0043]="cs_t0043 / no-GMC 语义（主）"
CACHE[rc2]="$REPO/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache"
LABEL[rc2]="rc2 (trial_0080ep3) / GMC 语义（对照）"

MODELS="${MODELS:-cs_t0043 rc2}"
rc_all=0

for m in $MODELS; do
  cache="${CACHE[$m]}"
  if [ ! -d "$cache" ]; then echo "[FATAL] $m 缓存不存在: $cache"; exit 3; fi
  cache="$(realpath "$cache")"          # ★ 绝对路径（config 块进 md5）
  n=$(ls "$cache"/*.pkl 2>/dev/null | wc -l)
  M="$OUT/$m"
  mkdir -p "$M"
  printf '%s\n' "${LABEL[$m]}" > "$M/label.txt"

  # ---- A：原脚本（只读、禁改）------------------------------------------
  A="$M/orig"
  mkdir -p "$A"
  if [ -f "$A/arms.json" ] && [ -d "$A/preds" ]; then
    echo "[$(date '+%F %T')] SKIP $m/A（原脚本产物已存在）"
  else
    cmdA=("$PY" manu/evaluation/system_sota_longquan.py
      --cache-root "$cache" --labels-root "$GT" --params-json "$PARAMS"
      --all-seqs --out-json "$A/arms.json" --out-preds-dir "$A/preds")
    printf '%s\n' "${cmdA[*]}" > "$A/command.txt"
    echo "[$(date '+%F %T')] RUN $m/A（原脚本）"
    t0=$(date +%s)
    "${cmdA[@]}" > "$A/run.log" 2>&1
    rca=$?
    echo "[$(date '+%F %T')] A rc=$rca 用时 $(( $(date +%s) - t0 ))s"
    [ "$rca" -ne 0 ] && { tail -20 "$A/run.log"; rc_all=1; }
  fi

  # ---- B：插件（开关关）----------------------------------------------
  B="$M/plug"
  mkdir -p "$B"
  if [ -f "$B/arms.json" ] && [ -d "$B/preds" ]; then
    echo "[$(date '+%F %T')] SKIP $m/B（插件产物已存在）"
  else
    cmdB=("$PY" manu/tracking/lookahead/lookahead_longquan.py
      --cache-root "$cache" --labels-root "$GT" --frames-root "$FRAMES" --frames-cache "$FC"
      --params-json "$PARAMS" --all-seqs --lookahead off --arms original
      --out-json "$B/arms.json" --out-preds-dir "$B/preds")
    printf '%s\n' "${cmdB[*]}" > "$B/command.txt"
    echo "[$(date '+%F %T')] RUN $m/B（插件 --lookahead off）"
    t0=$(date +%s)
    "${cmdB[@]}" > "$B/run.log" 2>&1
    rcb=$?
    echo "[$(date '+%F %T')] B rc=$rcb 用时 $(( $(date +%s) - t0 ))s"
    [ "$rcb" -ne 0 ] && { tail -20 "$B/run.log"; rc_all=1; }
  fi
done

echo "[$(date '+%F %T')] 跑批结束 ⇒ 比对：$PY manu/tracking/lookahead/server_overnight/compare_gate_a.py"
exit "$rc_all"
