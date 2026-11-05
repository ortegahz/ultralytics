#!/usr/bin/env bash
# ============================================================
# ★ P4 —— no-GMC (cs_t0043) 的 OSD 三臂对照视频（stale / replay / original）
#
#   目的：肉眼可查的证据。同一序列上并排看三臂：
#     stale   = 老逻辑（真吃 Δ record 滞后），**交付基线**
#     replay  = 外挂快推（look-ahead）
#     original= 零延迟天花板（oracle 上界，**不是基线**）
#
#   用法（服务器）：
#     cd /tmp/pycharm_project_10ae9e2e
#     bash manu/tracking/lookahead/server_overnight/run_osd_3arm_nogmc.sh
#     env SEQS="VIDEO00033_19700101_015000" bash .../run_osd_3arm_nogmc.sh   # 只跑一条
#     env MODEL=cs_t0043 DELTA=3 bash .../run_osd_3arm_nogmc.sh
#
#   ★ 幂等：已有产物跳过（被杀后重跑不浪费）；进度写 progress.txt。
#   ★ 渲染很慢时**按序列分批**：每条序列跑完写一次 progress.txt。
#   ★ 不改 manu/tracking/lookahead/render_osd_3arm.sh（禁改），本脚本是它的薄封装。
# ============================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
PY="${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}"
GT="${GT:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
FRAMES="${FRAMES:-/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg}"
FC="${FRAMES_CACHE:-/mnt/data/siping/frame_cache_lq}"
PARAMS="${PARAMS:-manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json}"
DELIVERY_SH="$REPO/manu/tracking/lookahead/run_delivery.sh"
OUT="${OUT:-$REPO/runs/lookahead/osd}"
THREADS="${THREADS:-6}"
MODEL="${MODEL:-cs_t0043}"
DELTA="${DELTA:-3}"
SEQS="${SEQS:-VIDEO00007_19700101_001324 VIDEO00033_19700101_015000}"

export PYTHONPATH="$REPO"
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

cd "$REPO" || exit 2

# ---- 模型包（缓存根绝对路径）---------------------------------------------
case "$MODEL" in
  cs_t0043) CACHE="/mnt/data/siping/runs/nogmc_t3/swa/gate/cs_t0043/ms_cache" ;;
  rc2)      CACHE="$REPO/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache" ;;
  trial_0049) CACHE="/mnt/data/siping/runs/nogmc_t3/trials/trial_0049/gate/epoch20/ms_cache" ;;
  *) echo "[FATAL] 未知 MODEL=$MODEL"; exit 3 ;;
esac
[ -d "$CACHE" ] || { echo "[FATAL] 缓存不存在: $CACHE"; exit 3; }
CACHE="$(realpath "$CACHE")"

WORK="$OUT/nogmc_${MODEL}_d${DELTA}"
VID="$OUT/videos"
mkdir -p "$WORK" "$VID"
PROG="$WORK/progress.txt"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$PROG"; }

# ---- CFG 真值：直接从交付脚本读取（多行续行 ⇒ 必须用 awk 取整块）----------
CFG_ASSIGN="$(awk '/^CFG="/ {f=1} f {print} f && /"[[:space:]]*$/ {exit}' "$DELIVERY_SH")"
[ -n "$CFG_ASSIGN" ] || { echo "[FATAL] 无法从 $DELIVERY_SH 解析 CFG"; exit 3; }
CFG=""
eval "$CFG_ASSIGN"
[ -n "$CFG" ] || { echo "[FATAL] CFG 解析为空"; exit 3; }
for _must in "--replay-roi-selfpred-max-px 16" "--replay-rollback-min-peak 0.90" \
             "--replay-polarity auto" "--replay-xcorr-uniq-gate 0.6" "--replay-mode perframe" \
             "--replay-abstain-peak-below 0.30" "--replay-rollback-max-frames 16"; do
  case " $CFG " in
    *" $_must "*) ;;
    *) echo "[FATAL] CFG 缺少必需旋钮 '$_must' ⇒ 解析被截断，拒绝跑"; exit 3 ;;
  esac
done
log "模型=$MODEL 缓存=$CACHE Δ=$DELTA"
log "CFG（源自 run_delivery.sh，${#CFG} 字符）: $CFG"

FRAME_ARGS=""
if [ -d "$FC" ]; then FRAME_ARGS="--frames-cache $FC"; log "帧缓存 memmap: $FC"; fi

SEQCSV="$(echo "$SEQS" | tr ' ' ',')"
# 插件跑的序列范围：默认跑**全部 8 序 gate 集**（preds 目录要能覆盖所有待渲染序列，
# 否则只跑一条序列时后面那条就没 preds 可渲染）。PLUGIN_SEQS="all" ⇒ --all-seqs；
# 否则用逗号分隔的序列列表。
PLUGIN_SEQS="${PLUGIN_SEQS:-all}"
if [ "$PLUGIN_SEQS" = "all" ]; then PLUGIN_ARG=(--all-seqs); else PLUGIN_ARG=(--seqs "$PLUGIN_SEQS"); fi

# ---- 三臂各跑一次插件（主臂由 --lookahead 决定 ⇒ 决定写哪套 preds）--------
#   off    → main_arm=original（零延迟天花板）
#   stale  → main_arm=stale（老逻辑，交付基线）
#   replay → main_arm=replay（外挂快推）；顺带把 original/stale 也算进 arms.json
run_arm(){  # $1=臂名  $2=--lookahead 值  $3=--arms 值
  local arm="$1" mode="$2" arms="$3"
  local O="$WORK/preds_$arm"
  if [ -f "$WORK/arms_$arm.json" ] && ls "$O"/*.json >/dev/null 2>&1; then
    log "SKIP 插件臂 $arm（产物已存在）"; return 0
  fi
  mkdir -p "$O"
  local cmd=("$PY" manu/tracking/lookahead/lookahead_longquan.py
    --cache-root "$CACHE" --labels-root "$GT" --frames-root "$FRAMES" $FRAME_ARGS
    --params-json "$PARAMS" "${PLUGIN_ARG[@]}"
    --lookahead "$mode" --arms "$arms" --delta "$DELTA" $CFG
    --out-json "$WORK/arms_$arm.json" --out-preds-dir "$O")
  printf '%s\n' "${cmd[*]}" > "$WORK/command_$arm.txt"
  log "RUN 插件臂 $arm (--lookahead $mode --arms $arms)"
  local t0=$(date +%s)
  "${cmd[@]}" > "$WORK/run_$arm.log" 2>&1
  local rc=$?
  log "DONE 插件臂 $arm rc=$rc 用时 $(( $(date +%s) - t0 ))s"
  [ "$rc" -ne 0 ] && { tail -20 "$WORK/run_$arm.log"; return 1; }
  return 0
}

rc_all=0
run_arm original off original   || rc_all=1
run_arm stale   stale stale     || rc_all=1
run_arm replay  replay original,stale,replay || rc_all=1

# ---- 渲染三臂（按序列分批；每条序列跑完写一次 progress.txt）--------------
for SEQ in $SEQS; do
  W="$OUT/windows/${SEQ}_3arm.json"
  if [ ! -f "$W" ]; then log "WARN $SEQ 没有窗口文件 $W ⇒ 跳过渲染"; continue; fi
  START=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['start'])" "$W")
  LIMIT=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['limit'])" "$W")
  log "RENDER $SEQ start=$START limit=$LIMIT"
  for arm in stale replay original; do
    MP4="$VID/${SEQ}_3arm_${arm}_${MODEL}_d${DELTA}.mp4"
    if [ -s "$MP4" ]; then log "SKIP 渲染 $SEQ/$arm（$MP4 已存在）"; continue; fi
    t0=$(date +%s)
    $PY manu/videos/render_sys_sota_longquan.py \
      --preds-dir "$WORK/preds_$arm" --seq "$SEQ" \
      --output "$MP4" --labels-root "$GT" --frames-root "$FRAMES" \
      --start "$START" --limit "$LIMIT" --codec ffmpeg:h264 \
      --out-json "$VID/${SEQ}_3arm_${arm}_${MODEL}_d${DELTA}.json" \
      > "$VID/${SEQ}_3arm_${arm}_${MODEL}_d${DELTA}.log" 2>&1
    r=$?
    if [ "$r" -ne 0 ]; then
      log "FAIL 渲染 $SEQ/$arm rc=$r"; tail -10 "$VID/${SEQ}_3arm_${arm}_${MODEL}_d${DELTA}.log"; rc_all=1
    else
      log "OK 渲染 $SEQ/$arm 用时 $(( $(date +%s) - t0 ))s size=$(stat -c%s "$MP4")B"
    fi
  done
  log "SEQ DONE $SEQ"
done

log "OSD_3ARM_NOGMC_DONE rc=$rc_all 产物在 $VID"
exit "$rc_all"
