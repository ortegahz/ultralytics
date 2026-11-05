#!/usr/bin/env bash
# ============================================================
# ★ 24 序 **cs_t0043（no-GMC）** 的 Gate A ——「关开关 ⇒ 逐位一致」
#
#   为什么 8 序的 Gate A 不够：8 序 Gate A 用的是**旧的 8 序 no-GMC 缓存**；
#   本轮 24 序用的是**新特征集建出来的新缓存**（`longquanshan_ir_gmc_median_nogmc`）
#   ⇒ 必须在这份**新缓存**上再过一次 Gate A，才能证明外挂没扰动跟踪资产。
#
#   A = 原脚本   manu/evaluation/system_sota_longquan.py（只读、禁改）
#   B = 插件开关关 manu/tracking/lookahead/lookahead_longquan.py --lookahead off --arms original
#   ⇒ 逐序列整份 JSON md5 必须全同（config / metrics / per_frame / payload / 资产）。
#
#   尺子1（同产物内四臂 source_frame.overall.f1 逐位相等）用 **24 序 cs_t0043 Δ=3** 的 arms.json；
#   它还没跑出来就等（最多等 2 小时），等不到则回退到 8 序那份（并在日志注明）。
#
#   启动（服务器）：
#     cd /tmp/pycharm_project_10ae9e2e
#     setsid nohup bash manu/tracking/lookahead/server_overnight/drive_gate_a_nogmc24.sh \
#        > runs/lookahead/gateA24_driver.log 2>&1 < /dev/null &
# ============================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
PY="${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}"
GT="${GT:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
FRAMES="${FRAMES:-/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg}"
FC="${FRAMES_CACHE:-/mnt/data/siping/frame_cache_lq}"
PARAMS="${PARAMS:-manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json}"
CACHE="${CACHE:-/mnt/data/siping/runs/nogmc_t3/swa/ms_cache_24seq}"
OUT="${OUT:-$REPO/runs/lookahead/gateA24}"
RULER24="$REPO/runs/lookahead/fi_nogmc24/cs_t0043/d3/arms.json"
RULER8="$REPO/runs/lookahead/fi_nogmc/cs_t0043/d3/arms.json"
THREADS="${THREADS:-8}"

export PYTHONPATH="$REPO"
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

cd "$REPO" || exit 2
mkdir -p "$OUT"
log(){ echo "[$(date '+%F %T')] [gateA24] $*"; }

# ---- 等缓存就绪（24 个 pkl）--------------------------------------------
log "等 24 序 no-GMC 缓存 $CACHE 就绪"
deadline=$(( $(date +%s) + 14400 ))
while :; do
  n=$(ls "$CACHE"/*.pkl 2>/dev/null | wc -l)
  [ "$n" -eq 24 ] && { log "缓存就绪（24 pkl）"; break; }
  if [ "$(date +%s)" -gt "$deadline" ]; then log "超时（4h），pkl=$n ⇒ 放弃"; exit 3; fi
  log "等待中 pkl=$n/24"
  sleep 60
done

cache="$(realpath "$CACHE")"     # ★ 必须绝对路径（config 块进 md5）
M="$OUT/cs_t0043"
mkdir -p "$M"
printf '%s\n' "cs_t0043 / no-GMC 语义 · **24 序官方口径**" > "$M/label.txt"

# ---- A：原脚本 ----------------------------------------------------------
A="$M/orig"; mkdir -p "$A"
if [ ! -f "$A/arms.json" ]; then
  cmdA=("$PY" manu/evaluation/system_sota_longquan.py
    --cache-root "$cache" --labels-root "$GT" --params-json "$PARAMS"
    --all-seqs --out-json "$A/arms.json" --out-preds-dir "$A/preds")
  printf '%s\n' "${cmdA[*]}" > "$A/command.txt"
  log "RUN A（原脚本，24 序）"
  t0=$(date +%s); "${cmdA[@]}" > "$A/run.log" 2>&1; rc=$?
  log "A rc=$rc 用时 $(( $(date +%s) - t0 ))s"
  [ $rc -ne 0 ] && tail -20 "$A/run.log"
else
  log "SKIP A（产物已存在）"
fi

# ---- B：插件（开关关）--------------------------------------------------
B="$M/plug"; mkdir -p "$B"
if [ ! -f "$B/arms.json" ]; then
  cmdB=("$PY" manu/tracking/lookahead/lookahead_longquan.py
    --cache-root "$cache" --labels-root "$GT" --frames-root "$FRAMES" --frames-cache "$FC"
    --params-json "$PARAMS" --all-seqs --lookahead off --arms original
    --out-json "$B/arms.json" --out-preds-dir "$B/preds")
  printf '%s\n' "${cmdB[*]}" > "$B/command.txt"
  log "RUN B（插件 --lookahead off，24 序）"
  t0=$(date +%s); "${cmdB[@]}" > "$B/run.log" 2>&1; rc=$?
  log "B rc=$rc 用时 $(( $(date +%s) - t0 ))s"
  [ $rc -ne 0 ] && tail -20 "$B/run.log"
else
  log "SKIP B（产物已存在）"
fi

# ---- 等尺子1 的 24 序 arms.json ---------------------------------------
ruler="$RULER8"
d2=$(( $(date +%s) + 7200 ))
while [ ! -f "$RULER24" ] && [ "$(date +%s)" -lt "$d2" ]; do sleep 60; done
if [ -f "$RULER24" ]; then ruler="$RULER24"; log "尺子1 用 24 序 arms.json"; else log "⚠️ 24 序尺子1 产物未就绪 ⇒ 回退 8 序"; fi

log "比对：compare_gate_a.py --gate-root $OUT"
"$PY" manu/tracking/lookahead/server_overnight/compare_gate_a.py \
  --gate-root "$OUT" --ruler-arms-json "$ruler" \
  --out-json "$OUT/gateA.json" --out-md "$OUT/gateA.md"
log "Gate A（24 序）结束 rc=$?"
