#!/usr/bin/env bash
# ============================================================
# 确定性夜间评测: 24 序列官方缓存 -> 原逻辑 / stale / 外挂 三臂 -> 对照表
# 不依赖 LLM, 即使 codex 会话挂掉也能产出数字。
# ============================================================
set -uo pipefail
ROOT="${ROOT:-/tmp/pycharm_project_10ae9e2e}"
OUT="$ROOT/runs/lookahead/server_out"
BUNDLE="$ROOT/manu/tracking/lookahead/server_overnight"
cd "$ROOT"; export PYTHONPATH="$ROOT"; export PYTHONUNBUFFERED=1
mkdir -p "$OUT"

CACHE_ROOT="$(cat "$OUT/cache_root.txt" 2>/dev/null | head -1)"
LABELS_ROOT="${LABELS_ROOT:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
PARAMS="$ROOT/manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json"
LKH="$ROOT/manu/tracking/lookahead/lookahead_longquan.py"

echo "[$(date '+%F %T')] CACHE_ROOT=$CACHE_ROOT"
if [ -z "$CACHE_ROOT" ] || [ ! -d "$CACHE_ROOT" ]; then
  echo "[$(date '+%F %T')] 未找到 24 序列 ms_cache -> 用官方一键流水线重建(GPU)"
  echo "[$(date '+%F %T')] bash manu/evaluation/run_trial0080_longquan_eval.sh"
  ( nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1 | awk -F, '{if ($1+0 > 2000) print 1; else print 0}' ) > "$OUT/gpu_busy.flag" 2>/dev/null || echo 0 > "$OUT/gpu_busy.flag"
  if [ "$(cat "$OUT/gpu_busy.flag" 2>/dev/null)" = "1" ]; then
    echo "[$(date '+%F %T')] GPU 正在被占用, 保持 WAIT_GPU=1 等它让出来"
    WAIT_GPU=1 bash manu/evaluation/run_trial0080_longquan_eval.sh
  else
    WAIT_GPU=0 bash manu/evaluation/run_trial0080_longquan_eval.sh
  fi
  CACHE_ROOT="$ROOT/runs/optuna_mix_ratio/trials/trial_0080/lq_eval/ms_cache"
  echo "${CACHE_ROOT}" > "$OUT/cache_root.txt"
  [ -d "$CACHE_ROOT" ] || { echo "[$(date '+%F %T')] FATAL: 重建失败"; exit 3; }
fi

run(){ echo "[$(date '+%F %T')] RUN $*"; "$@"; echo "[$(date '+%F %T')] EXIT=$?"; }

# ---------- 臂0: 原代码(带 track 臂, 无外挂) -- 与交付口径 94.747% 对齐 ----------
run python manu/evaluation/system_sota_longquan.py \
  --cache-root "$CACHE_ROOT" --labels-root "$LABELS_ROOT" --params-json "$PARAMS" \
  --all-seqs --out-json "$OUT/original_24.json" --out-preds-dir "$OUT/preds_original_24"

# ---------- 臂0b: 外挂关闭闸门(Gate A): 必须与臂0 逐位一致 ----------
run python "$LKH" --lookahead off --arms original \
  --cache-root "$CACHE_ROOT" --labels-root "$LABELS_ROOT" --params-json "$PARAMS" \
  --all-seqs --out-json "$OUT/gateA_lookahead_off.json" --out-preds-dir "$OUT/preds_gateA_off"

# ---------- 臂1: 纯检测基线(硬吃 320ms 滞后) ----------
run python "$LKH" --lookahead off --arms stale \
  --cache-root "$CACHE_ROOT" --labels-root "$LABELS_ROOT" --params-json "$PARAMS" \
  --all-seqs --out-json "$OUT/arm1_stale_24.json"

# ---------- 臂3: 外挂(perframe 最优配置) ----------
run python "$LKH" --lookahead replay --delta 8 --arms original,stale,replay \
  --replay-max-hold 8 --replay-fallback holdv --replay-anchor delta \
  --fast-win-size 48 --fast-bg-window 96 --no-replay-hold-use-ref \
  --cache-root "$CACHE_ROOT" --labels-root "$LABELS_ROOT" --params-json "$PARAMS" \
  --all-seqs --out-json "$OUT/arm3_replay_24.json" --out-preds-dir "$OUT/preds_arm3_replay"

# ---------- 对照表 ----------
run python "$ROOT/manu/tracking/lookahead/make_report_table.py" \
  --runs "$OUT/original_24.json" "$OUT/arm1_stale_24.json" "$OUT/arm3_replay_24.json" \
  --labels "原代码(交付口径)" "臂1 纯检测基线(硬吃320ms)" "臂3 外挂(rolling/perframe)" \
  > "$OUT/comparison_24.md"

echo "[$(date '+%F %T')] DONE. 产物在 $OUT"
ls -la "$OUT"
