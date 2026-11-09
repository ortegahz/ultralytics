#!/bin/bash
# ==============================================================================
# 启动 **「NPU int8 ≤ 20 ms」搜索**（周末长跑，后台常驻，与 SSH 会话完全解耦）
#
# 目标（用户 2026-11-06 定）：**NPU int8 推理 ≤ 20 ms** 且 **T1/T2/T3 下降非常微小**，
# deadline **下周一 09:00**；**宿主今天 18:30 关机**，**板卡今天 18:30 断电**。
#
# ⇒ 本脚本的三条设计主线：
#   1. **速度查表**：`--speed-table` 是今天（板卡在线时）实测的「结构 → rknn_run ms」谱系表。
#      派单前查表 ⇒ **>20 ms 的结构被直接剪枝，不浪费一次训练**（板卡周末已断电，没法再测）。
#   2. **精度优先**：约束是「≤20 ms」而不是「最快」⇒ 候选结构按**头层级**排序
#      （p1 320×320 > p2_fused 160×160 > p3 80×80）⇒ 20 ms 预算内尽量保住高分辨率定位能力。
#   3. **硬截止**：`MAX_HOURS` 按 deadline 反算 ⇒ 到点停派单，剩余在跑的臂跑完即收割。
#
# 用法（服务器 `ssh -p 32222 huangzhe@192.168.99.40`，`conda activate uav`）：
#   bash manu/training/launch_npu20ms_search.sh
#   MAX_HOURS=6 N_TRIALS=8 bash manu/training/launch_npu20ms_search.sh
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
OUT_ROOT="${OUT_ROOT:-/mnt/data/siping/runs/npu20ms}"
GPUS="${GPUS:-0,1,2,3}"
N_TRIALS="${N_TRIALS:-24}"
# 硬截止：从启动起算的上限（deadline 周一 09:00 ⇒ 19:15 起最多 ~13.5 h）
MAX_HOURS="${MAX_HOURS:-13}"
BATCH="${BATCH:-32}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
# ⚠️ **study 名带 _v2**（2026-11-06）：结构维度的 choices 改了（head_conv 加 48/40、
#    p2_out_scale 加 0.25）⇒ **旧 study 已记录 (64,32) / (1.0,0.75,0.5)**
#    ⇒ Optuna 的 CategoricalDistribution 不支持动态值空间 ⇒ 复用旧 study 会直接抛
#    `ValueError: CategoricalDistribution does not support dynamic value space.`
#    ⇒ **必须换新 study 名**。
STUDY_NAME="${STUDY_NAME:-mix_ratio_search_npu20ms_v2}"
PLAN="${PLAN:-manu/training/npu20ms_plan.json}"
SPEED_TABLE="${SPEED_TABLE:-/mnt/data/siping/runs/npu20ms/struct_speed_table.json}"
SPEED_LIMIT_MS="${SPEED_LIMIT_MS:-21.0}"
EPOCHS_DEFAULT="${EPOCHS_DEFAULT:-15}"     # 快筛用；plan 里逐臂可覆盖（终验臂写 20）

# ---- no-GMC 数据与参照系（与配方线、结构线完全一致）----
B=/mnt/data/siping/datasets/manu
LQ_DATA="${LQ_DATA:-$B/longquanshan_ft_gmc_median_nogmc/data.yaml}"
UAV_DATA="${UAV_DATA:-$B/uav_gmc_median_nogmc/data.yaml}"
REF_DIR="${REF_DIR:-/mnt/data/siping/runs/nogmc_ref_weekend}"
REFERENCE_FRAME="${REFERENCE_FRAME:-$REF_DIR/reference_frame.json}"
BASELINE_CACHE="${BASELINE_CACHE:-$REF_DIR/uav_cache_base.pkl}"
CS_T0043="${CS_T0043:-/mnt/data/siping/runs/nogmc_t3/swa/cs_t0043.pt}"
BASE_WEIGHTS="${BASE_WEIGHTS:-$CS_T0043}"

cd "$PROJECT_ROOT" || { echo "[FATAL] 进不去 $PROJECT_ROOT"; exit 1; }
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav || { echo "[FATAL] conda 激活失败"; exit 1; }

# ---- 前置闸门 ----
for f in "$REFERENCE_FRAME" "$BASELINE_CACHE" "$BASE_WEIGHTS" "$PLAN" "$SPEED_TABLE"; do
  [ -f "$f" ] || { echo "[ABORT] 缺文件：$f"; exit 1; }
done
[ -f "$UAV_DATA" ] && [ -f "$LQ_DATA" ] || { echo "[ABORT] no-GMC 数据集缺失"; exit 1; }
if ! grep -q "struct-space" manu/training/optuna_mix_ratio_search.py; then
  echo "[ABORT] dispatcher 里没有 --struct-space ⇒ 代码没同步好"; exit 1
fi
if ! grep -q "speed-table" manu/training/optuna_mix_ratio_search.py; then
  echo "[ABORT] dispatcher 里没有 --speed-table ⇒ 速度闸门不可用"; exit 1
fi

mkdir -p "$OUT_ROOT/logs"

# ---- 重复启动保护 ----
PID_FILE="$OUT_ROOT/dispatcher.pid"
if [ -f "$PID_FILE" ]; then
  OLD=$(cat "$PID_FILE" 2>/dev/null || echo "")
  if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
    echo "[ABORT] 搜索已在运行（PID $OLD）"; exit 1
  fi
  echo "[WARN] 陈旧 PID 文件（$OLD），覆盖"
fi

LOG="$OUT_ROOT/logs/dispatcher.log"
echo "[LAUNCH] 时间 $(date '+%F %T') | PID -> $PID_FILE | 日志 $LOG"
echo "         初始化权重 $BASE_WEIGHTS | 计划 $PLAN | 速度表 $SPEED_TABLE（上限 ${SPEED_LIMIT_MS} ms）"
echo "         MAX_HOURS=$MAX_HOURS | N_TRIALS=$N_TRIALS | epochs 默认 $EPOCHS_DEFAULT（plan 可逐臂覆盖）"

setsid nohup python -u manu/training/optuna_mix_ratio_search.py \
  --output-root "$OUT_ROOT" \
  --study-name "$STUDY_NAME" \
  --gpus "$GPUS" \
  --n-trials "$N_TRIALS" \
  --epochs "$EPOCHS_DEFAULT" \
  --batch "$BATCH" \
  --imgsz "$IMGSZ" \
  --workers "$WORKERS" \
  --max-hours "$MAX_HOURS" \
  --longquan-data "$LQ_DATA" \
  --uav-data "$UAV_DATA" \
  --weights "$BASE_WEIGHTS" \
  --baseline-cache "$BASELINE_CACHE" \
  --reference-frame "$REFERENCE_FRAME" \
  --struct-space \
  --speed-table "$SPEED_TABLE" \
  --speed-limit-ms "$SPEED_LIMIT_MS" \
  --enqueue-plan "$PLAN" \
  >> "$LOG" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
disown 2>/dev/null || true

sleep 6
if kill -0 "$NEW_PID" 2>/dev/null; then
  echo "[OK] 「NPU≤20ms」搜索已后台启动：PID $NEW_PID（setsid 独立 session，宿主/板卡断电都不影响）"
  echo "     监控： python manu/training/status_mix_ratio_search.py --out-root $OUT_ROOT \\"
  echo "             --study-name $STUDY_NAME --reference-frame $REFERENCE_FRAME"
  echo "     日志： tail -f $LOG"
else
  echo "[FAIL] 启动失败，看 $LOG"; tail -30 "$LOG"; exit 1
fi
