#!/bin/bash
# ==============================================================================
# 周末 no-GMC 复现实验 · 搜索启动器（后台常驻，与 SSH 会话完全解耦）
#
# 与 10-27/28 那轮 no-GMC 搜索的**关键差异**：
#   ① **产物落 /mnt/data**（持久盘）—— 上一轮产物只存在服务器 /tmp，10-29 被清掉，
#      99 行 CSV / SUMMARY / 全部权重一次性丢失（数据集因为在 /mnt/data 才活下来）。
#   ② **扩展搜索空间**（--extended-space）：+init_from / +epochs / +backbone_lr_scale
#      / +freeze_backbone_epochs —— 核心假设是「用 GMC RC2(trial_0080) 当初始化权重」。
#   ③ **每 trial 可换初始化权重**（--init-weights-map）与**每 trial 可换 epoch 数**。
#
# 底模两种（都由 --init-weights-map 提供）：
#   base474 = runs/optuna_p0_nas/trial_0474/weights/best.pt（去 P0）—— 旧实验起点
#   rc2     = runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt —— 新假设起点
#
# 前置（没有参照系包就不许开跑 —— 否则会拿 GMC 基线判 no-GMC 模型）：
#   python manu/training/measure_reference_frame.py \
#     --uav-data /mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml \
#     --ft-root  /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc \
#     --out-dir <REF_DIR> --label nogmc --gpus 0,1,2,3 \
#     --track2-fp-max-increment 60
#
# 用法（服务器 `ssh -p 32222 huangzhe@192.168.99.40`，`conda activate uav`）：
#   bash manu/training/launch_nogmc_weekend_search.sh
#   MAX_HOURS=40 OUT_ROOT=... bash manu/training/launch_nogmc_weekend_search.sh
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
RUNS_BASE="${RUNS_BASE:-/mnt/data/siping/runs}"
OUT_ROOT="${OUT_ROOT:-$RUNS_BASE/nogmc_weekend}"
REF_DIR="${REF_DIR:-$RUNS_BASE/nogmc_ref_weekend}"
GPUS="${GPUS:-0,1,2,3}"
N_TRIALS="${N_TRIALS:-4096}"
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-32}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
MAX_HOURS="${MAX_HOURS:-40}"
STUDY_NAME="${STUDY_NAME:-mix_ratio_search_nogmc_weekend}"
PLAN="${PLAN:-manu/training/nogmc_weekend_plan.json}"
INIT_MAP="${INIT_MAP:-manu/training/nogmc_weekend_init_weights.json}"

# ---- no-GMC 数据与参照系 ----
B=/mnt/data/siping/datasets/manu
LQ_DATA="${LQ_DATA:-$B/longquanshan_ft_gmc_median_nogmc/data.yaml}"
UAV_DATA="${UAV_DATA:-$B/uav_gmc_median_nogmc/data.yaml}"
REFERENCE_FRAME="${REFERENCE_FRAME:-$REF_DIR/reference_frame.json}"
BASELINE_CACHE="${BASELINE_CACHE:-$REF_DIR/uav_cache_base.pkl}"

cd "$PROJECT_ROOT" || { echo "[FATAL] 进不去 $PROJECT_ROOT"; exit 1; }
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav || { echo "[FATAL] conda 激活失败"; exit 1; }
export PYTHONPATH=.

# ---- 前置闸门：参照系包必须在位（否则会拿 GMC 基线判 no-GMC 模型）----
if [ ! -f "$REFERENCE_FRAME" ]; then
  echo "[ABORT] 缺参照系包：$REFERENCE_FRAME"
  echo "        先跑 measure_reference_frame.py（见本脚本头部）"
  exit 1
fi
if [ ! -f "$UAV_DATA" ] || [ ! -f "$LQ_DATA" ]; then
  echo "[ABORT] no-GMC 数据集缺失：$UAV_DATA / $LQ_DATA"
  exit 1
fi

mkdir -p "$OUT_ROOT/logs"

# ---- 重复启动保护 ----
PID_FILE="$OUT_ROOT/dispatcher.pid"
if [ -f "$PID_FILE" ]; then
  OLD=$(cat "$PID_FILE" 2>/dev/null || echo "")
  if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
    echo "[ABORT] 搜索已在运行（PID $OLD）。要重启请先停掉它"
    exit 1
  fi
  echo "[WARN] 发现陈旧 PID 文件（$OLD），将覆盖"
fi

LOG="$OUT_ROOT/logs/dispatcher.log"
echo "[LAUNCH] 时间 $(date '+%F %T') | PID 写入 $PID_FILE | 日志 $LOG"
echo "         输出根 $OUT_ROOT（★持久盘 /mnt/data）"
echo "         参照系 $REFERENCE_FRAME | 基线缓存 $BASELINE_CACHE"
echo "         对照臂清单 $PLAN | 初始化权重表 $INIT_MAP"
setsid nohup python -u manu/training/optuna_mix_ratio_search.py \
  --output-root "$OUT_ROOT" \
  --study-name "$STUDY_NAME" \
  --gpus "$GPUS" \
  --n-trials "$N_TRIALS" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --imgsz "$IMGSZ" \
  --workers "$WORKERS" \
  --max-hours "$MAX_HOURS" \
  --longquan-data "$LQ_DATA" \
  --uav-data "$UAV_DATA" \
  --baseline-cache "$BASELINE_CACHE" \
  --reference-frame "$REFERENCE_FRAME" \
  --extended-space \
  --init-weights-map "$INIT_MAP" \
  --enqueue-plan "$PLAN" \
  >> "$LOG" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
disown 2>/dev/null || true

sleep 5
if kill -0 "$NEW_PID" 2>/dev/null; then
  echo "[OK] 周末 no-GMC 搜索已后台启动：PID $NEW_PID（session 独立，不随 SSH 断开/本机关机而终止）"
  echo "     监控： python manu/training/status_mix_ratio_search.py --out-root $OUT_ROOT \\"
  echo "             --study-name $STUDY_NAME --reference-frame $REFERENCE_FRAME"
  echo "     日志： tail -f $LOG"
else
  echo "[FAIL] 启动失败，立刻查看 $LOG"
  tail -40 "$LOG"
  exit 1
fi
