#!/bin/bash
# ==============================================================================
# 启动 **no-GMC** 的 Optuna 混合比例超参搜索（后台常驻，与 SSH 会话完全解耦）
#
# 与 `launch_mix_ratio_search.sh`（GMC 版）**配置逐项完全一致**：
#   底模 / epochs / batch / imgsz / workers / gpus / 搜索空间 / TPE / 种子 trial /
#   门禁脚本 / 截止机制 / 断点续跑 / 状态查询 —— **只换三样**：
#     ① 训练集 / 测试集 → no-GMC 特征集
#     ② 基线缓存 + 参照系 → no-GMC 语义（`--reference-frame`）
#     ③ 输出根 + study 名 → runs/optuna_nogmc（独立 study.db，不污染 GMC 研究）
#
# 底模：runs/optuna_p0_nas/trial_0474/weights/best.pt（**去 P0**，与 RC2 trial_0080 同）
#
# 前置（没有参照系包就不许开跑 —— 否则会用 GMC 基线判 no-GMC 模型）：
#   1) python manu/training/measure_reference_frame.py \
#        --uav-data .../uav_gmc_median_nogmc/data.yaml \
#        --ft-root  .../longquanshan_ft_gmc_median_nogmc \
#        --out-dir runs/nogmc_ref --label nogmc --gpus 0,1,2,3
#
# 用法（服务器 `ssh -p 32222 huangzhe@192.168.99.40`，`conda activate uav`）：
#   bash manu/training/launch_nogmc_mix_ratio_search.sh              # 默认 19 小时
#   MAX_HOURS=8 bash manu/training/launch_nogmc_mix_ratio_search.sh  # 自定义时长
#   N_TRIALS=200 bash manu/training/launch_nogmc_mix_ratio_search.sh # 自定义 trial 数
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
OUT_ROOT="${OUT_ROOT:-runs/optuna_nogmc}"
GPUS="${GPUS:-0,1,2,3}"
N_TRIALS="${N_TRIALS:-4096}"
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-32}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
MAX_HOURS="${MAX_HOURS:-19}"
STUDY_NAME="${STUDY_NAME:-mix_ratio_search_nogmc}"

# ---- no-GMC 数据与参照系（**唯一与 GMC 版不同的地方**）----
B=/mnt/data/siping/datasets/manu
LQ_DATA="${LQ_DATA:-$B/longquanshan_ft_gmc_median_nogmc/data.yaml}"
UAV_DATA="${UAV_DATA:-$B/uav_gmc_median_nogmc/data.yaml}"
REF_DIR="${REF_DIR:-runs/nogmc_ref}"
REFERENCE_FRAME="${REFERENCE_FRAME:-$REF_DIR/reference_frame.json}"
BASELINE_CACHE="${BASELINE_CACHE:-$REF_DIR/uav_cache_base.pkl}"
BASE_WEIGHTS="${BASE_WEIGHTS:-runs/optuna_p0_nas/trial_0474/weights/best.pt}"

cd "$PROJECT_ROOT" || { echo "[FATAL] 进不去 $PROJECT_ROOT"; exit 1; }
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav || { echo "[FATAL] conda 激活失败"; exit 1; }

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
echo "         参照系 $REFERENCE_FRAME | 基线缓存 $BASELINE_CACHE"
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
  --weights "$BASE_WEIGHTS" \
  --baseline-cache "$BASELINE_CACHE" \
  --reference-frame "$REFERENCE_FRAME" \
  >> "$LOG" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
disown 2>/dev/null || true

sleep 3
if kill -0 "$NEW_PID" 2>/dev/null; then
  echo "[OK] no-GMC 搜索已后台启动：PID $NEW_PID（session 独立，不随 SSH 断开而终止）"
  echo "     监控： python manu/training/status_mix_ratio_search.py --out-root $OUT_ROOT \\"
  echo "             --study-name $STUDY_NAME --reference-frame $REFERENCE_FRAME"
  echo "     日志： tail -f $LOG"
else
  echo "[FAIL] 启动失败，立刻查看 $LOG"
  tail -30 "$LOG"
  exit 1
fi
