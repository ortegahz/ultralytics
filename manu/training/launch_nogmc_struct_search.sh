#!/bin/bash
# ==============================================================================
# 启动 **no-GMC 结构线** 的 Optuna 搜索（后台常驻，与 SSH 会话完全解耦）
#
# 目标（用户 2026-11-05）：在 `cs_t0043`（3 粒子同配方 Model Soup）基础上
#   **删掉 P2 分支 + 删掉 bbox 回归分支 ⇒ 退化成 YOLO26n 原生拓扑、只改成 heatmap**，
#   **all relu**，从 cs_t0043 初始化继续搜索，期望 **速度大幅提升 + 精度微量下降**。
#
# 与 `launch_nogmc_mix_ratio_search.sh`（no-GMC 配方线）**配置逐项一致**，只多三样：
#   ① `--struct-space`         打开结构搜索空间（no_p2/head_level/act/head_conv/late/unfreeze）
#   ② `--enqueue-plan`         注入 20 条**确定性**臂（2 复现对照 + 16 结构网格 + 2 配方邻域）
#   ③ `--weights` 指向 cs_t0043（**不是** trial_0474）⇒ 所有臂从 cs_t0043 初始化
#
# ⚠️ **为什么不开 --extended-space**：
#   开着会 suggest `init_from`（base474/rc2）并覆盖我们的 cs_t0043 初始化，
#   而且 `epochs` 会变成可采样维度 ⇒ 就不再是"在 cs_t0043 上微调"的受控口径。
#   不开 ⇒ epochs 固定为 `--epochs 20`（= cs_t0043 母配方的 epoch 数），
#   `backbone_lr_scale=1.0` / `freeze_backbone_epochs=0` 取 dispatcher 默认值
#   （**与 cs_t0043 母配方逐项一致**，已核对 optuna_summary.csv）。
#
# 前置（没有参照系包就不许开跑 —— 否则会用 GMC 基线判 no-GMC 模型）：
#   1) python manu/training/measure_reference_frame.py \
#        --uav-data .../uav_gmc_median_nogmc/data.yaml \
#        --ft-root  .../longquanshan_ft_gmc_median_nogmc \
#        --out-dir runs/nogmc_ref --label nogmc --gpus 0,1,2,3
#
# 用法（服务器 `ssh -p 32222 huangzhe@192.168.99.40`，`conda activate uav`）：
#   bash manu/training/launch_nogmc_struct_search.sh
#   MAX_HOURS=8 N_TRIALS=12 bash manu/training/launch_nogmc_struct_search.sh
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
# ★ 输出根统一放 /mnt/data/siping/runs/（用户 2026-11-05 定的长期约定：以后都放这里）
OUT_ROOT="${OUT_ROOT:-/mnt/data/siping/runs/nogmc_struct}"
GPUS="${GPUS:-0,1,2,3}"
N_TRIALS="${N_TRIALS:-20}"
EPOCHS="${EPOCHS:-20}"          # = cs_t0043 母配方的 epoch 数（微调口径）
BATCH="${BATCH:-32}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
# 任务 deadline：明天 09:00；搜索必须在那之前收口 ⇒ 默认给 12 h（~19:40 + 12h = 07:40）
MAX_HOURS="${MAX_HOURS:-12}"
STUDY_NAME="${STUDY_NAME:-mix_ratio_search_nogmc_struct}"
PLAN="${PLAN:-manu/training/nogmc_struct_plan.json}"

# ---- no-GMC 数据与参照系（与配方线完全一致）----
B=/mnt/data/siping/datasets/manu
LQ_DATA="${LQ_DATA:-$B/longquanshan_ft_gmc_median_nogmc/data.yaml}"
UAV_DATA="${UAV_DATA:-$B/uav_gmc_median_nogmc/data.yaml}"
# ⚠️ no-GMC 线实际用的参照系在 /mnt/data/siping/runs/nogmc_ref_weekend/
#    （项目内 runs/nogmc_ref 不存在 —— 实测确认）
REF_DIR="${REF_DIR:-/mnt/data/siping/runs/nogmc_ref_weekend}"
REFERENCE_FRAME="${REFERENCE_FRAME:-$REF_DIR/reference_frame.json}"
BASELINE_CACHE="${BASELINE_CACHE:-$REF_DIR/uav_cache_base.pkl}"

# ★ 初始化权重 = cs_t0043（3 粒子同配方 Model Soup）
CS_T0043="${CS_T0043:-/mnt/data/siping/runs/nogmc_t3/swa/cs_t0043.pt}"
BASE_WEIGHTS="${BASE_WEIGHTS:-$CS_T0043}"

cd "$PROJECT_ROOT" || { echo "[FATAL] 进不去 $PROJECT_ROOT"; exit 1; }
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav || { echo "[FATAL] conda 激活失败"; exit 1; }

# ---- 前置闸门 ----
if [ ! -f "$REFERENCE_FRAME" ]; then
  echo "[ABORT] 缺参照系包：$REFERENCE_FRAME"
  echo "        先跑 measure_reference_frame.py（见本脚本头部）"
  exit 1
fi
if [ ! -f "$UAV_DATA" ] || [ ! -f "$LQ_DATA" ]; then
  echo "[ABORT] no-GMC 数据集缺失：$UAV_DATA / $LQ_DATA"
  exit 1
fi
if [ ! -f "$BASE_WEIGHTS" ]; then
  echo "[ABORT] 初始化权重不存在：$BASE_WEIGHTS（cs_t0043）"
  exit 1
fi
if [ ! -f "$PLAN" ]; then
  echo "[ABORT] 计划文件缺失：$PLAN"
  exit 1
fi
# ★ 代码必须已含 --struct-space（dispatcher 是启动时导入的，**运行中改代码不生效**）
if ! grep -q "struct-space" manu/training/optuna_mix_ratio_search.py; then
  echo "[ABORT] dispatcher 里没有 --struct-space ⇒ 代码没同步好，先同步再启"
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
echo "         ★初始化权重 $BASE_WEIGHTS | ★计划 $PLAN（$(python -c "import json;print(len(json.load(open('$PLAN'))))") 条臂）"
echo "         ★epochs=$EPOCHS（cs_t0043 母配方）| MAX_HOURS=$MAX_HOURS | N_TRIALS=$N_TRIALS"

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
  --struct-space \
  --enqueue-plan "$PLAN" \
  >> "$LOG" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
disown 2>/dev/null || true

sleep 5
if kill -0 "$NEW_PID" 2>/dev/null; then
  echo "[OK] no-GMC 结构线搜索已后台启动：PID $NEW_PID（setsid 独立 session，不随 SSH 断开/宿主关机而终止）"
  echo "     监控： python manu/training/status_mix_ratio_search.py --out-root $OUT_ROOT \\"
  echo "             --study-name $STUDY_NAME --reference-frame $REFERENCE_FRAME"
  echo "     日志： tail -f $LOG"
  echo "     板卡测速： bash manu/pipeline/npu/run_hm_speed_board.sh（见报告）"
else
  echo "[FAIL] 启动失败，立刻查看 $LOG"
  tail -30 "$LOG"
  exit 1
fi
