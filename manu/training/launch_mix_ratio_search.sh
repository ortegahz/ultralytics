#!/bin/bash
# ==============================================================================
# 启动 Optuna 混合比例超参搜索（**后台常驻**，与 SSH 会话完全解耦）
#
# 为什么要 setsid + nohup
# ----------------------
# 宿主机/本机会在下班时断电、SSH 窗口会关闭。若搜索挂在 SSH 会话上，
# 会话一断就被 SIGHUP 杀掉 ⇒ 搜索白跑。
#   · setsid  → 新建独立 session/进程组，不受终端关闭影响
#   · nohup   → 忽略 SIGHUP
#   · disown  → 从 shell 作业表摘除
# 三重保险，保证搜索在服务器上一直跑到截止时间。
#
# 用法（服务器 `ssh -p 32222 huangzhe@192.168.99.40`，`conda activate uav`）：
#   bash manu/training/launch_mix_ratio_search.sh              # 默认 20 小时
#   MAX_HOURS=8 bash manu/training/launch_mix_ratio_search.sh  # 自定义时长
#   N_TRIALS=200 bash manu/training/launch_mix_ratio_search.sh # 自定义 trial 数
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
OUT_ROOT="${OUT_ROOT:-runs/optuna_mix_ratio}"
GPUS="${GPUS:-0,1,2,3}"
N_TRIALS="${N_TRIALS:-4096}"
EPOCHS="${EPOCHS:-3}"
MAX_HOURS="${MAX_HOURS:-20}"
STUDY_NAME="${STUDY_NAME:-mix_ratio_search}"

cd "$PROJECT_ROOT" || { echo "[FATAL] 进不去 $PROJECT_ROOT"; exit 1; }
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav || { echo "[FATAL] conda 激活失败"; exit 1; }

mkdir -p "$OUT_ROOT/logs"

# ---- 重复启动保护 ----
PID_FILE="$OUT_ROOT/dispatcher.pid"
if [ -f "$PID_FILE" ]; then
  OLD=$(cat "$PID_FILE" 2>/dev/null || echo "")
  if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
    echo "[ABORT] 搜索已在运行（PID $OLD）。要重启请先执行 stop_mix_ratio_search.sh"
    exit 1
  fi
  echo "[WARN] 发现陈旧 PID 文件（$OLD），将覆盖"
fi

LOG="$OUT_ROOT/logs/dispatcher.log"
echo "[LAUNCH] 时间 $(date '+%F %T') | PID 写入 $PID_FILE | 日志 $LOG"
setsid nohup python -u manu/training/optuna_mix_ratio_search.py \
  --output-root "$OUT_ROOT" \
  --study-name "$STUDY_NAME" \
  --gpus "$GPUS" \
  --n-trials "$N_TRIALS" \
  --epochs "$EPOCHS" \
  --max-hours "$MAX_HOURS" \
  >> "$LOG" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
disown 2>/dev/null || true

sleep 3
if kill -0 "$NEW_PID" 2>/dev/null; then
  echo "[OK] 搜索已后台启动：PID $NEW_PID（session 独立，不随 SSH 断开而终止）"
  echo "     监控： python manu/training/status_mix_ratio_search.py"
  echo "     停止： bash manu/training/stop_mix_ratio_search.sh"
  echo "     日志： tail -f $LOG"
else
  echo "[FAIL] 启动失败，立刻查看 $LOG"
  tail -30 "$LOG"
  exit 1
fi
