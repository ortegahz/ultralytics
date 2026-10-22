#!/bin/bash
# ==============================================================================
# 停止 Optuna 混合比例超参搜索（**可恢复**：重跑 launch 即从断点继续）
#
# 默认（优雅）：给调度器发 SIGTERM ⇒ 它停止派新 trial，并等待在跑的 trial 收尾
#   · 已完成的 trial 全部保存在 SQLite study 里（COMPLETE / PRUNED）
#   · 下次 launch 会自动跳过它们，从断点继续
# --force：立刻杀掉整个进程组（调度器 + 所有 worker），trial 标为 FAIL，
#          下次 launch 启动时自愈（陈旧 RUNNING → FAIL）
#
# 用法：
#   bash manu/training/stop_mix_ratio_search.sh            # 优雅停止
#   bash manu/training/stop_mix_ratio_search.sh --force    # 立即强杀
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
OUT_ROOT="${OUT_ROOT:-runs/optuna_mix_ratio}"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

cd "$PROJECT_ROOT" || { echo "[FATAL] 进不去 $PROJECT_ROOT"; exit 1; }
PID_FILE="$OUT_ROOT/dispatcher.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "[INFO] 没有 PID 文件（$PID_FILE）—— 搜索本来就没在跑"
  exit 0
fi
PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
if [ -z "$PID" ]; then
  echo "[INFO] PID 文件为空"; exit 0
fi

if ! kill -0 "$PID" 2>/dev/null; then
  echo "[INFO] 调度器 $PID 已不在运行（陈旧 PID 文件）"
  rm -f "$PID_FILE"
  exit 0
fi

if [ "$FORCE" = "1" ]; then
  echo "[FORCE] 杀掉整个进程组（调度器 + worker）…"
  kill -TERM -"$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null
  sleep 3
  kill -KILL -"$PID" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "[OK] 已强杀。重新启动即可从断点继续：bash manu/training/launch_mix_ratio_search.sh"
else
  echo "[STOP] 给调度器 $PID 发 SIGTERM（停止派新 trial，等待在跑的收尾）…"
  kill -TERM "$PID"
  echo "     等待在跑的 trial 收尾（最多 30 分钟，可用 --force 立即强杀）…"
  for i in $(seq 1 180); do
    if ! kill -0 "$PID" 2>/dev/null; then
      echo "[OK] 调度器已退出（用时约 ${i}0 秒）"
      rm -f "$PID_FILE"
      echo "     恢复：bash manu/training/launch_mix_ratio_search.sh（会跳过已完成的 trial）"
      exit 0
    fi
    sleep 10
  done
  echo "[WARN] 30 分钟仍未退出；如需强杀：bash manu/training/stop_mix_ratio_search.sh --force"
  exit 1
fi
