#!/bin/bash
# ==============================================================================
# 优雅切换：旧调度器收尾 → 用**新代码 + 5-epoch 协议**重启搜索
#
# 为什么需要它（2026-10-21 决策）
# ----------------------------
# 用户裁决：后续 trial 改跑 **5 epoch**（cosine 周期仍 5 ⇒ 与 RC1 同 LR 轨迹），
# 目的是**复现 RC1 的完整 5-epoch 曲线**（Ep3 与 Ep5 在同一次运行里），
# 并且让搜索直接测量"最终 5-epoch 交付模型"所处的那个区间。
#
# 为什么用"优雅切换"而不是直接杀
# ------------------------------
# 直接杀会丢掉在跑的 4 个 trial（约 1.5 GPU-小时）；优雅切换让它们跑完再换，
# **GPU 全程不闲置**。切换全程在服务器上，**与 SSH 会话解耦** ⇒ 本机断电无影响。
#
# 用法（服务器）：
#   setsid nohup bash manu/training/handoff_to_5epoch.sh >> runs/optuna_mix_ratio/logs/handoff.log 2>&1 &
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
OUT_ROOT="${OUT_ROOT:-runs/optuna_mix_ratio}"
GPUS="${GPUS:-0,1,2,3}"
N_TRIALS="${N_TRIALS:-4096}"
NEW_EPOCHS="${NEW_EPOCHS:-5}"
# 剩余预算小时数（原始 20h 预算的剩余部分，避免重启把截止时间推后）
NEW_MAX_HOURS="${NEW_MAX_HOURS:-17}"

cd "$PROJECT_ROOT" || { echo "[FATAL] 进不去 $PROJECT_ROOT"; exit 1; }
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav || { echo "[FATAL] conda 激活失败"; exit 1; }

PID_FILE="$OUT_ROOT/dispatcher.pid"
LOG="$OUT_ROOT/logs/dispatcher.log"

OLD=""
[ -f "$PID_FILE" ] && OLD=$(cat "$PID_FILE" 2>/dev/null || echo "")
if [ -z "$OLD" ] || ! kill -0 "$OLD" 2>/dev/null; then
  echo "[HANDOFF] 没有在跑的调度器（PID=${OLD:-无}）⇒ 直接按新协议启动"
else
  echo "[HANDOFF] 旧调度器 PID=$OLD ⇒ 发 SIGTERM（停止派新单，等在跑的收尾）"
  kill -TERM "$OLD" 2>/dev/null
  # 等旧调度器自己退出（它会等在跑的 trial 收尾，最多 1800s）
  for _ in $(seq 1 150); do
    kill -0 "$OLD" 2>/dev/null || break
    sleep 20
  done
  if kill -0 "$OLD" 2>/dev/null; then
    echo "[HANDOFF] 旧调度器 50 分钟仍未退出 ⇒ **放弃切换，保持原状**（不杀、不重启）"
    exit 1
  fi
  echo "[HANDOFF] 旧调度器已退出"
fi

# ---- 运行时配置：后续 trial 训练几个 epoch（cosine 周期由 --sched-epochs 保持 5）----
# ⚠️ 必须与命令行 --epochs 一致（运行时配置优先级更高）。
cat > "$OUT_ROOT/runtime_config.json" <<EOF
{
  "train_epochs": $NEW_EPOCHS
}
EOF
echo "[HANDOFF] 已写 $OUT_ROOT/runtime_config.json → {\"train_epochs\": $NEW_EPOCHS}"

# ---- 用新代码拉起（断点续跑：已完成的 trial 全部跳过）----
setsid nohup python -u manu/training/optuna_mix_ratio_search.py \
  --output-root "$OUT_ROOT" \
  --study-name mix_ratio_search \
  --gpus "$GPUS" \
  --n-trials "$N_TRIALS" \
  --epochs "$NEW_EPOCHS" \
  --max-hours "$NEW_MAX_HOURS" \
  >> "$LOG" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
disown 2>/dev/null || true

sleep 6
if kill -0 "$NEW_PID" 2>/dev/null; then
  echo "[HANDOFF] ✅ 新调度器已启动 PID=$NEW_PID（${NEW_EPOCHS}-epoch 协议，预算 ${NEW_MAX_HOURS}h）"
  echo "[HANDOFF]    监控： python manu/training/status_mix_ratio_search.py"
else
  echo "[HANDOFF] ❌ 新调度器启动失败，日志尾部："
  tail -30 "$LOG"
  exit 1
fi
