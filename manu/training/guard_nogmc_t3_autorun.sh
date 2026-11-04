#!/bin/bash
# ==============================================================================
# **autorun 守护**（2026-11-03 17:40 新增）
#
# 为什么需要它
# ============
# host 主机 18:30 关机 ⇒ 本机侧**再无人工干预**。
# 现状：调度器有看门狗（`run_nogmc_t3_pipeline.sh` 的 `WATCH_ONLY=1` 模式），
#       但 **autorun（自主收尾）本身无人守护** —— 它一旦崩了，04:10 之后
#       服务器上就再没有进程去产出 HANDOVER / 冻结 RC / 前沿地图 / 轮次报告。
#       ⇒ 本脚本是**服务器侧**独立会话（setsid + PPID=1）的兜底。
#
# 它做什么（每 CHECK_INTERVAL 秒一次）
# ======================================
#   1. `AUTORUN_DONE.json` 已存在 ⇒ 已正常完成 ⇒ 退出（不打扰）
#   2. autorun 进程存活 ⇒ 继续等（它内部有 FINAL_DEADLINE / GATE_DEADLINE 兜底）
#   3. autorun 进程消失、但还没完成 ⇒ 用**完全相同的环境变量**把它拉起来
#   4. 变体门禁进程仍在跑 ⇒ **不重启**（否则会起第二个 gate 并行写同一目录 ⇒ 竞态）
#   5. 超过 FINAL_DEADLINE ⇒ 不再重启（磁盘上已有「保险层」交付件，③ 已落盘）
#
# 约束
# ====
#   · **不改正在运行的 `run_nogmc_t3_autorun.sh`**（铁律：运行中的 bash 脚本禁止修改）
#   · 最多重启 `RESTART_MAX` 次；两次重启至少隔 `MIN_INTERVAL` 秒（防崩溃-重启循环）
#   · 零本机依赖：路径、产物、进程全在服务器本地
#
# 用法（服务器，setsid 让它脱离 SSH 会话）
#   cd /tmp/pycharm_project_10ae9e2e
#   setsid nohup bash manu/training/guard_nogmc_t3_autorun.sh \
#     >> /mnt/data/siping/runs/nogmc_t3/logs/autorun_guard.log 2>&1 < /dev/null &
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
RUNS_BASE="${RUNS_BASE:-/mnt/data/siping/runs}"
OUT_ROOT="${OUT_ROOT:-$RUNS_BASE/nogmc_t3}"
LOG="${LOG:-$OUT_ROOT/logs/autorun_guard.log}"

# ⚠️ 与原实例（PID 914121）**完全一致**的环境变量
#    （取自 /proc/914121/environ，2026-11-03 17:35 实测）
TRAIN_STOP="${TRAIN_STOP:-2026-11-04 04:50}"
GATE_DEADLINE="${GATE_DEADLINE:-2026-11-04 06:45}"
FINAL_DEADLINE="${FINAL_DEADLINE:-2026-11-04 07:45}"
WAIT_HARD_CAP_MIN="${WAIT_HARD_CAP_MIN:-1200}"

RESTART_MAX="${RESTART_MAX:-2}"        # 最多重启几次
MIN_INTERVAL="${MIN_INTERVAL:-900}"    # 两次重启至少隔多久（秒）
CHECK_INTERVAL="${CHECK_INTERVAL:-180}" # 检查间隔（秒）
DRY_RUN="${DRY_RUN:-0}"                # 1 = 只判定不动作（用于上线前自检）

cd "$PROJECT_ROOT" || exit 1
mkdir -p "$OUT_ROOT/logs"
DONE_FLAG="$OUT_ROOT/AUTORUN_DONE.json"
STATE_FILE="$OUT_ROOT/logs/autorun_guard.state"

say(){
  local m
  m="[$(date '+%F %T')] [guard] $*"
  printf '%s\n' "$m" >> "$LOG"
  return 0
}

# ---- 进程探测 -------------------------------------------------------------
# ⚠️ 用 `^bash manu/training/...` 锚定命令行开头，避免匹配到别的东西；
#    `pgrep` 从不报告自己，所以不会自匹配。
find_autorun(){
  pgrep -f "^bash manu/training/run_nogmc_t3_autorun.sh" 2>/dev/null | head -1
}
find_gate(){
  pgrep -f "^bash manu/training/gate_nogmc_t3_variants.sh" 2>/dev/null | head -1
}

# ---- 状态持久化（重启次数 / 上次重启时刻）--------------------------------
restarts=0
last_restart=0
if [ -f "$STATE_FILE" ]; then
  # shellcheck disable=SC1090
  . "$STATE_FILE"
fi
save_state(){
  printf 'restarts=%s\nlast_restart=%s\n' "$restarts" "$last_restart" > "$STATE_FILE"
}

say "════ autorun 守护启动 ════"
say "  产物根 $OUT_ROOT ｜ TRAIN_STOP=$TRAIN_STOP ｜ GATE_DEADLINE=$GATE_DEADLINE ｜ FINAL_DEADLINE=$FINAL_DEADLINE"
say "  检查间隔 ${CHECK_INTERVAL}s ｜ 最多重启 ${RESTART_MAX} 次 ｜ 最小重启间隔 ${MIN_INTERVAL}s ｜ DRY_RUN=$DRY_RUN"

while :; do
  now=$(date +%s)

  # ① 已完成 ⇒ 退出
  if [ -f "$DONE_FLAG" ]; then
    say "✅ 检测到 AUTORUN_DONE.json ⇒ 收尾已完成，守护退出"
    exit 0
  fi

  autorun_pid="$(find_autorun)"
  if [ -n "$autorun_pid" ]; then
    # ② autorun 存活 ⇒ 继续等
    if [ "$restarts" -gt 0 ] && [ "$last_restart" -gt 0 ]; then
      save_state
    fi
    say "· autorun 存活（PID=$autorun_pid）⇒ 等待中"
    sleep "$CHECK_INTERVAL"
    continue
  fi

  # ③ autorun 没了但还没完成
  gate_pid="$(find_gate)"
  if [ -n "$gate_pid" ]; then
    say "· autorun 已消失，但变体门禁仍在跑（PID=$gate_pid）⇒ 不重启（避免两个 gate 并行写同一目录）"
    sleep "$CHECK_INTERVAL"
    continue
  fi

  # ⑤ 超过 finalize 硬截止 ⇒ 不再重启
  final_s="$(date -d "$FINAL_DEADLINE" '+%s' 2>/dev/null || echo 0)"
  if [ "$final_s" -gt 0 ] && [ "$now" -ge "$final_s" ]; then
    say "⚠️ 已过 FINAL_DEADLINE（$FINAL_DEADLINE）⇒ 不再重启 autorun（磁盘上保留保险层交付件）"
    exit 0
  fi

  # 重启配额 / 间隔
  if [ "$restarts" -ge "$RESTART_MAX" ]; then
    say "⚠️ 已重启 $restarts 次（上限 $RESTART_MAX）⇒ 不再重启"
    exit 0
  fi
  if [ "$last_restart" -gt 0 ] && [ $((now - last_restart)) -lt "$MIN_INTERVAL" ]; then
    say "· 距上次重启不足 ${MIN_INTERVAL}s ⇒ 本轮不重启"
    sleep "$CHECK_INTERVAL"
    continue
  fi

  if [ "$DRY_RUN" = "1" ]; then
    say "[DRY_RUN] 判定为「需要重启 autorun」（would restart with TRAIN_STOP=$TRAIN_STOP GATE_DEADLINE=$GATE_DEADLINE FINAL_DEADLINE=$FINAL_DEADLINE）"
    sleep "$CHECK_INTERVAL"
    continue
  fi

  # 真正重启
  restarts=$((restarts + 1))
  last_restart="$now"
  save_state
  say "🔁 第 $restarts/$RESTART_MAX 次重启 autorun（env 与原实例一致）…"
  (
    cd "$PROJECT_ROOT" || exit 1
    export TRAIN_STOP GATE_DEADLINE FINAL_DEADLINE WAIT_HARD_CAP_MIN
    setsid nohup bash manu/training/run_nogmc_t3_autorun.sh \
      >> "$OUT_ROOT/logs/autorun.log" 2>&1 < /dev/null &
  )
  sleep 5
  new_pid="$(find_autorun)"
  if [ -n "$new_pid" ]; then
    say "✅ autorun 已重新拉起（PID=$new_pid）"
  else
    say "⚠️ 重启后 5s 内没看到 autorun 进程 ⇒ 下一轮再试"
  fi
  sleep "$CHECK_INTERVAL"
done
