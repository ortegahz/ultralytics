#!/bin/bash
# ==============================================================================
# run_seed_round_relay.sh —— **第二轮接力**：等第一轮收口 ⇒ 自动起「种子臂轮」
#
# 为什么需要第二轮（2026-11-06）
# --------------------------------
# 基线 `cs_t0043` 本身是**同配方 3 粒子 soup**（trial_0043/0049/0050）。
# 项目自己的结论是：「**交接口径（同超参多种子中位）是唯一能给出可信结论的办法**，
# n=2 仍脆 ⇒ **n=3 才可信**」（种子噪声 |ΔT1| 中位 0.00246 / |ΔT3| 中位 0.00300）。
#
# 而第一轮的 36 条臂里**只有 1 个配方有多种子**（两条 p1 对照臂）⇒ soup 口径无法成立。
# ⇒ 第二轮**只跑种子臂**：给母配方下的 320 输出主力结构各补 2 个种子
#    ⇒ 与第一轮的原臂凑成 **n=3** ⇒ 可以做同配方 soup。
#
# 为什么**另起一轮**而不是重启 dispatcher
# ----------------------------------------
# · dispatcher 的 `--n-trials` 是**启动参数、不可变** ⇒ 想中途加臂必须重启
#   ⇒ 重启会**杀掉正在跑的 4 条臂**（各已跑 ~2 h ⇒ 损失 ~6 卡时）；
# · 而第一轮跑完 36 臂自然收口（~29.3 h）⇒ 第二轮用一个**新的 out_root + 新 study 名**
#   ⇒ **零损失、零污染**（第一轮的 study.db 完整保留，便于对账）。
#
# 用法（服务器，conda uav）：
#     setsid nohup bash manu/training/run_seed_round_relay.sh \
#         > /mnt/data/siping/runs/npu20ms/seed_relay.log 2>&1 &
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
ROUND1_OUT="${ROUND1_OUT:-/mnt/data/siping/runs/npu20ms}"
ROUND2_OUT="${ROUND2_OUT:-/mnt/data/siping/runs/npu20ms_seed}"
ROUND1_STUDY="${ROUND1_STUDY:-mix_ratio_search_npu20ms_v2}"
ROUND2_STUDY="${ROUND2_STUDY:-mix_ratio_search_npu20ms_seed}"
POLL_SEC="${POLL_SEC:-120}"
DEADLINE_EPOCH="${DEADLINE_EPOCH:-0}"

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav || exit 1
mkdir -p "$ROUND2_OUT"

log() { echo "[SEED-RELAY $(date '+%F %T')] $*"; }

# ---- 1) 等第一轮 dispatcher 退出 ----
PID_FILE="$ROUND1_OUT/dispatcher.pid"
CUR_PID=""
[ -f "$PID_FILE" ] && CUR_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
log "第一轮 dispatcher pid = ${CUR_PID:-<无>}（输出根 $ROUND1_OUT）"
if [ -n "$CUR_PID" ]; then
  while kill -0 "$CUR_PID" 2>/dev/null; do
    if [ "$DEADLINE_EPOCH" -gt 0 ] && [ "$(date +%s)" -gt "$DEADLINE_EPOCH" ]; then
      log "超过 deadline，强制进入第二轮"; break
    fi
    sleep "$POLL_SEC"
  done
  log "第一轮已退出 ⇒ 进入第二轮"
else
  log "没有第一轮 PID ⇒ 直接进入第二轮"
fi

# ---- 2) 生成种子臂计划（从速度表读 ≤21ms 的 320 输出主力结构）----
log "生成种子臂计划"
cp -f manu/pipeline/npu/struct_speed_table.json "$ROUND2_OUT/struct_speed_table.json" 2>/dev/null
python manu/training/build_seed_arms.py \
  --speed-table "$ROUND2_OUT/struct_speed_table.json" \
  --out "$ROUND2_OUT/seed_arms.json" \
  --seed-base 3000 --seeds-per-struct 2 --limit-ms 21.0 || { log "种子臂计划生成失败"; exit 1; }
cp -f "$ROUND2_OUT/seed_arms.json" "$ROUND1_OUT/seed_arms.json" 2>/dev/null

# ---- 3) 起第二轮 dispatcher（新 out_root + 新 study 名）----
log "启动第二轮（种子臂轮）：$ROUND2_OUT / $ROUND2_STUDY"
# ★ N_TRIALS 必须等于种子臂条数（当前 = 3 个家族 × 2 种子 = 6）
N_SEED=$(python3 -c "import json;print(len(json.load(open('$ROUND2_OUT/seed_arms.json'))))")
log "种子臂条数 = $N_SEED"
OUT_ROOT="$ROUND2_OUT" \
STUDY_NAME="$ROUND2_STUDY" \
N_TRIALS="$N_SEED" \
MAX_HOURS="${MAX_HOURS:-8}" \
EPOCHS_DEFAULT=20 \
SPEED_TABLE="$ROUND2_OUT/struct_speed_table.json" \
SPEED_LIMIT_MS=21.0 \
PLAN="$ROUND2_OUT/seed_arms.json" \
  bash manu/training/launch_npu20ms_search.sh
rc=$?
log "第二轮启动返回 rc=$rc"
echo "{\"seed_round_ok\": $([ $rc -eq 0 ] && echo true || echo false), \"ts\": $(date +%s)}" \
  > "$ROUND2_OUT/seed_round_done.json"
exit $rc
