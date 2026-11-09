#!/bin/bash
# ==============================================================================
# final_harvest.sh —— **最终收割**：两轮跑完 ⇒ 建 soup ⇒ 门禁 ⇒ 出总表
#
# 流程
#   1) 等第一轮 dispatcher 退出（输出 ACCEPTANCE.md：单臂总表）
#   2) 等第二轮（种子臂轮）dispatcher 退出  ← ★ 竞态处理见下
#   3) 生成家族定义（build_struct_families.py）⇒ 建 soup（n=3 权重平均）
#   4) 对每个 soup 跑三轨门禁
#   5) 出**最终总表**（单臂 + soup，按 T3 排序）
#
# ★ 竞态（2026-11-06 修）：接力脚本与本脚本**都在等第一轮退出**
#   · 接力脚本：轮询第一轮 pid ⇒ 生成种子臂 ⇒ 起第二轮 ⇒ 写 `$R2/dispatcher.pid`
#   · 本脚本：轮询第一轮 pid ⇒ 跑 acceptance ⇒ 去看 `$R2/dispatcher.pid`
#   ⇒ 本脚本**比接力脚本先醒**是常态 ⇒ 直接「没有 pid 文件就跳过」会让 soup **拿不到 n=3**。
#   ⇒ 改成：**等 pid 文件出现**（默认最多等 2 h）再等它退出。
#   ⚠️ 注意：`$R2/seed_round_done.json` 只表示「第二轮**已启动**」（launcher 是后台起的），
#      **不是**「第二轮已跑完」⇒ 不能拿它当完成信号。
#
# ⚠️ 板卡 2026-11-06 18:30 断电 ⇒ 速度**查表**（结构决定速度，与权重无关；
#    已实证：探针 17.973 ms vs 真实训练权重 18.028 ms，差 0.055 ms）。
#
# 用法（服务器，conda uav）：
#     setsid nohup bash manu/training/final_harvest.sh \
#         > /mnt/data/siping/runs/npu20ms/final_harvest.log 2>&1 &
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
R1="${R1:-/mnt/data/siping/runs/npu20ms}"
R2="${R2:-/mnt/data/siping/runs/npu20ms_seed}"
STUDY1="${STUDY1:-mix_ratio_search_npu20ms_v2}"
STUDY2="${STUDY2:-mix_ratio_search_npu20ms_seed}"
SPEED_TABLE="${SPEED_TABLE:-$PROJECT_ROOT/manu/pipeline/npu/struct_speed_table.json}"
POLL_SEC="${POLL_SEC:-120}"
DEADLINE_EPOCH="${DEADLINE_EPOCH:-0}"
GATE_GPU="${GATE_GPU:-0}"
REQUIRE_N="${REQUIRE_N:-3}"                 # 每个家族至少几个成员（3 = 同配方 3 粒子）
R2_PID_WAIT_SEC="${R2_PID_WAIT_SEC:-7200}" # 等第二轮 pid 文件出现的上限（秒）
FAMILIES_JSON="${FAMILIES_JSON:-$R1/struct_families.json}"
SOUPS_DIR="${SOUPS_DIR:-$R1/soups}"          # ★ 复数：与 build_struct_soups / acceptance 的默认路径一致

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav || exit 1
mkdir -p "$R1/logs"

log() { echo "[HARVEST $(date '+%F %T')] $*"; }

# ==============================================================================
# 等一个 dispatcher 收口
#   $1 = 输出根 | $2 = 标签 | $3 = 等 pid 文件出现的秒数（0 = 不等）
# ==============================================================================
wait_dispatcher() {
  local out="$1" tag="$2" pid_wait="${3:-0}"
  local pf="$out/dispatcher.pid"
  local pid=""

  # ---- ★ 先等 pid 文件出现 ----
  if [ "$pid_wait" -gt 0 ]; then
    local t0=$(date +%s)
    while [ ! -s "$pf" ]; do
      if [ "$(date +%s)" -gt "$t0" ] && [ $(( $(date +%s) - t0 )) -gt "$pid_wait" ]; then
        log "$tag pid 文件 $pid_wait s 内没出现 ⇒ 不再等（可能接力没起来）"
        return 1
      fi
      sleep 30
    done
    log "$tag pid 文件已出现：$pf"
  fi

  [ -f "$pf" ] && pid=$(cat "$pf" 2>/dev/null || echo "")
  log "等 $tag 收口（pid=${pid:-<无>}，输出根 $out）"
  if [ -z "$pid" ]; then log "$tag 没有 pid 文件 ⇒ 跳过等待"; return 1; fi
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$DEADLINE_EPOCH" -gt 0 ] && [ "$(date +%s)" -gt "$DEADLINE_EPOCH" ]; then
      log "超过 deadline ⇒ 强制进入下一步"; break
    fi
    sleep "$POLL_SEC"
  done
  log "$tag 已退出"
  return 0
}

# ---- 1) 等第一轮 ----
wait_dispatcher "$R1" "第一轮" 0
# 第一轮的单臂总表（看门狗也会做一份，这里补一次确保存在）
python manu/training/npu20ms_acceptance.py \
  --root "$R1" --speed-table "$SPEED_TABLE" --study-name "$STUDY1" --limit-ms 21.0 \
  > "$R1/logs/acceptance_round1.log" 2>&1
log "第一轮单臂总表 -> $R1/ACCEPTANCE.md"

# ---- 2) 等第二轮（★ 等 pid 文件出现，接住接力脚本）----
wait_dispatcher "$R2" "第二轮" "$R2_PID_WAIT_SEC"

# ---- 2b) 第二轮单臂总表（★ 第二轮是新 out_root ⇒ 第一轮的验收脚本看不到它的臂）----
mkdir -p "$R2/logs"
python manu/training/npu20ms_acceptance.py \
  --root "$R2" --speed-table "$SPEED_TABLE" --study-name "$STUDY2" --limit-ms 21.0 \
  > "$R2/logs/acceptance_round2.log" 2>&1
log "第二轮单臂总表 -> $R2/ACCEPTANCE.md"

# ==============================================================================
# ---- 3) 生成家族定义 ⇒ 建 soup ----
# 家族定义必须**运行时生成**：第二轮的成员只有跑完才存在。
# 配对规则 = params 逐字段相等（种子在 arm['seed']，不在 params 里）⇒ 天然「同配方只差种子」。
# ==============================================================================
log "生成家族定义（第一轮母臂 + 第二轮种子臂，按 params 配对，要求 ≥ $REQUIRE_N 个成员）"
python manu/training/build_struct_families.py \
  --round1-root "$R1" --round1-study "$STUDY1" \
  --round2-root "$R2" --round2-study "$STUDY2" \
  --out "$FAMILIES_JSON" --require-n "$REQUIRE_N" \
  2>&1 | tee "$R1/logs/build_families.log"
FAM_RC=${PIPESTATUS[0]}
log "家族定义生成 rc=$FAM_RC -> $FAMILIES_JSON"

# ★ 兜底：n=3 凑不齐（种子臂有掉链子）⇒ 退到 n=2。
#   n=2 虽脆（项目结论：n=2 仍不可信），但**门禁会仲裁** ⇒ 有总比没有强。
if [ "$FAM_RC" -ne 0 ] && [ "$REQUIRE_N" -gt 2 ]; then
  log "⚠️ n=$REQUIRE_N 没凑出家族 ⇒ 退到 n=2 重试"
  python manu/training/build_struct_families.py \
    --round1-root "$R1" --round1-study "$STUDY1" \
    --round2-root "$R2" --round2-study "$STUDY2" \
    --out "$FAMILIES_JSON" --require-n 2 \
    2>&1 | tee "$R1/logs/build_families_n2.log"
  FAM_RC=${PIPESTATUS[0]}
  log "n=2 兜底 rc=$FAM_RC"
fi

if [ "$FAM_RC" -eq 0 ] && [ -s "$FAMILIES_JSON" ]; then
  mkdir -p "$SOUPS_DIR"
  log "建 soup（n=3 权重平均；结构自描述逐字段校验，不一致直接拒绝）"
  python manu/training/build_struct_soups.py \
    --out-root "$R1" --study-name "$STUDY1" \
    --families-json "$FAMILIES_JSON" \
    --out-dir "$SOUPS_DIR" \
    --speed-table "$SPEED_TABLE" \
    2>&1 | tee "$R1/logs/build_soup.log"
  log "建 soup rc=${PIPESTATUS[0]}"
else
  log "没有可用家族定义 ⇒ 跳过建 soup（单臂总表仍会出）"
fi

# ---- 4) 对 soup 跑门禁 ----
if ls "$SOUPS_DIR"/soup_*.pt >/dev/null 2>&1; then
  log "对 soup 跑三轨门禁（GPU $GATE_GPU；门禁参数与训练臂逐位一致）"
  OUT_ROOT="$R1" \
  SOUPS_DIR="$SOUPS_DIR" \
  CANDIDATES="$SOUPS_DIR/soup_candidates.json" \
  GPUS=1 \
  DEVICE="$GATE_GPU" \
  INFER_GPU="$GATE_GPU" \
  CACHE_GPUS="$GATE_GPU" \
    bash manu/training/gate_struct_soups.sh \
    2>&1 | tee "$R1/logs/gate_soups.log"
  log "soup 门禁 rc=${PIPESTATUS[0]}"
else
  log "没有 soup 产物 ⇒ 跳过 soup 门禁"
fi

# ---- 5) 出最终总表（单臂 + soup）----
log "出最终总表（单臂 + soup）"
python manu/training/npu20ms_acceptance.py \
  --root "$R1" --speed-table "$SPEED_TABLE" --study-name "$STUDY1" --limit-ms 21.0 \
  2>&1 | tee "$R1/logs/acceptance_final.log"

log "完成。产物：$R1/ACCEPTANCE.md + $R1/acceptance.json"
echo "{\"harvest_done\": true, \"ts\": $(date +%s)}" > "$R1/harvest_done.json"
