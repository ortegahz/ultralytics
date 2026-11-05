#!/usr/bin/env bash
# ============================================================
# ★ P3c · 候选组合的**跨 Δ 复核**（Δ=2/4/8）
#
#   目的：P3c 在 Δ=3（24 序官方口径）上筛出候选组合 c1
#         （--output-max-step-px 64 + --replay-xcorr-min-peak 0.15
#           + --replay-roi-selfpred-max-px 8，叠加出框门 m=6）；
#         定档条件① = 「24 序**全 Δ** 复核」⇒ 本脚本补 Δ∈{2,4,8}。
#
#   ★ 每个 Δ 独立走两阶段（**先门禁、后扫描**，与 Δ=3 同纪律）：
#       阶段1 `gated`：baseline（交付档 CFG + --replay-fov-margin-px 6）复现，
#                       与该 Δ 自己的参考产物**四臂逐位一致**（严格相等，无容差）
#                       ⇒ 不一致 ⇒ 该 Δ 整轮作废（扫描器引入了额外变量）。
#       阶段2 `extra`：跑候选组合（唯一变量 = 该 Δ 下的组合旋钮串）。
#     ⚠️ 每个 Δ 必须用**自己的** REF_ARMS：参考产物随 Δ 变，拿错 Δ 的参考
#        ⇒ 门禁会拿错基线比较 ⇒ 结论做反。
#
#   ★ 三个 Δ **并行**（各 Δ 互不共享 OUT / MANIFEST / 门禁参考）。
#
#   用法（服务器）：
#     cd /tmp/pycharm_project_10ae9e2e
#     bash manu/tracking/lookahead/server_overnight/run_knob_combo_delta_sweep.sh
#     env DELTAS="2 4 8" COMBO_TAG=c1_om64_pk015_ro8 \
#         COMBO_KNOBS="--output-max-step-px=64,--replay-xcorr-min-peak=0.15,--replay-roi-selfpred-max-px=8" \
#         bash .../run_knob_combo_delta_sweep.sh
# ============================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
SELF="$REPO/manu/tracking/lookahead/server_overnight/run_knob_scan_d3.sh"
DELTAS="${DELTAS:-2 4 8}"
CACHE_CS="${CACHE_CS:-/mnt/data/siping/runs/nogmc_t3/swa/ms_cache_24seq}"
FOV_ROOT="${FOV_ROOT:-$REPO/runs/lookahead}"
OUT_ROOT="${OUT_ROOT:-$REPO/runs/lookahead/fi_nogmc24}"
THREADS="${THREADS:-4}"
JOBS="${JOBS:-1}"
EXTRA_BASE_CFG="${EXTRA_BASE_CFG:---replay-fov-margin-px 6}"
COMBO_TAG="${COMBO_TAG:-c1_om64_pk015_ro8}"
COMBO_KNOBS="${COMBO_KNOBS:---output-max-step-px=64,--replay-xcorr-min-peak=0.15,--replay-roi-selfpred-max-px=8}"

cd "$REPO" || exit 2
mkdir -p "$OUT_ROOT"
LOGDIR="$OUT_ROOT/knobscan_p3c_dlogs"
mkdir -p "$LOGDIR"

run_delta() {
  local d="$1"
  local out="$OUT_ROOT/knobscan_p3c_d$d"
  local ref="$FOV_ROOT/fovgate24nogmc_d$d/runs/cs_t0043/d$d/fov__m6/arms.json"
  local glog="$LOGDIR/d$d.log"
  mkdir -p "$out"
  # ⚠️ 块内**不能写 exit**：{...} 里的 exit 会终止整个后台子 shell，
  #    后面写 rc 文件的语句就永远不会执行 ⇒ 汇总读到「缺 rc 文件」而误判成失败。
  #    ⇒ 用 rc 变量在块内传递，块外统一落盘。
  local rc=0
  {
    echo "===== Δ=$d 开始 $(date '+%F %T') ====="
    if [ ! -f "$ref" ]; then
      echo "[FATAL] Δ=$d 参考产物不存在: $ref"; rc=3
    fi
    if [ "$rc" -eq 0 ]; then
      # ---- 阶段1：baseline 逐位门禁 ----
      env OUT="$out" DELTA="$d" CACHE_CS="$CACHE_CS" EXTRA_BASE_CFG="$EXTRA_BASE_CFG" \
          REF_ARMS="$ref" GATE_LIST="cs_t0043|$d" THREADS="$THREADS" JOBS="$JOBS" \
          bash "$SELF" gated
      local grc=$?
      echo "[GATE] Δ=$d 门禁 rc=$grc"
      if [ "$grc" -ne 0 ]; then
        echo "[FATAL] Δ=$d 门禁未过 ⇒ 该 Δ 作废（不跑扫描）"; rc=3
      fi
    fi
    if [ "$rc" -eq 0 ]; then
      # ---- 阶段2：候选组合 ----
      # COMBO_SPECS 每项 = "tag|knobs"（knobs = "flag=value[,flag=value]"）⇒ 一次可跑多个组合
      local specs=""
      if [ -n "${COMBO_SPECS:-}" ]; then
        local cs
        for cs in $COMBO_SPECS; do
          specs="${specs:+$specs }cs_t0043|$d|${cs%%|*}|${cs#*|}"
        done
      else
        specs="cs_t0043|$d|$COMBO_TAG|$COMBO_KNOBS"
      fi
      env OUT="$out" DELTA="$d" CACHE_CS="$CACHE_CS" EXTRA_BASE_CFG="$EXTRA_BASE_CFG" \
          REF_ARMS="$ref" THREADS="$THREADS" JOBS="$JOBS" \
          EXTRA_SPECS="$specs" \
          bash "$SELF" extra
      local xrc=$?
      echo "[EXTRA] Δ=$d 组合 rc=$xrc"
      rc=$xrc
    fi
  } > "$glog" 2>&1
  echo "$rc" > "$LOGDIR/d$d.rc"
  return $rc
}

for d in $DELTAS; do
  run_delta "$d" &
done
wait

fail=0
for d in $DELTAS; do
  rc="$(cat "$LOGDIR/d$d.rc" 2>/dev/null || echo 99)"
  echo "Δ=$d rc=$rc"
  [ "$rc" = "0" ] || fail=1
done
echo "跨 Δ 复核完成 fail=$fail  日志: $LOGDIR"
exit $fail
