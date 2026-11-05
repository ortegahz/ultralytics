#!/usr/bin/env bash
# ============================================================
# ★ 24 序 no-GMC 收尾：等 driver 跑完 ⇒ 出全部报告 + 落档
#
#   为什么单独一个脚本：drive_nogmc24_pipeline.sh 正在运行，bash 会按字节偏移
#   增量读取脚本文件 ⇒ **绝不能编辑它**（会写坏执行流）。所以收尾另起一个进程。
#
#   为什么需要它：driver 的 Step 5 只重生成 make_fastinfer_nogmc_report.py 那一份，
#   FINAL_REPORT.md / house-style 逐序表 / 机器可核对的结果档都不在里面。
#
#   启动（服务器）：
#     cd /tmp/pycharm_project_10ae9e2e
#     setsid nohup bash manu/tracking/lookahead/server_overnight/finalize_nogmc24_report.sh \
#        > runs/lookahead/nogmc24_finalize.log 2>&1 < /dev/null &
# ============================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
PY="${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}"
OUT24="${OUT24:-$REPO/runs/lookahead/fi_nogmc24}"
ROOT8="${ROOT8:-$REPO/runs/lookahead/fi_nogmc}"
DRIVER_PID="${DRIVER_PID:-}"
TARGET="${TARGET:-cs_t0043}"
WAIT_DEADLINE="${WAIT_DEADLINE:-21600}"   # 最多等 6 小时
DELTAS="${DELTAS:-3 2 4 8}"

cd "$REPO" || exit 2
log(){ echo "[$(date '+%F %T')] [finalize] $*"; }

################################################################
# Step 1 —— 等 driver 出完数
################################################################
log "Step 1 等 driver 收尾（目标模型=$TARGET，Δ=$DELTAS）"
deadline=$(( $(date +%s) + WAIT_DEADLINE ))
while :; do
  have=0
  for d in $DELTAS; do
    [ -f "$OUT24/$TARGET/d$d/arms.json" ] && have=$((have+1))
  done
  driver_alive=0
  if [ -n "$DRIVER_PID" ] && kill -0 "$DRIVER_PID" 2>/dev/null; then driver_alive=1; fi
  if pgrep -f "drive_nogmc24_pipeline.sh" >/dev/null 2>&1; then driver_alive=1; fi
  if [ "$have" -eq "$(echo $DELTAS | wc -w)" ] && [ "$driver_alive" -eq 0 ]; then
    log "Step 1 完成：$TARGET 的 $have 个 Δ 产物齐了，driver 已退出"
    break
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "Step 1 超时（${WAIT_DEADLINE}s）：$have/$(echo $DELTAS | wc -w) 个产物，driver_alive=$driver_alive ⇒ 按现状继续"
    break
  fi
  log "Step 1 等待中：$have/$(echo $DELTAS | wc -w) 个产物，driver_alive=$driver_alive"
  sleep 60
done

################################################################
# Step 2 —— 记录脚本指纹（夜间 agent 可能改过报告脚本 ⇒ 明早要能对账）
################################################################
log "Step 2 记录报告脚本 md5（供明早对账）"
{
  md5sum manu/tracking/lookahead/make_fastinfer_nogmc_report.py \
         manu/tracking/lookahead/make_fastinfer_final_report.py \
         manu/tracking/lookahead/make_report_table.py \
         manu/tracking/lookahead/server_overnight/summarize_nogmc24_results.py 2>/dev/null
} | tee -a "$REPO/runs/lookahead/nogmc24_finalize.md5"

################################################################
# Step 3 —— 三份报告 + 机器可核对结果档
################################################################
log "Step 3 出报告"
"$PY" manu/tracking/lookahead/make_fastinfer_nogmc_report.py \
    --root "$ROOT8" --root-24 "$OUT24" \
    --gate-a "$REPO/runs/lookahead/gateA/gateA.json" \
    --byte-equal "$REPO/runs/lookahead/nogmc24_byte_equal.json" \
    --out-md "$ROOT8/report.md" --out-json "$ROOT8/report.json"
log "Step 3.1 nogmc report rc=$?"

"$PY" manu/tracking/lookahead/make_fastinfer_final_report.py \
    --root-8 "$ROOT8" --root-24 "$OUT24" \
    --gatea "$REPO/runs/lookahead/gateA/gateA.json" \
    --out-md "$ROOT8/FINAL_REPORT.md" --out-json "$ROOT8/final_report.json"
log "Step 3.2 final report rc=$?"

# house-style 逐序对照表（rc2 vs cs_t0043，同 Δ）
for d in $DELTAS; do
  if [ -f "$OUT24/rc2/d$d/arms.json" ] && [ -f "$OUT24/$TARGET/d$d/arms.json" ]; then
    "$PY" manu/tracking/lookahead/make_report_table.py \
      --runs "$OUT24/rc2/d$d/arms.json" "$OUT24/$TARGET/d$d/arms.json" \
      --labels "RC2 trial_0080ep3 / GMC 语义" "$TARGET / no-GMC 语义" \
      --title "Fast Infer × no-GMC 严格对照（24 序官方口径）" \
      --delta "$d" --model "rc2 / $TARGET" \
      --per-sequence --baseline stale \
      --out-md "$OUT24/report_house_style_d$d.md" > "$OUT24/report_house_style_d$d.log" 2>&1
    log "Step 3.3 house-style Δ=$d rc=$? → $OUT24/report_house_style_d$d.md"
  else
    log "Step 3.3 跳过 house-style Δ=$d（产物不齐）"
  fi
done

"$PY" manu/tracking/lookahead/server_overnight/summarize_nogmc24_results.py \
    --root-24 "$OUT24" \
    --out-md "$OUT24/summary.md" --out-json "$OUT24/summary.json" \
    --out-memory "$REPO/manu/memory/lookahead_nogmc24_results_20261104.md"
log "Step 3.4 结果档 rc=$?"

log "==== finalize 结束 ===="
log "24 序产物："
find "$OUT24" -name arms.json | sort | tee -a "$REPO/runs/lookahead/nogmc24_finalize.md5"
