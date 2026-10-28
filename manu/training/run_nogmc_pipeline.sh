#!/bin/bash
# ==============================================================================
# no-GMC RC2 全流程**无人值守**编排（跑在服务器上，**不依赖 SSH 会话 / 本机不断电**）
#
# 为什么要有它
# ------------
# 本机会在晚上断电（mcode 会话退出），但**服务器不会**。凡是需要跨越会话的步骤，
# 都必须落在服务器上、用 setsid nohup 常驻。本脚本把三步串成一条链：
#
#   ① 等 no-GMC 参照系测完（measure_reference_frame.py）
#   ② 启动 no-GMC 搜索（launch_nogmc_mix_ratio_search.sh，setsid nohup 常驻）
#   ③ natural 对照 trial_0000 出报告后，**自动**把它的指标写进参照系包的 rc1
#      （这样后续 trial 的 U(vs RC1) 用的就是 no-GMC 语义的新擂主，而不是 GMC 常量）
#   ④ 搜索结束 → 补采 CSV 漏行 → 生成 SUMMARY.md（明早直接读）
#
# 硬截止保护：到达 HARD_STOP 仍未结束 ⇒ SIGTERM 调度器（它有优雅处理：停派新 trial、等在跑的收尾）
#
# 用法（服务器）：
#   setsid nohup bash manu/training/run_nogmc_pipeline.sh > runs/nogmc_pipeline.out 2>&1 &
# ==============================================================================
set -uo pipefail

cd /tmp/pycharm_project_10ae9e2e || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav || exit 1
export PYTHONPATH=.

REF_DIR="${REF_DIR:-runs/nogmc_ref}"
OUT_ROOT="${OUT_ROOT:-runs/optuna_nogmc}"
STUDY_NAME="${STUDY_NAME:-mix_ratio_search_nogmc}"
PIPE_LOG="${PIPE_LOG:-runs/nogmc_pipeline.log}"
MAX_HOURS="${MAX_HOURS:-19}"
HARD_STOP="${HARD_STOP:-2026-10-28 07:00}"   # 硬截止（服务器本地时间）

mkdir -p runs
step() { echo "[$(date '+%F %T')] $*" | tee -a "$PIPE_LOG"; }

# ---------------------------------------------------------------- ① 等参照系
step "PIPELINE 启动 | 等 no-GMC 参照系（474 去 P0 三轨基线）…"
while [ ! -f "$REF_DIR/reference_frame.json" ]; do
  if ! pgrep -f "measure_reference_frame.py" >/dev/null 2>&1; then
    step "❌ 测量进程已退出但没有 $REF_DIR/reference_frame.json ⇒ 终止（详见 $REF_DIR/measure.log）"
    exit 1
  fi
  sleep 20
done
step "✅ 参照系就绪：$REF_DIR/reference_frame.json"
python - <<'PY' 2>&1 | tee -a "$PIPE_LOG"
import json, os
p = os.environ.get("REF_DIR", "runs/nogmc_ref") + "/reference_frame.json"
d = json.load(open(p, encoding="utf-8"))
b = d.get("frozen_baseline", {})
print(f"  T1 基线 F1={b.get('track1_f1')} FP={b.get('track1_fp')} | "
      f"T2 F1={b.get('track2_f1')} FP={b.get('track2_fp')} | T3 F1={b.get('track3_f1')} FP={b.get('track3_fp')}")
print(f"  rc1 = {d.get('rc1') or '（待填：natural 对照 trial_0000 落地后自动填）'}")
PY

# ---------------------------------------------------------------- ② 启动搜索
step "启动 no-GMC 搜索（MAX_HOURS=$MAX_HOURS，4 卡并行）…"
MAX_HOURS="$MAX_HOURS" bash manu/training/launch_nogmc_mix_ratio_search.sh 2>&1 | tee -a "$PIPE_LOG"
sleep 10
step "调度器 PID = $(cat "$OUT_ROOT/dispatcher.pid" 2>/dev/null)"

# ---------------------------------------------------------------- ③ 自动填 rc1
step "等 natural 对照 trial_0000 的门禁报告 → 自动填 rc1…"
RC1_DONE=0
while true; do
  GR="$OUT_ROOT/trials/trial_0000/gate/epoch3/gate_report.json"
  if [ -f "$GR" ]; then
    step "找到 natural 对照门禁报告，填 rc1："
    python manu/training/measure_reference_frame.py --out-dir "$REF_DIR" --fill-rc1-from "$GR" 2>&1 | tee -a "$PIPE_LOG"
    RC1_DONE=1
    break
  fi
  if ! pgrep -f "optuna_mix_ratio_search.py" >/dev/null 2>&1; then
    step "⚠️ 调度器已退出，natural 对照未产出门禁报告 ⇒ rc1 留空"
    break
  fi
  sleep 60
done
step "rc1 填充：$( [ "$RC1_DONE" = 1 ] && echo 完成 || echo 未完成 )"

# ---------------------------------------------------------------- ④ 等搜索结束
step "等搜索收尾…"
while pgrep -f "optuna_mix_ratio_search.py" >/dev/null 2>&1; do
  NOW_S=$(date '+%s')
  STOP_S=$(date -d "$HARD_STOP" '+%s' 2>/dev/null || echo 0)
  if [ "$STOP_S" -gt 0 ] && [ "$NOW_S" -gt "$STOP_S" ]; then
    step "⏰ 到达硬截止 $HARD_STOP ⇒ SIGTERM 调度器（优雅：停派新 trial，等在跑的收尾）"
    kill -TERM "$(cat "$OUT_ROOT/dispatcher.pid" 2>/dev/null)" 2>/dev/null
    sleep 180
    break
  fi
  sleep 60
done
step "调度器已退出"

# ---------------------------------------------------------------- ⑤ 收尾汇总
step "补采 CSV 漏行（调度器回收时可能漏写）…"
python manu/training/backfill_summary_rows.py \
  --out-root "$OUT_ROOT" \
  --storage "sqlite:///$PWD/$OUT_ROOT/study.db" \
  --study-name "$STUDY_NAME" --apply 2>&1 | tee -a "$PIPE_LOG"

step "生成汇总…"
python manu/training/summarize_nogmc_search.py \
  --out-root "$OUT_ROOT" --ref "$REF_DIR/reference_frame.json" --top 15 2>&1 | tee -a "$PIPE_LOG"

step "✅ PIPELINE 全部完成 —— 汇总见 $OUT_ROOT/SUMMARY.md"
