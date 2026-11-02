#!/bin/bash
# ==============================================================================
# 周末 no-GMC 复现实验 · **无人值守**全流程编排
# （跑在服务器上，**不依赖 SSH 会话 / 本机不断电** —— 本机 18:30 关机，服务器不关）
#
# 为什么必须有这个
# ----------------
# ① 本机（host）今天 18:30 关机 ⇒ 任何绑在本机会话里的进程都会死；
#    服务器 192.168.99.40 整周末在线 ⇒ **所有编排必须落在服务器、且彻底 setsid 脱离**。
# ② 上一轮（10-27/28）的产物只存在服务器 /tmp ⇒ 10-29 被清掉，99 行 CSV / SUMMARY /
#    全部权重一次性丢失。本轮**产物根改到 /mnt/data**（持久盘）+ 收尾再备份一份。
# ③ 搜索可能因为意外崩溃而中断 ⇒ 这里带**看门狗**：调度器不在了就自动断点续跑重启。
#
# 流程
# ----
#   ① 重建 no-GMC 参照系（474 去 P0 三轨基线 + 门禁阈值，ΔFP_T2 上限 60）
#   ② 启动扩展搜索（确定性对照臂优先 → 再 TPE）
#   ③ 看门狗：调度器意外退出 ⇒ 自动重启（study.db 断点续跑 + 自愈陈旧 RUNNING）
#   ④ natural 对照落地 ⇒ 自动把它的指标填进参照系包的 rc1
#   ⑤ 搜索到点 ⇒ 补采 CSV → 重判 → 出 SUMMARY
#   ⑥ 复现阶段：取配置级多种子榜前 N 名，各补到 3 个种子
#   ⑦ 最终重判 + 出 SUMMARY + 备份
#
# 用法（服务器）：
#   setsid nohup bash manu/training/run_nogmc_weekend_pipeline.sh \
#       > /mnt/data/siping/runs/nogmc_weekend_pipeline.out 2>&1 &
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
RUNS_BASE="${RUNS_BASE:-/mnt/data/siping/runs}"
OUT_ROOT="${OUT_ROOT:-$RUNS_BASE/nogmc_weekend}"
REF_DIR="${REF_DIR:-$RUNS_BASE/nogmc_ref_weekend}"
STUDY_NAME="${STUDY_NAME:-mix_ratio_search_nogmc_weekend}"
PIPE_LOG="${PIPE_LOG:-$RUNS_BASE/nogmc_weekend_pipeline.log}"
PLAN="${PLAN:-manu/training/nogmc_weekend_plan.json}"

# ---- 绝对时间锚点（服务器本地时间）----
# 搜索停止时刻（留出复现 + 收尾的时间）；到点后不再派新 trial
#   2026-10-30 是周五；周一 09:00 = 2026-11-02 09:00
#   搜索跑到 Sun 15:00（≈44 h）→ 复现 6 h → 收尾 1 h ⇒ 距周一 09:00 还有 ~11 h 缓冲
SEARCH_STOP="${SEARCH_STOP:-2026-11-01 15:00}"
# 复现阶段最长时长（小时）
PHASE3_MAX_HOURS="${PHASE3_MAX_HOURS:-6}"
# **终极硬停**：过了这个点，一切未收尾的进程一律 SIGTERM，然后强行出汇总
HARD_STOP="${HARD_STOP:-2026-11-02 06:00}"

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav || exit 1
export PYTHONPATH=.

mkdir -p "$RUNS_BASE" "$OUT_ROOT/logs"
step() { echo "[$(date '+%F %T')] $*" | tee -a "$PIPE_LOG"; }

STOP_S=$(date -d "$SEARCH_STOP" '+%s' 2>/dev/null || echo 0)
HARD_S=$(date -d "$HARD_STOP" '+%s' 2>/dev/null || echo 0)
step "PIPELINE 启动 | 输出根 $OUT_ROOT | 参照系 $REF_DIR"
step "  搜索停止 $SEARCH_STOP | 复现阶段 ≤${PHASE3_MAX_HOURS}h | 硬停 $HARD_STOP"

B=/mnt/data/siping/datasets/manu
LQ_DATA="$B/longquanshan_ft_gmc_median_nogmc/data.yaml"
UAV_DATA="$B/uav_gmc_median_nogmc/data.yaml"
REF="$REF_DIR/reference_frame.json"

# ---------------------------------------------------------------- ① 参照系
step "① 检查 no-GMC 参照系 …"
if [ ! -f "$REF" ]; then
  step "   参照系不存在 ⇒ 实测（474 去 P0 在 no-GMC 语义下的三轨基线），约 20~40 min"
  mkdir -p "$REF_DIR"
  python manu/training/measure_reference_frame.py \
    --uav-data "$UAV_DATA" \
    --ft-root  "$B/longquanshan_ft_gmc_median_nogmc" \
    --weights  runs/optuna_p0_nas/trial_0474/weights/best.pt \
    --out-dir  "$REF_DIR" --label nogmc --gpus 0,1,2,3 \
    --track2-fp-max-increment 60 2>&1 | tee -a "$PIPE_LOG"
  RC=${PIPESTATUS[0]}
  step "   measure_reference_frame rc=$RC"
  if [ ! -f "$REF" ]; then
    step "❌ 参照系仍未生成 ⇒ 终止（详见 $PIPE_LOG / $REF_DIR/measure.log）"
    exit 1
  fi
fi
python - "$REF" <<'PY' 2>&1 | tee -a "$PIPE_LOG"
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
b = d.get("frozen_baseline", {})
g = d.get("gate", {})
print(f"  T1 基线 F1={b.get('track1_f1')} FP={b.get('track1_fp')} | "
      f"T2 基线 F1={b.get('track2_f1')} FP={b.get('track2_fp')} | "
      f"T3 基线 F1={b.get('track3_f1')} FP={b.get('track3_fp')}")
print(f"  门禁：T2F1≥{g.get('track2_f1_min')} ΔFP_T2≤{g.get('track2_fp_max_increment')} "
      f"T1Δ≥{g.get('track1_min_delta')}")
print(f"  rc1 = {d.get('rc1') or '（待填）'}")
PY

# ---------------------------------------------------------------- ② 启动搜索
step "② 启动扩展搜索（确定性对照臂 → TPE）…"
LAUNCH_H=$(( (STOP_S - $(date '+%s')) / 3600 ))
[ "$LAUNCH_H" -lt 1 ] && LAUNCH_H=1
step "   搜索预算 ${LAUNCH_H} h（到 $SEARCH_STOP）"
OUT_ROOT="$OUT_ROOT" REF_DIR="$REF_DIR" STUDY_NAME="$STUDY_NAME" PLAN="$PLAN" \
  MAX_HOURS="$LAUNCH_H" bash manu/training/launch_nogmc_weekend_search.sh 2>&1 | tee -a "$PIPE_LOG"
sleep 15
step "   调度器 PID = $(cat "$OUT_ROOT/dispatcher.pid" 2>/dev/null)"

# ---------------------------------------------------------------- ③ 看门狗
step "③ 看门狗启动（调度器意外退出 ⇒ 自动断点续跑重启）"
LAUNCHES=0
while :; do
  NOW_S=$(date '+%s')
  if [ "$STOP_S" -gt 0 ] && [ "$NOW_S" -ge "$STOP_S" ]; then
    step "   到达搜索停止时刻 $SEARCH_STOP ⇒ 进入收尾"
    break
  fi
  if [ "$HARD_S" -gt 0 ] && [ "$NOW_S" -ge "$HARD_S" ]; then
    step "⏰ 到达硬停 $HARD_STOP ⇒ 停掉调度器并强制收尾"
    kill -TERM "$(cat "$OUT_ROOT/dispatcher.pid" 2>/dev/null)" 2>/dev/null
    sleep 60
    break
  fi
  PID=$(cat "$OUT_ROOT/dispatcher.pid" 2>/dev/null || echo "")
  if [ -z "$PID" ] || ! kill -0 "$PID" 2>/dev/null; then
    REMAIN_H=$(( (STOP_S - NOW_S) / 3600 ))
    [ "$REMAIN_H" -lt 1 ] && REMAIN_H=1
    LAUNCHES=$((LAUNCHES+1))
    step "⚠️ 调度器不在了（PID='$PID'）⇒ 第 $LAUNCHES 次自动重启，剩余预算 ${REMAIN_H}h"
    OUT_ROOT="$OUT_ROOT" REF_DIR="$REF_DIR" STUDY_NAME="$STUDY_NAME" PLAN="$PLAN" \
      MAX_HOURS="$REMAIN_H" bash manu/training/launch_nogmc_weekend_search.sh 2>&1 | tee -a "$PIPE_LOG"
    sleep 30
  fi
  sleep 60
done
step "   看门狗结束（共自动重启 $LAUNCHES 次）"

# ---------------------------------------------------------------- ④ rc1 自动填
step "④ 找 natural 对照 trial 的门禁报告 → 填 rc1 …"
RC1_DONE=0
for _try in $(seq 1 120); do
  GR=$(python - "$OUT_ROOT" <<'PY' 2>/dev/null
import json, sys, glob, os
root = sys.argv[1]
for res in sorted(glob.glob(os.path.join(root, "trials", "trial_*", "trial_result.json"))):
    try:
        d = json.load(open(res, encoding="utf-8"))
    except Exception:
        continue
    if d.get("mix_spec") == "natural" and d.get("gate_dir"):
        gp = os.path.join(d["gate_dir"], "gate_report.json")
        if os.path.isfile(gp):
            print(gp)
            break
PY
)
  if [ -n "${GR:-}" ]; then
    step "   找到 $GR ⇒ 填 rc1"
    python manu/training/measure_reference_frame.py --out-dir "$REF_DIR" --fill-rc1-from "$GR" 2>&1 | tee -a "$PIPE_LOG"
    RC1_DONE=1
    break
  fi
  NOW_S=$(date '+%s')
  if [ "$STOP_S" -gt 0 ] && [ "$NOW_S" -ge "$STOP_S" ]; then break; fi
  sleep 60
done
step "   rc1 填充：$( [ "$RC1_DONE" = 1 ] && echo 完成 || echo 未完成 )"

# ---------------------------------------------------------------- ⑤ 收尾汇总
step "⑤ 搜索收尾：补采 CSV → 重判 → 出 SUMMARY"
python manu/training/backfill_summary_rows.py \
  --out-root "$OUT_ROOT" \
  --storage "sqlite:///$OUT_ROOT/study.db" \
  --study-name "$STUDY_NAME" --apply 2>&1 | tee -a "$PIPE_LOG"
python manu/training/rejudge_with_reference.py \
  --out-root "$OUT_ROOT" --ref "$REF" --epochs 0 --apply 2>&1 | tee -a "$PIPE_LOG"
python manu/training/summarize_nogmc_search.py \
  --out-root "$OUT_ROOT" --ref "$REF" --top 20 2>&1 | tee -a "$PIPE_LOG"
cp -f "$OUT_ROOT/SUMMARY.md" "$RUNS_BASE/nogmc_weekend_SUMMARY_phase2.md" 2>/dev/null || true
cp -f "$OUT_ROOT/optuna_summary.csv" "$RUNS_BASE/nogmc_weekend_summary_phase2.csv" 2>/dev/null || true

# ---------------------------------------------------------------- ⑥ 复现阶段
step "⑥ 复现阶段：配置级多种子榜前 3 名，各补到 3 个种子"
REPL_PLAN="$OUT_ROOT/replication_plan.json"
python manu/training/plan_replication.py \
  --out-root "$OUT_ROOT" --top 3 --seeds 3 --out-json "$REPL_PLAN" 2>&1 | tee -a "$PIPE_LOG"
if [ -s "$REPL_PLAN" ]; then
  N_REPL=$(python -c "import json,sys; print(len(json.load(open('$REPL_PLAN'))))")
  step "   复现计划：$N_REPL 条 ⇒ 启动第二阶段调度器（≤${PHASE3_MAX_HOURS}h）"
  DONE_N=$(cd "$PROJECT_ROOT" && python - "$STUDY_NAME" "$OUT_ROOT/study.db" <<'PY'
import optuna, sys
try:
    s = optuna.load_study(study_name=sys.argv[1], storage="sqlite:///" + sys.argv[2])
    print(len([t for t in s.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]))
except Exception as e:
    print(0)
PY
)
  TOTAL_N=$((DONE_N + N_REPL))
  step "   done=$DONE_N ⇒ 本阶段 n-trials 设为 $TOTAL_N（只补跑复现臂，不再让 TPE 自由探索）"
  setsid nohup python -u manu/training/optuna_mix_ratio_search.py \
    --output-root "$OUT_ROOT" --study-name "$STUDY_NAME" --gpus 0,1,2,3 \
    --n-trials "$TOTAL_N" --epochs 3 --batch 32 --imgsz 640 --workers 8 \
    --max-hours "$PHASE3_MAX_HOURS" \
    --longquan-data "$LQ_DATA" --uav-data "$UAV_DATA" \
    --baseline-cache "$REF_DIR/uav_cache_base.pkl" --reference-frame "$REF" \
    --extended-space --init-weights-map manu/training/nogmc_weekend_init_weights.json \
    --enqueue-plan "$REPL_PLAN" \
    >> "$OUT_ROOT/logs/dispatcher_phase3.log" 2>&1 &
  P3=$!
  echo "$P3" > "$OUT_ROOT/dispatcher_phase3.pid"
  disown 2>/dev/null || true
  step "   复现阶段调度器 PID=$P3"
  # 等它跑完（到硬停为止）
  while kill -0 "$P3" 2>/dev/null; do
    NOW_S=$(date '+%s')
    if [ "$HARD_S" -gt 0 ] && [ "$NOW_S" -ge "$HARD_S" ]; then
      step "⏰ 到硬停 ⇒ 停掉复现阶段"
      kill -TERM "$P3" 2>/dev/null
      sleep 60
      break
    fi
    sleep 60
  done
  step "   复现阶段结束"
else
  step "   （复现计划为空 ⇒ 跳过）"
fi

# ---------------------------------------------------------------- ⑦ 最终汇总
step "⑦ 最终重判 + 汇总 + 备份"
python manu/training/backfill_summary_rows.py \
  --out-root "$OUT_ROOT" --storage "sqlite:///$OUT_ROOT/study.db" \
  --study-name "$STUDY_NAME" --apply 2>&1 | tee -a "$PIPE_LOG"
python manu/training/rejudge_with_reference.py \
  --out-root "$OUT_ROOT" --ref "$REF" --epochs 0 --apply 2>&1 | tee -a "$PIPE_LOG"
python manu/training/summarize_nogmc_search.py \
  --out-root "$OUT_ROOT" --ref "$REF" --top 20 2>&1 | tee -a "$PIPE_LOG"

STAMP=$(date '+%Y%m%d_%H%M')
BAK="$RUNS_BASE/nogmc_weekend_backup_$STAMP"
mkdir -p "$BAK"
for f in SUMMARY.md optuna_summary.csv study.db reference_frame.json; do
  [ -f "$OUT_ROOT/$f" ] && cp -f "$OUT_ROOT/$f" "$BAK/" 2>/dev/null
done
[ -f "$REF" ] && cp -f "$REF" "$BAK/" 2>/dev/null
cp -f "$OUT_ROOT/SUMMARY.md" "$RUNS_BASE/nogmc_weekend_SUMMARY_final.md" 2>/dev/null || true
cp -f "$OUT_ROOT/optuna_summary.csv" "$RUNS_BASE/nogmc_weekend_summary_final.csv" 2>/dev/null || true
step "✅ PIPELINE 全部完成"
step "   最终汇总：$OUT_ROOT/SUMMARY.md"
step "   备份：$BAK"
step "   常驻副本：$RUNS_BASE/nogmc_weekend_SUMMARY_final.md"
