#!/bin/bash
# ==============================================================================
# 定点打靶**断点续跑**（2026-11-02）
#
# 用途
# ----
# `run_nogmc_focus_pipeline.sh` 是"从零跑完整轮"的一次性脚本（阶段① 变体C 门禁
# → 阶段② 聚焦搜索 → 阶段③ 收尾）。但它**不可重入**：阶段① 会重跑 4 个 soup 门禁
# （≈ 40 分钟纯浪费），而且 host 重启后没人记得该敲哪条命令。
#
# 本脚本 = **只跑阶段② + 阶段③**，跳过已完成的阶段①：
#   · soup 权重与门禁报告已落盘（`soups/*/gate/gate_report.json`）⇒ 直接复用；
#   · study.db 存在 ⇒ 调度器自动跳过已跑过的臂（去重键 = 「参数集 + 种子」）；
#   · 训练截止时刻与主流水线**保持一致**（同样默认 11-03 06:30），避免续跑时多跑。
#
# 用法（服务器 `ssh -p 32222 huangzhe@192.168.99.40`，`conda activate uav`）：
#
# ```bash
# cd /tmp/pycharm_project_10ae9e2e
# bash manu/training/resume_nogmc_focus.sh
# ```
#
# 幂等：可反复执行（已跑完的臂会被去重跳过；看门狗逻辑与主流水线一致）。
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
RUNS_BASE="${RUNS_BASE:-/mnt/data/siping/runs}"
OLD_ROOT="${OLD_ROOT:-$RUNS_BASE/nogmc_weekend}"
OUT_ROOT="${OUT_ROOT:-$RUNS_BASE/nogmc_focus}"
REF_DIR="${REF_DIR:-$RUNS_BASE/nogmc_ref_weekend}"
LOG="${LOG:-$RUNS_BASE/nogmc_focus_pipeline.log}"
STUDY_NAME="${STUDY_NAME:-mix_ratio_search_nogmc_focus}"
PLAN="${PLAN:-manu/training/nogmc_focus_plan.json}"
INIT_MAP="${INIT_MAP:-manu/training/nogmc_weekend_init_weights.json}"
TRAIN_STOP="${TRAIN_STOP:-2026-11-03 06:30}"

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav || exit 1
export PYTHONPATH=.
mkdir -p "$OUT_ROOT/logs"
say(){ echo "[$(date '+%F %T')] [resume] $*" | tee -a "$LOG"; }

REF="$REF_DIR/reference_frame.json"
BASE="$REF_DIR/uav_cache_base.pkl"
[ -f "$REF" ] || { say "缺参照系包 $REF"; exit 1; }
[ -f "$BASE" ] || { say "缺基线缓存 $BASE"; exit 1; }

# ---- 阶段① 的产物是否已就绪（有则跳过，没有也不致命：soup 只是诊断性变体）----
N_SOUP=$(ls -d "$OUT_ROOT"/soups/*/gate/gate_report.json 2>/dev/null | wc -l)
say "════ 断点续跑启动 ════  训练截止 $TRAIN_STOP"
say "  阶段① soup 门禁报告：$N_SOUP 份（存在则复用，不重跑）"
say "  study.db：$([ -f "$OUT_ROOT/study.db" ] && echo 存在 || echo '不存在⇒从零开始')"
say "  plan 臂数：$(python3 -c "import json;print(len(json.load(open('$PLAN'))))" 2>/dev/null || echo '?')"

# ---- CHECK_ONLY：只校验前置条件，不启动调度器（安全自检用）--------------
if [ "${CHECK_ONLY:-0}" = "1" ]; then
  say "CHECK-ONLY：前置条件校验"
  [ -f "$OUT_ROOT/study.db" ] && say "  ✅ study.db 存在（续跑会按「参数集+种子」去重跳过已跑臂）" || say "  ⚠️ 无 study.db ⇒ 将从零开始"
  say "  ✅ 调度器命令：python manu/training/optuna_mix_ratio_search.py --output-root $OUT_ROOT --study-name $STUDY_NAME --gpus 0,1,2,3 --focus-space --enqueue-plan $PLAN"
  exit 0
fi

# ---------------------------------------------------------------- 阶段②：聚焦搜索
STOP_S=$(date -d "$TRAIN_STOP" '+%s' 2>/dev/null || echo 0)
LAUNCH_H=$(( (STOP_S - $(date '+%s')) / 3600 ))
[ "$LAUNCH_H" -lt 1 ] && LAUNCH_H=1
say "② 启动聚焦搜索（确定性臂 → 聚焦 TPE），预算 ${LAUNCH_H} h"

setsid nohup python -u manu/training/optuna_mix_ratio_search.py \
  --output-root "$OUT_ROOT" --study-name "$STUDY_NAME" \
  --gpus 0,1,2,3 --n-trials 4096 --epochs 3 --batch 32 --imgsz 640 --workers 8 \
  --max-hours "$LAUNCH_H" \
  --longquan-data /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc/data.yaml \
  --uav-data /mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml \
  --baseline-cache "$BASE" --reference-frame "$REF" \
  --extended-space --focus-space \
  --init-weights-map "$INIT_MAP" --enqueue-plan "$PLAN" \
  >> "$OUT_ROOT/logs/dispatcher.log" 2>&1 < /dev/null &
DISP_PID=$!
echo "$DISP_PID" > "$OUT_ROOT/dispatcher.pid"
disown 2>/dev/null || true
say "   调度器 PID=$DISP_PID"

# ---- 看门狗：调度器意外退出 ⇒ 自动断点续跑重启 ----
RESTARTS=0
while :; do
  NOW_S=$(date '+%s')
  [ "$STOP_S" -gt 0 ] && [ "$NOW_S" -ge "$STOP_S" ] && { say "   到达训练截止时刻"; break; }
  sleep 60
  if ! kill -0 "$DISP_PID" 2>/dev/null; then
    REMAIN=$(( (STOP_S - $(date '+%s')) / 3600 )); [ "$REMAIN" -lt 1 ] && REMAIN=1
    RESTARTS=$((RESTARTS+1))
    say "⚠️ 调度器退出 ⇒ 第 $RESTARTS 次自动重启（剩余 ${REMAIN}h）"
    setsid nohup python -u manu/training/optuna_mix_ratio_search.py \
      --output-root "$OUT_ROOT" --study-name "$STUDY_NAME" \
      --gpus 0,1,2,3 --n-trials 4096 --epochs 3 --batch 32 --imgsz 640 --workers 8 \
      --max-hours "$REMAIN" \
      --longquan-data /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc/data.yaml \
      --uav-data /mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml \
      --baseline-cache "$BASE" --reference-frame "$REF" \
      --extended-space --focus-space \
      --init-weights-map "$INIT_MAP" --enqueue-plan "$PLAN" \
      >> "$OUT_ROOT/logs/dispatcher.log" 2>&1 < /dev/null &
    DISP_PID=$!
    echo "$DISP_PID" > "$OUT_ROOT/dispatcher.pid"
    disown 2>/dev/null || true
    sleep 30
  fi
done
say "   训练阶段结束（看门狗共重启 $RESTARTS 次）"

# ---------------------------------------------------------------- 阶段③：收尾
say "③ 收尾：补采 → 重判 → 汇总 → 判定 → 备份"
python manu/training/backfill_summary_rows.py --out-root "$OUT_ROOT" \
  --storage "sqlite:///$OUT_ROOT/study.db" --study-name "$STUDY_NAME" --apply >> "$LOG" 2>&1
python manu/training/rejudge_with_reference.py --out-root "$OUT_ROOT" --ref "$REF" --epochs 0 --apply >> "$LOG" 2>&1
python manu/training/summarize_nogmc_search.py --out-root "$OUT_ROOT" --ref "$REF" --top 20 >> "$LOG" 2>&1
# ⚠️ **判定池要含周末**（交接口径 = 同超参多种子中位 ⇒ 同配方种子横跨两个 study）。
#    ⚠️ argparse 的 `--csv` 是 nargs="+" ⇒ **只能写一次 `--csv`、后面跟多个值**；
#    重复写 `--csv A --csv B` 只会保留最后一个（argparse 的 store 动作是覆盖不是追加）。
JUDGE_CSV_ARGS=(--csv "focus=$OUT_ROOT/optuna_summary.csv")
OLD_CSV="$OLD_ROOT/optuna_summary.csv"
[ -f "$OLD_CSV" ] && JUDGE_CSV_ARGS+=("weekend=$OLD_CSV")
python manu/training/judge_handover_candidate.py \
  "${JUDGE_CSV_ARGS[@]}" --ref "$REF" \
  --out-md "$OUT_ROOT/HANDOVER.md" --out-json "$OUT_ROOT/HANDOVER.json" \
  --min-seeds 2 >> "$LOG" 2>&1
cp -f "$OUT_ROOT/SUMMARY.md" "$RUNS_BASE/nogmc_focus_SUMMARY.md" 2>/dev/null
cp -f "$OUT_ROOT/optuna_summary.csv" "$RUNS_BASE/nogmc_focus_summary.csv" 2>/dev/null
cp -f "$OUT_ROOT/HANDOVER.md" "$RUNS_BASE/nogmc_focus_HANDOVER.md" 2>/dev/null
STAMP=$(date '+%Y%m%d_%H%M')
mkdir -p "$RUNS_BASE/nogmc_focus_backup_$STAMP"
cp -f "$OUT_ROOT/SUMMARY.md" "$OUT_ROOT/optuna_summary.csv" "$OUT_ROOT/study.db" \
      "$OUT_ROOT/HANDOVER.md" "$OUT_ROOT/HANDOVER.json" \
      "$RUNS_BASE/nogmc_focus_backup_$STAMP/" 2>/dev/null
say "✅ 断点续跑流水线完成；备份 $RUNS_BASE/nogmc_focus_backup_$STAMP"
