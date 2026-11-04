#!/bin/bash
# ==============================================================================
# no-GMC T3 study 的**收尾交付**（薄封装）—— 2026-11-03
#
# 它不重新实现收尾链，只是把 `finalize_nogmc_focus.sh` 的**参数**换成 T3 study 的：
#   · OUT_ROOT   = /mnt/data/siping/runs/nogmc_t3
#   · STUDY_NAME = mix_ratio_search_nogmc_t3
#   · SELF_TAG   = t3      （本 study 在判定器里的来源标签 ⇒ trial 名加 't3:' 前缀）
#   · 判定池 = **三份 CSV**（focus + weekend + t3）
#       ⇒ 交接口径是「同超参多种子中位」，而**同配方的种子横跨三个 study**
#          （周末 trial_0008 / focus 的 FOCUS_* / T3 的 INS_*）⇒ 只读一份永远凑不够 n≥2。
#   · T3 可达性 = **三个 study**（focus + weekend + t3）
#
# ⚠️ 旧 study 的默认行为**逐位不变**：不带任何环境变量跑 `finalize_nogmc_focus.sh`
#    仍然是 focus 那轮的判定池（focus + weekend）、备份名 nogmc_focus_backup_*、
#    根下拷贝 nogmc_focus_*。
#
# 用法（服务器，conda `uav`，项目根 /tmp/pycharm_project_10ae9e2e）：
#   bash manu/training/finalize_nogmc_t3.sh
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
RUNS_BASE="${RUNS_BASE:-/mnt/data/siping/runs}"
export RUNS_BASE
export PROJECT_ROOT
export OUT_ROOT="${OUT_ROOT:-$RUNS_BASE/nogmc_t3}"
export OLD_ROOT="${OLD_ROOT:-$RUNS_BASE/nogmc_weekend}"     # 周末那轮（种子来源之一）
export REF_DIR="${REF_DIR:-$RUNS_BASE/nogmc_ref_weekend}"
export STUDY_NAME="${STUDY_NAME:-mix_ratio_search_nogmc_t3}"
export RUN_TAG="${RUN_TAG:-nogmc_t3}"                        # 备份目录 / 根下拷贝的前缀
export SELF_TAG="${SELF_TAG:-t3}"                             # 本 study 的判定池标签
export RC_ID="${RC_ID:-Candidate_t3}"
export RC_NAME="${RC_NAME:-no-GMC T3 前沿（26 臂）最优交接候选}"
export LOG="${LOG:-$RUNS_BASE/nogmc_t3_finalize.log}"
export SWA_MARK="${SWA_MARK:-$RUNS_BASE/nogmc_t3/logs/gate_variants.log}"
# ---- 追加 focus 那轮（判定器 / 固化 / T3 可达性都收它）---------------------
export EXTRA_CSV="${EXTRA_CSV:-focus=$RUNS_BASE/nogmc_focus/optuna_summary.csv}"
export EXTRA_TRIALS_ROOT="${EXTRA_TRIALS_ROOT:-focus=$RUNS_BASE/nogmc_focus/trials}"
export EXTRA_STUDY="${EXTRA_STUDY:-focus=$RUNS_BASE/nogmc_focus/study.db:mix_ratio_search_nogmc_focus}"
# ---- 变体池（soup / SWA）：**不是 trial** ⇒ 必须走判定器的独立通道 ------------
export EXTRA_VARIANTS_CSV="${EXTRA_VARIANTS_CSV:-$RUNS_BASE/nogmc_t3/variants.csv}"
export VARIANTS_SWA_DIR="${VARIANTS_SWA_DIR:-$RUNS_BASE/nogmc_t3/swa}"
export EXTRA_VARIANTS_SWA_DIRS="${EXTRA_VARIANTS_SWA_DIRS:-$RUNS_BASE/nogmc_focus/swa}"
export VARIANTS_FAMILIES_JSON="${VARIANTS_FAMILIES_JSON:-manu/training/nogmc_t3_families.json}"

# ⚠️ `say` 定义在 finalize_nogmc_focus.sh 里 ⇒ 本脚本（不再 exec 它）必须自带一份
say(){ echo "[$(date '+%F %T')] $*" >> "${LOG:-/tmp/nogmc_t3_finalize.log}"; }

# ---- 先跑通用收尾（判定 / 固化 / T3 可达性 / 备份）--------------------------
bash "$PROJECT_ROOT/manu/training/finalize_nogmc_focus.sh"
FIN_RC=$?

# ==============================================================================
# 2026-11-03 追加：**前沿地图** + **轮次报告**（本机 18:30 关机 ⇒ 必须服务器侧自主产出）
# ==============================================================================
say "⑤b 前沿地图（T1–T3 前沿 + 逐旋钮汇率）"
python manu/training/analyze_t3_frontier.py \
  --study "focus=$RUNS_BASE/nogmc_focus/study.db:mix_ratio_search_nogmc_focus" \
  --study "weekend=$RUNS_BASE/nogmc_weekend/study.db:mix_ratio_search_nogmc_weekend" \
  --study "t3=$OUT_ROOT/study.db:$STUDY_NAME" \
  --csv "focus=$RUNS_BASE/nogmc_focus/optuna_summary.csv" \
  --csv "weekend=$RUNS_BASE/nogmc_weekend/optuna_summary.csv" \
  --csv "t3=$OUT_ROOT/optuna_summary.csv" \
  --ref "$REF_DIR/reference_frame.json" \
  --out-md "$OUT_ROOT/T3_FRONTIER.md" --out-json "$OUT_ROOT/T3_FRONTIER.json" \
  >> "$LOG" 2>&1
say "   rc=$?"

say "⑤c 轮次报告（ROUND_REPORT.md / .json + memory 回写）"
python manu/training/build_round_report.py \
  --out-root "$OUT_ROOT" --ref "$REF_DIR/reference_frame.json" \
  --memory-file "manu/memory/nogmc_t3_20261103.md" \
  >> "$LOG" 2>&1
say "   rc=$?"

say "⑤d 交付件自检（清单 / 判定自洽 / 时序 / 导航表）"
python manu/training/check_nogmc_t3_deliverables.py \
  --out-root "$OUT_ROOT" --ref "$REF_DIR/reference_frame.json" \
  > "$OUT_ROOT/DELIVERABLE_CHECK.txt" 2>&1
say "   rc=$? → $OUT_ROOT/DELIVERABLE_CHECK.txt"

exit $FIN_RC
