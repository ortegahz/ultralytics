#!/bin/bash
# ==============================================================================
# 定点打靶流水线（2026-11-02，窗口到 **周二 09:00**）
#
# 顺序：
#   ① 变体 C（Model Soups，零训练成本）—— 4 个融合权重各跑一次门禁
#   ② 变体 A/B（16 条确定性臂）—— 聚焦空间 TPE 启动（确定性臂优先，再 TPE 精修）
#   ③ 截止 → backfill → rejudge → SUMMARY → 备份 → 冻结权重
#
# 关键护栏（都来自周末的教训）：
#   · worker 调门禁前会**释放训练模型显存**（本轮新加）⇒ 门禁缓存重建不再 OOM
#   · 门禁**串行**跑（同一时刻只有 1 个门禁在重建缓存）⇒ 不再 4 并发抢显存
#   · 产物落 /mnt/data（物理盘）
#   · 看门狗：调度器意外退出 ⇒ 自动断点续跑重启
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
RUNS_BASE="${RUNS_BASE:-/mnt/data/siping/runs}"
OLD_ROOT="${OLD_ROOT:-$RUNS_BASE/nogmc_weekend}"          # 周末结果（参照系/基线/种子权重）
OUT_ROOT="${OUT_ROOT:-$RUNS_BASE/nogmc_focus}"             # 本轮产物
REF_DIR="${REF_DIR:-$RUNS_BASE/nogmc_ref_weekend}"
LOG="${LOG:-$RUNS_BASE/nogmc_focus_pipeline.log}"
STUDY_NAME="${STUDY_NAME:-mix_ratio_search_nogmc_focus}"
PLAN="${PLAN:-manu/training/nogmc_focus_plan.json}"
INIT_MAP="${INIT_MAP:-manu/training/nogmc_weekend_init_weights.json}"
# 训练截止时刻（服务器本地时间）——周二 06:30，留 2.5 h 给门禁+收尾
TRAIN_STOP="${TRAIN_STOP:-2026-11-03 06:30}"

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav || exit 1
export PYTHONPATH=.
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/soups"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

REF="$REF_DIR/reference_frame.json"
[ -f "$REF" ] || { say "缺参照系包 $REF"; exit 1; }
BASE="$REF_DIR/uav_cache_base.pkl"
[ -f "$BASE" ] || { say "缺基线缓存 $BASE"; exit 1; }

say "════ 定点打靶流水线启动 ════  训练截止 $TRAIN_STOP"

# ---------------------------------------------------------------- ① 变体 C：soup 门禁
say "① 变体 C（Model Soups）：4 个融合权重各跑一次门禁（串行，避免显存争抢）"
# ⚠️ 必须用**空格**分隔（`for S in $SUPS` 按空白切分；用冒号会导致一个 case 都不匹配
#    ⇒ SRC 未赋值 ⇒ set -u 直接退出脚本。2026-11-02 实测踩过。）
SUPS="soup_strict3 soup_strict3_nobn soup_high3 soup_high3_nobn"
for S in $SUPS; do
  case "$S" in
    soup_strict3)      SRC="$OLD_ROOT/trials/trial_0009/weights/epoch9.pt $OLD_ROOT/trials/trial_0025/weights/epoch6.pt $OLD_ROOT/trials/trial_0018/weights/epoch6.pt"; NOBN="" ;;
    soup_strict3_nobn) SRC="$OLD_ROOT/trials/trial_0009/weights/epoch9.pt $OLD_ROOT/trials/trial_0025/weights/epoch6.pt $OLD_ROOT/trials/trial_0018/weights/epoch6.pt"; NOBN="--nobn" ;;
    soup_high3)        SRC="$OLD_ROOT/trials/trial_0087/weights/epoch8.pt $OLD_ROOT/trials/trial_0113/weights/epoch8.pt $OLD_ROOT/trials/trial_0008/weights/epoch8.pt"; NOBN="" ;;
    soup_high3_nobn)   SRC="$OLD_ROOT/trials/trial_0087/weights/epoch8.pt $OLD_ROOT/trials/trial_0113/weights/epoch8.pt $OLD_ROOT/trials/trial_0008/weights/epoch8.pt"; NOBN="--nobn" ;;
  esac
  W="$OUT_ROOT/soups/$S.pt"
  if [ ! -f "$W" ]; then
    say "  融合 $S …"
    python manu/training/fuse_weights.py --inputs $SRC --out "$W" --name "$S" $NOBN >> "$LOG" 2>&1
  fi
  [ -f "$W" ] || { say "  ✗ $S 融合失败"; continue; }
  GD="$OUT_ROOT/soups/$S/gate"
  mkdir -p "$GD"
  say "  门禁 $S …"
  python manu/training/auto_epoch_gate_check.py \
    --weights "$W" --out-dir "$GD" \
    --gpus 4 --device 0 --infer-gpu 0 --cache-gpus 0,1,2,3 \
    --baseline-cache "$BASE" \
    --min-track2-f1 0.985 --max-track2-fp 112 --min-track1-f1 0.8958967482400269 \
    --uav-data /mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml \
    --ft-root /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc \
    --base-track1-f1 0.9008967482400269 --base-track1-tp 21499 --base-track1-fp 1118 \
    --base-track2-f1 0.981074481074481 --base-track2-tp 16070 --base-track2-fp 52 --base-track2-fn 568 \
    --base-track3-f1 0.7578120617110798 --base-track3-tp 6754 --base-track3-fp 536 --base-track3-fn 3781 \
    >> "$LOG" 2>&1
  say "  $S 门禁 rc=$?"
  sleep 10
done
say "① 变体 C 门禁完成"

# ---------------------------------------------------------------- ② 定点打靶搜索
STOP_S=$(date -d "$TRAIN_STOP" '+%s' 2>/dev/null || echo 0)
LAUNCH_H=$(( (STOP_S - $(date '+%s')) / 3600 ))
[ "$LAUNCH_H" -lt 1 ] && LAUNCH_H=1
say "② 启动聚焦搜索（16 条确定性臂 → 聚焦 TPE），预算 ${LAUNCH_H} h"

# 调度器：聚焦空间 + 扩展空间 + 确定性臂入队
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

# 看门狗：调度器意外退出 ⇒ 自动重启（断点续跑）
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

# ---------------------------------------------------------------- ③ 收尾
say "③ 收尾：补采 → 重判 → SUMMARY → 备份"
python manu/training/backfill_summary_rows.py --out-root "$OUT_ROOT" \
  --storage "sqlite:///$OUT_ROOT/study.db" --study-name "$STUDY_NAME" --apply >> "$LOG" 2>&1
python manu/training/rejudge_with_reference.py --out-root "$OUT_ROOT" --ref "$REF" --epochs 0 --apply >> "$LOG" 2>&1
python manu/training/summarize_nogmc_search.py --out-root "$OUT_ROOT" --ref "$REF" --top 20 >> "$LOG" 2>&1
cp -f "$OUT_ROOT/SUMMARY.md" "$RUNS_BASE/nogmc_focus_SUMMARY.md" 2>/dev/null
cp -f "$OUT_ROOT/optuna_summary.csv" "$RUNS_BASE/nogmc_focus_summary.csv" 2>/dev/null
STAMP=$(date '+%Y%m%d_%H%M')
mkdir -p "$RUNS_BASE/nogmc_focus_backup_$STAMP"
cp -f "$OUT_ROOT/SUMMARY.md" "$OUT_ROOT/optuna_summary.csv" "$OUT_ROOT/study.db" "$RUNS_BASE/nogmc_focus_backup_$STAMP/" 2>/dev/null
say "✅ 定点打靶流水线全部完成；备份 $RUNS_BASE/nogmc_focus_backup_$STAMP"
