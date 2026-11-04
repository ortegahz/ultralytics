#!/bin/bash
# ==============================================================================
# 跨种子 soup（同配方·不同种子·末 epoch 权重平均）的三轨门禁（2026-11-02）
#
# 与 gate_swa_variants.sh 的关系
# -----------------------------
# · `gate_swa_variants.sh` 先跑（沿 **epoch 轴** 的 SWA，6 个候选）；
# · 本脚本**等它跑完**再跑（沿 **种子轴** 的 soup，3 个家族）；
# · 两个都**等训练彻底结束**才开跑（铁律：评估不得与训练同卡）。
#
# 为什么排队而不是同时跑
# ----------------------
# 门禁窗口只有 06:30~08:00（每个候选约 10 min ⇒ 最多 9 个）。
# ⇒ 按"最可能命中"排序跑，**时间不够就自然截断**（低价值的排在后面被砍掉）。
#
# 用法：
#   nohup bash manu/training/gate_crossseed_soups.sh > .../logs/gate_crossseed.log 2>&1 &
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
RUNS_BASE="${RUNS_BASE:-/mnt/data/siping/runs}"
OUT_ROOT="${OUT_ROOT:-$RUNS_BASE/nogmc_focus}"
REF_DIR="${REF_DIR:-$RUNS_BASE/nogmc_ref_weekend}"
LOG="${LOG:-$OUT_ROOT/logs/gate_crossseed.log}"
TRAIN_STOP="${TRAIN_STOP:-2026-11-03 06:30}"
SWA_DIR="$OUT_ROOT/swa"
BASE="$REF_DIR/uav_cache_base.pkl"
STOP_S=$(date -d "$TRAIN_STOP" '+%s' 2>/dev/null || echo 0)

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav || exit 1
export PYTHONPATH=.
mkdir -p "$OUT_ROOT/logs" "$SWA_DIR/gate"
say(){ echo "[$(date '+%F %T')] [cs-gate] $*" | tee -a "$LOG"; }

[ -f "$BASE" ] || { say "缺基线缓存 $BASE"; exit 1; }

# ---- ① 等训练彻底结束 ---------------------------------------------------
say "等待训练结束（TRAIN_STOP=$TRAIN_STOP）…"
while :; do
  now=$(date '+%s')
  after_stop=0
  [ "$STOP_S" -gt 0 ] && [ "$now" -ge "$STOP_S" ] && after_stop=0
  [ "$STOP_S" -gt 0 ] && [ "$now" -ge "$STOP_S" ] && after_stop=1
  disp_alive=0
  if [ -f "$OUT_ROOT/dispatcher.pid" ]; then
    dpid=$(cat "$OUT_ROOT/dispatcher.pid" 2>/dev/null | tr -d '[:space:]')
    if [ -n "$dpid" ] && kill -0 "$dpid" 2>/dev/null; then disp_alive=1; fi
  fi
  if [ "$after_stop" = "1" ] && [ "$disp_alive" = "0" ]; then break; fi
  say "等待中：已过截止=$after_stop 调度器存活=$disp_alive"
  sleep 60
done

# ---- ② 等 SWA 门禁跑完（串行，不抢显存）--------------------------------
SWA_MARK="$OUT_ROOT/logs/gate_swa.log"
if [ -f "$SWA_MARK" ]; then
  say "等待 SWA 门禁收尾（最多 150 分钟，避免 SWA 脚本异常时死等）…"
  WAITED=0
  while [ "$WAITED" -lt 150 ]; do
    grep -q "SWA 变体门禁全部完成" "$SWA_MARK" && break
    sleep 60
    WAITED=$((WAITED+1))
  done
  if grep -q "SWA 变体门禁全部完成" "$SWA_MARK"; then
    say "SWA 门禁已完成 ⇒ 开始跨种子 soup"
  else
    say "⚠️ 等待 SWA 门禁超时 ⇒ 仍继续跑跨种子 soup（可能与 SWA 门禁并发 ⇒ 显存风险）"
  fi
fi

# ---- ③ 构建跨种子 soup（若本轮种子已跑完）------------------------------
say "构建跨种子 soup …"
python manu/training/build_crossseed_soups.py \
  --focus-root "$OUT_ROOT" --weekend-root "$RUNS_BASE/nogmc_weekend" \
  --out-dir "$SWA_DIR" >> "$LOG" 2>&1
say "  构建 rc=$?"

# ---- ④ 串行门禁（按"最可能命中"排序）----------------------------------
# 0077 是唯一 T3 已过 0.87 的配方；0082 的 T1 缺口最小；0008 次之
# ⚠️ **只门禁 manifest 里列出的产物**（构建器只收 state==COMPLETE 的 trial）⇒
#    避免把**上一次遗留的半成品 soup**（例如用半训练权重建出来的）当成候选去量。
MANIFEST="$SWA_DIR/crossseed_manifest.json"
for NAME in cs_t0077 cs_t0082 cs_t0008; do
  if [ -f "$MANIFEST" ]; then
    if ! grep -q "\"$NAME.pt\"" "$MANIFEST"; then
      say "跳过 $NAME（不在构建 manifest 里 ⇒ 不是本次构建的合法产物）"; continue
    fi
  fi
  W="$SWA_DIR/$NAME.pt"
  [ -f "$W" ] || { say "跳过 $NAME（未构建）"; continue; }
  GD="$SWA_DIR/gate/$NAME"
  if ls "$GD"/epoch*/gate_report.json >/dev/null 2>&1; then
    say "跳过 $NAME（已有门禁结果）"; continue
  fi
  mkdir -p "$GD"
  say "门禁 $NAME …"
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
  say "  $NAME 门禁 rc=$?"
  sleep 10
done

say "✅ 跨种子 soup 门禁完成；结果在 $SWA_DIR/gate/"
