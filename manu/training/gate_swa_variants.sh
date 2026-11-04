#!/bin/bash
# ==============================================================================
# SWA（同一 trial 内相邻 epoch 权重平均）变体的三轨门禁（2026-11-02 16:35）
#
# 为什么做这个
# ------------
# 变体 C 的 Model Soups 是**跨配置**平均（trial_0009 + trial_0025 + trial_0018），
# 而这三者的 T1/T3 画像差别很大 ⇒ 平均结果只是**在两端之间插值**
# （strict3: T1 0.8968/T3 0.8475；high3_nobn: T1 0.8938/T3 0.8747）⇒ 路线判负。
#
# **SWA 沿 epoch 轴平均是另一回事**：同一 run 的相邻 epoch 在**同一个损失盆地**里
# ⇒ 平均 = 降方差、不改变 T1/T3 的权衡位置，**通常还会小幅涨 F1**（SWA 的经典收益）。
# ⇒ **零训练成本**就能试，而且正对着当前的死结：
#    最近几个配置（trial_0077 / 0008 / 0082）的 T1 缺口只有 0.00015~0.00123，
#    而种子噪声是 0.0025 ⇒ 若 SWA 能把 T1 抬 0.001~0.003 且不掉 T3，
#    就可能**直接命中「T3 ≥ 0.87 且 T1 过线」**。
#
# 铁律遵守
# --------
# · **评估不得与训练同卡** ⇒ 本脚本**等训练彻底结束**（TRAIN_STOP 之后 + 调度器进程已退出）才开跑。
# · 门禁**串行**（同一时刻只跑 1 个），避免 4 并发抢显存。
# · 产物落 /mnt/data（物理盘）。
#
# 用法（服务器，conda `uav`，项目根 /tmp/pycharm_project_10ae9e2e）：
#   nohup bash manu/training/gate_swa_variants.sh > /mnt/data/siping/runs/nogmc_focus/logs/gate_swa.log 2>&1 &
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
RUNS_BASE="${RUNS_BASE:-/mnt/data/siping/runs}"
OUT_ROOT="${OUT_ROOT:-$RUNS_BASE/nogmc_focus}"
REF_DIR="${REF_DIR:-$RUNS_BASE/nogmc_ref_weekend}"
LOG="${LOG:-$OUT_ROOT/logs/gate_swa.log}"
TRAIN_STOP="${TRAIN_STOP:-2026-11-03 06:30}"
SWA_DIR="$OUT_ROOT/swa"
BASE="$REF_DIR/uav_cache_base.pkl"
STOP_S=$(date -d "$TRAIN_STOP" '+%s' 2>/dev/null || echo 0)

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav || exit 1
export PYTHONPATH=.
mkdir -p "$SWA_DIR/gate" "$OUT_ROOT/logs"
say(){ echo "[$(date '+%F %T')] [swa-gate] $*" | tee -a "$LOG"; }

[ -f "$BASE" ] || { say "缺基线缓存 $BASE"; exit 1; }

# ---- 等训练彻底结束 ------------------------------------------------------
# 判据：① 已过 TRAIN_STOP；② 调度器进程已退出（避免与训练抢卡）
say "等待训练结束（TRAIN_STOP=$TRAIN_STOP）…"
while :; do
  now=$(date '+%s')
  after_stop=0
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
say "训练已结束、GPU 已释放 ⇒ 开始串行门禁"

# ---- 串行门禁（按「最可能命中」排序）------------------------------------
# 排序理由：0077 是**唯一 T3 已过 0.87** 的配置（T1 差 0.00123）；0082 的 T1 缺口最小（0.00015）；
# 0008 次之；0113/0087 是 T3 最高的两条（T1 缺口 0.0016~0.0021）；0009 的 T1 余量最大但 T3 偏低。
ORDER="swa_0077_last3 swa_0008_last3 swa_0082_last3 swa_0113_last3 swa_0087_last3 swa_0009_last3"

for NAME in $ORDER; do
  W="$SWA_DIR/$NAME.pt"
  [ -f "$W" ] || { say "跳过 $NAME（权重不存在）"; continue; }
  GD="$SWA_DIR/gate/$NAME"
  if [ -f "$GD/gate_report.json" ] || ls "$GD"/epoch*/gate_report.json >/dev/null 2>&1; then
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

say "✅ SWA 变体门禁全部完成；结果在 $SWA_DIR/gate/"
