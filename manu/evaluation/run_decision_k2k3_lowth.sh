#!/bin/bash
# ==============================================================================
# 决策层 K2/K3 **跟进网格**：低阈值 × 更严判别门（纯 CPU、零 GPU）
#
# 为什么要有这一轮
# --------------
# 第一轮 K2/K3（th ∈ {0.26,0.28,0.30,0.32}）在 `cs_t0008p2`（**T1 过线**的那个）上
# 把 T3 从 0.869543 推到 **0.879942（+1.0399 pp）** ⇒ 判别门**确实**是杠杆，
# 且已把「兼得」缺口从 2.0457 pp 压到 **1.0058 pp**。
#
# 但第一轮**只砍虚警、不换回 TP**（TP 不变、FN 不变，只是 FP 被门滤掉）。
# 而 D1 诊断说 T3 的漏检是 `fn_absent=1530 / fn_compete=0` ⇒ **目标根本没冒头**。
# ⇒ 要再进一步，只能**把阈值降下来让更多候选冒头**，再用门把噪声滤掉：
#     低阈值（多候选 ⇒ 可能换回 TP）  ×  严判别门（area/sums ⇒ 滤掉噪声 FP）
#
# 口径纪律（不变）
# ----------------
#   * 零 GPU、零推理、零缓存重建；`reference_frame.json` 只读；
#   * 这是**评估集上搜出来的** ⇒ 属探索性证据，命中后须成组重测才能进交接池。
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
RUNS="${RUNS:-/mnt/data/siping/runs}"
OUT="${OUT:-$RUNS/nogmc_t3}"
LABELS="${LABELS:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
UNTIL="${UNTIL:-2026-11-05 13:00}"
TAG="${TAG:-lowth}"

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav || exit 1
export PYTHONPATH=.
mkdir -p "$OUT/logs"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$OUT/decision_sweep/${STAMP}_k2k3_${TAG}"
mkdir -p "$OUTDIR"

T3_SEQ="VIDEO00007_19700101_001324"
T2_SEQS="VIDEO00009_19700101_001531,VIDEO00029_19700101_014053,VIDEO00033_19700101_015000,VIDEO00034_19700101_015052,VIDEO00006_19700101_000703,VIDEO00008_19700101_001307,VIDEO00007_19700101_000818"

echo "[$(date '+%F %T')] [k2k3-$TAG] 输出目录 $OUTDIR"
python manu/evaluation/sweep_decision_layer.py \
  --arm t49="$OUT/trials/trial_0049/gate/epoch20/ms_cache" \
  --arm p2="$OUT/swa/gate/cs_t0008p2/ms_cache" \
  --track "t3=$T3_SEQ" \
  --track "t2=$T2_SEQS" \
  --labels-root "$LABELS" \
  --out-dir "$OUTDIR" \
  --stage k2k3 \
  --primary-track t3 \
  --gated-track t2 \
  --k2-th-list "0.18,0.20,0.22,0.24,0.26,0.28,0.30,0.32" \
  --k2-area-list "1,2,3,4,6,8,12,16,24" \
  --k2-sum-list "0,0.1,0.2,0.3,0.4,0.5" \
  --k2-sum-abs-list "0,5,10,20,40,80,160" \
  --k2-radius 768.0 \
  --until "$UNTIL" \
  --check-json "t49:t3=$OUT/trials/trial_0049/gate/epoch20/track3_blind.json" \
  --check-json "p2:t3=$OUT/swa/gate/cs_t0008p2/track3_blind.json"
rc=$?
echo "[$(date '+%F %T')] [k2k3-$TAG] rc=$rc ⇒ $OUTDIR"
exit $rc
