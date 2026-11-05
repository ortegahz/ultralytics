#!/bin/bash
# ==============================================================================
# 决策层 K2/K3 —— area / sum 判别式候选门（零 GPU，纯 CPU）
#
# 目的
# ----
#   K1（全局 th_base × dedup_radius 网格）已证伪为「平的」：best 只 +0.07 pp，
#   达不到 RC2 反推的「净赚需 > 1.21 TP/FP」门槛。
#   K2/K3 换了一个**质上不同**的旋钮：不是全局阈值平移，而是**逐候选**判别
#   ——用缓存里已经存了、但当前决策层丢弃掉的三个属性：
#       area（连通域像素数）、sums（域内热图积分能量）、tag（尺度来源 0/1/2）。
#   假说是「弱但真」的目标有可观的面积/能量，而噪点是小面积低能量；
#   全局阈值做不到这种区分，判别门可以。
#
# 口径纪律
# --------
#   * **零 GPU、零推理、零缓存重建**：只 pickle 读已落盘的 ms_cache；
#   * **参照系只读**：`reference_frame.json` 一个字节都不碰；
#   * 自检：--check-json 把基线算出来的 TP/FP/FN/F1 与冻结门禁报告逐位比对；
#   * 结论只作**探索性证据**：决策层改参数属口径变更，命中后须成组重测才能进交接池。
#
# 用法（服务器，conda uav，根 /tmp/pycharm_project_10ae9e2e）：
#   setsid nohup bash manu/evaluation/run_decision_k2k3.sh > /mnt/data/siping/runs/nogmc_t3/logs/decision_k2k3.out 2>&1 < /dev/null &
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
RUNS="${RUNS:-/mnt/data/siping/runs}"
OUT="${OUT:-$RUNS/nogmc_t3}"
LABELS="${LABELS:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
LOGDIR="$OUT/logs"
UNTIL="${UNTIL:-2026-11-05 13:00}"

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav || exit 1
export PYTHONPATH=.
mkdir -p "$LOGDIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$OUT/decision_sweep/${STAMP}_k2k3"
mkdir -p "$OUTDIR"

# T3 = 龙泉山盲测单序列；T2 = 金池 7 序列（硬门：F1 ≥ 0.985 且 FP ≤ 112）
T3_SEQ="VIDEO00007_19700101_001324"
T2_SEQS="VIDEO00009_19700101_001531,VIDEO00029_19700101_014053,VIDEO00033_19700101_015000,VIDEO00034_19700101_015052,VIDEO00006_19700101_000703,VIDEO00008_19700101_001307,VIDEO00007_19700101_000818"

T49_CACHE="$OUT/trials/trial_0049/gate/epoch20/ms_cache"
P2_CACHE="$OUT/swa/gate/cs_t0008p2/ms_cache"

echo "[$(date '+%F %T')] [k2k3] 输出目录 $OUTDIR"
python manu/evaluation/sweep_decision_layer.py \
  --arm t49="$T49_CACHE" \
  --arm p2="$P2_CACHE" \
  --track "t3=$T3_SEQ" \
  --track "t2=$T2_SEQS" \
  --labels-root "$LABELS" \
  --out-dir "$OUTDIR" \
  --stage k2k3 \
  --primary-track t3 \
  --gated-track t2 \
  --until "$UNTIL" \
  --check-json "t49:t3=$OUT/trials/trial_0049/gate/epoch20/track3_blind.json" \
  --check-json "t49:t2=$OUT/trials/trial_0049/gate/epoch20/track2_golden.json" \
  --check-json "p2:t3=$OUT/swa/gate/cs_t0008p2/track3_blind.json" \
  --check-json "p2:t2=$OUT/swa/gate/cs_t0008p2/track2_golden.json"
rc=$?
echo "[$(date '+%F %T')] [k2k3] rc=$rc ⇒ $OUTDIR"
exit $rc
