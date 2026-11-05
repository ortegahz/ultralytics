#!/bin/bash
# ==============================================================================
# 决策层 K2/K3 **样本外（out-of-sample）复核** —— 判别门到底是真收益还是评估集拟合？
#
# 为什么必须做这一轮
# ------------------
# 第一轮 K2/K3 用的是 `--sum-quantile-scope arm` ⇒ **per-tag 的 `sums` 分位阈值是从
# 「该臂 cache 根目录下所有序列的候选」算出来的**，而 T3 盲测序列
# （`VIDEO00007_19700101_001324`）**就在这个总体里** ⇒ 阈值有一部分是被测对象自己定的
# ⇒ **+1.0399 pp 里含评估集拟合成分**，不能直接当收益。
#
# 本轮改用 `--sum-quantile-scope exclude-primary`
#   ⇒ 分位阈值**只用金池 7 序列**（T2 golden pool）定，**盲测序列完全不参与**
#   ⇒ 阈值在部署时同样可以「用金池定好、搬到盲测/实机上用」⇒ **这是可交付的口径**。
#
# **自带对照**：`sum_mode=abs`（绝对能量门）的阈值是写死的常数，**与总体无关**
# ⇒ 第一轮里 `abs` 模式的行**本来就是样本外的** ⇒ 两轮一对比就能直接读出
#    「tagq 模式的收益里有多少是拟合来的」。
#
# 用法（服务器，conda uav，根 /tmp/pycharm_project_10ae9e2e）：
#   setsid nohup bash manu/evaluation/run_decision_k2k3_oos.sh > /mnt/data/siping/runs/nogmc_t3/logs/decision_k2k3_oos.out 2>&1 < /dev/null &
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
RUNS="${RUNS:-/mnt/data/siping/runs}"
OUT="${OUT:-$RUNS/nogmc_t3}"
LABELS="${LABELS:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
UNTIL="${UNTIL:-2026-11-05 13:30}"

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav || exit 1
export PYTHONPATH=.
mkdir -p "$OUT/logs"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$OUT/decision_sweep/${STAMP}_k2k3_oos"
mkdir -p "$OUTDIR"

T3_SEQ="VIDEO00007_19700101_001324"
T2_SEQS="VIDEO00009_19700101_001531,VIDEO00029_19700101_014053,VIDEO00033_19700101_015000,VIDEO00034_19700101_015052,VIDEO00006_19700101_000703,VIDEO00008_19700101_001307,VIDEO00007_19700101_000818"

echo "[$(date '+%F %T')] [k2k3-oos] 输出目录 $OUTDIR"
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
  --sum-quantile-scope exclude-primary \
  --k2-th-list "0.26,0.28,0.30,0.32" \
  --k2-area-list "1,2,3,4,6,8,12" \
  --k2-sum-list "0,0.05,0.1,0.2,0.3,0.4,0.5" \
  --k2-sum-abs-list "0,5,10,20,40" \
  --k2-radius 768.0 \
  --until "$UNTIL" \
  --check-json "t49:t3=$OUT/trials/trial_0049/gate/epoch20/track3_blind.json" \
  --check-json "p2:t3=$OUT/swa/gate/cs_t0008p2/track3_blind.json"
rc=$?
echo "[$(date '+%F %T')] [k2k3-oos] rc=$rc ⇒ $OUTDIR"
exit $rc
