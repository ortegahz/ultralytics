#!/bin/bash
# 交付档一键复现：冠军 `WTH15` + `--replay-polarity auto` + `--replay-xcorr-uniq-gate 0.6`
#             + 「不自信就不上报」`--replay-abstain-peak-below 0.30 --replay-abstain-disp-px 24`
#             + 「源帧无检测 ⇒ 往回退重推」`--replay-rollback-max-frames 16`
#               叠峰值置信门 `--replay-rollback-min-peak 0.90`（FAR 红线内的甜点）
#             + `--replay-roi-selfpred-max-px 16`（§33.6 24 序 2×2 因子隔离 + ROI 前推扫描：
#               FAR 红线余量 0.000162 → 0.000452（×2.8）、hit@5px +0.0233pp、F1 +0.0301pp）
# （8 序列 gate 集 + 24 序列官方口径）。
#
#   bash manu/tracking/lookahead/run_delivery.sh [8|24|both]
#
# ⚠️ 路径按**服务器**写（host 侧请改 --cache-root / --labels-root / --frames-root）。
#    服务器：cwd=/tmp/pycharm_project_10ae9e2e，python=conda env `uav`。
set -eu
cd "$(dirname "$0")/../../.."        # → 项目根（manu/tracking/lookahead → 上三级）
export PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}
C8=runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache
C24=runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache
GT="${GT:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
FRAMES="${FRAMES:-/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg}"
# 帧缓存（memmap）：sshfs 上读+解码 11.3 ms/帧，memmap ~0.1 ms/帧。
# **逐位安全**（§33.5：抽样 12 帧 np.array_equal 全等 + 同配置端到端逐位一致）⇒ 命中缓存即用它。
FC="${FRAMES_CACHE:-/mnt/data/siping/frame_cache_lq}"
FRAME_ARGS=""
if [ -d "$FC" ]; then
  FRAME_ARGS="--frames-cache $FC"
fi
PARAMS=manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json
OUT=runs/lookahead/final
WHICH=${1:-both}
mkdir -p "$OUT/final8" "$OUT/final24"

# 冠军 WTH15 的全部旋钮 + 三个新增档位：
#   --replay-polarity auto              目标极性（亮/暗）逐链自适应（§5.9）
#   --replay-xcorr-uniq-gate 0.6        相关峰歧义时的「二次意见」重配（§5.10）
#   --replay-abstain-peak-below 0.30    「不自信就不上报」：模板峰值低于 0.30 且
#   --replay-abstain-disp-px 24         链位移大于 24 px ⇒ 该帧不上报（§5.11）
# 其余一字不改。
CFG="--replay-mode perframe --replay-max-hold 8 --replay-fallback holdv --no-replay-hold-use-ref \
 --output-max-step-px 30 --replay-anchor delta --fast-win-size 52 --fast-bg-window 104 \
 --fast-th-k 2.5 --fast-min-energy 50 --replay-roi-mode selfpred --replay-roi-selfpred-window 3 \
 --replay-roi-selfpred-max-px 16 --replay-xcorr-radius 32 --replay-xcorr-template 32 \
 --replay-xcorr-th-k 1.5 --replay-polarity auto --replay-xcorr-uniq-gate 0.6 \
 --replay-abstain-peak-below 0.30 --replay-abstain-disp-px 24 \
 --replay-rollback-max-frames 16 --replay-rollback-min-peak 0.90"

if [ "$WHICH" = "8" ] || [ "$WHICH" = "both" ]; then
  echo "=== 8 序列（官方 gate 集） $(date '+%F %T') ==="
  $PY manu/tracking/lookahead/lookahead_longquan.py \
    --cache-root "$C8" --labels-root "$GT" --frames-root "$FRAMES" $FRAME_ARGS --params-json "$PARAMS" \
    --all-seqs --lookahead replay --arms original,stale,replay $CFG \
    --out-json "$OUT/final8/arms_delivery.json" --out-preds-dir "$OUT/final8/preds_delivery" \
    | tee "$OUT/final8/delivery.log"
fi

if [ "$WHICH" = "24" ] || [ "$WHICH" = "both" ]; then
  echo "=== 24 序列（官方口径） $(date '+%F %T') ==="
  $PY manu/tracking/lookahead/lookahead_longquan.py \
    --cache-root "$C24" --labels-root "$GT" --frames-root "$FRAMES" $FRAME_ARGS --params-json "$PARAMS" \
    --all-seqs --lookahead replay --arms original,stale,kinematic,replay $CFG \
    --out-json "$OUT/final24/arms_delivery.json" --out-preds-dir "$OUT/final24/preds_delivery" \
    | tee "$OUT/final24/delivery.log"
fi
echo "DELIVERY_DONE $(date '+%F %T')"
