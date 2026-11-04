#!/bin/bash
# 「新旋钮默认关 ⇒ 逐位一致」复核（--replay-rollback-max-frames /
# --replay-rollback-min-peak）。三组对照：
#   ① replay 臂（冠军配置，新旋钮**不传**）vs **上一版冠军**产物（rb0 备份）
#      ⇒ 证明新旋钮默认是「关」，且关着时行为与上一版冠军逐位相同
#   ② --lookahead off --arms original vs 零延迟参考产物（ref）
#   ③ replay 臂（**带上**交付档的新旋钮）vs 当前交付档产物
#      ⇒ 证明交付档产物可由交付脚本原样复现
#
#   bash manu/tracking/lookahead/check_newknobs_off.sh [8|24]
set -u
cd "$(dirname "$0")/../../.."        # → 项目根
export PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}
F=runs/lookahead/final
GT="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
FR="/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg"
PARAMS=manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json
ROLLBACK=${ROLLBACK:-"--replay-rollback-max-frames 16 --replay-rollback-min-peak 0.90"}
TAG=${1:-8}
case "$TAG" in
  8)  CACHE=runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache
      REF="$F/_prev_champion_backup/preds_delivery_8seq_rb0"; OFFREF=runs/lookahead/ref/preds_original_v3
      OUT="$F/newknob_off8"; DELIV="$F/final8/preds_delivery" ;;
  24) CACHE=runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache
      REF="$F/_prev_champion_backup/preds_delivery_24seq_rb0"; OFFREF="$F/ref_preds_original_v3_24seq"
      OUT="$F/newknob_off24"; DELIV="$F/final24/preds_delivery" ;;
  *) echo "usage: $0 [8|24]"; exit 2 ;;
esac
CFG="--replay-mode perframe --replay-max-hold 8 --replay-fallback holdv --no-replay-hold-use-ref \
 --output-max-step-px 30 --replay-anchor delta --fast-win-size 52 --fast-bg-window 104 \
 --fast-th-k 2.5 --fast-min-energy 50 --replay-roi-mode selfpred --replay-roi-selfpred-window 3 \
 --replay-roi-selfpred-max-px 4 --replay-xcorr-radius 32 --replay-xcorr-template 32 \
 --replay-xcorr-th-k 1.5 --replay-polarity auto --replay-xcorr-uniq-gate 0.6 \
 --replay-abstain-peak-below 0.30 --replay-abstain-disp-px 24"
mkdir -p "$OUT"
rc=0

# 逐序列比对 per_frame + metrics（**不能比整文件 md5**：config.cache_root 的
# 相对/绝对路径不同会造成假阳性）
cmp_dir() {  # $1=新产物目录 $2=参考目录 $3=标签
  same=0; diff=0; miss=0
  for f in "$1"/*.json; do
    n=$(basename "$f"); g="$2/$n"
    [ -e "$f" ] || continue
    if [ ! -f "$g" ]; then miss=$((miss+1)); continue; fi
    if $PY -c 'import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
sys.exit(0 if a.get("per_frame")==b.get("per_frame") and a.get("metrics")==b.get("metrics") else 1)' "$f" "$g"; then
      same=$((same+1))
    else
      echo "DIFFER  $n"; diff=$((diff+1))
    fi
  done
  echo "[$TAG 序] $3 per_frame+metrics IDENTICAL=$same DIFFER=$diff MISSING=$miss"
  [ "$diff" -eq 0 ] && [ "$miss" -eq 0 ] || rc=1
}

echo "=== [$TAG 序] ① replay 臂（新旋钮默认关）vs 上一版冠军产物 $REF ==="
$PY manu/tracking/lookahead/lookahead_longquan.py \
  --cache-root "$CACHE" --labels-root "$GT" --frames-root "$FR" --params-json "$PARAMS" \
  --all-seqs --lookahead replay --arms replay $CFG \
  --out-json "$OUT/arms.json" --out-preds-dir "$OUT/preds_replay" > "$OUT/replay.log" 2>&1 \
  || { echo "[FAIL] replay run"; tail -5 "$OUT/replay.log"; exit 1; }
cmp_dir "$OUT/preds_replay" "$REF" "replay(新旋钮关) vs 上一版冠军"

echo "=== [$TAG 序] ② --lookahead off --arms original vs $OFFREF ==="
$PY manu/tracking/lookahead/lookahead_longquan.py \
  --cache-root "$CACHE" --labels-root "$GT" --frames-root "$FR" --params-json "$PARAMS" \
  --all-seqs --lookahead off --arms original \
  --out-json "$OUT/arms_off.json" --out-preds-dir "$OUT/preds_off" > "$OUT/off.log" 2>&1 \
  || { echo "[FAIL] off run"; tail -5 "$OUT/off.log"; exit 1; }
cmp_dir "$OUT/preds_off" "$OFFREF" "lookahead-off vs 零延迟参考"

echo "=== [$TAG 序] ③ replay 臂（带上交付档新旋钮）vs 交付档产物 $DELIV ==="
$PY manu/tracking/lookahead/lookahead_longquan.py \
  --cache-root "$CACHE" --labels-root "$GT" --frames-root "$FR" --params-json "$PARAMS" \
  --all-seqs --lookahead replay --arms replay $CFG $ROLLBACK \
  --out-json "$OUT/arms_delivery.json" --out-preds-dir "$OUT/preds_delivery" > "$OUT/delivery.log" 2>&1 \
  || { echo "[FAIL] delivery run"; tail -5 "$OUT/delivery.log"; exit 1; }
cmp_dir "$OUT/preds_delivery" "$DELIV" "replay(带新旋钮) vs 交付档"

[ "$rc" -eq 0 ] && echo "NEWKNOBS_OFF_OK" || echo "NEWKNOBS_OFF_FAIL"
exit "$rc"
