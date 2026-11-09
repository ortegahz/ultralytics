#!/bin/bash
# ==============================================================================
# 结构线跨种子 soup 的三轨门禁（2026-11-06）
#
# 与 gate_crossseed_soups.sh 的关系
# ---------------------------------
# · `gate_crossseed_soups.sh` 门禁的是 **no-GMC 周末/焦点线**的 soup（家族写死在脚本里）；
# · 本脚本门禁的是 **结构线**（study `mix_ratio_search_npu20ms_v2`）的 soup
#   —— 候选来自 `build_struct_soups.py` 产出的 `soup_candidates.json`（**不写死家族**）。
#
# 为什么参数要照抄 dispatcher（optuna_mix_ratio_search.py:1291-1318）
# ---------------------------------------------------------------
#   门禁参数口径**必须与训练臂的门禁逐位一致**，否则 soup 的数字与臂的数字不可比：
#     --weights --out-dir --gpus --device --infer-gpu --cache-gpus --baseline-cache
#     --min-track2-f1 --max-track2-fp --min-track1-f1 --uav-data --ft-root
#     + 11 个 --base-track*（参照系包 reference_frame.json 的 gate_base）
#   阈值（min-track2-f1 / max-track2-fp / min-track1-f1）**从参照系包算**，
#   算法与 `optuna_mix_ratio_search.resolve_reference()` 一致：
#     max_track2_fp  = frozen_baseline.track2_fp + gate.track2_fp_max_increment
#     min_track1_f1  = frozen_baseline.track1_f1 + gate.track1_min_delta
#     min_track2_f1  = gate.track2_f1_min
#   ⇒ 对 no-GMC 参照系算出来正好是 0.985 / 112 / 0.8958967482400269
#     （与 gate_crossseed_soups.sh 里手写的三个数**逐位相同**）。
#
# 产物布局（★ 必须让 npu20ms_acceptance.py 直接收）
# -----------------------------------------------
#   <out-root>/trials/<soup_name>/gate/epoch<N>/gate_report.json
#   ⇒ acceptance 读的是 `<root>/trials/<trial_dir>/gate*/epoch*/gate_report.json`
#
# ⚠️ **GPU 可配**（默认挤在 2 号卡）：本轮 4 张训练卡都在跑 ⇒ 门禁推理要避开它们。
#
# ⚠️ **铁律一：评测脚本由用户执行** ⇒ 本脚本交给你手动跑，不自动执行。
#
# 用法（服务器，conda `uav`）：
#   bash manu/training/gate_struct_soups.sh
#   OUT_ROOT=/mnt/data/siping/runs/npu20ms INFER_GPU=2 CACHE_GPUS=2 \
#     bash manu/training/gate_struct_soups.sh
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
OUT_ROOT="${OUT_ROOT:-/mnt/data/siping/runs/npu20ms}"
SOUPS_DIR="${SOUPS_DIR:-$OUT_ROOT/soups}"
CANDIDATES="${CANDIDATES:-$SOUPS_DIR/soup_candidates.json}"
REF_DIR="${REF_DIR:-/mnt/data/siping/runs/nogmc_ref_weekend}"
REF_FRAME="${REF_FRAME:-$REF_DIR/reference_frame.json}"
BASELINE_CACHE="${BASELINE_CACHE:-$REF_DIR/uav_cache_base.pkl}"
B="${B:-/mnt/data/siping/datasets/manu}"
UAV_DATA="${UAV_DATA:-$B/uav_gmc_median_nogmc/data.yaml}"
LQ_DATA="${LQ_DATA:-$B/longquanshan_ft_gmc_median_nogmc/data.yaml}"
# ★ GPU 可配（本轮 4 张训练卡在跑 ⇒ 推理挤到空余显存多的卡上）
GPUS="${GPUS:-1}"
DEVICE="${DEVICE:-2}"
INFER_GPU="${INFER_GPU:-2}"
CACHE_GPUS="${CACHE_GPUS:-2}"
SKIP_TRACK1="${SKIP_TRACK1:-0}"
LOG="${LOG:-$OUT_ROOT/logs/gate_struct_soups.log}"

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav || exit 1
export PYTHONPATH=.
mkdir -p "$OUT_ROOT/logs" "$SOUPS_DIR"

say(){ echo "[$(date '+%F %T')] [struct-soup-gate] $*" | tee -a "$LOG"; }

[ -f "$CANDIDATES" ] || { say "[FATAL] 找不到候选清单 $CANDIDATES（先跑 build_struct_soups.py）"; exit 1; }
[ -f "$REF_FRAME" ]  || { say "[FATAL] 找不到参照系包 $REF_FRAME"; exit 1; }
[ -f "$BASELINE_CACHE" ] || { say "[FATAL] 找不到基线缓存 $BASELINE_CACHE"; exit 1; }

# ---- 参照系 ⇒ 门禁参数（11 个 --base-track* + 三个阈值）------------------------
# 写法照 rejudge_trial0003_t2t3.sh（python 先算出参数行，再拼进命令行）。
ARGS_FILE="$(mktemp /tmp/_struct_soup_gate_args.XXXXXX)"
trap 'rm -f "$ARGS_FILE"' EXIT
python - "$REF_FRAME" > "$ARGS_FILE" << 'PYEOF'
import json, sys

ref = json.loads(open(sys.argv[1], encoding="utf-8").read())
base = ref.get("gate_base") or {}
frozen = ref.get("frozen_baseline") or {}
gate = ref.get("gate") or {}

# 阈值：与 optuna_mix_ratio_search.resolve_reference() 同算法；
# 缺项时回落到 auto_epoch_gate_check.py 的模块常量 BASE。
min_track2_f1 = float(gate.get("track2_f1_min", 0.985))
fp_increment = float(gate.get("track2_fp_max_increment", 10))
track1_min_delta = float(gate.get("track1_min_delta", 0.0))
max_track2_fp = int(round(float(frozen.get("track2_fp", 26)) + fp_increment))
min_track1_f1 = float(frozen.get("track1_f1", 0.905736)) + track1_min_delta

out = []
pairs = [("track1_f1", "--base-track1-f1"), ("track1_tp", "--base-track1-tp"),
         ("track1_fp", "--base-track1-fp"), ("track2_f1", "--base-track2-f1"),
         ("track2_tp", "--base-track2-tp"), ("track2_fp", "--base-track2-fp"),
         ("track2_fn", "--base-track2-fn"), ("track3_f1", "--base-track3-f1"),
         ("track3_tp", "--base-track3-tp"), ("track3_fp", "--base-track3-fp"),
         ("track3_fn", "--base-track3-fn")]
for k, flag in pairs:
    if k in base:
        out += [flag, str(base[k])]
out += ["--min-track2-f1", str(min_track2_f1),
        "--max-track2-fp", str(max_track2_fp),
        "--min-track1-f1", str(min_track1_f1)]
print("\n".join(out))
PYEOF
say "门禁参数（来自 $REF_FRAME）：$(tr '\n' ' ' < "$ARGS_FILE")"

# ---- 候选清单 ⇒ name / epoch / 权重路径 ---------------------------------------
LIST_FILE="$(mktemp /tmp/_struct_soup_list.XXXXXX)"
trap 'rm -f "$ARGS_FILE" "$LIST_FILE"' EXIT
python - "$CANDIDATES" > "$LIST_FILE" << 'PYEOF'
import json, sys

d = json.loads(open(sys.argv[1], encoding="utf-8").read())
for c in d.get("candidates", []):
    name = c.get("name")
    ckpt = c.get("out")
    epoch = int(c.get("epoch") or 0)
    if not name or not ckpt:
        continue
    print(f"{name}\t{epoch}\t{ckpt}")
PYEOF

N_CAND=$(wc -l < "$LIST_FILE")
say "候选 soup $N_CAND 个（清单 $CANDIDATES）"

# ---- 串行门禁（一个一个跑，避免抢显存）--------------------------------------
while IFS=$'\t' read -r NAME EPOCH CKPT; do
  [ -n "${NAME:-}" ] || continue
  [ -f "$CKPT" ] || { say "跳过 $NAME（权重不存在：$CKPT）"; continue; }
  GD="$OUT_ROOT/trials/$NAME/gate/epoch$EPOCH"
  if [ -f "$GD/gate_report.json" ]; then
    say "跳过 $NAME（已有门禁结果：$GD/gate_report.json）"; continue
  fi
  mkdir -p "$GD"
  say "门禁 $NAME（epoch$EPOCH）→ $GD"
  EXTRA=""
  [ "$SKIP_TRACK1" = "1" ] && EXTRA="--skip-track1"
  python -u manu/training/auto_epoch_gate_check.py \
    --weights "$CKPT" --out-dir "$GD" \
    --gpus "$GPUS" --device "$DEVICE" --infer-gpu "$INFER_GPU" --cache-gpus "$CACHE_GPUS" \
    --baseline-cache "$BASELINE_CACHE" \
    --uav-data "$UAV_DATA" \
    --ft-root "$(dirname "$LQ_DATA")" \
    $EXTRA \
    $(tr '\n' ' ' < "$ARGS_FILE") \
    < /dev/null >> "$LOG" 2>&1
  rc=$?
  say "  $NAME 门禁 rc=$rc"
  sleep 10
done < "$LIST_FILE"

say "✅ 结构线 soup 门禁完成；报告在 $OUT_ROOT/trials/<soup_name>/gate/epoch<N>/"
say "   验收总表： python manu/training/npu20ms_acceptance.py --root $OUT_ROOT \\"
say "               --speed-table manu/pipeline/npu/struct_speed_table.json"
