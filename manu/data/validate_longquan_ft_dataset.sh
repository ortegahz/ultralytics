#!/usr/bin/env bash
# 龙泉山微调数据集 —— 精度对账闭环（一条命令跑完）
#
# 做什么：
#   1) 用新建数据集的 val 特征重建多尺度热图缓存（去 P0 ref 模型，与 ms_cache_nop0 同参）
#   2) 跑两条 val（黄金池 7 条 + 盲测 001324）的单帧臂，与 2026-10-20 冻结 baseline 对门禁
#   3) 跑数据集完整性校验（帧集合 / 标签 / GT 对账 / 特征健康度）
#
# 前置：数据集已构建到 $FT_ROOT
# 用法：
#   bash manu/data/validate_longquan_ft_dataset.sh
#   FT_ROOT=/path/to/dataset bash manu/data/validate_longquan_ft_dataset.sh
#   bash manu/data/validate_longquan_ft_dataset.sh \
#        --weights runs/mixed_ft_route_b/weights/epoch5.pt \
#        --out-dir runs/mixed_ft_route_b/gate/epoch5 \
#        --strict-gate 1
#
# ⚠️ 两种门禁语义，务必分清（方案 §五 的脚本片段默认落在第一种，会误判）：
#
#   ① **基线复现校验**（本脚本默认，`--expect-*`）
#      判据：|实测 F1 − 冻结基线 F1| < 5e-6 且 TP/FP 逐个相等。
#      用途：验证「**基线**能否被逐位复现」——Epoch-0 门禁、换语义后重锁 baseline。
#      **对任何微调后的权重都必然 FAIL**（数字本来就该变），因为它不是阈值判定。
#
#   ② **阈值门禁**（每-epoch 微调用）
#      判据：Track2 F1 ≥ 0.985 且 FP ≤ 36（一票否决）；Track1 F1 ≥ 去 P0 基线 0.905736；
#            Track3 观察 F1 vs 0.837331。
#      入口：manu/training/auto_epoch_gate_check.sh（= 方案 §五 指定的每-epoch 门禁脚本），
#            内部委托 manu/training/auto_epoch_gate_check.py 做**三轨全量**阈值门禁
#            （本脚本只覆盖 Track2/3，而方案 §一 明确要求 Track1 也要守）。
#
#   ⇒ **训练循环里的每-epoch 门禁请用 auto_epoch_gate_check.sh**；
#     本脚本用于「基线复现」与「数据集完整性」两条线。
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 2
export PYTHONPATH=.

# ------------------------------------------------------------------ 参数解析
# 方案 §五 要求的调用形式：--weights / --out-dir / --strict-gate
# （与 env 形式并存，env 仍可覆盖；命令行优先）
WEIGHTS_OVERRIDE=""
OUT_DIR_OVERRIDE=""
STRICT_GATE="${STRICT_GATE:-1}"
while [ $# -gt 0 ]; do
  case "$1" in
    --weights)     WEIGHTS_OVERRIDE="${2:-}"; shift 2 ;;
    --out-dir)     OUT_DIR_OVERRIDE="${2:-}"; shift 2 ;;
    --strict-gate) STRICT_GATE="${2:-1}"; shift 2 ;;
    --help|-h)
      echo "用法: bash manu/data/validate_longquan_ft_dataset.sh [--weights CKPT] [--out-dir DIR] [--strict-gate 0|1]"
      echo "  env 仍可覆盖: FT_ROOT / CACHE_ROOT / FT_EXPECT_* / WEIGHTS"
      exit 0 ;;
    *) echo "[WARN] 未知参数: $1（忽略）"; shift ;;
  esac
done

FT_ROOT="${FT_ROOT:-/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official}"
CACHE_ROOT="${CACHE_ROOT:-runs/ms_cache_ft_val_official}"
# --out-dir 决定本次产物落点；未给则沿用历史默认（runs/logs + runs/ms_cache_ft_val_official）
OUT_DIR="${OUT_DIR_OVERRIDE:-runs/logs}"
if [ -n "$OUT_DIR_OVERRIDE" ]; then
  CACHE_ROOT="$OUT_DIR_OVERRIDE/ms_cache"
fi
mkdir -p "$OUT_DIR"
# FT_EXPECT_GATES=0 时不挂 --expect-* 门禁（用于「换语义后重锁 Baseline」的场景：
# 旧门禁值属于上一套特征语义，新语义下必然不满足，不应据此判 FAIL）
FT_EXPECT_GATES="${FT_EXPECT_GATES:-1}"
# --strict-gate 0 ⇒ 关闭 --expect-* 门禁（只出数不判罚）
if [ "$STRICT_GATE" != "1" ]; then
  FT_EXPECT_GATES=0
fi
# 默认门禁 = 2026-10-21 冻结的 official 语义 Baseline v2（微调前现行基线）
FT_EXPECT_GOLDEN_F1="${FT_EXPECT_GOLDEN_F1:-0.9884359729262149}"
FT_EXPECT_GOLDEN_TP="${FT_EXPECT_GOLDEN_TP:-16283}"
FT_EXPECT_GOLDEN_FP="${FT_EXPECT_GOLDEN_FP:-26}"
FT_EXPECT_BLIND_F1="${FT_EXPECT_BLIND_F1:-0.8373308251032032}"
FT_EXPECT_BLIND_TP="${FT_EXPECT_BLIND_TP:-8012}"
FT_EXPECT_BLIND_FP="${FT_EXPECT_BLIND_FP:-590}"
D="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
P="manu/evaluation/system_sota_longquan_delivery_params.json"
WEIGHTS="${WEIGHTS_OVERRIDE:-runs/optuna_p0_nas/trial_0474/weights/best.pt}"
GOLDEN="VIDEO00009_19700101_001531,VIDEO00029_19700101_014053,VIDEO00033_19700101_015000,VIDEO00034_19700101_015052,VIDEO00006_19700101_000703,VIDEO00008_19700101_001307,VIDEO00007_19700101_000818"
BLIND="VIDEO00007_19700101_001324"

echo "=============================================================="
echo " 0. 环境"
echo "=============================================================="
python -c "import torch;print(' torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"
[ -d "$FT_ROOT/images/val" ] || { echo "[FAIL] 数据集不存在: $FT_ROOT"; exit 2; }
[ -f "$WEIGHTS" ] || { echo "[FAIL] 权重不存在: $WEIGHTS"; exit 2; }
echo " 数据集: $FT_ROOT"
echo " 缓存  : $CACHE_ROOT"
echo " 权重  : $WEIGHTS"
echo " 产物  : $OUT_DIR"
echo " 门禁  : strict-gate=$STRICT_GATE（FT_EXPECT_GATES=$FT_EXPECT_GATES）"

echo
echo "=============================================================="
echo " 1. 重建 val 多尺度热图缓存（去 P0 ref 模型，参数对齐 ms_cache_nop0）"
echo "=============================================================="
mkdir -p "$CACHE_ROOT" "$OUT_DIR"
NGPU=$(python -c "import torch;print(max(1,torch.cuda.device_count()))")
echo " GPU 数: $NGPU"
PIDS=()
for i in $(seq 0 $((NGPU-1))); do
  python manu/data/multiscale_heatmap_cache.py \
    --features-root "$FT_ROOT/images/val" \
    --cache-root "$CACHE_ROOT" \
    --shard "$i/$NGPU" \
    --scales 160,80 \
    --region-threshold 0.06 --main-threshold 0.22 \
    --fusion-dilate 32 --min-area 4 \
    --device "$i" --chunk-size 64 --batch-size 8 --cpu-threads 1 \
    --hm-weights "$WEIGHTS" --ablate-p0 \
    --overwrite > "$OUT_DIR/ft_val_cache_shard$i.log" 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
echo " 缓存 pkl 数: $(ls "$CACHE_ROOT"/*.pkl 2>/dev/null | wc -l) / 8"

echo
echo "=============================================================="
echo " 2. 精度对账（单帧臂 --no-pipeline）"
echo "=============================================================="
RC=0
GATE_ARGS_GOLDEN=(--expect-f1 "$FT_EXPECT_GOLDEN_F1" --expect-tp "$FT_EXPECT_GOLDEN_TP" --expect-fp "$FT_EXPECT_GOLDEN_FP")
GATE_ARGS_BLIND=(--expect-f1 "$FT_EXPECT_BLIND_F1" --expect-tp "$FT_EXPECT_BLIND_TP" --expect-fp "$FT_EXPECT_BLIND_FP")
if [ "$FT_EXPECT_GATES" != "1" ]; then
  GATE_ARGS_GOLDEN=()
  GATE_ARGS_BLIND=()
  echo " ⚠️ 已关闭 --expect-* 门禁（FT_EXPECT_GATES=$FT_EXPECT_GATES）：本次为重锁 Baseline"
fi

echo "--- 2.1 黄金池 7 条（期望 F1=$FT_EXPECT_GOLDEN_F1 TP=$FT_EXPECT_GOLDEN_TP FP=$FT_EXPECT_GOLDEN_FP）---"
python manu/evaluation/system_sota_longquan.py \
  --cache-root "$CACHE_ROOT" --labels-root "$D" --params-json "$P" \
  --seqs "$GOLDEN" \
  --out-json "$OUT_DIR/ft_golden_pool_single.json" \
  --out-preds-dir "$OUT_DIR/ft_preds_golden_single" \
  --no-pipeline \
  "${GATE_ARGS_GOLDEN[@]}" || RC=1

echo "--- 2.2 盲测 001324（期望 F1=$FT_EXPECT_BLIND_F1 TP=$FT_EXPECT_BLIND_TP FP=$FT_EXPECT_BLIND_FP）---"
python manu/evaluation/system_sota_longquan.py \
  --cache-root "$CACHE_ROOT" --labels-root "$D" --params-json "$P" \
  --seq "$BLIND" \
  --out-json "$OUT_DIR/ft_blind_001324_single.json" \
  --out-preds-dir "$OUT_DIR/ft_preds_001324_single" \
  --no-pipeline \
  "${GATE_ARGS_BLIND[@]}" || RC=1

echo
echo "=============================================================="
echo " 3. 数据集完整性校验"
echo "=============================================================="
python manu/data/check_longquan_ft_dataset.py \
  --dataset-root "$FT_ROOT" \
  --split-config manu/data/longquan_ft_split.json \
  --labels-root "$D" \
  --out-json "$OUT_DIR/longquan_ft_dataset_check.json" || RC=1

echo
echo "=============================================================="
if [ "$RC" -eq 0 ]; then echo " 结果: 全部通过 ✅"; else echo " 结果: 存在未通过项 ❌ (见上)"; fi
echo "=============================================================="
exit "$RC"
