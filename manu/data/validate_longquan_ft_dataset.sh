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
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 2
export PYTHONPATH=.

FT_ROOT="${FT_ROOT:-/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official}"
CACHE_ROOT="${CACHE_ROOT:-runs/ms_cache_ft_val_official}"
# FT_EXPECT_GATES=0 时不挂 --expect-* 门禁（用于「换语义后重锁 Baseline」的场景：
# 旧门禁值属于上一套特征语义，新语义下必然不满足，不应据此判 FAIL）
FT_EXPECT_GATES="${FT_EXPECT_GATES:-1}"
# 默认门禁 = 2026-10-21 冻结的 official 语义 Baseline v2（微调前现行基线）
FT_EXPECT_GOLDEN_F1="${FT_EXPECT_GOLDEN_F1:-0.9884359729262149}"
FT_EXPECT_GOLDEN_TP="${FT_EXPECT_GOLDEN_TP:-16283}"
FT_EXPECT_GOLDEN_FP="${FT_EXPECT_GOLDEN_FP:-26}"
FT_EXPECT_BLIND_F1="${FT_EXPECT_BLIND_F1:-0.8373308251032032}"
FT_EXPECT_BLIND_TP="${FT_EXPECT_BLIND_TP:-8012}"
FT_EXPECT_BLIND_FP="${FT_EXPECT_BLIND_FP:-590}"
D="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
P="manu/evaluation/system_sota_longquan_delivery_params.json"
WEIGHTS="runs/optuna_p0_nas/trial_0474/weights/best.pt"
GOLDEN="VIDEO00009_19700101_001531,VIDEO00029_19700101_014053,VIDEO00033_19700101_015000,VIDEO00034_19700101_015052,VIDEO00006_19700101_000703,VIDEO00008_19700101_001307,VIDEO00007_19700101_000818"
BLIND="VIDEO00007_19700101_001324"

echo "=============================================================="
echo " 0. 环境"
echo "=============================================================="
python -c "import torch;print(' torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"
[ -d "$FT_ROOT/images/val" ] || { echo "[FAIL] 数据集不存在: $FT_ROOT"; exit 2; }
echo " 数据集: $FT_ROOT"
echo " 缓存  : $CACHE_ROOT"

echo
echo "=============================================================="
echo " 1. 重建 val 多尺度热图缓存（去 P0 ref 模型，参数对齐 ms_cache_nop0）"
echo "=============================================================="
mkdir -p "$CACHE_ROOT" runs/logs
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
    --overwrite > "runs/logs/ft_val_cache_shard$i.log" 2>&1 &
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
  --out-json runs/logs/ft_golden_pool_single.json \
  --out-preds-dir runs/ft_preds_golden_single \
  --no-pipeline \
  "${GATE_ARGS_GOLDEN[@]}" || RC=1

echo "--- 2.2 盲测 001324（期望 F1=$FT_EXPECT_BLIND_F1 TP=$FT_EXPECT_BLIND_TP FP=$FT_EXPECT_BLIND_FP）---"
python manu/evaluation/system_sota_longquan.py \
  --cache-root "$CACHE_ROOT" --labels-root "$D" --params-json "$P" \
  --seq "$BLIND" \
  --out-json runs/logs/ft_blind_001324_single.json \
  --out-preds-dir runs/ft_preds_001324_single \
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
  --out-json runs/logs/longquan_ft_dataset_check.json || RC=1

echo
echo "=============================================================="
if [ "$RC" -eq 0 ]; then echo " 结果: 全部通过 ✅"; else echo " 结果: 存在未通过项 ❌ (见上)"; fi
echo "=============================================================="
exit "$RC"
