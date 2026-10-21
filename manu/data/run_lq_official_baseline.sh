#!/usr/bin/env bash
# 龙泉山 official 语义 Baseline v2 —— 一键跑完「校验 → 建缓存 → 跑双 val → 出对比表」
#
# 与 FPV 口径的差别只有一处：**特征语义**。
#   * FPV     = 相邻帧仿射连乘（既有 longquanshan_ir_gmc_median 的口径）
#   * OFFICIAL= 逐 lag 直接拟合（Anti-UAV 官方集 uav_gmc_median 的口径，已 125/125 逐字节同源验证）
#
# 用法：
#   bash manu/data/run_lq_official_baseline.sh
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
export PYTHONPATH=.

FT_ROOT="${FT_ROOT:-/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official}"
CACHE_ROOT="${CACHE_ROOT:-runs/ms_cache_ft_val_official}"
D="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
OLD_FT="/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median"

echo "############ 0) 环境与资产 ############"
python -c "import torch;print(' torch',torch.__version__,'cuda',torch.cuda.is_available(),'gpus',torch.cuda.device_count())"
[ -d "$FT_ROOT/images/val" ] || { echo "[FAIL] official 数据集不存在: $FT_ROOT"; exit 2; }
echo " official 数据集: $FT_ROOT"
echo " 缓存          : $CACHE_ROOT"

echo
echo "############ 1) 数据集完整性（GT 须对账 16,638 / 10,535）############"
python manu/data/check_longquan_ft_dataset.py \
  --dataset-root "$FT_ROOT" \
  --split-config manu/data/longquan_ft_split.json \
  --labels-root "$D" \
  --out-json runs/logs/longquan_ft_official_dataset_check.json || exit 2

echo
echo "############ 2) 建 val 多尺度热图缓存（去 P0 ref 模型，参数对齐 ms_cache_nop0）############"
mkdir -p "$CACHE_ROOT" runs/logs
NGPU=$(python -c "import torch;print(max(1,torch.cuda.device_count()))")
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
    --hm-weights runs/optuna_p0_nas/trial_0474/weights/best.pt --ablate-p0 \
    --overwrite > "runs/logs/ft_official_cache_shard$i.log" 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
echo " 缓存 pkl: $(ls "$CACHE_ROOT"/*.pkl 2>/dev/null | wc -l)/8"

echo
echo "############ 3) 跑双 val（不挂旧门禁：旧值属 FPV 语义）############"
P=manu/evaluation/system_sota_longquan_delivery_params.json
GOLDEN="VIDEO00009_19700101_001531,VIDEO00029_19700101_014053,VIDEO00033_19700101_015000,VIDEO00034_19700101_015052,VIDEO00006_19700101_000703,VIDEO00008_19700101_001307,VIDEO00007_19700101_000818"
BLIND="VIDEO00007_19700101_001324"
python manu/evaluation/system_sota_longquan.py \
  --cache-root "$CACHE_ROOT" --labels-root "$D" --params-json "$P" \
  --seqs "$GOLDEN" \
  --out-json runs/logs/ft_official_golden_pool_single.json \
  --out-preds-dir runs/ft_official_preds_golden_single --no-pipeline || exit 2
python manu/evaluation/system_sota_longquan.py \
  --cache-root "$CACHE_ROOT" --labels-root "$D" --params-json "$P" \
  --seq "$BLIND" \
  --out-json runs/logs/ft_official_blind_001324_single.json \
  --out-preds-dir runs/ft_official_preds_001324_single --no-pipeline || exit 2

echo
echo "############ 4) 生成 FPV vs official 对比表 ############"
# FPV 侧报告是历史产物；若专用副本不存在则从通用名补一份（避免被 official 结果覆盖）
[ -f runs/logs/ft_fpv_golden_pool_single.json ] || cp runs/logs/ft_golden_pool_single.json runs/logs/ft_fpv_golden_pool_single.json
[ -f runs/logs/ft_fpv_blind_001324_single.json ]   || cp runs/logs/ft_blind_001324_single.json runs/logs/ft_fpv_blind_001324_single.json
python manu/data/compare_lq_baselines.py \
  --fpv runs/logs/ft_fpv_golden_pool_single.json runs/logs/ft_fpv_blind_001324_single.json \
  --official runs/logs/ft_official_golden_pool_single.json runs/logs/ft_official_blind_001324_single.json \
  --out-md runs/logs/lq_baseline_semantics_compare.md \
  --out-json runs/logs/lq_baseline_semantics_compare.json || exit 2

echo
echo "############ 完成 ############"
