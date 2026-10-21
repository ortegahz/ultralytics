#!/usr/bin/env bash
# =============================================================================
# auto_epoch_gate_check.sh —— 方案 §五 指定的「每个 Epoch 结束后自动触发」门禁入口
#
# 两种调用形式都支持：
#   A) 方案 §五 原样的位置参数形式
#      bash manu/training/auto_epoch_gate_check.sh <CHECKPOINT_PATH> <EPOCH_IDX>
#   B) 旗标形式（与方案 §五 脚本片段里的 --weights/--out-dir/--strict-gate 对齐）
#      bash manu/training/auto_epoch_gate_check.sh \
#           --weights <CKPT> --out-dir <DIR> --strict-gate 1
#
# 实际执行：委托给 manu/training/auto_epoch_gate_check.py（三轨全量门禁）
#   Track1 Anti-UAV 官方大盘（8px 点对点）  ← 方案 §一 的「防宏观遗忘」红线
#   Track2 龙泉山黄金池 7 条（point-in-bbox）← 方案 §一 的「一票否决」门禁
#   Track3 盲测 001324（point-in-bbox）      ← 方案 §一 的观察轨
#   方案 §五 片段里写的 `validate_longquan_ft_dataset.sh --weights/--out-dir/--strict-gate`
#   现已支持该调用形式（见 manu/data/validate_longquan_ft_dataset.sh 头部），
#   但它只覆盖 Track2/3；本入口走三轨全量，因为方案 §一 明确要求 Track1 也要守。
#
# 退出码：0=通过   1=指标未达标（质量熔断）   3=基础设施失败（不是模型退化）
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
export PYTHONPATH=.

WEIGHTS=""
EPOCH_IDX=""
OUT_DIR=""
STRICT_GATE=1
GATE_GPUS="${GATE_GPUS:-}"
GATE_INFER_GPU="${GATE_INFER_GPU:-1}"
GATE_CACHE_GPUS="${GATE_CACHE_GPUS:-1,2,3}"

# ---- A) 位置参数形式 ----
if [ $# -ge 1 ] && [ "${1#--}" = "$1" ]; then
  WEIGHTS="$1"; shift
  if [ $# -ge 1 ] && [ "${1#--}" = "$1" ]; then EPOCH_IDX="$1"; shift; fi
fi
# ---- B) 旗标形式 ----
while [ $# -gt 0 ]; do
  case "$1" in
    --weights)     WEIGHTS="${2:-}"; shift 2 ;;
    --epoch)       EPOCH_IDX="${2:-}"; shift 2 ;;
    --out-dir)     OUT_DIR="${2:-}"; shift 2 ;;
    --strict-gate) STRICT_GATE="${2:-1}"; shift 2 ;;
    --gpus)        GATE_GPUS="${2:-}"; shift 2 ;;
    --infer-gpu)   GATE_INFER_GPU="${2:-}"; shift 2 ;;
    --cache-gpus)  GATE_CACHE_GPUS="${2:-}"; shift 2 ;;
    --help|-h)
      echo "用法:"
      echo "  bash manu/training/auto_epoch_gate_check.sh <CHECKPOINT_PATH> <EPOCH_IDX>"
      echo "  bash manu/training/auto_epoch_gate_check.sh --weights <CKPT> --out-dir <DIR> --strict-gate 1"
      exit 0 ;;
    *) echo "[WARN] 未知参数: $1（忽略）"; shift ;;
  esac
done

if [ -z "$WEIGHTS" ]; then
  echo "[FAIL] 未提供 --weights / <CHECKPOINT_PATH>"; exit 2
fi
if [ ! -f "$WEIGHTS" ]; then
  echo "[FAIL] 权重不存在: $WEIGHTS"; exit 2
fi
if [ -z "$OUT_DIR" ]; then
  OUT_DIR="runs/train/ft_official_dryrun/eval_epoch_${EPOCH_IDX:-0}"
fi
mkdir -p "$OUT_DIR"

echo "=============================================================="
echo " === [EPOCH ${EPOCH_IDX:-?}] 触发 Baseline v2 自动化门禁检验 ==="
echo "=============================================================="
echo "  权重  : $WEIGHTS"
echo "  产物  : $OUT_DIR"
echo "  门禁  : strict-gate=$STRICT_GATE"

CMD=(python manu/training/auto_epoch_gate_check.py
     --weights "$WEIGHTS"
     --out-dir "$OUT_DIR"
     --infer-gpu "$GATE_INFER_GPU"
     --cache-gpus "$GATE_CACHE_GPUS")
[ -n "$GATE_GPUS" ] && CMD+=(--gpus "$GATE_GPUS")
[ "$STRICT_GATE" = "1" ] && CMD+=(--strict-gate)

"${CMD[@]}"
GATE_STATUS=$?

echo
if [ "$GATE_STATUS" -eq 0 ]; then
  echo "🟢 [GATE PASSED] 三轨门禁全过（黄金池守门线完好 + Anti-UAV 守大盘未退化）"
elif [ "$GATE_STATUS" -eq 3 ]; then
  echo "⚠️ [GATE INFRA FAILURE] 门禁因基础设施失败退出（rc=3），**不是**指标未达标；请查 $OUT_DIR 日志"
else
  echo "🔴 [GATE FAILED] Epoch ${EPOCH_IDX:-?} 破坏了门禁线（Track1 退化 / 黄金池 F1<98.5% 或 FP>36）！"
  echo "触发熔断机制，建议调小学习率或回滚！"
fi
echo "  报告: $OUT_DIR/gate_report.json"
exit "$GATE_STATUS"
