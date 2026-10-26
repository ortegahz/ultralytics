#!/usr/bin/env bash
# =============================================================================
# run_ptq_pipeline.sh -- RK3588 PTQ (post-training quantisation) pipeline, one command
# =============================================================================
# 做什么（一条命令跑完，5 段）：
#   1) 01_build_calibration_sets.py   4 组 × 250 张无损 BMP 校准集
#   2) 02_batch_ptq_pc_screening.py   12 个 RKNN 配置 + PC 模拟器初筛 → 打印筛选表 + Top 2
#   3) 03_make_fast_eval_manifest.py  001324_sub2000 + golden_pool_sub1200 + 全量 BMP 铺量
#   4) 04_run_board_fast_eval.py      Top 2 上板 + PT 同口径 P/R/F1/FP + 仲裁
#   5) 05_ladder_verification.py      阶梯校验链 PT→ONNX→f16→int8 逐级可归因
#
# -----------------------------------------------------------------------------
# 🔴 关键常量说明：RKNN 的 mean_values / std_values 必须与 YOLO 预处理严格对应
# -----------------------------------------------------------------------------
# 本模型（trial_0080 = YOLO26HeatmapDetector）的输入是**预计算好的 3 通道特征图**，
# 不是原始灰度图。三个通道的磁盘顺序与模型顺序相反，链条如下：
#
#   磁盘（特征图）      [ I_t , GMC_diff , median_residual ]   ← build_full_median_dataset.py 写出
#   模型输入（forward） [ median_residual , GMC_diff , I_t ]   ← 训练时的顺序
#   板端 NPU 输入      uint8 NHWC，字节序 = reverse(cv2.imread(file))
#
# 关键事实：**stbi_load(file) 恰好等于 reverse(cv2.imread(file))**（stb_image 固定输出
# RGB，OpenCV 固定输出 BGR）。而训练管线做的是 cv2.imread 之后 [::-1]，也就是
# reverse(cv2.imread)。所以只要 BMP 是用
#       cv2.imwrite(dst, cv2.imread(src))
# 写出来的，板端喂进 NPU 的字节与 PC 端喂给 PyTorch 的张量**逐字节相同**。
# 中间若放 JPEG，libjpeg-turbo 与 stb_image 的解码结果不保证一致，量化损失就会被
# 解码差异污染 —— 这就是全部中间产物一律用 BMP 的原因（见 ptq_common.py 像素契约）。
#
# 归一化：
#   RKNN : mean_values=[[0,0,0]], std_values=[[255,255,255]]
#          ⇒ 图内烘焙 (x-0)/255，NPU 吃**原始 uint8 RGB**
#   PyTorch: uint8/255（float32）
#   ⇒ 两者数值相同，host 与板端看到的是同一批数字，下游任何差异都可归因于 NPU。
#
# 若把 mean/std 写成别的值（例如 mean=[[0.5,0.5,0.5]]），NPU 看到的就不是
# uint8/255，而是 (x-0.5)/255 —— 与训练分布差 0.5 个强度级，量化器会在错误的
# 动态范围上标定，表现为"板端整体偏暗/偏亮、目标峰值掉到阈值以下"。
#
# -----------------------------------------------------------------------------
# 用法
# -----------------------------------------------------------------------------
#   bash manu/pipeline/npu/ptq/run_ptq_pipeline.sh                # 全流程
#   STAGE=1 bash .../run_ptq_pipeline.sh                          # 只跑到校准集
#   STAGE=2 bash .../run_ptq_pipeline.sh                          # 只跑 PTQ 矩阵 + 筛选
#   VARIANT=p0_on bash .../run_ptq_pipeline.sh                    # 换模型变体
#   TOP2="a.rknn b.rknn" bash .../run_ptq_pipeline.sh              # 手动指定上板模型
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 2
export PYTHONPATH=.

STAGE="${STAGE:-all}"
VARIANT="${VARIANT:-p0_off}"
ALGOS="${ALGOS:-none kl_divergence mmse}"
SETS="${SETS:-A B C D}"
JOBS="${JOBS:-4}"
WORK="${WORK:-/mnt/data/siping/ptq_work}"
SCREEN_DIR="$WORK/screen/$VARIANT"
BOARD_NFS="${BOARD_NFS:-/run/media/manu/SSD960-LOCAL/nfs/ptq}"
FAST="$WORK/fast"

echo "=============================================================="
echo " RK3588 PTQ pipeline"
echo "   variant = $VARIANT"
echo "   sets    = $SETS"
echo "   algos   = $ALGOS"
echo "   jobs    = $JOBS   (mmse is memory-heavy and runs in a smaller pool)"
echo "   stage   = $STAGE"
echo "=============================================================="

# ---------------------------------------------------------------- 1. 校准集
if [ "$STAGE" = "all" ] || [ "$STAGE" = "1" ]; then
  echo
  echo "=== [1/5] 校准集：4 组 × 250 张无损 BMP ==="
  python -u manu/pipeline/npu/ptq/ptq_common.py          # 先跑自检闸
  python -u manu/pipeline/npu/ptq/01_build_calibration_sets.py --out "$WORK/calib" \
    || { echo "[FAIL] 校准集构建失败"; exit 3; }
  for k in A B C D; do
    n=$(grep -c . "$WORK/calib/set_$k.txt" 2>/dev/null || echo 0)
    [ "$n" = "250" ] || { echo "[FAIL] set_$k 有 $n 行，必须是 250"; exit 3; }
    echo "  set_$k : $n 行 ✓"
  done
fi

# ---------------------------------------------------------------- 2. PTQ 矩阵 + 初筛
if [ "$STAGE" = "all" ] || [ "$STAGE" = "2" ]; then
  echo
  echo "=== [2/5] PTQ 矩阵 + PC 模拟器初筛 ==="
  python -u manu/pipeline/npu/ptq/02_batch_ptq_pc_screening.py \
    --variant "$VARIANT" --sets $SETS --algos $ALGOS --jobs "$JOBS" \
    --out "$WORK/screen" || { echo "[FAIL] PTQ 矩阵失败"; exit 3; }
  echo
  echo "--- 筛选表（按 heatmap 平均余弦相似度排序）---"
  python - "$WORK/screen/screen_$VARIANT.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
rows = [r for r in d["results"] if r.get("status") == "ok"]
rows.sort(key=lambda r: -(r.get("cos_heatmap_mean") or 0))
print(f"{'tag':30s} {'cos_hm':>9s} {'cos_min':>9s} {'mse_hm':>11s} "
      f"{'F1':>8s} {'R':>8s} {'FP':>6s} {'build_s':>8s}")
for r in rows:
    print(f"{r['variant']}_{r['set']}_{r['algo']:22s} "
          f"{(r.get('cos_heatmap_mean') or 0):9.6f} "
          f"{(r.get('cos_heatmap_min') or 0):9.6f} "
          f"{(r.get('mse_heatmap_mean') or 0):11.3e} "
          f"{(r.get('f1') or 0):8.4f} {(r.get('recall') or 0):8.4f} "
          f"{(r.get('fp') if r.get('fp') is not None else -1):6d} "
          f"{str(r.get('build_seconds')):>8s}")
if len(rows) >= 2:
    print(f"\n[Top 2] 1) {rows[0]['variant']}_{rows[0]['set']}_{rows[0]['algo']}")
    print(f"         2) {rows[1]['variant']}_{rows[1]['set']}_{rows[1]['algo']}")
PY
fi

# ---------------------------------------------------------------- 3. 快速评测清单
if [ "$STAGE" = "all" ] || [ "$STAGE" = "3" ]; then
  echo
  echo "=== [3/5] 快速评测清单 + BMP 铺量 ==="
  python -u manu/pipeline/npu/ptq/03_make_fast_eval_manifest.py \
    --out "$FAST" --bmp-root "$WORK/bmp" --stage-full || exit 3
fi

# ---------------------------------------------------------------- 4. 板端 Tier2
if [ "$STAGE" = "all" ] || [ "$STAGE" = "4" ]; then
  echo
  echo "=== [4/5] 板端快速评测（Top 2 上板 + PT 同口径）==="
  if [ -n "${TOP2:-}" ]; then
    MODELS="$TOP2"
  else
    MODELS=$(ls "$WORK/screen"/*.rknn 2>/dev/null | head -2 | tr '\n' ' ')
  fi
  [ -n "$MODELS" ] || { echo "[FAIL] 没有可上板的 rknn 模型"; exit 3; }
  echo "  上板模型: $MODELS"
  for m in $MODELS; do
    echo "--- $m ---"
    python -u manu/pipeline/npu/ptq/04_run_board_fast_eval.py \
      --models "$m" --variant "$VARIANT" \
      --manifest "$FAST/001324_sub2000.json" --out "$WORK/tier2" \
      --board-nfs "$BOARD_NFS" || exit 3
  done
  echo
  echo "--- 仲裁：在 001324 上 Recall 跌幅最小 且 黄金池 FP 不激增 ---"
  python - "$WORK/tier2" <<'PY'
import json, sys, glob, os
best = None
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*_report.json"))):
    d = json.load(open(p))
    pt = d.get("arms", {}).get("pt_float")
    for name, arm in d.get("arms", {}).items():
        if name == "pt_float" or pt is None:
            continue
        drec = arm["recall"] - pt["recall"]
        dfp = arm["fp"] - pt["fp"]
        df1 = arm["f1"] - pt["f1"]
        print(f"{name:40s} dF1={df1:+.4f} dRecall={drec:+.4f} dFP={dfp:+d}")
        key = (drec, dfp)
        if best is None or key > best[0]:
            best = (key, name, df1, drec, dfp)
if best:
    print(f"\n[胜出] {best[1]}  dF1={best[2]:+.4f} dRecall={best[3]:+.4f} dFP={best[4]:+d}")
PY
fi

# ---------------------------------------------------------------- 5. 阶梯校验
if [ "$STAGE" = "all" ] || [ "$STAGE" = "5" ]; then
  echo
  echo "=== [5/5] 阶梯校验链：PT(p0_on) → PT(p0_off) → ONNX → f16 → int8 ==="
  ARGS=""
  for m in "$WORK/screen/${VARIANT}_A_none.rknn" "$WORK/screen/${VARIANT}_A_kl_divergence.rknn"; do
    [ -f "$m" ] && ARGS="$ARGS --board-model $(basename "$m" .rknn)=$m"
  done
  python -u manu/pipeline/npu/ptq/05_ladder_verification.py \
    --frames "$FAST/tier1_sensitive.json" --out "$WORK/ladder" $ARGS || exit 3
fi

echo
echo "=============================================================="
echo " 流水线结束  $(date '+%F %T')"
echo " 产物："
echo "   校准集   $WORK/calib/"
echo "   矩阵     $WORK/screen/"
echo "   清单     $FAST/"
echo "   板端报告 $WORK/tier2/"
echo "   阶梯     $WORK/ladder/"
echo "=============================================================="
