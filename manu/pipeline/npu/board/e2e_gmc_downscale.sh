#!/usr/bin/env bash
# =============================================================================
# e2e_gmc_downscale.sh -- 板卡端到端：GMC 拟合分辨率 320x256 (ds=2) vs 160x128 (ds=4)
#
# 背景：GMC 的拟合分辨率由 gmc_downscale 决定（冻结值 2 => 640x512 先缩到 320x256
# 再做角点检测与 LK 拟合，见 gmc_stream_ocl.cpp 的 FIT PARAMS 注释）。
# 再降 2 倍 => ds=4 => 160x128 => 角点与 LK 的像素量变成 1/4。
#
# ⚠️ 分辨率变了 => 变换变了 => 拼图对齐变了 => 检测结果也会变。
# 所以这个脚本必须同时给出**速度**和**检测结果**，不能只看速度。
#
# 参数与 CANON28_dec_rr_d3 完全一致（同一臂、同一拓扑），只改 --gmc-downscale。
# 三个臂**串行**跑（并发会互相污染 RKNN context）。
#
# 用法： bash e2e_gmc_downscale.sh [e2e二进制]
# =============================================================================
set -uo pipefail

BIN="${1:-/tmp/rknn_mosaic_e2e_pm}"
OUTROOT="/root/psearch/results"
mount | grep -q "on /root/psearch/results type tmpfs" || \
  mount -t tmpfs -o size=1G tmpfs "$OUTROOT"

ARGS=(--raw-root /mnt/manu/lq_ir_bmp --sequence VIDEO00007_19700101_001324
      --model /root/psearch/models/p0_off_A_none.rknn --kernel-dir /root/psearch/kernels
      --limit 0 --input-queue 64 --anchor-step 2 --gmc-phys-anchor 1 --gmc-workers 1
      --jitter-depth 3 --gmc-ring-slack 1 --cpu-affinity 0-7 --npu-workers 2
      --post-workers 1 --pace-ms 40 --gmc-min-stride 1
      --gmc-anchor-threads 4 --gmc-affinity-group 4-7
      --pre-affinity 0-3 --npu-stage-affinity 6-7
      --decoupled-stages 1 --buffer-a-depth 3 --buffer-b-depth 3
      --buf-a-mode dma --pre-threads 2)

for DS in 2 4; do
  ARM="GMCDS${DS}"
  OUT="$OUTROOT/$ARM"
  mkdir -p "$OUT"
  echo "===== $ARM (downscale=$DS) ====="
  timeout 1800 "$BIN" "${ARGS[@]}" --gmc-downscale "$DS" --out "$OUT" \
      > "$OUT/run.log" 2>&1
  echo "  exit=$?"
  grep -E '^(gmc_downscale|gmc_frames|det_frames|fps|npu_util_pct|dropped|features_dropped|detections_dropped)=' \
      "$OUT/timing.txt" 2>/dev/null | sed 's/^/  /'
  grep -E '^(gmc_fit|gmc_total|rknn_run|backproject_cc|post_wait|total_wall) ' "$OUT/timing.txt" 2>/dev/null | sed 's/^/  /'
  grep -E '^(gmc_fit_p50|gmc_fit_p95|gmc_fit_p99|gmc_fit_max|gmc_corner_mean)=' "$OUT/timing.txt" 2>/dev/null | sed 's/^/  /'
done

echo "===== 汇总 ====="
for DS in 2 4; do
  T="$OUTROOT/GMCDS${DS}/timing.txt"
  [ -f "$T" ] && echo "--- downscale=$DS ---" && \
    grep -E '^(fps|gmc_downscale|gmc_frames|det_frames|npu_util_pct)=' "$T" && \
    grep -E '^(gmc_fit|gmc_total|rknn_run|backproject_cc|post_wait|total_wall) ' "$T"
done
