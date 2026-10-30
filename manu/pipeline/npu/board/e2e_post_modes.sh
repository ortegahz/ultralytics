#!/usr/bin/env bash
# =============================================================================
# e2e_post_modes.sh -- 板卡端到端三模式对比（dense / sparse / gpu）
#
# 目的：量化后处理算法对**端到端吞吐**的影响。
#   后处理在解耦管线里跑在 POST 线程上；NPU 线程要等它出结果（post_wait）。
#   dense 路径每帧 37.8 ms（抢 A55 核），sparse 0.5 ms，gpu 0.9 ms ——
#   端到端 fps 的差别才是「值不值得换」的回答。
#
# 参数与 CANON28_dec_rr_d3 完全一致（同一臂、同一拓扑），只改后处理算法。
# 三个臂**串行**跑（并发会互相污染 RKNN context）。
#
# 用法： bash e2e_post_modes.sh [e2e二进制]
# =============================================================================
set -uo pipefail

BIN="${1:-/tmp/rknn_mosaic_e2e_pm}"
OUTROOT="/root/psearch/results"
# 结果目录挂在 tmpfs（板卡根分区 ext4 已坏，>1 MB 的文件会被写坏）
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

for MODE in dense sparse gpu; do
  ARM="POSTM_${MODE}"
  OUT="$OUTROOT/$ARM"
  mkdir -p "$OUT"
  echo "===== $ARM ====="
  MOSAIC_POST_MODE="$MODE" timeout 1800 "$BIN" "${ARGS[@]}" --out "$OUT" \
      > "$OUT/run.log" 2>&1
  echo "  exit=$?"
  grep -E '^(backproject_cc|post_wait|fps|total_wall|npu_util_pct|frames)=' "$OUT/timing.txt" 2>/dev/null | sed 's/^/  /'
  grep -E '^(backproject_cc|post_wait|fps|total_wall) ' "$OUT/timing.txt" 2>/dev/null | sed 's/^/  /'
done
echo "===== 汇总 ====="
for MODE in dense sparse gpu; do
  T="$OUTROOT/POSTM_${MODE}/timing.txt"
  [ -f "$T" ] && echo "--- $MODE ---" && grep -E 'backproject_cc|post_wait|^fps|^total_wall|npu_util_pct' "$T"
done
