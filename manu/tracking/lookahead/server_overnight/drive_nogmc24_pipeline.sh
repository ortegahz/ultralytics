#!/usr/bin/env bash
# ============================================================
# ★ 24 序 no-GMC **全链路 driver**（host 18:30 关机 ⇒ 必须服务器端自治跑完）
#
#   串起 24 序 no-GMC 从「特征集」到「24 序官方口径交付数字」的完整链路：
#     Step 0  等 host 启动的 24 序 no-GMC 特征集构建跑完（165,223 帧）
#     Step 1  **逐字节等价性自检**：8 条 gate 序列 vs 8 序 no-GMC（md5 必须全同）
#             —— 这是「8 序结论能不能外推到 24 序」的地基
#     Step 2  **发布**：自检通过 ⇒ `.build` 原子改名为正式目录
#     Step 3  **建 24 序 no-GMC 的 ms_cache**（GPU 1/2/3；GPU 0 有别人的进程，不碰）
#     Step 4  **24 序官方口径 Δ 扫描**（cs_t0043；rc2 Δ=2 已完成不重跑）
#     Step 5  重生成严格对比报告（8 序 + 24 序 + Gate A）
#
#   ★ 为什么是「等 + 发布」而不是「自己建」：host 已于 17:37 起建好特征集进程，
#     并发重建会写坏同一目录（脚本 cv2.imwrite 直写、无原子改名）。
#   ★ 为什么 Step 2 失败也要往下跑：即使逐字节自检不过，24 序实测数仍是交付判据，
#     只不过要额外标注「8 序外推前提不成立」。
#
#   启动（服务器）：
#     cd /tmp/pycharm_project_10ae9e2e
#     setsid nohup bash manu/tracking/lookahead/server_overnight/drive_nogmc24_pipeline.sh \
#        > runs/lookahead/nogmc24_pipeline.log 2>&1 < /dev/null &
# ============================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
PY="${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}"
STAGE="${STAGE:-/mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median_nogmc.build}"
CANON="${CANON:-/mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median_nogmc}"
OLD8="${OLD8:-/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc/images/val}"
CACHE="${CACHE:-/mnt/data/siping/runs/nogmc_t3/swa/ms_cache_24seq}"
W="${W:-/mnt/data/siping/runs/nogmc_t3/swa/cs_t0043.pt}"
GPUS="${GPUS:-1 2 3}"
NSHARD="${NSHARD:-3}"
DELTAS="${DELTAS:-3 2 4 8}"
EXPECTED="${EXPECTED:-165223}"
BUILD_LOG="${BUILD_LOG:-$REPO/runs/lookahead/nogmc24_build.log}"
OUT24="${OUT24:-$REPO/runs/lookahead/fi_nogmc24}"

cd "$REPO" || exit 2
log(){ echo "[$(date '+%F %T')] [driver] $*"; }

################################################################
# Step 0 —— 等特征集构建完成
################################################################
log "Step 0 等 24 序 no-GMC 特征集构建完成（期望 $EXPECTED 帧，日志 $BUILD_LOG）"
deadline=$(( $(date +%s) + 10800 ))   # 最多等 3 小时
while :; do
  n=0
  [ -d "$STAGE/images/train" ] && n=$(ls "$STAGE/images/train" 2>/dev/null | wc -l)
  if grep -q "exact count $EXPECTED/$EXPECTED" "$BUILD_LOG" 2>/dev/null; then
    log "Step 0 完成：构建日志已确认 exact count $EXPECTED/$EXPECTED（盘上 $n）"
    break
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "Step 0 超时（3 小时）—— 放弃等待，用现状 $n 帧继续"; break
  fi
  if ! pgrep -f "build_nogmc_rebuild_dataset.py" > /dev/null 2>&1; then
    log "Step 0 构建进程已消失（可能失败）—— 再等 30s 看日志是否落最终行"
    sleep 30
    if grep -q "exact count $EXPECTED/$EXPECTED" "$BUILD_LOG" 2>/dev/null; then
      log "Step 0 完成（延迟确认）"; break
    fi
    log "Step 0 判定为构建失败/未完成（盘上 $n 帧）—— 后续步骤按现状继续"
    break
  fi
  log "Step 0 进行中：$n/$EXPECTED"
  sleep 30
done

################################################################
# Step 1 —— 逐字节等价性自检（8 条 gate 序列）
################################################################
log "Step 1 逐字节等价性自检：新 24 序 no-GMC  vs  8 序 no-GMC（val）"
mkdir -p "$REPO/runs/lookahead"
"$PY" manu/tracking/lookahead/server_overnight/check_nogmc24_byte_equal.py \
    --new "$STAGE/images/train" --old "$OLD8" \
    --out-json "$REPO/runs/lookahead/nogmc24_byte_equal.json"
BYTE_RC=$?
log "Step 1 自检 rc=$BYTE_RC"

################################################################
# Step 2 —— 发布（.build → 正式目录）
################################################################
FEAT_ROOT="$STAGE"
if [ "$BYTE_RC" -eq 0 ]; then
  if [ -d "$CANON" ]; then
    cn=$(ls "$CANON/images/train" 2>/dev/null | wc -l)
    if [ "$cn" -eq "$EXPECTED" ]; then
      log "Step 2 正式目录已存在且帧数=$cn ⇒ 直接用正式目录，保留 .build 备查"
      FEAT_ROOT="$CANON"
    else
      bak="$CANON.partial.$(date +%Y%m%d_%H%M%S)"
      log "Step 2 正式目录帧数=$cn ≠ $EXPECTED ⇒ 改名让位到 $bak"
      mv "$CANON" "$bak" && mv "$STAGE" "$CANON" && FEAT_ROOT="$CANON"
    fi
  else
    log "Step 2 发布：mv $STAGE -> $CANON"
    if mv "$STAGE" "$CANON"; then FEAT_ROOT="$CANON"; else log "Step 2 发布失败，继续用 $STAGE"; fi
  fi
else
  log "Step 2 跳过发布（自检未过）⇒ 特征根用 $STAGE"
fi
log "Step 2 特征根 = $FEAT_ROOT/images/train"

################################################################
# Step 3 —— 建 24 序 no-GMC 的 ms_cache（GPU）
################################################################
log "Step 3 建 24 序 no-GMC 缓存：FEAT=$FEAT_ROOT/images/train OUT=$CACHE GPU=$GPUS shard=$NSHARD"
FEAT="$FEAT_ROOT/images/train" W="$W" OUT="$CACHE" GPUS="$GPUS" NSHARD="$NSHARD" \
  bash manu/tracking/lookahead/server_overnight/rebuild_fpv8_24seq_nogmc24_server.sh
CACHE_RC=$?
NPKL=$(ls "$CACHE"/*.pkl 2>/dev/null | wc -l)
log "Step 3 结束 rc=$CACHE_RC，pkl=$NPKL/24"

################################################################
# Step 4 —— 24 序官方口径 Δ 扫描（cs_t0043）
################################################################
if [ "$NPKL" -eq 24 ]; then
  log "Step 4 24 序 Δ 扫描：DELTAS=$DELTAS"
  CACHE="$CACHE" OUT="$OUT24" DELTAS="$DELTAS" \
    bash manu/tracking/lookahead/server_overnight/run_fastinfer_24seq_nogmc24_sweep.sh
  log "Step 4 结束 rc=$?"
else
  log "Step 4 跳过（缓存 pkl=$NPKL ≠ 24）"
fi

################################################################
# Step 5 —— 重生成严格对比报告
################################################################
log "Step 5 重生成报告"
"$PY" manu/tracking/lookahead/make_fastinfer_nogmc_report.py \
    --root "$REPO/runs/lookahead/fi_nogmc" --root-24 "$OUT24" \
    --gate-a "$REPO/runs/lookahead/gateA/gateA.json" \
    --out-md "$REPO/runs/lookahead/fi_nogmc/report.md" \
    --out-json "$REPO/runs/lookahead/fi_nogmc/report.json"
log "Step 5 结束 rc=$?"

log "==== driver 全链路结束 ===="
log "特征根: $FEAT_ROOT"
log "缓存:   $CACHE (pkl=$(ls "$CACHE"/*.pkl 2>/dev/null | wc -l)/24)"
log "产物:   $OUT24"
find "$OUT24" -name arms.json | sort
