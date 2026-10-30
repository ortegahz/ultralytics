#!/usr/bin/env bash
# =============================================================================
# measure_post_modes.sh -- 板卡后处理四模式实测（dense / sparse / verify / gpu）
#
# 用的是「实时 pipeline 测试程序」mosaic_api_test（一推一收的真实管线路径），
# 每种模式跑同一批帧，产出：
#   * 逐帧检测 TSV（供 PC 评分器打分 + 命中帧对比）
#   * 汇总行里的「后处理 X ms/帧」（库自报的 backproj_ms / frames）
#
# 部署方式：**走 NFS，不 scp 到板卡**。
#   板卡根分区 /dev/mmcblk0p6 的 ext4 已损坏（errors=continue 挂载，
#   htree 目录块 + journal 损坏）⇒ 板上 >1 MB 的二进制会被写坏
#   （同尺寸不同 md5，2026-10-28 曾出现 libOpenCL.so.1.0.0 的 .dynamic 被清零）。
#   NFS 导出是本机 /run/media/manu/SSD960-LOCAL/nfs ↔ 板卡 /mnt/manu。
#
# 用法：
#   bash measure_post_modes.sh [板卡IP] [帧数]
#   默认：192.168.0.64 600
# =============================================================================
set -uo pipefail

BOARD="${1:-192.168.0.64}"
PORT=22
LIMIT="${2:-600}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPU="$(cd "$HERE/.." && pwd)"
OUT="$NPU/out"
LOCAL_NFS="/run/media/manu/SSD960-LOCAL/nfs"
REMOTE_NFS="/mnt/manu/post_measure"

SEQ="VIDEO00007_19700101_001324"
MODEL="/mnt/manu/ptq/models/p0_off_A_none.rknn"
KERNEL="/root/kernels"
RAW="/mnt/manu/lq_ir_bmp"

SSH="ssh -p $PORT root@$BOARD"

die() { echo "[FATAL] $*" >&2; exit 1; }

# --- 前置检查 ---------------------------------------------------------------
[ -x "$OUT/mosaic_api_test" ] || die "缺 $OUT/mosaic_api_test（先跑 build_mosaic_pipeline_arm64.sh）"
[ -d "$LOCAL_NFS" ] || die "NFS 导出目录不存在：$LOCAL_NFS"
$SSH 'true' >/dev/null 2>&1 || die "板卡 $BOARD 不可达"

STAGE="$LOCAL_NFS/post_measure"
rm -rf "$STAGE" 2>/dev/null || true
mkdir -p "$STAGE"
cp -f "$OUT/mosaic_api_test" "$STAGE/mosaic_api_test"
chmod +x "$STAGE/mosaic_api_test"

echo "== 部署到 NFS（板卡侧 $REMOTE_NFS）=="
$SSH "mkdir -p $REMOTE_NFS" || die "板上建目录失败"
# 板上再拷到 tmpfs：NFS 上跑大二进制慢，而且要反复跑
$SSH "cp -f $REMOTE_NFS/mosaic_api_test /tmp/mosaic_api_test_pm && chmod +x /tmp/mosaic_api_test_pm && md5sum /tmp/mosaic_api_test_pm" \
  || die "拷贝到板卡 tmpfs 失败"
echo "  本地 md5: $(md5sum "$STAGE/mosaic_api_test" | cut -d' ' -f1)"

# --- 逐模式跑 ---------------------------------------------------------------
MODES="${*}"
for mode in dense sparse verify gpu; do
  echo
  echo "===== post-mode = $mode ====="
  TSV_REMOTE="$REMOTE_NFS/${mode}.tsv"
  $SSH "rm -f $TSV_REMOTE"
  # verify 模式跑两遍后处理（自检用），耗时会接近 dense 的两倍
  $SSH "/tmp/mosaic_api_test_pm --raw-root $RAW --sequence $SEQ --model $MODEL \
        --kernel-dir $KERNEL --limit $LIMIT --post-mode $mode --dump-tsv $TSV_REMOTE \
        2>&1 | tail -6" || echo "  [warn] $mode 退出码非 0"
  $SSH "ls -l $TSV_REMOTE 2>/dev/null; md5sum $TSV_REMOTE 2>/dev/null" | sed 's/^/  /'
done

echo
echo "===== 汇总（板卡后处理耗时）====="
for mode in dense sparse verify gpu; do
  printf '  %-8s ' "$mode"
  $SSH "/tmp/mosaic_api_test_pm --raw-root $RAW --sequence $SEQ --model $MODEL \
        --kernel-dir $KERNEL --limit $LIMIT --post-mode $mode 2>/dev/null \
        | grep -o '后处理 [^ ]* ms/帧' | tail -1" || echo "(无输出)"
done

echo
echo "TSV 在本机：$STAGE"
ls -l "$STAGE"/*.tsv 2>/dev/null
