#!/usr/bin/env bash
# =============================================================================
# measure_post_variance.sh -- 板卡后处理三模式的「逐帧耗时分布」（含方差）
#
# 用的是 mosaic_api_test（实时 pipeline，一推一收）。
# 每种模式跑同一批帧，输出：
#   * 逐帧后处理耗时的分布：均值 / 标准差 / p50 / p90 / p95 / p99 / min / max
#   * GPU 路径的分段：上传 / kernel / 回读 / 整段（各自分布）
#   * 逐帧检测 TSV（供命中帧对比与 PC 评分器）
#
# ⚠️ 一次只跑一个模式：并发会互相抢 A55 核，测出来的方差是假的。
# ⚠️ 必须等 e2e 跑完再跑本脚本（e2e 是端到端计时基准）。
#
# 用法： bash measure_post_variance.sh [帧数]
# =============================================================================
set -uo pipefail

LIMIT="${1:-600}"
NPU="/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/npu"
LOCAL_NFS="/run/media/manu/SSD960-LOCAL/nfs"
STAGE="$LOCAL_NFS/post_measure"
REMOTE="/mnt/manu/post_measure"

SSH="ssh -p 22 root@192.168.0.64"
SEQ="VIDEO00007_19700101_001324"
MODEL="/mnt/manu/ptq/models/p0_off_A_none.rknn"
KERNEL="/root/kernels"
RAW="/mnt/manu/lq_ir_bmp"

[ -x "$NPU/out/mosaic_api_test" ] || { echo "[FATAL] 缺 $NPU/out/mosaic_api_test"; exit 1; }
mkdir -p "$STAGE"
cp -f "$NPU/out/mosaic_api_test" "$STAGE/mosaic_api_test"
chmod +x "$STAGE/mosaic_api_test"
$SSH "cp -f $REMOTE/mosaic_api_test /tmp/mosaic_api_test_pm && chmod +x /tmp/mosaic_api_test_pm"
echo "板端 md5: $($SSH 'md5sum /tmp/mosaic_api_test_pm' | cut -d' ' -f1)"
echo "本地 md5: $(md5sum "$STAGE/mosaic_api_test" | cut -d' ' -f1)"
echo

for MODE in dense sparse gpu; do
  echo "===== $MODE ====="
  $SSH "/tmp/mosaic_api_test_pm --raw-root $RAW --sequence $SEQ --model $MODEL \
        --kernel-dir $KERNEL --limit $LIMIT --post-mode $MODE --dump-tsv $REMOTE/${MODE}_v2.tsv \
        2>&1 | grep -E '后处理|完成 '" | sed 's/^/  /'
done

echo
echo "===== 分段（GPU 模式）====="
$SSH "grep 'gpu post:' $REMOTE/gpu_v2.tsv.log 2>/dev/null | tail -2" || true
