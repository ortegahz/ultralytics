#!/bin/bash
# measure_hm_speed_delivery_verify_C1.sh -- 阶段C1：as-shipped 交付包 2500 帧全量验证
#   跑的是 /mnt/manu/mosaic_sdk/out/ 里**包内自带**的 .so 与探针（RPATH $ORIGIN 自找 .so）
#   目的：① 交付包的运行时回读是否打印全 9 字段
#         ② 换成板上重编 .so 后，融合输出是否与之前**逐字节**相同（功能等价性闸门）
set -uo pipefail
LOG=/root/verifyC1_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "=== [phaseC1] start $(date -Is) ==="
export LD_PRELOAD=/root/mosaic/libstatcompat.so
export LD_LIBRARY_PATH=/mnt/manu/mosaic_sdk/out:/root/mosaic/opencv:/root/mosaic:/root/hm_speed/lib:/lib:/lib/aarch64-linux-gnu:/usr/lib
cd /mnt/manu/mosaic_sdk
./out/mosaic_track_dump --dir /tmp/lqframes --model /mnt/manu/mosaic_sdk/models/p0_off_A_none.rknn \
  --kernel-dir /mnt/manu/mosaic_sdk/runtime/kernels --seq VIDEO00007_19700101_001324 \
  --out-prefix /tmp/ship_td --frames 2500 --npu-workers 1 --thr 0.02 --fusion 1 --tag-mask 7
echo "ship_rc=$?"
ls -l /tmp/ship_td*
echo "--- 融合输出 md5（对比 board_td_fused.tsv = 5a317dab22bb28497e93b3db77927270）---"
md5sum /tmp/ship_td_fused.tsv /tmp/ship_td_cands.tsv
echo "--- 候选输出 md5（对比 board_td_cands.tsv = 0e511a4c66a7f2842c79c0a9f73fe613）---"
cat /sys/kernel/debug/rknpu/load 2>/dev/null
dmesg 2>/dev/null | grep -i "job timeout\|soft reset" | tail -3
echo "=== [phaseC1] done $(date -Is) log=$LOG ==="
