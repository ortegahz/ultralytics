#!/bin/bash
# phaseC2b -- 【ABI/行为等价闸门·修正版】
#   踩坑记录：RUNPATH($ORIGIN) 的搜索顺序在 LD_LIBRARY_PATH **之后**
#             ⇒ phaseC2 里 /root/mosaic/libmosaic_pipeline.so 把 c2_newso/ 的那份盖掉了，
#             两个变体实际都跑了同一个 .so（C2 结果作废）。
#             本版把变体目录放在 LD_LIBRARY_PATH **最前面**，并用
#             LD_TRACE_LOADED_OBJECTS=1 实证实际加载的是哪一个 .so。
#   目的：交付树 09:10 产物 a8d56d26  vs  11:59 源码重编 b383291e 是否行为等价？
set -uo pipefail
LOG=/root/verifyC2b_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
export LD_PRELOAD=/root/mosaic/libstatcompat.so
mkdir -p /root/c2_oldso /root/c2_newso
cp -f /root/delivery_out/mosaic_track_dump /root/c2_oldso/
cp -f /root/delivery_out/mosaic_track_dump /root/c2_newso/
cp -f /root/mosaic/libmosaic_pipeline.so /root/c2_oldso/libmosaic_pipeline.so   # a8d56d26
cp -f /root/build/libmosaic_pipeline.so  /root/c2_newso/libmosaic_pipeline.so   # b383291e
echo "=== [phaseC2b] start $(date -Is) ==="
for v in oldso newso; do
  export LD_LIBRARY_PATH=/root/c2_$v:/root/mosaic/opencv:/root/hm_speed/lib:/lib:/lib/aarch64-linux-gnu:/usr/lib
  echo "############ variant=$v  target=$(md5sum /root/c2_$v/libmosaic_pipeline.so | cut -d' ' -f1) ############"
  echo "--- 实证实际加载的 .so ---"
  LD_TRACE_LOADED_OBJECTS=1 /root/c2_$v/mosaic_track_dump 2>&1 | grep -i "mosaic_pipeline.so" | head -3
  /root/c2_$v/mosaic_track_dump --dir /tmp/lqframes \
    --model /mnt/manu/mosaic_sdk/models/p0_off_A_none.rknn \
    --kernel-dir /mnt/manu/mosaic_sdk/runtime/kernels --seq VIDEO00007_19700101_001324 \
    --out-prefix /tmp/c2b_${v}_td --frames 250 --npu-workers 1 --thr 0.02 --fusion 1 --tag-mask 7 2>&1 \
    | grep -E "track fusion ON|pushed=|pipeline_frames=|FATAL|ERROR"
  echo "--- md5 ($v) ---"
  md5sum /tmp/c2b_${v}_td_fused.tsv /tmp/c2b_${v}_td_cands.tsv
done
echo "=== [phaseC2b] done $(date -Is) log=$LOG ==="
