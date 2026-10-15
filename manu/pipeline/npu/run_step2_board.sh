#!/bin/sh
# =============================================================================
# 板卡端 anchor_step=2 全量 GMC 特征生成循环
#
# 本文件是 2026-10-15 ARM GMC 精度评估实验的**板卡侧实际执行脚本**
# （实验记录 §5.1，memory.md §12.7 步骤 1，铁律九登记项）。
# 从板卡 /root/run_step2.sh 取回固化，此前只存在于板卡本地。
#
# 执行环境：RK3588 板卡 evm3588（ssh -p 22 root@192.168.0.64），root 身份。
# 二进制：/root/gmc_stream_ocl  md5 aa99da811356694d66a0bce6de108b92
#         （ELF machine 0xb7 = AArch64，支持 --out-ext bmp）
# 内核：  /root/kernels（warp_median_fused.cl + sortnet_generated.inc）
#
# 用法： scp -P 22 run_step2_board.sh root@192.168.0.64:/root/run_step2.sh
#        ssh -p 22 root@192.168.0.64 'chmod +x /root/run_step2.sh && \
#            nohup /root/run_step2.sh > /mnt/manu/armfeat_s2.log 2>&1 &'
#
# 输入清单：/mnt/manu/npu/out/demo/expected.tsv（24 行 `<seq>\t<期望帧数>`）
#   注意：期望帧数是「官方 val 帧数」，而本脚本每条序列会多产出前 2 帧
#   （板卡 feat k 对应 raw 第 k+1 个文件，官方 val 从 __000003 起算），
#   故总产出 31,661 而非 31,613；组装阶段丢弃 k<2 的 48 帧后对齐。
#
# 产出：/mnt/manu/armfeat_s2/<seq>/feat/<seq>_<k:06d>.bmp（512x512 24bpp）
# 耗时：全量约 75 min（24 序列，37~210 ms/frame，随角点密度变化）
# 结束标记：日志末尾输出 STEP2_ALL_DONE
# =============================================================================
set -u
OUTROOT=/mnt/manu/armfeat_s2
mkdir -p $OUTROOT
while IFS="$(printf '\t')" read -r seq want; do
  [ -n "$seq" ] || continue
  D=/mnt/manu/npu/out/demo/$seq
  RAWD=$D/raw/$seq
  nraw=$(ls "$RAWD" 2>/dev/null | wc -l)
  if [ "$nraw" -eq 0 ]; then echo "[$seq] SKIP no-raw"; continue; fi
  mkdir -p $OUTROOT/$seq/feat
  ( cd $D && /root/gmc_stream_ocl --raw-root raw --sequence "$seq" \
      --anchor-step 2 --out-dir $OUTROOT/$seq/feat --out-ext bmp \
      --fused gpu --kernel-dir /root/kernels --timing > gmc_s2.log 2>&1 )
  rc=$?
  n=$(ls $OUTROOT/$seq/feat/*.bmp 2>/dev/null | wc -l)
  echo "[$seq] rc=$rc feat=$n"
  grep -oE 'anchor_step=[0-9]+|corners mean=[0-9.]+|total=[0-9.]+ ms/frame' $D/gmc_s2.log | tr '\n' ' '
  echo ""
done < /mnt/manu/npu/out/demo/expected.tsv
chown -R 1000:1000 $OUTROOT 2>/dev/null
echo STEP2_ALL_DONE
