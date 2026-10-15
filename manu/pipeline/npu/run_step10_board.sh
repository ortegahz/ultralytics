#!/bin/sh
# =============================================================================
# 板卡端 anchor_step=10 全量 GMC 特征生成循环
#
# anchor_step=10 是 e2e demo（run_hm_demo.sh）实际使用的参数：
#   锚点网格 {2,12,22,32,42}，共 5 个锚点，其余 lag 由缓存 stride 连乘（最大深度 4）。
#   与 anchor_step=2（零连乘、逐 lag 直接拟合）**不等价**：
#   memory/cpp_port_rk3588.md 实测 400 帧中 354 帧特征不同，两序列均自第 23 帧起分歧
#   （第 0~22 帧为冷启动钳制区，step2 与 step10 全同）。
#   PC 侧对应实现的实测值为 F1 0.906050 / ΔF1 −0.000311（tree 臂，引用值）。
#
# 本脚本是 2026-10-15 ARM GMC 精度评估第二轮的板卡侧执行脚本
# （实验记录 §5.1，memory.md §13，rules.md 铁律九登记项）。
# 与 run_step2_board.sh（step=2 版）仅 `--anchor-step` 与输出目录不同。
#
# 执行环境：RK3588 板卡 evm3588（ssh -p 22 root@192.168.0.64），root 身份。
# 二进制：/root/gmc_stream_ocl  md5 aa99da811356694d66a0bce6de108b92
#         （ELF machine 0xb7 = AArch64，支持 --out-ext bmp）
# 内核：  /root/kernels（warp_median_fused.cl + sortnet_generated.inc）
#
# 用法： scp -P 22 run_step10_board.sh root@192.168.0.64:/root/run_step10.sh
#        ssh -p 22 root@192.168.0.64 'chmod +x /root/run_step10.sh && \
#            nohup /root/run_step10.sh > /mnt/manu/armfeat_s10.log 2>&1 &'
#
# 输入清单：/mnt/manu/npu/out/demo/expected.tsv（24 行 `<seq>\t<官方帧数>`）
# 产出：/mnt/manu/armfeat_s10/<seq>/feat/<seq>_<k:06d>.bmp（512x512 24bpp）
#       命名映射与 step=2 相同：<seq>_<k> -> 官方 <seq>__<k+1>，组装阶段丢弃 k<2 的 48 帧
# 结束标记：日志末尾输出 STEP10_ALL_DONE
# =============================================================================
set -u
OUTROOT=/mnt/manu/armfeat_s10
mkdir -p $OUTROOT
while IFS="$(printf '\t')" read -r seq want; do
  [ -n "$seq" ] || continue
  D=/mnt/manu/npu/out/demo/$seq
  RAWD=$D/raw/$seq
  nraw=$(ls "$RAWD" 2>/dev/null | wc -l)
  if [ "$nraw" -eq 0 ]; then echo "[$seq] SKIP no-raw"; continue; fi
  mkdir -p $OUTROOT/$seq/feat
  ( cd $D && /root/gmc_stream_ocl --raw-root raw --sequence "$seq" \
      --anchor-step 10 --out-dir $OUTROOT/$seq/feat --out-ext bmp \
      --fused gpu --kernel-dir /root/kernels --timing > gmc_s10.log 2>&1 )
  rc=$?
  n=$(ls $OUTROOT/$seq/feat/*.bmp 2>/dev/null | wc -l)
  echo "[$seq] rc=$rc feat=$n"
  grep -oE 'anchor_step=[0-9]+|corners mean=[0-9.]+|total=[0-9.]+ ms/frame' $D/gmc_s10.log | tr '\n' ' '
  echo ""
done < /mnt/manu/npu/out/demo/expected.tsv
chown -R 1000:1000 $OUTROOT 2>/dev/null
echo STEP10_ALL_DONE
