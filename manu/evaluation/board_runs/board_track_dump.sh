#!/bin/bash
# board_track_dump.sh -- RK3588 board: dump the FUSED tracks of the delivery parameter group.
#
# 执行机：RK3588 板卡（root@192.168.0.64 / evm3588，Ubuntu 20.04.6 focal，aarch64）
# 用法（在板上执行）：
#   bash board_track_dump.sh <frames_dir> <out_prefix> [tag_mask] [frames] [ids_file] \
#                             [cand_dedup_radius] [seq]
#
#   seq（可选，第 7 个位置参数）：
#     `--seq` 的取值。**它只是一个标签**，不参与选帧（选帧完全靠 `--dir` 扫目录），
#     只影响 md5-out 的文件名与前缀，不影响三份对账 TSV。
#     默认 = **帧目录 basename**（板端布局恒为 `<root>/<SEQ>`），
#     也可用环境变量 `SEQ_OVERRIDE` 覆盖；第 7 个位置参数优先级最高。
#
#   cand_dedup_radius（可选，第 6 个位置参数）：
#     **不填** ⇒ 不传该 flag ⇒ 新二进制用自己的默认值（20 px = 交付值），
#                旧二进制（无此开关）也能照常跑（行为 = 合并前的旧行为）。
#     填 `0`  ⇒ 显式关掉跨尺度候选合并（**A/B 对照臂**：FAR 0.3832，红线 12.8 倍）。
#     填 `20` ⇒ 显式开（交付臂：FAR 0.0488 / F1 0.6930591）。
#     ⚠️ 只有**用新源码重编**的二进制才认这个开关；旧二进制收到它会报用法错误。
#
# 前置（板卡上已完成）：
#   * /root/mosaic/runtime/ir_uav_det.rknn   交付模型（md5 1a3faf2f9459cad8a389ffa81810234c）
#   * /root/mosaic/runtime/kernels            GMC OpenCL kernel
#   * /root/mosaic/libstatcompat.so          OpenCV 4.8 的 stat 符号补丁（LD_PRELOAD；
#                                             **文件缺失时脚本只告警并继续**，不退出）
#   * /root/build/libmosaic_pipeline.so       板上重编的 .so（md5 b383291e31ccbd71c46430ca69869360）
#
# ⚠️ **跨尺度候选合并（2026-11-20 新增）需要 .so 与本脚本用的二进制都重编**：
#    合并逻辑在 `mosaic_pipeline.cpp`（进 .so），开关 `--cand-dedup-radius` 在
#    `mosaic_track_dump.cpp`（进二进制）⇒ **两个都要用新源码重编**，否则合并不会生效。
#    交付值 **20 px**（可用带 16~24；**不要**照抄服务器的 768）。
#
# ⚠️ 板卡串行铁律：同一时刻只允许一个 NPU 上下文（并发 ⇒ RKNPU job timeout ⇒ soft reset
#    ⇒ 只能物理断电）。本脚本名不含 rknn_mosaic/e2e_v/measure_hm_speed，服务器侧
#    board_t4_watch 的让路检查**不会**命中 ⇒ 跑之前先 `ps -eo comm | grep -E "rknn|e2e"` 确认空闲。
set -uo pipefail

DIR=${1:?frames dir}
PREFIX=${2:?out prefix}
TAG_MASK=${3:-1}
FRAMES=${4:-2500}
IDS=${5:-}
CAND_DEDUP_RADIUS=${6:-}
# `--seq` 只是标签：优先第 7 个位置参数 > 环境变量 SEQ_OVERRIDE > 帧目录 basename。
SEQ_ARG=${7:-${SEQ_OVERRIDE:-}}
if [ -z "$SEQ_ARG" ]; then SEQ_ARG=$(basename "${DIR%/}"); fi
if [ -z "$SEQ_ARG" ]; then
  echo "[FATAL] 无法从帧目录推导序列名（DIR='$DIR'），请显式传入第 7 个位置参数" >&2
  exit 2
fi

# stat 符号补丁：**存在才 preload**；缺失只告警继续（不因缺文件退出）。
STAT_COMPAT=/root/mosaic/libstatcompat.so
if [ -e "$STAT_COMPAT" ]; then
  export LD_PRELOAD="$STAT_COMPAT"
else
  echo "[WARN] 未找到 $STAT_COMPAT（OpenCV 4.8 的 stat 符号补丁）——不 preload 继续；" \
       "若板端报 undefined symbol: stat，请先在板上编译该补丁。" >&2
fi
export LD_LIBRARY_PATH=/root/build:/root/mosaic/opencv:/root/mosaic:/root/hm_speed/lib:/lib:/lib/aarch64-linux-gnu:/usr/lib

ARGS=(--dir "$DIR" --model /root/mosaic/runtime/ir_uav_det.rknn
      --kernel-dir /root/mosaic/runtime/kernels --seq "$SEQ_ARG"
      --out-prefix "$PREFIX" --frames "$FRAMES" --npu-workers 1 --thr 0.02
      --fusion 1 --tag-mask "$TAG_MASK")
if [ -n "$IDS" ]; then ARGS+=(--ids "$IDS"); fi
# 只在显式给了半径时才传该开关（老二进制不认识它）。
if [ -n "$CAND_DEDUP_RADIUS" ]; then ARGS+=(--cand-dedup-radius "$CAND_DEDUP_RADIUS"); fi

/root/rknn_mosaic_track_dump "${ARGS[@]}"
echo "rc=$?"
ls -l "${PREFIX}"_*.tsv
