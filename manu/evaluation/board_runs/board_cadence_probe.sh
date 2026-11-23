#!/bin/bash
# board_cadence_probe.sh -- RK3588 board: measure the REAL delivery cadence (fps / Δn).
#
# 执行机：RK3588 板卡（root@192.168.0.64）
# 用法（在板上执行）：bash board_cadence_probe.sh <frames_dir> <out_tsv> [duration_s] [seq]
#
#   seq（可选，第 4 个位置参数）：
#     `--seq` 的取值。**它只是一个标签**，不参与选帧（选帧完全靠 `--dir` 扫目录）。
#     默认 = **帧目录 basename**（板端布局恒为 `<root>/<SEQ>`），
#     也可用环境变量 `SEQ_OVERRIDE` 覆盖；第 4 个位置参数优先级最高。
#
# 语义：生产者按源帧率（25 fps）推帧，消费者收结果 —— 与嵌入式 app 的运行方式一致。
# Δn = 两次交付之间的**源帧号**差 = 一次跟踪器更新覆盖了多少个相机帧。
set -uo pipefail

DIR=${1:?frames dir}
OUT=${2:?out tsv}
DURATION=${3:-150}
# `--seq` 只是标签：优先第 4 个位置参数 > 环境变量 SEQ_OVERRIDE > 帧目录 basename。
SEQ_ARG=${4:-${SEQ_OVERRIDE:-}}
if [ -z "$SEQ_ARG" ]; then SEQ_ARG=$(basename "${DIR%/}"); fi
if [ -z "$SEQ_ARG" ]; then
  echo "[FATAL] 无法从帧目录推导序列名（DIR='$DIR'），请显式传入第 4 个位置参数" >&2
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

/root/rknn_mosaic_cadence_probe \
  --dir "$DIR" --model /root/mosaic/runtime/ir_uav_det.rknn \
  --kernel-dir /root/mosaic/runtime/kernels --seq "$SEQ_ARG" \
  --duration "$DURATION" --src-fps 25 --warmup 40 --frames 0 --npu-workers 1 --thr 0.02 \
  --preset realtime --fusion 1 --out "$OUT"
echo "rc=$?"
