#!/bin/bash
# board_prepare_frames.sh -- RK3588 board: extract 8-bit gray BMPs from the Longquan clip.
#
# 执行机：RK3588 板卡（root@192.168.0.64）
# 用法：bash board_prepare_frames.sh <video.mp4> <out_dir> [nb_frames]
#
# 探针/转储只吃 8bit 灰度 BMP；ffmpeg 的 %06d 从 000001 起 ⇒ 源帧号 = 文件名 - 1。
set -uo pipefail

MP4=${1:?video path}
OUT=${2:?out dir}
N=${3:-0}

rm -rf "$OUT"; mkdir -p "$OUT"
if [ "$N" -gt 0 ]; then
  ffmpeg -hide_banner -loglevel error -i "$MP4" -vf format=gray -vsync 0 -frames:v "$N" "$OUT/%06d.bmp"
else
  ffmpeg -hide_banner -loglevel error -i "$MP4" -vf format=gray -vsync 0 "$OUT/%06d.bmp"
fi
echo "frames=$(ls "$OUT" | wc -l)"
du -sh "$OUT"
