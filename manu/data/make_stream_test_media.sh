#!/usr/bin/env bash
# 生成嵌入式组推流测试素材：原始 mp4 + 按 BMP 处理链裁剪的 640x512 mp4
#
# BMP 处理链（manu/data/extract_video_frames.py + manu/pipeline/npu/ptq/39_convert_lq_ir_to_bmp.py）：
#   1) cv2.cvtColor(frame, COLOR_BGR2GRAY)   # BT.601 luma，先转灰度
#   2) crop ROI = x=320, y=28, w=1280, h=1024  # 去掉 1920x1080 红外画面的黑边/OSD
#   3) cv2.resize(roi, (640, 512), INTER_AREA)  # 2 倍面积平均缩放
#   4) 存 JPG(q=95) -> 再转 8bit 灰度 BMP(BI_RGB, 256 级灰度调色板)
#
# 用法: bash make_stream_test_media.sh <源.h264> <输出目录> [jpg序列目录]
set -euo pipefail

SRC="${1:?用法: $0 <源.h264> <输出目录> [jpg序列目录]}"
OUT="${2:?用法: $0 <源.h264> <输出目录> [jpg序列目录]}"
JPG_DIR="${3:-}"

SEQ="$(basename "${SRC%.*}")"
mkdir -p "$OUT"

# ---- 1. 原始 mp4：裸 H.264 elementary stream 无损封装（-c copy，像素级无损）----
ffmpeg -hide_banner -loglevel error -y -f h264 -i "$SRC" -c copy -movflags +faststart \
  "$OUT/${SEQ}_original_1920x1080.mp4"

# ---- 2. 640x512 视频：直接由 BMP 处理链产出的 JPG 序列编码，保证与 BMP 同源 ----
# 若未给 JPG 序列目录，则从源视频按 BMP 链重新抽帧（等价路径）
if [[ -z "$JPG_DIR" ]]; then
  JPG_DIR="$(mktemp -d)/frames"
  echo "[INFO] 未提供 JPG 序列，从源视频按 BMP 链重新抽帧 -> $JPG_DIR"
  python3 - "$SRC" "$JPG_DIR" <<'PY'
import sys, cv2, os
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
cap = cv2.VideoCapture(src)
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = frame[28:28 + 1024, 320:320 + 1280]
    out = cv2.resize(roi, (640, 512), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(dst, f"frame_{i:06d}.jpg"), out,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    i += 1
cap.release()
print(f"[DONE] {i} frames -> {dst}")
PY
fi

# 2a. 像素级无损灰度版（用于逐帧比对 BMP，maxdiff 应为 0）
ffmpeg -hide_banner -loglevel error -y -framerate 25 -start_number 0 \
  -i "$JPG_DIR/frame_%06d.jpg" -c:v libx264 -preset medium -qp 0 -pix_fmt gray \
  -threads 4 -movflags +faststart "$OUT/${SEQ}_640x512_lossless_gray.mp4"

# 2b. 推流用版（yuv420p，板端 VPU 硬解友好）
ffmpeg -hide_banner -loglevel error -y -framerate 25 -start_number 0 \
  -i "$JPG_DIR/frame_%06d.jpg" -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
  -threads 4 -movflags +faststart "$OUT/${SEQ}_640x512.mp4"

echo "[DONE] 输出目录: $OUT"
ls -la "$OUT"
