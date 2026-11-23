#!/bin/bash
# build_board_native.sh -- 在 **RK3588 板卡上原生**全量编译库 + 板卡验证探针。
#
# 为什么要它：交付包里的 `out/mosaic_track_dump` / `out/mosaic_cadence_probe`
#   是**板卡原生编译**的二进制（零 OpenCV 依赖 ⇒ 三个 OpenCV 变体共用同一份），
#   而 `build_lib.sh` 只编库、**不编探针** ⇒ 探针的重编入口就是本脚本。
#
# 执行机：RK3588 板卡（root@<board>，aarch64，需板卡自带 g++ 与 OpenCV）
# 用法（把本包整棵树放到板卡上，例如 /root/mosaic_sdk）：
#   cd /root/mosaic_sdk && bash build_board_native.sh
#
# 关键点：
#   * 探针链接时用 `-Wl,--allow-shlib-undefined` + 只链 `-lmosaic_pipeline` ⇒ **零 OpenCV NEEDED**
#     ⇒ 同一份二进制可在 4.6 / 4.8 / 4.2 三种板卡上直接用。
#   * 库链接时用板卡自带的 OpenCV 头与库 ⇒ `.so` 的 NEEDED 随板卡 OpenCV 版本变化
#     （4.8 板 ⇒ `.so.408`；4.6 板 ⇒ `.so.406`；4.2 板 ⇒ `.so.42`），**必须按目标板编译**。
#   * `-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations` 与交叉编译保持同一数值口径。
set -euo pipefail

HERE=${HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
OUT=${OUT:-$HERE/out}
SRC=${SRC:-$HERE/src/npu}
CXX=${CXX:-g++}
FP="-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations"

# 板卡 OpenCV 头目录（不固定：官方 BSP 在 /usr/include/opencv4，工程板常在 /root/cv480inc 等）
OCV_INC=${OCV_INC:-}
if [ -z "$OCV_INC" ]; then
  for c in /usr/include/opencv4 /usr/local/include/opencv4 /root/cv480inc /root/cv46inc /root/cv42inc; do
    [ -d "$c" ] && { OCV_INC=$c; break; }
  done
fi
[ -n "$OCV_INC" ] || { echo "[FATAL] 找不到板卡 OpenCV 头目录，请显式指定 OCV_INC=<dir>" >&2; exit 1; }

# 板卡 OpenCV 库目录（`.so` 在哪就在哪）
OCV_LIB=${OCV_LIB:-}
if [ -z "$OCV_LIB" ]; then
  for c in /usr/lib/aarch64-linux-gnu /usr/lib /root/mosaic/opencv; do
    ls "$c"/libopencv_core.so* >/dev/null 2>&1 && { OCV_LIB=$c; break; }
  done
fi
[ -n "$OCV_LIB" ] || { echo "[FATAL] 找不到板卡 OpenCV 库目录，请显式指定 OCV_LIB=<dir>" >&2; exit 1; }

# 其它板卡库目录（RKNN / RGA / OpenCL）
EXTRA_LIB=${EXTRA_LIB:-/usr/lib/aarch64-linux-gnu}
LNK=${LNK:-/root/lnk}

INC="-I$HERE/include -I$HERE/include/rga -I$HERE/src/cpp -I$HERE/src/opencl/host -I$OCV_INC"
FLAGS="-std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function $FP -fPIC -DCL_TARGET_OPENCL_VERSION=300"
mkdir -p "$OUT"

echo "--- 环境 ---"
echo "HERE=$HERE OUT=$OUT OCV_INC=$OCV_INC OCV_LIB=$OCV_LIB"
echo "--- 源文件 md5（交付基线核对）---"
md5sum "$SRC/mosaic_track_fusion.h" "$SRC/mosaic_track_fusion.cpp" \
       "$SRC/mosaic_pipeline.h" "$SRC/mosaic_pipeline.cpp" "$SRC/mosaic_track_dump.cpp"

echo "--- 编译中间产物 ---"
$CXX $FLAGS $INC -c "$SRC/mosaic_pipeline.cpp" -o "$OUT/mosaic_pipeline.o"
$CXX $FLAGS $INC -c "$SRC/mosaic_track_fusion.cpp" -o "$OUT/mosaic_track_fusion.o"

echo "--- 链接共享库 ---"
$CXX -shared -Wl,-soname,libmosaic_pipeline.so -Wl,--allow-shlib-undefined \
  -o "$OUT/libmosaic_pipeline.so" "$OUT/mosaic_pipeline.o" "$OUT/mosaic_track_fusion.o" \
  -L"$OCV_LIB" -L"$EXTRA_LIB" -L"$LNK" \
  -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video \
  -lopencv_imgproc -lopencv_core -lOpenCL -lrknnrt -lrga -lpthread -ldl -lm

echo "--- 链接板卡验证探针（零 OpenCV）---"
for t in mosaic_cadence_probe mosaic_track_dump; do
  $CXX -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function $FP \
    -I"$HERE/include" -I"$SRC" "$SRC/$t.cpp" -o "$OUT/$t" \
    -L"$OUT" -Wl,--allow-shlib-undefined -Wl,-rpath,'$ORIGIN' \
    -lmosaic_pipeline -lpthread -ldl -lm
done

echo "--- 产物 md5 ---"
md5sum "$OUT/libmosaic_pipeline.so" "$OUT/mosaic_track_dump" "$OUT/mosaic_cadence_probe"

echo "--- 跨尺度候选合并是否在库里 ---"
nm -D "$OUT/libmosaic_pipeline.so" | grep -i MergeCandidatesByRadius || echo "SYMBOL MISSING"
echo "--- 新 CLI 开关是否在探针里 ---"
strings "$OUT/mosaic_track_dump" | grep -i "cand-dedup-radius" || echo "FLAG MISSING"
echo "=== BUILD_BOARD_NATIVE_OK ==="
