#!/usr/bin/env bash
# ===========================================================================
# build_probe_only.sh -- cross-compile ONLY the GMC GPU probe (gmc_gpu_probe).
#
# Same flags as build_mosaic_pipeline_arm64.sh's gmc_gpu_probe target, but it
# does NOT rebuild the library or the certified binaries: the probe is a
# standalone measurement tool and must not drag the certified artifacts into
# a rebuild.
#
# Usage: ./build_probe_only.sh
# ===========================================================================
set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE="$(cd "$HERE/.." && pwd)"
OUT="$HERE/out"
STAGE="$HERE/.build-include"

TC="${RK3588_TOOLCHAIN:-/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu}"
INSTALL="${OPENCV_INSTALL:-/media/manu/1TB-Volume/workspace/opencv-4.10.0-arm64-install}"
LIBDIR="${BOARD_LIBS:-/media/manu/1TB-Volume/rk3588/boardlibs}"
TP_LIB="$INSTALL/lib/opencv4/3rdparty"

CXX="$TC/bin/aarch64-rockchip-linux-gnu-g++"
FP_FLAGS="-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations"
OPENCV_LIBS="-lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video -lopencv_imgproc -lopencv_core -ltegra_hal -lzlib"

mkdir -p "$OUT"
[ -x "$CXX" ] || { echo "[ERROR] cross compiler missing: $CXX" >&2; exit 1; }

echo "== building gmc_gpu_probe (standalone GMC feature-prep GPU probe) =="
"$CXX" -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
  $FP_FLAGS -DCL_TARGET_OPENCL_VERSION=300 \
  -I"$STAGE" -I"$STAGE/rga" -I"$PIPE/cpp" -I"$PIPE/opencl/host" -I"$INSTALL/include/opencv4" \
  "$HERE/gmc_gpu_probe.cpp" -o "$OUT/gmc_gpu_probe" \
  -L"$LIBDIR" -L"$INSTALL/lib" -L"$TP_LIB" $OPENCV_LIBS -lOpenCL -lpthread -ldl -lm
echo "[ok] $OUT/gmc_gpu_probe"
file "$OUT/gmc_gpu_probe" | cut -d, -f1-3
echo "    md5: $(md5sum "$OUT/gmc_gpu_probe" | cut -d' ' -f1)"
