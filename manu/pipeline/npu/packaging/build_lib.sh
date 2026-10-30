#!/usr/bin/env bash
# ===========================================================================
# build_lib.sh -- cross-compile ONLY the mosaic pipeline LIBRARY for RK3588.
#
#   output: out/mosaic_pipeline.o
#           out/libmosaic_pipeline.a
#
# This script is a self-contained extraction of the certified
# manu/pipeline/npu/build_mosaic_pipeline_arm64.sh (library half).  The
# compile flags are VERBATIM identical to that certified script -- only the
# paths changed (headers are now staged inside this package).
#
#   * -ffp-contract=off + explicit fast-math rejection (bit-exact port).
#   * NEVER -I/usr/include: the host headers are x86 and would poison the
#     build.  Khronos CL headers are staged privately under include/CL.
#   * export LC_ALL=C: readelf localises its "Shared library:" label.
#
# Usage: ./build_lib.sh
# ===========================================================================
set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/out"

TC="${RK3588_TOOLCHAIN:-/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu}"
INSTALL="${OPENCV_INSTALL:-/media/manu/1TB-Volume/workspace/opencv-4.10.0-arm64-install}"
LIBDIR="${BOARD_LIBS:-/media/manu/1TB-Volume/rk3588/boardlibs}"
RT_LIB_DIR="${RT_LIB_DIR:-/run/media/manu/SSD960-LOCAL/nfs/npu}"

CXX="$TC/bin/aarch64-rockchip-linux-gnu-g++"
AR="$TC/bin/aarch64-rockchip-linux-gnu-ar"

FP_FLAGS="-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations"
TP_LIB="$INSTALL/lib/opencv4/3rdparty"
OPENCV_LIBS="-lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video -lopencv_imgproc -lopencv_core -ltegra_hal -lzlib"

die() { echo "[ERROR] $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. environment gate -- fail early and loudly, never half-build
# ---------------------------------------------------------------------------
[ -x "$CXX" ] || die "cross compiler missing: $CXX"
[ -x "$AR" ]  || die "cross ar missing: $AR"
[ -d "$INSTALL/lib" ] || die "cross OpenCV missing (expected $INSTALL/lib)"
[ -f "$LIBDIR/libOpenCL.so" ] || die "board libOpenCL not staged in $LIBDIR"
[ -f "$RT_LIB_DIR/librknnrt.so" ] || die "board librknnrt.so not staged in $RT_LIB_DIR"
[ -f "$RT_LIB_DIR/librga.so" ] || die "board librga.so not staged in $RT_LIB_DIR"
[ -f "$HERE/src/npu/mosaic_pipeline.cpp" ] || die "missing source: src/npu/mosaic_pipeline.cpp"

mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# 1. compile + archive the library
# ---------------------------------------------------------------------------
LIB_CXXFLAGS="-std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
  $FP_FLAGS -DCL_TARGET_OPENCL_VERSION=300 \
  -I$HERE/include -I$HERE/include/rga -I$HERE/src/cpp -I$HERE/src/opencl/host -I$INSTALL/include/opencv4"

echo "== compiling libmosaic_pipeline.a =="
"$CXX" $LIB_CXXFLAGS -c "$HERE/src/npu/mosaic_pipeline.cpp" -o "$OUT/mosaic_pipeline.o"
rm -f "$OUT/libmosaic_pipeline.a"
"$AR" rcs "$OUT/libmosaic_pipeline.a" "$OUT/mosaic_pipeline.o"

echo
echo "== artifact =="
echo "  path : $OUT/libmosaic_pipeline.a"
echo "  size : $(stat -c%s "$OUT/libmosaic_pipeline.a") B"
echo "  md5  : $(md5sum "$OUT/libmosaic_pipeline.a" | cut -d' ' -f1)"
echo
echo "[ok] library built.  Next: bash build_demo.sh"
