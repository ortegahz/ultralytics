#!/usr/bin/env bash
# ===========================================================================
# build_gmc_ocl_arm64.sh -- cross-compile gmc_stream_ocl.cpp for the RK3588.
#
# This is the board build of the real feature-channel pipeline, not a
# paraphrase of it: the same source file that runs on x86, compiled for aarch64
# and linked against the cross-built OpenCV 4.10.0. gmc_stream_ocl.cpp already
# enumerates ".bmp" among its accepted image extensions, so no source change is
# needed to read the BMP sequence.
#
# Two link-time facts, both measured rather than assumed:
#
#  * OpenCL. ocl_host.h links against the ICD loader (libOpenCL.so.1). The board
#    does have one -- /usr/lib/aarch64-linux-gnu/libOpenCL.so.1, 34 KB -- and it
#    is in ldconfig, even though /etc/OpenCL/vendors/mali.icd names a
#    libMaliOpenCL.so.1 that does not exist. A link-time probe on the board
#    returned `arm_release_ver: g13p0-01eac0 ... context=OK`, so the loader does
#    reach the real Mali driver and no dlopen rewrite is needed.
#
#    The loader .so is staged under rk3588/boardlibs on the x86 host rather than
#    built: the cross sysroot has no OpenCL, and copying it to the board is not
#    an option (1.7 GiB free). At run time the binary resolves it from the
#    board's own ldconfig path.
#
#  * OpenCV. Linked statically against the cross build in
#    opencv-4.10.0-arm64-install, so the board needs no OpenCV installed.
#    tegra_hal (carotene) and zlib come along because core references both
#    unconditionally -- see build_opencv_parity.sh for the details.
#
# The floating-point contract is applied exactly as the x86 CMakeLists pins it:
#   -ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations
# plus the OpenCV build itself uses the same flags.
#
# Usage: ./build_gmc_ocl_arm64.sh
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE="${PIPE_DIR:-/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline}"
OUT="$HERE/out"

TC="${RK3588_TOOLCHAIN:-/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu}"
CXX="$TC/bin/aarch64-rockchip-linux-gnu-g++"
INSTALL="${OPENCV_INSTALL:-/media/manu/1TB-Volume/workspace/opencv-4.10.0-arm64-install}"
LIBDIR="${BOARD_LIBS:-/media/manu/1TB-Volume/rk3588/boardlibs}"
STAGE="$HERE/.build-include"

SRC="$PIPE/cpp/gmc_stream_ocl.cpp"
FP_FLAGS="-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations"
TP_LIB="$INSTALL/lib/opencv4/3rdparty"
OPENCL_LIBS="-lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video -lopencv_imgproc -lopencv_core -ltegra_hal -lzlib -lOpenCL -lpthread -ldl -lm"

die() { echo "[ERROR] $*" >&2; exit 1; }

mkdir -p "$OUT"
[ -x "$CXX" ] || die "cross compiler missing: $CXX"
[ -f "$SRC" ] || die "missing source: $SRC"
[ -d "$INSTALL/lib" ] || die "cross OpenCV missing (run build_opencv_arm64.sh)"
[ -f "$LIBDIR/libOpenCL.so" ] || die "board libOpenCL not staged (see script header)"
[ -f "$PIPE/opencl/kernels/warp_median_fused.cl" ] || die "kernel source missing"

# Khronos headers are staged privately rather than pulled from /usr/include:
# -I/usr/include would drag the x86 glibc headers into an aarch64 build.
if [ ! -d "$STAGE/CL" ]; then
  rm -rf "$STAGE"; mkdir -p "$STAGE"
  [ -d /usr/include/CL ] || die "host OpenCL headers missing"
  cp -r /usr/include/CL "$STAGE/CL"
  echo "[stage] Khronos headers -> $STAGE/CL"
fi

echo "== cross-compiling gmc_stream_ocl.cpp -> aarch64 =="
"$CXX" -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter \
  $FP_FLAGS \
  -DCL_TARGET_OPENCL_VERSION=300 \
  -I"$STAGE" \
  -I"$PIPE/cpp" \
  -I"$INSTALL/include/opencv4" \
  "$SRC" -o "$OUT/gmc_stream_ocl_arm64" \
  -L"$LIBDIR" -L"$INSTALL/lib" -L"$TP_LIB" $OPENCL_LIBS

echo
echo "[ok] $OUT/gmc_stream_ocl_arm64"
file "$OUT/gmc_stream_ocl_arm64" | cut -d, -f1-3
"$TC/bin/aarch64-rockchip-linux-gnu-readelf" -d "$OUT/gmc_stream_ocl_arm64" \
  | grep NEEDED | sed 's/^/    /'
