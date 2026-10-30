#!/usr/bin/env bash
# ===========================================================================
# build_demo.sh -- cross-compile ONLY the TEST PROGRAMS for RK3588.
#
#   outputs:
#     out/mosaic_api_test        the EXAMPLE DEMO third parties copy.
#                                 Calls ONLY the public C ABI:
#                                 open / push / collect / destroy.
#     out/mosaic_contract_test   the library's own API CONTRACT test.
#
# Both programs link out/libmosaic_pipeline.a, so run build_lib.sh FIRST.
#
# This script is a self-contained extraction of the certified
# manu/pipeline/npu/build_mosaic_pipeline_arm64.sh (demo half).  The link
# line is VERBATIM identical to that certified script -- only the source and
# artifact paths changed.
#
#   * links the BOARD's own librknnrt.so (2.3.0) and librga.so.2 (2.1.0).
#   * NEVER -I/usr/include: the host headers are x86 and would poison the
#     build.  Khronos CL headers are staged privately under include/CL.
#   * -ffp-contract=off + explicit fast-math rejection.
#   * export LC_ALL=C: readelf localises its "Shared library:" label.
#
# NOTE: this script deliberately contains NO deploy / scp / ssh logic.
#       Deployment is the operator's job -- see README.md section (3).
#
# Usage: ./build_demo.sh
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

FP_FLAGS="-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations"
TP_LIB="$INSTALL/lib/opencv4/3rdparty"
OPENCV_LIBS="-lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video -lopencv_imgproc -lopencv_core -ltegra_hal -lzlib"

die() { echo "[ERROR] $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. environment gate -- fail early and loudly, never half-build
# ---------------------------------------------------------------------------
[ -x "$CXX" ] || die "cross compiler missing: $CXX"
[ -d "$INSTALL/lib" ] || die "cross OpenCV missing (expected $INSTALL/lib)"
[ -f "$LIBDIR/libOpenCL.so" ] || die "board libOpenCL not staged in $LIBDIR"
[ -f "$RT_LIB_DIR/librknnrt.so" ] || die "board librknnrt.so not staged in $RT_LIB_DIR"
[ -f "$RT_LIB_DIR/librga.so" ] || die "board librga.so not staged in $RT_LIB_DIR"

# the library MUST exist first -- that is what we link against
[ -f "$OUT/libmosaic_pipeline.a" ] || \
  die "out/libmosaic_pipeline.a not found -- run 'bash build_lib.sh' FIRST"

[ -f "$HERE/src/npu/mosaic_api_test.cpp" ]      || die "missing source: src/npu/mosaic_api_test.cpp"
[ -f "$HERE/src/npu/mosaic_contract_test.cpp" ] || die "missing source: src/npu/mosaic_contract_test.cpp"

mkdir -p "$OUT"

LIB_CXXFLAGS="-std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
  $FP_FLAGS -DCL_TARGET_OPENCL_VERSION=300 \
  -I$HERE/include -I$HERE/include/rga -I$HERE/src/cpp -I$HERE/src/opencl/host -I$INSTALL/include/opencv4"

# ---------------------------------------------------------------------------
# 1. the EXAMPLE DEMO (open / push / collect / destroy)
# ---------------------------------------------------------------------------
echo "== building mosaic_api_test (the example demo) =="
"$CXX" $LIB_CXXFLAGS "$HERE/src/npu/mosaic_api_test.cpp" "$HERE/out/libmosaic_pipeline.a" -o "$HERE/out/mosaic_api_test" \
  -L"$RT_LIB_DIR" -lrknnrt -lrga -Wl,-rpath-link,"$RT_LIB_DIR" \
  -L"$LIBDIR" -L"$INSTALL/lib" -L"$TP_LIB" $OPENCV_LIBS -lOpenCL -lpthread -ldl -lm

# ---------------------------------------------------------------------------
# 2. the API CONTRACT test (the library's own gate)
# ---------------------------------------------------------------------------
echo "== building mosaic_contract_test (library contract test) =="
"$CXX" $LIB_CXXFLAGS "$HERE/src/npu/mosaic_contract_test.cpp" "$HERE/out/libmosaic_pipeline.a" -o "$HERE/out/mosaic_contract_test" \
  -L"$RT_LIB_DIR" -lrknnrt -lrga -Wl,-rpath-link,"$RT_LIB_DIR" \
  -L"$LIBDIR" -L"$INSTALL/lib" -L"$TP_LIB" $OPENCV_LIBS -lOpenCL -lpthread -ldl -lm

echo
echo "== artifacts =="
for f in "$OUT/mosaic_api_test" "$OUT/mosaic_contract_test"; do
  echo "  path : $f"
  echo "  size : $(stat -c%s "$f") B"
  echo "  md5  : $(md5sum "$f" | cut -d' ' -f1)"
done
echo
echo "[ok] demo programs built.  Copy out/* to the NFS mount -- see README.md section (3)."
