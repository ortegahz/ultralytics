#!/usr/bin/env bash
# ===========================================================================
# build_e2e_demo_arm64.sh -- cross-compile the single-process e2e demo.
#
# Merges the two build chains that run_hm_demo.sh used to invoke separately:
#
#   build_gmc_ocl_arm64.sh   -> cross OpenCV 4.10.0 + Khronos CL headers + board libOpenCL
#   build_rknn_heatmap_arm64.sh -> rknn_api.h + stb_image*, board librknnrt.so 2.3.0
#
# Both now land in ONE binary, because e2e_core.cpp #includes the two production
# sources with their main() compiled out. Everything else is unchanged:
#   * no /usr/include (rules.md 2b) -- every header is staged privately
#   * -ffp-contract=off + fast-math rejection, same as the x86 CMakeLists pins
#   * the BOARD's librknnrt.so, never the SDK's stale 1.4.0 copy, and the
#     DT_NEEDED entry is asserted to be the plain SONAME
#
# Usage: ./build_e2e_demo_arm64.sh
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE="${PIPE_DIR:-/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline}"
OUT="$HERE/out"
STAGE="$HERE/.build-include"

SDK="${RKNN_SDK:-/media/manu/1TB-Volume/rk3588/rk3588_sdk/EVM3588-A_20250815/02Linux软件资料/SDK&Image/SDK/evm3588_linux_sdk_v2.1.1.0/external/rknpu2}"
TC="${RK3588_TOOLCHAIN:-/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu}"
RT_LIB_DIR="${RT_LIB_DIR:-/run/media/manu/SSD960-LOCAL/nfs/npu}"
INSTALL="${OPENCV_INSTALL:-/media/manu/1TB-Volume/workspace/opencv-4.10.0-arm64-install}"
LIBDIR="${BOARD_LIBS:-/media/manu/1TB-Volume/rk3588/boardlibs}"

CXX="$TC/bin/aarch64-rockchip-linux-gnu-g++"
READELF="$TC/bin/aarch64-rockchip-linux-gnu-readelf"
SRC_DEMO="$HERE/e2e_demo.cpp"
SRC_CORE="$HERE/e2e_core.cpp"
BIN="$OUT/e2e_demo"
FP_FLAGS="-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations"
TP_LIB="$INSTALL/lib/opencv4/3rdparty"
OPENCV_LIBS="-lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video -lopencv_imgproc -lopencv_core"

die() { echo "[ERROR] $*" >&2; exit 1; }

mkdir -p "$OUT" "$STAGE"
[ -x "$CXX" ]  || die "cross compiler missing: $CXX"
[ -f "$SRC_DEMO" ] || die "missing source: $SRC_DEMO"
[ -f "$SRC_CORE" ] || die "missing source: $SRC_CORE"
[ -f "$RT_LIB_DIR/librknnrt.so" ] || die "board librknnrt.so not found in $RT_LIB_DIR"
[ -d "$INSTALL/lib" ] || die "cross OpenCV missing (run build_opencv_arm64.sh)"
[ -f "$LIBDIR/libOpenCL.so" ] || die "board libOpenCL not staged"

# ---- stage RKNN + stb headers ------------------------------------------------
for h in rknn_api.h; do
  [ -f "$SDK/runtime/RK3588/Linux/librknn_api/include/$h" ] || die "SDK header not found: $h"
  cp -f "$SDK/runtime/RK3588/Linux/librknn_api/include/$h" "$STAGE/$h"
done
for h in stb_image.h stb_image_write.h; do
  [ -f "$SDK/examples/3rdparty/stb/$h" ] || die "SDK header not found: $h"
  cp -f "$SDK/examples/3rdparty/stb/$h" "$STAGE/$h"
done
# ---- stage Khronos CL headers (never -I/usr/include) ------------------------
if [ ! -d "$STAGE/CL" ]; then
  [ -d /usr/include/CL ] || die "host OpenCL headers missing"
  cp -r /usr/include/CL "$STAGE/CL"
fi
echo "[stage] headers -> $STAGE"

echo "== cross-compiling -> aarch64 =="
# Both demos link the SAME e2e_core.cpp, so the serial and the pipelined build
# provably share one copy of the feature/inference frame bodies. That is what
# lets "outputs identical" be a structural claim rather than a hope.
for NAME in e2e_demo e2e_par_demo e2e_par2_demo; do
  SRC_DEMO="$HERE/$NAME.cpp"
  [ -f "$SRC_DEMO" ] || die "missing source: $SRC_DEMO"
  BIN="$OUT/$NAME"
  echo "-- $NAME"
  "$CXX" -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter \
    $FP_FLAGS \
    -DCL_TARGET_OPENCL_VERSION=300 \
    -I"$STAGE" \
    -I"$PIPE/cpp" \
    -I"$PIPE/opencl/host" \
    -I"$INSTALL/include/opencv4" \
    "$SRC_DEMO" "$SRC_CORE" -o "$BIN" \
    -L"$RT_LIB_DIR" -lrknnrt -Wl,-rpath-link,"$RT_LIB_DIR" \
    -L"$LIBDIR" -L"$INSTALL/lib" -L"$TP_LIB" \
    $OPENCV_LIBS -ltegra_hal -lzlib -lOpenCL -lpthread -ldl -lm || die "compile failed: $NAME"

  echo "[ok] $BIN"
  file "$BIN" | cut -d, -f1-3
  "$READELF" -d "$BIN" | grep -q 'librknnrt\.so' \
    || die "$NAME: DT_NEEDED does not contain plain librknnrt.so"
done

echo
echo "    NEEDED (e2e_demo):"
"$READELF" -d "$OUT/e2e_demo" | grep NEEDED | sed 's/^/      /'
echo "    [ok] all three resolve librknnrt.so via the board's own runtime"
echo "    [ok] copy with:"
echo "      scp $OUT/e2e_demo $OUT/e2e_par_demo $OUT/e2e_par2_demo root@192.168.0.64:/root/"
