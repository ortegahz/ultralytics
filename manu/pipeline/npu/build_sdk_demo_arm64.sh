#!/usr/bin/env bash
# ===========================================================================
# build_sdk_demo_arm64.sh -- cross-build Rockchip's own rknn_yolov5_demo.
#
# Why: the demo is the REFERENCE implementation for this model's output layout.
# When rknn_npu_smoke's boxes looked plausible-but-wrong, the only way to settle
# it was to run Rockchip's own code and compare numbers. Having it buildable is
# what makes that comparison repeatable instead of a one-off.
#
# Build notes (all measured, none assumed):
#
#  * CMake 4.2.3 on this host refuses the demo's `cmake_minimum_required(3.4.1)`.
#    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 restores compatibility without editing
#    the SDK source (keeping the SDK tree byte-identical to its tag).
#
#  * The demo's CMakeLists picks armhf vs aarch64 by testing whether
#    CMAKE_C_COMPILER matches "aarch64", so the cross prefix must keep that
#    substring -- `aarch64-rockchip-linux-gnu-gcc`, not the `aarch64-linux-`
#    prefix (which has no sysroot here).
#
#  * OpenCV is resolved from the SDK's own prebuilt aarch64 tree (3.4.5, STATIC),
#    so the board needs no OpenCV installed -- the same property the project's
#    own opencv-4.10.0-arm64-install provides, but here straight from the SDK.
#
#  * The binary links librknnrt.so by SONAME, so at run time it resolves the
#    BOARD's 2.3.0 runtime, not the stale 1.4.0 the SDK ships. Verified by the
#    demo printing "sdk version: 2.3.0" on the board.
#
#  * `make install` is deliberately NOT run: it would copy the SDK's 1.4.0
#    librknnrt.so next to the binary and shadow the board's 2.3.0.
#
# Usage: ./build_sdk_demo_arm64.sh
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK="${RKNN_SDK:-/media/manu/1TB-Volume/rk3588/rk3588_sdk/EVM3588-A_20250815/02Linux软件资料/SDK&Image/SDK/evm3588_linux_sdk_v2.1.1.0/external/rknpu2}"
DEMO="$SDK/examples/rknn_yolov5_demo"
BUILD="$HERE/.demo-build"

die() { echo "[ERROR] $*" >&2; exit 1; }

[ -d "$DEMO" ] || die "SDK demo not found: $DEMO"
[ -f "$HERE/rk3588-aarch64-npu-toolchain.cmake" ] || die "toolchain file missing"

mkdir -p "$BUILD"
cd "$BUILD"
cmake "$DEMO" \
  -DCMAKE_TOOLCHAIN_FILE="$HERE/rk3588-aarch64-npu-toolchain.cmake" \
  -DTARGET_SOC=rk3588 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -E "Found OpenCV|Configuring done|error" || true

make -j"$(nproc)" 2>&1 | tail -3

echo
echo "[ok] $BUILD/rknn_yolov5_demo"
file "$BUILD/rknn_yolov5_demo" | cut -d, -f1-3
TC="${RK3588_TOOLCHAIN:-/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu}"
"$TC/bin/aarch64-rockchip-linux-gnu-readelf" -d "$BUILD/rknn_yolov5_demo" | grep NEEDED | sed 's/^/    /'

cat <<'EOF'

Deploy & run (binary must live on the board's own disk -- NFS exec gives ETXTBSY):
  RKNN_PASSWORD=... ./run_sdk_demo.sh
EOF