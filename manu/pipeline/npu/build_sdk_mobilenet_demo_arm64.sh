#!/usr/bin/env bash
# ===========================================================================
# build_sdk_mobilenet_demo_arm64.sh -- cross-build Rockchip's own
# rknn_mobilenet_demo, the same way build_sdk_demo_arm64.sh builds the yolov5
# demo.
#
# Why a second SDK demo: the yolov5 demo is a DETECTOR (7.2 GFLOPs @ 640).
# The mobilenet demo is a 0.57 GFLOPs @ 224 CLASSIFIER, i.e. a ~13x lighter
# workload. Together they bracket the board's achievable GOPS and let us
# tell "the board is slow" apart from "this particular graph is slow".
#
# Same build invariants as build_sdk_demo_arm64.sh:
#   * out-of-tree build dir (.demo-build-mobilenet) -> SDK tree stays clean
#   * -DCMAKE_POLICY_VERSION_MINIMUM=3.5 for CMake 4.x vs cmake_minimum_required(3.4.1)
#   * SDK's own prebuilt aarch64 OpenCV 3.4.5 (STATIC) -> board needs no OpenCV
#   * NO `make install` -> the board's 2.3.0 librknnrt is not shadowed
#
# Usage: ./build_sdk_mobilenet_demo_arm64.sh
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK="${RKNN_SDK:-/media/manu/1TB-Volume/rk3588/rk3588_sdk/EVM3588-A_20250815/02Linux软件资料/SDK&Image/SDK/evm3588_linux_sdk_v2.1.1.0/external/rknpu2}"
DEMO="$SDK/examples/rknn_mobilenet_demo"
BUILD="$HERE/.demo-build-mobilenet"

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
echo "[ok] $BUILD/rknn_mobilenet_demo"
file "$BUILD/rknn_mobilenet_demo" | cut -d, -f1-3
TC="${RK3588_TOOLCHAIN:-/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu}"
"$TC/bin/aarch64-rockchip-linux-gnu-readelf" -d "$BUILD/rknn_mobilenet_demo" | grep NEEDED | sed 's/^/    /'

cat <<'EOF'

Deploy & run (binary must live on the board's own disk, not the NFS mount):
  ./run_sdk_demos_board.sh
EOF
