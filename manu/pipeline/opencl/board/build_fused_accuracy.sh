#!/usr/bin/env bash
# ===========================================================================
# build_fused_accuracy.sh -- cross-compile board_fused_accuracy for RK3588.
#
# Same reasoning as build_board_probe.sh: the tool dlopens OpenCL, so the only
# build-time requirement is a C++ compiler plus libdl. There is deliberately
# NO OpenCV here -- the reference is shipped as data because no arm64 OpenCV
# exists anywhere in this project (toolchain sysroot, vendor SDK, or board).
#
# Usage: ./build_fused_accuracy.sh [output_dir]
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TC="${RK3588_TOOLCHAIN:-/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu}"
CXX="$TC/bin/aarch64-rockchip-linux-gnu-g++"
OUT_DIR="${1:-$HERE/out}"
STAGE="$HERE/.build-include"

die() { echo "[ERROR] $*" >&2; exit 1; }

[ -x "$CXX" ] || die "cross compiler not found: $CXX"
[ -f "$HERE/board_fused_accuracy.cpp" ] || die "missing source"

# Stage the Khronos headers privately: -I/usr/include would drag the x86 glibc
# headers into the cross build.
if [ ! -d "$STAGE/CL" ]; then
  rm -rf "$STAGE"; mkdir -p "$STAGE"
  [ -d /usr/include/CL ] || die "host OpenCL headers missing: /usr/include/CL"
  cp -r /usr/include/CL "$STAGE/CL"
  echo "[stage] Khronos headers -> $STAGE/CL"
fi

mkdir -p "$OUT_DIR"
"$CXX" \
  -std=c++17 -O2 -Wall -Wextra \
  -DCL_TARGET_OPENCL_VERSION=300 \
  -ffp-contract=off -fno-fast-math \
  -I"$STAGE" -I"$HERE" \
  "$HERE/board_fused_accuracy.cpp" \
  -o "$OUT_DIR/board_fused_accuracy" \
  -ldl

echo "[ok] built $OUT_DIR/board_fused_accuracy"
file "$OUT_DIR/board_fused_accuracy"
