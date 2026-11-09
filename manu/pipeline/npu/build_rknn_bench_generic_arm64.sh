#!/usr/bin/env bash
# ===========================================================================
# build_rknn_bench_generic_arm64.sh -- cross-compile rknn_bench_generic.cpp
# for the RK3588.
#
# Why: the SDK's rknn_mobilenet_demo has NO timing loop (it runs the model
# once and prints Top5), and the SDK's rknn_yolov5_demo folds post_process and
# RGA into its "average run" number. This bench gives a clean, model-agnostic
# rknn_run-only figure so the two SDK models can be compared on the same
# footing -- and so a detector of ours can be measured with the SAME harness.
#
# Build invariants copied from build_rknn_smoke_arm64.sh (both are proven to
# run on the board against the board's librknnrt 2.3.0):
#   * link against the BOARD's librknnrt.so (pulled over NFS), not the SDK's
#     stale 1.4.0 copy
#   * headers staged privately from the SDK; never -I/usr/include
#   * project FP contract: -ffp-contract=off + explicit fast-math rejection
#
# Usage: ./build_rknn_bench_generic_arm64.sh
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/out"
STAGE="$HERE/.build-include"

SDK="${RKNN_SDK:-/media/manu/1TB-Volume/rk3588/rk3588_sdk/EVM3588-A_20250815/02Linux软件资料/SDK&Image/SDK/evm3588_linux_sdk_v2.1.1.0/external/rknpu2}"
TC="${RK3588_TOOLCHAIN:-/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu}"
RT_LIB_DIR="${RT_LIB_DIR:-/run/media/manu/SSD960-LOCAL/nfs/npu}"

CXX="$TC/bin/aarch64-rockchip-linux-gnu-g++"
READELF="$TC/bin/aarch64-rockchip-linux-gnu-readelf"
SRC="$HERE/rknn_bench_generic.cpp"
FP_FLAGS="-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations"

die() { echo "[ERROR] $*" >&2; exit 1; }

mkdir -p "$OUT" "$STAGE"
[ -x "$CXX" ] || die "cross compiler missing: $CXX"
[ -f "$SRC" ] || die "missing source: $SRC"
[ -f "$RT_LIB_DIR/librknnrt.so" ] || die "board librknnrt.so not found in $RT_LIB_DIR"

# --- stage the SDK header (private copy; SDK tree untouched) ----------------
cp -f "$SDK/runtime/RK3588/Linux/librknn_api/include/rknn_api.h" "$STAGE/rknn_api.h"
echo "[stage] rknn_api.h -> $STAGE"

echo "== cross-compiling rknn_bench_generic.cpp -> aarch64 =="
"$CXX" -std=c++17 -O2 -Wall -Wextra \
  $FP_FLAGS \
  -I"$STAGE" \
  "$SRC" -o "$OUT/rknn_bench_generic" \
  -L"$RT_LIB_DIR" -lrknnrt -Wl,-rpath-link,"$RT_LIB_DIR" -lpthread -ldl -lm

echo
echo "[ok] $OUT/rknn_bench_generic"
file "$OUT/rknn_bench_generic" | cut -d, -f1-3
echo "    NEEDED:"
"$READELF" -d "$OUT/rknn_bench_generic" | grep NEEDED | sed 's/^/      /'
