#!/usr/bin/env bash
# ===========================================================================
# build_board_probe.sh -- cross-compile board_cl_probe.cpp for RK3588 (aarch64)
#
# The probe is deliberately self-contained: it resolves OpenCL through dlopen at
# run time, so the only thing it needs from the toolchain is a C++ compiler and
# libdl. There is no -lOpenCL here, because the x86 cross toolchain's sysroot
# ships no libOpenCL and the vendor SDK carries only a buildroot recipe.
#
# Usage:
#   ./build_board_probe.sh [output_dir]
#
# Default output_dir is ./out next to this script.
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Rockchip buildroot toolchain, per manu/memory/rules.md sec. 2b. It lives only
# on the x86 host; never copy it onto the board (1.7 GiB free there).
TC="${RK3588_TOOLCHAIN:-/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu}"
CXX="$TC/bin/aarch64-rockchip-linux-gnu-g++"
OUT_DIR="${1:-$HERE/out}"
STAGE="$HERE/.build-include"

die() { echo "[ERROR] $*" >&2; exit 1; }

[ -x "$CXX" ] || die "cross compiler not found or not executable: $CXX"
[ -f "$HERE/board_cl_probe.cpp" ] || die "missing source: $HERE/board_cl_probe.cpp"

# --- stage the Khronos headers ------------------------------------------------
# They are plain C declarations with no architecture-specific content, so the
# host copy is correct for aarch64. They are staged into a private directory
# instead of pointing -I at /usr/include, because that would drag the x86 glibc
# headers into the cross build and corrupt it in a way that only shows up as
# bizarre compile errors.
rm -rf "$STAGE"
mkdir -p "$STAGE"
[ -d /usr/include/CL ] || die "host OpenCL headers missing: /usr/include/CL"
cp -r /usr/include/CL "$STAGE/CL"
echo "[stage] Khronos headers -> $STAGE/CL"

# --- compile ------------------------------------------------------------------
mkdir -p "$OUT_DIR"

# -O2 with no fast-math. The probe does no float arithmetic, but keeping the
# flags clean means the same recipe stays safe if the kernel work moves in here.
"$CXX" \
  -std=c++17 -O2 -Wall -Wextra \
  -DCL_TARGET_OPENCL_VERSION=300 \
  -ffp-contract=off -fno-fast-math \
  -I"$STAGE" \
  "$HERE/board_cl_probe.cpp" \
  -o "$OUT_DIR/board_cl_probe" \
  -ldl

echo "[ok] built $OUT_DIR/board_cl_probe"
file "$OUT_DIR/board_cl_probe"
echo
echo "Expected ELF: ARM aarch64, dynamically linked, interpreter /lib/ld-linux-aarch64.so.1"
echo "Note: this binary still needs an arm64 glibc >= the toolchain's 2.29 at run time;"
echo "      the board is Ubuntu, so that direction is safe."
