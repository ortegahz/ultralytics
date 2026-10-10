#!/usr/bin/env bash
# ===========================================================================
# build_opencv_parity.sh -- build gen_opencv_parity_case and opencv_parity for
# both x86 and arm64.
#
# Two binaries with deliberately different inputs:
#
#   gen_opencv_parity_case   x86 only. Needs imgcodecs to decode JPEG, and
#                            imgcodecs cannot be cross-compiled for this board
#                            (the sysroot has no zlib/libpng/libjpeg headers).
#                            Decoding once on x86 and shipping raw pixels means
#                            both sides read identical bytes instead of each
#                            decoding independently and hoping they agree.
#
#   opencv_parity            both. Links the six modules that were cross-built
#                            (core imgproc features2d video calib3d flann) and
#                            nothing else, which also proves the cross build is
#                            complete -- any missing operator is a link error
#                            here rather than a surprise on the board.
#
# The same floating-point contract applies to both, per the project rule.
#
# Usage: ./build_opencv_parity.sh
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/out"

TC="${RK3588_TOOLCHAIN:-/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu}"
CXX="$TC/bin/aarch64-rockchip-linux-gnu-g++"
INSTALL="${OPENCV_INSTALL:-/media/manu/1TB-Volume/workspace/opencv-4.10.0-arm64-install}"

FP_FLAGS="-ffp-contract=off -fno-fast-math"
# tegra_hal IS carotene: OpenCV names the NEON SIMD target `tegra_hal` but
# namespaces its symbols `carotene_o4t`, so it looks like a missing dependency
# rather than a library one. opencv_core references its split/merge HAL entry
# points unconditionally, so any link against the cross build needs this.
# It installs to lib/opencv4/3rdparty/, not lib/ -- ocv_install_target routes
# third-party targets there.
TP_LIB="$INSTALL/lib/opencv4/3rdparty"
# zlib is here for the same reason: even with WITH_JPEG=OFF, core's
# persistence.cpp references gzopen/gzread unconditionally for FileStorage.
# The library file is libzlib.a, hence -lzlib.
ARM_LIBS="-lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video -lopencv_imgproc -lopencv_core -ltegra_hal -lzlib -lpthread -ldl -lm"

die() { echo "[ERROR] $*" >&2; exit 1; }

mkdir -p "$OUT"
[ -x "$CXX" ] || die "cross compiler missing: $CXX"
[ -d "$INSTALL/lib" ] || die "cross OpenCV missing: $INSTALL/lib (run build_opencv_arm64.sh)"
[ -f "$TP_LIB/libtegra_hal.a" ] || die "carotene/tegra_hal missing: $TP_LIB/libtegra_hal.a"
[ -f "$TP_LIB/libzlib.a" ] || die "zlib missing: $TP_LIB/libzlib.a"

echo "== x86: gen_opencv_parity_case (needs imgcodecs) =="
g++ -std=c++11 -O2 $FP_FLAGS $(pkg-config --cflags opencv4) \
    "$HERE/gen_opencv_parity_case.cpp" -o "$OUT/gen_opencv_parity_case" \
    $(pkg-config --libs opencv4)

echo "== x86: opencv_parity (reference) =="
g++ -std=c++11 -O2 $FP_FLAGS $(pkg-config --cflags opencv4) \
    "$HERE/opencv_parity.cpp" -o "$OUT/opencv_parity_x86" \
    $(pkg-config --libs opencv4)

echo "== arm64: opencv_parity (against the cross-built OpenCV 4.10.0) =="
"$CXX" -std=c++11 -O2 $FP_FLAGS \
    -I"$INSTALL/include/opencv4" \
    "$HERE/opencv_parity.cpp" -o "$OUT/opencv_parity_arm64" \
    -L"$INSTALL/lib" -L"$TP_LIB" $ARM_LIBS

echo
echo "[ok]"
for f in gen_opencv_parity_case opencv_parity_x86 opencv_parity_arm64; do
  printf '  %-26s %8s  %s\n' "$f" "$(stat -c %s "$OUT/$f")" "$(file -b "$OUT/$f" | cut -d, -f1-2)"
done
