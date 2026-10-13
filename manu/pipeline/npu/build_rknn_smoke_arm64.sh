#!/usr/bin/env bash
# ===========================================================================
# build_rknn_smoke_arm64.sh -- cross-compile rknn_npu_smoke.cpp for the RK3588.
#
# Design constraints, each one measured on the board rather than assumed:
#
#  * No OpenCV, no RGA. The SDK's rknn_yolov5_demo links both, and the board has
#    neither: OpenCV exists only as the pip cv2.abi3 module (no headers, no
#    linkable .so, no .pc -- see memory/opencv_arm64_parity.md) and librga is
#    absent. So this program decodes JPEG and resizes with the SDK's header-only
#    stb_image / stb_image_resize and links nothing but librknnrt.
#
#  * Link against the board's OWN librknnrt.so, not the SDK's copy. The SDK ships
#    librknnrt.so 1.4.0 (2022-09-09) while the board runs 2.3.0 (2024-11-07);
#    linking the stale one would bake a 1.x ABI into the binary. The board's copy
#    is pulled to the host through the NFS exchange area (it is also what the
#    loader will actually resolve, since its SONAME is plain "librknnrt.so").
#    RT_LIB_DIR overrides the location; verify with `md5sum` that it is
#    56e425dcc7b59ab0e1aec685b178b99f for librknnrt.so v2.3.0.
#
#  * Floating-point contract is the project's: -ffp-contract=off plus explicit
#    rejection of fast-math, matching opencl/board/build_gmc_ocl_arm64.sh.
#
#  * Headers are staged privately from the SDK. Never add -I/usr/include: that
#    drags x86 glibc headers into an aarch64 build (recorded in rules.md 2b).
#
# Usage: ./build_rknn_smoke_arm64.sh
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/out"
STAGE="$HERE/.build-include"

SDK="${RKNN_SDK:-/media/manu/1TB-Volume/rk3588/rk3588_sdk/EVM3588-A_20250815/02Linux软件资料/SDK&Image/SDK/evm3588_linux_sdk_v2.1.1.0/external/rknpu2}"
TC="${RK3588_TOOLCHAIN:-/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu}"
RT_LIB_DIR="${RT_LIB_DIR:-/home/manu/nfs/npu}"

CXX="$TC/bin/aarch64-rockchip-linux-gnu-g++"
READELF="$TC/bin/aarch64-rockchip-linux-gnu-readelf"
SRC="$HERE/rknn_npu_smoke.cpp"
FP_FLAGS="-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations"

die() { echo "[ERROR] $*" >&2; exit 1; }

mkdir -p "$OUT" "$STAGE"
[ -x "$CXX" ]  || die "cross compiler missing: $CXX"
[ -f "$SRC" ]  || die "missing source: $SRC"
[ -f "$RT_LIB_DIR/librknnrt.so" ] || die \
  "board librknnrt.so not found in $RT_LIB_DIR -- copy it off the board first:
   ssh root@192.168.0.64 'cp /usr/lib/librknnrt.so /mnt/manu/npu/'"

# --- stage headers out of the SDK -----------------------------------------
for h in rknn_api.h; do
  [ -f "$SDK/runtime/RK3588/Linux/librknn_api/include/$h" ] || die "SDK header not found: $h"
  cp -f "$SDK/runtime/RK3588/Linux/librknn_api/include/$h" "$STAGE/$h"
done
for h in stb_image.h stb_image_resize.h stb_image_write.h; do
  [ -f "$SDK/examples/3rdparty/stb/$h" ] || die "SDK header not found: $h"
  cp -f "$SDK/examples/3rdparty/stb/$h" "$STAGE/$h"
done
echo "[stage] headers -> $STAGE"

echo "== cross-compiling rknn_npu_smoke.cpp -> aarch64 =="
"$CXX" -std=c++17 -O2 -Wall -Wextra \
  $FP_FLAGS \
  -I"$STAGE" \
  "$SRC" -o "$OUT/rknn_npu_smoke" \
  -L"$RT_LIB_DIR" -lrknnrt -Wl,-rpath-link,"$RT_LIB_DIR" -lpthread -ldl -lm

echo
echo "[ok] $OUT/rknn_npu_smoke"
file "$OUT/rknn_npu_smoke" | cut -d, -f1-3
echo "    NEEDED:"
"$READELF" -d "$OUT/rknn_npu_smoke" | grep NEEDED | sed 's/^/      /'

# Guard: the binary must depend on librknnrt.so by SONAME, not by absolute
# path, otherwise it would try to load the SDK's stale 1.4.0 copy.
if ! "$READELF" -d "$OUT/rknn_npu_smoke" | grep -q 'librknnrt\.so'; then
  die "DT_NEEDED does not contain plain librknnrt.so"
fi
echo "    [ok] resolves librknnrt.so via the board's own runtime"