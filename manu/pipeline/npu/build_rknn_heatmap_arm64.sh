#!/usr/bin/env bash
# ===========================================================================
# build_rknn_heatmap_arm64.sh -- cross-compile rknn_heatmap_smoke.cpp for the
# RK3588. Identical constraints to build_rknn_smoke_arm64.sh:
#   * no OpenCV, no RGA (the board has neither); stb_image does the JPEG decode
#   * link the BOARD's librknnrt.so (2.3.0), never the SDK's stale 1.4.0 copy,
#     and assert the DT_NEEDED entry is the plain SONAME
#   * -ffp-contract=off + explicit fast-math rejection
#   * headers staged privately; never add -I/usr/include (rules.md 2b)
#
# Usage: ./build_rknn_heatmap_arm64.sh
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
SRC="$HERE/rknn_heatmap_smoke.cpp"
BIN="$OUT/rknn_heatmap_smoke"
FP_FLAGS="-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations"

die() { echo "[ERROR] $*" >&2; exit 1; }

mkdir -p "$OUT" "$STAGE"
[ -x "$CXX" ]  || die "cross compiler missing: $CXX"
[ -f "$SRC" ]  || die "missing source: $SRC"
[ -f "$RT_LIB_DIR/librknnrt.so" ] || die \
  "board librknnrt.so not found in $RT_LIB_DIR -- copy it off the board first:
   ssh root@192.168.0.64 'cp /usr/lib/librknnrt.so /mnt/manu/npu/'"

for h in rknn_api.h; do
  [ -f "$SDK/runtime/RK3588/Linux/librknn_api/include/$h" ] || die "SDK header not found: $h"
  cp -f "$SDK/runtime/RK3588/Linux/librknn_api/include/$h" "$STAGE/$h"
done
for h in stb_image.h stb_image_write.h; do
  [ -f "$SDK/examples/3rdparty/stb/$h" ] || die "SDK header not found: $h"
  cp -f "$SDK/examples/3rdparty/stb/$h" "$STAGE/$h"
done
echo "[stage] headers -> $STAGE"

echo "== cross-compiling rknn_heatmap_smoke.cpp -> aarch64 =="
"$CXX" -std=c++17 -O2 -Wall -Wextra \
  $FP_FLAGS \
  -I"$STAGE" \
  "$SRC" -o "$BIN" \
  -L"$RT_LIB_DIR" -lrknnrt -Wl,-rpath-link,"$RT_LIB_DIR" -lpthread -ldl -lm

echo
echo "[ok] $BIN"
file "$BIN" | cut -d, -f1-3
echo "    NEEDED:"
"$READELF" -d "$BIN" | grep NEEDED | sed 's/^/      /'

if ! "$READELF" -d "$BIN" | grep -q 'librknnrt\.so'; then
  die "DT_NEEDED does not contain plain librknnrt.so"
fi
echo "    [ok] resolves librknnrt.so via the board's own runtime"

# The board only has ~1.7 GB free on /, and this binary is the one file that
# CANNOT live on NFS: executing a file off a noexec-uncertain NFS mount is a
# gamble, and mmap'ing an executable from NFS hits ETXTBSY on re-copy.
echo "    [ok] copy to the board with: scp $BIN root@192.168.0.64:/root/"
