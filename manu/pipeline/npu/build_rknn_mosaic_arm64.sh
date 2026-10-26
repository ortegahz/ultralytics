#!/usr/bin/env bash
# ===========================================================================
# build_rknn_mosaic_arm64.sh -- cross-compile rknn_mosaic_heatmap.cpp for the
# RK3588.
#
# Same discipline as build_rknn_heatmap_arm64.sh + build_rga_probe_arm64.sh:
#   * no OpenCV; stb_image does the BMP decode
#   * link the BOARD's own librknnrt.so (2.3.0) and librga.so.2 (2.1.0),
#     never the SDK's stale copies, and assert that the DT_NEEDED entries are
#     the plain SONAMEs (a path would silently pin the wrong library)
#   * -ffp-contract=off + explicit fast-math rejection
#   * headers staged privately in .build-include/; NEVER add -I/usr/include
#     (rules.md 2b -- the host's headers are x86 and would poison the build)
#   * -std=gnu++17 (librga's headers use GNU statement expressions)
#   * export LC_ALL=C: readelf localises its "Shared library:" label, and a
#     Chinese label makes the DT_NEEDED assertion below silently pass/fail
#
# Usage: ./build_rknn_mosaic_arm64.sh
# ===========================================================================
set -euo pipefail

# readelf localises its "Shared library:" label; force the C locale so the
# DT_NEEDED assertions below can match it.
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/out"
STAGE="$HERE/.build-include"

SDK="${RKNN_SDK:-/media/manu/1TB-Volume/rk3588/rk3588_sdk/EVM3588-A_20250815/02Linux软件资料/SDK&Image/SDK/evm3588_linux_sdk_v2.1.1.0/external/rknpu2}"
TC="${RK3588_TOOLCHAIN:-/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu}"
RT_LIB_DIR="${RT_LIB_DIR:-/run/media/manu/SSD960-LOCAL/nfs/npu}"
RGA_HDR_DIR="${RGA_HDR_DIR:-$STAGE/rga}"

CXX="$TC/bin/aarch64-rockchip-linux-gnu-g++"
READELF="$TC/bin/aarch64-rockchip-linux-gnu-readelf"
SRC="$HERE/rknn_mosaic_heatmap.cpp"
BIN="$OUT/rknn_mosaic_heatmap"
FP_FLAGS="-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations"

die() { echo "[ERROR] $*" >&2; exit 1; }

mkdir -p "$OUT" "$STAGE"
[ -x "$CXX" ]  || die "cross compiler missing: $CXX"
[ -f "$SRC" ]  || die "missing source: $SRC"

# ---- staged RKNN + stb headers --------------------------------------------
for h in rknn_api.h; do
  [ -f "$SDK/runtime/RK3588/Linux/librknn_api/include/$h" ] || die "SDK header not found: $h"
  cp -f "$SDK/runtime/RK3588/Linux/librknn_api/include/$h" "$STAGE/$h"
done
for h in stb_image.h; do
  [ -f "$SDK/examples/3rdparty/stb/$h" ] || die "SDK header not found: $h"
  cp -f "$SDK/examples/3rdparty/stb/$h" "$STAGE/$h"
done
echo "[stage] headers -> $STAGE"

# ---- staged RGA headers (the board's own set) ------------------------------
[ -d "$RGA_HDR_DIR" ] || die \
  "RGA headers not staged in $RGA_HDR_DIR.
   Fetch them from the board (they differ from the SDK debs):
     mkdir -p $RGA_HDR_DIR
     ssh root@192.168.0.64 'cd /usr/include/rga && tar cf - .' | tar xf - -C $RGA_HDR_DIR"
for h in im2d.h im2d_type.h im2d_single.h im2d_buffer.h im2d_common.h im2d_version.h \
         im2d_expand.h im2d_mpi.h rga.h RgaUtils.h; do
  [ -f "$RGA_HDR_DIR/$h" ] || die "missing staged RGA header: $RGA_HDR_DIR/$h"
done

# ---- staged board libraries -----------------------------------------------
# The board's own builds:
#   librknnrt.so   -> /usr/lib/librknnrt.so          (2.3.0)
#   librga.so.2    -> /usr/lib/aarch64-linux-gnu/librga.so.2 -> librga.so.2.1.0
# The SDK debs do NOT match what the board runs, so both are staged from the
# board into RT_LIB_DIR.
[ -f "$RT_LIB_DIR/librknnrt.so" ] || die \
  "board librknnrt.so not found in $RT_LIB_DIR -- copy it off the board first:
     ssh root@192.168.0.64 'cat /usr/lib/librknnrt.so' > $RT_LIB_DIR/librknnrt.so"
[ -f "$RT_LIB_DIR/librga.so" ] || die \
  "board librga.so not found in $RT_LIB_DIR -- stage it first:
     ssh root@192.168.0.64 'cat /usr/lib/aarch64-linux-gnu/librga.so.2.1.0' > $RT_LIB_DIR/librga.so.2.1.0
     ln -sf librga.so.2.1.0 $RT_LIB_DIR/librga.so"

echo "== cross-compiling rknn_mosaic_heatmap.cpp -> aarch64 =="
# gnu++17 (not c++17): librga's headers use GNU statement expressions.
# No -I/usr/include anywhere: the host headers are x86 and would poison the
# cross build (rules.md 2b).
"$CXX" -std=gnu++17 -O2 -Wall -Wextra \
  $FP_FLAGS \
  -I"$STAGE" \
  -I"$RGA_HDR_DIR" \
  "$SRC" -o "$BIN" \
  -L"$RT_LIB_DIR" -lrknnrt -lrga -Wl,-rpath-link,"$RT_LIB_DIR" -lpthread -ldl -lm

echo
echo "[ok] $BIN"
file "$BIN" | cut -d, -f1-3
echo "    NEEDED:"
"$READELF" -d "$BIN" | grep NEEDED | sed 's/^/      /'

# Strict SONAME assertions: the dependencies must be the bare SONAMEs, not
# paths and not versioned copies that may not exist on the board.
if ! "$READELF" -d "$BIN" | grep -qE 'Shared library: \[librknnrt\.so\]'; then
  die "DT_NEEDED does not contain the plain SONAME librknnrt.so"
fi
if ! "$READELF" -d "$BIN" | grep -qE 'Shared library: \[librga\.so\.2\]'; then
  die "DT_NEEDED does not contain the plain SONAME librga.so.2"
fi
echo "    [ok] resolves librknnrt.so + librga.so.2 via the board's own runtime"

# No host-specific C++ runtime may sneak in: libc++ would be a different
# toolchain's runtime.  libstdc++ from THIS toolchain is fine and expected.
if "$READELF" -d "$BIN" | grep -qE 'Shared library: \[libc\+\+\.so'; then
  die "DT_NEEDED contains libc++ -- that is a host-runtime difference, refuse to ship"
fi
echo "    [ok] no libc++ in DT_NEEDED"

echo "    md5: $(md5sum "$BIN" | cut -d' ' -f1)"
echo
echo "    copy to the board with: scp -P 22 $BIN root@192.168.0.64:/root/"
echo "    run on the board with:"
echo "      /root/rknn_mosaic_heatmap --model /mnt/manu/ptq/models/p0_off_A_none.rknn \\"
echo "                              --list  <list.txt> --out <out_dir> \\"
echo "                              [--start N] [--limit N] [--tiles 0,1,2] \\"
echo "                              [--canvas-mode mosaic|single] \\"
echo "                              [--input-mode set|io-mem-nchw|io-mem-nhwc] \\"
echo "                              [--cpu-affinity 4-7] [--timing <file>]"
