#!/usr/bin/env bash
# ===========================================================================
# build_demo.sh -- cross-compile ONLY the TEST PROGRAMS for RK3588.
#
#   outputs:
#     out/mosaic_api_test        the EXAMPLE DEMO third parties copy.
#                                 Calls ONLY the public C ABI:
#                                 open / push / collect / destroy.
#     out/mosaic_client_example  the MINIMAL INTEGRATION example -- the one
#                                 an embedded integrator copies.
#     out/mosaic_contract_test   the library's own API CONTRACT test.
#     out/rknn_mosaic_e2e        the END-TO-END test program: measures
#                                 throughput / latency (the 12.9 fps number).
#                                 Calls ONLY the public C ABI.
#
# All four programs link out/libmosaic_pipeline.a, so run build_lib.sh FIRST.
#
# This script is a self-contained extraction of the certified
# manu/pipeline/npu/build_mosaic_pipeline_arm64.sh (demo half).  The link
# line is VERBATIM identical to that certified script -- only the source and
# artifact paths changed.
#
#   * links the BOARD's own librknnrt.so (2.3.0) and librga.so.2 (2.1.0).
#   * NEVER -I/usr/include: the host headers are x86 and would poison the
#     build.  Khronos CL headers are staged privately under include/CL.
#   * -ffp-contract=off + explicit fast-math rejection.
#   * export LC_ALL=C: readelf localises its "Shared library:" label.
#
# NOTE: this script deliberately contains NO deploy / scp / ssh logic.
#       Deployment is the operator's job -- see README.md section (3).
#
# Usage: ./build_demo.sh
# ===========================================================================
set -euo pipefail
export LC_ALL=C

die() { echo "[ERROR] $*" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/out"

# ---------------------------------------------------------------------------
# 依赖定位（与 build_lib.sh 完全一致 ⇒ 两条脚本用的是同一套依赖）
#
# 【第三方依赖】默认用**交付包内自带的 third_party/**：
#   third_party/opencv/     交叉编译版 OpenCV（头 + 静态库 .a）
#   third_party/boardlibs/  板卡 libOpenCL.so
#   third_party/rknnrt/     板卡 librknnrt.so / librga.so
# 路径相对脚本自身 ⇒ **拷到哪个目录都能编**。
# 三个环境变量仍可覆盖（换板卡 / 换 BSP 时用）：
#   OPENCV_INSTALL / BOARD_LIBS / RT_LIB_DIR
#
# 【交叉编译器】不在包里（编译器没法随包发），用 RK3588_TOOLCHAIN 指定；
# 可以用 ':' 分隔多个候选，按顺序取第一个真的能用的。
# CROSS_CXX 可直接指定编译器（优先级最高）。
# ---------------------------------------------------------------------------
TC_DEFAULT="/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu"
TC_CANDIDATES="${RK3588_TOOLCHAIN:-$TC_DEFAULT}"

INSTALL="${OPENCV_INSTALL:-$HERE/third_party/opencv}"
LIBDIR="${BOARD_LIBS:-$HERE/third_party/boardlibs}"
RT_LIB_DIR="${RT_LIB_DIR:-$HERE/third_party/rknnrt}"

# 工具链里的编译器名字各家不一样（rockchip / none / 通用 aarch64-linux-gnu）。
# ⚠️ 光看「文件在不在」不够：buildroot 的 `*-g++` 是指向 `*.br_real` 的包装器，
#    改名/错链之后驱动自己就找不到后端 ⇒ 编译时才炸，报错还很难懂。
#    ⇒ 必须**真的跑一下**（预处理一个空文件：能过 = 驱动 + 前端 + 头路径都通）。
cxx_works() { [ -x "$1" ] && "$1" -E -x c++ - </dev/null >/dev/null 2>&1; }

find_cxx() {
  local tc="$1" n c
  for n in aarch64-rockchip-linux-gnu-g++ aarch64-none-linux-gnu-g++ aarch64-linux-gnu-g++; do
    if cxx_works "$tc/bin/$n"; then echo "$tc/bin/$n"; return 0; fi
  done
  # 兜底：bin 下任何 *-g++，且能真的跑起来
  while IFS= read -r c; do
    if cxx_works "$c"; then echo "$c"; return 0; fi
  done < <(find "$tc/bin" -maxdepth 1 -name '*-g++' -type f 2>/dev/null | LC_ALL=C sort)
  return 1
}

TC=""
IFS=':' read -r -a _tc_list <<< "$TC_CANDIDATES"
for _tc in "${_tc_list[@]}"; do
  [ -n "$_tc" ] || continue
  if find_cxx "$_tc" >/dev/null 2>&1; then TC="$_tc"; break; fi
done
[ -n "$TC" ] || die "找不到**可用**的交叉编译器。
       候选路径：$TC_CANDIDATES
       ⇒ 「可用」= 编译器能真的跑起来（驱动 + 前端 + 头路径都通）；
         目录里有 g++ 但跑不起来（常见于 buildroot 的 *.br_real 包装器被改名/错链）不算。
       ⇒ 请设置 RK3588_TOOLCHAIN=<工具链根目录>（可用 ':' 分隔多个，按顺序试），
         或直接设 CROSS_CXX=<aarch64 交叉 g++ 的绝对路径>。"

CXX="${CROSS_CXX:-}"
[ -n "$CXX" ] || CXX="$(find_cxx "$TC")"
[ -x "$CXX" ] || die "cross compiler 不可用：$CXX"

FP_FLAGS="-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations"
TP_LIB="$INSTALL/lib/opencv4/3rdparty"
OPENCV_LIBS="-lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video -lopencv_imgproc -lopencv_core -ltegra_hal -lzlib"

# ---------------------------------------------------------------------------
# 0. environment gate -- fail early and loudly, never half-build
# ---------------------------------------------------------------------------
[ -d "$INSTALL/lib" ] || die "cross OpenCV missing (expected $INSTALL/lib)
       ⇒ 包内默认路径是 <包>/third_party/opencv；若包不完整，跑 make_build_dir.sh 重新打包；
         或用 OPENCV_INSTALL=<交叉编译版 OpenCV 根目录> 指定。"
[ -f "$LIBDIR/libOpenCL.so" ] || die "board libOpenCL not staged in $LIBDIR
       ⇒ 包内默认路径是 <包>/third_party/boardlibs；或用 BOARD_LIBS=<目录> 指定。"
[ -f "$RT_LIB_DIR/librknnrt.so" ] || die "board librknnrt.so not staged in $RT_LIB_DIR
       ⇒ 包内默认路径是 <包>/third_party/rknnrt；或用 RT_LIB_DIR=<目录> 指定。"
[ -f "$RT_LIB_DIR/librga.so" ] || die "board librga.so not staged in $RT_LIB_DIR
       ⇒ 包内默认路径是 <包>/third_party/rknnrt；或用 RT_LIB_DIR=<目录> 指定。"

echo "  工具链   : $TC"
echo "  编译器   : $CXX"
echo "  OpenCV   : $INSTALL"
echo "  板卡库   : $LIBDIR"
echo "  RKNN 库  : $RT_LIB_DIR"

# the library MUST exist first -- that is what we link against
[ -f "$OUT/libmosaic_pipeline.a" ] || \
  die "out/libmosaic_pipeline.a not found -- run 'bash build_lib.sh' FIRST"

[ -f "$HERE/src/npu/mosaic_api_test.cpp" ]      || die "missing source: src/npu/mosaic_api_test.cpp"
[ -f "$HERE/src/npu/mosaic_client_example.cpp" ] || die "missing source: src/npu/mosaic_client_example.cpp"
[ -f "$HERE/src/npu/mosaic_contract_test.cpp" ] || die "missing source: src/npu/mosaic_contract_test.cpp"
[ -f "$HERE/src/npu/rknn_mosaic_e2e.cpp" ]     || die "missing source: src/npu/rknn_mosaic_e2e.cpp"

mkdir -p "$OUT"

LIB_CXXFLAGS="-std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
  $FP_FLAGS -DCL_TARGET_OPENCL_VERSION=300 \
  -I$HERE/include -I$HERE/include/rga -I$HERE/src/cpp -I$HERE/src/opencl/host -I$INSTALL/include/opencv4"

# ---------------------------------------------------------------------------
# 1. the EXAMPLE DEMO (open / push / collect / destroy)
# ---------------------------------------------------------------------------
echo "== building mosaic_api_test (the example demo) =="
"$CXX" $LIB_CXXFLAGS "$HERE/src/npu/mosaic_api_test.cpp" "$HERE/out/libmosaic_pipeline.a" -o "$HERE/out/mosaic_api_test" \
  -L"$RT_LIB_DIR" -lrknnrt -lrga -Wl,-rpath-link,"$RT_LIB_DIR" \
  -L"$LIBDIR" -L"$INSTALL/lib" -L"$TP_LIB" $OPENCV_LIBS -lOpenCL -lpthread -ldl -lm

# ---------------------------------------------------------------------------
# 1b. the MINIMAL INTEGRATION EXAMPLE (what an embedded integrator copies)
#     Same link line as the demo; only the source path differs.
# ---------------------------------------------------------------------------
echo "== building mosaic_client_example (the minimal integration example) =="
"$CXX" $LIB_CXXFLAGS "$HERE/src/npu/mosaic_client_example.cpp" "$HERE/out/libmosaic_pipeline.a" -o "$HERE/out/mosaic_client_example" \
  -L"$RT_LIB_DIR" -lrknnrt -lrga -Wl,-rpath-link,"$RT_LIB_DIR" \
  -L"$LIBDIR" -L"$INSTALL/lib" -L"$TP_LIB" $OPENCV_LIBS -lOpenCL -lpthread -ldl -lm

# ---------------------------------------------------------------------------
# 2. the API CONTRACT test (the library's own gate)
# ---------------------------------------------------------------------------
echo "== building mosaic_contract_test (library contract test) =="
"$CXX" $LIB_CXXFLAGS "$HERE/src/npu/mosaic_contract_test.cpp" "$HERE/out/libmosaic_pipeline.a" -o "$HERE/out/mosaic_contract_test" \
  -L"$RT_LIB_DIR" -lrknnrt -lrga -Wl,-rpath-link,"$RT_LIB_DIR" \
  -L"$LIBDIR" -L"$INSTALL/lib" -L"$TP_LIB" $OPENCV_LIBS -lOpenCL -lpthread -ldl -lm

# ---------------------------------------------------------------------------
# 3. the END-TO-END test program (throughput / latency measurement)
#    Same link line as the certified build_mosaic_pipeline_arm64.sh, only the
#    source and artifact paths changed.
# ---------------------------------------------------------------------------
echo "== building rknn_mosaic_e2e (the end-to-end test program) =="
"$CXX" $LIB_CXXFLAGS "$HERE/src/npu/rknn_mosaic_e2e.cpp" "$HERE/out/libmosaic_pipeline.a" -o "$HERE/out/rknn_mosaic_e2e" \
  -L"$RT_LIB_DIR" -lrknnrt -lrga -Wl,-rpath-link,"$RT_LIB_DIR" \
  -L"$LIBDIR" -L"$INSTALL/lib" -L"$TP_LIB" $OPENCV_LIBS -lOpenCL -lpthread -ldl -lm

echo
echo "== artifacts =="
for f in "$OUT/mosaic_api_test" "$OUT/mosaic_client_example" "$OUT/mosaic_contract_test" "$OUT/rknn_mosaic_e2e"; do
  echo "  path : $f"
  echo "  size : $(stat -c%s "$f") B"
  echo "  md5  : $(md5sum "$f" | cut -d' ' -f1)"
done
echo
echo "[ok] test programs built.  Copy out/* to the NFS mount -- see README.md section (3)."
