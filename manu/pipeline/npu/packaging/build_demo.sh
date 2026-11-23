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
#     out/mosaic_async_client_example
#                                 the MULTI-THREADED PRODUCER/CONSUMER template
#                                 (camera thread never blocks + strict
#                                 one-push-one-collect + integrator callback).
#     out/mosaic_contract_test   the library's own API CONTRACT test.
#     out/rknn_mosaic_e2e        the END-TO-END test program: measures
#                                 throughput / latency (the 12.9 fps number).
#                                 Calls ONLY the public C ABI.
#
# All four programs link **out/libmosaic_pipeline.so** (共享库),
# so run build_lib.sh FIRST.
#
# ---------------------------------------------------------------------------
# 【2026-11-11 新板交付版】
#   * 库从静态 .a 改成**共享 .so**：测试程序通过 -lmosaic_pipeline 链接，
#     并用 **-Wl,-rpath,'$ORIGIN'** 把自己旁边的 libmosaic_pipeline.so 找到
#     ⇒ 整包搬到任何目录都能跑，不需要设 LD_LIBRARY_PATH。
#   * **不链接任何随包的 .so**：OpenCV / RKNN Runtime / RGA / OpenCL
#     全部用**板卡自带**的动态库（与 build_lib.sh 同一套定位规则）。
#   * -ffp-contract=off + 显式拒绝 fast-math。
#   * export LC_ALL=C: readelf 本地化 "Shared library:" 标签。
#
# ---------------------------------------------------------------------------
# 【2026-11-20 stat shim (src/npu/stat_compat.c)】
#   WHY: the buildroot cross toolchain is **glibc 2.29**, which exports only
#     `__xstat(ver,...)` and NOT the public symbol `stat` (the public `stat`
#     lives in libc_nonshared.a as a *hidden* symbol); the board's OpenCV 4.8
#     `libopencv_core.so.408` references the PUBLIC symbol `stat` => linking
#     an EXECUTABLE fails with:
#         ld: .../libc_nonshared.a(stat.oS): hidden symbol `stat' ...
#                is referenced by DSO
#         ld: final link failed: Bad value
#     (Linking a SHARED library does not fail, because build_lib.sh passes
#      `-Wl,--allow-shlib-undefined`; only executables hit this.)
#   FIX: `src/npu/stat_compat.c` defines the public symbol `stat` and forwards
#     to glibc's real entry point `__xstat` (the same trick as the board's
#     `/root/mosaic/libstatcompat.so`). It is linked STATICALLY into every
#     executable => **no runtime dependency** (no LD_PRELOAD, no extra .so).
#     Behaviour is also correct on glibc >= 2.33 boards.
#
# NOTE: this script deliberately contains NO deploy / scp / ssh logic.
#       Deployment is the operator's job -- see README.md section (3).
#
# 用法: ./build_demo.sh
# ===========================================================================
set -euo pipefail
export LC_ALL=C

die() { echo "[ERROR] $*" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/out"

# ---------------------------------------------------------------------------
# 依赖定位（与 build_lib.sh 完全一致 ⇒ 两条脚本用的是同一套依赖）
#   全部指向**板卡自带的库**，包内没有任何 third_party 副本。
# ---------------------------------------------------------------------------
detect_multiarch() {
  for d in /usr/lib/aarch64-linux-gnu /usr/lib64 /usr/lib; do
    [ -d "$d" ] && { echo "$d"; return 0; }
  done
  echo /usr/lib
}
BOARD_SYS_LIBDIR="$(detect_multiarch)"

OPENCV_INCLUDE_DIR="${OPENCV_INCLUDE_DIR:-/usr/include/opencv4}"
OPENCV_LIB_DIR="${OPENCV_LIB_DIR:-$BOARD_SYS_LIBDIR}"
BOARD_LIBS="${BOARD_LIBS:-$BOARD_SYS_LIBDIR}"
RT_LIB_DIR="${RT_LIB_DIR:-/usr/lib}"
EXTRA_LIB_DIRS="${EXTRA_LIB_DIRS:-}"

if [ -n "${OPENCV_INSTALL:-}" ]; then
  [ -n "${OPENCV_INCLUDE_DIR:-}" ] || OPENCV_INCLUDE_DIR="$OPENCV_INSTALL/include/opencv4"
  [ -n "${OPENCV_LIB_DIR:-}" ]    || OPENCV_LIB_DIR="$OPENCV_INSTALL/lib"
fi

# ---------------------------------------------------------------------------
# 交叉编译器
# ---------------------------------------------------------------------------
# ⚠️ 2026-11-20 起：**不要再硬编码宿主机盘符**。
#    宿主机重装 / 换盘之后 `/media/manu/1TB-Volume` 会变成
#    `/run/media/manu/<UUID>` ⇒ 写死的路径必然失效（而且失效得很隐蔽：
#    脚本会在"找不到编译器"这一行才炸）。
#    ⇒ 改成**自动发现**：按 新式盘符 → 旧式盘符 → $HOME → 旧硬编码路径 顺序扫，
#       多个候选之间用 ':' 分隔（下方 find_cxx 会逐个真跑一遍做自检）。
TC_REL="rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu"
TC_DEFAULT=""
for _tc_root in /run/media/manu/* /media/manu/* "$HOME"; do
  [ -d "$_tc_root/$TC_REL" ] && TC_DEFAULT="${TC_DEFAULT:+:}$_tc_root/$TC_REL"
done
[ -n "$TC_DEFAULT" ] || TC_DEFAULT="/media/manu/1TB-Volume/$TC_REL"
TC_CANDIDATES="${RK3588_TOOLCHAIN:-$TC_DEFAULT}"

# ⚠️ 必须**真的跑一下**编译器（buildroot 的 `*-g++` 是 `*.br_real` 包装器）。
cxx_works() { [ -x "$1" ] && "$1" -E -x c++ - </dev/null >/dev/null 2>&1; }

find_cxx() {
  local tc="$1" n c
  for n in aarch64-rockchip-linux-gnu-g++ aarch64-none-linux-gnu-g++ aarch64-linux-gnu-g++; do
    if cxx_works "$tc/bin/$n"; then echo "$tc/bin/$n"; return 0; fi
  done
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
       ⇒ 请设置 RK3588_TOOLCHAIN=<工具链根目录>（可用 ':' 分隔多个，按顺序试），
         或直接设 CROSS_CXX=<aarch64 交叉 g++ 的绝对路径>。"

CXX="${CROSS_CXX:-}"
[ -n "$CXX" ] || CXX="$(find_cxx "$TC")"
[ -x "$CXX" ] || die "cross compiler 不可用：$CXX"

FP_FLAGS="-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations"

OPENCV_LIBS="-lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video -lopencv_imgproc -lopencv_core"

LIB_PATHS=(-L"$RT_LIB_DIR" -L"$BOARD_LIBS" -L"$OPENCV_LIB_DIR")
for d in $EXTRA_LIB_DIRS; do LIB_PATHS+=(-L"$d"); done
RPATH_LINKS=(-Wl,-rpath-link,"$RT_LIB_DIR" -Wl,-rpath-link,"$BOARD_LIBS" -Wl,-rpath-link,"$OPENCV_LIB_DIR")

# 板卡 OpenCV 是共享库，它自己还链着一长串传递依赖（libjpeg/libpng/libgdal/...）。
# 那些依赖**只在板卡运行时由 ldconfig 解析**，交叉链接时不必搬 ⇒ 允许共享库带
# 未定义符号。集成方只需提供板卡那 7 个 libopencv_*.so + 3 个板卡库。
RELAX="-Wl,--allow-shlib-undefined"

# ---------------------------------------------------------------------------
# 0. environment gate -- fail early and loudly, never half-build
# ---------------------------------------------------------------------------
[ -d "$OPENCV_INCLUDE_DIR/opencv2" ] || die "找不到板卡 OpenCV 头：$OPENCV_INCLUDE_DIR/opencv2
       ⇒ 用 OPENCV_INCLUDE_DIR=<目录> 覆盖（交叉编译时指向从板卡取回的副本）。"
_have_cv=0
for m in core imgproc imgcodecs calib3d features2d flann video; do
  ls "$OPENCV_LIB_DIR"/libopencv_$m.so* >/dev/null 2>&1 && _have_cv=$((_have_cv+1))
done
[ "$_have_cv" -eq 7 ] || die "板卡 OpenCV 动态库不全（$_have_cv/7）：$OPENCV_LIB_DIR"
[ -f "$BOARD_LIBS/libOpenCL.so" ] || [ -f "$BOARD_LIBS/libOpenCL.so.1" ] || die "找不到板卡 libOpenCL：$BOARD_LIBS"
[ -f "$RT_LIB_DIR/librknnrt.so" ] || die "找不到板卡 librknnrt.so：$RT_LIB_DIR"
[ -f "$BOARD_LIBS/librga.so" ] || [ -f "$BOARD_LIBS/librga.so.2" ] || die "找不到板卡 librga：$BOARD_LIBS"

echo "  工具链   : $TC"
echo "  编译器   : $CXX"
echo "  OpenCV头 : $OPENCV_INCLUDE_DIR"
echo "  OpenCV库 : $OPENCV_LIB_DIR"
echo "  板卡库   : $BOARD_LIBS"
echo "  RKNN 库  : $RT_LIB_DIR"

# the shared library MUST exist first -- that is what we link against
[ -f "$OUT/libmosaic_pipeline.so" ] || \
  die "out/libmosaic_pipeline.so not found -- run 'bash build_lib.sh' FIRST"

[ -f "$HERE/src/npu/mosaic_api_test.cpp" ]        || die "missing source: src/npu/mosaic_api_test.cpp"
[ -f "$HERE/src/npu/mosaic_client_example.cpp" ] || die "missing source: src/npu/mosaic_client_example.cpp"
[ -f "$HERE/src/npu/mosaic_async_client_example.cpp" ] || die "missing source: src/npu/mosaic_async_client_example.cpp"
[ -f "$HERE/src/npu/mosaic_contract_test.cpp" ]   || die "missing source: src/npu/mosaic_contract_test.cpp"
[ -f "$HERE/src/npu/rknn_mosaic_e2e.cpp" ]        || die "missing source: src/npu/rknn_mosaic_e2e.cpp"
[ -f "$HERE/src/npu/stat_compat.c" ]             || die "missing source: src/npu/stat_compat.c"

mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# The `stat` shim (public symbol `stat` for glibc < 2.33) -- it MUST be built
# BEFORE the link, and then goes into LINK_COMMON so every executable carries
# it.  Pure static object => no runtime dependency.
# ---------------------------------------------------------------------------
echo '== building stat_compat.o (glibc<2.33 `stat` shim) =='
"$CXX" -std=c++17 -O2 -fPIC -c "$HERE/src/npu/stat_compat.c" -o "$OUT/stat_compat.o"

LIB_CXXFLAGS="-std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
  $FP_FLAGS -DCL_TARGET_OPENCL_VERSION=300 \
  -I$HERE/include -I$HERE/include/rga -I$HERE/src/cpp -I$HERE/src/opencl/host -I$OPENCV_INCLUDE_DIR"

# 共享库链接行（五个测试程序共用）：
#   -L$OUT            → 找 libmosaic_pipeline.so
#   -lmosaic_pipeline → 链接它
#   -rpath,'$ORIGIN'  → 运行时从**自己所在目录**找它（整包可搬）
LINK_COMMON=(
  -L"$OUT" -lmosaic_pipeline -Wl,-rpath,'$ORIGIN'
  "$OUT/stat_compat.o"
  "${LIB_PATHS[@]}" "${RPATH_LINKS[@]}" $RELAX
  $OPENCV_LIBS -lOpenCL -lrknnrt -lrga -lpthread -ldl -lm
)

# ---------------------------------------------------------------------------
# 1. the EXAMPLE DEMO (open / push / collect / destroy)
# ---------------------------------------------------------------------------
echo "== building mosaic_api_test (the example demo) =="
"$CXX" $LIB_CXXFLAGS "$HERE/src/npu/mosaic_api_test.cpp" -o "$HERE/out/mosaic_api_test" "${LINK_COMMON[@]}"

# ---------------------------------------------------------------------------
# 1b. the MINIMAL INTEGRATION EXAMPLE (what an embedded integrator copies)
# ---------------------------------------------------------------------------
echo "== building mosaic_client_example (the minimal integration example) =="
"$CXX" $LIB_CXXFLAGS "$HERE/src/npu/mosaic_client_example.cpp" -o "$HERE/out/mosaic_client_example" "${LINK_COMMON[@]}"

# ---------------------------------------------------------------------------
# 1c. the MULTI-THREADED PRODUCER/CONSUMER TEMPLATE (the production-grade
#     integration skeleton: the camera thread never blocks, the worker keeps a
#     strict one-push-one-collect pairing, and the results are delivered to the
#     integrator's own callback).
# ---------------------------------------------------------------------------
echo "== building mosaic_async_client_example (the multi-threaded template) =="
"$CXX" $LIB_CXXFLAGS "$HERE/src/npu/mosaic_async_client_example.cpp" -o "$HERE/out/mosaic_async_client_example" "${LINK_COMMON[@]}"

# ---------------------------------------------------------------------------
# 2. the API CONTRACT test (the library's own gate)
# ---------------------------------------------------------------------------
echo "== building mosaic_contract_test (library contract test) =="
"$CXX" $LIB_CXXFLAGS "$HERE/src/npu/mosaic_contract_test.cpp" -o "$HERE/out/mosaic_contract_test" "${LINK_COMMON[@]}"

# ---------------------------------------------------------------------------
# 3. the END-TO-END test program (throughput / latency measurement)
# ---------------------------------------------------------------------------
echo "== building rknn_mosaic_e2e (the end-to-end test program) =="
"$CXX" $LIB_CXXFLAGS "$HERE/src/npu/rknn_mosaic_e2e.cpp" -o "$HERE/out/rknn_mosaic_e2e" "${LINK_COMMON[@]}"

echo
echo "== artifacts =="
for f in "$OUT/libmosaic_pipeline.so" "$OUT/mosaic_api_test" "$OUT/mosaic_client_example" "$OUT/mosaic_async_client_example" "$OUT/mosaic_contract_test" "$OUT/rknn_mosaic_e2e"; do
  echo "  path : $f"
  echo "  size : $(stat -c%s "$f") B"
  echo "  md5  : $(md5sum "$f" | cut -d' ' -f1)"
done
echo
echo "  -- 运行时依赖（板卡自带库，板卡 ldconfig 可解析）--"
for f in "$OUT/mosaic_api_test"; do
  readelf -d "$f" | grep NEEDED | sed 's/.*\[/    /;s/\]//'
done
echo
echo "[ok] test programs built.  Copy out/* to the NFS mount -- see README.md section (3)."
