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
#     out/mosaic_api_test_static  the SAME example demo, but linked against
#                                 **out/libmosaic_pipeline.a** (the STATIC
#                                 library) instead of the shared one.
#                                 This is the LIVING PROOF that the static
#                                 library links and runs; its output must be
#                                 byte-identical to out/mosaic_api_test.
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
# 【2026-11-25 新板（Orange Pi 5 / Debian 12）实测事实】
#   * 板上同时存在**两套 OpenCV**：
#       /usr/lib/libopencv_*.so.408                    ← 4.8（板上 app 与本 SDK 链的）
#       /usr/lib/aarch64-linux-gnu/libopencv_*.so.406   ← 4.6（Debian 系统包）
#     而板上的**头**只有 4.6（/usr/include/opencv4）⇒ 4.8 的头必须外部提供
#     （本包 include/opencv4 就是随包发布的那份 4.8 头）。
#     ⇒ 运行库目录按 **SONAME 后缀（ABI）** 选（OPENCV_ABI，默认 408），
#       头版本与 ABI 的配套关系**当场校验**；不配套就 die。
#   * OpenCL 只有 **ICD loader** /usr/lib/aarch64-linux-gnu/libOpenCL.so.1，
#     **没有 libOpenCL.so 开发软链** ⇒ -lOpenCL 链不出来 ⇒ 脚本退化成按
#     **绝对路径**直接链接那个版本化文件（NEEDED 仍是 libOpenCL.so.1）。
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
#
#   OPENCV_ABI          OpenCV 的 **SONAME 后缀**（408 = 4.8，406 = 4.6，
#                        410 = 4.10）。默认 408。
#   OPENCV_VERSION      期望的**头**版本，形如 4.8；默认由 OPENCV_ABI 推导。
#   OPENCV_INCLUDE_DIR  OpenCV **头**目录（含 opencv2/）。默认优先级：
#                        ① 用户显式指定 ② $HERE/include/opencv4（包内随包发布
#                        的 4.8 头）③ /usr/include/opencv4
#   OPENCV_LIB_DIR      OpenCV **动态库**目录（libopencv_*.so.$OPENCV_ABI）。
#                        默认按 ABI 自动探测（见 detect_opencv_lib_dir）。
#   BOARD_LIBS          板卡 libOpenCL.so.1 / librga.so 所在目录
#                        板卡默认 /usr/lib/aarch64-linux-gnu
#   RT_LIB_DIR          板卡 librknnrt.so 所在目录
#                        板卡默认 /usr/lib
#   EXTRA_LIB_DIRS      额外链接搜索路径（空格分隔），用于放 OpenCV 的传递依赖
#                        （libjpeg/libpng/libtiff/libgdal/...），交叉链接时需要
# ---------------------------------------------------------------------------
# ---- OpenCV 版本：ABI 是唯一真相来源 --------------------------------------
# OpenCV 的 SONAME 后缀 = major*100 + minor（4.6 → 406，4.8 → 408，4.10 → 410）
# ⇒ ABI 能唯一反推版本号 ⇒ 头校验与选目录共用同一套数字，不会各说各话。
OPENCV_ABI="${OPENCV_ABI:-408}"
case "$OPENCV_ABI" in
  ''|*[!0-9]*) die "OPENCV_ABI 必须是纯数字（OpenCV 的 SONAME 后缀，如 408/406/410）：$OPENCV_ABI" ;;
esac
_cv_abi_num=$((10#$OPENCV_ABI))
[ "$_cv_abi_num" -ge 100 ] || die "OPENCV_ABI 不合法（应为 major*100+minor，如 408）：$OPENCV_ABI"
_cv_abi_major=$((_cv_abi_num / 100))
_cv_abi_minor=$((_cv_abi_num % 100))
# 分工：OPENCV_ABI 决定**运行库**（选目录 + 校验 .so 的 SONAME）；
#       OPENCV_VERSION 决定**头**的期望版本（用户显式设置时以用户为准）。
OPENCV_VERSION="${OPENCV_VERSION:-$_cv_abi_major.$_cv_abi_minor}"

# ---- OpenCV 运行库目录：按 ABI 选，绝不按"目录是否存在"选 ------------------
# 新板上同时存在两套 OpenCV：
#   /usr/lib/libopencv_core.so.408                   ← 4.8（板上 app 与本 SDK 链的）
#   /usr/lib/aarch64-linux-gnu/libopencv_core.so.406  ← 4.6（Debian 系统包）
# 旧写法"哪个 multiarch 目录存在就选哪个"会选到 aarch64-linux-gnu（4.6）⇒
# 同一进程里同时装进 4.6 与 4.8 两套 OpenCV ⇒ cv::Mat 跨 .so 边界串味 ⇒ 事故。
# ⇒ 只接受**恰好含 libopencv_core.so.$OPENCV_ABI** 的目录。
OPENCV_LIB_CANDIDATES="/usr/lib /usr/lib/aarch64-linux-gnu /usr/lib64 /root/mosaic/opencv /home/orangepi/lib"

detect_opencv_lib_dir() {
  local d found
  for d in $OPENCV_LIB_CANDIDATES; do
    [ -d "$d" ] || continue
    [ -f "$d/libopencv_core.so.$OPENCV_ABI" ] && { echo "$d"; return 0; }
  done
  {
    echo "       需要 libopencv_core.so.$OPENCV_ABI（OPENCV_ABI=$OPENCV_ABI）"
    echo "       扫描过的候选目录 / 实际找到的 libopencv_core.so*："
    for d in $OPENCV_LIB_CANDIDATES; do
      if [ -d "$d" ]; then
        found="$(ls -1 "$d"/libopencv_core.so* 2>/dev/null | while IFS= read -r f; do basename "$f"; done | tr '\n' ' ')" || found=""
        echo "         $d : ${found:-(无)}"
      else
        echo "         $d : (目录不存在)"
      fi
    done
  } >&2
  die "按 OpenCV ABI 找不到板卡 OpenCV 运行库。
       ⇒ 板卡上可能只有**另一个次版本**的 OpenCV（例如只有 .so.406，而这里要 .so.408）。
       ⇒ 可用 OPENCV_LIB_DIR=<含 libopencv_core.so.$OPENCV_ABI 的目录> 覆盖，
         或用 OPENCV_ABI=<板卡实际提供的后缀> 换一套配套的头。"
}

# 用户显式指定的（含旧变量名 OPENCV_INSTALL 拆出来的）
_cv_inc_user="${OPENCV_INCLUDE_DIR:-}"
_cv_lib_user="${OPENCV_LIB_DIR:-}"

# 兼容旧变量名：OPENCV_INSTALL=<根>/... 时拆成 头目录 + 库目录
if [ -n "${OPENCV_INSTALL:-}" ]; then
  [ -n "$_cv_inc_user" ] || _cv_inc_user="$OPENCV_INSTALL/include/opencv4"
  [ -n "$_cv_lib_user" ] || _cv_lib_user="$OPENCV_INSTALL/lib"
fi

# 头目录：包内随包发布的 4.8 头优先（板上 /usr/include/opencv4 只有 4.6 的头）
if [ -n "$_cv_inc_user" ]; then
  OPENCV_INCLUDE_DIR="$_cv_inc_user"
elif [ -d "$HERE/include/opencv4" ]; then
  OPENCV_INCLUDE_DIR="$HERE/include/opencv4"
else
  OPENCV_INCLUDE_DIR="/usr/include/opencv4"
fi

# 运行库目录：按 ABI 探测（找不到就 die，并把每个候选目录里实际有的版本列出来）
if [ -n "$_cv_lib_user" ]; then
  OPENCV_LIB_DIR="$_cv_lib_user"
else
  OPENCV_LIB_DIR="$(detect_opencv_lib_dir)"
fi

BOARD_LIBS="${BOARD_LIBS:-/usr/lib/aarch64-linux-gnu}"
RT_LIB_DIR="${RT_LIB_DIR:-/usr/lib}"
EXTRA_LIB_DIRS="${EXTRA_LIB_DIRS:-}"

# ---- 头版本解析：CV_VERSION_MAJOR / CV_VERSION_MINOR ------------------------
# 从 opencv2/core/version.hpp 里取头版本（# 与 define 之间允许多余空格）
cv_version_of() {
  local f="$1/opencv2/core/version.hpp" maj min
  [ -f "$f" ] || return 0
  maj="$(sed -n 's/^[[:space:]]*#[[:space:]]*define[[:space:]]\{1,\}CV_VERSION_MAJOR[[:space:]]\{1,\}\([0-9]\{1,\}\).*/\1/p' "$f" | head -1)" || maj=""
  min="$(sed -n 's/^[[:space:]]*#[[:space:]]*define[[:space:]]\{1,\}CV_VERSION_MINOR[[:space:]]\{1,\}\([0-9]\{1,\}\).*/\1/p' "$f" | head -1)" || min=""
  [ -n "$maj" ] && [ -n "$min" ] && echo "$maj.$min"
  return 0
}

# ---- 运行库完整性：按**精确版本**判定 --------------------------------------
# 旧写法 `libopencv_$m.so*`（通配）会把 4.6 的 .so.406 也算成"有" ⇒ 明明链的
# 是 4.6 却以为对上了 4.8。⇒ 改成精确匹配 libopencv_$m.so.$OPENCV_ABI；无版本号
# 的 .so 软链也接受（新板 /usr/lib/libopencv_core.so 就是这种），但必须**核对它
# 的 SONAME** —— 链接器 -lopencv_<m> 吃的正是这个 .so，它的 SONAME 决定产物里
# 的 NEEDED，对不上就等于悄悄链了另一个次版本。
cv_module_ok() {
  local d="$1" m="$2" abi="$3" so son
  # ① 精确版本文件
  if [ -f "$d/libopencv_$m.so.$abi" ]; then
    so="$d/libopencv_$m.so"
    if [ -f "$so" ]; then
      son="$(readelf -d "$so" 2>/dev/null | sed -n 's/.*SONAME.*\[\([^]]*\)\].*/\1/p' | head -1)" || son=""
      [ "$son" = "libopencv_$m.so.$abi" ] && return 0
      return 1
    fi
    return 0
  fi
  # ② 只有无版本号的 .so（开发软链）⇒ 核对它的 SONAME
  so="$d/libopencv_$m.so"
  if [ -f "$so" ]; then
    son="$(readelf -d "$so" 2>/dev/null | sed -n 's/.*SONAME.*\[\([^]]*\)\].*/\1/p' | head -1)" || son=""
    [ "$son" = "libopencv_$m.so.$abi" ] && return 0
  fi
  return 1
}

# ---- OpenCL 链接项解析 ----------------------------------------------------
# 新板（Debian 12）只有 **ICD loader** /usr/lib/aarch64-linux-gnu/libOpenCL.so.1，
# **没有 libOpenCL.so 开发软链** ⇒ -lOpenCL 必然链不出来。
# ⇒ 退化：直接把这个版本化文件按**绝对路径**放进链接行。它的 SONAME 就是
#    libOpenCL.so.1 ⇒ 产物里的 NEEDED 仍是 libOpenCL.so.1，与 -lOpenCL 一致。
resolve_opencl_link() {
  if [ -f "$BOARD_LIBS/libOpenCL.so" ]; then
    echo "-lOpenCL"; return 0
  fi
  local f
  for f in "$BOARD_LIBS/libOpenCL.so.1" /lib/aarch64-linux-gnu/libOpenCL.so.1 /usr/lib/aarch64-linux-gnu/libOpenCL.so.1; do
    [ -f "$f" ] && { echo "$f"; return 0; }
  done
  die "找不到板卡 libOpenCL（既没有 libOpenCL.so，也没有 libOpenCL.so.1）。
       ⇒ 板卡只有运行库、没有开发软链时，脚本会退化成直接链接版本化的 .so.1；
         这里连 .so.1 都没有 ⇒ 板上可能压根没装 OpenCL ICD loader。
       ⇒ 可用 BOARD_LIBS=<目录> 覆盖。"
}
OPENCL_LINK="$(resolve_opencl_link)"

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
       ⇒ 这应当是**板卡**的 OpenCV 头（板卡上为 /usr/include/opencv4，只有 4.6；
         本 SDK 要的是 $OPENCV_VERSION 的头，包内 include/opencv4 就是那份）。
         交叉编译时必须指向**从板卡取回的那份副本**（fetch_board_libs.sh），
         不能用宿主机的 /usr（x86 的头会污染构建）。
         可用 OPENCV_INCLUDE_DIR=<目录> 覆盖。"

# ---- 头版本 ↔ 运行库 ABI 的一致性校验 --------------------------------------
# OpenCV 次版本会改 C++ ABI ⇒ 拿 4.6 的头去编 4.8 的运行库就是 ABI 错配，
# 症状是运行期莫名其妙崩在 cv::Mat 相关的调用上 ⇒ 必须在链接前**当场**挡住。
_cv_hdr_ver="$(cv_version_of "$OPENCV_INCLUDE_DIR")"
[ -n "$_cv_hdr_ver" ] || die "解析不出 OpenCV 头版本（CV_VERSION_MAJOR/MINOR）：$OPENCV_INCLUDE_DIR/opencv2/core/version.hpp
       ⇒ 这不是一个完整的 OpenCV 头目录。"
_cv_hdr_major="${_cv_hdr_ver%%.*}"
_cv_hdr_minor="${_cv_hdr_ver##*.}"
_cv_exp_major="${OPENCV_VERSION%%.*}"
_cv_exp_minor="${OPENCV_VERSION##*.}"
# MAJOR+MINOR 逐位比（4.8 → 48，4.10 → 410）：比的是**次版本**，补丁号不算
[ "$_cv_hdr_major$_cv_hdr_minor" = "$_cv_exp_major$_cv_exp_minor" ] || {
  _cv_board_ver="$(cv_version_of /usr/include/opencv4)"
  _cv_pkg_ver="$(cv_version_of "$HERE/include/opencv4")"
  die "OpenCV 头版本与运行库 ABI 不配套：
       头目录   : $OPENCV_INCLUDE_DIR → 实际 OpenCV $_cv_hdr_ver
       期望版本 : $OPENCV_VERSION（OPENCV_ABI=$OPENCV_ABI，ABI 前两位 ${OPENCV_ABI:0:2}）
       板上 /usr/include/opencv4 的版本: ${_cv_board_ver:-（本机未找到；这块板上是 4.6 的头）}
       ⇒ OpenCV 次版本会改 C++ ABI：$_cv_hdr_ver 的头配 $OPENCV_ABI 的运行库是错配，
         同一进程里会同时出现两套 OpenCV ⇒ cv::Mat 跨边界串味。
       ⇒ 必须显式指定 $OPENCV_VERSION 的头目录：
         OPENCV_INCLUDE_DIR=<含 opencv2/ 的 $OPENCV_VERSION 头目录>
         并用 OPENCV_ABI=$OPENCV_ABI 让运行库与它配套。
         包内 include/opencv4：${_cv_pkg_ver:-（这个包里没有）}"
}

_cv_missing=""
for m in core imgproc imgcodecs calib3d features2d flann video; do
  cv_module_ok "$OPENCV_LIB_DIR" "$m" "$OPENCV_ABI" || _cv_missing="$_cv_missing $m"
done
if [ -n "$_cv_missing" ]; then
  {
    echo "       缺（要 libopencv_*.so.$OPENCV_ABI）：$_cv_missing"
    echo "       目录里实际有的版本："
    for m in $_cv_missing; do
      _cv_found="$(ls -1 "$OPENCV_LIB_DIR"/libopencv_$m.so* 2>/dev/null | while IFS= read -r f; do basename "$f"; done | tr '\n' ' ')" || _cv_found=""
      echo "         libopencv_$m : ${_cv_found:-(无)}"
    done
  } >&2
  die "板卡 OpenCV 动态库不全 / 版本不对（OPENCV_ABI=$OPENCV_ABI）：$OPENCV_LIB_DIR
       ⇒ 新板上 4.8 在 /usr/lib（.so.408），Debian 的 4.6 在 /usr/lib/aarch64-linux-gnu（.so.406）。
       ⇒ 可用 OPENCV_LIB_DIR=<目录> 覆盖，或用 OPENCV_ABI=406 明确要 4.6 那套。"
fi

[ -f "$RT_LIB_DIR/librknnrt.so" ] || die "找不到板卡 librknnrt.so：$RT_LIB_DIR
       ⇒ 可用 RT_LIB_DIR=<目录> 覆盖。"
# RGA：按**绝对路径**参与链接，不用 `-lrga`。
#   🔴 为什么：`-L` 顺序是 RT_LIB_DIR → BOARD_LIBS → OPENCV_LIB_DIR，而板卡上
#      RT_LIB_DIR 默认就是 **/usr/lib** —— 那里有**另一份** librga.so
#      （Rockchip BSP 版，md5 与 multiarch 那份不同；app 的 ldd 解析到的也不是它）
#      ⇒ `-lrga` 会挑到 BSP 那份 ⇒ 必须用绝对路径锁定 BOARD_LIBS 这一份
#      （与 build_board_native.sh 同口径；链接器取其 SONAME ⇒ NEEDED 仍是 librga.so.2）。
if [ -f "$BOARD_LIBS/librga.so" ]; then
  RGA_LINK="$BOARD_LIBS/librga.so"
elif [ -f "$BOARD_LIBS/librga.so.2" ]; then
  RGA_LINK="$BOARD_LIBS/librga.so.2"
else
  die "找不到板卡 librga：$BOARD_LIBS
       ⇒ 可用 BOARD_LIBS=<目录> 覆盖（板卡上是 /usr/lib/aarch64-linux-gnu）。"
fi
[ -f "$HERE/src/npu/mosaic_pipeline.cpp" ] || die "missing source: src/npu/mosaic_pipeline.cpp"

echo "  工具链   : $TC"
echo "  编译器   : $CXX"
echo "  OpenCV头 : $OPENCV_INCLUDE_DIR (v$_cv_hdr_ver)"
echo "  OpenCV库 : $OPENCV_LIB_DIR (ABI $OPENCV_ABI)"
echo "  OpenCL   : $OPENCL_LINK"
echo "  板卡库   : $BOARD_LIBS"
echo "  RKNN 库  : $RT_LIB_DIR"
[ -n "$EXTRA_LIB_DIRS" ] && echo "  额外路径 : $EXTRA_LIB_DIRS"

# the shared library MUST exist first -- that is what we link against
[ -f "$OUT/libmosaic_pipeline.so" ] || \
  die "out/libmosaic_pipeline.so not found -- run 'bash build_lib.sh' FIRST"

[ -f "$HERE/src/npu/mosaic_api_test.cpp" ]        || die "missing source: src/npu/mosaic_api_test.cpp"
[ -f "$HERE/src/npu/mosaic_client_example.cpp" ] || die "missing source: src/npu/mosaic_client_example.cpp"
[ -f "$HERE/src/npu/mosaic_async_client_example.cpp" ] || die "missing source: src/npu/mosaic_async_client_example.cpp"
[ -f "$HERE/src/npu/mosaic_contract_test.cpp" ]   || die "missing source: src/npu/mosaic_contract_test.cpp"
[ -f "$HERE/src/npu/mosaic_tracked_cb_example.cpp" ] || die "missing source: src/npu/mosaic_tracked_cb_example.cpp"
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
  $OPENCV_LIBS "$OPENCL_LINK" -lrknnrt "$RGA_LINK" -lpthread -ldl -lm
)

# ---------------------------------------------------------------------------
# 1. the EXAMPLE DEMO (open / push / collect / destroy)
# ---------------------------------------------------------------------------
echo "== building mosaic_api_test (the example demo) =="
"$CXX" $LIB_CXXFLAGS "$HERE/src/npu/mosaic_api_test.cpp" -o "$HERE/out/mosaic_api_test" "${LINK_COMMON[@]}"

# ---------------------------------------------------------------------------
# 1a. the SAME demo linked against the STATIC library.
#
#     🔴 **必须按路径显式写 .a，不能写 -lmosaic_pipeline**：链接器在 -L 路径下
#        **优先选 libmosaic_pipeline.so** ⇒ 写了 -l 就会静默链成共享版，
#        「静态链接验证」变成一句空话（产物看起来正常，NEEDED 里却有 .so）。
#        ⇒ 这里只给 .a 的**完整路径**，且**不给** -Wl,-rpath,'$ORIGIN'
#           （静态链接的程序运行时不需要我们的 .so）。
#
#     ⚠️ stat_compat.o 必须带上：链可执行文件时 glibc<2.33 的 `stat` 是 hidden
#        符号，板卡 OpenCV 引用的是公开符号 `stat` => 不带就链不出来。
# ---------------------------------------------------------------------------
echo "== building mosaic_api_test_static (the example demo linked STATICALLY) =="
[ -f "$OUT/libmosaic_pipeline.a" ] || \
  die "out/libmosaic_pipeline.a not found -- run 'bash build_lib.sh' FIRST"
"$CXX" $LIB_CXXFLAGS "$HERE/src/npu/mosaic_api_test.cpp" -o "$OUT/mosaic_api_test_static" \
  "$OUT/libmosaic_pipeline.a" "$OUT/stat_compat.o" \
  "${LIB_PATHS[@]}" "${RPATH_LINKS[@]}" $RELAX \
  $OPENCV_LIBS "$OPENCL_LINK" -lrknnrt "$RGA_LINK" -lpthread -ldl -lm

# 自检：静态链接版**不得**再依赖 libmosaic_pipeline.so（依赖了就说明上面
# 那行不知怎么又链成了共享版 —— 静默退化，必须当场炸掉）
if readelf -d "$OUT/mosaic_api_test_static" 2>/dev/null | grep -q 'libmosaic_pipeline\.so'; then
  die "mosaic_api_test_static 仍然依赖 libmosaic_pipeline.so ==> 静态链接没有生效！"
fi
echo "  [ok] mosaic_api_test_static 的 NEEDED 里没有 libmosaic_pipeline.so（确为静态链接）"

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
# 1d. the TRACKED-RESULT CALLBACK EXAMPLE (v10.7 -- one target, ONE box)
# ---------------------------------------------------------------------------
echo "== building mosaic_tracked_cb_example (the tracked-callback example) =="
"$CXX" $LIB_CXXFLAGS "$HERE/src/npu/mosaic_tracked_cb_example.cpp" -o "$HERE/out/mosaic_tracked_cb_example" "${LINK_COMMON[@]}"

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
for f in "$OUT/libmosaic_pipeline.so" "$OUT/libmosaic_pipeline.a" "$OUT/mosaic_api_test" "$OUT/mosaic_api_test_static" "$OUT/mosaic_client_example" "$OUT/mosaic_async_client_example" "$OUT/mosaic_contract_test" "$OUT/mosaic_tracked_cb_example" "$OUT/rknn_mosaic_e2e"; do
  echo "  path : $f"
  echo "  size : $(stat -c%s "$f") B"
  echo "  md5  : $(md5sum "$f" | cut -d' ' -f1)"
done
echo
echo "  -- 运行时依赖（板卡自带库，板卡 ldconfig 可解析）--"
for f in "$OUT/mosaic_api_test" "$OUT/mosaic_api_test_static"; do
  readelf -d "$f" | grep NEEDED | sed 's/.*\[/    /;s/\]//'
  echo "    ----"
done
echo "  🔴 对照：mosaic_api_test（链 .so）**必须**有 libmosaic_pipeline.so；"
echo "     mosaic_api_test_static（链 .a）**必须没有** ⇒ 静态库真的生效了。"
echo
echo "[ok] test programs built.  Copy out/* to the NFS mount -- see README.md section (3)."
