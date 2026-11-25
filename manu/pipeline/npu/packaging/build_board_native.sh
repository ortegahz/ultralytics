#!/bin/bash
# build_board_native.sh -- 在 **RK3588 板卡上原生**全量编译库 + 板卡验证探针。
#
# 为什么要它：交付包里的 `out/mosaic_track_dump` / `out/mosaic_cadence_probe`
#   是**板卡原生编译**的二进制（零 OpenCV 依赖 ⇒ 三个 OpenCV 变体共用同一份），
#   而 `build_lib.sh` 只编库、**不编探针** ⇒ 探针的重编入口就是本脚本。
#
# 执行机：RK3588 板卡（root@<board>，aarch64，需板卡自带 g++ 与 OpenCV）
# 用法（把本包整棵树放到板卡上，例如 /root/mosaic_sdk）：
#   cd /root/mosaic_sdk && bash build_board_native.sh
#
# ---- 板卡默认（2026-11-25 实测：Orange Pi 5 / Debian 12 / glibc 2.36 / gcc 12.2）----
#   * OpenCV **运行库**有两套，必须按 ABI 挑：
#       - 4.8：/usr/lib/libopencv_*.so.408                          ← app 与本 SDK 真正用的那套
#       - 4.6：/usr/lib/aarch64-linux-gnu/libopencv_*.so.406         ← Debian 的 libopencv-core406
#     ⇒ 探测顺序 **/usr/lib 必须排在 /usr/lib/aarch64-linux-gnu 前面**，否则会链到 4.6。
#   * OpenCV **头**板上只有 4.6（/usr/include/opencv4，CV_VERSION_MINOR 6），
#     **没有 4.8 头** ⇒ 头 4.6 配库 4.8 = **ABI 错配**（结构体布局按 4.6 头编、按 4.8 库解释）
#     ⇒ 4.8 头必须外部提供：包内 `include/opencv4`（随包发布）或板上 `/root/cv480inc`。
#   * OpenCL：板上**只有** /usr/lib/aarch64-linux-gnu/libOpenCL.so.1（Debian 的 ICD loader），
#     **没有 libOpenCL.so 开发软链** ⇒ `-lOpenCL` 会链接失败 ⇒ 改用绝对路径参与链接。
#     真正的 GPU 实现是 /usr/lib/aarch64-linux-gnu/mali/libMaliOpenCL.so.1
#     （由 /etc/OpenCL/vendors/mali.icd 指定，链接期不需要它）。
#   * RGA：/usr/lib/aarch64-linux-gnu/librga.so → SONAME librga.so.2
#     （另有一份**不同**的 /usr/lib/librga.so —— Rockchip BSP 版；app 实际解析到的是 multiarch 那份）
#   * RKNN runtime：/usr/lib/librknnrt.so
#
# ---- 关键点 ----
#   * 探针链接时用 `-Wl,--allow-shlib-undefined` + 只链 `-lmosaic_pipeline` ⇒ **零 OpenCV NEEDED**
#     ⇒ 同一份二进制可在 4.6 / 4.8 / 4.2 三种板卡上直接用。
#   * 库链接时用板卡自带的 OpenCV 头与库 ⇒ `.so` 的 NEEDED 随板卡 OpenCV 版本变化
#     （4.8 板 ⇒ `.so.408`；4.6 板 ⇒ `.so.406`；4.2 板 ⇒ `.so.42`），**必须按目标板编译**。
#   * `-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations` 与交叉编译保持同一数值口径。
#
# ---- 环境变量（与 build_lib.sh / build_demo.sh **同名同义**）----
#   OPENCV_ABI          OpenCV 的 SONAME 后缀，默认 408（可选 406 / 410）
#   OPENCV_VERSION      期望的**头**版本，形如 4.8，默认由 OPENCV_ABI 推导（408→4.8、406→4.6、410→4.10）
#   OPENCV_INCLUDE_DIR  OpenCV 头目录（含 opencv2/）；默认按优先级自动选，见下面 OCV_INC_CANDIDATES
#   OPENCV_LIB_DIR      OpenCV 运行库目录；默认**按 ABI 探测**（只认 libopencv_core.so.$OPENCV_ABI）
#   BOARD_LIBS          板卡 OpenCL / RGA 所在目录，默认 /usr/lib/aarch64-linux-gnu
#   RT_LIB_DIR          板卡 librknnrt.so 所在目录，默认 /usr/lib
set -euo pipefail

die() { echo "[FATAL] $*" >&2; exit 1; }

HERE=${HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
OUT=${OUT:-$HERE/out}
SRC=${SRC:-$HERE/src/npu}
CXX=${CXX:-g++}
FP="-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations"

# ---------------------------------------------------------------------------
# OpenCV ABI 契约
# ---------------------------------------------------------------------------
OPENCV_ABI="${OPENCV_ABI:-408}"
case "$OPENCV_ABI" in
  406) OPENCV_VERSION_DEFAULT=4.6 ;;
  408) OPENCV_VERSION_DEFAULT=4.8 ;;
  410) OPENCV_VERSION_DEFAULT=4.10 ;;
  *) die "不支持的 OPENCV_ABI='$OPENCV_ABI'（可用 406 / 408 / 410）" ;;
esac
OPENCV_VERSION="${OPENCV_VERSION:-$OPENCV_VERSION_DEFAULT}"

# OpenCV 的 soversion 规则：major*100 + minor（4.6→406、4.8→408、4.10→410）
ocv_version_to_abi() {
  local v="$1" major minor
  major="${v%%.*}"
  minor="${v#*.}"; minor="${minor%%.*}"
  case "$major$minor" in *[!0-9]*) return 1 ;; esac
  [ -n "$major" ] && [ -n "$minor" ] || return 1
  echo $(( major * 100 + minor ))
}
_abi_from_ver="$(ocv_version_to_abi "$OPENCV_VERSION")" \
  || die "OPENCV_VERSION='$OPENCV_VERSION' 不是形如 4.8 的版本号"
# 与 build_lib.sh / build_demo.sh 同语义：**用户显式设置 OPENCV_VERSION 时以用户为准**
#   （头按 OPENCV_VERSION 校验、运行库按 OPENCV_ABI 选）⇒ 两者冲突只警告，不拦。
if [ "$_abi_from_ver" != "$OPENCV_ABI" ]; then
  echo "[WARN] OPENCV_VERSION=$OPENCV_VERSION 与 OPENCV_ABI=$OPENCV_ABI 不一致（$OPENCV_VERSION 对应 ABI $_abi_from_ver）" >&2
  echo "       ⇒ 头按 $OPENCV_VERSION 校验、运行库按 .$OPENCV_ABI 选；两者错配会编出错误 ABI 的产物。" >&2
fi

BOARD_LIBS="${BOARD_LIBS:-/usr/lib/aarch64-linux-gnu}"
RT_LIB_DIR="${RT_LIB_DIR:-/usr/lib}"

# ---------------------------------------------------------------------------
# OpenCV 头目录：用户显式 > 包内随包发布的 4.8 头 > 板上候选目录
# ---------------------------------------------------------------------------
# 候选顺序的理由：包内 include/opencv4 是**交付物自带**的 4.8 头（与 4.8 运行库同源），
#   板上 /root/cv480inc 是人工放到板上的 4.8 头，
#   /usr/include/opencv4 在新板上**只有 4.6**，所以排在最后（且会被版本校验挡下）。
OCV_INC_CANDIDATES=("$HERE/include/opencv4" /root/cv480inc /usr/local/include/opencv4 /usr/include/opencv4)

# 读一个头目录的版本（opencv2/core/version.hpp）
ocv_header_version() {
  local inc="$1" vh major minor
  vh="$inc/opencv2/core/version.hpp"
  [ -f "$vh" ] || return 1
  major="$(sed -n 's/^[[:space:]]*#define[[:space:]]\{1,\}CV_VERSION_MAJOR[[:space:]]\{1,\}\([0-9]\{1,\}\).*/\1/p' "$vh" | head -1)"
  minor="$(sed -n 's/^[[:space:]]*#define[[:space:]]\{1,\}CV_VERSION_MINOR[[:space:]]\{1,\}\([0-9]\{1,\}\).*/\1/p' "$vh" | head -1)"
  [ -n "$major" ] && [ -n "$minor" ] || return 1
  echo "$major.$minor"
}

# 头/库 ABI 错配时的统一报错（说清"为什么"+"怎么办"）
die_ocv_header_mismatch() {
  local found_dir="$1" found_ver="$2" sys_ver="${3:-}"
  {
    echo "[FATAL] OpenCV **头**与要链的运行库版本不一致 —— 这不是小问题："
    echo "        编译期结构体布局按**头**决定，运行期按**库**解释 ⇒ 版本不同就是踩内存。"
    echo "        期望 : OpenCV ${OPENCV_VERSION} 头（运行库 libopencv_core.so.${OPENCV_ABI}）"
    echo "        实际 : OpenCV ${found_ver} 头（${found_dir}）"
    if [ -n "$sys_ver" ]; then
      echo "        板上 \`/usr/include/opencv4\` 是 OpenCV ${sys_ver} 头，但本 SDK 必须链 ${OPENCV_VERSION}（app 链的是 ${OPENCV_VERSION}）"
    else
      echo "        板上没有 \`/usr/include/opencv4\` 头，但本 SDK 必须链 ${OPENCV_VERSION}（app 链的是 ${OPENCV_VERSION}）"
    fi
    echo "        ⇒ 请把 ${OPENCV_VERSION} 头放到板上（如 \`/root/cv480inc\`）并用 \`OPENCV_INCLUDE_DIR=\` 指定，"
    echo "          或用包内随包发布的 \`include/opencv4\`（本包位置：$HERE/include/opencv4）"
  } >&2
  exit 1
}

SYS_OCV_INC_VER="$(ocv_header_version /usr/include/opencv4 2>/dev/null || true)"

OPENCV_INCLUDE_DIR="${OPENCV_INCLUDE_DIR:-}"
if [ -n "$OPENCV_INCLUDE_DIR" ]; then
  [ -d "$OPENCV_INCLUDE_DIR/opencv2" ] \
    || die "OPENCV_INCLUDE_DIR=$OPENCV_INCLUDE_DIR 里没有 opencv2/ ⇒ 不是 OpenCV 头目录"
  _v="$(ocv_header_version "$OPENCV_INCLUDE_DIR" || true)"
  [ -n "$_v" ] || die "读不出 $OPENCV_INCLUDE_DIR/opencv2/core/version.hpp 的版本号"
  [ "$_v" = "$OPENCV_VERSION" ] \
    || die_ocv_header_mismatch "$OPENCV_INCLUDE_DIR" "$_v" "$SYS_OCV_INC_VER"
else
  _ocv_inc=""
  _ocv_inc_bad=""
  _ocv_inc_bad_ver=""
  for c in "${OCV_INC_CANDIDATES[@]}"; do
    [ -d "$c/opencv2" ] || continue
    _v="$(ocv_header_version "$c" || true)"
    if [ "$_v" = "$OPENCV_VERSION" ]; then
      _ocv_inc="$c"
      break
    fi
    if [ -z "$_ocv_inc_bad" ]; then
      _ocv_inc_bad="$c"
      _ocv_inc_bad_ver="${_v:-未知}"
    fi
  done
  if [ -z "$_ocv_inc" ]; then
    echo "[FATAL] 找不到 OpenCV ${OPENCV_VERSION} 头（对应运行库 libopencv_core.so.${OPENCV_ABI}）。" >&2
    echo "        扫描过的候选目录：" >&2
    for c in "${OCV_INC_CANDIDATES[@]}"; do
      if [ -d "$c/opencv2" ]; then
        _v="$(ocv_header_version "$c" || true)"
        echo "          $c  → OpenCV ${_v:-未知} 头" >&2
      else
        echo "          $c  → 不存在" >&2
      fi
    done
    die_ocv_header_mismatch "${_ocv_inc_bad:-/usr/include/opencv4}" "${_ocv_inc_bad_ver:-未知}" "$SYS_OCV_INC_VER"
  fi
  OPENCV_INCLUDE_DIR="$_ocv_inc"
fi

# ---------------------------------------------------------------------------
# OpenCV 运行库目录：**按 ABI 探测**（只认 libopencv_core.so.$OPENCV_ABI）
# ---------------------------------------------------------------------------
# 候选顺序的理由：新板 4.8 在 /usr/lib，4.6 在 multiarch 目录 ⇒ /usr/lib 必须在前，
#   否则 -lopencv_core 会解析到 4.6，产物 NEEDED 变成 .406，和 app 链的 .408 对不上。
OCV_LIB_CANDIDATES=(/usr/lib /usr/lib/aarch64-linux-gnu /usr/local/lib /root/mosaic/opencv)

ocv_soname() {
  readelf -d "$1" 2>/dev/null | sed -n 's/.*SONAME[^[]*\[\([^]]*\)\].*/\1/p' | head -1
}

ocv_lib_dir_matches() {
  local d="$1" soname
  [ -d "$d" ] || return 1
  # 命中判定 1：直接有带 ABI 后缀的库
  [ -e "$d/libopencv_core.so.$OPENCV_ABI" ] && return 0
  # 命中判定 2：无版本号的 libopencv_core.so，但其 SONAME 就是 .$OPENCV_ABI
  [ -e "$d/libopencv_core.so" ] || return 1
  soname="$(ocv_soname "$d/libopencv_core.so")"
  [ "$soname" = "libopencv_core.so.$OPENCV_ABI" ]
}

OPENCV_LIB_DIR="${OPENCV_LIB_DIR:-}"
if [ -n "$OPENCV_LIB_DIR" ]; then
  [ -d "$OPENCV_LIB_DIR" ] || die "OPENCV_LIB_DIR=$OPENCV_LIB_DIR 不是目录"
  ocv_lib_dir_matches "$OPENCV_LIB_DIR" \
    || die "OPENCV_LIB_DIR=$OPENCV_LIB_DIR 里没有 libopencv_core.so.$OPENCV_ABI（SONAME 也不匹配）⇒ 与 OPENCV_ABI=$OPENCV_ABI 对不上"
else
  for c in "${OCV_LIB_CANDIDATES[@]}"; do
    if ocv_lib_dir_matches "$c"; then
      OPENCV_LIB_DIR="$c"
      break
    fi
  done
fi
if [ -z "$OPENCV_LIB_DIR" ]; then
  {
    echo "[FATAL] 找不到 OpenCV ${OPENCV_VERSION} 运行库（ABI 后缀 .${OPENCV_ABI}）。" >&2
    echo "        当前 OPENCV_ABI=$OPENCV_ABI，扫描过的目录与各自实际存在的 libopencv_core.so.*：" >&2
    for c in "${OCV_LIB_CANDIDATES[@]}"; do
      if [ -d "$c" ]; then
        _found="$(ls "$c" 2>/dev/null | grep -E '^libopencv_core\.so' | tr '\n' ' ' || true)"
        echo "          $c  → ${_found:-（无 libopencv_core.so*）}" >&2
      else
        echo "          $c  → 目录不存在" >&2
      fi
    done
    echo "        ⇒ 用 OPENCV_LIB_DIR=<目录> 指定；本 SDK 要链的是 app 用的那套（.${OPENCV_ABI}）。" >&2
  }
  exit 1
fi

# ---------------------------------------------------------------------------
# OpenCL / RGA / RKNN 的链接项
# ---------------------------------------------------------------------------
# OpenCL：板上没有 libOpenCL.so 开发软链（只有 Debian ICD loader libOpenCL.so.1）
#   ⇒ `-lOpenCL` 必然失败 ⇒ 找不到开发软链时改用**绝对路径**参与链接
#   （链接器取文件里的 SONAME ⇒ NEEDED 仍是 libOpenCL.so.1，和 app 一致）。
OCL_LINK=""
if [ -e "$BOARD_LIBS/libOpenCL.so" ]; then
  OCL_LINK="-lOpenCL"
else
  for c in "$BOARD_LIBS" /lib/aarch64-linux-gnu /usr/lib/aarch64-linux-gnu; do
    if [ -e "$c/libOpenCL.so.1" ]; then
      OCL_LINK="$c/libOpenCL.so.1"
      break
    fi
  done
fi
[ -n "$OCL_LINK" ] \
  || die "找不到板卡 OpenCL：$BOARD_LIBS/libOpenCL.so(.1)、/lib/aarch64-linux-gnu/libOpenCL.so.1 都没有
       ⇒ 板上 OpenCL 是 Debian ICD loader（真实现由 /etc/OpenCL/vendors/mali.icd 指定），
         请用 BOARD_LIBS=<目录> 指定，或在板上造一个 libOpenCL.so -> libOpenCL.so.1 软链。"

# RGA：用 $BOARD_LIBS/librga.so（SONAME librga.so.2）。
#   注意 /usr/lib 里**还有一份不同的** librga.so（Rockchip BSP 版），而 -L 顺序里 /usr/lib 在前
#   （那里放 OpenCV 4.8）⇒ 若用 `-lrga` 会挑到 BSP 那份 ⇒ 这里用绝对路径锁定 multiarch 这份
#   （app 的 ldd 解析到的就是它）。
if [ -e "$BOARD_LIBS/librga.so" ]; then
  RGA_LINK="$BOARD_LIBS/librga.so"
elif [ -e "$BOARD_LIBS/librga.so.2" ]; then
  RGA_LINK="$BOARD_LIBS/librga.so.2"
else
  RGA_LINK=""
fi
[ -n "$RGA_LINK" ] \
  || die "找不到板卡 RGA：$BOARD_LIBS/librga.so(.2) 都没有
       ⇒ 注意 /home/orangepi/rockit/librga.so 是**另一份**（app 不用它），
         请用 BOARD_LIBS=<目录> 指定到 /usr/lib/aarch64-linux-gnu。"

if [ -e "$RT_LIB_DIR/librknnrt.so" ]; then
  RTKNN_LINK="-lrknnrt"
elif [ -e "$RT_LIB_DIR/librknnrt.so.2" ]; then
  RTKNN_LINK="$RT_LIB_DIR/librknnrt.so.2"
else
  RTKNN_LINK=""
fi
[ -n "$RTKNN_LINK" ] \
  || die "找不到板卡 RKNN runtime：$RT_LIB_DIR/librknnrt.so 都没有
       ⇒ 用 RT_LIB_DIR=<目录> 指定（新板上它在 /usr/lib）。"

# 其它板卡库目录（老板子上人工放 OpenCV 传递依赖的地方，存在才加）
LNK=${LNK:-/root/lnk}

# -L 顺序：**OpenCV 目录必须第一**（/usr/lib 里的 4.8 vs multiarch 里的 4.6）
LIB_PATHS=(-L"$OPENCV_LIB_DIR" -L"$RT_LIB_DIR" -L"$BOARD_LIBS")
[ -d "$LNK" ] && LIB_PATHS+=(-L"$LNK")

INC="-I$HERE/include -I$HERE/include/rga -I$HERE/src/cpp -I$HERE/src/opencl/host -I$OPENCV_INCLUDE_DIR"
FLAGS="-std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function $FP -fPIC -DCL_TARGET_OPENCL_VERSION=300"
mkdir -p "$OUT"

echo "--- 环境 ---"
echo "HERE=$HERE OUT=$OUT"
echo "OPENCV_ABI=$OPENCV_ABI  OPENCV_VERSION=$OPENCV_VERSION"
echo "OPENCV_INCLUDE_DIR=$OPENCV_INCLUDE_DIR"
echo "OPENCV_LIB_DIR=$OPENCV_LIB_DIR"
echo "BOARD_LIBS=$BOARD_LIBS  RT_LIB_DIR=$RT_LIB_DIR"
echo "OpenCL 链接项=$OCL_LINK   RGA 链接项=$RGA_LINK   RKNN 链接项=$RTKNN_LINK"
echo "--- 源文件 md5（交付基线核对）---"
md5sum "$SRC/mosaic_track_fusion.h" "$SRC/mosaic_track_fusion.cpp" \
       "$SRC/mosaic_pipeline.h" "$SRC/mosaic_pipeline.cpp" "$SRC/mosaic_track_dump.cpp"

echo "--- 编译中间产物 ---"
$CXX $FLAGS $INC -c "$SRC/mosaic_pipeline.cpp" -o "$OUT/mosaic_pipeline.o"
$CXX $FLAGS $INC -c "$SRC/mosaic_track_fusion.cpp" -o "$OUT/mosaic_track_fusion.o"

echo "--- 链接共享库 ---"
# --no-as-needed：Debian 的 gcc **默认 --as-needed**，会把链了但没用到的
#   libopencv_features2d / libopencv_flann 从 NEEDED 里丢掉；
#   而交付产物（交叉编译，buildroot 工具链不带 as-needed）保留 7 个 OpenCV 模块
#   ⇒ 加上这个开关，板卡原生重编出来的 .so 与交付产物的 NEEDED 集合保持一致。
$CXX -shared -Wl,-soname,libmosaic_pipeline.so -Wl,--allow-shlib-undefined \
  -Wl,--no-as-needed \
  -o "$OUT/libmosaic_pipeline.so" "$OUT/mosaic_pipeline.o" "$OUT/mosaic_track_fusion.o" \
  "${LIB_PATHS[@]}" \
  -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video \
  -lopencv_imgproc -lopencv_core "$OCL_LINK" "$RGA_LINK" "$RTKNN_LINK" -lpthread -ldl -lm

echo "--- 归档静态库（与共享库同源同对象文件）---"
find "$OUT" -maxdepth 1 -name 'libmosaic_pipeline.a' -delete 2>/dev/null || true
ar rcsD "$OUT/libmosaic_pipeline.a" "$OUT/mosaic_pipeline.o" "$OUT/mosaic_track_fusion.o" 2>/dev/null \
  || ar rcs "$OUT/libmosaic_pipeline.a" "$OUT/mosaic_pipeline.o" "$OUT/mosaic_track_fusion.o"
[ -f "$OUT/libmosaic_pipeline.a" ] || { echo "[FATAL] 静态库归档失败" >&2; exit 1; }

echo "--- 编译 stat 垫片（板卡自己的 glibc 版本决定要不要它）---"
$CXX -std=c++17 -O2 -fPIC -c "$SRC/stat_compat.c" -o "$OUT/stat_compat.o"

echo "--- 链接静态版 demo（板卡原生 g++ 链 .a）---"
$CXX -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function $FP \
  -Wl,--no-as-needed \
  -I"$HERE/include" -I"$HERE/include/rga" -I"$HERE/src/cpp" -I"$HERE/src/opencl/host" -I"$OPENCV_INCLUDE_DIR" \
  -DCL_TARGET_OPENCL_VERSION=300 \
  "$SRC/mosaic_api_test.cpp" -o "$OUT/mosaic_api_test_static" \
  "$OUT/libmosaic_pipeline.a" "$OUT/stat_compat.o" \
  "${LIB_PATHS[@]}" \
  -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video \
  -lopencv_imgproc -lopencv_core "$OCL_LINK" "$RGA_LINK" "$RTKNN_LINK" -lpthread -ldl -lm
if readelf -d "$OUT/mosaic_api_test_static" 2>/dev/null | grep -q 'libmosaic_pipeline\.so'; then
  echo "[FATAL] mosaic_api_test_static 仍然依赖 libmosaic_pipeline.so ==> 静态链接没生效" >&2
  exit 1
fi
echo "  [ok] 静态链接生效（NEEDED 里没有 libmosaic_pipeline.so）"

echo "--- 链接板卡验证探针（零 OpenCV）---"
for t in mosaic_cadence_probe mosaic_track_dump; do
  $CXX -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function $FP \
    -I"$HERE/include" -I"$SRC" "$SRC/$t.cpp" -o "$OUT/$t" \
    -L"$OUT" -Wl,--allow-shlib-undefined -Wl,-rpath,'$ORIGIN' \
    -lmosaic_pipeline -lpthread -ldl -lm
done

echo "--- 产物 md5 ---"
md5sum "$OUT/libmosaic_pipeline.so" "$OUT/libmosaic_pipeline.a" "$OUT/mosaic_track_dump" "$OUT/mosaic_cadence_probe" "$OUT/mosaic_api_test_static"

echo "--- 产物 NEEDED（确认与 app 链的是同一套 OpenCV）---"
readelf -d "$OUT/libmosaic_pipeline.so" | grep NEEDED || true

echo "--- 跨尺度候选合并是否在库里 ---"
nm -D "$OUT/libmosaic_pipeline.so" | grep -i MergeCandidatesByRadius || echo "SYMBOL MISSING"
echo "--- 新 CLI 开关是否在探针里 ---"
strings "$OUT/mosaic_track_dump" | grep -i "cand-dedup-radius" || echo "FLAG MISSING"
echo "=== BUILD_BOARD_NATIVE_OK ==="
