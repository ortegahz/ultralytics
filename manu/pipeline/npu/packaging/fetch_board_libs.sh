#!/usr/bin/env bash
# ===========================================================================
# fetch_board_libs.sh -- 把**板卡自带的**库与头取回宿主机，供交叉编译时链接。
#
# 为什么需要这个脚本
# ---------------------------------------------------------------------------
#   本包**不携带**任何第三方 .so/.a：OpenCV / RKNN Runtime / RGA / OpenCL 全部
#   链接**板卡上原来就有的**动态库。
#
#   但交叉编译是在**宿主机**上做的，而宿主机上的 /usr 是 x86 的
#   ⇒ 宿主机 /usr/include/opencv4（x86 的头）**绝对不能**拿来编 aarch64 的包；
#   宿主机上的同名 .so 也不是板卡那一份。
#
#   ⇒ 正确做法：先把**板卡那一份**（头 + .so）取回宿主机的一个暂存目录，
#     再把 build_lib.sh / build_demo.sh 的路径变量指向那个暂存目录。
#
# 取什么
# ---------------------------------------------------------------------------
#   <dest>/opencv/include/opencv4/   ← 板卡 /usr/include/opencv4      （编译期要头）
#   <dest>/opencv/lib/               ← 板卡 libopencv_*.so.406        （链接期要 .so）
#   <dest>/boardlibs/                ← 板卡 libOpenCL.so.1 + librga.so.2
#   <dest>/rknnrt/                   ← 板卡 librknnrt.so
#
#   ⚠️ OpenCV 的**传递依赖**（libjpeg/libpng/libtiff/libgdal/...）**不用取**：
#      它们只在板卡运行时由板卡 ldconfig 解析；交叉链接时用
#      `-Wl,--allow-shlib-undefined` 跳过（build_lib.sh 已内置）。
#
# 用法
# ---------------------------------------------------------------------------
#   bash fetch_board_libs.sh [--board root@192.168.0.211] [--dest <目录>]
#                            [--port 22] [--keep-closure]
#
#   典型流程：
#     bash fetch_board_libs.sh --dest ~/rk3588_boardlibs
#     export OPENCV_INCLUDE_DIR=~/rk3588_boardlibs/opencv/include/opencv4
#     export OPENCV_LIB_DIR=~/rk3588_boardlibs/opencv/lib
#     export BOARD_LIBS=~/rk3588_boardlibs/boardlibs
#     export RT_LIB_DIR=~/rk3588_boardlibs/rknnrt
#     bash build_lib.sh && bash build_demo.sh
#
#   脚本跑完会**直接把上面四个 export 打印出来**，照抄即可。
# ===========================================================================
set -euo pipefail
export LC_ALL=C

die() { echo "[ERROR] $*" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BOARD="${BOARD:-root@192.168.0.211}"
BOARD_PORT="${BOARD_PORT:-22}"
DEST=""
KEEP_CLOSURE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --board) BOARD="$2"; shift 2 ;;
    --port) BOARD_PORT="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --keep-closure) KEEP_CLOSURE=1; shift ;;
    -h|--help) sed -n '2,60p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

# 板卡多架构库目录不固定，自动探测
detect_multiarch() {
  for d in /usr/lib/aarch64-linux-gnu /usr/lib64 /usr/lib; do
    [ -d "$d" ] && { echo "$d"; return 0; }
  done
  echo /usr/lib
}

[ -n "$DEST" ] || DEST="$PWD/boardlibs"

# ssh：主机密钥必须钉死（交付脚本是以 root 身份去板卡执行二进制的）
KNOWN_HOSTS="$HERE/known_hosts"
[ -f "$KNOWN_HOSTS" ] || KNOWN_HOSTS="$(dirname "$HERE")/packaging/known_hosts"
[ -f "$KNOWN_HOSTS" ] || die "找不到 known_hosts（板卡主机密钥基线）：$HERE/known_hosts"
SSH_OPTS=(-F /dev/null -o UserKnownHostsFile="$KNOWN_HOSTS"
          -o StrictHostKeyChecking=yes -o LogLevel=ERROR -o ConnectTimeout=15)

echo "==========================================================="
echo " 取回板卡自带的库与头 -> $DEST"
echo "==========================================================="
echo "  板卡 : $BOARD:$BOARD_PORT"
echo "  暂存 : $DEST"
echo

mkdir -p "$DEST"
TMP_TAR="$DEST/.boardlibs.tar"
rm -f "$TMP_TAR"

echo "== [1/3] 在板卡上打包（头 + 7 个 OpenCV .so + OpenCL/RGA/RKNN）=="
if ! ssh "${SSH_OPTS[@]}" -p "$BOARD_PORT" "$BOARD" "bash -s -- '$KEEP_CLOSURE'" 2>&1 <<'REMOTE' > "$TMP_TAR"
set -uo pipefail
export LC_ALL=C
keep_closure="$1"

multiarch=""
for d in /usr/lib/aarch64-linux-gnu /usr/lib64 /usr/lib; do
  [ -d "$d" ] && { multiarch="$d"; break; }
done
[ -n "$multiarch" ] || multiarch=/usr/lib

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/opencv/include" "$stage/opencv/lib" "$stage/boardlibs" "$stage/rknnrt"

# --- OpenCV 头（编译期硬依赖）---
if [ -d /usr/include/opencv4 ]; then
  cp -r /usr/include/opencv4 "$stage/opencv/include/opencv4"
else
  echo "[ERROR] 板卡上找不到 /usr/include/opencv4" >&2
  exit 1
fi

# --- OpenCV 的 7 个模块（链接期硬依赖）---
#    拷贝**实体文件**并命名为 libopencv_<mod>.so（SONAME 仍是 .406），
#    这样 -lopencv_<mod> 能解析到，DT_NEEDED 仍然写 .406。
for m in core imgproc imgcodecs calib3d features2d flann video; do
  real=""
  for cand in "$multiarch/libopencv_${m}.so.406" "$multiarch/libopencv_${m}.so.408" "/usr/lib/libopencv_${m}.so.408"; do
    [ -e "$cand" ] && { real="$(readlink -f "$cand")"; break; }
  done
  if [ -z "$real" ]; then
    echo "[ERROR] 板卡上找不到 libopencv_${m}.so" >&2
    exit 1
  fi
  cp -L "$real" "$stage/opencv/lib/libopencv_${m}.so"
done

# --- 板卡自带的 OpenCL / RGA / RKNN Runtime ---
for f in "$multiarch/libOpenCL.so.1" "$multiarch/libOpenCL.so.1.0.0"; do
  [ -e "$f" ] && { cp -L "$f" "$stage/boardlibs/libOpenCL.so.1"; break; }
done
[ -f "$stage/boardlibs/libOpenCL.so.1" ] || { echo "[ERROR] 板卡上找不到 libOpenCL.so.1" >&2; exit 1; }
cp -L "$stage/boardlibs/libOpenCL.so.1" "$stage/boardlibs/libOpenCL.so"

for f in "$multiarch/librga.so.2" "$multiarch/librga.so.2.1.0"; do
  [ -e "$f" ] && { cp -L "$f" "$stage/boardlibs/librga.so.2"; break; }
done
[ -f "$stage/boardlibs/librga.so.2" ] || { echo "[ERROR] 板卡上找不到 librga.so.2" >&2; exit 1; }
cp -L "$stage/boardlibs/librga.so.2" "$stage/boardlibs/librga.so"

for f in /usr/lib/librknnrt.so "$multiarch/librknnrt.so"; do
  [ -e "$f" ] && { cp -L "$f" "$stage/rknnrt/librknnrt.so"; break; }
done
[ -f "$stage/rknnrt/librknnrt.so" ] || { echo "[ERROR] 板卡上找不到 librknnrt.so" >&2; exit 1; }

# --- 可选：OpenCV 的传递依赖闭包（一般不需要；--keep-closure 才做）---
if [ "$keep_closure" = "1" ]; then
  mkdir -p "$stage/opencv/deps"
  queue=()
  for f in "$stage"/opencv/lib/*.so "$stage"/boardlibs/libOpenCL.so.1 "$stage"/rknnrt/librknnrt.so "$stage"/boardlibs/librga.so.2; do
    queue+=("$f")
  done
  while [ ${#queue[@]} -gt 0 ]; do
    cur="${queue[0]}"; queue=("${queue[@]:1}")
    deps=$(ldd "$cur" 2>/dev/null | awk '$2=="=>" && $3 ~ /^\// {print $3}')
    for dep in $deps; do
      name="$(basename "$dep")"
      [ -e "$stage/opencv/deps/$name" ] && continue
      real="$(readlink -f "$dep" 2>/dev/null)" || continue
      [ -f "$real" ] || continue
      cp -L "$real" "$stage/opencv/deps/$name" 2>/dev/null || continue
      queue+=("$stage/opencv/deps/$name")
    done
  done
fi

# 版本信息（给宿主机的用户看）
{
  echo "### BOARD_OPENCV_VERSION"
  grep -E "CV_VERSION_(MAJOR|MINOR|REVISION)" /usr/include/opencv4/opencv2/core/version.hpp 2>/dev/null | sed 's/#define *//'
  echo "### BOARD_RKNNRT_VERSION"
  strings /usr/lib/librknnrt.so 2>/dev/null | grep -i "librknnrt version" | head -1
  echo "### BOARD_RGA_VERSION"
  strings "$multiarch/librga.so.2" 2>/dev/null | grep -i "rga_api version" | head -1
} > "$stage/VERSIONS.txt" 2>/dev/null || true

cd "$stage" && tar cf - .
REMOTE
then
  echo "[ERROR] 从板卡取库失败 —— 检查 ssh 连通性 / known_hosts 是否匹配这台板卡" >&2
  exit 1
fi

echo "== [2/3] 解包到宿主机暂存目录 =="
tar xf "$TMP_TAR" -C "$DEST"
rm -f "$TMP_TAR"

for m in core imgproc imgcodecs calib3d features2d flann video; do
  [ -f "$DEST/opencv/lib/libopencv_${m}.so" ] || die "解包后缺 libopencv_${m}.so"
done
[ -f "$DEST/boardlibs/libOpenCL.so.1" ] || die "解包后缺 libOpenCL.so.1"
[ -f "$DEST/boardlibs/librga.so.2" ]      || die "解包后缺 librga.so.2"
[ -f "$DEST/rknnrt/librknnrt.so" ]        || die "解包后缺 librknnrt.so"
[ -d "$DEST/opencv/include/opencv4/opencv2" ] || die "解包后缺 OpenCV 头"

echo "== [3/3] 版本核对 =="
if [ -f "$DEST/VERSIONS.txt" ]; then
  sed 's/^/  /' "$DEST/VERSIONS.txt"
fi
echo "  OpenCV 模块 : $(ls "$DEST/opencv/lib" | wc -l) 个"
echo "  暂存总大小  : $(du -sh "$DEST" | cut -f1)"
echo

cat <<EOF

===========================================================
 取回完成。**照抄下面这四行**再跑 build_lib.sh / build_demo.sh：

export OPENCV_INCLUDE_DIR="$DEST/opencv/include/opencv4"
export OPENCV_LIB_DIR="$DEST/opencv/lib"
export BOARD_LIBS="$DEST/boardlibs"
export RT_LIB_DIR="$DEST/rknnrt"

bash build_lib.sh && bash build_demo.sh

===========================================================
EOF
