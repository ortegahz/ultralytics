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
# ---- 目标板默认（2026-11-25 实测：Orange Pi 5 / Debian 12 / glibc 2.36）----
#   root@192.168.1.115
#   * OpenCV 运行库有两套，**必须按 ABI 挑**：
#       - 4.8：/usr/lib/libopencv_*.so.408                    ← app 与本 SDK 用的那套
#       - 4.6：/usr/lib/aarch64-linux-gnu/libopencv_*.so.406   ← Debian 的 libopencv-core406
#   * OpenCV **头**板上只有 4.6（/usr/include/opencv4）⇒ 头/库 ABI 错配，
#     脚本会**如实报告**并给出两条出路（--headers 外部带 4.8 头，或把 4.8 头放到板上）。
#   * OpenCL：板上只有 libOpenCL.so.1（Debian ICD loader），没有 libOpenCL.so 开发软链
#     ⇒ stage 里补一个 **libOpenCL.so -> libOpenCL.so.1 软链**（链接垫片）。
#   * RGA：/usr/lib/aarch64-linux-gnu/librga.so -> librga.so.2
#     （/usr/lib/librga.so 是 Rockchip BSP 的**另一份**，app 不用它 ⇒ 不取）
#
# 取什么
# ---------------------------------------------------------------------------
#   <dest>/opencv/include/opencv4/   ← 板卡 OpenCV 头（4.8 头需 --headers 带进来）
#   <dest>/opencv/lib/               ← 板卡 libopencv_<mod>.so.$OPENCV_ABI
#   <dest>/boardlibs/                ← libOpenCL.so.1(+垫片软链) + librga.so.2(+垫片软链)
#   <dest>/rknnrt/                   ← 板卡 librknnrt.so
#
#   ⚠️ OpenCV 的**传递依赖**（libjpeg/libpng/libtiff/libgdal/...）**不用取**：
#      它们只在板卡运行时由板卡 ldconfig 解析；交叉链接时用
#      `-Wl,--allow-shlib-undefined` 跳过（build_lib.sh 已内置）。
#
# 用法
# ---------------------------------------------------------------------------
#   bash fetch_board_libs.sh [--board root@192.168.1.115] [--dest <目录>]
#                            [--port 22] [--abi 408] [--headers <4.8头目录>]
#                            [--askpass <脚本>] [--known-hosts <文件>] [--keep-closure]
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
#
# ---- 认证（目标板只有密码认证）----
#   * 免密：直接 ssh（板卡上已配好 root 公钥时）
#   * 密码：BOARD_PASSWORD=qwe123（需要本机有 sshpass）
#   * 或用 SSH_ASKPASS 助手：--askpass <脚本路径>（脚本内部 setsid + SSH_ASKPASS_REQUIRE=force）
# ===========================================================================
set -euo pipefail
export LC_ALL=C

die() { echo "[ERROR] $*" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BOARD="${BOARD:-root@192.168.1.115}"
BOARD_PORT="${BOARD_PORT:-22}"
BOARD_PASSWORD="${BOARD_PASSWORD:-}"
OPENCV_ABI="${OPENCV_ABI:-408}"
DEST=""
HEADERS_DIR=""
ASKPASS_SCRIPT=""
KNOWN_HOSTS_OPT=""
KEEP_CLOSURE=0
STRICT_HEADERS=0

usage() { sed -n '2,80p' "${BASH_SOURCE[0]}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --board) BOARD="$2"; shift 2 ;;
    --port) BOARD_PORT="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --abi) OPENCV_ABI="$2"; shift 2 ;;
    --headers) HEADERS_DIR="$2"; shift 2 ;;
    --askpass) ASKPASS_SCRIPT="$2"; shift 2 ;;
    --known-hosts) KNOWN_HOSTS_OPT="$2"; shift 2 ;;
    --strict-headers) STRICT_HEADERS=1; shift ;;
    --keep-closure) KEEP_CLOSURE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

# OpenCV ABI → 期望的**头**版本（OpenCV 的 soversion 规则 major*100+minor）
case "$OPENCV_ABI" in
  406) OPENCV_VERSION=4.6 ;;
  408) OPENCV_VERSION=4.8 ;;
  410) OPENCV_VERSION=4.10 ;;
  *) die "不支持的 --abi '$OPENCV_ABI'（可用 406 / 408 / 410）" ;;
esac

[ -n "$DEST" ] || DEST="$PWD/boardlibs"

# ssh：主机密钥基线。默认用包里钉死的 known_hosts；目标板不在基线里时，
#   退化成"会话内 accept-new"（写进 $DEST/.known_hosts.session，**不动仓库里的基线**）。
KNOWN_HOSTS="${KNOWN_HOSTS_OPT:-$HERE/known_hosts}"
[ -f "$KNOWN_HOSTS" ] || KNOWN_HOSTS="$(dirname "$HERE")/packaging/known_hosts"
[ -f "$KNOWN_HOSTS" ] || die "找不到 known_hosts（板卡主机密钥基线）：$HERE/known_hosts"

SSH_OPTS=(-F /dev/null -o UserKnownHostsFile="$KNOWN_HOSTS"
          -o StrictHostKeyChecking=yes -o LogLevel=ERROR -o ConnectTimeout=15)

board_host="${BOARD##*@}"
host_pinned=0
if ssh-keygen -F "$board_host" -f "$KNOWN_HOSTS" >/dev/null 2>&1; then
  host_pinned=1
elif [ "$BOARD_PORT" != "22" ] && ssh-keygen -F "[$board_host]:$BOARD_PORT" -f "$KNOWN_HOSTS" >/dev/null 2>&1; then
  host_pinned=1
fi
SESSION_KH="$DEST/.known_hosts.session"
if [ "$host_pinned" -eq 0 ]; then
  mkdir -p "$DEST"
  : > "$SESSION_KH"
  SSH_OPTS=(-F /dev/null -o UserKnownHostsFile="$SESSION_KH"
            -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -o ConnectTimeout=15)
fi

# 认证：sshpass（有密码时）优先；否则 SSH_ASKPASS 助手；否则裸 ssh（免密）
SSH_CMD=(ssh "${SSH_OPTS[@]}" -p "$BOARD_PORT")
if [ -n "$BOARD_PASSWORD" ]; then
  command -v sshpass >/dev/null 2>&1 \
    || die "设了 BOARD_PASSWORD，但本机没有 sshpass ⇒ 请安装 sshpass，或改用 --askpass <脚本> 提供 SSH_ASKPASS 助手"
  SSH_CMD=(sshpass -p "$BOARD_PASSWORD" "${SSH_CMD[@]}")
elif [ -n "$ASKPASS_SCRIPT" ]; then
  [ -x "$ASKPASS_SCRIPT" ] || die "--askpass 指定的脚本不存在或不可执行：$ASKPASS_SCRIPT"
  export SSH_ASKPASS="$ASKPASS_SCRIPT"
  export SSH_ASKPASS_REQUIRE=force
  # setsid：让 ssh 脱离当前终端 ⇒ 走 SSH_ASKPASS 而不是终端提示
  SSH_CMD=(setsid -w ssh "${SSH_OPTS[@]}" -p "$BOARD_PORT")
fi

echo "==========================================================="
echo " 取回板卡自带的库与头 -> $DEST"
echo "==========================================================="
echo "  板卡 : $BOARD:$BOARD_PORT"
echo "  暂存 : $DEST"
echo "  OpenCV : ABI=.$OPENCV_ABI（期望头 ${OPENCV_VERSION}）"
if [ "$host_pinned" -eq 1 ]; then
  echo "  主机密钥 : 钉死基线 $KNOWN_HOSTS"
else
  echo "  主机密钥 : 不在基线里 ⇒ 会话内 accept-new（$SESSION_KH，不改仓库里的 known_hosts）"
fi
echo

mkdir -p "$DEST"
TMP_TAR="$DEST/.boardlibs.tar"
SSH_LOG="$DEST/.boardlibs.ssh.log"
find "$DEST" -maxdepth 1 -name '.boardlibs.tar' -delete 2>/dev/null || true

SKIP_HEADERS=0
[ -n "$HEADERS_DIR" ] && SKIP_HEADERS=1

echo "== [1/4] 在板卡上打包（头 + 7 个 OpenCV .so + OpenCL/RGA/RKNN）=="
if ! "${SSH_CMD[@]}" "$BOARD" \
      "bash -s -- '$KEEP_CLOSURE' '$OPENCV_ABI' '$OPENCV_VERSION' '$SKIP_HEADERS'" \
      <<'REMOTE' > "$TMP_TAR" 2> "$SSH_LOG"
set -uo pipefail
export LC_ALL=C
keep_closure="$1"; abi="$2"; want_ver="$3"; skip_headers="$4"

# 读一个 OpenCV 头目录的版本
ocv_ver() {
  local vh="$1/opencv2/core/version.hpp" major minor
  [ -f "$vh" ] || return 1
  major="$(sed -n 's/^[[:space:]]*#define[[:space:]]\{1,\}CV_VERSION_MAJOR[[:space:]]\{1,\}\([0-9]\{1,\}\).*/\1/p' "$vh" | head -1)"
  minor="$(sed -n 's/^[[:space:]]*#define[[:space:]]\{1,\}CV_VERSION_MINOR[[:space:]]\{1,\}\([0-9]\{1,\}\).*/\1/p' "$vh" | head -1)"
  [ -n "$major" ] && [ -n "$minor" ] || return 1
  echo "$major.$minor"
}

stage="$(mktemp -d)"
cleanup() { find "$stage" -mindepth 1 -delete 2>/dev/null; rmdir "$stage" 2>/dev/null; }
trap cleanup EXIT
mkdir -p "$stage/opencv/include" "$stage/opencv/lib" "$stage/boardlibs" "$stage/rknnrt"

# ---- OpenCV 头：优先命中"与目标 ABI 同版本"的目录（4.8 头放 /root/cv480inc 即可命中）----
hdr_match=""; hdr_fallback=""
for c in /root/cv480inc /usr/local/include/opencv4 /usr/include/opencv4 /usr/include/aarch64-linux-gnu/opencv4; do
  [ -d "$c/opencv2" ] || continue
  v="$(ocv_ver "$c" || true)"
  if [ "$v" = "$want_ver" ]; then hdr_match="$c"; break; fi
  [ -n "$hdr_fallback" ] || hdr_fallback="$c"
done
hdr_dir="$hdr_match"
[ -n "$hdr_dir" ] || hdr_dir="$hdr_fallback"
hdr_ver=""
[ -n "$hdr_dir" ] && hdr_ver="$(ocv_ver "$hdr_dir" || true)"

if [ "$skip_headers" != "1" ]; then
  if [ -n "$hdr_dir" ]; then
    cp -r "$hdr_dir" "$stage/opencv/include/opencv4"
  else
    echo "[WARN] 板卡上找不到 OpenCV 头目录" >&2
  fi
fi

# ---- OpenCV 运行库：**按 ABI 后缀取**（不再"目录里有什么版本就取什么"）----
#   候选顺序 /usr/lib 在前：新板 4.8 在 /usr/lib，4.6 在 multiarch 目录。
libdirs="/usr/lib /usr/lib/aarch64-linux-gnu /usr/lib64 /usr/local/lib"
for m in core imgproc imgcodecs calib3d features2d flann video; do
  real=""
  for d in $libdirs; do
    [ -e "$d/libopencv_${m}.so.$abi" ] && { real="$d/libopencv_${m}.so.$abi"; break; }
  done
  if [ -z "$real" ]; then
    for d in $libdirs; do
      cand="$d/libopencv_${m}.so"
      [ -e "$cand" ] || continue
      sn="$(readelf -d "$cand" 2>/dev/null | sed -n 's/.*SONAME[^[]*\[\([^]]*\)\].*/\1/p' | head -1)"
      [ "$sn" = "libopencv_${m}.so.$abi" ] && { real="$cand"; break; }
    done
  fi
  if [ -z "$real" ]; then
    echo "[ERROR] 板卡上找不到 libopencv_${m}.so.$abi（当前 ABI=.$abi）" >&2
    for d in $libdirs; do
      [ -d "$d" ] && echo "        $d: $(ls "$d" 2>/dev/null | grep -E "^libopencv_${m}\.so" | tr '\n' ' ')" >&2
    done
    exit 1
  fi
  cp -L "$real" "$stage/opencv/lib/libopencv_${m}.so.$abi"
  ln -sf "libopencv_${m}.so.$abi" "$stage/opencv/lib/libopencv_${m}.so"
done

# ---- 板卡自带的 OpenCL / RGA / RKNN Runtime ----
ocl=""
for d in /usr/lib/aarch64-linux-gnu /lib/aarch64-linux-gnu /usr/lib /lib; do
  [ -e "$d/libOpenCL.so.1" ] && { ocl="$d/libOpenCL.so.1"; break; }
done
[ -n "$ocl" ] || { echo "[ERROR] 板卡上找不到 libOpenCL.so.1" >&2; exit 1; }
cp -L "$ocl" "$stage/boardlibs/libOpenCL.so.1"
ocl_dev=""
for d in /usr/lib/aarch64-linux-gnu /usr/lib; do
  [ -e "$d/libOpenCL.so" ] && { ocl_dev="$d/libOpenCL.so"; break; }
done

# RGA 只取 multiarch 那份（SONAME librga.so.2，app 解析到的就是它）；
# /usr/lib/librga.so 是 Rockchip BSP 的**另一份**文件 ⇒ 不取。
rga=""
for d in /usr/lib/aarch64-linux-gnu /lib/aarch64-linux-gnu /usr/lib /lib; do
  [ -e "$d/librga.so.2" ] && { rga="$d/librga.so.2"; break; }
done
[ -n "$rga" ] || { echo "[ERROR] 板卡上找不到 librga.so.2" >&2; exit 1; }
cp -L "$rga" "$stage/boardlibs/librga.so.2"

rknn=""
for d in /usr/lib /lib /usr/lib/aarch64-linux-gnu; do
  [ -e "$d/librknnrt.so" ] && { rknn="$d/librknnrt.so"; break; }
done
[ -n "$rknn" ] || { echo "[ERROR] 板卡上找不到 librknnrt.so" >&2; exit 1; }
cp -L "$rknn" "$stage/rknnrt/librknnrt.so"

# ---- 可选：OpenCV 的传递依赖闭包（一般不需要；--keep-closure 才做）----
if [ "$keep_closure" = "1" ]; then
  mkdir -p "$stage/opencv/deps"
  queue=()
  for f in "$stage"/opencv/lib/*.so.* "$stage"/boardlibs/libOpenCL.so.1 "$stage"/rknnrt/librknnrt.so "$stage"/boardlibs/librga.so.2; do
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
  echo "### BOARD_OPENCV_HEADER_DIR"
  echo "${hdr_dir:-（板上没找到 OpenCV 头目录）}"
  echo "### BOARD_OPENCV_HEADER_VERSION"
  echo "${hdr_ver:-未知}"
  echo "### TARGET_OPENCV_ABI"
  echo ".$abi"
  echo "### TARGET_OPENCV_VERSION"
  echo "$want_ver"
  echo "### BOARD_OPENCL_DEV_LINK"
  echo "${ocl_dev:-（板上没有 libOpenCL.so 开发软链 ⇒ stage 里用垫片软链）}"
  echo "### BOARD_RKNNRT_VERSION"
  strings "$stage/rknnrt/librknnrt.so" 2>/dev/null | grep -i "librknnrt version" | head -1
  echo "### BOARD_RGA_VERSION"
  strings "$stage/boardlibs/librga.so.2" 2>/dev/null | grep -i "rga_api version" | head -1
} > "$stage/VERSIONS.txt" 2>/dev/null || true

cd "$stage" && tar cf - .
REMOTE
then
  echo "[ERROR] 从板卡取库失败 —— ssh 日志：" >&2
  sed 's/^/  /' "$SSH_LOG" >&2 || true
  exit 1
fi
if [ -s "$SSH_LOG" ]; then
  echo "  [ssh 提示]"; sed 's/^/    /' "$SSH_LOG"
fi

echo "== [2/4] 解包到宿主机暂存目录 =="
tar xf "$TMP_TAR" -C "$DEST"
find "$DEST" -maxdepth 1 -name '.boardlibs.tar' -delete 2>/dev/null || true

# --- OpenCV 头：优先用 --headers 指定的宿主机 4.8 头，否则用板上那份（并核对版本）---
board_hdr_dir="$(sed -n '/^### BOARD_OPENCV_HEADER_DIR$/{n;p;}' "$DEST/VERSIONS.txt" 2>/dev/null | tr -d '[:space:]')"
board_hdr_ver="$(sed -n '/^### BOARD_OPENCV_HEADER_VERSION$/{n;p;}' "$DEST/VERSIONS.txt" 2>/dev/null | tr -d '[:space:]')"
hdr_note=""
if [ -n "$HEADERS_DIR" ]; then
  hdr_src="$HEADERS_DIR"
  [ -d "$hdr_src/opencv2" ] || { [ -d "$hdr_src/opencv4/opencv2" ] && hdr_src="$hdr_src/opencv4"; }
  [ -d "$hdr_src/opencv2" ] || die "--headers $HEADERS_DIR 里找不到 opencv2/ ⇒ 不是 OpenCV 头目录"
  _hv="$(sed -n 's/^[[:space:]]*#define[[:space:]]\{1,\}CV_VERSION_MAJOR[[:space:]]\{1,\}\([0-9]\{1,\}\).*/\1/p' "$hdr_src/opencv2/core/version.hpp" 2>/dev/null | head -1)"
  _hv2="$(sed -n 's/^[[:space:]]*#define[[:space:]]\{1,\}CV_VERSION_MINOR[[:space:]]\{1,\}\([0-9]\{1,\}\).*/\1/p' "$hdr_src/opencv2/core/version.hpp" 2>/dev/null | head -1)"
  [ -n "$_hv" ] && [ -n "$_hv2" ] || die "--headers $hdr_src 里读不出 opencv2/core/version.hpp 的版本"
  _hv="$_hv.$_hv2"
  [ "$_hv" = "$OPENCV_VERSION" ] \
    || die "--headers 指定的头是 OpenCV $_hv，但目标 ABI 是 .$OPENCV_ABI（需要 ${OPENCV_VERSION} 头）"
  find "$DEST/opencv/include/opencv4" -mindepth 1 -delete 2>/dev/null || true
  rmdir "$DEST/opencv/include/opencv4" 2>/dev/null || true
  mkdir -p "$DEST/opencv/include"
  cp -r "$hdr_src" "$DEST/opencv/include/opencv4"
  hdr_note="来自 --headers $hdr_src（宿主机带进来的 OpenCV ${OPENCV_VERSION} 头）"
else
  hdr_note="来自板卡 $board_hdr_dir（OpenCV ${board_hdr_ver:-未知} 头）"
fi

# --- 链接垫片：板卡没有 libOpenCL.so / librga.so 的开发软链 ---
#    宿主机侧 build_lib.sh / build_demo.sh 用 -lOpenCL / -lrga ⇒ stage 里必须有这两个名字。
#    ⇒ 造软链指向真实文件（真实文件的 SONAME 不变 ⇒ 产物 NEEDED 仍是 libOpenCL.so.1 / librga.so.2）。
for pair in "libOpenCL.so:libOpenCL.so.1" "librga.so:librga.so.2"; do
  _ln="${pair%%:*}"; _tgt="${pair##*:}"
  [ -e "$DEST/boardlibs/$_tgt" ] || die "解包后缺 $_tgt（垫片软链没法建）"
  [ -L "$DEST/boardlibs/$_ln" ] || find "$DEST/boardlibs" -maxdepth 1 -name "$_ln" -delete 2>/dev/null || true
  ln -sfn "$_tgt" "$DEST/boardlibs/$_ln"
done

for m in core imgproc imgcodecs calib3d features2d flann video; do
  [ -e "$DEST/opencv/lib/libopencv_${m}.so" ] || die "解包后缺 libopencv_${m}.so"
done
[ -e "$DEST/boardlibs/libOpenCL.so.1" ] || die "解包后缺 libOpenCL.so.1"
[ -e "$DEST/boardlibs/librga.so.2" ]      || die "解包后缺 librga.so.2"
[ -e "$DEST/rknnrt/librknnrt.so" ]        || die "解包后缺 librknnrt.so"
[ -d "$DEST/opencv/include/opencv4/opencv2" ] || die "解包后缺 OpenCV 头"

echo "== [3/4] 版本核对 =="
if [ -f "$DEST/VERSIONS.txt" ]; then
  sed 's/^/  /' "$DEST/VERSIONS.txt"
fi

# 板上头 vs 目标 ABI：如实报告；不一致就**大声**说清两条出路（不拿 4.6 头悄悄凑数）
if [ -z "$HEADERS_DIR" ] && [ -n "$board_hdr_ver" ] && [ "$board_hdr_ver" != "$OPENCV_VERSION" ]; then
  {
    echo "  [头/库 ABI 错配 —— 必须处理]"
    echo "    板上 OpenCV 头 : $board_hdr_ver   （$board_hdr_dir）"
    echo "    本 SDK 要链的库 : OpenCV $OPENCV_VERSION（libopencv_core.so.$OPENCV_ABI，app 链的就是它）"
    echo "    ⇒ 用 $board_hdr_ver 头编 $OPENCV_VERSION 库 = 结构体布局不匹配 ⇒ 运行期踩内存。"
    echo "    出路 1：把 ${OPENCV_VERSION} 头放到板上（如 /root/cv480inc），重跑本脚本即可自动命中。"
    echo "    出路 2：从宿主机带进去："
    echo "           bash fetch_board_libs.sh --board $BOARD --dest $DEST --headers <本地 ${OPENCV_VERSION} 头目录>"
  } >&2
  {
    echo "板上 OpenCV 头是 $board_hdr_ver（$board_hdr_dir），但目标运行库是 libopencv_core.so.$OPENCV_ABI（OpenCV $OPENCV_VERSION）。"
    echo "头/库版本不一致会编出错误 ABI 的产物。两条出路："
    echo "  1) 把 ${OPENCV_VERSION} 头放到板上（如 /root/cv480inc），重跑 fetch_board_libs.sh。"
    echo "  2) 用 --headers <本地 ${OPENCV_VERSION} 头目录> 从宿主机带进去。"
  } > "$DEST/OPENCV_HEADERS_MISMATCH.txt"
  if [ "$STRICT_HEADERS" -eq 1 ]; then
    die "板上头版本 $board_hdr_ver ≠ 目标 $OPENCV_VERSION（--strict-headers）⇒ 详见 $DEST/OPENCV_HEADERS_MISMATCH.txt"
  fi
elif [ -z "$HEADERS_DIR" ]; then
  echo "  OpenCV 头版本 : $board_hdr_ver（与运行库 ABI .$OPENCV_ABI 一致）"
fi

echo "  OpenCV 头来源 : $hdr_note"
echo "  OpenCV 模块   : $(find "$DEST/opencv/lib" -maxdepth 1 -name 'libopencv_*.so' | wc -l) 个（每个都有 .so 软链 + .so.$OPENCV_ABI 实体）"
echo "  暂存总大小    : $(du -sh "$DEST" | cut -f1)"
echo "  --- 各 .so 的 SONAME（确认拿的是对的那一套）---"
for so in "$DEST"/opencv/lib/*.so.* "$DEST"/boardlibs/*.so.* "$DEST"/rknnrt/*.so; do
  [ -e "$so" ] || continue
  sn="$(readelf -d "$so" 2>/dev/null | sed -n 's/.*SONAME[^[]*\[\([^]]*\)\].*/\1/p' | head -1)"
  printf '    %-34s %s\n' "$(basename "$so")" "${sn:-（无 SONAME）}"
done
echo "  --- 链接垫片（板卡上没有，只为宿主机侧 -lOpenCL / -lrga 能链接）---"
for l in libOpenCL.so librga.so; do
  printf '    %-16s -> %s\n' "$l" "$(readlink "$DEST/boardlibs/$l")"
done
echo

cat <<EOF

===========================================================
 取回完成。**照抄下面这几行**再跑 build_lib.sh / build_demo.sh：

export OPENCV_ABI="$OPENCV_ABI"
export OPENCV_VERSION="$OPENCV_VERSION"
export OPENCV_INCLUDE_DIR="$DEST/opencv/include/opencv4"
export OPENCV_LIB_DIR="$DEST/opencv/lib"
export BOARD_LIBS="$DEST/boardlibs"
export RT_LIB_DIR="$DEST/rknnrt"

bash build_lib.sh && bash build_demo.sh

===========================================================
EOF
