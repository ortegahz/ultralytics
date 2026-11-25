#!/usr/bin/env bash
# ===========================================================================
# fetch_opencv_headers.sh -- 取一份**版本严格匹配**的 OpenCV 头到本地目录，
#                            供编译期使用（OPENCV_INCLUDE_DIR）。
#
# 为什么需要这个脚本
# ---------------------------------------------------------------------------
#   目标板（Orange Pi 5 / Debian 12）实测：
#     运行库 = 4.8    /usr/lib/libopencv_*.so.408（SONAME 后缀 408）
#     头     = 只有 4.6  /usr/include/opencv4（CV_VERSION_MINOR 6）
#   ⇒ 板卡原生构建会「4.6 头配 4.8 运行库」= **ABI 错配**：
#     OpenCV 次版本会改 C++ ABI（cv::Mat 布局 / 内联函数）
#     ⇒ 同一进程里会同时加载两套 OpenCV，cv::Mat 跨版本串味。
#   ⇒ 4.8 头必须**外部提供**。本脚本负责把它取回来（并**校验版本**）。
#
# 三种取法
# ---------------------------------------------------------------------------
#   1) 从**本地已有**的 4.8 头目录拷（最常见：开发机上从别处拿到的）：
#        bash fetch_opencv_headers.sh --src /path/to/opencv4 --dest ~/opencv4-headers
#   2) 从**板卡**拉（板上恰好有 4.8 头时）：
#        bash fetch_opencv_headers.sh --board root@192.168.1.115 \
#             --board-path /usr/include/opencv4 --dest ~/opencv4-headers
#   3) 板卡 ssh 要口令时：本脚本**不硬编码口令**，走 SSH_ASKPASS ——
#      自己写一个「回显口令」的小脚本（chmod 700），用 --askpass 指给它；
#      默认找 /tmp/board_askpass.sh。两者都没有 ⇒ 提示改用**免密 key**。
#
# 版本校验（**必须**精确匹配）
# ---------------------------------------------------------------------------
#   拿到的头若不是 --expect-abi（默认 408 ⇒ OpenCV 4.8）对应的版本 ⇒ 直接 die：
#   「用 4.6 头编 4.8 运行库会导致同一进程加载两套 OpenCV、cv::Mat 跨版本串味」。
#   宁可取不到头，也不要配错 ABI。
#
# 落地后怎么用
# ---------------------------------------------------------------------------
#   export OPENCV_INCLUDE_DIR=<dest>      # dest 里直接就是 opencv2/
#   export OPENCV_HEADERS_SRC=<dest>      # 给 make_build_dir.sh（打进交付包）
# ===========================================================================
set -euo pipefail
export LC_ALL=C

die() { echo "[ERROR] $*" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
用法: fetch_opencv_headers.sh [--src <目录>] [--board <host[:port]>]
                              [--board-path <板上的头目录>] [--dest <目录>]
                              [--expect-abi <408>] [--askpass <脚本路径>]

取一份**版本严格匹配**的 OpenCV 头到本地目录（默认 ./opencv4-headers）。
落地目录里直接就有 opencv2/ ⇒ 它本身就是 include 目录。

参数:
  --src <目录>           从**本地已有**的 OpenCV 头目录拷贝（最常见）
  --board <host[:port]>  从**板卡**拉（板上恰好有目标版本的头时）
  --board-path <目录>    板卡上的头目录（默认 /usr/include/opencv4）
  --port <端口>          ssh 端口（默认 22；--board host:port 优先）
  --dest <目录>          落地目录（默认 ./opencv4-headers）
  --expect-abi <408>     期望的 OpenCV SONAME 后缀（默认 408 ⇒ OpenCV 4.8）
  --askpass <脚本路径>   SSH 口令助手脚本（默认 /tmp/board_askpass.sh）
  -h, --help             显示本帮助

版本校验: 拿到的头若不是 --expect-abi 对应的版本 ⇒ 直接报错退出
          （用 4.6 头编 4.8 运行库会导致同一进程加载两套 OpenCV、
            cv::Mat 跨版本串味）。

落地后: export OPENCV_INCLUDE_DIR=<dest>
        export OPENCV_HEADERS_SRC=<dest>    # 打包进交付包时用
EOF
}

SRC=""
BOARD=""
BOARD_HOST=""
BOARD_PORT="22"
BOARD_PATH="/usr/include/opencv4"
DEST="./opencv4-headers"
EXPECT_ABI="408"
ASKPASS=""
KEEP_ASKPASS_DEFAULT=1

while [ $# -gt 0 ]; do
  case "$1" in
    --src)        [ $# -ge 2 ] || die "参数 $1 后面缺值"; SRC="$2"; shift 2 ;;
    --board)      [ $# -ge 2 ] || die "参数 $1 后面缺值"; BOARD="$2"; shift 2 ;;
    --board-path) [ $# -ge 2 ] || die "参数 $1 后面缺值"; BOARD_PATH="$2"; shift 2 ;;
    --port)       [ $# -ge 2 ] || die "参数 $1 后面缺值"; BOARD_PORT="$2"; shift 2 ;;
    --dest)       [ $# -ge 2 ] || die "参数 $1 后面缺值"; DEST="$2"; shift 2 ;;
    --expect-abi) [ $# -ge 2 ] || die "参数 $1 后面缺值"; EXPECT_ABI="$2"; shift 2 ;;
    --askpass)    [ $# -ge 2 ] || die "参数 $1 后面缺值"; ASKPASS="$2"; KEEP_ASKPASS_DEFAULT=0; shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *)            die "未知参数：$1（--help 看用法）" ;;
  esac
done

# ---------------------------------------------------------------------------
# 参数自检
# ---------------------------------------------------------------------------
[ -n "$SRC" ] || [ -n "$BOARD" ] || die "必须给 --src <目录> 或 --board <host[:port]>（--help 看用法）"
if [ -n "$SRC" ] && [ -n "$BOARD" ]; then
  die "--src 与 --board 只能给一个"
fi

case "$EXPECT_ABI" in
  [0-9][0-9][0-9]) ;;
  *) die "--expect-abi 必须是三位数字（OpenCV 运行库的 SONAME 后缀，如 408）：$EXPECT_ABI" ;;
esac
# SONAME 后缀 ⇒ 版本号：408 ⇒ 4.8、406 ⇒ 4.6、410 ⇒ 4.10
EXP_MAJOR="${EXPECT_ABI:0:1}"
EXP_MINOR=$((10#${EXPECT_ABI:1}))
EXP_VER="${EXP_MAJOR}.${EXP_MINOR}"

if [ -n "$BOARD" ]; then
  case "$BOARD" in
    *:*) BOARD_HOST="${BOARD%%:*}"; BOARD_PORT="${BOARD##*:}" ;;
    *)   BOARD_HOST="$BOARD" ;;
  esac
  [ -n "$BOARD_HOST" ] || die "--board 的主机名解析不出来：$BOARD"
fi

# ---------------------------------------------------------------------------
# 版本读取 / 校验
# ---------------------------------------------------------------------------
# 读 <dir>/opencv2/core/version.hpp → "MAJOR MINOR REVISION"（如 "4 8 0"）；读不到返回非 0
read_cv_version() {
  local hdr="$1/opencv2/core/version.hpp" major minor rev
  [ -f "$hdr" ] || return 1
  major=$(sed -nE 's/^[[:space:]]*#[[:space:]]*define[[:space:]]+CV_VERSION_MAJOR[[:space:]]+([0-9]+).*/\1/p' "$hdr" | head -1)
  minor=$(sed -nE 's/^[[:space:]]*#[[:space:]]*define[[:space:]]+CV_VERSION_MINOR[[:space:]]+([0-9]+).*/\1/p' "$hdr" | head -1)
  rev=$(sed -nE 's/^[[:space:]]*#[[:space:]]*define[[:space:]]+CV_VERSION_REVISION[[:space:]]+([0-9]+).*/\1/p' "$hdr" | head -1)
  [ -n "$major" ] && [ -n "$minor" ] || return 1
  printf '%s %s %s\n' "$major" "$minor" "${rev:-0}"
}

# 版本不符 / 读不出版本 ⇒ die（必须点名实际版本与期望版本）
check_cv_version() {
  local where="$1" raw="$2" major minor rev
  if [ -z "$raw" ]; then
    die "在 $where 找不到 opencv2/core/version.hpp
       ⇒ 读不出 CV_VERSION_MAJOR/MINOR，无法确认是不是 OpenCV $EXP_VER 的头
       ⇒ 拒绝使用（宁可取不到头，也不要配错 ABI）。"
  fi
  major="${raw%% *}"
  minor="${raw#* }"; minor="${minor%% *}"
  rev="${raw##* }"
  if [ "$major.$minor" != "$EXP_VER" ]; then
    die "这是 OpenCV $major.$minor 的头，与目标 ABI 不匹配（期望 OpenCV $EXP_VER，
       运行库 SONAME 后缀 libopencv_*.so.$EXPECT_ABI）；用 $major.$minor 头编
       $EXP_VER 运行库会导致同一进程加载两套 OpenCV、cv::Mat 跨版本串味
       ⇒ 拒绝使用。来源：$where"
  fi
  CV_MAJOR="$major"; CV_MINOR="$minor"; CV_REV="$rev"
}

# ---------------------------------------------------------------------------
# 临时目录清理（沙箱铁律：不用 rm -f / rm -rf）
# ---------------------------------------------------------------------------
TMP_TAR=""
STAGE_DIR=""
cleanup_tmp() {
  if [ -n "$TMP_TAR" ] && [ -e "$TMP_TAR" ]; then
    find "$TMP_TAR" -maxdepth 0 -delete 2>/dev/null || true
  fi
  if [ -n "$STAGE_DIR" ] && [ -d "$STAGE_DIR" ]; then
    find "$STAGE_DIR" -mindepth 1 -delete 2>/dev/null || true
    rmdir "$STAGE_DIR" 2>/dev/null || true
  fi
}
trap cleanup_tmp EXIT

echo "==========================================================="
echo " 取 OpenCV 头 -> $DEST"
echo "==========================================================="
echo "  目标 ABI   : $EXPECT_ABI  ⇒ 期望 OpenCV $EXP_VER"
if [ -n "$SRC" ]; then
  echo "  来源       : 本地目录 $SRC"
else
  echo "  来源       : 板卡 $BOARD_HOST:$BOARD_PORT  路径 $BOARD_PATH"
fi
echo "  落地目录   : $DEST"
echo

mkdir -p "$DEST"
DEST_ABS="$(cd "$DEST" && pwd)"
STAGE_DIR="$(mktemp -d "$DEST_ABS/.stage.XXXXXX")"

# ---------------------------------------------------------------------------
# 取头
# ---------------------------------------------------------------------------
if [ -n "$SRC" ]; then
  [ -d "$SRC" ] || die "--src 指定的目录不存在：$SRC"
  # 便利：--src 给的是**上一级**目录（含 opencv4/）时自动下探一层
  PICK="$SRC"
  if [ ! -f "$PICK/opencv2/core/version.hpp" ] && [ -f "$PICK/opencv4/opencv2/core/version.hpp" ]; then
    PICK="$SRC/opencv4"
    echo "  [--] $SRC 下没有 opencv2/ ⇒ 自动改用 $PICK"
  fi
  echo "== [1/2] 从本地目录拷贝 =="
  cp -r "$PICK"/. "$STAGE_DIR"/
  echo "== [2/2] 版本校验 =="
  raw="$(read_cv_version "$STAGE_DIR" || true)"
  check_cv_version "$PICK" "$raw"
else
  echo "== [1/2] 从板卡拉取 =="
  # --- 认证：本脚本**不硬编码口令** -------------------------------------
  #  有口令助手脚本 ⇒ SSH_ASKPASS + setsid -w（无 tty 也能过口令）
  #  没有 ⇒ 提示改用免密 key（ssh 走普通认证，可用 ssh-agent）
  ASK_EFF=""
  if [ "$KEEP_ASKPASS_DEFAULT" = "1" ]; then
    ASK_EFF="/tmp/board_askpass.sh"
  else
    ASK_EFF="$ASKPASS"
  fi
  if [ -n "$ASK_EFF" ] && [ -x "$ASK_EFF" ]; then
    echo "  口令助手   : $ASK_EFF（SSH_ASKPASS 方式）"
  else
    echo "  [--] 没找到可用的口令助手脚本（$ASK_EFF）"
    echo "  [--] ⇒ 改用**免密 key** 登录板卡（ssh-copy-id / ssh-agent）。"
    echo "  [--]    要口令方式就自己写一个回显口令的脚本（chmod 700）并用 --askpass 指给它。"
    ASK_EFF=""
  fi
  command -v ssh >/dev/null 2>&1 || die "找不到 ssh 命令"
  if [ -n "$ASK_EFF" ]; then
    command -v setsid >/dev/null 2>&1 || die "找不到 setsid（util-linux）：SSH_ASKPASS 方式需要它"
  fi

  SSH_OPTS=(-o ConnectTimeout=15 -o LogLevel=ERROR)
  if [ -f "$HERE/known_hosts" ]; then
    SSH_OPTS+=(-o UserKnownHostsFile="$HERE/known_hosts" -o StrictHostKeyChecking=yes)
  else
    SSH_OPTS+=(-o StrictHostKeyChecking=accept-new)
  fi

  TMP_TAR="$(mktemp)"
  if [ -n "$ASK_EFF" ]; then
    SSH_ASKPASS="$ASK_EFF" SSH_ASKPASS_REQUIRE=force DISPLAY="${DISPLAY:-:0}" \
      setsid -w ssh "${SSH_OPTS[@]}" -p "$BOARD_PORT" "$BOARD_HOST" \
        "tar cf - -C '$BOARD_PATH' ." > "$TMP_TAR" || \
      die "从板卡 $BOARD_HOST:$BOARD_PORT 取头失败（ssh 认证 / 网络 / 主机密钥 / 路径 $BOARD_PATH）"
  else
    ssh "${SSH_OPTS[@]}" -p "$BOARD_PORT" "$BOARD_HOST" \
      "tar cf - -C '$BOARD_PATH' ." > "$TMP_TAR" || \
      die "从板卡 $BOARD_HOST:$BOARD_PORT 取头失败（ssh 认证 / 网络 / 主机密钥 / 路径 $BOARD_PATH）"
  fi

  echo "== [2/2] 解包 + 版本校验 =="
  tar xf "$TMP_TAR" -C "$STAGE_DIR"
  raw="$(read_cv_version "$STAGE_DIR" || true)"
  check_cv_version "$BOARD_HOST:$BOARD_PATH" "$raw"
fi

# 校验通过 ⇒ 移进落地目录（**覆盖式**，可反复重跑）
#   mv 不能把目录覆盖到「已存在的非空目录」上 ⇒ 先把落地目录里**同名的一级条目**
#   删掉（只删同名条目，落地目录里的其它内容不动）。
while IFS= read -r top; do
  base="$(basename "$top")"
  if [ -e "$DEST_ABS/$base" ]; then
    find "$DEST_ABS/$base" -mindepth 0 -delete 2>/dev/null || true
  fi
done < <(find "$STAGE_DIR" -mindepth 1 -maxdepth 1)
find "$STAGE_DIR" -mindepth 1 -maxdepth 1 -exec mv {} "$DEST_ABS"/ \;
rmdir "$STAGE_DIR" 2>/dev/null || true
STAGE_DIR=""

# 落地后再确认一次版本（搬漏了/搬错了要当场发现）
raw="$(read_cv_version "$DEST_ABS" || true)"
check_cv_version "$DEST_ABS" "$raw"

n=$(find "$DEST_ABS" -type f | wc -l)
echo
echo "  文件数     : $n"
echo "  头版本     : OpenCV $CV_MAJOR.$CV_MINOR（CV_VERSION_MAJOR=$CV_MAJOR / CV_VERSION_MINOR=$CV_MINOR / CV_VERSION_REVISION=$CV_REV）"
echo "  落地目录   : $DEST_ABS"
echo

cat <<EOF
===========================================================
 取回完成。**照抄下面这行**再跑 build_lib.sh / build_demo.sh：

export OPENCV_INCLUDE_DIR="$DEST_ABS"

要把它**打包进交付包**（make_build_dir.sh 会拷到 include/opencv4/）：

export OPENCV_HEADERS_SRC="$DEST_ABS"

===========================================================
EOF
