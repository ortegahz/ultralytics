#!/usr/bin/env bash
# ===========================================================================
# copy_to_nfs.sh -- 把【整棵交付源码包】镜像到 NFS 交付目录（板卡 /mnt/manu）。
#
# 交付包 = manu/pipeline/npu/build/（由 make_build_dir.sh 生成）。脚本所在目录
# 就是包根；本脚本把整棵树（源码 + 第三方头 + kernel 资产 + **third_party/
# 第三方依赖** + 编译产物）原样镜像到 NFS 交付目录，再建 5 个兼容符号链接，
# 旧的平铺命令仍然可用。
#
# 用法:
#   bash copy_to_nfs.sh [--build] [--nfs <目录>] [--verify]
#                      [--board <host[:port]>] [--board-dest <板卡绝对路径>]
#
#   （无参）        整树镜像 + 逐文件 md5 回验 + 打印板卡侧路径与运行命令
#   --build        拷贝前先跑 build_lib.sh && build_demo.sh（失败即退出）
#   --nfs <目录>   覆盖 NFS 根目录（默认 /run/media/manu/SSD960-LOCAL/nfs）
#   --verify       拷贝后 ssh 到板卡跑契约测试 + demo + 最小集成示范（严格串行）
#   --board <目标[:端口]>    覆盖 ssh 目标（默认 root@192.168.0.64，端口 22）
#   --board-dest <绝对路径>  板卡侧交付目录（默认 /mnt/manu/mosaic_sdk）
#
# ⚠️ 安全铁律（板卡 /root 的 ext4 已损坏，>1 MB 的二进制会被写坏）：
#    * 绝不写板卡 /root —— 交付目录不允许落在 /root 下。
#    * 本机 NFS 根若是只读挂载（当前开发机就是），走【板卡 /tmp 中转】：
#      tar -> 板卡 /tmp (tmpfs 3.9G) -> cp -a -> 板卡交付目录。
#      中转只碰 /tmp，绝不往板卡 /root 落大文件。
#
# 🔴 安全铁律（板卡把 NFS【soft 挂载】(soft,proto=tcp)）：
#    * 从 soft 挂载上【读取 >1 MB 的二进制】会污染板卡客户端页缓存
#      （实测：同尺寸但内容被清零 3741 字节，首个差异 offset 5046273），
#      直接执行会 SIGILL(132) —— 板卡 dmesg 出现 "RKNPU job timeout" +
#      "soft reset"，看着像驱动中毒，其实不是。
#    * ⇒ 板卡侧要跑 >1 MB 的二进制，必须先流式落到板卡 /tmp（tmpfs 3.9 G）：
#      cat > /tmp/x && chmod +x，再用 md5 与【本机源文件】对账，一致才执行。
#      落在 /tmp 的临时文件用完即删（带前缀的唯一名，别误删板卡上别人的东西）。
#    * 判别口诀：「同尺寸、不同内容」= 传输/缓存损坏，不是算法 bug。
# ===========================================================================
set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认值（唯一允许写死的地方：默认 NFS 根 + 默认板卡）
DEFAULT_NFS_ROOT="/run/media/manu/SSD960-LOCAL/nfs"
DEFAULT_BOARD_DEST="/mnt/manu/mosaic_sdk"

NFS_ROOT="${NFS_ROOT:-$DEFAULT_NFS_ROOT}"
DEST="$NFS_ROOT/mosaic_sdk"
BOARD="${BOARD:-root@192.168.0.64}"
BOARD_PORT="${BOARD_PORT:-22}"
BOARD_DEST="${BOARD_DEST:-}"

# 板卡侧的中转目录（只在 /tmp 下；/tmp 是 tmpfs，3.9 G）
STAGE_DIR="/tmp/mosaic_sdk_stage"
STAGE_MD5="/tmp/mosaic_sdk_md5s.txt"

DO_BUILD=0
DO_VERIFY=0
LAST_OUT=""

die() { echo "[ERROR] $*" >&2; exit 1; }

# 真的探一次能不能写：`-w` 只看权限位，本机是 ro 挂载、服务端 export 是 ro、
# 或者根可写但目标子目录不可写，这三种情况 `-w` 都会误判。
nfs_root_writable() {
  local probe="$NFS_ROOT/.mosaic_write_probe_$$"
  if : > "$probe" 2>/dev/null; then
    rm -f "$probe"
    return 0
  fi
  return 1
}

usage() {
  cat <<'EOF'
用法: bash copy_to_nfs.sh [--build] [--nfs <目录>] [--verify]
                         [--board <host[:port]>] [--board-dest <板卡绝对路径>]

  --build              拷贝前先编译（build_lib.sh + build_demo.sh）
  --nfs <目录>         覆盖 NFS 根目录（默认 /run/media/manu/SSD960-LOCAL/nfs）
  --verify             拷贝后 ssh 到板卡跑契约测试 + demo + 最小集成示范（串行，非 0 退出即失败）
  --board <目标[:端口]>  覆盖 ssh 目标（默认 root@192.168.0.64，端口 22）
  --board-dest <路径>  板卡侧交付目录（默认 /mnt/manu/mosaic_sdk；
                       NFS 根不是默认根时必须显式给出）

例:
  bash copy_to_nfs.sh                        # 整树镜像 + md5 回验
  bash copy_to_nfs.sh --build --verify       # 编译 + 拷贝 + 上板验证
  bash copy_to_nfs.sh --nfs /tmp/test_nfs    # 换目标目录（不碰真 NFS）
  bash copy_to_nfs.sh --board root@1.2.3.4:2222
  bash copy_to_nfs.sh --board-dest /mnt/manu/mosaic_sdk
EOF
}

# ---------------------------------------------------------------------------
# 1. 解析参数
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --build)  DO_BUILD=1; shift ;;
    --verify) DO_VERIFY=1; shift ;;
    --nfs)
      [ $# -ge 2 ] || { echo "[ERROR] --nfs 缺少参数值" >&2; usage >&2; exit 2; }
      NFS_ROOT="$2"; DEST="$NFS_ROOT/mosaic_sdk"; shift 2 ;;
    --board)
      [ $# -ge 2 ] || { echo "[ERROR] --board 缺少参数值" >&2; usage >&2; exit 2; }
      BOARD="$2"; shift 2 ;;
    --board-dest)
      [ $# -ge 2 ] || { echo "[ERROR] --board-dest 缺少参数值" >&2; usage >&2; exit 2; }
      BOARD_DEST="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] 未知参数：$1" >&2
      usage >&2
      exit 2 ;;
  esac
done

# --board 允许 "host" 或 "host:port" 两种写法
case "$BOARD" in
  *:*)
    BOARD="${BOARD%:*}"
    BOARD_PORT="${BOARD##*:}" ;;
esac
[ -n "$BOARD" ] || die "--board 解析后目标为空"
[ -n "$BOARD_PORT" ] || die "--board 解析后端口为空"

# 板卡侧交付目录：只有默认 NFS 根才认 /mnt/manu/mosaic_sdk；
# 换了 NFS 根就只能让用户显式给（真的要用到板卡时缺了就报错退出）。
if [ -z "$BOARD_DEST" ] && [ "$NFS_ROOT" = "$DEFAULT_NFS_ROOT" ]; then
  BOARD_DEST="$DEFAULT_BOARD_DEST"
fi
if [ -n "$BOARD_DEST" ]; then
  case "$BOARD_DEST" in
    /*) ;;
    *) die "--board-dest 必须是绝对路径：$BOARD_DEST" ;;
  esac
  # 铁律：板卡 /root 的 ext4 已损坏，>1 MB 的二进制会被写坏
  case "$BOARD_DEST" in
    /root|/root/*) die "交付目录不能落在板卡 /root 下（ext4 已损坏）：$BOARD_DEST" ;;
  esac
fi

# 板卡侧路径只有在真要用到板卡（中转 / 上板验证）时才必须有
require_board_dest() {
  [ -n "$BOARD_DEST" ] && return 0
  die "板卡侧交付目录未知：NFS 根不是默认根（$DEFAULT_NFS_ROOT），请显式给出 --board-dest <板卡绝对路径>"
}

# --verify 要在板卡上跑程序，板卡侧路径必须先知道（早失败，别拷完再报错）
if [ "$DO_VERIFY" = "1" ]; then
  require_board_dest
fi

# ---------------------------------------------------------------------------
# 2. 定位包目录（源 = 整棵树）
#    交付态：脚本就躺在包根，脚本所在目录 = 包目录。
#    开发态：脚本原件在 packaging/ 下，包目录是同级的 build/（make_build_dir.sh 生成）。
# ---------------------------------------------------------------------------
PKG_DIR="$HERE"
if [ ! -d "$PKG_DIR/out" ] || [ ! -d "$PKG_DIR/src" ]; then
  _cand="$(dirname "$HERE")/build"
  if [ -d "$_cand/out" ] && [ -d "$_cand/src" ]; then
    PKG_DIR="$_cand"
  fi
fi
[ -d "$PKG_DIR/out" ] && [ -d "$PKG_DIR/src" ] || \
  die "包目录不完整（缺 out/ 或 src/）：$PKG_DIR —— 先跑 make_build_dir.sh"

# 期望集合：包内全部普通文件（相对包根）+ 5 个兼容别名
PKG_FILES=()
declare -A EXPECTED=()
declare -A EXPECTED_DIR=()
while IFS= read -r rel; do
  PKG_FILES+=("$rel")
  EXPECTED["$rel"]=1
  d="$(dirname "$rel")"
  while [ "$d" != "." ]; do
    EXPECTED_DIR["$d"]=1
    d="$(dirname "$d")"
  done
done < <(cd "$PKG_DIR" && find . -type f | sed 's|^\./||' | LC_ALL=C sort)
[ "${#PKG_FILES[@]}" -gt 0 ] || die "包目录里没有普通文件：$PKG_DIR"

# 兼容别名（旧命令仍然可用）：名字 -> 相对链接目标
ALIAS_PAIR=(
  "mosaic_api_test|out/mosaic_api_test"
  "mosaic_contract_test|out/mosaic_contract_test"
  "libmosaic_pipeline.a|out/libmosaic_pipeline.a"
  "mosaic_pipeline.h|src/npu/mosaic_pipeline.h"
  "kernels|runtime/kernels"
)
ALIAS_SPEC=""
for pair in "${ALIAS_PAIR[@]}"; do
  ALIAS_SPEC="$ALIAS_SPEC ${pair%%|*}=${pair##*|}"
done
declare -A ALIAS_NAME=()
for pair in "${ALIAS_PAIR[@]}"; do
  ALIAS_NAME["${pair%%|*}"]=1
done

# 交付包里必须可执行的脚本 / 可执行文件
EXEC_FILES=(
  "build_lib.sh"
  "build_demo.sh"
  "copy_to_nfs.sh"
  "out/mosaic_api_test"
  "out/mosaic_client_example"
  "out/mosaic_contract_test"
  "out/rknn_mosaic_e2e"
)

# 交付包必须齐全的关键文件
REQUIRED_FILES=(
  "README.md" "MANIFEST.md" "build_lib.sh" "build_demo.sh" "copy_to_nfs.sh"
  "src/npu/mosaic_pipeline.cpp" "src/npu/mosaic_pipeline.h"
  "src/npu/mosaic_core.h" "src/npu/mosaic_post_gpu.h"
  "src/npu/mosaic_api_test.cpp" "src/npu/mosaic_contract_test.cpp"
  "src/npu/rknn_mosaic_e2e.cpp" "src/npu/mosaic_client_example.cpp"
  "src/cpp/gmc_stream_ocl.cpp" "src/cpp/md5.h"
  "src/opencl/host/ocl_host.h" "src/opencl/host/post_local_max.cl"
  "include/rknn_api.h" "runtime/kernels/warp_median_fused.cl"
  "runtime/kernels/sortnet_generated.inc" "runtime/kernels/gmc_feature_gpu.cl"
  "out/mosaic_pipeline.o" "out/libmosaic_pipeline.a"
  "out/mosaic_api_test" "out/mosaic_contract_test"
  "out/rknn_mosaic_e2e" "out/mosaic_client_example"
  # 第三方依赖（**编译期硬依赖**，以前只存在于开发机的绝对路径里 ⇒ 包编不出来）
  "third_party/MANIFEST.md" "third_party/README.md"
  "third_party/opencv/include/opencv4/opencv2/core.hpp"
  "third_party/opencv/include/opencv4/opencv2/imgcodecs.hpp"
  "third_party/opencv/lib/libopencv_core.a"
  "third_party/opencv/lib/libopencv_imgproc.a"
  "third_party/opencv/lib/libopencv_imgcodecs.a"
  "third_party/opencv/lib/libopencv_calib3d.a"
  "third_party/opencv/lib/libopencv_features2d.a"
  "third_party/opencv/lib/libopencv_flann.a"
  "third_party/opencv/lib/libopencv_video.a"
  "third_party/opencv/lib/opencv4/3rdparty/libtegra_hal.a"
  "third_party/opencv/lib/opencv4/3rdparty/libzlib.a"
  "third_party/boardlibs/libOpenCL.so" "third_party/boardlibs/libOpenCL.so.1"
  "third_party/rknnrt/librknnrt.so"
  "third_party/rknnrt/librga.so" "third_party/rknnrt/librga.so.2"
)

# ssh 通用参数：本机 ~/.ssh 只读（known_hosts 写不了），但**主机密钥必须钉死**
# —— 交付脚本是以 root 身份去板卡上执行二进制的，不能接受
# "StrictHostKeyChecking=no" 那种「谁应答谁赢」。
#   * 交付包里带一份 known_hosts（make_build_dir.sh 一起拷进来）
#   * 开发态（脚本在 packaging/ 下）用同目录的那份
KNOWN_HOSTS="$HERE/known_hosts"
[ -f "$KNOWN_HOSTS" ] || KNOWN_HOSTS="$(dirname "$HERE")/packaging/known_hosts"
[ -f "$KNOWN_HOSTS" ] || die "找不到 known_hosts（板卡主机密钥基线）：$HERE/known_hosts"
SSH_OPTS=(-F /dev/null -o UserKnownHostsFile="$KNOWN_HOSTS"
          -o StrictHostKeyChecking=yes -o LogLevel=ERROR -o ConnectTimeout=15)
ssh_board() { ssh "${SSH_OPTS[@]}" -p "$BOARD_PORT" "$BOARD" "$@"; }

echo "== 一键拷贝（整树源码包）=="
echo "  包目录   : $PKG_DIR"
echo "  文件数   : ${#PKG_FILES[@]}"
echo "  NFS 根   : $NFS_ROOT"
echo "  本机目标 : $DEST"
echo "  板卡     : $BOARD:$BOARD_PORT"
echo "  板卡目标 : $BOARD_DEST"

# ---------------------------------------------------------------------------
# 3. 可选：先编译
# ---------------------------------------------------------------------------
if [ "$DO_BUILD" = "1" ]; then
  echo
  echo "== [1/5] 编译 =="
  bash "$PKG_DIR/build_lib.sh"  || die "build_lib.sh 失败（编译错误见上）"
  bash "$PKG_DIR/build_demo.sh" || die "build_demo.sh 失败（编译错误见上）"
  # ⚠️ 编译完之后**必须再刷一次交付文档**。make_build_dir.sh 生成的 README/MANIFEST
  #    里记着「out/ 产物的 md5」和「包内文件总数」：如果在编译**之前**打包，
  #    README 会写成「（尚未编译）」+ 文件数少 6 ⇒ **交付出去的 README 是错的**。
  #    （仓库里 ../packaging/make_build_dir.sh 存在时自动补这一步；交付态的包
  #      没有这个脚本，就只能由构建者自己再跑一次。）
  if [ -f "$PKG_DIR/../packaging/make_build_dir.sh" ]; then
    echo
    echo "  [刷新交付文档] 再跑一次 make_build_dir.sh（把新产物 md5 与文件总数写回 README/MANIFEST）"
    bash "$PKG_DIR/../packaging/make_build_dir.sh" || die "make_build_dir.sh 失败"
  else
    echo
    echo "  [提示] 本包是**交付态**（没有 ../packaging/make_build_dir.sh）"
    echo "         ⇒ 请自行再跑一次打包脚本，否则 README 里的产物 md5 会停留在「（尚未编译）」"
  fi
else
  echo
  echo "== [1/5] 跳过编译（用已有的 out/ 产物）=="
fi

# ---------------------------------------------------------------------------
# 4. 检查包内容是否齐全
# ---------------------------------------------------------------------------
echo
echo "== [2/5] 检查包内容 =="
for f in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$PKG_DIR/$f" ]; then
    case "$f" in
      third_party/*)
        die "缺少 $f —— 这个包缺【第三方依赖】（third_party/）。
       ⇒ 上一版交付包（69 个文件）没有 third_party/，在集成方机器上**编不出来**
         （构建脚本指向开发机的绝对路径）。
       ⇒ 处置：① 重新打包：bash packaging/make_build_dir.sh（会自动带上 third_party/）
                ② 或用 OPENCV_INSTALL / BOARD_LIBS / RT_LIB_DIR 三个环境变量
                   指向你机器上的交叉编译版 OpenCV / 板卡 libOpenCL / 板卡 librknnrt+librga。" ;;
      *)
        die "缺少 $f —— 先跑 build_lib.sh / build_demo.sh（--build 可自动做）" ;;
    esac
  fi
  echo "  [ok] $f"
done

# ---------------------------------------------------------------------------
# 5. 拷贝（整树镜像）
#    直写模式：NFS 根可写 -> 本机直接写 $DEST。
#    中转模式：NFS 根只读（当前开发机）-> tar 到板卡 /tmp，再 cp -a 到板卡交付目录。
# ---------------------------------------------------------------------------
echo
echo "== [3/5] 拷贝 =="

# 5a. 清理目标里的陈旧文件（不属于期望集合的旧文件，例如上一版遗留的平铺实文件）
sweep_stale() {
  local dest="$1" f rel
  while IFS= read -r f; do
    rel="${f#"$dest"/}"
    if [ -z "${EXPECTED[$rel]:-}" ]; then
      rm -f "$f"
      echo "  [stale] 已删除 $rel"
    fi
  done < <(find "$dest" -type f | LC_ALL=C sort)
  while IFS= read -r f; do
    rel="${f#"$dest"/}"
    if [ -z "${EXPECTED[$rel]:-}" ] && [ -z "${ALIAS_NAME[$rel]:-}" ]; then
      rm -f "$f"
      echo "  [stale] 已删除符号链接 $rel"
    fi
  done < <(find "$dest" -type l | LC_ALL=C sort)
}

# 5b. 清掉空目录（期望集合里没有的）：上一版遗留的 kernels/ 实目录会在这里消失
sweep_empty_dirs() {
  local dest="$1" f rel
  while IFS= read -r f; do
    [ "$f" = "$dest" ] && continue
    rel="${f#"$dest"/}"
    if [ -z "${EXPECTED_DIR[$rel]:-}" ]; then
      if rmdir "$f" 2>/dev/null; then
        echo "  [stale] 已删除空目录 $rel"
      fi
    fi
  done < <(find "$dest" -depth -type d | LC_ALL=C sort -r)
}

# 5c. 建 5 个兼容符号链接（相对路径）
#     建之前必须先删掉同名旧实文件/实目录：
#       - ln -sfn 覆盖不了实目录；
#       - cp 遇到同名符号链接会穿透进去，把 out/ 里的真文件写坏。
make_aliases() {
  local dest="$1" pair name target
  for pair in "${ALIAS_PAIR[@]}"; do
    name="${pair%%|*}"
    target="${pair##*|}"
    if [ -e "$dest/$name" ] && [ ! -L "$dest/$name" ]; then
      rm -rf "$dest/$name"
      echo "  [stale] 已删除 $name（旧实路径，改成兼容符号链接）"
    fi
    ln -sfn "$target" "$dest/$name"
  done
}

# 5d. 直写模式：整树拷贝到本机 $DEST
copy_direct() {
  local rel dst
  [ -e "$DEST" ] && [ ! -d "$DEST" ] && die "目标路径存在但不是目录：$DEST"
  mkdir -p "$DEST"
  sweep_stale "$DEST"
  for rel in "${PKG_FILES[@]}"; do
    dst="$DEST/$rel"
    mkdir -p "$(dirname "$dst")"
    if [ -L "$dst" ]; then
      rm -f "$dst"
      echo "  [stale] 已删除 $rel（旧符号链接）"
    elif [ -d "$dst" ]; then
      rm -rf "$dst"
      echo "  [stale] 已删除 $rel（旧目录）"
    fi
    cp -f "$PKG_DIR/$rel" "$dst"
  done
  for rel in "${EXEC_FILES[@]}"; do
    chmod +x "$DEST/$rel"
  done
  sweep_empty_dirs "$DEST"
  make_aliases "$DEST"
}

# 5e. 中转模式：本机 NFS 根只读 -> 经板卡 /tmp 中转
copy_relay() {
  local md5_list
  require_board_dest
  md5_list="$(mktemp)"
  # 本机清单用完即删（别在 /tmp 里漏一堆 tmp.XXXX）
  trap 'rm -f "$md5_list"' RETURN
  # 本机生成 md5 清单（路径相对包根，板卡侧按 board_dest 为基准校验）
  ( cd "$PKG_DIR" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 md5sum ) > "$md5_list"

  echo "  [1/3] tar 传整树 -> $BOARD:$BOARD_PORT:$STAGE_DIR（板卡 /tmp 中转）"
  tar -C "$PKG_DIR" -cf - . | ssh_board "rm -rf '$STAGE_DIR' && mkdir -p '$STAGE_DIR' && tar -C '$STAGE_DIR' -xf -"

  echo "  [2/3] 传 md5 清单 -> $STAGE_MD5"
  ssh_board "cat > '$STAGE_MD5'" < "$md5_list"

  echo "  [3/3] 板卡侧清理 + 拷贝 + 建别名 + md5 校验"
  # 整条命令作为一个字符串传给远端 shell：别名清单里有空格，
  # 分成多个参数会被远端 shell 重新拆开，只剩第一个别名。
  ssh_board "bash -s -- '$BOARD_DEST' '$STAGE_DIR' '$STAGE_MD5' '$ALIAS_SPEC'" <<'REMOTE'
set -euo pipefail
export LC_ALL=C
dest="$1"; stage="$2"; md5f="$3"; alias_spec="$4"

# 失败也要清理：/tmp 是板卡 tmpfs（3.9 G），失败一次就留 25 MB 垃圾
cleanup() { rm -rf "$stage" "$md5f" /tmp/mosaic_sdk_expected.txt /tmp/mosaic_sdk_expected_dirs.txt; }
trap cleanup EXIT

case "$stage" in /tmp/*) ;; *) echo "[ERROR] 中转目录必须落在 /tmp 下：$stage" >&2; exit 1 ;; esac
case "$dest"  in /root|/root/*) echo "[ERROR] 交付目录不能落在 /root 下：$dest" >&2; exit 1 ;; esac

exp=/tmp/mosaic_sdk_expected.txt
dirs=/tmp/mosaic_sdk_expected_dirs.txt
sed -E 's/^[0-9a-f]{32}  //' "$md5f" | sed 's|^\./||' | LC_ALL=C sort > "$exp"
: > "$dirs"
while IFS= read -r rel; do
  d="$(dirname "$rel")"
  while [ "$d" != "." ]; do
    echo "$d" >> "$dirs"
    d="$(dirname "$d")"
  done
done < "$exp"
LC_ALL=C sort -u "$dirs" -o "$dirs"

mkdir -p "$dest"

# 陈旧文件（不在期望集合里的旧文件，例如上一版遗留的平铺实文件）
while IFS= read -r f; do
  rel="${f#"$dest"/}"
  if ! grep -Fxq "$rel" "$exp"; then
    rm -f "$f"
    echo "  [stale] 已删除 $rel"
  fi
done < <(find "$dest" -type f | LC_ALL=C sort)

# 陈旧符号链接（5 个兼容别名要保留）
while IFS= read -r f; do
  rel="${f#"$dest"/}"
  keep=0
  for spec in $alias_spec; do
    if [ "${spec%%=*}" = "$rel" ]; then keep=1; fi
  done
  if [ "$keep" = "0" ]; then
    rm -f "$f"
    echo "  [stale] 已删除符号链接 $rel"
  fi
done < <(find "$dest" -type l | LC_ALL=C sort)

# 整树拷贝
# ⚠️ 必须先清掉【与别名同名的旧实路径】：cp -a 不能用符号链接覆盖实目录
#    （旧交付版在根目录留的是 kernels/ 实目录，升级路径必踩）
for spec in $alias_spec; do
  name="${spec%%=*}"
  if [ -e "$dest/$name" ] && [ ! -L "$dest/$name" ]; then
    rm -rf "$dest/$name"
    echo "  [stale] 已删除 $name（旧实路径，改成兼容符号链接）"
  fi
done
cp -a "$stage/." "$dest/"

# 空目录（期望集合里没有的）
while IFS= read -r f; do
  if [ "$f" = "$dest" ]; then continue; fi
  rel="${f#"$dest"/}"
  if ! grep -Fxq "$rel" "$dirs"; then
    if rmdir "$f" 2>/dev/null; then
      echo "  [stale] 已删除空目录 $rel"
    fi
  fi
done < <(find "$dest" -depth -type d | LC_ALL=C sort -r)

# 5 个兼容符号链接（相对路径）
for spec in $alias_spec; do
  name="${spec%%=*}"; target="${spec#*=}"
  if [ -e "$dest/$name" ] && [ ! -L "$dest/$name" ]; then
    rm -rf "$dest/$name"
    echo "  [stale] 已删除 $name（旧实路径，改成兼容符号链接）"
  fi
  ln -sfn "$target" "$dest/$name"
done

# md5 回验（清单路径相对 dest）
if ! out="$(cd "$dest" && md5sum -c "$md5f" 2>&1)"; then
  printf '%s\n' "$out" | grep -v ': OK$' >&2 || true
  echo "[ERROR] 板卡侧 md5 校验失败" >&2
  exit 1
fi
echo "  [ok] 板卡侧 md5 校验通过（$(printf '%s\n' "$out" | grep -c ': OK$') 个文件）"

# 清掉中转目录（/tmp 是 tmpfs，别留垃圾）
rm -rf "$stage" "$md5f" "$exp" "$dirs"
echo "  [ok] 板卡 /tmp 中转目录已清理"
REMOTE
}

if [ ! -d "$NFS_ROOT" ]; then
  # 根目录还没建：父目录可写就直写（脚本自己 mkdir），否则无从下手
  _parent="$(dirname "$NFS_ROOT")"
  [ -d "$_parent" ] && [ -w "$_parent" ] || \
    die "NFS 根不存在且父目录不可写：$NFS_ROOT（父目录 $_parent）"
  MODE="直写"
  echo "  模式：直写（NFS 根不存在，将新建：$NFS_ROOT）"
  copy_direct
elif nfs_root_writable; then
  MODE="直写"
  echo "  模式：直写（NFS 根可写：$NFS_ROOT）"
  copy_direct
else
  MODE="中转"
  echo "  模式：中转（NFS 根只读：$NFS_ROOT）-> 经板卡 /tmp 中转"
  copy_relay
fi

# ---------------------------------------------------------------------------
# 6. md5 回验（直写模式在本机做；中转模式已在板卡侧做完）
# ---------------------------------------------------------------------------
echo
echo "== [4/5] md5 回验 =="
if [ "$MODE" = "直写" ]; then
  bad=0
  for rel in "${PKG_FILES[@]}"; do
    sm="$(md5sum "$PKG_DIR/$rel" | cut -d' ' -f1)"
    dm="$(md5sum "$DEST/$rel" | cut -d' ' -f1)"
    if [ "$sm" != "$dm" ]; then
      echo "  [ERROR] md5 不一致：$rel（源=$sm 目标=$dm）" >&2
      bad=$((bad + 1))
    fi
  done
  [ "$bad" = "0" ] || die "md5 回验失败：$bad 个文件与源不一致 —— 写入可能损坏"
  echo "  [ok] ${#PKG_FILES[@]} 个普通文件 md5 全部一致"

  # 别名只校验链接目标字符串
  for pair in "${ALIAS_PAIR[@]}"; do
    name="${pair%%|*}"
    target="${pair##*|}"
    got="$(readlink "$DEST/$name")"
    [ "$got" = "$target" ] || die "兼容符号链接不对：$name -> $got（应为 $target）"
    echo "  [ok] $name -> $got"
  done

  # 镜像检查：目标里不该有期望集合之外的文件
  n_dst="$(find "$DEST" -type f | wc -l)"
  [ "$n_dst" = "${#PKG_FILES[@]}" ] || \
    die "目标里普通文件数 $n_dst != 源 ${#PKG_FILES[@]}（镜像不干净）"
  echo "  [ok] 目标是干净镜像（$n_dst 个普通文件 + 5 个兼容别名）"
else
  echo "  [ok] 板卡侧 md5 校验已在拷贝时完成"
fi

# ---------------------------------------------------------------------------
# 7. 打印目标清单 + 板卡侧路径 + 运行命令
# ---------------------------------------------------------------------------
echo
echo "== [5/5] 目标目录清单 =="
if [ "$MODE" = "直写" ]; then
  find "$DEST" -type f | LC_ALL=C sort | while IFS= read -r p; do
    printf "  %-52s %10s B  %s\n" "${p#"$DEST"/}" "$(stat -c%s "$p")" "$(md5sum "$p" | cut -d' ' -f1)"
  done
  find "$DEST" -type l | LC_ALL=C sort | while IFS= read -r p; do
    printf "  %-52s %10s    -> %s\n" "${p#"$DEST"/}" "symlink" "$(readlink "$p")"
  done
else
  ssh_board "bash -s -- '$BOARD_DEST'" <<'REMOTE'
set -euo pipefail
export LC_ALL=C
cd "$1"
find . -type f | LC_ALL=C sort | while IFS= read -r p; do
  printf "  %-52s %10s B  %s\n" "${p#./}" "$(stat -c%s "$p")" "$(md5sum "$p" | cut -d' ' -f1)"
done
find . -type l | LC_ALL=C sort | while IFS= read -r p; do
  printf "  %-52s %10s    -> %s\n" "${p#./}" "symlink" "$(readlink "$p")"
done
REMOTE
fi

echo
echo "== 板卡侧路径 =="
echo "  本机 NFS 根 : $NFS_ROOT"
if [ -z "$BOARD_DEST" ]; then
  echo "  ⚠️  板卡侧路径未知（NFS 根非默认根）：只有默认根才对应 /mnt/manu/mosaic_sdk"
  echo "     本机目标：$DEST"
  echo "     要打板卡请显式给 --board-dest <板卡绝对路径>"
elif [ "$NFS_ROOT" = "$DEFAULT_NFS_ROOT" ]; then
  echo "  板卡挂载点  : /mnt/manu   =>  $DEST  ==>  $BOARD_DEST/"
else
  echo "  ⚠️  非默认 NFS 根：板卡侧挂载点未知"
  echo "     本机目标：$DEST"
  echo "     板卡目标：$BOARD_DEST"
fi

# 板卡命令里的目标路径：已知就用真值，未知就留占位符
BD="${BOARD_DEST:-<board_dest>}"
echo
echo "== 板卡运行命令（权威写法：新目录布局）=="
cat <<EOF
ssh -p $BOARD_PORT $BOARD '$BD/out/mosaic_api_test \\
  --raw-root   /mnt/manu/lq_ir_bmp \\
  --sequence   VIDEO00007_19700101_001324 \\
  --model      /mnt/manu/ptq/models/p0_off_A_none.rknn \\
  --kernel-dir $BD/runtime/kernels \\
  --limit 12'
EOF
echo
echo "== 兼容别名（旧命令仍可用，等价写法）=="
cat <<EOF
ssh -p $BOARD_PORT $BOARD '$BD/mosaic_api_test \\
  --raw-root   /mnt/manu/lq_ir_bmp \\
  --sequence   VIDEO00007_19700101_001324 \\
  --model      /mnt/manu/ptq/models/p0_off_A_none.rknn \\
  --kernel-dir $BD/kernels \\
  --limit 12'
EOF

# ---------------------------------------------------------------------------
# 8. 可选：ssh 到板卡验证（严格串行：板卡 RKNN context 上限 2）
# ---------------------------------------------------------------------------
if [ "$DO_VERIFY" = "1" ]; then
  echo
  echo "== 板卡验证（严格串行：板卡 RKNN context 上限 2）=="
  echo "  目标: $BOARD:$BOARD_PORT"

  # 板卡上的 NFS 是【soft 挂载】（soft,proto=tcp,timeo=5,retrans=2）：
  # 刚往同一个 export 写了 25 MB 产物就立刻让板卡去读帧，可能撞上瞬时 I/O 错误
  # （实测：帧文件明明在，OpenCV 却报 "can't open/read file"）。
  # 🔴 更狠的一条：从 soft 挂载上【读 >1 MB 二进制】会污染板卡客户端页缓存，
  #    直接执行 SIGILL(132)（详见文件头铁律）。所以板卡侧要跑的每个 >1 MB
  #    二进制都必须先流式落到板卡 /tmp(tmpfs) + md5 对账，再执行。
  # ⇒ 先让写回落定，再跑；失败再自动重试一次（重试前先清板卡页缓存）。
  echo "  [等待 3 s 让 NFS 写回落定]"
  sleep 3

  # 板卡 /tmp 里的暂存前缀：带本机 PID，绝不撞板卡上别人的文件
  VERIFY_TMP_PREFIX="/tmp/mosaic_verify_$$"

  # 板卡侧跑一个二进制：NFS(soft) -> cat 到板卡 /tmp(tmpfs) -> chmod +x ->
  # md5 与【本机源文件】对账（不一致 = NFS 传输损坏，退出码 3）-> 执行 -> 删 /tmp。
  # 用法：run_board_test <标签> <板卡 out/ 里的文件名> [传给程序的参数...]
  run_board_test() {
    local label="$1" tag="$2" attempt rc out args want_md5
    shift 2
    want_md5="$(md5sum "$PKG_DIR/out/$tag" 2>/dev/null | cut -d' ' -f1)" || want_md5=""
    [ -n "$want_md5" ] || die "算不出本机源文件 md5：$PKG_DIR/out/$tag"
    args=""
    if [ $# -gt 0 ]; then args="$(printf '%q ' "$@")"; fi
    for attempt in 1 2; do
      if [ "$attempt" = "2" ]; then
        echo "  [retry] $label 第 1 次失败 —— 重试前先 sync + drop_caches 清掉板卡页缓存"
        echo "          （页缓存污染是持久的，不清缓存重试必然复现同一个 SIGILL）"
        if ! ssh_board "sync; echo 3 > /proc/sys/vm/drop_caches"; then
          echo "  [warn] 板卡 drop_caches 失败（/proc/sys 只读？），继续重试"
        fi
        sleep 5
      fi
      rc=0
      # ⚠️ `2>&1` 必须有：mosaic_contract_test 把 `[RESULT] N passed, 0 failed`
      #    打到 **stderr**（mosaic_contract_test.cpp:1098），只捕获 stdout 会让
      #    下面的 `grep '0 failed'` 必然落空 ⇒ 误报「没找到 0 failed」。
      out="$(ssh_board "bash -s -- '$VERIFY_TMP_PREFIX' '$tag' '$want_md5' '$BOARD_DEST/out/$tag' $args" 2>&1 <<'REMOTE'
set -uo pipefail
export LC_ALL=C
tmp_prefix="$1"; tag="$2"; want_md5="$3"; src="$4"; shift 4
tmp="${tmp_prefix}_${tag}"

# 用完就删（只删自己那一个带前缀的文件，不碰板卡上别人的东西）
cleanup() { rm -f "$tmp"; }
trap cleanup EXIT

[ -f "$src" ] || { echo "[ERROR] 板卡交付目录里没有可执行文件：$src" >&2; exit 1; }

# 铁律：板卡 NFS 是 soft 挂载，直接从 NFS 执行 >1 MB 二进制会污染页缓存 ->
# SIGILL(132)。必须先流式落到板卡 /tmp（tmpfs）再执行。
cat "$src" > "$tmp" || { echo "[ERROR] 从 NFS 读取 $src 失败" >&2; exit 1; }
chmod +x "$tmp" || { echo "[ERROR] chmod +x $tmp 失败" >&2; exit 1; }
[ -x "$tmp" ] || { echo "[ERROR] $tmp 没落到板卡 /tmp（或不可执行）" >&2; exit 1; }

got="$(md5sum "$tmp" | cut -d' ' -f1)"
if [ "$got" != "$want_md5" ]; then
  echo "[ERROR] NFS 传输损坏：$tag 落到板卡 /tmp 后 md5=$got，与本机源文件 md5=$want_md5 不一致" >&2
  echo "        （同尺寸、不同内容 = soft 挂载的传输/页缓存损坏，不是算法 bug）" >&2
  exit 3
fi
echo "  [stage] $tag -> $tmp（已落板卡 /tmp，md5 与本机源文件一致：$got）"

rc=0
"$tmp" "$@" || rc=$?
exit $rc
REMOTE
)" || rc=$?
      printf '%s\n' "$out" | grep -E '\[stage\]' || true
      printf '%s\n' "$out" | grep -vE '\[stage\]' | tail -n 12 || true
      echo "  退出码: $rc"
      if [ "$rc" = "0" ]; then
        LAST_OUT="$out"
        return 0
      fi
      if [ "$rc" = "3" ]; then
        die "NFS 传输损坏：$label 的二进制落到板卡 /tmp 后 md5 与本机源文件不一致 —— 同尺寸、不同内容 = 传输/页缓存损坏，不是算法 bug"
      fi
    done
    LAST_OUT="$out"
    echo "  [fail] $label 重试后仍失败 —— 两种可能："
    echo "         ① 疑似 NFS 页缓存损坏（已先落 /tmp + 已 drop_caches 仍失败）"
    echo "         ② 真回归（板卡侧逻辑/数据真的变了）"
    return 1
  }

  echo
  echo "  [1/4] 契约测试：mosaic_contract_test --limit 60（先落板卡 /tmp 再执行）"
  rc=0
  run_board_test "契约测试" "mosaic_contract_test" --limit 60 --kernel-dir "$BOARD_DEST/runtime/kernels" || rc=$?
  [ "$rc" = "0" ] || die "板卡契约测试失败（退出码 $rc）—— 疑似 NFS 页缓存损坏（已先落 /tmp + 已 drop_caches 仍失败）或真回归"
  if printf '%s\n' "$LAST_OUT" | grep -q '\[RESULT\] .* 0 failed'; then
    echo "  [ok] 契约测试 0 failed"
  else
    echo "  ⚠️  契约测试输出里没找到 '0 failed'，请人工核对上面的 [RESULT] 行"
  fi

  echo
  echo "  [2/4] demo：mosaic_api_test --limit 6（先落板卡 /tmp 再执行）"
  rc=0
  run_board_test "demo" "mosaic_api_test" --limit 6 --kernel-dir "$BOARD_DEST/runtime/kernels" || rc=$?
  [ "$rc" = "0" ] || die "板卡 demo 失败（退出码 $rc）—— 疑似 NFS 页缓存损坏（已先落 /tmp + 已 drop_caches 仍失败）或真回归"
  echo "  [ok] 板卡从交付目录跑通"

  # 第三步：最小集成示范（嵌入式组照着抄的那份）。⚠️ 它的接口是**四个位置
  # 参数**（帧目录 / 模型 / kernel 目录 / 帧数），**不是**命名开关 —— 与上面
  # 两步的 mosaic_contract_test / mosaic_api_test 不同，别照抄它们的写法。
  # **必须串行**：板卡 RKNN context 上限 2，并发跑会互相抢 context。跟前两步
  # 一样，二进制先落板卡 /tmp(tmpfs) + md5 对账再执行。
  echo
  echo "  [3/4] 最小集成示范：mosaic_client_example <帧目录> <模型> <kernel> 6（先落板卡 /tmp 再执行）"
  rc=0
  run_board_test "最小集成示范" "mosaic_client_example" \
    "/mnt/manu/lq_ir_bmp/VIDEO00007_19700101_001324" \
    "/mnt/manu/ptq/models/p0_off_A_none.rknn" \
    "$BOARD_DEST/runtime/kernels" \
    6 || rc=$?
  [ "$rc" = "0" ] || die "板卡最小集成示范失败（退出码 $rc）—— 疑似 NFS 页缓存损坏（已先落 /tmp + 已 drop_caches 仍失败）或真回归"
  echo "  [ok] 板卡从交付目录跑通（demo + 最小集成示范）"

  # -------------------------------------------------------------------------
  # [4/4] 新增门禁：**包内自带的第三方运行时库**在板卡上真的能用吗？
  #
  #   librknnrt.so / librga.so.2 / libOpenCL.so.1 这三个以前**不在包里**
  #   （只存在于开发机的绝对路径），现在打进 third_party/ ⇒ 必须验证
  #   「板卡能从包里加载它们」，否则这个依赖就白打包了。
  #
  #   ⚠️ 三个文件里 librknnrt.so 有 7.3 MB ⇒ 从 soft 挂载的 NFS 读会被
  #      页缓存污染 ⇒ 同样先落板卡 /tmp（tmpfs）+ md5 对账再加载。
  #   ⚠️ 板卡 BSP 里本来就有这三个库（ldconfig 能查到），但 LD_LIBRARY_PATH
  #      的搜索顺序**在系统路径之前** ⇒ 加载的一定是**包里的**那一份，
  #      这道门禁才有意义。
  # -------------------------------------------------------------------------
  echo
  echo "  [4/4] 包内第三方运行时库：落板卡 /tmp → md5 对账 → 用【包里的】库跑一遍"
  tp_w_rknn="$(md5sum "$PKG_DIR/third_party/rknnrt/librknnrt.so"       | cut -d' ' -f1)"
  tp_w_rga="$(md5sum "$PKG_DIR/third_party/rknnrt/librga.so.2"           | cut -d' ' -f1)"
  tp_w_cl="$(md5sum "$PKG_DIR/third_party/boardlibs/libOpenCL.so.1"     | cut -d' ' -f1)"
  tp_w_mce="$(md5sum "$PKG_DIR/out/mosaic_client_example"                | cut -d' ' -f1)"
  tp_tmp="/tmp/mosaic_tp_$$"
  tp_out="$(ssh_board "bash -s -- '$tp_tmp' '$BOARD_DEST' '$tp_w_rknn' '$tp_w_rga' '$tp_w_cl' '$tp_w_mce'" 2>&1 <<'REMOTE'
set -uo pipefail
export LC_ALL=C
tmp="$1"; dest="$2"; w_rknn="$3"; w_rga="$4"; w_cl="$5"; w_mce="$6"
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT

rm -rf "$tmp"; mkdir -p "$tmp"
cat "$dest/third_party/rknnrt/librknnrt.so"        > "$tmp/librknnrt.so"        || { echo "[ERROR] 从 NFS 读 librknnrt.so 失败" >&2; exit 1; }
cat "$dest/third_party/rknnrt/librga.so.2"          > "$tmp/librga.so.2"          || { echo "[ERROR] 从 NFS 读 librga.so.2 失败" >&2; exit 1; }
cat "$dest/third_party/boardlibs/libOpenCL.so.1"   > "$tmp/libOpenCL.so.1"      || { echo "[ERROR] 从 NFS 读 libOpenCL.so.1 失败" >&2; exit 1; }
cat "$dest/out/mosaic_client_example"              > "$tmp/mosaic_client_example" || { echo "[ERROR] 从 NFS 读 mosaic_client_example 失败" >&2; exit 1; }
chmod +x "$tmp/mosaic_client_example"

g_rknn="$(md5sum "$tmp/librknnrt.so"     | cut -d' ' -f1)"
g_rga="$(md5sum "$tmp/librga.so.2"       | cut -d' ' -f1)"
g_cl="$(md5sum "$tmp/libOpenCL.so.1"     | cut -d' ' -f1)"
g_mce="$(md5sum "$tmp/mosaic_client_example" | cut -d' ' -f1)"
[ "$g_rknn" = "$w_rknn" ] || { echo "[ERROR] NFS 传输损坏：librknnrt.so md5=$g_rknn ≠ $w_rknn" >&2; exit 3; }
[ "$g_rga"   = "$w_rga"   ] || { echo "[ERROR] NFS 传输损坏：librga.so.2 md5=$g_rga ≠ $w_rga" >&2; exit 3; }
[ "$g_cl"    = "$w_cl"    ] || { echo "[ERROR] NFS 传输损坏：libOpenCL.so.1 md5=$g_cl ≠ $w_cl" >&2; exit 3; }
[ "$g_mce"   = "$w_mce"   ] || { echo "[ERROR] NFS 传输损坏：mosaic_client_example md5=$g_mce ≠ $w_mce" >&2; exit 3; }
echo "  [stage] 包内 3 个运行时库 + 示范程序已落板卡 /tmp，md5 与本机源文件逐位一致"

# LD_LIBRARY_PATH 优先于系统路径 ⇒ 加载的一定是【包里的】那一份
export LD_LIBRARY_PATH="$tmp"
"$tmp/mosaic_client_example" \
  "/mnt/manu/lq_ir_bmp/VIDEO00007_19700101_001324" \
  "/mnt/manu/ptq/models/p0_off_A_none.rknn" \
  "$dest/runtime/kernels" \
  6
rc=$?
[ "$rc" = "0" ] && echo "  [ok] 已用【包内 third_party 的库】跑通最小集成示范（LD_LIBRARY_PATH=$tmp）"
exit $rc
REMOTE
)" || tp_rc=$?
  tp_rc="${tp_rc:-0}"
  printf '%s\n' "$tp_out" | grep -E '\[stage\]|\[ok\]' || true
  printf '%s\n' "$tp_out" | grep -vE '\[stage\]|\[ok\]' | tail -n 8 || true
  [ "$tp_rc" = "0" ] || die "包内第三方运行时库验证失败（退出码 $tp_rc）—— ① NFS 传输损坏（md5 对不上）② 包里的库在板卡上装不上"
  echo "  [ok] 包内 third_party 运行时库在板卡上可用"
fi

echo
echo "[ok] 全部完成（模式：$MODE）。"
