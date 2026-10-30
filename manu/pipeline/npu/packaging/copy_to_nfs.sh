#!/usr/bin/env bash
# ===========================================================================
# copy_to_nfs.sh -- 一键把编译产物拷贝到 NFS 挂载点（板卡 /mnt/manu）。
#
# 用法:
#   bash copy_to_nfs.sh [--build] [--nfs <目录>] [--verify] [--board <ssh目标>[:端口]]
#
#   （无参）    只拷贝 + 逐文件 md5 回验 + 打印板卡侧路径与运行命令
#   --build    拷贝前先跑 build_lib.sh && build_demo.sh（失败即退出）
#   --nfs <目录>  覆盖 NFS 根目录（默认 /run/media/manu/SSD960-LOCAL/nfs）
#   --verify   拷贝后 ssh 到板卡跑一次 demo（零参数，--limit 6）
#   --board <目标[:端口]>  覆盖 ssh 目标（默认 root@192.168.0.64，端口 22）
#
# ⚠️ 本脚本【绝不】scp / rsync 到板卡：
#    板卡根分区 /root 的 ext4 已损坏，>1 MB 的二进制会被写坏。
#    --verify 只在 NFS 上【执行】已有的程序，不往板卡写任何东西。
# ===========================================================================
set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NFS_ROOT="${NFS_ROOT:-/run/media/manu/SSD960-LOCAL/nfs}"
DEST="$NFS_ROOT/mosaic_sdk"
BOARD="${BOARD:-root@192.168.0.64}"
BOARD_PORT="${BOARD_PORT:-22}"

DO_BUILD=0
DO_VERIFY=0

die() { echo "[ERROR] $*" >&2; exit 1; }

usage() {
  cat <<'EOF'
用法: bash copy_to_nfs.sh [--build] [--nfs <目录>] [--verify] [--board <ssh目标>[:端口]]

  --build              拷贝前先编译（build_lib.sh + build_demo.sh）
  --nfs <目录>         覆盖 NFS 根目录（默认 /run/media/manu/SSD960-LOCAL/nfs）
  --verify             拷贝后 ssh 到板卡跑一次 demo（零参数，--limit 6）
  --board <目标[:端口]>  覆盖 ssh 目标（默认 root@192.168.0.64，端口 22）

例:
  bash copy_to_nfs.sh                        # 只拷贝 + md5 回验
  bash copy_to_nfs.sh --build --verify       # 编译 + 拷贝 + 上板验证
  bash copy_to_nfs.sh --nfs /tmp/test_nfs    # 换目标目录（不碰真 NFS）
  bash copy_to_nfs.sh --board root@1.2.3.4:2222
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

echo "== 一键拷贝 =="
echo "  包目录 : $HERE"
echo "  NFS 根 : $NFS_ROOT"
echo "  目标   : $DEST"

# ---------------------------------------------------------------------------
# 2. 检查 NFS 挂载点
# ---------------------------------------------------------------------------
[ -d "$NFS_ROOT" ] && [ -w "$NFS_ROOT" ] || \
  die "NFS 挂载点不存在或不可写：$NFS_ROOT（先挂载 NFS）"

# ---------------------------------------------------------------------------
# 3. 可选：先编译
# ---------------------------------------------------------------------------
if [ "$DO_BUILD" = "1" ]; then
  echo
  echo "== [1/5] 编译 =="
  bash "$HERE/build_lib.sh"  || die "build_lib.sh 失败（编译错误见上）"
  bash "$HERE/build_demo.sh" || die "build_demo.sh 失败（编译错误见上）"
else
  echo
  echo "== [1/5] 跳过编译（用已有的 out/ 产物）=="
fi

# ---------------------------------------------------------------------------
# 4. 检查待拷贝的文件是否齐全
# ---------------------------------------------------------------------------
echo
echo "== [2/5] 检查产物 =="
for f in out/mosaic_api_test out/mosaic_contract_test out/libmosaic_pipeline.a \
         src/npu/mosaic_pipeline.h runtime/kernels/warp_median_fused.cl \
         runtime/kernels/sortnet_generated.inc runtime/kernels/gmc_feature_gpu.cl \
         README.md MANIFEST.md copy_to_nfs.sh; do
  [ -f "$HERE/$f" ] || die "缺少 $f —— 先跑 build_lib.sh / build_demo.sh（--build 可自动做）"
  echo "  [ok] $f"
done

# ---------------------------------------------------------------------------
# 5. 拷贝
# ---------------------------------------------------------------------------
echo
echo "== [3/5] 拷贝到 $DEST =="
mkdir -p "$DEST/kernels"

# (源, 目标) 配对：拷贝与 md5 回验共用这一张表，不会漏检、也不会拷错地方
PAIRS=(
  "out/mosaic_api_test|$DEST/mosaic_api_test"
  "out/mosaic_contract_test|$DEST/mosaic_contract_test"
  "out/libmosaic_pipeline.a|$DEST/libmosaic_pipeline.a"
  "src/npu/mosaic_pipeline.h|$DEST/mosaic_pipeline.h"
  "README.md|$DEST/README.md"
  "MANIFEST.md|$DEST/MANIFEST.md"
  "copy_to_nfs.sh|$DEST/copy_to_nfs.sh"
  "runtime/kernels/gmc_feature_gpu.cl|$DEST/kernels/gmc_feature_gpu.cl"
  "runtime/kernels/warp_median_fused.cl|$DEST/kernels/warp_median_fused.cl"
  "runtime/kernels/sortnet_generated.inc|$DEST/kernels/sortnet_generated.inc"
)

for pair in "${PAIRS[@]}"; do
  src="${pair%%|*}"; dst="${pair##*|}"
  cp -f "$HERE/$src" "$dst"
done
chmod +x "$DEST/copy_to_nfs.sh" 2>/dev/null || true
echo "  [ok] 已拷贝 ${#PAIRS[@]} 个文件"

# ---------------------------------------------------------------------------
# 6. 逐文件 md5 回验（NFS 写坏了要立刻知道）
# ---------------------------------------------------------------------------
echo
echo "== [4/5] md5 回验 =="
for pair in "${PAIRS[@]}"; do
  src="${pair%%|*}"; dst="${pair##*|}"
  sm=$(md5sum "$HERE/$src" | cut -d' ' -f1)
  dm=$(md5sum "$dst" | cut -d' ' -f1)
  if [ "$sm" != "$dm" ]; then
    die "md5 回验失败：$dst（源=$sm 目标=$dm）—— NFS 写入可能损坏"
  fi
  echo "  [ok] ${dst#$DEST/}  $dm"
done

# ---------------------------------------------------------------------------
# 7. 打印目标清单 + 板卡侧路径 + 运行命令
# ---------------------------------------------------------------------------
echo
echo "== [5/5] 目标目录清单 =="
find "$DEST" -type f | sort | while read -r p; do
  printf "  %-46s %10s B  %s\n" "${p#$DEST/}" "$(stat -c%s "$p")" "$(md5sum "$p" | cut -d' ' -f1)"
done

echo
echo "== 板卡侧路径 =="
echo "  本机 NFS 根 : $NFS_ROOT"
if [ "$NFS_ROOT" = "/run/media/manu/SSD960-LOCAL/nfs" ]; then
  echo "  板卡挂载点  : /mnt/manu   =>  $DEST  ==>  /mnt/manu/mosaic_sdk/"
else
  echo "  ⚠️  非默认 NFS 根：板卡侧挂载点未知（只有默认根才对应 /mnt/manu）"
  echo "     本机目标：$DEST"
fi
echo
echo "== 板卡运行命令（可直接粘贴）=="
cat <<EOF
ssh -p $BOARD_PORT $BOARD '/mnt/manu/mosaic_sdk/mosaic_api_test \\
  --raw-root   /mnt/manu/lq_ir_bmp \\
  --sequence   VIDEO00007_19700101_001324 \\
  --model      /root/psearch/models/p0_off_A_none.rknn \\
  --kernel-dir /root/kernels \\
  --limit 12'
EOF

# ---------------------------------------------------------------------------
# 8. 可选：ssh 到板卡验证（只执行 NFS 上的程序，绝不往板卡写东西）
# ---------------------------------------------------------------------------
if [ "$DO_VERIFY" = "1" ]; then
  echo
  echo "== 板卡验证（ssh 执行 NFS 上的程序，不写板卡）=="
  echo "  目标: $BOARD:$BOARD_PORT"
  echo "  命令: /mnt/manu/mosaic_sdk/mosaic_api_test --limit 6"
  echo "  ------------------------------------------------"
  rc=0
  ssh -p "$BOARD_PORT" -o ConnectTimeout=15 "$BOARD" \
      '/mnt/manu/mosaic_sdk/mosaic_api_test --limit 6' || rc=$?
  echo "  ------------------------------------------------"
  echo "  板卡退出码: $rc"
  [ "$rc" = "0" ] || die "板卡验证失败（退出码 $rc）"
  echo "  [ok] 板卡从 NFS 跑通"
fi

echo
echo "[ok] 全部完成。"
