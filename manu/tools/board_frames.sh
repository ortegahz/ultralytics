#!/usr/bin/env bash
# board_frames.sh -- 板卡取帧的两条通路，按「板卡磁盘占用」与「板卡 ctx 占用」两个约束选择。
#
# ## 为什么需要它
# 板卡 /dev/root 只剩 1.4 G，/userdata 剩 12 G。全量三轨 60,610 帧 BMP 约 27 G
# ⇒ 装不下板卡，只能分块推。而板端同时只允许 2 个 RKNN ctx，板卡是稀缺资源。
#
# ## 两条通路
#
#   通路 A：服务器 →(SSH tar 流)→ 板卡 /userdata
#     - 板卡占用磁盘（每块需清理）
#     - 实测吞吐 84 MB/s（板卡 root 免密 key 写服务器 7.4T SSD）
#     - 优点：板卡本地读，无 NFS 延迟抖动；缺点：吃板卡磁盘
#
#   通路 B：服务器 →(本机 NFS 导出)→ 板卡 /mnt/manu 直读   ★ 推荐
#     - 板卡占用磁盘 = 0
#     - 本机 IP 192.168.0.115，NFS 导出目录 /run/media/manu/SSD960-LOCAL/nfs
#       板卡已把它挂在 /mnt/manu（nfs4, 880 G, 剩 601 G）
#     - 2026-10-23 已实测：workstation 写入 → 板卡 /mnt/manu 立即可见
#     - 缺点：NFS over TCP 在高并发下有抖动；单 ctx 顺序读实测无问题
#
# ## 用法
#   bash manu/tools/board_frames.sh probe                 # 探测两条通路可用性
#   bash manu/tools/board_frames.sh push  <n> <outdir>     # 服务器→本机NFS 推 n 帧
#   bash manu/tools/board_frames.sh board-push <n> <outdir> # 服务器→板卡 tar 流
#   bash manu/tools/board_frames.sh list                   # 板卡上的模型与帧
#   bash manu/tools/board_frames.sh clean                  # 清板卡 /userdata 里的帧
#
set -uo pipefail

SSH_SRV="ssh -p 32222 -o ConnectTimeout=10 -o BatchMode=yes huangzhe@192.168.99.40"
SSH_BRD="ssh -p 22 -o ConnectTimeout=10 -o BatchMode=yes root@192.168.0.64"
WORK="/mnt/data/siping/ptq_work"
NFS_LOCAL="/run/media/manu/SSD960-LOCAL/nfs/ptq_frames"
NFS_BOARD="/mnt/manu/ptq_frames"

die() { echo "ERROR: $*" >&2; exit 1; }

cmd_probe() {
  echo "== 板卡 ctx（必须为 0 才能起活）=="
  $SSH_BRD "ps -eo comm | grep -c '^rknn_heatmap'" 2>/dev/null | tail -1
  echo
  echo "== 板卡磁盘 =="
  $SSH_BRD "df -h /dev/root /userdata 2>/dev/null | tail -2" 2>/dev/null
  echo
  echo "== 板卡 host NFS =="
  $SSH_BRD "df -h /mnt/manu 2>/dev/null | tail -1; ls /mnt/manu >/dev/null 2>&1 && echo '  NFS 可读写' || echo '  NFS 不可用'" 2>/dev/null
  echo
  echo "== 本机 NFS 导出 =="
  df -h "$NFS_LOCAL" 2>/dev/null | tail -1 || df -h /run/media/manu/SSD960-LOCAL/nfs | tail -1
  echo
  echo "== 服务器可达性 =="
  $SSH_SRV "echo '  服务器 OK: ' \$(hostname)" 2>/dev/null
}

cmd_push() {
  local n="${1:?帧数}" outdir="${2:-$NFS_LOCAL}"
  mkdir -p "$outdir"
  echo "从服务器取 $n 帧 → 本机 NFS $outdir"
  $SSH_SRV "ls $WORK/bmp/*/*.bmp 2>/dev/null | head -$n | tar -C $WORK/bmp -cf - -T - 2>/dev/null" \
    | tar -C "$outdir" -xf - 2>/dev/null
  local got
  got=$(find "$outdir" -name '*.bmp' | wc -l)
  echo "  已落地 $got 帧（$(du -sh "$outdir" 2>/dev/null | cut -f1)）"
  echo "  板卡侧路径：$NFS_BOARD/$(basename "$outdir")"
}

cmd_board_push() {
  local n="${1:?帧数}" outdir="${2:-/userdata/ptq_frames}"
  echo "从服务器经 SSH tar 流推 $n 帧 → 板卡 $outdir"
  $SSH_BRD "mkdir -p $outdir" 2>/dev/null
  $SSH_SRV "ls $WORK/bmp/*/*.bmp 2>/dev/null | head -$n | tar -C $WORK/bmp -cf - -T - 2>/dev/null" \
    | $SSH_BRD "tar -C $outdir -xf - 2>/dev/null"
  echo "  板卡侧：$outdir"
  $SSH_BRD "ls $outdir | wc -l" 2>/dev/null
}

cmd_list() {
  echo "== 板卡模型 =="
  $SSH_BRD "ls -la /userdata/ptq_models/ 2>/dev/null" 2>/dev/null
  echo
  echo "== 板卡帧 =="
  $SSH_BRD "du -sh /userdata/ptq_frames 2>/dev/null; ls /userdata/ptq_frames 2>/dev/null | head" 2>/dev/null
  echo
  echo "== 本机 NFS 帧 =="
  du -sh "$NFS_LOCAL" 2>/dev/null; ls "$NFS_LOCAL" 2>/dev/null | head
}

cmd_clean() {
  echo "清理板卡 /userdata 里的帧（不动模型）"
  $SSH_BRD "rm -rf /userdata/ptq_frames/* 2>/dev/null; df -h /userdata | tail -1" 2>/dev/null
}

case "${1:-}" in
  probe)     cmd_probe ;;
  push)      shift; cmd_push "$@" ;;
  board-push) shift; cmd_board_push "$@" ;;
  list)      cmd_list ;;
  clean)     cmd_clean ;;
  *) echo "用法: board_frames.sh <probe|push|board-push|list|clean>"; exit 1 ;;
esac
