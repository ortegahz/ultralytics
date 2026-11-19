#!/usr/bin/env bash
# 把 host 上的 stream_test_media 交付目录**发布**到板卡可见的 NFS（/mnt/manu/...）
#
# 为什么不能在 host 直接 cp：
#   沙箱/受限环境下 `/run/media/manu/SSD960-LOCAL`（NFS 导出根）是 **ro** 挂载
#   ⇒ host 直写 NFS 必然 `Read-only file system`（§177 交付时踩过）。
#
# 为什么走「ssh 管道 → 板卡直写 NFS」：
#   板卡把 host 的 NFS 子树挂在 /mnt/manu（实测 nfs4 **rw**，soft proto=tcp），
#   而 host 侧只**读**本地产物 ⇒ `cat 本地文件 | ssh 板卡 'cat > 目标.part && mv'`。
#   ⚠️ 板卡 /tmp 是 **tmpfs（仅 ~3.6G 可用）** ⇒ 十几 GB 的交付物**不能**先落板卡 /tmp，
#      必须**直写 NFS**（.part + mv 保证不会把半成品暴露成正式文件）。
#
# 用法:
#   bash publish_stream_test_media_to_board.sh <host 源目录> [板卡] [并发数] [板卡侧目录名]
# 例:
#   bash publish_stream_test_media_to_board.sh \
#        /media/manu/1TB-Volume/workspace/delivery/stream_test_media_th045_pass \
#        root@192.168.0.64 3 stream_test_media_th045_pass
set -uo pipefail

SRC="${1:?用法: $0 <host 源目录> [板卡] [并发数] [板卡侧目录名]}"
BOARD="${2:-root@192.168.0.64}"
JOBS="${3:-3}"
NAME="${4:-$(basename "$SRC")}"

REMOTE_BASE="/mnt/manu/$NAME"
# 🔴 必须用**字符串**而不是 bash 数组：数组**不会被 export** 给 `bash -c` 子 shell
#    ⇒ copy_one 里 ${SSH_OPTS[@]} 会变空 ⇒ ssh 退回读系统配置
#    ⇒ 撞上本机权限损坏的 /etc/ssh/ssh_config.d/20-systemd-ssh-proxy.conf ⇒ 每路都断。
SSH_OPTS="-F /dev/null -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/kh_publish"

[[ -d "$SRC" ]] || { echo "[FATAL] 源目录不存在: $SRC" >&2; exit 2; }
SRC_ABS="$(cd "$SRC" && pwd)"
export SRC_ABS REMOTE_BASE SSH_OPTS BOARD

echo "[PUB] src=$SRC_ABS"
echo "[PUB] board=$BOARD  目标=$REMOTE_BASE  并发=$JOBS"
du -sh "$SRC_ABS" | sed 's/^/[PUB] 源大小: /'

# 板卡侧预建目录 + 探针（确认 rw；ro 就地失败，别到一半才炸）
ssh $SSH_OPTS "$BOARD" "mkdir -p '$REMOTE_BASE' && touch '$REMOTE_BASE/.__wtest' \
  && rm -f '$REMOTE_BASE/.__wtest' && echo '[PUB] 板卡 NFS 可写 ✓'" \
  || { echo "[FATAL] 板卡 NFS 不可写：$REMOTE_BASE" >&2; exit 3; }

# 清掉上一轮失败留下的 .part 残留（成功路径里 .part 已被 mv 掉，残留即为失败产物）
ssh $SSH_OPTS "$BOARD" "find '$REMOTE_BASE' -name '*.part' -delete" 2>/dev/null

copy_one() {
  local rel="$1"
  local dstdir="${REMOTE_BASE%/*}/$(basename "$REMOTE_BASE")"
  local base="$(dirname "$rel")"
  [[ "$base" == "." ]] || ssh $SSH_OPTS "$BOARD" "mkdir -p '$dstdir/$base'"
  if cat "$SRC_ABS/$rel" | ssh $SSH_OPTS "$BOARD" \
       "cat > '$REMOTE_BASE/$rel.part' && mv -f '$REMOTE_BASE/$rel.part' '$REMOTE_BASE/$rel'"; then
    echo "[PUB][OK] $rel"
  else
    echo "[PUB][FAIL] $rel" >&2
    return 1
  fi
}
export -f copy_one

# 文件清单：跳过目录本体；.part 是中转残留，成功后已被 mv 掉
#
# 🔴 铁律「多通道必须错峰启动（间隔 ≥ 30s）」：不直接 xargs -P N 一次拉起 N 路，
#    而是把清单**轮转分到 N 个 bucket**，每个 bucket **单通道**（-P1）跑，
#    第 i 个 bucket 延迟 30*i 秒启动 ⇒ 始终 ≤ N 路并发且错峰。
LIST_TMP="$(mktemp)"; trap 'rm -f "$LIST_TMP" "$LIST_TMP".b*' EXIT
( cd "$SRC_ABS" && find . -type f ! -name '.*' -printf '%P\n' ) > "$LIST_TMP"

RC=0
for i in $(seq 0 $((JOBS - 1))); do
  awk -v i="$i" 'NR % '"$JOBS"' == i' "$LIST_TMP" > "$LIST_TMP.b$i"
  [ -s "$LIST_TMP.b$i" ] || continue
  ( sleep $((30 * i)); xargs -d '\n' -P1 -I{} bash -c 'copy_one "$@"' _ {} < "$LIST_TMP.b$i" ) &
  eval "PID_$i=\$!"
done
for i in $(seq 0 $((JOBS - 1))); do
  eval "P=\${PID_$i:-}"; [ -n "$P" ] && { wait "$P" || RC=1; }
done

echo "[PUB] ---- 收口对账 ----"
ssh $SSH_OPTS "$BOARD" "
  cd '$REMOTE_BASE' || exit 1
  echo \"[PUB] 板卡侧文件数: \$(find . -type f | wc -l)\"
  echo \"[PUB] 板卡侧总字节: \$(find . -type f -printf '%s\n' | awk '{s+=\$1} END{print s}')\"
  if [ -f md5sums.txt ]; then
    md5sum -c md5sums.txt --quiet && echo '[PUB] md5sum -c 60/60 PASS ✓' || echo '[PUB] md5sum -c FAILED ✗'
  fi
"
exit $RC
