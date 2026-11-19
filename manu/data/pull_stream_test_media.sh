#!/usr/bin/env bash
# 把服务器上的推流素材交付目录并行拉回 host NFS（板卡 /mnt/manu/... 可见）
#
# 为什么不在服务器直接写 NFS：
#   host 的 /etc/exports 只放开 192.168.0.0/24，而服务器在 192.168.99.40
#   ⇒ NFS 挂载会被 ACL 拒绝；改 exports 需要 host root（沙箱内无 sudo）⇒ 只能 host 侧拉回。
#
# 为什么用并行 rsync：
#   单条 SSHFS 通道实测 ~7.8 MB/s（11 GB 要 24 min）；并行多通道可线性提速。
#
# 用法:
#   bash pull_stream_test_media.sh <服务器侧目录> <host NFS 目标目录> [并发数]
set -uo pipefail

SRC="${1:?用法: $0 <服务器侧目录> <host NFS 目标目录> [并发数]}"
DST="${2:?用法: $0 <服务器侧目录> <host NFS 目标目录> [并发数]}"
JOBS="${3:-6}"

MIRROR="${MIRROR:-/home/manu/mnt/data}"
[[ "$SRC" == /mnt/data/* ]] && SRC="$MIRROR${SRC#/mnt/data}"

[[ -d "$SRC" ]] || { echo "[FATAL] 源目录不存在: $SRC" >&2; exit 2; }
SRC_ABS="$SRC"
export SRC_ABS
mkdir -p "$DST" || exit 2

echo "[PULL] src=$SRC"
echo "[PULL] dst=$DST"
echo "[PULL] 并发=$JOBS"
du -sh "$SRC" 2>/dev/null | sed 's/^/[PULL] 源大小: /'

# 并行拉取：每条 SSHFS 通道约 7.8 MB/s ⇒ 多通道才能跑出带宽
#   （小文件用 rsync 一次收；大文件用 xargs 并行 cp，避免单流串行）
export SRC_ABS DST
copy_one() {
  local rel="$1"
  local s="$SRC_ABS/$rel"
  local d="$DST/$rel"
  mkdir -p "$(dirname "$d")"
  for attempt in 1 2 3; do
    if cp -f "$s" "$d.part" 2>/dev/null; then
      mv -f "$d.part" "$d" && return 0
    fi
    sleep $((attempt * 3))
  done
  echo "[PULL][FAIL] $rel" >&2
  return 1
}
export -f copy_one

# 源路径在 SRC 下，rel 是相对 SRC 的路径
( cd "$SRC" && find . -type f ! -path './_logs/*' ! -name '.*' -printf '%P\n' ) \
  | xargs -d '\n' -P "$JOBS" -I{} bash -c 'copy_one "$@"' _ {}

# 收口对账：逐文件比对字节数
python3 - "$SRC" "$DST" <<'PY'
import os, sys
src, dst = sys.argv[1], sys.argv[2]
bad, miss, n, total = [], [], 0, 0
for root, _, files in os.walk(src):
    rel_root = os.path.relpath(root, src)
    # 与拷贝阶段(find)的过滤谓词保持一致：排除 _logs/ 与隐藏文件
    if rel_root != "." and rel_root.split(os.sep)[0] == "_logs":
        continue
    for f in files:
        if f.startswith('.') or f.endswith('.log'):
            continue
        s = os.path.join(root, f)
        rel = os.path.relpath(s, src)
        d = os.path.join(dst, rel)
        if not os.path.isfile(d):
            miss.append(rel); continue
        ss, ds = os.path.getsize(s), os.path.getsize(d)
        n += 1; total += ss
        if ss != ds:
            bad.append(f"{rel}: {ds} != {ss}")
print(f"[PULL] 校验文件数={n}  总字节={total}")
if miss: print("[PULL] 缺失:", *miss, sep="\n  ")
if bad:  print("[PULL] 尺寸不符:", *bad, sep="\n  ")
print("[PULL] RESULT:", "OK" if not miss and not bad else "FAIL")
sys.exit(0 if (not miss and not bad) else 1)
PY
