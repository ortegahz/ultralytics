#!/usr/bin/env bash
# host 侧 memory 回填（server -> host）
#
# 为什么必须在 host 上执行：
#   - host（192.168.0.115）**没有 sshd** ⇒ 服务器无法主动推送；
#   - 服务器侧无 host 仓库副本、无 NFS 挂载（服务器是 NFS 服务端，方向是 server->board）；
#   - 故 host 开机恢复 192.168.0.x 路由后，由人在 host 终端主动拉取。
#
# 用法（在 host 的普通终端执行）:
#   bash ultralytics/manu/memory/delivery_20261119/HOST_BACKFILL.sh            # 拉取 + md5 复核
#   bash ultralytics/manu/memory/delivery_20261119/HOST_BACKFILL.sh --check    # 只预检，不拉取
#   RS=/path/to/ultralytics bash .../HOST_BACKFILL.sh                          # 手工指定 host 仓库根目录
#
# 服务器侧源: test-Super-Server 192.168.99.40:32222 (huangzhe)
set -euo pipefail

# The server sits on 192.168.99.0/24 while the host sits on 192.168.0.0/24 -- whether the host can
# route to 192.168.99.x is NOT verified from here, so probe the candidates instead of assuming one.
SERVER_CANDIDATES="${SERVER_CANDIDATES:-192.168.99.40}"
SERVER_IP="${SERVER_IP:-}"          # resolved by pick_server
SERVER_PORT=32222
SERVER_USER=huangzhe
SERVER_ROOT=/mnt/data/siping/runs/npu20ms/nano_search/repo
MANIFEST_REL=manu/memory/delivery_20261119/artifacts/memory_md5_manifest.txt
SSH_KEY="${SSH_KEY:-}"          # optional -i <key>; the server is pubkey-only (PasswordAuthentication no)
probe() {   # $1 = candidate ip -> prints a reason and returns non-zero if unusable
  local ip="$1"
  if ! (exec 3<>"/dev/tcp/${ip}/${SERVER_PORT}") 2>/dev/null; then
    echo "TCP ${ip}:${SERVER_PORT} 不可达（路由或防火墙）"; return 1
  fi
  # 诊断必须区分三种失败：路由不通 / 主机密钥没被记录 / 密钥没被授权。
  # 早前这里只报"密钥未授权"，而 BatchMode 下"Host key verification failed"也会走到同一分支 ⇒ 误诊。
  local err
  err="$(timeout 8 $SSH -o ConnectTimeout=6 "${SERVER_USER}@${ip}" true 2>&1)" && { echo "OK"; return 0; }
  case "$err" in
    *"Host key verification failed"*|*"REMOTE HOST IDENTIFICATION HAS CHANGED"*)
      echo "TCP 可达，但服务器主机密钥未记录/已变更（用 StrictHostKeyChecking=accept-new 放行）"; return 1 ;;
    *"Permission denied"*|*"publickey"*)
      echo "TCP 可达，但本机密钥未被授权（服务器 PasswordAuthentication=no，仅公钥）"; return 1 ;;
    *"timed out"*|*"Connection refused"*|*"No route"*)
      echo "TCP 握手失败（路由/防火墙）"; return 1 ;;
    *)
      echo "TCP 可达，但 ssh 失败：$(echo "$err" | tail -1)"; return 1 ;;
  esac
}
pick_server() {
  local ip reason
  for ip in $SERVER_CANDIDATES; do
    reason=$(probe "$ip") && { echo "$ip"; return 0; }
    echo "[CHECK] ${ip}: ${reason}" >&2
  done
  return 1
}
# accept-new：首次连服务器时自动记录主机密钥（否则 BatchMode 下直接失败，且会被误诊成"密钥未授权"）。
KNOWN_HOSTS="${KNOWN_HOSTS:-$HOME/.ssh/known_hosts}"
SSH="ssh -p ${SERVER_PORT} -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=${KNOWN_HOSTS}"
[ -n "$SSH_KEY" ] && SSH="$SSH -i $SSH_KEY -o IdentitiesOnly=yes"

# The host repo root: prefer an explicit RS, else derive it from this script's own location
# (<repo>/manu/memory/delivery_20261119/HOST_BACKFILL.sh -> <repo>), else fall back to the
# historical path.  The derived root is only accepted when it really is a repo root.
detect_root() {
  local here
  here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local up
  up=$(cd "$here/../../.." 2>/dev/null && pwd || true)
  if [ -n "$up" ] && [ -d "$up/manu/memory" ]; then
    echo "$up"
  elif [ -d "/media/manu/1TB-Volume/workspace/ultralytics/manu/memory" ]; then
    echo "/media/manu/1TB-Volume/workspace/ultralytics"
  fi
}
RS="${RS:-$(detect_root)}"
if [ -z "$RS" ]; then
  echo "[FAIL] 找不到 host 仓库根目录：请用 RS=<repo 根目录> 指定" >&2
  exit 2
fi

FILES="memory.md memory_compact.md memory/memory.md memory/memory_compact.md manu/memory/memory.md manu/memory/memory_compact.md"

echo "[INFO] 服务器   ${SERVER_USER}@${SERVER_IP:-<待探测>}:${SERVER_PORT}${SERVER_ROOT}"
echo "[INFO] host 仓库根目录 = $RS"
echo "[INFO] 待回填 ${FILES}"

if [ "${1:-}" = "--check" ]; then
  echo
  echo "[CHECK] 服务器连通性（候选：$SERVER_CANDIDATES）"
  if ! SERVER_IP=$(pick_server); then
    echo "[FAIL] 没有可用的服务器通道。" >&2
    echo "       排查顺序：① host 能否路由到 192.168.99.0/24；" >&2
    echo "       ② host 的公钥是否已加进服务器 /home/${SERVER_USER}/.ssh/authorized_keys；" >&2
    echo "       ③ 用 SSH_KEY=<私钥路径> 指定密钥。" >&2
    echo "       ④ 若都不通：直接在服务器侧把 6 份 memory 拷到 host（见交付包 osd_manifest.json 顶部说明）。" >&2
    exit 1
  fi
  echo "[CHECK] 可用服务器 = ${SERVER_IP}"
  $SSH "${SERVER_USER}@${SERVER_IP}" "ls -l ${SERVER_ROOT}/${MANIFEST_REL}" || {
    echo "[FAIL] 能登录但读不到 ${MANIFEST_REL}" >&2; exit 1; }
  echo
  echo "[CHECK] 本机目标目录"
  for f in $FILES; do
    if [ -f "$RS/$f" ]; then echo "  存在   $f  ($(md5sum "$RS/$f" | cut -d' ' -f1))"
    else echo "  将新建 $f"; fi
  done
  echo
  echo "[CHECK] 通过。去掉 --check 即可正式回填。"
  exit 0
fi

if ! SERVER_IP=$(pick_server); then
  echo "[FAIL] 没有可用的服务器通道（见 --check 的排查提示）" >&2
  exit 1
fi
echo "[INFO] 使用服务器 ${SERVER_IP}:${SERVER_PORT}"

# ---- 落地方式：只合并「§257 十八」这一块，绝不整文件覆盖 --------------------
# 为什么不整文件 rsync：host 上可能有并发线（升轨 / 视频质量指标线）写的 §258/§259/§260，
# 整文件覆盖会把它们换成服务器副本（可能更旧）⇒ 静默吞掉对方章节。
# 所以这里取回「合并工具 + 两份正文」，在 host 本机做章节级合并，块外字节逐字节保留。

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
DEL_REL="manu/memory/delivery_20261119"
MANIFEST_REL="$DEL_REL/artifacts/memory_md5_manifest.txt"

echo "[1/4] 从服务器取回合并工具 + 两份正文 + md5 清单"
mkdir -p "$RS/$DEL_REL/artifacts"
rsync -a --partial -e "$SSH" \
  "${SERVER_USER}@${SERVER_IP}:${SERVER_ROOT}/${DEL_REL}/merge_memory_section.py" "$TMP/merge_memory_section.py"
rsync -a --partial -e "$SSH" \
  "${SERVER_USER}@${SERVER_IP}:${SERVER_ROOT}/${DEL_REL}/artifacts/memory_section_257_18.md" "$TMP/memory_section_257_18.md"
rsync -a --partial -e "$SSH" \
  "${SERVER_USER}@${SERVER_IP}:${SERVER_ROOT}/${DEL_REL}/artifacts/memory_section_257_18_compact.md" "$TMP/memory_section_257_18_compact.md"
rsync -a --partial -e "$SSH" \
  "${SERVER_USER}@${SERVER_IP}:${SERVER_ROOT}/${MANIFEST_REL}" "$RS/${MANIFEST_REL}"
chmod +x "$TMP/merge_memory_section.py"

echo "[2/4] 合并「§257 十八」到本机 6 份 memory（只动这一块）"
cd "$RS"
MAIN_FILES="memory.md memory/memory.md manu/memory/memory.md"
COMPACT_FILES="memory_compact.md memory/memory_compact.md manu/memory/memory_compact.md"
for f in $MAIN_FILES; do
  mkdir -p "$(dirname "$f")"
  python3 "$TMP/merge_memory_section.py" "$f" "$TMP/memory_section_257_18.md" || exit 1
done
for f in $COMPACT_FILES; do
  mkdir -p "$(dirname "$f")"
  python3 "$TMP/merge_memory_section.py" "$f" "$TMP/memory_section_257_18_compact.md" || exit 1
done

echo "[3/4] 复核：正文 md5 与服务器清单一致 + 并发线章节仍在位"
EXPECT_FULL="$(md5sum "$TMP/memory_section_257_18.md" | cut -d' ' -f1)"
EXPECT_COMPACT="$(md5sum "$TMP/memory_section_257_18_compact.md" | cut -d' ' -f1)"
EXPECT_FULL="$EXPECT_FULL" EXPECT_COMPACT="$EXPECT_COMPACT" python3 - <<'PY'
import hashlib, os, pathlib
NL = chr(10)
def block(p):
    t = pathlib.Path(p).read_text(encoding='utf-8'); ls = t.splitlines(keepends=True)
    idx = [i for i, l in enumerate(ls) if l.startswith('## §')]
    for n, s in enumerate(idx):
        if ls[s].startswith('## §257 十八'):
            e = idx[n + 1] if n + 1 < len(idx) else len(ls)
            return ''.join(ls[s:e])
    return None
full, compact = os.environ['EXPECT_FULL'], os.environ['EXPECT_COMPACT']
rc = 0
for f in ['memory.md', 'memory/memory.md', 'manu/memory/memory.md',
          'memory_compact.md', 'memory/memory_compact.md', 'manu/memory/memory_compact.md']:
    p = pathlib.Path(f)
    if not p.exists():
        print('  MISSING ' + f); rc = 1; continue
    b = block(p)
    if b is None:
        print('  NO-SECTION ' + f); rc = 1; continue
    got = hashlib.md5((b.rstrip(NL) + NL).encode()).hexdigest()
    want = compact if 'compact' in f else full
    t = p.read_text(encoding='utf-8')
    keep = sum(1 for k in ('## §258', '## §259', '## §260') if k in t)
    ok = got == want and keep == 3
    print('  %-32s sec_md5=%s %s concurrent_kept=%d/3' % (f, got, 'OK' if got == want else 'MISMATCH', keep))
    if not ok:
        rc = 1
raise SystemExit(rc)
PY
[ $? -eq 0 ] || { echo "[BACKFILL FAIL] 复核未通过" >&2; exit 1; }

echo "[4/5] 拉取交付包（新参数档 + 对照表 + OSD 清单 + 证据文件）"
PKG_LOCAL="$RS/manu/memory/delivery_20261119"
mkdir -p "$PKG_LOCAL/artifacts"
rsync -a --partial -e "$SSH" \
  "${SERVER_USER}@${SERVER_IP}:${SERVER_ROOT}/${DEL_REL}/COMPARISON.md" \
  "${SERVER_USER}@${SERVER_IP}:${SERVER_ROOT}/${DEL_REL}/osd_manifest.json" \
  "${SERVER_USER}@${SERVER_IP}:${SERVER_ROOT}/${DEL_REL}/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json" \
  "$PKG_LOCAL/"
# r84: 推送路排掉 *.bak*，拉取路此前只排 __pycache__ ⇒ 拉取路会把 12 个 .bak*（含 4 个**旧版闸门脚本**）带到 host。
# 两路筛选规则现在一致。
rsync -a --partial --exclude '__pycache__' --exclude '*.bak*' -e "$SSH" \
  "${SERVER_USER}@${SERVER_IP}:${SERVER_ROOT}/${DEL_REL}/artifacts/" \
  "$PKG_LOCAL/artifacts/"
rsync -a --partial -e "$SSH" \
  "${SERVER_USER}@${SERVER_IP}:${SERVER_ROOT}/${DEL_REL}/merge_memory_section.py" \
  "$PKG_LOCAL/"
echo "[BACKFILL PKG] 交付包已落到 $PKG_LOCAL"

echo "[5/5] 附：整档 md5 对照（**信息项，不是闸门**）"
echo "        host 侧若保留了自己独有的并发章节，整档 md5 本就应与服务器不同。"
while read -r want path; do
  [ -z "${want:-}" ] && continue
  case "$want" in \#*) continue ;; esac
  got=$(md5sum "$path" 2>/dev/null | cut -d' ' -f1)
  if [ "$got" = "$want" ]; then echo "  相同   $path"; else echo "  不同   $path  host=$got server=$want"; fi
done < "$RS/${MANIFEST_REL}"

echo "[BACKFILL OK] 「§257 十八」已合并进 6 份 memory；并发线章节原样保留。"
