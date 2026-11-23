#!/usr/bin/env bash
# 服务器侧 memory 推送（server -> host）
#
# 为什么需要它：
#   - host（192.168.0.115）关机后恢复，host->server 的**拉取**方向在认证上走不通
#     （服务器 authorized_keys 只有 id_board 与 board-ptq 两把，没有 host 的公钥）；
#   - 但服务器持有 ~/.ssh/id_ed25519_host_zerg（ssh config 里就是 Host host-zerg）
#     ⇒ **服务器->主机**的推送方向有可用密钥；
#   - 所以两条路都留着：HOST_BACKFILL.sh（host 侧拉）+ 本脚本（server 侧推）。
#
# 安全性：绝不做整文件覆盖。只把本线拥有的「§257 十八」块写进 host 的 memory
#   （缺失->插入 / 旧版->原位替换 / 已一致->no-op），**块外字节逐字节保留**，
#   并发线的 §258 / §259 / §260 以及 host 独有的内容一律不动。
#
# 用法（服务器侧执行）:
#   bash manu/memory/delivery_20261119/HOST_PUSH.sh --check   # 只探测，不写入
#   bash manu/memory/delivery_20261119/HOST_PUSH.sh           # 正式推送 + 复核
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_ROOT="${SERVER_ROOT:-/mnt/data/siping/runs/npu20ms/nano_search/repo}"
SEARCH_DIR="${SEARCH_DIR:-/mnt/data/siping/runs/trial0080_onnx/search}"

HOST_ALIAS="${HOST_ALIAS:-host-zerg}"          # ssh config: zerg@192.168.0.115
HOST_FALLBACK="${HOST_FALLBACK:-zerg@192.168.0.115}"
HOST_PORT="${HOST_PORT:-22}"
REMOTE_ROOT="${REMOTE_ROOT:-}"                # 可显式指定，否则自动探测
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_host_zerg}"
# ssh 用 -p 指端口，scp 用 -P 指端口（-p 在 scp 里是"保留时间戳"）—— 两者必须分开。
# 服务器的 /home 是只读挂载 ⇒ ~/.ssh/known_hosts 写不进去 ⇒ 用可写的 UserKnownHostsFile。
KNOWN_HOSTS="${KNOWN_HOSTS:-/tmp/.manu_known_hosts}"
SSH_OPTS="-p ${HOST_PORT} -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=${KNOWN_HOSTS}"
SCP_OPTS="-P ${HOST_PORT} -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=${KNOWN_HOSTS}"
[ -f "$SSH_KEY" ] && { SSH_OPTS="$SSH_OPTS -i $SSH_KEY -o IdentitiesOnly=yes"; SCP_OPTS="$SCP_OPTS -i $SSH_KEY -o IdentitiesOnly=yes"; }

MERGE_TOOL="$HERE/merge_memory_section.py"
BODY_FULL="${BODY_FULL:-$HERE/artifacts/memory_section_257_18.md}"
BODY_COMPACT="${BODY_COMPACT:-$HERE/artifacts/memory_section_257_18_compact.md}"

for f in "$MERGE_TOOL" "$BODY_FULL" "$BODY_COMPACT"; do
  [ -f "$f" ] || { echo "[FAIL] 缺文件: $f" >&2; exit 2; }
done

MAIN_FILES="memory.md memory/memory.md manu/memory/memory.md"
COMPACT_FILES="memory_compact.md memory/memory_compact.md manu/memory/memory_compact.md"

USE_HOST=""
ssh_cmd() {  # $@ = remote command
  [ -n "$USE_HOST" ] && { ssh $SSH_OPTS "$USE_HOST" "$@"; return $?; }
  ssh $SSH_OPTS "$HOST_ALIAS" "$@" 2>/dev/null && return 0
  ssh $SSH_OPTS "$HOST_FALLBACK" "$@" 2>/dev/null
}

echo "[INFO] 服务器 $SERVER_ROOT"
echo "[INFO] 目标主机 ${HOST_ALIAS} (fallback ${HOST_FALLBACK})，密钥 ${SSH_KEY}"

if ssh $SSH_OPTS "$HOST_ALIAS" true 2>/dev/null; then
  USE_HOST="$HOST_ALIAS"
elif ssh $SSH_OPTS "$HOST_FALLBACK" true 2>/dev/null; then
  USE_HOST="$HOST_FALLBACK"
fi
if [ -z "$USE_HOST" ]; then
  echo "[FAIL] 主机不可达/认证不通。" >&2
  echo "       排查：① host 是否已开机并恢复到 192.168.0.x；" >&2
  echo "             ② host 的 sshd 是否在跑（${HOST_FALLBACK}:22）；" >&2
  echo "             ③ 改用 host 侧拉取：bash ultralytics/manu/memory/delivery_20261119/HOST_BACKFILL.sh --check" >&2
  exit 1
fi
echo "[INFO] 主机连通 OK"

if [ -z "$REMOTE_ROOT" ]; then
REMOTE_ROOT="$(ssh_cmd 'for c in /media/manu/1TB-Volume/workspace/ultralytics "$HOME/workspace/ultralytics" "$HOME/ultralytics"; do [ -d "$c/manu/memory" ] && { echo "$c"; exit 0; }; done; d=$(find /media /home -maxdepth 5 -type d -path "*/manu/memory" 2>/dev/null | head -1); [ -n "$d" ] && dirname "$(dirname "$d")"' || true)"
fi
if [ -z "${REMOTE_ROOT:-}" ]; then
  echo "[FAIL] 在主机上找不到 ultralytics 仓库根目录（找 */manu/memory）。" >&2
  echo "       请在主机上手工指定 REMOTE_ROOT 后重跑：REMOTE_ROOT=/path/to/ultralytics bash $0" >&2
  exit 2
fi
echo "[INFO] 主机仓库根目录 = $REMOTE_ROOT"

if [ "${1:-}" = "--check" ]; then
  echo "[CHECK] 主机上待合并的文件："
  for f in $MAIN_FILES $COMPACT_FILES; do
    ssh_cmd "if [ -f '$REMOTE_ROOT/$f' ]; then echo '  存在   $f'; else echo '  将新建 $f'; fi"
  done
  echo "[CHECK] 通过。去掉 --check 即可正式推送。"
  exit 0
fi

REMOTE_TMP="/tmp/.manu_memory_push_$$"
ssh_cmd "rm -rf '$REMOTE_TMP' && mkdir -p '$REMOTE_TMP'"
scp $SCP_OPTS "$MERGE_TOOL" "$BODY_FULL" "$BODY_COMPACT" "${USE_HOST}:$REMOTE_TMP/"
echo "[INFO] 工具与正文已上传到主机 $REMOTE_TMP"

ssh_cmd "cd '$REMOTE_ROOT' && for f in $MAIN_FILES; do python3 '$REMOTE_TMP/merge_memory_section.py' \"\$f\" '$REMOTE_TMP/memory_section_257_18.md'; done; for f in $COMPACT_FILES; do python3 '$REMOTE_TMP/merge_memory_section.py' \"\$f\" '$REMOTE_TMP/memory_section_257_18_compact.md'; done"
echo "[INFO] 合并完成"

echo "[INFO] 复核（主机侧）："
EXPECT_FULL="$(md5sum "$BODY_FULL" | cut -d' ' -f1)"
EXPECT_COMPACT="$(md5sum "$BODY_COMPACT" | cut -d' ' -f1)"
VERIFY_CMD="cd '$REMOTE_ROOT' && EXPECT_FULL='$EXPECT_FULL' EXPECT_COMPACT='$EXPECT_COMPACT' python3 - <<'PY'
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
    # 归一化成「恰好一个尾换行」再取 md5，才能与清单里的正文 md5 直接比对
    got = hashlib.md5((b.rstrip(NL) + NL).encode()).hexdigest()
    want = compact if 'compact' in f else full
    ok = got == want
    t = p.read_text(encoding='utf-8')
    keep = sum(1 for k in ('## §258', '## §259', '## §260') if k in t)
    flag = 'OK' if ok else 'MISMATCH-want-' + want
    print('  %-32s sec_md5=%s %s concurrent_kept=%d/3' % (f, got, flag, keep))
    if not ok or keep != 3:
        rc = 1
raise SystemExit(rc)
PY"
if ssh $SSH_OPTS "$USE_HOST" "$VERIFY_CMD"; then
  echo "[VERIFY OK] 主机侧 6 份文件：§257 十八 与服务器正文 md5 一致，且 §258/§259/§260 全部在位。"
else
  echo "[VERIFY FAIL] 主机侧复核未通过（见上表）。" >&2
  exit 1
fi
ssh_cmd "rm -rf '$REMOTE_TMP'"

# ---- 交付包一起推过去（参数档 / 对照表 / OSD 清单 / 证据文件）----------------
# 为什么必须一起推：交付包是「新参数档 + OSD 清单 + 对照表」的载体，
# 只推 memory 章节的话，host 上永远拿不到这轮的新参数档与对照表。
# 这里只写**本线自己的** delivery_20261119/ 目录，不碰并发线的任何 memory 章节。
echo "[INFO] 推交付包到 $REMOTE_ROOT/manu/memory/delivery_20261119/ ..."
PKG_REMOTE="$REMOTE_ROOT/manu/memory/delivery_20261119"
ssh_cmd "mkdir -p '$PKG_REMOTE/artifacts'"
# r84: 文档让 host 用户执行 `bash ultralytics/.../HOST_BACKFILL.sh`（拉取路），
# 但此前**两条路都不搬这个文件** ⇒ host 上根本没有它。随包一起推，拉取路才真的可执行。
scp $SCP_OPTS "$HERE/COMPARISON.md" "$HERE/osd_manifest.json" \
    "$HERE/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json" \
    "$HERE/HOST_BACKFILL.sh" \
    "${USE_HOST}:$PKG_REMOTE/"
# artifacts/ 按目录实际内容整搬：手写白名单会随证据增减漂移，host 上就会缺文件
#（例如「规范搜索产物」与 OSD 媒体核验脚本就没在白名单里，而 README 已把它们登记成交付物）。
# 只搬普通文件、排除 .bak 备份与 __pycache__，与拉取路 HOST_BACKFILL.sh 同口径。
mapfile -t PKG_ARTIFACTS < <(find "$HERE/artifacts" -maxdepth 1 -type f ! -name '*.bak*' -print | sort)
[ "${#PKG_ARTIFACTS[@]}" -gt 0 ] || { echo "[FAIL] artifacts 目录为空" >&2; exit 2; }
scp $SCP_OPTS "${PKG_ARTIFACTS[@]}" "${USE_HOST}:$PKG_REMOTE/artifacts/"
scp $SCP_OPTS "$HERE/merge_memory_section.py" "${USE_HOST}:$PKG_REMOTE/"
echo "[INFO] 交付包已推送"
ssh_cmd "cd '$PKG_REMOTE' && ls -1 . artifacts | head -40"

echo "[PUSH DONE] 主机侧 §257 十八 已合并；并发线章节由上面的 concurrent_kept 计数证明仍在；交付包已落 $PKG_REMOTE"
