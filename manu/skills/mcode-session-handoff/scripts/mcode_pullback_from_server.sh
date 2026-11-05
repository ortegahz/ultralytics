#!/usr/bin/env bash
# =============================================================================
# mcode 会话转移 —— server → host 回拉（逆向）
#
#   bash mcode_pullback_from_server.sh --session <id> \
#        [--server <host>] [--port N] [--user <u>] \
#        [--workspace <本机 workspace>] [--data-dir <本机 dataDir>] [--exec]
#
# 与 mcode_handoff_to_server.sh 的关系：
#   handoff  = host → server（把活儿交出去）
#   pullback = server → host（把活儿拿回来）
#   两者共用同一套搬运机制（mcode_handoff.py export/import），
#   方向只取决于 export 在哪台机器上跑、import 落到哪台机器上。
#
# ⚠️ 语义警告（详见 ../SKILL.md §10）：
#   推是**单向快照**，不是双向同步。回拉会用 server 版本**覆盖** host 版本，
#   host 上较新的轮次会丢失。回拉前请确认 host 侧没有未推送的进展。
#   本脚本默认**不覆盖**，先做 --dry-run 报告差异；确认后再加 --force。
# =============================================================================
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

SESSION=""
SERVER="192.168.99.40"
PORT="32222"
USER_="huangzhe"
WORKSPACE="/media/manu/1TB-Volume/workspace"
LOCAL_DATA_DIR=""
REMOTE_DATA_DIR="~/.minimax"
DO_EXEC="0"
EXEC_PROMPT="reply OK only"
FORCE="0"

while [ $# -gt 0 ]; do
  case "$1" in
    --session)       SESSION="$2"; shift 2 ;;
    --server)        SERVER="$2"; shift 2 ;;
    --port)          PORT="$2"; shift 2 ;;
    --user)          USER_="$2"; shift 2 ;;
    --workspace)     WORKSPACE="$2"; shift 2 ;;
    --data-dir)      LOCAL_DATA_DIR="$2"; shift 2 ;;
    --remote-data-dir) REMOTE_DATA_DIR="$2"; shift 2 ;;
    --exec)          DO_EXEC="1"; shift ;;
    --prompt)        EXEC_PROMPT="$2"; shift 2 ;;
    --force)         FORCE="1"; shift ;;
    -h|--help)       sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

[ -n "$SESSION" ] || { echo "❌ 缺少 --session" >&2; exit 2; }
[ -n "$LOCAL_DATA_DIR" ] || LOCAL_DATA_DIR="$HOME/.minimax"

SSH=(ssh -p "$PORT" -o BatchMode=yes -o ConnectTimeout=15 "${USER_}@${SERVER}")
SCP=(scp -P "$PORT" -o BatchMode=yes -o ConnectTimeout=15)

PACK="$(mktemp -t mcode_pullback_XXXXXX.json)"
REMOTE_PACK="/tmp/mcode_pullback_$$.json"
cleanup() { rm -f "$PACK"; }
trap cleanup EXIT

echo "═══ 1/4 export（服务器侧） ═══"
"${SSH[@]}" "export PATH=\$(ls -d \$HOME/.minimax-code/runtime/node-v*/bin 2>/dev/null | head -1):\$PATH; \
  mkdir -p /tmp/mcode_handoff_tools && \
  cd /tmp/mcode_handoff_tools && \
  python3 mcode_handoff.py export --session ${SESSION} --data-dir ${REMOTE_DATA_DIR} --out ${REMOTE_PACK}" || exit 1

echo "═══ 2/4 transfer（server → host） ═══"
"${SCP[@]}" "${USER_}@${SERVER}:${REMOTE_PACK}" "$PACK" || exit 1
LOCAL_MD5="$(md5sum "$PACK" | cut -d' ' -f1)"
REMOTE_MD5="$("${SSH[@]}" "md5sum ${REMOTE_PACK} | cut -d' ' -f1")"
if [ "$LOCAL_MD5" != "$REMOTE_MD5" ]; then
  echo "❌ md5 不一致：local=$LOCAL_MD5 remote=$REMOTE_MD5" >&2; exit 1
fi
echo "  md5 一致：$LOCAL_MD5"

echo "═══ 3/4 差异预览 ═══"
# 本机当前状态 vs 服务器版本
python3 - "$SELF_DIR" "$SESSION" "$PACK" "$LOCAL_DATA_DIR" <<'PY'
import json, subprocess, sys, os
self_dir, session, pack, local_data_dir = sys.argv[1:5]
pack = json.load(open(pack))
remote_msg = pack["messageCount"]
remote_rev = pack["activeRevision"]
local = subprocess.run(
    [sys.executable, os.path.join(self_dir, "mcode_handoff.py"), "verify",
     "--session", session, "--data-dir", local_data_dir],
    capture_output=True, text=True,
)
try:
    d = json.loads(local.stdout)
    local_msg = d.get("message_count")
    local_rev = d.get("active_revision")
except Exception:
    local_msg = local_rev = None
print(f"  本机   : msg={local_msg} rev={(local_rev or '-')[:24]}")
print(f"  服务器 : msg={remote_msg} rev={remote_rev[:24]}")
if local_msg is None:
    print("  ⇒ 本机没有该会话，直接导入即可")
elif local_msg == remote_msg and local_rev == remote_rev:
    print("  ⇒ 两侧完全一致，回拉是幂等的")
elif remote_msg > local_msg:
    print(f"  ⇒ 服务器更新（多 {remote_msg - local_msg} 条），回拉会推进本机")
else:
    print(f"  ⇒ ⚠️ 本机更新（多 {local_msg - remote_msg} 条），回拉会**覆盖掉本机较新的轮次**")
PY

if [ "$FORCE" != "1" ]; then
  echo "  （未加 --force，先不覆盖本机。要真正回拉请加 --force）"
  exit 0
fi

echo "═══ 4/4 import（本机） ═══"
python3 "$SELF_DIR/mcode_handoff.py" import --pack "$PACK" \
  --data-dir "$LOCAL_DATA_DIR" --workspace "$WORKSPACE" || exit 1

python3 "$SELF_DIR/mcode_handoff.py" verify --session "$SESSION" --data-dir "$LOCAL_DATA_DIR" || exit 1

if [ "$DO_EXEC" = "1" ]; then
  echo "═══ 附加：本机实际加载测试 ═══"
  cd "$WORKSPACE" || exit 1
  mcode exec --session "$SESSION" --output-format json --max-steps 1 "$EXEC_PROMPT" | tail -1
fi

echo "✅ 回拉完成：session ${SESSION} ← ${USER_}@${SERVER}"
