#!/usr/bin/env bash
# =============================================================================
# mcode 会话转移 —— host → server 一条命令端到端
#
#   bash mcode_handoff_to_server.sh --session <id> --server <host> [--port N] \
#        [--user <u>] [--workspace <目标机上的项目目录>] [--data-dir <目标机 dataDir>]
#
# 默认值：
#   --server    192.168.99.40
#   --port      32222
#   --user      huangzhe
#   --workspace /tmp/pycharm_project_10ae9e2e
#   --data-dir  ~/.minimax（目标机）
#
# 流程：export(本机) → scp(传输) → import(目标机) → verify(目标机)
#
# ⚠️ 目标机上必须有 node（指纹计算用）：
#    mcode 自带 node 在 ~/.minimax-code/runtime/node-*/bin/node，
#    本脚本会自动把它加进 PATH（服务器上系统 node 常常不存在）。
# =============================================================================
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

SESSION=""
SERVER="192.168.99.40"
PORT="32222"
USER_="huangzhe"
WORKSPACE="/tmp/pycharm_project_10ae9e2e"
REMOTE_DATA_DIR="~/.minimax"
DO_EXEC="0"
EXEC_PROMPT="reply OK only"

while [ $# -gt 0 ]; do
  case "$1" in
    --session)    SESSION="$2"; shift 2 ;;
    --server)     SERVER="$2"; shift 2 ;;
    --port)       PORT="$2"; shift 2 ;;
    --user)       USER_="$2"; shift 2 ;;
    --workspace)  WORKSPACE="$2"; shift 2 ;;
    --data-dir)   REMOTE_DATA_DIR="$2"; shift 2 ;;
    --exec)       DO_EXEC="1"; shift ;;
    --prompt)     EXEC_PROMPT="$2"; shift 2 ;;
    -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

[ -n "$SESSION" ] || { echo "❌ 缺少 --session" >&2; exit 2; }

PACK="$(mktemp -t mcode_handoff_pack_XXXXXX.json)"
REMOTE_PACK="/tmp/mcode_handoff_pack_$$.json"
cleanup() { rm -f "$PACK"; }
trap cleanup EXIT

SSH=(ssh -p "$PORT" -o BatchMode=yes -o ConnectTimeout=15 "${USER_}@${SERVER}")
SCP=(scp -P "$PORT" -o BatchMode=yes -o ConnectTimeout=15)

# 目标机上找 node（mcode 自带；服务器上系统 node 常常不存在）。
# ⚠️ `export PATH=$HOME/.../node-v*/bin:$PATH` 不做 glob 展开 ⇒ 必须先算出绝对路径。
REMOTE_NODE_BIN="$("${SSH[@]}" 'ls -d $HOME/.minimax-code/runtime/node-v*/bin 2>/dev/null | head -1')"
if [ -z "$REMOTE_NODE_BIN" ]; then
  echo "❌ 目标机上找不到 node（~/.minimax-code/runtime/node-v*/bin）" >&2
  echo "   指纹计算依赖 node，请先在目标机安装或修好 mcode runtime。" >&2
  exit 1
fi
echo "目标机 node：$REMOTE_NODE_BIN"

echo "═══ 1/4 export（本机） ═══"
python3 "$SELF_DIR/mcode_handoff.py" export --session "$SESSION" --out "$PACK" || exit 1

echo "═══ 2/4 transfer ═══"
"${SCP[@]}" "$PACK" "${USER_}@${SERVER}:${REMOTE_PACK}" || exit 1
LOCAL_MD5="$(md5sum "$PACK" | cut -d' ' -f1)"
REMOTE_MD5="$("${SSH[@]}" "md5sum ${REMOTE_PACK} | cut -d' ' -f1")"
if [ "$LOCAL_MD5" != "$REMOTE_MD5" ]; then
  echo "❌ md5 不一致：local=$LOCAL_MD5 remote=$REMOTE_MD5" >&2
  exit 1
fi
echo "  md5 一致：$LOCAL_MD5"

echo "═══ 3/4 import（目标机） ═══"
# 先把工具同步到目标机（指纹计算需要 node）
"${SCP[@]}" "$SELF_DIR/mcode_handoff.py" "$SELF_DIR/mcode_session_tool.mjs" \
  "${USER_}@${SERVER}:/tmp/mcode_handoff_tools/" || exit 1

"${SSH[@]}" "export PATH=${REMOTE_NODE_BIN}:\$PATH; \
  cd /tmp/mcode_handoff_tools && \
  python3 mcode_handoff.py import --pack ${REMOTE_PACK} --data-dir ${REMOTE_DATA_DIR} --workspace ${WORKSPACE}" || exit 1

echo "═══ 4/4 verify（目标机） ═══"
"${SSH[@]}" "export PATH=${REMOTE_NODE_BIN}:\$PATH; \
  cd /tmp/mcode_handoff_tools && \
  python3 mcode_handoff.py verify --session ${SESSION} --data-dir ${REMOTE_DATA_DIR}" || exit 1

if [ "$DO_EXEC" = "1" ]; then
  echo "═══ 附加：目标机实际加载测试 ═══"
  "${SSH[@]}" "cd ${WORKSPACE} && ~/.minimax-code/bin/mcode exec --session ${SESSION} \
     --output-format json --max-steps 1 '${EXEC_PROMPT}'" | tail -1
fi

echo "✅ 交接完成：session ${SESSION} → ${USER_}@${SERVER}:${WORKSPACE}"
