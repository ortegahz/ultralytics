#!/usr/bin/env bash
# ============================================================
# 一键：在【能 SSH 到服务器的机器】上执行
#   1) 打包本地代码  2) scp 上传  3) 服务器解包+校验  4) setsid 拉起夜间作业
# 用法: bash manu/tracking/lookahead/server_overnight/push_and_start.sh
# ============================================================
set -euo pipefail

SRC_REPO="${SRC_REPO:-/media/manu/1TB-Volume/workspace/ultralytics}"
SSH_HOST="${SSH_HOST:-huangzhe@192.168.99.40}"
REMOTE_ROOT="${REMOTE_ROOT:-/tmp/pycharm_project_10ae9e2e}"
SSH_PORT="${SSH_PORT:-32222}"   # 服务器 sshd 在 32222(历史记录确认), 22 是关闭的
SSH_OPTS=(-p "$SSH_PORT" -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

BUNDLE_REL="manu/tracking/lookahead/server_overnight"
REMOTE_LOG="$REMOTE_ROOT/$BUNDLE_REL/overnight.log"

echo "== [1/4] 打包本地代码 =="
TMP_TAR="$(mktemp /tmp/lookahead_bundle.XXXXXX.tar.gz)"
tar -C "$SRC_REPO" \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    -czf "$TMP_TAR" \
    manu/tracking \
    manu/memory/lookahead_extrapolator_20261102.md \
    manu/memory/rules.md
ls -lh "$TMP_TAR"

echo "== [2/4] 上传到 $SSH_HOST =="
scp "${SSH_OPTS[@]}" "$TMP_TAR" "$SSH_HOST:/tmp/lookahead_bundle.tar.gz"

echo "== [3/4] 服务器端解包 + 校验 =="
ssh "${SSH_OPTS[@]}" "$SSH_HOST" "set -e
cd '$REMOTE_ROOT'
tar -xzf /tmp/lookahead_bundle.tar.gz
echo '--- 外挂代码 ---'; ls manu/tracking/lookahead/
echo '--- 作业包 ---';   ls manu/tracking/lookahead/server_overnight/
echo '--- 基线 md5 (本地应为 3361e7b70e76d3a92346fd8fd9879dad) ---'
md5sum manu/evaluation/system_sota_longquan.py"

echo "== [4/4] 启动服务器夜间作业 (setsid nohup) =="
ssh "${SSH_OPTS[@]}" "$SSH_HOST" "cd '$REMOTE_ROOT' && setsid nohup bash $BUNDLE_REL/start_on_server.sh > $REMOTE_LOG 2>&1 < /dev/null & sleep 30; echo '--- overnight.log ---'; tail -50 '$REMOTE_LOG'"

echo "== 完成 =="
