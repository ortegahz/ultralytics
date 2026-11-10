#!/bin/bash
# 纵横AI RC2 时空轨迹档 —— 把服务器产物收进本地仓库并提交（**在 host 上跑**）
#
# 服务器的 /tmp/pycharm_project_10ae9e2e 与本地仓库是两份独立 checkout（inode 不同），
# 服务器端生成的划分清单/复核底稿不会自动出现在本地 ⇒ 经 SSHFS 桥拷回后提交。
set -uo pipefail

LOCAL=/media/manu/1TB-Volume/workspace/ultralytics
BRIDGE=/home/manu/mnt/pycharm_project_10ae9e2e
SERVER="huangzhe@192.168.99.40"
PORT=32222
FINAL_LOG=/mnt/data/siping/runs/zongheng_rc2_tracked_finalize.log
SPLIT_NAME=zongheng_ft_split_tracked.json
REVIEW_NAME=zongheng-tk-tracked.md

echo "[COLLECT] $(date '+%F %T') waiting for the tracked finalizer to finish..."
for i in $(seq 1 720); do
  ssh -p "$PORT" -o BatchMode=yes -o ConnectTimeout=10 "$SERVER" \
    "grep -q '\[FINALIZE\] done' '$FINAL_LOG'" 2>/dev/null && break
  sleep 30
done
sleep 10
echo "--- finalizer tail ---"
ssh -p "$PORT" -o BatchMode=yes "$SERVER" "tail -30 '$FINAL_LOG'" 2>/dev/null

echo
echo "[COLLECT] copying $SPLIT_NAME + $REVIEW_NAME from the SSHFS bridge"
mkdir -p "$LOCAL/manu/data" "$LOCAL/manu/docs/dataset"
COPIED=0
if [ -f "$BRIDGE/manu/data/$SPLIT_NAME" ]; then
  cp "$BRIDGE/manu/data/$SPLIT_NAME" "$LOCAL/manu/data/" && COPIED=1
fi
if [ -f "$BRIDGE/manu/docs/dataset/$REVIEW_NAME" ]; then
  cp "$BRIDGE/manu/docs/dataset/$REVIEW_NAME" "$LOCAL/manu/docs/dataset/" && COPIED=1
fi
echo "[COLLECT] copied=$COPIED"

echo
echo "[COLLECT] committing into the local repo"
cd "$LOCAL" || exit 2
git add "manu/data/$SPLIT_NAME" "manu/docs/dataset/$REVIEW_NAME" 2>/dev/null
if git diff --cached --quiet; then
  echo "[COLLECT] nothing new to commit"
else
  git -c user.name=manu -c user.email=manu@local commit -q -m "feat(manu): 纵横AI RC2 时空轨迹档的序列划分与人工复核底稿

由服务器端收尾脚本生成、经 SSHFS 桥拉回本地（58 条 mp4 全量 RC2 + 龙泉山时空轨迹后处理）：
- manu/data/$SPLIT_NAME：val / train / ignore 三桶，字段沿用 longquan_ft_split.json
- manu/docs/dataset/$REVIEW_NAME：人工复核底稿（每条预填跟踪覆盖率与平均分）" \
    && echo "[COLLECT] committed" || echo "[COLLECT] commit failed"
fi

echo
echo "[COLLECT] ==== FINAL STATE ===="
echo " 划分   : $LOCAL/manu/data/$SPLIT_NAME"
echo " 复核稿 : $LOCAL/manu/docs/dataset/$REVIEW_NAME"
cd "$LOCAL" && git log --oneline -1
echo "[COLLECT] done $(date '+%F %T')"
