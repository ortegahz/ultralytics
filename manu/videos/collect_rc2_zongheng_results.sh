#!/bin/bash
# 纵横AI RC2 OSD —— 渲染完成后：把划分产物收进**本地仓库**并提交（**在 host 上跑**）
#
# 为什么必须在 host 上做这一步：
#   服务器的 /tmp/pycharm_project_10ae9e2e 与本地仓库 /media/manu/1TB-Volume/workspace/ultralytics
#   是**两份独立 checkout**（inode 不同）⇒ 服务器端生成的 split / 复核底稿不会自动出现在本地仓库。
#   中间那层 SSHFS（host 上的 /home/manu/mnt/pycharm_project_10ae9e2e ↔ 服务器同一路径）才是桥。
#   ⇒ 收尾链路 = 服务器写 → SSHFS 桥 → host 拷进本地仓库 → 提交。
set -uo pipefail

SERVER="huangzhe@192.168.99.40"
PORT=32222
LOCAL=/media/manu/1TB-Volume/workspace/ultralytics
BRIDGE=/home/manu/mnt/pycharm_project_10ae9e2e
OUT=/mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_20261109
FINAL_LOG=/mnt/data/siping/runs/zongheng_rc2_osd_finalize.log

echo "[COLLECT] $(date '+%F %T') waiting for the render to finish..."
for i in $(seq 1 720); do
  n=$(ssh -p "$PORT" -o BatchMode=yes -o ConnectTimeout=10 "$SERVER" \
      'pgrep -cf "render_rc2_center_osd_video.py --input-root" || true' 2>/dev/null || echo 1)
  [ "${n:-1}" -eq 0 ] && break
  sleep 30
done
sleep 15

echo "[COLLECT] server state:"
ssh -p "$PORT" -o BatchMode=yes "$SERVER" "
D='$OUT'
echo \"  mp4          = \$(ls -1 \"\$D\"/*.mp4 2>/dev/null | wc -l) / 58\"
echo \"  frames jsonl = \$(ls -1 \"\$D/frames\"/*.jsonl 2>/dev/null | wc -l) / 58\"
echo \"  prelabels    = \$(ls -1 /mnt/data/siping/datasets/manu/纵横AI数据采集_preannot_rc2 2>/dev/null | wc -l) 序列\"
" 2>/dev/null

echo "[COLLECT] waiting for the finalizer to finish..."
for i in $(seq 1 120); do
  ssh -p "$PORT" -o BatchMode=yes "$SERVER" "grep -q '\[FINALIZE\] done' '$FINAL_LOG'" 2>/dev/null && break
  sleep 30
done
echo "--- finalizer tail ---"
ssh -p "$PORT" -o BatchMode=yes "$SERVER" "tail -22 '$FINAL_LOG'" 2>/dev/null

echo
echo "[COLLECT] copying the split + review sheet from the SSHFS bridge into the local checkout"
mkdir -p "$LOCAL/manu/data" "$LOCAL/manu/docs/dataset"
COPIED=0
if [ -f "$BRIDGE/manu/data/zongheng_ft_split.json" ]; then
  cp "$BRIDGE/manu/data/zongheng_ft_split.json" "$LOCAL/manu/data/" && COPIED=1
fi
if [ -f "$BRIDGE/manu/docs/dataset/zongheng-tk.md" ]; then
  cp "$BRIDGE/manu/docs/dataset/zongheng-tk.md" "$LOCAL/manu/docs/dataset/" && COPIED=1
fi
echo "[COLLECT] copied=$COPIED"

echo
echo "[COLLECT] committing into the local repo"
cd "$LOCAL" || exit 2
git add manu/data/zongheng_ft_split.json manu/docs/dataset/zongheng-tk.md 2>/dev/null
if git diff --cached --quiet; then
  echo "[COLLECT] nothing new to commit"
else
  git -c user.name=manu -c user.email=manu@local commit -q -m "feat(manu): 纵横AI 数据采集域 RC2 OSD 的序列划分与人工复核底稿

由服务器端收尾脚本生成、经 SSHFS 桥拉回本地（58 条 mp4 全量 RC2 单帧多尺度检测的结果）：
- manu/data/zongheng_ft_split.json：val / train / ignore 三桶，字段沿用 longquan_ft_split.json
- manu/docs/dataset/zongheng-tk.md：人工复核底稿（每条预填检测覆盖率与平均分）" \
    && echo "[COLLECT] committed" || echo "[COLLECT] commit failed"
fi

echo
echo "[COLLECT] ==== FINAL STATE ===="
echo " 视频目录   : $OUT"
echo " 每帧记录   : $OUT/frames"
echo " 标注候选   : /mnt/data/siping/datasets/manu/纵横AI数据采集_preannot_rc2"
echo " 本地划分   : $LOCAL/manu/data/zongheng_ft_split.json"
echo " 本地复核稿 : $LOCAL/manu/docs/dataset/zongheng-tk.md"
cd "$LOCAL" && git log --oneline -1
echo "[COLLECT] done $(date '+%F %T')"
