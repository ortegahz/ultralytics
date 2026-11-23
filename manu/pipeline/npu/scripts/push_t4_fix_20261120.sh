#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# 🟢 推送「板卡 T4 判据修复」到服务器（2026-11-20）
#
# 为什么需要本脚本：**沙箱出网被封**（ssh/scp 全部 `socket: Operation not permitted`）
#   ⇒ Codex 无法自行把代码送上服务器（铁律〇：本地优先 ⇒ 同步判据只能靠 **md5 双侧比对**）
#   ⇒ 交付形态 = **一条用户手工执行的命令**（铁律〇之三之二：Agent 不提交、不代跑重活）。
#
# 推什么（10 个文件，全部是本轮 T4 判据改动；相对仓库根）：
#   manu/pipeline/npu/t4_criterion.py            ★新增：T4 判据的唯一实现（纯函数）
#   manu/pipeline/npu/board_t4_watch.py           生产者：每次会话把基线臂放上板 + 写 pass_t4
#   manu/pipeline/npu/auto_promote_watch.py       消费者①：只信台账里已算好的 pass_t4
#   manu/pipeline/npu/promote_candidate.py        消费者②：只信台账里已算好的 board_pass_t4
#   manu/pipeline/npu/trackb_core.py              消费者③：Track-B 升轨链（原先只看绝对口径 ⇒ 断链）
#   manu/pipeline/npu/beat_report.py              报告侧：板端判定与升轨链同口径
#   manu/pipeline/npu/board_t4_rejudge.py         历史会话重判：改走同一判据
#   manu/pipeline/npu/run_morning_report.sh       早班报告 ⑧a-0：改读 pass_t4（不再自算绝对口径）
#   manu/pipeline/npu/tests/__init__.py           ★新增
#   manu/pipeline/npu/tests/test_t4_criterion.py  ★新增：33 条契约测试
#
# 做完三件事：
#   ① scp 打包上传 → 服务器解包进 `/tmp/pycharm_project_10ae9e2e`
#   ② **md5 双侧比对**（本地 vs 服务器）⇒ 不一致就报错退出（铁律〇 / AGENTS.md §5）
#   ③ **服务器侧跑契约测试**（纯 CPU、无 GPU、无 NPU、无训练、无板卡访问）
#      ⇒ 证明「落到服务器上的那份代码」本身是对的
#
# 用法（**在 host 的普通终端**跑，宿主机有网）：
#     bash manu/pipeline/npu/scripts/push_t4_fix_20261120.sh              # 推 + 校验 + 服务器自测
#     bash manu/pipeline/npu/scripts/push_t4_fix_20261120.sh --restart   # 额外：重启 3 个守望进程
#
# ⚠️ `--restart` 会 `pkill` 三个**守望**进程（board_t4_watch / auto_promote_watch /
#    baseline_guard）—— 它们由顶层守望 `day_supervisor_d1.sh` 自动重拉（用的是**新代码**）。
#    ⚠️ **不碰**任何训练 / 推理 / 评测进程（铁律一：不擅自启动重活）。
# ══════════════════════════════════════════════════════════════════════════
set -euo pipefail

SSH_HOST="huangzhe@192.168.99.40"
SSH_PORT="32222"
SSH_OPTS=(-F /dev/null -p "$SSH_PORT" -o ConnectTimeout=15 -o StrictHostKeyChecking=no)
REMOTE_ROOT="/tmp/pycharm_project_10ae9e2e"
PY_REMOTE="/home/huangzhe/anaconda3/envs/uav/bin/python"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
FILES=(
  "manu/pipeline/npu/t4_criterion.py"            # ★新增：判据唯一实现
  "manu/pipeline/npu/board_t4_watch.py"           # 生产者
  "manu/pipeline/npu/auto_promote_watch.py"       # 消费者①
  "manu/pipeline/npu/promote_candidate.py"        # 消费者②
  "manu/pipeline/npu/trackb_core.py"              # 消费者③（Track-B 升轨链）
  "manu/pipeline/npu/beat_report.py"              # 报告侧（口径统一）
  "manu/pipeline/npu/board_t4_rejudge.py"         # 历史会话重判（口径统一）
  "manu/pipeline/npu/run_morning_report.sh"       # 早班报告 ⑧a-0（改读 pass_t4）
  "manu/pipeline/npu/tests/__init__.py"
  "manu/pipeline/npu/tests/test_t4_criterion.py"  # ★新增：33 条契约测试
)
RESTART=0
[ "${1:-}" = "--restart" ] && RESTART=1

say(){ echo "[$(date "+%F %T")] $*"; }

say "=== ① 本地预检（文件都在吗）==="
for f in "${FILES[@]}"; do
  [ -f "$REPO_ROOT/$f" ] || { say "❌ 缺文件：$f"; exit 1; }
  say "   ok  $f  $(md5sum "$REPO_ROOT/$f" | cut -d' ' -f1)"
done

say "=== ② 打包 + 上传 ==="
TMP_TAR="$(mktemp /tmp/t4fix.XXXXXX.tar.gz)"
tar -C "$REPO_ROOT" -czf "$TMP_TAR" "${FILES[@]}"
scp "${SSH_OPTS[@]}" "$TMP_TAR" "$SSH_HOST:/tmp/t4fix.tar.gz"
say "   已上传 /tmp/t4fix.tar.gz（$(stat -c%s "$TMP_TAR") 字节）"

say "=== ③ 服务器解包 + md5 双侧比对 ==="
ssh "${SSH_OPTS[@]}" "$SSH_HOST" "set -e
  cd '$REMOTE_ROOT'
  tar -xzf /tmp/t4fix.tar.gz
  md5sum ${FILES[*]}
  rm -f /tmp/t4fix.tar.gz" > /tmp/t4fix_remote_md5.txt
LOCAL_MD5="$(cd "$REPO_ROOT" && md5sum "${FILES[@]}")"
REMOTE_MD5="$(grep -E '  manu/pipeline/npu/' /tmp/t4fix_remote_md5.txt || true)"
echo "--- 本地 ---"; echo "$LOCAL_MD5"
echo "--- 服务器 ---"; echo "$REMOTE_MD5"
if [ "$(echo "$LOCAL_MD5" | awk '{print $1}' | sort)" != "$(echo "$REMOTE_MD5" | awk '{print $1}' | sort)" ]; then
  say "❌ md5 双侧不一致 ⇒ **未落地**（铁律〇）"
  exit 1
fi
say "✅ md5 双侧一致（10/10）"

say "=== ④ 服务器侧契约测试（纯 CPU，无 GPU/NPU/训练/板卡）==="
ssh "${SSH_OPTS[@]}" "$SSH_HOST" "cd '$REMOTE_ROOT' && PYTHONPATH=. $PY_REMOTE -m unittest manu.pipeline.npu.tests.test_t4_criterion -v 2>&1 | tail -25"

if [ "$RESTART" = "1" ]; then
  say "=== ⑤ 重启三个守望进程（顶层守望会自动用新代码重拉）==="
  ssh "${SSH_OPTS[@]}" "$SSH_HOST" "pkill -f 'manu/pipeline/npu/board_t4_watch.py' || true
    pkill -f 'manu/pipeline/npu/auto_promote_watch.py' || true
    pkill -f 'baseline_guard.py' || true
    sleep 25
    ps -eo pid,etime,args | grep -E 'board_t4_watch|auto_promote_watch|baseline_guard' | grep -v grep || echo '（守望将被顶层守望拉起）'"
  say "   ⚠️ 若三个进程没被拉起，手工执行：ssh -p $SSH_PORT $SSH_HOST 'cd $REMOTE_ROOT && bash /mnt/data/siping/runs/npu20ms/nano_search/scripts/day_supervisor_d1.sh &'"
else
  say "=== ⑤ 未重启（加 --restart 可让三个守望进程用上新代码）==="
fi

rm -f "$TMP_TAR"
say "=== 完成 ==="
