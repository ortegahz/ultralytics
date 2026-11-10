#!/usr/bin/env bash
# ==============================================================================
# 最终板卡会话：一次测完全部候选（含交付件对照）
#   ① 归档板上 arms/ 里已有的探针臂（保证会话只测本轮臂）
#   ② 上传本轮候选的真训练权重 rknn（真权重 ⇒ 时延与结构探针一致，但可复核）
#   ③ 把交付件 t31_silu 从 arms_archive/ 拷回 arms/ 作**同会话对照**
#   ④ 一次会话测完（LOOPS=30 WARMUP=10 SETTLE=5），写 board_ms.json
# ==============================================================================
set -uo pipefail
REPO=/tmp/pycharm_project_10ae9e2e
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
WORK=/mnt/data/siping/runs/npu20ms/route2_final_gate
BOARD_KEY="-i /home/huangzhe/.ssh/id_board -o ConnectTimeout=10 -o StrictHostKeyChecking=no"
BOARD=root@192.168.0.64
BOARD_DIR=/root/hm_speed

log() { echo "[$(date +%H:%M:%S)] $*"; }

# 候选：name:rknn_path:stride（stride = 320 输出 ⇒ 2；160 输出 ⇒ 4）
CAND=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --candidate) CAND+=("$2"); shift 2 ;;
    *) echo "未知参数 $1" >&2; exit 2 ;;
  esac
done
[[ ${#CAND[@]} -eq 0 ]] && { echo "至少一个 --candidate name=path[:stride]" >&2; exit 2; }

cd "$REPO" && export PYTHONPATH=.

# ① 归档旧臂（先归档再上传！）
"$PY" - <<'PYEOF'
import sys
sys.path.insert(0, "/tmp/pycharm_project_10ae9e2e")
from manu.pipeline.npu import board_speed_search as B
B.archive_board_arms()
PYEOF

# ② 上传候选臂 + 写 .stride 边车
for spec in "${CAND[@]}"; do
  name="${spec%%=*}"; rest="${spec#*=}"
  rknn="${rest%%:*}"; stride="${rest##*:}"
  [[ "$stride" == "$rknn" ]] && stride=2
  [ -f "$rknn" ] || { log "跳过（文件不存在）：$rknn"; continue; }
  scp $BOARD_KEY "$rknn" "$BOARD:$BOARD_DIR/arms/${name}_int8.rknn"
  ssh $BOARD_KEY "$BOARD" "printf $stride > $BOARD_DIR/arms/${name}_int8.stride"
  log "上传 ${name}_int8.rknn (stride=$stride, $(stat -c%s "$rknn")B)"
done

# ③ 交付件对照：从 arms_archive 拷回 arms/
ssh $BOARD_KEY "$BOARD" "cd $BOARD_DIR && for f in arms_archive/t31_silu_int8.rknn arms_archive/t31_silu_int8.stride; do [ -e \"\$f\" ] && cp -f \"\$f\" arms/; done; ls arms/ | tr '\n' ' '"

# ④ 一次会话测完
log "开始板卡会话（一次测完，控制臂 base_int8.rknn 必带）…"
"$PY" - "$WORK" <<'PYEOF'
import json, sys
sys.path.insert(0, "/tmp/pycharm_project_10ae9e2e")
from manu.pipeline.npu import board_speed_search as B
work = sys.argv[1]
r1, r2, r3, txt = B.measure_on_board(work)
json.dump(dict(run=r1, dequant=r2, full=r3), open(f"{work}/board_ms.json", "w"),
          ensure_ascii=False, indent=2)
print("@@BOARD@@")
for k, v in sorted(r1.items()):
    print(f"{k:44s} {v:9.3f} ms")
PYEOF
log "DONE ⇒ $WORK/board_ms.json"
