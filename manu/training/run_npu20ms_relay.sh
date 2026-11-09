#!/bin/bash
# ==============================================================================
# 接力脚本：等「当前轮（结构 B hc64）」收口 ⇒ 自动生成周末计划 ⇒ 启动「NPU≤21ms」周末轮
#
# 为什么需要它（2026-11-06）
# --------------------------
# * 当前轮（study mix_ratio_search_nogmc_struct4，输出根 /mnt/data/siping/runs/nogmc_struct4）
#   在跑结构 B（p2_fused, 160 输出）的 10 臂，**约 13:00 收口**；
# * 周末轮（study mix_ratio_search_npu20ms）要抢在**周一 09:00 deadline** 前跑完 ~24 臂；
# * **宿主今天 18:30 关机** ⇒ 接力必须**全自动**，不能依赖 Agent 在线。
#
# ⇒ 本脚本在服务器上 detached 常驻：轮询当前 dispatcher 的 PID ⇒ 它一退出就：
#   1) 合并今天测的结构→速度表（含探针 3/4/5）到 /mnt/data/siping/runs/npu20ms/
#   2) 用 build_npu20ms_plan.py 按 ≤21 ms 闸门 + 「头层级优先/同层宽度降序」生成计划
#   3) 启动 launch_npu20ms_search.sh（setsid 独立 session）
#   4) 把结果写进 relay.log / relay_done.json
#
# 用法（服务器，conda uav）：
#   setsid nohup bash manu/training/run_npu20ms_relay.sh > /mnt/data/siping/runs/npu20ms_relay.log 2>&1 &
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
CUR_OUT="${CUR_OUT:-/mnt/data/siping/runs/nogmc_struct4}"
CUR_PID_FILE="$CUR_OUT/dispatcher.pid"
NEW_OUT="${NEW_OUT:-/mnt/data/siping/runs/npu20ms}"
SPEED_TABLE="$NEW_OUT/struct_speed_table.json"
TABLE_DIRS="${TABLE_DIRS:-/media/manu/1TB-Volume/rknn_work/hm_speed}"
LIMIT_MS="${LIMIT_MS:-21.0}"
PER_STRUCT="${PER_STRUCT:-2}"
POLL_SEC="${POLL_SEC:-45}"
DEADLINE_EPOCH="${DEADLINE_EPOCH:-0}"   # >0 时到点强制停（unix 秒）

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav || exit 1
mkdir -p "$NEW_OUT"

log() { echo "[RELAY $(date '+%F %T')] $*"; }

CUR_PID=""
if [ -f "$CUR_PID_FILE" ]; then
  CUR_PID=$(cat "$CUR_PID_FILE" 2>/dev/null || echo "")
fi
log "当前轮 dispatcher pid = ${CUR_PID:-<无>}（输出根 $CUR_OUT）"

# ---- 1) 等当前轮退出 ----
if [ -n "$CUR_PID" ]; then
  while kill -0 "$CUR_PID" 2>/dev/null; do
    if [ "$DEADLINE_EPOCH" -gt 0 ] && [ "$(date +%s)" -gt "$DEADLINE_EPOCH" ]; then
      log "超过 deadline，强制进入接力"
      break
    fi
    sleep "$POLL_SEC"
  done
  log "当前轮已退出（或超时）"
else
  log "没有当前轮 PID ⇒ 直接接力"
fi

# ---- 2) 合并结构→速度表 ----
# ⚠️ **表必须在仓库里**（宿主 /media/manu/1TB-Volume 是宿主盘，服务器上没有这个挂载！）
#    ⇒ 宿主在关机前把 probes*/arms.json 合并好写进仓库并同步，relay 从仓库读。
log "合并结构→速度表（仓库内 manu/pipeline/npu/struct_speed_table.json + 可选的宿主 probes*）"
python3 - << 'PYEOF'
import json, glob, os
from pathlib import Path
arms = {}
# ① 仓库内的合并表（宿主在关机前写好的）
repo_tbl = Path("manu/pipeline/npu/struct_speed_table.json")
if repo_tbl.is_file():
    try:
        for a in json.loads(repo_tbl.read_text()).get("arms", []):
            arms[a["name"]] = a
        print(f"  仓库表 {repo_tbl}: {len(arms)} 个结构")
    except Exception as e:
        print("  仓库表读取失败", e)
# ② 若宿主盘可达（同一台机器上跑时），再并入更新的 probes*/
for d in sorted(glob.glob("/media/manu/1TB-Volume/rknn_work/hm_speed/probes*")):
    p = Path(d) / "arms.json"
    if not p.is_file():
        continue
    try:
        for a in json.loads(p.read_text()):
            arms[a["name"]] = a
    except Exception as e:
        print("  skip", p, e)
allarms = list(arms.values())
pts = [(a["cycles"]["npu_cycles"]/1e6, a["cycles"]["rw_mb"], a["ms"])
       for a in allarms if a.get("ms") is not None]
def lstsq3(pts):
    A=[[1.0,p[0],p[1]] for p in pts]; y=[p[2] for p in pts]; n=3
    ATA=[[sum(A[k][i]*A[k][j] for k in range(len(A))) for j in range(n)] for i in range(n)]
    ATy=[sum(A[k][i]*y[k] for k in range(len(A))) for i in range(n)]
    M=[ATA[i][:]+[ATy[i]] for i in range(n)]
    for i in range(n):
        p=max(range(i,n),key=lambda r:abs(M[r][i])); M[i],M[p]=M[p],M[i]
        for r in range(n):
            if r!=i and M[i][i]!=0:
                f=M[r][i]/M[i][i]
                for c in range(i,n+1): M[r][c]-=f*M[i][c]
    return [M[i][n]/M[i][i] for i in range(n)]
pred=None
if len(pts)>=3:
    a,b,c=lstsq3(pts); pred={"a":a,"b":b,"c":c}
out=Path("/mnt/data/siping/runs/npu20ms/struct_speed_table.json")
if len(arms) < 3:
    print("  ⚠️ 结构表不足 3 个 ⇒ 速度闸门会失效（不会剪枝，但也不致命）")
out.write_text(json.dumps({"arms":allarms,"predictor":pred}, indent=2, ensure_ascii=False)+"\n")
print(f"  合并 {len(allarms)} 个结构（{len(pts)} 个有实测）-> {out}")
PYEOF

# ---- 3) 生成计划 ----
log "生成计划（闸门 ≤${LIMIT_MS} ms，每结构 ${PER_STRUCT} 个超参点）"
python manu/training/build_npu20ms_plan.py \
  --speed-table "$SPEED_TABLE" \
  --out manu/training/npu20ms_plan.json \
  --limit-ms "$LIMIT_MS" --per-struct "$PER_STRUCT" || { log "计划生成失败，退出"; exit 1; }
cp -f manu/training/npu20ms_plan.json "$NEW_OUT/npu20ms_plan.json"

# ---- 4) 启动周末轮 ----
log "启动「NPU≤21ms」周末轮"
OUT_ROOT="$NEW_OUT" \
SPEED_TABLE="$SPEED_TABLE" \
SPEED_LIMIT_MS="$LIMIT_MS" \
PLAN=manu/training/npu20ms_plan.json \
  bash manu/training/launch_npu20ms_search.sh
rc=$?
log "启动返回 rc=$rc"
echo "{\"relay_ok\": $([ $rc -eq 0 ] && echo true || echo false), \"ts\": $(date +%s)}" > "$NEW_OUT/relay_done.json"
exit $rc
