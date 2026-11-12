#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# 2026-11-12 夜搜启动脚本（host 18:00 关机 ⇒ 服务器自主跑到 2026-11-13 09:00）
#   ① 补回 R6 轮记录（并发写覆盖事故）+ 重算待复现队列
#   ② 启动调度器（--start-round 7，独立 SID）
#   ③ 写守护 state（code_md5 = 当前组合 md5，rounds = 已完成最大轮号）⇒ 整夜不触发热升级
#   ④ 启动守护 v2
#   ⑤ 核验：进程 PPID=1 / 独立 SID / 4 卡满载 / 日志有启动行
# 用法（服务器）：bash manu/pipeline/npu/launch_night_20261112.sh
# ══════════════════════════════════════════════════════════════════════════
set -u
REPO=/tmp/pycharm_project_10ae9e2e
W=/mnt/data/siping/runs/npu20ms/nano_search
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
DEADLINE="${1:-2026-11-13 09:00}"
START_ROUND="${2:-7}"
N_SEED="${3:-6}"
N_MUT="${4:-18}"
cd $REPO || exit 1
export PYTHONPATH=.

echo "════ [1/5] 补回 R6 轮记录 + 重算待复现队列 ════"
$PY manu/pipeline/npu/restore_r6_record.py || echo "  (restore 报错，继续)"

echo "════ [2/5] 台账自检 ════"
$PY - <<'PYEOF'
import json
L = json.load(open("/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json"))
print("  arms =", len(L.get("arms") or {}))
print("  rounds =", [(r.get("round"), r.get("ts")) for r in (L.get("rounds") or [])])
print("  pending =", len((L.get("search_state") or {}).get("replication_pending") or []))
for t in ("9863", "12053"):
    a = (L.get("arms") or {}).get(t) or {}
    print(f"  {t}: picked.nano={(a.get('picked') or {}).get('nano')} official.T2={(a.get('official') or {}).get('t2')}")
PYEOF

echo "════ [3/5] 启动调度器（start-round=${START_ROUND}, deadline=${DEADLINE}）════"
setsid nohup bash manu/pipeline/npu/run_adaptive_night.sh "${DEADLINE}" "${N_SEED}" "${N_MUT}" "${START_ROUND}" \
    >> /dev/null 2>&1 < /dev/null &
sleep 12
PID=$(pgrep -f "adaptive_runner.py --deadline" | head -1)
if [ -z "${PID:-}" ]; then echo "  ✘ 调度器没起来"; exit 1; fi
echo "  ✔ 调度器 PID=${PID}"
ps -o pid,ppid,sid,pgid,etime,cmd -p "$PID" | tail -1

echo "════ [4/5] 写守护 state + 启动守护 v2 ════"
CUR=$(md5sum manu/pipeline/npu/adaptive_runner.py manu/pipeline/npu/nano_search.py | md5sum | cut -d" " -f1)
COMP=$($PY -c "
import json
L=json.load(open('$W/nano_search_ledger.json'))
rs=[int(r.get('round',0)) for r in (L.get('rounds') or []) if isinstance(r,dict)]
print(max(rs) if rs else 0)
")
$PY -c "
import json,sys
json.dump({'code_md5':sys.argv[2],'rounds':int(sys.argv[3])}, open('$W/watchdog_state.json','w'))
" x "${CUR}" "${COMP}"
echo "  state: code_md5=${CUR} rounds=${COMP}"
rm -f "$W/watchdog_state.json"   # 让守护自己写 state ⇒ 日志里有「启动」行可核验
setsid nohup bash manu/pipeline/npu/run_adaptive_watchdog.sh "${DEADLINE}" 2700 "${CUR}" \
    >> /dev/null 2>&1 < /dev/null &
sleep 3
WPID=$(pgrep -f "run_adaptive_watchdog.sh" | head -1)
echo "  ✔ 守护 PID=${WPID:-未找到}"
[ -n "${WPID:-}" ] && ps -o pid,ppid,sid,pgid,etime,cmd -p "$WPID" | tail -1

echo "════ [5/5] 核验 ════"
sleep 20
echo "--- 进程（PPID 应为 1，SID 各自独立）---"
ps -eo pid,ppid,sid,pgid,etime,cmd | grep -E "adaptive_runner.py --deadline|run_adaptive_watchdog.sh|optuna_mix_ratio" | grep -v grep
echo "--- GPU ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "--- watchdog.log 尾 ---"
tail -n 3 $W/watchdog.log
echo "--- adaptive.log 尾 ---"
tail -n 8 $W/adaptive.log
echo "════ 启动完成 ════"
