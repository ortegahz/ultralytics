#!/bin/bash
# ==============================================================================
# 看门狗：等「NPU≤21ms」周末轮收口 ⇒ 自动跑验收 ⇒ 落盘最终交付物
#
# 为什么需要它
# ------------
# * 周末轮从 13:00 跑到 ~04:40（15.6 h），**宿主今天 18:30 关机**；
# * 验收（速度 + 三轨 + 相对 cs_t0043 的 Δ）必须**在无人值守时也能产出**；
# * ⇒ 本脚本在服务器上 setsid 常驻，轮询 dispatcher PID，退出即收割。
#
# 产出：
#   /mnt/data/siping/runs/npu20ms/acceptance.json          （机读）
#   /mnt/data/siping/runs/npu20ms/ACCEPTANCE.md           （人读总表）
#   /mnt/data/siping/runs/npu20ms/logs/watchdog.log        （本脚本日志）
#
# 用法（服务器）：
#   setsid nohup bash manu/training/npu20ms_watchdog.sh > /dev/null 2>&1 &
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
OUT_ROOT="${OUT_ROOT:-/mnt/data/siping/runs/npu20ms}"
PID_FILE="$OUT_ROOT/dispatcher.pid"
LOG="$OUT_ROOT/logs/watchdog.log"
SPEED_TABLE="${SPEED_TABLE:-$PROJECT_ROOT/manu/pipeline/npu/struct_speed_table.json}"
POLL_SEC="${POLL_SEC:-120}"
DEADLINE_EPOCH="${DEADLINE_EPOCH:-0}"   # >0 ⇒ 到点强制收割（即使 dispatcher 还活着）

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav || exit 1
mkdir -p "$OUT_ROOT/logs"

log() { echo "[WATCHDOG $(date '+%F %T')] $*" | tee -a "$LOG"; }

PID=""
for i in $(seq 1 30); do
  [ -f "$PID_FILE" ] && PID=$(cat "$PID_FILE" 2>/dev/null || echo "") && [ -n "$PID" ] && break
  sleep 20
done
log "周末轮 dispatcher pid = ${PID:-<无>}"

if [ -n "$PID" ]; then
  while kill -0 "$PID" 2>/dev/null; do
    if [ "$DEADLINE_EPOCH" -gt 0 ] && [ "$(date +%s)" -gt "$DEADLINE_EPOCH" ]; then
      log "超过 deadline，强制进入收割"
      break
    fi
    sleep "$POLL_SEC"
  done
fi
log "dispatcher 已退出（或超时）⇒ 开始收割"

python manu/training/npu20ms_acceptance.py \
  --root "$OUT_ROOT" \
  --speed-table "$SPEED_TABLE" \
  --study-name "${STUDY_NAME:-mix_ratio_search_npu20ms_v2}" \
  --limit-ms "${LIMIT_MS:-21.0}" \
  --t3-drop-max-pp "${T3_DROP_MAX_PP:-1.0}" \
  --t1-drop-max-pp "${T1_DROP_MAX_PP:-0.5}" \
  --t2-fp-max "${T2_FP_MAX:-10}" 2>&1 | tee -a "$LOG"

# ---- 人读版总表 ----
if [ -f "$OUT_ROOT/acceptance.json" ]; then
  python - << 'PYEOF' 2>&1 | tee -a "$LOG"
# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
R = Path(os.environ.get("OUT_ROOT", "/mnt/data/siping/runs/npu20ms"))
d = json.loads((R / "acceptance.json").read_text())
base = d["baseline"]
rows = d["results"]
rows_ok = [r for r in rows if str(r.get("verdict", "")).startswith("🎯")]
rows_sp = [r for r in rows if r.get("ms") is not None]
out = ["# 「NPU ≤ 21 ms 且精度下降微小」最终验收表", "",
       "基线 = cs_t0043（3 粒子同配方 Model Soup）："
       f"T1 {base['t1_f1']:.6f} | T2 {base['t2_f1']:.6f} (FP {base['t2_fp']}) | T3 {base['t3_f1']:.6f}", "",
       "| 臂 | NPU ms | T1 | ΔT1(pp) | T2 | ΔFP_T2 | T3 | ΔT3(pp) | T3FP | 判定 |",
       "|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|"]
def f(v, n=6):
    return ("%.*f" % (n, v)) if isinstance(v, float) else "-"
for r in sorted(rows, key=lambda x: (x.get("ms") is None, x.get("ms") or 0)):
    out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        r["trial"], f(r.get("ms"), 2), f(r.get("t1")), f(r.get("d1"), 3), f(r.get("t2")),
        ("%+d" % r["dFP2"]) if r.get("dFP2") is not None else "-",
        f(r.get("t3")), f(r.get("d3"), 3), r.get("t3fp") if r.get("t3fp") is not None else "-",
        r.get("verdict", "-")))
out.append("")
if rows_ok:
    out.append("## 🏆 达标候选（按 T3 降序）")
    for r in sorted(rows_ok, key=lambda x: -(x.get("t3") or 0)):
        out.append("- `%s` NPU %.2f ms | T1 %.6f | T2 %.6f | T3 %.6f" % (
            r["trial"], r["ms"], r.get("t1") or 0, r.get("t2") or 0, r.get("t3") or 0))
else:
    out.append("## （暂无同时满足「NPU ≤ 21 ms ∧ ΔT3 ≥ −1.0 pp ∧ ΔT1 ≥ −0.5 pp ∧ ΔFP_T2 ≤ +10」的候选）")
    if rows_sp:
        out.append("最快的几条（速度达标）：")
        for r in sorted([x for x in rows_sp if x.get("ms")], key=lambda x: x["ms"])[:8]:
            out.append("- `%s` NPU %.2f ms | ΔT1 %s pp | ΔT3 %s pp" % (
                r["trial"], r["ms"], f(r.get("d1"), 3), f(r.get("d3"), 3)))
(R / "ACCEPTANCE.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
PYEOF
  log "验收表 -> $OUT_ROOT/ACCEPTANCE.md"
fi
log "看门狗完成"
