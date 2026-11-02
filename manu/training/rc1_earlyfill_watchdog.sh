#!/bin/bash
# ==============================================================================
# rc1 早填看门狗（no-GMC 周末实验，2026-10-30）
#
# 为什么需要它
# ------------
# 主流水线 `run_nogmc_weekend_pipeline.sh` 的 rc1 自动填充排在**看门狗循环之后**，
# 也就是要等到搜索停止（11-01 15:00）才填 ⇒ 整个搜索期间 trial 的
# `utility_vs_rc1` 都是空的。
#
# ⚠️ **为什么不直接改主脚本**：主脚本**正在运行中**，bash 是按字节偏移增量读取脚本的，
#    改动已读区域会导致它执行错位的字节 ⇒ 可能整条流水线静默跑飞。
#    ⇒ 这里用一个**独立的一次性 watcher**，逻辑与主脚本 step ④ 完全一致（幂等）。
#
# 做什么
# ------
# 等到 **natural 对照 trial**（trial_0000，mix_spec=natural）的门禁报告落地，
# 立刻把它写进参照系包的 rc1 字段 ⇒ 之后所有 trial 的 utility_vs_rc1 都有值。
#
# 安全性
# ------
# · 幂等：重复执行只是把同一个 rc1 再写一遍；
# · 不碰主脚本、不碰调度器、不重启任何东西；
# · 找不到就退出（最多等 12 h）。
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
OUT_ROOT="${OUT_ROOT:-/mnt/data/siping/runs/nogmc_weekend}"
REF_DIR="${REF_DIR:-/mnt/data/siping/runs/nogmc_ref_weekend}"
REF="$REF_DIR/reference_frame.json"
LOG="${LOG:-/mnt/data/siping/runs/rc1_earlyfill.log}"
MAX_WAIT_S="${MAX_WAIT_S:-43200}"   # 12 h

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav || exit 1
export PYTHONPATH=.

mkdir -p "$(dirname "$LOG")"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "RC1 早填看门狗启动 | out=$OUT_ROOT ref=$REF"

T0=$(date '+%s')
while :; do
  NOW=$(date '+%s')
  [ $((NOW - T0)) -gt "$MAX_WAIT_S" ] && { say "⏰ 超过 ${MAX_WAIT_S}s ⇒ 退出"; exit 0; }

  GR=$(python - "$OUT_ROOT" <<'PY' 2>/dev/null
import json, os, sys, glob
root = sys.argv[1]
for res in sorted(glob.glob(os.path.join(root, "trials", "trial_*", "trial_result.json"))):
    try:
        d = json.load(open(res, encoding="utf-8"))
    except Exception:
        continue
    if d.get("mix_spec") == "natural" and d.get("gate_dir"):
        gp = os.path.join(d["gate_dir"], "gate_report.json")
        if os.path.isfile(gp):
            print(gp)
            break
PY
)
  if [ -n "${GR:-}" ]; then
    say "找到 natural 对照门禁报告：$GR ⇒ 填 rc1"
    python manu/training/measure_reference_frame.py --out-dir "$REF_DIR" --fill-rc1-from "$GR" 2>&1 | tee -a "$LOG"
    say "rc1 填充完成，退出"
    python - "$REF" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print("  rc1 =", json.dumps(d.get("rc1"), ensure_ascii=False))
PY
    exit 0
  fi
  sleep 30
done
