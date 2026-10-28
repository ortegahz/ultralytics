#!/bin/bash
# ==============================================================================
# no-GMC 搜索收尾后的「实测比修复 + 汇总刷新」守护（服务器常驻，不依赖本机）
#
# 为什么需要它
# ------------
# 2026-10-27 抓到：`optuna_mix_ratio_search.py` 统计"这批有多少 LQ 样本"时**硬编码**了
# GMC 目录名，换成 no-GMC 目录后永远匹配不上 ⇒ `measured_lq` 恒为 0、
# `measured_ratio` 变成 ~56926 的垃圾值。
#
# **采样本身是对的**（WeightedRandomSampler 用样本数构造权重，不看路径），
# 受影响的只有上报字段 ⇒ 不影响训练 / 权重 / 门禁三轨指标 / U 判定。
#
# 代码已修（从下一个 trial 起新 worker 就用新逻辑），但**当时已在跑的 4 个种子 trial
# （trial_0000~0003）用的是旧进程**，它们的 measured_ratio 仍是垃圾值。
# ⇒ 用**确定性重放**（`replay_measured_ratio.py`，已在 GMC 侧 104/104 逐位验证）
# 把这些 trial 的实测比复原，并刷新 SUMMARY.md。
#
# 时序：本脚本等**管线 SUMMARY.md 落地**（= 管线全部跑完）才动手，
#       避免与管线的补采/汇总并发写同一个 CSV。
#
# 用法（服务器）：
#   setsid nohup bash manu/training/nogmc_repair_after.sh > runs/nogmc_repair_after.out 2>&1 &
# ==============================================================================
set -uo pipefail

cd /tmp/pycharm_project_10ae9e2e || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav || exit 1
export PYTHONPATH=.

OUT_ROOT="${OUT_ROOT:-runs/optuna_nogmc}"
REF="${REF:-runs/nogmc_ref/reference_frame.json}"
RLOG="${RLOG:-runs/nogmc_repair_after.log}"
SUMMARY="$OUT_ROOT/SUMMARY.md"

mkdir -p runs
log() { echo "[$(date '+%F %T')] $*" | tee -a "$RLOG"; }

log "REPAIR 守护启动 | 等管线产出 $SUMMARY …"
waited=0
while [ ! -f "$SUMMARY" ]; do
    # 兜底：调度器与管线都没了、且已等够 10 分钟 ⇒ 自己也把汇总跑一遍
    if ! pgrep -f "optuna_mix_ratio_search.py" >/dev/null 2>&1 \
       && ! pgrep -f "run_nogmc_pipeline.sh" >/dev/null 2>&1; then
        if [ "$waited" -ge 600 ]; then
            log "⚠️ 调度器与管线均已退出且等满 10 分钟仍无 SUMMARY ⇒ 自行补跑汇总"
            break
        fi
    fi
    sleep 30
    waited=$((waited + 30))
done
log "✅ 管线产出完成（等待 ${waited}s），开始修复实测比"

log "---- ① 确定性重放复原实测比 ----"
python manu/training/replay_measured_ratio.py --repair --out-root "$OUT_ROOT" --apply 2>&1 | tee -a "$RLOG"

log "---- ① 之二 用最终参照系包重判所有 trial 的门禁与 U ----"
python manu/training/rejudge_with_reference.py --out-root "$OUT_ROOT" --ref "$REF" --apply 2>&1 | tee -a "$RLOG"

log "---- ② 刷新汇总（把修正后的实测比与判定写进 SUMMARY.md）----"
python manu/training/summarize_nogmc_search.py --out-root "$OUT_ROOT" --ref "$REF" --top 15 2>&1 | tee -a "$RLOG"

log "---- ③ 校验 ----"
OUT_ROOT="$OUT_ROOT" python - <<'PY' 2>&1 | tee -a "$RLOG"
import csv, os, pathlib
p = pathlib.Path(os.environ["OUT_ROOT"]) / "optuna_summary.csv"
bad = []
with p.open(encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        try:
            v = float(r.get("measured_ratio") or 0)
        except Exception:
            v = 0.0
        if v > 1000:
            bad.append((r.get("trial"), v))
print(f"[CHECK] {p} 实测比仍异常的行：{len(bad)}" + (f" -> {bad[:10]}" if bad else " ✅"))
PY

log "✅ REPAIR 守护全部完成"
