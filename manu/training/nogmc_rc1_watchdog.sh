#!/bin/bash
# ==============================================================================
# no-GMC 参照系 rc1 回填 —— **独立保险**（与 run_nogmc_pipeline.sh 的第③步并行）
#
# 为什么需要它
# ------------
# 无人值守管线 `run_nogmc_pipeline.sh` 的第③步负责把 natural 对照 trial_0000 的
# 门禁指标写进参照系包的 `rc1` 字段（后续所有 trial 的 U(vs 新擂主) 靠它）。
# 但那一步有一个已知的历史坑（2026-10-27 实测复现并修复）：
# `measure_reference_frame.py` 曾把 `--uav-data/--ft-root` 设为 argparse 必填，
# 而回填命令根本不传这俩 ⇒ argparse 退出码 2 ⇒ 回填静默失败，
# 且管线仍会打印「rc1 填充：完成」（RC1_DONE=1 是无条件赋值的）。
#
# 修好 argparse 后管线那步已能正常跑；但**19 小时的 GPU 产出不能只押在一处**。
# 本脚本作为**第二道保险**：
#   · 等 trial_0000 的门禁报告落地；
#   · **故意晚 180 秒**再动手 —— 让管线自己先填（避免两个进程并发写同一个
#     非原子 JSON 文件造成读半截的竞态）；
#   · 若管线已填好（rc1 非空）⇒ 只做校验并记录，不重复写；
#   · 若管线那步失败（rc1 仍为空）⇒ 由本脚本补填，并校验写入结果。
#
# 幂等：rc1 写入是同值覆盖，重复执行无副作用。
#
# 用法（服务器）：
#   setsid nohup bash manu/training/nogmc_rc1_watchdog.sh > runs/nogmc_rc1_watchdog.out 2>&1 &
# ==============================================================================
set -uo pipefail

cd /tmp/pycharm_project_10ae9e2e || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav || exit 1
export PYTHONPATH=.

REF_DIR="${REF_DIR:-runs/nogmc_ref}"
OUT_ROOT="${OUT_ROOT:-runs/optuna_nogmc}"
WLOG="${WLOG:-runs/nogmc_rc1_watchdog.log}"
GR="$OUT_ROOT/trials/trial_0000/gate/epoch3/gate_report.json"
RF="$REF_DIR/reference_frame.json"
DELAY="${DELAY:-180}"     # 让管线先填（规避并发写竞态）

mkdir -p runs
log() { echo "[$(date '+%F %T')] $*" | tee -a "$WLOG"; }

log "WATCHDOG 启动 | 等 natural 对照门禁报告：$GR"
while [ ! -f "$GR" ]; do
    # 调度器都没了还没报告 ⇒ 放弃（不是我们的锅，另行排查）
    if ! pgrep -f "optuna_mix_ratio_search.py" >/dev/null 2>&1; then
        log "❌ 调度器已退出且 $GR 不存在 ⇒ 放弃回填（rc1 留空）"
        exit 1
    fi
    sleep 30
done
log "✅ 门禁报告已出现，等 ${DELAY}s 让管线自己先填（规避并发写竞态）…"
sleep "$DELAY"

# ---- 管线是否已经填好了？----
already=$(python - <<'PY'
import json, pathlib
p = pathlib.Path("runs/nogmc_ref/reference_frame.json")
try:
    rc1 = (json.loads(p.read_text(encoding="utf-8")).get("rc1") or {})
    print("1" if rc1.get("track1_f1") is not None else "0")
except Exception:
    print("0")
PY
)

if [ "$already" = "1" ]; then
    log "✅ 管线已把 rc1 填好 ⇒ 本保险无需动作（只做校验）"
else
    log "⚠️ 管线那步没填上 rc1 ⇒ 由保险脚本补填"
    python manu/training/measure_reference_frame.py \
        --out-dir "$REF_DIR" --fill-rc1-from "$GR" 2>&1 | tee -a "$WLOG"
fi

# ---- 最终校验：rc1 必须非空且六个字段齐全 ----
python - <<'PY' 2>&1 | tee -a "$WLOG"
import json, pathlib, sys
p = pathlib.Path("runs/nogmc_ref/reference_frame.json")
d = json.loads(p.read_text(encoding="utf-8"))
rc1 = d.get("rc1") or {}
need = ("track1_f1","track1_fp","track2_f1","track2_fp","track3_f1","track3_fp")
miss = [k for k in need if rc1.get(k) is None]
if miss:
    print(f"[FAIL] rc1 仍不完整，缺字段：{miss}"); sys.exit(1)
print("[OK] rc1 校验通过：T1 F1=%s FP=%s | T2 F1=%s FP=%s | T3 F1=%s FP=%s" % tuple(rc1[k] for k in need))
PY
rc=$?
log "WATCHDOG 结束（校验退出码=$rc）"
exit $rc
