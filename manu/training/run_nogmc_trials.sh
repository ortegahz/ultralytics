#!/usr/bin/env bash
# no-GMC RC2 重训：4 卡并行启动器（natural 对照 / RC2 配置 / RC2 两个种子）
#
# 用法（服务器）：
#   bash manu/training/run_nogmc_trials.sh smoke    # 预检冒烟：1 trial，2 iters，跳过门禁
#   bash manu/training/run_nogmc_trials.sh launch   # 正式：4 卡并行跑 4 个 trial
#   bash manu/training/run_nogmc_trials.sh status   # 查看进度
#
# 前置：
#   1. no-GMC 特征集已建好（uav_gmc_median_nogmc / longquanshan_ft_gmc_median_nogmc）
#   2. 参照系包已测好：runs/nogmc_ref/reference_frame.json
#      （其中 rc1 需先用 natural 对照 trial 的 gate_report.json 填好）
set -euo pipefail

cd /tmp/pycharm_project_10ae9e2e
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
export PYTHONPATH=.

B=/mnt/data/siping/datasets/manu
LQ_DATA=$B/longquanshan_ft_gmc_median_nogmc/data.yaml
UAV_DATA=$B/uav_gmc_median_nogmc/data.yaml
REF_DIR=${REF_DIR:-runs/nogmc_ref}
TRIALS=$REF_DIR/trials
REF=$REF_DIR/reference_frame.json
BASE_CKPT=$REF_DIR/uav_cache_base.pkl
WEIGHTS=runs/optuna_p0_nas/trial_0474/weights/best.pt
MODE=${1:-status}

# ---- 共享超参（与 trial_0080 逐项一致；唯一变量是特征语义与参照系）----
COMMON=(
  --output-root "$TRIALS"
  --longquan-data "$LQ_DATA"
  --uav-data "$UAV_DATA"
  --weights "$WEIGHTS"
  --baseline-cache "$BASE_CKPT"
  --reference-frame "$REF"
  --epochs 3 --sched-epochs 5 --batch 32 --imgsz 640 --workers 8
  --gate-epochs last --gate-epoch 3
  --lr-scale 2.270182993957786
  --warmup-epochs 0.5877289830563912 --weight-decay 0.00026905793288047476
  --focal-beta 2.75 --offset-weight 0.40
  --mosaic 0.15 --scale 0.10 --translate 0.05
  --clip-max-norm 5.649530591182026 --freeze-bn 0
)

# trial 号 / GPU / mix / seed —— natural 对照与 RC2 的两种种子
#   90 = natural 对照（新 RC1 等价擂主的来源）
#   80 = RC2 = trial_0080 的配置（seed 0 + trial 80 ⇒ 采样 seed 80，与原 trial_0080 同）
#   91 / 92 = RC2 的另两个种子（多种子确认）
TRIALS_SPEC=(
  "90 0 natural 0"
  "80 1 w2.5 0"
  "91 2 w2.5 91"
  "92 3 w2.5 92"
)

mkdir -p "$TRIALS/logs"

case "$MODE" in
  smoke)
    echo "[SMOKE] 1 trial / 2 iters / 跳过门禁（只验证训练链路能跑通）"
    python -u manu/training/optuna_mix_ratio_search.py --worker \
      --trial-number 99 --gpu-id 0 "${COMMON[@]}" \
      --mix-spec w2.5 --seed 0 --max-iters 2 --skip-gate \
      2>&1 | tee "$TRIALS/logs/smoke.log"
    ;;

  launch)
    if [ ! -f "$REF" ]; then echo "[FAIL] 缺参照系包 $REF（先跑 measure_reference_frame.py）"; exit 1; fi
    if ! grep -q '"rc1": {' "$REF" || python -c "
import json,sys
d=json.load(open('$REF'))
sys.exit(0 if d.get('rc1') else 1)"; then
      echo "[FAIL] 参照系包的 rc1 为空 —— 先用 natural 对照 trial 填："
      echo "       python manu/training/measure_reference_frame.py --out-dir $REF_DIR --fill-rc1-from <natural 的 gate_report.json>"
      exit 1
    fi
    echo "[LAUNCH] 4 卡并行启动：natural 对照 + RC2 + 2 种子"
    for spec in "${TRIALS_SPEC[@]}"; do
      read -r num gpu mix seed <<<"$spec"
      log="$TRIALS/logs/trial_$(printf '%04d' "$num").log"
      echo "  → trial_$(printf '%04d' "$num") GPU$gpu mix=$mix seed=$seed | $log"
      nohup python -u manu/training/optuna_mix_ratio_search.py --worker \
        --trial-number "$num" --gpu-id "$gpu" "${COMMON[@]}" \
        --mix-spec "$mix" --seed "$seed" \
        > "$log" 2>&1 &
      echo "     PID=$!"
    done
    echo "[LAUNCH] 全部已后台启动；用 '$0 status' 看进度"
    ;;

  status)
    for spec in "${TRIALS_SPEC[@]}"; do
      read -r num gpu mix seed <<<"$spec"
      log="$TRIALS/logs/trial_$(printf '%04d' "$num").log"
      echo "=== trial_$(printf '%04d' "$num") (GPU$gpu mix=$mix seed=$seed) ==="
      if [ -f "$log" ]; then tail -3 "$log"; else echo "  (日志未生成)"; fi
      res="$TRIALS/trials/trial_$(printf '%04d' "$num")/trial_result.json"
      if [ -f "$res" ]; then
        python -c "
import json;d=json.load(open('$res'))
print('  verdict:', d.get('gate_verdict'), '| U:', d.get('utility'), '| U(vs RC1):', d.get('utility_vs_rc1'))
print('  metrics:', d.get('metrics'))"
      fi
    done
    # ---- RC2 交接判据：RC2 三次（0080/0091/0092）的 U 中位数 ≥ +0.045 ----
    echo
    echo "=== RC2 交接判据（U 中位数 ≥ +0.045；对照 = natural 对照 trial_0090 的 U） ==="
    python - "$TRIALS" <<'PY'
import json, statistics, sys
from pathlib import Path
trials = Path(sys.argv[1])
def load(n):
    p = trials / "trials" / f"trial_{n:04d}" / "trial_result.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return {"u": d.get("utility"), "u_rc1": d.get("utility_vs_rc1"),
            "verdict": d.get("gate_verdict"), "m": d.get("metrics")}
rows = {n: load(n) for n in (90, 80, 91, 92)}
for n, tag in ((90, "natural 对照"), (80, "RC2 seed80"), (91, "RC2 seed91"), (92, "RC2 seed92")):
    r = rows[n]
    if not r:
        print(f"  trial_{n:04d} ({tag}): 未完成"); continue
    m = r["m"] or {}
    print(f"  trial_{n:04d} ({tag}): verdict={r['verdict']} U={r['u']} U(vs RC1)={r['u_rc1']} "
          f"T1={m.get('track1_f1')} T2={m.get('track2_f1')} T3={m.get('track3_f1')}")
us = [rows[n]["u"] for n in (80, 91, 92) if rows[n] and rows[n]["u"] is not None]
if len(us) == 3:
    med = statistics.median(us)
    print(f"  ▶ RC2 三次 U = {us} | 中位数 = {med:+.6f} | 门槛 +0.045 → "
          f"{'✅ 达标' if med >= 0.045 else '❌ 未达标'}")
elif us:
    print(f"  ▶ 已完成的 RC2 U = {us}（需 3 个才能算中位数）")
else:
    print("  ▶ 尚无完成的 RC2 trial")
PY
    ;;

  *)
    echo "用法: $0 {smoke|launch|status}"; exit 1 ;;
esac
