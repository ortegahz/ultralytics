#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# 22i_run_pc_sim_qat_arms.sh -- PC-simulator evaluation of the four QAT arms.
#
# The simulator is CPU-bound (~1.5 s/frame) and the box has 80 cores, so the
# four arms run as PARALLEL processes -- one build+simulate process each,
# because the simulator refuses a model that was loaded with load_rknn()
# (it has to be built in the SAME process).
#
# Run (server, conda env `rknn`):
#   source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate rknn
#   cd /tmp/pycharm_project_10ae9e2e
#   setsid nohup bash manu/pipeline/npu/ptq/22i_run_pc_sim_qat_arms.sh \
#       > /mnt/data/siping/ptq_work/logs/qat_pc_sim.out 2>&1 < /dev/null &
set -uo pipefail

source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate rknn
cd /tmp/pycharm_project_10ae9e2e

W=/mnt/data/siping/ptq_work
LOGDIR=$W/logs
OUT=$W/qat/trial_0080/pc_sim
ARMS="a2_qat_learned a1_qat_maxclip a0_qat_float a1b_base_maxclip"
mkdir -p "$OUT" "$W/qat/trial_0080/pc_sim_rknn"

run_manifest () {
  local man=$1 tag=$2
  local pids=()
  for arm in $ARMS; do
    echo "[launch] $arm / $tag"
    python -u manu/pipeline/npu/ptq/22h_pc_sim_eval_qat_arms.py \
      --arm "$arm" \
      --manifest "$man" \
      --out "$OUT/$tag" \
      --rknn-dir "$W/qat/trial_0080/pc_sim_rknn" \
      --thresholds 0.30,0.45,0.50,0.55,0.60,0.70 \
      --collect-th 0.05 \
      --topk 100 \
      > "$LOGDIR/pcsim_${arm}_${tag}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; echo "[done] pid=$p rc=$?"; done
}

run_manifest "$W/fast/tier1_sensitive.json"      tier1_sensitive
run_manifest "$W/fast/golden_pool_sub1200.json"   golden_pool_sub1200

echo ""
echo "=== PC-simulator summary ==="
python - << "PYEOF"
import glob, json
rows = []
for f in sorted(glob.glob("/mnt/data/siping/ptq_work/qat/trial_0080/pc_sim/*/*.json")):
    d = json.load(open(f))
    for th, c in d["curve"].items():
        rows.append((f.split("pc_sim/")[1].split("/")[0], d["arm"], th,
                     c["f1"], c["precision"], c["recall"], c["tp"], c["fp"], c["fn"]))
print(f"{'manifest':<22}{'arm':<18}{'th':>5}{'F1':>9}{'P':>9}{'R':>9}{'TP':>7}{'FP':>7}{'FN':>7}")
for r in sorted(rows):
    print(f"{r[0]:<22}{r[1]:<18}{r[2]:>5}{r[3]:>9.4f}{r[4]:>9.4f}{r[5]:>9.4f}{r[6]:>7}{r[7]:>7}{r[8]:>7}")
PYEOF
echo "[ALL DONE]"
