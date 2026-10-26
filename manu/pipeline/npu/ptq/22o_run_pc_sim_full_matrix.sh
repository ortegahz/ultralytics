#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# 22o_run_pc_sim_full_matrix.sh -- the QAT accuracy matrix, SERVER SIDE ONLY.
#
# NO BOARD IS TOUCHED.  Every number below comes from the rknn-toolkit2 PC
# simulator (a CPU emulation of the RK3588 NPU), so this script can be re-run
# while the board is busy with somebody else's latency work.
#
# Phase 1  the PT float reference per manifest (conda env `uav`, GPU cuda:1)
# Phase 2  the five arms x three manifests through the simulator
#          (conda env `rknn`, CPU) -- all in parallel, one process per cell,
#          because the simulator refuses a model that was loaded with
#          load_rknn() and therefore has to build its own in-process.
#
# ARMS
#   a0_qat_float            QAT weights, no Q/DQ (fp16)
#   a1_qat_maxclip          QAT weights + max-calibrated clip
#   a2_qat_learned          QAT weights + LSQ-learned clip      (main candidate)
#   a1b_base_maxclip        BASE weights + max-calibrated clip
#   a2f_qatw_toolkit_int8   QAT weights under the toolkit's OWN full int8 recipe
#
# Run (server):
#   cd /tmp/pycharm_project_10ae9e2e
#   setsid nohup bash manu/pipeline/npu/ptq/22o_run_pc_sim_full_matrix.sh \
#       > /mnt/data/siping/ptq_work/logs/qat_pc_sim_matrix.out 2>&1 < /dev/null &
set -uo pipefail

REPO=/tmp/pycharm_project_10ae9e2e
W=/mnt/data/siping/ptq_work
LOGDIR=$W/logs
OUT=$W/qat/trial_0080/pc_sim
PTREF=$W/qat/trial_0080/pt_ref
mkdir -p "$OUT" "$PTREF" "$LOGDIR"

MANIFESTS="tier1_sensitive:fast/tier1_sensitive.json
golden_pool_sub1200:fast/golden_pool_sub1200.json
001324_sub2000:fast/001324_sub2000.json"

ARMS="a0_qat_float a1_qat_maxclip a2_qat_learned a1b_base_maxclip a2f_qatw_toolkit_int8"
THS="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70"

source /home/huangzhe/anaconda3/etc/profile.d/conda.sh

# 15 simulator processes share this box: every BLAS/OMP pool must be capped or
# one ONNX Runtime session grabs all 80 cores and starves the other cells.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

# ---------------------------------------------------------------- Phase 1 ----
echo "############ PHASE 1: PT float reference (env uav, cuda:1) ############"
conda activate uav
cd "$REPO"
for entry in $MANIFESTS; do
  tag=${entry%%:*}
  man=$W/${entry#*:}
  if [ -f "$PTREF/$tag.json" ]; then
    echo "--- PT reference $tag  (already done, skipping)"
    continue
  fi
  echo "--- PT reference $tag"
  python -u manu/pipeline/npu/ptq/22l_pt_float_reference.py \
    --manifest "$man" --device cuda:1 --variant p0_off --topk 100 \
    --thresholds "$THS" --out "$PTREF/$tag.json" \
    > "$LOGDIR/ptref_$tag.log" 2>&1
  echo "    rc=$?  $(tail -3 "$LOGDIR/ptref_$tag.log" | head -1)"
done

# ---------------------------------------------------------------- Phase 2 ----
echo "############ PHASE 2: PC simulator matrix (env rknn, CPU) ############"
conda activate rknn
cd "$REPO"

PIDS=""
for entry in $MANIFESTS; do
  tag=${entry%%:*}
  man=$W/${entry#*:}
  for arm in $ARMS; do
    echo "--- launch $arm / $tag"
    python -u manu/pipeline/npu/ptq/22h_pc_sim_eval_qat_arms.py \
      --arm "$arm" \
      --manifest "$man" \
      --out "$OUT/$tag" \
      --rknn-dir "$W/qat/trial_0080/pc_sim_rknn" \
      --thresholds "$THS" \
      --collect-th 0.05 \
      --topk 100 \
      --pt-ref "$PTREF/$tag.json" \
      --threads 2 \
      --verbose 1 \
      > "$LOGDIR/pcsim_${arm}_${tag}.log" 2>&1 &
    PIDS="$PIDS $!"
  done
done

echo "--- waiting for ${PIDS}"
for p in $PIDS; do wait "$p"; echo "    pid=$p rc=$?"; done

echo ""
echo "############ SUMMARY ############"
python - << "PYEOF"
import glob
import json
import os

rows = []
for f in sorted(glob.glob("/mnt/data/siping/ptq_work/qat/trial_0080/pc_sim/*/*.json")):
    d = json.load(open(f))
    tag = f.split("pc_sim/")[1].split("/")[0]
    for th, c in d.get("curve", {}).items():
        loss = d.get("loss_vs_pt_float", {}).get(th)
        rows.append((tag, d["arm"], th, c["f1"], c["precision"], c["recall"],
                     c["tp"], c["fp"], c["fn"], loss))
print(f"{'manifest':<22}{'arm':<26}{'th':>6}{'F1':>9}{'P':>9}{'R':>9}"
      f"{'TP':>7}{'FP':>7}{'FN':>7}{'loss':>9}")
for r in sorted(rows):
    ls = f"{r[9]:+.4f}" if r[9] is not None else "-"
    print(f"{r[0]:<22}{r[1]:<26}{r[2]:>6}{r[3]:>9.4f}{r[4]:>9.4f}{r[5]:>9.4f}"
          f"{r[6]:>7}{r[7]:>7}{r[8]:>7}{ls:>9}")
PYEOF
echo "[ALL DONE]"
