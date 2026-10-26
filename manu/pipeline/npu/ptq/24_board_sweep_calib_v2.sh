#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# 24_board_sweep_calib_v2.sh -- board threshold sweep for the CALIBRATION-SET
# v2 (set_3) int8 model, against the int8 baseline.
#
# WHY THIS EXISTS
# ---------------
# The calibration set is the one PTQ variable that was never closed:
#   set_A / set_B / set_C / set_D differ by only 0.048 F1 on the board, so
#   the "IR percentile stratification" dimension looked useless -- BUT that
#   was measured with the A/B/C/D family, all built from the SAME design.
#   The v2 candidate family (11_make_calib_variants.py) was never taken to
#   the board at all; set_3 is the best set on the PC simulator.
#
# On 2026-10-23 the pure-PTQ knob sweep measured (PC simulator, 100
# sensitive frames, same code path, same batch):
#   base   (set_A)  InputOperator INT8  cos_hm 0.97309  F1 0.7299
#   v2set3 (set_3)  InputOperator INT8  cos_hm 0.97339  F1 0.7800
# The calibration set is the ONLY surviving knob with a visible signal, and
# unlike hybrid it does NOT touch the graph input dtype (INT8 stays INT8,
# inputs_set 0.34 ms).
#
# ARMS
#   p0_off_3_normal          the calibration-set v2 (set_3) model
#   p0_off_A_normal          the ORIGINAL int8 baseline -- the exact artefact
#                            the 60,610-frame three-track evaluation was run
#                            with (md5 cedaeb74d066a64ba2c82003993defb1), so
#                            the numbers stay continuous with that run
#   p0_off_A_normal_rebuild  the SAME baseline rebuilt in the same batch as
#                            the v2 model.  It reproduces the original's
#                            cos_hm / mse / F1 on the simulator but NOT its
#                            .rknn bytes (md5 75d84fcd... vs cedaeb74...):
#                            the toolkit build is not byte-reproducible.  This
#                            third arm is the controlled A/B that isolates the
#                            calibration set from build non-determinism.
#
# Run (server, conda env `uav`):
#   source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
#   cd /tmp/pycharm_project_10ae9e2e
#   setsid nohup bash manu/pipeline/npu/ptq/24_board_sweep_calib_v2.sh \
#       > /mnt/data/siping/ptq_work/logs/calib_v2_sweep.out 2>&1 < /dev/null &
set -uo pipefail

source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd /tmp/pycharm_project_10ae9e2e

W=/mnt/data/siping/ptq_work
OUT=$W/calib_v2_sweep
mkdir -p "$OUT"

# The board runs ONE driver at a time (at most two RKNN contexts; a third
# one soft-resets the NPU).  Wait for any other board sweep to finish
# before taking the board.
echo "### waiting for the board to be free ###"
for _ in $(seq 1 240); do
  n=$(ps -eo args | grep -c "[1]9_board_threshold_sweep")
  [ "$n" -eq 0 ] && break
  echo "  another board sweep is running ($n), waiting..."
  sleep 15
done
echo "### board is free ###"

MODELS=""
add_model () {
  local src="$1" dst="$2"
  if [ ! -f "$src" ]; then echo "[FATAL] missing $src"; exit 1; fi
  cp -f "$src" "$W/models/$dst"
  MODELS="$MODELS $W/models/$dst"
  echo "### $dst <- $src  md5=$(md5sum "$src" | cut -d' ' -f1)"
}
mkdir -p "$W/models"

add_model "$W/screen_v2b/p0_off_3_normal.rknn"        p0_off_3_normal.rknn
add_model "$W/screen_n/p0_off_A_normal.rknn"           p0_off_A_normal.rknn
add_model "$W/screen_base/p0_off_A_normal.rknn"        p0_off_A_normal_rebuild.rknn

# re-stage onto the board so both sides agree byte-for-byte
for m in $MODELS; do
  scp -o StrictHostKeyChecking=no "$m" ptq-board:/mnt/manu/ptq/models/ >/dev/null || exit 1
  echo "  staged $(basename "$m") md5=$(md5sum "$m" | cut -d' ' -f1)"
done

run_sweep () {
  local man=$1 tag=$2
  echo ""
  echo "############################################################"
  echo "### CALIB-V2 SWEEP $tag   manifest=$man"
  echo "############################################################"
  python -u manu/pipeline/npu/ptq/19_board_threshold_sweep.py \
    --manifest "$man" \
    --models $MODELS \
    --collect-th 0.05 \
    --thresholds 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70 \
    --topk 100 \
    --variant p0_off \
    --pt-device cuda:3 \
    --board-timeout 900 \
    --out "$OUT/$tag" 2>&1
  echo "[sweep] $tag rc=$?"
}

run_sweep "$W/fast/tier1_sensitive.json"      tier1_sensitive
run_sweep "$W/fast/001324_sub2000.json"       001324_sub2000
run_sweep "$W/fast/golden_pool_sub1200.json"  golden_pool_sub1200

echo ""
echo "### done -> $OUT ###"
