#!/usr/bin/env bash
# =============================================================================
# 24_screening_round.sh -- one screening ROUND over (calib set x quant method).
#
# WHY THIS EXISTS
# ---------------
# The end-to-end F1 of an int8 arm is decided by TWO things that were never
# crossed:
#   * the CALIBRATION SET (its activation/output range decides where the
#     quantiser clips), and
#   * the quantisation GRANULARITY (quantized_method: channel / layer /
#     group{SIZE}).
# Measured on 2026-10-23 (PC simulator, 100 sensitive frames, end_to_end.f1):
#   set_A + channel 0.7299 | set_A + layer 0.7512 | set_3 + channel 0.7800
#   set_B 0.7103 | set_C 0.6726 | kl_divergence A/B/C/D 0.0000
# The tensor metrics (cos_heatmap) do NOT predict that ranking -- set_B has the
# BEST cos_hm (0.9824) and one of the WORST F1 (0.7103). So the ranking metric
# has to be the end-to-end F1, and every candidate has to be scored that way.
#
# The second thing this script is careful about is COST:
#   * a build is ~6-10 min (toolkit, CPU) -- never more than a few arms;
#   * the PC simulator runs at ~1.3 s/frame, so it is only used for the cheap
#     100-frame proxy (tier1_sensitive), NOT for the 2000-frame tracks;
#   * the BOARD is ~17x faster than the simulator for end-to-end scoring
#     (2000 frames: 152 s on the board vs ~43 min in the simulator), so the
#     finalists are scored on the board (04_run_board_fast_eval.py) and the
#     threshold curve comes free from re-scoring its TSV (19_board_threshold_sweep.py).
#
# Usage:
#   24_screening_round.sh <calib_root> <out_root> "<set>:<method>[:<algo>]" ...
#   e.g. 24_screening_round.sh /mnt/data/siping/ptq_work/calib2 \
#          /mnt/data/siping/ptq_work/screen_r1 3:layer 1:layer 3:group64
# =============================================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
WORK="${WORK:-/mnt/data/siping/ptq_work}"
CALIB_ROOT="${1:?calib root}"
OUT_ROOT="${2:?out root}"
shift 2
ARMS=("$@")
[ ${#ARMS[@]} -gt 0 ] || { echo "[FATAL] no arms given"; exit 2; }

# One build at a time per arm, all arms in parallel: the toolkit build is
# essentially single-threaded, so this is the cheapest way to use the box
# without starving whatever else is running on it.
JOBS="${JOBS:-1}"
FRAMES="${FRAMES:-$WORK/fast/tier1_sensitive.json}"
LOGDIR="$WORK/logs"
mkdir -p "$OUT_ROOT" "$LOGDIR"

# The rknn-toolkit2 lives in the `rknn` env (python3.10, toolkit 2.3.2) -- the
# `uav` env does NOT have it (ModuleNotFoundError: No module named 'rknn').
PY="${PY:-/home/huangzhe/anaconda3/envs/rknn/bin/python}"
[ -x "$PY" ] || { echo "[FATAL] toolkit python not found: $PY"; exit 2; }
cd "$REPO"

pids=()
for arm in "${ARMS[@]}"; do
    set_key="${arm%%:*}"; rest="${arm#*:}"
    method="${rest%%:*}"; algo="${rest#*:}"
    [ "$algo" = "$method" ] && algo="normal"
    tag="${set_key}_${method}_${algo}"
    log="$LOGDIR/screen_${tag}.log"
    echo "[launch] set=$set_key method=$method algo=$algo -> $OUT_ROOT/$tag  (log $log)"
    (
        "$PY" -u manu/pipeline/npu/ptq/02_batch_ptq_pc_screening.py \
            --variant p0_off --onnx-mode float \
            --sets "$set_key" --algos "$algo" \
            --calib-root "$CALIB_ROOT" \
            --out "$OUT_ROOT" \
            --frames-json "$FRAMES" \
            --quant-method "$method" \
            --jobs "$JOBS" > "$log" 2>&1
        echo "[done rc=$?] set=$set_key method=$method algo=$algo" >> "$LOGDIR/screen_round_done.log"
    ) &
    pids+=($!)
done

fail=0
for p in "${pids[@]}"; do
    wait "$p" || fail=1
done
echo "[round] done (fail=$fail)"
exit $fail
