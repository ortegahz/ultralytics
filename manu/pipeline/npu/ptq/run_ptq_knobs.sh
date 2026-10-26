#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_ptq_knobs.sh -- the PURE-PTQ sweep over the toolkit knobs that were
# never exercised before 2026-10-23.
#
# ## Why the sweep is shaped the way it is
#
# The mechanism measured on 2026-10-22 is an ADDITIVE noise floor on the
# heatmap: on the Anti-UAV full track the int8 arm loses 643 true positives
# and gains only 109 false positives (noise shaves the peaks), while on the
# 001324 blind track it GAINS 77 true positives and 356 false positives (noise
# lifts the weak targets and the background alike).
#
# The obvious toolkit answer to a noise floor is hybrid quantisation, but
# the 2026-10-16 measurement (memory archive/02 section 28) closed it:
#   quantized_hybrid_level = 1/2/3  ->  the graph INPUT becomes FLOAT16,
#   internal memory 105,600 KB (= fp16, int8 is 20,400 KB), and the board
#   measures `input attr: type=1` -> inputs_set 45.41 ms instead of 0.34 ms.
# Root cause: the toolkit binds "dtype of the graph input" to "compute
# precision"; there is no switch that keeps the input int8 while the inside
# is mixed.  The same happens to the QDQ/--no-quant path (the QAT arms on
# 2026-10-23 also came out with `InputOperator FLOAT16`).
#
# => THE GATE for every build in this sweep is the dtype of the INPUT
#    operator.  A model whose input is FLOAT16 buys its accuracy back with
#    a ~45 ms (small cores) / ~14.8 ms (big cores) input-set penalty and
#    5x the board memory, so it is NOT a valid substitute for an int8
#    deliverable.  The dtype is read straight out of the build log's
#    operator table (--verbose-build), so the gate costs no board time.
#
# ## The knobs still in play (none of them changes the input dtype)
#
#   quantized_method = group{SIZE}   weight scales per group of SIZE
#                                     channels -- finer than per-channel, so
#                                     a smaller WEIGHT quantisation error.
#                                     {SIZE} must be a multiple of 32 in
#                                     [32,256] (toolkit docstring).
#   calibration set v2 (set_3)        the best set on the PC simulator, but
#                                     it has NEVER been on the board.
#   baseline (same batch)             a controlled A/B against the artefact
#                                     that the whole 60,610-frame evaluation
#                                     was run with.
#
# hybrid_quantization_step1/2 (the official two-step route) and
# quantized_hybrid_level are NOT deliverable candidates -- they only exist
# here as a diagnostic (hyb1_diag) to answer one open question: did
# toolkit 2.3.2 change anything about the hybrid input dtype compared to
# 2.3.0?  That answer decides whether the whole hybrid branch is closed.
# ---------------------------------------------------------------------------
set -uo pipefail
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate rknn
cd /tmp/pycharm_project_10ae9e2e
W=/mnt/data/siping/ptq_work
LOG=$W/logs
mkdir -p "$LOG"

run() {
  local name="$1"; shift
  echo "[$(date '+%H:%M:%S')] === $name ==="
  # start from an empty log: the fp16-tensor count and the InputOperator
  # dtype in the summary are read out of this log, a stale log would poison
  # both
  : > "$LOG/knob_$name.log"
  python -u manu/pipeline/npu/ptq/02_batch_ptq_pc_screening.py \
    --variant p0_off --onnx-mode float --algos normal --jobs 1 \
    --verbose-build "$@" \
    >> "$LOG/knob_$name.log" 2>&1
  # $? MUST be captured on its own line: a command substitution such as
  # $(date) inside the next echo would overwrite it and report rc=0 for a
  # crashed build.  That bug silently hid the first (failed) sweep.
  local rc=$?
  echo "[$(date '+%H:%M:%S')] $name rc=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "[$(date '+%H:%M:%S')] $name FAILED rc=$rc -- see $LOG/knob_$name.log"
  else
    grep -E "^p0_off" "$LOG/knob_$name.log" | tail -3
    local nfp16
    nfp16=$(grep -c "Meet hybrid type" "$LOG/knob_$name.log")
    local indt outdir
    outdir="${!#}"          # the --out dir is the last argument of every call
    indt=$(grep -rhoE "InputOperator +[A-Z0-9]+" "$outdir"/*.log 2>/dev/null | head -1 | awk '{print $2}')
    echo "[$(date '+%H:%M:%S')] $name input_op_dtype=${indt:-UNKNOWN} float16_tensors=$nfp16"
  fi
}

# --- group A: weight quantisation granularity ------------------------------
run grp32 --calib-root $W/calib --sets A \
    --quant-method group32 --out $W/screen_grp32 &
PA=$!
run grp64 --calib-root $W/calib --sets A \
    --quant-method group64 --out $W/screen_grp64 &
PB=$!
# --- group B: the calibration set v2 (set_3 was best on the simulator but
#     has NEVER been on the board) -------------------------------------------
run v2set3 --calib-root $W/calib2 --sets 3 --out $W/screen_v2b &
PC=$!
# --- group C: the SAME-batch baseline (no knob) -----------------------------
# Built in the same batch with the same code path, so the comparison is a
# controlled A/B and not a comparison against a differently-built artefact.
run base --calib-root $W/calib --sets A --out $W/screen_base &
PD=$!
# --- group D: DIAGNOSTIC ONLY, not a deliverable candidate ------------------
# Question: does toolkit 2.3.2 (vs 2.3.0 in the 2026-10-16 measurement)
# still turn the graph input into FLOAT16 under hybrid quantisation?
run hyb1_diag --calib-root $W/calib --sets A \
    --quant-hybrid-level 1 --auto-hybrid-cos 0.98 --out $W/screen_hyb1_diag &
PE=$!
wait $PA $PB $PC $PD $PE
echo "[$(date '+%H:%M:%S')] ALL KNOB BUILDS DONE"

echo
printf "%-14s %-30s %-9s %-9s %-12s %8s %8s %10s %7s %8s %6s\n" \
  "VARIANT" "TAG" "STATUS" "INPUT_OP" "GATE" "cos_hm" "cos_min" "mse" "F1" "build_s" "fp16"
for pair in "base:screen_base" "grp32:screen_grp32" "grp64:screen_grp64" \
            "v2set3:screen_v2b" "hyb1_diag:screen_hyb1_diag"; do
  name="${pair%%:*}"; d="${pair##*:}"
  python3 - "$W/$d/screen_p0_off.json" "$d" "$W/$d" <<'PY'
import glob, json, os, re, sys
path, d, log = sys.argv[1], sys.argv[2], sys.argv[3]
nfp16, indt = 0, None
text = ""
for fn in sorted(glob.glob(os.path.join(log, "*.log"))):
    with open(fn, "r", errors="ignore") as fh:
        text += fh.read()
nfp16 = text.count("Meet hybrid type")
m = re.search(r"InputOperator\s+(\w+)", text)
indt = m.group(1) if m else None
gate = {"INT8": "OK int8-input", "FLOAT16": "REJECT fp16-input"}.get(indt, "UNKNOWN")
try:
    with open(path) as fh:
        data = json.load(fh)
except Exception as exc:
    print("%-14s %-30s %-9s %-9s %-12s (%s)" % (d, "-", "no_json", indt or "-", gate, exc))
    raise SystemExit
for r in data.get("results", []):
    print("%-14s %-30s %-9s %-9s %-12s %8.5f %8.5f %10.3e %7.4f %8.0f %6d" % (
        d, r.get("tag") or "-", r.get("status"), r.get("input_op_dtype") or indt or "-",
        gate, r.get("cos_heatmap_mean") or 0, r.get("cos_heatmap_min") or 0,
        r.get("mse_heatmap_mean") or 0, r.get("f1") or 0, r.get("build_seconds") or 0, nfp16))
PY
done
