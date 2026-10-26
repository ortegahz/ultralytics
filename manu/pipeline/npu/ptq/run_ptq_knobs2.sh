#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_ptq_knobs2.sh -- the SECOND pure-PTQ batch: the two knobs that are
# still untried AND do not turn the graph input into FLOAT16.
#
# Batch 1 (run_ptq_knobs.sh) closed two more axes with hard toolkit errors:
#   quantized_method='group{SIZE}'
#     -> "config: The quantize_method = 'group32' only support when
#         quantized_dtype = 'w4a16'"   and rk3588 only accepts w8a8
#         (w4a16 / w8a16 are rejected by the platform), so the whole
#         weight-granularity axis is closed on this target.
#   quantized_hybrid_level=1/2/3
#     -> the graph input becomes FLOAT16 (2026-10-16 measurement), i.e. it
#        buys accuracy back with the fp16 input-set cost.
#
# That leaves exactly two knobs that change the COMPUTATION but keep the
# input INT8:
#
#   optimization_level (default 3)
#     It decides how hard the graph is optimised (fusion), and the fusion
#     result decides where the quantisation nodes sit -- i.e. it can move
#     the activation quantisation noise itself.  Levels 0/1/2 have never
#     been built for this model.
#
#   enable_flash_attention (default False)
#     The per-layer SQNR analysis of 2026-10-16 (memory archive/02 sec. 28.3)
#     ranked the attention Softmax among the three WORST tensors
#     (28.73 dB) -- an op whose numerics a different attention kernel
#     changes.  RKNN's SDPA path is a different kernel on rk3588.
#
#   quantized_method='layer' (per-tensor, coarser than the per-channel we
#     always used) is included as the REVERSE control of the granularity
#     axis: it should be worse, and if it is not, the per-channel scales
#     are not doing anything.
#
# Same gate as batch 1: the dtype of the INPUT operator, read from the
# build log's operator table (--verbose-build), costs no board time.
# Only INT8-input models are allowed to go on the board.
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
  : > "$LOG/knob_$name.log"
  python -u manu/pipeline/npu/ptq/02_batch_ptq_pc_screening.py \
    --variant p0_off --onnx-mode float --algos normal --jobs 1 \
    --verbose-build "$@" \
    >> "$LOG/knob_$name.log" 2>&1
  local rc=$?
  echo "[$(date '+%H:%M:%S')] $name rc=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "[$(date '+%H:%M:%S')] $name FAILED rc=$rc -- see $LOG/knob_$name.log"
  else
    grep -E "^p0_off" "$LOG/knob_$name.log" | tail -3
    local nfp16 indt
    nfp16=$(grep -c "Meet hybrid type" "$LOG/knob_$name.log")
    outdir="${!#}"          # the --out dir is the last argument of every call
    indt=$(grep -rhoE "InputOperator +[A-Z0-9]+" "$outdir"/*.log 2>/dev/null | head -1 | awk '{print $2}')
    echo "[$(date '+%H:%M:%S')] $name input_op_dtype=${indt:-UNKNOWN} float16_tensors=$nfp16"
  fi
}

run olv0 --calib-root $W/calib --sets A \
    --optimization-level 0 --out $W/screen_olv0 &
PA=$!
run olv2 --calib-root $W/calib --sets A \
    --optimization-level 2 --out $W/screen_olv2 &
PB=$!
run fa --calib-root $W/calib --sets A \
    --enable-flash-attention --out $W/screen_fa &
PC=$!
run layer --calib-root $W/calib --sets A \
    --quant-method layer --out $W/screen_layer &
PD=$!
wait $PA $PB $PC $PD
echo "[$(date '+%H:%M:%S')] ALL KNOB BUILDS DONE"

echo
printf "%-14s %-30s %-9s %-9s %-12s %8s %8s %10s %7s %8s %6s\n" \
  "VARIANT" "TAG" "STATUS" "INPUT_OP" "GATE" "cos_hm" "cos_min" "mse" "F1" "build_s" "fp16"
for pair in "olv0:screen_olv0" "olv2:screen_olv2" "fa:screen_fa" "layer:screen_layer"; do
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
