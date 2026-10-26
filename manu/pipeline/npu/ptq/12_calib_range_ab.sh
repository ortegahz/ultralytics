#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 12_calib_range_ab.sh -- the calibration-range A/B, run on the SERVER.
#
# WHY
# ---
# The trial-0080 int8 model cannot represent heatmap values above 0.2399:
#   board: out[0] type=INT8 zp=-128 scale=0.00094072  ->  range [0, 0.2399]
#   decision threshold = 0.3  ->  every peak is clipped below it  ->  0 detections
#   board probe: th=0.3 -> 0/500 frames, th=0.1 -> 389/500, score_max=0.2399
#
# But the SAME calibration frames, run through the FLOAT ONNX, give heatmap
# peaks of 0.85..0.97 (10_probe_calib_peak.py) => an output scale of ~0.0075.
# So the toolkit's calibration pass did not see what ONNX Runtime sees.
#
# Three candidate causes, three builds (algo=normal, build only -- no
# simulator), all from the SAME float32 ONNX so the only variable is the
# calibration list:
#
#   bmp    set_A.txt   250 frames, 24-bit BMP (the set we shipped)
#   png    set_A.txt   the same 250 frames re-encoded as PNG
#   t0474  set_t0474.txt  200 Anti-UAV train feature JPGs (the frozen
#                         trial-0474 list that produced scale=0.00365272)
#
# READ THE RESULT ON THE BOARD: each model's out[0] zp/scale is printed by
# rknn_heatmap_smoke at startup.  That scale is the answer.
#
# Usage (on the server):
#   bash 12_calib_range_ab.sh <out_dir>
# ---------------------------------------------------------------------------
set -uo pipefail

REPO=/tmp/pycharm_project_10ae9e2e
OUT="${1:-/mnt/data/siping/ptq_work/calib_ab}"
LOGS=/mnt/data/siping/ptq_work/logs
PY=/home/huangzhe/anaconda3/envs/rknn/bin/python
ONNX=$REPO/runs/onnx/trial_0080_hm_nop0.onnx

mkdir -p "$OUT" "$LOGS"

run_one() {
  local tag="$1" calib_root="$2" set_key="$3"
  cd "$REPO" || exit 1
  "$PY" -u manu/pipeline/npu/ptq/02_batch_ptq_pc_screening.py \
      --variant p0_off --onnx-mode float \
      --calib-root "$calib_root" --sets "$set_key" --algos normal --jobs 1 \
      --build-only --out "$OUT" > "$LOGS/calabab_${tag}.log" 2>&1
  echo "[done] $tag rc=$?"
}

# the driver names the output <variant>_<set>_<algo>.rknn -> rename afterwards
run_one bmp   /mnt/data/siping/ptq_work/calib  A
run_one png   /mnt/data/siping/ptq_work/calib2 A
run_one t0474 /mnt/data/siping/ptq_work/calib2 t0474

cd "$OUT" || exit 1
for f in p0_off_A_normal.rknn; do
  [ -f "$f" ] && mv -f "$f" p0_off_bmp_normal.rknn
done
[ -f p0_off_A_normal.rknn ] && mv -f p0_off_A_normal.rknn p0_off_png_normal.rknn
[ -f p0_off_t0474_normal.rknn ] && mv -f p0_off_t0474_normal.rknn p0_off_t0474_normal.rknn
ls -la "$OUT"
echo "[calib_ab] finished -> $OUT"
