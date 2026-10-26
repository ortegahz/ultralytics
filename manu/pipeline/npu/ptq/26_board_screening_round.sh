#!/usr/bin/env bash
# =============================================================================
# 26_board_screening_round.sh -- rank int8 candidates ON THE BOARD.
#
# WHY THE BOARD AND NOT THE PC SIMULATOR
# --------------------------------------
# Measured 2026-10-23: on the PC simulator (100 sensitive frames) the
# calibration-set v2 model (set_3) scored F1 0.7800 against the baseline's
# 0.7299 -- a +0.050 gap. On the BOARD (001324, 2000 frames, collect_th=0.05)
# the two are IDENTICAL at their own best threshold:
#
#   set_3 + channel : best F1 0.7090 @ 0.45   (at 0.30: F1 0.6505, TP 1367, FP 517)
#   set_A + channel : best F1 0.7090 @ 0.45   (at 0.30: F1 0.6448, TP 1359, FP 527)
#
# => the simulator's 100-frame end-to-end F1 does NOT transfer (+0.050 on the
# simulator, 0.000 on the board). Ranking on the simulator picks the wrong
# model. The board is the ground truth, and it is CHEAP enough to screen on:
# 2000 frames = 153 s, 2500 frames = 190 s at 13 fps.
#
# SCREENING MUST COVER BOTH FAILURE MODES
# --------------------------------------
# The quantisation loss has two opposite signatures (archive 07, §56.4):
#   T1 Anti-UAV : peak clipping  -> 643 fewer TP, 109 more FP  (MISSES)
#   T3 001324    : raised floor   -> 213 more TP, 1474 more FP (FALSE ALARMS)
# A calibration set that covers the strong peaks (the v2 family) can only help
# T1; on T3 it is neutral at best. Screening on T3 alone would therefore throw
# away exactly the arms that fix T1. This script screens BOTH:
#   T1 = fast/anti_uav_sub2500.json (2500 frames, ~190 s/arm)
#   T3 = fast/001324_sub2000.json  (2000 frames, ~153 s/arm)
#
# Usage:
#   26_board_screening_round.sh <out_root> <model.rknn> [model.rknn ...]
# Env:
#   COLLECT_TH (0.05)   THRESHOLDS (0.30..0.70)   WAIT_MIN (40)
# =============================================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
W="${W:-/mnt/data/siping/ptq_work}"
OUT_ROOT="${1:?out root}"
shift
MODELS=("$@")
[ ${#MODELS[@]} -gt 0 ] || { echo "[FATAL] no models given"; exit 2; }

COLLECT_TH="${COLLECT_TH:-0.05}"
THRESHOLDS="${THRESHOLDS:-0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70}"
WAIT_MIN="${WAIT_MIN:-40}"
mkdir -p "$OUT_ROOT"

source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd "$REPO"

# ---- the board allows TWO RKNN contexts; a third one soft-resets the NPU ----
# So we wait until no board demo is alive at all, and we do not start while
# another sweep driver is running.
#
# The mutual exclusion between screening rounds is a LOCK DIRECTORY, not a ps
# grep: `ps -eo args | grep -c "<this script>"` counts the launcher's OWN
# command line (it contains the script path), so the process would wait for
# itself forever. mkdir is atomic, so it is the right primitive here.
LOCK="${LOCK:-/tmp/ptq_board_screen.lock}"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "[FATAL] another screening round holds $LOCK -- refusing to start"
    exit 4
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

wait_for_board() {
    echo "### waiting for the board to be free ###"
    for _ in $(seq 1 "$WAIT_MIN"); do
        n=$(ps -eo args | grep -c "[1]9_board_threshold_sweep")
        # `grep -c` exits 1 when it finds nothing, so the remote side must
        # swallow that status; otherwise the "|| echo 1" fallback appends a
        # second line and the count becomes "0\n1" (an integer error).
        b=$(timeout 20 ssh -o ConnectTimeout=10 -o BatchMode=yes ptq-board \
                'ps -eo comm | grep -c "^rknn_heatmap" || true' < /dev/null 2>/dev/null \
                | tr -d '[:space:]')
        b=${b:-1}
        if [ "$n" -eq 0 ] && [ "$b" -eq 0 ]; then
            echo "### board is free ###"
            return 0
        fi
        echo "  busy: sweep=$n board_ctx=$b -- waiting 15 s"
        sleep 15
    done
    echo "[FATAL] board never became free in $WAIT_MIN rounds"
    return 1
}

MANIFESTS="${MANIFESTS:-fast/anti_uav_sub2500.json fast/001324_sub2000.json}"
for manifest in $MANIFESTS; do
    name=$(basename "$manifest" .json)
    wait_for_board || exit 3
    echo
    echo "############################################################"
    echo "### BOARD SCREEN $name   manifest=$W/$manifest"
    echo "############################################################"
    python -u manu/pipeline/npu/ptq/19_board_threshold_sweep.py \
        --manifest "$W/$manifest" \
        --models "${MODELS[@]}" \
        --collect-th "$COLLECT_TH" \
        --thresholds "$THRESHOLDS" \
        --skip-pt \
        --board-models-dir "${BOARD_MODELS_DIR:-/mnt/manu/ptq/models}" \
        --out "$OUT_ROOT/$name" 2>&1 | tee "$OUT_ROOT/${name}.log"
done

echo
echo "### ranking (best F1 per arm per track) ###"
python - "$OUT_ROOT" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = {}
for j in sorted(root.glob("*/*sweep*.json")) + sorted(root.glob("*/sweep*.json")):
    try:
        d = json.load(open(j))
    except Exception:
        continue
    track = d.get("manifest_name") or j.parent.name
    for tag, arm in (d.get("arms") or {}).items():
        cur = arm.get("curve") or []
        if not cur:
            continue
        best = max(cur, key=lambda c: c["f1"])
        rows.setdefault(tag, {})[track] = (best["f1"], best.get("th"))
for tag, tr in sorted(rows.items(), key=lambda kv: -sum(v[0] for v in kv[1].values())/len(kv[1])):
    parts = " ".join("%s=%.4f@%.2f" % (k, v[0], v[1]) for k, v in sorted(tr.items()))
    mean = sum(v[0] for v in tr.values())/len(tr)
    print("%-34s mean=%.4f  %s" % (tag, mean, parts))
EOF
