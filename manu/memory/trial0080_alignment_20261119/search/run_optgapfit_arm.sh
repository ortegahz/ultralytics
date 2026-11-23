#!/usr/bin/env bash
# trial_0080 + v3 + Δn 重搜出的最优 gapfit 参数：self-check -> track 臂 -> 双对账
set -uo pipefail
cd /mnt/data/siping/runs/npu20ms/nano_search/repo || exit 2
export PYTHONPATH=/mnt/data/siping/ptq_work/ortgpu:/mnt/data/siping/runs/npu20ms/nano_search/repo
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
OUT=/mnt/data/siping/runs/trial0080_onnx/eval
CACHE=/mnt/data/siping/runs/trial0080_onnx/ms_cache
GT="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
REC="/mnt/data/siping/datasets/manu/龙泉山"
PY=~/anaconda3/envs/uav/bin/python
PARAMS=manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json

echo "[1/5] self-check（显式基线：trial_0080 单帧臂）"
$PY manu/evaluation/system_sota_longquan.py \
  --cache-root "$CACHE" --labels-root "$GT" --all-seqs \
  --self-check --self-check-baseline manu/evaluation/system_sota_longquan_trial0080_singleframe_gate_baseline.json \
  2>&1 | tee "$OUT/self_check_optgapfit.log"
GATE_RC=${PIPESTATUS[0]}
echo "[self-check rc=$GATE_RC]"

echo
echo "[2/5] optgapfit 臂评测"
$PY manu/evaluation/system_sota_longquan.py \
  --cache-root "$CACHE" --labels-root "$GT" \
  --params-json "$PARAMS" --all-seqs \
  --out-json "$OUT/track_all_v3_optgapfit.json" \
  --out-preds-dir "$OUT/preds_v3_optgapfit" \
  2>&1 | tee "$OUT/track_all_v3_optgapfit.log" | grep -E "^\[SEQ\]|^\[OVERALL\]|^\[GATE\]"

echo
echo "[2b/5] 可复现性检查：同一份缓存上用【v3 原参】跑出的 track_all_v3.json 对官方 v3 记录"
echo "          （|dF1| <= 0.1pp 才是流水线没坏的证据；换了参数的臂当然对不上，这不是错误）"
$PY manu/evaluation/cmp_seq_vs_official.py \
  --eval-json "$OUT/track_all_v3.json" \
  --official-json "$REC/osd_sys_sota_trial0080_v3_20261022/search_json_for_render.json" \
  --out-json "$OUT/cmp_v3_rebuilt_vs_official_v3.json" \
  --require-gt-identical --max-df1-pp 0.1 \
  2>&1 | tee "$OUT/cmp_v3_rebuilt_vs_official_v3.log"
REPRO_RC=${PIPESTATUS[0]}
echo "[cmp reproducibility rc=$REPRO_RC]"

echo
echo "[3/5] 与官方 v3 记录逐序列对账（真闸门）"
$PY manu/evaluation/cmp_seq_vs_official.py \
  --eval-json "$OUT/track_all_v3_optgapfit.json" \
  --official-json "$REC/osd_sys_sota_trial0080_v3_20261022/search_json_for_render.json" \
  --out-json "$OUT/cmp_v3_optgapfit_vs_official_v3.json" \
  --require-gt-identical --max-df1-pp 0.1 \
  2>&1 | tee "$OUT/cmp_v3_optgapfit_vs_official_v3.log"
CMP_RC=${PIPESTATUS[0]}
echo "[cmp vs official rc=$CMP_RC]"

echo
echo "[4/5] 与『重建缓存上的 v3』对账（同一份缓存 => 纯参数效应）"
$PY manu/evaluation/cmp_seq_vs_official.py \
  --eval-json "$OUT/track_all_v3_optgapfit.json" \
  --official-json "$OUT/search_json_for_render_v3_rebuilt.json" \
  --out-json "$OUT/cmp_v3_optgapfit_vs_rebuilt_v3.json" \
  --require-gt-identical \
  2>&1 | tee "$OUT/cmp_v3_optgapfit_vs_rebuilt_v3.log"
CMP2_RC=${PIPESTATUS[0]}
echo "[cmp vs rebuilt rc=$CMP2_RC]"

echo
echo "[5/5] 汇总 OVERALL"
$PY - "$OUT/track_all_v3_optgapfit.json" "$OUT/track_all_v3.json" <<'PYEOF'
import json, sys
from pathlib import Path
for path in sys.argv[1:]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    o = d.get("overall", {})
    print(f"[OVERALL] {path}")
    for k in ("tp", "fp", "fn", "gt", "f1", "recall", "precision", "far_per_frame"):
        if k in o:
            print(f"   {k:14s} = {o[k]}")
PYEOF

echo
echo "[done] $(date +%H:%M:%S)  gate_rc=$GATE_RC repro_rc=$REPRO_RC cmp_official_rc=$CMP_RC cmp_rebuilt_rc=$CMP2_RC"
