#!/usr/bin/env bash
# trial_0080 × 跳帧适配参数(delta_n) 的 OSD 重生成（观察/对照用，不改任何既有产物）
#
# 用户 2026-11-19：「用最优的稀疏参数，使用 0080 模型，生成一个新的 osd 目录，
# 也分 pass 和 upass，明天可以跟 v3 对比。也就是用这个参数生成 osd，
# 只是把模型换成了跟板卡对应的 0080。」
#
# 参数来源（**现成冻结件，不臆造**）：
#   manu/evaluation/system_sota_longquan_trial0080_delta_n_params.json
#     confirm_consec=2  confirm_gap_tol=6  max_age_phys=10  r_slope=3.5  match_dist=16.0
#
# 模型 = 板卡对应的那份 0080：交付 rknn `p0_off_A_none.rknn`（md5 1a3faf2f…）的源图
#   → 热图缓存 /mnt/data/siping/runs/trial0080_onnx/ms_cache（24 pkl，ONNX Runtime 重建）
#   ⇒ **不重跑热图推理、不占板卡**（纯 CPU，读缓存）。
#
# 产物：
#   评测  /mnt/data/siping/runs/trial0080_onnx/eval/track_all_deltan_0080.json
#   预测  /mnt/data/siping/runs/trial0080_onnx/eval/preds_deltan_0080/
#   OSD   /mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_deltan_20261120/
#          （pass/ + un-pass/，阈值 F1 >= 88.0，与 v3 同口径）
#   对账  <OSD>/cmp_deltan_vs_v3.json
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
export PYTHONPATH=.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

EVAL_ROOT=/mnt/data/siping/runs/trial0080_onnx/eval
CACHE=/mnt/data/siping/runs/trial0080_onnx/ms_cache
GT="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
PARAMS=manu/evaluation/system_sota_longquan_trial0080_delta_n_params.json
OUT_JSON="$EVAL_ROOT/track_all_deltan_0080.json"
PREDS="$EVAL_ROOT/preds_deltan_0080"
OSD_DIR="/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_deltan_20261120"
V3_OSD="/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_20261022"
JOBS="${JOBS:-6}"
OUT_CUT="${OUT_CUT:-88.0}"
PY=/home/huangzhe/anaconda3/envs/llm/bin/python

echo "== 0. 前置检查 =="
for p in "$CACHE" "$GT" "$PARAMS"; do
  [ -e "$p" ] || { echo "[FAIL] 缺失: $p"; exit 2; }
done
echo " 缓存 pkl 数: $(ls "$CACHE"/*.pkl 2>/dev/null | wc -l) / 24"
if [ -e "$OSD_DIR" ] && [ -n "$(ls -A "$OSD_DIR" 2>/dev/null)" ]; then
  echo "[FAIL] 输出目录已存在且非空: $OSD_DIR —— 绝不覆盖既有 OSD"; exit 1
fi

echo "== 1. 带 track 臂评估（参数 = delta_n 跳帧适配件；模型 = 板卡对应 0080 缓存）=="
$PY manu/evaluation/system_sota_longquan.py \
  --cache-root "$CACHE" --labels-root "$GT" --params-json "$PARAMS" \
  --all-seqs --out-json "$OUT_JSON" --out-preds-dir "$PREDS" \
  2>&1 | tee "$EVAL_ROOT/track_all_deltan_0080.log" | grep -E "^\[SEQ\]|^\[OVERALL\]|^\[GATE\]"

echo "== 2. 渲染 OSD（新目录，pass/un-pass，阈值 F1 >= $OUT_CUT）=="
PREDS_DIR="$PREDS" SEARCH_JSON="$OUT_JSON" OUT_DIR="$OSD_DIR" \
OUT_CUT="$OUT_CUT" JOBS="$JOBS" \
  bash manu/videos/render_trial0080_longquan_osd.sh

echo "== 3. 与 v3 逐序列对账 =="
$PY manu/evaluation/cmp_seq_vs_official.py \
  --eval-json "$OUT_JSON" \
  --official-json "$V3_OSD/search_json_for_render.json" \
  --out-json "$OSD_DIR/cmp_deltan_vs_v3.json"

echo "== 4. 汇总 =="
NP=$(ls -1 "$OSD_DIR"/pass/*.mp4 2>/dev/null | wc -l)
NU=$(ls -1 "$OSD_DIR"/un-pass/*.mp4 2>/dev/null | wc -l)
echo " pass=$NP  un-pass=$NU  合计=$((NP+NU))/24"
$PY - <<PYEOF
import json
a=json.load(open("$OUT_JSON")); b=json.load(open("$V3_OSD/search_json_for_render.json"))
oa=a.get("overall",{}); ob=b.get("best_global",{}).get("metrics",{})
print("deltan F1=%.6f%%  fp=%s  far/frame=%s" % (float(oa.get("f1",0))*100, oa.get("fp"), oa.get("far_per_frame")))
print("v3     F1=%.6f%%  fp=%s  far/frame=%s" % (float(ob.get("f1",0))*100, ob.get("fp"), ob.get("far_per_frame")))
PYEOF
du -sh "$OSD_DIR"
echo "== DONE =="
