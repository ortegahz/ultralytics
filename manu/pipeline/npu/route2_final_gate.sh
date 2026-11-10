#!/usr/bin/env bash
# ==============================================================================
# 路线二候选臂的**收官编排**：官方三轨门禁 → int8 rknn → 一次板卡会话 → 联合门禁
# ==============================================================================
# 用法：
#   bash manu/pipeline/npu/route2_final_gate.sh \
#       --candidate 9645=/path/epoch10.pt \
#       --candidate 9647=/path/epoch10.pt \
#       --budget-ms 30 --nano-min 0.6978
#
# 设计要点：
#   ① **官方三轨门禁 = 权威选型口径**（T1 UAV / T2 龙泉山黄金池 / T3 blind），
#      每个候选一条独立门禁，**T2 缓存按候选分片并行**（GPU 空闲时 ~10 min/候选）。
#   ② **NanoBench（no-GMC）= 微小目标分层的补充口径**，与官方三轨**分列报告、严禁混读**。
#   ③ **板卡一次会话测完全部候选**（驱动脆弱：长会话 → timeout → soft reset → rknn_init 132，
#      只能物理断电恢复）⇒ 必须先归档 arms/ 再上传（board_speed_search.archive_board_arms）。
#   ④ 联合判定：int8 ≤ budget 且 nano ≥ nano_min ⇒ PASS（dual_constraint_gate.py）。
# ==============================================================================
set -uo pipefail
REPO=/tmp/pycharm_project_10ae9e2e
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
PY_RKNN=/home/huangzhe/anaconda3/envs/rknn/bin/python
RUNS=/mnt/data/siping/runs/npu20ms
WORK="$RUNS/route2_final_gate"
GATE_ROOT=/mnt/data/siping/runs/npu20ms_offcal/gate
CALIB="$RUNS/_chain_test/calib_nogmc_train_server.txt"

BUDGET_MS=30
NANO_MIN=0.6978
GATE_GPUS="0,1,2,3"
declare -a CANDIDATES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --candidate) CANDIDATES+=("$2"); shift 2 ;;
    --budget-ms) BUDGET_MS="$2"; shift 2 ;;
    --nano-min)  NANO_MIN="$2"; shift 2 ;;
    --gate-gpus) GATE_GPUS="$2"; shift 2 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done
[[ ${#CANDIDATES[@]} -eq 0 ]] && { echo "至少要给一个 --candidate" >&2; exit 2; }

mkdir -p "$WORK"/{onnx,arms}
cd "$REPO" && export PYTHONPATH=.

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- ① 官方三轨门禁（每个候选一条，缓存并行分片） ----
GPU_ARR=($(echo "$GATE_GPUS" | tr ',' ' '))
n_gpu=${#GPU_ARR[@]}
declare -a GATE_PIDS=()
for i in "${!CANDIDATES[@]}"; do
  name="${CANDIDATES[$i]%%=*}"; w="${CANDIDATES[$i]##*=}"
  out="$GATE_ROOT/r2_${name}"
  mkdir -p "$out"
  g="${GPU_ARR[$i % n_gpu]}"
  log "官方三轨门禁 → $out（GPU $g，权重 $(basename "$w")）"
  ( "$PY" -u manu/training/auto_epoch_gate_check.py \
      --weights "$w" --out-dir "$out" \
      --infer-gpu "$g" --cache-gpus "$g" --gpus 1 \
      > "$out/gate.log" 2>&1; echo "rc=$?" > "$out/gate.rc" ) &
  GATE_PIDS+=($!)
done
log "等 $(( ${#GATE_PIDS[@]} )) 条门禁跑完（各含 T1 推理 + 龙泉山缓存重建 + T2/T3 判定）…"
for p in "${GATE_PIDS[@]}"; do wait "$p"; done

# ---- ② 导 int8 rknn ----
declare -a ARMS=()
for c in "${CANDIDATES[@]}"; do
  name="${c%%=*}"; w="${c##*=}"
  onnx="$WORK/onnx/${name}.onnx"; rknn="$WORK/arms/${name}_int8.rknn"
  if [ ! -f "$rknn" ]; then
    "$PY" manu/pipeline/npu/hm_export_onnx.py --weights "$w" --out "$onnx" \
      --act relu --ablate-p0 --skip-check > "$WORK/export_${name}.log" 2>&1
    "$PY_RKNN" manu/pipeline/npu/hm_convert_rknn.py --onnx "$onnx" --out "$rknn" \
      --calib-list "$CALIB" --calib-list-only > "$WORK/rknn_${name}.log" 2>&1
  fi
  if [ -f "$rknn" ]; then
    stride=$("$PY" -c "import torch;d=torch.load('$w',map_location='cpu',weights_only=False);print(int(d.get('stride',2) or 2))" 2>/dev/null)
    [[ -z "$stride" ]] && stride=2
    log "rknn 就绪：${name} (stride=$stride)"
    ARMS+=("${name}|${rknn}|${stride}")
  else
    log "rknn 失败：${name}"
  fi
done

# ---- ③ 一次板卡会话测完全部候选 ----
log "板卡一次会话测速（先归档旧臂再上传）…"
"$PY" - "$WORK" "${ARMS[@]}" <<'PYEOF'
import sys
sys.path.insert(0, "/tmp/pycharm_project_10ae9e2e")
from manu.pipeline.npu import board_speed_search as B
work = sys.argv[1]
built = []
for spec in sys.argv[2:]:
    name, rknn, stride = spec.split("|")
    built.append(dict(name=name, rknn=rknn, stride=int(stride), note="", hc=0,
                      head_src="", late=0, sc=1.0))
B.archive_board_arms()      # ⚠️ 必须先归档再上传
B.upload(built)
r1, _r2, _r3, _txt = B.measure_on_board(work)
import json
json.dump(r1, open(f"{work}/board_ms.json", "w"), ensure_ascii=False, indent=2)
print("@@BOARD@@", json.dumps(r1, ensure_ascii=False))
PYEOF

# ---- ④ 联合门禁（速度 × nano 召回） ----
log "联合门禁（预算 ${BUDGET_MS} ms / nano 下限 ${NANO_MIN}）…"
CAND_ARGS=""
BOARD_ARGS=""
# 板卡实测 ms（arm_<name>_int8 → name）⇒ 喂给联合门禁的速度门
if [ -f "$WORK/board_ms.json" ]; then
  BOARD_ARGS=$("$PY" - "$WORK/board_ms.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
items = []
for k, v in d.items():
    if k.startswith("arm_") and k.endswith("_int8"):
        items.append(f"{k[4:-5]}={v}")
print(",".join(items))
PYEOF
)
fi
for c in "${CANDIDATES[@]}"; do
  name="${c%%=*}"
  pkl="$RUNS/tiny_ref/w1_t${name}_ep10.pkl"
  [ -f "$pkl" ] || pkl="$RUNS/tiny_ref/w1_t${name}.pkl"
  [ -f "$pkl" ] && CAND_ARGS="$CAND_ARGS --candidate ${name}=${pkl}"
done
if [ -n "$CAND_ARGS" ]; then
  "$PY" manu/evaluation/dual_constraint_gate.py $CAND_ARGS \
    ${BOARD_ARGS:+--board-ms "$BOARD_ARGS"} \
    --baseline /mnt/data/siping/runs/nogmc_ref_weekend/uav_cache_base.pkl \
    --budget-ms "$BUDGET_MS" --nano-min "$NANO_MIN" \
    --out "$WORK/dual_gate.json" 2>&1 | tail -30
else
  log "没有候选的 no-GMC 缓存 ⇒ 跳过联合门禁（先跑 route2_acceptance.py 建缓存）"
fi
log "DONE ⇒ $WORK"
