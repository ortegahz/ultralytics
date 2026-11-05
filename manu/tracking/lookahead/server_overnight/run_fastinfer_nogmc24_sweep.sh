#!/usr/bin/env bash
# ============================================================
# ★ 24 序官方口径 Δ 扫描 —— **no-GMC 特征语义**（P0b 接力的第 ③ 步）
#
#   这是 `run_fastinfer_24seq_sweep.sh`（**只读参考，禁改**）的薄参数化：
#   唯一变量 = `--cache-root` 指向**用 no-GMC 特征重建**的 24 序 ms_cache。
#   其余（CFG 直接从 run_delivery.sh 取、GT / 帧 / 帧缓存 / 工作点 params /
#   --delta 列表 / --arms 四臂）**逐字不变** ⇒ 与 rc2 的 24 序扫描构成严格 A/B。
#
#   前置条件（务必先做完，见 runs/lookahead/fi_nogmc/_p0b/GPU_REQUIRED.md）：
#     ① 24 序 no-GMC 特征集已建好；
#     ② 24 序 no-GMC ms_cache 已用 **GPU** 重建（CPU 建的与 GPU 建的**不逐位相同**）。
#
#   ★ 口径自证（**默认 = no-GMC 语义缓存**）：
#     默认 CACHE = /mnt/data/siping/runs/nogmc_t3/swa/ms_cache_24seq
#       （no-GMC 特征 + cs_t0043 模型包建出，见 rebuild_fpv8_24seq_nogmc24_server.sh）
#     ⇒ **模型包与缓存语义必须配对**：模型包是在**建缓存**那一步经 --hm-weights 烤进 pkl 的，
#       本脚本不再单独接 ckpt ⇒ 换缓存 = 换整套 {特征语义, 模型包} 口径；
#       拿 no-GMC 模型包去读 GMC 语义的缓存（或反之）⇒ 口径串味、结论作废。
#     若要跑 rc2 / GMC 语义，必须**显式覆盖** CACHE（该缓存对应 rc2 / trial_0080ep3 + GMC 特征，
#     与上面的 no-GMC 缓存**不可互换**；REPO 默认 /tmp/pycharm_project_10ae9e2e）：
#       CACHE=$REPO/runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache bash .../run_fastinfer_nogmc24_sweep.sh
#
#   ★ 默认 OUT 不再写进 runs/lookahead/fi_nogmc24/：
#     fi_nogmc24/ 是**交付判据产物目录**（summarize_nogmc24_results.py 以它为 root、
#     pack_delivery_nogmc_fi.sh 从它打包）⇒ 扫描跑批产物落进去会污染交付口径，
#     故默认改写到独立目录 fi_nogmc24_sweep_runs/nogmc/（非交付产物，勿被当成交付口径读）。
#
#   用法（服务器，有 GPU 时）：
#     cd /tmp/pycharm_project_10ae9e2e
#     bash manu/tracking/lookahead/server_overnight/run_fastinfer_nogmc24_sweep.sh   # 默认即 no-GMC 口径
#     CACHE=/path/to/nogmc24/ms_cache bash manu/tracking/lookahead/server_overnight/run_fastinfer_nogmc24_sweep.sh
#     env CACHE=... DELTAS="3" bash .../run_fastinfer_nogmc24_sweep.sh
#
#   ★ 幂等：已有 arms.json 的 Δ 跳过。
# ============================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
PY="${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}"
GT="${GT:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
FRAMES="${FRAMES:-/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg}"
FC="${FRAMES_CACHE:-/mnt/data/siping/frame_cache_lq}"
PARAMS="${PARAMS:-manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json}"
DELIVERY_SH="$REPO/manu/tracking/lookahead/run_delivery.sh"
OUT="${OUT:-$REPO/runs/lookahead/fi_nogmc24_sweep_runs/nogmc}"
THREADS="${THREADS:-12}"
DELTAS="${DELTAS:-2 3 4 8}"
CACHE="${CACHE:-/mnt/data/siping/runs/nogmc_t3/swa/ms_cache_24seq}"

export PYTHONPATH="$REPO"
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

cd "$REPO" || exit 2
mkdir -p "$OUT"

CFG_ASSIGN="$(awk '/^CFG="/ {f=1} f {print} f && /"[[:space:]]*$/ {exit}' "$DELIVERY_SH")"
[ -n "$CFG_ASSIGN" ] || { echo "[FATAL] 无法从 $DELIVERY_SH 解析 CFG"; exit 3; }
CFG=""
eval "$CFG_ASSIGN"
[ -n "$CFG" ] || { echo "[FATAL] CFG 解析为空"; exit 3; }
for _must in "--replay-roi-selfpred-max-px 16" "--replay-rollback-min-peak 0.90" \
             "--replay-polarity auto" "--replay-xcorr-uniq-gate 0.6" "--replay-mode perframe" \
             "--replay-abstain-peak-below 0.30" "--replay-rollback-max-frames 16"; do
  case " $CFG " in
    *" $_must "*) ;;
    *) echo "[FATAL] CFG 缺少必需旋钮 '$_must' ⇒ 解析被截断，拒绝跑"; exit 3 ;;
  esac
done
echo "[$(date '+%F %T')] CFG（源自 run_delivery.sh，${#CFG} 字符）: $CFG"

FRAME_ARGS=""
if [ -d "$FC" ]; then FRAME_ARGS="--frames-cache $FC"; echo "[$(date '+%F %T')] 帧缓存 memmap: $FC"; fi

NPKL=$(ls "$CACHE"/*.pkl 2>/dev/null | wc -l)
echo "[$(date '+%F %T')] 24 序 no-GMC 缓存 $CACHE pkl=$NPKL（期望 24）"
if [ "$NPKL" -ne 24 ]; then echo "[FATAL] 24 序缓存 pkl 数=$NPKL ≠ 24"; exit 3; fi

for d in $DELTAS; do
  O="$OUT/d$d"
  if [ -f "$O/arms.json" ]; then echo "[$(date '+%F %T')] SKIP Δ=$d（已有产物）"; continue; fi
  mkdir -p "$O"
  echo "[$(date '+%F %T')] ===== RUN 24seq no-GMC Δ=$d ====="
  t0=$(date +%s)
  "$PY" manu/tracking/lookahead/lookahead_longquan.py \
    --cache-root "$CACHE" --labels-root "$GT" --frames-root "$FRAMES" $FRAME_ARGS \
    --params-json "$PARAMS" --all-seqs \
    --lookahead replay --arms original,stale,kinematic,replay \
    --delta "$d" $CFG \
    --out-json "$O/arms.json" --out-preds-dir "$O/preds" \
    > "$O/run.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] DONE 24seq no-GMC Δ=$d rc=$rc 用时 $(( $(date +%s) - t0 ))s" | tee -a "$OUT/progress.txt"
  [ "$rc" -ne 0 ] && tail -25 "$O/run.log"
done

echo "[$(date '+%F %T')] ALL DONE. 产物在 $OUT"
find "$OUT" -name arms.json | sort
