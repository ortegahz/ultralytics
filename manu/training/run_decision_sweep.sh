#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# 服务器端自主链：**决策层旋钮扫参**（零 GPU / 纯 CPU）。
#
#   等 GPU 链不冲突 → 跑 K1 → 跑 K2/K3 → 出汇总
#
# 做什么
# ------
# 在**已落盘**的多尺度热图缓存（ms_cache）上扫决策层旋钮：
#   K1   (th_base × dedup_radius) 网格 —— 单帧臂上从未测过；
#   K2/K3 判别式候选门（area_min × sum_min × th_base）—— 用缓存里存了但决策层丢掉的
#         area / sums / tag 字段区分「弱但真」与「噪声」。
# 三个臂同时扫（成组应用同一组参数）：
#   nogmc_t0049  no-GMC 基线臂（trial_0049 epoch20）
#   cs_t0008p2   no-GMC 交叉种子 SWA 臂（T1 合规那条）
#   rc2_gmc_t0080  RC2 带 GMC 参照臂（只读观测，用来验证「网格是平的」并复现 4 个已知锚点）
#
# 不做什么（硬边界）
# ----------------
#   * **不启动任何推理 / 缓存重建 / 训练 / 门禁脚本** —— 本链只读 pkl，纯 CPU；
#   * **不碰冻结参照系** `/mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json`；
#   * **不删文件**：需要清理时 mv 到带时间戳的 _trash_*；
#   * **不改** auto_epoch_gate_check.py / system_sota_longquan.py / eval_multiscale_cache_th.py。
#
# 对齐 manu/memory/rules.md 铁律十一
# ----------------------------------
#   * 墙钟**硬截止**（UNTIL，默认 "2026-11-05 07:00"）：到点停止，已跑出的结果照常落盘；
#   * **每个子步骤独立 rc 可查**（rc/stepNN.rc + rc/_steps.tsv），外层 rc=0 不代表每个产物都是新的；
#   * 产物落盘后**核对时间戳/字节数**，避免「静默停在旧档」；
#   * **幂等**：每次运行写独立时间戳目录，重复运行不覆盖旧产物；封存前先 chmod 0644（封存副本是 0444）。
#
# 用法
# ----
#   bash manu/training/run_decision_sweep.sh
#   UNTIL="2026-11-05 04:00" WAIT_BUDGET_MIN=30 bash manu/training/run_decision_sweep.sh
#   SKIP_WAIT=1 bash manu/training/run_decision_sweep.sh      # 不等 GPU 链（纯 CPU，低优先级）

set -uo pipefail

# ----------------------------------------------------------------------------------------------
# 配置（全部可用环境变量覆盖）
# ----------------------------------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
CONDA_SH="${CONDA_SH:-/home/huangzhe/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-uav}"
LABELS_ROOT="${LABELS_ROOT:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
RUNS_ROOT="${RUNS_ROOT:-/mnt/data/siping/runs/nogmc_t3}"
OUT_ROOT="${OUT_ROOT:-${RUNS_ROOT}/decision_sweep}"
UNTIL="${UNTIL:-2026-11-05 07:00}"
WAIT_BUDGET_MIN="${WAIT_BUDGET_MIN:-180}"
POLL_SEC="${POLL_SEC:-60}"
NICENESS="${NICENESS:-10}"
SKIP_WAIT="${SKIP_WAIT:-0}"
SWEEP_REL="manu/evaluation/sweep_decision_layer.py"

T3_SEQ="VIDEO00007_19700101_001324"
T2_SEQS="VIDEO00009_19700101_001531,VIDEO00029_19700101_014053,VIDEO00033_19700101_015000,VIDEO00034_19700101_015052,VIDEO00006_19700101_000703,VIDEO00008_19700101_001307,VIDEO00007_19700101_000818"

ARM_T0049="${RUNS_ROOT}/trials/trial_0049/gate/epoch20/ms_cache"
ARM_CST8P2="${RUNS_ROOT}/swa/gate/cs_t0008p2/ms_cache"
ARM_RC2="${PROJECT_ROOT}/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache"

# GPU 链的进程特征：训练臂 / 缓存重建 / 门禁。命中即认为「GPU 链还在跑」。
GPU_CHAIN_PATTERN="${GPU_CHAIN_PATTERN:-auto_epoch_gate_check.py|multiscale_heatmap_cache.py|train.py|train_ddp.py|optuna_mix_ratio_search.py|run_nogmc_tail_autorun.sh|guard_nogmc_t3_autorun.sh}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_ROOT}/${STAMP}"
RC_DIR="${OUT_DIR}/rc"
LOG_DIR="${OUT_DIR}/logs"
SWEEP_DIR="${OUT_DIR}/sweep"
SEAL_DIR="${OUT_DIR}/SEALED"
mkdir -p "${RC_DIR}" "${LOG_DIR}" "${SWEEP_DIR}" "${SEAL_DIR}"

# ----------------------------------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------------------------------
log() { printf '[%s] %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$*"; }

record_rc() {  # $1 = 步骤名, $2 = rc
  printf '%s\n' "$2" > "${RC_DIR}/$1.rc"
  printf '%s\t%s\t%s\n' "$1" "$2" "$(date +'%Y-%m-%d %H:%M:%S')" >> "${RC_DIR}/_steps.tsv"
  log "[STEP] $1 rc=$2"
}

now_epoch() { date +%s; }

deadline_epoch() { date -d "$1" +%s 2>/dev/null || echo 0; }

gpu_chain_busy() { pgrep -f "${GPU_CHAIN_PATTERN}" >/dev/null 2>&1; }

gpu_chain_snapshot() {
  pgrep -af "${GPU_CHAIN_PATTERN}" 2>/dev/null | head -20
}

seal_file() {  # $1 = 待封存文件（幂等：0444 的旧副本先放开权限）
  local src="$1"
  [ -f "$src" ] || return 1
  local dst="${SEAL_DIR}/$(basename "$src")"
  [ -f "$dst" ] && chmod 0644 "$dst"
  cp -f "$src" "$dst" || return 1
  chmod 0444 "$dst" || return 1
  log "[SEAL] $src -> $dst ($(stat -c%s "$dst") bytes, $(stat -c%y "$dst" | cut -d. -f1))"
  return 0
}

# ----------------------------------------------------------------------------------------------
# 步骤 0：preflight
# ----------------------------------------------------------------------------------------------
log "=== 决策层扫参自主链启动 ==="
log "PROJECT_ROOT=${PROJECT_ROOT}"
log "OUT_DIR=${OUT_DIR}"
log "UNTIL=${UNTIL}"
log "WAIT_BUDGET_MIN=${WAIT_BUDGET_MIN}"

preflight_rc=0
for path in "${CONDA_SH}" "${LABELS_ROOT}" "${SWEEP_REL}" "${PROJECT_ROOT}/${SWEEP_REL}"; do
  if [ ! -e "$path" ]; then
    log "[PREFLIGHT] 缺失: ${path}"
    preflight_rc=1
  fi
done
for arm_dir in "${ARM_T0049}" "${ARM_CST8P2}" "${ARM_RC2}"; do
  if [ ! -d "$arm_dir" ]; then
    log "[PREFLIGHT] 臂缓存缺失: ${arm_dir}"
    preflight_rc=1
  else
    n_pkl=$(find "$arm_dir" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l)
    log "[PREFLIGHT] 臂缓存 ${arm_dir} -> ${n_pkl} 个 pkl"
    if [ "$n_pkl" -lt 8 ]; then
      log "[PREFLIGHT] 警告: 该臂 pkl 少于 8 个（golden 7 + blind 1），评测会缺序列"
    fi
  fi
done
record_rc 00_preflight "${preflight_rc}"
if [ "${preflight_rc}" -ne 0 ]; then
  log "[FATAL] preflight 失败，链中止"
  exit 3
fi

# ----------------------------------------------------------------------------------------------
# 步骤 1：等 GPU 链不冲突
# ----------------------------------------------------------------------------------------------
wait_start="$(date +'%Y-%m-%d %H:%M:%S')"
if [ "${SKIP_WAIT}" = "1" ]; then
  log "[WAIT] SKIP_WAIT=1，跳过等待"
  wait_rc=0
else
  budget_end=$(( $(now_epoch) + WAIT_BUDGET_MIN * 60 ))
  waited=0
  while gpu_chain_busy; do
    if [ "$(now_epoch)" -ge "${budget_end}" ]; then
      log "[WAIT] 等待预算 ${WAIT_BUDGET_MIN} 分钟用尽；本链为纯 CPU 且以 nice ${NICENESS} 运行，继续"
      break
    fi
    if [ "$(deadline_epoch "${UNTIL}")" -ne 0 ] && [ "$(now_epoch)" -ge "$(deadline_epoch "${UNTIL}")" ]; then
      log "[WAIT] 已到墙钟截止 ${UNTIL}，不再等待"
      break
    fi
    log "[WAIT] GPU 链仍在运行，等待中（累计 ${waited}s）"
    sleep "${POLL_SEC}"
    waited=$(( waited + POLL_SEC ))
  done
  wait_rc=0
fi
wait_end="$(date +'%Y-%m-%d %H:%M:%S')"
{
  echo "wait_start=${wait_start}"
  echo "wait_end=${wait_end}"
  echo "gpu_chain_pattern=${GPU_CHAIN_PATTERN}"
  echo "gpu_chain_snapshot_at_start:"
  gpu_chain_snapshot
} > "${OUT_DIR}/gpu_wait.txt"
log "[WAIT] 等待区间 ${wait_start} -> ${wait_end}（快照见 gpu_wait.txt）"
record_rc 01_wait_gpu "${wait_rc}"

# ----------------------------------------------------------------------------------------------
# 步骤 2：跑扫参（K1 → K2/K3，同一脚本同一次调用）
# ----------------------------------------------------------------------------------------------
source "${CONDA_SH}" && conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}" || { log "[FATAL] 进不去 PROJECT_ROOT=${PROJECT_ROOT}"; exit 3; }
if ! python -c "import numpy, scipy" >/dev/null 2>&1; then
  log "[FATAL] conda 环境 ${CONDA_ENV} 里没有 numpy/scipy，链中止"
  exit 3
fi
log "[ENV] python=$(command -v python) numpy=$(python -c 'import numpy;print(numpy.__version__)')"

# 两个口径各跑一遍：
#   oos_primary  能量分位表的总体**排除盲测序列** ⇒ T3 读数对自己 out-of-sample（主口径）
#   arm_scope   能量分位表取该臂整个 cache_root ⇒ 敏感性对照（分位表含 T3 自身）
SCOPES="${SCOPES:-oos_primary arm_scope}"

run_sweep() {  # $1 = 口径名, $2 = --sum-quantile-scope 取值
  local scope_name="$1" scope_value="$2" out="${OUT_DIR}/sweep_${scope_name}"
  mkdir -p "${out}"
  SWEEP_CMD=(
    python "${PROJECT_ROOT}/${SWEEP_REL}"
    --arm "nogmc_t0049=${ARM_T0049}"
    --arm "cs_t0008p2=${ARM_CST8P2}"
    --arm "rc2_gmc_t0080=${ARM_RC2}"
    --track "t3=${T3_SEQ}"
    --track "t2=${T2_SEQS}"
    --labels-root "${LABELS_ROOT}"
    --out-dir "${out}"
    --stage all
    --primary-track t3
    --gated-track t2
    --gate-f1-min 0.985
    --gate-fp-max 112
    --sum-quantile-scope "${scope_value}"
    --until "${UNTIL}"
    --check-json "nogmc_t0049:t3=${ARM_T0049%/ms_cache}/track3_blind.json"
    --check-json "nogmc_t0049:t2=${ARM_T0049%/ms_cache}/track2_golden.json"
    --check-json "cs_t0008p2:t3=${ARM_CST8P2%/ms_cache}/track3_blind.json"
    --check-json "cs_t0008p2:t2=${ARM_CST8P2%/ms_cache}/track2_golden.json"
    --check-json "rc2_gmc_t0080:t3=${ARM_RC2%/ms_cache}/track3_blind.json"
    --check-json "rc2_gmc_t0080:t2=${ARM_RC2%/ms_cache}/track2_golden.json"
  )
  log "[SWEEP:${scope_name}] ${SWEEP_CMD[*]}"
  {
    echo "$ ${SWEEP_CMD[*]}"
    echo "started_at=$(date +'%Y-%m-%d %H:%M:%S')"
  } > "${LOG_DIR}/sweep_${scope_name}_cmd.txt"
  nice -n "${NICENESS}" "${SWEEP_CMD[@]}" > "${LOG_DIR}/sweep_${scope_name}.log" 2>&1
  return $?
}

sweep_rc=0
step_index=2
for scope_name in ${SCOPES}; do
  case "${scope_name}" in
    oos_primary) scope_value="exclude-primary" ;;
    arm_scope)   scope_value="arm" ;;
    *)           scope_value="${scope_name}" ;;
  esac
  run_sweep "${scope_name}" "${scope_value}"
  rc=$?
  record_rc "$(printf '%02d' "${step_index}")_sweep_${scope_name}" "${rc}"
  log "[SWEEP:${scope_name}] 结束 rc=${rc}，日志 ${LOG_DIR}/sweep_${scope_name}.log"
  [ "${rc}" -ne 0 ] && sweep_rc="${rc}"
  step_index=$(( step_index + 1 ))
done

# ----------------------------------------------------------------------------------------------
# 步骤 3：核对产物（时间戳/字节数，防「静默停在旧档」）
# ----------------------------------------------------------------------------------------------
verify_rc=0
for scope_name in ${SCOPES}; do
  for artefact in sweep_k1.csv sweep_k2k3.csv sweep_all.csv sweep_results.json SUMMARY.md; do
    path="${OUT_DIR}/sweep_${scope_name}/${artefact}"
    if [ ! -s "$path" ]; then
      log "[VERIFY] 缺失或为空: ${path}"
      verify_rc=1
      continue
    fi
    log "[VERIFY] ${scope_name}/${artefact}  bytes=$(stat -c%s "$path")  mtime=$(stat -c%y "$path" | cut -d. -f1)"
  done
done
record_rc "$(printf '%02d' "${step_index}")_verify" "${verify_rc}"
step_index=$(( step_index + 1 ))

# ----------------------------------------------------------------------------------------------
# 步骤 4：封存（幂等）+ 出汇总
# ----------------------------------------------------------------------------------------------
seal_rc=0
for scope_name in ${SCOPES}; do
  for artefact in sweep_k1.csv sweep_k2k3.csv sweep_all.csv sweep_results.json SUMMARY.md; do
    seal_file "${OUT_DIR}/sweep_${scope_name}/${artefact}" || seal_rc=1
  done
done
record_rc "$(printf '%02d' "${step_index}")_seal" "${seal_rc}"

log "============================================================"
log " 决策层扫参链结束"
log " 产物目录: ${OUT_DIR}"
log " 密封副本: ${SEAL_DIR}"
log " 各步 rc  : ${RC_DIR}/_steps.tsv"
log "============================================================"

for scope_name in ${SCOPES}; do
  if [ -s "${OUT_DIR}/sweep_${scope_name}/SUMMARY.md" ]; then
    log ""
    log "---------------- SUMMARY.md[${scope_name}]（节选） ----------------"
    head -90 "${OUT_DIR}/sweep_${scope_name}/SUMMARY.md"
    log "------------------------------------------------"
  fi
done

for scope_name in ${SCOPES}; do
  python - "${OUT_DIR}/sweep_${scope_name}/sweep_results.json" "${scope_name}" <<'PYANSWER' 2>/dev/null || true
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
meta = payload["meta"]
rows = payload["rows"]
primary = meta["primary_track"]
print(f"---------------- 关键答案 [{sys.argv[2]}] ----------------")
print(f"[能量分位口径] scope={meta.get('sum_quantile_scope')} 排除={meta.get('sum_quantile_excluded')}")
for check in meta.get("self_check", []):
    print(f"[自检] {check['arm']}/{check['track']}: {check['verdict']}  "
          f"frozen TP/FP/FN/F1={check['expected_tp']}/{check['expected_fp']}/{check['expected_fn']}/"
          f"{check['expected_f1']:.6f}")
for arm, bases in meta.get("baselines", {}).items():
    for track, base in bases.items():
        print(f"[基线] {arm}/{track}: TP={base['tp']} FP={base['fp']} FN={base['fn']} F1={base['f1']:.6f}")
k1 = [r for r in rows if r["stage"] == "k1"]
for arm in sorted({r["arm"] for r in k1}):
    best = max((r for r in k1 if r["arm"] == arm), key=lambda r: r[f"{primary}_f1"])
    print(f"[K1最优] {arm}: radius={best['radius']:g} th={best['th_base']:g} "
          f"{primary}_F1={best[f'{primary}_f1']:.6f} ΔF1={best.get(f'd_{primary}_f1', 0)*100:+.4f} pp "
          f"TP/FP/FN={best[f'{primary}_tp']}/{best[f'{primary}_fp']}/{best[f'{primary}_fn']}")
k2 = [r for r in rows if r["stage"].startswith("k2k3")]
for arm in sorted({r["arm"] for r in k2}):
    arm_rows = [r for r in k2 if r["arm"] == arm]
    best = max(arm_rows, key=lambda r: r[f"{primary}_f1"])
    improved = [r for r in arm_rows if r.get(f"d_{primary}_f1", -1) > 0]
    strict = [r for r in arm_rows if r.get("bar_strict_1p21_pass")]
    gated = [r for r in arm_rows if r.get("t2_gate_pass")]
    both = [r for r in arm_rows if r.get("t2_gate_pass") and r.get(f"d_{primary}_f1", -1) > 0]
    print(f"[K2K3最优] {arm}: {primary}_F1={best[f'{primary}_f1']:.6f} "
          f"ΔF1={best.get(f'd_{primary}_f1', 0)*100:+.4f} pp "
          f"(gate_stage={best['gate_stage']} area_min={best['area_min']} "
          f"sum_mode={best['sum_mode']} sum={best['sum_value']:g} th={best['th_base']:g}) | "
          f"净赚比={best.get('bar_fp_per_tp_cost', float('nan')):.3f} | "
          f"ΔF1>0: {len(improved)}/{len(arm_rows)} | 过1.21门槛: {len(strict)}/{len(arm_rows)} | "
          f"T2仍过门禁: {len(gated)}/{len(arm_rows)} | 同时满足ΔF1>0且T2过门禁: {len(both)}/{len(arm_rows)}")
print("----------------------------------------")
PYANSWER
done

exit $(( sweep_rc != 0 ? sweep_rc : (verify_rc != 0 ? verify_rc : 0) ))
