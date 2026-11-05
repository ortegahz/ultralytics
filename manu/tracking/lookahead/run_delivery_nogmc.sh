#!/usr/bin/env bash
# ============================================================
# ★ 交付档一键复现 —— **no-GMC 推荐档**（`cs_t0043` · Δ=3 · 出框门 m=6）
#
#   推荐档（结论出处：runs/lookahead/final/nogmc_fi_20261104/FINAL_REPORT.md §12）：
#     模型包   = `cs_t0043`（**no-GMC 特征语义**）
#     假设延时 = **Δ=3**（120 ms @25 Hz）
#     交付臂   = `replay`
#     出框门   = `--replay-fov-margin-px 6`
#     实测（24 序官方口径，replay 臂，current_frame.overall）：
#       F1 **94.8094%** · Recall **92.8507%** · Precision **96.8525%** · FAR **0.024337** 帧⁻¹
#       hit@5px **86.9718%** · CLE 池化 p95 **7.0663**
#     交付三门（F1≥0.88 / Recall≥0.85 / FAR≤0.030）= **PASS**
#
#   ★ 为什么**单开这个入口**、而不是改 `run_delivery.sh`：
#     `run_delivery.sh` 的 `CFG=` 块是**全仓 CFG 单一真值**，被 **12 个脚本**用
#     `awk '/^CFG="/ {f=1} f {print} f && /"[[:space:]]*$/ {exit}'` 逐字解析
#     （run_fastinfer_24seq_sweep.sh / run_fastinfer_24seq_nogmc24_sweep.sh /
#      run_fastinfer_nogmc24_sweep.sh / run_fov_gate_scan.sh / run_knob_scan_d3*.sh /
#      run_osd_3arm_nogmc.sh / run_reg_oracle_dump.sh / run_rollback_gate_scan.sh /
#      run_workpoint_audit.sh …）⇒ 动它（哪怕只多一个旋钮）= 连锁改口径 = 大面积断裂。
#     而且老入口指向的是 **rc2 / GMC 语义缓存**、CFG 里**没有出框门**
#     ⇒ 照 `run_delivery.sh` 跑**复现不出**上面这个推荐档。
#     本脚本是**并列**入口：老入口一个字不动，CFG 仍从它取（唯一真值不分裂）。
#
#   ★ 与参考产物的**逐位门禁**：
#     跑完后用一段内嵌 `python3 -` 把**本脚本新产物**的 replay 臂指标，与**已验收的参考产物**
#     `runs/lookahead/fovgate24nogmc/runs/cs_t0043/d3/fov__m6/arms.json`（可用 `REF_ARMS` 覆盖）
#     做**逐位相等**比对（严格相等、无容差）：
#       current_frame.overall : f1 / recall / precision / far_per_frame / tp / fp / fn / det / gt
#       servo.overall         : hit_rate_5px / cle_pooled_p50 / cle_pooled_p95 / cle_pooled_p99
#                               / cle_pooled_max / cle_pooled_samples
#     不一致 ⇒ **exit 非 0**（数字对不上 = 口径漂了，绝不能当复现成功）。
#     参考产物缺失 ⇒ 明确告警 + **跳过**该门禁（不因此 exit 非 0），日志写清
#     「参考产物缺失 ⇒ 未做逐位门禁」。
#
#   ★ 怎么复现**严格对比报告**：
#     本脚本产物落在 `<OUT>/runs/<model>/d<Δ>/fov__m<m>/arms.json` —— 与
#     `runs/lookahead/fovgate24nogmc/` **同一目录约定** ⇒ 可直接喂给报告生成器（它只读产物、不重算）：
#       python manu/tracking/lookahead/make_fastinfer_final_report.py \
#         --fovgate-24-nogmc "$OUT" \
#         --out-md  runs/lookahead/final/<批次>/FINAL_REPORT.md \
#         --out-json runs/lookahead/final/<批次>/final_report.json
#
#   ★ 口径纪律：**模型包与缓存语义必须配对**（模型包在建缓存那一步经 `--hm-weights` 烤进 pkl，
#     本脚本不再单独接 ckpt）⇒ 换 CACHE 就是换整套 {特征语义, 模型包} 口径；
#     拿 no-GMC 模型包去读 GMC 语义的缓存（或反之）⇒ 口径串味、结论作废。
#     绝不能用 rc2 的 GMC 缓存（`.../trial_0080/lq_eval_fpv8/ms_cache`）跑这个交付档。
#
#   用法（服务器：cwd=/tmp/pycharm_project_10ae9e2e，conda env `uav`）：
#     bash manu/tracking/lookahead/run_delivery_nogmc.sh              # 真跑（重活，服务器）
#     DRY_RUN=1 bash manu/tracking/lookahead/run_delivery_nogmc.sh    # 只打印命令行，不跑 python
#     bash manu/tracking/lookahead/run_delivery_nogmc.sh --dry-run   # 同上
#
#   ★ 幂等：目标 `arms.json` 已存在且非空 ⇒ 打印 `SKIP`、**不重跑重活**；
#     随后的逐位门禁**照跑**（只读 JSON，秒级）⇒ 门禁不过仍 exit 非 0
#     （缓存里的产物若已漂移，必须报出来，不能被 SKIP 掩盖）。
#
#   ⚠️ 路径按**服务器**写（host 侧请覆盖 CACHE / GT / FRAMES / REPO）。
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
用法: bash manu/tracking/lookahead/run_delivery_nogmc.sh [--dry-run]

  --dry-run   只打印最终要执行的完整命令行（CFG 已展开），不执行任何 python
              （等价于环境变量 DRY_RUN=1）

可覆盖的环境变量（默认值见脚本内注释）：
  REPO / PY / CACHE / GT / FRAMES / FRAMES_CACHE / PARAMS
  DELTA(=3) / MARGIN(=6) / MODEL(=cs_t0043) / OUT / REF_ARMS / THREADS / DRY_RUN
USAGE
}

DRY_RUN="${DRY_RUN:-0}"
for _arg in "$@"; do
  case "$_arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[FATAL] 未知参数: $_arg（支持 --dry-run / --help）"; usage; exit 2 ;;
  esac
done

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
PY="${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}"
PY3="${PY3:-python3}"
CACHE="${CACHE:-/mnt/data/siping/runs/nogmc_t3/swa/ms_cache_24seq}"
GT="${GT:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
FRAMES="${FRAMES:-/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg}"
FC="${FRAMES_CACHE:-/mnt/data/siping/frame_cache_lq}"
PARAMS="${PARAMS:-manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json}"
DELTA="${DELTA:-3}"
MARGIN="${MARGIN:-6}"
MODEL="${MODEL:-cs_t0043}"
OUT="${OUT:-$REPO/runs/lookahead/delivery_nogmc_d3}"
THREADS="${THREADS:-12}"
# CFG 唯一真值 = 与本脚本同目录的 run_delivery.sh（服务器上仍是 $REPO/manu/tracking/lookahead/run_delivery.sh）
DELIVERY_SH="${DELIVERY_SH:-$SCRIPT_DIR/run_delivery.sh}"
# 已验收的参考产物（逐位门禁的对标基准；可用 REF_ARMS 覆盖）
REF_ARMS="${REF_ARMS:-$REPO/runs/lookahead/fovgate24nogmc/runs/$MODEL/d$DELTA/fov__m$MARGIN/arms.json}"

export PYTHONPATH="$REPO"
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

O="$OUT/runs/$MODEL/d$DELTA/fov__m$MARGIN"
ARMS="$O/arms.json"
PREDS="$O/preds"

# ---- 唯一 CFG 真值：直接从 run_delivery.sh 取（多行续行 ⇒ awk 取整块再 eval）----
if [ ! -f "$DELIVERY_SH" ]; then echo "[FATAL] 找不到 $DELIVERY_SH"; exit 3; fi
CFG_ASSIGN="$(awk '/^CFG="/ {f=1} f {print} f && /"[[:space:]]*$/ {exit}' "$DELIVERY_SH")"
if [ -z "$CFG_ASSIGN" ]; then echo "[FATAL] 无法从 $DELIVERY_SH 解析 CFG"; exit 3; fi
CFG=""
eval "$CFG_ASSIGN"
if [ -z "$CFG" ]; then echo "[FATAL] CFG 解析为空"; exit 3; fi
for _must in "--replay-roi-selfpred-max-px 16" "--replay-rollback-min-peak 0.90" \
             "--replay-polarity auto" "--replay-xcorr-uniq-gate 0.6" "--replay-mode perframe" \
             "--replay-abstain-peak-below 0.30" "--replay-rollback-max-frames 16"; do
  case " $CFG " in
    *" $_must "*) ;;
    *) echo "[FATAL] CFG 缺少必需旋钮 '$_must' ⇒ 解析被截断，拒绝跑"; exit 3 ;;
  esac
done
echo "[$(date '+%F %T')] CFG（源自 run_delivery.sh，${#CFG} 字符）: $CFG"
# 出框门**不进 CFG 块**（与 run_fov_gate_scan.sh 一致）⇒ 单开 --replay-fov-margin-px 传参
case " $CFG " in
  *" --replay-fov-margin-px "*) echo "[FATAL] CFG 里混进了出框门 ⇒ 会与 --replay-fov-margin-px 打架，拒绝跑"; exit 3 ;;
esac

FRAME_ARGS=""
if [ -d "$FC" ]; then
  FRAME_ARGS="--frames-cache $FC"
  echo "[$(date '+%F %T')] 帧缓存 memmap: $FC"
else
  echo "[$(date '+%F %T')] ⚠️ 帧缓存不存在 $FC ⇒ 走原始帧（慢）"
fi

NPKL=$(ls "$CACHE"/*.pkl 2>/dev/null | wc -l)
echo "[$(date '+%F %T')] 24 序 no-GMC 缓存 $CACHE pkl=$NPKL（期望 24）"
if [ "$NPKL" -ne 24 ]; then
  if [ "$DRY_RUN" = "1" ]; then
    echo "[$(date '+%F %T')] [DRY-RUN] 缓存 pkl 数=$NPKL ≠ 24 ⇒ 干跑模式不 exit（服务器上应为 24）"
  else
    echo "[FATAL] 24 序缓存 pkl 数=$NPKL ≠ 24"; exit 3
  fi
fi
case "$CACHE" in
  *trial_0080*) echo "[$(date '+%F %T')] ⚠️ CACHE 路径含 trial_0080 ⇒ 可能是 rc2 / GMC 语义缓存，交付档必须用 no-GMC 缓存" ;;
esac

if [ -d "$REPO" ]; then
  cd "$REPO" || exit 2
else
  if [ "$DRY_RUN" = "1" ]; then
    echo "[$(date '+%F %T')] [DRY-RUN] REPO=$REPO 不存在（host 侧）⇒ 不 cd，用当前目录拼命令"
  else
    echo "[FATAL] REPO=$REPO 不存在 ⇒ 请在服务器上跑（/tmp/pycharm_project_10ae9e2e）"; exit 2
  fi
fi

# ---- 命令行（干跑打印的就是这一条，逐 token 展开 CFG）----
CMD=("$PY" manu/tracking/lookahead/lookahead_longquan.py
     --cache-root "$CACHE" --labels-root "$GT" --frames-root "$FRAMES")
if [ -n "$FRAME_ARGS" ]; then CMD+=(--frames-cache "$FC"); fi
CMD+=(--params-json "$PARAMS" --all-seqs
      --lookahead replay --arms original,stale,kinematic,replay
      --delta "$DELTA")
read -r -a _CFG_ARGS <<< "$CFG"
CMD+=("${_CFG_ARGS[@]}")
CMD+=(--replay-fov-margin-px "$MARGIN"
      --out-json "$ARMS" --out-preds-dir "$PREDS")

echo "[$(date '+%F %T')] 交付档 = 模型包 $MODEL · Δ=$DELTA · 出框门 m=$MARGIN · 臂 replay"
echo "[$(date '+%F %T')] 产物 = $ARMS"
echo "[$(date '+%F %T')] 逐位门禁参考 = $REF_ARMS"

if [ "$DRY_RUN" = "1" ]; then
  echo "[$(date '+%F %T')] ===== DRY-RUN：完整命令行（不执行）====="
  printf '%s\n' "${CMD[*]}"
  echo "[$(date '+%F %T')] ===== DRY-RUN 结束（未执行任何 python，未创建任何目录）====="
  exit 0
fi

if [ -s "$ARMS" ]; then
  echo "[$(date '+%F %T')] SKIP $MODEL Δ=$DELTA m=$MARGIN（已有产物，不重跑；逐位门禁照跑）"
else
  mkdir -p "$O"
  echo "[$(date '+%F %T')] ===== RUN 24seq no-GMC $MODEL Δ=$DELTA m=$MARGIN ====="
  t0=$(date +%s)
  "${CMD[@]}" > "$O/run.log" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "[$(date '+%F %T')] DONE 24seq no-GMC $MODEL Δ=$DELTA m=$MARGIN rc=$rc 用时 $((t1-t0))s" | tee -a "$OUT/progress.txt"
  if [ $rc -ne 0 ]; then
    tail -25 "$O/run.log"
    echo "[FATAL] 评测失败 rc=$rc ⇒ 无产物可比"; exit 5
  fi
fi

# ---- ★ 逐位门禁：新产物 vs 已验收参考产物（严格相等，无容差）----
"$PY3" - "$ARMS" "$REF_ARMS" <<'PY'
import json
import os
import sys

new_path, ref_path = sys.argv[1], sys.argv[2]
CUR_FIELDS = ("f1", "recall", "precision", "far_per_frame", "tp", "fp", "fn", "det", "gt")
SRV_FIELDS = ("hit_rate_5px", "cle_pooled_p50", "cle_pooled_p95", "cle_pooled_p99",
              "cle_pooled_max", "cle_pooled_samples")
PCT = {"f1": "F1%", "recall": "Recall%", "precision": "Precision%", "hit_rate_5px": "hit@5px%"}


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def replay_rows(doc):
    arm = doc["arms"]["replay"]
    cur = arm["current_frame"]["overall"]
    srv = arm["servo"]["overall"]
    rows = {k: cur.get(k) for k in CUR_FIELDS}
    rows.update({k: srv.get(k) for k in SRV_FIELDS})
    return rows


def show(key, val):
    if val is None:
        return "None"
    if key in PCT:
        return "%.4f%%" % (val * 100.0)
    return repr(val)


if not os.path.isfile(new_path):
    print("[GATE][FAIL] 新产物缺失: %s" % new_path)
    sys.exit(3)
if not os.path.isfile(ref_path):
    print("[GATE][WARN] 参考产物缺失: %s" % ref_path)
    print("[GATE][WARN] 参考产物缺失 ⇒ 未做逐位门禁（跳过，不影响退出码）")
    sys.exit(0)

try:
    new = replay_rows(load(new_path))
    ref = replay_rows(load(ref_path))
except (KeyError, TypeError, ValueError) as exc:
    print("[GATE][FAIL] 产物解析失败: %r" % (exc,))
    sys.exit(3)

print("[GATE] 新产物  : %s" % new_path)
print("[GATE] 参考产物: %s" % ref_path)
print("[GATE] 逐位比对 replay 臂 %d 个字段（严格相等，无容差）" % len(new))
print("   %-20s %26s %26s  判定" % ("字段", "新产物", "参考产物"))
bad = []
for key in list(CUR_FIELDS) + list(SRV_FIELDS):
    a, b = new.get(key), ref.get(key)
    same = a == b
    if not same:
        bad.append((key, a, b))
    print("   %-20s %26s %26s  %s" % (key, show(key, a), show(key, b), "OK" if same else "DIFF"))

if bad:
    print("[GATE][FAIL] 逐位不一致 %d 处：" % len(bad))
    for key, a, b in bad:
        print("    %s: new=%r ref=%r" % (key, a, b))
    sys.exit(3)
print("[GATE][PASS] replay 臂 %d 字段与参考产物逐位一致" % len(new))

f1, rec, far = new.get("f1"), new.get("recall"), new.get("far_per_frame")
if None not in (f1, rec, far):
    verdict = "PASS" if (f1 >= 0.88 and rec >= 0.85 and far <= 0.030) else "FAIL"
    print("[GATE] 交付三门（F1≥0.88 / Recall≥0.85 / FAR≤0.030）: "
          "F1 %.4f%% · Recall %.4f%% · FAR %.6f 帧⁻¹ ⇒ %s" % (f1 * 100.0, rec * 100.0, far, verdict))
sys.exit(0)
PY
gate_rc=$?
echo "[$(date '+%F %T')] 逐位门禁 rc=$gate_rc"
exit $gate_rc
