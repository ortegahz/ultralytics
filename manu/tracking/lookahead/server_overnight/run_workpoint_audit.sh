#!/usr/bin/env bash
# ============================================================
# ★ 工作点对齐审计（work-point alignment audit）扫描器
#
#   目的：
#     回答「cs_t0043（no-GMC）在 Δ=3 replay 上的优势是真的，
#            还是只是 v3 工作点（th_base=0.48 / th_ground=0.26）碰巧适合它？」
#     ⇒ 在 rc2 与 cs_t0043 两个模型包上，把 (th_base × th_ground) 工作点网格
#       跑成 FAR–F1 曲线，然后**在同一 FAR 下**做线性插值对比，
#       把「工作点偏差」从「模型能力差异」里剔出去。
#
#   ★ 受控 A/B（唯一变量 = 工作点）：
#     - 同一 8 序列 gate 集 / 同一 28,997 帧 / 同一 GT / 同一帧缓存
#     - 同一插件 CFG：**直接从 run_delivery.sh 读取**（多行反斜杠续行，单行 grep 会截断）
#     - 同一缓存（不重跑推理，纯 CPU 读缓存）
#     - 每个网格点一个**独立的新 params 文件**（v3 主文件只读，绝不就地改）
#
#   ★ 幂等：产物已存在就跳过 ⇒ 可断点续跑。
#
#   用法（服务器）：
#     cd /tmp/pycharm_project_10ae9e2e
#     bash manu/tracking/lookahead/server_overnight/run_workpoint_audit.sh
#     env MODELS="rc2" DELTAS="3" bash .../run_workpoint_audit.sh   # 单模型单 Δ
# ============================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
PY="${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}"
GT="${GT:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
FRAMES="${FRAMES:-/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg}"
FC="${FRAMES_CACHE:-/mnt/data/siping/frame_cache_lq}"
V3_PARAMS="manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json"
WP_DIR="manu/tracking/lookahead/workpoint_params"
OUT="${OUT:-$REPO/runs/lookahead/fi_nogmc/workpoint}"
DELIVERY_SH="$REPO/manu/tracking/lookahead/run_delivery.sh"

MODELS="${MODELS:-rc2 cs_t0043}"
DELTAS="${DELTAS:-3 8}"
TH_BASES="${TH_BASES:-0.40 0.44 0.48 0.52 0.56}"
TH_GROUNDS="${TH_GROUNDS:-0.22 0.26 0.30}"
JOBS="${JOBS:-4}"          # 并行度封顶 4（机器 80 核，但已有 24 序扫描占 ~12 线程）
THREADS="${THREADS:-6}"    # OMP_NUM_THREADS

export PYTHONPATH="$REPO"
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

cd "$REPO" || exit 2
mkdir -p "$OUT" "$WP_DIR" "$OUT/.rc"

# ---- 唯一 CFG 真值：直接从交付脚本读取，杜绝配置漂移 ----------------------
# ⚠️ run_delivery.sh 的 CFG 是**多行反斜杠续行**的赋值，单行 grep 会把它截断。
#    ⇒ 用 awk 取完整赋值块（直到行尾是 " 的那行），再交给 bash 自己 eval。
if [ ! -f "$DELIVERY_SH" ]; then echo "[FATAL] 找不到 $DELIVERY_SH"; exit 3; fi
CFG_ASSIGN="$(awk '/^CFG="/ {f=1} f {print} f && /"[[:space:]]*$/ {exit}' "$DELIVERY_SH")"
if [ -z "$CFG_ASSIGN" ]; then echo "[FATAL] 无法从 $DELIVERY_SH 解析 CFG"; exit 3; fi
CFG=""
eval "$CFG_ASSIGN"
if [ -z "$CFG" ]; then echo "[FATAL] CFG 解析为空"; exit 3; fi
# 防截断断言：交付档的关键旋钮必须全部在位（少一个 ⇒ 解析被截断 ⇒ 拒绝跑）
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
if [ -d "$FC" ]; then
  FRAME_ARGS="--frames-cache $FC"
  echo "[$(date '+%F %T')] 帧缓存 memmap: $FC"
else
  echo "[$(date '+%F %T')] ⚠️ 帧缓存不存在 $FC ⇒ 走原始帧（慢 11.3 ms/帧）"
fi

# ---- 生成网格 params 文件（幂等；v3 主文件只读，绝不就地改）--------------
# 每个网格点一个独立文件，只改 params.th_base 与 params.th_ground，
# 其余参数与 v3 一字不改；_provenance 里补一句来源说明。
if [ ! -f "$V3_PARAMS" ]; then echo "[FATAL] 找不到 v3 主文件 $V3_PARAMS"; exit 3; fi
export WP_V3_PARAMS="$V3_PARAMS" WP_DIR WP_TH_BASES="$TH_BASES" WP_TH_GROUNDS="$TH_GROUNDS"
"$PY" <<'PYGEN'
import json
import os
from pathlib import Path

src = Path(os.environ["WP_V3_PARAMS"])
dst_dir = Path(os.environ["WP_DIR"])
th_bases = os.environ["WP_TH_BASES"].split()
th_grounds = os.environ["WP_TH_GROUNDS"].split()

base = json.loads(src.read_text(encoding="utf-8"))
base_params = base["params"]
for tb in th_bases:
    for tg in th_grounds:
        params = dict(base_params)
        params["th_base"] = float(tb)
        params["th_ground"] = float(tg)
        # 断言：除这两个键外，其余键集合与取值与 v3 完全一致
        others = {k: v for k, v in params.items() if k not in ("th_base", "th_ground")}
        assert others == {k: v for k, v in base_params.items() if k not in ("th_base", "th_ground")}
        payload = dict(base)
        prov = dict(base["_provenance"])
        prov["workpoint_audit_source"] = (
            f"工作点对齐审计网格点：复制自 {src.name}（v3 主文件只读，未改动），"
            f"仅改 params.th_base {base_params['th_base']} -> {float(tb)} 与 "
            f"params.th_ground {base_params['th_ground']} -> {float(tg)}；其余参数一字不改。"
        )
        payload["_provenance"] = prov
        payload["params"] = params
        out = dst_dir / f"wp_thb{tb}_thg{tg}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[PARAMS] {out}")
PYGEN
if [ $? -ne 0 ]; then echo "[FATAL] params 文件生成失败"; exit 3; fi

# ---- 任务表 --------------------------------------------------------------
JOBS_FILE="$OUT/.jobs.txt"
: > "$JOBS_FILE"
for m in $MODELS; do
  for d in $DELTAS; do
    for tb in $TH_BASES; do
      for tg in $TH_GROUNDS; do
        echo "$m $d $tb $tg" >> "$JOBS_FILE"
      done
    done
  done
done
N_TOTAL=$(wc -l < "$JOBS_FILE")
echo "[$(date '+%F %T')] ===== 计划：$N_TOTAL 次（模型=[$MODELS] Δ=[$DELTAS] th_base=[$TH_BASES] th_ground=[$TH_GROUNDS]） ====="

# ---- 并行执行（xargs -P 4）------------------------------------------------
export OUT WP_DIR PY GT FRAMES FRAME_ARGS CFG
export CACHE_RC2="$REPO/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache"
export CACHE_CS_T0043="/mnt/data/siping/runs/nogmc_t3/swa/gate/cs_t0043/ms_cache"

xargs -P "$JOBS" -n 1 -d '\n' bash -c '
  set -u
  read -r m d tb tg <<< "$1"
  case "$m" in
    rc2)      CACHE_ROOT="$CACHE_RC2" ;;
    cs_t0043) CACHE_ROOT="$CACHE_CS_T0043" ;;
    *) echo "[FATAL] 未知模型 $m"; exit 97 ;;
  esac
  odir="$OUT/${m}_d${d}/${tb}_${tg}"
  arms="$odir/arms.json"
  rcfile="$OUT/.rc/${m}_d${d}_${tb}_${tg}.rc"
  if [ -f "$arms" ]; then
    # 幂等跳过：保留已有 rc 记录；若无，则用产物里的 elapsed_seconds 补一条（不抹掉真实用时）
    if [ ! -f "$rcfile" ]; then
      el=$(sed -n "s/.*\"elapsed_seconds\": *\([0-9.]*\).*/\1/p" "$arms" | head -1)
      printf "%s %s %s %s 0 %s 1 %s\n" "$m" "$d" "$tb" "$tg" "${el%%.*}" "$(date "+%F %T")" > "$rcfile"
    fi
    echo "[$(date "+%F %T")] SKIP  $m D=$d th_base=$tb th_ground=$tg (arms.json exists)"
    exit 0
  fi
  mkdir -p "$odir"
  params="$WP_DIR/wp_thb${tb}_thg${tg}.json"
  if [ ! -f "$params" ]; then
    echo "[$(date "+%F %T")] FAIL  $m D=$d th_base=$tb th_ground=$tg (missing params)"
    printf "%s %s %s %s 98 0 0 %s\n" "$m" "$d" "$tb" "$tg" "$(date "+%F %T")" > "$rcfile"
    exit 98
  fi
  t0=$(date +%s)
  "$PY" manu/tracking/lookahead/lookahead_longquan.py \
    --cache-root "$CACHE_ROOT" --labels-root "$GT" --frames-root "$FRAMES" $FRAME_ARGS \
    --params-json "$params" --all-seqs \
    --lookahead replay --arms original,stale,kinematic,replay \
    --delta "$d" $CFG \
    --out-json "$arms" --out-preds-dir "$odir/preds" \
    > "$odir/run.log" 2>&1
  rc=$?
  t1=$(date +%s)
  printf "%s %s %s %s %s %s 0 %s\n" "$m" "$d" "$tb" "$tg" "$rc" "$((t1-t0))" "$(date "+%F %T")" > "$rcfile"
  echo "[$(date "+%F %T")] DONE  $m D=$d th_base=$tb th_ground=$tg rc=$rc elapsed=$((t1-t0))s"
  if [ "$rc" -ne 0 ]; then tail -20 "$odir/run.log"; fi
  exit 0
' _ < "$JOBS_FILE"

# ---- 汇总 progress.txt（幂等；从每个作业的 rc 文件重建，避免并行写串行）----
{
  echo "# 工作点对齐审计 —— 每次运行的 rc 与用时"
  echo "# 生成时间: $(date '+%F %T')  |  执行机: 服务器 test-Super-Server"
  echo "# model delta th_base th_ground rc elapsed_s skipped ts"
  for f in $(ls "$OUT/.rc"/*.rc 2>/dev/null | sort); do cat "$f"; done
} > "$OUT/progress.txt"

N_BAD=$(awk '!/^#/ && $5 != 0 {c++} END {print c+0}' "$OUT/progress.txt")
N_SKIP=$(awk '!/^#/ && $7 == 1 {c++} END {print c+0}' "$OUT/progress.txt")
N_OK=$(awk '!/^#/ && $5 == 0 {c++} END {print c+0}' "$OUT/progress.txt")
echo "[$(date '+%F %T')] ALL DONE. 成功=$N_OK  非零 rc=$N_BAD  跳过(幂等)=$N_SKIP"
echo "[$(date '+%F %T')] 进度: $OUT/progress.txt"
find "$OUT" -name arms.json | sort
