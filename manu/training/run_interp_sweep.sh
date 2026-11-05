#!/bin/bash
# ==============================================================================
# 跨家族权重插值 / 分层插值 / TIES / 多锚点混合 —— 零训练自主扫参链
# （2026-11-04 · goal「方向一」）
#
# 目标
# ----
#   W(α) = α·W_trial_0049 + (1−α)·W_cs_t0008p2
# 找是否存在一个 α 临界点，让 **T1 稳在红线 0.8958967 之上** 且 **T3 落在 0.895~0.900**。
#
# 本链覆盖 **6 个零训练搜索空间**：
#   W1 linear    全局线性插值 α ∈ {0,0.1,...,1.0}（α=0/1 是端点自检）
#   W2 ties      TIES-Merging（修剪→符号选举→不相交平均），密度 0.2 / 0.5 / 0.8
#   W3 groups    **分层插值**：early/mid/head 三个参数组各自的 α
#   W4 simplex   **多锚点混合**（权重空间单纯形）：T3 家族 × T1 家族的凸组合
#   W5 swa       **沿训练轨迹的 SWA/EMA**：trial_0049 的 ep14~ep20 / ep16~ep20 …
#   W6 extrap    **外插**（α<0 或 α>1）：检验两端之外是否还有更优点
#
# 铁律
# ----
#   * **评估不得与训练同卡** ⇒ 本链**等阶段 E（clip 实验）完全结束**才开跑；
#   * **墙钟硬截止**（默认 2026-11-05 07:30）⇒ 到点自动收尾，不无限跑；
#   * 每个子步骤 rc 独立可查、落盘幂等；
#   * 单条门禁失败**不打断整链**（`set -uo pipefail`，不用 -e）。
#
# 用法（服务器，conda uav，根 /tmp/pycharm_project_10ae9e2e）
#   nohup bash manu/training/run_interp_sweep.sh > /mnt/data/siping/runs/nogmc_t3/logs/interp_sweep.out 2>&1 &
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
RUNS_BASE="${RUNS_BASE:-/mnt/data/siping/runs}"
OUT="${OUT:-$RUNS_BASE/nogmc_t3}"
REF="${REF:-$RUNS_BASE/nogmc_ref_weekend}"
INTERP="$OUT/interp"
LOG="$OUT/logs/interp_sweep.log"
DEADLINE="${DEADLINE:-2026-11-05 07:30}"
GPUS="${GPUS:-0,1,2,3}"            # 本链**只用**这些卡（与训练链按卡分区）
WORKERS="${WORKERS:-4}"            # 并行门禁数（= 同时占用的卡数）
WAIT_PHASE_E_MIN="${WAIT_PHASE_E_MIN:-240}"
# PASS2_ONLY=1 ⇒ 跳过第一轮（权重已在 w1..w8 / 门禁已在 CSV 里），只跑**第二轮细化**。
# 用于修掉 argparse 负值坑之后补跑被吞掉的细化轮，不必重跑 46 条第一轮门禁。
PASS2_ONLY="${PASS2_ONLY:-0}"

# ---- 两个锚点 ----
A_T49="/mnt/data/siping/runs/nogmc_t3/trials/trial_0049/weights/epoch20.pt"
B_P2="/mnt/data/siping/runs/nogmc_t3/swa/cs_t0008p2.pt"
# ---- 多锚点用的其它端点 ----
A_T50="/mnt/data/siping/runs/nogmc_t3/trials/trial_0050/weights/epoch20.pt"
A_T43="/mnt/data/siping/runs/nogmc_t3/trials/trial_0043/weights/epoch20.pt"
T3SOUP="/mnt/data/siping/runs/nogmc_t3/swa/cs_t0043.pt"
P2="/mnt/data/siping/runs/nogmc_t3/swa/cs_t0008p2.pt"
T49D="/mnt/data/siping/runs/nogmc_t3/trials/trial_0049/weights"

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav || exit 1
export PYTHONPATH=.
mkdir -p "$OUT/logs" "$INTERP"

say(){ printf '[%s] [interp] %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; echo "[$(date '+%F %T')] [interp] $*"; }
deadline_pass(){ [ "$(date +%s)" -lt "$(date -d "$DEADLINE" '+%s')" ]; }

CSV="$INTERP/interp_sweep.csv"
[ -f "$CSV" ] || echo "name,weight,alpha,mode,t1_f1,t1_margin_pp,t2_f1,t2_fp,t3_f1,t3_vs_base_pp,t3_vs_rc2_pp,t3_fp,verdict" > "$CSV"

# ==============================================================================
# ① 等阶段 E（clip 实验 + cs_t0033clip 门禁 + finalize）完全结束 ⇒ 不撞卡
# ==============================================================================
say "════ 等阶段 E 结束（训练 + 门禁都空出来才开跑）════"
for i in $(seq 1 "$WAIT_PHASE_E_MIN"); do
  if grep -q "阶段 E 完成" "$OUT/logs/phaseE.log" 2>/dev/null; then
    say "✅ 阶段 E 已完成"; break
  fi
  # 训练进程是否还在
  # NOTE: `pgrep -fc` prints "0" **and** exits 1 when nothing matches, so `|| echo 0`
  # would append a second line and make the count "0\n0" -- never equal to "0", which
  # silently stalled the wait loop for 30 minutes.  `| wc -l` always yields one number.
  nproc=$(pgrep -f "optuna_mix_ratio_search.py --worker" 2>/dev/null | wc -l)
  if [ "$nproc" = "0" ] && [ "$i" -gt 2 ]; then
    say "   训练进程已清零（nproc=0）⇒ 额外等 3 分钟让阶段 E 的门禁收尾"
    sleep 180
    if grep -q "阶段 E 完成" "$OUT/logs/phaseE.log" 2>/dev/null; then say "✅ 阶段 E 已完成"; break; fi
  fi
  say "   等待中：阶段 E 未完成（训练进程 $nproc 个）"
  sleep 60
done
deadline_pass || { say "❌ 等待阶段 E 期间到硬截止 ⇒ 中止"; exit 1; }

# 训练卡确认空闲（铁律：评估不得与训练同卡）—— 独立再等一次 worker 进程清零
for i in $(seq 1 60); do
  np=$(pgrep -f "optuna_mix_ratio_search.py --worker" 2>/dev/null | wc -l)
  [ "$np" = "0" ] && break
  say "   等训练 worker 退出（还有 $np 个）"
  sleep 30
done
NG=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
say "服务器 GPU 数 = $NG ｜ 本链用卡 = $GPUS ｜ 并行门禁 worker = $WORKERS"

# ==============================================================================
# ② 造权重（CPU，不占卡）
# ==============================================================================
say "① 造插值/合并权重（CPU）"
BUILD="$INTERP/build.log"

py_build(){ python manu/training/build_weight_interpolation.py "$@" >> "$BUILD" 2>&1; echo "rc=$?"; }

# W1 全局线性插值（含端点自检 α=0/1）
if [ "$PASS2_ONLY" != "1" ]; then
say "   W1 linear α ∈ {0,0.1,...,1.0}"
py_build --mode linear --a "$A_T49" --b "$B_P2" \
  --alphas 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --out-dir "$INTERP/w1" --prefix interp_lin | tee -a "$LOG"

# W2 TIES
say "   W2 TIES（density 0.2/0.5/0.8）"
for d in 0.2 0.5 0.8; do
  py_build --mode ties --a "$A_T49" --b "$B_P2" --ties-density "$d" \
    --out-dir "$INTERP/w2" --prefix interp_ties | tee -a "$LOG"
done

# W3 分层插值（early:mid:head）
say "   W3 分层插值（early/mid/head 三组独立 α）"
py_build --mode groups --a "$A_T49" --b "$B_P2" \
  --alphas "0.4:0.6:0.8,0.6:0.6:0.8,0.8:0.6:0.6,0.6:0.6:0.4,0.8:0.8:0.4,0.4:0.4:0.8,0.2:0.6:0.9,0.9:0.6:0.2" \
  --out-dir "$INTERP/w3" --prefix interp_grp | tee -a "$LOG"

# W4 多锚点混合（权重空间单纯形）
say "   W4 多锚点混合（simplex）"
py_build --mode simplex --anchors "1=$A_T49,1=$A_T50,2=$P2" \
  --out-dir "$INTERP/w4" --prefix interp_mix_t3fam2_x_p2 | tee -a "$LOG"
py_build --mode simplex --anchors "1=$A_T49,1=$A_T50,1=$A_T43,3=$P2" \
  --out-dir "$INTERP/w4" --prefix interp_mix_t3fam3_x_p2 | tee -a "$LOG"
py_build --mode simplex --anchors "1=$T3SOUP,1=$P2" \
  --out-dir "$INTERP/w4" --prefix interp_mix_t3soup_x_p2 | tee -a "$LOG"
py_build --mode simplex --anchors "2=$A_T49,1=$A_T50,3=$P2" \
  --out-dir "$INTERP/w4" --prefix interp_mix_t49heavy_x_p2 | tee -a "$LOG"
py_build --mode simplex --anchors "1=$A_T49,1=$T3SOUP,2=$P2" \
  --out-dir "$INTERP/w4" --prefix interp_mix_t49_tsoup_x_p2 | tee -a "$LOG"

# W5 沿训练轨迹 SWA/EMA（trial_0049 / trial_0050 的末段 epoch 平均）
say "   W5 轨迹 SWA/EMA"
py_build --mode simplex --anchors "1=$T49D/epoch18.pt,1=$T49D/epoch19.pt,1=$T49D/epoch20.pt" \
  --out-dir "$INTERP/w5" --prefix interp_swa_t49_ep18_20 | tee -a "$LOG"
py_build --mode simplex --anchors "1=$T49D/epoch14.pt,1=$T49D/epoch15.pt,1=$T49D/epoch16.pt,1=$T49D/epoch17.pt,1=$T49D/epoch18.pt,1=$T49D/epoch19.pt,1=$T49D/epoch20.pt" \
  --out-dir "$INTERP/w5" --prefix interp_swa_t49_ep14_20 | tee -a "$LOG"
py_build --mode simplex --anchors "1=$A_T49,1=$A_T50" \
  --out-dir "$INTERP/w5" --prefix interp_soup_t49_t50 | tee -a "$LOG"

# W6 外插
say "   W6 外插（α = -0.1 / 1.1）"
py_build --mode linear --a "$A_T49" --b "$B_P2" --alphas=-0.1,1.1 \
  --out-dir "$INTERP/w6" --prefix interp_extrap | tee -a "$LOG"

# BN 统计量取单端的变体（诊断显示 BN 是差异最大的部分 ⇒ 值得单独试）
say "   W7 BN 取端点 A 的变体（α ∈ {0.2..0.8}）—— head 的 BN 差异最大，值得单独扫"
py_build --mode linear --a "$A_T49" --b "$B_P2" --alphas 0.2,0.3,0.4,0.5,0.6,0.7,0.8 --nobn \
  --out-dir "$INTERP/w7" --prefix interp_nobn | tee -a "$LOG"

# W8 head 模块手术（离散换源）：哪些 head 子模块带来 T3、哪些带来 T1
say "   W8 head 模块手术（整块换源）"
py_build --mode surgery --a "$A_T49" --b "$B_P2" --from-b "head.feat_conv." \
  --out-dir "$INTERP/w8" --prefix interp_surg_featconv | tee -a "$LOG"
py_build --mode surgery --a "$A_T49" --b "$B_P2" --from-b "head.offset." \
  --out-dir "$INTERP/w8" --prefix interp_surg_offset | tee -a "$LOG"
py_build --mode surgery --a "$A_T49" --b "$B_P2" --from-b "head.heatmap." \
  --out-dir "$INTERP/w8" --prefix interp_surg_heatmap | tee -a "$LOG"
py_build --mode surgery --a "$A_T49" --b "$B_P2" --from-b "head.feat_conv.,head.offset." \
  --out-dir "$INTERP/w8" --prefix interp_surg_featoff | tee -a "$LOG"
py_build --mode surgery --a "$A_T49" --b "$B_P2" --from-b "head.heatmap.,fuse_p1." \
  --out-dir "$INTERP/w8" --prefix interp_surg_heatfuse | tee -a "$LOG"
py_build --mode surgery --a "$A_T49" --b "$B_P2" --from-b "head." \
  --out-dir "$INTERP/w8" --prefix interp_surg_allhead | tee -a "$LOG"
py_build --mode surgery --a "$A_T49" --b "$B_P2" --from-b "head.,fuse_p1." \
  --out-dir "$INTERP/w8" --prefix interp_surg_headfuse | tee -a "$LOG"
fi   # PASS2_ONLY

# ==============================================================================
# ③ 排队 + 4 卡并行门禁
# ==============================================================================
QUEUE="$INTERP/queue.txt"
: > "$QUEUE"
# 优先级：先跑最可能命中的（线性 α 粗扫 → 分层 → 单纯形 → TIES → SWA → 外插）
for w in w1 w7 w8 w4 w3 w5 w2 w6; do
  for f in "$INTERP"/$w/*.pt; do [ -f "$f" ] && echo "$(basename "$f" .pt)|$f" >> "$QUEUE"; done
done
NQ=$(wc -l < "$QUEUE")
say "② 门禁队列 $NQ 条（$WORKERS 卡并行，单条约 20~25 min ⇒ 约 $(( NQ / WORKERS * 22 )) min）"

# 记录本次队列（供后续细化与追溯）
cp "$QUEUE" "$INTERP/queue_pass1.txt"

run_gate(){   # $1=name  $2=weight  $3=gpu
  local name="$1" w="$2" gpu="$3"
  local gd="$INTERP/gate/$name"
  mkdir -p "$gd"
  python manu/training/auto_epoch_gate_check.py \
    --weights "$w" --out-dir "$gd" \
    --gpus 1 --device "$gpu" --infer-gpu "$gpu" --cache-gpus "$gpu" \
    --baseline-cache "$REF/uav_cache_base.pkl" \
    --min-track2-f1 0.985 --max-track2-fp 112 --min-track1-f1 0.8958967482400269 \
    --uav-data /mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml \
    --ft-root /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc \
    --base-track1-f1 0.9008967482400269 --base-track1-tp 21499 --base-track1-fp 1118 \
    --base-track2-f1 0.981074481074481 --base-track2-tp 16070 --base-track2-fp 52 --base-track2-fn 568 \
    --base-track3-f1 0.7578120617110798 --base-track3-tp 6754 --base-track3-fp 536 --base-track3-fn 3781 \
    > "$OUT/logs/gate_interp_${name}.log" 2>&1
  local rc=$?
  python3 - "$gd/gate_report.json" "$name" "$w" "$CSV" <<'PY' >> "$CSV" 2>>"$LOG"
import json, sys, os
rp, name, w, csvp = sys.argv[1:5]
if not os.path.exists(rp):
    print("%s,%s,,,,,,,,,,,,GATE_MISSING" % (name, w)); raise SystemExit(0)
d = json.load(open(rp))
t1 = d["tracks"]["track1_uav"]["finetune_peak"]; t2 = d["tracks"]["track2_golden"]; t3 = d["tracks"]["track3_blind"]
RL, BASE3, RC23 = 0.8958967482400269, 0.7578120617110798, 0.8976937315523539
print("%s,%s,%s,%s,%.9f,%+.4f,%.9f,%d,%.9f,%+.4f,%+.4f,%d,%s" % (
    name, w, "", "", t1["f1"], (t1["f1"]-RL)*100, t2["f1"], t2["fp"], t3["f1"],
    (t3["f1"]-BASE3)*100, (t3["f1"]-RC23)*100, t3["fp"], d.get("verdict")))
PY
  say "   [$gpu] $name 门禁 rc=$rc ⇒ $(tail -1 "$CSV")"
}

say "③ 开始门禁"
if [ "$PASS2_ONLY" != "1" ]; then
i=0
while IFS='|' read -r name w; do
  [ -z "$name" ] && continue
  if ! deadline_pass; then say "❌ 到硬截止 ⇒ 停止排队"; break; fi
  gpu=$(echo "$GPUS" | cut -d, -f$(( (i % WORKERS) + 1 )))
  # 简单并行：每 WORKERS 条起一个后台 worker
  if [ $((i % WORKERS)) -eq 0 ]; then
    # 等本批全部结束
    wait
  fi
  ( run_gate "$name" "$w" "$gpu" ) &
  i=$((i+1))
done < "$QUEUE"
wait
say "④ 第一轮门禁完成"
fi   # PASS2_ONLY

# ==============================================================================
# ④ 自动细化：找 T1 过线且 T3 最高的点 ⇒ 在它附近加密 α
# ==============================================================================
say "⑤ 自动细化（找兼得点附近加密）"
if [ "$PASS2_ONLY" != "1" ]; then
python3 - <<'PY' 2>&1 | tee -a "$LOG"
import csv, json, os
from pathlib import Path
INTERP = Path("/mnt/data/siping/runs/nogmc_t3/interp")
rows = list(csv.DictReader(open(INTERP / "interp_sweep.csv")))
rows = [r for r in rows if r.get("t1_f1") not in (None, "", "GATE_MISSING")]
if not rows:
    print("  无有效结果 ⇒ 跳过细化"); raise SystemExit(0)
for r in rows:
    r["t1"] = float(r["t1_f1"]); r["t3"] = float(r["t3_f1"])
RL = 0.8958967482400269
ok = [r for r in rows if r["t1"] >= RL]
print("  有效 %d 条｜T1 过线 %d 条" % (len(rows), len(ok)))
if ok:
    best = max(ok, key=lambda r: r["t3"])
    print("  ⭐ T1 过线且 T3 最高：%s  T1=%.9f（余量 %+.4f pp）T3=%.9f" %
          (best["name"], best["t1"], (best["t1"]-RL)*100, best["t3"]))
else:
    best = max(rows, key=lambda r: r["t3"])
    print("  ⚠️ 无 T1 过线点；T3 最高：%s T1=%.9f T3=%.9f" % (best["name"], best["t1"], best["t3"]))
# 生成细化队列：在最佳 α 附近（±0.15，步长 0.025）补线性插值点
done = {r["name"] for r in rows}
A = "/mnt/data/siping/runs/nogmc_t3/trials/trial_0049/weights/epoch20.pt"
B = "/mnt/data/siping/runs/nogmc_t3/swa/cs_t0008p2.pt"
base_alpha = None
try:
    base_alpha = float(best["name"].split("_a")[-1].replace("m", "-").replace("p", "+"))
except Exception:
    pass
q = []
if base_alpha is not None:
    for k in range(-6, 7):
        al = round(base_alpha + k * 0.025, 4)
        if al < -0.15 or al > 1.15:
            continue
        tag = "a%+.3f" % al
        name = "interp_lin_%s" % tag.replace("+", "p").replace("-", "m")
        if name in done:
            continue
        q.append((name, al))
(INTERP / "queue_pass2_alphas.txt").write_text(
    "".join("%s\n" % a for _, a in q), encoding="utf-8")
print("  细化候选 %d 个：%s" % (len(q), ", ".join("%.4f" % a for _, a in q)))
PY
fi   # PASS2_ONLY（细化点已定 ⇒ 复用既有 queue_pass2_alphas.txt）

# 造细化权重并跑第二轮
if [ -s "$INTERP/queue_pass2_alphas.txt" ]; then
  AL=$(paste -sd, "$INTERP/queue_pass2_alphas.txt")
  if [ -n "$AL" ]; then
    say "   造细化权重 α = $AL"
    # NOTE: must use ``--alphas=VALUE`` (with ``=``).  A bare ``--alphas "$AL"`` makes
    # argparse treat a negative leading value (``-0.15,...``) as an option and abort with
    # "expected one argument" -- that silently killed the 2026-11-04 17:xx refinement round
    # (build failed -> empty queue -> "第二轮门禁 0 条").
    python manu/training/build_weight_interpolation.py --mode linear --a "$A_T49" --b "$B_P2" \
      --alphas="$AL" --out-dir "$INTERP/w9" --prefix interp_lin >> "$BUILD" 2>&1
    if [ ! -d "$INTERP/w9" ] || ! ls "$INTERP"/w9/*.pt >/dev/null 2>&1; then
      say "❌ 细化权重建权失败（见 $BUILD）⇒ 跳过第二轮"
      : > "$QUEUE"
    else
      : > "$QUEUE"
      for f in "$INTERP"/w9/*.pt; do [ -f "$f" ] && echo "$(basename "$f" .pt)|$f" >> "$QUEUE"; done
    fi
    say "   第二轮门禁 $(wc -l < "$QUEUE") 条"
    i=0
    while IFS='|' read -r name w; do
      [ -z "$name" ] && continue
      if ! deadline_pass; then say "❌ 到硬截止 ⇒ 停止排队"; break; fi
      gpu=$(echo "$GPUS" | cut -d, -f$(( (i % WORKERS) + 1 )))
      if [ $((i % WORKERS)) -eq 0 ]; then wait; fi
      ( run_gate "$name" "$w" "$gpu" ) &
      i=$((i+1))
    done < "$QUEUE"
    wait
  fi
fi

# ==============================================================================
# ⑤ 汇总
# ==============================================================================
say "⑥ 汇总"
python3 - <<'PY' 2>&1 | tee -a "$LOG"
import csv
from pathlib import Path
INTERP = Path("/mnt/data/siping/runs/nogmc_t3/interp")
rows = list(csv.DictReader(open(INTERP / "interp_sweep.csv")))
rows = [r for r in rows if r.get("t1_f1") not in (None, "", "GATE_MISSING")]
RL = 0.8958967482400269
def f(r, k):
    try: return float(r[k])
    except Exception: return None
rows.sort(key=lambda r: (-(f(r, "t1_f1") or 0)))
out = ["# 跨家族权重插值扫参结果（2026-11-04）", "",
       "| 产物 | T1 | T1 余量(pp) | T2 F1 | T2 FP | T3 | T3 vs base(pp) | T3 vs RC2(pp) | T3 FP | verdict |",
       "|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|"]
for r in rows:
    out.append("| `%s` | %.9f | %+.4f | %.9f | %s | %.9f | %s | %s | %s | %s |" % (
        r["name"], f(r, "t1_f1"), (f(r, "t1_f1") - RL) * 100, f(r, "t2_f1"), r["t2_fp"],
        f(r, "t3_f1"), r["t3_vs_base_pp"], r["t3_vs_rc2_pp"], r["t3_fp"], r["verdict"]))
out += ["", "**判据**：T1 ≥ 0.8958967482400269（硬门）且 T3 ≥ 0.87（观察轨）。", ""]
(INTERP / "interp_sweep.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
PY
say "✅ 插值扫参链完成（产物：$INTERP/interp_sweep.md / interp_sweep.csv）"
