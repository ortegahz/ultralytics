#!/bin/bash
# ==============================================================================
# 训练空间臂链（2026-11-04 夜 · no-GMC 方向一的第二类搜索空间）
#
# 目的
# ----
# 权重空间（插值/手术/TIES/单纯形）都是**零训练**的探索；
# 这条链补上**训练空间**：真正去改「T1 与 T3 互斥」的结构性成因。
#
# 三组臂
# ------
#  A 组｜**T1 修复微调**（最高价值）：从 T3 冠军 `trial_0049` epoch20 起手短训，
#       试探**能否把 T1 拉回红线而 T3 只掉一点**。
#       A1 最小扰动（4 ep, lr 0.5）｜A2 稍强（4 ep, lr 1.0）｜
#       A3 ⭐ **只动头**（6 ep, lr 1.0, backbone_lr_scale 0.1 + 前 2 ep 冻骨干）
#       —— 依据：权重空间诊断显示 trial_0049 与 cs_t0008p2 的 **early/mid 组余弦 0.999**，
#          差异几乎全在 head ⇒ T1 修复**可能只需动 head**，不动深层的 T3 特征。
#  B 组｜**T3 配方空间延伸**：§2.2 结论「最优解贴着取值上界 ⇒ 空间被截断」⇒ 必须往上界外扩。
#       B1 lr 4.0 @ 20 ep｜B2 lr 3.5 @ 30 ep（起点 = `soup_strict3`，与 trial_0043 同谱系）
#  C 组｜**T3 家族加种子**：把 `w2.0·lr3.5·20ep·soup_strict3-init` 家族补到 5~6 种子
#       ⇒ 抬升该配方的**中位 T3**（决策单元是配置中位，不是单条最高）
#
# 资源
# ----
#  · 本链**只用 GPU 0 / GPU 1**（另一条插值链占 GPU 2/3）⇒ 绝不撞卡。
#  · 每条臂训练完**自带门禁**（worker 内建 `--gate-epochs last`），训练模型先释放再起门禁
#    ⇒ 评估不与训练同卡（铁律）。
#  · 墙钟硬截止 2026-11-05 07:00 ⇒ 到点停止排队并落盘。
#
# 口径
# ----
#  · 参照系包 `nogmc_ref_weekend/reference_frame.json` **只读**；
#  · 门禁红线：T1 ≥ 0.8958967482400269 / T2 F1 ≥ 0.985 且 FP ≤ 112 / T3 ≥ 0.87（观察轨）。
#
# 用法（服务器，conda uav，根 /tmp/pycharm_project_10ae9e2e）：
#   setsid nohup bash manu/training/run_t3_train_arms.sh > /mnt/data/siping/runs/nogmc_t3/logs/train_arms.out 2>&1 < /dev/null &
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
OUT="${OUT:-/mnt/data/siping/runs/nogmc_t3}"
REF="${REF:-/mnt/data/siping/runs/nogmc_ref_weekend}"
LOG="$OUT/logs/train_arms.log"
DEADLINE="${DEADLINE:-2026-11-05 07:00}"
GPUS="${GPUS:-0,1}"
WAIT_PHASE_E_MIN="${WAIT_PHASE_E_MIN:-240}"
NOGMC_LQ=/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc/data.yaml
NOGMC_UAV=/mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml
RC2_W=/tmp/pycharm_project_10ae9e2e/runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt
SOUP3=/mnt/data/siping/runs/nogmc_focus/soups/soup_strict3.pt
T49EP20=/mnt/data/siping/runs/nogmc_t3/trials/trial_0049/weights/epoch20.pt

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav || exit 1
export PYTHONPATH=.
mkdir -p "$OUT/logs"
say(){ printf '[%s] [arms] %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; echo "[$(date '+%F %T')] [arms] $*"; }
deadline_pass(){ [ "$(date +%s)" -lt "$(date -d "$DEADLINE" '+%s')" ]; }

# ==============================================================================
# ① 等阶段 E 结束（训练与门禁都空出来）
# ==============================================================================
say "════ 等阶段 E 结束 + 训练 worker 全部退出 ════"
for i in $(seq 1 "$WAIT_PHASE_E_MIN"); do
  done_marker=0
  grep -q "阶段 E 完成" "$OUT/logs/phaseE.log" 2>/dev/null && done_marker=1
  np=$(pgrep -fc "optuna_mix_ratio_search.py --worker" 2>/dev/null || echo 0)
  if [ "$done_marker" = "1" ] && [ "$np" = "0" ]; then say "✅ 阶段 E 完成且无训练进程"; break; fi
  say "   等待中：阶段E完成=$done_marker 训练进程=$np"
  sleep 60
done
deadline_pass || { say "❌ 等待期间到硬截止 ⇒ 中止"; exit 1; }

# ==============================================================================
# ② 臂队列（优先级从高到低；时间不够时末尾被自然截断）
#    字段：trial_no|gpu|seed|epochs|lr_scale|mix_spec|init_weights|init_key|bblr|freeze_bb
# ==============================================================================
QUEUE=$(mktemp)
cat > "$QUEUE" <<EOF
55|0|391|4|0.5|w3.0|$T49EP20|t49ep20|1.0|0
56|1|392|4|1.0|w3.0|$T49EP20|t49ep20|1.0|0
57|0|393|6|1.0|w3.0|$T49EP20|t49ep20|0.1|2
58|1|394|20|4.0|w2.0|$SOUP3|soup_strict3|1.0|0
59|0|395|20|3.5|w2.0|$SOUP3|soup_strict3|1.0|0
60|1|396|20|3.5|w2.0|$SOUP3|soup_strict3|1.0|0
61|0|397|30|3.5|w2.0|$SOUP3|soup_strict3|1.0|0
EOF
say "① 臂队列 $(wc -l < "$QUEUE") 条："
while IFS='|' read -r tn gpu seed ep lr mix init ik bblr fz; do
  say "   t$tn  gpu$gpu  ${ep}ep  lr=$lr  mix=$mix  init=$ik  bblr=$bblr  freeze_bb=$fz  seed=$seed"
done < "$QUEUE"

# ==============================================================================
# ③ 跑臂（每轮最多 2 条并行，分别占 GPU 0 / 1）
# ==============================================================================
run_arm(){   # $1..$9 与队列字段顺序一致
  local tn="$1" gpu="$2" seed="$3" ep="$4" lr="$5" mix="$6" init="$7" ik="$8" bblr="$9"
  local fz="${10}"
  say "▶ t$tn 起跑（gpu$gpu, ${ep}ep, lr=$lr, mix=$mix, init=$ik, bblr=$bblr, freeze_bb=$fz, seed=$seed）"
  python -u manu/training/optuna_mix_ratio_search.py --worker \
    --trial-number "$tn" --gpu-id "$gpu" \
    --output-root "$OUT" \
    --longquan-data "$NOGMC_LQ" \
    --uav-data "$NOGMC_UAV" \
    --weights "$init" --gate-script manu/training/auto_epoch_gate_check.py \
    --baseline-cache "$REF/uav_cache_base.pkl" \
    --epochs "$ep" --batch 32 --imgsz 640 --workers 8 --gate-epochs last --gate-epoch 0 \
    --seed "$seed" --mix-spec "$mix" --lr-scale "$lr" --warmup-epochs 1.5 \
    --weight-decay 0.000269 --focal-beta 2.75 --offset-weight 0.4 \
    --mosaic 0.15 --scale 0.1 --translate 0.05 --clip-max-norm 5.65 \
    --freeze-bn 0 --backbone-lr-scale "$bblr" --freeze-backbone-epochs "$fz" \
    --init-key "$ik" --sched-epochs "$ep" --reference-frame "$REF/reference_frame.json" \
    >> "$OUT/logs/worker_t${tn}.log" 2>&1
  local rc=$?
  say "◀ t$tn 结束 rc=$rc"
  python3 - "$OUT/trials/trial_$(printf '%04d' "$tn")/trial_result.json" "$tn" <<'PY' 2>>"$LOG"
import json, sys, os
p, tn = sys.argv[1], sys.argv[2]
if not os.path.exists(p):
    print("   t%s 无 trial_result.json" % tn); raise SystemExit(0)
d = json.load(open(p)); m = d.get("metrics") or {}
print("   t%s  gate_verdict=%s  T1=%s  T2=%s(fp %s)  T3=%s(fp %s)  utility=%s" % (
    tn, d.get("gate_verdict"), m.get("track1_f1"), m.get("track2_f1"), m.get("track2_fp"),
    m.get("track3_f1"), m.get("track3_fp"), d.get("utility")))
PY
  return 0
}

say "② 开始跑臂（每轮 2 条并行 ⇒ GPU 0 / 1）"
PIDS=()
while IFS='|' read -r tn gpu seed ep lr mix init ik bblr fz; do
  [ -z "$tn" ] && continue
  if ! deadline_pass; then say "❌ 到硬截止 ⇒ 停止排队（剩余臂被截断）"; break; fi
  say "   派发 t$tn → GPU $gpu"
  run_arm "$tn" "$gpu" "$seed" "$ep" "$lr" "$mix" "$init" "$ik" "$bblr" "$fz" &
  PIDS+=($!)
  # 同时在跑的最多 2 条
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do sleep 20; done
done < "$QUEUE"
wait
say "③ 全部臂结束"

# ==============================================================================
# ④ 汇总本链结果
# ==============================================================================
say "④ 汇总"
python3 - <<'PY' 2>&1 | tee -a "$LOG"
import json, glob, os
OUT = "/mnt/data/siping/runs/nogmc_t3"
RL = 0.8958967482400269
rows = []
for p in sorted(glob.glob(OUT + "/trials/trial_00*/gate/*/gate_report.json")):
    n = int(p.split("/trial_")[1].split("/")[0])
    if n < 55:
        continue
    try:
        d = json.load(open(p))
    except Exception:
        continue
    try:
        t1 = d["tracks"]["track1_uav"]["finetune_peak"]["f1"]
        t2 = d["tracks"]["track2_golden"]["f1"]; t2fp = d["tracks"]["track2_golden"]["fp"]
        t3 = d["tracks"]["track3_blind"]["f1"]
    except Exception:
        continue
    rows.append((n, t1, t2, t2fp, t3, d.get("verdict")))
if not rows:
    print("  （无臂产物）")
else:
    print("| trial | T1 | T1 余量(pp) | T1 过线 | T2 F1/FP | T2 过线 | T3 |")
    print("|:--|--:|--:|:--:|--:|:--:|--:|")
    for n, t1, t2, t2fp, t3, v in sorted(rows, key=lambda r: -r[4]):
        print("| trial_%04d | %.9f | %+.4f | %s | %.9f/%d | %s | %.9f |" % (
            n, t1, (t1 - RL) * 100, "✅" if t1 >= RL else "❌", t2, t2fp,
            "✅" if (t2 >= 0.985 and t2fp <= 112) else "❌", t3))
    best = max(rows, key=lambda r: r[4])
    print("\n  T3 最高：trial_%04d  T3=%.9f  T1=%.9f（%s）" % (
        best[0], best[4], best[1], "过线" if best[1] >= RL else "不过线"))
    ok = [r for r in rows if r[1] >= RL]
    if ok:
        b = max(ok, key=lambda r: r[4])
        print("  T1 过线者中 T3 最高：trial_%04d  T3=%.9f  T1 余量 %+.4f pp" % (
            b[0], b[4], (b[1] - RL) * 100))
    else:
        print("  ⚠️ 本链没有任何 T1 过线的臂")
PY
say "✅ 训练臂链完成"
