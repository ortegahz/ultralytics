#!/bin/bash
# ==============================================================================
# 定点打靶**收尾交付**（2026-11-02）
#
# 把「训练结束 → 出交接口径报告 → 固化 RC → 备份」串成**一条命令**，
# 避免周二早上（用户上班前）手忙脚乱漏步骤。
#
# 顺序（每一步都落盘 + 打日志）：
#   ① backfill   —— 从 study.db 补齐 CSV 行
#   ② rejudge    —— 用最终参照系重判（**并回填 metrics**，见 §8.5）
#   ③ summarize  —— 生成 SUMMARY.md
#   ④ judge      —— **交接口径判定**（严格通道 / 多种子中位 / 噪声带 / 阻塞项）
#   ⑤ seal       —— 按判定结果固化 Release Candidate（只封严格 PASS + 中位 trial）
#   ⑥ 备份 + 打印摘要
#
# 用法（服务器 `ssh -p 32222 huangzhe@192.168.99.40`，`conda activate uav`）：
#
# ```bash
# cd /tmp/pycharm_project_10ae9e2e
# bash manu/training/finalize_nogmc_focus.sh
# ```
#
# 幂等：可重复执行。`SKIP_SEAL=1` 可跳过固化（当判定结果为"无严格 PASS"时，
# 固化脚本会自己以退出码 4 拒绝，不会误封 PASS_RELAXED）。
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
RUNS_BASE="${RUNS_BASE:-/mnt/data/siping/runs}"
# ⚠️ **RUN_TAG / SELF_TAG 的默认值就是 focus 那轮的旧行为**（逐位不变）：
#   RUN_TAG  —— 决定备份目录名与拷贝到 runs/ 根下的文件名（nogmc_focus_*）
#   SELF_TAG —— 本 study 在**判定器**里的来源标签（trial 名会加 'TAG:' 前缀）
#               ⚠️ 必须是**唯一**标签：两个 study 都有 trial_0009 ⇒ 同名不同物
#   其它 study 用 EXTRA_CSV / EXTRA_TRIALS_ROOT / EXTRA_STUDY 追加进来。
RUN_TAG="${RUN_TAG:-nogmc_focus}"
SELF_TAG="${SELF_TAG:-focus}"
EXTRA_CSV="${EXTRA_CSV:-}"                # 形如 "t3=/mnt/data/siping/runs/nogmc_t3/optuna_summary.csv"
EXTRA_TRIALS_ROOT="${EXTRA_TRIALS_ROOT:-}"  # 形如 "t3=/mnt/data/siping/runs/nogmc_t3/trials"
EXTRA_STUDY="${EXTRA_STUDY:-}"            # 形如 "t3=/mnt/data/siping/runs/nogmc_t3/study.db:mix_ratio_search_nogmc_t3"
OUT_ROOT="${OUT_ROOT:-$RUNS_BASE/nogmc_focus}"
OLD_ROOT="${OLD_ROOT:-$RUNS_BASE/nogmc_weekend}"      # 周末那轮（**必须一起进判定池**）
REF_DIR="${REF_DIR:-$RUNS_BASE/nogmc_ref_weekend}"
LOG="${LOG:-$RUNS_BASE/${RUN_TAG}_finalize.log}"
STUDY_NAME="${STUDY_NAME:-mix_ratio_search_nogmc_focus}"
RC_ID="${RC_ID:-Candidate_v2}"
RC_NAME="${RC_NAME:-no-GMC 定点打靶（41 臂）最优交接候选}"

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav || exit 1
export PYTHONPATH=.
mkdir -p "$OUT_ROOT/logs"
say(){ echo "[$(date '+%F %T')] [finalize] $*" | tee -a "$LOG"; }

REF="$REF_DIR/reference_frame.json"
[ -f "$REF" ] || { say "缺参照系包 $REF"; exit 1; }
CSV="$OUT_ROOT/optuna_summary.csv"
[ -f "$CSV" ] || { say "缺 $CSV（先跑完训练与 summarize）"; exit 1; }

# ---- 判定池：**周末 + 本轮**两份 CSV ----------------------------------
# ⚠️ 为什么必须合并（2026-11-02 15:35 实测发现）：
#    交接口径 = **同超参多种子中位**。同配方的种子**横跨两个 study** ——
#    周末 trial_0009（严格 PASS 里 T3 最高 0.857880）与本轮 FOCUS_T0009_SEED2
#    是**同一个配方**（w3.0·lr1.0·15ep·rc2·bb1）⇒ 只有合并才能凑够 n≥2。
#    只读本轮 CSV ⇒ ① 候选池里少了周末的 33 条严格 PASS（含 T3 最高的 trial_0009）；
#    ② 同配方种子被拆成两组 ⇒ n 永远算不到 2 ⇒ 交接口径永远不成立。
#    ⚠️ 两个 study **都有 trial_0009 但不是同一条** ⇒ 工具会给 trial 名加 'TAG:' 前缀。
OLD_CSV="$OLD_ROOT/optuna_summary.csv"
CSV_ARGS=(--csv "$SELF_TAG=$CSV")
ROOT_ARGS=(--trials-root "$SELF_TAG=$OUT_ROOT/trials")
if [ -f "$OLD_CSV" ]; then
  CSV_ARGS+=("weekend=$OLD_CSV")
  ROOT_ARGS+=(--trials-root "weekend=$OLD_ROOT/trials")
  say "判定池：$SELF_TAG + weekend 两份 CSV（合并同配方种子）"
else
  say "⚠️ 找不到周末 CSV $OLD_CSV ⇒ 只用本轮（**多种子口径可能不成立**）"
fi

# ---- 额外判定池（可选；默认空 ⇒ **旧行为逐位不变**）----------------------
#   EXTRA_CSV         形如 "t3=/path/optuna_summary.csv"      ⇒ 判定器多收一份 CSV
#   EXTRA_TRIALS_ROOT 形如 "t3=/path/trials"                  ⇒ 固化时多一个 trials 根
#   EXTRA_STUDY       形如 "t3=/path/study.db:study_name"     ⇒ T3 可达性多一个 study
# ⚠️ 判定器的 `--csv` 是 **nargs="+"**（一个 --csv 多个值）⇒ 只能往 CSV_ARGS 里追加。
for SPEC in $EXTRA_CSV; do
  [ -n "$SPEC" ] || continue
  CSV_ARGS+=("$SPEC")
  say "判定池追加 CSV：$SPEC"
done
for SPEC in $EXTRA_TRIALS_ROOT; do
  [ -n "$SPEC" ] || continue
  ROOT_ARGS+=(--trials-root "$SPEC")
  say "固化 trials 根追加：$SPEC"
done

say "════ 收尾交付开始 ════"

# ① backfill
say "① backfill 补齐 CSV 行"
python manu/training/backfill_summary_rows.py --out-root "$OUT_ROOT" \
  --storage "sqlite:///$OUT_ROOT/study.db" --study-name "$STUDY_NAME" --apply >> "$LOG" 2>&1
say "   rc=$?"

# ② rejudge（含 metrics 回填）
say "② rejudge 权威重判（回填 metrics）"
python manu/training/rejudge_with_reference.py --out-root "$OUT_ROOT" --ref "$REF" \
  --epochs 0 --apply >> "$LOG" 2>&1
say "   rc=$?"

# ③ summarize
say "③ 生成 SUMMARY.md"
python manu/training/summarize_nogmc_search.py --out-root "$OUT_ROOT" --ref "$REF" --top 20 >> "$LOG" 2>&1
say "   rc=$?"

# ④ 交接口径判定
say "④ 交接口径判定（严格通道 / 多种子中位 / 噪声带 / 阻塞项）"
# ④-0 **变体池**（跨种子 soup / 同 run 相邻 epoch 平均）：
#      ⚠️ 变体**不在 study 里**（不是 trial）⇒ 判定器按配方分组时**看不见它们**
#      ⇒ 实测：cs_t0008（T1 0.8959005 / T2 FP 42 / T3 0.868572 / verdict=PASS）
#      是**严格 PASS**，却是全局 U 最高的产物 ⇒ 不送进去就等于交接时被无视。
#      ⇒ 这里先把变体的门禁结果汇成 CSV，再作为**独立通道**送进判定器：
#         变体**单独建组**（绝不混进同配方种子中位 ⇒ 不污染配置中位数），
#         硬门禁用它自己的三轨指标判，多种子证据取**源家族**的种子数。
VARIANT_ARGS=()
if [ -n "${EXTRA_VARIANTS_CSV:-}" ]; then
  VARIANT_BUILDER="manu/training/build_nogmc_variants_csv.py"
  if [ -f "$VARIANT_BUILDER" ] && [ -d "${VARIANTS_SWA_DIR:-}" ]; then
    say "④-0 生成变体 CSV（soup / SWA → 判定器独立通道）"
    VARIANTS_BUILD_ARGS=(--swa-dir "${VARIANTS_SWA_DIR}")
    for D in ${EXTRA_VARIANTS_SWA_DIRS:-}; do VARIANTS_BUILD_ARGS+=(--swa-dir "$D"); done
    # ⚠️ **必须传 --study-map**（2026-11-03 18:30 修）：
    #    `build_nogmc_variants_csv.py` 读 `<swa-dir>/crossseed_manifest.json` 里的
    #    **旧家族**（如 t0077 / t0082）来算「多种子证据数」，而 manifest 的结构是
    #    `{fam: {study_tag: [成员]}}` ⇒ 必须有 tag→(root, study) 的映射才能展开成员。
    #    不传 ⇒ `study_map` 为空 ⇒ **所有成员被 continue 跳过** ⇒ 家族定义拿不到
    #    ⇒ `seed_evidence_n` 退化成 **1**（实测 cs_t0077 / cs_t0082 各有 3 个成员，
    #       却都被记成 1 ⇒ **种子证据被低估** ⇒ 判定链里被降级）。
    #    ⇒ 传 weekend / focus 两个 tag（focus 侧的历史家族主要在这两个 study 里）。
    VARIANTS_BUILD_ARGS+=(--study-map "weekend=${RUNS_BASE}/nogmc_weekend:mix_ratio_search_nogmc_weekend")
    VARIANTS_BUILD_ARGS+=(--study-map "focus=${RUNS_BASE}/nogmc_focus:mix_ratio_search_nogmc_focus")
    python "$VARIANT_BUILDER" "${VARIANTS_BUILD_ARGS[@]}" \
      --families-json "${VARIANTS_FAMILIES_JSON:-manu/training/nogmc_t3_families.json}" \
      --ref "$REF" --out "$EXTRA_VARIANTS_CSV" >> "$LOG" 2>&1
    say "   rc=$? → $EXTRA_VARIANTS_CSV"
  fi
  if [ -f "$EXTRA_VARIANTS_CSV" ]; then
    VARIANT_ARGS=(--variants-csv "$EXTRA_VARIANTS_CSV")
    say "④-0 变体池已接入：$EXTRA_VARIANTS_CSV"
  else
    say "⚠️ 变体 CSV 不存在 ⇒ 本次判定**不含变体**（严格 PASS 的 soup 会被无视）"
  fi
fi
python manu/training/judge_handover_candidate.py "${CSV_ARGS[@]}" ${VARIANT_ARGS[@]+"${VARIANT_ARGS[@]}"} --ref "$REF" \
  --out-md "$OUT_ROOT/HANDOVER.md" --out-json "$OUT_ROOT/HANDOVER.json" \
  --min-seeds 2 >> "$LOG" 2>&1
JUDGE_RC=$?
say "   判定器 rc=$JUDGE_RC（0=可交接 / 3=有候选但种子数不足 / 4=无严格 PASS）"

# ④b SWA 变体（沿 epoch 轴权重平均，零训练成本）—— 等它门禁跑完再整合
#     ⚠️ SWA 产物**不在 study 里** ⇒ judge/seal 都看不到它 ⇒ 必须单独并入报告。
#     ⚠️ 也可能**命中双目标**（T1 过线 + T3 ≥ 0.87）⇒ 命中则它就是最优产物。
SWA_MARK="${SWA_MARK:-$OUT_ROOT/logs/gate_swa.log}"
if [ -f "$SWA_MARK" ]; then
  say "④b 等 SWA 门禁收尾（最多 ${SWA_WAIT_MIN:-45} 分钟）"
  WAITED=0
  while [ "$WAITED" -lt "${SWA_WAIT_MIN:-45}" ]; do
    grep -q "SWA 变体门禁全部完成" "$SWA_MARK" && break
    sleep 60
    WAITED=$((WAITED+1))
  done
fi
say "④b 整合 SWA 结果（并入 HANDOVER.md 附录 C；命中则自动封存）"
# ⚠️ **重封前必须先解除只读**（2026-11-03 06:33 实测踩到）：
#    封存副本是 0444（防篡改）⇒ 第二次封存同一个 rc_id 时 `shutil.copy2` 直接抛
#    **PermissionError** ⇒ 整个收尾的固化步骤 rc=1 失败。
#    ⇒ 覆盖前先 chmod +w（只影响"能不能被覆盖"，封存后仍会再置回 0444）。
for D in "$RUNS_BASE/CANDIDATES/$RC_ID" "$RUNS_BASE/CANDIDATES/${RC_ID}_swa"; do
  [ -d "$D" ] && { chmod -R u+w "$D" 2>/dev/null; say "   已解除只读：$D"; }
done
python manu/training/incorporate_swa_results.py \
  --out-root "$OUT_ROOT" --ref "$REF" \
  --handover-md "$OUT_ROOT/HANDOVER.md" \
  --candidates-dir "$RUNS_BASE/CANDIDATES" \
  --rc-id "${RC_ID}_swa" --seal >> "$LOG" 2>&1
say "   整合 rc=$?"
SWA_HIT=0
if [ -f "$OUT_ROOT/SWA_SUMMARY.json" ]; then
  SWA_HIT=$(python3 -c "import json;print(json.load(open('$OUT_ROOT/SWA_SUMMARY.json')).get('n_hit',0))" 2>/dev/null || echo 0)
fi
say "   SWA 命中数 = $SWA_HIT"

# ⑤ 固化 RC
if [ "${SKIP_SEAL:-0}" = "1" ]; then
  say "⑤ 跳过固化（SKIP_SEAL=1）"
elif [ "$JUDGE_RC" = "4" ]; then
  say "⑤ 无严格 PASS ⇒ **不固化任何 RC**（口径铁律：宁可交缺口清单也不封放宽通道）"
elif [ "${SWA_HIT:-0}" -gt 0 ] 2>/dev/null; then
  say "⑤ ★ **SWA 变体命中双目标（已由 ④b 封存）⇒ 不再封 trial 侧候选**（避免两个 RC 造成歧义）"
else
  say "⑤ 固化 Release Candidate：$RC_ID"
  python manu/training/seal_candidate.py \
    --handover-json "$OUT_ROOT/HANDOVER.json" "${CSV_ARGS[@]}" "${ROOT_ARGS[@]}" \
    --out-dir "$RUNS_BASE/CANDIDATES" \
    --rc-id "$RC_ID" --name "$RC_NAME" --ref "$REF" >> "$LOG" 2>&1
  say "   固化 rc=$?"
fi

# ⑤b T3 可达性分析（交付件之一）
#   · 用 **study.db 原始 float** 重扫受控对（CSV 的格式化浮点会凭空造出/抹掉受控对）
#   · 给出「每换 1 单位 T3 要付多少 T1」的**汇率排序** + **种子噪声带**
#   · ⇒ 回答「T3 ≥ 0.87 在这套先验空间里到底可达不可达」
say "⑤b T3 可达性分析（受控对重扫 + 汇率排序 + 种子噪声带）"
# ⚠️ **必须用独立的参数数组**（2026-11-03 08:5x 修）：
#    `analyze_t3_reachability.py` 的 `--csv` 是 **action="append"**（每个 CSV 一个 --csv），
#    而 `judge_handover_candidate.py` / `seal_candidate.py` 的是 **nargs="+"**（一个 --csv 多个值）。
#    ⇒ 复用 judge 的 CSV_ARGS ⇒ argparse 报 "unrecognized arguments: weekend=..." ⇒ **rc=2**，
#       T3_REACHABILITY.md 停在旧档（交付件残缺）。
STUDY_ARGS=(--study "$SELF_TAG=$OUT_ROOT/study.db:$STUDY_NAME")
T3_CSV_ARGS=(--csv "$SELF_TAG=$CSV")
[ -f "$OLD_CSV" ] && T3_CSV_ARGS+=(--csv "weekend=$OLD_CSV")
[ -f "$OLD_ROOT/study.db" ] && STUDY_ARGS+=(--study "weekend=$OLD_ROOT/study.db:mix_ratio_search_nogmc_weekend")
# 额外 study（可选；默认空 ⇒ 旧行为不变）。⚠️ 这里的 --csv/--study 是 **action="append"**
#    ⇒ 每个 CSV 一个 --csv、每个 study 一个 --study（**不能**复用 judge 的 nargs="+" 数组）。
for SPEC in $EXTRA_CSV; do
  [ -n "$SPEC" ] || continue
  T3_CSV_ARGS+=(--csv "$SPEC")
done
for SPEC in $EXTRA_STUDY; do
  [ -n "$SPEC" ] || continue
  STUDY_ARGS+=(--study "$SPEC")
done
python manu/training/analyze_t3_reachability.py \
  "${T3_CSV_ARGS[@]}" "${STUDY_ARGS[@]}" \
  --ref "$REF" \
  --out-md "$OUT_ROOT/T3_REACHABILITY.md" \
  --out-json "$OUT_ROOT/T3_REACHABILITY.json" >> "$LOG" 2>&1
say "   T3 可达性 rc=$?"

# ⑥ 备份 + 摘要
STAMP=$(date '+%Y%m%d_%H%M')
BAK="$RUNS_BASE/${RUN_TAG}_backup_$STAMP"
mkdir -p "$BAK"
for f in SUMMARY.md optuna_summary.csv study.db HANDOVER.md HANDOVER.json SWA_SUMMARY.json \
         T3_REACHABILITY.md T3_REACHABILITY.json crossseed_manifest.json; do
  cp -f "$OUT_ROOT/$f" "$BAK/" 2>/dev/null
done
cp -f "$CSV" "$RUNS_BASE/${RUN_TAG}_summary.csv" 2>/dev/null
cp -f "$OUT_ROOT/SUMMARY.md" "$RUNS_BASE/${RUN_TAG}_SUMMARY.md" 2>/dev/null
[ -f "$OUT_ROOT/HANDOVER.md" ] && cp -f "$OUT_ROOT/HANDOVER.md" "$RUNS_BASE/${RUN_TAG}_HANDOVER.md" 2>/dev/null
say "✅ 收尾完成；备份 $BAK"

say "──────── 交接摘要 ────────"
if [ -f "$OUT_ROOT/HANDOVER.md" ]; then
  # ⚠️ **2026-11-04 11:10 修**：本脚本开了 `set -o pipefail`（第 26 行），而这行是**脚本的最后一条命令**
  #    ⇒ `sed | head -40` 里 head 读够 40 行就退出 ⇒ sed 收到 SIGPIPE 退 141
  #    ⇒ pipefail 把 141 传成**整个 finalize 的退出码** ⇒ 上层续跑链全部记成「finalize rc=141 失败」
  #    （实际交付件全部正常产出，只是退出码脏了）。
  #    ⇒ 加 `|| true` 吞掉管道里的 SIGPIPE，不影响真正的失败语义。
  sed -n '/## 3. 判定链/,/## 6/p' "$OUT_ROOT/HANDOVER.md" | head -40 || true
fi
