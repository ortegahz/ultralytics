#!/usr/bin/env bash
# 装配交付包：Fast Infer × no-GMC（第 2 夜收尾）
# 产物根：runs/lookahead/final/nogmc_fi_20261104/
# 用法：bash manu/tracking/lookahead/server_overnight/pack_delivery_nogmc_fi.sh
set -uo pipefail

REPO="/tmp/pycharm_project_10ae9e2e"
cd "$REPO" || exit 1

OUT="runs/lookahead/final/nogmc_fi_20261104"
mkdir -p "$OUT"

copy() {  # copy <src> <dst-name>
  if [ -f "$1" ]; then
    cp -f "$1" "$OUT/$2" && echo "[OK]   $2"
  else
    echo "[MISS] $1  (缺失，跳过)"; MISSING=$((MISSING+1))
  fi
}
MISSING=0

echo "=== 1/4 三份权威报告 ==="
copy runs/lookahead/fi_nogmc/FINAL_REPORT.md              FINAL_REPORT.md
copy runs/lookahead/fi_nogmc/final_report.json              FINAL_REPORT.json
copy runs/lookahead/fi_nogmc/report.md                      report.md
copy runs/lookahead/fi_nogmc/report.json                    report.json
copy runs/lookahead/fi_nogmc/report_house_style.md          report_house_style.md
copy runs/lookahead/fi_nogmc24/report_house_style.md        report_house_style_24seq_d3.md
copy runs/lookahead/fi_nogmc24/report_house_style_d2.md     report_house_style_24seq_d2.md
copy runs/lookahead/fi_nogmc24/report_house_style_d4.md     report_house_style_24seq_d4.md
copy runs/lookahead/fi_nogmc24/report_house_style_d8.md     report_house_style_24seq_d8.md

echo "=== 2/4 24 序汇总 / 门禁 / 口径自证 ==="
copy runs/lookahead/fi_nogmc24/summary.md                   summary_24seq.md
copy runs/lookahead/fi_nogmc24/summary.json                 summary_24seq.json
# ★ §8 产物索引引用的 8 序「全量指标真源」：仓内在 fi_nogmc/_facts/，
#   不入包 ⇒ 只拿包的评审者按 FINAL_REPORT.md §8 找不到 ⇒ 补进包根（与 24 序对称命名）。
copy runs/lookahead/fi_nogmc/_facts/summary.md             summary_8seq.md
copy runs/lookahead/fi_nogmc/_facts/all_metrics.json       summary_8seq.json
copy runs/lookahead/fi_nogmc/cfg_shared_8seq.json           cfg_shared_8seq.json
copy runs/lookahead/fi_nogmc/ruler1_gate_check.json         ruler1_gate_check.json
copy runs/lookahead/nogmc24_cfg_identical.json             nogmc24_cfg_identical.json
copy runs/lookahead/nogmc24_byte_equal.json                nogmc24_byte_equal.json
copy runs/lookahead/gateA/gateA.json                       gateA_8seq.json
copy runs/lookahead/gateA24/gateA.json                     gateA_24seq.json
# ★ 同上：Gate A 的人读报告（此前只 copy 了 .json）⇒ §8 索引的 .md 在包内定位不到。
copy runs/lookahead/gateA/gateA.md                        gateA_8seq.md
copy runs/lookahead/gateA24/gateA.md                      gateA_24seq.md
copy runs/lookahead/fi_nogmc/workpoint/audit.json           workpoint_audit.json
copy runs/lookahead/fi_nogmc/workpoint/audit.md             workpoint_audit.md

echo "=== 3/4 六份出框门受控 A/B 报告 ==="
copy runs/lookahead/fovgate/report.md                       fovgate_8seq_cs_t0043.md
copy runs/lookahead/fovgate/report.json                     fovgate_8seq_cs_t0043.json
copy runs/lookahead/fovgate24/report.md                     fovgate_24seq_rc2_d3.md
copy runs/lookahead/fovgate24/report.json                   fovgate_24seq_rc2_d3.json
copy runs/lookahead/fovgate24nogmc/report.md                fovgate_24seq_cs_t0043_d3.md
copy runs/lookahead/fovgate24nogmc/report.json              fovgate_24seq_cs_t0043_d3.json
copy runs/lookahead/fovgate24nogmc_d2/report.md             fovgate_24seq_cs_t0043_d2.md
copy runs/lookahead/fovgate24nogmc_d2/report.json           fovgate_24seq_cs_t0043_d2.json
copy runs/lookahead/fovgate24nogmc_d8/report.md             fovgate_24seq_cs_t0043_d8.md
copy runs/lookahead/fovgate24nogmc_d8/report.json           fovgate_24seq_cs_t0043_d8.json
copy runs/lookahead/fovgate24rc2_d8/report.md               fovgate_24seq_rc2_d8.md
copy runs/lookahead/fovgate24rc2_d8/report.json             fovgate_24seq_rc2_d8.json
copy runs/lookahead/fovgate24nogmc_d4/report.md             fovgate_24seq_cs_t0043_d4.md
copy runs/lookahead/fovgate24nogmc_d4/report.json           fovgate_24seq_cs_t0043_d4.json

echo "=== 3b/4 24 序 Δ=3 专属旋钮扫描报告 ==="
copy runs/lookahead/fi_nogmc24/knobscan_d3/report.md        knobscan_24seq_d3.md
copy runs/lookahead/fi_nogmc24/knobscan_d3/report.json      knobscan_24seq_d3.json

echo "=== 3b2/4 夜 6 新机理受控 A/B 报告（13 变体全表） ==="
copy runs/lookahead/fi_nogmc24/knobscan_p3b/report.md        knobscan_p3b_24seq.md
copy runs/lookahead/fi_nogmc24/knobscan_p3b/report.json      knobscan_p3b_24seq.json

echo "=== 3c/4 24 序 Δ 依赖性 + 惰性旋钮机理报告 ==="
copy runs/lookahead/fi_nogmc24/delta_dep/report.md          delta_dep_24seq.md
copy runs/lookahead/fi_nogmc24/delta_dep/report.json        delta_dep_24seq.json

echo "=== 3d/4 夜 6 帧级缺口诊断（证据链整包） ==="
mkdir -p "$OUT/gapdiag"
for f in gap_report.md gap_report.json gapdiag_frames.json gapdiag_frames.py \
         diag_fp_attrib.json diag_fp_attrib.md diag_gap_attribution.json \
         diag_miss_funnel.json diag_gap_buckets.json; do
  copy "runs/lookahead/fi_nogmc24/gapdiag/$f" "gapdiag/$f"
done

echo "=== 3e/4 夜 7 位移修正族 oracle 上界 + 候选族结案（证据链整包） ==="
mkdir -p "$OUT/oracle_accel"
for f in closure_report.md closure_report.json closure.md closure.json \
         oracle_bounds.md oracle_bounds.json oracle_bounds.py closure.py; do
  copy "runs/lookahead/fi_nogmc24/oracle_accel/$f" "oracle_accel/$f"
done

echo "=== 3f/4 夜 8 配准/歧义族 oracle 上界 + 候选族结案（证据链整包） ==="
mkdir -p "$OUT/reg_oracle"
for f in closure_report.md closure_report.json reg_oracle.md reg_oracle.json dump_summary.json; do
  copy "runs/lookahead/fi_nogmc24/reg_oracle/$f" "reg_oracle/$f"
done

echo "=== 3g/4 夜 8 回退链峰值门受控 A/B（唯一变量 = --replay-rollback-min-peak） ==="
mkdir -p "$OUT/rbgate"
for f in progress.txt; do
  copy "runs/lookahead/rbgate24nogmc/$f" "rbgate/$f"
done
for d in runs/lookahead/rbgate24nogmc/runs/cs_t0043/d3/rb__*; do
  [ -d "$d" ] || continue
  tag="$(basename "$d")"
  mkdir -p "$OUT/rbgate/$tag"
  copy "$d/arms.json" "rbgate/$tag/arms.json"
done

echo "=== 3h/4 夜 7~10 独立审计（§16 / §17 三份审计报告 + 原始发现留痕） ==="
# ★ 审计脚本**按内容锚点定位**报告行号 ⇒ 报告改版不再让审计崩溃；
#   `_round7_findings.md` = 原始发现（勿覆盖），同名无后缀 = 修复后复验版。
mkdir -p "$OUT/audit_round7"
for f in audit_sec16_gap.md audit_sec16_gap_round7_findings.md \
         audit_sec17_reg_oracle.md audit_sec17_reg_oracle_round7_findings.md \
         audit_sec17_rbgate.md audit_sec17_rbgate_round7_findings.md \
         audit_sec5_acceptance.md audit_sec5_deliverables.md \
         audit_pack_selfsufficient.md mutation_test_selfsufficient.md; do
  copy "runs/lookahead/fi_nogmc/audit_round7/$f" "audit_round7/$f"
done

echo "=== 4/4 OSD / 独立审计 / 代码 ==="
copy runs/lookahead/osd/osd_3arm_cs_t0043_d3.md             osd_3arm_cs_t0043_d3.md
copy runs/lookahead/osd/osd_3arm_cs_t0043_d3.json           osd_3arm_cs_t0043_d3.json

# OSD **视频本体**（肉眼可查的三臂对照证据）：host 只拉 runs/lookahead/{fi_nogmc,final}/
# ⇒ 视频必须进交付包，否则 §P4 的视觉证据随 osd/ 目录一起漏掉。
mkdir -p "$OUT/osd_videos"
for f in runs/lookahead/osd/videos/*_cs_t0043_d3.mp4 \
         runs/lookahead/osd/videos/*_cs_t0043_d3.json \
         runs/lookahead/osd/videos/*_cs_t0043_d3.log; do
  [ -f "$f" ] || continue
  cp -f "$f" "$OUT/osd_videos/$(basename "$f")" && echo "[OK]   osd_videos/$(basename "$f")"
done
copy runs/lookahead/fi_nogmc/_audit_round2_report_numbers.md audit_round2_report_numbers.md
copy runs/lookahead/fi_nogmc/_audit_round3_report_numbers.md audit_round3_report_numbers.md
copy runs/lookahead/fi_nogmc/_audit_round4_report_numbers.md audit_round4_report_numbers.md
copy runs/lookahead/fi_nogmc/_audit_round5_plugin_code.md       audit_round5_plugin_code.md
copy runs/lookahead/fi_nogmc/_audit_round6_sec15.md            audit_round6_sec15.md
copy runs/lookahead/fi_nogmc/_coldstart_repro.md                coldstart_repro.md

mkdir -p "$OUT/code"
for f in make_fastinfer_final_report.py make_fastinfer_nogmc_report.py \
         make_report_table.py make_fov_gate_report.py make_knob_scan_24seq_report.py \
         make_knob_scan_it2_report.py make_workpoint_audit.py \
         make_delta_dependence_report.py make_accel_closure_report.py \
         make_reg_oracle_report.py diag_reg_oracle.py \
         audit_sec16_gap.py audit_sec17_reg_oracle.py audit_sec17_rbgate.py \
         audit_sec5_acceptance.py audit_sec5_deliverables.py \
         audit_pack_selfsufficient.py; do
  copy "manu/tracking/lookahead/$f" "code/$f"
done
for f in summarize_nogmc24_results.py check_sweep_cfg_shared.py check_ruler1_gate.py \
         run_fov_gate_scan.sh run_fastinfer_nogmc_sweep.sh run_knob_scan_d3.sh \
         run_fastinfer_nogmc24_sweep.sh regen_reports_nogmc_fi.sh \
         run_reg_oracle_dump.sh run_rollback_gate_scan.sh \
         coldstart_verify_delivery.sh mutation_test_selfsufficient.py \
         verify_all.sh; do
  copy "manu/tracking/lookahead/server_overnight/$f" "code/$f"
done

echo "=== 5/5 仓镜像 repo/（交付包可独立复跑全部审计） ==="
# host 只拉 runs/lookahead/{fi_nogmc,final}/ ⇒ **24 序真源 arms.json、§17 帧级 oracle dump、
# 回退门产物、OSD 视频都不在拉取范围内** ⇒ 在包内按**仓内相对路径**镜像一份：
# 包内 code/ 的五份审计脚本会自定位到本镜像 ⇒ host 拿到包就能独立复跑、复现结论。
MIRROR="$OUT/repo"
MIRROR_N=0
rm -rf "$MIRROR"            # 清掉上一轮镜像（否则会把自己嵌套进去）
mirror() {  # mirror <仓内相对路径>
  local rel="$1"
  case "$rel" in "$OUT"/*) return;; esac   # 交付包自身不参与镜像（否则自包含）
  [ -f "$rel" ] || { echo "[MISS] $rel  (镜像缺失)"; MISSING=$((MISSING+1)); return; }
  mkdir -p "$MIRROR/$(dirname "$rel")"
  cp -f "$rel" "$MIRROR/$rel" && MIRROR_N=$((MIRROR_N+1))
}
mirror_dir() {  # mirror_dir <仓内相对目录>：镜像该目录**下所有文件**（不进子目录）
  local rel="$1" f
  [ -d "$rel" ] || { echo "[MISS] $rel  (镜像目录缺失)"; MISSING=$((MISSING+1)); return; }
  for f in "$rel"/*; do [ -f "$f" ] && mirror "$f"; done
}

# ① 全部 arms.json —— 报告与审计的**唯一真源**
while IFS= read -r f; do mirror "$f"; done < <(find runs/lookahead -name arms.json -not -path "$OUT/*" | sort)
# ② §17 帧级 oracle dump（jsonl）—— 独立重算的输入
mirror_dir runs/lookahead/fi_nogmc24/reg_oracle
mirror_dir runs/lookahead/fi_nogmc24/oracle_accel
mirror_dir runs/lookahead/fi_nogmc24/gapdiag
# ③ OSD 三臂视频（§P4 视觉证据；审计会检查该目录存在）
for f in runs/lookahead/osd/videos/*_cs_t0043_d3.*; do [ -f "$f" ] && mirror "$f"; done
# ④ 其余被审计直接读取的文件
for f in \
  runs/lookahead/fi_nogmc/FINAL_REPORT.md \
  runs/lookahead/fi_nogmc/report.md \
  runs/lookahead/fi_nogmc/report_house_style.md \
  runs/lookahead/fi_nogmc/ruler1_gate_check.json \
  runs/lookahead/fi_nogmc/workpoint/audit.json \
  runs/lookahead/fi_nogmc/workpoint/audit.md \
  runs/lookahead/nogmc24_byte_equal.json \
  runs/lookahead/nogmc24_cfg_identical.json \
  runs/lookahead/fi_nogmc24/knobscan_d3/manifest.tsv \
  runs/lookahead/fi_nogmc24/knobscan_p3b/manifest.tsv \
  runs/lookahead/fi_nogmc24/report_house_style_d2.md \
  runs/lookahead/fi_nogmc24/report_house_style_d3.md \
  runs/lookahead/fi_nogmc24/report_house_style_d4.md \
  runs/lookahead/fi_nogmc24/report_house_style_d8.md \
  runs/lookahead/gateA/gateA.json \
  runs/lookahead/gateA24/gateA.json \
  runs/lookahead/rbgate24nogmc/progress.txt ; do
  mirror "$f"
done
# ⑤ **报告重生成**（不只是审计）所需的静态输入 —— host 冷启动要能**逐位重生成全部 35 份报告**：
#   · cfg_shared_8seq.json      → FINAL_REPORT §11「8 序 gate 集 CFG」段
#   · fi_nogmc/knobscan/report.* → FINAL_REPORT §6「Δ=3 专属旋钮扫描」（8 序）
#   · knobscan_d3 的 gate__baseline **preds/** → delta_dep §1.3「限幅截断率」
#     （直接从 preds/*.json 的 per_frame 算 ⇒ 不重跑推理，但**必须带原始 preds**）
mirror runs/lookahead/fi_nogmc/cfg_shared_8seq.json
mirror runs/lookahead/fi_nogmc/knobscan/report.json
mirror runs/lookahead/fi_nogmc/knobscan/report.md
for _d in 2 3 4 8; do
  mirror_dir "runs/lookahead/fi_nogmc24/knobscan_d3/runs/cs_t0043/d${_d}/gate__baseline/preds"
done

cat > "$MIRROR/README.md" <<EOF
# 仓镜像（\`repo/\`）—— 交付包内可独立复跑全部审计

- **为什么有它**：host 只拉 \`runs/lookahead/{fi_nogmc,final}/\` ⇒ **24 序真源 \`arms.json\`、
  §17 帧级 oracle dump、回退门产物、OSD 视频都不在拉取范围内**。没有本镜像，包外的审计脚本
  读不到真源，"严格的精度对比报告" 就只能看已生成的 md，**无法独立复核**。
- **怎么用**：本目录按**仓内相对路径**镜像（\`runs/lookahead/...\`），包内 \`code/\` 的五份审计脚本
  会自定位到本镜像 ⇒ 在包外直接跑：

  \`\`\`bash
  cd <交付包>
  python code/audit_sec5_deliverables.py   # §5 三份交付物核验
  python code/audit_sec5_acceptance.py     # prompt §5 八项要求逐项验收
  python code/audit_sec16_gap.py           # §16 缺口审计
  python code/audit_sec17_reg_oracle.py    # §17 配准族
  python code/audit_sec17_rbgate.py        # 回退链峰值门
  python code/audit_pack_selfsufficient.py # 交付包自足性（校验**包级 + 镜像级**两份 MD5SUMS）
  python code/mutation_test_selfsufficient.py  # 审计的**判别力**证明（14 个用例）
  bash   code/verify_all.sh                  # **一键复核**（清单 + 六份审计 + 变异测试）
  \`\`\`

- **两层锁 + 一条锚定链**：包根 \`MD5SUMS.txt\`（覆盖包内每个文件，**含 \`repo/MD5SUMS.txt\`**）
  与镜像 \`repo/MD5SUMS.txt\`（覆盖镜像每个文件）互为备份，且**包级清单锚住镜像清单**
  ⇒ 只改镜像清单，包级立刻发现（早先打包器把 \`repo/MD5SUMS.txt\` 一起排除 ⇒ 镜像清单无锚）。
- **一键复核**：\`bash code/verify_all.sh\` 跑完「包级清单 + 镜像清单 + 包内六份审计 + 变异测试」
  并给出一行结论；★ 全程在**临时副本**里跑 ⇒ **不改动交付包本体**。
- **运行环境**：**Python ≥ 3.8**（系统 \`python3\` 即可；已在 **3.8.10** 与 **3.10.20** 双版本实测）。
  交付链刻意只用 3.8 通用语法/API —— 早前审计脚本用了 \`str.removeprefix\`（3.9+），
  在 python3.8 的机器上**必崩**（实测 AttributeError）⇒ host 复核会直接失败；现已改为 3.7+ 通用写法。
- **随包脚本的适用范围**：\`code/\` 下 \`diag_reg_oracle.py\` 是**服务端诊断脚本**
  （要 numpy/cv2/scipy + 完整 \`manu/\` 仓）⇒ **在交付包里跑不起来**，随包仅作留档；
  其余脚本（六份审计 + 报告生成器 + 一键复核 + 冷启动）在包内均可独立复跑。
- **复核脚本的「宁可红不可绿」闸**：缺 \`md5sum\`、Python < 3.8、清单「零通过项」、
  沙箱路径越界 ⇒ 一律 \`[FATAL]\` 退出（**绝不**把「没验」输出成「验过」）。
- **host 侧两条命令**（**不依赖服务器任何绝对路径**，脚本自定位）：
  \`\`\`bash
  bash code/verify_all.sh            # 一键复核（清单 + 六份审计 + 变异测试）⇒ 不改动交付包本体
  bash code/coldstart_verify_delivery.sh   # 冷启动复现（**副本沙箱**里重生成全部报告）⇒ 同样不改动交付包本体
  \`\`\`

- **对账**：\`MD5SUMS.txt\` 是镜像内每个文件的 md5（相对本目录）；\`audit_pack_selfsufficient.py\`
  会把镜像逐文件与仓内原件对账（md5 逐位相等）。
- 本镜像**只读**：审计脚本的产物写在 \`repo/runs/lookahead/fi_nogmc/audit_round7/\` 下。
EOF
# ★ 清单**最后**生成：README.md 也要进清单（否则「覆盖完整」判据会把 README 误判为多余文件）
# ★ 只排除**自己**这一份清单（`-not -path "./MD5SUMS.txt"`）：早先 `-not -name "MD5SUMS*"`
#   把 `repo/MD5SUMS.txt` 也排掉了 ⇒ 包级清单**锚不住**镜像清单 ⇒ 两把锁其实只剩一把。
( cd "$MIRROR" && find . -type f -not -path "./MD5SUMS.txt" -print0 | sort -z | xargs -0 md5sum > MD5SUMS.txt )
echo "  镜像文件数：$MIRROR_N   镜像体积：$(du -sh "$MIRROR" | cut -f1)"
echo "[OK]   $MIRROR"

echo "=== 校验和清单（镜像入库后重算） ==="
# ★ 清掉 __pycache__/*.pyc：脚本被 import 时会生成，既不是交付物、也不该进校验清单
find "$OUT" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null
find "$OUT" -name "*.pyc" -delete 2>/dev/null
( cd "$OUT" && find . -type f -not -path "./MD5SUMS.txt" -print0 | sort -z | xargs -0 md5sum > MD5SUMS.txt )
echo "[OK]   $OUT/MD5SUMS.txt"
echo
echo "=== 交付包内容 ==="
ls -la "$OUT"
echo
echo "文件数：$(find "$OUT" -type f | wc -l)   缺失源文件数：$MISSING"
echo "PACK_DIR=$OUT"
