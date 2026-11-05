#!/usr/bin/env bash
# 重新生成 Fast Infer × no-GMC 的全部报告（纯读产物，不重跑任何评估）
# 用法：bash manu/tracking/lookahead/server_overnight/regen_reports_nogmc_fi.sh
# ⚠️ **执行顺序即依赖顺序**：
#    knob24 → knobscan_p3b → **accel_closure(§16)** → **reg_oracle_closure(§17)** → delta_dep → **工作点** → 权威报告 → house-style → 出框门。
#    FINAL_REPORT.md 的 §13 要读 knobscan_d3/report.json、§14 要读 delta_dep/report.json、
#    §11 要读 workpoint/audit.json、§16 要读 oracle_accel/closure_report.json、
#    §17 要读 reg_oracle/closure_report.json ⇒ 这些必须**先生成**，
#    否则 FINAL_REPORT 会读到上一轮产物
#    （新键缺失 ⇒ 静默回退；工作点审计缺失 ⇒ 直接「跳过本节」静默降级）。
set -uo pipefail

# ★ **异机可跑**：REPO/PY 默认值优先级 = ① 显式环境变量 ② 服务器仓根（存在则用 ⇒ 行为逐位不变）
#   ③ 包内 repo/ 镜像（host 随包跑）④ 系统 python3（交付链已兼容 **3.8+**）。
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${REPO:-}" ]; then
  if [ -d "/tmp/pycharm_project_10ae9e2e/runs/lookahead" ]; then
    REPO="/tmp/pycharm_project_10ae9e2e"
  elif [ -d "$SELF_DIR/../repo/runs/lookahead" ]; then
    REPO="$(cd "$SELF_DIR/../repo" && pwd)"
  else
    REPO="/tmp/pycharm_project_10ae9e2e"
  fi
fi
if [ -z "${PY:-}" ]; then
  if [ -x "/home/huangzhe/anaconda3/envs/uav/bin/python" ]; then
    PY="/home/huangzhe/anaconda3/envs/uav/bin/python"
  else
    PY="python3"
  fi
fi
cd "$REPO" || exit 1

# ★ **交付包只读闸**：所有 `--out-*` 都是**相对路径**（相对 `$REPO`）⇒ 若 `$REPO`
#   指向交付包内的 `repo/` 镜像，这个脚本会**直接写脏交付包**（包级+镜像级 md5 双 FAIL）。
#   包镜像根目录带 `MD5SUMS.txt`（仓根没有）⇒ 用它当「这是交付包镜像」的判据。
#   合法用法：仓根（服务器）或**冷启动沙箱** `$SB/repo`（无 MD5SUMS.txt）⇒ 不触发。
if [ -f "$REPO/MD5SUMS.txt" ]; then
  echo "[FATAL] \$REPO 指向交付包镜像（$REPO）⇒ 本脚本会写脏交付包 ⇒ 拒绝执行。"
  echo "       请改用仓根（服务器）或冷启动沙箱（\$SB/repo）作为 \$REPO。"
  exit 2
fi

# 生成器代码目录：**仓内优先，其次脚本自身目录**（交付包 code/）⇒ 同一份脚本仓内/包内两用
if [ -f "$REPO/manu/tracking/lookahead/make_fastinfer_final_report.py" ]; then
  CODE_DIR="$REPO/manu/tracking/lookahead"
elif [ -f "$SELF_DIR/make_fastinfer_final_report.py" ]; then
  CODE_DIR="$SELF_DIR"
else
  echo "[FATAL] 找不到生成器代码：$REPO/manu/tracking/lookahead/ 与 $SELF_DIR/ 均无 make_fastinfer_final_report.py"; exit 3
fi
echo "[$(date '+%F %T')] REPO=$REPO  CODE_DIR=$CODE_DIR"
export PYTHONPATH="$CODE_DIR:$REPO"

R8="runs/lookahead/fi_nogmc"
R24="runs/lookahead/fi_nogmc24"
rc_all=0

run() {  # run <说明> <命令...>
  local tag="$1"; shift
  "$@" > "/tmp/regen_${tag}.log" 2>&1
  local rc=$?
  echo "[$( [ $rc -eq 0 ] && echo OK || echo FAIL )] $tag (rc=$rc)"
  [ $rc -ne 0 ] && { tail -5 "/tmp/regen_${tag}.log"; rc_all=1; }
  return 0
}

echo "=== 1/9 24 序 Δ=3 专属旋钮扫描报告 ==="
run knob24 "$PY" "$CODE_DIR/make_knob_scan_24seq_report.py" \
  --root runs/lookahead/fi_nogmc24/knobscan_d3 \
  --ref-arms runs/lookahead/fovgate24nogmc/runs/cs_t0043/d3/fov__m6/arms.json \
  --model cs_t0043 --target-delta 3 \
  --out-md runs/lookahead/fi_nogmc24/knobscan_d3/report.md \
  --out-json runs/lookahead/fi_nogmc24/knobscan_d3/report.json

echo "=== 2/9 夜 6 新机理受控 A/B 报告（13 变体全表；FINAL_REPORT §15 读它） ==="
run knobscan_p3b "$PY" "$CODE_DIR/make_knob_scan_24seq_report.py" \
  --root runs/lookahead/fi_nogmc24/knobscan_p3b \
  --ref-arms runs/lookahead/fovgate24nogmc/runs/cs_t0043/d3/fov__m6/arms.json \
  --model cs_t0043 --target-delta 3 \
  --out-md runs/lookahead/fi_nogmc24/knobscan_p3b/report.md \
  --out-json runs/lookahead/fi_nogmc24/knobscan_p3b/report.json

echo "=== 3/9 位移修正族 oracle 上界 + 候选族结案报告（FINAL_REPORT §16 读它） ==="
# ★ 输入 oracle_bounds.json / closure.json 是**静态诊断产物**（与 gapdiag 同类：不在 regen 内重算），
#   但**必须存在且非空** ⇒ 否则 §16 会静默缺失（下面的硬门禁也会拦）。
run accel_closure "$PY" "$CODE_DIR/make_accel_closure_report.py" \
  --root runs/lookahead/fi_nogmc24/oracle_accel \
  --arms runs/lookahead/fovgate24nogmc/runs/cs_t0043/d3/fov__m6/arms.json \
  --out-md runs/lookahead/fi_nogmc24/oracle_accel/closure_report.md \
  --out-json runs/lookahead/fi_nogmc24/oracle_accel/closure_report.json

echo "=== 4/9 夜 8 配准/歧义族 oracle 上界 + 候选族结案报告（FINAL_REPORT §17 读它） ==="
# ★ 输入 reg_oracle.json 是**静态诊断产物**（由 diag_reg_oracle.py --analyze 生成，与 gapdiag 同类）
#   ⇒ 不在 regen 内重算，但**必须存在且非空**（下面的硬门禁也会拦）。
run reg_oracle_closure "$PY" "$CODE_DIR/make_reg_oracle_report.py" \
  --root runs/lookahead/fi_nogmc24/reg_oracle \
  --rbgate-root runs/lookahead/rbgate24nogmc \
  --out-md runs/lookahead/fi_nogmc24/reg_oracle/closure_report.md \
  --out-json runs/lookahead/fi_nogmc24/reg_oracle/closure_report.json

echo "=== 5/9 24 序 Δ 依赖性 + 惰性旋钮机理报告 ==="
run delta_dep "$PY" "$CODE_DIR/make_delta_dependence_report.py" \
  --root runs/lookahead/fi_nogmc24/knobscan_d3 \
  --d8-root runs/lookahead/fi_nogmc24/knobscan_d8dead \
  --rc2-root runs/lookahead/fi_nogmc24/knobscan_rc2_d3 \
  --out-md runs/lookahead/fi_nogmc24/delta_dep/report.md \
  --out-json runs/lookahead/fi_nogmc24/delta_dep/report.json

echo "=== 6/9 工作点对齐审计（★ 必须在权威报告之前：FINAL_REPORT §11 读它） ==="
run workpoint "$PY" "$CODE_DIR/make_workpoint_audit.py" \
  --root runs/lookahead/fi_nogmc/workpoint \
  --out-md runs/lookahead/fi_nogmc/workpoint/audit.md \
  --out-json runs/lookahead/fi_nogmc/workpoint/audit.json

echo "=== 7/9 权威报告 ==="
# 硬门禁：权威报告要消费的输入必须存在且非空（否则它会「跳过本节」静默降级）
#   ★ gapdiag/gap_report.json 是**帧级诊断的合成产物**（由 diag_*.json + gapdiag_frames.json 汇总），
#     与 gateA.json / nogmc24_byte_equal.json 同类：**静态诊断输入，不在 regen 内重算**，
#     但**必须存在且非空**，否则 FINAL_REPORT §15 会静默缺失。
for _need in runs/lookahead/fi_nogmc/workpoint/audit.json \
             runs/lookahead/fi_nogmc24/knobscan_d3/report.json \
             runs/lookahead/fi_nogmc24/knobscan_p3b/report.json \
             runs/lookahead/fi_nogmc24/gapdiag/gap_report.json \
             runs/lookahead/fi_nogmc24/delta_dep/report.json \
             runs/lookahead/gateA/gateA.json \
             runs/lookahead/gateA24/gateA.json \
             runs/lookahead/nogmc24_byte_equal.json \
             runs/lookahead/fi_nogmc24/oracle_accel/oracle_bounds.json \
             runs/lookahead/fi_nogmc24/oracle_accel/closure.json \
             runs/lookahead/fi_nogmc24/oracle_accel/closure_report.json \
             runs/lookahead/fi_nogmc24/reg_oracle/reg_oracle.json \
             runs/lookahead/fi_nogmc24/reg_oracle/closure_report.json; do
  if [ ! -s "$_need" ]; then
    echo "[FATAL] 权威报告的输入缺失或为空：$_need ⇒ 拒绝生成（会静默降级）"; rc_all=1; break
  fi
done
[ $rc_all -ne 0 ] && { echo "整体 rc_all=$rc_all"; exit $rc_all; }

run final_report "$PY" "$CODE_DIR/make_fastinfer_final_report.py" \
  --root-8 "$R8" --root-24 "$R24" \
  --gatea runs/lookahead/gateA/gateA.json \
  --out-md "$R8/FINAL_REPORT.md" --out-json "$R8/final_report.json"

run nogmc_report "$PY" "$CODE_DIR/make_fastinfer_nogmc_report.py" \
  --root "$R8" --root-24 "$R24" \
  --gate-a runs/lookahead/gateA/gateA.json \
  --byte-equal runs/lookahead/nogmc24_byte_equal.json \
  --out-md "$R8/report.md" --out-json "$R8/report.json"

echo "=== 8/9 house-style 逐序对照表 ==="
run hs_8seq "$PY" "$CODE_DIR/make_report_table.py" \
  --runs "$R8/rc2/d3/arms.json" "$R8/cs_t0043/d3/arms.json" "$R8/trial_0049/d3/arms.json" \
  --labels "RC2 trial_0080ep3 / GMC 语义" "cs_t0043 / no-GMC 语义" "trial_0049 ep20 / no-GMC 语义" \
  --title "Fast Infer × no-GMC 严格对照（Δ=3）" --delta 3 --model "rc2 / cs_t0043 / trial_0049" \
  --per-sequence --baseline stale --out-md "$R8/report_house_style.md"

for d in 2 3 4 8; do
  [ -f "$R24/rc2/d$d/arms.json" ] && [ -f "$R24/cs_t0043/d$d/arms.json" ] || { echo "[SKIP] house-style Δ=$d（产物不齐）"; continue; }
  run "hs_24seq_d$d" "$PY" "$CODE_DIR/make_report_table.py" \
    --runs "$R24/rc2/d$d/arms.json" "$R24/cs_t0043/d$d/arms.json" \
    --labels "RC2 trial_0080ep3 / GMC 语义" "cs_t0043 / no-GMC 语义" \
    --title "Fast Infer × no-GMC 严格对照（24 序官方口径）" --delta "$d" --model "rc2 / cs_t0043" \
    --per-sequence --baseline stale --out-md "$R24/report_house_style_d$d.md"
done
cp -f "$R24/report_house_style_d3.md" "$R24/report_house_style.md"

echo "=== 9/9 出框门受控 A/B 报告（7 份） ==="
run fov_8seq "$PY" "$CODE_DIR/make_fov_gate_report.py" \
  --root runs/lookahead/fovgate --ref-root "$R8" \
  --title "P3 新机理 · 出框门（FOV margin gate）受控 A/B · 8 序 gate 集" \
  --out-md runs/lookahead/fovgate/report.md --out-json runs/lookahead/fovgate/report.json
run fov_24rc2_d3 "$PY" "$CODE_DIR/make_fov_gate_report.py" \
  --root runs/lookahead/fovgate24 --ref-root "$R24" \
  --title "P3 新机理 · 出框门受控 A/B · 24 序 rc2 (GMC) · Δ=3" \
  --out-md runs/lookahead/fovgate24/report.md --out-json runs/lookahead/fovgate24/report.json
run fov_24cs_d3 "$PY" "$CODE_DIR/make_fov_gate_report.py" \
  --root runs/lookahead/fovgate24nogmc --ref-root "$R24" \
  --title "P3 新机理 · 出框门受控 A/B · 24 序 no-GMC (cs_t0043) · Δ=3" \
  --out-md runs/lookahead/fovgate24nogmc/report.md --out-json runs/lookahead/fovgate24nogmc/report.json
run fov_24cs_d2 "$PY" "$CODE_DIR/make_fov_gate_report.py" \
  --root runs/lookahead/fovgate24nogmc_d2 --ref-root "$R24" \
  --title "P3 新机理 · 出框门受控 A/B · 24 序 no-GMC (cs_t0043) · Δ=2（甜点档）" \
  --out-md runs/lookahead/fovgate24nogmc_d2/report.md --out-json runs/lookahead/fovgate24nogmc_d2/report.json
run fov_24cs_d8 "$PY" "$CODE_DIR/make_fov_gate_report.py" \
  --root runs/lookahead/fovgate24nogmc_d8 --ref-root "$R24" \
  --title "P3 新机理 · 出框门受控 A/B · 24 序 no-GMC (cs_t0043) · Δ=8（红线救援档）" \
  --out-md runs/lookahead/fovgate24nogmc_d8/report.md --out-json runs/lookahead/fovgate24nogmc_d8/report.json
run fov_24rc2_d8 "$PY" "$CODE_DIR/make_fov_gate_report.py" \
  --root runs/lookahead/fovgate24rc2_d8 --ref-root "$R24" \
  --title "P3 新机理 · 出框门受控 A/B · 24 序 rc2 (GMC) · Δ=8（红线边缘档）" \
  --out-md runs/lookahead/fovgate24rc2_d8/report.md --out-json runs/lookahead/fovgate24rc2_d8/report.json
run fov_24cs_d4 "$PY" "$CODE_DIR/make_fov_gate_report.py" \
  --root runs/lookahead/fovgate24nogmc_d4 --ref-root "$R24" \
  --title "P3 新机理 · 出框门受控 A/B · 24 序 no-GMC (cs_t0043) · Δ=4（补齐档）" \
  --out-md runs/lookahead/fovgate24nogmc_d4/report.md --out-json runs/lookahead/fovgate24nogmc_d4/report.json

echo
echo "整体 rc_all=$rc_all"
exit $rc_all
