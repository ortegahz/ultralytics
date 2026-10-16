#!/bin/sh
# =============================================================================
# int8 vs fp16 精度对照编排（在训练服务器上执行）
#
# 背景：把 NPU 推理后端从 int8 换成 fp16，其他一切不变。四臂共用同一批 PC-BMP
# 特征、同一 harvest（--th 0.08 / --topk 80）、同一 15 档阈值、同一 8px 匹配。
# 唯一变量是**模型的数值精度**。
#
#   臂1  PC-FP32 +P0   runs/gmc_eval/pc_bmp_full_cache.pkl      冻结基准
#   臂2  PC-FP32 -P0   runs/gmc_eval/pc_bmp_nop0_cache.pkl
#   臂3  NPU-int8 -P0  runs/gmc_eval/arm_npu_nop0_cache.pkl      昨天的 int8 臂
#   臂4  NPU-fp16 -P0  runs/gmc_eval/arm_npu_f16_nop0_cache.pkl  本脚本生成
#
# eval_arm_pc_gmc_ab.py 一次只吃两臂，所以跑两次两两对比：
#   对比F  臂2 -> 臂4   fp16 相对 FP32 的代价
#   对比I  臂3 -> 臂4   ** fp16 相对 int8 挽回多少（用户要的那张表）**
#
# 自校准闸门：eval 脚本只校验 A 臂峰值。两个期望值都**从既有 JSON 取，不手填**：
#   臂2 期望 <- runs/gmc_eval/three_arm/cmp2_arm2_vs_arm3.json 的 arm_a.peak
#   臂3 期望 <- runs/gmc_eval/three_arm/cmp2_arm2_vs_arm3.json 的 arm_b.peak
#   （即昨天已经验过、可复现的那两个数；复现不了就 exit 2）
# =============================================================================
set -eu

ROOT=${ROOT:-/tmp/pycharm_project_10ae9e2e}
PY=${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}
GE=$ROOT/runs/gmc_eval
TSV_ROOT=${TSV_ROOT:-$ROOT/runs/npu_pcfeat_f16_tsv}   # 板上 fp16 TSV 已 rsync 至此
OUT=$GE/int8_vs_f16
PREV=$GE/three_arm/cmp2_arm2_vs_arm3.json

A1=$GE/pc_bmp_full_cache.pkl          # 臂1 PC-FP32 +P0
A2=$GE/pc_bmp_nop0_cache.pkl          # 臂2 PC-FP32 -P0
A3=$GE/arm_npu_nop0_cache.pkl         # 臂3 NPU-int8 -P0
A4=$GE/arm_npu_f16_nop0_cache.pkl     # 臂4 NPU-fp16 -P0

mkdir -p "$OUT"

echo "############ [STEP 1] fp16 TSV -> 臂4 cache pkl ############"
if [ -f "$A4" ]; then
    echo "[SKIP] 臂4 已存在: $A4"
else
    "$PY" "$ROOT/manu/diagnostics/npu_tsv_to_cache.py" \
        --tsv-root "$TSV_ROOT" \
        --gt-cache "$A2" \
        --out "$A4" \
        --expect-total 31613
fi

echo
echo "############ [STEP 2] 取闸门期望值（昨天已验证的两个峰）############"
[ -f "$PREV" ] || { echo "[FATAL] 缺 $PREV，无法取期望值"; exit 2; }
set -- $("$PY" -c "
import json
d = json.load(open('$PREV'))
a2, a3 = d['arm_a']['peak'], d['arm_b']['peak']
print('%.6f %d %d' % (a2['f1'], a2['tp'], a2['fp']))
print('%.6f %d %d' % (a3['f1'], a3['tp'], a3['fp']))
")
E2_F1=$1; E2_TP=$2; E2_FP=$3; E3_F1=$4; E3_TP=$5; E3_FP=$6
echo "[闸门] 臂2 PC-FP32-P0 期望 F1=$E2_F1 TP=$E2_TP FP=$E2_FP"
echo "[闸门] 臂3 NPU-int8-P0 期望 F1=$E3_F1 TP=$E3_TP FP=$E3_FP"

echo
echo "############ [STEP 3] 对比F: 臂2 -> 臂4（fp16 相对 FP32 的代价）############"
"$PY" "$ROOT/manu/evaluation/eval_arm_pc_gmc_ab.py" \
    --arm-a "$A2" --label-a "PC-FP32-P0" \
    --arm-b "$A4" --label-b "NPU-fp16-P0" \
    --tol 8.0 --expect-f1 "$E2_F1" --expect-tp "$E2_TP" --expect-fp "$E2_FP" \
    --out-json "$OUT/cmpF_arm2_vs_arm4.json" | tee "$OUT/cmpF_arm2_vs_arm4.txt"

echo
echo "############ [STEP 4] 对比I: 臂3 -> 臂4（fp16 相对 int8 挽回多少）############"
"$PY" "$ROOT/manu/evaluation/eval_arm_pc_gmc_ab.py" \
    --arm-a "$A3" --label-a "NPU-int8-P0" \
    --arm-b "$A4" --label-b "NPU-fp16-P0" \
    --tol 8.0 --expect-f1 "$E3_F1" --expect-tp "$E3_TP" --expect-fp "$E3_FP" \
    --out-json "$OUT/cmpI_arm3_vs_arm4.json" | tee "$OUT/cmpI_arm3_vs_arm4.txt"

echo
echo "############ [STEP 5] 四臂峰值对照表 ############"
"$PY" - "$OUT" "$GE/three_arm/cmp3_arm1_vs_arm3.json" <<'PYEOF'
import json, sys, pathlib
out, prev = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
r2 = json.loads((out / "cmpF_arm2_vs_arm4.json").read_text())
rI = json.loads((out / "cmpI_arm3_vs_arm4.json").read_text())
p  = json.loads(prev.read_text())

# 臂1 必须来自 cmp3（臂1 vs 臂3）的 A 臂。**不能**用 cmp2：那份是「臂2 vs 臂3」，
# 它的 arm_a 是臂2 而不是臂1，取错会得到「臂1 == 臂2」的假零代价。
src = [
    ("臂1  PC-FP32 +P0", p["arm_a"]["label"], p["arm_a"]["peak"]),
    ("臂2  PC-FP32 -P0", r2["arm_a"]["label"], r2["arm_a"]["peak"]),
    ("臂3  NPU-int8 -P0", rI["arm_a"]["label"], rI["arm_a"]["peak"]),
    ("臂4  NPU-fp16 -P0", r2["arm_b"]["label"], r2["arm_b"]["peak"]),
]
want = ["PC-FP32+P0", "PC-FP32-P0", "NPU-int8-P0", "NPU-fp16-P0"]
for (lab, src_label, _), w in zip(src, want):
    if src_label != w:
        raise SystemExit(f"[FATAL] {lab} 取错来源：JSON 里标签是 '{src_label}'，应为 '{w}'")

arms = [(lab, pk) for lab, _, pk in src]
print(f"{'臂':<18} {'源标签':<13} {'th':>5} {'F1':>10} {'P%':>7} {'R%':>7} {'TP':>7} {'FP':>7}")
print("-" * 82)
for (lab, src_label, pk) in src:
    print(f"{lab:<18} {src_label:<13} {pk['th']:>5.2f} {pk['f1']:>10.6f} "
          f"{pk['precision']*100:>7.2f} {pk['recall']*100:>7.2f} {pk['tp']:>7} {pk['fp']:>7}")

base = arms[0][1]
a2, a3, a4 = arms[1][1], arms[2][1], arms[3][1]
print("-" * 66)
print(f"{'分解':<18} {'dF1':>10} {'dTP':>6}")
print("-" * 66)
print(f"{'去 P0 (1->2)':<18} {a2['f1']-base['f1']:>+10.6f} {a2['tp']-base['tp']:>+6}")
print(f"{'int8 (2->3)':<18} {a3['f1']-a2['f1']:>+10.6f} {a3['tp']-a2['tp']:>+6}")
print(f"{'fp16 (2->4)':<18} {a4['f1']-a2['f1']:>+10.6f} {a4['tp']-a2['tp']:>+6}")
print(f"{'fp16挽回 (3->4)':<18} {a4['f1']-a3['f1']:>+10.6f} {a4['tp']-a3['tp']:>+6}")
print(f"{'合计 (1->3)':<18} {a3['f1']-base['f1']:>+10.6f} {a3['tp']-base['tp']:>+6}")
print(f"{'合计 (1->4)':<18} {a4['f1']-base['f1']:>+10.6f} {a4['tp']-base['tp']:>+6}")
print("-" * 66)
d_i8 = a2['f1'] - a3['f1']
d_f16 = a2['f1'] - a4['f1']
print(f"int8 残留损失 (臂2-臂3) = {d_i8:.6f}")
print(f"fp16 残留损失 (臂2-臂4) = {d_f16:.6f}")
if d_i8 > 0:
    print(f"fp16 挽回比例 = {(a4['f1']-a3['f1'])/d_i8*100:.1f}%")
else:
    # 臂3 不劣于臂2 时比例无定义（负损失或零损失），不能打印一个假百分比
    print("fp16 挽回比例 = 无定义（臂3 未低于臂2，比例分母非正）")
PYEOF

echo
echo "INT8_VS_F16_EVAL_DONE"
