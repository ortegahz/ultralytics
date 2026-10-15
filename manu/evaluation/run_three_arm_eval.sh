#!/bin/sh
# =============================================================================
# 三臂受控 A/B 编排 —— NPU int8 去 P0 vs PC SOTA
#
# 三臂共用同一批 PC-BMP 特征、同一 harvest（--conf-thresh 0.08 / --top-k 80）、
# 同一 15 档阈值、同一 8px 匹配。唯一变量是推理后端。
#
#   臂1 PC-FP32 + P0   runs/gmc_eval/pc_bmp_full_cache.pkl    （冻结基准）
#   臂2 PC-FP32 - P0   runs/gmc_eval/pc_bmp_nop0_cache.pkl
#   臂3 NPU-int8 - P0  runs/gmc_eval/arm_npu_nop0_cache.pkl   （本脚本生成）
#
# eval_arm_pc_gmc_ab.py 一次只吃两臂，所以跑 **3 次两两对比** 来分解代价：
#   对比1  臂1 -> 臂2   去 P0 的代价（FP32 域）
#   对比2  臂2 -> 臂3   int8 量化的代价（已去 P0 域）
#   对比3  臂1 -> 臂3   合计损失  ← 用户要的 headline
#
# 自校准闸门：eval 脚本只校验 **A 臂** 峰值。对比 1/3 以臂1（A）为 A 臂，
# 传入冻结 headline（F1 0.902316 / TP 21601 / FP 1167）做硬校验，复现不了
# 就 exit 2，保证后面所有结论都建立在可复现的基准上。
# 对比 2 以臂2（A）为 A 臂，其期望值从对比 1 的 JSON 里取，不手填。
# =============================================================================
set -eu

ROOT=${ROOT:-/tmp/pycharm_project_10ae9e2e}
PY=${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}
GE=$ROOT/runs/gmc_eval
TSV_ROOT=${TSV_ROOT:-$ROOT/runs/npu_pcfeat_tsv}     # 板上 TSV 已 rsync 至此
OUT=$GE/three_arm

A1=$GE/pc_bmp_full_cache.pkl          # PC-FP32 + P0
A2=$GE/pc_bmp_nop0_cache.pkl          # PC-FP32 - P0
A3=$GE/arm_npu_nop0_cache.pkl         # NPU-int8 - P0

mkdir -p "$OUT"

echo "############ [STEP 1] TSV -> 臂3 cache pkl ############"
if [ -f "$A3" ]; then
    echo "[SKIP] 臂3 已存在: $A3"
else
    "$PY" "$ROOT/manu/diagnostics/npu_tsv_to_cache.py" \
        --tsv-root "$TSV_ROOT" \
        --gt-cache "$A2" \
        --out "$A3" \
        --expect-total 31613
fi

echo
echo "############ [STEP 2] 对比1: 臂1 -> 臂2（去 P0 的代价, FP32 域）############"
"$PY" "$ROOT/manu/evaluation/eval_arm_pc_gmc_ab.py" \
    --arm-a "$A1" --label-a "PC-FP32+P0" \
    --arm-b "$A2" --label-b "PC-FP32-P0" \
    --tol 8.0 --expect-f1 0.902316 --expect-tp 21601 --expect-fp 1167 \
    --out-json "$OUT/cmp1_arm1_vs_arm2.json" | tee "$OUT/cmp1_arm1_vs_arm2.txt"

echo
echo "############ [STEP 3] 对比2: 臂2 -> 臂3（int8 量化的代价, 已去 P0 域）############"
# 期望值从对比1的 B 臂峰值取，避免手抄
EXP2=$("$PY" -c "
import json,sys
d=json.load(open('$OUT/cmp1_arm1_vs_arm2.json'))
p=d['arm_b']['peak']
print('%.6f %d %d' % (p['f1'], p['tp'], p['fp']))
")
set -- $EXP2
echo "[对比2 闸门期望 臂2 峰值] F1=$1 TP=$2 FP=$3"
"$PY" "$ROOT/manu/evaluation/eval_arm_pc_gmc_ab.py" \
    --arm-a "$A2" --label-a "PC-FP32-P0" \
    --arm-b "$A3" --label-b "NPU-int8-P0" \
    --tol 8.0 --expect-f1 "$1" --expect-tp "$2" --expect-fp "$3" \
    --out-json "$OUT/cmp2_arm2_vs_arm3.json" | tee "$OUT/cmp2_arm2_vs_arm3.txt"

echo
echo "############ [STEP 4] 对比3: 臂1 -> 臂3（合计损失, headline）############"
"$PY" "$ROOT/manu/evaluation/eval_arm_pc_gmc_ab.py" \
    --arm-a "$A1" --label-a "PC-FP32+P0" \
    --arm-b "$A3" --label-b "NPU-int8-P0" \
    --tol 8.0 --expect-f1 0.902316 --expect-tp 21601 --expect-fp 1167 \
    --out-json "$OUT/cmp3_arm1_vs_arm3.json" | tee "$OUT/cmp3_arm1_vs_arm3.txt"

echo
echo "############ [STEP 5] 汇总三臂峰值对照表 ############"
"$PY" - "$OUT" <<'PYEOF'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
r2 = json.loads((out / "cmp2_arm2_vs_arm3.json").read_text())
r3 = json.loads((out / "cmp3_arm1_vs_arm3.json").read_text())

# 臂1 从对比3的 A 臂取，臂2 从对比2的 A 臂取，臂3 从对比3的 B 臂取
arms = [
    ("臂1  PC-FP32 +P0", r3["arm_a"]["peak"]),
    ("臂2  PC-FP32 -P0", r2["arm_a"]["peak"]),
    ("臂3  NPU-int8 -P0", r3["arm_b"]["peak"]),
]
print(f"{'臂':<18} {'th':>5} {'F1':>10} {'P%':>7} {'R%':>7} {'TP':>7} {'FP':>7}")
print("-" * 66)
for lab, p in arms:
    print(f"{lab:<18} {p['th']:>5.2f} {p['f1']:>10.6f} "
          f"{p['precision']*100:>7.2f} {p['recall']*100:>7.2f} {p['tp']:>7} {p['fp']:>7}")

base = arms[0][1]
print("-" * 66)
print(f"{'对比':<18} {'dF1':>10} {'dTP':>6}")
print("-" * 66)
for lab, p in arms[1:]:
    print(f"{lab:<18} {p['f1']-base['f1']:>+10.6f} {p['tp']-base['tp']:>+6}")
print("-" * 66)
print("去 P0 代价 (臂1->臂2) = %+.6f   (TP %+d)" % (
    arms[1][1]["f1"]-base["f1"], arms[1][1]["tp"]-base["tp"]))
print("int8 代价 (臂2->臂3) = %+.6f   (TP %+d)" % (
    arms[2][1]["f1"]-arms[1][1]["f1"], arms[2][1]["tp"]-arms[1][1]["tp"]))
print("合计    (臂1->臂3) = %+.6f   (TP %+d)" % (
    arms[2][1]["f1"]-base["f1"], arms[2][1]["tp"]-base["tp"]))
PYEOF

echo
echo "THREE_ARM_EVAL_DONE"
