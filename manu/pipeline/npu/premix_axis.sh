#!/bin/bash
# 两阶段预混轴：640001@5 × [ (1-r)·1100296@1 + r·1100296@2 ]
#   动机：@1 端 T3 落在好台阶(.9485822) 但 T1 偏低(.890527)；@2 端 T1 更高(.890823) 但 T3 差一档(.9484730)
#         ⇒ 把 @2 掺进 @1（r=5%~20%）有望「保住 T3 好台阶 + 抬 T1」⇒ 一次命中六项。
#   stage1: parent=1100296@1, child=1100296@2, w∈{0.8,0.85,0.9,0.95}  (w 给 parent ⇒ @2 占比 5~20%)
#   stage2: parent=640001@5, child=<premix_sid>:5, w∈{0.55,0.555,0.56}
#   注意：stage1 家族 = bs1100296:*（不计入关A）⇒ 打上 dummy official 让门禁队列跳过，省门禁算力
set -u
W=/mnt/data/siping/runs/npu20ms/nano_search
REPO=/tmp/pycharm_project_10ae9e2e
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
cd $REPO || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
log(){ echo "[$(date "+%F %T")] $*" >> $W/premix_axis.log; }
END="2026-11-18 07:00"
# ---- stage 1 ----
log "stage1 开始：1100296@1 × 1100296@2 预混"
timeout 3000 $PY -u manu/pipeline/npu/night_soup_batch.py --gpu 3 --parents 1100296:1 \
  --children-arms "1100296:2" --weights "0.8,0.85,0.9,0.95" --sid-base 1260000 --sid-span 100 \
  --max-gates 0 --gate-family-quota 2 --mem-gate-gb 44 --deadline "$END" >> $W/soup_new_1260000.log 2>&1
SIDS=$(grep -ao "sid=12[0-9]*" $W/soup_new_1260000.log | sed "s/sid=//" | sort -u | tr "\n" "," | sed "s/,$//")
log "stage1 完成，预混臂号 = $SIDS"
if [ -z "$SIDS" ]; then log "stage1 无产出 ⇒ 退出"; exit 0; fi
# ---- stage 2：640001@5 × 预混 ----
KIDS=$(for s in ${SIDS//,/ }; do echo -n "$s:5,"; done | sed "s/,$//")
log "stage2 开始：640001@5 × [$KIDS]"
timeout 5000 $PY -u manu/pipeline/npu/night_soup_batch.py --gpu 3 --parents 640001:5 \
  --children-arms "$KIDS" --weights "0.545,0.55,0.555,0.56,0.565" --sid-base 1270000 --sid-span 500 \
  --max-gates 0 --gate-family-quota 6 --mem-gate-gb 44 --deadline "$END" >> $W/soup_new_1270000.log 2>&1
log "stage2 完成"
# ---- stage1 臂打 dummy official（t1/t3 非 None）⇒ 门禁队列跳过它们，省算力 ----
$PY - <<PYEOF
import sys; sys.path.insert(0,"/tmp/pycharm_project_10ae9e2e")
import manu.pipeline.npu.nano_search as NS
led=NS.load_ledger()
n=0
for k,st in (led.get("arms") or {}).items():
    if not k.startswith("126"): continue
    if (st.get("soup_family") or "").startswith("bs1100296"):
        st["official"]={"t1":0.0,"t3":0.0,"t2_fp":0,"note":"premix-intermediate (skip gate)"}
        st["six_ok"]=False; n+=1
NS.save_ledger(led); print("marked",n)
PYEOF
log "预混轴全部完成"
