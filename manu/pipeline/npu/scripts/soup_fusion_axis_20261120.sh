#!/bin/bash
# 🟢🟢 2026-11-20 17:45 SB4「细网格汤融合轴」—— 两实例并行，父本集合互斥，臂号段互斥
# 🔴 动机（实测依据）：
#   ① 台账结构性互斥：950 个 nano4 富臂里 T1≥基线 **0 个**；49 个 T1≥基线的臂**全是汤臂**
#      ⇒ 「nano 富」与「T1 过线」互斥 ⇒ 必须靠**汤**。
#   ② 线性上界不够：14963@4(dT +21.3/dN −22) 配任何 nano 富臂，nano 增益上界仅 +2 阶（需 +22）。
#   ③ 同族汤实证压缩 nano 代价约 5 倍（同族汤臂 nano 缺口中位 −11 vs 训练臂 −57）⇒ 值得细网格实测。
#   ④ nano 源：cp 0.15~0.30 甜区 ⇒ 11042xx 家族 nano/nano4 双过基线且**从未门禁**（T1 未知）。
# 🟢 **不训练**（只 checkpoint 加权平均 + NanoBench）⇒ 不抢训练轴 GPU；**独立进程、绝不做 pkill**
#    ⇒ 崩了不影响 826 臂训练轴；台账 save_ledger 带跨进程 flock + 读-合并-写事务 ⇒ 并发写不丢。
# 🟢 幂等：按 (members, weights) 跳过已做组合；臂号独占段 + flock 原子占用 ⇒ 续跑不撞号。
# 📌 双实例（父本/号段互斥 ⇒ 零组合重叠）：
#    A 跨家族行：parents=14963:4,15510:4,1104642:5（child 仍 nano 富集臂）｜号段 2109000+
#    B 同族互汤行：parents=1104200:1,1104203:1,1104205:1（11042xx 家族内部互汤）｜号段 2109500+
# 🔴 2026-11-20 17:44 实测（跨家族行）：(14963@4,1104200@1) w=0.10/0.15/0.20/0.25
#    ⇒ nano −24/−50/−100/**−200** 阶，**远差于线性预测（+12 阶）** ⇒ 跨家族权重平均对 nano 召回
#    是**破坏性**的（两臂分属不同 lineage/盆地）⇒ 故 A/B 分实例、**B 同族行优先开跑**。
set -u
REPO=/tmp/pycharm_project_10ae9e2e; PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search; END="2026-11-23 07:00"
cd "$REPO" || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
TAG=${SB4_TAG:-A}
log(){ echo "[$(date "+%F %T")] $*" >> $W/soup_fusion_axis_$TAG.log; }
# 📌 铁律：任何「会 pkill 再拉起」的脚本都必须加 flock（cron 与手工是两个竞争拉起器 ⇒ 双实例 ⇒ 台账污染）
LOCK=$W/.soup_fusion_axis_$TAG.lock
exec 9>"$LOCK"
if ! flock -n 9; then log "🔴 已有另一个 SB4-$TAG 实例在跑 ⇒ 退出（防双实例）"; exit 0; fi
log "🔒 已取得 SB4-$TAG 锁 $LOCK"
CHILDREN="1104200:1,1104203:1,1104204:1,1104205:1,1104207:1"
WEIGHTS="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90"
case "$TAG" in
  A) PARENTS="14963:4,15510:4,1104642:5"; SIDBASE=2109000; SIDSPAN=400; GPU=${SB4_GPU:-1} ;;
  B) PARENTS="1104200:1,1104203:1,1104205:1"; SIDBASE=2109500; SIDSPAN=400; GPU=${SB4_GPU:-2} ;;
  *) echo "未知 TAG=$TAG" >&2; exit 2 ;;
esac
LOGF=$W/soup_fusion_axis_$TAG.log
while :; do
  now=$(date +%s); end=$(date -d "$END" +%s)
  [ "$now" -lt "$end" ] || { log "[截止] 退出"; break; }
  avail=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
  if [ "${avail:-0}" -lt 40960 ]; then log "[闸门] avail=${avail}MB < 40960MB ⇒ 等 120s"; sleep 120; continue; fi
  BEFORE=$(wc -l < $LOGF 2>/dev/null || echo 0)
  log "[轮] SB4-$TAG 启动 parents=$PARENTS children=$CHILDREN sid=[$SIDBASE,$((SIDBASE+SIDSPAN))) gpu=$GPU"
  timeout 7200 nice -n 15 $PY -u manu/pipeline/npu/night_soup_batch.py \
    --parents "$PARENTS" --children-arms "$CHILDREN" --weights "$WEIGHTS" \
    --sid-base $SIDBASE --sid-span $SIDSPAN --gpu $GPU \
    --max-gates 12 --gate-family-quota 40 --mem-gate-gb 40 --deadline "$END" \
    >> $LOGF 2>&1 9>&-
  rc=$?
  log "[轮] SB4-$TAG 结束 rc=$rc"
  # 幂等：已做组合会被跳过 ⇒ 若本轮**新增臂数 = 0** ⇒ 组合已跑完 ⇒ 退出，别空转
  NEW=$(tail -n +$((BEFORE+1)) $LOGF 2>/dev/null | grep -c "⇒ nano=" || true)
  log "[轮] SB4-$TAG 本轮新增臂数=$NEW"
  if [ "${NEW:-0}" -eq 0 ]; then log "[完成] 无新增组合 ⇒ SB4-$TAG 全部跑完 ⇒ 退出"; break; fi
  sleep 30
done
log "SB4-$TAG 轴结束"
