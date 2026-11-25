#!/bin/bash
# 🟢 D5 波 · GPU1（2026-11-18 16:10 新建）—— GPU1 此前只跑门禁推理（25% 利用率）⇒ 本波补满算力
#   顺序：h22（Track-B × a 扫描 ⇒ 把 1544008 缺的 T1 找回来）→ h26（T1/T3/nano4 三供体混合汤）
#         → h25（换极小目标专家 1104407@1 ⇒ 攻严格轨天花板）→ h27（T1 供体家族超细扫）→ h23（2 成员 nano4 极值区）
#   铁律：flock 互斥 + PID 判活 + 起跑前查 MemAvailable（<40GB ⇒ 等 120s）+ OMP/MKL/OPENBLAS/NUMEXPR/NUMBA 全钉 4
set -u
REPO=/tmp/pycharm_project_10ae9e2e; PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search; END="2026-11-30 09:00"
cd $REPO || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
exec 9>"$W/.lock_d5_gpu1"
flock -n 9 || { echo "[$(date "+%F %T")] [D5-GPU1] 已有实例在跑 ⇒ 退出" >> "$W/d5_gpu1.log"; exit 0; }
log(){ echo "[$(date "+%F %T")] $*" >> $W/d5_gpu1.log; }
echo $$ > "$W/.pid_synth_gpu1"
cleanup_synth1(){ [ "$(cat "$W/.pid_synth_gpu1" 2>/dev/null)" = "$$" ] && rm -f "$W/.pid_synth_gpu1"; }
trap cleanup_synth1 EXIT
trap 'cleanup_synth1; exit 143' INT TERM
run3(){ # run3 <sid_base> <span> <recipes> <logfile>
  timeout 14400 $PY -u manu/pipeline/npu/night_soup3.py --gpu 1 --sid-base "$1" --sid-span "$2" \
    --max-gates 0 --mem-gate-gb 40 --deadline "$END" --recipes "$3" >> "$4" 2>&1 9>&-
}
clean_orphans(){
  for _p in $(ps -eo pid,ppid,args | grep "multiscale_heatmap_cache.py" | grep -v grep | awk '$2==1{print $1}'); do
    kill -9 "$_p" 2>/dev/null && echo "[清理] 孤儿缓存构建 pid=$_p" >> "$W/gate_orphan_clean.log"
  done
}
while :; do
  now=$(date +%s); end=$(date -d "$END" +%s)
  [ "$now" -lt "$end" ] || { log "[截止] $END ⇒ 退出"; break; }
  avail=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
  if [ "${avail:-0}" -lt 40960 ]; then log "[闸门] avail=${avail}MB < 40960MB ⇒ 等 120s"; sleep 120; continue; fi
  # 🟢🟢🟢 2026-11-19 16:45 —— **波次让位闸门**（内存优先给门禁）
  #   背景：门禁是**真瓶颈**（3 通道 × 8 臂/轮 ≈ 24 臂/小时），而波次出货 ~3 臂/分钟
  #         ⇒ 生产速度是消费速度的 10 倍 ⇒ 波次库存足够跑一整夜。
  #   代价：每条波占 1 个 `cache_trial0474_inferences`（~5.3GB）⇒ 4 条波 = 21GB
  #         ⇒ 把 MemAvailable 压到 40GB 以下 ⇒ **门禁壳自己的闸门（avail<40GB）会把门禁卡死**。
  #   规则：① `$W/.pause_waves` 存在 ⇒ 本轮让位（手动开关）；② avail < 60GB ⇒ 本轮让位。
  #   ⇒ 让位时**不退出进程**（保持 alive ⇒ 守望不会反复重启，也不会刷爆重启计数）。
  if [ -f "$W/.pause_waves" ]; then log "[让位] .pause_waves 存在 ⇒ 本轮跳过（内存优先给门禁）"; sleep 300; continue; fi
  if [ "${avail:-0}" -lt 61440 ]; then log "[让位] avail=${avail}MB < 60GB ⇒ 本轮让位给门禁"; sleep 300; continue; fi
  clean_orphans
  log "[轮] D5-GPU1 开始 avail=${avail}MB"
  # ★★★ 2026-11-18 16:35 改序：h32/h33/h34 = **F1×T1 交汇窗口**（今晚最核心发现）排最前
  # 🆕 2026-11-19 周末搜索空间（近线两族细扫 + 二阶汤 + nano4 金矿）
  # 🆕 2026-11-19 12:10 **w8 突破轴**（排在 w7 之前，优先级最高）
  #    目标：**打破 nano 天花板 .7242439** ⇒ 升轨判据是「全 >= 基线」，必须有轴严格更好才叫超越
  #    做法：基线族 {1614000@5,1104407@1} / A 族 + 9 个「nano 供体」（nano > 基线）剂量扫描
  RND8=$(cat $W/.w8_round_1 2>/dev/null || echo 0)
    # 🆕 2026-11-19 11:55 **w9 突破轴**（真针 1102033@1 + 老牌 14445/14049 + 官方三轴供体）
  RND9=$(cat $W/.w9_round_1 2>/dev/null || echo 0)
    # 🆕 2026-11-19 12:05 **w9b 突破轴**（高 N1 剂量 + nano 同族伙伴 + 老牌强臂主导）
  RND9B=$(cat $W/.w9b_round_1 2>/dev/null || echo 0)
    # 🆕 2026-11-19 12:20 **w10 前沿剃刀边精扫**（T3 差 5.2e-5 / F1 差 8.8e-5）
  RND10=$(cat $W/.w10_round_1 2>/dev/null || echo 0)
    # 🆕 2026-11-19 12:20 **w11 四臂合璧**（A族双雄 + T3/T2fp 供体 + F1 供体）
  RND11=$(cat $W/.w11_round_1 2>/dev/null || echo 0)
    # 🆕 2026-11-19 12:50 **w12 种子关攻坚**（bw07 扩族 + bw01 扩族 + nano4↑ 族修 T1/T3）
  # 🆕 2026-11-19 13:00 **w13 针尖精扫轴**（1102033 针族：nano4 高 1.2e-2，只差 nano 一格）
  RND13=$(cat $W/.w13_round_1 2>/dev/null || echo 0)
  # 🆕 2026-11-20 15:50 **T1 供体微剂量轴**（§269：唯一 100% 未开采、且真正对得上 T1 缺口的区域）
  #   · 实测：台账里**含任何 dT1>0 成员的汤臂 = 0 个** ⇒ 这块地完全没测过（不是「试过不行」）
  #   · 既有 w13 的 ax_t1donor/ax_t1donor6 挂在 **FAM5 骨架**上，而 FAM5 自身 dT1 = −9.26e-5
  #     ⇒ 任何供体的 T1 余量都补不上 ⇒ 那两轴**注定过不了 T1 关**（实测 35 臂全差 T1）
  #   · 本轴用**基线骨架**（dT1 恰好 = 0）+ 微剂量正 T1 供体 ⇒ dT1 直接转正
  #   · 臂号段 2050000~2053999（实测台账 max sid = 2030026 ⇒ 该段空闲）
  export T1D_MAX_OUT=24
  REC=$($PY $W/scripts/gen_t1donor_recipes.py 1); [ -n "$REC" ] && run3 2050000 4000 "$REC" $W/soup3_t1donor_gpu1.log
  # 🆕 2026-11-20 16:10 **T1 骨架补偿轴**（轴⑭ · gen_t1fix_recipes.py）
  #   · 骨架：现成汤里「**只差 T1、但 nano/nano4 有余量**」的骨架（2000321/2010351/1920134/1920144）
  #     典型 2000321：dN +2.36e-4、dN4 +4.73e-4、dF1 +3.25e-5、dT3 +1.61e-4、**dT1 −5.54e-5**、t2fp=3
  #   · 供体：T1 正增益供体**微剂量**（14963@4：dT1 +3.95e-4、dN −5.2e-3、dN4 +9.45e-4、dF1 +3.58e-3）
  #   · 实测：台账里「**同时**含 dT1>0 与 dT3>0 成员」的汤臂 = **0** ⇒ 这块地 100% 未开采
  #   · 臂号段 2060000~2063999（2050000=轴⑨，2100000+=训练轴 ⇒ 三段互不撞号）
  export T1FIX_MAX_OUT=24
  REC=$($PY $W/scripts/gen_t1fix_recipes.py 1); [ -n "$REC" ] && run3 2060000 4000 "$REC" $W/soup3_t1fix_gpu1.log
  REC=$($PY $W/scripts/gen_w13_recipes.py 1 $RND13)
  if [ -n "$REC" ]; then run3 2000000 4000 "$REC" $W/soup3_w13_gpu1.log; fi
  echo $((RND13+1)) > $W/.w13_round_1
  RND12=$(cat $W/.w12_round_1 2>/dev/null || echo 0)
  REC=$($PY $W/scripts/gen_w12_recipes.py 1 $RND12)
  if [ -n "$REC" ]; then run3 1960000 4000 "$REC" $W/soup3_w12_gpu1.log; fi
  echo $((RND12+1)) > $W/.w12_round_1
REC=$($PY $W/scripts/gen_w11_recipes.py 1 $RND11)
  if [ -n "$REC" ]; then run3 1920000 4000 "$REC" $W/soup3_w11_gpu1.log; fi
  echo $((RND11+1)) > $W/.w11_round_1
REC=$($PY $W/scripts/gen_w10_recipes.py 1 $RND10)
  if [ -n "$REC" ]; then run3 1880000 4000 "$REC" $W/soup3_w10_gpu1.log; fi
  echo $((RND10+1)) > $W/.w10_round_1
REC=$($PY $W/scripts/gen_w9b_recipes.py 1 $RND9B)
  if [ -n "$REC" ]; then run3 1840000 4000 "$REC" $W/soup3_w9b_gpu1.log; fi
  echo $((RND9B+1)) > $W/.w9b_round_1
REC=$($PY $W/scripts/gen_w9_recipes.py 1 $RND9)
  if [ -n "$REC" ]; then run3 1800000 4000 "$REC" $W/soup3_w9_gpu1.log; fi
  echo $((RND9+1)) > $W/.w9_round_1
REC=$($PY $W/scripts/gen_w8_recipes.py 1 $RND8)
  if [ -n "$REC" ]; then run3 1760000 4000 "$REC" $W/soup3_w8_d5_gpu1.log; fi
  echo $((RND8+1)) > $W/.w8_round_1
  # 🆕 2026-11-19 11:45 **w7 高精度扫描**（排在循环体首位）
  #    依据：基线族（含 T3 供体 1102002）密集网格 + A 族 × 9 个 T3 供体微剂量交叉
  #    round 轮换剂量阶梯 ⇒ 每轮巡游产出新配方（否则一轮跑完就空转）
  RND=$(cat $W/.w7_round_1 2>/dev/null || echo 0)
  REC=$($PY $W/scripts/gen_w7_recipes.py 1 $RND)
  if [ -n "$REC" ]; then run3 1720000 4000 "$REC" $W/soup3_w7_d5_gpu1.log; fi
  echo $((RND+1)) > $W/.w7_round_1
  REC=$($PY $W/scripts/gen_w6_recipes.py 1); [ -n "$REC" ] && run3 1694000 400 "$REC" $W/soup3_w61.log
  REC=$($PY $W/scripts/gen_w5_recipes.py 1); [ -n "$REC" ] && run3 1674000 400 "$REC" $W/soup3_w51.log
  REC=$($PY $W/scripts/gen_weekend_recipes.py w2); [ -n "$REC" ] && run3 1642000 300 "$REC" $W/soup3_w2.log
  REC=$($PY $W/scripts/gen_dyn_recipes.py 1 24); [ -n "$REC" ] && run3 1654000 400 "$REC" $W/soup3_dyn1.log
  # 🆕 2026-11-19 动态前沿汤（每次巡游重读台账 ⇒ 搜索跟着前沿走，而不是跑死列表）
  REC=$($PY $W/scripts/gen_d5_recipes.py h60); [ -n "$REC" ] && run3 1630000 200 "$REC" $W/soup3_h60.log
  REC=$($PY $W/scripts/gen_d5_recipes.py h32); [ -n "$REC" ] && run3 1605000 200 "$REC" $W/soup3_h32.log
  REC=$($PY $W/scripts/gen_d5_recipes.py h33); [ -n "$REC" ] && run3 1606000 200 "$REC" $W/soup3_h33.log
  REC=$($PY $W/scripts/gen_d5_recipes.py h35); [ -n "$REC" ] && run3 1608000 200 "$REC" $W/soup3_h35.log
  REC=$($PY $W/scripts/gen_d5_recipes.py h36); [ -n "$REC" ] && run3 1609000 200 "$REC" $W/soup3_h36.log
  REC=$($PY $W/scripts/gen_d5_recipes.py h37); [ -n "$REC" ] && run3 1610000 200 "$REC" $W/soup3_h37.log
  REC=$($PY $W/scripts/gen_d5_recipes.py h26); [ -n "$REC" ] && run3 1601000 200 "$REC" $W/soup3_h26.log
  REC=$($PY $W/scripts/gen_d5_recipes.py h34); [ -n "$REC" ] && run3 1607000 200 "$REC" $W/soup3_h34.log
  REC=$($PY $W/scripts/gen_d5_recipes.py h23); [ -n "$REC" ] && run3 1604000 200 "$REC" $W/soup3_h23.log
  REC=$($PY $W/scripts/gen_d5_recipes.py h27); [ -n "$REC" ] && run3 1603000 200 "$REC" $W/soup3_h27.log
  log "[轮] D5-GPU1 结束 ⇒ 睡 60s 再巡（幂等：已合成组合自动跳过）"
  sleep 60 9>&-
done
