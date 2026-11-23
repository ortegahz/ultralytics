#!/bin/bash
# 🟢🟢 T1 收口扫网 · **合成通道**（2026-11-23 新增）
#
# ★ 它做什么：按 `gen_t1close_recipes.py` 的细网格配方，**只做「合成 + NanoBench」**
#   （`night_soup3.py --max-gates 0` ⇒ **一格门禁机时都不占**），
#   把新臂写进台账 ⇒ **门禁通道 A/B/C 每轮重新排序时会自动把它们挑走**（`gate_prio.py`）。
#
# ★ 为什么这么做（实测依据，2026-11-23 上午）：
#   ① 六项判据里 **T1 是唯一真缺口**：FAM5 家族 89 个已门禁臂里大批是 5/6，
#      **只差 T1 0.5~1.2 阶**（5e-5~1.2e-4 ≈ 1~4 个检出）；T1 台阶量化 ≈1.5e-5 ⇒ 离散细胞
#      ⇒ **只能密集采样去撞**。
#   ② T1 杠杆 = **A 叶纯汤比例**（1603002/1610001/1603004/1609001，自身 T1 高基线 0.5~0.85 阶）。
#   ③ 🔴 **w3 上探假设已被证伪**（w3>.2116 的 91 个已门禁臂，T3 无一超过基线）⇒ 只做邻域细扫。
#   ④ 🔴 **分辨率地板**：展平 maxΔ<0.012 ⇒ 读数与基线逐位相同；0.012~0.030 ⇒ T1/T3 最大增益 = 0
#      ⇒ 合成配方必须落在**实测有产出的盒子**内（对照：FAM5 家族 maxΔ 0.055~0.095）。
#
# ★ 资源纪律：
#   · flock 互斥（同名锁 ⇒ 绝不双实例）
#   · **按 PID 文件判活**，守望脚本据此拉起；**禁 pkill -f**
#   · 起跑前查 MemAvailable（< 40GB ⇒ 等 120s）
#   · 臂号独占段（`--sid-base/--sid-span`）⇒ 与其它通道零撞号
#   · 合成通道**不占门禁通道额度**（门禁通道铁律 ≤3 不受影响）
#
# 用法（服务器）：
#   T1CLOSE_GPU=3 T1CLOSE_PARITY=0 setsid nohup bash $W/scripts/t1close_synth.sh >> $W/t1close_gpu3.log 2>&1 &
#   T1CLOSE_GPU=2 T1CLOSE_PARITY=1 setsid nohup bash $W/scripts/t1close_synth.sh >> $W/t1close_gpu2.log 2>&1 &
set -u
REPO=/tmp/pycharm_project_10ae9e2e; PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search; END="${T1CLOSE_END:-2026-11-23 21:00}"
GPU="${T1CLOSE_GPU:-3}"
PARITY="${T1CLOSE_PARITY:-0}"          # 0/1 ⇒ 与另一条合成通道**分片互斥**（零重复）
NCHUNK="${T1CLOSE_NCHUNK:-8}"
SIDBASE="${T1CLOSE_SID_BASE:-2130000}"     # 213xxxx 段 = nanolift 轴独占（212xxxx 已被 bridge 轴占满）
SIDSPAN="${T1CLOSE_SID_SPAN:-2000}"
MAXBATCH="${T1CLOSE_MAX_BATCH:-5}"         # nanolift 轴共 6 批（0~5）
# 🔴 2026-11-23 10:35 **默认生成器 = nanolift 轴**（A 叶四元组 × 桥接锚交叉配方）：
#   ① FAM5 细网格轴已证伪（125 个已门禁臂 dT1 ∈ [−22.23, −3.68]e-5，无一触到 0）
#   ② bridge 轴已探到 nano4 墙（外推到 w=0.45 的臂 dNano4 ∈ [−70.9, −117.9]e-5 全负）
#   ⇒ 唯一未被实测覆盖的坐标 = **「dNano4≥0 ∧ dT1>0」同时成立**（全库 0 臂）
#     ⇒ nanolift batch5 = A 叶四元组 {1100275,1102004,1104407,1623003} × 桥接锚 1612001
# ⚠️ **cron 复活只传 3 个 env**（t1close_cron.sh: T1CLOSE_GPU/PARITY/END）
#    ⇒ **默认值必须写死在这里**，否则 cron 一拉起就退回旧轴（本轮踩过：10:25 实例跑的是 bridge 轴）
GEN="${T1CLOSE_GEN:-$W/scripts/gen_nanolift_recipes.py}"   # 配方生成器可换
cd "$REPO" || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
LOG=$W/t1close_synth_gpu${GPU}.log
log(){ echo "[$(date "+%F %T")] $*" >> "$LOG"; }
exec 9>"$W/.lock_t1close_gpu${GPU}"
flock -n 9 || { log "🔴 已有实例在跑 ⇒ 退出"; exit 0; }
echo $$ > "$W/.pid_t1close_gpu${GPU}"

# 🔴🔴 2026-11-23 10:29~10:31 事故修复：**包装 shell 死掉时必须把 worker 一起带走**
#
#   坑：flock 挂在**包装 shell**上，真正干活的是子进程 `night_soup3.py`（经 timeout）。
#        包装 shell 一死（SIGHUP / SIGTERM / 被守望收割）⇒ flock 立即释放
#        ⇒ 守望按「PID 文件已死」拉起第二个实例
#        ⇒ **两个 night_soup3 拿同一个 --sid-base、同一批配方并行跑**。
#        实测证据：pid 819483 与 820183 的 `--recipes` 逐字节相同（10:29:40 与 10:30:03 同时在跑），
#        ⇒ 重复烧掉一整个 GPU 槽位 + 并发写同一个 `epoch5.pt` / `soup3_<sid>.pkl`。
#   修法（三条一起）：
#     ① `trap '' HUP`：**忽略 SIGHUP** —— 手工 ssh 起的通道不该因为「会话断了」被杀
#        （本轮事故就是这么来的：手工 ssh 启动 ⇒ 包装 shell 收到 SIGHUP ⇒ worker 变孤儿）。
#     ② 任何退出路径（EXIT/INT/TERM）都**按 PID 先杀 worker**，再释放锁 ⇒
#        「锁的生命周期」严格覆盖「worker 的生命周期」⇒ 不会出现「锁没了 worker 还活着」。
#     ③ worker PID 单独落文件 ⇒ 守望/运维能看出「有没有残留的孤儿 worker」。
#        （`night_soup3.py` 侧另有一道号段独占锁兜底，见该文件 acquire_sid_range_lock）
WORKER_PID=""
CLEANED=0
cleanup(){
  # 幂等：INT/TERM 的 trap 会先跑一次 cleanup，随后 exit 触发 EXIT trap 再跑一次
  [ "$CLEANED" -eq 1 ] && return 0
  CLEANED=1
  if [ -n "$WORKER_PID" ] && kill -0 "$WORKER_PID" 2>/dev/null; then
    log "🔴 收尾：杀 worker pid=$WORKER_PID（防 worker 变孤儿继续烧卡）"
    kill -TERM "$WORKER_PID" 2>/dev/null
    for _i in $(seq 1 20); do kill -0 "$WORKER_PID" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$WORKER_PID" 2>/dev/null && { kill -9 "$WORKER_PID" 2>/dev/null; sleep 1; }
  fi
  [ -f "$W/.pid_t1close_worker_gpu${GPU}" ] && \
    [ "$(cat "$W/.pid_t1close_worker_gpu${GPU}" 2>/dev/null)" = "$WORKER_PID" ] && \
    rm -f "$W/.pid_t1close_worker_gpu${GPU}"
  [ "$(cat "$W/.pid_t1close_gpu${GPU}" 2>/dev/null)" = "$$" ] && rm -f "$W/.pid_t1close_gpu${GPU}"
}
trap cleanup EXIT
trap 'cleanup; exit 143' INT TERM
trap '' HUP
log "🔒 启动 t1close 合成通道 gpu=$GPU parity=$PARITY nchunk=$NCHUNK sid=[$SIDBASE,$((SIDBASE+SIDSPAN))) END=$END pid=$$"
for B in $(seq 0 "$MAXBATCH"); do
  for C in $(seq 0 $((NCHUNK-1))); do
    # 分片互斥：两个实例各吃一半（NCHUNK 为偶数时严格不重叠）
    if [ $(( C % 2 )) -ne "$PARITY" ]; then continue; fi
    now=$(date +%s); end=$(date -d "$END" +%s)
    [ "$now" -lt "$end" ] || { log "[截止] $END ⇒ 退出"; exit 0; }
    avail=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
    if [ "${avail:-0}" -lt 40960 ]; then log "[闸门] avail=${avail}MB < 40960MB ⇒ 等 120s"; sleep 120; continue; fi
    # 🔴 配方先过**分辨率地板过滤器**（maxΔ<0.030 ⇒ 读数必与基线同格 ⇒ 纯烧 GPU）
    REC=$($PY "$GEN" "$B" "$C" "$NCHUNK" 2>>"$LOG" | $PY "$W/scripts/filter_live_recipes.py" 2>>"$LOG")
    [ -n "$REC" ] || continue
    N=$(printf '%s' "$REC" | tr ';' '\n' | grep -c . || true)
    log "[轮] batch=$B chunk=$C/$NCHUNK 配方 $N 条 ⇒ 开始合成（GPU$GPU）"
    # 🔴 后台跑 + wait ⇒ ① 能拿到 worker PID ② 收到信号时 wait 立刻返回并触发 trap 清理
    timeout 5400 $PY -u manu/pipeline/npu/night_soup3.py --gpu "$GPU" --recipes "$REC" \
      --sid-base "$SIDBASE" --sid-span "$SIDSPAN" --max-gates 0 --mem-gate-gb 40 \
      --deadline "$END" >> "$LOG" 2>&1 9>&- &
    WORKER_PID=$!
    echo "$WORKER_PID" > "$W/.pid_t1close_worker_gpu${GPU}"
    wait "$WORKER_PID"; rc=$?
    WORKER_PID=""
    rm -f "$W/.pid_t1close_worker_gpu${GPU}"
    log "[轮] batch=$B chunk=$C 结束 rc=$rc"
  done
done
log "t1close 合成通道结束"
