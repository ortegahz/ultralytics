#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# 🟢 周末**新成员生产线**启动器（2026-11-20）—— 汤的空间已证伪饱和后的唯一出路
#
# 📌 为什么要起训练轴（§268 §18 实测结论，不是猜测）
#   · 汤的权重空间：**空集**。1,154 个有完整六项读数的臂里，
#     「其余五项 ≥ 基线 且 dT1 ≥ 0 且 dT3 > 0」的臂 = **0 个**。
#   · 成员层面同样封死：292 个训练成员里只有 19 个 dT1 > 0，**全部 dNano < 0**
#     ⇒ 现有成员的凸包**不含**任何支配基线的点。
#   ⇒ **冠军只能来自新训练的成员**；而「从当前基线热启动的 2-epoch 微调」是
#     唯一能在**不牺牲 nano/nano4** 的前提下把 T1/T3 往上推的动作
#     （起点六项 = 基线本身 ⇒ 任何净改善都是严格更优）。
#   ⇒ 训练锚点已改成 **V3 = {基线 v3.1 汤, 基线的 4 个成员, 1543009}**，
#     抛光配方以**基线自己的配方**（2-epoch / lr_scale=0.3 / anchor_l2=0.001 /
#     cp=0.05 / sw=4.0）为中心。
#
# ⚠️ **本脚本会启动训练**（铁律一：训练/推理由用户手工执行，Agent 不代劳）。
# ⚠️ 幂等：会先杀掉旧训练轴及其训练子进程再拉起（否则两个实例写同一个臂号目录）。
# 用法：
#     bash weekend_train_axis_20261120.sh --dry-run    # 只打印将要做什么
#     bash weekend_train_axis_20261120.sh              # 真执行
# ══════════════════════════════════════════════════════════════════════════
set -u
W=/mnt/data/siping/runs/npu20ms/nano_search
REPO=/tmp/pycharm_project_10ae9e2e
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# 🔴 截止时刻 = 2026-11-23 08:00（比波次 08:30 早 30 min）
#   ⇒ 最后一条臂能跑完训练 + 官方门禁，不会在 08:30 被腰斩留下半截读数。
DL="2026-11-23 08:00"
# 🔴 并发 2（不是 4/5）：实测每个训练臂 **PSS 10.8~11.1 GB**（8 个 dataloader worker）
#   ⇒ 4 臂 = 44 GB ⇒ 加上波次/门禁后 MemAvailable 会跌破
#      **40 GB 守望闸门**（`day_supervisor_d1.sh:69`）与波次壳的 40/60 GB 闸门
#   ⇒ 波次会被拖停，**得不偿失**。2 臂 ≈ 22 GB ⇒ 余量安全。
# 🔴🔴 2026-11-20 16:20 **并发从 2 收敛到 1（默认 1，可用环境变量 ARMS 覆盖）**
#   背景：实测（16:07~16:16）本脚本的 **2 臂** 与「4 波次 + 3 门禁 + 缓存构建」叠加后
#         MemAvailable 从 90 GB 一路掉到 **29 GB**（训练子进程 22 个，RSS 合计 48 GB；
#         另有 6 个 multiscale_heatmap_cache ≈ 5 GB/个）。
#   风险闸门：`nano_watchdog.sh` 在 **<25 GB 杀最新门禁批次**、**<12 GB 全停**（门禁+缓存构建）
#         ⇒ 29 GB 的地板离 25 GB 只有 4 GB 余量 ⇒ 一旦叠加波峰就可能踩到「门禁被杀」。
#   为什么降到 1 臂不心疼：**门禁是真瓶颈（3 通道 ≈ 24 臂/小时）**，而：
#         · 训练臂产能 ≈ 1.25 臂/小时（48 min/臂）⇒ **远低于门禁消费速度** ⇒ 1 臂就够喂饱门禁；
#         · 汤轴（轴⑨/⑭）候选 159 + 831 条 ⇒ 门禁的**供给**从来不缺；
#         · 训练轴的价值在于「**唯一能产出新冠军成员**的通道」（§268 §18：汤空间已证伪饱和），
#           而**产出速度**不是瓶颈 ⇒ 1 臂足够。
#   ⇒ 腾出的 ~16 GB 全部回给门禁与波次 ⇒ 地板回到 ~40 GB，离两条刹车线都有余量。
ARMS=${ARMS:-3}
GPUS="0,2,3"   # 🔴🔴🔴 2026-11-20 17:05 **3 臂 × 3 卡**（§280 周末重排）
#   理由：① 汤（权重平均）空间已实测封死（轴⑨/⑭ 30 臂三轴全过 0 个）⇒ 4 条波次壳全停
#         ⇒ 4 张卡的 GPU 与 ~25 GB 内存全部让给训练轴；
#      ② 训练轴是**唯一**能产出新冠军成员的通道（§268 §18）⇒ 周末唯一值得烧的算力；
#      ③ 并发 3 臂（每臂 PSS ~11 GB ⇒ 稳态 ~33 GB）+ 3 条门禁通道（≤15 GB）
#         ⇒ 稳态 MemAvailable ≈ 60~70 GB（在外部视频质量作业结束后）；
#      ④ 内存不足时**自动降档**：night_train_axis.py 的 MEM_START_MIN_GB 门槛（42→26 GB）
#         会在起臂前拦 ⇒ 压力下自动退到 1 臂，压力解除后自动升到 3 臂。
#   ⚠ 若要回滚到 1 臂：ARMS=1 bash weekend_train_axis_20261120.sh
POLISH=600     # 实测 P 臂 ≈ 9.2 min（服务器日志）⇒ 4 并发 65 h ≈ 1700 臂上限；600 是安全预算
# 🔴🔴🔴 2026-11-20 17:02 **L5 臂预算 = 28**（§282：T1 供体「二代长训」轴）
#   · 7 个实测 T1 供体锚点 × {lr1.5, lr3.5} × {cp0.05, cp0.15} = 28 臂，臂号段 2107000~2107027
#   · 2026-11-20 17:20 追加锚点 **1610001@5**（SB3 汤供体）：**nano 只差基线 −2 阶**、
#     F1 +7 阶、T1 +4 阶、t2fp 3 ⇒ 是「nano 缺口最小」的供体 ⇒ cp 补 nano 成功率最高
#     （cp0.15 实测 nano +35 阶 ⇒ 足以补平 −2 阶并反超）⇒ 与其余 6 个「5 epoch 训练代」
#     供体（nano −22~−83 阶）构成**互补**的锚点区域
#   · 目的：全台账 dT1>0 的成员**全部**出自 5 epoch 的 BASE_A2 代 ⇒ PT1（2 epoch 轻抛）够不到那个尺度
#   · 代价：L5 臂 = 5 epoch ≈ P 臂（2 epoch）的 2.5 倍机时 ⇒ 28 臂 ≈ 3 并发下 ~3.5 h
L5ARMS=28
# 🔴🔴🔴 2026-11-20 17:25 **L5B 臂预算 = 42**（§282：L5 的 lr/cp **补网格**）
#   · 7 锚点 × {lr 0.75, 2.5} × {cp 0.05, 0.10, 0.15} = 42 臂，臂号段 2108000~2108041（**全新段 ⇒ 零映射风险**）
#   · 为什么必须有：L5 只测了网格四角 ⇒ **cp 0.10（≈ nano +23 阶 ≈ 14963@4 的 −22 阶缺口）
#     与 lr 0.75/2.5 两个尺度完全没测**；且本机 18:00 关机 ⇒ 周一前**无人能按 L5 结果临时加轴**
#   · 代价：42 臂 ≈ 5 epoch 训练 ⇒ 3 并发下约 7 h（吃掉周末尾部的空转）
L5BARMS=42

say(){ echo "[$(date "+%F %T")] $*"; }
run(){ if [ "$DRY" = "1" ]; then say "DRY-RUN: $*"; else eval "$@"; fi; }

# 🔴🔴🔴 2026-11-20 17:20 **加锁：禁止两个实例同时拉起训练轴**
#   【现场事故 17:16】手工重启训练轴与 `heartbeat.sh` 的 cron（每 2 min）**撞车**
#   ⇒ 两条路径**各自**执行本脚本 ⇒ 各自 `pkill` 后各自 `setsid nohup` 拉起
#   ⇒ **同时跑了 2 个 `night_train_axis.py`**（pid 2333228 / 2333283，pgid 各自独立）
#   ⇒ 两个实例**写同一批臂号目录**（`trials/trial_<sid>/weights/` + 台账）
#   ⇒ 这正是脚本注释里写的、也是 memory §281 §6 红线里的「台账静默污染」入口。
#   修法：整个脚本体用 `flock -n` 串行化 ⇒ 并发调用者**立刻退出**（不 pkill、不拉起）。
#   ⚠ 锁必须覆盖「pkill → 拉起 → 自检」全过程 ⇒ 用文件描述符 9 持锁（脚本退出才释放）。
LOCK="$W/.train_axis_launch.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  say "🔴 已有另一个训练轴拉起流程在跑（$LOCK 被占）⇒ 本次**不重复拉起**（防双实例）"
  exit 0
fi
say "🔒 已取得拉起锁 $LOCK"

say "=== 周末训练轴启动（DRY=$DRY）==="
say "   截止=$DL｜并发=$ARMS｜GPU=$GPUS｜抛光臂预算=$POLISH"

# ── ① 干净重启：先停 runner，再停训练子进程（否则两个实例写同一臂号目录）──
say "① 停旧训练轴 runner"
run "pkill -f '[n]ight_train_axis.py' 2>/dev/null"
sleep 4
say "② 停训练子进程（optuna_mix_ratio_search --worker）"
run "pkill -f '[o]ptuna_mix_ratio_search' 2>/dev/null"
sleep 3
left=$(ps -eo args | grep -c "[o]ptuna_mix_ratio_search" || true)
say "   残留训练子进程 = $left（应为 0）"

# ── ③ 拉起 ──
say "③ 拉起训练轴"
if [ "$DRY" = "1" ]; then
  say "DRY-RUN: cd $REPO && setsid nohup $PY -u manu/pipeline/npu/night_train_axis.py --arms $ARMS --gpus $GPUS --stagger 60 --deadline '$DL' --polish-arms $POLISH --pt1-arms 30 --l5-arms $L5ARMS --l5b-arms $L5BARMS >> $W/night_train_axis.log 2>&1 < /dev/null &"
else
  cd "$REPO" || exit 1
  export PYTHONPATH=$REPO
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
  setsid nohup $PY -u manu/pipeline/npu/night_train_axis.py \
    --arms $ARMS --gpus "$GPUS" --stagger 60 --deadline "$DL" --polish-arms $POLISH \
    --pt1-arms 30 --l5-arms $L5ARMS --l5b-arms $L5BARMS \
    >> "$W/night_train_axis.log" 2>&1 < /dev/null 9>&- &
  echo $! > "$W/.pid_train_axis"
  say "   pid=$(cat "$W/.pid_train_axis")"
fi

# ── ④ 自检 ──
sleep 25
say "④ 自检：训练轴进程"
ps -eo pid,etime,args | grep -E "night_train_axis|optuna_mix_ratio_search" | grep -v grep \
  | sed 's/--longquan-data[^ ]*//; s/--uav-data[^ ]*//; s/--weights [^ ]*/--weights .../; s/--baseline-cache [^ ]*/--baseline-cache .../' \
  | cut -c1-240 | sed 's/^/   /'
say "④ 自检：GPU 水位"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/   /'
say "④ 自检：MemAvailable"
free -m | awk '/^Mem:/{printf "   MemAvailable = %s MB\n", $7}'
say "④ 自检：臂号段（应为 2100000+ ⇒ 与基线族成员 1104407/1102001/1102004 不撞号）"
grep -c . "$W/arm_recipe_registry.json" 2>/dev/null | sed 's/^/   注册表条目数 = /'
say "=== 完成：训练轴已起（截止 $DL）｜升轨需**手工**跑 promote_candidate（见 memory §268 §18）==="
