#!/bin/bash
# 夜间训练轴启动器（幂等 + 干净重启）+ 守望链路 deadline 滚动
#   🔴 为什么要「干净重启」：pkill 只杀 runner，**训练子进程会变孤儿**继续跑 ⇒
#      新实例再起同名 trial ⇒ 两个进程写同一个 trial 目录 ⇒ checkpoint 互相覆盖。
#   🔴 deadline 必须**滚动**：写死 2026-11-17 09:00 ⇒ 服务器一重启就过期 ⇒ 轴起不来。
set -u
W=/mnt/data/siping/runs/npu20ms/nano_search
R=$W/repo
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
# 截止 = 明天 09:00（跨夜跑满）
DL=$(date -d "tomorrow 09:00" +"%Y-%m-%d %H:%M")
MRDATE=$(date -d "tomorrow 09:00" +"%Y-%m-%d")
echo "本轮 deadline = $DL｜晨报 = $MRDATE 09:10"
pkill -f "[n]ight_train_axis.py" 2>/dev/null && echo "停 runner" || echo "无 runner"
pkill -f "[o]ptuna_mix_ratio_search" 2>/dev/null && echo "停训练子进程" || echo "无训练子进程"
sleep 4
cd "$R" || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH=$R
setsid nohup $PY -u manu/pipeline/npu/night_train_axis.py \
  --arms 5 --gpus 0,1,2,3 --stagger 60 --deadline "$DL" \
  >> "$W/night_train_axis.log" 2>&1 < /dev/null &
echo "拉起新实例 pid=$!"
# 看门狗（分级刹车覆盖训练轴）
pkill -f "[l]oad_watchdog.sh" 2>/dev/null && echo "停旧看门狗" || echo "无旧看门狗"
sleep 2
setsid nohup bash "$W/scripts/load_watchdog.sh" >> /tmp/load_watchdog.log 2>&1 < /dev/null &
echo "看门狗 pid=$!"
# 守望链路：deadline 滚动（旧的 2026-11-17 09:00 已过期 ⇒ 升轨/板卡会停摆）
pkill -f "[a]uto_promote_watch.py" 2>/dev/null && echo "停旧升轨守望" || echo "无旧升轨守望"
pkill -f "[b]oard_t4_watch.py" 2>/dev/null && echo "停旧板卡守望" || echo "无旧板卡守望"
sleep 2
setsid nohup $PY -u manu/pipeline/npu/auto_promote_watch.py --watch --interval 600 --deadline "$DL" >> "$W/auto_promote_watch.log" 2>&1 < /dev/null &
setsid nohup $PY -u manu/pipeline/npu/board_t4_watch.py --watch --interval 600 --deadline "$DL" >> "$W/board_t4_watch.log" 2>&1 < /dev/null &
echo "升轨/板卡守望 pid=$!/$!"
# 晨报（明早 09:10）
pkill -f "[r]un_morning_report" 2>/dev/null && echo "停旧晨报" || echo "无旧晨报"
sleep 1
setsid nohup bash manu/pipeline/npu/run_morning_report2.sh "$MRDATE 09:10" >> "$W/morning_report_runner.log" 2>&1 < /dev/null &
echo "晨报 pid=$!"
