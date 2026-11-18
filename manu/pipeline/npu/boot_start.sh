#!/bin/bash
# 服务器开机自启（幂等）：看门狗 + 守望链路 + **夜间训练轴**（2026-11-16 §195 起）
#   🟢 换轴说明：策展汤轴已被证伪（穷举 C(52,5)=0 组过线 / 加权 LP=-0.07σ / 汤≈成员均值 ⇒ F1 天花板=基线）
#      ⇒ 开机自启不再拉「3 条策展汤通道」，改拉「训练轴」（6 臂，4 卡）。
set -u
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
REPO=/tmp/pycharm_project_10ae9e2e
W=/mnt/data/siping/runs/npu20ms/nano_search
S=$W/scripts
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p "$W"
echo "[$(date '+%F %T')] boot_start 执行"
# /tmp 会被重启清空 ⇒ 先把软链补回来
if [ ! -e "$REPO" ]; then ln -s "$W/repo" "$REPO"; echo "已重建软链 $REPO"; fi
pgrep -f "[l]oad_watchdog.sh" >/dev/null || { setsid nohup bash "$S/load_watchdog.sh" >> /tmp/load_watchdog.log 2>&1 < /dev/null & }
cd "$REPO" 2>/dev/null || { echo "REPO_MISSING"; exit 1; }
DL=$(date -d "tomorrow 09:00" +"%Y-%m-%d %H:%M")   # 🔴 滚动 deadline：写死日期 ⇒ 重启后轴起不来
MRDATE=$(date -d "tomorrow 09:00" +"%Y-%m-%d")
pgrep -f "[a]uto_promote_watch.py" >/dev/null || { setsid nohup $PY -u manu/pipeline/npu/auto_promote_watch.py --watch --interval 600 --deadline "$DL" >> "$W/auto_promote_watch.log" 2>&1 < /dev/null & }
pgrep -f "[b]oard_t4_watch.py"   >/dev/null || { setsid nohup $PY -u manu/pipeline/npu/board_t4_watch.py --watch --interval 600 --deadline "$DL" >> "$W/board_t4_watch.log" 2>&1 < /dev/null & }
pgrep -f "[s]oup_sync_loop.sh"   >/dev/null || { setsid nohup bash "$S/soup_sync_loop.sh" >> "$W/soup_sync.log" 2>&1 < /dev/null & }
pgrep -f "[r]un_morning_report"  >/dev/null || { setsid nohup bash manu/pipeline/npu/run_morning_report2.sh "$MRDATE 09:10" >> "$W/morning_report_runner.log" 2>&1 < /dev/null & }
# ---- 训练轴（替代原 3 条策展通道）----
if [ -f "$S/start_train_axis.sh" ]; then
  if pgrep -f "[n]ight_train_axis.py" >/dev/null; then
    echo "训练轴已在跑 ⇒ boot_start 不重复拉起"
  else
    M=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
    if [ "${M:-0}" -ge 60 ]; then
      setsid nohup bash "$S/start_train_axis.sh" >> "$W/boot_start.log" 2>&1 < /dev/null &
      echo "MemAvailable=${M}GB ⇒ 起训练轴"
    else
      echo "MemAvailable=${M}GB < 60 ⇒ 暂不起训练轴（守铁律）"
    fi
  fi
fi
echo "[$(date '+%F %T')] boot_start 完成"
