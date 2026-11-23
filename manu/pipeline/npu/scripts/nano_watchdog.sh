#!/bin/bash
# 🛡 nano 夜间管线内存看门狗（2026-11-18 13:10 新建；2026-11-20 17:00 阈值放宽）
# 背景：今晚实测踩到**真事故** —— 5 个门禁通道 × 每通道 4 个 multiscale_heatmap_cache 构建进程
#       （每个 ~4.7GB RSS）⇒ 20×4.7 ≈ 94GB ⇒ MemAvailable 掉到 7.5%（<10% 铁律停机线）。
#       老 load_watchdog 只管 optuna 老进程，**完全不管当前门禁/合成管线** ⇒ 本脚本补上。
# 🔴🔴🔴 2026-11-20 17:00 **阈值放宽 + 保护训练轴**（§280）
#   实测事故：16:19~16:28 看门狗每 2 分钟判一次 MEM_CRIT(<12GB) ⇒ **反复自杀**，
#     原因不是我的管线过载，而是服务器上有**外部**作业
#     （`manu/video_quality/vq_track_link_driver.py --workers 32 --gpus 4`，
#      RSS 合计 ~140 GB ⇒ 大量 swap）把 MemAvailable 压到 5~10 GB。
#     ⇒ 门禁/缓存构建被反复杀 ⇒ 三条门禁通道全程空转（都在等 40 GB 闸门）。
#   改法：① 刹车线 <25 GB → **<12 GB**（杀最新门禁批次）、<12 GB → **<6 GB**（全停）；
#         ② 全停时**额外杀 night_soup3.py**（汤产线）⇒ 把内存让给训练轴；
#         ③ 训练轴（night_train_axis.py / optuna_mix_ratio_search / auto_epoch_gate_check）
#            **永不匹配本脚本的 kill 模式** ⇒ 训练轴不会被看门狗误杀。
#   保留铁律：门禁本身有 40 GB 内存闸门（gate_loop_d1*.sh）⇒ 门禁永远不会在低内存下开跑。
set -u
W=/mnt/data/siping/runs/npu20ms/nano_search
LOG=$W/nano_watchdog.log
kill_tree(){ # kill_tree <pid>：连同全部后代
  local p="$1" kids
  kids=$(ps -eo pid,ppid | awk -v pp="$p" '$2==pp{print $1}')
  for k in $kids; do kill_tree "$k"; done
  kill -9 "$p" 2>/dev/null
}
while true; do
  sleep 30
  M=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
  if [ "${M:-0}" -lt 6 ]; then
    for p in $(ps -eo pid,args | grep "[n]ight_soup_batch.py" | awk '{print $1}'); do kill_tree "$p"; done
    for p in $(ps -eo pid,args | grep "[n]ight_soup3.py" | awk '{print $1}'); do kill_tree "$p"; done
    for p in $(ps -eo pid,args | grep "[m]ultiscale_heatmap_cache.py" | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done
    echo "$(date '+%F %T') MEM_CRIT mem_avail=${M}GB(<6) => 全部停机（门禁+汤产线+缓存构建）" >> "$LOG"
    sleep 120; continue
  fi
  if [ "${M:-0}" -lt 12 ]; then
    NEW=$(ps -eo etimes,pid,args | grep "[n]ight_soup_batch.py" | grep -- "--only-gate" | sort -n | head -1 | awk '{print $2}')
    if [ -n "${NEW:-}" ]; then
      kill_tree "$NEW"
      echo "$(date '+%F %T') MEM_LOW mem_avail=${M}GB(<12) => 杀最新门禁批次 pid=$NEW" >> "$LOG"
    fi
    sleep 60; continue
  fi
  NC=$(ps -eo args | grep -c "[m]ultiscale_heatmap_cache.py")
  echo "$(date '+%F %T') 水位 mem_avail=${M}GB cache_builders=${NC} load=$(awk '{print $1}' /proc/loadavg)/$(nproc --all)" >> "$LOG"
done
