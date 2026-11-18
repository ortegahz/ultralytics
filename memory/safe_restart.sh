#!/bin/bash
# 服务器侧「安全重启」v3（跑在服务器上）
# 事故背景：8 通道策展 + adaptive → 每通道派生 7 进程 × torch 默认 80 线程 ⇒ 125G 内存打满 + writeback 风暴
#           ⇒ 整机僵死（ping/TCP 全无），只能物理断电。
# v3 变更：① 关键脚本落盘到 $W/scripts（防 /tmp 被清空）② 装 systemd 开机自愈（host 断电也不怕）
#           ③ 限并发 3 + OMP=4 + nice10 ④ load/mem 看门狗 ⑤ 自带验收报告
set -u
LOG=/tmp/safe_restart.log
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
REPO=/tmp/pycharm_project_10ae9e2e
W=/mnt/data/siping/runs/npu20ms/nano_search
S=$W/scripts
mkdir -p "$W" "$S" 2>/dev/null
exec >> $LOG 2>&1
echo "=== [$(date '+%F %T')] safe_restart v3 开始 ==="
cat /proc/loadavg; free -g | head -2

# ---- 互斥锁：防止 host 链 与 板卡链 同时触发 ⇒ 双份通道超内存 ----
LOCKD=$W/.safe_restart.lock
if mkdir -p "$LOCKD" 2>/dev/null; then
  echo $$ > "$LOCKD/pid"; date +%s > "$LOCKD/ts"
elif [ -f "$LOCKD/ts" ] && [ $(( $(date +%s) - $(cat "$LOCKD/ts") )) -gt 1800 ]; then
  rm -rf "$LOCKD" && mkdir -p "$LOCKD" && echo $$ > "$LOCKD/pid" && date +%s > "$LOCKD/ts"
  echo "陈旧锁已清理"
else
  echo "已有 safe_restart 在执行（锁 $LOCKD）⇒ 本次退出，避免双份通道"
  exit 0
fi
trap 'rm -rf "$LOCKD"' EXIT

kill_heavy() {
  pkill -9 -f "curated_[l]oop.sh"                2>/dev/null
  pkill -9 -f "basin_soup_[t]argeted.py"          2>/dev/null
  pkill -9 -f "multiscale_heatmap_[c]ache.py"    2>/dev/null
  pkill -9 -f "cache_trial0474_[i]nferences.py"   2>/dev/null
  pkill -9 -f "dual_constraint_[g]ate.py"         2>/dev/null
  pkill -9 -f "adaptive_[r]unner.py"             2>/dev/null
  pkill -9 -f "run_adaptive_[n]ight.sh"          2>/dev/null
  pkill -9 -f "curate[4]_append.py"              2>/dev/null
  pkill -9 -f "cos_[p]ool.py"                    2>/dev/null
}
kill_heavy
sleep 4
echo "--- 清场后 ---"; cat /proc/loadavg; free -g | head -2

# ---------- 0) 仓库健康检查 ----------
if [ ! -d "$REPO/manu/pipeline/npu" ]; then
  echo "!!! REPO_MISSING: $REPO 不存在（/tmp 可能被清空）"
  ls -d /tmp/pycharm_project_* 2>/dev/null
  if [ -d "$W/repo_backup/pipeline_npu" ]; then
    echo "--- 从 $W/repo_backup 恢复 pipeline/npu ---"
    mkdir -p "$REPO/manu/pipeline" && cp -a "$W/repo_backup/pipeline_npu" "$REPO/manu/pipeline/" && echo "恢复完成"
  fi
fi

# ---------- 1) 关键脚本落盘（防 /tmp 清空） ----------
for f in refresh_sets.sh soup_sync_loop.sh load_watchdog.sh curated_loop.sh; do
  [ -f "/tmp/$f" ] && cp -f "/tmp/$f" "$S/$f" 2>/dev/null
done
[ -f "$W/curated_loop.sh" ] && cp -f "$W/curated_loop.sh" "$S/curated_loop.sh" 2>/dev/null

# 看门狗（自包含版，落盘到 $S）
cat > "$S/load_watchdog.sh" << 'EOW'
#!/bin/bash
# 负载/内存看门狗：MemAvailable<8GB ⇒ 杀全部策展；load1>48 且在跑臂>3 ⇒ 杀 heatmap/cache 派生
while true; do
  sleep 60
  L=$(awk '{print $1}' /proc/loadavg)
  M=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
  if [ "${M:-0}" -lt 8 ]; then
    pkill -9 -f "curated_[l]oop.sh"; pkill -9 -f "basin_soup_[t]argeted.py"
    pkill -9 -f "multiscale_heatmap_[c]ache.py"; pkill -9 -f "cache_trial0474_[i]nferences.py"
    pkill -9 -f "dual_constraint_[g]ate.py"
    echo "$(date '+%F %T') MEM_CRIT ${M}GB -> 杀全部策展通道" >> /tmp/load_watchdog.log
    sleep 900; continue
  fi
  if awk "BEGIN{exit !($L > 48)}"; then
    N=$(pgrep -f "[b]asin_soup_targeted.py" | wc -l)
    if [ "$N" -gt 3 ]; then
      pkill -9 -f "multiscale_heatmap_[c]ache.py"; pkill -9 -f "cache_trial0474_[i]nferences.py"
      echo "$(date '+%F %T') LOAD ${L} MEM ${M}GB -> 杀掉 heatmap/cache 派生" >> /tmp/load_watchdog.log
    fi
  fi
done
EOW
chmod +x "$S/load_watchdog.sh"

# ---------- 2) 开机自愈（systemd + rc.local 双保险） ----------
cat > "$S/boot_start.sh" << 'EOB'
#!/bin/bash
# 服务器开机自启：看门狗 + 守望链路 + 3 条策展通道（幂等）
set -u
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
REPO=/tmp/pycharm_project_10ae9e2e
W=/mnt/data/siping/runs/npu20ms/nano_search
S=$W/scripts
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p "$W"
echo "[$(date '+%F %T')] boot_start 执行"
pgrep -f "[l]oad_watchdog.sh" >/dev/null || { setsid nohup bash "$S/load_watchdog.sh" >> /tmp/load_watchdog.log 2>&1 < /dev/null & }
cd "$REPO" 2>/dev/null || { echo "REPO_MISSING"; exit 1; }
pgrep -f "[a]uto_promote_watch.py" >/dev/null || { setsid nohup $PY -u manu/pipeline/npu/auto_promote_watch.py --watch --interval 600 --deadline "2026-11-17 09:00" >> "$W/auto_promote_watch.log" 2>&1 < /dev/null & }
pgrep -f "[b]oard_t4_watch.py"   >/dev/null || { setsid nohup $PY -u manu/pipeline/npu/board_t4_watch.py --watch --interval 600 --deadline "2026-11-17 09:00" >> "$W/board_t4_watch.log" 2>&1 < /dev/null & }
pgrep -f "[s]oup_sync_loop.sh"   >/dev/null || { setsid nohup bash "$S/soup_sync_loop.sh" >> "$W/soup_sync.log" 2>&1 < /dev/null & }
pgrep -f "[r]un_morning_report_v2.sh" >/dev/null || { setsid nohup bash manu/pipeline/npu/run_morning_report_v2.sh "2026-11-17 09:10" >> "$W/morning_report_runner.log" 2>&1 < /dev/null & }
pgrep -f "[r]efresh_sets.sh"    >/dev/null || { setsid nohup bash "$S/refresh_sets.sh" >> /tmp/refresh_sets.log 2>&1 < /dev/null & }
cd "$W"
L="$S/curated_loop.sh"; [ -f "$L" ] || L="$W/curated_loop.sh"
if [ -f "$L" ]; then
  if [ "$(pgrep -f '[c]urated_loop.sh' | wc -l)" -ge 3 ]; then
    echo "已有通道在跑 ⇒ boot_start 不重复拉起"
  else
    setsid nohup nice -n 10 bash "$L" 0 860000 0 c0 >> "$W/curated_c0.log" 2>&1 < /dev/null &
    setsid nohup nice -n 10 bash "$L" 0 870000 1 c1 >> "$W/curated_c1.log" 2>&1 < /dev/null &
    setsid nohup nice -n 10 bash "$L" 1 880000 2 c2 >> "$W/curated_c2.log" 2>&1 < /dev/null &
  fi
fi
echo "[$(date '+%F %T')] boot_start 完成"
# 延迟 15 分钟起 1 个训练家族（仅 GPU2，且要求可用内存 >= 60G ⇒ 保守启动）
( sleep 900
  M=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
  if [ "${M:-0}" -ge 60 ] && ! pgrep -f "[r]un_adaptive_night.sh" >/dev/null; then
    cd "$REPO" || exit 0
    sed -i 's/--gpus 0,1,2,3/--gpus 2/' manu/pipeline/npu/run_adaptive_night.sh 2>/dev/null
    setsid nohup nice -n 15 bash manu/pipeline/npu/run_adaptive_night.sh "2026-11-17 09:00" 6 18 22 >> "$W/adaptive.log" 2>&1 < /dev/null &
    echo "[$(date '+%F %T')] MemAvailable=${M}GB => 起 GPU2 训练家族"
  else
    echo "[$(date '+%F %T')] MemAvailable=${M}GB 或已有训练家族 => 暂不起 adaptive"
  fi ) >> /tmp/boot_start.log 2>&1 &
EOB
chmod +x "$S/boot_start.sh"

cat > /etc/systemd/system/nanopipe.service << 'EOSV'
[Unit]
Description=Nano Search Pipeline (self-heal on boot)
After=network.target
[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash /mnt/data/siping/runs/npu20ms/nano_search/scripts/boot_start.sh
[Install]
WantedBy=multi-user.target
EOSV
systemctl daemon-reload >/dev/null 2>&1
SUDO=""
[ "$(id -u)" != "0" ] && { sudo -n true 2>/dev/null && SUDO="sudo -n"; }
$SUDO systemctl daemon-reload >/dev/null 2>&1
$SUDO systemctl enable nanopipe.service >/dev/null 2>&1 && echo "systemd 开机自启已装" || echo "systemd 不可用（无 root/sudo）"
RC=/etc/rc.local
if [ -f "$RC" ]; then
  grep -q "boot_start.sh" "$RC" || $SUDO sed -i "s|^exit 0|\"$S/boot_start.sh\" \&\nexit 0|" "$RC" 2>/dev/null
  $SUDO chmod +x "$RC" 2>/dev/null
  echo "rc.local 已写入"
else
  printf '#!/bin/sh\n"%s/boot_start.sh" &\nexit 0\n' "$S" > /tmp/rc.local.new 2>/dev/null && \
  { $SUDO cp /tmp/rc.local.new "$RC" && $SUDO chmod +x "$RC" && echo "rc.local 已创建"; } || echo "rc.local 不可写"
fi

# ---------- 3) 看门狗 ----------
pgrep -f "[l]oad_watchdog.sh" >/dev/null || { setsid nohup bash "$S/load_watchdog.sh" >> /tmp/load_watchdog.log 2>&1 < /dev/null & echo "拉起 load_watchdog"; }

# ---------- 4) 守望链路 ----------
cd "$REPO" 2>/dev/null
pgrep -f "[a]uto_promote_watch.py" >/dev/null || { setsid nohup $PY -u manu/pipeline/npu/auto_promote_watch.py --watch --interval 600 --deadline "2026-11-17 09:00" >> "$W/auto_promote_watch.log" 2>&1 < /dev/null & echo "拉起 auto_promote_watch"; }
pgrep -f "[b]oard_t4_watch.py"   >/dev/null || { setsid nohup $PY -u manu/pipeline/npu/board_t4_watch.py --watch --interval 600 --deadline "2026-11-17 09:00" >> "$W/board_t4_watch.log" 2>&1 < /dev/null & echo "拉起 board_t4_watch"; }
pgrep -f "[s]oup_sync_loop.sh"   >/dev/null || { setsid nohup bash "$S/soup_sync_loop.sh" >> "$W/soup_sync.log" 2>&1 < /dev/null & echo "拉起 soup_sync_loop"; }
pgrep -f "[r]un_morning_report_v2.sh" >/dev/null || { setsid nohup bash manu/pipeline/npu/run_morning_report_v2.sh "2026-11-17 09:10" >> "$W/morning_report_runner.log" 2>&1 < /dev/null & echo "拉起 morning_report(11-17 09:10)"; }
pgrep -f "[r]efresh_sets.sh"    >/dev/null || { setsid nohup bash "$S/refresh_sets.sh" >> /tmp/refresh_sets.log 2>&1 < /dev/null & echo "拉起 refresh_sets"; }

# ---------- 5) 策展通道（3 条，限线程 + 降权） ----------
cd "$W"
L="$S/curated_loop.sh"; [ -f "$L" ] || L="$W/curated_loop.sh"
if [ -f "$L" ]; then
  if [ "$(pgrep -f '[c]urated_loop.sh' | wc -l)" -ge 3 ]; then
    echo "已有 $(pgrep -f '[c]urated_loop.sh' | wc -l) 条通道在跑 ⇒ 不重复拉起"
  else
    setsid nohup nice -n 10 bash "$L" 0 860000 0 c0 >> "$W/curated_c0.log" 2>&1 < /dev/null &
    setsid nohup nice -n 10 bash "$L" 0 870000 1 c1 >> "$W/curated_c1.log" 2>&1 < /dev/null &
    setsid nohup nice -n 10 bash "$L" 1 880000 2 c2 >> "$W/curated_c2.log" 2>&1 < /dev/null &
    echo "拉起 3 条策展通道 (GPU0x2/GPU1x1, OMP=4, nice10)"
  fi
else
  echo "!! curated_loop.sh 缺失"
fi

# ---------- 6) 仓库备份（防 /tmp 清空） ----------
mkdir -p "$W/repo_backup" 2>/dev/null
cp -a "$REPO/manu/pipeline/npu" "$W/repo_backup/" 2>/dev/null && echo "pipeline/npu 已备份到 $W/repo_backup" || echo "仓库备份失败"

# ---------- 7) 验收报告 ----------
sleep 10
echo "=== 验收 [$(date '+%F %T')] ==="
echo "通道数: $(pgrep -f '[c]urated_loop.sh' | wc -l)  看门狗: $(pgrep -f '[l]oad_watchdog.sh' | wc -l)"
echo "守望: promote=$(pgrep -f '[a]uto_promote_watch.py' | wc -l) t4=$(pgrep -f '[b]oard_t4_watch.py' | wc -l) soup=$(pgrep -f '[s]oup_sync_loop.sh' | wc -l) report=$(pgrep -f '[r]un_morning_report_v2.sh' | wc -l) refresh=$(pgrep -f '[r]efresh_sets.sh' | wc -l)"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
cat /proc/loadavg; free -g | head -2
echo "台账: $(ls -la $W/nano_search_ledger.json 2>/dev/null)"
echo "=== safe_restart v3 结束 ==="
