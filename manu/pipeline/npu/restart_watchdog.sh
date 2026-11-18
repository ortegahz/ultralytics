#!/bin/bash
# 独立重启看门狗（不动训练轴）；启动后打印 md5 以便核验
set -u
W=/mnt/data/siping/runs/npu20ms/nano_search
pkill -f "[l]oad_watchdog.sh" 2>/dev/null && echo "停旧看门狗" || echo "无旧看门狗"
sleep 2
setsid nohup bash "$W/scripts/load_watchdog.sh" >> /tmp/load_watchdog.log 2>&1 < /dev/null &
sleep 2
echo "看门狗 md5: $(md5sum "$W/scripts/load_watchdog.sh" | cut -d' ' -f1)"
pgrep -af "[l]oad_watchdog.sh" | head -2
