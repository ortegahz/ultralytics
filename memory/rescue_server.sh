#!/bin/bash
# host 侧自愈扫描器 v3：每 ~10s 扫 192.168.99.0/24:32222；服务器一通 ⇒ 清场 + 安全重启 + 备份 + 装开机自启
LOG=/tmp/rescue_server.log
echo "[$(date '+%F %T')] 自愈扫描启动 v3 (pid $$)" >> $LOG
HB=0
while true; do
  IPS=$(python3 - << 'PY'
import socket, concurrent.futures
def probe(ip):
    s = socket.socket(); s.settimeout(0.6)
    try:
        s.connect((ip, 32222)); return ip
    except Exception:
        return None
    finally:
        s.close()
with concurrent.futures.ThreadPoolExecutor(max_workers=120) as ex:
    print(" ".join(r for r in ex.map(probe, [f"192.168.99.{j}" for j in range(1, 255)]) if r))
PY
)
  HB=$((HB+1))
  if [ $((HB % 18)) -eq 0 ]; then
    echo "[$(date '+%F %T')] 心跳: 仍在扫描 (候选: ${IPS:-无})" >> $LOG
  fi
  for IP in $IPS; do
    OUT=$(timeout 40 ssh -F /dev/null -o ConnectTimeout=12 -o BatchMode=yes -o StrictHostKeyChecking=no -p 32222 huangzhe@$IP \
      'pkill -9 -f "curated_[l]oop.sh"; pkill -9 -f "basin_soup_[t]argeted.py"; pkill -9 -f "multiscale_heatmap_[c]ache.py"; pkill -9 -f "cache_trial0474_[i]nferences.py"; pkill -9 -f "dual_constraint_[g]ate.py"; pkill -9 -f "adaptive_[r]unner.py"; echo RESCUED; cat /proc/loadavg; free -g | head -2' 2>/dev/null)
    if echo "$OUT" | grep -q RESCUED; then
      echo "[$(date '+%F %T')] 服务器恢复 @$IP ⇒ 清场: $(echo "$OUT" | tr '\n' ' ')" >> $LOG
      timeout 300 ssh -F /dev/null -o ConnectTimeout=20 -o BatchMode=yes -p 32222 huangzhe@$IP 'bash -s' < /tmp/safe_restart.sh >> $LOG 2>&1
      echo "[$(date '+%F %T')] safe_restart 执行完毕" >> $LOG
      if [ -x /tmp/post_recover.sh ]; then
        timeout 900 /tmp/post_recover.sh "$IP" >> $LOG 2>&1
        echo "[$(date '+%F %T')] post_recover 执行完毕" >> $LOG
      fi
      sleep 600
      break
    fi
  done
  sleep 10
done
