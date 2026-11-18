#!/bin/sh
# 板卡侧兜底触发器：服务器一恢复就自动清场 + 拉起受限并发流水线（host 断电也能自愈）
KEY=/root/.ssh/id_ed25519_ptq_server
SRV=huangzhe@192.168.99.40
PORT=32222
LOG=/root/watch_server_recover.log
SSHOPT="-F /dev/null -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 -p $PORT"
while true; do
  if ssh $SSHOPT $SRV 'echo UP' 2>/dev/null | grep -q UP; then
    echo "[$(date)] 服务器可达 => 触发恢复" >> $LOG
    ssh $SSHOPT $SRV 'bash -s' < /root/server_recover_payload.sh >> $LOG 2>&1
    sleep 1800
  else
    sleep 30
  fi
done
