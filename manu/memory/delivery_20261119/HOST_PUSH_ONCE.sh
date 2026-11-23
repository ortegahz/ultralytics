#!/usr/bin/env bash
# 服务器侧一次性推送尝试（幂等，可在每次值守/投递时无脑调用）。
#
# 为什么不是常驻守望：训练/交付容器里**没有可用的守护机制**
#   - cron 未运行，且 /var/spool/cron、/etc/cron.d 不可写，也没有 crond 二进制；
#   - 后台进程（nohup / setsid / screen）都会在**当前 shell 会话结束时被回收**（实测两次）。
# ⇒ 常驻守望在这台机器上活不过一次 exec；改成"每次值守时调一次"即可达成同样的自动闭环。
#
# 退出码：0=推送成功并复核通过；3=host 不可达（正常，等下次）；1=推送失败（详见日志）
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${PUSH_LOG:-/mnt/data/siping/runs/trial0080_onnx/search/host_push_watch.log}"
HOST_ADDR="${HOST_ADDR:-192.168.0.115}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

if ! ping -c1 -W2 "$HOST_ADDR" >/dev/null 2>&1; then
  log "[WAIT] ${HOST_ADDR} 无响应（host 仍关机），本次跳过"
  exit 3
fi

log "[UP] ${HOST_ADDR} 可达，尝试推送"
if bash "$HERE/HOST_PUSH.sh" >> "$LOG" 2>&1; then
  log "[SUCCESS] 推送完成并通过主机侧复核"
  exit 0
fi
log "[FAIL] 推送未通过（详见日志上文）"
exit 1
