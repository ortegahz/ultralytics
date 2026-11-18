#!/bin/bash
# host 侧：服务器恢复后的资产回灌 + 状态核验（把服务器上的关键产物拉回 host，防 /tmp 清空）
IP=${1:-192.168.99.40}
H=/media/manu/1TB-Volume/workspace/ultralytics/server_backup
mkdir -p "$H"
SSH="ssh -F /dev/null -o ConnectTimeout=12 -o BatchMode=yes -o StrictHostKeyChecking=no -p 32222 huangzhe@$IP"
SCP="scp -F /dev/null -o ConnectTimeout=12 -o BatchMode=yes -o StrictHostKeyChecking=no -P 32222"

echo "=== [$(date '+%F %T')] 服务器 @$IP 状态核验 ==="
$SSH 'echo "--- uptime/load ---"; uptime; free -g | head -2; echo "--- gpu ---"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; echo "--- 进程计数 ---"; echo "channels=$(pgrep -f "[c]urated_loop.sh" | wc -l) watchdog=$(pgrep -f "[l]oad_watchdog.sh" | wc -l) promote=$(pgrep -f "[a]uto_promote_watch.py" | wc -l) t4=$(pgrep -f "[b]oard_t4_watch.py" | wc -l) soup=$(pgrep -f "[s]oup_sync_loop.sh" | wc -l) report=$(pgrep -f "[r]un_morning_report_v2.sh" | wc -l) refresh=$(pgrep -f "[r]efresh_sets.sh" | wc -l)"; echo "--- 重活进程（应<=6）---"; pgrep -af "[b]asin_soup_targeted.py|[m]ultiscale_heatmap_cache.py|[c]ache_trial0474_inferences.py|[d]ual_constraint_gate.py" | wc -l; echo "--- 关键文件 ---"; ls -la /tmp/pycharm_project_10ae9e2e/manu/memory/memory.md 2>&1 | tail -1; wc -l /tmp/pycharm_project_10ae9e2e/manu/memory/memory.md 2>/dev/null; ls -la /mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json 2>&1 | tail -1; ls /mnt/data/siping/runs/npu20ms/nano_search/curated/ 2>/dev/null | head -12; echo "--- systemd 自启 ---"; systemctl is-enabled nanopipe.service 2>&1 | tail -1; echo "--- 台账家族统计 ---"; /home/huangzhe/anaconda3/envs/uav/bin/python - << "PYEOF"
import json
try:
    d=json.load(open("/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json"))
    arms=d.get("arms",d)
    fam={}
    six=0
    for k,v in (arms.items() if isinstance(arms,dict) else []):
        try:
            m=v.get("metrics",v)
            ok=all([m.get("nano",0)>0,m.get("nano4",0)>0,m.get("f1",0)>0,m.get("t1",0)>0,m.get("t3",0)>0,m.get("t2_fp",9)<=5])
        except Exception:
            ok=False
        if ok:
            six+=1
            f=v.get("family","?"); fam[f]=fam.get(f,0)+1
    print("six_pass_arms:",six,"families:",fam)
    print("baseline_v2:",d.get("baseline_v2",{}).get("trial") if isinstance(d.get("baseline_v2"),dict) else d.get("baseline_v2"))
except Exception as e:
    print("ledger read fail:",e)
PYEOF' 2>&1 | tail -30

echo "=== 回灌资产到 host ==="
$SCP -r huangzhe@$IP:/tmp/pycharm_project_10ae9e2e/manu/pipeline/npu "$H/" 2>&1 | tail -2
$SCP -r huangzhe@$IP:/tmp/pycharm_project_10ae9e2e/manu/memory "$H/" 2>&1 | tail -2
$SCP -r huangzhe@$IP:/mnt/data/siping/runs/npu20ms/nano_search/scripts "$H/" 2>&1 | tail -2
$SCP -r huangzhe@$IP:/mnt/data/siping/runs/npu20ms/nano_search/curated "$H/" 2>&1 | tail -2
$SCP huangzhe@$IP:/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json "$H/" 2>&1 | tail -2
echo "=== 推送最新脚本到服务器 \$W/scripts ==="
$SCP /tmp/safe_restart.sh huangzhe@$IP:/mnt/data/siping/runs/npu20ms/nano_search/scripts/ 2>&1 | tail -1
$SCP /tmp/rescue_server.sh huangzhe@$IP:/mnt/data/siping/runs/npu20ms/nano_search/scripts/ 2>&1 | tail -1
echo "=== CLI 探测（供下一步决策）==="
$SSH 'cd /tmp/pycharm_project_10ae9e2e && /home/huangzhe/anaconda3/envs/uav/bin/python -u manu/pipeline/npu/basin_soup_targeted.py --help 2>&1 | head -40; echo "--- adaptive_runner ---"; /home/huangzhe/anaconda3/envs/uav/bin/python -u manu/pipeline/npu/adaptive_runner.py --help 2>&1 | head -40' 2>&1 | tail -60
echo "=== [$(date '+%F %T')] 回灌完成 ==="
