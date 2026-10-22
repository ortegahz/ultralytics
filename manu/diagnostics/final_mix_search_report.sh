#!/usr/bin/env bash
# 混合比例搜索 · 最终收口报告（一条命令出全部结论）
#
# 执行机：服务器（需要读 runs/ 产物与 GPU 状态）
# 用法：  bash manu/diagnostics/final_mix_search_report.sh [输出文件]
# 默认输出： /tmp/final_mix_report_<时间戳>.txt
#
# 内容：
#   [0] 调度器存活 / 在跑 trial / 已结算数
#   [1] 严格过线候选 Top（U 重算，不信 CSV 列）
#   [2] 按混合比例分组（同口径对局）
#   [3] 全口径分组（含旧调度口径）
#   [4] 分档统计对账（bootstrap CI / 效应量 / 三轨分解）
#   [5] 对照组复现判决（EXACT / BAND / NOT_REPRO）
#   [6] 候选权重 md5（上线核验）
set -uo pipefail

OUT="${1:-/tmp/final_mix_report_$(date +%H%M%S).txt}"
REMOTE="cd /tmp/pycharm_project_10ae9e2e && source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav &&"

{
echo "################################################################"
echo "# 混合比例搜索 · 最终收口报告"
echo "# 生成时间(服务器): $(date '+%F %T')"
echo "################################################################"
echo
echo "[0] 调度器状态"
ssh -p 32222 huangzhe@192.168.99.40 "$REMOTE
  echo '  dispatcher: '\$(ps -eo pid,ppid,etime,args | grep 'optuna_mix_ratio_search.py --output-root' | grep -v grep | head -1)
  echo '  已结算: '\$(ls runs/optuna_mix_ratio/trials/*/trial_result.json 2>/dev/null | wc -l) 个 trial
  echo '  在跑: '\$(ps -eo args | grep -o 'trial-number [0-9]*' | sort -u | tr '\n' ' ')
  echo
  echo '  GPU:'; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/    /'
" 2>&1 | grep -v "WARNING\|post-quantum\|store now\|may need"

echo
echo "[1] 补采：把已结算但没进 optuna_summary.csv 的 trial 补回汇总表"
echo "    （调度器截止时刻的 [WAIT] 块只等不收割 ⇒ 最后一批可能丢行）"
ssh -p 32222 huangzhe@192.168.99.40 "$REMOTE python manu/training/backfill_summary_rows.py --apply" 2>&1 | grep -v "WARNING\|post-quantum\|store now\|may need"

echo
echo "[2] 严格过线候选 Top-20（U 一律重算）"
ssh -p 32222 huangzhe@192.168.99.40 "$REMOTE python manu/training/status_mix_ratio_search.py --top 20 --strict-only" 2>&1 | grep -v "WARNING\|post-quantum\|store now\|may need"

echo
echo "[3] 同口径分组对局（RC1-Ep3 口径）"
ssh -p 32222 huangzhe@192.168.99.40 "$REMOTE python manu/training/status_mix_ratio_search.py --all-calibers" 2>&1 | grep -v "WARNING\|post-quantum\|store now\|may need"

echo
echo "[4] 分档统计对账（bootstrap CI 4000 / RC1原值超参子集 / 效应量 / 三轨分解）"
ssh -p 32222 huangzhe@192.168.99.40 "$REMOTE python manu/training/analyze_mix_spec_results.py --boot 4000" 2>&1 | grep -v "WARNING\|post-quantum\|store now\|may need"

echo
echo "[5] 对照组复现判决"
ssh -p 32222 huangzhe@192.168.99.40 "$REMOTE python manu/training/check_control_repro.py --all" 2>&1 | grep -v "WARNING\|post-quantum\|store now\|may need"

echo
echo "[6] 上线候选权重 md5（与 /mnt/data 备份交叉核验）"
ssh -p 32222 huangzhe@192.168.99.40 "cd /tmp/pycharm_project_10ae9e2e && BK=/mnt/data/siping/backup/mix_ratio_search_20261022 && for t in 0028 0037 0063 0065 0070 0080 0082; do
  f=runs/optuna_mix_ratio/trials/trial_\$t/weights/epoch3.pt
  b=\$BK/weights/trial_\$t'_epoch3.pt'
  m1=\$( [ -f \"\$f\" ] && md5sum \"\$f\" | cut -d' ' -f1 || echo MISSING )
  m2=\$( [ -f \"\$b\" ] && md5sum \"\$b\" | cut -d' ' -f1 || echo MISSING )
  if [ \"\$m1\" = \"\$m2\" ]; then s='✅ 与备份一致'; elif [ \"\$m2\" = MISSING ]; then s='⚠️ 备份缺失'; else s='⚠️ 与备份不一致'; fi
  echo \"  trial_\$t  \$m1  \$s\"
done
echo '  RC1      '\$(md5sum runs/mixed_ft_route_b/RC1/epoch3.pt | cut -d' ' -f1)'  '\$( [ -f \$BK/weights/RC1_epoch3.pt ] && md5sum \$BK/weights/RC1_epoch3.pt | cut -d' ' -f1 || echo MISSING)" 2>&1 | grep -v "WARNING\|post-quantum\|store now\|may need"

echo
echo "################################################################"
echo "# 报告结束"
echo "################################################################"
} | tee "$OUT"

echo
echo "📄 报告已写入：$OUT"
