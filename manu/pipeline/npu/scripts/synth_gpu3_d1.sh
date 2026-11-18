#!/bin/bash
# GPU3 白天波 D1 第二段：三路汤 round2（补家族票数）+ 1100277 高权重**细档**扫描
#   · 三路汤 round2：bw10/bw15/bw20 各 5 配方 ⇒ 每族 ≥5 成员 ⇒ 关A(n≥4) 有票可攒
#   · 2 路高权重段：**按 1% 权重档**取 4~5 个细点（档内同族 ⇒ 同族票可攒）
set -u
REPO=/tmp/pycharm_project_10ae9e2e; PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search; END="2026-11-19 08:00"
cd $REPO || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
RECIPES="640001:5:0.30,1100296:1:0.60,1100277:2:0.10;640001:5:0.35,1100296:1:0.55,1100277:2:0.10;640001:5:0.40,1100296:1:0.50,1100277:2:0.10;640001:5:0.50,1100296:1:0.40,1100277:2:0.10;640001:5:0.30,1100296:1:0.55,1100277:2:0.15;640001:5:0.35,1100296:1:0.50,1100277:2:0.15;640001:5:0.40,1100296:1:0.45,1100277:2:0.15;640001:5:0.45,1100296:1:0.40,1100277:2:0.15;640001:5:0.50,1100296:1:0.35,1100277:2:0.15;640001:5:0.35,1100296:1:0.45,1100277:2:0.20;640001:5:0.45,1100296:1:0.35,1100277:2:0.20;640001:5:0.50,1100296:1:0.30,1100277:2:0.20"
HIW="0.7750,0.7775,0.7800,0.7825,0.7850,0.7950,0.7975,0.8000,0.8025,0.8050,0.8150,0.8175,0.8200,0.8225,0.8250,0.8400,0.8425,0.8450,0.8750,0.8775,0.8800,0.8825"
timeout 5400 $PY -u manu/pipeline/npu/night_soup3.py --gpu 3 --sid-base 1450000 --sid-span 400 \
  --max-gates 0 --mem-gate-gb 40 --deadline "$END" --recipes "$RECIPES" >> $W/soup3_round2.log 2>&1
timeout 9000 $PY -u manu/pipeline/npu/night_soup_batch.py --gpu 3 --parents 640001:5 --children-arms "1100277:2" \
  --weights "$HIW" --sid-base 1460000 --sid-span 200 --max-gates 0 --gate-family-quota 8 \
  --mem-gate-gb 40 --deadline "$END" >> $W/soup_new_1460000.log 2>&1
