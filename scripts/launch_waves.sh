#!/bin/bash
# 夜间接力 v5 启动器（2026-11-17 15:20）：4 条**单家族集中打击**通道（全部 parent=640001:5）
#   GPU0 冠军池×5档权重 ｜ GPU1 干净温和池×3档 ｜ GPU2 老干净盆地池×2档 ｜ GPU3 递归自喂波
#   子池互不重叠 ⇒ 绝不重复造同一个模型；每个通道都含 **w=0.55** ⇒ 命中都落在
#   原始家族 `bs640001:w55`（已有 1 票 1106580）⇒ 满足升轨双关（家族 n≥4 + 同权重 ≥2）。
W=/mnt/data/siping/runs/npu20ms/nano_search
cd "$W" || exit 1
LA="setsid nohup bash"
CHAMP_A="1106580:5,1106512:5,1106562:5,1112006:5,1100296:1"
CHAMP_B="98902:5,98913:5,98905:5,600004:5,300001:5,620002:5"
CHAMP_C="12679:5,69914:5,69902:5,69906:5,99002:5,99100:5,98903:5,99000:5,98901:5,99101:5"
$LA scripts/night_wave.sh 0 1120000 "$CHAMP_A" "0.55,0.6,0.65,0.7,0.75" 44 36 >> night_wave_launch.log 2>&1 < /dev/null &
sleep 30
$LA scripts/night_wave.sh 1 1130000 "$CHAMP_B" "0.55,0.6,0.65" 44 36 >> night_wave_launch.log 2>&1 < /dev/null &
sleep 30
$LA scripts/night_wave.sh 2 1140000 "$CHAMP_C" "0.55,0.6" 44 36 >> night_wave_launch.log 2>&1 < /dev/null &
sleep 30
$LA scripts/night_recursion.sh >> night_recursion.log 2>&1 < /dev/null &
sleep 5
echo "waves launched"
