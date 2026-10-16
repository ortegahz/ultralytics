#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# Head 分支合并 (merge_head_branches) —— f16 / int8 双臂 NPU 速度 A/B runbook
#
# 背景：trial0474 stride=2 @640 下 head.heatmap.0 与 head.offset.0 两个 3x3 C=64
# 卷积各 3.775 G MAC（合计 7.55 G / 全网 21.48 G 的 35%），且都把同一份
# 320x320x64 特征读一遍。合并为单个 Conv2d(64,128,3) 数学等价、零重训，
# 省的是**一次 320x320x64 输入读取**，不是乘法（总 MAC 不变）。
#
# 本脚本只做「速度」判定，不做精度判定（ONNX 层等价已由
# manu/pipeline/npu/verify_merge_head_branches.py 证明逐位相同）。
#
# 内建的铁律约束：
#   * 铁律零·本地优先      —— 只在本地 checkout 构建/编辑，不直改服务器代码
#   * 铁律·板卡测量前置    —— 每次测量前必须先清残留进程（3 上下文会 IOMMU fault 掉线）
#   * §四·9e               —— 构建日志 fallback 警告必须 0 条、CPU 计算算子必须 0 个
#   * §四·9f/§四·10        —— 禁止 MAC 外推时延、禁止反解时延；只认 rknn_run 墙钟
#   * NFS 口径              —— 模型与可执行文件走板卡本地盘；经 /mnt/manu 的耗时不计入算子时延
#   * COLLECT_PERF_MASK    —— 一律不加 --perf（实测让 rknn_run 慢 54%，读数不可用）
#
# 用法：
#   bash manu/pipeline/npu/run_merge_head_speed_ab.sh onnx     # 阶段1 导出 ONNX（本地）
#   bash manu/pipeline/npu/run_merge_head_speed_ab.sh build    # 阶段2 在训练服务器构建 f16+int8
#   bash manu/pipeline/npu/run_merge_head_speed_ab.sh board    # 阶段3 上板测速（f16 + int8）
#   bash manu/pipeline/npu/run_merge_head_speed_ab.sh all

set -euo pipefail

REPO="/media/manu/1TB-Volume/workspace/ultralytics"
WORK="/media/manu/1TB-Volume/rknn_work/hm/merge_head"
ONNX_DIR="$WORK/onnx"
REMOTE="huangzhe@192.168.99.40"
PORT=32222
REMOTE_ROOT="/tmp/pycharm_project_10ae9e2e"
REMOTE_WORK="$REMOTE_ROOT/runs/merge_head"
BOARD_PY="$REPO/manu/pipeline/opencl/board/board_run.py"
CKPT="/home/manu/mnt/pycharm_project_10ae9e2e/runs/optuna_p0_nas/trial_0474/weights/best.pt"
CALIB_IMG="/home/manu/mnt/data/siping/datasets/manu/uav_gmc_median/images/val/01_4485_1167-2666__000100.jpg"
FEAT_JPG="/home/manu/mnt/data/siping/datasets/manu/uav_gmc_median/images/val/01_4485_1167-2666__000100.jpg"
UAV_PY="/home/manu/anaconda3/envs/uav/bin/python"
# rknn-toolkit2 2.3.0 —— 与板卡 librknnrt commit c949ad889d 同版本。
# ⚠️ conda-envs/rknn 是 1.4.0，案卷已定其为弃用版，禁止用于转换。
RKNN_PY="/media/manu/1TB-Volume/conda-envs/rknn230/bin/python"
BOARD_PY_PY="/usr/bin/python3"   # 需要 pexpect；uav 环境没有，系统 python3 有

# 板端工作目录一律本地盘（/root），避免 NFS 往返污染时延
B_LOCAL=/root/merge_head

stage_onnx() {
  echo "=== [1/3] 导出 ONNX（本地，用真实 trial_0474 权重）==="
  mkdir -p "$ONNX_DIR"
  for m in "" "--merge-head-branches"; do
    tag=$([ -z "$m" ] && echo base || echo merge)
    echo "--- $tag ---"
    # --skip-check: 本机 uav 环境 torch 1.13.1 配 numpy 2.2.6，ORT 对账无法跑（案卷已载明）
    "$UAV_PY" "$REPO/manu/pipeline/npu/hm_export_onnx.py" \
      --ablate-p0 $m --skip-check \
      --out "$ONNX_DIR/trial0474_hm_nop0_${tag}.onnx" \
      2>&1 | grep -E "GATE|matched|wrote|nodes"
  done
  echo
  echo "--- 核验合并确实生效（head Conv 数 / initializer 数）---"
  "$UAV_PY" - <<'PYEOF'
import onnx
for t in ("base", "merge"):
    g = onnx.load(f"/media/manu/1TB-Volume/rknn_work/hm/merge_head/onnx/trial0474_hm_nop0_{t}.onnx").graph
    hc = [n.name for n in g.node if n.op_type == "Conv" and "/head/" in n.name]
    has = [n.name for n in g.node if n.name in ("/model/head/branch3x3/Conv", "/model/head/Split")]
    print(f"  {t:6s} nodes={len(g.node)} init={len(g.initializer)} headConv={len(hc)} merge_nodes={has}")
PYEOF
}

stage_build() {
  echo "=== [2/3] 本地构建 RKNN（rknn-toolkit2 2.3.0；f16 + int8 × base + merge = 4 个）==="
  echo "--- 版本闸门：必须 2.3.0（1.4.0 是弃用版）---"
  "$RKNN_PY" -c "import importlib.metadata as m; v=m.version('rknn-toolkit2'); print('[toolkit]', v); assert v.startswith('2.3'), 'toolkit 必须是 2.3.x'"

  cd "$REPO"
  for t in base merge; do
    for q in int8 fp16; do
      EXTRA=""
      [ "$q" = "fp16" ] && EXTRA="--no-quant"
      LOG="$WORK/build_${t}_${q}.log"
      echo "--- build $t / $q -> $LOG ---"
      "$RKNN_PY" manu/pipeline/npu/hm_convert_rknn.py \
        --onnx "$ONNX_DIR/trial0474_hm_nop0_${t}.onnx" \
        --out  "$WORK/trial0474_hm_nop0_${t}_${q}.rknn" \
        --calib-list "$WORK/calib_train_features.txt" \
        $EXTRA > "$LOG" 2>&1 || { echo "[FATAL] $t/$q 构建失败，日志尾："; tail -20 "$LOG"; return 1; }
      grep -E "^\[OK\]|^\[calib\]|^\[quant\]" "$LOG" | head -4
    done
  done

  echo
  echo "=== §四·9e 上板闸门：fallback 警告 0 条 + CPU 计算算子 0 个 ==="
  for t in base merge; do for q in int8 fp16; do
    L="$WORK/build_${t}_${q}.log"
    fb=$(grep -ic 'fallback' "$L" || true)
    [ -s "$L" ] || { printf "  %-26s [FATAL] 日志缺失\n" "$(basename $L)"; return 1; }
    printf "  %-26s fallback=%s  %s\n" "$(basename $L)" "$fb" \
      "$([ "$fb" -eq 0 ] && echo PASS || echo 'FAIL <-- 不可上板')"
  done; done

  echo
  echo "--- 构建日志里的 NPU cycles / Conv MAC / 内部内存（合并前后应几乎相同）---"
  for t in base merge; do for q in int8 fp16; do
    printf "  %-6s %-5s " "$t" "$q"
    grep -oiE "NPU cycles[^|]*|Total Internal Memory Size[^,]*|Conv MAC[^,]*" "$WORK/build_${t}_${q}.log" | head -3 | tr '\n' ' '
    echo
  done; done
  ls -la "$WORK"/*.rknn
}

stage_board() {
  echo "=== [3/3] 上板测速（f16 + int8 × base + merge）==="
  [ -f "$WORK/trial0474_hm_nop0_base_int8.rknn" ] || { echo "[FATAL] 先跑 build 阶段"; exit 1; }

  echo "--- 交叉编译 arm64 smoke ---"
  bash "$REPO/manu/pipeline/npu/build_rknn_heatmap_arm64.sh" | tail -3

  echo "--- 投递到板卡本地盘 /root ---"
  # 目录必须先建：scp 在目标目录不存在时报 "dest open ...: No such file or directory"，
  # 不会被 key auth 或 BatchMode 救回来。
  "$BOARD_PY_PY" "$BOARD_PY" --exec "mkdir -p $B_LOCAL && ls -ld $B_LOCAL"
  "$BOARD_PY_PY" "$BOARD_PY" \
    --put "$REPO/manu/pipeline/npu/out/rknn_heatmap_smoke:$B_LOCAL/rknn_heatmap_smoke" \
    --put "$FEAT_JPG:$B_LOCAL/feat.jpg" \
    --put "$WORK/trial0474_hm_nop0_base_int8.rknn:$B_LOCAL/base_int8.rknn" \
    --put "$WORK/trial0474_hm_nop0_base_fp16.rknn:$B_LOCAL/base_fp16.rknn" \
    --put "$WORK/trial0474_hm_nop0_merge_int8.rknn:$B_LOCAL/merge_int8.rknn" \
    --put "$WORK/trial0474_hm_nop0_merge_fp16.rknn:$B_LOCAL/merge_fp16.rknn" \
    --exec "chmod 755 $B_LOCAL/rknn_heatmap_smoke && df -h / | tail -1"

  # 🔴 铁律：板卡测量前置必须先清残留进程。3 个 NPU 上下文会越过 P=2 上限 →
  #    failed to submit + IOMMU page fault + soft reset + 掉线，只能断电。
  KILL="for p in e2e_par2_demo e2e_par_demo e2e_demo rknn_heatmap_smoke; do pkill -9 -f \$p 2>/dev/null; done; sleep 1; ps -eo pid,etime,args | grep -E '[r]knn_|[e]2e_' || echo '[CLEAN] 无残留进程'"

  for m in base merge; do
    for q in int8 fp16; do
      echo
      echo "===== ${m} / ${q} ====="
      "$BOARD_PY_PY" "$BOARD_PY" --exec "$KILL"
      # 不加 --perf：RKNN_FLAG_COLLECT_PERF_MASK 实测让 rknn_run 慢 54%，读数不可用
      # --warmup 10 + --loops 30，与案卷热图检测器口径一致
      "$BOARD_PY_PY" "$BOARD_PY" --exec \
        "cd $B_LOCAL && ./rknn_heatmap_smoke ./${m}_${q}.rknn ./feat.jpg --th 0.25 --stride 2 --topk 80 --warmup 10 --loops 30 --tag ${m}_${q} 2>&1 | tail -30"
    done
  done

  echo
  echo "=== 汇总：以 base/int8 的 rknn_run 为 100% ==="
  "$UAV_PY" - <<'PYEOF'
import re, subprocess, os
print("  从上一步输出里抄 rknn_run 中位数；判定口径：")
print("  * 只认 rknn_run 墙钟（§四·10 禁止反解）")
print("  * 不看端到端（含 GMC/letterbox，与本次改动无关）")
print("  * MAC 不变是预期：合并省搬运不省乘法，若 rknn_run 持平则说明")
print("    RKNN 把 channel Split 实现成了等量拷贝 —— 这本身是有价值的负结果")
PYEOF
}

case "${1:-all}" in
  onnx)  stage_onnx ;;
  build) stage_build ;;
  board) stage_board ;;
  all)   stage_onnx; stage_build; stage_board ;;
  *)     echo "usage: $0 {onnx|build|board|all}"; exit 2 ;;
esac
