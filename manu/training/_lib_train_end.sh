#!/bin/bash
# ==============================================================================
# _lib_train_end.sh —— 「训练结束 + GPU 空闲」的**唯一真值**判据（2026-11-03）
#
# 为什么单独抽一个库
# ------------------
# 铁律：**评估不得与训练同卡**。判定「训练真的停了」要同时看三件事：
#   ① 墙钟已过训练截止（runtime_config.json 的 stop_at）；
#   ② 调度器进程（<out-root>/dispatcher.pid）已退出；
#   ③ **4 张卡上没有训练进程**（nvidia-smi 的 compute-apps + ps 交叉验证）。
# 这三件事在 `gate_*.sh` 与 `run_nogmc_t3_autorun.sh` 里都要用
# ⇒ 抽成库，避免两处实现漂移（漂移 = 某天在训练同卡时开了门禁）。
#
# 用法
# ----
#   source manu/training/_lib_train_end.sh
#   wait_train_end "$OUT_ROOT" "$TRAIN_STOP" 180 || exit $?
#
# 返回码
# ------
#   0 = 可以评估（GPU 已释放、无训练进程）
#   3 = 超过硬上限仍未满足（**调用方必须中止**，绝不允许"带训练开评估"）
# ==============================================================================

# 训练进程的特征串（用 [.] 技巧 ⇒ grep 自身不会误匹配）
TRAIN_PROC_RE="optuna_mix_ratio_search[.]py|train[.]py|mixed_ft_train[.]py|finetune_median_4gpu[.]py|finetune_pixelshuffle[.]py|finetune_gaussian_relaxation[.]py|finetune_soft_iou[.]py"

# ---- 正在占 GPU 的进程 PID（空 = 没有）-------------------------------------
gpu_busy_pids(){
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  # ⚠️ **不能写 `tr -d '[:space:]'`**（会连**换行**一起删 ⇒ 多个 PID 被粘成
  #    一个 "908557908558908559908560" ⇒ 循环只看到一个**不存在**的 PID
  #    ⇒ `gpu_has_train` **永远返回空** ⇒ 「GPU 上有训练进程」这条安全检查形同虚设。
  #    只清 CR，靠 `grep -E '^[0-9]+$'` 逐行过滤。
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | tr -d '\r' | grep -E '^[0-9]+$' || true
}

# ---- 正在训练的进程 PID（空 = 没有）----------------------------------------
train_pids(){
  ps -eo pid,args 2>/dev/null \
    | grep -E "$TRAIN_PROC_RE" \
    | awk -v self="$$" '$1 != self {print $1}' || true
}

# ---- 占着 GPU 的进程里**有没有训练进程**（有则打印该 PID）------------------
gpu_has_train(){
  local pids pid cl
  pids="$(gpu_busy_pids)"
  for pid in $pids; do
    [ -r "/proc/$pid/cmdline" ] || continue
    cl="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    case "$cl" in
      *train*|*optuna_mix_ratio_search*|*mixed_ft*) echo "$pid"; return 0 ;;
    esac
  done
  return 1
}

# ---- 调度器是否还活着-------------------------------------------------------
dispatcher_alive(){
  local out_root="$1" dpid=""
  [ -f "$out_root/dispatcher.pid" ] || { echo 0; return 0; }
  dpid="$(tr -d '[:space:]' < "$out_root/dispatcher.pid" 2>/dev/null || true)"
  if [ -n "$dpid" ] && kill -0 "$dpid" 2>/dev/null; then echo 1; else echo 0; fi
}

# ---- 阻塞等待「训练结束 + GPU 空闲」----------------------------------------
#   $1 = out_root（study 产物根，找 dispatcher.pid）
#   $2 = train_stop（墙钟截止，格式同 date -d）
#   $3 = hard_cap_min（硬上限分钟数；超过 ⇒ 返回 3）
wait_train_end(){
  local out_root="$1" train_stop="$2" hard_cap_min="${3:-180}"
  local stop_s now_s after_stop disp_alive tp gp waited=0
  stop_s=$(date -d "$train_stop" '+%s' 2>/dev/null || echo 0)
  while :; do
    now_s=$(date '+%s')
    after_stop=1
    [ "$stop_s" -gt 0 ] && [ "$now_s" -lt "$stop_s" ] && after_stop=0
    disp_alive="$(dispatcher_alive "$out_root")"
    tp="$(train_pids)"
    gp="$(gpu_has_train)"
    if [ "$after_stop" = "1" ] && [ "$disp_alive" = "0" ] && [ -z "$tp" ] && [ -z "$gp" ]; then
      echo "[WAIT] ✅ 训练已结束、GPU 已释放（已过截止=$after_stop 调度器存活=$disp_alive）"
      return 0
    fi
    waited=$((waited + 1))
    if [ "$waited" -ge "$((hard_cap_min * 2))" ]; then
      echo "[WAIT] ❌ 超过硬上限 ${hard_cap_min} 分钟仍未满足："
      echo "       已过截止=$after_stop 调度器存活=$disp_alive"
      echo "       训练进程=${tp:-无} 占卡训练进程=${gp:-无}"
      echo "       ⇒ 返回 3：**不得在训练同卡时评估**"
      return 3
    fi
    echo "[WAIT] 等待中：已过截止=$after_stop 调度器存活=$disp_alive 训练进程=${tp:-无} 占卡训练进程=${gp:-无}"
    sleep 30
  done
}
