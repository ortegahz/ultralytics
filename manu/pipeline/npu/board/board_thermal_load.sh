#!/bin/bash
# 板卡三域压测负载发生器（RK3588 / Orange Pi CM5）
#
#   board_thermal_load.sh start [duration_s] [npu_procs] [cpu_stress] [gpu]
#   board_thermal_load.sh stop
#   board_thermal_load.sh status
#
# CPU: stress --cpu <核数>（纯用户态计算，模拟系统侧其它业务）
# NPU: N 个 mosaic_async_client_example（**真实交付负载**：rknn 推理 + GMC + 拼帧）
# GPU: glmark2-drm（可选；默认关 —— 真实业务 GPU 负载只有 GMC fit，约 1 %）
#
# 所有进程 setsid 脱离 ssh 会话，PID 落在 /tmp/bts_load/ 下，便于 stop。
#
# ⚠️ 2026-11-13 新增（起因：mosaic 客户端启动时偶发触发 Rockchip RGA 驱动内核 Oops，
#    "rga_mm: This is not a buffer that has been imported, handle = -1" ⇒ NULL 解引用，
#    进程被内核打死或卡死 ⇒ NPU 实际没在跑，温控数据会失真）：
#      1) CLIENT_RESTART=1 时，守护循环内嵌**存活看门狗**：日志 [rx] 行数 60 s 不增长 ⇒ 判定卡死 ⇒ kill -9 重启；
#      2) start 结束前做**起飞自检**：NPU 必须真的在出帧（rx 增长 + rknpu Core0>0），
#         否则重启客户端重试，最多 3 次，仍失败则报错退出（避免跑出一场"假满载"）。
set -u

ACTION="${1:-start}"
DURATION="${2:-4200}"      # 压测时长（秒）
NPU_PROCS="${3:-1}"        # mosaic 客户端进程数
CPU_STRESS="${4:-1}"       # 1 = 起 stress
GPU_ON="${5:-0}"           # 1 = 起 glmark2
STRESS_CORES="${STRESS_CORES:-$(nproc)}"   # stress 占用的核数（判别实验用 4 等）
NFRAMES="${NFRAMES:-6000}"        # 单次客户端跑多少帧（跑完即退）
CLIENT_RESTART="${CLIENT_RESTART:-0}"  # 1 = 客户端跑完自动重启，直到压测截止（保持 NPU 连续负载）
WATCHDOG_INTERVAL="${WATCHDOG_INTERVAL:-15}"  # 看门狗采样间隔（秒）
WATCHDOG_STALL="${WATCHDOG_STALL:-4}"        # 连续多少次不增长就判定卡死（4×15 s = 60 s）
RESTART_GAP="${RESTART_GAP:-2}"              # 客户端重启间隔（秒）

BTS_DIR="${BTS_DIR:-/root/bts}"
RUN_DIR="${RUN_DIR:-$BTS_DIR/load}"
CLIENT="${CLIENT:-$BTS_DIR/mosaic_async_client_example}"
MODEL="${MODEL:-$BTS_DIR/p0_off_A_none.rknn}"
SEQ=VIDEO00007_19700101_001324
LOCAL_FRAMES="${LOCAL_FRAMES:-}"      # 默认空 ⇒ 直接读 NFS，不做本地拷贝
NFS_FRAMES="${NFS_FRAMES:-/mnt/manu/lq_ir_bmp/$SEQ}"
KERNEL_DIR="${KERNEL_DIR:-$BTS_DIR/kernels}"
FPS=25
INFLIGHT=4

mkdir -p "$RUN_DIR"

log() { echo "[load $(date +%H:%M:%S)] $*"; }

pick_frames_dir() {
    if [ -n "$LOCAL_FRAMES" ] && [ -d "$LOCAL_FRAMES" ] && [ "$(ls -1 "$LOCAL_FRAMES" 2>/dev/null | wc -l)" -gt 100 ]; then
        echo "$LOCAL_FRAMES"
    else
        echo "$NFS_FRAMES"
    fi
}

rx_count() { grep -c '\[rx\]' "$1" 2>/dev/null || echo 0; }

rknpu_core0() {
    sed -n 's/.*Core0: *\([0-9][0-9]*\)%.*/\1/p' /sys/kernel/debug/rknpu/load 2>/dev/null | head -1
}

do_stop() {
    local n=0
    for f in "$RUN_DIR"/*.pid; do
        [ -e "$f" ] || continue
        p="$(cat "$f")"
        if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
            kill -TERM "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null
            n=$((n + 1))
        fi
        rm -f "$f"
    done
    # 兜底：按命令行特征再清一遍。
    # ⚠️ 必须锚定行首：守护进程是 `sh -c "... /root/bts/mosaic_async_client_example ..."`，
    #    它的 cmdline 里含有客户端路径，若用 `pkill -f "[m]osaic_async_client_example"`
    #    会把**别的 RUN_DIR 的守护进程一起杀掉**（本次踩到：两套负载并发 + 客户端无守护）。
    #    `^/root/bts/mosaic_async_client_example` 只匹配真正以该路径开头的客户端进程。
    pkill -f "^${CLIENT}" 2>/dev/null
    pkill -f "[b]ts_load_stress" 2>/dev/null
    pkill -f "[g]lmark2" 2>/dev/null
    log "已停止 $n 个负载进程（并做了一遍特征兜底清理）"
}

start_client() {
    local idx="$1"
    local end_epoch="$2"
    if [ "$CLIENT_RESTART" = "1" ]; then
        setsid nohup sh -c "
            end=$end_epoch
            while [ \$(date +%s) -lt \"\$end\" ]; do
                '$CLIENT' --frames-dir '$FRAMES_DIR' --model '$MODEL' \\
                    --kernel-dir '$KERNEL_DIR' --out '$RUN_DIR/out_$idx' --seq '$SEQ' \\
                    --mode push-all --frames '$NFRAMES' --fps '$FPS' --inflight '$INFLIGHT' \\
                    >> '$RUN_DIR/mosaic_$idx.log' 2>&1 &
                cpid=\$!
                echo \$cpid > '$RUN_DIR/mosaic_$idx.clientpid'
                last=-1
                stall=0
                while :; do
                    st=\$(ps -o stat= -p \$cpid 2>/dev/null | tr -d ' ')
                    [ -z \"\$st\" ] && break
                    case \"\$st\" in Z*) break;; esac
                    sleep $WATCHDOG_INTERVAL
                    cur=\$(grep -c '\[rx\]' '$RUN_DIR/mosaic_$idx.log' 2>/dev/null || echo 0)
                    if [ \"\$cur\" = \"\$last\" ]; then
                        stall=\$((stall+1))
                        if [ \$stall -ge $WATCHDOG_STALL ]; then
                            echo \"[load \$(date +%H:%M:%S)] 客户端 pid=\$cpid 卡死（[rx] 停在 \$cur 不再增长）⇒ kill -9 并重启\" >> '$RUN_DIR/watchdog_$idx.log'
                            kill -9 \$cpid 2>/dev/null
                            break
                        fi
                    else
                        stall=0
                    fi
                    last=\$cur
                done
                wait \$cpid 2>/dev/null
                sleep $RESTART_GAP
            done
        " > /dev/null 2>&1 &
    else
        setsid nohup "$CLIENT" \
            --frames-dir "$FRAMES_DIR" \
            --model "$MODEL" \
            --kernel-dir "$KERNEL_DIR" \
            --out "$RUN_DIR/out_$idx" \
            --seq "$SEQ" \
            --mode push-all \
            --frames "$NFRAMES" \
            --fps "$FPS" \
            --inflight "$INFLIGHT" > "$RUN_DIR/mosaic_$idx.log" 2>&1 &
        echo $! > "$RUN_DIR/mosaic_$idx.clientpid"
    fi
    echo $! > "$RUN_DIR/mosaic_$idx.pid"
}

# 起飞自检：NPU 必须真的在出帧，否则这场压测没有意义
verify_npu_alive() {
    local logf="$RUN_DIR/mosaic_0.log"
    local tries=0 before after rk cpid
    while [ "$tries" -lt 3 ]; do
        sleep 12
        before=$(rx_count "$logf")
        sleep 12
        after=$(rx_count "$logf")
        rk=$(rknpu_core0)
        if [ "${after:-0}" -gt "${before:-0}" ] && [ "${rk:-0}" -gt 0 ]; then
            log "✅ 起飞自检通过：[rx] ${before} → ${after} 帧，rknpu Core0=${rk}%"
            return 0
        fi
        tries=$((tries + 1))
        log "⚠️ 起飞自检第 ${tries}/3 次失败：[rx] ${before} → ${after}，rknpu Core0=${rk:-?}（客户端疑似卡死 ⇒ 杀掉让守护重启）"
        cpid=$(cat "$RUN_DIR/mosaic_0.clientpid" 2>/dev/null)
        if [ -n "$cpid" ]; then
            kill -9 "$cpid" 2>/dev/null && log "   已 kill -9 客户端 pid=$cpid"
        fi
    done
    log "❌ 起飞自检 3 次均失败：NPU 未真正起跑，本轮压测无效"
    return 1
}

do_status() {
    echo "== 负载进程 =="
    for f in "$RUN_DIR"/*.pid; do
        [ -e "$f" ] || continue
        p="$(cat "$f")"
        if kill -0 "$p" 2>/dev/null; then s="运行中"; else s="已退出"; fi
        echo "  $(basename "$f" .pid): pid=$p $s"
    done
    echo "== 实时负载 =="
    for z in /sys/class/thermal/thermal_zone*; do
        printf "  %-18s %s C\n" "$(cat "$z"/type)" "$(( $(cat "$z"/temp) / 1000 ))"
    done
    printf "  GPU util   : %s %%\n" "$(cat /sys/devices/platform/fb000000.gpu/utilisation 2>/dev/null)"
    printf "  NPU load   : %s %%\n" "$(cat /sys/class/devfreq/fdab0000.npu/load 2>/dev/null)"
    echo "  rknpu/load : $(cat /sys/kernel/debug/rknpu/load 2>/dev/null | tr '\n' ' ')"
    echo "  [rx] 帧数  : $(rx_count "$RUN_DIR/mosaic_0.log")"
    echo "  CPU per-core util (2s 窗口):"
    head -9 /proc/stat | tail -8 | awk '{print $1}' | while read c; do
        a=$(grep "^$c " /proc/stat | awk '{print $2+$3+$4+$5+$6+$7+$8}')
        sleep 0.2
        b=$(grep "^$c " /proc/stat | awk '{print $2+$3+$4+$5+$6+$7+$8}')
        echo "    $c"
    done
}

case "$ACTION" in
  start)
    do_stop >/dev/null 2>&1
    FRAMES_DIR="$(pick_frames_dir)"
    log "帧目录: $FRAMES_DIR"
    log "参数: duration=${DURATION}s npu_procs=$NPU_PROCS cpu_stress=$CPU_STRESS gpu=$GPU_ON"

    if [ "$CPU_STRESS" = "1" ]; then
        setsid nohup stress --cpu "$STRESS_CORES" --timeout "$DURATION" > "$RUN_DIR/stress.log" 2>&1 &
        echo $! > "$RUN_DIR/stress.pid"
        log "CPU 压测: stress --cpu $STRESS_CORES (pid $(cat "$RUN_DIR/stress.pid"))"
    fi

    i=0
    END_EPOCH=$(( $(date +%s) + DURATION ))
    while [ "$i" -lt "$NPU_PROCS" ]; do
        start_client "$i" "$END_EPOCH"
        log "NPU 客户端 #$i: pid $(cat "$RUN_DIR/mosaic_$i.pid")"
        i=$((i + 1))
    done

    if [ "$GPU_ON" = "1" ]; then
        setsid nohup sh -c 'while :; do glmark2-drm --benchmark glmark2 >/dev/null 2>&1; done' > "$RUN_DIR/glmark.log" 2>&1 &
        echo $! > "$RUN_DIR/glmark.pid"
        log "GPU 压测: glmark2-drm 循环 (pid $(cat "$RUN_DIR/glmark.pid"))"
    fi

    sleep 3
    if [ "$NPU_PROCS" -ge 1 ]; then
        if ! verify_npu_alive; then
            do_stop
            exit 3
        fi
    fi
    do_status
    ;;
  stop)
    do_stop
    ;;
  status)
    do_status
    ;;
  *)
    echo "用法: $0 {start|stop|status} [duration] [npu_procs] [cpu_stress] [gpu]" >&2
    exit 2
    ;;
esac
