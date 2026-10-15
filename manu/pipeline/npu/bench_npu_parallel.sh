#!/bin/sh
# =============================================================================
# RK3588 NPU 并行度扫描 —— 回答「能不能开 6 个 / 9 个进程」
#
# 方法
#   每个 worker 跑同样帧数 F，各自写自己的 TSV，墙钟取最慢的那个。
#   吞吐 = P*F / 墙钟。
#
# 两个防坑设计
#   1) 每档 P 用**不同的帧区间**（BASE + P*100 起）。否则 P=1 跑完的帧已经
#      在页缓存里，P=2/3... 读的就是内存，会把 NFS 瓶颈掩盖掉、虚高。
#   2) 同时统计 rc!=0 的 worker 数，以及该档期间 dmesg 新增的
#      "job timeout" / "Unable to handle kernel" 行数。
#      —— 之前串行跑就踩到过 RKNPU 6s job timeout -> soft reset -> rknn_run failed，
#         光看速度不看稳定性会得出错误结论。
#
# 计时用 /proc/uptime（busybox date 不支持 %N）。
# =============================================================================
set -u

NPU=/root/rknn_heatmap_smoke
MODEL=/mnt/manu/npu/data/trial0474_hm_nop0_int8.rknn
TH=0.08
TOPK=80
FEAT=/mnt/manu/pcfeat_bmp/images
SEQ=${SEQ:-5_1}
F=${F:-60}
BASE=${BASE:-300}
OUT=/mnt/manu/pbench
P_LIST=${P_LIST:-"1 2 3 4 6 9"}

mkdir -p "$OUT"
[ -f "$MODEL" ] || { echo "[FATAL] model missing"; exit 2; }
[ -x "$NPU" ]    || { echo "[FATAL] $NPU not exec";  exit 2; }

now_ms() { awk '{printf "%d", $1*1000}' /proc/uptime; }

base_fps=0   # P=1 的吞吐，用来算 speedup

printf 'P\tframes\twall_ms\ttput_fps\tspeedup\tbad_rc\tdmesg_to\tdmesg_crash\n'

for P in $P_LIST; do
    # ---- 本档专用帧区间（避开上一档预热的页缓存）
    #      id 从 3 起；P=1 起点 BASE，之后每档往后挪 P*100，保证互不重叠且都是冷的
    start=$((BASE + P * 100))
    L=$OUT/frames_p${P}.txt
    : > "$L"
    n=0
    while [ "$n" -lt "$F" ]; do
        echo "$FEAT/${SEQ}__$(printf '%06d' $((start + n))).bmp" >> "$L"
        n=$((n + 1))
    done

    # ---- 前置校验：清单里每个特征都必须真实存在。
    # 不校验的话，越界的 BASE 会让 NPU 立刻 rc!=0 退出，
    # 脚本照样打出一行"120 帧 / 250ms / 480fps"的假数据（已踩过一次）。
    miss=0
    while read -r fp; do
        [ -f "$fp" ] || { echo "[FATAL] 特征不存在: $fp"; miss=$((miss + 1)); }
    done < "$L"
    [ "$miss" -eq 0 ] || { echo "[FATAL] P=$P 缺 $miss 个特征，检查 BASE/SEQ/帧数"; exit 3; }

    d0=$(dmesg 2>/dev/null | wc -l)

    t0=$(now_ms)
    w=0
    while [ "$w" -lt "$P" ]; do
        (
            "$NPU" "$MODEL" --batch "$L" --th "$TH" --topk "$TOPK" \
                --detections-out "$OUT/t_p${P}_w${w}.tsv" \
                > "$OUT/log_p${P}_w${w}.txt" 2>&1
            echo $? > "$OUT/rc_p${P}_w${w}"
        ) &
        w=$((w + 1))
    done
    wait
    t1=$(now_ms)

    bad=0
    w=0
    while [ "$w" -lt "$P" ]; do
        rc=$(cat "$OUT/rc_p${P}_w${w}" 2>/dev/null || echo 99)
        [ "$rc" = "0" ] || bad=$((bad + 1))
        w=$((w + 1))
    done

    tail_dmesg=$(dmesg 2>/dev/null | tail -n +$((d0 + 1)))
    to=$(printf '%s' "$tail_dmesg" | grep -c "job timeout")
    cr=$(printf '%s' "$tail_dmesg" | grep -c "Unable to handle kernel")

    wall=$((t1 - t0))
    total=$((P * F))
    tput=$((total * 100000 / wall))          # fps x100
    fps_int=$((tput / 100)); fps_frac=$((tput % 100))

    if [ "$base_fps" -eq 0 ]; then
        base_fps=$tput
        sp="1.00"
    else
        sp=$((tput * 100 / base_fps))        # speedup x100
        sp="$((sp / 100)).$((sp % 100))"
    fi

    printf '%s\t%s\t%s\t%s.%02d\t%s\t%s\t%s\t%s\n' \
        "$P" "$total" "$wall" "$fps_int" "$fps_frac" "$sp" "$bad" "$to" "$cr"
    printf '   [P=%s] frames=%s start_id=%s wall=%sms fps=%s.%02d bad_rc=%s dmesg_timeout=%s crash=%s\n' \
        "$P" "$total" "$start" "$wall" "$fps_int" "$fps_frac" "$bad" "$to" "$cr" \
        >> "$OUT/bench_detail.log"
done

echo "BENCH_DONE"
