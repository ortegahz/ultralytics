#!/bin/sh
# ===========================================================================
# hm_val_board_loop.sh -- run on the RK3588 board, one sequence at a time.
#
#   raw BMP -> gmc_stream_ocl (CPU fit + Mali fused tail) -> feat/*.bmp
#           -> rknn_heatmap_smoke (RKNN int8, Trial 0474 w/o P0) -> npu_det.tsv
#
# SERIAL, and that is deliberate: each stage keeps its own honest timer, which is
# what makes the speed report attributable rather than merely fast.
#
# RESUMABLE. A sequence is skipped when its TSV already holds exactly the expected
# number of frames. Progress is appended to progress.tsv after every sequence, so a
# crash costs at most one sequence and the host can poll without guessing.
#
# Do NOT pipe the binaries: `gmc_stream_ocl ... | tail` makes `$?` the exit code of
# tail, which silently turns a [FATAL] into a success. Every rc here is captured
# into a variable first.
#
# Inputs (both already on the board via NFS):
#   /mnt/manu/npu/out/demo/expected.tsv   <seq>\t<expected_frame_count>
#   /mnt/manu/npu/out/demo/<seq>/raw/<seq>/*.bmp
#   /mnt/manu/npu/out/demo/<seq>/feat_named/list_board.txt
# ===========================================================================
set -u

ROOT=/mnt/manu/npu/out/demo
MODEL=data/trial0474_hm_nop0_int8.rknn
TH=0.25
STRIDE=2
KERNELS=/root/kernels
GMC=/root/gmc_stream_ocl
NPU=/root/rknn_heatmap_smoke

cd /mnt/manu/npu || exit 2
[ -f "$MODEL" ] || { echo "[FATAL] model missing: $MODEL"; exit 2; }

progress=$ROOT/progress.tsv
: > "$progress"

echo "=== val board loop start $(date '+%F %T') ==="
echo "model: $MODEL  th=$TH stride=$STRIDE"

total_seq=0
done_seq=0
fail_seq=0

while IFS="$(printf '\t')" read -r seq want; do
    [ -n "${seq:-}" ] || continue
    total_seq=$((total_seq + 1))
    D=$ROOT/$seq
    # hm_val_prepare_links.py writes the board batch list at <seq>/feat_named/list_board.txt
    # (same path run_hm_demo.sh uses) -- NOT feat_named/<seq>/...
    batch_list=$D/feat_named/list_board.txt

    # ---- checkpoint 1: an exactly-sized TSV means this sequence is already done
    if [ -f "$D/npu_det.tsv" ]; then
        have=$(wc -l < "$D/npu_det.tsv")
        if [ "$have" -eq "$want" ]; then
            printf '%s\tSKIP\t%s\t%s\n' "$seq" "$have" "-" >> "$progress"
            echo "[$seq] SKIP (TSV 已是 $have 行)"
            done_seq=$((done_seq + 1))
            continue
        fi
    fi

    # ---- stage A: GMC (skipped when a previous run already produced every frame)
    nraw=$(ls "$D/raw/$seq" 2>/dev/null | wc -l)
    nfeat=$(ls "$D/feat" 2>/dev/null | wc -l)
    if [ "$nraw" -gt 0 ] && [ "$nfeat" -eq "$nraw" ]; then
        echo "[$seq] GMC SKIP (feat=$nfeat 已等于 raw=$nraw)"
    else
        echo "[$seq] GMC start (已有 feat=$nfeat, raw=$nraw, 期望 $want)"
        mkdir -p "$D/feat"
        ( cd "$D" && "$GMC" --raw-root raw --sequence "$seq" --out-dir feat \
            --out-ext bmp --fused gpu --kernel-dir "$KERNELS" --timing \
            --profile-csv gmc_profile.csv > gmc_run.log 2>&1 )
        gmc_rc=$?
        nfeat=$(ls "$D/feat" 2>/dev/null | wc -l)
        echo "[$seq] GMC rc=$gmc_rc feat=$nfeat"

        if [ "$gmc_rc" -ne 0 ]; then
            printf '%s\tGMC_FAIL\t%s\t%s\n' "$seq" "$nfeat" "$gmc_rc" >> "$progress"
            fail_seq=$((fail_seq + 1))
            continue
        fi
    fi

    # ---- stage B: NPU
    if [ ! -f "$batch_list" ]; then
        printf '%s\tNO_LIST\t%s\t%s\n' "$seq" "$nfeat" "$batch_list" >> "$progress"
        echo "[$seq] 缺 $batch_list，跳过 NPU"
        fail_seq=$((fail_seq + 1))
        continue
    fi
    echo "[$seq] NPU start ($(wc -l < "$batch_list") 帧)"
    "$NPU" "$MODEL" --batch "$batch_list" --th "$TH" \
        --stride "$STRIDE" --detections-out "$D/npu_det.tsv" \
        > "$D/npu_run.log" 2> "$D/npu_run.err"
    npu_rc=$?
    ntsv=$(wc -l < "$D/npu_det.tsv" 2>/dev/null || echo 0)
    echo "[$seq] NPU rc=$npu_rc tsv=$ntsv (期望 $want)"

    if [ "$npu_rc" -ne 0 ]; then
        printf '%s\tNPU_FAIL\t%s\t%s\n' "$seq" "$ntsv" "$npu_rc" >> "$progress"
        fail_seq=$((fail_seq + 1))
        continue
    fi
    if [ "$ntsv" -ne "$want" ]; then
        # a short TSV is a FAILED run, never a pass with fewer frames
        printf '%s\tSHORT_TSV\t%s\t%s\n' "$seq" "$ntsv" "$want" >> "$progress"
        echo "[$seq] ⚠ TSV 行数 $ntsv != 期望 $want"
        fail_seq=$((fail_seq + 1))
        continue
    fi

    printf '%s\tOK\t%s\t%s\n' "$seq" "$ntsv" "$nfeat" >> "$progress"
    done_seq=$((done_seq + 1))
done < "$ROOT/expected.tsv"

echo "=== val board loop end $(date '+%F %T') ==="
echo "total=$total_seq ok=$done_seq fail=$fail_seq"
echo "ALL_DONE_MARKER"