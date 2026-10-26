#!/usr/bin/env bash
# Reproduce the RK3588 f16 multi-scale mosaic experiment on the two Longquanshan val tracks.
#
#   track2_golden : the 7 golden-pool val sequences (T2)  -- 17,187 frames
#   track3_blind  : the blind hard-case hold-out (T3)      -- 11,810 frames
#
# Modes:
#   bash manu/pipeline/npu/run_mosaic_lq_val.sh board-t3      # board arm, T3      (~28 min)
#   bash manu/pipeline/npu/run_mosaic_lq_val.sh board-golden  # board arm, golden  (~41 min)
#   bash manu/pipeline/npu/run_mosaic_lq_val.sh pc            # PC-side mosaic control (f32)
#   bash manu/pipeline/npu/run_mosaic_lq_val.sh report        # assemble the comparison tables
#
# Environment (see manu/memory/MOSAIC_RK3588_20261026.md):
#   board  : ssh -p 22 root@192.168.0.64      (NFS /mnt/manu -> host /run/media/manu/SSD960-LOCAL/nfs)
#   server : ssh -p 32222 huangzhe@192.168.99.40  (/tmp/pycharm_project_10ae9e2e, conda env `uav`)
#   mirror : /home/manu/mnt/pycharm_project_10ae9e2e  == the server tree over SSHFS
set -euo pipefail

REPO=/media/manu/1TB-Volume/workspace/ultralytics
MIRROR=/home/manu/mnt/pycharm_project_10ae9e2e
NPU="$REPO/manu/pipeline/npu"

BOARD_MODEL=/mnt/manu/ptq/models/p0_off_A_none.rknn          # md5 1a3faf2f9459cad8a389ffa81810234c
HOST_OUT=/run/media/manu/SSD960-LOCAL/nfs/mosaic_out         # the board's NFS tree, seen from the workstation
SERVER=/tmp/pycharm_project_10ae9e2e

GOLDEN="VIDEO00006_19700101_000703 VIDEO00007_19700101_000818 VIDEO00008_19700101_001307 \
        VIDEO00009_19700101_001531 VIDEO00029_19700101_014053 VIDEO00033_19700101_015000 \
        VIDEO00034_19700101_015052"
BLIND="VIDEO00007_19700101_001324"

mode="${1:-}"
case "$mode" in

build)
  (cd "$NPU" && ./build_rknn_mosaic_arm64.sh)
  echo "== md5 (local / SSHFS / server / board) =="
  md5sum "$NPU/out/rknn_mosaic_heatmap" "$MIRROR/manu/pipeline/npu/out/rknn_mosaic_heatmap"
  scp -q -P 22 "$NPU/out/rknn_mosaic_heatmap" root@192.168.0.64:/root/
  ssh -p 22 root@192.168.0.64 'md5sum /root/rknn_mosaic_heatmap; ps -eo comm | grep -c "^rknn_"'
  ;;

board-t3)
  S=$BLIND; OUT=/mnt/manu/mosaic_out/full_mosaic_v2
  ssh -p 22 root@192.168.0.64 "ps -eo comm | grep -c '^rknn_'"   # must be 0
  ssh -p 22 root@192.168.0.64 "
    mkdir -p $OUT
    ls /mnt/manu/ptq/bmp/$S/*.bmp | sort > /root/manifest_$S.txt
    /root/rknn_mosaic_heatmap --model $BOARD_MODEL --list /root/manifest_$S.txt --out $OUT \
      --canvas-mode mosaic --input-mode set --cpu-affinity 4-7 --timing $OUT/timing.txt \
      > $OUT/stdout.log 2> $OUT/stderr.log"
  ls -la "$HOST_OUT/full_mosaic_v2/"
  ;;

board-golden)
  OUT=/mnt/manu/mosaic_out/golden
  ssh -p 22 root@192.168.0.64 "ps -eo comm | grep -c '^rknn_'"   # must be 0
  ssh -p 22 root@192.168.0.64 "
    mkdir -p $OUT
    for S in $GOLDEN ; do
      ls /mnt/manu/ptq/bmp/\$S/*.bmp | sort > /root/manifest_\$S.txt
      echo \"=== \$S frames=\$(wc -l < /root/manifest_\$S.txt) \$(date +%T) ===\"
      mkdir -p $OUT/\$S
      /root/rknn_mosaic_heatmap --model $BOARD_MODEL --list /root/manifest_\$S.txt --out $OUT/\$S \
        --canvas-mode mosaic --input-mode set --cpu-affinity 4-7 --timing $OUT/\$S/timing.txt \
        > $OUT/\$S.stdout.log 2> $OUT/\$S.stderr.log
      echo \"    rc=\$? done \$(date +%T) tsv=\$(wc -l < $OUT/\$S/\${S}_mosaic.tsv)\"
    done
    echo ALL_GOLDEN_DONE"
  ;;

pc)
  # The PC-side mosaic control: PT float32, ONE forward, the SAME canvas, the SAME BMP pixels.
  # All sequences of both tracks share ONE out-root per interpolation, so 38 can take a single dir.
  for S in $BLIND $GOLDEN; do
    ssh -p 32222 huangzhe@192.168.99.40 "
      source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
      cd $SERVER
      python manu/pipeline/npu/ptq/37_mosaic_pc_control.py --seq $S --batch-size 32 \
        --thumb-interp area   --score --out-root /mnt/data/siping/ptq_work/mosaic_pc_area_all
      python manu/pipeline/npu/ptq/37_mosaic_pc_control.py --seq $S --batch-size 32 \
        --thumb-interp linear --score --out-root /mnt/data/siping/ptq_work/mosaic_pc_linear_all"
  done
  ;;

report)
  # pull the board TSVs + timings into the server tree, then assemble the two-track table
  mkdir -p "$MIRROR/manu/pipeline/npu/ptq/board_tsv" "$MIRROR/manu/pipeline/npu/ptq/board_timing_dir"
  for S in $BLIND; do
    cp -f "$HOST_OUT/full_mosaic_v2/${S}_mosaic.tsv" "$MIRROR/manu/pipeline/npu/ptq/board_tsv/${S}_mosaic.tsv"
    mkdir -p "$MIRROR/manu/pipeline/npu/ptq/board_timing_dir/$S"
    cp -f "$HOST_OUT/full_mosaic_v2/timing.txt" "$MIRROR/manu/pipeline/npu/ptq/board_timing_dir/$S/timing.txt"
  done
  for S in $GOLDEN; do
    cp -f "$HOST_OUT/golden/$S/${S}_mosaic.tsv" "$MIRROR/manu/pipeline/npu/ptq/board_tsv/${S}_mosaic.tsv"
    mkdir -p "$MIRROR/manu/pipeline/npu/ptq/board_timing_dir/$S"
    cp -f "$HOST_OUT/golden/$S/timing.txt" "$MIRROR/manu/pipeline/npu/ptq/board_timing_dir/$S/timing.txt"
  done
  md5sum "$MIRROR/manu/pipeline/npu/ptq/board_tsv/"*.tsv
  ssh -p 32222 huangzhe@192.168.99.40 "
    source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
    cd $SERVER
    python manu/pipeline/npu/ptq/38_mosaic_final_report.py \
      --group track3_blind=$BLIND \
      --group track2_golden=$(echo "$GOLDEN" | tr ' ' ',') \
      --board-tsv-dir manu/pipeline/npu/ptq/board_tsv \
      --pc-area-tsv-dir /mnt/data/siping/ptq_work/mosaic_pc_area_all \
      --pc-linear-tsv-dir /mnt/data/siping/ptq_work/mosaic_pc_linear_all \
      --board-timing-dir manu/pipeline/npu/ptq/board_timing_dir \
      --out /mnt/data/siping/ptq_work/final_report_lq"
  ;;

*)
  sed -n '2,16p' "$0"
  exit 1
  ;;
esac
