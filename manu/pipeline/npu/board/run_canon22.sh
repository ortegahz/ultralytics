#!/usr/bin/env bash
# CANON22: the same topology as CANON21 (the prescription arm) but with a
# DEEPER Buffer A.  The measurement that motivates it: CANON21 made 7195 GMC
# groups and dropped 1567 of them (21.8%) at Buffer A, while the NPU stage was
# only 73% busy -- the GMC's BURSTS overflow a 3-slot ring.
#
#   GMC 4-7 (4 anchor workers) | Pre+Post 0-3 (A55) | Stage 2 6-7
#   --buffer-a-depth 8   (8 x 2,457,600 B = 19.7 MB of DMA per NPU service)
set -uo pipefail
while ps -eo args | grep -q "[r]knn_mosaic_e2e_v6"; do sleep 10; done
sleep 5
for f in /root/psearch/.burn_pids_*; do
  [ -f "${f}" ] || continue
  while read -r p; do [ -n "${p}" ] && kill "${p}" 2>/dev/null; done < "${f}"
  : > "${f}"
done
ps -eo pid,args | grep "[y]es" | grep -v sftp | grep -v ClearAll | awk '{print $1}' |
  while read p; do kill "$p" 2>/dev/null; done
sleep 3
echo "=== leftover burners (want 0): $(ps -eo args | grep '[y]es' | grep -v sftp | grep -vc ClearAll) ==="

BASE_COMMON="--raw-root /mnt/manu/lq_ir_bmp --sequence VIDEO00007_19700101_001324
 --model /root/psearch/models/p0_off_A_none.rknn --kernel-dir /root/psearch/kernels
 --limit 0 --input-queue 64 --anchor-step 2 --gmc-phys-anchor 1 --gmc-workers 1
 --jitter-depth 3 --gmc-ring-slack 1 --cpu-affinity 0-7 --npu-workers 2
 --post-workers 1 --pace-ms 40 --gmc-min-stride 1"

exec /root/psearch/arms_v10.sh /root/rknn_mosaic_e2e_v6 CANON22_dec_bufa16
