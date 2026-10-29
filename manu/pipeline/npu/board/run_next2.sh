#!/usr/bin/env bash
# Wait for the current (contaminated) e2e to finish, sweep the orphaned
# burners, then re-run the three arms in a CLEAN environment.
while ps -eo args | grep -q "[r]knn_mosaic_e2e_v6"; do sleep 10; done
sleep 5
# kill every burner child recorded by any earlier arm
for f in /root/psearch/.burn_pids_*; do
  [ -f "${f}" ] || continue
  while read -r p; do [ -n "${p}" ] && kill "${p}" 2>/dev/null; done < "${f}"
  : > "${f}"
done
# and anything still running
ps -eo pid,args | grep "[y]es" | grep -v sftp | grep -v ClearAll | awk '{print $1}' |
  while read p; do kill "$p" 2>/dev/null; done
sleep 3
echo "=== leftover burners (want 0): $(ps -eo args | grep '[y]es' | grep -v sftp | grep -vc ClearAll) ==="
exec /root/psearch/arms_v10.sh /root/rknn_mosaic_e2e_v6 CANON21_dec_a55 CANON19_dec_jit67 CANON20_fused_jit67
