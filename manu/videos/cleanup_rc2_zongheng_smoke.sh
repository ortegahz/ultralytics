#!/bin/bash
# Remove the RC2 OSD smoke-test output directories (they were probes, not deliverables).
set -u
DATA_ROOT=/mnt/data/siping/datasets/manu
for d in "$DATA_ROOT/纵横AI数据采集_osd_rc2_smoke" "$DATA_ROOT/纵横AI数据采集_osd_rc2_smoke2"; do
  if [ -d "$d" ]; then
    echo "[CLEAN] $d ($(ls -1 "$d" 2>/dev/null | wc -l) entries)"
    rm -rf "$d"
  fi
done
echo "[CLEAN] remaining 纵横AI dirs:"
ls -1d "$DATA_ROOT"/纵横AI数据采集* 2>/dev/null
