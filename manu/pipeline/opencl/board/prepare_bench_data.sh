#!/usr/bin/env bash
# Convert the chosen sequences to BMP locally, from the SSHFS corpus.
#
# LOCAL FIRST, then scp: the mount is a fragile SSHFS link that has already
# dropped twice this session, and decoding 2000 JPEGs directly on the board is
# impossible anyway (no usable codec there -- see convert_seq_to_bmp.py's header).
# Converting here also means the board never has to reproduce cvtColor, which is
# what keeps the two sides byte-comparable.
set -uo pipefail

SRC="/home/manu/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg"
OUT="/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/opencl/board/.bench_bmp"
FRAMES=400
PY=/home/manu/anaconda3/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Chosen by MEASURED Shi-Tomasi corner count at the pipeline's own half-res
# (pick_sequences.py), spanning the 9.03x texture spread of the 24-sequence corpus.
# A speed ratio is meaningless without the texture it was measured at, so the
# benchmark must cover the range rather than pick four names that sound typical.
SEQS="VIDEO00034_19700101_015052 \
      VIDEO00028_19700101_013631 \
      VIDEO00032_19700101_014453 \
      VIDEO00002_19700101_001415 \
      VIDEO00036_19700101_015740"

mkdir -p "$OUT"
for s in $SEQS; do
  if [ -d "$OUT/$s" ]; then
    echo "skip $s (already converted)"
    continue
  fi
  echo "=== converting $s ==="
  "$PY" "$HERE/convert_seq_to_bmp.py" --seq-dir "$SRC/$s" --out-dir "$OUT/$s" --frames "$FRAMES" \
      2>&1 | tail -2
done

echo
echo "=== result ==="
du -sh "$OUT"
for s in $SEQS; do printf "%-34s %s frames\n" "$s" "$(ls "$OUT/$s" 2>/dev/null | wc -l)"; done