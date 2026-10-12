#!/usr/bin/env bash
# npy_diff_summary.sh -- aggregate per-frame |a-b| over a dump directory pair.
#
# The first version of this used awk string comparison for the max, which is
# LEXICOGRAPHIC: "51" < "6", so any sequence containing a two-digit difference
# silently reported "6" as the worst frame. Every max in the earlier tables was
# therefore wrong whenever it reached double digits. Fixed here with an explicit
# +0 so the comparison is numeric, and the max is also tracked WITH the frame it
# came from -- a bare number cannot be re-checked against the dump, and this one
# could not be.
set -u
A=$1   # reference dump dir
B=$2   # candidate dump dir
LABEL=${3:-comparison}

tot=0
for r in $(ls "$A"/*_feat.npy); do
  b="$B/$(basename "$r")"
  [ -f "$b" ] || continue
  /tmp/npy_absdiff "$r" "$b"
  tot=$((tot+1))
done | awk -v label="$LABEL" '
{ c=$1
  mx[c]=substr($2,5)+0; mn[c]=substr($3,6)+0; ex[c]=substr($5,7)+0
  k[c]++
  if (mx[c] > M[c]) { M[c]=mx[c]; F[c]=NR }
  S[c]+=mn[c]; E[c]+=ex[c] }
END{ printf "%s  (%d frames)\n", label, k["Ch0"]
     printf "  %-4s %-6s %-12s %-10s %s\n", "chan", "max", "mean|diff|", "exact%", "worst-frame-line"
     printf "  %-4s %-6d %-12.5f %-10.4f %d\n", "Ch0", M["Ch0"], S["Ch0"]/k["Ch0"], E["Ch0"]/k["Ch0"], F["Ch0"]
     printf "  %-4s %-6d %-12.5f %-10.4f %d\n", "Ch1", M["Ch1"], S["Ch1"]/k["Ch1"], E["Ch1"]/k["Ch1"], F["Ch1"]
     printf "  %-4s %-6d %-12.5f %-10.4f %d\n", "Ch2", M["Ch2"], S["Ch2"]/k["Ch2"], E["Ch2"]/k["Ch2"], F["Ch2"] }'
