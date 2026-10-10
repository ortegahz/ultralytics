#!/usr/bin/env bash
# ===========================================================================
# probe_opencv_on_board.sh -- does the RK3588 have a usable OpenCV C++ SDK?
#
# Runs on the board. The board has gcc/g++/make/cmake, so everything is
# compiled and executed natively there -- nothing is cross-compiled, which
# means the answer comes from the board's own toolchain and its own filesystem
# rather than from an inference about them.
#
# Two independent pieces of evidence are collected, because they can disagree
# and the disagreement is the point:
#   A. an ACTUAL compile+link attempt of a minimal OpenCV program
#   B. board_opencv_probe, a filesystem + dlopen survey that needs no headers
# If A fails and B agrees, the absence is real. If A somehow succeeds, B
# explains why.
#
# Usage: copy to the board and run:
#   bash probe_opencv_on_board.sh
# ===========================================================================
set -uo pipefail

echo "======================================================================"
echo " RK3588 OpenCV C++ SDK probe"
echo "======================================================================"

echo
echo "--- [1/5] board toolchain ---"
for t in gcc g++ cc make cmake pkg-config; do
  p=$(command -v "$t" 2>/dev/null) && echo "  $t      : $p" || echo "  $t      : (absent)"
done
(gcc --version 2>/dev/null | head -1) | sed 's/^/  gcc ver : /'
(g++ --version 2>/dev/null | head -1) | sed 's/^/  g++ ver : /'

echo
echo "--- [2/5] package manager ---"
if command -v dpkg >/dev/null 2>&1; then
  n=$(dpkg -l 2>/dev/null | grep -ci opencv || true)
  echo "  dpkg opencv packages : $n"
  dpkg -l 2>/dev/null | grep -i opencv | head -5 | sed 's/^/    /'
else
  echo "  dpkg absent"
fi
if command -v apt-cache >/dev/null 2>&1; then
  echo "  apt-cache policy libopencv-dev:"
  apt-cache policy libopencv-dev 2>/dev/null | head -4 | sed 's/^/    /'
fi
if command -v pkg-config >/dev/null 2>&1; then
  echo "  pkg-config opencv4   : $(pkg-config --modversion opencv4 2>&1 || echo 'not found')"
  echo "  pkg-config opencv     : $(pkg-config --modversion opencv 2>&1 || echo 'not found')"
fi

echo
echo "--- [3/5] python cv2 (NOT the same thing as a C++ SDK) ---"
python3 -c 'import cv2; print("  import cv2 :", cv2.__version__); print("  location  :", cv2.__file__)' 2>&1 | sed 's/^/  /'
python3 - <<'PY' 2>&1 | sed 's/^/  /'
try:
    import cv2, os, glob
    d = os.path.dirname(cv2.__file__)
    for so in sorted(glob.glob(os.path.join(d, "**", "*.so*"), recursive=True)):
        print("  wheel .so :", so)
except Exception as e:
    print("  cv2 probe failed:", e)
PY

echo
echo "--- [4/5] ACTUAL compile+link attempt of a minimal OpenCV program ---"
TMP=$(mktemp -d)
cat > "$TMP/t.cpp" <<'EOF'
#include <opencv2/core.hpp>
#include <cstdio>
int main() {
  cv::Mat m(4, 4, CV_8UC1, cv::Scalar(7));
  std::printf("  compiled+linked OK: cv::Mat %dx%d, build=%s\n",
              m.cols, m.rows, CV_VERSION);
  return 0;
}
EOF

echo "  source:"; sed 's/^/    /' "$TMP/t.cpp"

echo "  --- try 1: g++ -lopencv_core ---"
if g++ -std=c++17 -O2 "$TMP/t.cpp" -o "$TMP/t1" -lopencv_core 2>"$TMP/e1"; then
  echo "  RESULT: SUCCESS"; "$TMP/t1" | sed 's/^/  /'
else
  echo "  RESULT: FAILED"
  head -4 "$TMP/e1" | sed 's/^/    /'
fi

echo "  --- try 2: g++ -lopencv_core -I/usr/include/opencv4 ---"
if g++ -std=c++17 -O2 -I/usr/include/opencv4 "$TMP/t.cpp" -o "$TMP/t2" -lopencv_core 2>"$TMP/e2"; then
  echo "  RESULT: SUCCESS"; "$TMP/t2" | sed 's/^/  /'
else
  echo "  RESULT: FAILED"
  head -4 "$TMP/e2" | sed 's/^/    /'
fi

echo "  --- try 3: header-only existence check ---"
for d in /usr/include/opencv4 /usr/local/include/opencv4; do
  [ -d "$d" ] && echo "    $d : PRESENT" || echo "    $d : absent"
done
[ -f /usr/include/opencv4/opencv2/core.hpp ] && echo "    core.hpp : FOUND" \
  || echo "    core.hpp : not found anywhere standard"

rm -rf "$TMP"

echo
echo "--- [5/5] filesystem + dlopen survey (board_opencv_probe) ---"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$HERE/board_opencv_probe.cpp" ]; then
  if g++ -std=c++17 -O2 "$HERE/board_opencv_probe.cpp" -o "$HERE/board_opencv_probe" -ldl 2>"$HERE/.cc.log"; then
    echo "  probe compiled natively, running..."
    "$HERE/board_opencv_probe" | sed 's/^/  /'
  else
    echo "  probe failed to compile:"; head -10 "$HERE/.cc.log" | sed 's/^/    /'
  fi
else
  echo "  board_opencv_probe.cpp not found next to this script"
fi

echo
echo "======================================================================"
echo " done"
echo "======================================================================"
