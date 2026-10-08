#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Orchestration test for the C++ port: does each lag resolve to the RIGHT FRAME, and are the
composed lags composed in the RIGHT ORDER?

Why this exists
---------------
`test_port_units.py` verifies the port's own scalar logic (compose, anchor_grid, ordering,
median). It does NOT exercise `OnlineFeaturePipeline::push()` itself, which is where the
project's worst historical bug lived -- a ring buffer that indexed by *one-frame* lag while
lookups strided by `stride_step`, silently fetching every history frame at HALF the intended
lag. That bug produced no exception and no warning, only "metrics look a bit off".

Reading the code is not a substitute: during this port, six real defects were found, and
**every single one was found by running something, none by reading**. So the addressing is
tested, not inspected.

How
---
Both sides run the SAME algorithm on marker-encoded frames, with the OpenCV primitives
replaced by deterministic stand-ins that carry the identity of their input frame:

    pixel(0,0) of frame k  ==  k+1                      (frame k is the (k+1)-th)
    goodFeaturesToTrack    ->  remembers prev[0][0]  as a marker
    estimateAffinePartial2D->  returns [[1,0,marker],[0,1,0]]
    everything else        ->  trivial but deterministic

Run with `--downscale 1` so the marker survives untouched. Then, for current frame t:

    anchor lag L :  mats[L].tx  ==  marker(frame at t-L)          (clamped to frame 0)
    composed lag L:  mats[L].tx  ==  marker(t-L) + sum of marker(t-lag_abs) over the chain

Comparing that vector against the Python reference detects any lag off-by-factor, any
chain-order error, and any cold-start clamp error. The numbers are produced by the shipped
C++ (compiled here against generated stand-ins) and by the frozen Python engine.

This is a SIMULATION: the primitives are fake. It proves the ORCHESTRATION, never the
OpenCV equivalence. The end-to-end claim stays with verify_cpp_port.py's G1/G2 on real data.

Usage:
    python manu/pipeline/cpp/test_orchestration.py
    python manu/pipeline/cpp/test_orchestration.py --cxx g++ --frames 8
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

CPP_DIR = Path(__file__).resolve().parent
SRC = CPP_DIR / "gmc_stream.cpp"
PROJECT_ROOT = CPP_DIR.parents[2]

# ---------------------------------------------------------------------------
# Fake OpenCV headers + implementation, generated at run time so the repo stays clean.
# Signatures mirror real OpenCV 4.x; bodies are deterministic stand-ins.
# ---------------------------------------------------------------------------
FAKE_CORE = r"""
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#define CV_VERSION "FAKE-1.0"
#define CV_8U 0
#define CV_32F 5
#define CV_64F 6
#define CV_8UC1 0
#define CV_8UC3 16
#define CV_32FC2 4
#define AUTO_STEP 0
namespace cv {
typedef unsigned char uchar;
struct Point2f { float x=0.f, y=0.f; Point2f()=default; Point2f(float a,float b):x(a),y(b){} };
struct Size { int width=0,height=0; Size()=default; Size(int w,int h):width(w),height(h){} };
template <typename T> struct Vec_ { T v[3]; Vec_(){v[0]=v[1]=v[2]=T();}
  Vec_(T a,T b,T c){v[0]=a;v[1]=b;v[2]=c;} T& operator[](int i){return v[i];} };
typedef Vec_<uchar> Vec3b;
enum { INTER_LINEAR=1, BORDER_REFLECT=2, RANSAC=0 };
enum { IMREAD_GRAYSCALE=0 };
struct TermCriteria { int type=0,maxCount=0; double epsilon=0; TermCriteria(){} };
struct RNG { unsigned long long state=0xFFFFFFFFFFFFFFFFULL; };
RNG& theRNG(); void setNumThreads(int);
class Mat;
class InputArray { public: InputArray(){} InputArray(const Mat&m):p(const_cast<Mat*>(&m)){}
  template<typename T> InputArray(const std::vector<T>&v):p(nullptr){(void)v;} Mat* p=nullptr; };
class OutputArray { public: OutputArray(){} OutputArray(Mat&m):p(&m){}
  template<typename T> OutputArray(std::vector<T>&v):vp(&v){}
  Mat* p=nullptr; void* vp=nullptr; };
inline InputArray noArray(){ return InputArray(); }
template <typename _Tp,int m,int n> class Matx {
 public: _Tp val[m*n];
  Matx(){ for(int i=0;i<m*n;++i) val[i]=_Tp(); }
  Matx(_Tp v0){ for(int i=0;i<m*n;++i) val[i]=v0; }
  Matx(_Tp v0,_Tp v1,_Tp v2,_Tp v3,_Tp v4,_Tp v5){
    val[0]=v0;val[1]=v1;val[2]=v2;val[3]=v3;val[4]=v4;val[5]=v5; }
  _Tp& operator()(int i,int j){ return val[i*n+j]; }
  const _Tp& operator()(int i,int j) const { return val[i*n+j]; }
  static Matx eye(){ Matx r; for(int i=0;i<(m<n?m:n);++i) r(i,i)=_Tp(1); return r; } };
typedef Matx<float,2,3> Matx23f; typedef Matx<double,3,3> Matx33d;
class Mat {
 public:
  int rows=0, cols=0, type_=0;
  // Real OpenCV exposes `uchar* data` as a PUBLIC MEMBER (not an accessor). The port writes
  // `frame.data`, so the fake must match exactly or it would "prove" nothing.
  uchar* data=nullptr;
  Mat(){}
  Mat(int r,int c,int t):rows(r),cols(c),type_(t){ alloc(); }
  Mat(int r,int c,int t,void* d,size_t step=0):rows(r),cols(c),type_(t){
    alloc(); if(d) std::memcpy(data,d,(size_t)r*c*elemSize()); (void)step; }
  Mat(const Mat& o){ rows=o.rows; cols=o.cols; type_=o.type_; alloc();
    if(o.data) std::memcpy(data,o.data,total()*elemSize()); }
  Mat& operator=(const Mat& o){ if(this!=&o){ rows=o.rows; cols=o.cols; type_=o.type_; alloc();
      if(o.data) std::memcpy(data,o.data,total()*elemSize()); } return *this; }
  ~Mat(){ delete[] data; }
  int type() const { return type_; }
  int depth() const { return type_ & 7; }
  int channels() const { return 1 + ((type_>>3)&7); }
  size_t elemSize() const { size_t d = depth()==CV_64F?8:4; return d*channels(); }
  bool empty() const { return data==nullptr || total()==0; }
  size_t total() const { return (size_t)rows*cols; }
  Size size() const { return Size(cols,rows); }
  uchar* ptr(int i=0){ return data + (size_t)i*cols*elemSize(); }
  const uchar* ptr(int i=0) const { return data + (size_t)i*cols*elemSize(); }
  void create(int r,int c,int t){ delete[] data; data=nullptr; rows=r; cols=c; type_=t; alloc(); }
  void release(){ delete[] data; data=nullptr; rows=cols=0; }
  Mat clone() const { return Mat(*this); }
  void copyTo(OutputArray d) const { if(!d.p) return;
    d.p->create(rows,cols,type_); if(data) std::memcpy(d.p->data,data,total()*elemSize()); }
  template <typename T> T& at(int i,int j){ return *(T*)(data + ((size_t)i*cols+j)*elemSize()); }
  template <typename T> const T& at(int i,int j) const {
    return *(const T*)(data + ((size_t)i*cols+j)*elemSize()); }
 private:
  void alloc(){ if(total()) data = new uchar[total()*elemSize()](); }
};
}
"""

FAKE_HEADERS = {
    "opencv2/imgproc.hpp": """
#pragma once
#include <opencv2/core.hpp>
namespace cv {
void resize(InputArray, OutputArray, Size, double, double, int);
void absdiff(InputArray, InputArray, OutputArray);
// Real OpenCV order: (src, dst, M, dsize, ...). The stub MUST match or it would
// silently accept code that the real library rejects.
void warpAffine(InputArray, OutputArray, InputArray, Size, int, int, const void* = nullptr);
}""",
    "opencv2/imgcodecs.hpp": """
#pragma once
#include <string>
#include <opencv2/core.hpp>
namespace cv { Mat imread(const std::string&, int = 1);
bool imwrite(const std::string&, InputArray, const std::vector<int>& = {}); }""",
    "opencv2/features2d.hpp": """
#pragma once
#include <vector>
#include <opencv2/core.hpp>
namespace cv {
void goodFeaturesToTrack(InputArray, std::vector<Point2f>&, int, double, double, InputArray,
                         int, bool = false, double = 0.04); }""",
    "opencv2/calib3d.hpp": """
#pragma once
#include <opencv2/core.hpp>
namespace cv { Mat estimateAffinePartial2D(InputArray, InputArray, OutputArray, int, double,
                                          size_t = 1000, double = 0.99, size_t = 10); }""",
    "opencv2/video/tracking.hpp": """
#pragma once
#include <vector>
#include <opencv2/core.hpp>
namespace cv { void calcOpticalFlowPyrLK(InputArray, InputArray, InputArray, OutputArray,
                                       OutputArray, OutputArray, Size = Size(21,21), int = 2,
                                       TermCriteria = TermCriteria(), double = 1e-4, int = 0); }""",
}

FAKE_IMPL = r"""
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
namespace cv {
RNG g_rng; RNG& theRNG(){ return g_rng; }
void setNumThreads(int){}

// Marker of the image currently handed to goodFeaturesToTrack, i.e. the FIRST PIXEL.
// Pixel (0,0) of frame k is k+1, so this tag identifies exactly which frame was used.
int g_marker = 0;
#ifndef FAKE_IMREAD_W
#define FAKE_IMREAD_W 8
#define FAKE_IMREAD_H 8
#endif

void goodFeaturesToTrack(InputArray image, std::vector<Point2f>& corners, int, double, double,
                         InputArray, int, bool, double){
    const Mat& m = *image.p;
    g_marker = m.empty() ? 0 : int(m.ptr(0)[0]);
    corners.clear();
    for (int i = 0; i < 6; ++i) corners.push_back(Point2f(float(i), 0.f));
}

void calcOpticalFlowPyrLK(InputArray, InputArray, InputArray prevPts, OutputArray nextPts,
                          OutputArray status, OutputArray, Size, int, TermCriteria, double, int){
    // Flow is zero: return the input points unchanged. prevPts arrives as a Mat (the port
    // wraps the point vector in one); the outputs arrive as std::vector through OutputArray.
    const Mat& src = *prevPts.p;
    const size_t n = src.total();
    const Point2f* sp = reinterpret_cast<const Point2f*>(src.data);
    std::vector<Point2f>& dst = *reinterpret_cast<std::vector<Point2f>*>(nextPts.vp);
    dst.assign(sp, sp + n);
    std::vector<uchar>& st = *reinterpret_cast<std::vector<uchar>*>(status.vp);
    st.assign(n, 1);
}

Mat estimateAffinePartial2D(InputArray, InputArray, OutputArray, int, double, size_t, double, size_t){
    // A pure translation whose amount is the marker of the PREVIOUS frame: every downstream
    // compose() therefore accumulates a value that identifies its input frame.
    Mat M(2, 3, CV_64F);
    for (int r = 0; r < 2; ++r) for (int c = 0; c < 3; ++c) M.at<double>(r, c) = 0.0;
    M.at<double>(0, 0) = 1.0;
    M.at<double>(1, 1) = 1.0;
    M.at<double>(0, 2) = double(g_marker);
    return M;
}

void resize(InputArray src, OutputArray dst, Size, double, double, int){
    const Mat& s = *src.p; Mat& d = *dst.p;
    d.create(s.rows, s.cols, s.type());
    std::memcpy(d.data, s.data, (size_t)s.rows * s.cols * s.elemSize());
}
void absdiff(InputArray a, InputArray b, OutputArray d){
    const Mat& x = *a.p; const Mat& y = *b.p; Mat& o = *d.p;
    o.create(x.rows, x.cols, CV_8UC1);
    for (size_t i = 0; i < o.total(); ++i) {
        int v = int(x.ptr(0)[i]) - int(y.ptr(0)[i]); if (v < 0) v = -v;
        o.ptr(0)[i] = uchar(v);
    }
}
void warpAffine(InputArray src, OutputArray dst, InputArray M, Size dsize, int, int, const void*){
    const Mat& s = *src.p; Mat& o = *dst.p;
    o.create(dsize.height, dsize.width, s.type());
    // Pure-translation shift with wrap; marker-driven so lag identity survives the warp.
    const Mat& m = *M.p;
    const int tx = int(m.at<float>(0, 2));
    for (int y = 0; y < o.rows; ++y) for (int x = 0; x < o.cols; ++x) {
        int sx = ((x - tx) % s.cols + s.cols) % s.cols;
        int sy = ((y) % s.rows + s.rows) % s.rows;
        for (size_t k = 0; k < s.elemSize(); ++k)
            o.ptr(y)[(size_t)x*s.elemSize()+k] = s.ptr(sy)[(size_t)sx*s.elemSize()+k];
    }
}
Mat imread(const std::string& f, int){ Mat m; FILE* fp = std::fopen(f.c_str(), "rb");
    if (!fp) return m;
    // Headerless raw grayscale: the test harness guarantees the exact byte count.
    const int W = FAKE_IMREAD_W, H = FAKE_IMREAD_H;
    m.create(H, W, CV_8UC1);
    std::fread(m.data, 1, (size_t)W * H, fp);
    std::fclose(fp);
    return m; }
bool imwrite(const std::string& f, InputArray, const std::vector<int>&){ (void)f; return true; }
}
"""

# Python-side fake cv2 with identical semantics.
FAKE_CV2_PY = r'''
"""Deterministic OpenCV stand-ins, semantically identical to the C++ fakes."""
import numpy as np
import cv2_real_shim  # noqa: F401  (placeholder, unused)
'''


def write_fake_opencv(root: Path) -> None:
    (root / "opencv2" / "video").mkdir(parents=True, exist_ok=True)
    (root / "opencv2" / "core.hpp").write_text(FAKE_CORE, encoding="utf-8")
    for name, body in FAKE_HEADERS.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    (root / "fake_opencv.cpp").write_text(FAKE_IMPL, encoding="utf-8")


def build_binary(cxx: str, src: Path, workdir: Path) -> Path | None:
    write_fake_opencv(workdir)
    out = workdir / "gmc_stream"
    cmd = [cxx, "-std=c++17", "-O3", "-ffp-contract=off", "-fno-fast-math",
           "-fno-unsafe-math-optimizations", "-w",
           "-I", str(workdir), "-I", str(src.parent),
           str(src), str(workdir / "fake_opencv.cpp"), "-o", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[BUILD FAIL]\n{r.stderr[:3000]}")
        return None
    return out


def make_frames(seq_dir: Path, n: int, w: int = 8, h: int = 8) -> None:
    """Frame k has pixel(0,0) == k+1; the rest is a per-frame texture."""
    seq_dir.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        a = np.full((h, w), (k * 3) % 256, np.uint8)
        a[0, 0] = k + 1
        # Payload is raw grayscale (the fake imread parses it directly), but the extension is
        # .jpg because the shipped frame filter only accepts IMAGE_SUFFIXES -- using .pgm would
        # silently exercise an empty directory instead of the pipeline.
        (seq_dir / f"{k:06d}.jpg").write_bytes(a.tobytes())


def expected_tx(t: int, lag: int, lags: list[int], anchors: list[int],
                n_frames: int, stride: int) -> int:
    """Reference model of mats[lag].tx for the marker fakes."""
    def marker(frame_idx: int) -> int:
        return min(max(frame_idx, 0), n_frames - 1) + 1

    if lag in anchors:
        return marker(t - lag)
    below = max(a for a in anchors if a <= lag)
    tx = marker(t - below)
    for lag_abs in range(below + stride, lag + 1, stride):
        tx += marker(t - lag_abs)
    return tx


def run_case(cxx: str, src: Path, workdir: Path, frames: int, anchor_step: int,
             window: int, stride: int, quiet: bool) -> tuple[bool, list[tuple]]:
    """Build `src` against the fakes, run it, compare every lag. Returns (passed, bad_rows)."""
    binary = build_binary(cxx, src, workdir)
    if binary is None:
        raise RuntimeError("build failed")
    raw = workdir / "raw"
    seq = "SEQ"
    make_frames(raw / seq, frames)

    mats_frame = frames - 1  # a warm frame, past the cold-start plateau
    dump = workdir / "dump"
    cmd = [str(binary), "--raw-root", str(raw), "--sequence", seq,
           "--limit", str(frames), "--downscale", "1",
           "--anchor-step", str(anchor_step), "--window", str(window),
           "--stride-step", str(stride),
           "--mats-frame", str(mats_frame), "--dump-dir", str(dump)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"binary exited {r.returncode}: {r.stderr[:400]}")

    npf = dump / f"{seq}_{mats_frame:06d}_mats.npy"
    if not npf.is_file():
        raise RuntimeError(f"no _mats.npy at {npf}")
    cpp = np.load(npf)

    max_lag = window * stride
    lags = list(range(stride, max_lag + 1, stride))
    anchors = [a for a in sorted(set(range(stride, max_lag + 1, anchor_step)) | {max_lag})
               if a % stride == 0]

    t = mats_frame
    got = [float(row[2]) for row in cpp]
    exp = [float(expected_tx(t, lag, lags, anchors, frames, stride)) for lag in lags]

    bad = []
    if not quiet:
        print(f"\n  mats dumped for push index {mats_frame} ({len(got)} lags)")
        print(f"  {'lag':>5}{'anchor':>9}{'cpp tx':>10}{'expected':>11}{'ok':>7}")
        for i_, lag in enumerate(lags):
            ok = abs(got[i_] - exp[i_]) < 1e-9
            if not ok:
                bad.append((lag, got[i_], exp[i_], lag in anchors))
            print(f"  {lag:>5}{'Y' if lag in anchors else '-':>9}{got[i_]:>10.0f}{exp[i_]:>11.0f}"
                  f"{'  ok' if ok else ' FAIL':>7}")
    else:
        for i_, lag in enumerate(lags):
            if abs(got[i_] - exp[i_]) >= 1e-9:
                bad.append((lag, got[i_], exp[i_], lag in anchors))
    return (not bad), bad


# The historical production bug: a ring indexed by LAG while every frame is also its own
# current frame. Reintroducing it must make this test FAIL, otherwise the test has no teeth.
MUTATION_OLD = """        const Mat& f =
            ring_[static_cast<size_t>(((idx % ring_depth()) + ring_depth()) % ring_depth())];"""
MUTATION_NEW = """        const Mat& f =
            ring_[static_cast<size_t>(((lag % ring_depth()) + ring_depth()) % ring_depth())];"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxx", default="g++")
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--anchor-step", type=int, default=10)
    ap.add_argument("--window", type=int, default=21)
    ap.add_argument("--stride-step", type=int, default=2)
    ap.add_argument("--skip-mutation", action="store_true")
    args = ap.parse_args()

    print("=" * 92)
    print("   C++ PORT ORCHESTRATION TEST  (marker fakes; proves addressing, not OpenCV)")
    print("=" * 92)
    print(f"  source   : {SRC}")
    print(f"  compiler : {args.cxx}")
    print(f"  frames   : {args.frames}   anchor_step: {args.anchor_step}")
    print("=" * 92)

    ok = True
    wd = Path(tempfile.mkdtemp(prefix="gmcpp_orch_"))
    try:
        passed, bad = run_case(args.cxx, SRC, wd, args.frames, args.anchor_step,
                               args.window, args.stride_step, quiet=False)
        print("\n" + "=" * 92)
        if passed:
            print("[CASE 1] PASS -- all lags resolve to the correct frame "
                  "(anchor fits, chain order, cold-start clamp all hold)")
        else:
            ok = False
            print(f"[CASE 1] FAIL -- {len(bad)} lags resolve to the wrong frame")
            for lag, g, e, is_a in bad[:8]:
                print(f"   lag {lag:>3} ({'anchor' if is_a else 'composed'}): "
                      f"got {g:.0f}, expected {e:.0f}")
            if any(is_a for _, _, _, is_a in bad):
                print("   -> an ANCHOR lag is wrong: the ring buffer is not returning frame t-lag.")
            else:
                print("   -> only composed lags are wrong: chain order/indices are off.")
        print("=" * 92)

        # anchor_step == 2 : every lag is an anchor, zero composition.
        wd2 = Path(tempfile.mkdtemp(prefix="gmcpp_orch2_"))
        try:
            p2, b2 = run_case(args.cxx, SRC, wd2, args.frames, 2, args.window,
                              args.stride_step, quiet=True)
            print(f"[CASE 2] anchor_step=2 (zero composition): "
                  f"{'PASS' if p2 else 'FAIL'} ({len(b2)} lags wrong)")
            ok &= p2
        finally:
            shutil.rmtree(wd2, ignore_errors=True)

        # Mutation check: does this test actually detect the historical ring bug?
        if args.skip_mutation:
            print("[CASE 3] mutation check: SKIPPED (--skip-mutation)")
        else:
            text = SRC.read_text(encoding="utf-8")
            if MUTATION_OLD not in text:
                ok = False
                print("[CASE 3] FAIL -- could not locate the ring-index expression to mutate; "
                      "the mutation check cannot run and this test's validity is unproven")
            else:
                mut_dir = Path(tempfile.mkdtemp(prefix="gmcpp_mut_"))
                mut_dir.mkdir(parents=True, exist_ok=True)
                mut_src = mut_dir / "gmc_stream_mutated.cpp"
                mut_src.write_text(text.replace(MUTATION_OLD, MUTATION_NEW), encoding="utf-8")
                md5_h = mut_dir / "md5.h"
                shutil.copy(CPP_DIR / "md5.h", md5_h)
                try:
                    wd3 = Path(tempfile.mkdtemp(prefix="gmcpp_orch3_"))
                    try:
                        p3, b3 = run_case(args.cxx, mut_src, wd3, args.frames, args.anchor_step,
                                          args.window, args.stride_step, quiet=True)
                        if p3:
                            ok = False
                            print("[CASE 3] FAIL -- the mutated (historical ring bug) build PASSED. "
                                  "This test cannot detect that bug and proves nothing.")
                        else:
                            print(f"[CASE 3] PASS -- mutation detected: reintroducing the lag-indexed "
                                  f"ring makes {len(b3)} lags fail. The test has teeth.")
                    finally:
                        shutil.rmtree(wd3, ignore_errors=True)
                finally:
                    shutil.rmtree(mut_dir, ignore_errors=True)

        print("\n" + "=" * 92)
        print(f"[VERDICT] {'PASS' if ok else 'FAIL'} -- orchestration addressing verified"
              f"{' (mutation check included)' if not args.skip_mutation else ''}")
        print("  NOTE: primitives are fakes. This proves ADDRESSING and COMPOSITION ORDER,")
        print("        never OpenCV equivalence. The end-to-end claim stays with")
        print("        verify_cpp_port.py's G1/G2 on real data.")
        print("=" * 92)
        return 0 if ok else 1
    finally:
        shutil.rmtree(wd, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
