// C++ port of the streaming three-channel feature engine for Trial 0474.
//
//   Ch0 = I_t
//   Ch1 = |I_t - W(I_{t-2})|                        (lag 2 exactly; never 1, never 4)
//   Ch2 = (I_t - B_t)^+,  B_t = median{W(I_{t-2k})}, k = 1..21
//
// Python golden reference (bit-for-bit target):
//   manu/pipeline/streaming_feature_pipeline.py   (OnlineFeaturePipeline)
// whose primitives are imported from
//   manu/data/build_nogmc_median_dataset.py       (_compose, anchor_grid)
//   manu/data/build_sample_median_dataset.py       (FastGMCEstimator)
//
// ===========================================================================
// gmc_stream_ocl.cpp -- gmc_stream.cpp with an OpenCL fused tail, plus a
// CPU/GPU A-B that runs BOTH arms on the SAME fits inside one binary.
//
// WHY A FORK AND NOT AN EDIT OF gmc_stream.cpp
//   gmc_stream.cpp is the validated CPU baseline (400/400 bit-exact against the
//   Python engine, 396/396 against the frozen uav_gmc_median set). Editing it in
//   place risks the thing it exists to certify. This file is a superset: with
//   --fused cpu its MD5 stream must equal gmc_stream.cpp's exactly, and that
//   equality is the proof the fork is faithful. Verify it before trusting any
//   GPU number from this binary.
//
// WHAT MOVED TO THE GPU (and nothing else)
//   CPU keeps: Shi-Tomasi, pyramidal LK, RANSAC partial-affine, the anchor grid
//              and the bounded chain composition.
//   GPU takes: the 21 inverse re-projections, the 21-element register-resident
//              median, and the [Ch0,Ch1,Ch2] plane-major assembly.
//   The fits are computed ONCE per push and handed to both arms, so the A/B can
//   never pass by feeding the two arms different inputs.
//
// ---------------------------------------------------------------------------
// Why this port can be bit-exact
// ---------------------------------------------------------------------------
// The Python reference is not an algorithm re-implementation either: cv2 is a thin
// binding over the very same OpenCV C++ translation units. Shi-Tomasi, pyramidal
// Lucas-Kanade, RANSAC partial-affine, warpAffine and resize all live in libopencv_*
// compiled C++. So this port calls *those same C++ entry points with identical
// arguments, in an identical call order, on an identical OpenCV build*; the primitive
// results are the same machine code, hence bit-identical.
//
// What is actually ported here is the ORCHESTRATION -- the part Python adds on top:
//   * absolute-indexed ring buffer (never list shifting, never lag-indexed),
//   * the anchor grid and the bounded chain composition,
//   * the exact float64 homogeneous compose + guards of `_compose`,
//   * the odd-length median and the float32 clip of Ch2,
//   * the channel assembly order.
//
// ---------------------------------------------------------------------------
// Floating-point contract (violating any of these breaks bit-exactness)
// ---------------------------------------------------------------------------
//  1. `_compose` runs in float64 with a k-ascending dot product, exactly like
//     `np.eye(3) @ np.eye(3)` on 2x3-embedded transforms. Build with -ffp-contract=off
//     so the compiler may not fuse `acc + a*b` into an FMA and change the rounding.
//  2. No -Ofast / -ffast-math: those enable reassociation, which breaks associativity
//     and silently changes the chain result.
//  3. OpenCV must be the same build/version as the Python side, and both sides run
//     single-threaded (cv2.setNumThreads(1) / cv::setNumThreads(1)). Under multiple
//     OpenCV threads RANSAC and Lucas-Kanade are not bit-reproducible: an anchor fit can
//     flip between two equally plausible solutions, shifting a whole warped edge.
//  4. RANSAC consumes OpenCV's global RNG (theRNG()). By default both processes issue
//     the same draws in the same order, which matches. `--rng-seed-per-fit` reseeds before
//     every fit for hardening against any unrelated RNG consumption -- but if you enable
//     it you MUST mirror it on the Python side; it will no longer reproduce the frozen
//     files, which were built without it.
//  5. The median is taken over an ODD count (21), so it is a selection, not an average:
//     exact in any implementation. An even count would be an average and would instead have
//     to match numpy's tie and overflow rules.
//
// ---------------------------------------------------------------------------
// anchor_step semantics
// ---------------------------------------------------------------------------
//   --anchor-step 10  (default) anchors {2,12,22,32,42}, depth 4, 5 anchor fits/frame
//                     -> F1 0.906050, dF1 -0.000311 vs native. The shipping config.
//   --anchor-step 2   anchors {2,4,...,42}, depth 0, ZERO composition, 21 anchor
//                     fits/frame. This is the ONLY setting that is *exactly* equivalent
//                     to the frozen native (SOTA) pipeline, because every lag gets its
//                     own long-baseline direct fit. Run this config against
//                     `uav_gmc_median` to prove the port reproduces SOTA features.
//
// Build:
//   cmake -S manu/pipeline/cpp -B manu/pipeline/cpp/build -DCMAKE_BUILD_TYPE=Release
//   cmake --build manu/pipeline/cpp/build -j
//
// Run:
//   ./manu/pipeline/cpp/build/gmc_stream
//       --raw-root /mnt/data/siping/datasets/manu/anti-uav
//       --sequence wg2022_ir_052_split_08 --limit 200 --anchor-step 2
//       --md5-out runs/cpp_port/step2.md5 --dump-dir runs/cpp_port/step2_npy

#include "md5.h"

// OpenCL fused tail (warp + 21-frame median + 3-channel assembly).
// The module lives next door; see manu/pipeline/opencl/README.md for the full
// rationale behind every arithmetic choice in the kernel.
#include "../opencl/host/ocl_host.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <array>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <dirent.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/stat.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace {

using cv::Mat;
using cv::Matx23f;
using cv::Matx33d;

// np.eye(2, 3, dtype=np.float32)
const Matx23f kIdentity(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);

// -----------------------------------------------------------------------------
// natural_key: a byte-exact port of the Python comparator used to order frames.
//   re.split(r"(\d+)", stem) -> [int(p) if p.isdigit() else p.lower() for p in parts]
// Python compares those lists element-wise; digit runs compare numerically, text runs
// compare case-folded. Mixed-type positions cannot occur within one sequence folder.
// -----------------------------------------------------------------------------
std::vector<std::pair<int, std::string>> tokenize_stem(const std::string& stem) {
    // Must reproduce `re.split(r"(\d+)", stem)` exactly. Because the pattern has a CAPTURING
    // group, re.split always yields the text before each digit run AND the trailing text,
    // including EMPTY ones: "000001" -> ['', '000001', '']. A tokenizer that drops the empty
    // text segments yields a different token list and therefore a different sort order.
    std::vector<std::pair<int, std::string>> out;  // (is_digit, text)
    size_t i = 0;
    while (true) {
        size_t j = i;
        while (j < stem.size() && !std::isdigit(static_cast<unsigned char>(stem[j]))) ++j;
        std::string text = stem.substr(i, j - i);  // may legitimately be empty
        for (char& c : text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        out.emplace_back(0, text);
        if (j >= stem.size()) break;
        size_t k = j;
        while (k < stem.size() && std::isdigit(static_cast<unsigned char>(stem[k]))) ++k;
        out.emplace_back(1, stem.substr(j, k - j));
        i = k;
    }
    return out;
}

bool natural_less(const std::string& a, const std::string& b) {
    const auto ta = tokenize_stem(a);
    const auto tb = tokenize_stem(b);
    const size_t n = std::min(ta.size(), tb.size());
    for (size_t i = 0; i < n; ++i) {
        if (ta[i].first != tb[i].first) return ta[i].first < tb[i].first;  // digits first
        if (ta[i].first == 1) {
            const long long va = std::strtoll(ta[i].second.c_str(), nullptr, 10);
            const long long vb = std::strtoll(tb[i].second.c_str(), nullptr, 10);
            if (va != vb) return va < vb;
        } else if (ta[i].second != tb[i].second) {
            return ta[i].second < tb[i].second;
        }
    }
    return ta.size() < tb.size();
}

bool is_image_ext(const std::string& p) {
    if (p.size() < 4) return false;
    std::string e = p.substr(p.size() - 4);
    for (char& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return e == ".jpg" || e == ".jpeg" || e == ".png" || e == ".bmp";
}

bool is_dir(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::vector<std::string> list_dir(const std::string& dir, bool images_only) {
    std::vector<std::string> names;
    DIR* d = opendir(dir.c_str());
    if (!d) return names;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        const std::string n = e->d_name;
        if (n == "." || n == "..") continue;
        if (images_only && !is_image_ext(n)) continue;
        names.push_back(n);
    }
    closedir(d);
    return names;
}

// Same candidate order as the frozen builder: <root>/<seq>, Data/val, val, Data/train,
// train, then a recursive scan as the last resort.
bool find_sequence_folder(const std::string& raw_root, const std::string& seq, std::string* out) {
    const std::string cands[] = {raw_root + "/" + seq,
                                 raw_root + "/Data/val/" + seq,
                                 raw_root + "/val/" + seq,
                                 raw_root + "/Data/train/" + seq,
                                 raw_root + "/train/" + seq};
    for (const auto& c : cands) {
        if (is_dir(c)) {
            *out = c;
            return true;
        }
    }
    std::vector<std::string> queue{raw_root};
    while (!queue.empty()) {
        const std::string cur = queue.back();
        queue.pop_back();
        for (const std::string& n : list_dir(cur, false)) {
            const std::string p = cur + "/" + n;
            if (!is_dir(p)) continue;
            if (n == seq) {
                *out = p;
                return true;
            }
            queue.push_back(p);
        }
    }
    return false;
}

// -----------------------------------------------------------------------------
// FastGMCEstimator -- identical parameters, identical call order.
//
// FIT PARAMS (the shipped defaults reproduce the frozen configuration exactly;
// every one of them is a runtime option so the accuracy/perf curve can be
// walked without recompiling):
//   downscale   2     half-res already: 640x512 -> 320x256 before any fitting
//   max_corners 600   Shi-Tomasi cap
//   win         15    LK window
//   max_level   2     LK pyramid depth
//   iters       20    LK max iterations (epsilon 0.03, unchanged)
//
// PER-FRAME CACHE (--pyr-cache)
// Three intermediates are pure functions of ONE input image: the half-res
// resize, the Shi-Tomasi corners, and the optical-flow pyramid. Each frame is
// the "prev" side of ~3.25 fits per push (5 anchor lags 2/12/22/32/42 plus the
// cached one-step fits), so all three were recomputed from identical bytes on
// every fit. Measured on the 60-frame x86 sequence: Shi-Tomasi is ~66% of fit,
// the pyramid build ~3%, LK ~27%. So the cache is worth much more than the
// pyramid it is named after.
//
// Why the cached LK is bit-identical, not approximately so:
//   * the resize and the corner detector are pure functions of the same bytes;
//   * calcOpticalFlowPyrLK accepts an already-built pyramid whenever the
//     InputArray kind is STD_VECTOR_MAT (lkpyramid.cpp:1302/1330), and detects
//     precomputed gradients via the odd level count + channel/depth test
//     (lkpyramid.cpp:1309), switching to lvlStep=2 and reusing the derivative
//     planes instead of recomputing Scharr every call (lkpyramid.cpp:1392).
//     buildOpticalFlowPyramid's defaults -- withDerivatives=true,
//     pyrBorder=BORDER_REFLECT_101, derivBorder=BORDER_CONSTANT -- are exactly
//     the arguments calc() passes internally, so the planes are the same bytes;
//   * points stay in UNPADDED image coordinates: pyramid level 0 is an ROI view
//     into the padded buffer and LKTrackerInvoker indexes relative to that view
//     (lkpyramid.cpp:204-220), so no coordinate shift is applied.
//
// The cache is keyed by ABSOLUTE frame index. A plain "slot is filled" boolean
// is wrong and fails silently: slot(43) aliases slot(0), the flag left by frame
// 0 is still set, and frame 43 gets served frame 0's pixels -- no crash, no
// assert, just stale data. Hence the tag arrays, which are also what makes the
// ring correct when it wraps.
// -----------------------------------------------------------------------------
class FastGMCEstimator {
public:
    // Which per-frame intermediates the cache owns. Split out so a single part
    // can be ablated: a bit-exactness claim has to survive turning each piece
    // on by itself, not just all three together.
    enum : int { kCacheResize = 1, kCacheCorners = 2, kCachePyr = 4,
                 kCacheAll = 1 | 2 | 4 };

    FastGMCEstimator(int downscale, bool cache, int ring_depth, int max_corners, int win,
                     int max_level, int iters, int parts)
        : downscale_(downscale),
          cache_(cache),
          parts_(parts),
          max_corners_(max_corners),
          win_(win),
          max_level_(max_level),
          iters_(iters),
          ring_(static_cast<size_t>(std::max(ring_depth, 1))) {
        if (cache_) {
            small_.resize(ring_);
            corners_.resize(ring_);
            pyr_.resize(ring_);
            tag_small_.assign(ring_, -1);
            src_small_.assign(ring_, nullptr);
            tag_corners_.assign(ring_, -1);
            tag_pyr_.assign(ring_, -1);
        }
    }

    // Honest memory accounting: with maxLevel=2 the cached pyramid carries three
    // image planes plus three derivative planes, so this is not a small buffer.
    size_t cache_bytes() const {
        if (!cache_) return 0;
        size_t n = 0;
        for (size_t i = 0; i < ring_; ++i) {
            n += static_cast<size_t>(small_[i].total() * small_[i].elemSize());
            for (const Mat& m : pyr_[i])
                if (!m.empty()) n += static_cast<size_t>(m.total() * m.elemSize());
            n += corners_[i].capacity() * sizeof(cv::Point2f);
        }
        return n;
    }
    long resize_builds() const { return n_resize_builds_; }
    long corner_builds() const { return n_corner_builds_; }
    long pyr_builds() const { return n_pyr_builds_; }
    long cache_hits() const { return n_hits_; }
    long cache_lookups() const { return n_lookups_; }

    // ---------------------------------------------------------------------
    // THE SPLIT FIT: an LK stage and a RANSAC stage.
    //
    // WHY THE SPLIT EXISTS
    //   The reference fit is ONE call: LK, then RANSAC.  RANSAC consumes
    //   OpenCV's global RNG (theRNG()), so the ORDER of the fits is part of
    //   the output: five anchors fitted back to back draw five RNG blocks in a
    //   fixed order, and a fit that draws a different block can land on the
    //   other of two equally plausible solutions and shift a whole warped edge.
    //   That is why the FITS may not be run concurrently.
    //
    //   But the LK stage consumes NO RNG and is a pure function of (prev
    //   image, curr image, corners).  So the fit can be split:
    //     * the LK stage of every anchor runs CONCURRENTLY -- each anchor
    //       still runs the very same single-threaded OpenCV code on the very
    //       same bytes, so every output bit is the bit it was before; and
    //     * the RANSAC stages run SEQUENTIALLY, in the anchor order, in the
    //       thread that ran them before -- so the global RNG is drawn from in
    //       exactly the same order as before, and the output is bit-identical.
    //
    //   Nothing else changes: the OpenCV call signatures, the thresholds, the
    //   gate and the order of the surrounding bookkeeping are all untouched.
    //
    // THE CACHE IS SHARED MUTABLE STATE
    //   The per-frame cache (resize / corners / pyramid) is written by whoever
    //   fills it, so it is touched ONLY by prepare_fit(), which runs SERIALLY
    //   before any LK starts.  The LK stage then reads a snapshot of it and
    //   writes nothing at all -- no counters, no tags, no buffers -- so five
    //   anchors can run it at once without a data race.
    // ---------------------------------------------------------------------

    // The inputs of one anchor's LK stage: a READ-ONLY snapshot of the
    // per-frame cache, taken while nothing else is running.
    struct LkPlan {
        Mat prev_small, curr_small;             // headers onto the cached pixels
        std::vector<cv::Point2f> prev_corners;  // the source corners
        std::vector<Mat> prev_pyr, curr_pyr;    // the pyramids (when prebuilt)
        bool prebuilt_pyr = false;              // the reference's call shape
        bool enough_corners = false;            // >= 6 corners, else IDENTITY
    };

    // The LK stage's output: the status-filtered point pairs the RANSAC stage
    // consumes.  `ok == false` reproduces the reference's early return: the
    // fit is IDENTITY and NO RANSAC runs, so no RNG is drawn at all.
    struct LkPairs {
        std::vector<cv::Point2f> p0, p1;
        bool ok = false;
    };

    // SERIAL.  Fills `plan` with everything the LK stage needs, touching the
    // cache in exactly the order -- and with exactly the early exits -- the
    // reference's compute_affine() touched it.  The order matters for the
    // cache telemetry (builds / hits) and for the pyramid: the reference
    // returns BEFORE building the pyramid when there are fewer than 6
    // corners, and so does this.
    void prepare_fit(long prev_abs, long curr_abs, const Mat& prev_gray,
                     const Mat& curr_gray, LkPlan* plan) const {
        if (!cache_) {
            // The reference's no-cache branch: plain resize, fresh corners,
            // and the LK builds its own pyramid internally.
            //
            // The resize target size comes from the CURRENT frame -- exactly
            // as the reference's compute_affine() took it (it sized BOTH
            // resizes from curr_gray).  plain_small() would size each resize
            // from its own image, which is the same thing while the frames
            // share a shape (the pipeline rejects a shape change) but is not
            // the same code, so it is spelled out here.
            const int ds = downscale_;
            if (ds > 1) {
                const cv::Size small(curr_gray.cols / ds, curr_gray.rows / ds);
                cv::resize(prev_gray, plan->prev_small, small, 0, 0, cv::INTER_LINEAR);
                cv::resize(curr_gray, plan->curr_small, small, 0, 0, cv::INTER_LINEAR);
            } else {
                plan->prev_small = prev_gray;
                plan->curr_small = curr_gray;
            }
            detect(plan->prev_small, plan->prev_corners);
            ++n_corner_builds_;
            plan->prebuilt_pyr = false;
            plan->enough_corners = plan->prev_corners.size() >= 6;
            return;
        }
        small_of(prev_abs, prev_gray, plan->prev_small);
        small_of(curr_abs, curr_gray, plan->curr_small);
        // goodFeaturesToTrack: maxCorners=600, qualityLevel=0.01, minDistance=4, blockSize=3
        corners_of(prev_abs, plan->prev_small, plan->prev_corners);
        plan->enough_corners = plan->prev_corners.size() >= 6;
        if (!plan->enough_corners) return;   // the reference returns here
        pyr_of(prev_abs, plan->prev_small, plan->prev_pyr);
        pyr_of(curr_abs, plan->curr_small, plan->curr_pyr);
        plan->prebuilt_pyr = true;
    }

    // The LK stage.  Reads the snapshot, touches NO shared state and NO RNG,
    // so it may run on several anchors at once.  Every OpenCV argument is
    // the reference's, in the reference's call shape.
    LkPairs run_lk(const LkPlan& plan) const {
        LkPairs out;
        if (!plan.enough_corners) return out;
        // Wrap as (N,1) CV_32FC2 -- the same layout cv2 hands over from the Python side.
        const Mat pts_prev_mat(static_cast<int>(plan.prev_corners.size()), 1, CV_32FC2,
                               const_cast<cv::Point2f*>(plan.prev_corners.data()));
        std::vector<cv::Point2f> pts_curr;
        std::vector<unsigned char> status;
        std::vector<float> err;
        // winSize=(15,15), maxLevel=2; flags/criteria/minEigThreshold left at their defaults
        if (iters_ != 20) {
            // Only pay for a non-default criterion when it was actually asked for;
            // the default path below keeps the exact shipped call signature.
            cv::TermCriteria crit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, iters_, 0.03);
            if (!plan.prebuilt_pyr) {
                cv::calcOpticalFlowPyrLK(plan.prev_small, plan.curr_small, pts_prev_mat, pts_curr,
                                          status, err, cv::Size(win_, win_), max_level_, crit);
            } else {
                cv::calcOpticalFlowPyrLK(plan.prev_pyr, plan.curr_pyr, pts_prev_mat, pts_curr,
                                          status, err, cv::Size(win_, win_), max_level_, crit);
            }
        } else if (!plan.prebuilt_pyr) {
            cv::calcOpticalFlowPyrLK(plan.prev_small, plan.curr_small, pts_prev_mat, pts_curr, status,
                                      err, cv::Size(win_, win_), max_level_);
        } else {
            cv::calcOpticalFlowPyrLK(plan.prev_pyr, plan.curr_pyr, pts_prev_mat, pts_curr, status, err,
                                      cv::Size(win_, win_), max_level_);
        }

        out.p0.reserve(plan.prev_corners.size());
        out.p1.reserve(plan.prev_corners.size());
        for (size_t i = 0; i < plan.prev_corners.size(); ++i) {
            if (i < status.size() && status[i] == 1) {
                out.p0.push_back(plan.prev_corners[i]);
                out.p1.push_back(pts_curr[i]);
            }
        }
        if (out.p0.size() < 6) return LkPairs();   // the reference returns IDENTITY here
        out.ok = true;
        return out;
    }

    // The RANSAC stage.  This is the ONLY part of a fit that draws from
    // OpenCV's global RNG, so it must run in the anchor order and in ONE
    // thread -- which is exactly what the caller does.
    Matx23f ransac_stage(const LkPairs& lk) const {
        if (!lk.ok) return kIdentity;
        Mat inliers;
        const Mat M = (ransac_max_iters_ > 0)
                          ? cv::estimateAffinePartial2D(lk.p0, lk.p1, inliers, cv::RANSAC, 3.0,
                                                        ransac_max_iters_)
                          : cv::estimateAffinePartial2D(lk.p0, lk.p1, inliers, cv::RANSAC, 3.0);
        if (M.empty()) return kIdentity;

        double v[6];
        for (int i = 0; i < 6; ++i) v[i] = M.at<double>(i / 3, i % 3);
        Matx23f H(static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2]),
                  static_cast<float>(v[3]), static_cast<float>(v[4]), static_cast<float>(v[5]));
        if (downscale_ > 1) {
            // float32 * float32(ds): matches numpy's weak-scalar semantics exactly.
            H(0, 2) = H(0, 2) * static_cast<float>(downscale_);
            H(1, 2) = H(1, 2) * static_cast<float>(downscale_);
        }
        return H;
    }

    // The reference fit, unchanged: prepare -> LK -> RANSAC.  The serial
    // anchor path and the one-step fits both go through here.
    Matx23f compute_affine(long prev_abs, long curr_abs, const Mat& prev_gray,
                           const Mat& curr_gray) const {
        LkPlan plan;
        prepare_fit(prev_abs, curr_abs, prev_gray, curr_gray, &plan);
        return ransac_stage(run_lk(plan));
    }

    Mat warp(const Mat& img, const Matx23f& H) const {
        Mat M(2, 3, CV_32F);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j) M.at<float>(i, j) = H(i, j);
        Mat dst;
        // NOTE: real OpenCV order is (src, dst, M, dsize, ...) -- the matrix comes BEFORE dsize.
        cv::warpAffine(img, dst, M, cv::Size(img.cols, img.rows), cv::INTER_LINEAR,
                       cv::BORDER_REFLECT);
        return dst;
    }

private:
    // Slot index for an ABSOLUTE frame number. The ring is max_lag+1 deep, which
    // is exactly the set of frames a single push can reference, so two live
    // frames can never alias onto one entry.
    size_t slot_of(long abs) const {
        const long m = static_cast<long>(ring_);
        return static_cast<size_t>(((abs % m) + m) % m);
    }

    static bool g_verify_cache() {
        static const bool v = std::getenv("GMC_VERIFY_CACHE") != nullptr;
        return v;
    }

    void small_of(long abs, const Mat& gray, Mat& out) const {
        if (!(parts_ & kCacheResize)) { plain_small(gray, out); return; }
        ++n_lookups_;
        const size_t s = slot_of(abs);
        if (tag_small_[s] == abs) {
            // GMC_VERIFY_CACHE=1 re-derives every hit and compares. A cache that
            // silently serves the wrong frame produces plausible output and a
            // different MD5, which is very hard to localise by reasoning; this
            // makes it fail loudly instead. Off by default (it costs a resize
            // and a comparison per hit).
            if (g_verify_cache()) {
                Mat fresh;
                plain_small(gray, fresh);
                Mat d = fresh != small_[s];
                const int total = cv::countNonZero(d);
                if (total != 0) {
                    int fx = -1, fy = -1;
                    for (int y = 0; y < d.rows && fx < 0; ++y)
                        for (int x = 0; x < d.cols; ++x)
                            if (d.at<uchar>(y, x)) { fx = x; fy = y; break; }
                    std::fprintf(stderr,
                                 "[CACHE-BUG] small abs=%ld slot=%zu gray=%p(%dx%d) "
                                 "filled_from=%p diffpx=%d first@(%d,%d)\n",
                                 abs, s, (const void*)gray.data, gray.cols, gray.rows,
                                 src_small_[s], total, fx, fy);
                    std::exit(3);
                }
            }
            ++n_hits_;
            out = small_[s];
            return;
        }
        {
            const cv::Size small(gray.cols / downscale_, gray.rows / downscale_);
            if (downscale_ > 1) cv::resize(gray, small_[s], small, 0, 0, cv::INTER_LINEAR);
            else small_[s] = gray;
        }
        tag_small_[s] = abs;
        src_small_[s] = gray.data;
        tag_corners_[s] = -1;  // the resize changed, so the corners are stale
        tag_pyr_[s] = -1;
        ++n_resize_builds_;
        out = small_[s];
    }

    void plain_small(const Mat& gray, Mat& out) const {
        if (downscale_ > 1) {
            const cv::Size small(gray.cols / downscale_, gray.rows / downscale_);
            cv::resize(gray, out, small, 0, 0, cv::INTER_LINEAR);
        } else {
            out = gray;
        }
    }

    // Detect corners under the budget, and count the frames where the budget
    // actually bound.
    //
    // goodFeaturesToTrack returns corners in descending Shi-Tomasi response and
    // stops as soon as maxCorners is reached, so asking for `budget + 1` and
    // dropping the last point yields exactly the same top-`budget` set a plain
    // maxCorners=budget call returns -- while telling apart "this frame has
    // exactly `budget` corners" from "this frame had more and we cut it". Without
    // the probe, `size() == max_corners_` cannot distinguish the two and the
    // trigger statistic would be an unverified guess.
    //
    // The equivalence is an assumption about OpenCV's ordering, so it is not
    // taken on faith: --corner-probe 0 restores the plain call and the zero-
    // regression run compares the two MD5 streams on inputs where the budget
    // never binds.
    void detect(const Mat& small, std::vector<cv::Point2f>& out) const {
        ++n_corner_det_;
        if (!probe_) {
            cv::goodFeaturesToTrack(small, out, max_corners_, 0.01, 4, cv::noArray(), 3);
            n_corner_pts_ += static_cast<long>(out.size());
            return;
        }
        std::vector<cv::Point2f> all;
        cv::goodFeaturesToTrack(small, all, max_corners_ + 1, 0.01, 4, cv::noArray(), 3);
        if (static_cast<int>(all.size()) > max_corners_) {
            ++n_corner_capped_;
            n_corner_dropped_ += static_cast<long>(all.size()) - max_corners_;
            all.resize(static_cast<size_t>(max_corners_));
        }
        n_corner_pts_ += static_cast<long>(all.size());
        out.swap(all);
    }

    void corners_of(long abs, const Mat& small, std::vector<cv::Point2f>& out) const {
        if (!(parts_ & kCacheCorners)) {
            detect(small, out);
            ++n_corner_builds_;
            return;
        }
        ++n_lookups_;
        const size_t s = slot_of(abs);
        if (tag_corners_[s] == abs) { ++n_hits_; out = corners_[s]; return; }
        detect(small, corners_[s]);
        tag_corners_[s] = abs;
        ++n_corner_builds_;
        out = corners_[s];
    }

    void pyr_of(long abs, const Mat& small, std::vector<Mat>& out) const {
        if (!(parts_ & kCachePyr)) {
            cv::buildOpticalFlowPyramid(small, out, cv::Size(win_, win_), max_level_, true);
            ++n_pyr_builds_;
            return;
        }
        ++n_lookups_;
        const size_t s = slot_of(abs);
        if (tag_pyr_[s] == abs) { ++n_hits_; out = pyr_[s]; return; }
        // Defaults are withDerivatives=true, pyrBorder=BORDER_REFLECT_101,
        // derivBorder=BORDER_CONSTANT -- identical to what calcOpticalFlowPyrLK
        // passes internally, so the planes are the same bytes.
        cv::buildOpticalFlowPyramid(small, pyr_[s], cv::Size(win_, win_), max_level_, true);
        tag_pyr_[s] = abs;
        ++n_pyr_builds_;
        out = pyr_[s];
    }

    int downscale_;
    bool cache_;
    int parts_;
    int max_corners_, win_, max_level_, iters_;
    size_t ring_;
    mutable std::vector<Mat> small_;
    mutable std::vector<std::vector<cv::Point2f>> corners_;
    mutable std::vector<std::vector<Mat>> pyr_;
    mutable std::vector<long> tag_small_, tag_corners_, tag_pyr_;
    mutable std::vector<const void*> src_small_;
    mutable long n_resize_builds_ = 0, n_corner_builds_ = 0, n_pyr_builds_ = 0;
    mutable long n_hits_ = 0, n_lookups_ = 0;
    // Corner-budget triggers. Frames whose detected corner count exceeded the
    // budget, and how many points were dropped to bring it back under.
    mutable long n_corner_capped_ = 0, n_corner_dropped_ = 0;
    // Corner census: how many detections ran, and how many points they returned.
    // This is the pipeline's OWN count, which is not the same number as the one
    // pick_sequences.py reports -- that tool samples a different frame set with
    // its own detector settings, and on VIDEO00002 the two disagree by ~1.8x
    // (336.4 sampled vs >=601 actual). Quoting the sampled one as "the" corner
    // count is how a calibration number ends up cited against the wrong thing.
    mutable long n_corner_det_ = 0, n_corner_pts_ = 0;
    bool probe_ = true;

public:
    void set_probe(bool on) { probe_ = on; }
    long corner_capped() const { return n_corner_capped_; }
    long corner_dropped() const { return n_corner_dropped_; }
    long corner_dets() const { return n_corner_det_; }
    // Mean corners PER DETECTION. Equal to the true mean only while the budget
    // never binds; once it binds, this pins at the budget and the real count is
    // higher -- pair it with corner_capped() and never quote it alone.
    double corner_mean() const {
        return n_corner_det_ ? (double)n_corner_pts_ / (double)n_corner_det_ : 0.0;
    }

    // RANSAC sampling cap for the affine solve.  0 = OFF: the reference call
    // is made with its own default arguments, which is what the bit-exactness
    // contract requires.  A positive value caps the sampling iterations, so a
    // pathological frame cannot spin; the fit then returns the best transform
    // found so far.  ENABLING IT CHANGES THE NUMBERS, so it is off by default
    // and must be reported as a deviation when used.
    void set_ransac_max_iters(int n) { ransac_max_iters_ = n > 0 ? n : 0; }
    int ransac_max_iters() const { return ransac_max_iters_; }

private:
    int ransac_max_iters_ = 0;   // 0 == the reference's own default
};

Matx33d to33(const Matx23f& m) {
    Matx33d h = Matx33d::eye();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j) h(i, j) = static_cast<double>(m(i, j));
    return h;
}

// `_compose(new, acc)`: `new` applied after `acc`, in homogeneous form, float64.
Matx23f compose(const Matx23f& nw, const Matx23f& ac) {
    const Matx33d A = to33(nw);
    const Matx33d B = to33(ac);
    Matx33d H;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += A(i, k) * B(k, j);
            H(i, j) = s;
        }
    }
    const double n = H(2, 2);
    if (!std::isfinite(n) || std::fabs(n) < 1e-12) return kIdentity;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) H(i, j) /= n;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (!std::isfinite(H(i, j))) return kIdentity;
    Matx23f out;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j) out(i, j) = static_cast<float>(H(i, j));
    return out;
}

bool all_finite(const Matx23f& m) {
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            if (!std::isfinite(m(i, j))) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Fit plausibility gate.
//
// WHY THIS EXISTS (measured, not theoretical)
//   On 龙泉山 business-domain data the GMC fit produces a FINITE but physically
//   absurd transform on some frames: translation up to 4863 px on a 640x512
//   frame. all_finite() passes those. The damage is downstream and silent:
//   required_pad() faithfully reports ~4863, FusedGpu::configure() is grow-only,
//   the device image becomes 10366 x 10238 = 106 Mpixel per slot, and 45 slots
//   are ~4.8 GB -- which the board's global OOM killer takes as
//   file-rss:7623308kB while host anon-rss is only 60 MB. The process dies with
//   no assertion, no error, no non-zero rc from the pipeline's own logic.
//
//   The fit itself is NOT diverging: measured scale sqrt(det) stayed in
//   [0.995, 1.004] and lag-2 direct fits stayed at 1.4..5.7 px on the same
//   sequence. A few frames hand back an outlier, and the outliers -- not the
//   typical case -- are what kill the run.
//
// WHY IDENTITY, NOT A CLAMP
//   The alternatives were considered and rejected. Clamping the translation to
//   the cap still yields a wildly wrong warp; better to emit W(.) = IDENTITY for
//   that one frame, which is exactly the control arm the A/B harness already
//   validates, and let the temporal median downstream absorb the single bad
//   frame. A rejected fit is a LOUD, counted, reportable event --
//   `stats_.n_fit_rejected` exists for precisely this -- rather than an
//   unbounded allocation.
//
//   The bound is one frame width: measured legitimate motion on this corpus
//   peaks near 76 px on a 640 px frame, while the observed outliers are 3551 and
//   4863. kDefaultMaxShiftFrac of 1.0 leaves ~8x headroom over real motion and
//   still caps the device image at ~3.4 Mpixel/slot (~155 MB for 45 slots).
// ---------------------------------------------------------------------------
struct FitGate {
    bool ok = true;
    const char* why = "";
};

// Fraction of frame width a transform may translate before it is treated as an
// outlier. 1.0 == "cannot move further than the frame is wide".
static constexpr double kDefaultMaxShiftFrac = 1.0;
static constexpr double kDefaultMinScale = 0.5;
static constexpr double kDefaultMaxScale = 2.0;

FitGate check_fit(const Matx23f& m, int W, int H, double max_shift_frac,
                  double min_scale, double max_scale) {
    if (!all_finite(m)) return {false, "non-finite"};
    const double a = m(0, 0), b = m(0, 1), c = m(1, 0), d = m(1, 1);
    const double det = a * d - b * c;
    const double scale = std::sqrt(std::fabs(det));
    if (!(scale >= min_scale && scale <= max_scale))
        return {false, "scale"};
    const double tx = m(0, 2), ty = m(1, 2);
    const double lim = max_shift_frac * static_cast<double>(std::max(W, H));
    if (!(std::fabs(tx) <= lim && std::fabs(ty) <= lim)) return {false, "shift"};

    // Translation and scale are NOT independent for the thing that actually
    // matters. required_pad() measures where the transform sends the FRAME
    // CORNERS, and that is scale * extent + translation -- so a transform with
    // scale 2.0 and translation 640 px on a 640 px frame passes both independent
    // bounds while mapping a corner to ~1980 px, which demands ~1340 px of pad.
    // Measured: four frames reached required_pad 1032..1463 and tripped the device
    // cap while every constituent fit had passed the gate. Independent bounds are
    // therefore not sufficient; bound the corner extent itself.
    //
    // Affine extrema over a rectangle are attained at the corners, so 4
    // evaluations are exact -- the same argument required_pad() already relies on.
    double worst = 0.0;
    for (int cx = 0; cx < 2; ++cx) {
        for (int cy = 0; cy < 2; ++cy) {
            const double X = cx ? W : 0.0;
            const double Y = cy ? H : 0.0;
            const double sx = a * X + b * Y + tx;
            const double sy = c * X + d * Y + ty;
            worst = std::max(worst, std::max(std::fabs(sx - X), std::fabs(sy - Y)));
        }
    }
    // Allow the same headroom the translation bound allows, measured on the
    // displacement rather than the absolute coordinate.
    if (!(worst <= lim)) return {false, "extent"};
    return {true, ""};
}

// anchor_grid(stride_step, anchor_step, max_lag)
std::vector<int> anchor_grid(int stride_step, int anchor_step, int max_lag) {
    if (anchor_step % stride_step != 0) {
        std::fprintf(stderr,
                     "[FATAL] anchor_step=%d must be a multiple of stride_step=%d so the chain "
                     "recursion lands exactly on the anchors\n",
                     anchor_step, stride_step);
        std::exit(2);
    }
    std::set<int> s;
    for (int lag = stride_step; lag <= max_lag; lag += anchor_step) s.insert(lag);
    s.insert(max_lag);
    std::vector<int> out;
    for (int lag : s)
        if (lag % stride_step == 0) out.push_back(lag);
    return out;
}

// -----------------------------------------------------------------------------
// OnlineFeaturePipeline

// -----------------------------------------------------------------------------
// FusedGpu -- device-side ring + the single fused pass.
//
// Memory model. The kernel takes 22 image arguments (21 history + current) and
// each one is a pointer into a ring indexed by ABSOLUTE frame index modulo
// ring_depth. At push t the live set is {t, t-2, t-4, ..., t-42}; those 22
// indices are pairwise distinct modulo (max_lag+1) = 43, so 43 slots suffice
// and exactly ONE new frame crosses the bus per push (256 KB at 512x512). Using
// 21 slots indexed by lag instead would force all 21 history slices to be
// rewritten every frame -- 5.5 MB/frame -- which on RK3588 costs more than the
// arithmetic this kernel exists to save.
// -----------------------------------------------------------------------------
class FusedGpu {
public:
    struct DevPick {
        cl_platform_id plat = nullptr;
        cl_device_id dev = nullptr;
    };
    static DevPick pick() {
        DevPick d;
        d.dev = ocl::pick_device(&d.plat);
        return d;
    }

    FusedGpu(const std::string& kernel_dir, bool hw_linear, bool fp64_coord, int slots)
        : dev_(pick()),
          ctx_(dev_.dev),
          samp_(ocl::make_sampler(ctx_.get(), hw_linear)),
          slots_(slots),
          hw_linear_(hw_linear),
          slot_key_(static_cast<size_t>(slots), kNoKey) {
        ocl::describe(dev_.dev, dev_.plat);
        const std::string ksrc = ocl::read_file(kernel_dir + "/warp_median_fused.cl");
        const std::string isrc = ocl::read_file(kernel_dir + "/sortnet_generated.inc");
        prog_ = ocl::build_program(ctx_.get(), dev_.dev, ocl::splice_sortnet(ksrc, isrc), "fused",
                                   hw_linear, fp64_coord);
        ocl::pick_gray_format(ctx_.get(), &fmt_, &chan_);
        cl_int e;
        k_.reset(clCreateKernel(prog_.get(), "warp_median_fused", &e));
        OCL_CHECK(e, "clCreateKernel(warp_median_fused)");
        std::printf("[GPU] kernel built | sampling=%s | fp64_coord=%d | ring slots=%d\n",
                    hw_linear_ ? "CLK_FILTER_LINEAR" : "CLK_FILTER_NEAREST+manual2x2",
                    fp64_coord ? 1 : 0, slots_);
    }

    int W() const { return W_; }
    int H() const { return H_; }
    int pad() const { return pad_; }
    // The extent the device images are actually sized for. Equals pad() unless
    // pre-allocation reserved more than the current transforms need.
    int alloc_pad() const { return alloc_pad_; }
    const char* channel_note() const { return chan_ == 4 ? "CL_RGBA" : "CL_R8/CL_R"; }

    // Track the CURRENT pad, not the historical maximum -- but QUANTISED.
    //
    // It used to be grow-only: "allocate at the largest pad seen so far and only
    // grow", because the pad swings with the warp and reallocating every frame
    // would be absurd. That reasoning is right about CHURN and wrong about
    // MAGNITUDE. On 龙泉山 the camera pans continuously, so the historical max is
    // reached early and never released: measured devpad 2 -> 313 -> 3551 -> 4863
    // across one sequence, turning a 644x516 image (0.18 ms to upload) into
    // 7742x7614 (42 ms) and finally into a 4.8 GB allocation the OOM killer ends.
    //
    // required_pad() is exact for the transforms it is handed (affine extrema are
    // attained at the corners), which is what makes shrinking safe: the new pad is
    // always sufficient for the current warp.
    //
    // WHY GROW-ONLY AGAIN, BUT NOT THE ORIGINAL GROW-ONLY
    //
    // The original bug was never "grow-only". It was grow-only with NO UPPER BOUND
    // on the transform: required_pad tracked a finite outlier of 4863 px on a
    // 640x512 frame, the image became 10366x10238 = 106 Mpixel per slot, and 45
    // slots were ~4.8 GB -- which the board's OOM killer took as file-rss with
    // host anon-rss still at 60 MB.
    //
    // The bound now comes from the fit plausibility gate, which caps translation
    // at 1.0 frame width. That makes pad <= ~1.0 W, i.e. at most ~(2W)x(2H) =
    // 3.4 Mpixel per slot and ~155 MB for all 45 slots -- bounded by construction,
    // without ever shrinking.
    //
    // SHRINKING WAS TRIED AND MEASURED, TWICE, AND BOTH TIMES IT WAS WORSE:
    //   * multiplicative band [0.7x, 1.4x]  -> 60.9 ms upload spikes, 41 ms residual
    //   * quantised + shrink-at-half        -> 33..400 reallocations per 400 frames
    // The reason is mechanical: required_pad() is the max over 21 lags of the
    // CURRENT transforms, so it fluctuates. Any scheme that lets the buffer shrink
    // gives that fluctuation somewhere to oscillate, and every oscillation
    // reallocates all 45 device images. On a panning gimbal the requirement is
    // monotonically non-decreasing in practice, so growing only costs at most one
    // reallocation per power-of-two boundary (~6 per sequence) and cannot thrash.
    //
    // So: grow-only, quantised to powers of two so the growth is stepped rather
    // than continuous, bounded from above by the fit gate rather than by luck.
    static int quantise_pad(int p) {
        if (p < kPadQuantum) return kPadQuantum;
        if (p >= kMaxPadDevice) return kMaxPadDevice;
        int q = kPadQuantum;
        while (q < p && q < kMaxPadDevice) q <<= 1;
        return q;
    }

    void configure(int w, int h, int pad) {
        if (pad > kMaxPadDevice)
            std::fprintf(stderr,
                         "[WARN] required_pad=%d exceeds the %d device cap. The fit "
                         "gate is not holding; output beyond the capped region is "
                         "undefined.\n", pad, kMaxPadDevice);
        const int need = std::min(pad, kMaxPadDevice);
        int bucket = quantise_pad(need);

        // ------------------------------------------------------------------------
        // PAD PINNING -- the missing half of pre-allocation, added 2026-10-12.
        //
        // Pre-allocation removes the allocation EVENT. It does NOT remove the slot
        // INVALIDATION event, and on the board that turned out to be the larger of
        // the two. Measured with --prealloc-pad 1024 on RK3588, VIDEO00034 still
        // breached: devrealloc fell 5 -> 1, yet max only fell 53.50 -> 43.16 ms and
        // the worst frame was still frame 169. Cause, from the forensics:
        //
        //   img_up p50 1.56 / p99 16.47 / max 31.06, and the 2..10 ms band is EMPTY
        //   (296/300 frames under 2 ms, 2 frames over 20 ms). That is a discrete
        //   event, not a tail.
        //
        // The event is pad_ GROWTH, and it costs the same way a reallocation did:
        // pad_ is the border copyMakeBorder adds AND the content offset the kernel
        // samples at, so the moment it grows, all resident slots sit at the wrong
        // offset and every one of them must be dropped and re-uploaded -- see the
        // note above. That path returns early and never touches n_reconfigure_, so
        // `devrealloc == 1` proved the allocation was stable and told us nothing
        // about the ring.
        //
        // Pinning pad_ to the pre-allocation cap removes the event at its source:
        // pad_ starts at the cap and can never grow, so pad_grew is never true
        // after the first configure() and the ring is never invalidated.
        //
        // COST, and the design note above guessed it wrong. That note said uploading
        // at the allocation would "scale H2D by cap/bucket (32x at cap 1024 vs
        // bucket 32) and trade a 41 ms spike for a permanent tax". The board says
        // otherwise, and the board is the only thing that can price a bus:
        //
        //   VIDEO00002 (pad 32,   0.405 Mpixel)  img_up p50 0.37 ms
        //   VIDEO00034 (pad 512,  2.555 Mpixel)  img_up p50 1.56 ms
        //   VIDEO00028 (pad 1024, 6.881 Mpixel)  img_up p50 2.45 ms   <- measured
        //
        // 17x the bytes for 6.6x the time: the H2D path is bandwidth-saturated and
        // dominated by fixed cost, so the "permanent tax" is ~2 ms, not ~32x.
        // Pinning costs under 2.5 ms of p50 on the three sequences that were below
        // the cap and removes a 43 ms breach. Memory is unchanged -- the images are
        // ALREADY allocated at the cap; this only stops uploading less than we own.
        //
        // Output must be bit-identical: the kernel's offset argument is pad_, and
        // with a larger border the warp window is further inside the same
        // allocation, which is what the kernel has always been told to expect.
        // Verified by the two-arm zero-regression protocol, not by argument.
        // ------------------------------------------------------------------------
        if (pin_pad_ && prealloc_cap_ > 0)
            bucket = std::max(bucket, quantise_pad(prealloc_cap_));

        // The ALLOCATION extent is now decoupled from the pad actually used.
        //
        // pad_ keeps its old meaning and its old monotonicity: it is the border
        // copyMakeBorder adds and the content offset the kernel samples at. It
        // must never shrink, because cached slots hold frames uploaded at the
        // pad_ that was current when they were uploaded -- lowering pad_ would
        // make the kernel read a cached frame at the wrong offset. That is
        // unchanged from before.
        //
        // What pre-allocation removes is the allocation EVENT. With a cap set,
        // every device image is created once at (W + 2*cap) x (H + 2*cap); pad_
        // then climbs to the largest quantised requirement seen and can never
        // exceed the cap. Measured cost of the events this deletes: VIDEO00034
        // reallocated 5 times per 400 frames and the worst one cost 41.6 ms of
        // img_up (frame 169, total 53.50 ms) -- the only frame in 2000 that
        // breached 40 ms for a reason unrelated to fit cost.
        //
        // The upload region stays the PADDED size, never the allocated size --
        // unless --pin-pad is set, which is measured, not estimated. An earlier
        // version of this comment rejected uploading at the allocation on the
        // grounds that it "would scale H2D by cap/bucket (32x at cap 1024 vs
        // bucket 32) and trade a 41 ms spike for a permanent tax". The board
        // refuted the 32x, because the H2D path saturates and is dominated by
        // fixed cost:
        //
        //   VIDEO00002 pad 32   (0.405 Mpixel)  img_up p50 0.37 ms
        //   VIDEO00034 pad 512  (2.555 Mpixel)  img_up p50 1.56 ms
        //   VIDEO00028 pad 1024 (6.881 Mpixel)  img_up p50 2.45 ms
        //
        // 17x the bytes for 6.6x the time. The "permanent tax" is ~2 ms, and it
        // buys 28.81 ms off VIDEO00034's max. Only the board can price a bus.
        // See the PAD PINNING block in configure() and report section 5A.5.
        // to the kernel on every dispatch, in launch().
        const int want_alloc = (prealloc_cap_ > 0)
                                   ? std::max(bucket, quantise_pad(prealloc_cap_))
                                   : bucket;
        // Every resident slot holds a frame uploaded at the pad_ that was current
        // when it was uploaded, and the kernel locates content at offset pad_. So
        // pad_ is shared state: the moment it grows, the resident frames are at
        // the WRONG offset and must be dropped and re-uploaded, even when the
        // allocation itself does not change.
        //
        // Getting this wrong is invisible to the CPU arm (it never touches the
        // device images, so its MD5 is unchanged) and shows up only as a degraded
        // GPU-vs-CPU A/B score. It was exactly this bug when pre-allocation was
        // first added: the images were sized at pad 1024 from the start, so the
        // pad grew 32 -> 256 without a reallocation, the slots were kept, and
        // VIDEO00028's Ch0 exact-match fell from 46.55% to 12.86%.
        const bool pad_grew = bucket > pad_;
        if (w == W_ && h == H_ && want_alloc <= alloc_pad_) {
            if (pad_grew) {
                pad_ = bucket;
                std::fill(slot_key_.begin(), slot_key_.end(), kNoKey);
                key_to_slot_.clear();
            }
            return;                                     // sufficient: keep it
        }
        ++n_reconfigure_;
        W_ = w; H_ = h;
        if (pad_grew) pad_ = bucket;                    // monotonic; never shrinks
        alloc_pad_ = want_alloc;
        std::fill(slot_key_.begin(), slot_key_.end(), kNoKey);   // images are new
        key_to_slot_.clear();
        imgs_.clear();
        for (int i = 0; i < slots_; ++i)
            imgs_.emplace_back(ocl::make_image2d(ctx_.get(), fmt_,
                                                  (size_t)W_ + 2 * (size_t)alloc_pad_,
                                                  (size_t)H_ + 2 * (size_t)alloc_pad_));
        cl_int e;
        if (!mats_buf_) {
            // No CL_MEM_COPY_HOST_PTR: the 21 matrices are re-uploaded every push,
            // and pairing the flag with a null host_ptr is CL_INVALID_HOST_PTR.
            mats_buf_.reset(clCreateBuffer(ctx_.get(), CL_MEM_READ_ONLY,
                                           (size_t)kWindow * 6 * sizeof(float), nullptr, &e));
            OCL_CHECK(e, "clCreateBuffer(mats)");
            out_buf_.reset(clCreateBuffer(ctx_.get(), CL_MEM_WRITE_ONLY,
                                          (size_t)3 * W_ * H_, nullptr, &e));
            OCL_CHECK(e, "clCreateBuffer(out)");
        }
    }
    // Number of times the device images were reallocated. A non-trivial count is
    // the cost of shrinking made visible, so it is reported rather than hidden.
    long reconfigures() const { return n_reconfigure_; }

    // Returns the device slot holding `gray`, uploading only if it is not already
    // resident.
    //
    // Slots are keyed by CONTENT IDENTITY, never by `key % N`. A modulo ring looks
    // equivalent but is not: during cold start frame_at_lag() clamps every lag
    // that reaches past the first frame onto first_, so the CURRENT frame and a
    // CLAMPED HISTORY frame can share a residue. The modulo version then sees
    // "slot already holds key K" and skips the upload, leaving the previous
    // frame's content in place -- 31% of Ch0 pixels wrong, off by up to 80, and
    // it disappears after warm-up. Keying by identity makes that unrepresentable.
    size_t put(long key, const Mat& gray) {
        auto it = key_to_slot_.find(key);
        if (it != key_to_slot_.end()) return it->second;   // already resident
        size_t slot;
        auto free_it = std::find(slot_key_.begin(), slot_key_.end(), kNoKey);
        if (free_it != slot_key_.end()) {
            slot = (size_t)(free_it - slot_key_.begin());
        } else {
            // Evict the oldest key. Live keys span [t-max_lag, t] plus the one
            // clamped first-frame key, so slots_ = max_lag + 3 never gets here.
            slot = 0;
            for (size_t i = 1; i < slot_key_.size(); ++i)
                if (slot_key_[i] < slot_key_[slot]) slot = i;
            key_to_slot_.erase(slot_key_[slot]);
        }
        cv::Mat padded;
        cv::copyMakeBorder(gray, padded, pad_, pad_, pad_, pad_, cv::BORDER_REFLECT);
        if (chan_ == 4) {
            cv::Mat wide;
            cv::cvtColor(padded, wide, cv::COLOR_GRAY2RGBA);
            padded = wide;
        }
        const size_t o[3] = {0, 0, 0};
        const size_t r[3] = {(size_t)padded.cols, (size_t)padded.rows, 1};
        OCL_CHECK(clEnqueueWriteImage(ctx_.queue(), imgs_[slot].get(), CL_TRUE, o, r, 0, 0,
                                      padded.data, 0, nullptr, nullptr),
                  "clEnqueueWriteImage");
        slot_key_[slot] = key;
        key_to_slot_[key] = slot;
        return slot;
    }

    // `slots_for_lag[k]` is the device slot already holding the frame the kernel
    // must read for lag index k (0 = lag 2, 20 = lag 42); `cur_slot` holds I_t.
    void launch(const std::vector<cl_mem>& slots_for_lag, cl_mem cur_slot) {
        cl_int e;
        for (size_t k = 0; k < slots_for_lag.size(); ++k)
            OCL_CHECK(clSetKernelArg(k_.get(), (cl_uint)k, sizeof(cl_mem), &slots_for_lag[k]),
                      "clSetKernelArg(hist)");
        OCL_CHECK(clSetKernelArg(k_.get(), kWindow, sizeof(cl_mem), &cur_slot), "clSetKernelArg(cur)");
        cl_mem mb = mats_buf_.get();
        OCL_CHECK(clSetKernelArg(k_.get(), kWindow + 1, sizeof(cl_mem), &mb), "clSetKernelArg(mats)");
        cl_mem ob = out_buf_.get();
        OCL_CHECK(clSetKernelArg(k_.get(), kWindow + 2, sizeof(cl_mem), &ob), "clSetKernelArg(out)");
        OCL_CHECK(clSetKernelArg(k_.get(), kWindow + 3, sizeof(int), &W_), "clSetKernelArg(W)");
        OCL_CHECK(clSetKernelArg(k_.get(), kWindow + 4, sizeof(int), &H_), "clSetKernelArg(H)");
        OCL_CHECK(clSetKernelArg(k_.get(), kWindow + 5, sizeof(int), &pad_), "clSetKernelArg(pad)");
        cl_sampler sm = samp_.get();
        OCL_CHECK(clSetKernelArg(k_.get(), kWindow + 6, sizeof(cl_sampler), &sm), "clSetKernelArg(samp)");
        (void)e;
        // Profiling event. clEnqueueNDRangeKernel returns as soon as the command is
        // QUEUED, so wall-clock around the call measures nothing; the actual device
        // work is charged to whatever blocking call comes next. Without this the
        // kernel reports 0.04 ms while the readback reports 116 ms, which is the
        // compute hiding inside the synchronisation. Querying
        // CL_PROFILING_COMMAND_START/END is the only portable way to split them.
        // Timing, and WHY it is done this way.
        //
        // clEnqueueNDRangeKernel returns once the command is QUEUED, so a
        // wall-clock timer around it measures the enqueue, not the work: on this
        // stack the kernel "took" 0.04 ms while the following blocking readback
        // reported 116 ms -- the compute was hiding inside the synchronisation.
        //
        // CL_PROFILING_COMMAND_START/END is the textbook answer but is unusable
        // on this stack: it requires CL_QUEUE_PROFILING_ENABLE, which
        // clCreateCommandQueueWithProperties here rejects with CL_INVALID_VALUE,
        // and clCreateEvent is not even declared in this distro's cl.h. So:
        // enqueue, then clFinish, and attribute the elapsed time to the kernel.
        // The enqueue is microseconds and download() runs afterwards, so the split
        // between compute and transfer stays honest.
        const size_t g[2] = {(size_t)W_, (size_t)H_};
        const auto k0 = std::chrono::steady_clock::now();
        OCL_CHECK(clEnqueueNDRangeKernel(ctx_.queue(), k_.get(), 2, nullptr, g, nullptr, 0, nullptr,
                                         nullptr),
                  "clEnqueueNDRangeKernel");
        OCL_CHECK(clFinish(ctx_.queue()), "clFinish");
        last_kernel_ns_ = std::chrono::duration<double, std::nano>(
                              std::chrono::steady_clock::now() - k0)
                              .count();
    }

    double last_kernel_ns() const { return last_kernel_ns_; }

    void upload_mats(const float* m21) {
        OCL_CHECK(clEnqueueWriteBuffer(ctx_.queue(), mats_buf_.get(), CL_TRUE, 0,
                                       (size_t)kWindow * 6 * sizeof(float), m21, 0, nullptr, nullptr),
                  "clEnqueueWriteBuffer(mats)");
    }

    void download(Mat& out) {   // out is 3 x (H*W) CV_8UC1, matching gmc_stream.cpp's layout
        const size_t plane = (size_t)W_ * (size_t)H_;
        out.create(3, (int)plane, CV_8UC1);
        for (int c = 0; c < 3; ++c)
            OCL_CHECK(clEnqueueReadBuffer(ctx_.queue(), out_buf_.get(), CL_TRUE, (size_t)c * plane,
                                          plane, out.ptr<uint8_t>(c), 0, nullptr, nullptr),
                      "clEnqueueReadBuffer");
    }

    cl_mem slot(size_t i) const { return imgs_[i].get(); }

    long device_bytes() const {
        return (long)imgs_.size() * (long)(W_ + 2 * alloc_pad_) * (long)(H_ + 2 * alloc_pad_) * chan_;
    }

    // Reserve room for `cap` px of padding up front so the frame loop never
    // creates or destroys a device image. 0 (the default) disables it and
    // restores the grow-only-on-demand behaviour exactly.
    void set_prealloc_cap(int cap) { prealloc_cap_ = cap < 0 ? 0 : cap; }
    // Pin pad_ to the cap instead of letting it climb. Off by default so that
    // every previously published configuration keeps its exact previous meaning;
    // the zero-regression run compares it against that, not against itself.
    void set_pin_pad(bool on) { pin_pad_ = on; }

private:
    static constexpr int kWindow = 21;
    DevPick dev_;
    ocl::Context ctx_;
    ocl::Program prog_;
    ocl::Kernel k_;
    ocl::Sampler samp_;
    ocl::Mem mats_buf_, out_buf_;
    std::vector<ocl::Mem> imgs_;
    cl_image_format fmt_{};
    int chan_ = 1;
    int W_ = 0, H_ = 0, pad_ = 0;
    // Extent the device images are sized for. Equal to pad_ unless pre-allocation
    // reserved a larger cap, in which case pad_ climbs inside it and no
    // reallocation can occur.
    int alloc_pad_ = 0, prealloc_cap_ = 0;
    bool pin_pad_ = false;
    int slots_ = 43;
    bool hw_linear_ = true;
    std::vector<long> slot_key_;
    std::map<long, size_t> key_to_slot_;
    double last_kernel_ns_ = 0.0;
    static constexpr long kNoKey = -(1LL << 60);
    // Backstop only -- see configure(). 1024 px on a 640x512 frame is already
    // ~8x more than the corpus ever legitimately needs; the fit gate is what
    // actually bounds this in normal operation.
    static constexpr int kMaxPadDevice = 1024;
    static constexpr int kPadQuantum = 32;
    long n_reconfigure_ = 0;
};

// Pad needed so that every bilinear tap of every warp lands inside the padded
// frame. For an affine map over a rectangle the extremes are attained at the
// corners, so 4 evaluations per matrix replace a full sweep.
int required_pad(const std::vector<cv::Mat>& inv, int W, int H) {
    int p = 2;   // an identity warp still needs the 2x2 tap support inside
    for (const cv::Mat& m : inv) {
        for (int c = 0; c < 4; ++c) {
            const double X = (c & 1) ? W : 0.0;
            const double Y = (c & 2) ? H : 0.0;
            const double sx = m.at<double>(0, 0) * X + m.at<double>(0, 1) * Y + m.at<double>(0, 2);
            const double sy = m.at<double>(1, 0) * X + m.at<double>(1, 1) * Y + m.at<double>(1, 2);
            p = std::max(p, (int)std::ceil(std::max(-std::floor(sx), std::floor(sx) + 2.0 - W)));
            p = std::max(p, (int)std::ceil(std::max(-std::floor(sy), std::floor(sy) + 2.0 - H)));
        }
    }
    return p;
}

// -----------------------------------------------------------------------------
// Pin the calling thread to a cpu set. On RK3588 the four big cores are 4-7
// (Cortex-A76) and the little ones 0-3 (Cortex-A55, ~1.8 GHz vs ~2.4 GHz), and
// the default Linux scheduler is free to place OpenCV's worker threads on
// whichever it likes -- which makes run-to-run numbers drift. Pinning removes
// that as a variable instead of hoping the scheduler cooperates.
// Returns a short description for the run log, or an error string on failure.
std::string bind_cpus(const std::vector<int>& cpus) {
    if (cpus.empty()) return "none";
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus) CPU_SET(c, &set);
    const int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc != 0) return "FAILED(" + std::to_string(rc) + ")";
    std::string s;
    for (int c : cpus) {
        if (!s.empty()) s += ",";
        s += std::to_string(c);
    }
    return s;
}

std::vector<int> parse_cpu_list(const std::string& spec) {
    std::vector<int> out;
    if (spec == "none") return out;
    if (spec == "a76") return {4, 5, 6, 7};
    if (spec == "a55") return {0, 1, 2, 3};
    if (spec == "all") {
        const long n = sysconf(_SC_NPROCESSORS_ONLN);
        for (long i = 0; i < n; ++i) out.push_back(static_cast<int>(i));
        return out;
    }
    size_t i = 0;
    while (i <= spec.size()) {
        size_t j = spec.find(',', i);
        if (j == std::string::npos) j = spec.size();
        if (j > i) out.push_back(std::atoi(spec.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return out;
}

// ===========================================================================
// STAGE LEDGER for the channel-preparation (preprocess) path.
//
// WHY PER-FRAME AND NOT ONE AVERAGE
//   A single wall-clock total is not evidence on RK3588: it hides which of the
//   five physical stages owns the jitter, and an ownerless average cannot be
//   optimised or budgeted. Worse, it hides the stages that are NOT being timed
//   at all -- which is exactly the bug this ledger was written to catch (see the
//   `img_up` probe below).
//
// WHY THESE SIX STAGES
//   They are chosen so they TILE. Every timestamp pair is adjacent, so
//   sum(stages) == total by construction and no cost can hide in a seam:
//
//     1 prep      ring push + shape admission              (CPU)
//     2 fit       Shi-Tomasi + LK + RANSAC + chain compose (CPU A76 xN)
//     3 img_up    copyMakeBorder + clEnqueueWriteImage     (CPU -> bus -> GPU)
//     4 mats_up   21 affine matrices via WriteBuffer        (CPU -> bus -> GPU)
//     5 kernel    warpAffine(manual 2x2) + 21-frame median  (Mali-G610)
//     6 readback  3-plane clEnqueueReadBuffer              (GPU -> bus -> CPU)
//
//   Stage 3 existed in NO earlier accounting even though it moves one full
//   640x512 image per push; it was being silently charged to nothing. The
//   residual check in the report is what proves the tiling held on real data.
// ===========================================================================
struct FrameProfile {
    long idx = 0;           // absolute frame index (t_, 1-based)
    int seq_id = 0;         // which --sequence this row belongs to
    int seq_frame = 0;      // 0-based position WITHIN that sequence (for warm-up cuts)
    double io = 0.0;        // ms  decoding the frame from disk (stage 0)
    double prep = 0.0;      // ms
    double fit = 0.0;       // ms
    double warp = 0.0;      // ms  CPU arm only: 22x cv::warpAffine
    double median = 0.0;    // ms  CPU arm only: 21-frame median + residual
    double img_up = 0.0;    // ms
    double mats_up = 0.0;   // ms
    double kernel = 0.0;    // ms  (enqueue + clFinish)
    double readback = 0.0;  // ms
    double total = 0.0;     // ms  (measured prep-entry -> return, NOT the stage sum)
    double end_to_end = 0.0;// ms  (alias kept explicit; report checks stage sum vs this)
};

// One stage's distribution. Mean alone would hide a bimodal population, which is
// exactly what DVFS and scheduler preemption produce, so the tail is carried.
struct StageStats {
    double mean = 0, p50 = 0, p95 = 0, p99 = 0, max = 0, min = 0, sd = 0, sum = 0;
    long n = 0;
};

// Nearest-rank percentile over an already-sorted vector.
// Nearest rank, not interpolated: every number in the report must be a value some
// frame ACTUALLY took. An interpolated P99 is not an observation, and the whole
// point of a WCET column is that it is one.
static double pct_nearest_rank(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    const double n = static_cast<double>(sorted.size());
    long rank = static_cast<long>(std::ceil(q * n));
    if (rank < 1) rank = 1;
    if (rank > static_cast<long>(sorted.size())) rank = static_cast<long>(sorted.size());
    return sorted[static_cast<size_t>(rank - 1)];
}

static StageStats summarize(std::vector<double> v) {
    StageStats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.n = static_cast<long>(v.size());
    for (double x : v) s.sum += x;
    s.mean = s.sum / static_cast<double>(v.size());
    double acc = 0.0;
    for (double x : v) acc += (x - s.mean) * (x - s.mean);
    // Population SD. The run IS the population here -- we are not estimating a
    // larger process, so dividing by (n-1) would overstate spread slightly.
    s.sd = std::sqrt(acc / static_cast<double>(v.size()));
    s.p50 = pct_nearest_rank(v, 0.50);
    s.p95 = pct_nearest_rank(v, 0.95);
    s.p99 = pct_nearest_rank(v, 0.99);
    s.min = v.front();
    s.max = v.back();
    return s;
}

// ---------------------------------------------------------------------------
// A tiny fixed-size worker pool.
//
// WHY IT EXISTS
//   The anchor LK stage of one frame is split across a few workers.  The pool
//   is PERMANENT (created with the pipeline, destroyed with it): a per-frame
//   thread spawn would cost tens of microseconds per thread per frame -- small
//   against a 30 ms fit, but pointless, because the helpers have nothing to
//   do between frames but wait.
//
// THE CALLING THREAD PARTICIPATES
//   It takes work from the same counter as the helpers, so the split is
//   balanced by construction and does not depend on how many anchors there
//   are.  With 5 anchors and 5 workers every worker gets exactly one.
//
// THE WORK IS THE LK STAGE ONLY
//   Nothing here draws from OpenCV's global RNG, and nothing here writes
//   shared state -- that is what makes the pool safe for the bit-exactness
//   contract.  The RANSAC stage runs afterwards, in the caller, in the anchor
//   order.
// ---------------------------------------------------------------------------
class AnchorPool {
public:
    // `helpers` is the number of EXTRA threads; the calling thread is one of
    // the workers.  `cpus` is the core list the workers are spread over:
    // helper i takes cpus[(i+1) % cpus.size()], because cpus[0] belongs to
    // the calling thread.
    AnchorPool(int helpers, const std::vector<int>& cpus)
        : cpus_(cpus), helpers_(helpers < 0 ? 0 : helpers) {
        threads_.reserve(static_cast<size_t>(helpers_));
        for (int i = 0; i < helpers_; ++i)
            threads_.emplace_back([this, i] { helper_loop(i); });
    }
    ~AnchorPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            ++epoch_;
        }
        cv_.notify_all();
        for (std::thread& t : threads_)
            if (t.joinable()) t.join();
    }
    AnchorPool(const AnchorPool&) = delete;
    AnchorPool& operator=(const AnchorPool&) = delete;

    int helpers() const { return helpers_; }

    // Runs fn(i) for every i in [0, n).  Returns when all of them are done.
    void run(int n, const std::function<void(int)>& fn) {
        if (helpers_ <= 0 || n <= 1) {
            for (int i = 0; i < n; ++i) fn(i);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(m_);
            fn_ = &fn;
            n_ = n;
            next_.store(0, std::memory_order_relaxed);
            done_ = 0;
            ++epoch_;
        }
        cv_.notify_all();
        drain(&fn, n);
        std::unique_lock<std::mutex> lk(m_);
        cvd_.wait(lk, [this] { return done_ == helpers_; });
    }

private:
    void drain(const std::function<void(int)>* fn, int n) {
        for (;;) {
            const int i = next_.fetch_add(1, std::memory_order_relaxed);
            if (i >= n) break;
            (*fn)(i);
        }
    }

    void helper_loop(int tid) {
        if (!cpus_.empty()) {
            const int cpu = cpus_[static_cast<size_t>(tid + 1) % cpus_.size()];
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu, &set);
            if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0)
                std::fprintf(stderr,
                             "[WARN] anchor worker %d could not be pinned to core %d: %s\n", tid,
                             cpu, std::strerror(errno));
        }
        uint64_t seen = 0;
        for (;;) {
            const std::function<void(int)>* fn = nullptr;
            int n = 0;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return epoch_ != seen; });
                seen = epoch_;
                if (stop_) return;
                fn = fn_;
                n = n_;
            }
            drain(fn, n);
            {
                std::lock_guard<std::mutex> lk(m_);
                ++done_;
            }
            cvd_.notify_all();
        }
    }

    std::mutex m_;
    std::condition_variable cv_, cvd_;
    std::vector<std::thread> threads_;
    std::vector<int> cpus_;
    int helpers_ = 0;
    bool stop_ = false;
    uint64_t epoch_ = 0;
    const std::function<void(int)>* fn_ = nullptr;
    int n_ = 0;
    int done_ = 0;
    std::atomic<int> next_{0};
};

// The sentinel of the history ring's companion array (ring_id_): "this slot
// holds no frame".  -1 is safe: no frame index is ever negative.
static constexpr long kNoRingIndex = -1;

class OnlineFeaturePipeline {
public:
    struct Stats {
        double t_fit = 0, t_warp = 0, t_median = 0, t_total = 0;
        // The cost of COPYING the history window into the private snapshot.
        // It is a real per-frame cost, so it is measured instead of being
        // hidden inside the fit's residual.
        double t_snapshot = 0;
        long n_push = 0, n_fit = 0, n_warp = 0, n_compose = 0;
        // Fit-gate telemetry. A rejected fit is a real event: it is the difference
        // between "this frame got W(.) = IDENTITY" and "this frame blew the device
        // allocation up", so it must be counted and printed, never just handled.
        long n_fit_rejected = 0, n_fit_rejected_shift = 0, n_fit_rejected_other = 0;
        // GPU arm, segmented. upload / kernel / download are separated because on
        // RK3588 the ring's one-frame-per-push upload and the 3-plane readback are
        // bus traffic, not arithmetic, and they are what decides whether fusing
        // this tail is actually a win.
        double t_gpu = 0, t_gpu_upload = 0, t_gpu_kernel = 0, t_gpu_read = 0;
        double t_gpu_kernprof = 0;   // true device time from CL_PROFILING_COMMAND_*
    };

    enum Backend { kCpu, kGpu, kBoth };

    OnlineFeaturePipeline(int window, int stride_step, int anchor_step, int downscale,
                          bool rng_seed_per_fit, bool nogmc, FusedGpu* gpu, Backend backend,
                          bool pyr_cache, int max_corners, int fit_win, int fit_level,
                          int fit_iters, int cache_parts, int threads, int ring_slack = 0)
        : gpu_(gpu),
          backend_(backend),
          window_(window),
          stride_step_(stride_step),
          max_lag_(window * stride_step),
          rng_seed_per_fit_(rng_seed_per_fit),
          nogmc_(nogmc),
          ring_slack_(ring_slack < 0 ? 0 : ring_slack) {
        // 1 thread is the bit-exactness contract; >1 trades reproducibility for
        // throughput. RANSAC and the LK inner loop are not bit-reproducible
        // under a thread pool (see the floating-point contract in the header),
        // so any value above 1 MUST be reported alongside a transform deviation,
        // never as a bare speed number.
        cv::setNumThreads(threads > 0 ? threads : 1);
        anchors_ = anchor_grid(stride_step_, anchor_step, max_lag_);
        for (int a : anchors_) anchor_set_.insert(a);
        for (int lag = stride_step_; lag <= max_lag_; lag += stride_step_) {
            lags_.push_back(lag);
            if (anchor_set_.count(lag)) continue;
            int below = 0;
            for (int a : anchors_)
                if (a <= lag) below = a;
            anchor_below_[lag] = below;
        }
        // The cache ring must hold every frame one push can reference: the
        // deepest anchor lag, and the same for the one-step fits.
        est_.reset(new FastGMCEstimator(downscale, pyr_cache, max_lag_ + 1, max_corners,
                                        fit_win, fit_level, fit_iters, cache_parts));
        // The history ring is max_lag + 1 deep by default: that is exactly the
        // set of frames ONE push can reference, so the single-engine path
        // never needs a lookahead.
        //
        // `ring_slack` deepens it for the multi-engine case.  WHY: when several
        // engines share the history and each computes a DIFFERENT frame, the
        // producer of the history may run ahead of the engines.  Seeding frame
        // S overwrites the slot that held frame S - ring_depth, and the engine
        // that is about to compute frame J still needs frames J-42..J.  With
        // ring_depth == max_lag+1 the producer may not run ahead AT ALL; with
        // `ring_slack` extra slots it may run `ring_slack` frames ahead of the
        // oldest frame any engine still owes.  Without the slack the fit would
        // silently use OVERWRITTEN pixels -- wrong numbers, no crash.
        ring_.assign(static_cast<size_t>(max_lag_ + 1 + ring_slack_), Mat());
        // The companion array: WHICH frame index each ring slot currently
        // holds.  In the physical anchoring mode this is what turns a MISSING
        // frame into a DETECTED event instead of a silently wrong history
        // frame (see frame_at_lag).  It is maintained in BOTH modes -- one
        // store per seeded frame -- and only READ in the physical mode.
        ring_id_.assign(ring_.size(), kNoRingIndex);
        // max_lag + 1 keys can be live at once, plus the clamped first-frame key.
        gpu_slots_ = max_lag_ + 3;
        // The computation snapshot: one REFERENCE to every history frame a
        // SINGLE computation reads.  Slot 0 is the frame being computed (lag
        // 0); slot k is the frame at lag lags_[k-1].  ZERO COPY -- the slots
        // hold shallow Mats, so nothing is allocated or duplicated here; they
        // only pin the pixels for the duration of the computation.
        snap_.resize(lags_.size() + 1);
        snap_ok_.assign(snap_.size(), 0);
    }

    const std::vector<int>& anchors() const { return anchors_; }
    const std::vector<int>& lags() const { return lags_; }
    long ring_depth() const { return max_lag_ + 1 + ring_slack_; }

    // ---------------------------------------------------------------------
    // THE ANCHOR LK STAGE'S PARALLELISM.
    //
    // `threads` counts the WORKERS of the anchor LK stage, the calling
    // (engine) thread INCLUDED: 1 == the reference's serial anchor loop,
    // 5 == one worker per anchor.  The RANSAC stage is NOT affected: it
    // always runs afterwards, in this thread, in the anchor order, so the
    // global RNG is drawn from exactly as before and the output stays
    // bit-identical.
    //
    // `cpus` is the core list the workers are spread over.  Helper i takes
    // cpus[(i+1) % cpus.size()], because cpus[0] belongs to the calling
    // thread.  When the list is EMPTY the workers inherit the engine's
    // affinity -- and if the engine is pinned to ONE core, they all share
    // it and there is NO speedup.  That is not a bug, it is the whole
    // reason the caller must be given a GROUP of cores.
    // ---------------------------------------------------------------------
    void set_anchor_threads(int threads, const std::vector<int>& cpus) {
        anchor_threads_ = threads > 0 ? threads : 1;
        anchor_cpus_ = cpus;
        pool_.reset();
        if (anchor_threads_ > 1) {
            pool_.reset(new AnchorPool(anchor_threads_ - 1, anchor_cpus_));
            if (anchor_cpus_.empty()) {
                std::fprintf(stderr,
                             "[WARN] the anchor LK stage wants %d workers but no core list was "
                             "given: they inherit the engine's affinity, so a single-core engine "
                             "gets NO speedup from them\n",
                             anchor_threads_);
            }
        }
    }
    int anchor_threads() const { return anchor_threads_; }
    int anchor_workers() const { return pool_ ? anchor_threads_ : 1; }
    // Cache telemetry, so a timing claim can be checked against how much work
    // was actually avoided rather than assumed.
    long est_resize_builds() const { return est_->resize_builds(); }
    long est_corner_builds() const { return est_->corner_builds(); }
    long est_pyr_builds() const { return est_->pyr_builds(); }
    long est_cache_hits() const { return est_->cache_hits(); }
    long est_cache_lookups() const { return est_->cache_lookups(); }
    size_t est_cache_bytes() const { return est_->cache_bytes(); }
    // Gate configuration is a runtime value, not a constant: the bound is in
    // frame widths, so the same numbers mean different things on different
    // sensors, and tuning it must not require a rebuild.
    void set_fit_gate(double max_shift_frac, double min_scale, double max_scale) {
        gate_max_shift_frac_ = max_shift_frac;
        gate_min_scale_ = min_scale;
        gate_max_scale_ = max_scale;
    }
    Stats& stats() { return stats_; }
    const std::array<long, 3>& diff_max() const { return dmax_; }
    double diff_mae(int c) const { return dn_[c] ? dsum_[c] / (double)dn_[c] : 0.0; }
    double diff_exact_frac(int c) const {
        return dn_[c] ? (double)dexact_[c] / (double)dn_[c] : 1.0;
    }
    long diff_frames() const { return dframes_; }
    double diff_abs_sum(int c) const { return dsum_[c]; }
    long diff_pixels(int c) const { return dn_[c]; }
    long diff_exact_count(int c) const { return dexact_[c]; }
    int gpu_pad() const { return gpu_ ? gpu_->pad() : 0; }
    int gpu_alloc_pad() const { return gpu_ ? gpu_->alloc_pad() : 0; }
    long gpu_reconfigures() const { return gpu_ ? gpu_->reconfigures() : 0; }
    void gpu_prealloc(int cap) { if (gpu_) gpu_->set_prealloc_cap(cap); }
    void gpu_pin_pad(bool on) { if (gpu_) gpu_->set_pin_pad(on); }
    void set_corner_probe(bool on) { if (est_) est_->set_probe(on); }
    long corner_capped() const { return est_ ? est_->corner_capped() : 0; }
    long corner_dropped() const { return est_ ? est_->corner_dropped() : 0; }
    long corner_cap_frames() const { return n_corner_cap_frames_; }
    long corner_dets() const { return est_ ? est_->corner_dets() : 0; }
    double corner_mean() const { return est_ ? est_->corner_mean() : 0.0; }
    // RANSAC iteration cap.  0 = OFF: the reference call is made with its
    // own default arguments, which is what the bit-exactness contract
    // requires.  A positive value caps the sampling iterations, so a
    // pathological frame (a scene full of wild features) cannot spin: the
    // fit then returns the best transform found so far.  ENABLING IT CHANGES
    // THE NUMBERS, so it is off by default and must be reported as a
    // deviation when used.
    //
    // The solve itself lives in FastGMCEstimator, so the value is FORWARDED
    // to it: keeping a copy here without forwarding it would leave the knob
    // silently inert.
    void set_ransac_max_iters(int n) {
        ransac_max_iters_ = n > 0 ? n : 0;
        if (est_) est_->set_ransac_max_iters(ransac_max_iters_);
    }
    int ransac_max_iters() const { return ransac_max_iters_; }

    // ---------------------------------------------------------------------
    // THE PHYSICAL-FRAME-NUMBER ANCHOR (opt-in, OFF by default).
    //
    // WHAT IT CHANGES: the index the engine anchors its history on.  With it
    // OFF (the default) the index is the CONSUMPTION counter -- one per frame
    // the engine ingested -- which is what this engine has always used.  With
    // it ON the index is the camera's PHYSICAL frame number, the id the
    // producer assigned to the frame, so "lag 2" means "two PHYSICAL frames
    // back" and a frame that never arrived is a DETECTABLE hole instead of a
    // silently shifted window.
    //
    // WHY IT IS OFF BY DEFAULT: with no frame ever missing, the two
    // indexings are the same sequence shifted by a constant, so they select
    // the same pixels; but the switch exists so the new behaviour can be
    // validated on the board before it becomes the default.
    // ---------------------------------------------------------------------
    void set_phys_anchor(bool on) { phys_anchor_ = on; }
    bool phys_anchor() const { return phys_anchor_; }
    // A MISSING physical history frame, counted per LOOKUP (one frame can
    // miss several lags) and per FRAME (the number of frames whose history
    // had at least one hole).  Both stay 0 in the consumption-counter mode:
    // there the index sequence is dense by construction, so there is no hole
    // to detect -- which is exactly why the hole used to be silent.
    long n_history_gap() const { return n_history_gap_; }
    long n_history_gap_frames() const { return n_history_gap_frames_; }
    // Did the frame that was computed LAST have a hole in its history?
    bool history_gap() const { return gap_this_frame_; }
    // The history reads that were NOT served from the computation snapshot.
    //
    // WHY it is a first-class number: the snapshot is a SAFETY NET, not the
    // correctness argument.  A read that misses the snapshot falls back to the
    // ring's own path, so the numbers stay right even if the snapshot is
    // incomplete -- but then the snapshot is not doing its job, and the run
    // must say so instead of looking clean.  On a zero-drop sequence this
    // counter is 0; anything else means the snapshot's coverage is wrong.
    long n_snapshot_miss() const { return n_snapshot_miss_; }

    void set_fit_deadline(double ms, bool hold) {
        fit_deadline_ms_ = ms;
        fit_deadline_hold_ = hold;
    }
    long n_fit_deadline() const { return n_fit_deadline_; }
    // Composed lags rejected by the plausibility gate, counted separately
    // from rejected FITS: "one bad anchor" and "a bad chain" are different
    // failure modes and must not share one statistic.
    long n_chain_rejected() const { return n_chain_rejected_; }
    int width() const { return shape_w_; }
    int height() const { return shape_h_; }
    long gpu_bytes() const { return gpu_ ? gpu_->device_bytes() : 0; }

    // ---- stage ledger access -------------------------------------------------
    // profiling is opt-in and OFF by default: the per-frame vector grows without
    // bound and the probes themselves cost a clock read each, so a plain accuracy
    // run must not pay for a timing run.
    void set_profiling(bool on) { profiling_ = on; }
    bool profiling() const { return profiling_; }
    void set_seq_id(int id) { seq_id_ = id; }
    void set_seq_frame(int i) { seq_frame_ = i; }
    // Frame decode cost is paid by the caller, between pushes, and belongs in the
    // channel-prep budget: on the board the corpus lives on eMMC and decoding is
    // not free. Leaving it out would report a "end-to-end" number that silently
    // excludes a stage the delivery actually has to pay.
    void set_frame_io_ms(double ms) { io_ms_ = ms; }
    const std::vector<FrameProfile>& profiles() const { return prof_; }
    void clear_profiles() { prof_.clear(); }

    size_t state_bytes() const {
        size_t n = 0;
        for (const Mat& f : ring_)
            if (!f.empty()) n += static_cast<size_t>(f.total() * f.elemSize());
        n += steps_.size() * sizeof(Matx23f);
        // The computation snapshot: one private copy of every history frame
        // the current computation reads.  It is part of the engine's memory,
        // so it is part of the engine's memory REPORT.
        for (const Mat& f : snap_)
            if (!f.empty()) n += static_cast<size_t>(f.total() * f.elemSize());
        if (!snap_first_.empty())
            n += static_cast<size_t>(snap_first_.total() * snap_first_.elemSize());
        return n;
    }

    // ---------------------------------------------------------------------
    // Two-step frame API.
    //
    //   seed_frame(abs, frame)   store the frame in the history ring
    //   process_frame(abs, ...)  compute the feature for the frame at `abs`
    //
    // WHY: the per-frame computation depends ONLY on pixels.  The anchor fits
    // compare the frame at `abs` against the frame at `abs-lag`; the other
    // lags are composed from those fits INSIDE the same frame; the GPU tail
    // consumes the same `mats` map.  Nothing reads another frame's COMPUTED
    // result.  So several engines can be fed the SAME frames and each of them
    // can compute a DIFFERENT frame -- that is how the feature stage runs on
    // more than one core without touching the maths or the bit-exactness
    // contract (each engine still runs OpenCV single-threaded, so every fit is
    // the same computation the single-engine path performs).
    //
    // push() below is the original single-step API expressed as the two steps,
    // so the PC reference keeps its exact behaviour.
    // ---------------------------------------------------------------------
    void seed_frame(long abs, const Mat& frame) {
        if (frame.type() != CV_8UC1) {
            std::fprintf(stderr, "[FATAL] frames must be uint8 grayscale\n");
            std::exit(2);
        }
        if (shape_known_ && (frame.rows != shape_h_ || frame.cols != shape_w_)) {
            std::fprintf(stderr, "[FATAL] frame size changed: got (H,W)=(%d,%d), expected (%d,%d)\n",
                         frame.rows, frame.cols, shape_h_, shape_w_);
            std::exit(2);
        }
        shape_h_ = frame.rows;
        shape_w_ = frame.cols;
        shape_known_ = true;

        if (first_.empty()) {
            first_ = frame;
            // The index of the FIRST frame that was seeded: the cold-start
            // clamp, i.e. the frame the history falls back to when a lag
            // reaches past the start of the stream.
            first_idx_ = abs;
        }
        // NOTE: `t_` is NOT touched here.  `t_` is the frame the engine is
        // COMPUTING, and only the computing thread may change it: the
        // producer of the history may run ahead (multi-engine), and a write
        // here would move the history window under the running fit -- wrong
        // pixels, wrong transforms, no error.
        const size_t slot = static_cast<size_t>(((abs % ring_depth()) + ring_depth()) % ring_depth());
        ring_[slot] = frame;
        // WHICH frame this slot holds.  The companion array is what makes a
        // missing frame detectable: a lag looks its frame up by the index it
        // asks for, and if the slot's index is not that one, the frame is not
        // in the ring -- it never arrived.
        ring_id_[slot] = abs;
    }

    // Compute the feature for the frame at absolute index `abs`.
    // `mark_entry` is when the frame entered the stage: the ledger measures
    // from there, so the caller decides whether the queue wait counts.
    Mat process_frame(long abs, std::vector<Matx23f>* dump_mats,
                      std::chrono::steady_clock::time_point mark_entry) {
        // Every helper below (frame_at_lag, abs_at_lag, step) is relative to
        // the frame being computed, not to the newest frame that was seeded.
        t_ = abs;
        // The per-frame hole flag: the history of THIS frame starts clean.
        gap_this_frame_ = false;

        // The frame's own work starts HERE, so the snapshot copy below is
        // part of the frame's measured cost and not of the queue wait.
        const auto mark0 = std::chrono::steady_clock::now();

        // THE SNAPSHOT, BEFORE ANY FIT.  From here on this computation reads
        // its history from a private copy, so the producer of the ring cannot
        // change the pixels under it -- not by a tuned amount of lookahead,
        // but structurally.
        //
        // It is timed on its own because it is a REAL per-frame cost (one
        // memcpy of the whole history window) and a claim about the frame's
        // cost that left it out would be wrong.
        const auto mark_snap = std::chrono::steady_clock::now();
        build_snapshot();
        stats_.t_snapshot += std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - mark_snap).count();

        // The frame being computed IS the newest one in the ring (it was
        // seeded at `abs`), so it is the lag-0 frame.  Same pixels, same
        // Mat header as the `frame` argument the old push() took.
        const Mat& frame = frame_at_lag(0);

        // Open this frame's ledger row here, before any stage work, so the GPU tail
        // can fill its own stages into the SAME record (it is a separate function
        // with no return path back here).
        if (profiling_) {
            cur_ = FrameProfile();
            cur_.idx = t_;
            cur_.seq_id = seq_id_;
            cur_.seq_frame = seq_frame_;
            cur_.io = io_ms_;
            cur_.prep = std::chrono::duration<double, std::milli>(mark0 - mark_entry).count();
        }

        std::map<int, Matx23f> mats;
        if (!nogmc_) {
        // ---- anchors: direct fit on the anchor grid ----
        const auto mark_anchor = std::chrono::steady_clock::now();

        // Deadline guard: give up on the remaining anchors once this frame's fit
        // budget is spent. Off unless --fit-deadline-ms is set. When it fires the
        // whole frame degrades at once -- a half-fitted frame would leave
        // anchor_below() dangling and silently mix fresh and stale transforms.
        bool blown = false;
        if (pool_ && anchors_.size() > 1) {
            // -----------------------------------------------------------------
            // THE PARALLEL ANCHOR PATH: LK concurrent, RANSAC sequential.
            //
            // Phase 1 (serial): the cache work.  prepare_fit() is the only
            //   writer of the shared cache, and it walks the anchors in the
            //   reference's order, so the cache ends up in exactly the state
            //   the reference's loop left it in.
            // Phase 2 (parallel): the LK stage only.  It reads the snapshot
            //   and writes nothing -- no RNG, no counters, no buffers -- so
            //   the anchors may run it at once.
            // Phase 3 (serial, anchor order): the RANSAC stage and the gate.
            //   This is the only consumer of the global RNG, and it consumes
            //   it in the reference's order, so the draws -- and therefore
            //   the fits -- are the reference's.
            //
            // The deadline is checked in phase 3, before each RANSAC.  It can
            // no longer be checked BETWEEN the anchors' LKs (they run
            // concurrently), so a frame that used to blow after its second
            // anchor now blows after the whole LK phase.  The OUTCOME is the
            // same -- the whole frame degrades -- but WHICH frames blow
            // depends on the wall clock, as it always did.
            // -----------------------------------------------------------------
            if (plans_.size() != anchors_.size()) plans_.resize(anchors_.size());
            for (size_t i = 0; i < anchors_.size(); ++i) {
                const int lag = anchors_[i];
                est_->prepare_fit(abs_at_lag(lag), t_, frame_at_lag(lag), frame, &plans_[i]);
            }
            lks_.resize(anchors_.size());
            const std::function<void(int)> lk_job = [this](int i) {
                const size_t k = static_cast<size_t>(i);
                lks_[k] = est_->run_lk(plans_[k]);
            };
            pool_->run(static_cast<int>(anchors_.size()), lk_job);
            for (size_t i = 0; i < anchors_.size(); ++i) {
                if (fit_deadline_ms_ > 0.0) {
                    const double spent = std::chrono::duration<double, std::milli>(
                                             std::chrono::steady_clock::now() - mark_anchor)
                                             .count();
                    if (spent > fit_deadline_ms_) { blown = true; break; }
                }
                const int lag = anchors_[i];
                if (rng_seed_per_fit_) cv::theRNG().state = 0xFFFFFFFFu;
                mats[lag] = finish_fit(est_->ransac_stage(lks_[i]));
                stats_.n_fit += 1;
            }
        } else {
            // The reference's serial anchor loop, untouched.
            for (int lag : anchors_) {
                if (fit_deadline_ms_ > 0.0) {
                    const double spent = std::chrono::duration<double, std::milli>(
                                             std::chrono::steady_clock::now() - mark_anchor)
                                             .count();
                    if (spent > fit_deadline_ms_) { blown = true; break; }
                }
                mats[lag] = fit_similarity(abs_at_lag(lag), t_, frame_at_lag(lag), frame);
                stats_.n_fit += 1;
            }
        }
        if (blown) {
            ++n_fit_deadline_;
            // 'hold' reuses the last frame that was fitted SUCCESSFULLY -- not
            // literally the previous frame. When the budget is blown on every
            // single frame there is no successful frame to hold, so 'hold' falls
            // back to IDENTITY and collapses onto the other mode. That is the
            // intended safety direction (never propagate a degraded warp forward),
            // but it means the two modes are indistinguishable in a run where the
            // guard fires 100% of the time -- check fitdeadline before reading
            // anything into a hold-vs-identity md5 comparison.
            //
            // Both are wrong warps. The choice is a policy knob, not a
            // correctness claim, and the trigger count is reported either way.
            if (fit_deadline_hold_ && have_last_mats_ && last_mats_.size() == lags_.size()) {
                for (size_t k = 0; k < lags_.size(); ++k) mats[lags_[k]] = last_mats_[k];
            } else {
                for (int lag : lags_) mats[lag] = kIdentity;
            }
            const auto after_blown = std::chrono::steady_clock::now();
            stats_.t_fit +=
                std::chrono::duration<double, std::milli>(after_blown - mark_anchor).count();
            // Do NOT remember this frame as the "last good" one.
        } else {
        const auto after_anchor = std::chrono::steady_clock::now();
        stats_.t_fit += std::chrono::duration<double, std::milli>(after_anchor - mark_anchor).count();

        // ---- every other lag: bounded composition from the nearest anchor below ----
        for (int lag : lags_) {
            if (anchor_set_.count(lag)) continue;
            const int below = anchor_below_.at(lag);
            Matx23f acc = mats[below];
            for (int lag_abs = below + stride_step_; lag_abs <= lag; lag_abs += stride_step_) {
                acc = compose(step(t_ - lag_abs), acc);
                stats_.n_compose += 1;
            }
            mats[lag] = acc;
        }
        stats_.t_fit += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                  after_anchor)
                            .count();
        last_mats_.assign(lags_.size(), kIdentity);
        for (size_t k = 0; k < lags_.size(); ++k) {
            auto it = mats.find(lags_[k]);
            if (it != mats.end()) last_mats_[k] = it->second;
        }
        have_last_mats_ = true;
        if (dump_mats) {
            dump_mats->clear();
            for (const auto& kv : mats) dump_mats->push_back(kv.second);
        }
        }  // !blown
        // Distinct FRAMES on which the corner budget bound, as opposed to the
        // estimator's count of truncated DETECTIONS. Both are reported because
        // they answer different questions and conflating them is how a "triggered
        // 1236 times" line gets misread as "1236 bad frames": one frame can need
        // several corner sets (prev + curr per anchor), and a frame is what a
        // reviewer actually wants to count.
        if (est_) {
            const long cc = est_->corner_capped();
            if (cc > corner_capped_prev_) { ++n_corner_cap_frames_; corner_capped_prev_ = cc; }
        }
        }  // !nogmc_

        // ---- fused tail: warp + median + 3-channel assembly ----
        // The fits above were computed ONCE and both arms consume the same
        // `mats` map, so an A/B pass can never come from the two arms having
        // been fed different inputs.
        Mat out;
        const auto mark_warp = std::chrono::steady_clock::now();
        if (backend_ != kGpu) {
            // CPU arm: byte-for-byte the operations the validated gmc_stream.cpp
            // performs. In --fused cpu this block, and therefore the MD5 stream,
            // must be identical to that binary's.
            Mat ch1;
            std::vector<Mat> history;
            history.reserve(lags_.size());
            if (nogmc_) {
                // W(.) = IDENTITY: the no-GMC arm of the A/B harness. Zero fits, zero warps.
                cv::absdiff(frame, frame_at_lag(stride_step_), ch1);
                for (int lag : lags_) history.push_back(frame_at_lag(lag));
            } else {
                cv::absdiff(frame, est_->warp(frame_at_lag(stride_step_), mats[stride_step_]), ch1);
                for (int lag : lags_) history.push_back(est_->warp(frame_at_lag(lag), mats[lag]));
                stats_.n_warp += static_cast<long>(lags_.size()) + 1;
            }
            const auto after_warp = std::chrono::steady_clock::now();
            stats_.t_warp += std::chrono::duration<double, std::milli>(after_warp - mark_warp).count();
            if (profiling_)
                cur_.warp = std::chrono::duration<double, std::milli>(after_warp - mark_warp).count();

            // ---- median + residual ----
            const auto mark_med = std::chrono::steady_clock::now();
            Mat ch2;
            median_residual(frame, history, ch2);
            const auto after_med = std::chrono::steady_clock::now();
            stats_.t_median += std::chrono::duration<double, std::milli>(after_med - mark_med).count();
            if (profiling_)
                cur_.median = std::chrono::duration<double, std::milli>(after_med - mark_med).count();

            // Plane-major assembly: row p of a 3 x (H*W) Mat is exactly numpy's out[p].
            const size_t plane = static_cast<size_t>(frame.rows) * static_cast<size_t>(frame.cols);
            out = Mat(3, static_cast<int>(plane), CV_8UC1);
            std::memcpy(out.ptr(0), frame.data, plane);
            std::memcpy(out.ptr(1), ch1.data, plane);
            std::memcpy(out.ptr(2), ch2.data, plane);
        }

        if (backend_ != kCpu) {
            const auto g0 = std::chrono::steady_clock::now();
            gpu_fused_tail(frame, mats, out_gpu_);
            stats_.t_gpu +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g0).count();
            if (backend_ == kGpu) {
                out = out_gpu_;
            } else {
                diff_planes(out, out_gpu_);
            }
        }

        stats_.n_push += 1;
        const auto mark_done = std::chrono::steady_clock::now();
        stats_.t_total += std::chrono::duration<double, std::milli>(mark_done - mark0).count();

        if (profiling_) {
            using ms = std::chrono::duration<double, std::milli>;
            // `end_to_end` is MEASURED (prep entry -> here), never derived from the
            // stages. That independence is the point: the report recomputes
            // prep+fit+img_up+mats_up+kernel+readback and prints the gap against
            // this number, so an un-tiled stage cannot hide. A ledger that defines
            // total as its own sum would report a perfect residual by construction
            // and prove nothing.
            cur_.end_to_end = ms(mark_done - mark_entry).count() + io_ms_;
            // fit is the residual between the ring push and the fused tail, so ANY
            // work this binary does that is not separately attributed still lands in
            // `fit` instead of disappearing. Stages 1-6 therefore tile exactly.
            cur_.fit = ms(mark_warp - mark0).count();
            cur_.total = cur_.end_to_end;
            prof_.push_back(cur_);
        }

        for (auto it = steps_.begin(); it != steps_.end();) {
            if (it->first < t_ - max_lag_) it = steps_.erase(it);
            else ++it;
        }
        // The frame-level hole count: a frame whose history had at least one
        // missing physical frame.  Counted separately from the per-lookup
        // count, because "21 missing history frames" and "21 frames with a
        // hole in the history" are different questions.
        if (gap_this_frame_) ++n_history_gap_frames_;
        return out;
    }

    // Consume one grayscale frame; returns a 3 x (H*W) CV_8UC1 plane-major view of (3,H,W).
    // The original single-step API: seed the ring, then compute that frame.
    Mat push(const Mat& frame, std::vector<Matx23f>* dump_mats) {
        const auto mark_entry = std::chrono::steady_clock::now();
        const long abs = t_ + 1;
        seed_frame(abs, frame);
        return process_frame(abs, dump_mats, mark_entry);
    }

private:
    // ---------------------------------------------------------------------
    // GPU arm. One fused pass replaces 22 warpAffine calls + 21 full-frame Mats
    // + a per-pixel nth_element median + three memcpys.
    //
    // `mats` holds FORWARD transforms exactly as gmc_stream.cpp passes them to
    // cv::warpAffine, which inverts internally. We invert here too, in double,
    // because that inversion is part of the semantics being reproduced. Note
    // OpenCV 4.10's invertAffineTransform returns the OUTPUT in the INPUT's
    // element type, so reading it through at<double> without a convertTo strides
    // 8 bytes over a 4-byte layout and yields denormals.
    // ---------------------------------------------------------------------
    void gpu_fused_tail(const Mat& frame, const std::map<int, Matx23f>& mats, Mat& out) {
        const int W = frame.cols, H = frame.rows;

        // 1. the 21 transforms the kernel needs, inverted. In nogmc mode W(.) is
        //    IDENTITY, so synthesise it rather than skipping the GPU arm.
        std::vector<cv::Mat> inv;
        inv.reserve(lags_.size());
        std::array<float, 21 * 6> mf{};
        for (size_t k = 0; k < lags_.size(); ++k) {
            const int lag = lags_[k];
            Matx23f Hm = kIdentity;
            if (!nogmc_) {
                auto it = mats.find(lag);
                if (it != mats.end()) Hm = it->second;
                // Gate the COMPOSED transform too, not just the direct fits.
                //
                // fit_similarity() gates each fit, but 17 of the 21 lags are
                // composed from anchor fits by bounded chain composition, and a
                // composition of individually-plausible steps is not necessarily
                // plausible: measured required_pad of 1032..1463 px on a 640x512
                // frame even though every constituent fit passed the 1.0-frame
                // gate. Without this second gate those frames warp outside the
                // allocated image -- the device cap stops the OOM, but the output
                // there is undefined rather than merely degraded.
                //
                // So: same gate, same identity fallback, counted separately,
                // because "one bad anchor" and "a bad chain" are different
                // failure modes and should not share one statistic.
                const FitGate gc = check_fit(Hm, W, H, gate_max_shift_frac_,
                                             gate_min_scale_, gate_max_scale_);
                if (!gc.ok) {
                    n_chain_rejected_ += 1;
                    Hm = kIdentity;
                }
            }
            cv::Mat M32(2, 3, CV_32F);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 3; ++j) M32.at<float>(i, j) = Hm(i, j);
            cv::Mat Mi, Mi64;
            cv::invertAffineTransform(M32, Mi);
            Mi.convertTo(Mi64, CV_64F);

            // Gate the INVERSE, which is the matrix required_pad() actually
            // consumes and the kernel actually samples with.
            //
            // Gating only the forward transform is not enough, and the residual
            // case is instructive: a forward scale of 0.5 sits exactly on the
            // allowed floor, but its inverse is 2.0, and the inverse doubles the
            // corner extent -- so a transform that passes every forward check can
            // still demand ~1280 px of pad. Measured: two frames still tripped the
            // device cap after the forward corner-extent gate was added.
            //
            // Identity survives inverting (it is its own inverse), so rejecting
            // here degrades to W(.) = IDENTITY exactly as the forward gate does.
            Matx23f Inv;
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 3; ++j) Inv(i, j) = static_cast<float>(Mi64.at<double>(i, j));
            const FitGate gi = check_fit(Inv, W, H, gate_max_shift_frac_,
                                         gate_min_scale_, gate_max_scale_);
            if (!gi.ok) {
                n_chain_rejected_ += 1;
                cv::Mat Id = cv::Mat::eye(2, 3, CV_64F);
                Mi64 = Id;
            }
            inv.push_back(Mi64);
            for (int j = 0; j < 6; ++j) mf[k * 6 + j] = (float)Mi64.at<double>(j / 3, j % 3);
        }

        const int pad = required_pad(inv, W, H);

        // 2. resident frames. frame_at_lag() clamps any lag reaching past the
        //    first frame onto first_ (absolute index 1), so the effective index is
        //    max(1, t - lag) -- the same rule, not a guess.
        //
        // The stage-3 marker goes BEFORE configure() on purpose: configure() can
        // reallocate all 45 device images when the quantised pad bucket changes,
        // and that allocation belongs to the H2D side of the ledger. With the
        // marker after it, the reallocation showed up as an unattributed
        // 41 ms tiling residual -- invisible in every stage, visible only as a
        // residual, which is exactly how a defect hides.
        const auto p0 = std::chrono::steady_clock::now();
        gpu_->configure(W, H, pad);

        // Content key for a lag: the absolute index frame_at_lag() would use, or
        // kFirstKey when it clamps onto first_. t_ itself is the current frame's
        // key. These are identities, so two different frames can never collide.
        constexpr long kFirstKey = -1;

        // Stage 3 covers configure() + copyMakeBorder + clEnqueueWriteImage: every
        // CPU->device move the tail performs, including the occasional device-image
        // reallocation.
        std::vector<cl_mem> slots;
        slots.reserve(lags_.size());
        for (size_t k = 0; k < lags_.size(); ++k) {
            const long idx = t_ - lags_[k];
            const long key = (idx <= clamp_lo()) ? kFirstKey : idx;
            slots.push_back(gpu_->slot(gpu_->put(key, frame_at_lag(lags_[k]))));
        }
        const size_t cur_slot = gpu_->put(t_, frame);

        const auto u0 = std::chrono::steady_clock::now();
        gpu_->upload_mats(mf.data());
        const auto u1 = std::chrono::steady_clock::now();

        gpu_->launch(slots, gpu_->slot(cur_slot));
        const auto u2 = std::chrono::steady_clock::now();
        stats_.t_gpu_kernprof += (double)gpu_->last_kernel_ns() / 1.0e6;   // ns -> ms

        gpu_->download(out);
        const auto u3 = std::chrono::steady_clock::now();
        using ms = std::chrono::duration<double, std::milli>;
        // Legacy accumulators keep their original meaning (mats upload only) so the
        // already-published [SEQTIMING] numbers do not silently change meaning.
        stats_.t_gpu_upload += ms(u1 - u0).count();
        stats_.t_gpu_kernel += ms(u2 - u1).count();
        stats_.t_gpu_read += ms(u3 - u2).count();
        if (profiling_) {
            cur_.img_up = ms(u0 - p0).count();
            cur_.mats_up = ms(u1 - u0).count();
            cur_.kernel = ms(u2 - u1).count();
            cur_.readback = ms(u3 - u2).count();
        }
    }

    // Per-channel A/B between the CPU arm and the GPU arm of THIS push.
    void diff_planes(const Mat& a, const Mat& b) {
        const long plane = a.cols;
        for (int c = 0; c < 3; ++c) {
            const uint8_t* pa = a.ptr<uint8_t>(c);
            const uint8_t* pb = b.ptr<uint8_t>(c);
            long sum = 0, nexact = 0;
            for (long i = 0; i < plane; ++i) {
                const long t = std::labs((long)pa[i] - (long)pb[i]);
                if (t) sum += t; else ++nexact;
                if (t > dmax_[c]) dmax_[c] = t;
            }

            dsum_[c] += (double)sum;
            dn_[c] += plane;
            dexact_[c] += nexact;
        }
        ++dframes_;
    }

    // ---------------------------------------------------------------------
    // THE COMPUTATION SNAPSHOT -- THE TWO-LEVEL CACHE.
    //
    // WHY it exists: the fit reads the history WHILE it runs, and the
    // producer of the ring runs in ANOTHER thread.  Without a snapshot the
    // pixels a fit sees depend on WHEN it looks: the producer may overwrite
    // the slot between two reads of the SAME computation.  That is why the
    // ring is deepened by `ring_slack` -- a TUNED guarantee that the producer
    // stays `ring_slack` frames ahead of the oldest owed frame, which happens
    // to be just enough.  The snapshot makes the same guarantee STRUCTURAL:
    // the frames this computation will read are copied OUT of the ring
    // BEFORE the first fit, so what the producer does afterwards cannot reach
    // this computation at all.
    //
    // The set of frames is MEASURED, not guessed.  An instrumented run over a
    // 60-frame sequence (anchor-step 2 and 10, the CPU tail and the GPU tail,
    // with and without the physical anchor) showed that ONE computation reads
    // exactly
    //     {t} U {t-2, t-4, ..., t-42}
    // -- the frame being computed plus every lag of the median window -- and
    // nothing else.  The composed one-step fits stay inside that set: they ask
    // for the lags `lag_abs` and `lag_abs - stride_step`, and both are always
    // lags of the window.  The largest depth ever read is 42 (= max_lag_); a
    // 44 was never observed.
    //
    // The snapshot is PER INSTANCE and is written ONLY by the thread that is
    // computing (process_frame).  seed_frame -- the producer's -- never
    // touches it, so the two levels cannot race over it.
    // ---------------------------------------------------------------------

    // Take REFERENCES to the frames THIS computation will read out of the
    // ring.  Called once per frame, at the very top of process_frame(), before
    // any fit.  ZERO COPY: nothing is duplicated, the pixels stay where they
    // are and only the read right is pinned (see hold_snapshot_slot).
    void build_snapshot() {
        hold_snapshot_slot(0, t_);
        for (size_t i = 0; i < lags_.size(); ++i)
            hold_snapshot_slot(i + 1, t_ - lags_[i]);
        // THE COLD-START CLAMP TARGET.  When the deepest lag reaches past the
        // start of the stream, the read is served by the FIRST frame.  It is a
        // member of its own -- written once, with the first seeded frame, and
        // never changed again -- so a reference to it is taken on first use and
        // NEVER refreshed.  The snapshot's content is then EXACTLY the set of
        // physical frames this computation reads: the window's lags, the frame
        // being computed, and the clamp target.
        if (t_ - max_lag_ <= clamp_lo() && !snap_first_ok_ && !first_.empty()) {
            // ZERO COPY, like the slots above: a reference to the first
            // frame's pixels, which are written once and never changed.
            snap_first_ = first_;
            snap_first_ok_ = 1;
        }
    }

    void hold_snapshot_slot(size_t slot, long idx) {
        // A slot that is not filled keeps whatever it held: the validity lives
        // in a separate flag, not in the Mat.
        snap_ok_[slot] = 0;
        if (idx <= clamp_lo()) {
            // The cold-start clamp: the read is served by the clamp target,
            // which the snapshot holds separately (snap_first_).
            return;
        }
        const size_t s = static_cast<size_t>(((idx % ring_depth()) + ring_depth()) % ring_depth());
        if (phys_anchor_ && ring_id_[s] != idx) {
            // A HOLE IN THE HISTORY: the frame is not in the ring at all.
            // The slot is left INVALID on purpose, so the read takes the
            // ring's own path, which detects the hole, counts it and falls
            // back to the nearest older frame -- exactly as it did before
            // the snapshot existed.
            return;
        }
        const Mat& src = ring_[s];
        if (src.empty()) return;   // never seeded: the read falls back too
        // ZERO COPY: the snapshot takes a REFERENCE to the ring's pixels.
        //
        // OpenCV's Mat::operator= is a SHALLOW copy -- it bumps the refcount
        // and copies the data POINTER; there is no clone() (the pipeline does
        // the same thing at mosaic_pipeline.cpp:77-85, and relies on it).  A
        // shallow copy is exactly what is needed, because the thing that must
        // not change is the PIXELS the fit reads, and the producer never
        // writes into a buffer it has handed over: it replaces the ring SLOT
        // with another frame (ring_[s] = frame), which only rebinds the
        // pointer.  A reference taken here therefore keeps the pixels of THIS
        // frame alive and unchanged for the whole computation, while the
        // producer stays free to move the ring on.
        //
        // A DEEP copy (copyTo) was the wrong tool twice over.  It moved ~7 MB
        // per frame through the CPU for nothing, and -- because it copies
        // whatever the slot holds AT THAT MOMENT -- it would silently freeze a
        // WRONG frame into the snapshot whenever the slot had already been
        // replaced, hiding the very event the snapshot is meant to expose.
        snap_[slot] = src;
        snap_ok_[slot] = 1;
    }

    // The snapshot slot of a lag, or -1 when the lag is not on the window's
    // grid.  The grid is exact -- the window's lags are stride_step apart --
    // so the slot is the lag divided by the stride, not a search.
    int snapshot_slot(int lag) const {
        if (lag < 0 || lag % stride_step_ != 0) return -1;
        const int s = lag / stride_step_;
        if (s >= static_cast<int>(snap_.size())) return -1;
        return s;
    }

    // Indexed by ABSOLUTE frame index modulo depth. Shifting a ring by one slot per push
    // would index by *one-frame* lag while lag lookups stride by stride_step, silently
    // fetching every history frame at half the intended lag.
    //
    // NOT const any more: in the physical anchoring mode this DETECTS a hole
    // in the history (the frame the lag asks for is not in the ring) and
    // counts it.  In the consumption-counter mode the body is the original
    // one, statement for statement.
    //
    // THE SNAPSHOT comes FIRST: a frame whose read right was PINNED out of
    // the ring before the first fit is served from that pinned reference, so
    // the producer cannot change the pixels under this computation.  A MISS
    // falls back to the ring's own path -- the hole detection and the
    // nearest-older fallback below -- and is COUNTED, so an incomplete
    // snapshot is visible instead of silently degrading the run.
    const Mat& frame_at_lag(int lag) {
        const long idx = static_cast<long>(t_) - lag;
        if (idx <= clamp_lo()) {
            // The cold-start clamp: the read is served by the FIRST frame,
            // which the snapshot holds a reference to (see build_snapshot).
            // That reference is taken once and never refreshed, because
            // `first_` itself is written once and never changed.
            if (snap_first_ok_) return snap_first_;
            return first_;
        }
        const int slot = snapshot_slot(lag);
        if (slot >= 0 && snap_ok_[slot]) return snap_[slot];
        ++n_snapshot_miss_;
        const size_t s = static_cast<size_t>(((idx % ring_depth()) + ring_depth()) % ring_depth());
        if (phys_anchor_ && ring_id_[s] != idx) {
            // A HOLE IN THE PHYSICAL HISTORY.  The frame at `idx` was never
            // delivered (the camera queue dropped it), so this slot holds a
            // DIFFERENT frame -- using it would register against the wrong
            // pixels, silently.  So: count it, flag the frame, and fall back
            // to the nearest AVAILABLE older frame.
            ++n_history_gap_;
            gap_this_frame_ = true;
            return nearest_older(idx);
        }
        const Mat& f = ring_[s];
        return f.empty() ? first_ : f;
    }

    // The nearest AVAILABLE older frame, used when the frame a lag asks for is
    // missing.  The search is bounded by the window THIS frame owns
    // ([t-max_lag, t]): the ingest may not write those slots (the window
    // guard keeps it ring_slack frames ahead of the oldest owed frame), so
    // reading them is safe -- and a frame older than that may already have
    // been overwritten.
    const Mat& nearest_older(long idx) {
        const long lo = std::max(first_idx_, static_cast<long>(t_) - max_lag_);
        for (long j = idx; j >= lo; --j) {
            const size_t s = static_cast<size_t>(((j % ring_depth()) + ring_depth()) % ring_depth());
            if (ring_id_[s] != j) continue;   // this slot holds another frame
            if (ring_[s].empty()) continue;
            return ring_[s];
        }
        return first_;
    }

    // The index the history falls back to when a lag reaches past the start of
    // the stream.  The consumption-counter mode keeps the original rule (0);
    // the physical mode clamps to the FIRST frame's index, so the frame that
    // is actually read and the index that names it agree.
    long clamp_lo() const { return phys_anchor_ ? first_idx_ : 0; }

    // H_{abs_index -> abs_index + stride_step}, cached by absolute index.
    Matx23f step(int abs_index) {
        auto it = steps_.find(abs_index);
        if (it != steps_.end()) return it->second;
        const auto mark = std::chrono::steady_clock::now();
        // frame_at_lag(lag) yields absolute index max(1, t_ - lag). Here the lags
        // are (t_ - abs_index) and (t_ - abs_index - stride_step_), so prev is
        // absolute index abs_index and curr is abs_index + stride_step -- in that
        // order. Swapping the two tags silently feeds the cache the wrong frame
        // (the anchors stay correct, only the composed lags move), so the order
        // here is load-bearing, not cosmetic.
        const Matx23f m = fit_similarity(clamp_idx(abs_index),
                                         clamp_idx(abs_index + stride_step_),
                                         frame_at_lag(t_ - abs_index),
                                         frame_at_lag(t_ - abs_index - stride_step_));
        stats_.t_fit += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                  mark)
                            .count();
        stats_.n_fit += 1;
        steps_[abs_index] = m;
        return m;
    }

    // The gate half of a fit: the finiteness check, the plausibility gate and
    // the rejection counters.  Split out of fit_similarity() so the parallel
    // anchor path can run the RANSAC stage on its own and still apply
    // exactly the same gate, in the same order, with the same counters.
    Matx23f finish_fit(const Matx23f& m) {
        if (!all_finite(m)) {
            stats_.n_fit_rejected += 1;
            return kIdentity;
        }
        // Plausibility gate. See check_fit() for the measured justification: a
        // finite outlier here is what produced a 4.8 GB device image and an OOM
        // kill with no error. Counted, never silently swallowed.
        const FitGate g = check_fit(m, shape_w_, shape_h_, gate_max_shift_frac_,
                                     gate_min_scale_, gate_max_scale_);
        if (!g.ok) {
            stats_.n_fit_rejected += 1;
            if (std::strcmp(g.why, "shift") == 0 || std::strcmp(g.why, "extent") == 0)
                stats_.n_fit_rejected_shift += 1;
            else stats_.n_fit_rejected_other += 1;
            return kIdentity;
        }
        return m;
    }

    Matx23f fit_similarity(long prev_abs, long curr_abs, const Mat& prev, const Mat& curr) {
        if (rng_seed_per_fit_) cv::theRNG().state = 0xFFFFFFFFu;
        const Matx23f m = est_->compute_affine(prev_abs, curr_abs, prev, curr);
        return finish_fit(m);
    }

    // Absolute index of the frame frame_at_lag(lag) returns. t_ is 0-based (the
    // first pushed frame is absolute index 0, and first_ IS that frame), and
    // frame_at_lag clamps anything deeper onto it, so the cache tag must be
    // clamped to 0 as well. Clamping to 1 instead silently labels the clamped
    // frame with the NEXT frame's index; the very next push then hits that
    // mislabelled slot and is served the wrong pixels.
    long abs_at_lag(int lag) const { return clamp_idx(t_ - lag); }
    static long clamp_abs(long abs) { return abs > 0 ? abs : 0; }
    // The clamp, in the anchoring mode in force.  The consumption-counter mode
    // keeps the original rule (0); the physical mode clamps to the FIRST
    // frame's index, so the cache tag always names the frame that is
    // actually read.
    long clamp_idx(long idx) const {
        if (!phys_anchor_) return clamp_abs(idx);
        return idx > first_idx_ ? idx : first_idx_;
    }

    // np.median(stack(history), axis=0).astype(float32) -> clip(curr - bg, 0, 255).astype(uint8)
    void median_residual(const Mat& curr, const std::vector<Mat>& history, Mat& out) const {
        const int n = static_cast<int>(history.size());
        out.create(curr.rows, curr.cols, CV_8U);
        if (n < 5) {  // the builder only forms the median when history has >= 5 entries
            curr.copyTo(out);
            return;
        }
        // The original loop is kept for the single-threaded case and is NOT just
        // an optimisation fallback: routing it through parallel_for_ measured
        // 90.8 ms/frame against 66.4 ms serial on the board, because the body
        // then runs through an indirect call and `history[i].at()` stops hoisting
        // its data pointer. Since threads=1 is the default, taking that hit would
        // have made the default path slower than before this change.
        if (cv::getNumThreads() <= 1) {
            std::vector<uint8_t> col(static_cast<size_t>(n));
            for (int x = 0; x < curr.cols; ++x) {
                for (int y = 0; y < curr.rows; ++y) {
                    for (int i = 0; i < n; ++i) col[i] = history[i].at<uint8_t>(y, x);
                    std::nth_element(col.begin(), col.begin() + n / 2, col.end());
                    const float bg = static_cast<float>(col[n / 2]);
                    float v = static_cast<float>(curr.at<uint8_t>(y, x)) - bg;
                    if (v < 0.0f) v = 0.0f;
                    if (v > 255.0f) v = 255.0f;
                    out.at<uint8_t>(y, x) = static_cast<uint8_t>(v);
                }
            }
            return;
        }

        // Parallelised over columns, keeping the original x-outer/y-inner order so
        // the traversal is unchanged.
        //
        // Why this loop needed doing by hand: it is our own scalar code, so
        // cv::setNumThreads cannot reach it. It was measured at 64 ms/frame --
        // 67% of the whole CPU arm -- while cv::warpAffine next to it went
        // 69 -> 28 ms purely from threading. That asymmetry was the tell.
        //
        // Bit-exactness is preserved because every output pixel is an
        // independent selection over its own 21 values: partitioning the columns
        // cannot change any pixel's result, and n is odd so the median is a
        // selection rather than an average (no summation order to perturb).
        // `col` lives inside the lambda so each worker owns its scratch.
        cv::parallel_for_(cv::Range(0, curr.cols), [&](const cv::Range& xs) {
            std::vector<uint8_t> col(static_cast<size_t>(n));
            for (int x = xs.start; x < xs.end; ++x) {
                for (int y = 0; y < curr.rows; ++y) {
                    for (int i = 0; i < n; ++i) col[i] = history[i].at<uint8_t>(y, x);
                    // n is odd (21) so the median is a selection, not an average: exact anywhere.
                    std::nth_element(col.begin(), col.begin() + n / 2, col.end());
                    const float bg = static_cast<float>(col[n / 2]);
                    float v = static_cast<float>(curr.at<uint8_t>(y, x)) - bg;
                    if (v < 0.0f) v = 0.0f;
                    if (v > 255.0f) v = 255.0f;
                    out.at<uint8_t>(y, x) = static_cast<uint8_t>(v);
                }
            }
        });
    }

    FusedGpu* gpu_ = nullptr;
    Backend backend_ = kCpu;
    int gpu_slots_ = 43;
    Mat out_gpu_;
    std::array<long, 3> dmax_{0, 0, 0};
    std::array<double, 3> dsum_{0.0, 0.0, 0.0};
    std::array<long, 3> dn_{0, 0, 0};
    std::array<long, 3> dexact_{0, 0, 0};
    long dframes_ = 0;

    int window_, stride_step_, max_lag_;
    // Extra history-ring slots, so a producer of the history may run ahead of
    // the engine that consumes it (multi-engine, different frames).  0 keeps
    // the original single-engine behaviour exactly.
    int ring_slack_ = 0;
    int ransac_max_iters_ = 0;   // 0 == the reference's own default
    bool rng_seed_per_fit_;
    bool nogmc_;
    bool shape_known_ = false;
    int shape_h_ = 0, shape_w_ = 0;
    std::vector<int> anchors_, lags_;
    std::set<int> anchor_set_;
    std::map<int, int> anchor_below_;
    std::unique_ptr<FastGMCEstimator> est_;
    // ---- the anchor LK stage's parallelism ----
    // pool_ is NULL when the anchor loop is the reference's serial one.
    // plans_ / lks_ are the per-anchor scratch: they are REUSED across frames
    // so the parallel path does not allocate per frame.
    std::unique_ptr<AnchorPool> pool_;
    int anchor_threads_ = 1;
    std::vector<int> anchor_cpus_;
    std::vector<FastGMCEstimator::LkPlan> plans_;
    std::vector<FastGMCEstimator::LkPairs> lks_;
    std::vector<Mat> ring_;
    // WHICH frame index each ring slot holds (see frame_at_lag).
    std::vector<long> ring_id_;
    // ---- the computation snapshot (the two-level cache) ----
    // REFERENCES to the history frames the CURRENT computation reads.  ZERO
    // COPY: each slot holds a shallow Mat (a data pointer plus a refcount), so
    // the pixels are neither moved nor duplicated -- the snapshot only PINS
    // the read right for the duration of this computation.
    // Written only by the computing thread (process_frame); seed_frame never
    // touches it.
    std::vector<Mat> snap_;
    // Which snapshot slots hold a valid reference.  A flag and not an empty
    // Mat, so a slot that is not filled keeps what it had.
    std::vector<unsigned char> snap_ok_;
    // The cold-start clamp target: a reference to the FIRST frame, which the
    // snapshot holds so that EVERY history read is served from the pinned
    // set.  Filled once and never refreshed.
    Mat snap_first_;
    bool snap_first_ok_ = false;
    // The history reads that were not served from the snapshot (see
    // n_snapshot_miss).
    long n_snapshot_miss_ = 0;
    std::unordered_map<int, Matx23f> steps_;
    Mat first_;
    // The index of the first frame that was seeded: the cold-start clamp.
    long first_idx_ = 0;
    long t_ = -1;
    // THE PHYSICAL ANCHOR (opt-in) and the hole counter it needs.  OFF by
    // default: the engine anchors on the consumption counter, as it always
    // has, and the counters stay 0 because that index sequence is dense.
    bool phys_anchor_ = false;
    long n_history_gap_ = 0;
    long n_history_gap_frames_ = 0;
    bool gap_this_frame_ = false;
    Stats stats_;
    bool profiling_ = false;
    int seq_id_ = 0;
    int seq_frame_ = 0;
    double io_ms_ = 0.0;
    double gate_max_shift_frac_ = kDefaultMaxShiftFrac;
    double gate_min_scale_ = kDefaultMinScale;
    double gate_max_scale_ = kDefaultMaxScale;
    long n_chain_rejected_ = 0;
    // Fit deadline guard. OFF by default (fit_deadline_ms_ <= 0): the guard makes
    // the output depend on wall-clock, i.e. on machine load, so the same footage
    // would decode differently on an idle and a loaded board. The deterministic
    // handles -- the pre-allocated device images and the corner budget -- are the
    // ones relied on for the 40 ms bound; this exists for deployments that accept
    // load-dependent output in exchange for a hard per-frame ceiling, and every
    // trigger is counted so the cost is never silent.
    double fit_deadline_ms_ = 0.0;
    bool fit_deadline_hold_ = false;      // false -> IDENTITY, true -> reuse last good
    long n_fit_deadline_ = 0;
    long n_corner_cap_frames_ = 0, corner_capped_prev_ = 0;
    std::vector<Matx23f> last_mats_;
    bool have_last_mats_ = false;
    FrameProfile cur_;               // the in-flight frame, filled across functions
    std::vector<FrameProfile> prof_;
};

// -----------------------------------------------------------------------------
// npy v1.0 writer so numpy can np.load the dump directly.
// -----------------------------------------------------------------------------
bool write_npy_u8(const std::string& path, const uint8_t* data, size_t count,
                  const std::vector<size_t>& shape) {
    std::string shape_s;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) shape_s += ", ";
        shape_s += std::to_string(shape[i]);
    }
    std::string dict = "{'descr': '|u1', 'fortran_order': False, 'shape': (" + shape_s + "), }";
    while ((10 + dict.size() + 1) % 64 != 0) dict.push_back(' ');
    dict.push_back('\n');
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "[WARN] cannot write %s\n", path.c_str());
        return false;
    }
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    std::fwrite(magic, 1, 8, f);
    const uint16_t hlen = static_cast<uint16_t>(dict.size());
    std::fwrite(&hlen, 2, 1, f);
    std::fwrite(dict.data(), 1, dict.size(), f);
    std::fwrite(data, 1, count, f);
    std::fclose(f);
    return true;
}

// float32 twin of write_npy_u8; the transform matrices are cast to float32 by _compose.
bool write_npy_f32(const std::string& path, const float* data, size_t count,
                   const std::vector<size_t>& shape) {
    std::string shape_s;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) shape_s += ", ";
        shape_s += std::to_string(shape[i]);
    }
    std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + shape_s + "), }";
    while ((10 + dict.size() + 1) % 64 != 0) dict.push_back(' ');
    dict.push_back('\n');
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "[WARN] cannot write %s\n", path.c_str());
        return false;
    }
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    std::fwrite(magic, 1, 8, f);
    const uint16_t hlen = static_cast<uint16_t>(dict.size());
    std::fwrite(&hlen, 2, 1, f);
    std::fwrite(dict.data(), 1, dict.size(), f);
    std::fwrite(data, sizeof(float), count, f);
    std::fclose(f);
    return true;
}

void mkdir_p(const std::string& p) {
    std::string acc;
    for (size_t i = 0; i <= p.size(); ++i) {
        if (i == p.size() || p[i] == '/') {
            if (!acc.empty() && !is_dir(acc)) ::mkdir(acc.c_str(), 0775);
        }
        if (i < p.size()) acc.push_back(p[i]);
    }
}

void usage() {
    std::printf(
        "gmc_stream -- bit-exact C++ port of the Trial 0474 streaming feature engine\n"
        "\n"
        "  --raw-root PATH        raw dataset root (mandatory)\n"
        "  --sequence NAME        sequence folder name; repeatable\n"
        "  --limit N              max frames per sequence (0 = all)\n"
        "  --mode MODE            tree (default, W(.) = fitted GMC) or nogmc (W(.) = IDENTITY,\n"
        "                         the A/B harness's control arm: 0 fits, 0 warps)\n"
        "  --anchor-step N        10 = shipping config (5 anchors, depth 4)\n"
        "                         2  = exact-equivalence config (21 anchors, depth 0)\n"
        "  --window N             median window (frozen = 21)\n"
        "  --stride-step N        lag stride (frozen = 2)\n"
        "  --downscale N          GMC estimation downscale (frozen = 2)\n"
        "  --threads N            OpenCV worker threads (default 1). N>1 is NOT\n"
        "                         bit-reproducible: always quote a transform deviation\n"
        "                         alongside any speed claim made with it.\n"
        "  --anchor-threads N     the anchor LK stage's workers, THIS thread included\n"
        "                         (default 1 = the reference's serial anchor loop).\n"
        "                         The 5 anchor LK stages then run CONCURRENTLY; the\n"
        "                         RANSAC stages still run SEQUENTIALLY, in the anchor\n"
        "                         order, in this thread -- so the global RNG is drawn\n"
        "                         from exactly as before and the OUTPUT IS\n"
        "                         BIT-IDENTICAL.  This is NOT --threads: OpenCV's own\n"
        "                         internal threading is untouched (still 1).\n"
        "  --anchor-cpus LIST     the cores the anchor workers are spread over\n"
        "                         ('a76', 'a55', 'all', 'none', or '4,5,6,7').  Without\n"
        "                         it they inherit this thread's affinity -- so a\n"
        "                         single-core engine shares its core with them and\n"
        "                         gains NOTHING from the parallelism.\n"
        "  --affinity LIST        pin this thread: 'a76' (=4-7, big), 'a55' (=0-3, little),\n"
        "                         'all', 'none', or an explicit list like '4,5,6,7'.\n"
        "                         RK3588: CPU0-3 = Cortex-A55 ~1.8GHz, CPU4-7 = A76 ~2.4GHz\n"
        "  --pyr-cache            reuse the per-frame half-res image, Shi-Tomasi corners and\n"
        "                         optical-flow pyramid across fits. Keyed by ABSOLUTE frame index.\n"
        "                         Bit-exact vs the uncached path (same MD5) -- see the header.\n"
        "  --cache-parts N        ablation mask for --pyr-cache: 1=resize 2=corners 4=pyramid\n"
        "                         (default 7 = all three)\n"
        "  --fit-corners N        Shi-Tomasi maxCorners (frozen = 600). Lower = less work,\n"
        "                         lower LK cost; frames above it are counted as cornercap.\n"
        "  --corner-probe 0|1      count frames that hit --fit-corners exactly (default 1)\n"
        "  --prealloc-pad N       create the device images ONCE, sized for N px of pad, so\n"
        "                         the frame loop can never reallocate (0 = off). Costs\n"
        "                         (W+2N)x(H+2N)xslots device bytes; the upload region stays\n"
        "                         the padded size, so H2D is unaffected.\n"
        "  --pin-pad 0|1          with --prealloc-pad N, also pin pad_ to N from the first\n"
        "                         frame (default 0). --prealloc-pad alone removes the\n"
        "                         ALLOCATION event but NOT the RING INVALIDATION event:\n"
        "                         pad_ growth drops every resident slot and forces a mass\n"
        "                         re-upload. On RK3588 that left VIDEO00034 breaching at\n"
        "                         43.16 ms with devrealloc already down to 1. Pinning costs\n"
        "                         <2.5 ms of p50 (measured: pad-1024 uploads run 2.45 ms vs\n"
        "                         0.37 ms at pad 32 -- the H2D path saturates) and removes\n"
        "                         the breach. Output is bit-identical.\n"
        "  --fit-deadline-ms F    give up on the remaining fits after F ms of fitting in one\n"
        "                         frame (0 = off, the default). OFF is deliberate: a wall-clock\n"
        "                         trigger makes the output depend on machine load. Use only if\n"
        "                         a hard per-frame ceiling outweighs reproducibility.\n"
        "  --fit-degrade MODE     what --fit-deadline-ms falls back to: identity (W=IDENTITY)\n"
        "                         or hold (reuse the previous frame's transforms)\n"
        "                         fewer tracked points.\n"
        "  --fit-win N            LK window size (frozen = 15)\n"
        "  --fit-level N          LK pyramid depth (frozen = 2). Lower = cheaper, less\n"
        "                         capture range for fast motion.\n"
        "  --fit-iters N          LK max iterations (frozen = 20, epsilon 0.03)\n"
        "  --fit-ransac-iters N   RANSAC sampling cap for the affine solve (0 = the\n"
        "                         reference's own default).  A positive value caps the\n"
        "                         sampling iterations: ENABLING IT CHANGES THE NUMBERS,\n"
        "                         so it must be reported as a deviation with the results.\n"
        "  --out-dir PATH         write [Ch0,Ch1,Ch2] stacked HxWx3 JPGs (builder layout)\n"
        "  --out-ext jpg|png|bmp  codec for --out-dir (default jpg, the historical behaviour).\n"
        "                         LOSSLESS CODEC REQUIRED for any accuracy reconciliation.\n"
        "                         On RK3588 use bmp: that OpenCV build carries only the\n"
        "                         built-in BMP codec -- both png and jpg writers are absent\n"
        "                         (imwrite fails with 'could not find a writer'), and jpg\n"
        "                         would additionally be lossy on Ch1/Ch2.\n"
        "  --dump-dir PATH        write per-frame .npy (3,H,W); --mats-frame also gets _mats.npy\n"
        "  --mats-all PATH        log every frame's 21x6 transform chain as raw float32\n"
        "                         (~30 KB / 60 frames, vs ~59 MB for --dump-dir)\n"
        "  --mats-frame N         which push writes _mats.npy (default 0; pick a warm frame to\n"
        "                         inspect the transform chain away from the cold-start plateau)\n"
        "  --md5-out PATH         append per-frame md5 lines (seq, index, hash)\n"
        "  --rng-seed-per-fit     reseed theRNG before each fit; MUST be mirrored in Python\n"
        "  --timing               print the fit/warp/median breakdown with fits/frm\n"
        "  --profile-csv PATH     enable the per-frame stage ledger and write raw rows\n"
        "                         (prep/fit/warp/median/img_up/mats_up/kernel/readback)\n"
        "  --profile-warmup N     frames at the head of EACH sequence excluded from the\n"
        "                         statistics (default 100); raw CSV keeps them\n"
        "  --fit-gate-shift F    reject a GMC fit whose translation exceeds F x frame\n"
        "                         width (default 1.0; 0 disables the gate)\n"
        "  --fit-gate-scale F    max |scale-1| allowed (default 1.0 -> [0.5,2])\n"
        "\n"
        "  --fused cpu|gpu|both   which arm computes the warp+median+assembly tail\n"
        "                         (default both: run BOTH on the same fits and A/B them)\n"
        "  --kernel-dir PATH      where warp_median_fused.cl + sortnet_generated.inc live\n"
        "                         (default ../opencl/kernels)\n"
        "  --hw-linear 0|1        0 = CLK_FILTER_NEAREST + manual 2x2 blend (DEFAULT; measured\n"
        "                             required -- PoCL's CLK_FILTER_LINEAR cannot even reproduce a\n"
        "                             plain integer-coordinate copy, Ch0 Max|Diff| = 34)\n"
        "                         1 = CLK_FILTER_LINEAR sampler. RE-MEASURE ON MALI: PoCL's\n"
        "                             failure says nothing about Mali's accuracy.\n"
        "  --fp64-coord 0|1       DIAGNOSTIC only; never on RK3588 (Mali has no FP64)\n");
}

}  // namespace

// ===========================================================================
// Stage-ledger reporting.
//
// One row per physical stage, with the tail carried: a mean alone would hide the
// bimodal population that DVFS and scheduler preemption produce, which is the
// entire reason this file exists.
//
// The TILING CHECK is the load-bearing part. `resid` is
// end_to_end - sum(stages), computed per frame. Because the stages are adjacent
// timestamp pairs, this should sit near zero; a non-zero residual means either
// unattributed work exists or the probes are wrong, and either way the run's
// numbers are not yet trustworthy. It is printed, not asserted.
// ===========================================================================
static void print_profile_stats(const char* label, const std::vector<FrameProfile>& rows,
                                int warmup, OnlineFeaturePipeline::Backend backend) {
    if (rows.empty()) return;

    std::vector<double> prep, fit, warp, median, img_up, mats_up, kernel, readback, total, resid;
    std::vector<double> io;
    long dropped = 0;
    for (const FrameProfile& p : rows) {
        if (p.seq_frame < warmup) { ++dropped; continue; }
        io.push_back(p.io);
        prep.push_back(p.prep);
        fit.push_back(p.fit);
        warp.push_back(p.warp);
        median.push_back(p.median);
        img_up.push_back(p.img_up);
        mats_up.push_back(p.mats_up);
        kernel.push_back(p.kernel);
        readback.push_back(p.readback);
        total.push_back(p.total);
        resid.push_back(p.total - (p.io + p.prep + p.fit + p.warp + p.median + p.img_up +
                                   p.mats_up + p.kernel + p.readback));
    }
    if (total.empty()) {
        std::printf("[PROFILE] %s: all %zu frames fell inside the warm-up window\n", label,
                    rows.size());
        return;
    }

    struct Row { const char* name; StageStats s; const char* owner; };
    std::vector<Row> table;
    table.push_back({"0 frame decode (eMMC/BMP)", summarize(io), "CPU + storage"});
    table.push_back({"1 prep (ring push)", summarize(prep), "CPU A76"});
    table.push_back({"2 fit (LK+RANSAC+chain)", summarize(fit), "CPU A76 xN"});
    table.push_back({"2b warp (x22)", summarize(warp), "CPU A76 xN"});
    table.push_back({"2c median (21-frame)", summarize(median), "CPU A76 xN"});
    table.push_back({"3 img_up (H2D image)", summarize(img_up), "bus / Mali"});
    table.push_back({"4 mats_up (H2D 21 mats)", summarize(mats_up), "bus / Mali"});
    table.push_back({"5 kernel (warp+median)", summarize(kernel), "Mali-G610"});
    table.push_back({"6 readback (D2H 3-plane)", summarize(readback), "bus / CPU"});

    const StageStats tot = summarize(total);
    const StageStats rs = summarize(resid);

    std::printf("\n=========== STAGE LEDGER: %s ===========\n", label);
    std::printf("frames used=%ld  (warm-up discarded=%ld of %zu)  backend=%s\n", tot.n, dropped,
                rows.size(), backend == OnlineFeaturePipeline::kCpu
                                 ? "cpu" : (backend == OnlineFeaturePipeline::kGpu ? "gpu" : "both"));
    std::printf("%-26s %9s %8s %8s %8s %9s %9s  %s\n", "stage", "mean", "p50", "p95", "p99",
                "max", "stddev", "owner");
    std::printf("%-26s %9s %8s %8s %8s %9s %9s  %s\n", "--------------------------", "---------",
                "--------", "--------", "--------", "---------", "---------", "----------");
    char line[512];
    for (const Row& r : table) {
        std::snprintf(line, sizeof(line),
                      "%-26s %9.3f %8.3f %8.3f %8.3f %9.3f %9.3f  %s", r.name, r.s.mean, r.s.p50,
                      r.s.p95, r.s.p99, r.s.max, r.s.sd, r.owner);
        std::printf("%s\n", line);
    }
    std::printf("%-26s %9.3f %8.3f %8.3f %8.3f %9.3f %9.3f  %s\n", "== END-TO-END TOTAL ==",
                tot.mean, tot.p50, tot.p95, tot.p99, tot.max, tot.sd,
                tot.p50 > 0.0 ? "steady fps" : "-");
    std::printf("%-26s %9.3f %8.3f %8.3f %8.3f %9.3f %9.3f  %s\n", "TILING RESIDUAL",
                rs.mean, rs.p50, rs.p95, rs.p99, rs.max, rs.sd,
                (std::fabs(rs.mean) < 1e-3 && std::fabs(rs.max) < 1e-2) ? "TILES" : "CHECK ME");
    std::printf("steady-state fps @p50 = %.2f | @p99 = %.2f | @max = %.2f\n",
                tot.p50 > 0.0 ? 1000.0 / tot.p50 : 0.0, tot.p99 > 0.0 ? 1000.0 / tot.p99 : 0.0,
                tot.max > 0.0 ? 1000.0 / tot.max : 0.0);
    std::printf("[budget] 40ms red line: p99 %s (%.1f ms), max %s (%.1f ms)\n",
                tot.p99 <= 40.0 ? "PASS" : "FAIL", tot.p99,
                tot.max <= 40.0 ? "PASS" : "FAIL", tot.max);
}

// Raw per-frame rows, warm-up frames INCLUDED. The statistics deliberately drop
// them; the raw trace must not, or a cold-start bug becomes unreproducible.
static void write_profile_csv(const char* path, const std::vector<FrameProfile>& rows) {
    FILE* f = std::fopen(path, "w");
    if (!f) {
        std::fprintf(stderr, "[FATAL] cannot write %s\n", path);
        std::exit(2);
    }
    std::fprintf(f, "seq_id,frame_idx,seq_frame,io_ms,prep_ms,fit_ms,warp_ms,median_ms,"
                    "img_up_ms,mats_up_ms,kernel_ms,readback_ms,total_ms,resid_ms\n");
    for (const FrameProfile& p : rows) {
        const double sum = p.io + p.prep + p.fit + p.warp + p.median + p.img_up + p.mats_up +
                           p.kernel + p.readback;
        std::fprintf(f, "%d,%ld,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                     p.seq_id, p.idx, p.seq_frame, p.io, p.prep, p.fit, p.warp, p.median, p.img_up,
                     p.mats_up, p.kernel, p.readback, p.total, p.total - sum);
    }
    std::fclose(f);
}

// ---------------------------------------------------------------------------
// Reuse hook. Defining GMC_STREAM_OCL_NO_MAIN lets e2e_core.cpp #include this
// file and drive the very same classes (OnlineFeaturePipeline / FastGMCEstimator /
// FusedGpu) from a single-process demo. The main() body below is NOT touched --
// the point is that the e2e demo and this binary execute the identical code, so
// their features are byte-identical by construction rather than by re-derivation.
// ---------------------------------------------------------------------------
#ifndef GMC_STREAM_OCL_NO_MAIN
int main(int argc, char** argv) {
    std::string why;
    if (!gmcpp::md5_self_test(&why)) {
        std::fprintf(stderr, "[FATAL] built-in md5 self-test failed (%s). Refusing to run: a "
                             "fingerprint bug must never be mistaken for a feature mismatch.\n",
                     why.c_str());
        return 3;
    }

    std::string raw_root, out_dir, dump_dir, md5_out, mode = "tree";
    // Output codec for --out-dir. "jpg" is the historical default and stays the
    // default, so every existing invocation is byte-identical. "png" is lossless
    // and is what the frozen uav_gmc_median feature set is stored in -- a JPEG
    // round-trip (q95 with 4:2:0 chroma subsampling) perturbs Ch1/Ch2 exactly
    // where the heatmap head is most sensitive, so any accuracy reconciliation
    // must run with png.
    std::string out_ext = "jpg";
    std::vector<std::string> sequences;
    int limit = 0, anchor_step = 10, window = 21, stride_step = 2, downscale = 2;
    long mats_frame = 0;
    std::string mats_all_path;
    bool rng_seed_per_fit = false, want_timing = false;
    // Stage-ledger controls. CSV empty => profiling OFF (the default), so an
    // accuracy run pays nothing. warmup frames are dropped from the statistics
    // (but still computed and still in the CSV) so cold-start plateaus and Mali
    // driver JIT do not contaminate the steady-state numbers.
    std::string profile_csv;
    int profile_warmup = 100;
    // Fit gate. Defaults are the measured-safe values; 0 disables the gate and is
    // only for reproducing the pre-fix OOM on purpose.
    double gate_shift = kDefaultMaxShiftFrac;
    double gate_scale = 1.0;   // max |scale - 1|; 1.0 -> [0,2] clamped to [0.5,2]
    double gate_min = kDefaultMinScale, gate_max = kDefaultMaxScale;
    bool pyr_cache = false;
    int threads = 1;
    // The anchor LK stage's workers (the calling thread INCLUDED).  1 == the
    // reference's serial anchor loop.  The RANSAC stage is NOT affected: it
    // always runs afterwards, in this thread, in the anchor order, so the
    // global RNG is drawn from exactly as before and the output stays
    // bit-identical.  It is SEPARATE from --threads, which is OpenCV's own
    // internal threading (and >1 there breaks bit-reproducibility).
    int anchor_threads = 1;
    std::string anchor_cpus;
    std::string affinity = "none";
    int cache_parts = 7;  // resize | corners | pyramid
    // Fit knobs. Defaults are the frozen values, so omitting every flag
    // reproduces the shipped baseline exactly (same MD5).
    int fit_corners = 600, fit_win = 15, fit_level = 2, fit_iters = 20;
    // RANSAC sampling cap.  0 = OFF: the reference call keeps its own default
    // arguments, so omitting the flag reproduces the shipped baseline exactly.
    int fit_ransac_iters = 0;
    // ---- 40 ms guarantee: the two deterministic handles, plus one opt-in ----
    //
    // prealloc_cap: create the device images once, sized for this much padding,
    // so the frame loop can never reallocate. 0 = off (grow-on-demand).
    // corner_probe: count exactly which frames hit the --fit-corners budget.
    // fit_deadline: wall-clock fit guard. OFF by default -- see the note on
    // fit_deadline_ms_; it makes output depend on machine load.
    int prealloc_cap = 0, corner_probe = 1;
    bool pin_pad = false;
    double fit_deadline = 0.0;
    std::string fit_degrade = "identity";
    std::string fused = "both", kernel_dir = "../opencl/kernels";
    int hw_linear = 0, fp64_coord = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "[FATAL] %s needs a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--raw-root") raw_root = next("--raw-root");
        else if (a == "--sequence") sequences.push_back(next("--sequence"));
        else if (a == "--mode") mode = next("--mode");
        else if (a == "--limit") limit = std::atoi(next("--limit").c_str());
        else if (a == "--mats-frame") mats_frame = std::atol(next("--mats-frame").c_str());
        else if (a == "--mats-all") mats_all_path = next("--mats-all");
        else if (a == "--anchor-step") anchor_step = std::atoi(next("--anchor-step").c_str());
        else if (a == "--window") window = std::atoi(next("--window").c_str());
        else if (a == "--stride-step") stride_step = std::atoi(next("--stride-step").c_str());
        else if (a == "--downscale") downscale = std::atoi(next("--downscale").c_str());
        else if (a == "--pyr-cache") pyr_cache = true;
        else if (a == "--threads") threads = std::atoi(next("--threads").c_str());
        else if (a == "--anchor-threads") anchor_threads = std::atoi(next("--anchor-threads").c_str());
        else if (a == "--anchor-cpus") anchor_cpus = next("--anchor-cpus");
        else if (a == "--affinity") affinity = next("--affinity");
        else if (a == "--cache-parts") cache_parts = std::atoi(next("--cache-parts").c_str());
        else if (a == "--fit-corners") fit_corners = std::atoi(next("--fit-corners").c_str());
        else if (a == "--prealloc-pad") prealloc_cap = std::atoi(next("--prealloc-pad").c_str());
        else if (a == "--pin-pad") pin_pad = std::atoi(next("--pin-pad").c_str()) != 0;
        else if (a == "--corner-probe") corner_probe = std::atoi(next("--corner-probe").c_str());
        else if (a == "--fit-deadline-ms") fit_deadline = std::atof(next("--fit-deadline-ms").c_str());
        else if (a == "--fit-degrade") fit_degrade = next("--fit-degrade");
        else if (a == "--fit-win") fit_win = std::atoi(next("--fit-win").c_str());
        else if (a == "--fit-level") fit_level = std::atoi(next("--fit-level").c_str());
        else if (a == "--fit-iters") fit_iters = std::atoi(next("--fit-iters").c_str());
        else if (a == "--fit-ransac-iters")
            fit_ransac_iters = std::atoi(next("--fit-ransac-iters").c_str());
        else if (a == "--out-dir") out_dir = next("--out-dir");
        else if (a == "--out-ext") out_ext = next("--out-ext");
        else if (a == "--dump-dir") dump_dir = next("--dump-dir");
        else if (a == "--md5-out") md5_out = next("--md5-out");
        else if (a == "--rng-seed-per-fit") rng_seed_per_fit = true;
        else if (a == "--timing") want_timing = true;
        else if (a == "--profile-csv") profile_csv = next("--profile-csv");
        else if (a == "--profile-warmup") profile_warmup = std::atoi(next("--profile-warmup").c_str());
        else if (a == "--fit-gate-shift") gate_shift = std::atof(next("--fit-gate-shift").c_str());
        else if (a == "--fit-gate-scale") { gate_scale = std::atof(next("--fit-gate-scale").c_str()); }
        else if (a == "--fused") fused = next("--fused");
        else if (a == "--kernel-dir") kernel_dir = next("--kernel-dir");
        else if (a == "--hw-linear") hw_linear = std::atoi(next("--hw-linear").c_str());
        else if (a == "--fp64-coord") fp64_coord = std::atoi(next("--fp64-coord").c_str());
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else {
            std::fprintf(stderr, "[FATAL] unknown argument: %s\n", a.c_str());
            usage();
            return 2;
        }
    }

    if (out_ext != "jpg" && out_ext != "png" && out_ext != "bmp") {
        std::fprintf(stderr,
                     "[FATAL] --out-ext must be jpg, png or bmp (got '%s'). Use bmp on the\n"
                     "        board: its OpenCV has no png/jpg writer.\n",
                     out_ext.c_str());
        return 2;
    }
    if (raw_root.empty() || sequences.empty()) {
        usage();
        return 2;
    }
    if (!is_dir(raw_root)) {
        std::fprintf(stderr, "[FATAL] raw root not a directory: %s\n", raw_root.c_str());
        return 2;
    }
    if (mode != "tree" && mode != "nogmc") {
        std::fprintf(stderr, "[FATAL] --mode must be 'tree' or 'nogmc', got '%s'\n", mode.c_str());
        return 2;
    }
    OnlineFeaturePipeline::Backend backend = OnlineFeaturePipeline::kCpu;
    if (fused == "cpu") backend = OnlineFeaturePipeline::kCpu;
    else if (fused == "gpu") backend = OnlineFeaturePipeline::kGpu;
    else if (fused == "both") backend = OnlineFeaturePipeline::kBoth;
    else {
        std::fprintf(stderr, "[FATAL] --fused must be 'cpu', 'gpu' or 'both', got '%s'\n",
                     fused.c_str());
        return 2;
    }
    std::unique_ptr<FusedGpu> gpu;
    if (backend != OnlineFeaturePipeline::kCpu) {
        std::fprintf(stderr, "[BUILD] backend=%s anchor_step=%d opencv=%s\n", fused.c_str(),
                     anchor_step, CV_VERSION);
        gpu.reset(new FusedGpu(kernel_dir, hw_linear != 0, fp64_coord != 0,
                               window * stride_step + 3));
    }
    if (!out_dir.empty()) mkdir_p(out_dir);
    if (!dump_dir.empty()) mkdir_p(dump_dir);

    gmcpp::MD5 seq_hash_all;
    long total_push = 0;
    long g_diff_frames = 0;
    long g_dmax[3] = {0, 0, 0};
    double g_dsum[3] = {0.0, 0.0, 0.0};
    long g_dn[3] = {0, 0, 0};
    long g_dex[3] = {0, 0, 0};
    int W0 = 0, H0 = 0, gpu_pad_max = 0;
    long gpu_bytes = 0;
    // The pipeline is rebuilt per sequence (each gets a clean ring + GPU slots),
    // so the ledger rows have to be lifted out here to become one run-wide table.
    std::vector<FrameProfile> all_prof;
    int seq_counter = 0;

    for (const std::string& seq : sequences) {
        std::string seq_dir;
        if (!find_sequence_folder(raw_root, seq, &seq_dir)) {
            std::fprintf(stderr, "[FATAL] sequence folder not found for %s under %s\n",
                         seq.c_str(), raw_root.c_str());
            return 2;
        }
        std::vector<std::string> names = list_dir(seq_dir, true);
        std::sort(names.begin(), names.end(), natural_less);
        if (limit > 0 && static_cast<int>(names.size()) > limit) names.resize(static_cast<size_t>(limit));
        if (names.empty()) {
            std::fprintf(stderr, "[FATAL] no frames in %s\n", seq_dir.c_str());
            return 2;
        }

        const std::string aff = bind_cpus(parse_cpu_list(affinity));
        std::fprintf(stderr, "[THREADS]   requested=%d affinity=%s (effective opencv threads=%d)\n",
                     threads, aff.c_str(), cv::getNumThreads());
        OnlineFeaturePipeline pipe(window, stride_step, anchor_step, downscale, rng_seed_per_fit,
                                    mode == "nogmc", gpu.get(), backend, pyr_cache, fit_corners,
                                    fit_win, fit_level, fit_iters, cache_parts, threads);
        pipe.set_profiling(!profile_csv.empty());
        pipe.set_seq_id(seq_counter);
        // The anchor LK stage's workers.  The RANSAC stage stays serial in the
        // anchor order, so the output is bit-identical either way.
        if (anchor_threads > 1) {
            std::vector<int> cpus;
            if (!anchor_cpus.empty() && anchor_cpus != "none")
                cpus = parse_cpu_list(anchor_cpus);
            pipe.set_anchor_threads(anchor_threads, cpus);
            std::printf("[ANCHOR-THREADS] %d workers on the anchor LK stage "
                        "(RANSAC stays serial in the anchor order: the output is "
                        "bit-identical)\n",
                        anchor_threads);
        }
        if (gate_shift > 0.0)
            pipe.set_fit_gate(gate_shift, gate_min, gate_max);
        else
            std::printf("[FIT-GATE] DISABLED (--fit-gate-shift 0): a finite outlier "
                        "transform will drive required_pad unbounded. Reproduces the "
                        "pre-fix OOM by design.\n");
        std::printf("[FIT-GATE] shift<=%.2f x frame  scale in [%.2f,%.2f]\n", gate_shift,
                    gate_min, gate_max);
        pipe.gpu_prealloc(prealloc_cap);
        pipe.gpu_pin_pad(pin_pad);
        pipe.set_corner_probe(corner_probe != 0);
        pipe.set_fit_deadline(fit_deadline, fit_degrade == "hold");
        pipe.set_ransac_max_iters(fit_ransac_iters);
        if (fit_ransac_iters > 0)
            std::printf("[RANSAC-ITERS] maxIters=%d -- ENABLING IT CHANGES THE NUMBERS; "
                        "report it with the results\n",
                        fit_ransac_iters);
        if (prealloc_cap > 0)
            std::printf("[PREALLOC]  device images reserved once for pad<=%d "
                        "(no runtime reallocation)\n",
                        prealloc_cap);
        // The startup banner must never contain the literal "[HANDLE]": the
        // census parser matches that token, and a banner line would be counted
        // as a trigger row.
        if (pin_pad) {
            if (prealloc_cap <= 0)
                std::printf("[PIN-PAD] requested without --prealloc-pad; ignored\n");
            else
                std::printf("[PIN-PAD]  pad pinned to %d from the first frame, so the "
                            "ring is never invalidated (no mass re-upload)\n",
                            prealloc_cap);
        }
        if (fit_deadline > 0.0)
            std::printf("[FIT-DEADLINE] %.2f ms, degrade=%s -- OUTPUT BECOMES "
                        "LOAD-DEPENDENT; every trigger is counted in the per-sequence "
                        "handle census\n",
                        fit_deadline, fit_degrade.c_str());
        else
            std::printf("[FIT-DEADLINE] off (deterministic mode: output does not "
                        "depend on machine load)\n");
        if (!profile_csv.empty())
            std::printf("[PROFILE] stage ledger ON -> %s (warmup discarded=%d frames)\n",
                        profile_csv.c_str(), profile_warmup);
        if (sequences.size() == 1) {
            std::printf("ring_depth=%ld lags=%d..%d anchors=", pipe.ring_depth(),
                        pipe.lags().front(), pipe.lags().back());
            for (size_t i = 0; i < pipe.anchors().size(); ++i)
                std::printf("%s%d", i ? "," : "", pipe.anchors()[i]);
            std::printf("\n");
            std::printf("[BUILD] window=%d stride_step=%d anchor_step=%d downscale=%d mode=%s opencv=%s "
                        "rng_seed_per_fit=%d\n",
                        window, stride_step, anchor_step, downscale, mode.c_str(), CV_VERSION,
                        rng_seed_per_fit ? 1 : 0);
            std::printf("[NOTE] mode=%s is %s\n", mode.c_str(),
                        mode == "nogmc"
                            ? "the A/B CONTROL arm: W(.) = IDENTITY, 0 fits, 0 warps"
                            : (anchor_step == 2
                                   ? "the EXACT-EQUIVALENCE arm (composition depth 0): compare "
                                     "its output against uav_gmc_median to prove SOTA reproduction"
                                   : "the shipping arm; it is NOT equivalent to the frozen native "
                                     "set"));
        }

        gmcpp::MD5 seq_md5;
        char line[128];
        FILE* fmo = nullptr;
        FILE* fmo_all = nullptr;
        long n_mats_logged = 0;
        if (!mats_all_path.empty()) {
            fmo_all = std::fopen(mats_all_path.c_str(), "wb");
            if (!fmo_all) {
                std::fprintf(stderr, "[FATAL] cannot write %s\n", mats_all_path.c_str());
                std::exit(2);
            }
        }
        if (!md5_out.empty()) fmo = std::fopen(md5_out.c_str(), "ab");

        for (size_t fi = 0; fi < names.size(); ++fi) {
            const std::string path = seq_dir + "/" + names[fi];
            pipe.set_seq_frame((int)fi);
            const auto io_t0 = std::chrono::steady_clock::now();
            const Mat frame = cv::imread(path, cv::IMREAD_GRAYSCALE);
            pipe.set_frame_io_ms(
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - io_t0)
                    .count());
            if (frame.empty()) {
                std::fprintf(stderr, "[FATAL] failed to read %s\n", path.c_str());
                if (fmo) std::fclose(fmo);
        if (fmo_all) {
            std::fclose(fmo_all);
            std::printf("[MATS-ALL]  %s frames=%ld floats=%ld\n", mats_all_path.c_str(),
                        n_mats_logged, n_mats_logged * pipe.lags().size() * 6);
        }
                return 2;
            }
            std::vector<Matx23f> mats;
            // `mats` is only populated when push() is handed somewhere to put it, so the
            // --mats-all log needs it too -- otherwise it silently writes an empty file.
            const bool want_mats = !dump_dir.empty() || fmo_all != nullptr;
            const Mat out = pipe.push(frame, want_mats ? &mats : nullptr);

            // The comparison channel is the PRE-ENCODE array. Comparing decoded JPGs would
            // only measure JPEG quantisation, a storage artefact, not feature drift.
            gmcpp::MD5 f;
            f.update(reinterpret_cast<const uint8_t*>(out.data),
                     static_cast<size_t>(out.total()));
            const std::string fh = f.hex();
            seq_md5.update(fh);
            seq_hash_all.update(fh);
            ++total_push;
            if (fmo) {
                std::snprintf(line, sizeof(line), "%s\t%06zu\t%s\n", seq.c_str(), fi, fh.c_str());
                std::fputs(line, fmo);
            }
            if (want_timing && (fi + 1) % 50 == 0) {
                const auto& s = pipe.stats();
                const double n = static_cast<double>(std::max<long>(1, s.n_push));
                std::printf("[TIMING] %s frame %zu/%zu  fit=%.2f warp=%.2f median=%.2f total=%.2f ms"
                            "  fits/frm=%.2f warps/frm=%.2f\n",
                            seq.c_str(), fi + 1, names.size(), s.t_fit / n,
                            s.t_warp / n, s.t_median / n, s.t_total / n,
                            static_cast<double>(s.n_fit) / n, static_cast<double>(s.n_warp) / n);
            }
            if (!out_dir.empty()) {
                // Builder layout: HxWx3 stacked in [Ch0, Ch1, Ch2] order.
                Mat hwc(frame.rows, frame.cols, CV_8UC3);
                for (int y = 0; y < frame.rows; ++y)
                    for (int x = 0; x < frame.cols; ++x) {
                        const int p = y * frame.cols + x;
                        hwc.at<cv::Vec3b>(y, x)[0] = out.at<uint8_t>(0, p);
                        hwc.at<cv::Vec3b>(y, x)[1] = out.at<uint8_t>(1, p);
                        hwc.at<cv::Vec3b>(y, x)[2] = out.at<uint8_t>(2, p);
                    }
                // The sequence MUST be part of the filename: a multi-sequence run writes into
                // one --out-dir, and a push-index-only name makes every sequence overwrite the
                // previous one's frames. That silently corrupts the G2 comparison.
                char nm[256];
                std::snprintf(nm, sizeof(nm), "%s_%06zu.%s", seq.c_str(), fi,
                              out_ext.c_str());
                cv::imwrite(out_dir + "/" + nm, hwc);
            }
            if (!dump_dir.empty()) {
                char nm[128];
                std::snprintf(nm, sizeof(nm), "/%s_%06zu", seq.c_str(), fi);
                const size_t plane = static_cast<size_t>(frame.rows) * static_cast<size_t>(frame.cols);
                write_npy_u8(dump_dir + nm + "_feat.npy", reinterpret_cast<const uint8_t*>(out.data),
                             3 * plane,
                             {3, static_cast<size_t>(frame.rows), static_cast<size_t>(frame.cols)});
                if (static_cast<long>(fi) == mats_frame && !mats.empty()) {
                    // Row order matches the Python `mats` dict key order (sorted lags).
                    std::vector<float> flat;
                    flat.reserve(mats.size() * 6);
                    for (const Matx23f& m : mats)
                        for (int i = 0; i < 2; ++i)
                            for (int j = 0; j < 3; ++j) flat.push_back(m(i, j));
                    write_npy_f32(dump_dir + nm + "_mats.npy", flat.data(), flat.size(),
                                  {mats.size(), 6});
                }
                // Whole-run transform log: 21 lags x 6 floats per frame is ~30 KB
                // for 60 frames, versus ~59 MB for the feature planes. When the
                // question is "how far did the fit move", the transform chain is
                // the thing that moved, and it is orders of magnitude cheaper to
                // ship than the channels it produces.
            }
            if (fmo_all && !mats.empty()) {
                for (const Matx23f& m : mats)
                    for (int i = 0; i < 2; ++i)
                        for (int j = 0; j < 3; ++j)
                            std::fwrite(&m(i, j), sizeof(float), 1, fmo_all);
                ++n_mats_logged;
            }
        }
        if (fmo) std::fclose(fmo);

        const auto& s = pipe.stats();
        const double n = static_cast<double>(std::max<long>(1, s.n_push));
        if (backend == OnlineFeaturePipeline::kBoth) {
            g_diff_frames += pipe.diff_pixels(0);
            for (int c = 0; c < 3; ++c) {
                g_dmax[c] = std::max(g_dmax[c], pipe.diff_max()[c]);
                // Accumulate raw sums/counts, never a mean of means: the pixel
                // count is the same for all channels so the weights are equal,
                // but averaging per-sequence MAEs silently loses the frame count.
                g_dsum[c] += pipe.diff_abs_sum(c);
                g_dn[c] += pipe.diff_pixels(c);
                g_dex[c] += pipe.diff_exact_count(c);
            }
            g_diff_frames += pipe.diff_pixels(0);
        }
        if (gpu) { gpu_pad_max = std::max(gpu_pad_max, gpu->pad()); gpu_bytes = gpu->device_bytes();
                   if (!W0) { W0 = pipe.width(); H0 = pipe.height(); } }
        std::printf("[SEQ] %-32s frames=%-6zu md5=%s  fits/frm=%.2f warps/frm=%.2f "
                    "compose/frm=%.2f snapms/frm=%.2f snapmiss=%ld  state=%.2f MiB  devpad=%d  "
                    "devallocpad=%d  devrealloc=%ld",
                    seq.c_str(), names.size(), seq_md5.hex().c_str(),
                    static_cast<double>(s.n_fit) / n, static_cast<double>(s.n_warp) / n,
                    static_cast<double>(s.n_compose) / n, s.t_snapshot / n,
                    pipe.n_snapshot_miss(),
                    pipe.state_bytes() / 1048576.0,
                    pipe.gpu_pad(), pipe.gpu_alloc_pad(), pipe.gpu_reconfigures());
        // ---- special-handle trigger census ---------------------------------
        // Every fallback path that changed a frame's output is counted here, in
        // one block, so "how often did the pipeline fall back and why" is a
        // number you can read instead of a claim.
        //
        // The block's FORMAT is frozen: guarantee_report.py parses it with a
        // strict regex, and a silently non-matching regex would print a table
        // of zeros -- i.e. certify a guarantee it never observed.  So the
        // snapshot's counters live in the [SEQ] segment instead.
        std::printf(" | [HANDLE] cornercap=%ldf/%ldcalls(%ldpts) fitgate=%ld chain=%ld "
                    "fitdeadline=%ld | corners mean=%.1f over %ld dets (budget %d)",
                    pipe.corner_cap_frames(), pipe.corner_capped(), pipe.corner_dropped(),
                    s.n_fit_rejected, pipe.n_chain_rejected(), pipe.n_fit_deadline(),
                    pipe.corner_mean(), pipe.corner_dets(), fit_corners);
        if (s.n_fit_rejected) {
            std::printf("[FIT-GATE] %-28s rejected=%ld/%ld fits (%.3f%%) "
                        "[shift=%ld other=%ld] -> those frames used W(.)=IDENTITY\n",
                        seq.c_str(), s.n_fit_rejected, s.n_fit,
                        100.0 * (double)s.n_fit_rejected / (double)std::max<long>(1, s.n_fit),
                        s.n_fit_rejected_shift, s.n_fit_rejected_other);
        }
        if (pipe.n_chain_rejected()) {
            std::printf("[FIT-GATE] %-28s rejected=%ld composed lags -> W(.)=IDENTITY\n",
                        seq.c_str(), pipe.n_chain_rejected());
        }
        if (want_timing) {
            std::printf("[SEQTIMING] %-32s fit=%.3f warp=%.3f median=%.3f total=%.3f ms/frame",
                        seq.c_str(), s.t_fit / n, s.t_warp / n, s.t_median / n, s.t_total / n);
            if (backend != OnlineFeaturePipeline::kCpu)
                std::printf(" | gpu=%.3f (up=%.3f enq=%.3f KERN=%.3f read=%.3f)",
                            s.t_gpu / n, s.t_gpu_upload / n, s.t_gpu_kernel / n,
                            s.t_gpu_kernprof / n, s.t_gpu_read / n);
            std::printf("\n");
            if (pyr_cache) {
                std::printf("[CACHE]     %-32s builds: resize=%ld corner=%ld pyr=%ld | "
                            "hits=%ld | %.1f MiB resident\n",
                            seq.c_str(), pipe.est_resize_builds(), pipe.est_corner_builds(),
                            pipe.est_pyr_builds(), pipe.est_cache_hits(),
                            pipe.est_cache_bytes() / 1048576.0);
            }
            if (backend == OnlineFeaturePipeline::kBoth) {
                const double cpu_tail = (s.t_warp + s.t_median) / n;
                const double gpu_tail = s.t_gpu / n;
                std::printf("[SPEEDUP] %-32s fused tail: CPU %.3f ms -> GPU %.3f ms  "
                            "(x%.2f, kernel-only x%.2f)\n",
                            seq.c_str(), cpu_tail, gpu_tail,
                            gpu_tail > 0.0 ? cpu_tail / gpu_tail : 0.0,
                            s.t_gpu_kernprof > 0.0 ? cpu_tail / (s.t_gpu_kernprof / n) : 0.0);
            }
        }
        if (backend == OnlineFeaturePipeline::kBoth) {
            std::printf("[A/B] %-32s Ch0 max=%ld MAE=%.5f exact=%.4f%% | Ch1 max=%ld MAE=%.5f "
                        "exact=%.4f%% | Ch2 max=%ld MAE=%.5f exact=%.4f%%\n",
                        seq.c_str(), pipe.diff_max()[0], pipe.diff_mae(0),
                        100.0 * pipe.diff_exact_frac(0), pipe.diff_max()[1], pipe.diff_mae(1),
                        100.0 * pipe.diff_exact_frac(1), pipe.diff_max()[2], pipe.diff_mae(2),
                        100.0 * pipe.diff_exact_frac(2));
        }
        // Lift this sequence's ledger rows into the run-wide table, then print the
        // per-sequence stage picture. Warm-up frames are EXCLUDED from these stats
        // (but stay in the CSV) because the first pushes of a fresh pipeline are the
        // ring-refill / slot-fill / Mali-JIT plateau, not steady state.
        if (pipe.profiling()) {
            print_profile_stats(seq.c_str(), pipe.profiles(), profile_warmup, backend);
            all_prof.insert(all_prof.end(), pipe.profiles().begin(), pipe.profiles().end());
        }
        ++seq_counter;
    }

    if (!profile_csv.empty()) {
        print_profile_stats("ALL-SEQUENCES", all_prof, profile_warmup, backend);
        write_profile_csv(profile_csv.c_str(), all_prof);
        std::printf("[PROFILE-CSV] wrote %s (%zu rows)\n", profile_csv.c_str(), all_prof.size());
    }

    std::printf("[TOTAL] sequences=%zu frames=%ld\n", sequences.size(), total_push);
    if (backend == OnlineFeaturePipeline::kBoth) {
        std::printf("\n================ CPU/GPU A-B ================\n");
        std::printf("backend=%s  sampling=%s  frame=%dx%d  device pad=%d  ring bytes=%.1f MiB\n",
                    fused.c_str(), hw_linear ? "CLK_FILTER_LINEAR" : "NEAREST+manual2x2", W0, H0,
                    gpu_pad_max, gpu_bytes / 1048576.0);
        std::printf("frames compared : %ld\n", g_diff_frames / 3);
        std::printf("channel |   Max|Diff| |     MAE |    exact %%\n");
        for (int c = 0; c < 3; ++c) {
            const double mae = g_dn[c] ? g_dsum[c] / (double)g_dn[c] : 0.0;
            const double ex = g_dn[c] ? (double)g_dex[c] / (double)g_dn[c] : 1.0;
            std::printf("  Ch%d   | %11ld | %9.5f | %9.4f%%\n", c, g_dmax[c], mae, 100.0 * ex);
        }
        std::printf("threshold       | %11d | %9.5f\n", 1, 0.05);
        std::printf("===============================================\n");
    }
    if (want_timing && backend != OnlineFeaturePipeline::kCpu) {
        std::printf("[TIMING NOTE] Segmented fit / fused-tail / total, same fits for both arms.\n");
        if (fused == "both")
            std::printf("  CPU fused tail = warp + median (as measured by gmc_stream.cpp).\n"
                        "  GPU fused tail = upload + kernel + download, i.e. INCLUDING the bus\n"
                        "  traffic the CPU arm does not pay. kernel-only speedup is printed\n"
                        "  per sequence above for the arithmetic-only comparison.\n");
        std::printf("  PoCL executes this on the CPU; its absolute numbers say nothing about\n"
                    "  the RK3588 budget. Use them only as a relative sanity check.\n");
    }
    std::printf("[MD5-SUM] %s\n", seq_hash_all.hex().c_str());
    if (want_timing) {
        std::printf("[TIMING NOTE] This is a third independent cost measurement, segmented as\n"
                    "  fit / warp / median with the operation counts printed alongside.\n"
                    "  Report this set only together with a cross-validated set; never quote a\n"
                    "  single tool's figure as the embedded budget (see gmc_net_pending.md 13.4).\n");
    }
    return 0;
}
#endif  // GMC_STREAM_OCL_NO_MAIN