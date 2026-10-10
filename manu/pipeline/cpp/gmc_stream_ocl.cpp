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
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <array>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <dirent.h>
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

    Matx23f compute_affine(long prev_abs, long curr_abs, const Mat& prev_gray,
                           const Mat& curr_gray) const {
        const int h = curr_gray.rows;
        const int w = curr_gray.cols;
        const int ds = downscale_;
        Matx23f H = kIdentity;

        Mat prev_small, curr_small;
        if (!cache_) {
            if (ds > 1) {
                const cv::Size small(w / ds, h / ds);
                cv::resize(prev_gray, prev_small, small, 0, 0, cv::INTER_LINEAR);
                cv::resize(curr_gray, curr_small, small, 0, 0, cv::INTER_LINEAR);
            } else {
                prev_small = prev_gray;
                curr_small = curr_gray;
            }
        } else {
            small_of(prev_abs, prev_gray, prev_small);
            small_of(curr_abs, curr_gray, curr_small);
        }

        // goodFeaturesToTrack: maxCorners=600, qualityLevel=0.01, minDistance=4, blockSize=3
        std::vector<cv::Point2f> pts_prev;
        if (!cache_) {
            cv::goodFeaturesToTrack(prev_small, pts_prev, max_corners_, 0.01, 4, cv::noArray(), 3);
            ++n_corner_builds_;
        } else {
            corners_of(prev_abs, prev_small, pts_prev);
        }
        if (pts_prev.size() < 6) return H;

        // Wrap as (N,1) CV_32FC2 -- the same layout cv2 hands over from the Python side.
        const Mat pts_prev_mat(static_cast<int>(pts_prev.size()), 1, CV_32FC2,
                               const_cast<cv::Point2f*>(pts_prev.data()));
        std::vector<cv::Point2f> pts_curr;
        std::vector<unsigned char> status;
        std::vector<float> err;
        // winSize=(15,15), maxLevel=2; flags/criteria/minEigThreshold left at their defaults
        if (iters_ != 20) {
            // Only pay for a non-default criterion when it was actually asked for;
            // the default path below keeps the exact shipped call signature.
            cv::TermCriteria crit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, iters_, 0.03);
            if (!cache_) {
                cv::calcOpticalFlowPyrLK(prev_small, curr_small, pts_prev_mat, pts_curr, status,
                                          err, cv::Size(win_, win_), max_level_, crit);
            } else {
                pyr_of(prev_abs, prev_small, pyr_prev_);
                pyr_of(curr_abs, curr_small, pyr_curr_);
                cv::calcOpticalFlowPyrLK(pyr_prev_, pyr_curr_, pts_prev_mat, pts_curr, status,
                                          err, cv::Size(win_, win_), max_level_, crit);
            }
        } else if (!cache_) {
            cv::calcOpticalFlowPyrLK(prev_small, curr_small, pts_prev_mat, pts_curr, status, err,
                                      cv::Size(win_, win_), max_level_);
        } else {
            pyr_of(prev_abs, prev_small, pyr_prev_);
            pyr_of(curr_abs, curr_small, pyr_curr_);
            cv::calcOpticalFlowPyrLK(pyr_prev_, pyr_curr_, pts_prev_mat, pts_curr, status, err,
                                      cv::Size(win_, win_), max_level_);
        }

        std::vector<cv::Point2f> p0, p1;
        p0.reserve(pts_prev.size());
        p1.reserve(pts_prev.size());
        for (size_t i = 0; i < pts_prev.size(); ++i) {
            if (i < status.size() && status[i] == 1) {
                p0.push_back(pts_prev[i]);
                p1.push_back(pts_curr[i]);
            }
        }
        if (p0.size() < 6) return H;

        Mat inliers;
        const Mat M = cv::estimateAffinePartial2D(p0, p1, inliers, cv::RANSAC, 3.0);
        if (M.empty()) return H;

        double v[6];
        for (int i = 0; i < 6; ++i) v[i] = M.at<double>(i / 3, i % 3);
        H = Matx23f(static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2]),
                    static_cast<float>(v[3]), static_cast<float>(v[4]), static_cast<float>(v[5]));
        if (ds > 1) {
            // float32 * float32(ds): matches numpy's weak-scalar semantics exactly.
            H(0, 2) = H(0, 2) * static_cast<float>(ds);
            H(1, 2) = H(1, 2) * static_cast<float>(ds);
        }
        return H;
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

    void corners_of(long abs, const Mat& small, std::vector<cv::Point2f>& out) const {
        if (!(parts_ & kCacheCorners)) {
            cv::goodFeaturesToTrack(small, out, max_corners_, 0.01, 4, cv::noArray(), 3);
            ++n_corner_builds_;
            return;
        }
        ++n_lookups_;
        const size_t s = slot_of(abs);
        if (tag_corners_[s] == abs) { ++n_hits_; out = corners_[s]; return; }
        cv::goodFeaturesToTrack(small, corners_[s], max_corners_, 0.01, 4, cv::noArray(), 3);
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
    mutable std::vector<Mat> pyr_prev_, pyr_curr_;  // scratch aliases for the LK call
    mutable long n_resize_builds_ = 0, n_corner_builds_ = 0, n_pyr_builds_ = 0;
    mutable long n_hits_ = 0, n_lookups_ = 0;
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
    const char* channel_note() const { return chan_ == 4 ? "CL_RGBA" : "CL_R8/CL_R"; }

    // Grow-only. The pad needed for BORDER_REFLECT emulation swings with the
    // warp (37..57 px on real GMC fits), so reallocating every frame would be
    // absurd; allocate at the largest pad seen so far and only grow.
    void configure(int w, int h, int pad) {
        const bool same_dims = (w == W_ && h == H_);
        if (same_dims && pad <= pad_) { W_ = w; H_ = h; return; }
        const int p = std::max(pad, pad_ > 0 ? pad_ : 2);
        W_ = w; H_ = h; pad_ = p;
        std::fill(slot_key_.begin(), slot_key_.end(), kNoKey);   // images are new
        key_to_slot_.clear();
        imgs_.clear();
        for (int i = 0; i < slots_; ++i)
            imgs_.emplace_back(ocl::make_image2d(ctx_.get(), fmt_,
                                                  (size_t)W_ + 2 * (size_t)pad_,
                                                  (size_t)H_ + 2 * (size_t)pad_));
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
        return (long)imgs_.size() * (long)(W_ + 2 * pad_) * (long)(H_ + 2 * pad_) * chan_;
    }

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
    int slots_ = 43;
    bool hw_linear_ = true;
    std::vector<long> slot_key_;
    std::map<long, size_t> key_to_slot_;
    double last_kernel_ns_ = 0.0;
    static constexpr long kNoKey = -(1LL << 60);
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

class OnlineFeaturePipeline {
public:
    struct Stats {
        double t_fit = 0, t_warp = 0, t_median = 0, t_total = 0;
        long n_push = 0, n_fit = 0, n_warp = 0, n_compose = 0;
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
                          int fit_iters, int cache_parts, int threads)
        : gpu_(gpu),
          backend_(backend),
          window_(window),
          stride_step_(stride_step),
          max_lag_(window * stride_step),
          rng_seed_per_fit_(rng_seed_per_fit),
          nogmc_(nogmc) {
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
        ring_.assign(static_cast<size_t>(max_lag_ + 1), Mat());
        // max_lag + 1 keys can be live at once, plus the clamped first-frame key.
        gpu_slots_ = max_lag_ + 3;
    }

    const std::vector<int>& anchors() const { return anchors_; }
    const std::vector<int>& lags() const { return lags_; }
    long ring_depth() const { return max_lag_ + 1; }
    // Cache telemetry, so a timing claim can be checked against how much work
    // was actually avoided rather than assumed.
    long est_resize_builds() const { return est_->resize_builds(); }
    long est_corner_builds() const { return est_->corner_builds(); }
    long est_pyr_builds() const { return est_->pyr_builds(); }
    long est_cache_hits() const { return est_->cache_hits(); }
    long est_cache_lookups() const { return est_->cache_lookups(); }
    size_t est_cache_bytes() const { return est_->cache_bytes(); }
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
    int width() const { return shape_w_; }
    int height() const { return shape_h_; }
    long gpu_bytes() const { return gpu_ ? gpu_->device_bytes() : 0; }

    size_t state_bytes() const {
        size_t n = 0;
        for (const Mat& f : ring_)
            if (!f.empty()) n += static_cast<size_t>(f.total() * f.elemSize());
        n += steps_.size() * sizeof(Matx23f);
        return n;
    }

    // Consume one grayscale frame; returns a 3 x (H*W) CV_8UC1 plane-major view of (3,H,W).
    Mat push(const Mat& frame, std::vector<Matx23f>* dump_mats) {
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

        if (first_.empty()) first_ = frame;
        ++t_;
        ring_[static_cast<size_t>(((t_ % ring_depth()) + ring_depth()) % ring_depth())] = frame;

        const auto mark0 = std::chrono::steady_clock::now();

        std::map<int, Matx23f> mats;
        if (!nogmc_) {
        // ---- anchors: direct fit on the anchor grid ----
        const auto mark_anchor = std::chrono::steady_clock::now();
        for (int lag : anchors_) {
            mats[lag] = fit_similarity(abs_at_lag(lag), t_, frame_at_lag(lag), frame);
            stats_.n_fit += 1;
        }
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
        if (dump_mats) {
            dump_mats->clear();
            for (const auto& kv : mats) dump_mats->push_back(kv.second);
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

            // ---- median + residual ----
            const auto mark_med = std::chrono::steady_clock::now();
            Mat ch2;
            median_residual(frame, history, ch2);
            stats_.t_median += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                          mark_med)
                                   .count();

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
        stats_.t_total +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mark0).count();

        for (auto it = steps_.begin(); it != steps_.end();) {
            if (it->first < t_ - max_lag_) it = steps_.erase(it);
            else ++it;
        }
        return out;
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
            }
            cv::Mat M32(2, 3, CV_32F);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 3; ++j) M32.at<float>(i, j) = Hm(i, j);
            cv::Mat Mi, Mi64;
            cv::invertAffineTransform(M32, Mi);
            Mi.convertTo(Mi64, CV_64F);
            inv.push_back(Mi64);
            for (int j = 0; j < 6; ++j) mf[k * 6 + j] = (float)Mi64.at<double>(j / 3, j % 3);
        }

        const int pad = required_pad(inv, W, H);

        // 2. resident frames. frame_at_lag() clamps any lag reaching past the
        //    first frame onto first_ (absolute index 1), so the effective index is
        //    max(1, t - lag) -- the same rule, not a guess.
        gpu_->configure(W, H, pad);

        // Content key for a lag: the absolute index frame_at_lag() would use, or
        // kFirstKey when it clamps onto first_. t_ itself is the current frame's
        // key. These are identities, so two different frames can never collide.
        constexpr long kFirstKey = -1;
        std::vector<cl_mem> slots;
        slots.reserve(lags_.size());
        for (size_t k = 0; k < lags_.size(); ++k) {
            const long idx = t_ - lags_[k];
            const long key = (idx <= 0) ? kFirstKey : idx;
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
        stats_.t_gpu_upload += ms(u1 - u0).count();
        stats_.t_gpu_kernel += ms(u2 - u1).count();
        stats_.t_gpu_read += ms(u3 - u2).count();
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

    // Indexed by ABSOLUTE frame index modulo depth. Shifting a ring by one slot per push
    // would index by *one-frame* lag while lag lookups stride by stride_step, silently
    // fetching every history frame at half the intended lag.
    const Mat& frame_at_lag(int lag) const {
        const long idx = static_cast<long>(t_) - lag;
        if (idx <= 0) return first_;
        const Mat& f =
            ring_[static_cast<size_t>(((idx % ring_depth()) + ring_depth()) % ring_depth())];
        return f.empty() ? first_ : f;
    }

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
        const Matx23f m = fit_similarity(clamp_abs(abs_index),
                                         clamp_abs(abs_index + stride_step_),
                                         frame_at_lag(t_ - abs_index),
                                         frame_at_lag(t_ - abs_index - stride_step_));
        stats_.t_fit += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                  mark)
                            .count();
        stats_.n_fit += 1;
        steps_[abs_index] = m;
        return m;
    }

    Matx23f fit_similarity(long prev_abs, long curr_abs, const Mat& prev, const Mat& curr) {
        if (rng_seed_per_fit_) cv::theRNG().state = 0xFFFFFFFFu;
        const Matx23f m = est_->compute_affine(prev_abs, curr_abs, prev, curr);
        if (!all_finite(m)) return kIdentity;
        return m;
    }

    // Absolute index of the frame frame_at_lag(lag) returns. t_ is 0-based (the
    // first pushed frame is absolute index 0, and first_ IS that frame), and
    // frame_at_lag clamps anything deeper onto it, so the cache tag must be
    // clamped to 0 as well. Clamping to 1 instead silently labels the clamped
    // frame with the NEXT frame's index; the very next push then hits that
    // mislabelled slot and is served the wrong pixels.
    long abs_at_lag(int lag) const { return clamp_abs(t_ - lag); }
    static long clamp_abs(long abs) { return abs > 0 ? abs : 0; }

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
    bool rng_seed_per_fit_;
    bool nogmc_;
    bool shape_known_ = false;
    int shape_h_ = 0, shape_w_ = 0;
    std::vector<int> anchors_, lags_;
    std::set<int> anchor_set_;
    std::map<int, int> anchor_below_;
    std::unique_ptr<FastGMCEstimator> est_;
    std::vector<Mat> ring_;
    std::unordered_map<int, Matx23f> steps_;
    Mat first_;
    long t_ = -1;
    Stats stats_;
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
        "  --affinity LIST        pin this thread: 'a76' (=4-7, big), 'a55' (=0-3, little),\n"
        "                         'all', 'none', or an explicit list like '4,5,6,7'.\n"
        "                         RK3588: CPU0-3 = Cortex-A55 ~1.8GHz, CPU4-7 = A76 ~2.4GHz\n"
        "  --pyr-cache            reuse the per-frame half-res image, Shi-Tomasi corners and\n"
        "                         optical-flow pyramid across fits. Keyed by ABSOLUTE frame index.\n"
        "                         Bit-exact vs the uncached path (same MD5) -- see the header.\n"
        "  --cache-parts N        ablation mask for --pyr-cache: 1=resize 2=corners 4=pyramid\n"
        "                         (default 7 = all three)\n"
        "  --fit-corners N        Shi-Tomasi maxCorners (frozen = 600). Lower = less work,\n"
        "                         fewer tracked points.\n"
        "  --fit-win N            LK window size (frozen = 15)\n"
        "  --fit-level N          LK pyramid depth (frozen = 2). Lower = cheaper, less\n"
        "                         capture range for fast motion.\n"
        "  --fit-iters N          LK max iterations (frozen = 20, epsilon 0.03)\n"
        "  --out-dir PATH         write [Ch0,Ch1,Ch2] stacked HxWx3 JPGs (builder layout)\n"
        "  --dump-dir PATH        write per-frame .npy (3,H,W); --mats-frame also gets _mats.npy\n"
        "  --mats-all PATH        log every frame's 21x6 transform chain as raw float32\n"
        "                         (~30 KB / 60 frames, vs ~59 MB for --dump-dir)\n"
        "  --mats-frame N         which push writes _mats.npy (default 0; pick a warm frame to\n"
        "                         inspect the transform chain away from the cold-start plateau)\n"
        "  --md5-out PATH         append per-frame md5 lines (seq, index, hash)\n"
        "  --rng-seed-per-fit     reseed theRNG before each fit; MUST be mirrored in Python\n"
        "  --timing               print the fit/warp/median breakdown with fits/frm\n"
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

int main(int argc, char** argv) {
    std::string why;
    if (!gmcpp::md5_self_test(&why)) {
        std::fprintf(stderr, "[FATAL] built-in md5 self-test failed (%s). Refusing to run: a "
                             "fingerprint bug must never be mistaken for a feature mismatch.\n",
                     why.c_str());
        return 3;
    }

    std::string raw_root, out_dir, dump_dir, md5_out, mode = "tree";
    std::vector<std::string> sequences;
    int limit = 0, anchor_step = 10, window = 21, stride_step = 2, downscale = 2;
    long mats_frame = 0;
    std::string mats_all_path;
    bool rng_seed_per_fit = false, want_timing = false;
    bool pyr_cache = false;
    int threads = 1;
    std::string affinity = "none";
    int cache_parts = 7;  // resize | corners | pyramid
    // Fit knobs. Defaults are the frozen values, so omitting every flag
    // reproduces the shipped baseline exactly (same MD5).
    int fit_corners = 600, fit_win = 15, fit_level = 2, fit_iters = 20;
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
        else if (a == "--affinity") affinity = next("--affinity");
        else if (a == "--cache-parts") cache_parts = std::atoi(next("--cache-parts").c_str());
        else if (a == "--fit-corners") fit_corners = std::atoi(next("--fit-corners").c_str());
        else if (a == "--fit-win") fit_win = std::atoi(next("--fit-win").c_str());
        else if (a == "--fit-level") fit_level = std::atoi(next("--fit-level").c_str());
        else if (a == "--fit-iters") fit_iters = std::atoi(next("--fit-iters").c_str());
        else if (a == "--out-dir") out_dir = next("--out-dir");
        else if (a == "--dump-dir") dump_dir = next("--dump-dir");
        else if (a == "--md5-out") md5_out = next("--md5-out");
        else if (a == "--rng-seed-per-fit") rng_seed_per_fit = true;
        else if (a == "--timing") want_timing = true;
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
            const Mat frame = cv::imread(path, cv::IMREAD_GRAYSCALE);
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
                std::snprintf(nm, sizeof(nm), "%s_%06zu.jpg", seq.c_str(), fi);
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
                    "compose/frm=%.2f  state=%.2f MiB\n",
                    seq.c_str(), names.size(), seq_md5.hex().c_str(),
                    static_cast<double>(s.n_fit) / n, static_cast<double>(s.n_warp) / n,
                    static_cast<double>(s.n_compose) / n, pipe.state_bytes() / 1048576.0);
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