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

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <dirent.h>
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
// -----------------------------------------------------------------------------
class FastGMCEstimator {
public:
    explicit FastGMCEstimator(int downscale) : downscale_(downscale) {}

    Matx23f compute_affine(const Mat& prev_gray, const Mat& curr_gray) const {
        const int h = curr_gray.rows;
        const int w = curr_gray.cols;
        const int ds = downscale_;
        Matx23f H = kIdentity;

        Mat prev_small, curr_small;
        if (ds > 1) {
            const cv::Size small(w / ds, h / ds);
            cv::resize(prev_gray, prev_small, small, 0, 0, cv::INTER_LINEAR);
            cv::resize(curr_gray, curr_small, small, 0, 0, cv::INTER_LINEAR);
        } else {
            prev_small = prev_gray;
            curr_small = curr_gray;
        }

        // goodFeaturesToTrack: maxCorners=600, qualityLevel=0.01, minDistance=4, blockSize=3
        std::vector<cv::Point2f> pts_prev;
        cv::goodFeaturesToTrack(prev_small, pts_prev, 600, 0.01, 4, cv::noArray(), 3);
        if (pts_prev.size() < 6) return H;

        // Wrap as (N,1) CV_32FC2 -- the same layout cv2 hands over from the Python side.
        const Mat pts_prev_mat(static_cast<int>(pts_prev.size()), 1, CV_32FC2,
                               const_cast<cv::Point2f*>(pts_prev.data()));
        std::vector<cv::Point2f> pts_curr;
        std::vector<unsigned char> status;
        std::vector<float> err;
        // winSize=(15,15), maxLevel=2; flags/criteria/minEigThreshold left at their defaults
        cv::calcOpticalFlowPyrLK(prev_small, curr_small, pts_prev_mat, pts_curr, status, err,
                                  cv::Size(15, 15), 2);

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
    int downscale_;
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
class OnlineFeaturePipeline {
public:
    struct Stats {
        double t_fit = 0, t_warp = 0, t_median = 0, t_total = 0;
        long n_push = 0, n_fit = 0, n_warp = 0, n_compose = 0;
    };

    OnlineFeaturePipeline(int window, int stride_step, int anchor_step, int downscale,
                          bool rng_seed_per_fit, bool nogmc)
        : window_(window),
          stride_step_(stride_step),
          max_lag_(window * stride_step),
          rng_seed_per_fit_(rng_seed_per_fit),
          nogmc_(nogmc) {
        cv::setNumThreads(1);  // mandatory for bit-reproducibility
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
        est_.reset(new FastGMCEstimator(downscale));
        ring_.assign(static_cast<size_t>(max_lag_ + 1), Mat());
    }

    const std::vector<int>& anchors() const { return anchors_; }
    const std::vector<int>& lags() const { return lags_; }
    long ring_depth() const { return max_lag_ + 1; }
    Stats& stats() { return stats_; }

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
            mats[lag] = fit_similarity(frame_at_lag(lag), frame);
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

        // ---- warps ----
        const auto mark_warp = std::chrono::steady_clock::now();
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
        Mat out(3, static_cast<int>(plane), CV_8UC1);
        std::memcpy(out.ptr(0), frame.data, plane);
        std::memcpy(out.ptr(1), ch1.data, plane);
        std::memcpy(out.ptr(2), ch2.data, plane);

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
        const Matx23f m = fit_similarity(frame_at_lag(t_ - abs_index),
                                         frame_at_lag(t_ - abs_index - stride_step_));
        stats_.t_fit += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                  mark)
                            .count();
        stats_.n_fit += 1;
        steps_[abs_index] = m;
        return m;
    }

    Matx23f fit_similarity(const Mat& prev, const Mat& curr) {
        if (rng_seed_per_fit_) cv::theRNG().state = 0xFFFFFFFFu;
        const Matx23f m = est_->compute_affine(prev, curr);
        if (!all_finite(m)) return kIdentity;
        return m;
    }

    // np.median(stack(history), axis=0).astype(float32) -> clip(curr - bg, 0, 255).astype(uint8)
    void median_residual(const Mat& curr, const std::vector<Mat>& history, Mat& out) const {
        const int n = static_cast<int>(history.size());
        out.create(curr.rows, curr.cols, CV_8U);
        if (n < 5) {  // the builder only forms the median when history has >= 5 entries
            curr.copyTo(out);
            return;
        }
        std::vector<uint8_t> col(static_cast<size_t>(n));
        for (int x = 0; x < curr.cols; ++x) {
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
    }

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
        "  --out-dir PATH         write [Ch0,Ch1,Ch2] stacked HxWx3 JPGs (builder layout)\n"
        "  --dump-dir PATH        write per-frame .npy (3,H,W); --mats-frame also gets _mats.npy\n"
        "  --mats-frame N         which push writes _mats.npy (default 0; pick a warm frame to\n"
        "                         inspect the transform chain away from the cold-start plateau)\n"
        "  --md5-out PATH         append per-frame md5 lines (seq, index, hash)\n"
        "  --rng-seed-per-fit     reseed theRNG before each fit; MUST be mirrored in Python\n"
        "  --timing               print the fit/warp/median breakdown with fits/frm\n");
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
    bool rng_seed_per_fit = false, want_timing = false;

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
        else if (a == "--anchor-step") anchor_step = std::atoi(next("--anchor-step").c_str());
        else if (a == "--window") window = std::atoi(next("--window").c_str());
        else if (a == "--stride-step") stride_step = std::atoi(next("--stride-step").c_str());
        else if (a == "--downscale") downscale = std::atoi(next("--downscale").c_str());
        else if (a == "--out-dir") out_dir = next("--out-dir");
        else if (a == "--dump-dir") dump_dir = next("--dump-dir");
        else if (a == "--md5-out") md5_out = next("--md5-out");
        else if (a == "--rng-seed-per-fit") rng_seed_per_fit = true;
        else if (a == "--timing") want_timing = true;
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
    if (!out_dir.empty()) mkdir_p(out_dir);
    if (!dump_dir.empty()) mkdir_p(dump_dir);

    gmcpp::MD5 seq_hash_all;
    long total_push = 0;

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

        OnlineFeaturePipeline pipe(window, stride_step, anchor_step, downscale, rng_seed_per_fit,
                                    mode == "nogmc");
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
        if (!md5_out.empty()) fmo = std::fopen(md5_out.c_str(), "ab");

        for (size_t fi = 0; fi < names.size(); ++fi) {
            const std::string path = seq_dir + "/" + names[fi];
            const Mat frame = cv::imread(path, cv::IMREAD_GRAYSCALE);
            if (frame.empty()) {
                std::fprintf(stderr, "[FATAL] failed to read %s\n", path.c_str());
                if (fmo) std::fclose(fmo);
                return 2;
            }
            std::vector<Matx23f> mats;
            const Mat out = pipe.push(frame, dump_dir.empty() ? nullptr : &mats);

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
            }
        }
        if (fmo) std::fclose(fmo);

        const auto& s = pipe.stats();
        const double n = static_cast<double>(std::max<long>(1, s.n_push));
        std::printf("[SEQ] %-32s frames=%-6zu md5=%s  fits/frm=%.2f warps/frm=%.2f "
                    "compose/frm=%.2f  state=%.2f MiB\n",
                    seq.c_str(), names.size(), seq_md5.hex().c_str(),
                    static_cast<double>(s.n_fit) / n, static_cast<double>(s.n_warp) / n,
                    static_cast<double>(s.n_compose) / n, pipe.state_bytes() / 1048576.0);
        if (want_timing) {
            std::printf("[SEQTIMING] %-32s fit=%.3f warp=%.3f median=%.3f total=%.3f ms/frame\n",
                        seq.c_str(), s.t_fit / n, s.t_warp / n, s.t_median / n,
                        s.t_total / n);
        }
    }

    std::printf("[TOTAL] sequences=%zu frames=%ld\n", sequences.size(), total_push);
    std::printf("[MD5-SUM] %s\n", seq_hash_all.hex().c_str());
    if (want_timing) {
        std::printf("[TIMING NOTE] This is a third independent cost measurement, segmented as\n"
                    "  fit / warp / median with the operation counts printed alongside.\n"
                    "  Report this set only together with a cross-validated set; never quote a\n"
                    "  single tool's figure as the embedded budget (see gmc_net_pending.md 13.4).\n");
    }
    return 0;
}