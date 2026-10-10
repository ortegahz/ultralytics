// ===========================================================================
// opencv_parity.cpp -- run the feature-channel CPU chain on whatever machine it
// is compiled for, and dump every intermediate to disk so two builds can be
// compared element by element.
//
// Why not just diff a final image
// ------------------------------
// The point is to find out *where* x86 and arm64 stop agreeing. A single final
// MAE cannot tell you whether Shi-Tomasi drifted, or LK's pyramid diverged, or
// RANSAC flipped to a different local optimum. So each operator's raw output is
// written separately, in the order the real pipeline calls them, and the
// comparison script reports one verdict per stage.
//
// This mirrors gmc_stream_ocl.cpp exactly: same downscale, same
// goodFeaturesToTrack arguments, same LK window/levels, same RANSAC threshold,
// same warpAffine flags. A parity result is only meaningful if it exercises the
// code that will actually ship.
//
// Deliberate determinism controls (both are mandated by gmc_stream_ocl.cpp's
// own header, not added for convenience):
//   cv::setNumThreads(1)  -- under multiple threads RANSAC and LK are not
//                            bit-reproducible; the host has 12 cores and the
//                            board has 8, so leaving it default would make the
//                            thread count an uncontrolled variable.
//   theRNG()              -- estimateAffinePartial2D consumes it.
//
// No imgcodecs: inputs are raw grayscale, so the arm64 build needs exactly the
// six modules that were cross-compiled.
// ===========================================================================

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <chrono>

#include <sys/stat.h>
#include <sys/types.h>

// Frozen shipping config, copied from gmc_stream_ocl.cpp's defaults.
static constexpr int   kDownscale   = 2;
static constexpr int   kWindow      = 21;   // median history length
static constexpr int   kMaxCorners  = 600;
static constexpr double kQualityLvl = 0.01;
static constexpr int   kMinDistance = 4;
static constexpr int   kBlockSize   = 3;
static constexpr int   kLkWinSize   = 15;
static constexpr int   kLkMaxLevel  = 2;
static constexpr double kRansacReproj = 3.0;

namespace {

// ===========================================================================
// Per-operator timing
// ===========================================================================
// Two rules make these numbers worth anything:
//
//  1. Dumping is I/O, not compute. Every operator writes 0.3-2 MB here, which
//     is far more time than the operator itself. Timing a run that also dumps
//     would report the filesystem. So bench_runs() repeats the pipeline with
//     dumping disabled and only the last pass writes.
//
//  2. The first call is not representative: OpenCV allocates pools, builds
//     lookup tables and the CPU clocks are still ramping. Passes are therefore
//     accumulated per operator and reported as min and median, not mean -- a
//     single cold pass would dominate a mean.
//
// Note what is deliberately NOT timed: nothing. The board is an 8-core RK3588
// and this measures exactly the CPU half of the feature channel, which is the
// part that had never been measured on real hardware before.
class Bench {
public:
    using Clock = std::chrono::steady_clock;

    void add(const char* name, double ms) { samples_[name].push_back(ms); }

    void report() const {
        if (samples_.empty()) return;
        std::printf("\n=== per-operator CPU timing (%zu frames x %d timed passes, "
                    "dumping disabled) ===\n", frames_, frames_ ? committed_ : 0);
        std::printf("%-16s %10s %10s %10s %10s\n",
                    "operator", "min ms", "median ms", "mean ms", "max ms");
        std::printf("%s\n", std::string(60, '-').c_str());
        double total_median = 0.0;
        for (const auto& kv : order_) {
            const std::vector<double>& v = samples_.at(kv);
            if (v.empty()) continue;
            std::vector<double> s = v;
            std::sort(s.begin(), s.end());
            const double mn = s.front();
            const double md = s[s.size() / 2];
            double sum = 0.0;
            for (double x : s) sum += x;
            std::printf("%-16s %10.4f %10.4f %10.4f %10.4f\n",
                        kv.c_str(), mn, md, sum / s.size(), s.back());
            total_median += md;
        }
        std::printf("%s\n", std::string(60, '-').c_str());
        std::printf("%-16s %10s %10.4f\n", "SUM (median)", "", total_median);
        std::printf("\nPer-frame figures above are one GMC fit: downscale, Shi-Tomasi,\n"
                    "pyramidal LK, RANSAC partial-affine, warp, and the 21-sample\n"
                    "median. Single-threaded (setNumThreads(1), mandatory for\n"
                    "bit-reproducibility), so this is not a parallel-throughput number.\n");
    }

    struct Timer {
        Clock::time_point t0 = Clock::now();
        double ms() const {
            return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        }
    };

    void setFrames(size_t n) { frames_ = n; }

    void countPass() { ++committed_; }

private:
    std::map<std::string, std::vector<double>> samples_;
    std::vector<std::string> order_ = {
        "resize", "gftt", "lk", "affine", "warp", "median", "frame_total"};
    int committed_ = 0;
    size_t frames_ = 0;
};

struct Dumper {
    std::string dir;
    int frame = 0;
    // Timing runs must not dump: each stage writes 0.3-2 MB over NFS, which
    // would dominate the very numbers being measured.
    bool enabled = true;

    void open(const std::string& d) { dir = d; frame = 0; }

    static std::string tag(const char* name, int f) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%s_f%04d", name, f);
        return buf;
    }

    // Writes a header line next to the binary blob so the comparison side never
    // has to guess the depth from the file size.
    void put(const char* name, const cv::Mat& m) {
        if (!enabled) return;
        if (m.empty()) {
            std::printf("  [%-14s] EMPTY\n", name);
            return;
        }
        const std::string base = dir + "/" + tag(name, frame);
        cv::Mat cont = m.isContinuous() ? m : m.clone();
        cv::Mat out;
        cont.reshape(1, 1).copyTo(out);

        const std::string bin = base + ".bin";
        const std::string hdr = base + ".txt";
        std::vector<unsigned char> raw(out.total() * out.elemSize());
        std::memcpy(raw.data(), out.data, raw.size());

        FILE* f = std::fopen(bin.c_str(), "wb");
        if (!f) { std::fprintf(stderr, "[ERROR] cannot write %s\n", bin.c_str()); std::exit(2); }
        std::fwrite(raw.data(), 1, raw.size(), f);
        std::fclose(f);

        FILE* h = std::fopen(hdr.c_str(), "w");
        if (h) {
            std::fprintf(h, "name=%s\nrows=%d\ncols=%d\nchannels=%d\ndepth=%d\n"
                            "elems=%lld\nbytes=%zu\n",
                         name, cont.rows, cont.cols, cont.channels(), cont.depth(),
                         (long long)cont.total(), raw.size());
            std::fclose(h);
        }
        std::printf("  [%-14s] %4d x %4d ch=%d depth=%d  elems=%lld\n", name,
                    cont.rows, cont.cols, cont.channels(), cont.depth(),
                    (long long)cont.total());
    }

    // For containers that are not Mats (std::vector<Point2f> and friends).
    void put_vec(const char* name, const std::vector<cv::Point2f>& v) {
        if (v.empty()) { std::printf("  [%-14s] EMPTY\n", name); return; }
        const cv::Mat m(static_cast<int>(v.size()), 1, CV_32FC2,
                        const_cast<cv::Point2f*>(v.data()));
        put(name, m);
    }
    void put_u8(const char* name, const std::vector<unsigned char>& v) {
        if (v.empty()) { std::printf("  [%-14s] EMPTY\n", name); return; }
        const cv::Mat m(static_cast<int>(v.size()), 1, CV_8UC1,
                        const_cast<unsigned char*>(v.data()));
        put(name, m);
    }
    void put_f32(const char* name, const std::vector<float>& v) {
        if (v.empty()) { std::printf("  [%-14s] EMPTY\n", name); return; }
        const cv::Mat m(static_cast<int>(v.size()), 1, CV_32FC1,
                        const_cast<float*>(v.data()));
        put(name, m);
    }
};

cv::Matx23f kIdentityMat() {
    cv::Matx23f h = cv::Matx23f::eye();
    return h;
}

// Verbatim port of gmc_stream_ocl.cpp's FastGMCEstimator::compute_affine, with a
// Dumper threaded through so each operator's raw output is observable.
cv::Matx23f compute_affine(const cv::Mat& prev_gray, const cv::Mat& curr_gray,
                           Dumper& d, Bench* bench) {
    // One timer per operator. d.put() calls sit OUTSIDE every timer so that
    // the NFS writes are never counted as operator time.
    Bench::Timer t;
    auto lap = [&](const char* op, Bench::Timer& timer) {
        if (bench) bench->add(op, timer.ms());
        timer = Bench::Timer();
    };

    const int h = curr_gray.rows;
    const int w = curr_gray.cols;
    const int ds = kDownscale;
    cv::Matx23f H = kIdentityMat();

    cv::Mat prev_small, curr_small;
    if (ds > 1) {
        const cv::Size small(w / ds, h / ds);
        cv::resize(prev_gray, prev_small, small, 0, 0, cv::INTER_LINEAR);
        cv::resize(curr_gray, curr_small, small, 0, 0, cv::INTER_LINEAR);
    } else {
        prev_small = prev_gray;
        curr_small = curr_gray;
    }
    lap("resize", t);
    d.put("resize_prev", prev_small);
    d.put("resize_curr", curr_small);

    std::vector<cv::Point2f> pts_prev;
    cv::goodFeaturesToTrack(prev_small, pts_prev, kMaxCorners, kQualityLvl,
                            kMinDistance, cv::noArray(), kBlockSize);
    lap("gftt", t);
    d.put_vec("gftt", pts_prev);
    if (pts_prev.size() < 6) return H;

    const cv::Mat pts_prev_mat(static_cast<int>(pts_prev.size()), 1, CV_32FC2,
                               const_cast<cv::Point2f*>(pts_prev.data()));
    std::vector<cv::Point2f> pts_curr;
    std::vector<unsigned char> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prev_small, curr_small, pts_prev_mat, pts_curr,
                              status, err, cv::Size(kLkWinSize, kLkWinSize), kLkMaxLevel);
    lap("lk", t);
    d.put_vec("lk_curr", pts_curr);
    d.put_u8("lk_status", status);
    d.put_f32("lk_err", err);

    std::vector<cv::Point2f> p0, p1;
    p0.reserve(pts_prev.size());
    p1.reserve(pts_prev.size());
    for (size_t i = 0; i < pts_prev.size(); ++i) {
        if (i < status.size() && status[i] == 1) {
            p0.push_back(pts_prev[i]);
            p1.push_back(pts_curr[i]);
        }
    }
    d.put_vec("lk_p0", p0);
    d.put_vec("lk_p1", p1);
    if (p0.size() < 6) return H;

    cv::Mat inliers;
    const cv::Mat M = cv::estimateAffinePartial2D(p0, p1, inliers, cv::RANSAC, kRansacReproj);
    lap("affine", t);
    d.put("affine_M", M);
    d.put("affine_inl", inliers);
    if (M.empty()) return H;

    double v[6];
    for (int i = 0; i < 6; ++i) v[i] = M.at<double>(i / 3, i % 3);
    H = cv::Matx23f(static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2]),
                    static_cast<float>(v[3]), static_cast<float>(v[4]), static_cast<float>(v[5]));
    if (ds > 1) {
        H(0, 2) = H(0, 2) * static_cast<float>(ds);
        H(1, 2) = H(1, 2) * static_cast<float>(ds);
    }
    d.put("fit_H", cv::Mat(H));
    return H;
}

cv::Mat warp(const cv::Mat& img, const cv::Matx23f& H, Dumper& d, Bench* bench) {
    cv::Mat M(2, 3, CV_32F);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j) M.at<float>(i, j) = H(i, j);
    cv::Mat dst;
    Bench::Timer t;
    cv::warpAffine(img, dst, M, cv::Size(img.cols, img.rows), cv::INTER_LINEAR,
                   cv::BORDER_REFLECT);
    if (bench) bench->add("warp", t.ms());
    d.put("warp_dst", dst);
    return dst;
}

// np.median(stack(history), axis=0).astype(float32) -> clip(curr - bg, 0, 255).astype(uint8)
//
// n is 21 -- odd -- so the median is a selection, not an average, and the
// whole stage is integer-only. That matters for this comparison: given
// identical warped inputs, median_out MUST come out bit-identical on both
// architectures. It is therefore the control stage: if it differs, the inputs
// feeding it already differed and any downstream error is not about median.
cv::Mat median_residual(const cv::Mat& curr, const std::vector<cv::Mat>& history) {
    const int n = static_cast<int>(history.size());
    cv::Mat out;
    out.create(curr.rows, curr.cols, CV_8U);
    if (n < 5) {  // the builder only forms the median when history has >= 5 entries
        curr.copyTo(out);
        return out;
    }
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
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir, tagname = "unknown", in_dir;
    int limit = 0;
    int bench_runs = 5;      // timed passes; the final pass dumps
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "[ERROR] %s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--in-dir") in_dir = next("--in-dir");
        else if (a == "--out-dir") dir = next("--out-dir");
        else if (a == "--tag") tagname = next("--tag");
        else if (a == "--limit") limit = std::atoi(next("--limit").c_str());
        else if (a == "--bench") bench_runs = std::atoi(next("--bench").c_str());
    }
    if (in_dir.empty() || dir.empty()) {
        std::fprintf(stderr,
                     "usage: %s --in-dir DIR --out-dir DIR [--tag NAME] [--limit N]\n"
                     "  in-dir  must contain frame_0000.gray ... (raw 8UC1, 640x512)\n"
                     "  --bench N  timed passes with dumping OFF (default 5), then one\n"
                     "             dumping pass; use --bench 0 for a correctness-only run\n", argv[0]);
        return 2;
    }

    cv::setNumThreads(1);   // mandatory for bit-reproducibility

    std::printf("=== opencv_parity (%s) ===\n", tagname.c_str());
    std::printf("OpenCV version : %s\n", CV_VERSION);
    const std::string build_info = cv::getBuildInformation();
    const std::string cpu_line = cv::getCPUFeaturesLine();
    std::printf("build info head: %s\n", build_info.c_str());
    std::printf("numThreads     : %d\n", cv::getNumThreads());
    std::printf("CPU count      : %d\n", cv::getNumberOfCPUs());
    std::printf("CPU features   : %s\n", cpu_line.c_str());
    std::printf("RNG state      : 0x%08llx\n",
                (unsigned long long)cv::theRNG().state);

    Dumper d;
    // Create the output directory rather than assuming it exists: on the board
    // the tree is on NFS and may not have been prepared by hand, and failing
    // only at the first file write hides the real cause behind a fopen error.
    ::mkdir(dir.c_str(), 0777);
    d.open(dir);

    // Load frames.
    std::vector<cv::Mat> frames;
    for (int i = 0; ; ++i) {
        char p[512];
        std::snprintf(p, sizeof p, "%s/frame_%04d.gray", in_dir.c_str(), i);
        FILE* f = std::fopen(p, "rb");
        if (!f) break;
        cv::Mat m(512, 640, CV_8UC1);
        size_t got = std::fread(m.data, 1, m.total(), f);
        std::fclose(f);
        if (got != m.total()) { std::fprintf(stderr, "[ERROR] short read %s\n", p); return 2; }
        frames.push_back(m);
        if (limit > 0 && (int)frames.size() >= limit) break;
    }
    if (frames.size() < 2) {
        std::fprintf(stderr, "[ERROR] need >= 2 frames in %s (found %zu)\n",
                     in_dir.c_str(), frames.size());
        return 2;
    }
    std::printf("frames loaded  : %zu (640x512 8UC1)\n\n", frames.size());

    std::vector<cv::Mat> history;
    int med_frames = 0;
    int dumped = 0;

    // Two modes in one binary:
    //   --bench 0  correctness pass -- dump everything, time nothing meaningful
    //   --bench N  N timed passes with dumping OFF, then one final dumping
    //              pass whose samples are discarded. The reported figures are
    //              therefore free of I/O, and the dumps are byte-identical to
    //              what a plain correctness run produces.
    for (int pass = 0; pass <= bench_runs; ++pass) {
        const bool dumping = (pass == bench_runs);
        d.enabled = dumping;
        history.clear();
        med_frames = 0;
        if (dumping) d.frame = 0;

        Bench bench;
        if (!dumping) bench.setFrames(frames.size() - 1);
        Bench::Timer pass_t;

        for (size_t f = 1; f < frames.size(); ++f) {
            d.frame = static_cast<int>(f);
            if (dumping) std::printf("frame %d\n", d.frame);

            const cv::Matx23f H =
                compute_affine(frames[f - 1], frames[f], d, dumping ? nullptr : &bench);
            const cv::Mat warped = warp(frames[f], H, d, dumping ? nullptr : &bench);

            history.push_back(warped);
            if (static_cast<int>(history.size()) > kWindow) history.erase(history.begin());
            if (static_cast<int>(history.size()) == kWindow) {
                Bench::Timer mt;
                const cv::Mat med = median_residual(frames[f], history);
                if (!dumping) bench.add("median", mt.ms());
                d.put("median_out", med);
                ++med_frames;
            }
            if (dumping) std::printf("\n");
        }

        if (!dumping) {
            bench.add("frame_total", pass_t.ms());
            bench.countPass();
            bench.report();
        } else {
            std::printf("=== done: %zu frames, %d median outputs ===\n",
                        frames.size() - 1, med_frames);
            dumped = 1;
        }
    }
    if (!dumped) std::printf("=== done ===\n");
    return 0;
}
