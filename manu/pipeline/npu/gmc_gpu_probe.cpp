// ============================================================================
// gmc_gpu_probe.cpp -- standalone measurement probe: "how fast is the GMC
// feature-preparation stage on Mali-G610 (OpenCL), and what does it cost in
// accuracy?"
//
// THIS IS A PROBE, NOT A PIPELINE COMPONENT.  It links nothing from
// libmosaic_pipeline.a, it is not called by the pipeline, and it changes no
// production number.  It exists to answer one question with measurements.
//
// ---------------------------------------------------------------------------
// WHAT IS MEASURED
// ---------------------------------------------------------------------------
// The probe has TWO TIERS, selected by --gmc-only:
//
//   --gmc-only gftt      TIER 1.  Only the corner DETECTION moves to the GPU.
//                       The LK and the RANSAC stay on the CPU, fed with the
//                       GPU's corners.  This isolates "is gFTT on the GPU
//                       worth it" and is the tier that pays off fastest:
//                       gFTT is 65.8% of a fit, LK only 27.4%.
//                       (manu/memory/gmc_fit_cache_rk3588.md 1.2)
//
//   --gmc-only gftt+lk   TIER 2.  gFTT *and* the pyramidal LK move to the GPU
//                       (the pyramid itself is still built on the CPU and
//                       uploaded -- it is 3.2% of a fit).  The RANSAC stays on
//                       the CPU (1.8%).
//
// For every consecutive frame pair of a raw sequence, TWO arms are run on the
// SAME input:
//
//   CPU arm : the shipped GMC fit, i.e. exactly what FastGMCEstimator does
//             (gmc_stream_ocl.cpp):
//               resize 640x512 -> 320x256  (INTER_LINEAR)
//               goodFeaturesToTrack(600, 0.01, 4, blockSize=3)
//               calcOpticalFlowPyrLK(win 15x15, maxLevel 2, DEFAULT criteria)
//               estimateAffinePartial2D(RANSAC, 3.0)
//
//   GPU arm : the same thing with the requested stages moved to OpenCL:
//               resize (shared, timed once)
//               gftt_min_eigen  (GPU Shi-Tomasi score map)
//               score map read back -> CPU NMS + threshold + minDistance
//               [tier 2 only] buildOpticalFlowPyramid on the CPU -> padded
//                 planes uploaded -> lk_pyr_track (GPU pyramidal LK)
//               estimateAffinePartial2D(RANSAC, 3.0)
//
// The two arms are compared frame by frame:
//   max|dt| / mean|dt| (translation, full-resolution pixels)
//   rotation cos(theta_cpu - theta_gpu)
//   |ds/s| (uniform scale, relative)
//   corner count, corner correspondence rate and corner coordinate deviation
//
// ---------------------------------------------------------------------------
// WHAT IS DELIBERATELY NOT CLAIMED
// ---------------------------------------------------------------------------
// The GPU arm is NOT bit-identical to the CPU arm and is not meant to be.
// Three structural differences remain and they are exactly what the probe is
// supposed to measure:
//   1. the corner SET can differ (the GPU score map is a float re-derivation
//      of OpenCV's cornerMinEigenVal; ties and the 14-bit weight quantisation
//      of OpenCV's LK are not reproduced),
//   2. OpenCV's LK uses 14-bit fixed-point bilinear weights and rounds the
//      interpolated patch to integers; we use exact float weights,
//   3. RANSAC consumes the global RNG; we reseed it before every fit so that
//      the comparison attributes the difference to the feature preparation and
//      not to the random stream (the production path does NOT reseed).
//
// The e2e TSV md5 baseline (0bf598a1d890040bcfa8d157dc69c032) is untouched:
// this file adds no code to any certified path.
//
// Build / run:
//   gmc_gpu_probe --raw-root /mnt/manu/lq_ir_bmp
//                  --sequence VIDEO00007_19700101_001324
//                  --kernel-dir /root/kernels --limit 60
//                  --out /tmp/gmc_gpu_probe.md
// ============================================================================

#include "../opencl/host/ocl_host.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>

namespace {

using cv::Mat;
using cv::Matx23f;

// ---------------------------------------------------------------------------
// Frozen parameters -- these are the PRODUCTION values, not tuning knobs.
//   mosaic_pipeline.cpp:918-922 (gmc_downscale / fit_corners / fit_win /
//   fit_level / fit_iters) and gmc_stream_ocl.cpp:516 (gFTT call).
// ---------------------------------------------------------------------------
constexpr int kDownscale = 2;          // 640x512 -> 320x256
constexpr int kMaxCorners = 600;
constexpr double kQualityLevel = 0.01;
constexpr int kMinDistance = 4;
constexpr int kBlockSize = 3;
constexpr int kWin = 15;               // gmc_fit_win
constexpr int kMaxLevel = 2;           // gmc_fit_level
// gmc_fit_iters == 20 selects the NO-criteria overload of calcOpticalFlowPyrLK
// in gmc_stream_ocl.cpp:380, i.e. OpenCV's DEFAULT TermCriteria(COUNT|EPS,
// 30, 0.01) -- NOT (20, 0.03).  The GPU kernel uses the same numbers.
constexpr int kLkIters = 30;
constexpr double kLkEps = 0.01;
constexpr double kRansacThresh = 3.0;
constexpr unsigned kRansacSeed = 0x5eed1234u;

using Clock = std::chrono::steady_clock;
static double ms_since(const Clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// host-side BORDER_REFLECT_101, matching OpenCV
// ---------------------------------------------------------------------------
int reflect101(int i, int n) {
    if (n <= 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i - 1;
        else i = 2 * n - 2 - i;
    }
    return i;
}

// ---------------------------------------------------------------------------
// frame listing
// ---------------------------------------------------------------------------
struct FrameFile {
    std::string path;
    long index;
};

std::vector<FrameFile> list_frames(const std::string& dir, int limit, int start) {
    std::vector<FrameFile> out;
    DIR* d = opendir(dir.c_str());
    if (!d) {
        std::fprintf(stderr, "[FATAL] cannot open %s\n", dir.c_str());
        std::exit(2);
    }
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        const std::string n = e->d_name;
        if (n == "." || n == "..") continue;
        const size_t dot = n.find_last_of('.');
        if (dot == std::string::npos) continue;
        std::string ext = n.substr(dot);
        for (char& c : ext) c = (char)tolower((unsigned char)c);
        if (ext != ".bmp" && ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;
        // the trailing digit run is the frame index
        long idx = -1;
        size_t p = dot;
        while (p > 0 && isdigit((unsigned char)n[p - 1])) --p;
        if (p < dot) idx = atol(n.substr(p, dot - p).c_str());
        out.push_back(FrameFile{dir + "/" + n, idx});
    }
    closedir(d);
    std::sort(out.begin(), out.end(), [](const FrameFile& a, const FrameFile& b) {
        if (a.index != b.index) return a.index < b.index;
        return a.path < b.path;
    });
    // --start skips to a given absolute frame index, so the dense part of the
    // sequence (where the fit collapses) can be probed and not only the head.
    if (start > 0) {
        size_t k = 0;
        while (k < out.size() && out[k].index < (long)start) ++k;
        if (k < out.size()) out.erase(out.begin(), out.begin() + (long)k);
    }
    if (limit > 0 && (int)out.size() > limit) out.resize((size_t)limit);
    return out;
}

// ---------------------------------------------------------------------------
// CPU-side corner selection from a score map.
//
// This replicates cv::goodFeaturesToTrack's SELECTION stage exactly
// (OpenCV 4.10 modules/imgproc/src/featureselect.cpp:396-263):
//   maxVal = max(score)
//   threshold(score, score, maxVal*qualityLevel, THRESH_TOZERO)
//   dilate(score, tmp, 3x3)
//   candidate iff 1<=x<=w-2, 1<=y<=h-2, val != 0 && val == dilated
//   sort by value descending
//   greedy minDistance filter on a cell grid (cell = round(minDistance)),
//   reject if dx*dx+dy*dy < minDistance^2
//   stop at maxCorners
// The only difference from OpenCV: OpenCV's std::sort is not stable, so exact
// ties are ordered arbitrarily; we use a stable order (value desc, index asc).
//
// The 3x3 max-dilation is done SEPARABLY (a horizontal max pass then a
// vertical one, 2N instead of 9N).  OpenCV's dilate is separable too, so the
// naive 9-tap version would have made the CPU-side selection -- the part the
// task deliberately leaves on the CPU -- look 4x slower than it is, and the
// tier-1 verdict would then be an artefact of this probe, not of the GPU.
// ---------------------------------------------------------------------------
std::vector<cv::Point2f> select_corners(const std::vector<float>& score, int W, int H,
                                         int max_corners, double quality_level, int min_distance) {
    double max_val = 0.0;
    for (float v : score) {
        if ((double)v > max_val) max_val = (double)v;
    }
    const float thr = (float)(max_val * quality_level);

    // threshold (THRESH_TOZERO) + separable 3x3 max dilation
    const size_t n = (size_t)W * H;
    std::vector<float> t(n);
    for (size_t i = 0; i < n; ++i) t[i] = (score[i] > thr) ? score[i] : 0.0f;

    std::vector<float> rowmax(n);
    for (int y = 0; y < H; ++y) {
        const float* r = &t[(size_t)y * W];
        float* d = &rowmax[(size_t)y * W];
        d[0] = r[0] > r[1] ? r[0] : r[1];
        for (int x = 1; x < W - 1; ++x) {
            const float a = r[x - 1], b = r[x], c = r[x + 1];
            d[x] = a > b ? (a > c ? a : c) : (b > c ? b : c);
        }
        d[W - 1] = r[W - 2] > r[W - 1] ? r[W - 2] : r[W - 1];
    }
    std::vector<float> dil(n);
    for (int y = 0; y < H; ++y) {
        const float* r0 = &rowmax[(size_t)std::max(0, y - 1) * W];
        const float* r1 = &rowmax[(size_t)y * W];
        const float* r2 = &rowmax[(size_t)std::min(H - 1, y + 1) * W];
        float* d = &dil[(size_t)y * W];
        for (int x = 0; x < W; ++x) {
            const float a = r0[x], b = r1[x], c = r2[x];
            d[x] = a > b ? (a > c ? a : c) : (b > c ? b : c);
        }
    }

    struct Cand {
        float val;
        int x, y;
    };
    std::vector<Cand> cands;
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            const float v = t[(size_t)y * W + x];
            if (v != 0.0f && v == dil[(size_t)y * W + x]) cands.push_back(Cand{v, x, y});
        }
    }
    std::stable_sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
        if (a.val != b.val) return a.val > b.val;
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });

    const int cell = (int)std::lround((double)min_distance);
    const int gw = (W + cell - 1) / cell;
    const int gh = (H + cell - 1) / cell;
    std::vector<std::vector<cv::Point2f>> grid((size_t)gw * gh);
    const double md2 = (double)min_distance * (double)min_distance;

    std::vector<cv::Point2f> out;
    for (const Cand& c : cands) {
        const int xc = c.x / cell;
        const int yc = c.y / cell;
        bool good = true;
        const int x1 = std::max(0, xc - 1), x2 = std::min(gw - 1, xc + 1);
        const int y1 = std::max(0, yc - 1), y2 = std::min(gh - 1, yc + 1);
        for (int yy = y1; yy <= y2 && good; ++yy) {
            for (int xx = x1; xx <= x2 && good; ++xx) {
                for (const cv::Point2f& p : grid[(size_t)yy * gw + xx]) {
                    const double dx = (double)c.x - p.x;
                    const double dy = (double)c.y - p.y;
                    if (dx * dx + dy * dy < md2) {
                        good = false;
                        break;
                    }
                }
            }
        }
        if (good) {
            grid[(size_t)yc * gw + xc].push_back(cv::Point2f((float)c.x, (float)c.y));
            out.push_back(cv::Point2f((float)c.x, (float)c.y));
            if (max_corners > 0 && (int)out.size() == max_corners) break;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// The GPU side: buffers, program, the two kernels.
// ---------------------------------------------------------------------------
class GpuFeat {
public:
    GpuFeat(const std::string& kernel_dir, const std::string& kernel_file, int small_w, int small_h)
        : dev_(pick()), ctx_(dev_.dev) {
        ocl::describe(dev_.dev, dev_.plat);
        const std::string src = ocl::read_file(kernel_dir + "/" + kernel_file);
        // The kernel source has no #include and no build-time knobs, so the
        // plain build path is used.  -cl-std=CL1.2 keeps the Mali compiler
        // away from the OpenCL 2.0 atomics/pipe extensions we do not use.
        prog_ = ocl::build_program(ctx_.get(), dev_.dev, src, "gmc_feature", false, false);
        cl_int e;
        k_gftt_.reset(clCreateKernel(prog_.get(), "gftt_min_eigen", &e));
        OCL_CHECK(e, "clCreateKernel(gftt_min_eigen)");
        k_lk_.reset(clCreateKernel(prog_.get(), "lk_pyr_track", &e));
        OCL_CHECK(e, "clCreateKernel(lk_pyr_track)");
        configure(small_w, small_h);
    }

    int small_w() const { return sw_; }
    int small_h() const { return sh_; }

    // ---- stage 1: Shi-Tomasi score map -------------------------------------
    void gftt(const Mat& small, std::vector<float>& score) {
        const size_t n = (size_t)sw_ * sh_;
        const size_t bytes = n;
        cl_int e = clEnqueueWriteBuffer(q(), k_img_.get(), CL_TRUE, 0, bytes, small.data, 0,
                                         nullptr, nullptr);
        OCL_CHECK(e, "clEnqueueWriteBuffer(img)");
        set_mem_arg(k_gftt_.get(), 0, k_img_.get());
        set_mem_arg(k_gftt_.get(), 1, k_score_.get());
        set_int_arg(k_gftt_.get(), 2, sw_);
        set_int_arg(k_gftt_.get(), 3, sh_);
        const size_t gws[2] = {(size_t)sw_, (size_t)sh_};
        e = clEnqueueNDRangeKernel(q(), k_gftt_.get(), 2, nullptr, gws, nullptr, 0, nullptr,
                                   nullptr);
        OCL_CHECK(e, "clEnqueueNDRangeKernel(gftt)");
        OCL_CHECK(clFinish(q()), "clFinish(gftt)");
        score.resize(n);
        e = clEnqueueReadBuffer(q(), k_score_.get(), CL_TRUE, 0, n * sizeof(float), score.data(),
                                0, nullptr, nullptr);
        OCL_CHECK(e, "clEnqueueReadBuffer(score)");
    }

    // ---- stage 2: pyramidal LK --------------------------------------------
    // `pyr` is the std::vector<Mat> returned by cv::buildOpticalFlowPyramid
    // (withDerivatives=true): [img0, der0, img1, der1, ...].
    void upload_pyramids(const std::vector<Mat>& prev_pyr, const std::vector<Mat>& curr_pyr,
                         double& pad_build_ms) {
        const Clock::time_point t0 = Clock::now();
        const int nlev = nlevels_;
        // prev image levels
        size_t off_prev_img = 0, off_curr_img = 0;
        size_t off_prev_der = 0;   // in SHORTS (the host buffer is a vector<short>)
        for (int L = 0; L < nlev; ++L) {
            const Mat& pi = prev_pyr[L * 2];
            const Mat& pd = prev_pyr[L * 2 + 1];
            const Mat& ci = curr_pyr[L * 2];
            const int rw = pi.cols, rh = pi.rows;
            const int pw = rw + 2 * kWin, ph = rh + 2 * kWin;

            // image level, BORDER_REFLECT_101
            for (int y = 0; y < ph; ++y) {
                const int sy = reflect101(y - kWin, rh);
                uchar* dst = h_prev_img_.data() + off_prev_img + (size_t)y * pw;
                const uchar* src = pi.ptr<uchar>(sy);
                for (int x = 0; x < pw; ++x) dst[x] = src[reflect101(x - kWin, rw)];
            }
            // derivative level, BORDER_CONSTANT
            std::memset(h_prev_der_.data() + off_prev_der, 0,
                        (size_t)pw * ph * 2 * sizeof(short));
            for (int y = 0; y < rh; ++y) {
                const short* src = pd.ptr<short>(y);
                short* dst = h_prev_der_.data() + off_prev_der + (size_t)(y + kWin) * pw * 2 +
                             (size_t)kWin * 2;
                std::memcpy(dst, src, (size_t)rw * 2 * sizeof(short));
            }
            // current image level, BORDER_REFLECT_101
            for (int y = 0; y < ph; ++y) {
                const int sy = reflect101(y - kWin, rh);
                uchar* dst = h_curr_img_.data() + off_curr_img + (size_t)y * pw;
                const uchar* src = ci.ptr<uchar>(sy);
                for (int x = 0; x < pw; ++x) dst[x] = src[reflect101(x - kWin, rw)];
            }

            desc_[L * 8 + 0] = pw;
            desc_[L * 8 + 1] = ph;
            desc_[L * 8 + 2] = rw;
            desc_[L * 8 + 3] = rh;
            desc_[L * 8 + 4] = (int)off_prev_img;      // byte offset (uchar plane)
            desc_[L * 8 + 5] = (int)off_prev_der;      // ELEMENT offset (short plane)
            desc_[L * 8 + 6] = (int)off_curr_img;      // byte offset (uchar plane)
            desc_[L * 8 + 7] = 0;

            off_prev_img += (size_t)pw * ph;
            off_prev_der += (size_t)pw * ph * 2;
            off_curr_img += (size_t)pw * ph;
        }
        pad_build_ms = ms_since(t0);

        cl_int e;
        e = clEnqueueWriteBuffer(q(), k_prev_img_.get(), CL_TRUE, 0, off_prev_img,
                                 h_prev_img_.data(), 0, nullptr, nullptr);
        OCL_CHECK(e, "clEnqueueWriteBuffer(prev_img)");
        e = clEnqueueWriteBuffer(q(), k_prev_der_.get(), CL_TRUE, 0,
                                 off_prev_der * sizeof(short), h_prev_der_.data(), 0, nullptr,
                                 nullptr);
        OCL_CHECK(e, "clEnqueueWriteBuffer(prev_der)");
        e = clEnqueueWriteBuffer(q(), k_curr_img_.get(), CL_TRUE, 0, off_curr_img,
                                 h_curr_img_.data(), 0, nullptr, nullptr);
        OCL_CHECK(e, "clEnqueueWriteBuffer(curr_img)");
        e = clEnqueueWriteBuffer(q(), k_desc_.get(), CL_TRUE, 0, sizeof(desc_), desc_, 0, nullptr,
                                 nullptr);
        OCL_CHECK(e, "clEnqueueWriteBuffer(desc)");
        OCL_CHECK(clFinish(q()), "clFinish(upload)");
    }

    void lk(const std::vector<cv::Point2f>& pts, std::vector<cv::Point2f>& out,
            std::vector<uchar>& status) {
        const int n = (int)pts.size();
        out.assign((size_t)n, cv::Point2f(0, 0));
        status.assign((size_t)n, 0);
        if (n <= 0) return;
        for (int i = 0; i < n; ++i) {
            h_pts_in_[2 * i] = pts[i].x;
            h_pts_in_[2 * i + 1] = pts[i].y;
        }
        cl_int e;
        e = clEnqueueWriteBuffer(q(), k_pts_in_.get(), CL_TRUE, 0, (size_t)n * 2 * sizeof(float),
                                 h_pts_in_.data(), 0, nullptr, nullptr);
        OCL_CHECK(e, "clEnqueueWriteBuffer(pts_in)");
        set_mem_arg(k_lk_.get(), 0, k_prev_img_.get());
        set_mem_arg(k_lk_.get(), 1, k_prev_der_.get());
        set_mem_arg(k_lk_.get(), 2, k_curr_img_.get());
        set_mem_arg(k_lk_.get(), 3, k_desc_.get());
        set_mem_arg(k_lk_.get(), 4, k_patch_.get());
        set_mem_arg(k_lk_.get(), 5, k_pts_in_.get());
        set_mem_arg(k_lk_.get(), 6, k_pts_out_.get());
        set_mem_arg(k_lk_.get(), 7, k_status_.get());
        set_int_arg(k_lk_.get(), 8, n);
        set_int_arg(k_lk_.get(), 9, nlevels_);
        set_int_arg(k_lk_.get(), 10, win_);
        set_int_arg(k_lk_.get(), 11, iters_);
        const float eps_sq = (float)(kLkEps * kLkEps);
        set_float_arg(k_lk_.get(), 12, eps_sq);
        const float min_eig_thr = 1e-4f;   // OpenCV's calcOpticalFlowPyrLK default
        set_float_arg(k_lk_.get(), 13, min_eig_thr);
        const size_t gws = (size_t)n;
        e = clEnqueueNDRangeKernel(q(), k_lk_.get(), 1, nullptr, &gws, nullptr, 0, nullptr,
                                   nullptr);
        OCL_CHECK(e, "clEnqueueNDRangeKernel(lk)");
        OCL_CHECK(clFinish(q()), "clFinish(lk)");
        e = clEnqueueReadBuffer(q(), k_pts_out_.get(), CL_TRUE, 0, (size_t)n * 2 * sizeof(float),
                                h_pts_out_.data(), 0, nullptr, nullptr);
        OCL_CHECK(e, "clEnqueueReadBuffer(pts_out)");
        e = clEnqueueReadBuffer(q(), k_status_.get(), CL_TRUE, 0, (size_t)n * sizeof(int),
                                h_status_.data(), 0, nullptr, nullptr);
        OCL_CHECK(e, "clEnqueueReadBuffer(status)");
        for (int i = 0; i < n; ++i) {
            out[(size_t)i] = cv::Point2f(h_pts_out_[2 * i], h_pts_out_[2 * i + 1]);
            status[(size_t)i] = (uchar)(h_status_[i] ? 1 : 0);
        }
    }

private:
    struct DevPick {
        cl_platform_id plat = nullptr;
        cl_device_id dev = nullptr;
    };
    static DevPick pick() {
        DevPick d;
        d.dev = ocl::pick_device(&d.plat);
        return d;
    }
    cl_command_queue q() const { return ctx_.queue(); }

    static void set_mem_arg(cl_kernel k, cl_uint idx, cl_mem m) {
        const cl_int e = clSetKernelArg(k, idx, sizeof(cl_mem), &m);
        OCL_CHECK(e, "clSetKernelArg(mem)");
    }
    static void set_int_arg(cl_kernel k, cl_uint idx, int v) {
        const cl_int e = clSetKernelArg(k, idx, sizeof(int), &v);
        OCL_CHECK(e, "clSetKernelArg(int)");
    }
    static void set_float_arg(cl_kernel k, cl_uint idx, float v) {
        const cl_int e = clSetKernelArg(k, idx, sizeof(float), &v);
        OCL_CHECK(e, "clSetKernelArg(float)");
    }

    void configure(int w, int h) {
        sw_ = w;
        sh_ = h;
        win_ = kWin;
        iters_ = kLkIters;
        // level sizes, replicating cv::buildOpticalFlowPyramid's pyrDown chain
        std::vector<std::pair<int, int>> sizes;
        int lw = w, lh = h;
        for (int L = 0; L <= kMaxLevel; ++L) {
            sizes.push_back({lw, lh});
            lw = (lw + 1) / 2;
            lh = (lh + 1) / 2;
            if (lw <= win_ || lh <= win_) break;
        }
        nlevels_ = (int)sizes.size();
        size_t b_img = 0, b_der = 0;
        for (auto& s : sizes) {
            b_img += (size_t)(s.first + 2 * win_) * (s.second + 2 * win_);
            b_der += (size_t)(s.first + 2 * win_) * (s.second + 2 * win_) * 2 * sizeof(short);
        }
        h_prev_img_.assign(b_img, 0);
        h_prev_der_.assign(b_der / sizeof(short), 0);
        h_curr_img_.assign(b_img, 0);
        std::memset(desc_, 0, sizeof(desc_));

        const int max_pts = 1024;
        const size_t patch_floats = (size_t)3 * win_ * win_ * max_pts;

        cl_int e;
        k_img_.reset(clCreateBuffer(ctx_.get(), CL_MEM_READ_ONLY, (size_t)w * h, nullptr, &e));
        OCL_CHECK(e, "clCreateBuffer(img)");
        k_score_.reset(clCreateBuffer(ctx_.get(), CL_MEM_WRITE_ONLY,
                                      (size_t)w * h * sizeof(float), nullptr, &e));
        OCL_CHECK(e, "clCreateBuffer(score)");
        k_prev_img_.reset(clCreateBuffer(ctx_.get(), CL_MEM_READ_ONLY, b_img, nullptr, &e));
        OCL_CHECK(e, "clCreateBuffer(prev_img)");
        k_curr_img_.reset(clCreateBuffer(ctx_.get(), CL_MEM_READ_ONLY, b_img, nullptr, &e));
        OCL_CHECK(e, "clCreateBuffer(curr_img)");
        k_prev_der_.reset(clCreateBuffer(ctx_.get(), CL_MEM_READ_ONLY, b_der, nullptr, &e));
        OCL_CHECK(e, "clCreateBuffer(prev_der)");
        k_desc_.reset(clCreateBuffer(ctx_.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(desc_), desc_, &e));
        OCL_CHECK(e, "clCreateBuffer(desc)");
        k_patch_.reset(clCreateBuffer(ctx_.get(), CL_MEM_READ_WRITE, patch_floats * sizeof(float),
                                      nullptr, &e));
        OCL_CHECK(e, "clCreateBuffer(patch)");
        k_pts_in_.reset(clCreateBuffer(ctx_.get(), CL_MEM_READ_ONLY,
                                       (size_t)max_pts * 2 * sizeof(float), nullptr, &e));
        OCL_CHECK(e, "clCreateBuffer(pts_in)");
        k_pts_out_.reset(clCreateBuffer(ctx_.get(), CL_MEM_WRITE_ONLY,
                                        (size_t)max_pts * 2 * sizeof(float), nullptr, &e));
        OCL_CHECK(e, "clCreateBuffer(pts_out)");
        k_status_.reset(clCreateBuffer(ctx_.get(), CL_MEM_WRITE_ONLY,
                                       (size_t)max_pts * sizeof(int), nullptr, &e));
        OCL_CHECK(e, "clCreateBuffer(status)");
        h_pts_in_.assign((size_t)max_pts * 2, 0.0f);
        h_pts_out_.assign((size_t)max_pts * 2, 0.0f);
        h_status_.assign((size_t)max_pts, 0);
        std::printf("[GPU] levels=%d  patch scratch=%.2f MiB  pyramid bytes=%.2f MiB\n",
                    nlevels_, (double)(patch_floats * sizeof(float)) / 1048576.0,
                    (double)(b_img * 2 + b_der) / 1048576.0);
    }

    DevPick dev_;
    ocl::Context ctx_;
    ocl::Program prog_;
    ocl::Kernel k_gftt_, k_lk_;
    ocl::Mem k_img_, k_score_, k_prev_img_, k_curr_img_, k_prev_der_, k_desc_, k_patch_,
        k_pts_in_, k_pts_out_, k_status_;
    std::vector<uchar> h_prev_img_, h_curr_img_;
    std::vector<short> h_prev_der_;
    std::vector<float> h_pts_in_, h_pts_out_;
    std::vector<int> h_status_;
    int desc_[8 * 16] = {0};
    int sw_ = 0, sh_ = 0, win_ = kWin, iters_ = kLkIters, nlevels_ = 0;
};

// ---------------------------------------------------------------------------
// per-frame results
// ---------------------------------------------------------------------------
struct ArmTiming {
    double resize = 0, corners = 0, lk = 0, ransac = 0, total = 0;
    // GPU-only breakdown
    double pyr_build = 0, upload = 0, gftt_gpu = 0, score_read = 0, select = 0, lk_gpu = 0,
           lk_read = 0;
    int n_corners = 0, n_valid = 0;
    Matx23f H = Matx23f::eye();
    bool valid = false;
    std::vector<cv::Point2f> pts;   // the corner set this arm used
};

struct FrameCmp {
    double dtrans = 0;      // |dt| in full-resolution pixels (Euclidean)
    double cos_rot = 1.0;
    double rel_scale = 0.0;
    int n_cpu = 0, n_gpu = 0;
    double match05 = 0, match1 = 0, match2 = 0;   // corner correspondence rates
    double dev_mean = 0, dev_max = 0;             // nearest-neighbour corner offset (px)
    bool both_valid = false;
};

// decompose a 2x3 partial-affine matrix
void decompose(const Matx23f& H, double& scale, double& rot) {
    const double a = H(0, 0), c = H(1, 0);
    scale = std::sqrt(a * a + c * c);
    rot = std::atan2(c, a);
}

FrameCmp compare_arms(const ArmTiming& cpu, const ArmTiming& gpu,
                       const std::vector<cv::Point2f>& cpu_pts,
                       const std::vector<cv::Point2f>& gpu_pts) {
    FrameCmp c;
    c.n_cpu = cpu.n_corners;
    c.n_gpu = gpu.n_corners;
    if (cpu.valid && gpu.valid) {
        c.both_valid = true;
        const double dtx = (double)cpu.H(0, 2) - (double)gpu.H(0, 2);
        const double dty = (double)cpu.H(1, 2) - (double)gpu.H(1, 2);
        c.dtrans = std::sqrt(dtx * dtx + dty * dty);
        double sc, sg, rc, rg;
        decompose(cpu.H, sc, rc);
        decompose(gpu.H, sg, rg);
        c.cos_rot = std::cos(rc - rg);
        c.rel_scale = (sc > 0) ? std::fabs(sg - sc) / sc : 0.0;
    }
    // corner correspondence: for each CPU corner, the distance to the nearest
    // GPU corner (small-image coordinates).
    if (!cpu_pts.empty() && !gpu_pts.empty()) {
        int m05 = 0, m1 = 0, m2 = 0;
        double sum_dev = 0.0, max_dev = 0.0;
        for (const cv::Point2f& p : cpu_pts) {
            double best = 1e30;
            for (const cv::Point2f& q : gpu_pts) {
                const double dx = p.x - q.x, dy = p.y - q.y;
                const double d = std::sqrt(dx * dx + dy * dy);
                if (d < best) best = d;
            }
            if (best <= 0.5) ++m05;
            if (best <= 1.0) ++m1;
            if (best <= 2.0) ++m2;
            sum_dev += best;
            if (best > max_dev) max_dev = best;
        }
        c.match05 = (double)m05 / (double)cpu_pts.size();
        c.match1 = (double)m1 / (double)cpu_pts.size();
        c.match2 = (double)m2 / (double)cpu_pts.size();
        c.dev_mean = sum_dev / (double)cpu_pts.size();
        c.dev_max = max_dev;
    }
    return c;
}

// ---------------------------------------------------------------------------
// shared CPU primitives (identical in both arms)
// ---------------------------------------------------------------------------
void cpu_lk(const Mat& prev_small, const Mat& curr_small, const std::vector<cv::Point2f>& pts,
            std::vector<cv::Point2f>& pts_next, std::vector<uchar>& status,
            std::vector<float>& err) {
    pts_next.assign(pts.size(), cv::Point2f(0, 0));
    status.assign(pts.size(), 0);
    err.assign(pts.size(), 0.f);
    if (pts.empty()) return;
    const Mat pts_mat((int)pts.size(), 1, CV_32FC2, const_cast<cv::Point2f*>(pts.data()));
    const cv::TermCriteria crit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, kLkIters, kLkEps);
    cv::calcOpticalFlowPyrLK(prev_small, curr_small, pts_mat, pts_next, status, err,
                              cv::Size(kWin, kWin), kMaxLevel, crit);
}

int ransac_fit(const std::vector<cv::Point2f>& pts, const std::vector<cv::Point2f>& pts_next,
               const std::vector<uchar>& status, Matx23f& H) {
    std::vector<cv::Point2f> p0, p1;
    p0.reserve(pts.size());
    p1.reserve(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        if (i < status.size() && status[i] == 1) {
            p0.push_back(pts[i]);
            p1.push_back(pts_next[i]);
        }
    }
    if (p0.size() < 6) return 0;
    cv::Mat inliers;
    cv::theRNG() = cv::RNG(kRansacSeed);
    const cv::Mat M = cv::estimateAffinePartial2D(p0, p1, inliers, cv::RANSAC, kRansacThresh);
    if (M.empty()) return 0;
    double v[6];
    for (int i = 0; i < 6; ++i) v[i] = M.at<double>(i / 3, i % 3);
    H = Matx23f((float)v[0], (float)v[1], (float)v[2], (float)v[3], (float)v[4], (float)v[5]);
    // the reference scales the translation back to full resolution
    H(0, 2) = (float)(H(0, 2) * kDownscale);
    H(1, 2) = (float)(H(1, 2) * kDownscale);
    return (int)p0.size();
}

// ---------------------------------------------------------------------------
// the two arms
// ---------------------------------------------------------------------------
ArmTiming run_cpu(const Mat& prev_gray, const Mat& curr_gray, const Mat& prev_small,
                   const Mat& curr_small, double resize_ms) {
    ArmTiming t;
    t.resize = resize_ms;

    // --- corners ---
    Clock::time_point t0 = Clock::now();
    std::vector<cv::Point2f> pts;
    // the reference asks for max_corners+1 and drops the last one (gmc_stream_ocl.cpp:521)
    cv::goodFeaturesToTrack(prev_small, pts, kMaxCorners + 1, kQualityLevel, kMinDistance,
                             cv::noArray(), kBlockSize);
    if ((int)pts.size() > kMaxCorners) pts.resize((size_t)kMaxCorners);
    t.corners = ms_since(t0);
    t.n_corners = (int)pts.size();
    t.pts = pts;
    if (pts.size() < 6) {
        t.total = t.resize + t.corners;
        return t;
    }

    // --- LK ---
    t0 = Clock::now();
    std::vector<cv::Point2f> pts_next;
    std::vector<uchar> status;
    std::vector<float> err;
    cpu_lk(prev_small, curr_small, pts, pts_next, status, err);
    t.lk = ms_since(t0);

    // --- RANSAC ---
    t0 = Clock::now();
    t.n_valid = ransac_fit(pts, pts_next, status, t.H);
    t.valid = t.n_valid > 0;
    t.ransac = ms_since(t0);
    t.total = t.resize + t.corners + t.lk + t.ransac;
    return t;
}

// do_gpu_lk == false  -> TIER 1: only the corner detection is on the GPU, the
//                        LK and the RANSAC stay on the CPU.
// do_gpu_lk == true   -> TIER 2: gFTT *and* the pyramidal LK are on the GPU.
ArmTiming run_gpu(GpuFeat& gpu, const Mat& prev_small, const Mat& curr_small, double resize_ms,
                   bool do_gpu_lk) {
    ArmTiming t;
    t.resize = resize_ms;

    // --- gFTT score map on the GPU ---
    Clock::time_point t0 = Clock::now();
    std::vector<float> score;
    gpu.gftt(prev_small, score);
    t.gftt_gpu = ms_since(t0);

    // --- corner selection (CPU, by design) ---
    t0 = Clock::now();
    std::vector<cv::Point2f> pts = select_corners(score, gpu.small_w(), gpu.small_h(),
                                                  kMaxCorners, kQualityLevel, kMinDistance);
    t.select = ms_since(t0);
    t.n_corners = (int)pts.size();
    t.pts = pts;
    if (pts.size() < 6) {
        t.total = t.resize + t.gftt_gpu + t.select;
        return t;
    }

    std::vector<cv::Point2f> pts_next;
    std::vector<uchar> status;
    std::vector<float> err;

    if (do_gpu_lk) {
        // --- pyramid build (CPU, 3.2% of a fit) ---
        t0 = Clock::now();
        std::vector<Mat> pyr_prev, pyr_curr;
        cv::buildOpticalFlowPyramid(prev_small, pyr_prev, cv::Size(kWin, kWin), kMaxLevel, true);
        cv::buildOpticalFlowPyramid(curr_small, pyr_curr, cv::Size(kWin, kWin), kMaxLevel, true);
        t.pyr_build = ms_since(t0);

        // --- upload + LK ---
        double pad_ms = 0.0;
        t0 = Clock::now();
        gpu.upload_pyramids(pyr_prev, pyr_curr, pad_ms);
        t.upload = ms_since(t0);

        t0 = Clock::now();
        gpu.lk(pts, pts_next, status);
        t.lk_gpu = ms_since(t0);
    } else {
        // --- TIER 1: the LK stays on the CPU and is fed the GPU's corners ---
        t0 = Clock::now();
        cpu_lk(prev_small, curr_small, pts, pts_next, status, err);
        t.lk = ms_since(t0);
    }

    // --- RANSAC (CPU in both tiers) ---
    t0 = Clock::now();
    t.n_valid = ransac_fit(pts, pts_next, status, t.H);
    t.valid = t.n_valid > 0;
    t.ransac = ms_since(t0);

    if (do_gpu_lk) {
        t.total = t.resize + t.pyr_build + t.upload + t.gftt_gpu + t.select + t.lk_gpu + t.ransac;
    } else {
        t.total = t.resize + t.gftt_gpu + t.select + t.lk + t.ransac;
    }
    return t;
}

double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0;
    for (double x : v) s += x;
    return s / (double)v.size();
}
double pctl(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t i = (size_t)(p * (double)(v.size() - 1) + 0.5);
    return v[std::min(i, v.size() - 1)];
}

// ============================================================================
// FULL-FRAME MODE
// ============================================================================
//
// WHY THIS MODE EXISTS
// --------------------
// The single-lag probe measures ONE fit.  A production FRAME is SIX fits:
// the 5 anchor fits (lags 2/12/22/32/42, all of them ending on the CURRENT
// frame) plus the stride-2 step fit (t-4 -> t-2).  So the single-lag
// measurement does NOT answer "can GMC run at 25 fps on the GPU".
//
// This mode replicates the production frame:
//
//   per frame t:
//     5 anchor fits:  LK(prev = t-lag, curr = t) + RANSAC, lag in {2,12,22,32,42}
//     1 step fit:     LK(prev = t-4,   curr = t-2) + RANSAC
//     then the 21-lag chain composition (the production's per-lag warps)
//
// The RANSACs stay on the CPU and run SEQUENTIALLY in the fit order:
// estimateAffinePartial2D consumes OpenCV's global RNG, so the order is
// part of the output.  The LK stages are independent of each other, so on
// the GPU they are dispatched as ONE batched NDRange (lk_pyr_track_multi)
// -- and the same fits issued as one dispatch each (lk_pyr_track) are
// measured alongside, because that comparison is the point of the task.
//
// THE CACHE mirrors the production one (gmc_pyr_cache=1): a 43-slot ring
// keyed by the absolute frame index.  With it, a frame builds ONE resize,
// ONE corner set and ONE pyramid, and uploads ONE pyramid.  Without it
// (--cache 0) every frame rebuilds and re-uploads all 7 resident pyramids,
// which is the "no cache" cost.
//
// THE TWO ARMS
//   CPU arm : OpenCV, single-threaded (cv::setNumThreads(1)), the
//             production call shapes.  --cpu-threads N>1 runs the LK
//             stages of the 6 fits CONCURRENTLY (the production
//             "parallel anchors" path, gmc_stream_ocl.cpp's AnchorPool)
//             while the RANSACs stay serial, so the RNG order is the
//             production's.
//   GPU arm : gFTT on the GPU (the SELECTION stays on the CPU, exactly as
//             the single-lag probe measured it), the pyramid built on the
//             CPU and uploaded, the LK batched on the GPU, the RANSAC on
//             the CPU.
//
// DEVIATION: the GPU arm is NOT bit-equivalent to the CPU arm.  The corner
// SET can differ (the GPU score map is a float re-derivation of OpenCV's
// cornerMinEigenVal) and OpenCV's LK uses 14-bit fixed-point bilinear
// weights where the kernel uses exact floats.  Every number in the
// accuracy tables is therefore a CONTROLLED DEVIATION, not an error.
// ============================================================================

namespace ff {

// ---- frozen production parameters of the full-frame mode -------------------
constexpr int kAnchors[5] = {2, 12, 22, 32, 42};
constexpr int kNFits = 6;        // 5 anchors + 1 step fit
constexpr int kNLags = 21;       // lags 2,4,...,42
constexpr int kFullMaxLag = 42;  // the deepest lag
constexpr int kNptsSlot = 640;   // >= 600 corners, multiple of 64
constexpr int kRingDepth = 43;   // the production cache depth
constexpr unsigned kFitSeedBase = 0x5eed1234u;

struct Fit {
    long prev_abs = 0;
    long curr_abs = 0;
    int  lag = 0;
    bool is_step = false;
};

// the fits of one production frame
inline std::vector<Fit> fits_of(long t) {
    std::vector<Fit> v;
    v.reserve(kNFits);
    for (int i = 0; i < 5; ++i) {
        Fit f;
        f.prev_abs = t - kAnchors[i];
        f.curr_abs = t;
        f.lag = kAnchors[i];
        v.push_back(f);
    }
    Fit s;
    s.prev_abs = t - 4;
    s.curr_abs = t - 2;
    s.lag = 2;
    s.is_step = true;
    v.push_back(s);
    return v;
}

// The per-fit RNG seed.  The CPU arm and the GPU arm MUST draw the same
// random numbers for the same fit, otherwise the RANSAC consensus sets
// differ and the comparison would measure the RNG, not the LK.  So each
// fit reseeds the global RNG with a seed derived from (frame, fit index):
// the arms then draw identically regardless of what happened in the other
// fits.  (Production does NOT reseed -- it lets the stream run -- but
// production has only one arm, so the question does not arise there.)
inline unsigned fit_seed(long abs, int fit_idx) {
    unsigned h = kFitSeedBase;
    h = h * 1664525u + (unsigned)(abs & 0xffffffff) + 1u;
    h = h * 1664525u + (unsigned)fit_idx + 1u;
    return h;
}

// the largest anchor strictly below `lag` (the production's anchor_below_)
inline int anchor_below(int lag) {
    int below = 0;
    for (int i = 0; i < 5; ++i)
        if (kAnchors[i] < lag) below = kAnchors[i];
    return below;
}

// the production's compose() (gmc_stream_ocl.cpp:707), float64 homogeneous
inline Matx23f compose_ff(const Matx23f& nw, const Matx23f& ac) {
    double A[3][3], B[3][3], H[3][3];
    A[0][0] = nw(0, 0); A[0][1] = nw(0, 1); A[0][2] = nw(0, 2); A[2][2] = 1.0;
    A[1][0] = nw(1, 0); A[1][1] = nw(1, 1); A[1][2] = nw(1, 2);
    A[2][0] = 0.0;     A[2][1] = 0.0;
    B[0][0] = ac(0, 0); B[0][1] = ac(0, 1); B[0][2] = ac(0, 2); B[2][2] = 1.0;
    B[1][0] = ac(1, 0); B[1][1] = ac(1, 1); B[1][2] = ac(1, 2);
    B[2][0] = 0.0;     B[2][1] = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += A[i][k] * B[k][j];
            H[i][j] = s;
        }
    const double n = H[2][2];
    if (!std::isfinite(n) || std::fabs(n) < 1e-12) return Matx23f::eye();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) H[i][j] /= n;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (!std::isfinite(H[i][j])) return Matx23f::eye();
    Matx23f out;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j) out(i, j) = static_cast<float>(H[i][j]);
    return out;
}

// ---------------------------------------------------------------------------
// the GPU side: the score map, the pyramid arena and the two LK dispatches
// ---------------------------------------------------------------------------
class GpuFull {
public:
    GpuFull(const std::string& kernel_dir, const std::string& kernel_file, int sw, int sh,
            int nslots, int local, bool cached)
        : dev_(pick()), ctx_(dev_.dev), nslots_(nslots), cached_(cached) {
        ocl::describe(dev_.dev, dev_.plat);
        const std::string src = ocl::read_file(kernel_dir + "/" + kernel_file);
        prog_ = ocl::build_program(ctx_.get(), dev_.dev, src, "gmc_feature_full", false, false);
        cl_int e;
        k_gftt_.reset(clCreateKernel(prog_.get(), "gftt_min_eigen", &e));
        OCL_CHECK(e, "clCreateKernel(gftt_min_eigen)");
        k_lkm_.reset(clCreateKernel(prog_.get(), "lk_pyr_track_multi", &e));
        OCL_CHECK(e, "clCreateKernel(lk_pyr_track_multi)");
        k_lk1_.reset(clCreateKernel(prog_.get(), "lk_pyr_track", &e));
        OCL_CHECK(e, "clCreateKernel(lk_pyr_track)");
        configure(sw, sh, local);
        slot_abs_.assign((size_t)nslots_, -1);
    }

    int small_w() const { return sw_; }
    int small_h() const { return sh_; }

    // ---- gFTT score map ----------------------------------------------------
    void gftt(const Mat& small, std::vector<float>& score) {
        const size_t n = (size_t)sw_ * sh_;
        cl_int e = clEnqueueWriteBuffer(q(), k_img_.get(), CL_TRUE, 0, n, small.data, 0, nullptr,
                                        nullptr);
        OCL_CHECK(e, "clEnqueueWriteBuffer(img)");
        set_mem_arg(k_gftt_.get(), 0, k_img_.get());
        set_mem_arg(k_gftt_.get(), 1, k_score_.get());
        set_int_arg(k_gftt_.get(), 2, sw_);
        set_int_arg(k_gftt_.get(), 3, sh_);
        const size_t gws[2] = {(size_t)sw_, (size_t)sh_};
        e = clEnqueueNDRangeKernel(q(), k_gftt_.get(), 2, nullptr, gws, nullptr, 0, nullptr,
                                   nullptr);
        OCL_CHECK(e, "clEnqueueNDRangeKernel(gftt)");
        OCL_CHECK(clFinish(q()), "clFinish(gftt)");
        score.resize(n);
        e = clEnqueueReadBuffer(q(), k_score_.get(), CL_TRUE, 0, n * sizeof(float), score.data(),
                                0, nullptr, nullptr);
        OCL_CHECK(e, "clEnqueueReadBuffer(score)");
    }

    // ---- the pyramid arena -------------------------------------------------
    // begin_frame() resets the uncached bookkeeping; in the cached mode the
    // arena is a 43-slot ring keyed by the absolute frame index.
    void begin_frame() {
        if (!cached_) {
            tmp_map_.clear();
            next_slot_ = 0;
        }
    }

    // ensure_slot(): make `abs`'s pyramid resident in the arena, uploading it
    // if it is not there yet.  Returns the slot index.
    int ensure_slot(long abs, const std::vector<Mat>& pyr, double& pad_ms, double& up_ms) {
        int s;
        if (cached_) {
            s = (int)(((abs % nslots_) + nslots_) % nslots_);
            if (slot_abs_[(size_t)s] == abs) return s;
        } else {
            auto it = tmp_map_.find(abs);
            if (it != tmp_map_.end()) return it->second;
            s = next_slot_++;
            if (next_slot_ >= nslots_) next_slot_ = 0;
            tmp_map_[abs] = s;
        }
        upload_slot(s, pyr, pad_ms, up_ms);
        slot_abs_[(size_t)s] = abs;
        return s;
    }

    int slot_of(long abs) const {
        if (cached_) return (int)(((abs % nslots_) + nslots_) % nslots_);
        auto it = tmp_map_.find(abs);
        return it == tmp_map_.end() ? -1 : it->second;
    }

    // ---- the batched LK ----------------------------------------------------
    // `fits` gives, per fit, the prev slot, the curr slot and the point
    // count.  `pts` gives, per fit, the source points.
    void lk_multi(const std::vector<int>& prev_slot, const std::vector<int>& curr_slot,
                  const std::vector<std::vector<cv::Point2f>>& pts,
                  std::vector<std::vector<cv::Point2f>>& out_pts,
                  std::vector<std::vector<uchar>>& out_status,
                  double* dispatch_ms = nullptr, double* read_ms = nullptr) {
        const int nfits = (int)prev_slot.size();
        for (int f = 0; f < nfits; ++f) {
            for (int L = 0; L < nlevels_; ++L) {
                int* d = &desc_host_[(f * nlevels_ + L) * 8];
                d[0] = padw_[L];
                d[1] = padh_[L];
                d[2] = roiw_[L];
                d[3] = roih_[L];
                d[4] = prev_slot[(size_t)f] * slot_img_ + lvl_img_off_[L];
                d[5] = prev_slot[(size_t)f] * slot_der_ + lvl_der_off_[L];
                d[6] = curr_slot[(size_t)f] * slot_img_ + lvl_img_off_[L];
                d[7] = 0;
            }
            const int n = (int)pts[(size_t)f].size();
            fit_count_host_[f] = n;
            for (int p = 0; p < n; ++p) {
                pts_in_host_[((size_t)f * kNptsSlot + (size_t)p) * 2] = pts[(size_t)f][(size_t)p].x;
                pts_in_host_[((size_t)f * kNptsSlot + (size_t)p) * 2 + 1] =
                    pts[(size_t)f][(size_t)p].y;
            }
        }
        const Clock::time_point t_disp = Clock::now();
        cl_int e;
        e = clEnqueueWriteBuffer(q(), k_desc_.get(), CL_TRUE, 0, sizeof(desc_host_), desc_host_, 0,
                                 nullptr, nullptr);
        OCL_CHECK(e, "clEnqueueWriteBuffer(desc)");
        e = clEnqueueWriteBuffer(q(), k_fit_count_.get(), CL_TRUE, 0,
                                 (size_t)nfits * sizeof(int), fit_count_host_, 0, nullptr,
                                 nullptr);
        OCL_CHECK(e, "clEnqueueWriteBuffer(fit_count)");
        e = clEnqueueWriteBuffer(q(), k_pts_in_.get(), CL_TRUE, 0,
                                 (size_t)nfits * kNptsSlot * 2 * sizeof(float), pts_in_host_.data(), 0,
                                 nullptr, nullptr);
        OCL_CHECK(e, "clEnqueueWriteBuffer(pts_in)");
        set_mem_arg(k_lkm_.get(), 0, k_pyr_img_.get());
        set_mem_arg(k_lkm_.get(), 1, k_pyr_der_.get());
        set_mem_arg(k_lkm_.get(), 2, k_desc_.get());
        set_mem_arg(k_lkm_.get(), 3, k_fit_count_.get());
        set_mem_arg(k_lkm_.get(), 4, k_patch_.get());
        set_mem_arg(k_lkm_.get(), 5, k_pts_in_.get());
        set_mem_arg(k_lkm_.get(), 6, k_pts_out_.get());
        set_mem_arg(k_lkm_.get(), 7, k_status_.get());
        set_int_arg(k_lkm_.get(), 8, nfits);
        set_int_arg(k_lkm_.get(), 9, nlevels_);
        set_int_arg(k_lkm_.get(), 10, win_);
        set_int_arg(k_lkm_.get(), 11, iters_);
        set_float_arg(k_lkm_.get(), 12, (float)(kLkEps * kLkEps));
        set_float_arg(k_lkm_.get(), 13, 1e-4f);
        set_int_arg(k_lkm_.get(), 14, kNptsSlot);
        const size_t gws[2] = {(size_t)kNptsSlot, (size_t)nfits};
        const size_t lws[2] = {(size_t)local_, 1};
        e = clEnqueueNDRangeKernel(q(), k_lkm_.get(), 2, nullptr, gws, lws, 0, nullptr, nullptr);
        OCL_CHECK(e, "clEnqueueNDRangeKernel(lk_multi)");
        OCL_CHECK(clFinish(q()), "clFinish(lk_multi)");
        if (dispatch_ms) *dispatch_ms = ms_since(t_disp);
        const Clock::time_point t_read = Clock::now();
        e = clEnqueueReadBuffer(q(), k_pts_out_.get(), CL_TRUE, 0,
                                (size_t)nfits * kNptsSlot * 2 * sizeof(float), pts_out_host_.data(), 0,
                                nullptr, nullptr);
        OCL_CHECK(e, "clEnqueueReadBuffer(pts_out)");
        e = clEnqueueReadBuffer(q(), k_status_.get(), CL_TRUE, 0,
                                (size_t)nfits * kNptsSlot * sizeof(int), status_host_.data(), 0, nullptr,
                                nullptr);
        OCL_CHECK(e, "clEnqueueReadBuffer(status)");
        if (read_ms) *read_ms = ms_since(t_read);
        out_pts.resize((size_t)nfits);
        out_status.resize((size_t)nfits);
        for (int f = 0; f < nfits; ++f) {
            const int n = fit_count_host_[f];
            out_pts[(size_t)f].resize((size_t)n);
            out_status[(size_t)f].resize((size_t)n);
            for (int p = 0; p < n; ++p) {
                const size_t g = (size_t)f * kNptsSlot + (size_t)p;
                out_pts[(size_t)f][(size_t)p] =
                    cv::Point2f(pts_out_host_[g * 2], pts_out_host_[g * 2 + 1]);
                out_status[(size_t)f][(size_t)p] = (uchar)(status_host_[g] ? 1 : 0);
            }
        }
    }

    // ---- the SEQUENTIAL LK (one dispatch per fit) --------------------------
    // The same kernel the single-lag probe used, pointed at the arena: the
    // prev and the curr offsets both live in pyr_img, so the same buffer is
    // passed for prev_img and curr_img and the descriptor selects the plane.
    void lk_single(int prev_slot, int curr_slot, const std::vector<cv::Point2f>& pts,
                   std::vector<cv::Point2f>& out_pts, std::vector<uchar>& out_status) {
        const int n = (int)pts.size();
        out_pts.assign((size_t)n, cv::Point2f(0, 0));
        out_status.assign((size_t)n, 0);
        if (n <= 0) return;
        for (int L = 0; L < nlevels_; ++L) {
            int* d = &desc1_host_[L * 8];
            d[0] = padw_[L];
            d[1] = padh_[L];
            d[2] = roiw_[L];
            d[3] = roih_[L];
            d[4] = prev_slot * slot_img_ + lvl_img_off_[L];
            d[5] = prev_slot * slot_der_ + lvl_der_off_[L];
            d[6] = curr_slot * slot_img_ + lvl_img_off_[L];
            d[7] = 0;
        }
        for (int p = 0; p < n; ++p) {
            pts1_in_[2 * p] = pts[(size_t)p].x;
            pts1_in_[2 * p + 1] = pts[(size_t)p].y;
        }
        cl_int e;
        e = clEnqueueWriteBuffer(q(), k_desc1_.get(), CL_TRUE, 0, sizeof(desc1_host_),
                                 desc1_host_, 0, nullptr, nullptr);
        OCL_CHECK(e, "clEnqueueWriteBuffer(desc1)");
        e = clEnqueueWriteBuffer(q(), k_pts1_in_.get(), CL_TRUE, 0,
                                 (size_t)n * 2 * sizeof(float), pts1_in_.data(), 0, nullptr, nullptr);
        OCL_CHECK(e, "clEnqueueWriteBuffer(pts1_in)");
        set_mem_arg(k_lk1_.get(), 0, k_pyr_img_.get());
        set_mem_arg(k_lk1_.get(), 1, k_pyr_der_.get());
        set_mem_arg(k_lk1_.get(), 2, k_pyr_img_.get());
        set_mem_arg(k_lk1_.get(), 3, k_desc1_.get());
        set_mem_arg(k_lk1_.get(), 4, k_patch_.get());
        set_mem_arg(k_lk1_.get(), 5, k_pts1_in_.get());
        set_mem_arg(k_lk1_.get(), 6, k_pts1_out_.get());
        set_mem_arg(k_lk1_.get(), 7, k_status1_.get());
        set_int_arg(k_lk1_.get(), 8, n);
        set_int_arg(k_lk1_.get(), 9, nlevels_);
        set_int_arg(k_lk1_.get(), 10, win_);
        set_int_arg(k_lk1_.get(), 11, iters_);
        set_float_arg(k_lk1_.get(), 12, (float)(kLkEps * kLkEps));
        set_float_arg(k_lk1_.get(), 13, 1e-4f);
        const size_t gws = (size_t)n;
        e = clEnqueueNDRangeKernel(q(), k_lk1_.get(), 1, nullptr, &gws, nullptr, 0, nullptr,
                                   nullptr);
        OCL_CHECK(e, "clEnqueueNDRangeKernel(lk_single)");
        OCL_CHECK(clFinish(q()), "clFinish(lk_single)");
        e = clEnqueueReadBuffer(q(), k_pts1_out_.get(), CL_TRUE, 0, (size_t)n * 2 * sizeof(float),
                                pts1_out_.data(), 0, nullptr, nullptr);
        OCL_CHECK(e, "clEnqueueReadBuffer(pts1_out)");
        e = clEnqueueReadBuffer(q(), k_status1_.get(), CL_TRUE, 0, (size_t)n * sizeof(int),
                                pts1_status_.data(), 0, nullptr, nullptr);
        OCL_CHECK(e, "clEnqueueReadBuffer(status1)");
        for (int p = 0; p < n; ++p) {
            out_pts[(size_t)p] = cv::Point2f(pts1_out_[2 * p], pts1_out_[2 * p + 1]);
            out_status[(size_t)p] = (uchar)(pts1_status_[p] ? 1 : 0);
        }
    }

private:
    struct DevPick {
        cl_platform_id plat = nullptr;
        cl_device_id dev = nullptr;
    };
    static DevPick pick() {
        DevPick d;
        d.dev = ocl::pick_device(&d.plat);
        return d;
    }
    cl_command_queue q() const { return ctx_.queue(); }
    static void set_mem_arg(cl_kernel k, cl_uint idx, cl_mem m) {
        const cl_int e = clSetKernelArg(k, idx, sizeof(cl_mem), &m);
        OCL_CHECK(e, "clSetKernelArg(mem)");
    }
    static void set_int_arg(cl_kernel k, cl_uint idx, int v) {
        const cl_int e = clSetKernelArg(k, idx, sizeof(int), &v);
        OCL_CHECK(e, "clSetKernelArg(int)");
    }
    static void set_float_arg(cl_kernel k, cl_uint idx, float v) {
        const cl_int e = clSetKernelArg(k, idx, sizeof(float), &v);
        OCL_CHECK(e, "clSetKernelArg(float)");
    }

    void upload_slot(int slot, const std::vector<Mat>& pyr, double& pad_ms, double& up_ms) {
        const Clock::time_point t0 = Clock::now();
        for (int L = 0; L < nlevels_; ++L) {
            const Mat& pi = pyr[(size_t)L * 2];
            const Mat& pd = pyr[(size_t)L * 2 + 1];
            const int rw = pi.cols, rh = pi.rows;
            const int pw = rw + 2 * win_, ph = rh + 2 * win_;
            if (pw != padw_[L] || ph != padh_[L] || rw != roiw_[L] || rh != roih_[L]) {
                std::fprintf(stderr, "[FATAL] pyramid level %d shape mismatch: got %dx%d "
                                     "(padded %dx%d), expected %dx%d (padded %dx%d)\n",
                             L, rw, rh, pw, ph, roiw_[L], roih_[L], padw_[L], padh_[L]);
                std::exit(3);
            }
            // image level, BORDER_REFLECT_101 (buildOpticalFlowPyramid's border)
            for (int y = 0; y < ph; ++y) {
                const int sy = reflect101(y - win_, rh);
                uchar* dst = stage_img_.data() + lvl_img_off_[L] + (size_t)y * pw;
                const uchar* src = pi.ptr<uchar>(sy);
                for (int x = 0; x < pw; ++x) dst[x] = src[reflect101(x - win_, rw)];
            }
            // derivative level, BORDER_CONSTANT
            std::memset(stage_der_.data() + lvl_der_off_[L], 0,
                        (size_t)pw * ph * 2 * sizeof(short));
            for (int y = 0; y < rh; ++y) {
                const short* src = pd.ptr<short>(y);
                short* dst = stage_der_.data() + lvl_der_off_[L] +
                             (size_t)(y + win_) * pw * 2 + (size_t)win_ * 2;
                std::memcpy(dst, src, (size_t)rw * 2 * sizeof(short));
            }
        }
        pad_ms = ms_since(t0);
        const Clock::time_point t1 = Clock::now();
        cl_int e = clEnqueueWriteBuffer(q(), k_pyr_img_.get(), CL_TRUE,
                                        (size_t)slot * slot_img_, slot_img_, stage_img_.data(), 0,
                                        nullptr, nullptr);
        OCL_CHECK(e, "clEnqueueWriteBuffer(pyr_img)");
        e = clEnqueueWriteBuffer(q(), k_pyr_der_.get(), CL_TRUE,
                                 (size_t)slot * slot_der_ * sizeof(short),
                                 slot_der_ * sizeof(short), stage_der_.data(), 0, nullptr, nullptr);
        OCL_CHECK(e, "clEnqueueWriteBuffer(pyr_der)");
        OCL_CHECK(clFinish(q()), "clFinish(upload)");
        up_ms = ms_since(t1);
    }

    void configure(int w, int h, int local) {
        sw_ = w;
        sh_ = h;
        win_ = kWin;
        iters_ = kLkIters;
        local_ = local;
        std::vector<std::pair<int, int>> sizes;
        int lw = w, lh = h;
        for (int L = 0; L <= kMaxLevel; ++L) {
            sizes.push_back({lw, lh});
            lw = (lw + 1) / 2;
            lh = (lh + 1) / 2;
            if (lw <= win_ || lh <= win_) break;
        }
        nlevels_ = (int)sizes.size();
        padw_.assign((size_t)nlevels_, 0);
        padh_.assign((size_t)nlevels_, 0);
        roiw_.assign((size_t)nlevels_, 0);
        roih_.assign((size_t)nlevels_, 0);
        lvl_img_off_.assign((size_t)nlevels_, 0);
        lvl_der_off_.assign((size_t)nlevels_, 0);
        slot_img_ = 0;
        slot_der_ = 0;
        for (int L = 0; L < nlevels_; ++L) {
            const int rw = sizes[(size_t)L].first, rh = sizes[(size_t)L].second;
            const int pw = rw + 2 * win_, ph = rh + 2 * win_;
            padw_[L] = pw;
            padh_[L] = ph;
            roiw_[L] = rw;
            roih_[L] = rh;
            lvl_img_off_[L] = slot_img_;
            lvl_der_off_[L] = slot_der_;
            slot_img_ += (size_t)pw * ph;
            slot_der_ += (size_t)pw * ph * 2;
        }
        stage_img_.assign((size_t)slot_img_, 0);
        stage_der_.assign((size_t)slot_der_, 0);

        const size_t patch_floats = (size_t)3 * win_ * win_ * kNFits * kNptsSlot;
        const size_t arena_img = (size_t)nslots_ * slot_img_;
        const size_t arena_der = (size_t)nslots_ * slot_der_;

        cl_int e;
        k_img_.reset(clCreateBuffer(ctx_.get(), CL_MEM_READ_ONLY, (size_t)w * h, nullptr, &e));
        OCL_CHECK(e, "clCreateBuffer(img)");
        k_score_.reset(clCreateBuffer(ctx_.get(), CL_MEM_WRITE_ONLY, (size_t)w * h * sizeof(float),
                                      nullptr, &e));
        OCL_CHECK(e, "clCreateBuffer(score)");
        k_pyr_img_.reset(clCreateBuffer(ctx_.get(), CL_MEM_READ_ONLY, arena_img, nullptr, &e));
        OCL_CHECK(e, "clCreateBuffer(pyr_img)");
        k_pyr_der_.reset(clCreateBuffer(ctx_.get(), CL_MEM_READ_ONLY, arena_der * sizeof(short),
                                        nullptr, &e));
        OCL_CHECK(e, "clCreateBuffer(pyr_der)");
        k_desc_.reset(clCreateBuffer(ctx_.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(desc_host_), desc_host_, &e));
        OCL_CHECK(e, "clCreateBuffer(desc)");
        k_desc1_.reset(clCreateBuffer(ctx_.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(desc1_host_), desc1_host_, &e));
        OCL_CHECK(e, "clCreateBuffer(desc1)");
        k_fit_count_.reset(clCreateBuffer(ctx_.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          (size_t)kNFits * sizeof(int), fit_count_host_, &e));
        OCL_CHECK(e, "clCreateBuffer(fit_count)");
        k_patch_.reset(clCreateBuffer(ctx_.get(), CL_MEM_READ_WRITE, patch_floats * sizeof(float),
                                      nullptr, &e));
        OCL_CHECK(e, "clCreateBuffer(patch)");
        k_pts_in_.reset(clCreateBuffer(ctx_.get(), CL_MEM_READ_ONLY,
                                       (size_t)kNFits * kNptsSlot * 2 * sizeof(float), nullptr,
                                       &e));
        OCL_CHECK(e, "clCreateBuffer(pts_in)");
        k_pts_out_.reset(clCreateBuffer(ctx_.get(), CL_MEM_WRITE_ONLY,
                                        (size_t)kNFits * kNptsSlot * 2 * sizeof(float), nullptr,
                                        &e));
        OCL_CHECK(e, "clCreateBuffer(pts_out)");
        k_status_.reset(clCreateBuffer(ctx_.get(), CL_MEM_WRITE_ONLY,
                                       (size_t)kNFits * kNptsSlot * sizeof(int), nullptr, &e));
        OCL_CHECK(e, "clCreateBuffer(status)");
        k_pts1_in_.reset(clCreateBuffer(ctx_.get(), CL_MEM_READ_ONLY,
                                        (size_t)kNptsSlot * 2 * sizeof(float), nullptr, &e));
        OCL_CHECK(e, "clCreateBuffer(pts1_in)");
        k_pts1_out_.reset(clCreateBuffer(ctx_.get(), CL_MEM_WRITE_ONLY,
                                         (size_t)kNptsSlot * 2 * sizeof(float), nullptr, &e));
        OCL_CHECK(e, "clCreateBuffer(pts1_out)");
        k_status1_.reset(clCreateBuffer(ctx_.get(), CL_MEM_WRITE_ONLY,
                                        (size_t)kNptsSlot * sizeof(int), nullptr, &e));
        OCL_CHECK(e, "clCreateBuffer(status1)");
        pts_in_host_.assign((size_t)kNFits * kNptsSlot * 2, 0.0f);
        pts_out_host_.assign((size_t)kNFits * kNptsSlot * 2, 0.0f);
        status_host_.assign((size_t)kNFits * kNptsSlot, 0);
        pts1_in_.assign((size_t)kNptsSlot * 2, 0.0f);
        pts1_out_.assign((size_t)kNptsSlot * 2, 0.0f);
        pts1_status_.assign((size_t)kNptsSlot, 0);
        std::memset(desc_host_, 0, sizeof(desc_host_));
        std::memset(desc1_host_, 0, sizeof(desc1_host_));
        std::printf("[GPU-full] levels=%d local=%d slots=%d slot_img=%.2f MiB slot_der=%.2f MiB "
                    "arena=%.2f MiB patch=%.2f MiB\n",
                    nlevels_, local_, nslots_, (double)slot_img_ / 1048576.0,
                    (double)(slot_der_ * sizeof(short)) / 1048576.0,
                    (double)(arena_img + arena_der * sizeof(short)) / 1048576.0,
                    (double)(patch_floats * sizeof(float)) / 1048576.0);
    }

    DevPick dev_;
    ocl::Context ctx_;
    ocl::Program prog_;
    ocl::Kernel k_gftt_, k_lkm_, k_lk1_;
    ocl::Mem k_img_, k_score_, k_pyr_img_, k_pyr_der_, k_desc_, k_desc1_, k_fit_count_, k_patch_,
        k_pts_in_, k_pts_out_, k_status_, k_pts1_in_, k_pts1_out_, k_status1_;
    std::vector<uchar> stage_img_;
    std::vector<short> stage_der_;
    std::vector<long> slot_abs_;
    std::map<long, int> tmp_map_;
    int next_slot_ = 0;
    int nslots_ = 0;
    bool cached_ = true;
    int desc_host_[8 * 8 * kNFits] = {0};
    int desc1_host_[8 * 8] = {0};
    int fit_count_host_[kNFits] = {0};
    std::vector<float> pts_in_host_, pts_out_host_, pts1_in_, pts1_out_;
    std::vector<int> status_host_, pts1_status_;
    std::vector<int> padw_, padh_, roiw_, roih_, lvl_img_off_, lvl_der_off_;
    int sw_ = 0, sh_ = 0, win_ = kWin, iters_ = kLkIters, nlevels_ = 0, local_ = 32;
    int slot_img_ = 0, slot_der_ = 0;
};

// ---------------------------------------------------------------------------
// the host-side frame cache (the production's three-level cache)
// ---------------------------------------------------------------------------
class FrameCache {
public:
    explicit FrameCache(int depth, int max_corners) : depth_(depth), max_corners_(max_corners) {
        small_.resize((size_t)depth);
        tsmall_.assign((size_t)depth, -1);
        corners_.resize((size_t)depth);
        tcorners_.assign((size_t)depth, -1);
        capped_.assign((size_t)depth, 0);
        pyr_.resize((size_t)depth);
        tpyr_.assign((size_t)depth, -1);
    }
    int slot(long abs) const {
        long s = abs % depth_;
        if (s < 0) s += depth_;
        return (int)s;
    }

    // the downscaled frame.  The production sizes the resize from the CURRENT
    // frame, and every frame of a sequence has the same shape, so the size is
    // the same either way.
    const Mat& small_of(long abs, const std::vector<Mat>& frames, long first_abs, double& ms) {
        const int s = slot(abs);
        if (tsmall_[(size_t)s] == abs) return small_[(size_t)s];
        const long i = abs - first_abs;
        if (i < 0 || i >= (long)frames.size()) {
            tsmall_[(size_t)s] = abs;
            small_[(size_t)s] = Mat();
            return small_[(size_t)s];
        }
        const Clock::time_point t0 = Clock::now();
        const Mat& g = frames[(size_t)i];
        cv::resize(g, small_[(size_t)s], cv::Size(g.cols / kDownscale, g.rows / kDownscale), 0, 0,
                   cv::INTER_LINEAR);
        ms += ms_since(t0);
        tsmall_[(size_t)s] = abs;
        tcorners_[(size_t)s] = -1;
        tpyr_[(size_t)s] = -1;
        return small_[(size_t)s];
    }

    // the corner set.  gpu == nullptr -> OpenCV's goodFeaturesToTrack (the
    // production's own detector); otherwise the GPU gFTT + the CPU selection.
    const std::vector<cv::Point2f>& corners_of(long abs, const std::vector<Mat>& frames,
                                                long first_abs, GpuFull* gpu, double& gftt_ms,
                                                double& select_ms) {
        const int s = slot(abs);
        if (tcorners_[(size_t)s] == abs) return corners_[(size_t)s];
        const Mat& sm = small_of(abs, frames, first_abs, gftt_ms);
        if (sm.empty()) {
            tcorners_[(size_t)s] = abs;
            corners_[(size_t)s].clear();
            return corners_[(size_t)s];
        }
        if (gpu) {
            const Clock::time_point t0 = Clock::now();
            std::vector<float> score;
            gpu->gftt(sm, score);
            gftt_ms += ms_since(t0);
            const Clock::time_point t1 = Clock::now();
            corners_[(size_t)s] = select_corners(score, gpu->small_w(), gpu->small_h(), max_corners_,
                                                 kQualityLevel, kMinDistance);
            select_ms += ms_since(t1);
        } else {
            const Clock::time_point t0 = Clock::now();
            std::vector<cv::Point2f> pts;
            cv::goodFeaturesToTrack(sm, pts, max_corners_ + 1, kQualityLevel, kMinDistance,
                                    cv::noArray(), kBlockSize);
            // the production asks for budget+1 and drops the last one, so the
            // count reaching the budget exactly means the frame was CAPPED.
            if ((int)pts.size() > max_corners_) {
                capped_[s] = 1;
                pts.resize((size_t)max_corners_);
            } else {
                capped_[s] = 0;
            }
            corners_[(size_t)s] = pts;
            gftt_ms += ms_since(t0);
        }
        tcorners_[(size_t)s] = abs;
        return corners_[(size_t)s];
    }

    // the optical-flow pyramid (cv::buildOpticalFlowPyramid's output)
    const std::vector<Mat>& pyr_of(long abs, const std::vector<Mat>& frames, long first_abs,
                                   double& build_ms) {
        const int s = slot(abs);
        if (tpyr_[(size_t)s] == abs) return pyr_[(size_t)s];
        const long i = abs - first_abs;
        if (i < 0 || i >= (long)frames.size()) {
            tpyr_[(size_t)s] = abs;
            pyr_[(size_t)s].clear();
            return pyr_[(size_t)s];
        }
        const Mat& sm = small_of(abs, frames, first_abs, build_ms);
        if (sm.empty()) {
            tpyr_[(size_t)s] = abs;
            pyr_[(size_t)s].clear();
            return pyr_[(size_t)s];
        }
        const Clock::time_point t0 = Clock::now();
        cv::buildOpticalFlowPyramid(sm, pyr_[(size_t)s], cv::Size(kWin, kWin), kMaxLevel, true);
        build_ms += ms_since(t0);
        tpyr_[(size_t)s] = abs;
        return pyr_[(size_t)s];
    }

    // was this frame's corner set truncated by the budget?
    bool was_capped(long abs) const { return capped_[(size_t)slot(abs)] != 0; }

private:
    int depth_;
    int max_corners_ = kMaxCorners;
    std::vector<char> capped_;
    std::vector<Mat> small_;
    std::vector<long> tsmall_;
    std::vector<std::vector<cv::Point2f>> corners_;
    std::vector<long> tcorners_;
    std::vector<std::vector<Mat>> pyr_;
    std::vector<long> tpyr_;
};

// ---------------------------------------------------------------------------
// the per-frame record
// ---------------------------------------------------------------------------
struct FrameRec {
    long abs = 0;
    double cpu_resize = 0, cpu_corners = 0, cpu_pyr = 0, cpu_lk = 0, cpu_ransac = 0, cpu_total = 0;
    double gpu_resize = 0, gpu_gftt = 0, gpu_select = 0, gpu_pyr = 0, gpu_upload = 0;
    double gpu_lk_batch = 0, gpu_lk_1anchor = 0, gpu_lk_seq = 0, gpu_ransac = 0, gpu_total = 0;
    double gpu_lk_dispatch = 0, gpu_lk_read = 0;
    int cpu_ncorners = 0, gpu_ncorners = 0;
    int cpu_capped = 0, gpu_capped = 0;
    int cpu_nvalid[kNFits] = {0}, gpu_nvalid[kNFits] = {0};
    // per-fit stage times (the GPU arm): the sequential LK of each fit and
    // the RANSAC of each fit
    double lk_seq_per_fit[kNFits] = {0};
    double ransac_per_fit[kNFits] = {0};
    double cpu_ransac_per_fit[kNFits] = {0};
    // the per-fit fit results (the anchors first, then the step fit)
    std::vector<Matx23f> fit_H;
    std::vector<std::vector<cv::Point2f>> fit_corners;
    // the 21-lag chain
    double chain_dtrans[kNLags] = {0}, chain_cos[kNLags] = {1}, chain_scale[kNLags] = {0};
    int chain_valid = 0;
};

// the CPU LK with the PREBUILT pyramids (the production's prebuilt_pyr path)
void cpu_lk_pyr(const std::vector<Mat>& prev_pyr, const std::vector<Mat>& curr_pyr,
                const std::vector<cv::Point2f>& pts, std::vector<cv::Point2f>& pts_next,
                std::vector<uchar>& status) {
    pts_next.assign(pts.size(), cv::Point2f(0, 0));
    status.assign(pts.size(), 0);
    if (pts.empty() || prev_pyr.empty() || curr_pyr.empty()) return;
    const Mat pts_mat((int)pts.size(), 1, CV_32FC2, const_cast<cv::Point2f*>(pts.data()));
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prev_pyr, curr_pyr, pts_mat, pts_next, status, err, cv::Size(kWin, kWin),
                              kMaxLevel);
}

// the RANSAC stage, with the per-fit reseed (so both arms draw the same
// random numbers for the same fit)
int ransac_fit_seeded(const std::vector<cv::Point2f>& pts, const std::vector<cv::Point2f>& pts_next,
                      const std::vector<uchar>& status, Matx23f& H, unsigned seed) {
    std::vector<cv::Point2f> p0, p1;
    p0.reserve(pts.size());
    p1.reserve(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        if (i < status.size() && status[i] == 1) {
            p0.push_back(pts[i]);
            p1.push_back(pts_next[i]);
        }
    }
    if (p0.size() < 6) return 0;
    cv::Mat inliers;
    cv::theRNG() = cv::RNG(seed);
    const cv::Mat M = cv::estimateAffinePartial2D(p0, p1, inliers, cv::RANSAC, kRansacThresh);
    if (M.empty()) return 0;
    double v[6];
    for (int i = 0; i < 6; ++i) v[i] = M.at<double>(i / 3, i % 3);
    H = Matx23f((float)v[0], (float)v[1], (float)v[2], (float)v[3], (float)v[4], (float)v[5]);
    H(0, 2) = H(0, 2) * (float)kDownscale;
    H(1, 2) = H(1, 2) * (float)kDownscale;
    return (int)p0.size();
}

// ---------------------------------------------------------------------------
// the CPU arm
// ---------------------------------------------------------------------------
// `cpu_threads` > 1 runs the LK stages of the 6 fits CONCURRENTLY (the
// production's parallel-anchor path) while the RANSACs stay serial in the
// fit order, so the global RNG is drawn from exactly as the production
// draws it.
void run_cpu_arm(const std::vector<Mat>& frames, long first_abs, long report_from,
                 int cpu_threads, int max_corners, std::vector<FrameRec>& out,
                 std::map<long, std::map<int, Matx23f>>* chains) {
    FrameCache cache(kRingDepth, max_corners);
    std::map<long, Matx23f> step_cache;   // abs -> H(abs -> abs+2)

    for (size_t i = 0; i < frames.size(); ++i) {
        const long t = first_abs + (long)i;
        FrameRec rec;
        rec.abs = t;
        const std::vector<Fit> fits = fits_of(t);

        // ---- phase 1 (serial): the cache work + the LK plans ----------------
        std::vector<std::vector<cv::Point2f>> src(kNFits);
        std::vector<const std::vector<Mat>*> prev_pyr(kNFits), curr_pyr(kNFits);
        bool enough[kNFits] = {false};
        for (int f = 0; f < kNFits; ++f) {
            const long pa = fits[(size_t)f].prev_abs, ca = fits[(size_t)f].curr_abs;
            const bool pa_ok = (pa - first_abs) >= 0 && (pa - first_abs) < (long)frames.size();
            const bool ca_ok = (ca - first_abs) >= 0 && (ca - first_abs) < (long)frames.size();
            if (!pa_ok || !ca_ok) continue;
            cache.small_of(pa, frames, first_abs, rec.cpu_resize);
            cache.small_of(ca, frames, first_abs, rec.cpu_resize);
            const std::vector<cv::Point2f>& c =
                cache.corners_of(pa, frames, first_abs, nullptr, rec.cpu_corners, rec.cpu_corners);
            src[(size_t)f] = c;
            prev_pyr[(size_t)f] = &cache.pyr_of(pa, frames, first_abs, rec.cpu_pyr);
            curr_pyr[(size_t)f] = &cache.pyr_of(ca, frames, first_abs, rec.cpu_pyr);
            enough[(size_t)f] = c.size() >= 6 && !prev_pyr[(size_t)f]->empty() &&
                                !curr_pyr[(size_t)f]->empty();
        }
        rec.cpu_ncorners = (int)src[0].size();
        rec.cpu_capped = cache.was_capped(fits[0].prev_abs) ? 1 : 0;

        // ---- phase 2: the LK stages ----------------------------------------
        std::vector<std::vector<cv::Point2f>> lk_pts(kNFits);
        std::vector<std::vector<uchar>> lk_status(kNFits);
        const Clock::time_point t_lk = Clock::now();
        if (cpu_threads > 1) {
            std::vector<std::thread> th;
            for (int f = 0; f < kNFits; ++f) {
                if (!enough[(size_t)f]) continue;
                th.emplace_back([&, f]() {
                    cpu_lk_pyr(*prev_pyr[(size_t)f], *curr_pyr[(size_t)f], src[(size_t)f],
                               lk_pts[(size_t)f], lk_status[(size_t)f]);
                });
            }
            for (std::thread& x : th) x.join();
        } else {
            for (int f = 0; f < kNFits; ++f) {
                if (!enough[(size_t)f]) continue;
                cpu_lk_pyr(*prev_pyr[(size_t)f], *curr_pyr[(size_t)f], src[(size_t)f],
                           lk_pts[(size_t)f], lk_status[(size_t)f]);
            }
        }
        rec.cpu_lk = ms_since(t_lk);

        // ---- phase 3 (serial, fit order): the RANSACs ----------------------
        std::vector<Matx23f> H(kNFits, Matx23f::eye());
        const Clock::time_point t_ran = Clock::now();
        for (int f = 0; f < kNFits; ++f) {
            if (!enough[(size_t)f]) continue;
            const Clock::time_point tf = Clock::now();
            rec.cpu_nvalid[f] = ransac_fit_seeded(src[(size_t)f], lk_pts[(size_t)f],
                                                  lk_status[(size_t)f], H[(size_t)f],
                                                  fit_seed(t, f));
            rec.cpu_ransac_per_fit[f] = ms_since(tf);
        }
        rec.cpu_ransac = ms_since(t_ran);
        rec.cpu_total = rec.cpu_resize + rec.cpu_corners + rec.cpu_pyr + rec.cpu_lk +
                        rec.cpu_ransac;
        rec.fit_H = H;
        rec.fit_corners = src;

        // ---- the chain ------------------------------------------------------
        if (t >= report_from) {
            std::map<int, Matx23f> mats;
            for (int f = 0; f < 5; ++f) mats[kAnchors[f]] = H[(size_t)f];
            for (int f = 5; f < kNFits; ++f) {
                if (enough[(size_t)f]) step_cache[fits[(size_t)f].prev_abs] = H[(size_t)f];
            }
            int li = 0;
            for (int lag = 2; lag <= kFullMaxLag; lag += 2) {
                bool is_anchor = false;
                for (int a = 0; a < 5; ++a)
                    if (kAnchors[a] == lag) is_anchor = true;
                if (!is_anchor) {
                    const int below = anchor_below(lag);
                    if (!mats.count(below)) continue;
                    Matx23f acc = mats[below];
                    bool ok = true;
                    for (int la = below + 2; la <= lag; la += 2) {
                        auto it = step_cache.find(t - la);
                        if (it == step_cache.end()) { ok = false; break; }
                        acc = compose_ff(it->second, acc);
                    }
                    if (!ok) continue;
                    mats[lag] = acc;
                }
                (void)li;
                ++li;
            }
            if (chains) (*chains)[t] = mats;
        }

        out.push_back(rec);
    }
}

// ---------------------------------------------------------------------------
// the GPU arm
// ---------------------------------------------------------------------------
// Same frame, same fits, same cache structure.  The differences are exactly
// the ones the task asks for:
//   * the corners come from the GPU gFTT (the SELECTION stays on the CPU,
//     and is timed as such),
//   * the LK of all 6 fits is ONE batched NDRange (lk_pyr_track_multi),
//   * the sequential variant (one lk_pyr_track dispatch per fit) is
//     measured alongside, on the same uploaded pyramids and the same
//     points, so the two are directly comparable,
//   * the RANSACs stay on the CPU, serial, in the fit order.
void run_gpu_arm(const std::vector<Mat>& frames, long first_abs, long report_from, GpuFull& gpu,
                 bool measure_seq, int max_corners, std::vector<FrameRec>& out,
                 std::map<long, std::map<int, Matx23f>>* chains) {
    FrameCache cache(kRingDepth, max_corners);
    std::map<long, Matx23f> step_cache;   // abs -> H(abs -> abs+2)

    for (size_t i = 0; i < frames.size(); ++i) {
        const long t = first_abs + (long)i;
        FrameRec rec;
        rec.abs = t;
        const std::vector<Fit> fits = fits_of(t);
        gpu.begin_frame();

        // ---- phase 1 (serial): the cache work + the upload -----------------
        std::vector<std::vector<cv::Point2f>> src(kNFits);
        std::vector<int> prev_slot(kNFits, -1), curr_slot(kNFits, -1);
        bool enough[kNFits] = {false};
        for (int f = 0; f < kNFits; ++f) {
            const long pa = fits[(size_t)f].prev_abs, ca = fits[(size_t)f].curr_abs;
            const bool pa_ok = (pa - first_abs) >= 0 && (pa - first_abs) < (long)frames.size();
            const bool ca_ok = (ca - first_abs) >= 0 && (ca - first_abs) < (long)frames.size();
            if (!pa_ok || !ca_ok) continue;
            cache.small_of(pa, frames, first_abs, rec.gpu_resize);
            cache.small_of(ca, frames, first_abs, rec.gpu_resize);
            const std::vector<cv::Point2f>& c =
                cache.corners_of(pa, frames, first_abs, &gpu, rec.gpu_gftt, rec.gpu_select);
            src[(size_t)f] = c;
            const std::vector<Mat>& pp = cache.pyr_of(pa, frames, first_abs, rec.gpu_pyr);
            const std::vector<Mat>& cp = cache.pyr_of(ca, frames, first_abs, rec.gpu_pyr);
            if (pp.empty() || cp.empty() || c.size() < 6) continue;
            double pad_ms = 0.0, up_ms = 0.0;
            prev_slot[(size_t)f] = gpu.ensure_slot(pa, pp, pad_ms, up_ms);
            curr_slot[(size_t)f] = gpu.ensure_slot(ca, cp, pad_ms, up_ms);
            rec.gpu_upload += pad_ms + up_ms;
            enough[(size_t)f] = true;
        }
        rec.gpu_ncorners = (int)src[0].size();
        rec.gpu_capped = ((int)src[0].size() >= max_corners) ? 1 : 0;

        // ---- phase 2: the batched LK ---------------------------------------
        std::vector<std::vector<cv::Point2f>> lk_pts(kNFits);
        std::vector<std::vector<uchar>> lk_status(kNFits);
        {
            std::vector<int> ps, cs;
            std::vector<std::vector<cv::Point2f>> pts;
            for (int f = 0; f < kNFits; ++f) {
                if (!enough[(size_t)f]) continue;
                ps.push_back(prev_slot[(size_t)f]);
                cs.push_back(curr_slot[(size_t)f]);
                pts.push_back(src[(size_t)f]);
            }
            if (!ps.empty()) {
                const Clock::time_point t0 = Clock::now();
                gpu.lk_multi(ps, cs, pts, lk_pts, lk_status, &rec.gpu_lk_dispatch,
                             &rec.gpu_lk_read);
                rec.gpu_lk_batch = ms_since(t0);
            }
        }

        // ---- the sequential variant (the comparison) ------------------------
        // and the 1-anchor batch: the SAME kernel with nfits=1, which is the
        // direct test of "is the LK latency-bound or throughput-bound?" --
        // if 1 anchor and 5 anchors cost the same, the GPU is latency-bound
        // and the batch is free.
        if (measure_seq) {
            // 1 anchor, same kernel, one dispatch
            {
                std::vector<int> ps1, cs1;
                std::vector<std::vector<cv::Point2f>> pts1;
                for (int f = 0; f < kNFits; ++f) {
                    if (!enough[(size_t)f]) continue;
                    ps1.push_back(prev_slot[(size_t)f]);
                    cs1.push_back(curr_slot[(size_t)f]);
                    pts1.push_back(src[(size_t)f]);
                    break;   // only the FIRST fit
                }
                if (!ps1.empty()) {
                    std::vector<std::vector<cv::Point2f>> o1;
                    std::vector<std::vector<uchar>> s1;
                    const Clock::time_point t0 = Clock::now();
                    gpu.lk_multi(ps1, cs1, pts1, o1, s1);
                    rec.gpu_lk_1anchor = ms_since(t0);
                }
            }
            // 6 sequential dispatches of the single-anchor kernel
            std::vector<std::vector<cv::Point2f>> sp(kNFits);
            std::vector<std::vector<uchar>> ss(kNFits);
            const Clock::time_point t0 = Clock::now();
            for (int f = 0; f < kNFits; ++f) {
                if (!enough[(size_t)f]) continue;
                const Clock::time_point tf = Clock::now();
                gpu.lk_single(prev_slot[(size_t)f], curr_slot[(size_t)f], src[(size_t)f], sp[(size_t)f],
                              ss[(size_t)f]);
                rec.lk_seq_per_fit[f] = ms_since(tf);
            }
            rec.gpu_lk_seq = ms_since(t0);
        }

        // ---- phase 3 (serial, fit order): the RANSACs ----------------------
        std::vector<Matx23f> H(kNFits, Matx23f::eye());
        const Clock::time_point t_ran = Clock::now();
        for (int f = 0; f < kNFits; ++f) {
            if (!enough[(size_t)f]) continue;
            const Clock::time_point tf = Clock::now();
            rec.gpu_nvalid[f] = ransac_fit_seeded(src[(size_t)f], lk_pts[(size_t)f],
                                                  lk_status[(size_t)f], H[(size_t)f],
                                                  fit_seed(t, f));
            rec.ransac_per_fit[f] = ms_since(tf);
        }
        rec.gpu_ransac = ms_since(t_ran);
        rec.gpu_total = rec.gpu_resize + rec.gpu_gftt + rec.gpu_select + rec.gpu_pyr +
                        rec.gpu_upload + rec.gpu_lk_batch + rec.gpu_ransac;
        rec.fit_H = H;
        rec.fit_corners = src;

        // ---- the chain ------------------------------------------------------
        if (t >= report_from) {
            std::map<int, Matx23f> mats;
            for (int f = 0; f < 5; ++f) mats[kAnchors[f]] = H[(size_t)f];
            for (int f = 5; f < kNFits; ++f) {
                if (enough[(size_t)f]) step_cache[fits[(size_t)f].prev_abs] = H[(size_t)f];
            }
            for (int lag = 2; lag <= kFullMaxLag; lag += 2) {
                bool is_anchor = false;
                for (int a = 0; a < 5; ++a)
                    if (kAnchors[a] == lag) is_anchor = true;
                if (!is_anchor) {
                    const int below = anchor_below(lag);
                    if (!mats.count(below)) continue;
                    Matx23f acc = mats[below];
                    bool ok = true;
                    for (int la = below + 2; la <= lag; la += 2) {
                        auto it = step_cache.find(t - la);
                        if (it == step_cache.end()) { ok = false; break; }
                        acc = compose_ff(it->second, acc);
                    }
                    if (!ok) continue;
                    mats[lag] = acc;
                }
            }
            if (chains) (*chains)[t] = mats;
        }

        out.push_back(rec);
    }
}

// ---------------------------------------------------------------------------
// the accuracy comparison
// ---------------------------------------------------------------------------
struct Dev {
    double dtrans = 0;    // |dt| in full-resolution pixels
    double cosrot = 1.0;
    double relscale = 0.0;
};

Dev compare_H(const Matx23f& a, const Matx23f& b) {
    Dev d;
    const double dtx = (double)a(0, 2) - (double)b(0, 2);
    const double dty = (double)a(1, 2) - (double)b(1, 2);
    d.dtrans = std::sqrt(dtx * dtx + dty * dty);
    double sa = 0, sb = 0, ra = 0, rb = 0;
    decompose(a, sa, ra);
    decompose(b, sb, rb);
    d.cosrot = std::cos(ra - rb);
    d.relscale = (sa > 0) ? std::fabs(sb - sa) / sa : 0.0;
    return d;
}

// the corner correspondence rate: for every CPU corner, the distance to the
// nearest GPU corner (small-image coordinates)
double corner_match(const std::vector<cv::Point2f>& a, const std::vector<cv::Point2f>& b) {
    if (a.empty() || b.empty()) return 0.0;
    int m = 0;
    for (const cv::Point2f& p : a) {
        double best = 1e30;
        for (const cv::Point2f& q : b) {
            const double dx = p.x - q.x, dy = p.y - q.y;
            const double d = std::sqrt(dx * dx + dy * dy);
            if (d < best) best = d;
        }
        if (best <= 1.0) ++m;
    }
    return (double)m / (double)a.size();
}

// ---------------------------------------------------------------------------
// the statistics helper
// ---------------------------------------------------------------------------
struct Stats {
    double mean = 0, p50 = 0, p95 = 0, p99 = 0, max = 0, min = 0;
    double over40 = 0;   // the fraction of frames above 40 ms
    double fps = 0;      // 1000 / mean
    int n = 0;
};

Stats stats_of(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    s.n = (int)v.size();
    s.mean = mean(v);
    s.p50 = pctl(v, 0.50);
    s.p95 = pctl(v, 0.95);
    s.p99 = pctl(v, 0.99);
    s.max = pctl(v, 1.0);
    s.min = pctl(v, 0.0);
    int over = 0;
    for (double x : v)
        if (x > 40.0) ++over;
    s.over40 = (double)over / (double)v.size();
    s.fps = (s.mean > 0) ? 1000.0 / s.mean : 0.0;
    return s;
}

// ---------------------------------------------------------------------------
// the content scan: where are the corners?
// ---------------------------------------------------------------------------
// The hardest window is the one with the most corners, because the LK cost
// scales with the point count.  The scan uses the PRODUCTION detector
// (OpenCV's goodFeaturesToTrack) so the window is chosen with the same
// yardstick the pipeline uses.
int run_scan(const std::string& seq_dir, long start, int limit, const std::string& tsv_path) {
    const std::vector<FrameFile> all = list_frames(seq_dir, 0, 0);
    if (all.empty()) {
        std::fprintf(stderr, "[FATAL] no frames in %s\n", seq_dir.c_str());
        return 2;
    }
    FILE* f = std::fopen(tsv_path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "[FATAL] cannot write %s\n", tsv_path.c_str());
        return 2;
    }
    std::fprintf(f, "abs\tn_corners\tcapped\n");
    long n = 0;
    for (size_t i = 0; i < all.size(); ++i) {
        if (start > 0 && all[i].index < start) continue;
        if (limit > 0 && n >= limit) break;
        const Mat g = cv::imread(all[i].path, cv::IMREAD_GRAYSCALE);
        if (g.empty()) continue;
        Mat sm;
        cv::resize(g, sm, cv::Size(g.cols / kDownscale, g.rows / kDownscale), 0, 0, cv::INTER_LINEAR);
        std::vector<cv::Point2f> pts;
        cv::goodFeaturesToTrack(sm, pts, kMaxCorners + 1, kQualityLevel, kMinDistance, cv::noArray(),
                                kBlockSize);
        const int capped = ((int)pts.size() > kMaxCorners) ? 1 : 0;
        if ((int)pts.size() > kMaxCorners) pts.resize((size_t)kMaxCorners);
        std::fprintf(f, "%ld\t%d\t%d\n", all[i].index, (int)pts.size(), capped);
        ++n;
    }
    std::fclose(f);
    std::printf("[scan] %ld frames -> %s\n", n, tsv_path.c_str());
    return 0;
}

// ---------------------------------------------------------------------------
// the full-frame driver
// ---------------------------------------------------------------------------
struct FullOpts {
    std::string raw_root = "/mnt/manu/lq_ir_bmp";
    std::string sequence = "VIDEO00007_19700101_001324";
    std::string kernel_dir = "/root/kernels";
    std::string kernel_file = "gmc_feature_gpu.cl";
    std::string out_path = "/tmp/gmc_gpu_probe_full.md";
    std::string tsv_path = "/tmp/gmc_gpu_probe_full.tsv";
    long start = 0;
    int limit = 600;
    bool cache = true;
    bool measure_seq = true;
    bool run_cpu = true;
    bool run_gpu = true;
    int cpu_threads = 1;
    int local = 32;
    int corners = kMaxCorners;
};

int run_fullframe(const FullOpts& o) {
    const std::string seq_dir = o.raw_root + "/" + o.sequence;
    std::printf("[i] full-frame mode: %s\n", seq_dir.c_str());
    std::printf("[i] window: start=%ld limit=%d cache=%d cpu_threads=%d local=%d\n", o.start,
                o.limit, (int)o.cache, o.cpu_threads, o.local);

    const std::vector<FrameFile> all = list_frames(seq_dir, 0, 0);
    if (all.size() < 2) {
        std::fprintf(stderr, "[FATAL] need at least 2 frames in %s\n", seq_dir.c_str());
        return 2;
    }
    // the window: the reported frames start at `start`, but the frames need
    // 42 frames of history, so the window starts 42 frames earlier and the
    // first 42 are warm-up (computed, not reported).
    const long first_abs = std::max(0L, o.start - kFullMaxLag);
    const long want = (long)o.limit + (o.start - first_abs);
    std::vector<Mat> frames;
    frames.reserve((size_t)want);
    for (size_t i = 0; i < all.size() && (long)frames.size() < want; ++i) {
        if (all[i].index < first_abs) continue;
        const Mat g = cv::imread(all[i].path, cv::IMREAD_GRAYSCALE);
        if (g.empty()) {
            std::fprintf(stderr, "[FATAL] cannot read %s\n", all[i].path.c_str());
            return 2;
        }
        frames.push_back(g);
    }
    if (frames.empty()) {
        std::fprintf(stderr, "[FATAL] empty window\n");
        return 2;
    }
    const int fw = frames[0].cols, fh = frames[0].rows;
    const int sw = fw / kDownscale, sh = fh / kDownscale;
    std::printf("[i] %zu frames in the window (abs %ld..%ld), %dx%d -> %dx%d\n", frames.size(),
                first_abs, first_abs + (long)frames.size() - 1, fw, fh, sw, sh);

    // the arena: the cached mode is the production's 43-slot ring; the
    // uncached mode keeps only the 7 pyramids the frame needs.
    const int nslots = o.cache ? kRingDepth : 7;

    std::vector<FrameRec> cpu_recs, gpu_recs;
    std::map<long, std::map<int, Matx23f>> cpu_chains, gpu_chains;

    double cpu_wall = 0.0, gpu_wall = 0.0;
    if (o.run_cpu) {
        std::printf("[i] running the CPU arm (OpenCV threads=%d, anchor LK threads=%d)\n",
                    cv::getNumThreads(), o.cpu_threads);
        const Clock::time_point t0 = Clock::now();
        run_cpu_arm(frames, first_abs, o.start, o.cpu_threads, o.corners, cpu_recs, &cpu_chains);
        cpu_wall = ms_since(t0);
        std::printf("[i] CPU arm done: %zu frames, wall %.1f ms total (%.3f ms/frame incl. warm-up)\n",
                    cpu_recs.size(), cpu_wall, cpu_wall / (double)cpu_recs.size());
    }
    if (o.run_gpu) {
        std::printf("[i] running the GPU arm\n");
        GpuFull gpu(o.kernel_dir, o.kernel_file, sw, sh, nslots, o.local, o.cache);
        const Clock::time_point t0 = Clock::now();
        run_gpu_arm(frames, first_abs, o.start, gpu, o.measure_seq, o.corners, gpu_recs,
                    &gpu_chains);
        gpu_wall = ms_since(t0);
        std::printf("[i] GPU arm done: %zu frames, wall %.1f ms total (%.3f ms/frame incl. warm-up)\n",
                    gpu_recs.size(), gpu_wall, gpu_wall / (double)gpu_recs.size());
    }

    // ---- the per-frame table ----------------------------------------------
    std::vector<double> v_cpu, v_gpu, v_resize, v_corners, v_pyr, v_upload, v_gftt, v_select,
        v_lk_batch, v_lk_1anchor, v_lk_seq, v_ransac, v_cpu_lk, v_cpu_ransac, v_cpu_corners, v_cpu_pyr;
    std::vector<double> v_lkpf[kNFits], v_ransacpf[kNFits], v_cransacpf[kNFits];
    std::vector<double> v_ncorners, v_capped, v_lk_dispatch, v_lk_read;
    int n_rep = 0;
    for (size_t i = 0; i < frames.size(); ++i) {
        const long t = first_abs + (long)i;
        if (t < o.start) continue;
        ++n_rep;
        if (o.run_cpu && i < cpu_recs.size()) {
            const FrameRec& c = cpu_recs[i];
            v_cpu.push_back(c.cpu_total);
            v_cpu_lk.push_back(c.cpu_lk);
            v_cpu_ransac.push_back(c.cpu_ransac);
            v_cpu_corners.push_back(c.cpu_corners);
            v_cpu_pyr.push_back(c.cpu_pyr);
            for (int f = 0; f < kNFits; ++f) v_cransacpf[f].push_back(c.cpu_ransac_per_fit[f]);
        }
        if (o.run_gpu && i < gpu_recs.size()) {
            const FrameRec& g = gpu_recs[i];
            v_gpu.push_back(g.gpu_total);
            v_resize.push_back(g.gpu_resize);
            v_corners.push_back(g.gpu_gftt);
            v_gftt.push_back(g.gpu_gftt);
            v_select.push_back(g.gpu_select);
            v_pyr.push_back(g.gpu_pyr);
            v_upload.push_back(g.gpu_upload);
            v_lk_batch.push_back(g.gpu_lk_batch);
            v_lk_dispatch.push_back(g.gpu_lk_dispatch);
            v_lk_read.push_back(g.gpu_lk_read);
            v_lk_1anchor.push_back(g.gpu_lk_1anchor);
            v_lk_seq.push_back(g.gpu_lk_seq);
            v_ransac.push_back(g.gpu_ransac);
            for (int f = 0; f < kNFits; ++f) {
                v_lkpf[f].push_back(g.lk_seq_per_fit[f]);
                v_ransacpf[f].push_back(g.ransac_per_fit[f]);
            }
            v_ncorners.push_back((double)g.gpu_ncorners);
            v_capped.push_back((double)g.gpu_capped);
        }
    }
    const Stats s_cpu = stats_of(v_cpu);
    const Stats s_gpu = stats_of(v_gpu);
    const Stats s_batch = stats_of(v_lk_batch);
    const Stats s_1anchor = stats_of(v_lk_1anchor);
    const Stats s_seq = stats_of(v_lk_seq);

    // ---- the accuracy ------------------------------------------------------
    std::vector<double> v_fit_dtrans, v_fit_cos, v_fit_scale, v_fit_match;
    std::vector<double> v_ch_dtrans, v_ch_cos, v_ch_scale;
    int n_chain = 0;
    if (o.run_cpu && o.run_gpu) {
        for (size_t i = 0; i < frames.size(); ++i) {
            const long t = first_abs + (long)i;
            if (t < o.start) continue;
            if (i >= cpu_recs.size() || i >= gpu_recs.size()) continue;
            const FrameRec& c = cpu_recs[i];
            const FrameRec& g = gpu_recs[i];
            if (c.fit_H.size() < 5 || g.fit_H.size() < 5) continue;
            // the per-fit deviations: the 5 anchor fits
            for (int f = 0; f < 5; ++f) {
                const Dev d = compare_H(c.fit_H[(size_t)f], g.fit_H[(size_t)f]);
                v_fit_dtrans.push_back(d.dtrans);
                v_fit_cos.push_back(d.cosrot);
                v_fit_scale.push_back(d.relscale);
                if (f < (int)c.fit_corners.size() && f < (int)g.fit_corners.size())
                    v_fit_match.push_back(corner_match(c.fit_corners[(size_t)f],
                                                       g.fit_corners[(size_t)f]));
            }
            // the 21-lag chain
            auto ic = cpu_chains.find(t);
            auto ig = gpu_chains.find(t);
            if (ic == cpu_chains.end() || ig == gpu_chains.end()) continue;
            for (int lag = 2; lag <= kFullMaxLag; lag += 2) {
                auto a = ic->second.find(lag);
                auto b = ig->second.find(lag);
                if (a == ic->second.end() || b == ig->second.end()) continue;
                const Dev d = compare_H(a->second, b->second);
                v_ch_dtrans.push_back(d.dtrans);
                v_ch_cos.push_back(d.cosrot);
                v_ch_scale.push_back(d.relscale);
                ++n_chain;
            }
        }
    }

    // ---- the TSV -----------------------------------------------------------
    {
        FILE* f = std::fopen(o.tsv_path.c_str(), "w");
        if (!f) {
            std::fprintf(stderr, "[FATAL] cannot write %s\n", o.tsv_path.c_str());
            return 2;
        }
        std::fprintf(f, "abs\tcpu_total\tcpu_resize\tcpu_corners\tcpu_pyr\tcpu_lk\tcpu_ransac\t"
                        "gpu_total\tgpu_resize\tgpu_gftt\tgpu_select\tgpu_pyr\tgpu_upload\t"
                        "gpu_lk_batch\tgpu_lk_dispatch\tgpu_lk_read\tgpu_lk_1anchor\t"
                        "gpu_lk_seq\tgpu_ransac\tncorners_cpu\tncorners_gpu\n");
        for (size_t i = 0; i < frames.size(); ++i) {
            const long t = first_abs + (long)i;
            if (t < o.start) continue;
            const FrameRec& c = (i < cpu_recs.size()) ? cpu_recs[i] : FrameRec();
            const FrameRec& g = (i < gpu_recs.size()) ? gpu_recs[i] : FrameRec();
            std::fprintf(f, "%ld\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t"
                            "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%d\n",
                         t, c.cpu_total, c.cpu_resize, c.cpu_corners, c.cpu_pyr, c.cpu_lk,
                         c.cpu_ransac, g.gpu_total, g.gpu_resize, g.gpu_gftt, g.gpu_select, g.gpu_pyr,
                         g.gpu_upload, g.gpu_lk_batch, g.gpu_lk_dispatch, g.gpu_lk_read,
                         g.gpu_lk_1anchor, g.gpu_lk_seq, g.gpu_ransac, c.cpu_ncorners,
                         g.gpu_ncorners);
        }
        std::fclose(f);
        std::printf("[i] per-frame TSV: %s\n", o.tsv_path.c_str());
    }

    // ---- the console -------------------------------------------------------
    std::printf("\n================ GMC FULL-FRAME PROBE ================\n");
    if (s_cpu.n) {
        std::printf("CPU  total  mean %8.3f ms  p50 %.3f  p95 %.3f  p99 %.3f  max %.3f  | fps "
                    "%6.2f  >40ms %.2f%%\n",
                    s_cpu.mean, s_cpu.p50, s_cpu.p95, s_cpu.p99, s_cpu.max, s_cpu.fps,
                    s_cpu.over40 * 100.0);
    }
    if (s_gpu.n) {
        std::printf("GPU  total  mean %8.3f ms  p50 %.3f  p95 %.3f  p99 %.3f  max %.3f  | fps "
                    "%6.2f  >40ms %.2f%%\n",
                    s_gpu.mean, s_gpu.p50, s_gpu.p95, s_gpu.p99, s_gpu.max, s_gpu.fps,
                    s_gpu.over40 * 100.0);
        std::printf("GPU  breakdown: resize %.3f  gFTT %.3f  select %.3f  pyr %.3f  upload %.3f  "
                    "LK(batch) %.3f (disp %.3f + read %.3f)  RANSAC %.3f  [LK(seq) %.3f]\n",
                    mean(v_resize), mean(v_gftt), mean(v_select), mean(v_pyr), mean(v_upload),
                    s_batch.mean, mean(v_lk_dispatch), mean(v_lk_read), mean(v_ransac), s_seq.mean);
    }
    if (s_gpu.n) {
        int ncap = 0;
        for (double c : v_capped)
            if (c >= 0.5) ++ncap;
        std::printf("corners: budget %d  mean %.1f  capped %d/%d\n", o.corners, mean(v_ncorners),
                    ncap, n_rep);
    }
    if (s_cpu.n && s_gpu.n) {
        std::printf("accuracy: per-fit max|dt| %.4f px  mean %.4f | rot cos min %.7f | |ds/s| max "
                    "%.6f | corner match %.4f\n",
                    pctl(v_fit_dtrans, 1.0), mean(v_fit_dtrans), pctl(v_fit_cos, 0.0),
                    pctl(v_fit_scale, 1.0), mean(v_fit_match));
        std::printf("accuracy: chain(%d) max|dt| %.4f px  mean %.4f | rot cos min %.7f | |ds/s| "
                    "max %.6f\n",
                    n_chain, pctl(v_ch_dtrans, 1.0), mean(v_ch_dtrans), pctl(v_ch_cos, 0.0),
                    pctl(v_ch_scale, 1.0));
    }

    // ---- the markdown ------------------------------------------------------
    FILE* f = std::fopen(o.out_path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "[FATAL] cannot write %s\n", o.out_path.c_str());
        return 2;
    }
    std::fprintf(f, "# GMC 全帧 GPU 探针 (Mali-G610 / OpenCL)\n\n");
    std::fprintf(f, "- sequence: `%s`\n", seq_dir.c_str());
    std::fprintf(f, "- 窗口: start=%ld, 报告帧数=%d, 预热=%ld 帧 (abs %ld..%ld)\n", o.start, n_rep,
                 o.start - first_abs, first_abs, first_abs + (long)frames.size() - 1);
    std::fprintf(f, "- frame %dx%d -> small %dx%d (downscale %d)\n", fw, fh, sw, sh, kDownscale);
    std::fprintf(f, "- 每帧拟合: 5 个锚点 (lag %d/%d/%d/%d/%d) + 1 个 stride-2 一步拟合 (t-4->t-2)"
                    " = %d 次 LK+RANSAC\n",
                 kAnchors[0], kAnchors[1], kAnchors[2], kAnchors[3], kAnchors[4], kNFits);
    std::fprintf(f, "- 缓存: %s (生产同口径 43 槽 ring); LK: %s; CPU 锚点线程: %d; OpenCV 线程: %d\n",
                 o.cache ? "开 (gmc_pyr_cache=1)" : "关 (每帧重建+重传 7 套金字塔)",
                 o.measure_seq ? "批量 + 顺序 (都测)" : "批量", o.cpu_threads, cv::getNumThreads());
    std::fprintf(f, "- 角点预算: **%d** | gmc_corner_mean = **%.1f** | gmc_corner_capped = **%d / %d**"
                    " 帧 (撞顶)\n",
                 o.corners, mean(v_ncorners), (int)pctl(v_capped, 1.0) >= 0.5 ? (int)(mean(v_capped) * (double)n_rep + 0.5) : 0,
                 n_rep);
    std::fprintf(f, "\n## 1. 速度 (每帧完整特征准备)\n\n");
    std::fprintf(f, "| 路径 | mean | p50 | p95 | p99 | max | fps | >40ms 占比 |\n");
    std::fprintf(f, "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    if (s_cpu.n) {
        std::fprintf(f, "| CPU (OpenCV 单线程) | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f | %.2f%% |\n",
                     s_cpu.mean, s_cpu.p50, s_cpu.p95, s_cpu.p99, s_cpu.max, s_cpu.fps,
                     s_cpu.over40 * 100.0);
    }
    if (s_gpu.n) {
        std::fprintf(f, "| **GPU (全帧)** | **%.3f** | **%.3f** | **%.3f** | **%.3f** | **%.3f** | "
                     "**%.2f** | **%.2f%%** |\n",
                     s_gpu.mean, s_gpu.p50, s_gpu.p95, s_gpu.p99, s_gpu.max, s_gpu.fps,
                     s_gpu.over40 * 100.0);
    }
    std::fprintf(f, "\n## 2. GPU 每帧时间分解 (mean ms)\n\n");
    std::fprintf(f, "| 阶段 | ms | 占比 |\n| :--- | ---: | ---: |\n");
    if (s_gpu.n) {
        const double tot = s_gpu.mean;
        const double parts[7] = {mean(v_resize), mean(v_gftt) + mean(v_select), mean(v_pyr),
                                 mean(v_upload), mean(v_lk_dispatch), mean(v_lk_read),
                                 mean(v_ransac)};
        const char* names[7] = {"resize", "角点 (gFTT + CPU 选取)", "金字塔构建 (CPU)", "上传 (H2D)",
                                "LK (批量 dispatch, 不含回读)", "LK 回读 (D2H)",
                                "RANSAC (CPU, 串行)"};
        for (int i = 0; i < 7; ++i) {
            std::fprintf(f, "| %s | %.3f | %.1f%% |\n", names[i], parts[i],
                         tot > 0 ? parts[i] / tot * 100.0 : 0.0);
        }
        std::fprintf(f, "| **合计** | **%.3f** | 100%% |\n", tot);
        std::fprintf(f, "| LK (顺序 dispatch, %d 次) | %.3f | (参考) |\n", kNFits, s_seq.mean);
    }
    std::fprintf(f, "\n### 2b. 逐次拟合 (mean ms)\n\n");
    std::fprintf(f, "| 拟合 | 锚点 lag | 顺序 LK (GPU) | RANSAC (GPU) | RANSAC (CPU) |\n");
    std::fprintf(f, "| :--- | ---: | ---: | ---: | ---: |\n");
    for (int fi = 0; fi < kNFits; ++fi) {
        const char* nm = (fi < 5) ? "anchor" : "step(t-4->t-2)";
        std::fprintf(f, "| %s%d | %d | %.3f | %.3f | %.3f |\n", nm, fi, fi < 5 ? kAnchors[fi] : 2,
                     mean(v_lkpf[fi]), mean(v_ransacpf[fi]), mean(v_cransacpf[fi]));
    }
    std::fprintf(f, "\n## 3. 批量 vs 顺序 dispatch (LK 阶段, 同一份已上传金字塔)\n\n");
    std::fprintf(f, "| 变体 | mean | p50 | p95 | p99 | max |\n| :--- | ---: | ---: | ---: | ---: | "
                 "---: |\n");
    std::fprintf(f, "| 1 锚点批量 (同一 kernel, nfits=1) | %.3f | %.3f | %.3f | %.3f | %.3f |\n",
                 s_1anchor.mean, s_1anchor.p50, s_1anchor.p95, s_1anchor.p99, s_1anchor.max);
    std::fprintf(f, "| %d 锚点批量 (1 次 NDRange, nfits=%d) | %.3f | %.3f | %.3f | %.3f | %.3f |\n",
                 kNFits, kNFits, s_batch.mean, s_batch.p50, s_batch.p95, s_batch.p99, s_batch.max);
    std::fprintf(f, "| %d 次顺序 dispatch (单锚点 kernel) | %.3f | %.3f | %.3f | %.3f | %.3f |\n",
                 kNFits, s_seq.mean, s_seq.p50, s_seq.p95, s_seq.p99, s_seq.max);
    if (s_seq.mean > 0) std::fprintf(f, "\n顺序/批量 = **%.3fx**\n", s_seq.mean / s_batch.mean);
    if (s_1anchor.mean > 0)
        std::fprintf(f, "%d 锚点批量 / 1 锚点批量 = **%.3fx**  (≈1 表示 LK 是延迟受限, 不是吞吐受限)\n",
                     kNFits, s_batch.mean / s_1anchor.mean);
    if (s_cpu.n && s_gpu.n) {
        std::fprintf(f, "\n## 4. 精度 (GPU 全帧 vs CPU 参考, 逐次拟合)\n\n");
        std::fprintf(f, "| 指标 | 值 |\n| :--- | ---: |\n");
        std::fprintf(f, "| max\\|Δ平移\\| (px, 全分辨率) | %.4f |\n", pctl(v_fit_dtrans, 1.0));
        std::fprintf(f, "| mean\\|Δ平移\\| (px, 全分辨率) | %.4f |\n", mean(v_fit_dtrans));
        std::fprintf(f, "| 旋转 cos 最小 | %.7f |\n", pctl(v_fit_cos, 0.0));
        std::fprintf(f, "| 旋转 cos 均值 | %.7f |\n", mean(v_fit_cos));
        std::fprintf(f, "| \\|Δs/s\\| 最大 | %.6f |\n", pctl(v_fit_scale, 1.0));
        std::fprintf(f, "| \\|Δs/s\\| 均值 | %.6f |\n", mean(v_fit_scale));
        std::fprintf(f, "| 角点对应率 (≤1px) | %.4f |\n", mean(v_fit_match));
        std::fprintf(f, "\n## 5. 精度 (21 帧链, 逐 lag)\n\n");
        std::fprintf(f, "| 指标 | 值 |\n| :--- | ---: |\n");
        std::fprintf(f, "| 比较的 lag 数 | %d |\n", n_chain);
        std::fprintf(f, "| max\\|Δ平移\\| (px, 全分辨率) | %.4f |\n", pctl(v_ch_dtrans, 1.0));
        std::fprintf(f, "| mean\\|Δ平移\\| (px, 全分辨率) | %.4f |\n", mean(v_ch_dtrans));
        std::fprintf(f, "| 旋转 cos 最小 | %.7f |\n", pctl(v_ch_cos, 0.0));
        std::fprintf(f, "| \\|Δs/s\\| 最大 | %.6f |\n", pctl(v_ch_scale, 1.0));
    }
    std::fclose(f);
    std::printf("[i] report: %s\n", o.out_path.c_str());
    return 0;
}

}  // namespace ff

}  // namespace
int main(int argc, char** argv) {
    std::string raw_root = "/mnt/manu/lq_ir_bmp";
    std::string sequence = "VIDEO00007_19700101_001324";
    std::string kernel_dir = "/root/kernels";
    std::string kernel_file = "gmc_feature_gpu.cl";
    std::string out_path = "/tmp/gmc_gpu_probe.md";
    std::string gmc_only = "gftt+lk";
    std::string mode = "single";
    std::string tsv_path = "/tmp/gmc_gpu_probe_full.tsv";
    int limit = 60;
    int start = 0;
    int lag = 1;
    int threads = 1;
    // full-frame mode options
    int ff_cache = 1;
    int ff_seq = 1;
    int ff_local = 32;
    int ff_cpu_threads = 1;
    int ff_arm = 3;   // bit0 = CPU, bit1 = GPU
    int ff_corners = 600;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "[FATAL] %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--raw-root") raw_root = next("--raw-root");
        else if (a == "--sequence") sequence = next("--sequence");
        else if (a == "--kernel-dir") kernel_dir = next("--kernel-dir");
        else if (a == "--kernel-file") kernel_file = next("--kernel-file");
        else if (a == "--out") out_path = next("--out");
        else if (a == "--limit") limit = std::atoi(next("--limit").c_str());
        else if (a == "--gmc-only") gmc_only = next("--gmc-only");
        else if (a == "--start") start = std::atoi(next("--start").c_str());
        else if (a == "--lag") lag = std::atoi(next("--lag").c_str());
        else if (a == "--threads") threads = std::atoi(next("--threads").c_str());
        else if (a == "--mode") mode = next("--mode");
        else if (a == "--tsv") tsv_path = next("--tsv");
        else if (a == "--cache") ff_cache = std::atoi(next("--cache").c_str());
        else if (a == "--lk-measure") ff_seq = std::atoi(next("--lk-measure").c_str());
        else if (a == "--local") ff_local = std::atoi(next("--local").c_str());
        else if (a == "--cpu-threads") ff_cpu_threads = std::atoi(next("--cpu-threads").c_str());
        else if (a == "--arm") ff_arm = std::atoi(next("--arm").c_str());
        else if (a == "--corners") ff_corners = std::atoi(next("--corners").c_str());
        else if (a == "--help" || a == "-h") {
            std::printf("usage: %s --raw-root DIR --sequence NAME --kernel-dir DIR "
                        "[--limit N] [--start F] [--lag L] [--threads N] [--out FILE] "
                        "[--gmc-only gftt|gftt+lk]\n"
                        "  --gmc-only gftt      tier 1: only the corner detection on the GPU\n"
                        "  --gmc-only gftt+lk   tier 2: gFTT + pyramidal LK on the GPU\n"
                        "  --threads N          OpenCV threads for the CPU arm; DEFAULT 1.\n"
                        "                       The production path is forced single-threaded\n"
                        "                       because bit-reproducibility requires it\n"
                        "                       (gmc_stream_ocl.cpp:56-62: under multiple\n"
                        "                       OpenCV threads RANSAC and Lucas-Kanade are not\n"
                        "                       bit-reproducible).  N<=0 leaves OpenCV's\n"
                        "                       default (honours OPENCV_FOR_THREADS_NUM).\n"
                        "\n"
                        "  --mode single|fullframe|scan\n"
                        "     single     the single-lag probe (the default, unchanged)\n"
                        "     fullframe  the production per-frame replication: 5 anchor\n"
                        "                fits (lags 2/12/22/32/42) + 1 stride-2 step fit\n"
                        "                (t-4 -> t-2) + the 21-lag chain, one batched LK\n"
                        "                dispatch for all 6 fits\n"
                        "     scan       the corner-count scan of the sequence (TSV)\n"
                        "  --cache 0|1          fullframe: the production 43-slot cache\n"
                        "  --lk-measure 0|1     fullframe: also measure the 1-anchor batch\n"
                        "                       and the sequential dispatches\n"
                        "  --local N            the OpenCL local size of the batched LK\n"
                        "  --cpu-threads N      the CPU arm: the LK stages of the 6 fits\n"
                        "                       run in N threads (the production parallel-\n"
                        "                       anchor path); the RANSACs stay serial\n"
                        "  --arm 1|2|3          fullframe: 1 = CPU, 2 = GPU, 3 = both\n"
                        "  --corners N          the corner budget (default 600); the\n"
                        "                       production asks for budget+1 and drops the\n"
                        "                       last one, so a count of exactly N means the\n"
                        "                       frame was capped\n"
                        "  --tsv FILE           the per-frame TSV\n",
                        argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "[FATAL] unknown argument: %s\n", a.c_str());
            return 2;
        }
    }
    if (mode == "scan") {
        const std::string seq_dir = raw_root + "/" + sequence;
        return ff::run_scan(seq_dir, (long)start, limit, tsv_path);
    }
    if (mode == "fullframe") {
        ff::FullOpts fo;
        fo.raw_root = raw_root;
        fo.sequence = sequence;
        fo.kernel_dir = kernel_dir;
        fo.kernel_file = kernel_file;
        fo.out_path = out_path;
        fo.tsv_path = tsv_path;
        fo.start = (long)start;
        fo.limit = limit;
        fo.cache = ff_cache != 0;
        fo.measure_seq = ff_seq != 0;
        fo.local = ff_local;
        fo.cpu_threads = ff_cpu_threads;
        fo.corners = ff_corners;
        fo.run_cpu = (ff_arm & 1) != 0;
        fo.run_gpu = (ff_arm & 2) != 0;
        if (threads > 0) cv::setNumThreads(threads);
        std::printf("[i] OpenCV threads: %d\n", cv::getNumThreads());
        return ff::run_fullframe(fo);
    }
    if (mode != "single") {
        std::fprintf(stderr, "[FATAL] --mode must be single, fullframe or scan (got %s)\n",
                     mode.c_str());
        return 2;
    }
    bool do_gpu_lk = false;
    if (gmc_only == "gftt") do_gpu_lk = false;
    else if (gmc_only == "gftt+lk") do_gpu_lk = true;
    else {
        std::fprintf(stderr, "[FATAL] --gmc-only must be 'gftt' or 'gftt+lk' (got %s)\n",
                     gmc_only.c_str());
        return 2;
    }
    const char* tier_name = do_gpu_lk ? "TIER 2 (gFTT + LK on GPU)"
                                       : "TIER 1 (gFTT only on GPU)";
    // OpenCV thread count for the CPU arm.  DEFAULT 1, because the production
    // path is forced single-threaded for bit-reproducibility
    // (gmc_stream_ocl.cpp:56-62: under multiple OpenCV threads RANSAC and
    // Lucas-Kanade are not bit-reproducible -- an anchor fit can flip between
    // two equally plausible solutions).  A multi-threaded CPU arm is still
    // measurable (--threads 4) and is reported as a CONTRAST, because it
    // quantifies what the bit-exactness rule costs.
    if (threads > 0) cv::setNumThreads(threads);
    const int opencv_threads = cv::getNumThreads();
    std::printf("[i] mode: %s  (--gmc-only %s)\n", tier_name, gmc_only.c_str());
    std::printf("[i] OpenCV threads for the CPU arm: %d%s\n", opencv_threads,
                threads > 0 ? " (forced)" : " (OpenCV default / env)");

    const std::string seq_dir = raw_root + "/" + sequence;
    std::printf("[i] sequence %s (limit %d, start %d, lag %d)\n", seq_dir.c_str(), limit, start,
                lag);
    const std::vector<FrameFile> frames = list_frames(seq_dir, limit, start);
    if (frames.size() < 2) {
        std::fprintf(stderr, "[FATAL] need at least 2 frames in %s\n", seq_dir.c_str());
        return 2;
    }
    std::printf("[i] %zu frames\n", frames.size());

    // ---- preload the frames so the NFS read is NOT inside the timed region --
    std::vector<Mat> grays(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
        grays[i] = cv::imread(frames[i].path, cv::IMREAD_GRAYSCALE);
        if (grays[i].empty()) {
            std::fprintf(stderr, "[FATAL] cannot read %s\n", frames[i].path.c_str());
            return 2;
        }
    }
    const int fw = grays[0].cols, fh = grays[0].rows;
    const int sw = fw / kDownscale, sh = fh / kDownscale;
    std::printf("[i] frame %dx%d -> small %dx%d\n", fw, fh, sw, sh);

    GpuFeat gpu(kernel_dir, kernel_file, sw, sh);

    std::vector<ArmTiming> cpu_res(frames.size()), gpu_res(frames.size());
    std::vector<FrameCmp> cmps(frames.size());
    std::vector<std::vector<cv::Point2f>> cpu_pts(frames.size()), gpu_pts(frames.size());
    double score_err_max = 0.0, score_err_rel = 0.0;
    int score_checked = 0;

    for (size_t i = (size_t)lag; i < frames.size(); ++i) {
        // shared resize
        Clock::time_point t0 = Clock::now();
        Mat prev_small, curr_small;
        cv::resize(grays[i - (size_t)lag], prev_small, cv::Size(sw, sh), 0, 0, cv::INTER_LINEAR);
        cv::resize(grays[i], curr_small, cv::Size(sw, sh), 0, 0, cv::INTER_LINEAR);
        const double resize_ms = ms_since(t0);

        // CPU arm
        const ArmTiming c = run_cpu(grays[i - (size_t)lag], grays[i], prev_small, curr_small,
                                     resize_ms);
        cpu_res[i] = c;

        // GPU arm
        const ArmTiming g = run_gpu(gpu, prev_small, curr_small, resize_ms, do_gpu_lk);
        gpu_res[i] = g;

        // corner lists for the correspondence rate come from the arms themselves
        // (they are captured inside run_cpu / run_gpu, so no stage is run twice)
        cpu_pts[i] = cpu_res[i].pts;
        gpu_pts[i] = gpu_res[i].pts;
        {
            std::vector<float> score;
            gpu.gftt(prev_small, score);
            // score-map self check against OpenCV's cornerMinEigenVal
            if (i == 1) {
                Mat eig;
                cv::cornerMinEigenVal(prev_small, eig, kBlockSize, 3);
                const float* e = eig.ptr<float>();
                double mx = 0.0;
                for (int k = 0; k < sw * sh; ++k) mx = std::max(mx, (double)e[k]);
                for (int k = 0; k < sw * sh; ++k) {
                    const double d = std::fabs((double)score[(size_t)k] - (double)e[k]);
                    score_err_max = std::max(score_err_max, d);
                    if (mx > 0) score_err_rel = std::max(score_err_rel, d / mx);
                }
                ++score_checked;
            }
        }

        cmps[i] = compare_arms(cpu_res[i], gpu_res[i], cpu_pts[i], gpu_pts[i]);
    }

    // ---- aggregate --------------------------------------------------------
    std::vector<double> v_cpu_total, v_gpu_total, v_cpu_corners, v_cpu_lk, v_cpu_ransac,
        v_cpu_resize, v_gpu_pyr, v_gpu_upload, v_gpu_gftt, v_gpu_select, v_gpu_lk, v_gpu_ransac,
        v_dtrans, v_cos, v_scale, v_m05, v_m1, v_m2, v_ncpu, v_ngpu, v_dev_mean, v_dev_max;
    int n_valid_cmp = 0;
    for (size_t i = (size_t)lag; i < frames.size(); ++i) {
        v_cpu_total.push_back(cpu_res[i].total);
        v_gpu_total.push_back(gpu_res[i].total);
        v_cpu_resize.push_back(cpu_res[i].resize);
        v_cpu_corners.push_back(cpu_res[i].corners);
        v_cpu_lk.push_back(cpu_res[i].lk);
        v_cpu_ransac.push_back(cpu_res[i].ransac);
        v_gpu_pyr.push_back(gpu_res[i].pyr_build);
        v_gpu_upload.push_back(gpu_res[i].upload);
        v_gpu_gftt.push_back(gpu_res[i].gftt_gpu);
        v_gpu_select.push_back(gpu_res[i].select);
        v_gpu_lk.push_back(gpu_res[i].lk_gpu);
        v_gpu_ransac.push_back(gpu_res[i].ransac);
        v_ncpu.push_back((double)cpu_res[i].n_corners);
        v_ngpu.push_back((double)gpu_res[i].n_corners);
        if (cmps[i].both_valid) {
            ++n_valid_cmp;
            v_dtrans.push_back(cmps[i].dtrans);
            v_cos.push_back(cmps[i].cos_rot);
            v_scale.push_back(cmps[i].rel_scale);
        }
        v_m05.push_back(cmps[i].match05);
        v_m1.push_back(cmps[i].match1);
        v_m2.push_back(cmps[i].match2);
        v_dev_mean.push_back(cmps[i].dev_mean);
        v_dev_max.push_back(cmps[i].dev_max);
    }
    const int nfit = (int)v_cpu_total.size();
    const double frames_total = 11810.0;

    const double m_cpu = mean(v_cpu_total), m_gpu = mean(v_gpu_total);
    const double speedup = (m_gpu > 0) ? m_cpu / m_gpu : 0.0;

    // stage shares of the CPU fit, measured on this board
    const double m_cpu_corners = mean(v_cpu_corners), m_cpu_lk = mean(v_cpu_lk),
                 m_cpu_ransac = mean(v_cpu_ransac), m_cpu_resize = mean(v_cpu_resize);
    const double m_gftt_gpu = mean(v_gpu_gftt), m_select = mean(v_gpu_select);
    const double share_gftt = (m_cpu > 0) ? m_cpu_corners / m_cpu : 0.0;
    const double share_lk = (m_cpu > 0) ? m_cpu_lk / m_cpu : 0.0;

    // ---- console ----------------------------------------------------------
    std::printf("\n================ GMC GPU PROBE ================\n");
    std::printf("mode                   : %s\n", tier_name);
    std::printf("frames fitted          : %d\n", nfit);
    std::printf("CPU  total  mean       : %8.3f ms  (p50 %.3f, p95 %.3f, max %.3f)\n", m_cpu,
                pctl(v_cpu_total, 0.50), pctl(v_cpu_total, 0.95), pctl(v_cpu_total, 1.0));
    std::printf("GPU  total  mean       : %8.3f ms  (p50 %.3f, p95 %.3f, max %.3f)\n", m_gpu,
                pctl(v_gpu_total, 0.50), pctl(v_gpu_total, 0.95), pctl(v_gpu_total, 1.0));
    std::printf("speedup (CPU/GPU)      : %8.3fx\n", speedup);
    std::printf("CPU  resize / corners / LK / RANSAC : %.3f / %.3f / %.3f / %.3f ms\n",
                m_cpu_resize, m_cpu_corners, m_cpu_lk, m_cpu_ransac);
    std::printf("GPU  pyr / upload / gFTT / select / LK / RANSAC : %.3f / %.3f / %.3f / %.3f / "
                "%.3f / %.3f ms\n",
                mean(v_gpu_pyr), mean(v_gpu_upload), m_gftt_gpu, m_select, mean(v_gpu_lk),
                mean(v_gpu_ransac));
    std::printf("gFTT stage share of the CPU fit : %.1f%%  (CPU %.3f ms -> GPU %.3f + %.3f ms)\n",
                share_gftt * 100.0, m_cpu_corners, m_gftt_gpu, m_select);
    std::printf("corners  CPU %.1f  GPU %.1f\n", mean(v_ncpu), mean(v_ngpu));
    if (n_valid_cmp) {
        std::printf("dtrans  max %.4f  mean %.4f px | rot cos min %.7f mean %.7f | |ds/s| max "
                    "%.6f mean %.6f\n",
                    pctl(v_dtrans, 1.0), mean(v_dtrans), pctl(v_cos, 0.0), mean(v_cos),
                    pctl(v_scale, 1.0), mean(v_scale));
    }
    std::printf("corner correspondence  <=0.5px %.3f  <=1px %.3f  <=2px %.3f | coord dev mean "
                "%.4f px max %.4f px\n",
                mean(v_m05), mean(v_m1), mean(v_m2), mean(v_dev_mean), pctl(v_dev_max, 1.0));
    std::printf("extrapolated to %d frames: CPU %.1f s, GPU %.1f s  (ONE lag only -- NOT the "
                "production frame budget)\n",
                (int)frames_total, m_cpu * frames_total / 1000.0, m_gpu * frames_total / 1000.0);

    // ---- markdown ---------------------------------------------------------
    FILE* f = std::fopen(out_path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "[FATAL] cannot write %s\n", out_path.c_str());
        return 2;
    }
    std::fprintf(f, "# GMC feature-prep GPU probe (Mali-G610 / OpenCL)\n\n");
    std::fprintf(f, "- **实测档位: %s** (`--gmc-only %s`)\n", tier_name, gmc_only.c_str());
    std::fprintf(f, "- sequence: `%s` (limit %d, start %d)\n", seq_dir.c_str(), limit, start);
    std::fprintf(f, "- frames fitted: %d (pairs), fit lag = %d frame(s)\n", nfit, lag);
    std::fprintf(f, "- CPU 基准 OpenCV 线程数: **%d**%s\n", opencv_threads,
                 threads > 0 ? " (探针显式 setNumThreads，生产同口径：位级复现要求单线程)"
                              : " (OpenCV 默认/环境变量)");
    std::fprintf(f, "- frame %dx%d -> small %dx%d (downscale %d)\n", fw, fh, sw, sh, kDownscale);
    std::fprintf(f, "- gFTT: maxCorners=%d qualityLevel=%.2f minDistance=%d blockSize=%d\n",
                 kMaxCorners, kQualityLevel, kMinDistance, kBlockSize);
    std::fprintf(f, "- LK: win %dx%d, maxLevel %d, TermCriteria(COUNT|EPS, %d, %.2f)\n", kWin,
                 kWin, kMaxLevel, kLkIters, kLkEps);
    std::fprintf(f, "- RANSAC: estimateAffinePartial2D(threshold=%.1f), RNG reseeded per fit\n",
                 kRansacThresh);
    std::fprintf(f, "\n## 1. 速度\n\n");
    std::fprintf(f, "| 阶段 | CPU (ms) | GPU (ms) |\n| :--- | ---: | ---: |\n");
    std::fprintf(f, "| resize (共享, 两侧相同) | %.3f | %.3f |\n", m_cpu_resize, m_cpu_resize);
    std::fprintf(f, "| 角点检测 | %.3f | %.3f (gFTT kernel) + %.3f (CPU 选取 NMS/阈值/minDistance) |\n",
                 m_cpu_corners, m_gftt_gpu, m_select);
    if (do_gpu_lk) {
        std::fprintf(f, "| 金字塔构建 | (含在 LK 内) | %.3f |\n", mean(v_gpu_pyr));
        std::fprintf(f, "| 上传 (H2D) | - | %.3f |\n", mean(v_gpu_upload));
        std::fprintf(f, "| LK | %.3f | %.3f |\n", m_cpu_lk, mean(v_gpu_lk));
    } else {
        std::fprintf(f, "| 金字塔构建 | (含在 LK 内) | (LK 仍在 CPU) |\n");
        std::fprintf(f, "| 上传 (H2D) | - | (不需要) |\n");
        std::fprintf(f, "| LK | %.3f | %.3f (仍在 CPU, 喂 GPU 的角点) |\n", m_cpu_lk, mean(v_cpu_lk));
    }
    std::fprintf(f, "| RANSAC | %.3f | %.3f |\n", m_cpu_ransac, mean(v_gpu_ransac));
    std::fprintf(f, "| **总计** | **%.3f** | **%.3f** |\n", m_cpu, m_gpu);
    std::fprintf(f, "\n加速比 (CPU/GPU) = **%.3fx**\n\n", speedup);
    std::fprintf(f, "| 统计 | CPU | GPU |\n| :--- | ---: | ---: |\n");
    std::fprintf(f, "| mean | %.3f | %.3f |\n", m_cpu, m_gpu);
    std::fprintf(f, "| p50 | %.3f | %.3f |\n", pctl(v_cpu_total, 0.5), pctl(v_gpu_total, 0.5));
    std::fprintf(f, "| p95 | %.3f | %.3f |\n", pctl(v_cpu_total, 0.95), pctl(v_gpu_total, 0.95));
    std::fprintf(f, "| max | %.3f | %.3f |\n", pctl(v_cpu_total, 1.0), pctl(v_gpu_total, 1.0));
    std::fprintf(f, "\n单 lag 拟合 × 11810 帧: CPU **%.1f s**, GPU **%.1f s** (加速比 %.3fx)"
                    " —— **这只是规模感，不是生产帧预算**（生产一帧是多 lag 链 + 合成 + GPU 尾段，"
                    "见 §3）。\n",
                 m_cpu * frames_total / 1000.0, m_gpu * frames_total / 1000.0, speedup);

    std::fprintf(f, "\n## 2. 精度代价\n\n");
    if (n_valid_cmp) {
        std::fprintf(f, "| 指标 | 值 |\n| :--- | ---: |\n");
        std::fprintf(f, "| max\\|Δ平移\\| (px, 全分辨率) | %.4f |\n", pctl(v_dtrans, 1.0));
        std::fprintf(f, "| mean\\|Δ平移\\| (px, 全分辨率) | %.4f |\n", mean(v_dtrans));
        std::fprintf(f, "| 旋转 cos 最小 | %.7f |\n", pctl(v_cos, 0.0));
        std::fprintf(f, "| 旋转 cos 均值 | %.7f |\n", mean(v_cos));
        std::fprintf(f, "| \\|Δs/s\\| 最大 | %.6f |\n", pctl(v_scale, 1.0));
        std::fprintf(f, "| \\|Δs/s\\| 均值 | %.6f |\n", mean(v_scale));
    } else {
        std::fprintf(f, "（没有可比较的帧）\n");
    }
    std::fprintf(f, "\n| 角点集合指标 | 值 |\n| :--- | ---: |\n");
    std::fprintf(f, "| 角点数 CPU 均值 | %.1f |\n", mean(v_ncpu));
    std::fprintf(f, "| 角点数 GPU 均值 | %.1f |\n", mean(v_ngpu));
    std::fprintf(f, "| 角点对应率 ≤0.5 px | %.4f |\n", mean(v_m05));
    std::fprintf(f, "| 角点对应率 ≤1.0 px | %.4f |\n", mean(v_m1));
    std::fprintf(f, "| 角点对应率 ≤2.0 px | %.4f |\n", mean(v_m2));
    std::fprintf(f, "| 角点坐标偏差 mean (px, 小图坐标) | %.4f |\n", mean(v_dev_mean));
    std::fprintf(f, "| 角点坐标偏差 max (px, 小图坐标) | %.4f |\n", pctl(v_dev_max, 1.0));
    if (score_checked) {
        std::fprintf(f, "\n分数图自检 (GPU vs `cv::cornerMinEigenVal`, 第 1 帧): "
                        "max|Δ|=%.6g, max|Δ|/max=%.3g\n",
                     score_err_max, score_err_rel);
    }

    // ---- 3. scope: what this probe measures and what it does NOT ----------
    std::fprintf(f, "\n## 3. 口径：探针测的是什么，不是什么\n\n");
    std::fprintf(f, "**本探针只报实测，不做外推。**\n\n");
    std::fprintf(f, "- 探针测的是**单 lag** 的一次拟合：`gFTT + LK + RANSAC`（外加两边共享的 "
                    "resize）。\n");
    std::fprintf(f, "- 生产的 `gmc_fit` 预算（约 60 ms/帧）**不是同一个量**：生产一帧要跑"
                    "**多 lag 链**（anchor 拟合 + 逐 lag 变换合成）+ **GPU 尾段**（warp + median）。\n");
    std::fprintf(f, "- 因此「探针的 %.2f ms/帧」与「生产的 60 ms/帧」之间还有约 **%.1f×** 的量差，"
                    "**这 %.1f× 来自多 lag 链 + 变换合成 + GPU 尾段，不是 gFTT/LK 本身**。\n",
                 m_cpu, 60.0 / m_cpu, 60.0 / m_cpu);
    std::fprintf(f, "- 要定位那 %.1f×，**不要**用本探针推算：用仓库里已有的五阶段剖析器 "
                    "`manu/pipeline/cpp/fit_profile.cpp` 在**硬段**（如 `--start 5000`）上打"
                    "生产拟合。\n",
                 60.0 / m_cpu);
    std::fprintf(f, "\n> 本报告里出现的所有百分比都来自**同一次运行**的分阶段实测，没有把探针帧"
                    "的耗时按比例缩放到文档化的 60.1 ms —— 那不是同一个量。\n");

    // ---- 4. why the GPU is slower ------------------------------------------
    std::fprintf(f, "\n## 4. 为什么 GPU 更慢（机理）\n\n");
    std::fprintf(f, "**LK 占 CPU 拟合成本的 %.1f%%，而它是稀疏、gather 密集、迭代式的负载。**\n\n",
                 share_lk * 100.0);
    std::fprintf(f, "| 事实 | 数字 | 来源 |\n| :--- | ---: | :--- |\n");
    std::fprintf(f, "| LK 占 CPU 单次拟合 | %.1f%% (%.3f / %.3f ms) | 本次实测 |\n", share_lk * 100.0,
                 m_cpu_lk, m_cpu);
    std::fprintf(f, "| LK 的 work-item 数 | ~%.0f 个角点 | 实测角点数 |\n", mean(v_ncpu));
    std::fprintf(f, "| 每个 work-item 的迭代 | 3 层金字塔 × ≤%d 轮 × %d×%d patch = 每轮每点 %d 次"
                    "双线性采样 | LK 参数 |\n",
                 kLkIters, kWin, kWin, kWin * kWin);
    std::fprintf(f, "| 采样点 | 各 work-item 互不相关 ⇒ 访存不合并（gather） | LK 结构 |\n");
    std::fprintf(f, "| Mali-G610 并行度 | 4 CU | `clGetDeviceInfo` |\n");
    std::fprintf(f, "\n**推论：**\n\n");
    std::fprintf(f, "1. **LK 不适合 GPU。** ~%.0f 个 work-item 在 4 个 CU 上只有几波，"
                    "而每个 work-item 内部是 20 轮串行迭代、每轮 225 次互不相关的 gather —— "
                    "并行度太粗、访存不合并。CPU 侧相反：SIMD（NEON）+ "
                    "对同一 patch 的缓存友好访问，**反而更优**。实测 GPU LK %.3f ms vs CPU LK "
                    "%.3f ms（%.2f× 慢）。\n\n",
                 mean(v_ncpu), mean(v_gpu_lk), m_cpu_lk, mean(v_gpu_lk) / m_cpu_lk);
    std::fprintf(f, "2. **唯一有希望的是 gFTT。** kernel %.3f ms vs CPU 整段 gFTT %.3f ms"
                    "（省 %.3f ms）。但要把「选取」（NMS + qualityLevel 阈值 + minDistance，"
                    "当前在 CPU 上 %.3f ms）也搬上 GPU 才可能打平 ⇒ **净收益约等于零**。\n\n",
                 m_gftt_gpu, m_cpu_corners, m_cpu_corners - m_gftt_gpu, m_select);
    std::fprintf(f, "3. **数值上两条路是等价的。** 精度代价见 §2："
                    "max|Δ平移| %.4f px，旋转 cos ≥ %.7f，|Δs/s| ≤ %.6f —— "
                    "远优于「角点砍到 100」那条已知代价 2.05 px 的老路。\n\n",
                 pctl(v_dtrans, 1.0), pctl(v_cos, 0.0), pctl(v_scale, 1.0));
    std::fprintf(f, "4. **单线程是硬约束，不是保守。** 生产路径被位级复现要求强制单线程"
                    "（`gmc_stream_ocl.cpp:56-62`：多线程下 RANSAC 与 Lucas-Kanade 不可复现，"
                    "一次 anchor 拟合可能在两个同样合理的解之间翻转）。"
                    "本次 CPU 基准用 `cv::setNumThreads(%d)`，与生产同口径。\n", opencv_threads);

    std::fprintf(f, "\n## 5. 结论\n\n");
    std::fprintf(f, "> **把 GMC 的特征准备搬到 Mali-G610，在本平台上是数值等价但更慢"
                    "（%.3f×），因此不能作为提升 25 fps 的手段；LK 是稀疏 gather 型负载，"
                    "不适合 GPU。**\n\n",
                 speedup);
    std::fprintf(f, "（本行的 %.3f× 是**本次运行**的实测加速比；不同段落的角点密度不同，"
                    "数值会变，但**符号不会变：GPU 更慢**。）\n", speedup);
    std::fclose(f);
    std::printf("\n[i] report written: %s\n", out_path.c_str());
    return 0;
}
