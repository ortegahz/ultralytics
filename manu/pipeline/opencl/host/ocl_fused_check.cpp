// ============================================================================
// ocl_fused_check.cpp -- accuracy reconciliation for warp_median_fused.cl.
//
// Runs the CPU reference and the OpenCL fused kernel on the SAME frames with the
// SAME affine matrices and diffs the result channel by channel. This isolates
// kernel numerics from the feature pipeline: no Shi-Tomasi, no LK, no RANSAC,
// so a failure here is unambiguously the kernel's, not the estimator's.
//
// CPU reference is byte-for-byte the operations gmc_stream.cpp performs:
//     warped[k] = cv::warpAffine(hist_k, M_k, (W,H), INTER_LINEAR, BORDER_REFLECT)
//     Ch0       = I_t
//     Ch1       = |I_t - warped[0]|
//     Ch2       = max(0, I_t - median(warped[0..20]))
// Note warpAffine receives the FORWARD transform and inverts it internally, so
// the kernel is handed invertAffineTransform(M) -- computed the same way.
//
// Gates:  Max|Diff| <= 1 per channel,  MAE < 0.05 over the whole frame.
//
// Build:  see manu/pipeline/opencl/CMakeLists.txt
//
// PC verification host. See RK3588 notes in ocl_host.h and warp_median_fused.cl.
// ============================================================================

#include "ocl_host.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

static const int WINDOW = 21;

static bool has_ext(const std::string& p, const char* e) {
    const size_t n = p.size();
    return n >= std::strlen(e) && std::strcmp(p.c_str() + (n - std::strlen(e)), e) == 0;
}

static std::vector<std::string> list_frames(const std::string& dir) {
    std::vector<std::string> out;
    for (const auto& d : std::filesystem::directory_iterator(dir))
        if (d.is_regular_file() && (has_ext(d.path().string(), ".jpg") || has_ext(d.path().string(), ".png") ||
                               has_ext(d.path().string(), ".bmp")))
            out.push_back(d.path().string());
    std::sort(out.begin(), out.end());   // zero-padded names -> lexicographic == temporal
    return out;
}

// ---------------------------------------------------------------------------
// GPU-side state: 22 padded single-channel images, one per history frame plus
// the current one. Allocated once per resolution, then reused via clEnqueueWriteImage.
// ---------------------------------------------------------------------------
struct FusedRunner {
    ocl::Context ctx;
    ocl::Program prog;
    ocl::Kernel k_;
    ocl::Sampler samp;
    ocl::Mem mats_buf, out_buf;
    std::vector<ocl::Mem> imgs;                 // [0..20] history, [21] current
    int W = 0, H = 0, pad = -1;
    std::vector<float> mats;                     // 21 * 6
    bool hw_linear = true;
    cl_image_format fmt_{};
    int chan_ = 1;                               // 1 = CL_R8/CL_R, 4 = CL_RGBA

    FusedRunner(cl_device_id d, const std::string& kernel_src, const std::string& inc_src,
                bool hw, bool fp64 = false)
        : ctx(d), samp(ocl::make_sampler(ctx.get(), hw)), mats(WINDOW * 6), hw_linear(hw) {
        prog = ocl::build_program(ctx.get(), d, ocl::splice_sortnet(kernel_src, inc_src), "fused",
                                  hw, fp64);
        ocl::pick_gray_format(ctx.get(), &fmt_, &chan_);
        cl_int e;
        k_.reset(clCreateKernel(prog.get(), "warp_median_fused", &e));
        OCL_CHECK(e, "clCreateKernel(warp_median_fused)");
    }

    void resize(int w, int h, int p) {
        if (w == W && h == H && p == pad) return;
        W = w; H = h; pad = p;
        imgs.clear();
        for (int i = 0; i <= WINDOW; ++i)
            imgs.emplace_back(ocl::make_image2d(ctx.get(), fmt_,
                                                (size_t)W + 2 * (size_t)pad,
                                                (size_t)H + 2 * (size_t)pad));
        cl_int e;
        mats_buf.reset(clCreateBuffer(ctx.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      mats.size() * sizeof(float), mats.data(), &e));
        OCL_CHECK(e, "clCreateBuffer(mats)");
        out_buf.reset(clCreateBuffer(ctx.get(), CL_MEM_WRITE_ONLY,
                                     (size_t)3 * W * H, nullptr, &e));
        OCL_CHECK(e, "clCreateBuffer(out)");
    }

    // `padded` is always an 8-bit single-channel BORDER_REFLECT-padded frame.
    // On a device that only offers CL_RGBA it is widened here; read_imagef().x
    // is the red channel either way, so the kernel is identical in both cases.
    void upload(int slot, const cv::Mat& padded_in) {
        cv::Mat wide;
        const cv::Mat* src = &padded_in;
        if (chan_ == 4) {
            cv::cvtColor(padded_in, wide, cv::COLOR_GRAY2RGBA);
            src = &wide;
        }
        const size_t o[3] = {0, 0, 0};
        const size_t r[3] = {(size_t)src->cols, (size_t)src->rows, 1};
        OCL_CHECK(clEnqueueWriteImage(ctx.queue(), imgs[(size_t)slot].get(), CL_TRUE, o, r, 0, 0,
                                      src->data, 0, nullptr, nullptr),
                  "clEnqueueWriteImage");
    }

    void upload_mats(const std::vector<float>& m) {
        OCL_CHECK(clEnqueueWriteBuffer(ctx.queue(), mats_buf.get(), CL_TRUE, 0,
                                       m.size() * sizeof(float), m.data(), 0, nullptr, nullptr),
                  "clEnqueueWriteBuffer(mats)");
    }

    // @pads: padded frames for lags 2,4,...,42 followed by the current frame
    void run(std::vector<cv::Mat>& out) {
        cl_int e;
        for (int k = 0; k < WINDOW; ++k) {
            cl_mem im = imgs[(size_t)k].get();   // copy: get() returns by value
            e = clSetKernelArg(k_.get(), (cl_uint)k, sizeof(cl_mem), &im);
            OCL_CHECK(e, "clSetKernelArg(hist)");
        }
        cl_mem cur = imgs[(size_t)WINDOW].get();
        e = clSetKernelArg(k_.get(), WINDOW, sizeof(cl_mem), &cur);
        OCL_CHECK(e, "clSetKernelArg(cur)");
        cl_mem mb = mats_buf.get();
        e = clSetKernelArg(k_.get(), WINDOW + 1, sizeof(cl_mem), &mb);
        OCL_CHECK(e, "clSetKernelArg(mats)");
        cl_mem ob = out_buf.get();
        e = clSetKernelArg(k_.get(), WINDOW + 2, sizeof(cl_mem), &ob);
        OCL_CHECK(e, "clSetKernelArg(out)");
        e = clSetKernelArg(k_.get(), WINDOW + 3, sizeof(int), &W);
        OCL_CHECK(e, "clSetKernelArg(W)");
        e = clSetKernelArg(k_.get(), WINDOW + 4, sizeof(int), &H);
        OCL_CHECK(e, "clSetKernelArg(H)");
        e = clSetKernelArg(k_.get(), WINDOW + 5, sizeof(int), &pad);
        OCL_CHECK(e, "clSetKernelArg(pad)");
        cl_sampler sm = samp.get();
        e = clSetKernelArg(k_.get(), WINDOW + 6, sizeof(cl_sampler), &sm);
        OCL_CHECK(e, "clSetKernelArg(samp)");

        const size_t g[2] = {(size_t)W, (size_t)H};
        e = clEnqueueNDRangeKernel(ctx.queue(), k_.get(), 2, nullptr, g, nullptr, 0, nullptr, nullptr);
        OCL_CHECK(e, "clEnqueueNDRangeKernel");

        out.resize(3);
        const size_t plane = (size_t)W * (size_t)H;
        for (int c = 0; c < 3; ++c) {
            out[(size_t)c].create(H, W, CV_8UC1);
            OCL_CHECK(clEnqueueReadBuffer(ctx.queue(), out_buf.get(), CL_TRUE, (size_t)c * plane,
                                          plane, out[(size_t)c].data, 0, nullptr, nullptr),
                      "clEnqueueReadBuffer");
        }
    }

};

// ---------------------------------------------------------------------------
static cv::Mat median_plane(const std::vector<cv::Mat>& s) {
    const int h = s[0].rows, w = s[0].cols;
    cv::Mat out(h, w, CV_8UC1);
    std::vector<uint8_t> col(s.size());
    for (int x = 0; x < w; ++x)
        for (int y = 0; y < h; ++y) {
            for (size_t i = 0; i < s.size(); ++i) col[i] = s[i].at<uint8_t>(y, x);
            std::nth_element(col.begin(), col.begin() + (int)(col.size() / 2), col.end());
            out.at<uint8_t>(y, x) = col[col.size() / 2];
        }
    return out;
}

struct Diff {
    long maxdiff = 0;
    double mae = 0;
    int bx = -1, by = -1;
    long hist[8] = {0};      // |diff| == 0,1,2,3,4,5,6, >=7
    // Positions of every |diff| >= 2 pixel. Reported because the cause differs by
    // location: pixels hugging the frame edge implicate the border/pad path,
    // pixels scattered through the interior implicate float32 coordinate
    // precision against OpenCV's own float32 arithmetic. The Max|Diff| number
    // alone cannot distinguish the two.
    std::vector<std::pair<int, int>> hot;
};

// A |diff| histogram is not decoration. "Max|Diff| = 2, MAE = 0.003" and
// "Max|Diff| = 2, MAE = 1.4" have the same headline and completely different
// causes -- the first is a handful of quantisation-boundary ties, the second is
// a systematic rounding mismatch. The gate is per-pixel AND per-image, so both
// numbers have to be visible, not just the max.
static Diff compare(const cv::Mat& a, const cv::Mat& b) {
    Diff d;
    long n = 0, sum = 0;
    for (int y = 0; y < a.rows; ++y)
        for (int x = 0; x < a.cols; ++x) {
            const long t = std::abs((long)a.at<uint8_t>(y, x) - (long)b.at<uint8_t>(y, x));
            sum += t; ++n;
            d.hist[t < 7 ? t : 7]++;
            if (t > d.maxdiff) { d.maxdiff = t; d.bx = x; d.by = y; }
            if (t >= 2) d.hot.emplace_back(x, y);
        }
    d.mae = (double)sum / (double)n;
    return d;
}

/* BORDER_REFLECT is not decoration here -- it IS OpenCV's border rule, verified
 * by tools/probe_border.cpp (model B: reflect each tap independently, 0/96000
 * wrong). The kernel samples this padded image at (sx + pad), so the padding has
 * to be wide enough for the largest tap excursion, which a rotating warp pushes
 * to ~50 px outside the frame. See required_pad(). */
static cv::Mat pad_border(const cv::Mat& m, int p) {
    if (p <= 0) return m;
    cv::Mat o;
    cv::copyMakeBorder(m, o, p, p, p, p, cv::BORDER_REFLECT);
    return o;
}

// The exact bilinear footprint of an affine map over a rectangle is reached at
// the corners, so the pad needed for exact CLAMP_TO_EDGE==BORDER_REFLECT
// behaviour is computable from 4 evaluations per matrix instead of a full sweep.
static int required_pad(const std::vector<cv::Mat>& inv, int W, int H) {
    // p starts at 2: even an identity warp needs two rows/columns of border so
    // the bilinear tap support always lands inside the padded image and
    // CLK_ADDRESS_CLAMP_TO_EDGE never silently alters a value.
    int p = 2;
    for (const cv::Mat& m : inv) {
        if (m.empty() || m.rows != 2 || m.cols != 3 || m.type() != CV_64FC1) {
            std::fprintf(stderr, "[FATAL] inverse matrix is not a 2x3 CV_64F "
                                 "(rows=%d cols=%d type=%d)\n",
                         m.rows, m.cols, m.type());
            std::exit(2);
        }
        const double a = m.at<double>(0, 0), b = m.at<double>(0, 1), c = m.at<double>(0, 2);
        const double d = m.at<double>(1, 0), e = m.at<double>(1, 1), f = m.at<double>(1, 2);
        if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) ||
            !std::isfinite(d) || !std::isfinite(e) || !std::isfinite(f)) {
            std::fprintf(stderr, "[FATAL] non-finite inverse matrix\n");
            std::exit(2);
        }
        for (int corner = 0; corner < 4; ++corner) {
            const double X = (corner & 1) ? W : 0.0;
            const double Y = (corner & 2) ? H : 0.0;
            const double sx = a * X + b * Y + c;
            const double sy = d * X + e * Y + f;
            // Computed in double and only then narrowed: an (int) cast on an
            // unbounded double silently produced INT_MIN+510 here and asked the
            // driver for a 4294966788-pixel image, which came back as a FORMAT
            // error and looked like a channel-order problem.
            const double need_x_lo = -std::floor(sx);
            const double need_x_hi = std::floor(sx) + 2.0 - W;
            const double need_y_lo = -std::floor(sy);
            const double need_y_hi = std::floor(sy) + 2.0 - H;
            p = std::max(p, (int)std::ceil(std::max(need_x_lo, need_x_hi)));
            p = std::max(p, (int)std::ceil(std::max(need_y_lo, need_y_hi)));
        }
    }
    return p;
}

static void usage() {
    std::printf(
        "ocl_fused_check --raw-root DIR --seq NAME [--frames N] [--hw-linear 0|1]\n"
        "                 [--disp PX] [--seed N] [--dump-worst DIR]\n\n"
        "  --raw-root DIR   root containing train/<SEQ>/000001.jpg (anti-uav)\n"
        "  --seq NAME       sequence name, repeatable\n"
        "  --frames N       frames to test per sequence (default 8, must exceed max lag 42)\n"
        "  --hw-linear      1 = CLK_FILTER_LINEAR sampler, 0 = manual 2x2 blend\n"
        "  --fp64-coord     1 = DIAGNOSTIC: form the source coordinate in double\n"
        "                    (PoCL only; never on Mali, which has no FP64)\n"
        "  --disp PX        max |displacement| of the synthetic GMC matrices (default 6)\n"
        "  --seed N         RNG seed for the synthetic transforms\n");
}

int main(int argc, char** argv) {
    std::string raw_root, dump_worst;
    std::vector<std::string> seqs;
    int frames = 8, seed = 7, disp = 6;
    bool hw = true, fp64 = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if (a == "--raw-root") raw_root = next();
        else if (a == "--seq") seqs.push_back(next());
        else if (a == "--frames") frames = std::stoi(next());
        else if (a == "--hw-linear") hw = std::stoi(next()) != 0;
        else if (a == "--fp64-coord") fp64 = std::stoi(next()) != 0;
        else if (a == "--disp") disp = std::stoi(next());
        else if (a == "--seed") seed = std::stoi(next());
        else if (a == "--dump-worst") dump_worst = next();
        else { usage(); return 2; }
    }
    if (raw_root.empty() || seqs.empty()) { usage(); return 2; }
    if (frames <= 42 + 1) {
        std::fprintf(stderr, "[FATAL] --frames must exceed max lag 42 so every history lag is real\n");
        return 2;
    }

    cl_platform_id plat = nullptr;
    cl_device_id dev = ocl::pick_device(&plat);
    ocl::describe(dev, plat);
    std::printf("[OCL] sampling path : %s\n",
                hw ? "CLK_FILTER_LINEAR (hardware 2x2)"
                   : "CLK_FILTER_NEAREST + manual 2x2 blend");
    std::printf("[OCL] opengl/etc    : (not interop; images are uploaded directly)\n\n");

    const std::string ksrc = ocl::read_file("kernels/warp_median_fused.cl");
    const std::string isrc = ocl::read_file("kernels/sortnet_generated.inc");
    FusedRunner run(dev, ksrc, isrc, hw, fp64);

    cv::setNumThreads(1);   // the CPU reference must not vary run to run

    long gmax[3] = {0, 0, 0};
    double gsum[3] = {0, 0, 0}, gmae[3] = {0, 0, 0};
    long ghist[3][8] = {{0}};
    struct Hot { int ch, t, x, y, w, h; };
    std::vector<Hot> hot;
    long gn = 0, gnpx = 0;
    int worst_frame = -1; long worst = 0;
    int max_pad_seen = 0;
    bool gate_ok = true;

    for (const std::string& seq : seqs) {
        const std::string dir = raw_root + "/train/" + seq;
        std::vector<std::string> files;
        try {
            files = list_frames(dir);
        } catch (...) {}
        if (files.empty()) {
            std::fprintf(stderr, "[FATAL] no frames under %s\n", dir.c_str());
            return 2;
        }
        if ((int)files.size() > frames) files.resize((size_t)frames);

        std::vector<cv::Mat> imgs;
        for (const auto& f : files) {
            cv::Mat m = cv::imread(f, cv::IMREAD_GRAYSCALE);
            if (m.empty()) { std::fprintf(stderr, "[FATAL] read %s\n", f.c_str()); return 2; }
            imgs.push_back(m);
        }
        const int W = imgs[0].cols, H = imgs[0].rows;

        for (size_t t = (size_t)43; t < imgs.size(); ++t) {
            // ---- 21 lagged frames + 21 GMC-like similarity transforms ----
            cv::RNG rng((unsigned)(seed * 1000003u + (unsigned)t));
            std::vector<cv::Mat> fwd, inv;
            for (int k = 0; k < WINDOW; ++k) {
                const double ang = (rng.uniform(0.0, 1.0) - 0.5) * (disp * 0.02);
                const double sc = 1.0 + (rng.uniform(0.0, 1.0) - 0.5) * (disp * 0.01);
                const double tx = (rng.uniform(0.0, 1.0) - 0.5) * 2.0 * disp;
                const double ty = (rng.uniform(0.0, 1.0) - 0.5) * 2.0 * disp;
                cv::Mat M(2, 3, CV_32F);
                M.at<float>(0, 0) = (float)(sc * std::cos(ang));
                M.at<float>(0, 1) = (float)(-sc * std::sin(ang));
                M.at<float>(0, 2) = (float)tx;
                M.at<float>(1, 0) = (float)(sc * std::sin(ang));
                M.at<float>(1, 1) = (float)(sc * std::cos(ang));
                M.at<float>(1, 2) = (float)ty;
                fwd.push_back(M);
                // warpAffine inverts M internally; mirror that exactly.
                //
                // OpenCV 4.10's invertAffineTransform returns the OUTPUT in the
                // INPUT's element type, so a CV_32F input yields a CV_32F result
                // (not CV_64F as the docs suggest). Normalise to CV_64F here:
                // reading the 32-bit result through at<double> strides 8 bytes
                // over a 4-byte layout and silently yields denormals, which then
                // detonate further downstream as a 4-billion-pixel image request.
                cv::Mat Mi, Mi64;
                cv::invertAffineTransform(M, Mi);
                Mi.convertTo(Mi64, CV_64F);
                inv.push_back(Mi64);
            }

            // ---- CPU reference ----
            std::vector<cv::Mat> warped(WINDOW);
            for (int k = 0; k < WINDOW; ++k) {
                const size_t li = (size_t)t - (size_t)(2 * k + 2);
                const cv::Mat& src = imgs[li];          // t > 42 guarantees li >= 1
                cv::warpAffine(src, warped[(size_t)k], fwd[(size_t)k], cv::Size(W, H),
                               cv::INTER_LINEAR, cv::BORDER_REFLECT);
            }
            cv::Mat ch1;
            cv::absdiff(imgs[t], warped[0], ch1);
            const cv::Mat bg = median_plane(warped);
            cv::Mat ch2(H, W, CV_8UC1);   // rows=H, cols=W -- (W, H) is an out-of-range
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    const int v = (int)imgs[t].at<uint8_t>(y, x) - (int)bg.at<uint8_t>(y, x);
                    ch2.at<uint8_t>(y, x) = (uint8_t)(v > 0 ? v : 0);
                }
            std::vector<cv::Mat> ref{imgs[t], ch1, ch2};

            // ---- GPU ----
            if (t == (size_t)43) {
                std::printf("[BUILD] inv[0] = [%g %g %g / %g %g %g]\n",
                            inv[0].at<double>(0, 0), inv[0].at<double>(0, 1), inv[0].at<double>(0, 2),
                            inv[0].at<double>(1, 0), inv[0].at<double>(1, 1), inv[0].at<double>(1, 2));
                std::printf("[BUILD] fwd[0] = [%g %g %g / %g %g %g]\n",
                            fwd[0].at<float>(0, 0), fwd[0].at<float>(0, 1), fwd[0].at<float>(0, 2),
                            fwd[0].at<float>(1, 0), fwd[0].at<float>(1, 1), fwd[0].at<float>(1, 2));
            }
            const int pad = required_pad(inv, W, H);
            max_pad_seen = std::max(max_pad_seen, pad);

            run.resize(W, H, pad);
            for (int k = 0; k < WINDOW; ++k) {
                const size_t li = (size_t)t - (size_t)(2 * k + 2);
                run.upload(k, pad_border(imgs[li], pad));
            }
            run.upload(WINDOW, pad_border(imgs[t], pad));
            std::vector<float> mf((size_t)WINDOW * 6);
            for (int k = 0; k < WINDOW; ++k)
                for (int j = 0; j < 6; ++j)
                    mf[(size_t)k * 6 + (size_t)j] =
                        (float)inv[(size_t)k].at<double>(j / 3, j % 3);
            run.upload_mats(mf);

            std::vector<cv::Mat> got;
            run.run(got);

            // ---- diff ----
            Diff per[3];
            for (int c = 0; c < 3; ++c) {
                per[c] = compare(ref[(size_t)c], got[(size_t)c]);
                gmax[c] = std::max(gmax[c], per[c].maxdiff);
                gsum[c] += per[c].mae;                       // per-frame MAE
                gmae[c] = std::max(gmae[c], per[c].mae);
                ghist[c][0] += per[c].hist[0];
                for (int i = 1; i < 8; ++i) ghist[c][i] += per[c].hist[i];
                gnpx += (long)W * (long)H;
                if (per[c].maxdiff > 1) gate_ok = false;
                for (auto& q : per[c].hot)
                    hot.push_back({(int)c, (int)t, q.first, q.second, W, H});
                if (per[c].maxdiff > worst) { worst = per[c].maxdiff; worst_frame = (int)t; }
            }
            ++gn;
            if (gn <= 6 || per[2].maxdiff > 1)
                std::printf("[SEQ %-22s t=%-4zu pad=%-3d Ch0 d=%ld m=%.4f | "
                            "Ch1 d=%ld m=%.4f | Ch2 d=%ld m=%.4f\n",
                            seq.c_str(), t, pad, per[0].maxdiff, per[0].mae, per[1].maxdiff,
                            per[1].mae, per[2].maxdiff, per[2].mae);
        }
    }

    if (gn == 0) { std::fprintf(stderr, "[FATAL] no frames compared\n"); return 2; }
    std::printf("\n================ ACCURACY GATE ================\n");
    std::printf("frames compared : %ld   pixels/frame total : %ld   sampling : %s   max pad : %d\n",
                gn, gnpx, hw ? "HW_LINEAR" : "MANUAL_2x2", max_pad_seen);
    std::printf("channel |   Max|Diff| |     MAE |  worst-frame MAE\n");
    for (int c = 0; c < 3; ++c)
        std::printf("  Ch%d   | %11ld | %9.5f | %15.5f\n", c, gmax[c],
                    gsum[c] / (double)gn, gmae[c]);
    std::printf("\n|diff| distribution over all compared pixels:\n");
    std::printf("channel |      0 |      1 |      2 |      3 |     4-6 |    >=7\n");
    for (int c = 0; c < 3; ++c)
        std::printf("  Ch%d   | %6ld | %6ld | %6ld | %6ld | %7ld | %6ld\n", c, ghist[c][0],
                    ghist[c][1], ghist[c][2], ghist[c][3], ghist[c][4] + ghist[c][5] +
                    ghist[c][6], ghist[c][7]);
    std::printf("\npixels with |diff| >= 2  (channel, frame t, x, y, dist-to-edge):\n");
    for (size_t i = 0; i < hot.size() && i < 40; ++i) {
        const Hot& q = hot[i];
        std::printf("   Ch%d t=%-4d (%4d,%4d)  d_edge=%d\n", q.ch, q.t, q.x, q.y,
                    std::min(std::min(q.x, q.y), std::min(q.w - 1 - q.x, q.h - 1 - q.y)));
    }
    if (hot.size() > 40) std::printf("   ... %zu more\n", hot.size() - 40);
    std::printf("\nthreshold       | %11d | %9.5f\n", 1, 0.05);
    std::printf("verdict         : %s  (worst single-pixel |diff| = %ld at frame t=%d)\n",
                gate_ok ? "PASS" : "FAIL", worst, worst_frame);
    std::printf("==================================================\n");
    return gate_ok ? 0 : 1;
}