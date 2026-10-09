/* ============================================================================
 * probe_coord.cpp -- which float32 expression does cv::warpPerspective ACTUALLY
 *                    evaluate for the source coordinate?
 *
 * ----------------------------------------------------------------------------
 * WHY
 * ----------------------------------------------------------------------------
 * With float32 coordinates the fused kernel matches the CPU golden reference on
 * all but ~6 pixels in 5.5M, where it is off by exactly 2 grey levels. Switching
 * the kernel to fp64 coordinates changes nothing and slightly WORSENS the exact
 * count -- which is the tell: OpenCV is itself evaluating in float32, so the
 * remaining disagreement is in the *evaluation order*, not the precision. Two
 * IEEE-754 float32 expressions that are mathematically equal can disagree by an
 * ulp, and an ulp at a 1/32 quantisation tie flips the cell -- worth ~8 grey
 * levels on a textured frame.
 *
 * Rather than guess OpenCV's exact expression, enumerate the plausible ones and
 * let cv::warpAffine pick the winner. Pure CPU: no OpenCL, no GPU, runs in a
 * second, so the search is cheap enough to be exhaustive over a small space.
 *
 * Candidates differ in association order and in whether FMA contraction is
 * modelled. OpenCL C gives LLVM freedom to contract a*b+c into fma(), so the
 * host models that too rather than assuming it away.
 *
 * PC-ONLY OFFLINE TOOL -- not part of the RK3588 runtime.
 * ==========================================================================*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

enum Mode {
    M_LTR = 0,       // (m0*x + m1*y) + m2
    M_M1,            // m1*y + (m0*x + m2)
    M_M0,            // m0*x + (m1*y + m2)
    M_FMA_INNER,     // fma(m0, x, fma(m1, y, m2))
    M_FMA_OUTER,     // fma(m1, y, fma(m0, x, m2))
    M_F64,            // double throughout (invert in double)
    M_COUNT
};

static const char* MODE_N[M_COUNT] = {"(m0*x + m1*y) + m2",   "m1*y + (m0*x + m2)",
                                      "m0*x + (m1*y + m2)",   "fma(m0,x, fma(m1,y, m2))",
                                      "fma(m1,y, fma(m0,x, m2))",
                                      "DOUBLE coordinate"};

static inline double coordd(double m0, double m1, double m2, double x, double y) {
    return (m0 * x + m1 * y) + m2;
}

static inline uint8_t model_d(const cv::Mat& im, double sx, double sy) {
    const int x0 = (int)std::floor(sx);
    const int y0 = (int)std::floor(sy);
    const double qx = std::floor((sx - x0) * 32.0 + 0.5) / 32.0;
    const double qy = std::floor((sy - y0) * 32.0 + 0.5) / 32.0;
    if (x0 < 0 || y0 < 0 || x0 + 1 >= im.cols || y0 + 1 >= im.rows) return 0;
    const double p00 = im.at<uint8_t>(y0, x0), p10 = im.at<uint8_t>(y0, x0 + 1);
    const double p01 = im.at<uint8_t>(y0 + 1, x0), p11 = im.at<uint8_t>(y0 + 1, x0 + 1);
    const double t = p00 * (1 - qx) * (1 - qy) + p10 * qx * (1 - qy) +
                     p01 * (1 - qx) * qy + p11 * qx * qy;
    return (uint8_t)std::max(0, std::min(255, (int)std::floor(t + 0.5)));
}

static inline float coord(float m0, float m1, float m2, float x, float y, int mode) {
    switch (mode) {
        case M_LTR:        return (m0 * x + m1 * y) + m2;
        case M_M1:         return m1 * y + (m0 * x + m2);
        case M_M0:         return m0 * x + (m1 * y + m2);
        case M_FMA_INNER:  return std::fma(m0, x, std::fma(m1, y, m2));
        case M_FMA_OUTER:  return std::fma(m1, y, std::fma(m0, x, m2));
        default:           return 0.0f;
    }
}

static inline uint8_t model(const cv::Mat& im, float sx, float sy, int mode) {
    (void)mode;   // single model after the F64 split; kept in the signature for symmetry
    const int x0 = (int)std::floor(sx);
    const int y0 = (int)std::floor(sy);
    const float qx = std::floor((sx - (float)x0) * 32.0f + 0.5f) * (1.0f / 32.0f);
    const float qy = std::floor((sy - (float)y0) * 32.0f + 0.5f) * (1.0f / 32.0f);
    if (x0 < 0 || y0 < 0 || x0 + 1 >= im.cols || y0 + 1 >= im.rows) return 0;
    const float p00 = im.at<uint8_t>(y0, x0), p10 = im.at<uint8_t>(y0, x0 + 1);
    const float p01 = im.at<uint8_t>(y0 + 1, x0), p11 = im.at<uint8_t>(y0 + 1, x0 + 1);
    const float t = p00 * (1 - qx) * (1 - qy) + p10 * qx * (1 - qy) +
                    p01 * (1 - qx) * qy + p11 * qx * qy;
    return (uint8_t)std::max(0, std::min(255, (int)std::floor(t + 0.5f)));
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "";
    if (dir.empty()) { std::printf("usage: probe_coord DIR\n"); return 2; }

    std::vector<cv::Mat> im;
    for (int i = 1; i <= 60 && std::string(argv[1]) != ""; ++i) {
        char b[512];
        std::snprintf(b, sizeof(b), "%s/%06d.jpg", dir.c_str(), i);
        cv::Mat m = cv::imread(b, cv::IMREAD_GRAYSCALE);
        if (!m.empty()) im.push_back(m);
    }
    if (im.empty()) { std::fprintf(stderr, "no frames under %s\n", dir.c_str()); return 2; }
    const int W = im[0].cols, H = im[0].rows;
    std::printf("loaded %zu frames  %dx%d\n\n", im.size(), W, H);

    // GMC-like similarity transforms, same family the pipeline produces.
    cv::RNG rng(2024);
    long exact[M_COUNT] = {0}, sum[M_COUNT] = {0}, n = 0, maxd[M_COUNT] = {0};
    long d2[M_COUNT] = {0};
    long shown = 0;

    for (int trial = 0; trial < 24; ++trial) {
        const double ang = (rng.uniform(0.0, 1.0) - 0.5) * 0.12;
        const double sc = 1.0 + (rng.uniform(0.0, 1.0) - 0.5) * 0.06;
        const double tx = (rng.uniform(0.0, 1.0) - 0.5) * 12.0;
        const double ty = (rng.uniform(0.0, 1.0) - 0.5) * 12.0;
        cv::Mat M(2, 3, CV_32F);
        M.at<float>(0, 0) = (float)(sc * std::cos(ang));
        M.at<float>(0, 1) = (float)(-sc * std::sin(ang));
        M.at<float>(0, 2) = (float)tx;
        M.at<float>(1, 0) = (float)(sc * std::sin(ang));
        M.at<float>(1, 1) = (float)(sc * std::cos(ang));
        M.at<float>(1, 2) = (float)ty;

        // warpAffine inverts M internally; mirror that and keep the float32 type
        // the inversion produces, because that is the matrix OpenCV then uses.
        cv::Mat Mi;
        cv::invertAffineTransform(M, Mi);
        cv::Mat Mi64;
        Mi.convertTo(Mi64, CV_64F);
        cv::Mat dst;
        cv::warpAffine(im[0], dst, M, im[0].size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);

        const float a = Mi.at<float>(0, 0), b = Mi.at<float>(0, 1), c = Mi.at<float>(0, 2);
        const float d = Mi.at<float>(1, 0), e = Mi.at<float>(1, 1), f = Mi.at<float>(1, 2);
        const double A = Mi64.at<double>(0, 0), B = Mi64.at<double>(0, 1), C = Mi64.at<double>(0, 2);
        const double D = Mi64.at<double>(1, 0), E = Mi64.at<double>(1, 1), F = Mi64.at<double>(1, 2);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const float fx = (float)x, fy = (float)y;
                const int ref = dst.at<uint8_t>(y, x);
                // Pixels whose tap base leaves the image are counted for n but
                // excluded from every mode, because the reference used
                // BORDER_CONSTANT here while the fused kernel emulates
                // BORDER_REFLECT through a padded image. Mixing the two border
                // models is what made an earlier run of this probe report
                // Max|Diff| = 145 on a model that is otherwise correct.
                const double csx = coordd(A, B, C, (double)x, (double)y);
                const double csy = coordd(D, E, F, (double)x, (double)y);
                const int bx = (int)std::floor(csx), by = (int)std::floor(csy);
                if (bx < 0 || by < 0 || bx + 1 >= W || by + 1 >= H) continue;
                ++n;
                for (int m = 0; m < M_F64; ++m) {
                    const float sx = coord(a, b, c, fx, fy, m);
                    const float sy = coord(d, e, f, fx, fy, m);
                    const long t = std::abs((long)ref - (long)model(im[0], sx, sy, m));
                    if (t == 0) ++exact[m];
                    if (t >= 2) ++d2[m];
                    if (t > maxd[m]) maxd[m] = t;
                    sum[m] += t;
                }
                {
                    const double sx = coordd(A, B, C, (double)x, (double)y);
                    const double sy = coordd(D, E, F, (double)x, (double)y);
                    const long t = std::abs((long)ref - (long)model_d(im[0], sx, sy));
                    if (t == 0) ++exact[M_F64];
                    if (t >= 2) ++d2[M_F64];
                    if (t > maxd[M_F64]) maxd[M_F64] = t;
                    sum[M_F64] += t;
                    if (t > shown && shown < 12) {
                        shown++;
                        std::printf("  [mismatch] (%d,%d) ref=%d model=%d |d|=%ld  "
                                    "sx=%.9f sy=%.9f fracx=%.9f fracy=%.9f\n",
                                    x, y, ref, (int)model_d(im[0], sx, sy), t, sx, sy,
                                    sx - std::floor(sx), sy - std::floor(sy));
                    }
                }
            }
    }
    std::printf("%-30s %12s %10s %10s %10s\n", "coordinate expression", "exact%", "|d|>=2",
                "Max|D|", "MAE");
    for (int m = 0; m < M_F64; ++m)
        std::printf("%-30s %11.5f%% %10ld %10ld %10.5f\n", MODE_N[m],
                    100.0 * (double)exact[m] / (double)n, d2[m], maxd[m],
                    (double)sum[m] / (double)n);
    std::printf("\n(samples: %ld)\n", n);
    return 0;
}