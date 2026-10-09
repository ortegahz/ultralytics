/* ============================================================================
 * probe_interp.cpp  --  EMPIRICALLY pin down cv::warpAffine's exact CV_8UC1
 *                       INTER_LINEAR arithmetic.
 *
 * ----------------------------------------------------------------------------
 * WHY THIS EXISTS
 * ----------------------------------------------------------------------------
 * The acceptance gate is MAE < 0.05 over the whole output image. That is a much
 * tighter demand than "Max|Diff| <= 1", and it targets SYSTEMATIC bias, not noise:
 *
 *     Ch2 = max(0, I_t - median{W(I_{t-2k})})
 *
 * If the GPU rounds differently from OpenCV, every warped sample carries a
 * rounding error. The median of 21 such errors does NOT average them away --
 * truncation is always downward, so the errors are sign-correlated and the
 * median inherits roughly the same offset. A systematic -0.5 therefore survives
 * into Ch2 as an MAE of about 0.5: ten times over gate.
 *
 * So the kernel must replicate OpenCV's arithmetic, not approximate it. Which
 * arithmetic it is, is a fact about OpenCV internals. The previous porting stage
 * produced 11 silent defects from guessing at exactly this kind of fact, so:
 * measure it.
 *
 * ----------------------------------------------------------------------------
 * SWEPT
 * ----------------------------------------------------------------------------
 *   position : exact | floor(sx)+trunc(frac*32)/32 | floor(sx)+round(frac*32)/32
 *   weights  : float | 14-bit fixed | 16-bit fixed
 *   output   : round-half-up | truncate
 *   origin   : (x, y) | (x+0.5, y+0.5)      -- the -0.5 half-pixel question
 * = 36 models. For each: exact-match fraction, Max|Diff|, MAE over a random-noise
 * image (worst case for discrimination) warped by a systematic sub-pixel sweep.
 *
 * PC-ONLY OFFLINE TOOL -- not part of the RK3588 runtime.
 * ==========================================================================*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

enum PosQ { PQ_NONE = 0, PQ_TRUNC32 = 1, PQ_ROUND32 = 2, PQ_COUNT = 3 };
enum WM { W_FLOAT = 0, W_I14 = 1, W_I16 = 2, W_COUNT = 3 };
enum OutR { OR_ROUND = 0, OR_TRUNC = 1, OR_COUNT = 2 };

static inline uint8_t model(const cv::Mat& im, double sx, double sy, int posq, int wm, int orr) {
    const int x0 = (int)std::floor(sx);
    const int y0 = (int)std::floor(sy);
    double fx = sx - x0, fy = sy - y0;
    if (fx < 0) fx = 0;
    if (fy < 0) fy = 0;
    if (fx > 1) fx = 1;
    if (fy > 1) fy = 1;

    double qx = fx, qy = fy;
    if (posq == PQ_TRUNC32) {
        qx = std::min(1.0, std::floor(fx * 32.0) / 32.0);
        qy = std::min(1.0, std::floor(fy * 32.0) / 32.0);
    } else if (posq == PQ_ROUND32) {
        qx = std::min(1.0, std::floor(fx * 32.0 + 0.5) / 32.0);
        qy = std::min(1.0, std::floor(fy * 32.0 + 0.5) / 32.0);
    }

    const int p00 = im.at<uint8_t>(y0, x0), p10 = im.at<uint8_t>(y0, x0 + 1);
    const int p01 = im.at<uint8_t>(y0 + 1, x0), p11 = im.at<uint8_t>(y0 + 1, x0 + 1);

    int v;
    if (wm == W_FLOAT) {
        const double t = p00 * (1 - qx) * (1 - qy) + p10 * qx * (1 - qy) +
                         p01 * (1 - qx) * qy + p11 * qx * qy;
        v = (orr == OR_ROUND) ? (int)std::floor(t + 0.5) : (int)std::floor(t);
    } else {
        const int B = (wm == W_I14) ? 14 : 16;
        const int ONE = 1 << B;
        const int wx0 = (int)std::floor(ONE * (1.0 - qx)), wx1 = ONE - wx0;
        const int wy0 = (int)std::floor(ONE * (1.0 - qy)), wy1 = ONE - wy0;
        const long s = (long)wx0 * p00 + (long)wx1 * p10 + (long)wy0 * p01 + (long)wy1 * p11;
        v = (orr == OR_ROUND) ? (int)((s + (ONE >> 1)) >> B) : (int)(s >> B);
    }
    return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

struct Acc {
    long n = 0, exact = 0, maxdiff = 0;
    double sumabs = 0.0;
    void add(int ref, int got) {
        const long d = std::abs(ref - got);
        ++n;
        if (d == 0) ++exact;
        if (d > maxdiff) maxdiff = d;
        sumabs += (double)d;
    }
};

int main() {
    const int W = 96, H = 96;
    cv::Mat img(H, W, CV_8UC1);
    cv::RNG rng(12345);
    rng.fill(img, cv::RNG::UNIFORM, 0, 256);

    std::vector<cv::Mat> Ms;
    std::vector<const char*> Mn;
    for (int k = -96; k <= 96; ++k) {   // 1/64 px steps across +/- 1.5 px
        const double t = k / 64.0;
        Ms.push_back((cv::Mat_<double>(2, 3) << 1.0, 0.0, t, 0.0, 1.0, t * 0.6));
        Mn.push_back("trans");
    }
    for (int k = 0; k < 40; ++k) {     // random rotation/scale/translation
        const double a = (rng.uniform(0.0, 1.0) - 0.5) * 0.3;
        const double s = 1.0 + (rng.uniform(0.0, 1.0) - 0.5) * 0.2;
        const double tx = (rng.uniform(0.0, 1.0) - 0.5) * 4.0;
        const double ty = (rng.uniform(0.0, 1.0) - 0.5) * 4.0;
        Ms.push_back((cv::Mat_<double>(2, 3) << s * std::cos(a), -s * std::sin(a), tx,
                      s * std::sin(a), s * std::cos(a), ty));
        Mn.push_back("affine");
    }

    static Acc acc[2][PQ_COUNT][W_COUNT][OR_COUNT];
    /* Coordinate-precision dimension.
     *
     * The oracle (probe_interp2) already proved the STRUCTURE: w = round(frac*32)/32,
     * round half-up, no half-pixel offset. The residual mismatches were ~255/32, i.e.
     * exactly one quantisation cell against a maximum noise gradient -- the signature
     * of a rounding-boundary disagreement, not of a wrong weight table. The one
     * remaining candidate is the precision OpenCV uses when it forms src = M^-1 * [x,y,1].
     * A GPU computes this in float32 natively, so if double is wrong here the kernel
     * must use float32 to match -- and it costs nothing to find out which. */
    static Acc accf[2][PQ_COUNT][W_COUNT][OR_COUNT];

    for (size_t m = 0; m < Ms.size(); ++m) {
        // warpAffine(src, dst, M, ...) yields dst(x,y) <- src(M^-1 * [x,y,1]);
        // OpenCV inverts M internally, so we must invert here too to know which
        // source pixel it actually sampled.
        cv::Mat Mi;
        cv::invertAffineTransform(Ms[m], Mi);
        const double a0 = Mi.at<double>(0, 0), a1 = Mi.at<double>(0, 1), a2 = Mi.at<double>(0, 2);
        const double b0 = Mi.at<double>(1, 0), b1 = Mi.at<double>(1, 1), b2 = Mi.at<double>(1, 2);

        cv::Mat dst;
        cv::warpAffine(img, dst, Ms[m], img.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);

        for (int y = 2; y < H - 3; ++y) {
            for (int x = 2; x < W - 3; ++x) {
                for (int hp = 0; hp < 2; ++hp) {
                    const double dx = x + (hp ? 0.5 : 0.0);
                    const double dy = y + (hp ? 0.5 : 0.0);
                    const double sxd = a0 * dx + a1 * dy + a2;
                    const double syd = b0 * dx + b1 * dy + b2;
                    const float sxf = (float)sxd, syf = (float)syd;
                    const int x0 = (int)std::floor(sxd), y0 = (int)std::floor(syd);
                    if (x0 < 1 || y0 < 1 || x0 + 2 >= W || y0 + 2 >= H) continue;
                    const int ref = dst.at<uint8_t>(y, x);
                    for (int pq = 0; pq < PQ_COUNT; ++pq)
                        for (int wm = 0; wm < W_COUNT; ++wm)
                            for (int orr = 0; orr < OR_COUNT; ++orr) {
                                acc[hp][pq][wm][orr].add(ref, model(img, sxd, syd, pq, wm, orr));
                                accf[hp][pq][wm][orr].add(ref, model(img, sxf, syf, pq, wm, orr));
                            }
                }
            }
        }
    }

    static const char* pq_n[] = {"pos=exact  ", "pos=trunc32", "pos=round32"};
    static const char* w_n[] = {"w=float", "w=int14 ", "w=int16 "};
    static const char* o_n[] = {"out=round", "out=trunc"};

    for (int hp = 0; hp < 2; ++hp) {
        std::printf("\n=== origin = %s ===\n", hp ? "(x+0.5, y+0.5)  [half-pixel]" : "(x, y)      [no offset]");
        std::printf("%-11s %-9s %-10s | %18s | %18s\n", "position", "weights", "output",
                    "exact%  (f64 coord)", "exact%  (f32 coord)");
        for (int pq = 0; pq < PQ_COUNT; ++pq)
            for (int wm = 0; wm < W_COUNT; ++wm)
                for (int orr = 0; orr < OR_COUNT; ++orr) {
                    const Acc& A = acc[hp][pq][wm][orr];
                    const Acc& B = accf[hp][pq][wm][orr];
                    std::printf("%-11s %-9s %-10s | %8.4f%% d=%-6ld | %8.4f%% d=%-6ld\n",
                                pq_n[pq], w_n[wm], o_n[orr],
                                100.0 * (double)A.exact / (double)A.n, A.maxdiff,
                                100.0 * (double)B.exact / (double)B.n, B.maxdiff);
                }
    }
    return 0;
}