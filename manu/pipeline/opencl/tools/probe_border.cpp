/* ============================================================================
 * probe_border.cpp -- which two source texels does cv::warpAffine actually
 *                     read for an out-of-frame bilinear tap?
 *
 * ----------------------------------------------------------------------------
 * WHY
 * ----------------------------------------------------------------------------
 * Two candidate border models both look plausible and only one can be right:
 *
 *   A "reflect the base"    base' = reflect(base); taps = (src[base'], src[base'+1])
 *     -- what cv::borderInterpolate() would suggest
 *   B "reflect both taps"   taps = (src[reflect(base)], src[reflect(base+1)])
 *     -- what cv::copyMakeBorder(..., BORDER_REFLECT) gives you for free
 *
 * They differ exactly when the tap base is outside the frame, and the fused
 * kernel measured Ch1 Max|Diff| = 5 with model B versus 239 with model A on
 * 640-wide sequences. One of them is simply wrong; the rest of the border
 * correctness (and whether the host needs a big pad or a fixed pad of 1)
 * follows from the answer.
 *
 * METHOD
 * ------
 * Put a pure x-translation on a random-noise image so a handful of leading
 * columns sample off the left edge, then read dst(x) for those x directly. The
 * four tap values are known exactly, so each candidate predicts one integer and
 * the winner is the one that reproduces OpenCV.
 *
 * PC-ONLY OFFLINE TOOL -- not part of the RK3588 runtime.
 * ==========================================================================*/

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdio>

static int reflect(int p, int n) {
    if (n <= 1) return 0;
    const int period = 2 * n;
    int r = p % period;
    if (r < 0) r += period;
    return (r < n) ? r : (period - 1 - r);
}

// qx is the quantised sub-texel position in [0,1] (multiples of 1/32).
static int model(const unsigned char* img, int W, int H, int bx, int by, float qx, float qy,
                 int variant) {
    auto at = [&](int x, int y) { return (float)img[(size_t)y * W + x]; };
    int x0 = bx, y0 = by;
    if (variant == 0) {           // A: reflect the base, then base+1
        x0 = reflect(bx, W);
        y0 = reflect(by, H);
    } else if (variant == 1) {    // B: reflect each tap independently
        x0 = reflect(bx, W);
        y0 = reflect(by, H);
    }
    // variant 0 and 1 differ only in how the "+1" tap is resolved
    const int xi0 = (variant == 0) ? x0 : reflect(bx, W);
    const int xi1 = (variant == 0) ? x0 + 1 : reflect(bx + 1, W);
    const int yi0 = (variant == 0) ? y0 : reflect(by, H);
    const int yi1 = (variant == 0) ? y0 + 1 : reflect(by + 1, H);

    if (xi0 < 0 || yi0 < 0 || xi1 >= W || yi1 >= H) return -1;   // unusable
    const float p00 = at(xi0, yi0), p10 = at(xi1, yi0);
    const float p01 = at(xi0, yi1), p11 = at(xi1, yi1);
    const float t = p00 * (1 - qx) * (1 - qy) + p10 * qx * (1 - qy) +
                    p01 * (1 - qx) * qy + p11 * qx * qy;
    return (int)std::max(0.0f, std::min(255.0f, std::floor(t + 0.5f)));
}

int main() {
    const int W = 48, H = 48;
    cv::Mat img(H, W, CV_8UC1);
    cv::RNG rng(99);
    rng.fill(img, cv::RNG::UNIFORM, 0, 256);

    long hitA = 0, hitB = 0, tot = 0, badA = 0, badB = 0;

    for (int t8 = 1; t8 <= 200; ++t8) {          // 1/32 pixel steps -> exact sub-texel
        const double dx = t8 / 32.0;
        cv::Mat M = (cv::Mat_<double>(2, 3) << 1.0, 0.0, dx, 0.0, 1.0, 0.0);
        cv::Mat Mi;
        cv::invertAffineTransform(M, Mi);
        cv::Mat dst;
        cv::warpAffine(img, dst, M, img.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);

        for (int y = 4; y < H - 4; ++y)
            for (int x = 0; x < 12; ++x) {       // leading columns: taps off the edge
                const double sxd = Mi.at<double>(0, 0) * x + Mi.at<double>(0, 1) * y + Mi.at<double>(0, 2);
                const double syd = Mi.at<double>(1, 0) * x + Mi.at<double>(1, 1) * y + Mi.at<double>(1, 2);
                const int bx = (int)std::floor(sxd), by = (int)std::floor(syd);
                const float qx = (float)std::floor((sxd - bx) * 32.0 + 0.5) / 32.0f;
                const float qy = (float)std::floor((syd - by) * 32.0 + 0.5) / 32.0f;
                const int a = model(img.data, W, H, bx, by, qx, qy, 0);
                const int b = model(img.data, W, H, bx, by, qx, qy, 1);
                if (a < 0 || b < 0) continue;
                const int ref = dst.at<uint8_t>(y, x);
                ++tot;
                if (a == ref) ++hitA; else ++badA;
                if (b == ref) ++hitB; else ++badB;
            }
    }
    std::printf("out-of-frame tap samples compared : %ld\n", tot);
    std::printf("  A  reflect base, then base+1      : %ld exact, %ld wrong (%.3f%% wrong)\n",
                hitA, badA, 100.0 * (double)badA / (double)tot);
    std::printf("  B  reflect each tap independently : %ld exact, %ld wrong (%.3f%% wrong)\n",
                hitB, badB, 100.0 * (double)badB / (double)tot);
    return 0;
}