/* ============================================================================
 * probe_interp2.cpp -- EXTRACT cv::warpAffine's actual bilinear weights with a
 *                       step-edge oracle, instead of guessing at its source.
 *
 * ----------------------------------------------------------------------------
 * WHY AN ORACLE
 * ----------------------------------------------------------------------------
 * probe_interp.cpp narrowed 36 candidate models down to exactly one structure:
 *
 *     src = M^-1 * [x, y, 1]        (NO half-pixel offset)
 *     position quantised to 1/32, ROUNDED
 *     output rounded half-up
 *
 * at 99.58% exact. The residual 0.42% still reaches Max|Diff| = 8 on pure random
 * noise, so some further detail of the weight table is wrong. Guessing at that
 * table from memory is what produced 11 silent defects last stage, so read it
 * off the library directly.
 *
 * THE ORACLE
 * ----------
 * A vertical step edge is an exact readout of the horizontal weight:
 *
 *     I(x, y) = (x >= X0) ? 255 : 0
 *     M       = [[1, 0, -phi], [0, 1, 0]]      ->  src = (x + phi, y)
 *     read dst at x = X0 - 1, y = interior
 *
 *   src x = X0 - 1 + phi, so floor(sx) = X0 - 1, and the 2x2 tap is
 *   (0, 255) horizontally. src y = y exactly, so there is no vertical mixing.
 *   =>  out = f(255 * w_right(phi))
 *
 * So out/255 IS OpenCV's effective right-hand weight, read directly. Sweeping
 * phi over 1024 sub-steps prints the exact quantisation staircase. The same
 * construction transposed gives the vertical weight. This also separates
 * "position is quantised" from "the weight table is non-uniform" -- the first
 * probe could not tell those apart.
 *
 * PC-ONLY OFFLINE TOOL -- not part of the RK3588 runtime.
 * ==========================================================================*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    const int W = 128, H = 128, X0 = 64, Y0 = 64, STEPS = 1024;

    // ---------------- horizontal weight oracle ----------------
    cv::Mat img(H, W, CV_8UC1, cv::Scalar(0));
    img.colRange(X0, W).setTo(cv::Scalar(255));

    std::vector<int> hx(STEPS + 1), hy(STEPS + 1);
    for (int k = 0; k <= STEPS; ++k) {
        const double phi = (double)k / STEPS;

        cv::Mat M = (cv::Mat_<double>(2, 3) << 1.0, 0.0, -phi, 0.0, 1.0, 0.0);
        cv::Mat d;
        cv::warpAffine(img, d, M, img.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        hx[k] = d.at<uint8_t>(H / 2, X0 - 1);

        cv::Mat Mv = (cv::Mat_<double>(2, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, -phi);
        cv::warpAffine(img, d, Mv, img.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        // src y = y + phi -> use an image with a horizontal edge for the y readout
        cv::Mat _unused = d;
        (void)_unused;
        hy[k] = -1;
    }

    // vertical oracle needs a horizontal edge
    cv::Mat vimg(W, H, CV_8UC1, cv::Scalar(0));   // indexed [y][x] below
    vimg.rowRange(Y0, H).setTo(cv::Scalar(255));
    for (int k = 0; k <= STEPS; ++k) {
        const double phi = (double)k / STEPS;
        cv::Mat Mv = (cv::Mat_<double>(2, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, -phi);
        cv::Mat d;
        cv::warpAffine(vimg, d, Mv, vimg.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        hy[k] = d.at<uint8_t>(Y0 - 1, W / 2);
    }

    auto report = [&](const char* tag, const std::vector<int>& v) {
        std::printf("\n--- %s effective weight staircase (out = f(255*w)) ---\n", tag);
        // count distinct plateaus
        int changes = 0;
        for (int k = 1; k <= STEPS; ++k) if (v[k] != v[k - 1]) ++changes;
        std::printf("  %d distinct levels over 1024 sub-steps\n", changes + 1);
        std::printf("  %6s %10s %10s %10s %10s %8s\n", "k", "phi", "out", "out/255",
                    "model_rnd32", "match");
        int agree_rnd32 = 0, agree_trunc32 = 0, agree_exact = 0;
        for (int k = 0; k <= STEPS; k += 16) {
            const double phi = (double)k / STEPS;
            const int m_rnd = (int)std::floor(255.0 * std::floor(phi * 32.0 + 0.5) / 32.0 + 0.5);
            const int m_trn = (int)std::floor(255.0 * std::floor(phi * 32.0) / 32.0 + 0.5);
            const int m_exa = (int)std::floor(255.0 * phi + 0.5);
            agree_rnd32 += (v[k] == m_rnd);
            agree_trunc32 += (v[k] == m_trn);
            agree_exact += (v[k] == m_exa);
            if (k <= 256 || k % 128 == 0)
                std::printf("  %6d %10.6f %10d %10.6f %10d %8s\n", k, phi, v[k],
                            v[k] / 255.0, m_rnd, v[k] == m_rnd ? "ok" : "XX");
        }
        const int n = STEPS / 16 + 1;
        std::printf("  agreement over %d sampled k:  round32=%d  trunc32=%d  exact=%d\n", n,
                    agree_rnd32, agree_trunc32, agree_exact);
    };

    report("HORIZONTAL", hx);
    report("VERTICAL  ", hy);
    return 0;
}