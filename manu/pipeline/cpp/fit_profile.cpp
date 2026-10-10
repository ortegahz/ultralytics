// ===========================================================================
// fit_profile.cpp -- where does GMC Fit actually spend its time, and what does
// per-frame caching buy without changing the result?
//
// Production fit = FastGMCEstimator::compute_affine (gmc_stream.cpp):
//   resize x2 -> goodFeaturesToTrack -> calcOpticalFlowPyrLK -> RANSAC.
//
// Three per-frame quantities are recomputed on every single fit even though
// they depend only on ONE input image, and each image is the "prev" side of
// many fits in a frame (one per anchor lag 2/12/22/32/42, plus the cached
// one-step fits the chain composition needs):
//
//   1. the half-res resize        -- pure function of that image
//   2. the Shi-Tomasi corners    -- pure function of that image
//   3. the optical-flow pyramid   -- pure function of that image
//
// calcOpticalFlowPyrLK internally rebuilds (3) for BOTH images on every call,
// and then recomputes the Scharr derivative planes per level per call. Passing
// an already-built pyramid avoids both: calc() accepts a STD_VECTOR_MAT input
// (lkpyramid.cpp:1302/1330) and detects precomputed gradients via the odd
// level count + channel/depth test (lkpyramid.cpp:1309), switching to lvlStep=2
// and reusing the derivative planes (lkpyramid.cpp:1392).
// buildOpticalFlowPyramid's defaults -- withDerivatives=true,
// pyrBorder=BORDER_REFLECT_101, derivBorder=BORDER_CONSTANT -- are exactly what
// calc() passes when it builds them itself, so the pyramids are the same bytes.
//
// Coordinates: pyramid level 0 is an ROI view into the padded buffer and
// LKTrackerInvoker indexes relative to that view, so points stay in UNPADDED
// image coordinates (lkpyramid.cpp:204-220). No coordinate shift is applied.
//
// Arms:
//   A  today's path: resize + corners + RANSAC + LK(pyramid built inside), per fit
//   B  pyramid reuse only
//   C  pyramid reuse + corner cache + resize cache
//
// All three must agree bit-for-bit; the transform comparison at the end is the
// check, not the proof.
// ===========================================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <dirent.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

using cv::Mat;
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point a) {
    return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
}

static bool is_image_ext(const std::string& p) {
    if (p.size() < 4) return false;
    std::string e = p.substr(p.size() - 4);
    for (char& c : e) c = (char)tolower((unsigned char)c);
    return e == ".bmp" || e == ".png" || e == ".jpg" || e == ".jpeg";
}
static std::vector<std::string> list_images(const std::string& dir) {
    std::vector<std::string> names;
    DIR* d = opendir(dir.c_str());
    if (!d) { std::fprintf(stderr, "[FATAL] cannot open %s\n", dir.c_str()); std::exit(2); }
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        const std::string n = e->d_name;
        if (n == "." || n == "..") continue;
        if (is_image_ext(n)) names.push_back(n);
    }
    closedir(d);
    std::sort(names.begin(), names.end());
    return names;
}

struct Arm {
    const char* name = "";
    double resize = 0, gftt = 0, lk = 0, ransac = 0, pyrbuild = 0;
    long n_fits = 0, n_ok = 0, n_corners = 0, n_surv = 0, n_inliers = 0;
    double ratio_sum = 0, ratio_n = 0;
    double total() const { return resize + gftt + lk + ransac + pyrbuild; }
};

// Resize + corner detection. Cached in arms that keep a per-frame cache.
static void make_small(const Mat& g, int ds, Mat& out) {
    if (ds <= 1) out = g;
    else cv::resize(g, out, cv::Size(g.cols / ds, g.rows / ds), 0, 0, cv::INTER_LINEAR);
}
static void make_corners(const Mat& small, std::vector<cv::Point2f>& pts) {
    pts.clear();
    cv::goodFeaturesToTrack(small, pts, 600, 0.01, 4, cv::noArray(), 3);
}
static void ransac_stage(const std::vector<cv::Point2f>& pts,
                         const std::vector<cv::Point2f>& nc,
                         const std::vector<unsigned char>& st, Arm& a) {
    std::vector<cv::Point2f> p0, p1;
    for (size_t i = 0; i < pts.size(); ++i)
        if (i < st.size() && st[i] == 1) { p0.push_back(pts[i]); p1.push_back(nc[i]); }
    a.n_surv += (long)p0.size();
    if (p0.size() < 6) return;
    Mat inl;
    // RANSAC draws from the global RNG. Arms run back-to-back in one process,
    // so without a reseed each arm would start from a different RNG state and
    // the inlier statistics would not be comparable at all. Reseeding per fit
    // makes every arm's fit k see identical randomness.
    cv::theRNG().state = 0xFFFFFFFFu;
    auto t0 = Clock::now();
    const Mat M = cv::estimateAffinePartial2D(p0, p1, inl, cv::RANSAC, 3.0);
    a.ransac += ms_since(t0);
    if (M.empty()) return;
    const long ninl = inl.empty() ? 0 : cv::countNonZero(inl.reshape(1));
    a.n_inliers += ninl;
    if (!inl.empty() && inl.total()) { a.ratio_sum += (double)ninl / (double)inl.total(); ++a.ratio_n; }
    ++a.n_ok;
}

int main(int argc, char** argv) {
    const char* dir = argc > 1 ? argv[1] : "/home/manu/mnt/nfs/ocvparity/bmp";
    const int ds = argc > 2 ? atoi(argv[2]) : 2;
    const int maxLevel = argc > 3 ? atoi(argv[3]) : 2;
    cv::setNumThreads(1);  // same contract as production

    auto names = list_images(dir);
    if (names.empty()) { std::fprintf(stderr, "[FATAL] no images in %s\n", dir); return 2; }
    std::vector<Mat> frames;
    for (auto& n : names) frames.push_back(cv::imread(std::string(dir) + "/" + n, cv::IMREAD_GRAYSCALE));
    const int N = (int)frames.size();
    std::printf("[SETUP] dir=%s frames=%d downscale=%d maxLevel=%d threads=1 opencv=%s\n",
                dir, N, ds, maxLevel, CV_VERSION);
    std::printf("[SETUP] frame=%dx%d\n", frames[0].cols, frames[0].rows);

    const std::vector<int> anchors{2, 12, 22, 32, 42};  // frozen --anchor-step 10
    const cv::Size win(15, 15);

    // Per-frame caches, sized to the production ring depth.
    const int RING = 43;
    std::vector<Mat> small_c(RING);
    std::vector<std::vector<cv::Point2f>> corner_c(RING);
    std::vector<std::vector<Mat>> pyr_c(RING);
    // The cache is keyed by ABSOLUTE frame index, not by a "slot is filled"
    // boolean. With a plain boolean the ring fills once and then never
    // recomputes: slot(43) == slot(0), the flag left by frame 0 is still true,
    // and frame 43 is silently served frame 0's pixels. That failure is silent
    // -- no crash, no assert, just stale data -- so every slot records WHICH
    // absolute frame it currently holds.
    std::vector<int> tag_small(RING, -1), tag_corner(RING, -1), tag_pyr(RING, -1);
    long n_small_miss = 0, n_corner_miss = 0, n_pyr_miss = 0;

    Arm A{"A: today"}, B{"B: +pyr reuse"}, C{"C: +corner&resize cache"};

    auto slot = [&](int abs) { return ((abs % RING) + RING) % RING; };

    auto ensure_small = [&](int abs, Arm& a) -> const Mat& {
        const int s = slot(abs);
        if (tag_small[s] == abs) return small_c[s];
        auto t0 = Clock::now();
        make_small(frames[abs], ds, small_c[s]);
        a.resize += ms_since(t0);
        tag_small[s] = abs;
        ++n_small_miss;
        return small_c[s];
    };
    auto ensure_corner = [&](int abs, const Mat& sm, Arm& a) -> const std::vector<cv::Point2f>& {
        const int s = slot(abs);
        if (tag_corner[s] == abs) return corner_c[s];
        auto t0 = Clock::now();
        make_corners(sm, corner_c[s]);
        a.gftt += ms_since(t0);
        tag_corner[s] = abs;
        ++n_corner_miss;
        return corner_c[s];
    };
    auto ensure_pyr = [&](int abs, const Mat& sm, Arm& a) -> const std::vector<Mat>& {
        const int s = slot(abs);
        if (tag_pyr[s] == abs) return pyr_c[s];
        auto t0 = Clock::now();
        cv::buildOpticalFlowPyramid(sm, pyr_c[s], win, maxLevel, /*withDerivatives=*/true);
        a.pyrbuild += ms_since(t0);
        tag_pyr[s] = abs;
        ++n_pyr_miss;
        return pyr_c[s];
    };

    // arm B's own pyramid ring (arm C uses pyr_c).
    std::vector<std::vector<Mat>> pyrB(RING);
    std::vector<int> tag_pyrB(RING, -1);
    auto ensure_pyrB = [&](int abs, const Mat& sm, Arm& a) -> const std::vector<Mat>& {
        const int s = slot(abs);
        if (tag_pyrB[s] == abs) return pyrB[s];
        auto t0 = Clock::now();
        cv::buildOpticalFlowPyramid(sm, pyrB[s], win, maxLevel, /*withDerivatives=*/true);
        a.pyrbuild += ms_since(t0);
        tag_pyrB[s] = abs;
        return pyrB[s];
    };

    long n_pairs = 0;
    long cache_check = 0, cache_bad_small = 0, cache_bad_corners = 0;
    for (int t = 1; t <= N; ++t) {
        for (int lag : anchors) {
            const int pa = t - lag, ca = t - 1;  // absolute indices
            if (pa < 0) continue;
            ++n_pairs;

            // ---------------- arm A: today's path, nothing cached -------------
            {
                Mat ps, cs;
                auto t0 = Clock::now(); make_small(frames[pa], ds, ps); A.resize += ms_since(t0);
                t0 = Clock::now();    make_small(frames[ca], ds, cs); A.resize += ms_since(t0);
                std::vector<cv::Point2f> pts;
                t0 = Clock::now();    make_corners(ps, pts);          A.gftt   += ms_since(t0);
                A.n_corners += (long)pts.size();
                if (pts.size() >= 6) {
                    Mat pm((int)pts.size(), 1, CV_32FC2, pts.data());
                    std::vector<cv::Point2f> nc; std::vector<unsigned char> st; std::vector<float> er;
                    t0 = Clock::now();
                    cv::calcOpticalFlowPyrLK(ps, cs, pm, nc, st, er, win, maxLevel);
                    A.lk += ms_since(t0);
                    ++A.n_fits;
                    ransac_stage(pts, nc, st, A);
                }
            }

            // ---------------- arm B: pyramid reuse only ----------------------
            {
                Mat ps, cs;
                auto t0 = Clock::now(); make_small(frames[pa], ds, ps); B.resize += ms_since(t0);
                t0 = Clock::now();    make_small(frames[ca], ds, cs); B.resize += ms_since(t0);
                std::vector<cv::Point2f> pts;
                t0 = Clock::now();    make_corners(ps, pts);          B.gftt   += ms_since(t0);
                B.n_corners += (long)pts.size();
                if (pts.size() >= 6) {
                    Mat pm((int)pts.size(), 1, CV_32FC2, pts.data());
                    std::vector<cv::Point2f> nc; std::vector<unsigned char> st; std::vector<float> er;
                    // arm B keeps its own pyramid ring so its timing is not
                    // charged for (or credited to) arm C's cache state.
                    ensure_pyrB(pa, ps, B);
                    ensure_pyrB(ca, cs, B);
                    t0 = Clock::now();
                    cv::calcOpticalFlowPyrLK(pyrB[slot(pa)], pyrB[slot(ca)], pm, nc, st, er, win, maxLevel);
                    B.lk += ms_since(t0);
                    ++B.n_fits;
                    ransac_stage(pts, nc, st, B);
                }
            }

            // ---------------- arm C: pyramid + corner + resize cache ----------
            {
                const Mat& ps = ensure_small(pa, C);
                const Mat& cs = ensure_small(ca, C);
                const std::vector<cv::Point2f>& pts = ensure_corner(pa, ps, C);
                C.n_corners += (long)pts.size();

                // Self-check: the cache must return exactly what a fresh
                // computation would. Verified on every cache hit, not sampled.
                {
                    Mat fresh_s; make_small(frames[pa], ds, fresh_s);
                    if (fresh_s.size() != ps.size() || cv::countNonZero(fresh_s != ps) != 0)
                        ++cache_bad_small;
                    std::vector<cv::Point2f> fresh_c;
                    make_corners(fresh_s, fresh_c);
                    if (fresh_c.size() != pts.size()) ++cache_bad_corners;
                    else for (size_t i = 0; i < fresh_c.size(); ++i)
                        if (fresh_c[i].x != pts[i].x || fresh_c[i].y != pts[i].y) {
                            ++cache_bad_corners;
                            break;
                        }
                    ++cache_check;
                }

                if (pts.size() >= 6) {
                    Mat pm((int)pts.size(), 1, CV_32FC2, const_cast<cv::Point2f*>(pts.data()));
                    std::vector<cv::Point2f> nc; std::vector<unsigned char> st; std::vector<float> er;
                    ensure_pyr(pa, ps, C);
                    ensure_pyr(ca, cs, C);
                    auto t0 = Clock::now();
                    cv::calcOpticalFlowPyrLK(pyr_c[slot(pa)], pyr_c[slot(ca)], pm, nc, st, er, win, maxLevel);
                    C.lk += ms_since(t0);
                    ++C.n_fits;
                    ransac_stage(pts, nc, st, C);
                }
            }
        }
    }

    // ---- clean bit-exactness check (own counters, no timing) ---------------
    long cmp = 0; double maxPt = 0, maxSt = 0, maxEr = 0;
    for (int t = 1; t <= N; ++t) {
        for (int lag : anchors) {
            const int pa = t - lag, ca = t - 1;
            if (pa < 0) continue;
            Mat ps, cs;
            make_small(frames[pa], ds, ps);
            make_small(frames[ca], ds, cs);
            std::vector<cv::Point2f> pts;
            make_corners(ps, pts);
            if (pts.size() < 6) continue;
            Mat pm((int)pts.size(), 1, CV_32FC2, pts.data());
            std::vector<cv::Point2f> n1, n2;
            std::vector<unsigned char> s1, s2;
            std::vector<float> e1, e2;
            cv::calcOpticalFlowPyrLK(ps, cs, pm, n1, s1, e1, win, maxLevel);
            std::vector<Mat> p1, p2;
            cv::buildOpticalFlowPyramid(ps, p1, win, maxLevel, true);
            cv::buildOpticalFlowPyramid(cs, p2, win, maxLevel, true);
            auto pts2 = pts;  // same gFTT result, used to keep the two runs independent
            Mat pm2((int)pts2.size(), 1, CV_32FC2, pts2.data());
            cv::calcOpticalFlowPyrLK(p1, p2, pm2, n2, s2, e2, win, maxLevel);
            ++cmp;
            if (n1.size() != n2.size()) { maxPt = 1e9; break; }
            for (size_t i = 0; i < n1.size(); ++i) {
                maxPt = std::max(maxPt, (double)std::abs(n1[i].x - n2[i].x));
                maxPt = std::max(maxPt, (double)std::abs(n1[i].y - n2[i].y));
                maxSt = std::max(maxSt, (double)std::abs((int)s1[i] - (int)s2[i]));
                maxEr = std::max(maxEr, (double)std::abs(e1[i] - e2[i]));
            }
        }
    }

    const double tA = A.total(), tB = B.total(), tC = C.total();
    std::printf("\n================= FIT BREAKDOWN (%ld fits) =================\n", n_pairs);
    auto line = [&](const char* k, double a, double b, double c) {
        std::printf("  %-22s %8.2f %5.1f%% | %8.2f %5.1f%% | %8.2f %5.1f%%\n", k,
                    a, 100 * a / tA, b, 100 * b / tB, c, 100 * c / tC);
    };
    std::printf("  %-22s %8s %6s | %8s %6s | %8s %6s\n", "", "A ms", "%", "B ms", "%", "C ms", "%");
    line("resize",       A.resize,  B.resize,  C.resize);
    line("goodFeatures", A.gftt,    B.gftt,    C.gftt);
    line("LK",           A.lk,      B.lk,      C.lk);
    line("pyr build",    A.pyrbuild,B.pyrbuild,C.pyrbuild);
    line("RANSAC",       A.ransac,  B.ransac,  C.ransac);
    std::printf("  %-22s %8.2f       | %8.2f       | %8.2f\n", "TOTAL", tA, tB, tC);
    std::printf("  %-22s %12s | %7.3fx    | %7.3fx\n", "speedup", "", tA / tB, tA / tC);
    std::printf("  cache misses: resize=%ld corner=%ld pyramid=%ld  (fits=%ld)\n",
                n_small_miss, n_corner_miss, n_pyr_miss, n_pairs);

    std::printf("\n================= ACCURACY OBSERVABLES =================\n");
    std::printf("  cache self-check: %ld hits, bad_resize=%ld bad_corners=%ld -> %s\n",
                cache_check, cache_bad_small, cache_bad_corners,
                (cache_bad_small == 0 && cache_bad_corners == 0) ? "CACHE EXACT" : "CACHE CORRUPT");
    std::printf("  raw counts:\n");
    for (const Arm* a : {&A, &B, &C})
        std::printf("    %-28s n_fits=%ld n_corners=%ld n_surv=%ld n_inliers=%ld\n",
                    a->name, a->n_fits, a->n_corners, a->n_surv, a->n_inliers);
    for (const Arm* a : {&A, &B, &C}) {
        std::printf("  %-28s corners/fits=%6.1f  surv=%5.2f%%  inlier=%5.2f%%  ratio=%.5f  ok=%ld\n",
                    a->name, a->n_corners / (double)a->n_fits,
                    100.0 * a->n_surv / a->n_corners, 100.0 * a->n_inliers / a->n_corners,
                    a->ratio_n ? a->ratio_sum / a->ratio_n : 0.0, a->n_ok);
    }

    std::printf("\n================= BIT-EXACTNESS =================\n");
    std::printf("  compared fits : %ld\n", cmp);
    std::printf("  max |dpoint|  : %.9g px\n", maxPt);
    std::printf("  max |dstatus| : %.9g\n", maxSt);
    std::printf("  max |derr|    : %.9g\n", maxEr);
    std::printf("  VERDICT       : %s\n",
                (maxPt == 0 && maxSt == 0 && maxEr == 0) ? "BIT-IDENTICAL" : "DIVERGES");
    return 0;
}