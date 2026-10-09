// ===========================================================================
// gen_fused_case.cpp -- build accuracy test cases for warp_median_fused.cl
//                      on the RK3588.  RUNS ON THE x86 HOST, NEEDS OpenCV.
//
// Why this runs here and not on the board: the CPU reference must come from
// real cv::warpAffine, because the whole bit-exactness argument for the fused
// kernel is "the GPU reproduces what OpenCV would have produced". Reproducing
// OpenCV's 1/32 interpolation quantisation and BORDER_REFLECT semantics by hand
// on the board would mean validating my reimplementation of OpenCV instead of
// the kernel -- a circular test. So the reference is computed here, with real
// OpenCV, and shipped to the board as data. See fused_case_format.h.
//
// CPU reference (identical semantics to host/ocl_fused_check.cpp):
//   warped[k] = cv::warpAffine(frame[t-(2k+2)], fwd[k], (W,H), INTER_LINEAR,
//                              BORDER_REFLECT)
//   Ch0       = I_t
//   Ch1       = absdiff(I_t, warped[0])
//   Ch2       = max(0, I_t - median(warped[0..20]))
//
// The GPU takes a different route on purpose: it samples a PRE-PADDED copy
// with CLK_ADDRESS_CLAMP_TO_EDGE, which bakes the same border model in ahead
// of time. required_pad() sizes the pad so the two routes must agree.
//
// Usage:
//   gen_fused_case --seq <dir> --out <file.bin> [--cases N] [--t0 T]
//                  [--transform gmc|synth] [--disp px]
// ===========================================================================

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <dirent.h>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "fused_case_format.h"

static const int WINDOW = FKC_WINDOW;

static void die(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::fprintf(stderr, "[FATAL] ");
  std::vfprintf(stderr, fmt, ap);
  std::fprintf(stderr, "\n");
  va_end(ap);
  std::exit(2);
}

// ---------------------------------------------------------------------------
// Frame loading
// ---------------------------------------------------------------------------
// POSIX dirent rather than cv::filesystem: the OpenCV alias moved from cv::fs
// to cv::filesystem across the 4.x series, and this tool should not break on
// that. The frame names are zero-padded, so a plain lexicographic sort is also
// the numeric order.
static std::vector<std::string> list_frames(const std::string& dir) {
  std::vector<std::string> out;
  DIR* d = opendir(dir.c_str());
  if (!d) die("cannot open directory %s", dir.c_str());
  while (struct dirent* e = readdir(d)) {
    const std::string name = e->d_name;
    if (name.size() > 4 && name.compare(name.size() - 4, 4, ".jpg") == 0)
      out.push_back(dir + "/" + name);
  }
  closedir(d);
  std::sort(out.begin(), out.end());
  return out;
}

static cv::Mat load_gray(const std::string& path) {
  cv::Mat m = cv::imread(path, cv::IMREAD_GRAYSCALE);
  if (m.empty()) die("cannot read image: %s", path.c_str());
  return m;
}

// ---------------------------------------------------------------------------
// Similarity transform, matching the primitives gmc_stream.cpp uses:
//   goodFeaturesToTrack(600, 0.01, 4, blockSize=3)
//   calcOpticalFlowPyrLK
//   estimateAffinePartial2D(..., RANSAC, 3.0)
// on a 1/8 downscale, then normalised to the full-resolution frame.
//
// Returns false when the estimate is unusable; the caller then falls back to
// identity and records it, because a silently-dropped lag would make the case
// look cleaner than it is.
// ---------------------------------------------------------------------------
static bool similarity(const cv::Mat& prev, const cv::Mat& curr, cv::Mat& fwd) {
  const int W = prev.cols, H = prev.rows;
  cv::Mat ps, cs;
  cv::resize(prev, ps, cv::Size(W / 8, H / 8), 0, 0, cv::INTER_AREA);
  cv::resize(curr, cs, cv::Size(W / 8, H / 8), 0, 0, cv::INTER_AREA);

  std::vector<cv::Point2f> p0, p1;
  cv::goodFeaturesToTrack(ps, p0, 600, 0.01, 4, cv::noArray(), 3);
  if (p0.size() < 8) return false;

  std::vector<uchar> status;
  std::vector<float> err;
  cv::calcOpticalFlowPyrLK(ps, cs, p0, p1, status, err, cv::Size(21, 21), 3);

  std::vector<cv::Point2f> a, b;
  for (size_t i = 0; i < p0.size(); ++i)
    if (status[i]) { a.push_back(p0[i]); b.push_back(p1[i]); }
  if (a.size() < 8) return false;

  cv::Mat M = cv::estimateAffinePartial2D(a, b, cv::noArray(), cv::RANSAC, 3.0);
  if (M.empty() || M.rows != 2 || M.cols != 3) return false;

  fwd = cv::Mat::zeros(2, 3, CV_32F);
  for (int j = 0; j < 3; ++j) {
    const double v = M.at<double>(0, j);
    if (!std::isfinite(v)) return false;
    fwd.at<float>(0, j) = (float)v;
  }
  for (int j = 0; j < 3; ++j) {
    const double v = M.at<double>(1, j);
    if (!std::isfinite(v)) return false;
    fwd.at<float>(1, j) = (float)v;
  }
  return true;
}

// OpenCV 4.10's invertAffineTransform returns the OUTPUT in the INPUT's element
// type, so a CV_32F input yields CV_32F, not the CV_64F the docs suggest.
// Reading that 4-byte result through at<double> strides 8 bytes over it and
// yields denormals, which later detonate as a 4-billion-pixel image request.
// Hence the explicit normalisation, copied from ocl_fused_check.cpp.
static cv::Mat invert64(const cv::Mat& fwd) {
  cv::Mat Mi, Mi64;
  cv::invertAffineTransform(fwd, Mi);
  Mi.convertTo(Mi64, CV_64F);
  return Mi64;
}

static cv::Mat synth_forward(double disp, cv::RNG& rng) {
  const double ang = (rng.uniform(0.0, 1.0) - 0.5) * (disp * 0.02);
  const double sc = 1.0 + (rng.uniform(0.0, 1.0) - 0.5) * (disp * 0.01);
  const double tx = (rng.uniform(0.0, 1.0) - 0.5) * 2.0 * disp;
  const double ty = (rng.uniform(0.0, 1.0) - 0.5) * 2.0 * disp;
  cv::Mat M = cv::Mat::zeros(2, 3, CV_32F);
  M.at<float>(0, 0) = (float)(sc * std::cos(ang));
  M.at<float>(0, 1) = (float)(-sc * std::sin(ang));
  M.at<float>(0, 2) = (float)tx;
  M.at<float>(1, 0) = (float)(sc * std::sin(ang));
  M.at<float>(1, 1) = (float)(sc * std::cos(ang));
  M.at<float>(1, 2) = (float)ty;
  return M;
}

// ---------------------------------------------------------------------------
// required_pad -- ported verbatim in behaviour from ocl_fused_check.cpp.
// p starts at 2: even an identity warp needs two rows/cols of border so the
// bilinear tap support lands inside the padded image and CLAMP_TO_EDGE never
// silently alters a value.
// ---------------------------------------------------------------------------
static int required_pad(const std::vector<cv::Mat>& inv, int W, int H) {
  int p = 2;
  for (const cv::Mat& m : inv) {
    if (m.empty() || m.rows != 2 || m.cols != 3 || m.type() != CV_64FC1)
      die("inverse matrix is not a 2x3 CV_64F (rows=%d cols=%d type=%d)", m.rows,
          m.cols, m.type());
    const double a = m.at<double>(0, 0), b = m.at<double>(0, 1), c = m.at<double>(0, 2);
    const double d = m.at<double>(1, 0), e = m.at<double>(1, 1), f = m.at<double>(1, 2);
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) ||
        !std::isfinite(d) || !std::isfinite(e) || !std::isfinite(f))
      die("non-finite inverse matrix");
    for (int corner = 0; corner < 4; ++corner) {
      const double X = (corner & 1) ? W : 0.0;
      const double Y = (corner & 2) ? H : 0.0;
      const double sx = a * X + b * Y + c;
      const double sy = d * X + e * Y + f;
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

int main(int argc, char** argv) {
  std::string seq_dir, out_path;
  int n_cases = 3, t0 = -1, disp = 8;
  std::string transform = "gmc";

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char* what) -> std::string {
      if (i + 1 >= argc) die("%s needs a value", what);
      return argv[++i];
    };
    if (a == "--seq") seq_dir = need("--seq");
    else if (a == "--out") out_path = need("--out");
    else if (a == "--cases") n_cases = std::atoi(need("--cases").c_str());
    else if (a == "--t0") t0 = std::atoi(need("--t0").c_str());
    else if (a == "--disp") disp = std::atoi(need("--disp").c_str());
    else if (a == "--transform") transform = need("--transform");
    else die("unknown argument: %s", a.c_str());
  }
  if (seq_dir.empty() || out_path.empty()) {
    std::fprintf(stderr,
                 "usage: gen_fused_case --seq <dir> --out <file.bin> "
                 "[--cases N] [--t0 T] [--transform gmc|synth] [--disp px]\n");
    return 2;
  }
  if (transform != "gmc" && transform != "synth")
    die("--transform must be gmc or synth");

  std::vector<std::string> files = list_frames(seq_dir);
  if (files.empty()) die("no .jpg frames under %s", seq_dir.c_str());
  // t must exceed 42 so that lag k=20, i.e. t-(2*20+2) = t-42, stays >= 0.
  const int first_t = (t0 > 0) ? t0 : 43;
  if (first_t - 42 < 0) die("--t0 too small");
  if (first_t >= (int)files.size()) die("--t0 %d beyond sequence length %zu", first_t, files.size());
  if (first_t + n_cases > (int)files.size())
    die("need %d cases from t=%d but sequence has %zu frames", n_cases, first_t, files.size());

  std::printf("[gen] sequence : %s (%zu frames)\n", seq_dir.c_str(), files.size());
  std::printf("[gen] transform: %s%s\n", transform.c_str(),
              transform == "synth" ? " (synthetic displacement)" : " (real GMC)");

  // Load the whole span once; the network mount makes per-frame reads costly.
  std::vector<cv::Mat> imgs;
  for (int i = first_t - 42; i < first_t + n_cases; ++i) imgs.push_back(load_gray(files[i]));
  const int W = imgs[0].cols, H = imgs[0].rows;
  for (const cv::Mat& m : imgs)
    if (m.cols != W || m.rows != H) die("mixed resolutions in this sequence");
  std::printf("[gen] frames   : %dx%d, %zu loaded\n", W, H, imgs.size());

  // pad is a property of the whole bundle, so size it from the worst case over
  // all cases rather than per case -- one header, one pad, simpler contract.
  std::vector<std::vector<cv::Mat>> all_fwd(n_cases), all_inv(n_cases);
  int pad = 2;
  uint32_t total_gmc_failed = 0;
  for (int c = 0; c < n_cases; ++c) {
    const int t_global = first_t + c;         // index into the full sequence
    const int t_local = 42 + c;               // index into `imgs`
    cv::RNG rng((unsigned)(t_global * 1000003u));
    for (int k = 0; k < WINDOW; ++k) {
      const int li = t_local - (2 * k + 2);
      cv::Mat fwd;
      bool ok = true;
      if (transform == "gmc") {
        ok = similarity(imgs[li], imgs[t_local], fwd);
        if (!ok) {
          // Fall back to identity but record it: a silently-dropped lag would
          // make the case look cleaner than it is.
          std::fprintf(stderr,
                       "[gen] WARNING: GMC failed for lag k=%d (t=%d), identity used\n",
                       k, t_global);
          fwd = cv::Mat::eye(2, 3, CV_32F);
        }
      } else {
        fwd = synth_forward(disp, rng);
      }
      if (!ok) ++total_gmc_failed;
      all_fwd[c].push_back(fwd);
      all_inv[c].push_back(invert64(fwd));
    }
    pad = std::max(pad, required_pad(all_inv[c], W, H));
  }
  if (pad > 4096) die("required_pad exploded to %d -- a matrix is wrong", pad);
  std::printf("[gen] pad      : %d (covers the largest tap excursion)\n", pad);
  if (total_gmc_failed)
    std::printf("[gen] WARNING  : %u GMC estimates failed, fell back to identity\n",
                total_gmc_failed);

  const int pw = W + 2 * pad, ph = H + 2 * pad;
  const size_t frame_bytes = (size_t)pw * (size_t)ph;
  const size_t ref_bytes = (size_t)3 * W * H;
  const size_t per_case = (size_t)WINDOW * 6 * sizeof(float) +
                          (size_t)(WINDOW + 1) * frame_bytes + ref_bytes;
  std::printf("[gen] payload  : %.1f MiB total (%.1f MiB/case, padded %dx%d)\n",
              per_case * n_cases / 1048576.0, per_case / 1048576.0, pw, ph);

  FILE* f = std::fopen(out_path.c_str(), "wb");
  if (!f) die("cannot open %s for writing", out_path.c_str());

  FkcHeader h{};
  std::memcpy(h.magic, FKC_MAGIC, 4);
  h.W = (uint32_t)W; h.H = (uint32_t)H; h.pad = (uint32_t)pad;
  h.n_cases = (uint32_t)n_cases; h.window = (uint32_t)WINDOW; h.reserved = 0;
  std::fwrite(&h, sizeof(h), 1, f);

  std::vector<uint8_t> buf(frame_bytes);
  std::vector<float> mats((size_t)WINDOW * 6);

  for (int c = 0; c < n_cases; ++c) {
    const int t_local = 42 + c;
    const int t_global = first_t + c;

    // ---- CPU reference, real OpenCV -------------------------------------
    std::vector<cv::Mat> warped(WINDOW);
    for (int k = 0; k < WINDOW; ++k) {
      const int li = t_local - (2 * k + 2);
      cv::warpAffine(imgs[li], warped[(size_t)k], all_fwd[c][(size_t)k],
                     cv::Size(W, H), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    }
    cv::Mat ch1;
    cv::absdiff(imgs[t_local], warped[0], ch1);
    const cv::Mat bg = median_plane(warped);
    cv::Mat ch2(H, W, CV_8UC1);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        const int v = (int)imgs[t_local].at<uint8_t>(y, x) - (int)bg.at<uint8_t>(y, x);
        ch2.at<uint8_t>(y, x) = (uint8_t)(v > 0 ? v : 0);
      }

    // ---- metadata -------------------------------------------------------
    double motion = 0;
    for (int k = 0; k < WINDOW; ++k) {
      const cv::Mat& f_ = all_fwd[c][(size_t)k];
      motion += std::hypot((double)f_.at<float>(0, 2), (double)f_.at<float>(1, 2));
    }
    motion /= WINDOW;

    FkcCaseMeta m{};
    std::snprintf(m.seq, sizeof(m.seq), "%s", seq_dir.c_str());
    m.t = (uint32_t)t_global;
    m.gmc_failed = 0;
    m.motion_px = (float)motion;

    // ---- mats (float32 inverse affine, 21 x 6) --------------------------
    for (int k = 0; k < WINDOW; ++k)
      for (int j = 0; j < 6; ++j)
        mats[(size_t)k * 6 + (size_t)j] =
            (float)all_inv[c][(size_t)k].at<double>(j / 3, j % 3);

    std::fwrite(&m, sizeof(m), 1, f);
    std::fwrite(mats.data(), sizeof(float), mats.size(), f);

    // ---- padded frames: 21 lagged, then the current ---------------------
    // BORDER_REFLECT here is the whole point of the pad: it pre-bakes the same
    // border model the CPU reference gets from warpAffine's BORDER_REFLECT, so
    // the GPU's CLAMP_TO_EDGE sampler never has to invent one.
    for (int k = 0; k <= WINDOW; ++k) {
      const cv::Mat& src = imgs[k == WINDOW ? t_local : (t_local - (2 * k + 2))];
      cv::Mat padded;
      cv::copyMakeBorder(src, padded, pad, pad, pad, pad, cv::BORDER_REFLECT);
      std::memcpy(buf.data(), padded.data, frame_bytes);
      std::fwrite(buf.data(), 1, frame_bytes, f);
    }

    // ---- reference: plane-major [Ch0, Ch1, Ch2] --------------------------
    const cv::Mat* planes[3] = {&imgs[t_local], &ch1, &ch2};
    for (int p = 0; p < 3; ++p)
      std::fwrite(planes[p]->data, 1, (size_t)W * H, f);

    std::printf("[gen] case %d/%d  t=%d  motion=%.2f px  ch2 mean=%.2f\n", c + 1, n_cases,
                t_global, motion, cv::mean(ch2)[0]);
  }
  std::fclose(f);

  // Report the size actually on disk rather than recomputing it.
  std::FILE* probe = std::fopen(out_path.c_str(), "rb");
  long bytes = 0;
  if (probe) { std::fseek(probe, 0, SEEK_END); bytes = std::ftell(probe); std::fclose(probe); }
  std::printf("[gen] wrote    : %s (%.1f MiB on disk)\n", out_path.c_str(),
              (double)bytes / 1048576.0);
  std::printf("[gen] done\n");
  return 0;
}
