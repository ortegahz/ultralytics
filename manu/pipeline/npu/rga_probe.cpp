// ===========================================================================
// rga_probe.cpp -- RK3588 RGA (im2d v2 API) capability probe.
//
// Grounded in the official SDK material (external/linux-rga in the RK3588 SDK):
//   docs/Rockchip_Developer_Guide_RGA_CN.md
//   docs/Rockchip_FAQ_RGA_CN.md
//   samples/im2d_api_demo/rgaImDemo.cpp
//   samples/im2d_slt/sources/dma_alloc.cpp
//   kernel/include/uapi/linux/dma-heap.h
//
// MEASURED FACTS this probe encodes (board: RK3588, librga API 1.10.0_[9]):
//   * The board DOES have RGA: /usr/lib/aarch64-linux-gnu/librga.so.2,
//     /dev/rga, kernel module "rga v1.3.1", cores RGA2-Enhance + RGA3.
//   * querystring() reports: scale limit 0.0625~16, byte stride 16,
//     max input 8192x8192, max output 8128x8128, RGB_888 in/out supported.
//   * The ONLY buffer path that actually moves data on this board is
//       importbuffer_virtualaddr(va, w, h, fmt)  ->  wrapbuffer_handle(h, ...)
//     (what the official Linux demo does).  Passing a raw dma-buf fd through
//     wrapbuffer_fd_t() returns SUCCESS but moves no data at all.
//   * importbuffer_virtualaddr() captures the buffer CONTENT at import time,
//     so a buffer must be filled BEFORE it is imported.
//   * imcheck()/imcheck_t() must NOT be used on this librga: it prints
//     "src has not fd and address for render" (it validates the zeroed pat
//     buffer) and the real operation that follows then silently does nothing.
//     Byte budgets are checked explicitly instead.
//
// The probe answers, by measurement:
//   1. Does RGA's 4x / 8x downscale equal OpenCV INTER_AREA (block mean)?
//      The PC baseline builds the 160/80 thumbnails with INTER_AREA, so a
//      mismatch is a semantic change, not a rounding difference.
//   2. Can RGA assemble the 640x640 mosaic canvas, and does it match a CPU
//      reference byte for byte?
//   3. What does each step cost (import / resize / blit / fill)?
//
// Nothing here touches the NPU, so it cannot disturb the 2-context RKNN budget.
//
// BUILD  ./build_rga_probe_arm64.sh
// RUN    /root/rga_probe --only resize,mosaic,timing --iters 100
// ===========================================================================

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "im2d.h"
#include "RgaUtils.h"

namespace {

const int kNativeW = 640;  // feature width  (native pixels)
const int kNativeH = 512;  // feature height (native pixels)
const int kCanvas = 640;   // NPU canvas edge
const int kFmt = RK_FORMAT_RGB_888;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
double NowMs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

const char* StatusName(IM_STATUS s) {
  switch (s) {
    case IM_STATUS_SUCCESS: return "SUCCESS";
    case IM_STATUS_NOERROR: return "NOERROR";
    case IM_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
    case IM_STATUS_OUT_OF_MEMORY: return "OUT_OF_MEMORY";
    case IM_STATUS_INVALID_PARAM: return "INVALID_PARAM";
    case IM_STATUS_ILLEGAL_PARAM: return "ILLEGAL_PARAM";
    case IM_STATUS_ERROR_VERSION: return "ERROR_VERSION";
    case IM_STATUS_FAILED: return "FAILED";
    default: return "UNKNOWN";
  }
}

im_rect MakeRect(int x, int y, int w, int h) {
  im_rect r;
  r.x = x;
  r.y = y;
  r.width = w;
  r.height = h;
  return r;
}

// ---------------------------------------------------------------------------
// RGA buffer: exact-size allocation + import.
//
// The "exact size" and the "fill before import" rules are what make this work
// on the board's librga (see the header comment).
// ---------------------------------------------------------------------------
struct RgaBuf {
  uint8_t* data = nullptr;
  size_t size = 0;
  int w = 0, h = 0, fmt = 0;
  rga_buffer_handle_t handle = -1;
};

bool BufMake(RgaBuf* b, int w, int h, int fmt) {
  size_t bytes = (size_t)w * (size_t)h * (size_t)get_bpp_from_format(fmt);
  void* p = nullptr;
  if (posix_memalign(&p, 4096, bytes) != 0) return false;
  memset(p, 0, bytes);
  b->data = (uint8_t*)p;
  b->size = bytes;
  b->w = w;
  b->h = h;
  b->fmt = fmt;
  b->handle = -1;
  return true;
}

void BufFree(RgaBuf* b) {
  if (b->handle > 0) releasebuffer_handle(b->handle);
  if (b->data) free(b->data);
  b->data = nullptr;
  b->handle = -1;
}

// Import AFTER the content is final.  Returns false on failure.
bool BufImport(RgaBuf* b) {
  if (b->handle > 0) releasebuffer_handle(b->handle);
  b->handle = importbuffer_virtualaddr(b->data, b->w, b->h, b->fmt);
  return b->handle > 0;
}

rga_buffer_t BufAs(const RgaBuf& b) { return wrapbuffer_handle(b.handle, b.w, b.h, b.fmt); }

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

// OpenCV INTER_AREA equivalent for an exact integer downscale factor.
std::vector<uint8_t> RefAreaDown(const std::vector<uint8_t>& src, int sw, int sh, int dw, int dh,
                                 int ch) {
  std::vector<uint8_t> dst((size_t)dw * dh * ch, 0);
  const int bx = sw / dw, by = sh / dh;
  if (bx < 1 || by < 1) return dst;
  for (int y = 0; y < dh; ++y)
    for (int x = 0; x < dw; ++x)
      for (int c = 0; c < ch; ++c) {
        int sum = 0;
        for (int j = 0; j < by; ++j)
          for (int i = 0; i < bx; ++i)
            sum += src[((size_t)(y * by + j) * sw + (x * bx + i)) * ch + c];
        int n = bx * by;
        dst[((size_t)y * dw + x) * ch + c] = (uint8_t)((sum + n / 2) / n);
      }
  return dst;
}

// OpenCV INTER_LINEAR equivalent for uint8: half-pixel centred, BORDER_REPLICATE,
// 5-bit fixed-point weights (INTER_BITS = 5).
std::vector<uint8_t> RefBilinear(const std::vector<uint8_t>& src, int sw, int sh, int dw, int dh,
                                  int ch) {
  const int kBits = 5;
  std::vector<uint8_t> dst((size_t)dw * dh * ch, 0);
  std::vector<int> x0(dw), x1(dw), ax(dw);
  for (int x = 0; x < dw; ++x) {
    double fsx = (x + 0.5) * (double)sw / dw - 0.5;
    int sx = (int)std::floor(fsx);
    double fx = fsx - sx;
    if (fsx < 0) { sx = 0; fx = 0; }
    else if (fsx > sw - 1) { sx = sw - 1; fx = 0; }
    x0[x] = sx;
    x1[x] = std::min(sx + 1, sw - 1);
    ax[x] = std::min(kBits, std::max(0, (int)(fx * (1 << kBits) + 0.5)));
  }
  for (int y = 0; y < dh; ++y) {
    double fsy = (y + 0.5) * (double)sh / dh - 0.5;
    int sy = (int)std::floor(fsy);
    double fy = fsy - sy;
    if (fsy < 0) { sy = 0; fy = 0; }
    else if (fsy > sh - 1) { sy = sh - 1; fy = 0; }
    int y1 = std::min(sy + 1, sh - 1);
    int ay = std::min(kBits, std::max(0, (int)(fy * (1 << kBits) + 0.5)));
    for (int x = 0; x < dw; ++x)
      for (int c = 0; c < ch; ++c) {
        const uint8_t* p00 = &src[((size_t)sy * sw + x0[x]) * ch + c];
        const uint8_t* p01 = &src[((size_t)sy * sw + x1[x]) * ch + c];
        const uint8_t* p10 = &src[((size_t)y1 * sw + x0[x]) * ch + c];
        const uint8_t* p11 = &src[((size_t)y1 * sw + x1[x]) * ch + c];
        int top = (p00[0] * ((1 << kBits) - ax[x]) + p01[0] * ax[x] + (1 << (kBits - 1))) >> kBits;
        int bot = (p10[0] * ((1 << kBits) - ax[x]) + p11[0] * ax[x] + (1 << (kBits - 1))) >> kBits;
        int v = (top * ((1 << kBits) - ay) + bot * ay + (1 << (kBits - 1))) >> kBits;
        dst[((size_t)y * dw + x) * ch + c] = (uint8_t)v;
      }
  }
  return dst;
}

struct Diff {
  int max_abs = 0;
  double mean_abs = 0;
};

Diff Compare(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  Diff d;
  size_t n = std::min(a.size(), b.size());
  long long acc = 0;
  for (size_t i = 0; i < n; ++i) {
    int v = std::abs((int)a[i] - (int)b[i]);
    if (v > d.max_abs) d.max_abs = v;
    acc += v;
  }
  d.mean_abs = n ? (double)acc / (double)n : 0.0;
  return d;
}

// ---------------------------------------------------------------------------
// mosaic canvas layout (the geometry under test)
//   native : 640x512 at (0,0)       1:1
//   thumb160: 160x128 at (0,512)    1:4
//   thumb80 : 80x64   at (160,512)  1:8
// ---------------------------------------------------------------------------
struct Rect {
  int x, y, w, h;
};

const Rect kNativeRect{0, 0, 640, 512};
const Rect kT160Rect{0, 512, 160, 128};
const Rect kT80Rect{160, 512, 80, 64};
const uint8_t kBg = 114;  // the value the PC letterbox pad uses

void CpuBuildCanvas(const std::vector<uint8_t>& feat, std::vector<uint8_t>* canvas, int bg) {
  canvas->assign((size_t)kCanvas * kCanvas * 3, (uint8_t)bg);
  for (int y = 0; y < kNativeRect.h; ++y)
    memcpy(&(*canvas)[((size_t)(kNativeRect.y + y) * kCanvas + kNativeRect.x) * 3],
           &feat[(size_t)y * kNativeW * 3], (size_t)kNativeRect.w * 3);
  auto t160 = RefAreaDown(feat, kNativeW, kNativeH, kT160Rect.w, kT160Rect.h, 3);
  for (int y = 0; y < kT160Rect.h; ++y)
    memcpy(&(*canvas)[((size_t)(kT160Rect.y + y) * kCanvas + kT160Rect.x) * 3],
           &t160[(size_t)y * kT160Rect.w * 3], (size_t)kT160Rect.w * 3);
  auto t80 = RefAreaDown(feat, kNativeW, kNativeH, kT80Rect.w, kT80Rect.h, 3);
  for (int y = 0; y < kT80Rect.h; ++y)
    memcpy(&(*canvas)[((size_t)(kT80Rect.y + y) * kCanvas + kT80Rect.x) * 3],
           &t80[(size_t)y * kT80Rect.w * 3], (size_t)kT80Rect.w * 3);
}

struct Timing {
  double avg_ms = 0;
  double min_ms = 0;
};

template <typename F>
Timing TimeIt(int iters, F&& fn) {
  Timing t;
  double best = 1e30, total = 0;
  for (int i = 0; i < iters; ++i) {
    double a = NowMs();
    fn();
    double dt = NowMs() - a;
    total += dt;
    best = std::min(best, dt);
  }
  t.avg_ms = total / iters;
  t.min_ms = best;
  return t;
}

void FillSynthetic(std::vector<uint8_t>* img, int w, int h) {
  img->assign((size_t)w * h * 3, 0);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      uint8_t* p = &(*img)[((size_t)y * w + x) * 3];
      p[0] = (uint8_t)((x * 255) / (w > 1 ? w - 1 : 1));
      p[1] = (uint8_t)((y * 255) / (h > 1 ? h - 1 : 1));
      p[2] = (uint8_t)(((x + y) * 255) / (w + h - 2));
    }
  const int blobs[6][3] = {{100, 100, 12}, {300, 200, 8},  {500, 90, 20},
                            {200, 400, 6}, {420, 350, 10}, {60, 300, 5}};
  for (auto& b : blobs)
    for (int dy = -b[2]; dy <= b[2]; ++dy)
      for (int dx = -b[2]; dx <= b[2]; ++dx) {
        int x = b[0] + dx, y = b[1] + dy;
        if (x < 0 || y < 0 || x >= w || y >= h) continue;
        if (dx * dx + dy * dy > b[2] * b[2]) continue;
        uint8_t* p = &(*img)[((size_t)y * w + x) * 3];
        p[0] = 255; p[1] = 255; p[2] = 255;
      }
}

void Section(const char* title) {
  printf("\n================================================================\n");
  printf("  %s\n", title);
  printf("================================================================\n");
  fflush(stdout);
}

// ---------------------------------------------------------------------------
// shared state
// ---------------------------------------------------------------------------
std::vector<uint8_t> g_src;
int g_srcW = kNativeW, g_srcH = kNativeH;
int g_iters = 50;

// ===========================================================================
// sections
// ===========================================================================
void SecQuery() {
  Section("1. Hardware capability query (official querystring API)");
  const char* names[] = {"RGA_VENDOR",    "RGA_VERSION",     "RGA_MAX_INPUT",
                          "RGA_MAX_OUTPUT", "RGA_SCALE_LIMIT", "RGA_BYTE_STRIDE",
                          "RGA_INPUT_FORMAT", "RGA_OUTPUT_FORMAT", "RGA_FEATURE"};
  const int codes[] = {RGA_VENDOR, RGA_VERSION, RGA_MAX_INPUT, RGA_MAX_OUTPUT, RGA_SCALE_LIMIT,
                       RGA_BYTE_STRIDE, RGA_INPUT_FORMAT, RGA_OUTPUT_FORMAT, RGA_FEATURE};
  for (int i = 0; i < 9; ++i) {
    const char* s = querystring(codes[i]);
    printf("  %-18s : %s", names[i], s ? s : "(null)");
    fflush(stdout);
  }
  printf("imcheckHeader      : %s (%d)\n", StatusName(imcheckHeader()), (int)imcheckHeader());
  fflush(stdout);
}

void SecCopy() {
  Section("2. imcopy over the working buffer path");
  RgaBuf src, dst;
  if (!BufMake(&src, g_srcW, g_srcH, kFmt) || !BufMake(&dst, g_srcW, g_srcH, kFmt)) {
    printf("  alloc failed\n");
    return;
  }
  memcpy(src.data, g_src.data(), (size_t)g_srcW * g_srcH * 3);
  if (!BufImport(&src) || !BufImport(&dst)) {
    printf("  import failed\n");
    BufFree(&src);
    BufFree(&dst);
    return;
  }
  IM_STATUS st = imcopy(BufAs(src), BufAs(dst));
  bool ok = memcmp(src.data, dst.data, (size_t)g_srcW * g_srcH * 3) == 0;
  printf("  imcopy(%dx%d RGB_888) -> %s  payload=%s\n", g_srcW, g_srcH, StatusName(st),
         ok ? "IDENTICAL" : "differs");
  fflush(stdout);
  BufFree(&src);
  BufFree(&dst);
}

void SecResize() {
  Section("3. Resize matrix: does RGA downscale == OpenCV INTER_AREA?");
  struct Case { int sw, sh, dw, dh; const char* label; };
  Case cases[] = {
      {640, 512, 320, 256, "1/2"},
      {640, 512, 160, 128, "1/4  (thumb160)"},
      {640, 512, 80, 64, "1/8  (thumb80)"},
      {320, 320, 640, 640, "2x   (heatmap 320->640)"},
      {160, 128, 640, 512, "4x   (thumb160 -> native)"},
      {80, 64, 640, 512, "8x   (thumb80 -> native)"},
  };
  printf("  %-26s %-10s %-11s %-11s %-9s %s\n", "case", "status", "vs AREA", "vs BILIN", "maxdiff",
         "closer to");
  for (auto& c : cases) {
    RgaBuf src, dst;
    if (!BufMake(&src, c.sw, c.sh, kFmt) || !BufMake(&dst, c.dw, c.dh, kFmt)) continue;
    // fill the source with a slice of the test image
    for (int y = 0; y < c.sh && y < g_srcH; ++y) {
      int copy_w = std::min(c.sw, g_srcW);
      memcpy(src.data + (size_t)y * c.sw * 3, g_src.data() + (size_t)y * g_srcW * 3, (size_t)copy_w * 3);
    }
    if (!BufImport(&src) || !BufImport(&dst)) {
      printf("  %-26s import failed\n", c.label);
      BufFree(&src);
      BufFree(&dst);
      continue;
    }
    IM_STATUS st = imresize(BufAs(src), BufAs(dst), 0, 0, INTER_LINEAR);
    if (st != IM_STATUS_SUCCESS && st != IM_STATUS_NOERROR) {
      printf("  %-26s %s (%d)\n", c.label, StatusName(st), (int)st);
    } else {
      std::vector<uint8_t> out(dst.data, dst.data + dst.size);
      // The reference must run on a properly strided sub-image: g_src rows are
      // g_srcW*3 bytes apart, not sw*3.
      std::vector<uint8_t> sub((size_t)c.sw * c.sh * 3, 0);
      for (int y = 0; y < c.sh; ++y)
        memcpy(&sub[(size_t)y * c.sw * 3], &g_src[(size_t)y * g_srcW * 3], (size_t)c.sw * 3);
      std::vector<uint8_t> area = RefAreaDown(sub, c.sw, c.sh, c.dw, c.dh, 3);
      std::vector<uint8_t> bil = RefBilinear(sub, c.sw, c.sh, c.dw, c.dh, 3);
      Diff da = Compare(out, area), db = Compare(out, bil);
      printf("  %-26s %-10s %-11.4f %-11.4f %-9d %s\n", c.label, "ok", da.mean_abs, db.mean_abs,
             std::min(da.max_abs, db.max_abs), da.mean_abs <= db.mean_abs ? "INTER_AREA" : "BILINEAR");
    }
    fflush(stdout);
    BufFree(&src);
    BufFree(&dst);
  }
}

void SecInterp() {
  Section("4. Interpolation mode (nearest vs linear)");
  const int modes[] = {INTER_NEAREST, INTER_LINEAR};
  const char* mn[] = {"INTER_NEAREST", "INTER_LINEAR"};
  for (int m = 0; m < 2; ++m) {
    RgaBuf src, dst;
    if (!BufMake(&src, 640, 512, kFmt) || !BufMake(&dst, 160, 128, kFmt)) continue;
    memcpy(src.data, g_src.data(), (size_t)640 * 512 * 3);
    if (!BufImport(&src) || !BufImport(&dst)) {
      BufFree(&src);
      BufFree(&dst);
      continue;
    }
    IM_STATUS st = imresize(BufAs(src), BufAs(dst), 0, 0, modes[m]);
    std::vector<uint8_t> out(dst.data, dst.data + dst.size);
    Diff db = Compare(out, RefBilinear(g_src, 640, 512, 160, 128, 3));
    Diff da = Compare(out, RefAreaDown(g_src, 640, 512, 160, 128, 3));
    printf("  %-16s -> %-14s vs_bilinear_mean=%.4f  vs_area_mean=%.4f\n", mn[m], StatusName(st),
           db.mean_abs, da.mean_abs);
    fflush(stdout);
    BufFree(&src);
    BufFree(&dst);
  }
}

void SecCore() {
  Section("5. RGA core selection (RGA3 has a 128x128 minimum dimension)");
  for (int core = 0; core <= 2; ++core) {
    IM_STATUS cst = imconfig(IM_CONFIG_SCHEDULER_CORE, core);
    printf("  imconfig(SCHEDULER_CORE, %d) -> %s (%d)\n", core, StatusName(cst), (int)cst);
    fflush(stdout);
    struct Case { int sw, sh, dw, dh; const char* label; };
    Case cases[] = {{640, 512, 160, 128, "1/4 -> 160x128"}, {640, 512, 80, 64, "1/8 -> 80x64"}};
    for (auto& c : cases) {
      RgaBuf src, dst;
      if (!BufMake(&src, c.sw, c.sh, kFmt) || !BufMake(&dst, c.dw, c.dh, kFmt)) continue;
      memcpy(src.data, g_src.data(), (size_t)std::min(src.size, (size_t)g_src.size()));
      if (!BufImport(&src) || !BufImport(&dst)) {
        BufFree(&src);
        BufFree(&dst);
        continue;
      }
      IM_STATUS st = imresize(BufAs(src), BufAs(dst), 0, 0, INTER_LINEAR);
      bool moved = false;
      for (size_t i = 0; i < dst.size; ++i)
        if (dst.data[i] != 0) { moved = true; break; }
      printf("      %-16s -> %-14s (%d)  wrote_data=%s\n", c.label, StatusName(st), (int)st,
             moved ? "yes" : "no");
      fflush(stdout);
      BufFree(&src);
      BufFree(&dst);
    }
  }
  imconfig(IM_CONFIG_SCHEDULER_CORE, 0);
}

void SecMosaic() {
  Section("6. Mosaic canvas assembly (640x640) via improcess: RGA vs CPU reference");
  std::vector<uint8_t> ref;
  CpuBuildCanvas(g_src, &ref, kBg);

  RgaBuf feat, canvas, pat;
  if (!BufMake(&feat, kNativeW, kNativeH, kFmt) || !BufMake(&canvas, kCanvas, kCanvas, kFmt) ||
      !BufMake(&pat, 4, 4, kFmt)) {
    printf("  alloc failed\n");
    return;
  }
  memcpy(feat.data, g_src.data(), (size_t)kNativeW * kNativeH * 3);
  // The background is CONSTANT across frames -> fill it once at init, not per
  // frame.  imfill only supports RGBA, and the canvas has to stay RGB_888 for
  // the NPU input, so the one-time fill is a plain memset.
  memset(canvas.data, kBg, canvas.size);
  if (!BufImport(&feat) || !BufImport(&canvas) || !BufImport(&pat)) {
    printf("  import failed\n");
    return;
  }

  rga_buffer_t patb = BufAs(pat);
  // 1) native 1:1 into the canvas
  IM_STATUS st1 = improcess(BufAs(feat), BufAs(canvas), patb,
                            MakeRect(0, 0, kNativeW, kNativeH), MakeRect(0, 0, kNativeW, kNativeH),
                            MakeRect(0, 0, 0, 0), 0);
  printf("  improcess native 1:1      -> %s (%d)\n", StatusName(st1), (int)st1);
  fflush(stdout);
  // 2) 1/4 thumbnail
  IM_STATUS st2 = improcess(BufAs(feat), BufAs(canvas), patb,
                            MakeRect(0, 0, kNativeW, kNativeH),
                            MakeRect(kT160Rect.x, kT160Rect.y, kT160Rect.w, kT160Rect.h),
                            MakeRect(0, 0, 0, 0), 0);
  printf("  improcess thumb160 (1/4)   -> %s (%d)\n", StatusName(st2), (int)st2);
  fflush(stdout);
  // 3) 1/8 thumbnail
  IM_STATUS st3 = improcess(BufAs(feat), BufAs(canvas), patb,
                            MakeRect(0, 0, kNativeW, kNativeH),
                            MakeRect(kT80Rect.x, kT80Rect.y, kT80Rect.w, kT80Rect.h),
                            MakeRect(0, 0, 0, 0), 0);
  printf("  improcess thumb80  (1/8)   -> %s (%d)\n", StatusName(st3), (int)st3);
  fflush(stdout);

  std::vector<uint8_t> got(canvas.data, canvas.data + canvas.size);
  auto region_diff = [&](Rect r) {
    std::vector<uint8_t> a, b;
    for (int y = 0; y < r.h; ++y) {
      const uint8_t* pa = &got[((size_t)(r.y + y) * kCanvas + r.x) * 3];
      const uint8_t* pb = &ref[((size_t)(r.y + y) * kCanvas + r.x) * 3];
      a.insert(a.end(), pa, pa + (size_t)r.w * 3);
      b.insert(b.end(), pb, pb + (size_t)r.w * 3);
    }
    return Compare(a, b);
  };
  Diff dn = region_diff(kNativeRect), d1 = region_diff(kT160Rect), d8 = region_diff(kT80Rect);
  printf("  native  region vs CPU ref: max=%d mean=%.4f  %s\n", dn.max_abs, dn.mean_abs,
         dn.max_abs == 0 ? "(byte-identical)" : "(DIFFERS)");
  printf("  thumb160 region vs CPU ref: max=%d mean=%.4f\n", d1.max_abs, d1.mean_abs);
  printf("  thumb80  region vs CPU ref: max=%d mean=%.4f\n", d8.max_abs, d8.mean_abs);
  fflush(stdout);

  BufFree(&feat);
  BufFree(&canvas);
  BufFree(&pat);
}

void SecTiming() {
  Section("7. Timing: RGA vs CPU reference");
  RgaBuf feat, dst;
  if (!BufMake(&feat, 640, 512, kFmt) || !BufMake(&dst, 160, 128, kFmt)) {
    printf("  alloc failed\n");
    return;
  }
  memcpy(feat.data, g_src.data(), (size_t)640 * 512 * 3);
  BufImport(&feat);
  BufImport(&dst);

  {
    Timing t = TimeIt(g_iters, [&] { imresize(BufAs(feat), BufAs(dst), 0, 0, INTER_LINEAR); });
    printf("  RGA  imresize 640x512 -> 160x128   : avg %.3f ms  min %.3f ms\n", t.avg_ms, t.min_ms);
  }
  {
    Timing t = TimeIt(g_iters, [&] {
      std::vector<uint8_t> v = RefAreaDown(g_src, 640, 512, 160, 128, 3);
      (void)v;
    });
    printf("  CPU  area     640x512 -> 160x128   : avg %.3f ms  min %.3f ms (incl. alloc)\n",
           t.avg_ms, t.min_ms);
  }
  {
    RgaBuf s8, d8;
    BufMake(&s8, 160, 128, kFmt);
    BufMake(&d8, 640, 512, kFmt);
    memcpy(s8.data, dst.data, (size_t)160 * 128 * 3);
    BufImport(&s8);
    BufImport(&d8);
    Timing t = TimeIt(g_iters, [&] { imresize(BufAs(s8), BufAs(d8), 0, 0, INTER_LINEAR); });
    printf("  RGA  imresize 160x128 -> 640x512   : avg %.3f ms  min %.3f ms\n", t.avg_ms, t.min_ms);
    BufFree(&s8);
    BufFree(&d8);
  }
  {
    Timing t = TimeIt(g_iters, [&] {
      std::vector<uint8_t> v = RefBilinear(g_src, 160, 128, 640, 512, 3);
      (void)v;
    });
    printf("  CPU  bilinear 160x128 -> 640x512   : avg %.3f ms  min %.3f ms (incl. alloc)\n",
           t.avg_ms, t.min_ms);
  }
  // The import cost matters for the per-frame path: it has to be re-done for
  // every frame whose content changed.
  {
    Timing t = TimeIt(g_iters, [&] { BufImport(&feat); });
    printf("  RGA  importbuffer_virtualaddr     : avg %.3f ms  min %.3f ms\n", t.avg_ms, t.min_ms);
  }
  {
    Timing t = TimeIt(g_iters, [&] {
      std::vector<uint8_t> c;
      CpuBuildCanvas(g_src, &c, kBg);
      (void)c;
    });
    printf("  CPU  full mosaic assembly          : avg %.3f ms  min %.3f ms (incl. alloc)\n",
           t.avg_ms, t.min_ms);
  }
  {
    // RGA full mosaic assembly: fill + native blit + 2 resize + 2 translate
    RgaBuf canvas, t160, t80;
    BufMake(&canvas, kCanvas, kCanvas, kFmt);
    BufMake(&t160, kT160Rect.w, kT160Rect.h, kFmt);
    BufMake(&t80, kT80Rect.w, kT80Rect.h, kFmt);
    BufImport(&canvas);
    BufImport(&t160);
    BufImport(&t80);
    Timing t = TimeIt(g_iters, [&] {
      memset(canvas.data, kBg, canvas.size);
      imfill(BufAs(canvas), MakeRect(0, 0, kCanvas, kCanvas), 0);
      imtranslate(BufAs(feat), BufAs(canvas), 0, 0);
      imresize(BufAs(feat), BufAs(t160), 0, 0, INTER_LINEAR);
      imtranslate(BufAs(t160), BufAs(canvas), 0, 512);
      imresize(BufAs(feat), BufAs(t80), 0, 0, INTER_LINEAR);
      imtranslate(BufAs(t80), BufAs(canvas), 160, 512);
    });
    printf("  RGA  full mosaic assembly          : avg %.3f ms  min %.3f ms\n", t.avg_ms, t.min_ms);
    BufFree(&canvas);
    BufFree(&t160);
    BufFree(&t80);
  }
  BufFree(&feat);
  BufFree(&dst);
}

void SecNotes() {
  Section("8. Static facts from the SDK developer guide");
  printf("  RK3588 (Orion) = RGA2-Enhance + RGA3\n");
  printf("    RGA2-Enhance : src 2x2..8192x8192  dst 2x2..4096x4096  scale 1/16..16\n");
  printf("    RGA3         : src 128x128..8176x8176 dst 128x128..8128x8128 scale 1/8..8\n");
  printf("  strictest alignment on RK3588:\n");
  printf("    RGBA8888 -> width stride 4-aligned\n");
  printf("    RGB888   -> width stride 16-aligned\n");
  printf("    RGB565   -> width stride 8-aligned\n");
  printf("  All RGA formats are 8-bit fixed point: there is NO float32 format, so\n");
  printf("  the float32 heatmap reprojection cannot use RGA without a lossy\n");
  printf("  quantise step (0..255 over a 0..1 heatmap = 1/255 quantisation).\n");
}

struct SectionDef {
  const char* name;
  void (*fn)();
  const char* title;
};

const SectionDef kSections[] = {
    {"query", SecQuery, "hardware capability query"},
    {"copy", SecCopy, "imcopy"},
    {"resize", SecResize, "resize matrix"},
    {"interp", SecInterp, "interpolation"},
    {"core", SecCore, "RGA core selection"},
    {"mosaic", SecMosaic, "mosaic assembly"},
    {"timing", SecTiming, "timing"},
    {"notes", SecNotes, "static facts"},
};
const int kNumSections = (int)(sizeof(kSections) / sizeof(kSections[0]));

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);  // line buffered: partial output survives a crash

  bool run_all = true;
  bool list = false;
  std::vector<std::string> wanted;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--iters") && i + 1 < argc) g_iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--only") && i + 1 < argc) {
      run_all = false;
      std::string csv = argv[++i];
      size_t p = 0;
      while (p < csv.size()) {
        size_t q = csv.find(',', p);
        if (q == std::string::npos) q = csv.size();
        wanted.push_back(csv.substr(p, q - p));
        p = q + 1;
      }
    } else if (!strcmp(argv[i], "--list")) {
      list = true;
    } else if (!strcmp(argv[i], "--help")) {
      printf("usage: rga_probe [--iters N] [--only s1,s2] [--list]\n");
      for (int i = 0; i < kNumSections; ++i)
        printf("  section: %-8s %s\n", kSections[i].name, kSections[i].title);
      return 0;
    }
  }
  if (g_iters < 1) g_iters = 1;

  if (list) {
    for (int i = 0; i < kNumSections; ++i)
      printf("%-8s %s\n", kSections[i].name, kSections[i].title);
    return 0;
  }

  printf("rga_probe: RK3588 RGA (im2d v2) capability probe\n");
  printf("  compiled api version : %s\n", RGA_API_VERSION);
  printf("  iterations           : %d\n", g_iters);
  fflush(stdout);

  FillSynthetic(&g_src, g_srcW, g_srcH);
  printf("  source               : synthetic gradient+blobs (%dx%d)\n", g_srcW, g_srcH);
  fflush(stdout);

  for (int i = 0; i < kNumSections; ++i) {
    bool run = run_all;
    for (auto& w : wanted)
      if (w == kSections[i].name) run = true;
    if (!run) continue;
    kSections[i].fn();
    fflush(stdout);
  }

  Section("probe finished");
  return 0;
}
