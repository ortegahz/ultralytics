// ===========================================================================
// rknn_mosaic_heatmap.cpp -- RK3588 multi-scale ("mosaic") heatmap inference.
//
// ONE rknn_run per frame: the SAME feature frame is packed into three scales
// inside a single 640x640 canvas, the NPU emits one 320x320 heatmap, and the
// back-projection + connected-component detection runs on the CPU in float32.
//
//   canvas (640x640, RGB888, background 114)
//     +-------------------------------+
//     | native 640x512 @ (0,0)        |  tag 0, 1:1
//     |                               |
//     +-------+-------+---------------+  y = 512
//     | t160  | t80   |  background   |  tag 1 = 160x128 @ (0,512), 1/4 down
//     | 160x128 @ (0,512)             |  tag 2 = 80x64   @ (160,512), 1/8 down
//     +-------+-------+---------------+
//
// The three tiles are produced by THREE RGA `improcess` calls from the same
// 640x512 source buffer -- no CPU image copy per frame (the stb_image buffer
// is imported into RGA directly).
//
// THE PIXEL CONTRACT (do not "fix" this, it is already right):
//   the BMP on disk stores BGR (cv2.imwrite(cv2.imread(src))), and stb_image's
//   stbi_load returns RGB, whose byte order is EXACTLY the PC side's
//   model_order_bmp() == cv2.imread(bmp)[..., ::-1].  So the bytes stb hands
//   over are already in the model's channel order
//   [median residual, GMC diff, I_t] -- the C++ side must NOT flip anything.
//
// KNOWN CALIBER DIFFERENCE (documented, not hidden):
//   the PC side's thumb canvases have a background of "disk-order ch0 (==I_t)
//   filled with the mean of resized ch0, ch1/ch2 == 0", i.e. in model order
//   (residual=0, diff=0, I_t=mean).  THIS layout uses a flat 114 background
//   for the whole canvas.  The native tile therefore sees a 114 background on
//   the PC side and on the board alike (the PC letterbox pad is 114), but the
//   two thumbnail tiles do NOT have the same background.  That is a real
//   caliber difference between the two arms and must not be reported as if
//   the two were identical.
//
// THE "single" ABLATION ARM (--canvas-mode single):
//   a strict replica of the PC side's native-scale letterbox: the 640x512
//   frame at 1:1 into canvas (0,64), everything else padded with 114.  It is
//   the baseline that separates "fp16 / board quantisation loss" from
//   "mosaic packing loss".
//   PHASE NOTE (MEASURED against cv2 4.10 on the host, not assumed): the PC
//   side enlarges 320->640 FIRST and crops canvas rows 64..575 afterwards,
//   while this program crops the 320-cell rows [32,288) FIRST and enlarges
//   2x afterwards.  The two sample the SAME positions -- the half-pixel
//   mapping is identical -- and they differ only in the edge clamp at the
//   sub-rectangle border, i.e. in the FIRST and LAST output row (2 of 512):
//   max |delta| ~2.7e-2 heatmap units, mean ~1.7e-5.  So this arm is close
//   to, but NOT claimed to be bit-identical to, the PC side.
//
// OUTPUT (TSV, one line per frame, contract with the Python side):
//   <stem>\t<ndet>\t<tag> <cx> <cy> <w> <h> <area> <peak> <sum>\t...
//   cx/cy are NATIVE 640x512 pixel coordinates (not canvas, not letterbox).
//
// LOGGING: the board's librga is built WITH debug logging and prints an
// "im2d_rga_impl rga_dump_opt(...)" block to STDOUT on every call, so this
// program writes EVERY log line to stderr and every result to a FILE, and
// detaches stdout at startup.
//
// BUILD: ./build_rknn_mosaic_arm64.sh
// RUN:   rknn_mosaic_heatmap --model <m.rknn> --list <list.txt> --out <dir>
// ===========================================================================
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sched.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "rknn_api.h"

#define STB_IMAGE_IMPLEMENTATION
// The staged third-party stb_image.h trips -Wtype-limits on an internal
// STBI_ASSERT ("unsigned >= 0").  That warning comes from the header, not
// from this program, so it is suppressed for that header ONLY -- the rest of
// the translation unit stays at -Wall -Wextra with zero warnings.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "stb_image.h"
#pragma GCC diagnostic pop

#include "im2d.h"
#include "RgaUtils.h"

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define MOSAIC_HAVE_NEON 1
#endif

namespace {

// ---------------------------------------------------------------------------
// Geometry / detection constants (contracted or measured, none guessed)
// ---------------------------------------------------------------------------
constexpr int kCanvas = 640;          // NPU canvas edge
constexpr int kSrcW = 640;            // feature frame width  (native pixels)
constexpr int kSrcH = 512;            // feature frame height (native pixels)
constexpr int kSrcC = 3;              // feature frame channels
constexpr uint8_t kPad = 114;         // the PC letterbox pad value (3 channels)
constexpr int kRgaFmt = RK_FORMAT_RGB_888;

constexpr double kRegionThreshold = 0.06;  // PC side --region-threshold
constexpr int kMinArea = 4;                // PC side --min-area
constexpr int kMaxDet = 100;               // PC side --max-detections
constexpr int kDilate = 1;                 // PC side per-scale rows are NOT dilated
static_assert(kDilate == 1, "the PC side's per-scale rows are undilated; a dilate > 1 would "
                             "change the area/sum semantics");

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------
double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

const char* tensor_type_name(rknn_tensor_type t) {
  switch (t) {
    case RKNN_TENSOR_FLOAT32: return "FLOAT32";
    case RKNN_TENSOR_FLOAT16: return "FLOAT16";
    case RKNN_TENSOR_INT8:    return "INT8";
    case RKNN_TENSOR_UINT8:   return "UINT8";
    case RKNN_TENSOR_INT16:   return "INT16";
    default:                  return "OTHER";
  }
}

const char* tensor_fmt_name(rknn_tensor_format f) {
  switch (f) {
    case RKNN_TENSOR_NCHW: return "NCHW";
    case RKNN_TENSOR_NHWC: return "NHWC";
    default:               return "OTHER";
  }
}

const char* rga_status_name(IM_STATUS s) {
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

bool read_file(const char* path, std::vector<uint8_t>* out) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  const long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  out->resize((size_t)(n > 0 ? n : 0));
  const size_t got = out->empty() ? 0 : fread(out->data(), 1, out->size(), f);
  fclose(f);
  return got == out->size();
}

// ---------------------------------------------------------------------------
// fp16 helpers -- COPIED VERBATIM from the certified reference
// (manu/pipeline/npu/rknn_heatmap_smoke.cpp, lines 135-338) so that this
// program converts bytes exactly like the single-scale board arm does.
// They are duplicated rather than #included because that file's helpers sit
// in an anonymous namespace and several of them are unused here, which would
// trip -Wunused-function (this build must be warning free).
// ---------------------------------------------------------------------------
uint16_t f32_to_f16_bits(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
  uint32_t man = x & 0x7FFFFFu;
  if (((x >> 23) & 0xFF) == 0xFF) {
    return (uint16_t)(sign | 0x7C00u | (man ? (0x200u | (man >> 13)) : 0u));
  }
  if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
  if (exp <= 0) {
    if (exp < -10) return (uint16_t)sign;
    man |= 0x800000u;
    const uint32_t shift = (uint32_t)(14 - exp);
    uint32_t half = man >> shift;
    if ((man >> (shift - 1)) & 1u) {
      if ((man & ((1u << (shift - 1)) - 1u)) || (half & 1u)) half++;
    }
    return (uint16_t)(sign | half);
  }
  uint32_t half = (uint32_t)(exp << 10) | (man >> 13);
  if ((man >> 12) & 1u) {
    if ((man & 0xFFFu) || (half & 1u)) half++;
  }
  return (uint16_t)(sign | half);
}

#ifdef MOSAIC_HAVE_NEON
inline uint32x4_t fp16_bits_from_u32(uint32x4_t v, uint32x4_t bias, uint32x4_t zero) {
  const uint32x4_t f = vreinterpretq_u32_f32(vcvtq_f32_u32(v));
  const uint32x4_t h = vsubq_u32(vshrq_n_u32(f, 13), bias);
  const uint32x4_t isz = vceqq_u32(v, zero);
  return vbicq_u32(h, isz);
}
#endif

// FAST PATH for the integer case (div == 1, i.e. the tensor holds the raw
// byte values -- which is what the baseline rknn_inputs_set path feeds the
// model).  Every integer 0..255 is exactly representable in fp16, and its
// bit pattern has a closed form:
//
//     fp16(n) = (f32_bits(n) >> 13) - 0x1C000     for n > 0
//     fp16(0) = 0
//
// which turns the conversion into ~7 integer NEON ops per 4 elements and
// needs no FP16 ISA (the RK3588's Cortex-A55 does not implement FP16).
void u8_to_f16_int(const uint8_t* src, uint16_t* dst, size_t n) {
#ifdef MOSAIC_HAVE_NEON
  const uint32x4_t bias = vdupq_n_u32(0x1C000u);
  const uint32x4_t zero = vdupq_n_u32(0);
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const uint8x16_t b = vld1q_u8(src + i);
    const uint16x8_t h = vmovl_u8(vget_low_u8(b));
    const uint16x8_t h2 = vmovl_u8(vget_high_u8(b));
    const uint32x4_t v0 = vmovl_u16(vget_low_u16(h));
    const uint32x4_t v1 = vmovl_u16(vget_high_u16(h));
    const uint32x4_t v2 = vmovl_u16(vget_low_u16(h2));
    const uint32x4_t v3 = vmovl_u16(vget_high_u16(h2));
    const uint32x4_t o0 = fp16_bits_from_u32(v0, bias, zero);
    const uint32x4_t o1 = fp16_bits_from_u32(v1, bias, zero);
    const uint32x4_t o2 = fp16_bits_from_u32(v2, bias, zero);
    const uint32x4_t o3 = fp16_bits_from_u32(v3, bias, zero);
    vst1_u16(dst + i, vmovn_u32(o0));
    vst1_u16(dst + i + 4, vmovn_u32(o1));
    vst1_u16(dst + i + 8, vmovn_u32(o2));
    vst1_u16(dst + i + 12, vmovn_u32(o3));
  }
  for (; i < n; i++) {
    const float f = (float)src[i];
    uint32_t b;
    memcpy(&b, &f, 4);
    const uint32_t o = src[i] == 0 ? 0u : ((b >> 13) - 0x1C000u);
    dst[i] = (uint16_t)o;
  }
#else
  for (size_t i = 0; i < n; i++) {
    const float f = (float)src[i];
    uint32_t b;
    memcpy(&b, &f, 4);
    dst[i] = (uint16_t)(src[i] == 0 ? 0u : ((b >> 13) - 0x1C000u));
  }
#endif
}

// value = u8 / div, written as fp16.  The obvious intrinsic
// (vcvtq_f16_f32) needs the optional FP16 extension, which the RK3588's
// Cortex-A55 does NOT implement, so the f32->f16 step is integer NEON.
void u8_to_f16(const uint8_t* src, uint16_t* dst, size_t n, float div) {
  if (div == 1.0f) { u8_to_f16_int(src, dst, n); return; }
#ifdef MOSAIC_HAVE_NEON
  const float inv = 1.0f / div;
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const uint16x8_t u16 = vmovl_u8(vld1_u8(src + i));
    const uint32x4_t u32lo = vmovl_u16(vget_low_u16(u16));
    const uint32x4_t u32hi = vmovl_u16(vget_high_u16(u16));
    const float32x4_t flo = vmulq_n_f32(vcvtq_f32_u32(u32lo), inv);
    const float32x4_t fhi = vmulq_n_f32(vcvtq_f32_u32(u32hi), inv);
    const uint32x4_t xlo = vreinterpretq_u32_f32(flo);
    const uint32x4_t xhi = vreinterpretq_u32_f32(fhi);
    const uint32x4_t signlo = vshlq_n_u32(vandq_u32(xlo, vdupq_n_u32(0x80000000u)), 16);
    const uint32x4_t signhi = vshlq_n_u32(vandq_u32(xhi, vdupq_n_u32(0x80000000u)), 16);
    const uint32x4_t elo = vsubq_u32(vshrq_n_u32(xlo, 23), vdupq_n_u32(112));
    const uint32x4_t ehi = vsubq_u32(vshrq_n_u32(xhi, 23), vdupq_n_u32(112));
    const uint32x4_t mlo = vorrq_u32(vandq_u32(xlo, vdupq_n_u32(0x7FFFFFu)), vdupq_n_u32(0x800000u));
    const uint32x4_t mhi = vorrq_u32(vandq_u32(xhi, vdupq_n_u32(0x7FFFFFu)), vdupq_n_u32(0x800000u));
    const uint32x4_t mslo = vandq_u32(vshrq_n_u32(mlo, 13), vdupq_n_u32(0x3FFu));
    const uint32x4_t mshi = vandq_u32(vshrq_n_u32(mhi, 13), vdupq_n_u32(0x3FFu));
    const uint32x4_t remlo = vandq_u32(mlo, vdupq_n_u32(0x1FFFu));
    const uint32x4_t remhi = vandq_u32(mhi, vdupq_n_u32(0x1FFFu));
    const uint32x4_t gtlo = vcgtq_u32(remlo, vdupq_n_u32(0x1000u));
    const uint32x4_t gthi = vcgtq_u32(remhi, vdupq_n_u32(0x1000u));
    const uint32x4_t eqlo = vceqq_u32(remlo, vdupq_n_u32(0x1000u));
    const uint32x4_t eqhi = vceqq_u32(remhi, vdupq_n_u32(0x1000u));
    const uint32x4_t inclo = vandq_u32(vorrq_u32(gtlo, vandq_u32(eqlo, vandq_u32(mslo, vdupq_n_u32(1u)))),
                                      vdupq_n_u32(1u));
    const uint32x4_t inchi = vandq_u32(vorrq_u32(gthi, vandq_u32(eqhi, vandq_u32(mshi, vdupq_n_u32(1u)))),
                                      vdupq_n_u32(1u));
    const uint32x4_t hlo = vaddq_u32(vshlq_n_u32(elo, 10), vaddq_u32(mslo, inclo));
    const uint32x4_t hhi = vaddq_u32(vshlq_n_u32(ehi, 10), vaddq_u32(mshi, inchi));
    const uint32x4_t zlo = vceqq_u32(xlo, vdupq_n_u32(0));
    const uint32x4_t zhi = vceqq_u32(xhi, vdupq_n_u32(0));
    const uint32x4_t rlo = vorrq_u32(signlo, vbicq_u32(hlo, zlo));
    const uint32x4_t rhi = vorrq_u32(signhi, vbicq_u32(hhi, zhi));
    vst1_u16(dst + i, vmovn_u32(rlo));
    vst1_u16(dst + i + 4, vmovn_u32(rhi));
  }
  for (; i < n; i++) dst[i] = f32_to_f16_bits((float)src[i] * (1.0f / div));
#else
  const float inv = 1.0f / div;
  for (size_t i = 0; i < n; i++) dst[i] = f32_to_f16_bits((float)src[i] * inv);
#endif
}

float fp16_to_f32(uint16_t h) {
  const uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f;
  uint32_t man = h & 0x3ff;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) { bits = sign << 31; }
    else {
      int e = -1;
      do { e++; man <<= 1; } while (!(man & 0x400));
      man &= 0x3ff;
      bits = (sign << 31) | ((uint32_t)(127 - 15 - e) << 23) | (man << 13);
    }
  } else if (exp == 0x1f) {
    bits = (sign << 31) | 0x7f800000u | (man << 13);
  } else {
    bits = (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13);
  }
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

// Transpose an NHWC canvas (H*W*3, interleaved) into NCHW (3*H*W, planar).
void nhwc_to_nchw_f16(const uint16_t* src, uint16_t* dst, int hw, int c) {
  for (int ch = 0; ch < c; ch++)
    for (int y = 0; y < hw; y++)
      for (int x = 0; x < hw; x++)
        dst[(size_t)ch * hw * hw + (size_t)y * hw + x] = src[((size_t)y * hw + x) * c + ch];
}

// Dequantize one whole output tensor into float32, whatever the runtime hands
// back.  (COPIED from rknn_heatmap_smoke.cpp:364.)
bool dequant(const rknn_output& o, const rknn_tensor_attr& a, std::vector<float>* dst) {
  size_t n = 0;
  if (a.n_dims > 0) {
    n = 1;
    for (uint32_t d = 0; d < a.n_dims; d++) n *= (size_t)a.dims[d];
  }
  if (n == 0) n = o.size / 4;
  dst->resize(n);
  const float sc = a.scale;
  const int zp = a.zp;
  switch (a.type) {
    case RKNN_TENSOR_FLOAT32:
      memcpy(dst->data(), o.buf, n * 4);
      return true;
    case RKNN_TENSOR_FLOAT16: {
      const uint16_t* p = static_cast<const uint16_t*>(o.buf);
      for (size_t i = 0; i < n; i++) (*dst)[i] = fp16_to_f32(p[i]);
      return true;
    }
    case RKNN_TENSOR_INT8: {
      const int8_t* p = static_cast<const int8_t*>(o.buf);
      for (size_t i = 0; i < n; i++) (*dst)[i] = ((float)p[i] - (float)zp) * sc;
      return true;
    }
    case RKNN_TENSOR_UINT8: {
      const uint8_t* p = static_cast<const uint8_t*>(o.buf);
      for (size_t i = 0; i < n; i++) (*dst)[i] = ((float)p[i] - (float)zp) * sc;
      return true;
    }
    case RKNN_TENSOR_INT16: {
      const int16_t* p = static_cast<const int16_t*>(o.buf);
      for (size_t i = 0; i < n; i++) (*dst)[i] = ((float)p[i] - (float)zp) * sc;
      return true;
    }
    default:
      return false;
  }
}

// ---------------------------------------------------------------------------
// RGA buffer: exact-size allocation + import.
//
// MEASURED ON THE BOARD (manu/pipeline/npu/RGA_RK3588_NOTES.md):
//   * importbuffer_virtualaddr() -> wrapbuffer_handle() is the ONLY path that
//     actually moves data (a raw dma-buf fd through wrapbuffer_fd_t() returns
//     SUCCESS and writes nothing).
//   * the buffer CONTENT is captured at import time => fill BEFORE import.
//   * the buffer size must be EXACTLY w*h*bpp (over-allocating turns the
//     operation into a no-op).
//   * imcheck() must NOT be used on this librga: it validates a zeroed pat
//     buffer, prints "537:src has not fd and address for render" and makes
//     the operation that follows silently do nothing.  Byte budgets are
//     checked explicitly instead (CheckBudget).
//   * imfill() only supports RGBA, so the RGB_888 canvas background is a
//     plain memset (the background is frame-invariant, so it is filled ONCE).
//   * imtranslate() requires src/dst of equal size, so the tiles have to be
//     placed by improcess() (scale + place in one call).
//   * RGB_888 requires the WIDTH to be 16-aligned: 640 / 160 / 80 all are.
// ---------------------------------------------------------------------------
struct RgaBuf {
  uint8_t* data = nullptr;
  size_t size = 0;
  int w = 0, h = 0, fmt = 0;
  rga_buffer_handle_t handle = -1;
};

im_rect MakeRect(int x, int y, int w, int h) {
  im_rect r;
  r.x = x;
  r.y = y;
  r.width = w;
  r.height = h;
  return r;
}

// Allocate an exactly-sized, page-aligned buffer.  Over-allocating makes the
// RGA operation a no-op, so the size is exactly w*h*bpp.
bool BufAlloc(RgaBuf* b, int w, int h, int fmt) {
  const size_t bytes = (size_t)w * (size_t)h * (size_t)get_bpp_from_format(fmt);
  void* p = nullptr;
  if (posix_memalign(&p, 4096, bytes) != 0 || p == nullptr) return false;
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

// Import AFTER the content is final.
bool BufImport(RgaBuf* b) {
  if (b->handle > 0) releasebuffer_handle(b->handle);
  b->handle = importbuffer_virtualaddr(b->data, b->w, b->h, b->fmt);
  return b->handle > 0;
}

rga_buffer_t BufAs(const RgaBuf& b) { return wrapbuffer_handle(b.handle, b.w, b.h, b.fmt); }

// The explicit byte-budget check that replaces imcheck().
bool CheckBudget(const RgaBuf& b, const char* who) {
  const size_t need = (size_t)b.w * (size_t)b.h * (size_t)get_bpp_from_format(b.fmt);
  if (need > b.size) {
    fprintf(stderr, "[FATAL] RGA byte budget violated for %s: need %zu B, have %zu B\n", who, need,
            b.size);
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Tile layout
// ---------------------------------------------------------------------------
struct Tile {
  int tag = 0;        // 0 = native, 1 = thumb160, 2 = thumb80
  int x = 0, y = 0, w = 0, h = 0;   // canvas rectangle
  bool enabled = true;
};

// The heatmap sub-rectangle that belongs to a tile.
//
//   x_hm = x_canvas * hm_w / 640     (== x_canvas / 2 when hm_w == 320)
//
//   mosaic: native  -> rows [0,256)  cols [0,320)
//            thumb160-> rows [256,320) cols [0,80)
//            thumb80 -> rows [256,288) cols [80,120)
//   single: native  -> rows [32,288)  cols [0,320)
void HeatRect(const Tile& t, int hm_w, int hm_h, int* rx0, int* ry0, int* rx1, int* ry1) {
  *rx0 = (int)((long long)t.x * hm_w / kCanvas);
  *ry0 = (int)((long long)t.y * hm_h / kCanvas);
  *rx1 = (int)(((long long)(t.x + t.w) * hm_w + kCanvas - 1) / kCanvas);
  *ry1 = (int)(((long long)(t.y + t.h) * hm_h + kCanvas - 1) / kCanvas);
  if (*rx1 > hm_w) *rx1 = hm_w;
  if (*ry1 > hm_h) *ry1 = hm_h;
  if (*rx1 <= *rx0) *rx0 = (*rx1 > 0) ? *rx1 - 1 : 0;
  if (*ry1 <= *ry0) *ry0 = (*ry1 > 0) ? *ry1 - 1 : 0;
}

// ---------------------------------------------------------------------------
// Back-projection: float32 bilinear resize with half-pixel centres.
//
// This is cv2.resize's INTER_LINEAR semantics on a FLOAT32 image:
//     fx = (x + 0.5) * (src_w / dst_w) - 0.5
//     sx = floor(fx), alpha = fx - sx, edges clamped
// The formula is the same one rknn_heatmap_smoke.cpp::letterbox() uses for
// the uint8 canvas, but without the uint8 rounding step (the heatmap is
// float32 and must not be quantised).
//
// src_stride is the number of floats between two source ROWS.  It is NOT
// sw: a tile's heatmap sub-rectangle is a WINDOW of the 320-wide heatmap,
// so its rows are still 320 floats apart.  Reading it with a packed stride
// would silently walk into the neighbouring tile's cells.
// ---------------------------------------------------------------------------
void ResizeBilinearF32(const float* src, int src_stride, int sw, int sh, float* dst, int dw,
                       int dh) {
  const double sx_scale = (double)sw / (double)dw;
  const double sy_scale = (double)sh / (double)dh;
  for (int y = 0; y < dh; y++) {
    const double fy = (y + 0.5) * sy_scale - 0.5;
    const int sy = (int)std::floor(fy);
    const float ay = (float)(fy - sy);
    const int y0 = std::max(0, std::min(sh - 1, sy));
    const int y1 = std::max(0, std::min(sh - 1, sy + 1));
    const float* r0 = src + (size_t)y0 * src_stride;
    const float* r1 = src + (size_t)y1 * src_stride;
    float* drow = dst + (size_t)y * dw;
    for (int x = 0; x < dw; x++) {
      const double fx = (x + 0.5) * sx_scale - 0.5;
      const int sx = (int)std::floor(fx);
      const float ax = (float)(fx - sx);
      const int x0 = std::max(0, std::min(sw - 1, sx));
      const int x1 = std::max(0, std::min(sw - 1, sx + 1));
      const float top = r0[x0] * (1.0f - ax) + r0[x1] * ax;
      const float bot = r1[x0] * (1.0f - ax) + r1[x1] * ax;
      drow[x] = top * (1.0f - ay) + bot * ay;
    }
  }
}

// ---------------------------------------------------------------------------
// Connected components -- a faithful port of the PC side's
// manu/data/multiscale_heatmap_cache.py:205 detect_components()
//
//   mask   = (heatmap >= threshold)          [>=, not >]
//   dilate = 1 -> NO dilation
//   cv2.connectedComponentsWithStats(mask, connectivity=8)
//   area  = pixel count of the component
//   w, h  = bounding box extent (xmax-xmin+1 / ymax-ymin+1)
//   cx,cy = arithmetic mean of ALL component pixel coordinates
//   peak  = max of the ORIGINAL heatmap values inside the component
//   sum   = sum of the ORIGINAL heatmap values inside the component
//   sort  = by sum DESCENDING (stable), truncated to `limit`
//
// The label ids are numbered in the order of FIRST RASTER OCCURRENCE, which
// is what cv2 does after its union-find finalisation.  That matters only for
// ties in `sum`, where the Python side keeps the insertion (label) order
// because list.sort() is stable.
// ---------------------------------------------------------------------------
struct Comp {
  int tag = 0;
  float cx = 0, cy = 0;
  int w = 0, h = 0;
  int area = 0;
  float peak = 0;
  float sum = 0;
};

int UfFind(std::vector<int>* parent, int x) {
  int r = x;
  while ((*parent)[(size_t)x] != x) x = (*parent)[(size_t)x];
  r = x;
  while ((*parent)[(size_t)x] != x) {
    const int nx = (*parent)[(size_t)x];
    (*parent)[(size_t)x] = r;
    x = nx;
  }
  return r;
}

// The root is always the SMALLEST provisional label of the class, and the
// provisional labels are handed out in raster order of first occurrence, so
// the roots sorted ascending == the cv2 label order.
void UfUnite(std::vector<int>* parent, int a, int b) {
  a = UfFind(parent, a);
  b = UfFind(parent, b);
  if (a == b) return;
  if (a < b) (*parent)[(size_t)b] = a;
  else (*parent)[(size_t)a] = b;
}

struct CcWorkspace {
  std::vector<int32_t> labels;      // W*H
  std::vector<int> parent;          // union-find, 1-based (index 0 unused)
  std::vector<int> root_to_final;
};

void DetectComponents(CcWorkspace* ws, const float* hm, int W, int H, double threshold,
                      int min_area, int limit, std::vector<Comp>* out) {
  out->clear();
  const size_t n = (size_t)W * H;
  ws->labels.assign(n, 0);
  ws->parent.assign(1, 0);
  int nlab = 0;

  // ---- pass 1: provisional labelling (8-connectivity, raster order) --------
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      const size_t idx = (size_t)y * W + x;
      // numpy compares float32 against the python float 0.06 in float64, so
      // the comparison is done in double here as well.
      if (!((double)hm[idx] >= threshold)) { ws->labels[idx] = 0; continue; }
      int nb[4];
      int nn = 0;
      if (x > 0) nb[nn++] = (int)(idx - 1);              // left
      if (y > 0) {
        nb[nn++] = (int)(idx - W);                      // up
        if (x > 0) nb[nn++] = (int)(idx - W - 1);       // up-left
        if (x + 1 < W) nb[nn++] = (int)(idx - W + 1);   // up-right
      }
      int lbl = 0;
      for (int i = 0; i < nn; i++) {
        const int l = ws->labels[(size_t)nb[i]];
        if (l == 0) continue;
        if (lbl == 0 || l < lbl) lbl = l;
      }
      if (lbl == 0) {
        lbl = ++nlab;
        ws->parent.push_back(lbl);  // parent[lbl] == lbl
      }
      ws->labels[idx] = lbl;
      for (int i = 0; i < nn; i++) {
        const int l = ws->labels[(size_t)nb[i]];
        if (l != 0 && l != lbl) UfUnite(&ws->parent, lbl, l);
      }
    }
  }
  if (nlab == 0) return;

  // ---- final label ids in first-raster-occurrence order --------------------
  ws->root_to_final.assign((size_t)nlab + 1, 0);
  int ncomp = 0;
  for (int l = 1; l <= nlab; l++) {
    if (UfFind(&ws->parent, l) == l) ws->root_to_final[(size_t)l] = ++ncomp;
  }

  // ---- pass 2: per-component statistics -----------------------------------
  std::vector<int> cnt((size_t)ncomp + 1, 0);
  std::vector<int> xmin((size_t)ncomp + 1, W), xmax((size_t)ncomp + 1, -1);
  std::vector<int> ymin((size_t)ncomp + 1, H), ymax((size_t)ncomp + 1, -1);
  std::vector<double> sx((size_t)ncomp + 1, 0.0), sy((size_t)ncomp + 1, 0.0);
  std::vector<double> ssum((size_t)ncomp + 1, 0.0);
  std::vector<float> speak((size_t)ncomp + 1, -1.0e30f);

  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      const size_t idx = (size_t)y * W + x;
      const int l = ws->labels[idx];
      if (l == 0) continue;
      const int f = ws->root_to_final[(size_t)UfFind(&ws->parent, l)];
      cnt[(size_t)f]++;
      sx[(size_t)f] += (double)x;
      sy[(size_t)f] += (double)y;
      ssum[(size_t)f] += (double)hm[idx];
      if (hm[idx] > speak[(size_t)f]) speak[(size_t)f] = hm[idx];
      if (x < xmin[(size_t)f]) xmin[(size_t)f] = x;
      if (x > xmax[(size_t)f]) xmax[(size_t)f] = x;
      if (y < ymin[(size_t)f]) ymin[(size_t)f] = y;
      if (y > ymax[(size_t)f]) ymax[(size_t)f] = y;
    }
  }

  // The `sum` is accumulated in double and rounded to float32 once at the
  // end.  numpy sums a float32 array with pairwise summation and returns a
  // float32; the double accumulator is at least as accurate, and the
  // difference is far below the 1e-4 printing resolution.
  out->reserve((size_t)ncomp);
  for (int f = 1; f <= ncomp; f++) {
    if (cnt[(size_t)f] < min_area) continue;
    Comp c;
    c.cx = (float)(sx[(size_t)f] / (double)cnt[(size_t)f]);
    c.cy = (float)(sy[(size_t)f] / (double)cnt[(size_t)f]);
    c.w = xmax[(size_t)f] - xmin[(size_t)f] + 1;
    c.h = ymax[(size_t)f] - ymin[(size_t)f] + 1;
    c.area = cnt[(size_t)f];
    c.peak = speak[(size_t)f];
    c.sum = (float)ssum[(size_t)f];
    out->push_back(c);
  }
  // sort by sum DESCENDING, stable (python's list.sort is stable)
  std::stable_sort(out->begin(), out->end(),
                   [](const Comp& a, const Comp& b) { return a.sum > b.sum; });
  if ((int)out->size() > limit) out->resize((size_t)limit);
}

// ---------------------------------------------------------------------------
// NPU context
// ---------------------------------------------------------------------------
struct NpuCtx {
  rknn_context ctx = 0;
  uint32_t n_out = 0;
  std::vector<rknn_tensor_attr> out_attr;
  int in_h = 0, in_w = 0, in_c = 0;
  int hm_h = 0, hm_w = 0;
  bool input_is_f16 = false;
  bool input_is_f32 = false;
  bool input_is_int8 = false;
  int input_zp = 0;
  rknn_tensor_mem* io_mem = nullptr;
};

enum class InputMode { kSet, kIoMemNchw, kIoMemNhwc };
enum class CanvasMode { kMosaic, kSingle };

// ---------------------------------------------------------------------------
// Timing (8 buckets, per frame)
// ---------------------------------------------------------------------------
struct Timing {
  double decode = 0;
  double rga_import = 0;
  double rga_improcess = 0;
  double upload = 0;
  double run = 0;
  double outputs = 0;
  double backproj = 0;
  double write = 0;
  int frames = 0;
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
struct Options {
  std::string model;
  std::string list;
  std::string out;
  std::string timing;
  int start = 0;
  int limit = 0;            // 0 == no limit
  std::vector<int> tiles;   // empty == all
  std::string cpu_affinity;
  InputMode input_mode = InputMode::kSet;
  CanvasMode canvas_mode = CanvasMode::kMosaic;
};

void PrintUsage(const char* argv0) {
  fprintf(stderr,
          "usage: %s --model <model.rknn> --list <manifest.txt> --out <dir>\n"
          "          [--start N] [--limit N]\n"
          "          [--tiles 0,1,2]        only emit these tiles (default: all)\n"
          "          [--cpu-affinity LIST]  e.g. 4-7\n"
          "          [--timing <file>]      default <out>/timing.txt\n"
          "          [--input-mode {set,io-mem-nchw,io-mem-nhwc}]   default: set\n"
          "          [--canvas-mode {mosaic,single}]                default: mosaic\n"
          "\n"
          "  mosaic : canvas = native(0,0,640,512) + thumb160(0,512,160,128)\n"
          "                            + thumb80(160,512,80,64), background 114\n"
          "  single : canvas = the PC-side native letterbox replica,\n"
          "           native(0,64,640,512), background 114 (tag 0 only)\n",
          argv0);
}

bool ParseTiles(const char* s, std::vector<int>* out, std::string* err) {
  out->clear();
  const char* p = s;
  while (*p) {
    char* e = nullptr;
    const long v = strtol(p, &e, 10);
    if (e == p) { *err = std::string("bad --tiles value: ") + s; return false; }
    if (v < 0 || v > 2) {
      *err = std::string("--tiles only accepts 0,1,2 (got ") + std::to_string(v) + ")";
      return false;
    }
    out->push_back((int)v);
    p = e;
    if (*p == ',') p++;
    else if (*p != '\0') { *err = std::string("bad --tiles value: ") + s; return false; }
  }
  return true;
}

bool ApplyCpuAffinity(const std::string& spec) {
  cpu_set_t set;
  CPU_ZERO(&set);
  const char* p = spec.c_str();
  while (*p) {
    int a = 0, b = 0;
    char* e = nullptr;
    a = (int)strtol(p, &e, 10);
    if (e == p) { p++; continue; }
    b = a;
    if (*e == '-') { p = e + 1; b = (int)strtol(p, &e, 10); }
    for (int c = a; c <= b && c < CPU_SETSIZE; c++) CPU_SET(c, &set);
    p = e;
  }
  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    fprintf(stderr, "[WARN] sched_setaffinity(%s) failed: %s\n", spec.c_str(), strerror(errno));
    return false;
  }
  fprintf(stderr, "[i] cpu affinity pinned to %s\n", spec.c_str());
  return true;
}

// ---------------------------------------------------------------------------
// Sequence name: the directory the BMPs live in (the board driver writes one
// list per sequence, e.g. list_VIDEO00007_19700101_001324.txt).  Falls back
// to the list file's own stem.
// ---------------------------------------------------------------------------
std::string DeriveSeqName(const std::string& list_path, const std::vector<std::string>& frames) {
  if (!frames.empty()) {
    const std::string& p = frames[0];
    const size_t slash = p.find_last_of('/');
    if (slash != std::string::npos && slash + 1 < p.size()) {
      const std::string dir = p.substr(0, slash);
      const size_t slash2 = dir.find_last_of('/');
      const std::string name = (slash2 == std::string::npos) ? dir : dir.substr(slash2 + 1);
      if (!name.empty() && name != "." && name != "..") return name;
    }
  }
  const size_t slash = list_path.find_last_of('/');
  std::string stem = (slash == std::string::npos) ? list_path : list_path.substr(slash + 1);
  const size_t dot = stem.find_last_of('.');
  if (dot != std::string::npos && dot > 0) stem = stem.substr(0, dot);
  if (stem.rfind("list_", 0) == 0) stem = stem.substr(5);
  return stem.empty() ? std::string("seq") : stem;
}

std::string BaseName(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  return (slash == std::string::npos) ? path : path.substr(slash + 1);
}

std::string StemOf(const std::string& path) {
  std::string b = BaseName(path);
  const size_t dot = b.find_last_of('.');
  if (dot != std::string::npos && dot > 0) b = b.substr(0, dot);
  return b;
}

void EnsureDir(const std::string& dir) {
  if (dir.empty()) return;
  std::string cur;
  size_t i = 0;
  if (dir[0] == '/') { cur = "/"; i = 1; }
  while (i <= dir.size()) {
    const size_t slash = dir.find('/', i);
    const std::string part = dir.substr(i, (slash == std::string::npos ? dir.size() : slash) - i);
    if (!part.empty()) {
      if (!cur.empty() && cur.back() != '/') cur += "/";
      cur += part;
      mkdir(cur.c_str(), 0755);
    }
    if (slash == std::string::npos) break;
    i = slash + 1;
  }
}

}  // namespace

// ===========================================================================
int main(int argc, char** argv) {
  Options opt;
  bool have_tiles = false;

  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") { PrintUsage(argv[0]); return 0; }
    else if (a == "--model" && i + 1 < argc) opt.model = argv[++i];
    else if (a == "--list" && i + 1 < argc) opt.list = argv[++i];
    else if (a == "--out" && i + 1 < argc) opt.out = argv[++i];
    else if (a == "--timing" && i + 1 < argc) opt.timing = argv[++i];
    else if (a == "--start" && i + 1 < argc) opt.start = atoi(argv[++i]);
    else if (a == "--limit" && i + 1 < argc) opt.limit = atoi(argv[++i]);
    else if (a == "--tiles" && i + 1 < argc) {
      std::string err;
      if (!ParseTiles(argv[++i], &opt.tiles, &err)) {
        fprintf(stderr, "[FATAL] %s\n", err.c_str());
        return 2;
      }
      have_tiles = true;
    }
    else if (a == "--cpu-affinity" && i + 1 < argc) opt.cpu_affinity = argv[++i];
    else if (a == "--input-mode" && i + 1 < argc) {
      const std::string m = argv[++i];
      if (m == "set") opt.input_mode = InputMode::kSet;
      else if (m == "io-mem-nchw") opt.input_mode = InputMode::kIoMemNchw;
      else if (m == "io-mem-nhwc") opt.input_mode = InputMode::kIoMemNhwc;
      else { fprintf(stderr, "[FATAL] unknown --input-mode: %s\n", m.c_str()); return 2; }
    }
    else if (a == "--canvas-mode" && i + 1 < argc) {
      const std::string m = argv[++i];
      if (m == "mosaic") opt.canvas_mode = CanvasMode::kMosaic;
      else if (m == "single") opt.canvas_mode = CanvasMode::kSingle;
      else { fprintf(stderr, "[FATAL] unknown --canvas-mode: %s\n", m.c_str()); return 2; }
    }
    else {
      fprintf(stderr, "[FATAL] unknown option: %s\n", argv[i]);
      PrintUsage(argv[0]);
      return 2;
    }
  }

  if (opt.model.empty() || opt.list.empty() || opt.out.empty()) {
    fprintf(stderr, "[FATAL] --model, --list and --out are all required\n");
    PrintUsage(argv[0]);
    return 2;
  }
  if (opt.start < 0) opt.start = 0;
  if (opt.limit < 0) opt.limit = 0;

  // The single-scale arm only has the native tile.
  if (opt.canvas_mode == CanvasMode::kSingle && have_tiles) {
    for (int t : opt.tiles) {
      if (t != 0) {
        fprintf(stderr,
                "[FATAL] --canvas-mode single only has the native tile (tag 0); "
                "--tiles %d is not available in that mode.\n",
                t);
        return 2;
      }
    }
  }

  // The board's librga prints its debug dump to STDOUT.  Detach stdout so the
  // contract "logs on stderr, results in files" cannot be broken by a
  // library.
  if (FILE* devnull = fopen("/dev/null", "w")) {
    dup2(fileno(devnull), STDOUT_FILENO);
    fclose(devnull);
    fprintf(stderr, "[i] stdout detached (the board's librga prints RGA debug dumps there)\n");
  }

  if (!opt.cpu_affinity.empty()) ApplyCpuAffinity(opt.cpu_affinity);

  // ---- frame list ---------------------------------------------------------
  std::vector<std::string> frames;
  {
    FILE* f = fopen(opt.list.c_str(), "r");
    if (!f) {
      fprintf(stderr, "[FATAL] cannot read list %s: %s\n", opt.list.c_str(), strerror(errno));
      return 1;
    }
    char line[2048];
    while (fgets(line, sizeof(line), f)) {
      std::string s(line);
      while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
      if (!s.empty()) frames.push_back(s);
    }
    fclose(f);
  }
  if ((int)frames.size() <= opt.start) {
    fprintf(stderr, "[FATAL] list has %zu frames, --start %d skips all of them\n", frames.size(),
            opt.start);
    return 1;
  }
  frames.erase(frames.begin(), frames.begin() + opt.start);
  if (opt.limit > 0 && (int)frames.size() > opt.limit) frames.resize((size_t)opt.limit);
  if (frames.empty()) { fprintf(stderr, "[FATAL] no frames to process\n"); return 1; }

  const std::string seqname = DeriveSeqName(opt.list, frames);
  EnsureDir(opt.out);
  const std::string tsv_path = opt.out + "/" + seqname + "_mosaic.tsv";
  const std::string timing_path = opt.timing.empty() ? (opt.out + "/timing.txt") : opt.timing;

  // ---- tiles --------------------------------------------------------------
  std::vector<Tile> tiles;
  if (opt.canvas_mode == CanvasMode::kMosaic) {
    // tag, x, y, w, h
    tiles.push_back({0, 0, 0, 640, 512, true});      // native, 1:1
    tiles.push_back({1, 0, 512, 160, 128, true});    // thumb160, 1/4 down
    tiles.push_back({2, 160, 512, 80, 64, true});    // thumb80, 1/8 down
  } else {
    tiles.push_back({0, 0, 64, 640, 512, true});     // native letterbox replica
  }
  if (have_tiles) {
    for (Tile& t : tiles) {
      bool want = false;
      for (int sel : opt.tiles) if (sel == t.tag) want = true;
      t.enabled = want;
    }
  }
  bool any_tile = false;
  for (const Tile& t : tiles) if (t.enabled) any_tile = true;
  if (!any_tile) { fprintf(stderr, "[FATAL] --tiles selected no tile at all\n"); return 2; }

  // ---- NPU ----------------------------------------------------------------
  std::vector<uint8_t> model;
  if (!read_file(opt.model.c_str(), &model)) {
    fprintf(stderr, "[FATAL] cannot read model %s\n", opt.model.c_str());
    return 1;
  }
  NpuCtx npu;
  if (rknn_init(&npu.ctx, model.data(), (uint32_t)model.size(), 0, nullptr) != 0) {
    fprintf(stderr, "[FATAL] rknn_init failed\n");
    return 1;
  }
  {
    rknn_sdk_version ver;
    memset(&ver, 0, sizeof(ver));
    if (rknn_query(npu.ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver)) == 0)
      fprintf(stderr, "[i] rknn runtime api=%s driver=%s\n", ver.api_version, ver.drv_version);
  }
  rknn_input_output_num io;
  memset(&io, 0, sizeof(io));
  rknn_query(npu.ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
  if (io.n_input != 1 || io.n_output < 1) {
    fprintf(stderr, "[FATAL] expected 1 input / >=1 outputs, got %u / %u\n", io.n_input, io.n_output);
    return 1;
  }
  rknn_tensor_attr in_attr;
  memset(&in_attr, 0, sizeof(in_attr));
  in_attr.index = 0;
  rknn_query(npu.ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
  if (in_attr.fmt == RKNN_TENSOR_NCHW) {
    npu.in_c = (int)in_attr.dims[1];
    npu.in_h = (int)in_attr.dims[2];
    npu.in_w = (int)in_attr.dims[3];
  } else {
    npu.in_h = (int)in_attr.dims[1];
    npu.in_w = (int)in_attr.dims[2];
    npu.in_c = (int)in_attr.dims[3];
  }
  npu.input_is_f16 = (in_attr.type == RKNN_TENSOR_FLOAT16);
  npu.input_is_f32 = (in_attr.type == RKNN_TENSOR_FLOAT32);
  npu.input_is_int8 = (in_attr.type == RKNN_TENSOR_INT8);
  npu.input_zp = in_attr.zp;
  fprintf(stderr, "[i] input %dx%dx%d fmt=%s type=%s zp=%d scale=%.8f\n", npu.in_h, npu.in_w, npu.in_c,
          tensor_fmt_name(in_attr.fmt), tensor_type_name(in_attr.type), in_attr.zp,
          (double)in_attr.scale);
  if (npu.in_h != kCanvas || npu.in_w != kCanvas || npu.in_c != kSrcC) {
    fprintf(stderr, "[FATAL] model input is %dx%dx%d, expected %dx%dx%d\n", npu.in_h, npu.in_w,
            npu.in_c, kCanvas, kCanvas, kSrcC);
    return 1;
  }
  npu.n_out = io.n_output;
  npu.out_attr.assign(npu.n_out, rknn_tensor_attr{});
  for (uint32_t i = 0; i < npu.n_out; i++) {
    memset(&npu.out_attr[i], 0, sizeof(rknn_tensor_attr));
    npu.out_attr[i].index = i;
    rknn_query(npu.ctx, RKNN_QUERY_OUTPUT_ATTR, &npu.out_attr[i], sizeof(npu.out_attr[i]));
    fprintf(stderr, "[i] out[%u] dims=[", i);
    for (uint32_t d = 0; d < npu.out_attr[i].n_dims; d++)
      fprintf(stderr, "%u%s", npu.out_attr[i].dims[d], d + 1 < npu.out_attr[i].n_dims ? "," : "");
    fprintf(stderr, "] fmt=%s type=%s\n", tensor_fmt_name(npu.out_attr[i].fmt),
            tensor_type_name(npu.out_attr[i].type));
  }
  {
    const rknn_tensor_attr& a = npu.out_attr[0];
    if (a.n_dims != 4 || a.dims[0] != 1 || a.dims[1] != 1) {
      fprintf(stderr, "[FATAL] out[0] is not [1,1,H,W]\n");
      return 1;
    }
    npu.hm_h = (int)a.dims[2];
    npu.hm_w = (int)a.dims[3];
  }
  fprintf(stderr, "[i] heatmap grid %dx%d (canvas %dx%d, stride %d)\n", npu.hm_w, npu.hm_h, kCanvas,
          kCanvas, kCanvas / npu.hm_w);
  for (const Tile& t : tiles) {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    HeatRect(t, npu.hm_w, npu.hm_h, &x0, &y0, &x1, &y1);
    fprintf(stderr,
            "[i] tile tag=%d canvas(%d,%d,%d,%d) -> heatmap rows[%d,%d) cols[%d,%d)  %dx\n", t.tag,
            t.x, t.y, t.w, t.h, y0, y1, x0, x1, (x1 - x0) * kSrcW / (t.w ? t.w : 1));
  }

  // ---- zero-copy input buffer (optional) ----------------------------------
  if (opt.input_mode != InputMode::kSet) {
    const uint32_t mem_sz = in_attr.size_with_stride ? in_attr.size_with_stride : in_attr.size;
    npu.io_mem = rknn_create_mem(npu.ctx, mem_sz);
    if (!npu.io_mem) {
      fprintf(stderr, "[FATAL] rknn_create_mem(%u) failed\n", mem_sz);
      return 1;
    }
    if (rknn_set_io_mem(npu.ctx, npu.io_mem, &in_attr) != 0) {
      fprintf(stderr, "[FATAL] rknn_set_io_mem failed\n");
      return 1;
    }
    fprintf(stderr, "[i] io-mem input bound: %p, %u B (attr size %u)\n", npu.io_mem->virt_addr,
            mem_sz, in_attr.size);
  }

  // ---- RGA buffers --------------------------------------------------------
  RgaBuf canvas, pat, stage;
  if (!BufAlloc(&canvas, kCanvas, kCanvas, kRgaFmt) || !BufAlloc(&pat, 4, 4, kRgaFmt) ||
      !BufAlloc(&stage, kSrcW, kSrcH, kRgaFmt)) {
    fprintf(stderr, "[FATAL] RGA buffer allocation failed\n");
    return 1;
  }
  if (!CheckBudget(canvas, "canvas") || !CheckBudget(pat, "pat") || !CheckBudget(stage, "stage")) {
    return 1;
  }
  // The canvas background is frame-invariant -> filled ONCE.  imfill() only
  // supports RGBA, so the RGB_888 canvas is filled with memset.
  memset(canvas.data, kPad, canvas.size);

  // The decoded frame is imported into RGA straight from stb_image's buffer
  // (zero CPU image copy).  The board's librga is picky about the buffer, so
  // the FIRST frame verifies that the RGA really wrote the native tile; if it
  // did not (the known "silent no-op" failure mode), the program switches to
  // an aligned staging buffer for the rest of the run rather than emitting
  // garbage.
  bool use_staging = false;
  bool verified = false;

  // ---- output files -------------------------------------------------------
  FILE* tsv = fopen(tsv_path.c_str(), "w");
  if (!tsv) {
    fprintf(stderr, "[FATAL] cannot open %s: %s\n", tsv_path.c_str(), strerror(errno));
    return 1;
  }
  fprintf(stderr, "[i] TSV    -> %s\n", tsv_path.c_str());
  fprintf(stderr, "[i] timing -> %s\n", timing_path.c_str());
  fprintf(stderr, "[i] canvas-mode=%s input-mode=%s frames=%zu seq=%s\n",
          opt.canvas_mode == CanvasMode::kMosaic ? "mosaic" : "single",
          opt.input_mode == InputMode::kSet ? "set"
          : (opt.input_mode == InputMode::kIoMemNchw ? "io-mem-nchw" : "io-mem-nhwc"),
          frames.size(), seqname.c_str());

  // ---- per-frame state ----------------------------------------------------
  std::vector<float> hm;
  std::vector<rknn_output> outs;
  std::vector<float> native_map((size_t)kSrcW * kSrcH, 0.0f);
  std::vector<Comp> comps;
  CcWorkspace cc;
  std::vector<uint16_t> f16_tmp;
  std::vector<Comp> all_comps;

  Timing tm;
  const double wall0 = now_ms();

  for (size_t fi = 0; fi < frames.size(); fi++) {
    const std::string& path = frames[fi];
    const std::string stem = StemOf(path);

    // ---- 1. decode --------------------------------------------------------
    const double t0 = now_ms();
    int w = 0, h = 0, comp = 0;
    uint8_t* pixels = stbi_load(path.c_str(), &w, &h, &comp, 3);
    if (!pixels) {
      fprintf(stderr, "[FATAL] stbi_load failed: %s\n", path.c_str());
      return 1;
    }
    // PIXEL CONTRACT: the BMP stores BGR, stb returns RGB, and that RGB byte
    // order IS the PC side's model_order_bmp() == cv2.imread(bmp)[..., ::-1],
    // i.e. the model's channel order [residual, diff, I_t].  No flip here.
    if (w != kSrcW || h != kSrcH || comp != kSrcC) {
      fprintf(stderr, "[FATAL] %s: expected %dx%dx%d, got %dx%dx%d\n", path.c_str(), kSrcW, kSrcH,
              kSrcC, w, h, comp);
      return 1;
    }
    const double t_decode = now_ms() - t0;

    // ---- 2. RGA: import + improcess --------------------------------------
    const uint8_t* src_pixels = pixels;
    if (use_staging) {
      memcpy(stage.data, pixels, stage.size);
      src_pixels = stage.data;
    }
    double t_import = 0.0, t_improc = 0.0;
    bool rga_ok = true;
    for (int attempt = 0; attempt < 2 && rga_ok; attempt++) {
      const double i0 = now_ms();
      // The source buffer must be imported AFTER its content is final.
      const rga_buffer_handle_t src_handle =
          importbuffer_virtualaddr(const_cast<uint8_t*>(src_pixels), kSrcW, kSrcH, kRgaFmt);
      const bool canvas_ok = BufImport(&canvas);
      const double i1 = now_ms();
      t_import = i1 - i0;
      if (src_handle <= 0 || !canvas_ok) {
        fprintf(stderr, "[FATAL] RGA import failed (src=%d dst=%d)\n", (int)(src_handle > 0),
                (int)canvas_ok);
        if (src_handle > 0) releasebuffer_handle(src_handle);
        rga_ok = false;
        break;
      }
      const rga_buffer_t srcb = wrapbuffer_handle(src_handle, kSrcW, kSrcH, kRgaFmt);

      const double m0 = now_ms();
      for (const Tile& t : tiles) {
        const IM_STATUS st = improcess(srcb, BufAs(canvas), BufAs(pat), MakeRect(0, 0, kSrcW, kSrcH),
                                       MakeRect(t.x, t.y, t.w, t.h), MakeRect(0, 0, 0, 0), 0);
        if (st != IM_STATUS_SUCCESS && st != IM_STATUS_NOERROR) {
          fprintf(stderr, "[FATAL] improcess tile %d -> %s (%d)\n", t.tag, rga_status_name(st),
                  (int)st);
          rga_ok = false;
          break;
        }
      }
      t_improc = now_ms() - m0;
      releasebuffer_handle(src_handle);
      if (!rga_ok) break;

      if (!verified) {
        // One-shot self-check: the native tile is a 1:1 blit, so the canvas
        // region MUST equal the source byte for byte.  This catches the
        // "RGA returned SUCCESS but wrote nothing" failure mode.
        const Tile* nat = &tiles[0];
        if (nat->tag == 0 && nat->w == kCanvas) {
          const size_t off_bytes = (size_t)nat->y * kCanvas * kSrcC;
          const size_t len = (size_t)nat->h * kCanvas * kSrcC;
          const bool same = memcmp(canvas.data + off_bytes, src_pixels, len) == 0;
          if (!same && !use_staging) {
            fprintf(stderr,
                    "[WARN] the RGA did not write the native tile from the stb_image buffer; "
                    "switching to an aligned staging buffer for the rest of the run.\n");
            use_staging = true;
            memcpy(stage.data, pixels, stage.size);
            src_pixels = stage.data;
            rga_ok = true;  // retry this frame with the staging buffer
            continue;
          }
          if (!same) {
            fprintf(stderr,
                    "[FATAL] the RGA did not write the native tile even from the staging buffer; "
                    "the mosaic canvas would be garbage.\n");
            return 1;
          }
          fprintf(stderr, "[i] RGA native-tile self-check: OK\n");
        }
        verified = true;
      }
    }
    if (!rga_ok) return 1;
    free(pixels);

    // ---- 3. upload ---------------------------------------------------------
    const size_t canvas_elems = canvas.size;
    const double u0 = now_ms();
    if (opt.input_mode == InputMode::kSet) {
      // The default path: a plain uint8 NHWC upload, structurally identical
      // to the certified single-scale board arm (the runtime does the
      // uint8 -> fp16 widening itself).
      rknn_input in{};
      in.index = 0;
      in.pass_through = 0;
      in.fmt = RKNN_TENSOR_NHWC;
      in.type = RKNN_TENSOR_UINT8;
      in.size = (uint32_t)canvas_elems;
      in.buf = canvas.data;
      if (rknn_inputs_set(npu.ctx, 1, &in) != 0) {
        fprintf(stderr, "[FATAL] rknn_inputs_set failed\n");
        return 1;
      }
    } else {
      // Zero-copy path: the host fills the bound input tensor itself.
      //
      // WHY THIS MAY NOT MATCH `set` BIT FOR BIT (read before the board A/B):
      //   * `set` hands the runtime a uint8 NHWC canvas and lets librknnrt do
      //     the uint8 -> fp16 widening AND the layout conversion into the
      //     model's internal tensor format.
      //   * `io-mem-*` makes the host write the tensor itself, so the host now
      //     decides the layout.  For an fp16 graph the byte values are
      //     identical (every integer 0..255 is exact in fp16, and
      //     u8_to_f16_int is the closed form), but the LAYOUT must match what
      //     the bound tensor expects: RKNN_QUERY_INPUT_ATTR reports the
      //     tensor's own fmt, and the two io-mem modes cover the two
      //     possibilities (NHWC and NCHW).
      //   * If the two arms disagree, the difference is therefore in the
      //     LAYOUT the host wrote, not in the pixel values -- check
      //     in_attr.fmt against the mode you picked.  A disagreement that
      //     survives both layouts is a real difference and must be reported,
      //     not averaged away.
      uint8_t* dst = (uint8_t*)npu.io_mem->virt_addr;
      if (npu.input_is_f16) {
        if (opt.input_mode == InputMode::kIoMemNchw) {
          f16_tmp.resize(canvas_elems);
          u8_to_f16(canvas.data, f16_tmp.data(), canvas_elems, 1.0f);
          nhwc_to_nchw_f16(f16_tmp.data(), (uint16_t*)dst, kCanvas, kSrcC);
        } else {
          u8_to_f16(canvas.data, (uint16_t*)dst, canvas_elems, 1.0f);
        }
      } else if (npu.input_is_f32) {
        // A float32 graph consumes x/255 (its std_values are baked in).
        float* fdst = (float*)dst;
        for (size_t i = 0; i < canvas_elems; i++) fdst[i] = (float)canvas.data[i] / 255.0f;
      } else if (npu.input_is_int8) {
        // An INT8 input holds CODES: with scale=1/255 and zp=-128 the code of
        // a host byte x is x + zp.
        int8_t* d8 = (int8_t*)dst;
        for (size_t i = 0; i < canvas_elems; i++)
          d8[i] = (int8_t)((int)canvas.data[i] + npu.input_zp);
      } else {
        memcpy(dst, canvas.data, canvas_elems);
      }
    }
    const double t_upload = now_ms() - u0;
    // NOTE ON THE COST OF THIS BUCKET (for table C):
    //   in `set` mode this bucket is librknnrt's uint8 -> fp16 widening of all
    //   640*640*3 elements, which measured 45.4 ms/frame on an RK3588 A55
    //   (1.8 GHz) and ~11-15 ms on an A76 (2.35 GHz) -- it is the single
    //   largest HOST cost of the fp16 arm, not the NPU.
    //   in the io-mem modes the conversion is ours (~0.8 ms, integer NEON),
    //   but the runtime then stages the buffer into NPU memory inside
    //   rknn_run, so the saving can be a net LOSS.  Do not read the io-mem
    //   numbers as "the upload is free".

    // ---- 4. run + outputs --------------------------------------------------
    outs.assign(npu.n_out, rknn_output{});
    for (uint32_t i = 0; i < npu.n_out; i++) {
      outs[i].index = i;
      outs[i].want_float = 0;
    }
    const double r0 = now_ms();
    if (rknn_run(npu.ctx, nullptr) != 0) {
      fprintf(stderr, "[FATAL] rknn_run failed\n");
      return 1;
    }
    const double r1 = now_ms();
    if (rknn_outputs_get(npu.ctx, npu.n_out, outs.data(), nullptr) != 0) {
      fprintf(stderr, "[FATAL] rknn_outputs_get failed\n");
      return 1;
    }
    const double r2 = now_ms();
    // Only the heatmap is dequantised: the offset output is not used by the
    // connected-component detector (the PC side's detect_components does not
    // use it either).
    if (!dequant(outs[0], npu.out_attr[0], &hm)) {
      fprintf(stderr, "[FATAL] dequant of the heatmap output failed\n");
      rknn_outputs_release(npu.ctx, npu.n_out, outs.data());
      return 1;
    }
    // NOTE: this segment covers rknn_outputs_get + dequant ONLY.  It must not
    // start at r0, or the rknn_run time would be counted twice and
    // sum_of_segments would exceed total_wall.
    const double t_outputs = now_ms() - r1;
    rknn_outputs_release(npu.ctx, npu.n_out, outs.data());
    const double t_run = r1 - r0;
    (void)r2;

    // ---- 5. back-projection + connected components -------------------------
    const double b0 = now_ms();
    all_comps.clear();
    for (const Tile& t : tiles) {
      if (!t.enabled) continue;
      int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
      HeatRect(t, npu.hm_w, npu.hm_h, &x0, &y0, &x1, &y1);
      const int sw = x1 - x0, sh = y1 - y0;
      if (sw <= 0 || sh <= 0) continue;
      // The tile's heatmap sub-rectangle, enlarged back onto the native grid.
      // The source rows are still npu.hm_w floats apart -- the sub-rectangle
      // is a WINDOW of the heatmap, not a packed image.
      ResizeBilinearF32(hm.data() + (size_t)y0 * npu.hm_w + x0, npu.hm_w, sw, sh,
                        native_map.data(), kSrcW, kSrcH);
      DetectComponents(&cc, native_map.data(), kSrcW, kSrcH, kRegionThreshold, kMinArea, kMaxDet,
                       &comps);
      for (const Comp& c : comps) {
        Comp tagged = c;
        tagged.tag = t.tag;
        all_comps.push_back(tagged);
      }
    }
    const double t_backproj = now_ms() - b0;

    // ---- 6. write ----------------------------------------------------------
    const double w0 = now_ms();
    fprintf(tsv, "%s\t%zu", stem.c_str(), all_comps.size());
    for (const Comp& c : all_comps) {
      fprintf(tsv, "\t%d %.4f %.4f %d %d %d %.6f %.4f", c.tag, (double)c.cx, (double)c.cy, c.w, c.h,
              c.area, (double)c.peak, (double)c.sum);
    }
    fprintf(tsv, "\n");
    fflush(tsv);
    const double t_write = now_ms() - w0;

    tm.decode += t_decode;
    tm.rga_import += t_import;
    tm.rga_improcess += t_improc;
    tm.upload += t_upload;
    tm.run += t_run;
    tm.outputs += t_outputs;
    tm.backproj += t_backproj;
    tm.write += t_write;
    tm.frames++;
  }

  const double wall = now_ms() - wall0;
  fclose(tsv);

  // ---- timing report ------------------------------------------------------
  const int nf = tm.frames > 0 ? tm.frames : 1;
  const double fps = (wall > 0.0) ? (1000.0 * (double)tm.frames / wall) : 0.0;
  {
    FILE* tf = fopen(timing_path.c_str(), "w");
    if (!tf) {
      fprintf(stderr, "[WARN] cannot open timing file %s: %s\n", timing_path.c_str(),
              strerror(errno));
    } else {
      fprintf(tf, "# rknn_mosaic_heatmap timing\n");
      fprintf(tf, "canvas_mode=%s\n",
              opt.canvas_mode == CanvasMode::kMosaic ? "mosaic" : "single");
      fprintf(tf, "input_mode=%s\n",
              opt.input_mode == InputMode::kSet ? "set"
              : (opt.input_mode == InputMode::kIoMemNchw ? "io-mem-nchw" : "io-mem-nhwc"));
      fprintf(tf, "seq=%s\n", seqname.c_str());
      fprintf(tf, "frames=%d\n", tm.frames);
      fprintf(tf, "%-18s %14s %14s %8s\n", "segment", "total_ms", "avg_ms/frame", "frames");
      const struct { const char* name; double v; } segs[] = {
          {"decode", tm.decode},        {"rga_import", tm.rga_import},
          {"rga_improcess", tm.rga_improcess}, {"fp16_or_upload", tm.upload},
          {"rknn_run", tm.run},        {"outputs_get", tm.outputs},
          {"backproject_cc", tm.backproj}, {"write", tm.write},
      };
      for (const auto& s : segs)
        fprintf(tf, "%-18s %14.3f %14.4f %8d\n", s.name, s.v, s.v / (double)nf, tm.frames);
      const double sum_seg = tm.decode + tm.rga_import + tm.rga_improcess + tm.upload + tm.run +
                             tm.outputs + tm.backproj + tm.write;
      fprintf(tf, "%-18s %14.3f %14.4f %8d\n", "sum_of_segments", sum_seg, sum_seg / (double)nf,
              tm.frames);
      fprintf(tf, "%-18s %14.3f %14.4f %8d\n", "total_wall", wall, wall / (double)nf, tm.frames);
      fprintf(tf, "fps=%.3f\n", fps);
      fclose(tf);
    }
  }
  fprintf(stderr, "\n[timing] %d frames, total wall %.1f ms (%.3f ms/frame), %.3f fps\n", tm.frames,
          wall, wall / (double)nf, fps);
  fprintf(stderr, "[timing] decode %.3f  rga_import %.3f  rga_improcess %.3f  upload %.3f  "
                  "run %.3f  outputs %.3f  backproj %.3f  write %.3f  (ms/frame)\n",
          tm.decode / (double)nf, tm.rga_import / (double)nf, tm.rga_improcess / (double)nf,
          tm.upload / (double)nf, tm.run / (double)nf, tm.outputs / (double)nf,
          tm.backproj / (double)nf, tm.write / (double)nf);

  BufFree(&canvas);
  BufFree(&pat);
  BufFree(&stage);
  if (npu.io_mem) rknn_destroy_mem(npu.ctx, npu.io_mem);
  rknn_destroy(npu.ctx);
  return 0;
}
