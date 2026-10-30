// ===========================================================================
// mosaic_core.h -- the RK3588 f16 multi-scale mosaic inference core.
//
// WHY THIS FILE EXISTS
// --------------------
// rknn_mosaic_heatmap.cpp (the BMP-driven binary) and rknn_mosaic_e2e.cpp
// (the end-to-end demo: raw infrared -> GMC/OpenCL feature -> RGA mosaic ->
// NPU -> back-projection -> detections) MUST run the SAME inference code, or
// their numbers cannot be compared.  The e2e demo therefore includes this
// header instead of re-implementing the RGA/NPU/back-projection path; that is
// the same discipline e2e_core.cpp uses to share gmc_stream_ocl.cpp and
// rknn_heatmap_smoke.cpp.
//
// A Session owns ONE RKNN context and ONE RGA canvas, so several Sessions can
// run concurrently in several threads -- which is exactly what the two-NPU
// e2e demo does (the board tolerates exactly two concurrent contexts).
//
// The body below was moved verbatim out of rknn_mosaic_heatmap.cpp's
// anonymous namespace; the per-frame path was lifted into ProcessFrame() with
// the decode step replaced by a caller-supplied model-order RGB buffer, and
// the TSV write replaced by FormatTsvLine().  Both binaries are verified
// byte-for-byte against the archived 28,997-frame run.
// ===========================================================================
#pragma once
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
#include <pthread.h>
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

namespace mosaic {

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

// ---------------------------------------------------------------------------
// Three plane-major 8-bit planes -> ONE interleaved 3-channel buffer, in the
// model order [residual, diff, I_t] = [c2, c1, c0].
//
// WHY THIS EXISTS (measured, not guessed): the scalar version of this loop
// wrote 983,040 bytes with a stride-3 store in a byte-at-a-time loop.  On the
// board that cost ~15 ms of CPU per frame INSIDE the GMC service -- and the
// GMC service was the stage that could not keep up with a 25 fps camera, so
// this was not a cosmetic cost: it pushed the feature stage below 25 fps and
// the camera queue overflowed (which breaks the GMC registration chain).
//
// The NEON path does the same 16 pixels at a time with vst3q_u8, which is the
// hardware interleaving store.  THE BYTES ARE IDENTICAL -- this is a pure
// speed change and the bit-exactness contract is re-verified on the full
// sequence after it (the detection TSV md5 must stay 0bf598a1d890040bcfa8d157dc69c032).
// ---------------------------------------------------------------------------
inline void Interleave3Planes(const uint8_t* c0, const uint8_t* c1, const uint8_t* c2,
                               size_t plane, uint8_t* dst) {
#ifdef MOSAIC_HAVE_NEON
  size_t i = 0;
  for (; i + 16 <= plane; i += 16) {
    uint8x16x3_t v;
    v.val[0] = vld1q_u8(c2 + i);  // R = residual
    v.val[1] = vld1q_u8(c1 + i);  // G = diff
    v.val[2] = vld1q_u8(c0 + i);  // B = I_t
    vst3q_u8(dst + i * 3, v);
  }
  for (; i < plane; ++i) {
    dst[i * 3 + 0] = c2[i];
    dst[i * 3 + 1] = c1[i];
    dst[i * 3 + 2] = c0[i];
  }
#else
  for (size_t i = 0; i < plane; ++i) {
    dst[i * 3 + 0] = c2[i];
    dst[i * 3 + 1] = c1[i];
    dst[i * 3 + 2] = c0[i];
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
  bool input_nchw = false;   // the model's input tensor layout
  int input_zp = 0;
  rknn_tensor_mem* io_mem = nullptr;
  // The queried INPUT tensor attribute, kept so a caller can bind ANY
  // pre-allocated buffer to this context with rknn_set_io_mem().  The
  // DECOUPLED pipeline needs exactly that: the pre-process stage writes the
  // widened fp16 straight into a Buffer A slot and the NPU stage binds that
  // slot to its own context, one slot per frame.
  rknn_tensor_attr in_attr;
};

enum class InputMode {
  kSet,       // uint8 NHWC upload; the runtime does the uint8->fp16 widening
  kSetF16,    // WE do the uint8->fp16 widening (integer NEON) and hand the
              // runtime a FLOAT16 buffer, so the runtime only COPIES instead
              // of widening.  Same numbers, a fraction of the host time.
  kIoMemNchw,
  kIoMemNhwc,
  // APPENDED (v10).  The same widened fp16 buffer as kSetF16, but handed over
  // with pass_through=1: the runtime performs NO conversion and NO copy of
  // the layout, it takes the bytes as they are.  That is only legal when the
  // buffer already IS the model's layout (NHWC fp16 for our model), so the
  // mode refuses an NCHW model instead of silently feeding garbage.
  // Used by the host-memory arm of the decoupled pipeline (buf_a_mode=1).
  kSetF16Pt
};
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
// PER-SERVICE CPU AFFINITY.
//
// The process-wide affinity above is applied with sched_setaffinity(0), which
// only narrows the CALLING thread; the worker threads inherit the mask when
// they are created, so they all end up on the SAME set of cores.
//
// That is not good enough for a pipeline whose stages have very different
// profiles: a GMC fit is a 60 ms CPU burst that MUST NOT be preempted (a
// preempted fit does not just take longer, it makes the whole camera queue
// overflow), while an NPU service spends most of its time blocked on the NPU
// hardware.  Measured on the board: with everything on cores 4-7 the two GMC
// engine threads each got only ~0.85 of a core and the feature stage ran at
// 23.3 fps -- BELOW the 25 fps the camera delivers.
//
// So each service gets its own core list.  The thread pins ITSELF with
// pthread_setaffinity_np, which is the only way to place an individual
// thread.
// ---------------------------------------------------------------------------

// "4" -> {4};  "4,5" -> {4,5};  "4-7" -> {4,5,6,7};  "" -> {} (empty = no list).
inline bool ParseCpuList(const std::string& spec, std::vector<int>* out) {
  out->clear();
  const char* p = spec.c_str();
  while (*p) {
    int a = 0, b = 0;
    char* e = nullptr;
    a = (int)strtol(p, &e, 10);
    if (e == p) { p++; continue; }
    b = a;
    if (*e == '-') { p = e + 1; b = (int)strtol(p, &e, 10); }
    for (int c = a; c <= b && c < CPU_SETSIZE; c++) out->push_back(c);
    p = e;
    if (*p == ',') p++;
  }
  return !out->empty();
}

// Pin the CALLING thread to exactly one core.  Returns false (with a warning)
// if the kernel refuses -- for example a core id that is not in the process's
// inherited mask.
inline bool PinCurrentThreadToCpu(int cpu) {
  if (cpu < 0 || cpu >= CPU_SETSIZE) return false;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
    fprintf(stderr, "[WARN] pthread_setaffinity_np(core %d) failed: %s\n", cpu, strerror(errno));
    return false;
  }
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




// ---------------------------------------------------------------------------
// Session: everything ONE inference worker owns.
//
// ONE Session == ONE RKNN context.  The board tolerates exactly TWO concurrent
// RKNN contexts (a third trips the 6 s job timeout and soft-resets the NPU),
// so the e2e demo runs two Sessions in two threads.  Every mutable buffer
// below is therefore per-Session and never shared between threads.
//
// The RGA canvas is ALSO per-Session: RGA is a single hardware engine, so two
// Sessions compositing at the same time serialise inside the driver, but
// their canvases are independent buffers and no frame can be overwritten
// while another one is still being uploaded.
// ---------------------------------------------------------------------------
// Scratch for the back-projection stage.
//
// It belongs to whoever RUNS that stage, not to the Session: the stage may
// run on a different thread than the NPU that produced the heatmap, and two
// Sessions must never share it.  The library gives the post-processing
// service its own workspace per worker thread.
// ===== SPARSE CORE BEGIN =====
// ---------------------------------------------------------------------------
// 稀疏后处理（sparse back-projection）
// ---------------------------------------------------------------------------
//
// WHY
// ---
// dense 路径对每个 tile 都做「640x512 全图双线性上采样 + 全图 8 连通域」：
// 三个 tile 合计约 98 万次双线性求值 + 约 200 万次连通域像素访问，
// 板卡实测这一段 43.88 ms/帧，而且它在关键路径上（post_wait）。
// 但热图里真正过阈值的像素通常只有几百个 —— 其余的求值全是白做的。
//
// 数学依据
// --------
// 双线性值是 4 个源值的凸组合（权重非负、和为 1），所以在实数域里
// 「4 个源值全 < 阈值 ⇒ 该 dst 像素必然 < 阈值」。于是只需要：
//   STEP 1  扫源窗口，找出「可能过阈值」的像素
//   STEP 2  用逆映射把它们展开成 native 侧的候选矩形（位图）
//   STEP 3  只在候选位上做双线性求值，保留确实过阈值的像素（光栅序）
//   STEP 4  在这些像素上跑与 dense 完全相同的两遍 8 连通域
// 输出的 Comp 序列与 dense 逐字节一致。
//
// 浮点余量（最容易踩的坑）
// ----------------------
// 「凸组合 ⇒ 必然 < 阈值」在实数域成立，但 dense 侧是 FLOAT32 算完再和
// double 阈值比较；float32 的舍入有可能把「刚好低于阈值」的结果抬到阈值
// 之上（相对误差最大 6u，u = 2^-24）。所以候选集合不能用真实阈值构造，
// 必须用一个低一个安全余量的「源阈值」：
//     src_threshold = threshold - max(1e-9, 128u * threshold)
// 这样「4 个源值全 < src_threshold ⇒ 算出来的 dst 必然 < threshold」成立，
// 候选集合是「真正过阈值像素」的严格超集；而最终判定仍然走 dense 的
// 同一条 double 比较 ((double)value >= threshold)，所以输出一致。
//
// 另一个坑：dense 的比较是 ((double)float32值 >= (double)0.06)，
// 而 double(0.06) != float32(0.06)（float32(0.06) = 0.059999998658895493，
// 比 double(0.06) 小 1.34e-9），所以不能直接用 vcgeq_f32(v, 0.06f) 判。
// 正解见 SparseCeilF32()：取「不小于 src_threshold 的最小 float32」，
// 于是 float32 的比较与 double 的比较对任意 float32 v 完全等价。
// ===========================================================================

enum PostMode {
  kPostModeDense = 0,   // 全图双线性 + 全图连通域（默认，行为与原来完全一样）
  kPostModeSparse = 1,  // 稀疏：只求值「有可能过阈值」的 native 像素
  kPostModeVerify = 2,  // 自检：dense 与 sparse 同时跑并逐字段比对
  kPostModeGpu = 3,     // OpenCL 局部极大路径（Mali GPU）
};


// float32 的单位舍入 u = 2^-24。
constexpr double kSparseU = 5.9604644775390625e-08;
// 双线性链路里最多 6 次有效舍入（top/bot 各 2 乘 1 加，dst 1 乘 1 加），
// 相对误差 <= 6u ≈ 3.6e-7。取 128u 的相对余量（≈7.6e-6），
// 保证「4 个源值全 < src_threshold ⇒ 算出来的 dst 必然 < threshold」。
constexpr double kSparseURel = 128.0 * kSparseU;

// 候选位图：640*512 bit = 40 KB，每行 10 个 uint64。
constexpr int kSparseRowWords = kSrcW / 64;                 // 10
constexpr int kSparseCandWords = kSparseRowWords * kSrcH;   // 5120
// scratch 的初始容量（只增不减，稳态零分配）。
constexpr size_t kSparseCapFg = 1 << 16;       // 前景像素
constexpr size_t kSparseCapLabels = 1 << 16;   // 临时标号 / 并查集
constexpr size_t kSparseCapComps = 1 << 15;    // 组件统计
constexpr size_t kSparseCapCols = 1024;        // 一行里命中的列号（sw <= 640）

// 一个前景像素：native 坐标 + native 值 + 遍 1 的临时标号。
struct SparsePixel {
  int16_t x = 0;
  int16_t y = 0;
  float v = 0;        // 与 dense 的 native_map 里那个 float32 逐位相同
  int32_t label = 0;  // 遍 1 的临时标号
};

// 一个组件的统计量。累加器类型必须与 dense 完全一致：
// sx / sy / ssum 用 double，speak 用 float —— 否则 cx/cy/sum 的
// float 位模式会不一样。
struct SparseStat {
  int32_t cnt = 0;
  int32_t xmin = 0, xmax = 0, ymin = 0, ymax = 0;
  double sx = 0.0, sy = 0.0, ssum = 0.0;
  float speak = 0.0f;
};

// 源阈值：比真实阈值低一个浮点安全余量（理由见上面的分析）。
inline double SparseSrcThreshold(double threshold) {
  const double eps = std::max(1e-9, threshold * kSparseURel);
  return threshold - eps;
}

// ceil_f32(t)：不小于 t（按 double 比较）的最小 float32。
// 有了它，对任意 float32 v 都有
//     (v >= ceil_f32(t))  <=>  ((double)v >= t)
// 于是 NEON 的 vcgeq_f32 与 dense 的 double 比较完全等价。
inline float SparseCeilF32(double t) {
  float f = (float)t;
  if (std::isnan(f)) return f;
  if ((double)f < t) f = std::nextafterf(f, INFINITY);
  return f;
}

// 源列 j 会被哪些 native 列用到（闭区间 [lo, hi]）。
//
//   g(X) = floor((X+0.5)*sx_scale-0.5) 是 native 列 X 对应的源列下标
//   源列 j 以 ix0 的身份出现：X ∈ [A(j), B(j))
//   源列 j 以 ix1 的身份出现：X ∈ [A(j-1), B(j-1))，而 B(j-1) == A(j)，
//     两段正好首尾相接
//   => 合并后：X ∈ [A(j-1), B(j))
//   A(j-1) = (j-0.5)/sx_scale-0.5 ，B(j) = (j+1.5)/sx_scale-0.5
// 两端的夹取也自然被覆盖：j==0 时 ix<0 会被夹到 0（A(-1)<0 ⇒ lo=0），
// j==sw-1 时 ix+1 会被夹到 sw-1（B(sw-1)>dw-1 ⇒ hi=dw-1）。
inline void SparseSrcRange(double scale, int j, int dst_len, int* lo, int* hi) {
  const double a = ((double)j - 0.5) / scale - 0.5;   // A(j-1)
  const double b = ((double)j + 1.5) / scale - 0.5;   // B(j)
  int l = (int)std::ceil(a);
  int h = (int)std::ceil(b) - 1;
  if (l < 0) l = 0;
  if (h > dst_len - 1) h = dst_len - 1;
  if (h < l) h = l;
  *lo = l;
  *hi = h;
}

// 在位图的一行上把 [lo, hi] 置 1。
inline void SparseSetRange(uint64_t* row, int lo, int hi) {
  const int w0 = lo >> 6, w1 = hi >> 6;
  if (w0 == w1) {
    row[w0] |= (~0ULL << (lo & 63)) & (~0ULL >> (63 - (hi & 63)));
    return;
  }
  row[w0] |= (~0ULL << (lo & 63));
  for (int w = w0 + 1; w < w1; w++) row[w] = ~0ULL;
  row[w1] |= (~0ULL >> (63 - (hi & 63)));
}

// 扫窗口的一行，把「>= src_threshold」的列号写进 cols，返回个数。
//
// NEON 只做快速排除：整块 4 lane 都没命中时一个都不写；
// 命中的 lane 仍然用 double 精确复核（(double)v >= src_threshold），
// 与 dense 的判定完全一致。f32_thr 是 SparseCeilF32(src_threshold)，
// 所以 NEON 的比较与 double 的比较本身也是等价的（不会有漏判）。
inline int SparseScanRow(const float* row, int sw, float f32_thr, double src_threshold,
                         int* cols) {
  int n = 0;
  int x = 0;
  (void)f32_thr;   // 非 NEON 构建下用不到
#ifdef MOSAIC_HAVE_NEON
  const float32x4_t thr = vdupq_n_f32(f32_thr);
  for (; x + 4 <= sw; x += 4) {
    const float32x4_t v = vld1q_f32(row + x);
    const uint32x4_t m = vcgeq_f32(v, thr);
    if (vgetq_lane_u32(m, 0)) {
      const float vv = vgetq_lane_f32(v, 0);
      if ((double)vv >= src_threshold) cols[n++] = x + 0;
    }
    if (vgetq_lane_u32(m, 1)) {
      const float vv = vgetq_lane_f32(v, 1);
      if ((double)vv >= src_threshold) cols[n++] = x + 1;
    }
    if (vgetq_lane_u32(m, 2)) {
      const float vv = vgetq_lane_f32(v, 2);
      if ((double)vv >= src_threshold) cols[n++] = x + 2;
    }
    if (vgetq_lane_u32(m, 3)) {
      const float vv = vgetq_lane_f32(v, 3);
      if ((double)vv >= src_threshold) cols[n++] = x + 3;
    }
  }
#endif
  for (; x < sw; x++) {
    const float vv = row[x];
    if ((double)vv >= src_threshold) cols[n++] = x;
  }
  return n;
}

// float32 的逐位比较（区分 +0.0 / -0.0，NaN 也按位模式比）。
inline bool SparseSameF32(float a, float b) {
  uint32_t ua = 0, ub = 0;
  memcpy(&ua, &a, sizeof(ua));
  memcpy(&ub, &b, sizeof(ub));
  return ua == ub;
}

struct PostWorkspace {
  std::vector<float> native_map;
  CcWorkspace cc;
  std::vector<Comp> comps;

  // ---- 稀疏路径的 scratch（InitPostWorkspace 里一次性分配，运行时复用）----
  int post_mode = kPostModeDense;             // 0=dense 1=sparse 2=verify
  std::vector<uint64_t> sp_cand;               // 候选位图 640*512 bit = 40 KB
  std::vector<int32_t> sp_labels;              // 相对 bbox 的临时标号数组
  std::vector<SparsePixel> sp_fg;              // 前景像素（光栅序）
  std::vector<int> sp_parent;                  // 并查集 parent（1-based，0 号不用）
  std::vector<int> sp_root_final;              // root -> 最终标号
  std::vector<SparseStat> sp_stats;            // 每组件的统计量
  std::vector<int> sp_cols;                    // 一行里命中的列号

  // ---- 自检模式（post_mode == kPostModeVerify）的状态，外部可读 ----
  std::vector<Comp> verify_dense;              // dense 的结果（对照基准）
  std::vector<Comp> verify_sparse;             // sparse 的结果
  long verify_frames = 0;                      // 累计比对的帧数
  long verify_mismatch_count = 0;              // 累计不一致次数（外部可读）
  int verify_first_diff_printed = 0;           // 只打印第一处差异，避免刷屏

  // ---- GPU 路径的分段耗时（post_mode == kPostModeGpu）----
  double gpu_upload_ms = 0.0;
  double gpu_kernel_ms = 0.0;
  double gpu_readback_ms = 0.0;
  double gpu_total_ms = 0.0;
  long gpu_frames = 0;
  long gpu_peaks = 0;
  long gpu_peaks_dropped = 0;
  int gpu_fallback_warned = 0;                 // 只警告一次
  // 逐帧分段采样（算方差 / 分布用；稳态不分配，reserve 过了）
  std::vector<double> gpu_upload_samples;
  std::vector<double> gpu_kernel_samples;
  std::vector<double> gpu_readback_samples;
  std::vector<double> gpu_total_samples;
};

// Size the back-projection scratch.  The Session gets one of these from
// InitSession; a service that runs the stage on its own thread needs its own
// (the scratch belongs to the stage, not to the session).
inline void InitPostWorkspace(PostWorkspace* ws) {
  if (!ws) return;
  ws->native_map.assign((size_t)kSrcW * kSrcH, 0.0f);
  // 稀疏路径的 scratch：一次性分配好，运行时不再 malloc / new。
  ws->sp_cand.assign(kSparseCandWords, 0ULL);
  ws->sp_labels.assign((size_t)kSrcW * kSrcH, 0);
  ws->sp_fg.reserve(kSparseCapFg);
  ws->sp_parent.reserve(kSparseCapLabels);
  ws->sp_root_final.reserve(kSparseCapLabels);
  ws->sp_stats.reserve(kSparseCapComps);
  ws->sp_cols.assign(kSparseCapCols, 0);
  ws->verify_dense.reserve(1024);
  ws->verify_sparse.reserve(1024);
}

// ---------------------------------------------------------------------------
// SparseBackProjectTile: 稀疏后处理一个 tile。
//
// 输出的 Comp 序列与 dense 的 DetectComponents 逐字段一致：
//   * mask 判定用同一条 double 比较 ((double)value >= threshold)
//   * 临时标号按光栅序发放、并查集规则复用 UfFind/UfUnite
//   * 最终标号按 root 升序编号（== cv2 的 label 序）
//   * 遍 2 的统计累加也在光栅序上做，累加器类型与 dense 相同
//   * 过滤 / 排序 / 截断与 dense 相同
// ---------------------------------------------------------------------------
inline void SparseBackProjectTile(const float* src, int src_stride, int sw, int sh,
                                  double threshold, int min_area, int limit, PostWorkspace* ws,
                                  std::vector<Comp>* out) {
  out->clear();
  if (sw <= 0 || sh <= 0) return;

  // 兜底：workspace 没经过 InitPostWorkspace 时自建 scratch（只可能发生在
  // 测试代码里；正常路径下这一段永远不成立，因此没有运行期分配）。
  if (ws->sp_cand.size() != (size_t)kSparseCandWords) {
    ws->sp_cand.assign(kSparseCandWords, 0ULL);
    ws->sp_labels.assign((size_t)kSrcW * kSrcH, 0);
    ws->sp_fg.reserve(kSparseCapFg);
    ws->sp_parent.reserve(kSparseCapLabels);
    ws->sp_root_final.reserve(kSparseCapLabels);
    ws->sp_stats.reserve(kSparseCapComps);
  }
  if (ws->sp_cols.size() < (size_t)sw) ws->sp_cols.assign((size_t)sw, 0);

  const double src_threshold = SparseSrcThreshold(threshold);
  const float f32_thr = SparseCeilF32(src_threshold);
  const double sx_scale = (double)sw / (double)kSrcW;
  const double sy_scale = (double)sh / (double)kSrcH;

  // ---- STEP 1 + STEP 2：源窗口扫描 -> native 候选位图 ----------------------
  uint64_t* cand = ws->sp_cand.data();
  memset(cand, 0, sizeof(uint64_t) * (size_t)kSparseCandWords);
  int bx0 = kSrcW, by0 = kSrcH, bx1 = -1, by1 = -1;

  int* cols = ws->sp_cols.data();
  for (int hy = 0; hy < sh; hy++) {
    const float* row = src + (size_t)hy * src_stride;
    const int ncols = SparseScanRow(row, sw, f32_thr, src_threshold, cols);
    if (ncols == 0) continue;
    int ylo = 0, yhi = 0;
    SparseSrcRange(sy_scale, hy, kSrcH, &ylo, &yhi);
    // 同一行里相邻的命中列，其 native 区间也是相邻的 -> 合并成段，
    // 减少位图写入次数（长条斑块时效果明显）。
    int run_lo = -1, run_hi = -1;
    for (int i = 0; i < ncols; i++) {
      int xlo = 0, xhi = 0;
      SparseSrcRange(sx_scale, cols[i], kSrcW, &xlo, &xhi);
      if (run_lo < 0) {
        run_lo = xlo;
        run_hi = xhi;
        continue;
      }
      if (xlo <= run_hi + 1) {
        if (xhi > run_hi) run_hi = xhi;
        continue;
      }
      // flush 上一段（bbox 要用被写入的那一段 run_lo/run_hi）
      if (run_lo < bx0) bx0 = run_lo;
      if (run_hi > bx1) bx1 = run_hi;
      if (ylo < by0) by0 = ylo;
      if (yhi > by1) by1 = yhi;
      for (int y = ylo; y <= yhi; y++) SparseSetRange(cand + (size_t)y * kSparseRowWords, run_lo, run_hi);
      run_lo = xlo;
      run_hi = xhi;
    }
    if (run_lo >= 0) {
      if (run_lo < bx0) bx0 = run_lo;
      if (run_hi > bx1) bx1 = run_hi;
      if (ylo < by0) by0 = ylo;
      if (yhi > by1) by1 = yhi;
      for (int y = ylo; y <= yhi; y++) SparseSetRange(cand + (size_t)y * kSparseRowWords, run_lo, run_hi);
    }
  }
  if (bx1 < bx0) return;   // 没有任何候选 -> 没有前景

  // ---- STEP 3：在候选位里按光栅序求值 ------------------------------------
  const int bw = bx1 - bx0 + 1;
  const int bh = by1 - by0 + 1;
  // 标号数组（相对 bbox）先清零；只清 bbox 区域，不是全图。
  memset(ws->sp_labels.data(), 0, sizeof(int32_t) * (size_t)bw * (size_t)bh);

  std::vector<SparsePixel>& fg = ws->sp_fg;
  fg.clear();
  const int w0 = bx0 >> 6, w1 = bx1 >> 6;
  for (int y = by0; y <= by1; y++) {
    // 与 ResizeBilinearF32 逐字一致的每行常量
    const double fy = (y + 0.5) * sy_scale - 0.5;
    const int sy = (int)std::floor(fy);
    const float ay = (float)(fy - sy);
    const int y0 = std::max(0, std::min(sh - 1, sy));
    const int y1 = std::max(0, std::min(sh - 1, sy + 1));
    const float* r0 = src + (size_t)y0 * src_stride;
    const float* r1 = src + (size_t)y1 * src_stride;
    const uint64_t* crow = cand + (size_t)y * kSparseRowWords;
    for (int w = w0; w <= w1; w++) {
      uint64_t bits = crow[w];
      if (w == w0) bits &= ~0ULL << (bx0 & 63);
      if (w == w1) bits &= ~0ULL >> (63 - (bx1 & 63));
      while (bits) {
        const int b = __builtin_ctzll(bits);
        bits &= bits - 1;
        const int x = (w << 6) + b;
        // ---- 与 ResizeBilinearF32 逐字一致的求值 ----
        const double fx = (x + 0.5) * sx_scale - 0.5;
        const int sx = (int)std::floor(fx);
        const float ax = (float)(fx - sx);
        const int ix0 = std::max(0, std::min(sw - 1, sx));
        const int ix1 = std::max(0, std::min(sw - 1, sx + 1));
        const float top = r0[ix0] * (1.0f - ax) + r0[ix1] * ax;
        const float bot = r1[ix0] * (1.0f - ax) + r1[ix1] * ax;
        const float v = top * (1.0f - ay) + bot * ay;
        // dense 的判定：((double)hm[idx] >= threshold)
        if (!((double)v >= threshold)) continue;
        SparsePixel p;
        p.x = (int16_t)x;
        p.y = (int16_t)y;
        p.v = v;
        p.label = 0;
        fg.push_back(p);
      }
    }
  }
  if (fg.empty()) return;

  // ---- STEP 4a：遍 1，临时标号（8 连通，光栅序） ---------------------------
  // 与 dense 完全一样的规则：邻居只看 left / up / up-left / up-right
  // （up-right 在光栅序里更靠前，已经处理过）。
  ws->sp_parent.clear();
  ws->sp_parent.push_back(0);   // 0 号不用（与 dense 的 parent.assign(1,0) 等价）
  int nlab = 0;
  for (size_t i = 0; i < fg.size(); i++) {
    const int X = (int)fg[i].x;
    const int Y = (int)fg[i].y;
    const int lx = X - bx0, ly = Y - by0;
    int nb[4];
    int nn = 0;
    if (X > bx0) nb[nn++] = ly * bw + (lx - 1);            // left
    if (Y > by0) {
      nb[nn++] = (ly - 1) * bw + lx;                        // up
      if (X > bx0) nb[nn++] = (ly - 1) * bw + (lx - 1);     // up-left
      if (X < bx1) nb[nn++] = (ly - 1) * bw + (lx + 1);     // up-right
    }
    int lbl = 0;
    for (int k = 0; k < nn; k++) {
      const int l = ws->sp_labels[(size_t)nb[k]];
      if (l == 0) continue;
      if (lbl == 0 || l < lbl) lbl = l;
    }
    if (lbl == 0) {
      lbl = ++nlab;
      ws->sp_parent.push_back(lbl);   // parent[lbl] == lbl
    }
    fg[i].label = lbl;
    ws->sp_labels[(size_t)(ly * bw + lx)] = lbl;
    for (int k = 0; k < nn; k++) {
      const int l = ws->sp_labels[(size_t)nb[k]];
      if (l != 0 && l != lbl) UfUnite(&ws->sp_parent, lbl, l);
    }
  }
  if (nlab == 0) return;

  // ---- 最终标号：按 root 升序编号（== cv2 的 label 序） -------------------
  if (ws->sp_root_final.size() < (size_t)nlab + 1) ws->sp_root_final.resize((size_t)nlab + 1);
  int ncomp = 0;
  for (int l = 1; l <= nlab; l++) {
    if (UfFind(&ws->sp_parent, l) == l) ws->sp_root_final[(size_t)l] = ++ncomp;
  }

  // ---- STEP 4b：遍 2，按光栅序统计 ----------------------------------------
  if (ws->sp_stats.size() < (size_t)ncomp + 1) ws->sp_stats.resize((size_t)ncomp + 1);
  for (int f = 1; f <= ncomp; f++) {
    SparseStat& st = ws->sp_stats[(size_t)f];
    st.cnt = 0;
    st.xmin = kSrcW;
    st.xmax = -1;
    st.ymin = kSrcH;
    st.ymax = -1;
    st.sx = 0.0;
    st.sy = 0.0;
    st.ssum = 0.0;
    st.speak = -1.0e30f;
  }
  for (size_t i = 0; i < fg.size(); i++) {
    const int l = fg[i].label;
    const int f = ws->sp_root_final[(size_t)UfFind(&ws->sp_parent, l)];
    SparseStat& st = ws->sp_stats[(size_t)f];
    const int X = (int)fg[i].x;
    const int Y = (int)fg[i].y;
    st.cnt++;
    st.sx += (double)X;
    st.sy += (double)Y;
    st.ssum += (double)fg[i].v;
    if (fg[i].v > st.speak) st.speak = fg[i].v;
    if (X < st.xmin) st.xmin = X;
    if (X > st.xmax) st.xmax = X;
    if (Y < st.ymin) st.ymin = Y;
    if (Y > st.ymax) st.ymax = Y;
  }

  // ---- 过滤 / 排序 / 截断：与 dense 一致 ---------------------------------
  out->reserve((size_t)ncomp);
  for (int f = 1; f <= ncomp; f++) {
    const SparseStat& st = ws->sp_stats[(size_t)f];
    if (st.cnt < min_area) continue;
    Comp c;
    c.cx = (float)(st.sx / (double)st.cnt);
    c.cy = (float)(st.sy / (double)st.cnt);
    c.w = st.xmax - st.xmin + 1;
    c.h = st.ymax - st.ymin + 1;
    c.area = st.cnt;
    c.peak = st.speak;
    c.sum = (float)st.ssum;
    out->push_back(c);
  }
  // 排序：按 sum 降序，且**稳定**（dense 侧用的是 std::stable_sort）。
  //
  // 为什么不用 std::stable_sort：它会为归并**分配一块 O(n) 的临时缓冲**，
  // 那是稀疏路径稳态下**唯一剩下的一次 malloc**
  // （host 端实测：dense 每次调用 10 次分配，sparse 1 次 —— 那 1 次就是它）。
  // 而这里的候选数极小（实测每帧 2~4 个，上限 max_detections=100），
  // 插入排序的 O(n^2) 完全不是问题。
  //
  // 稳定性：只有当「前一个.sum < 当前.sum」时才往后挪 ⇒ **相等时保持原有相对顺序**，
  // 与 stable_sort 一致 ⇒ 输出逐字节相同。
  //
  // n 很大（病态帧，几千个碎块）时退回 stable_sort，避免 O(n^2)。
  if (out->size() <= 64) {
    for (size_t i = 1; i < out->size(); i++) {
      const Comp key = (*out)[i];
      size_t j = i;
      while (j > 0 && (*out)[j - 1].sum < key.sum) {
        (*out)[j] = (*out)[j - 1];
        j--;
      }
      (*out)[j] = key;
    }
  } else {
    std::stable_sort(out->begin(), out->end(),
                     [](const Comp& a, const Comp& b) { return a.sum > b.sum; });
  }
  if ((int)out->size() > limit) out->resize((size_t)limit);
}

// 自检模式：逐字段比对 dense 与 sparse 的结果。
// 不一致就累加计数，并打印第一处差异（tile、comps 下标、字段名、两边的值）。
inline void VerifyCompareComps(int tile_tag, const std::vector<Comp>& dense,
                               const std::vector<Comp>& sparse, PostWorkspace* ws) {
  const size_t n = dense.size() > sparse.size() ? dense.size() : sparse.size();
  for (size_t i = 0; i < n; i++) {
    if (i >= dense.size() || i >= sparse.size()) {
      ws->verify_mismatch_count++;
      if (!ws->verify_first_diff_printed) {
        ws->verify_first_diff_printed = 1;
        fprintf(stderr,
                "[VERIFY][MISMATCH] tile=%d comp[%zu] field=count dense=%zu sparse=%zu\n", tile_tag,
                i, dense.size(), sparse.size());
      }
      continue;
    }
    const Comp& a = dense[i];
    const Comp& b = sparse[i];
    const char* field = nullptr;
    if (a.tag != b.tag) field = "tag";
    else if (!SparseSameF32(a.cx, b.cx)) field = "cx";
    else if (!SparseSameF32(a.cy, b.cy)) field = "cy";
    else if (a.w != b.w) field = "w";
    else if (a.h != b.h) field = "h";
    else if (a.area != b.area) field = "area";
    else if (!SparseSameF32(a.peak, b.peak)) field = "peak";
    else if (!SparseSameF32(a.sum, b.sum)) field = "sum";
    if (!field) continue;
    ws->verify_mismatch_count++;
    if (!ws->verify_first_diff_printed) {
      ws->verify_first_diff_printed = 1;
      uint32_t bits_cx = 0, bits_sx = 0, bits_cy = 0, bits_sy = 0, bits_sum = 0, bits_ssum = 0;
      memcpy(&bits_cx, &a.cx, 4);
      memcpy(&bits_sx, &b.cx, 4);
      memcpy(&bits_cy, &a.cy, 4);
      memcpy(&bits_sy, &b.cy, 4);
      memcpy(&bits_sum, &a.sum, 4);
      memcpy(&bits_ssum, &b.sum, 4);
      fprintf(stderr,
              "[VERIFY][MISMATCH] tile=%d comp[%zu] field=%s "
              "dense=(tag=%d cx=%.9g/0x%08x cy=%.9g/0x%08x w=%d h=%d area=%d peak=%.9g sum=%.9g/0x%08x) "
              "sparse=(tag=%d cx=%.9g/0x%08x cy=%.9g/0x%08x w=%d h=%d area=%d peak=%.9g sum=%.9g/0x%08x)\n",
              tile_tag, i, field, a.tag, (double)a.cx, bits_cx, (double)a.cy, bits_cy, a.w, a.h,
              a.area, (double)a.peak, (double)a.sum, bits_sum, b.tag, (double)b.cx, bits_sx,
              (double)b.cy, bits_sy, b.w, b.h, b.area, (double)b.peak, (double)b.sum, bits_ssum);
    }
  }
}
// ===== SPARSE CORE END =====

struct Session {
  Options opt;
  std::vector<Tile> tiles;
  bool have_tiles = false;

  NpuCtx npu;
  RgaBuf canvas, pat, stage;
  bool use_staging = false;
  bool verified = false;

  // Detection thresholds.  They default to the frozen constants so the
  // BMP-driven binary keeps its exact behaviour; the library may override
  // them through the public C ABI.
  double region_threshold = kRegionThreshold;
  int min_area = kMinArea;
  int max_detections = kMaxDet;

  std::vector<float> hm;
  std::vector<rknn_output> outs;
  PostWorkspace post;
  std::vector<uint16_t> f16_tmp;
  // The transpose scratch for the NCHW upload.  It is MUTABLE because the
  // widening helper takes a const Session (the pre-process stage only ever
  // reads the canvas) and still needs a scratch buffer for the NCHW path.
  mutable std::vector<uint16_t> f16_nchw;   // transpose scratch for the NCHW upload
};

// ---------------------------------------------------------------------------
// GPU 后处理的接入点（注册钩子，不用反向 include）
//
// mosaic_post_gpu.h include 了本文件（它要用 Session / Comp），
// 所以本文件**不能** include mosaic_post_gpu.h（会成环）。
// 于是这里只声明一个函数指针类型 + 一个 inline 变量；
// 真正实现 GPU 路径的编译单元（mosaic_pipeline.cpp）在 create() 时把它注册进来。
// 没注册时 GPU 模式会回退 dense 并打一行警告，不会崩。
// ---------------------------------------------------------------------------
struct GpuPostTiming {
  double upload_ms = 0;    // 热图 -> GPU buffer（含计数器清零）
  double kernel_ms = 0;    // 派发 -> clFinish() 返回
  double readback_ms = 0;  // 阻塞回读候选点
  double total_ms = 0;     // 整个调用（含 CPU 侧反投影组装）
  int peaks = 0;           // 通过过滤的候选数
  int peaks_dropped = 0;   // 超过 max_peaks 被丢掉的局部极大
};

using GpuPostHook = bool (*)(const Session*, const std::vector<float>&, std::vector<Comp>*,
                             GpuPostTiming*);

// C++17 的 inline 变量：每个 TU 看到的是同一个对象。
inline GpuPostHook g_gpu_post_hook = nullptr;

// The tile layout.  Shared by both entry points, so the canvas can never be
// composed with a different tile set than the one the model was checked
// against.  VERBATIM from the original InitSession.
inline bool LayoutTiles(Session* s, CanvasMode mode, const std::vector<int>& only_tiles,
                        bool have_tiles) {
  if (mode == CanvasMode::kMosaic) {
    // tag, x, y, w, h
    s->tiles.push_back({0, 0, 0, 640, 512, true});      // native, 1:1
    s->tiles.push_back({1, 0, 512, 160, 128, true});    // thumb160, 1/4 down
    s->tiles.push_back({2, 160, 512, 80, 64, true});    // thumb80, 1/8 down
  } else {
    s->tiles.push_back({0, 0, 64, 640, 512, true});     // native letterbox replica
  }
  if (have_tiles) {
    for (Tile& t : s->tiles) {
      bool want = false;
      for (int sel : only_tiles) if (sel == t.tag) want = true;
      t.enabled = want;
    }
  }
  bool any_tile = false;
  for (const Tile& t : s->tiles) if (t.enabled) any_tile = true;
  if (!any_tile) {
    fprintf(stderr, "[FATAL] --tiles selected no tile at all\n");
    return false;
  }
  return true;
}

// The RGA buffers and the canvas background.  Shared by both entry points.
// VERBATIM from the original InitSession.
inline bool AllocCanvasBuffers(Session* s) {
  if (!BufAlloc(&s->canvas, kCanvas, kCanvas, kRgaFmt) || !BufAlloc(&s->pat, 4, 4, kRgaFmt) ||
      !BufAlloc(&s->stage, kSrcW, kSrcH, kRgaFmt)) {
    fprintf(stderr, "[FATAL] RGA buffer allocation failed\n");
    return false;
  }
  if (!CheckBudget(s->canvas, "canvas") || !CheckBudget(s->pat, "pat") ||
      !CheckBudget(s->stage, "stage")) {
    return false;
  }
  // The canvas background is frame-invariant -> filled ONCE.  imfill() only
  // supports RGBA, so the RGB_888 canvas is filled with memset.
  memset(s->canvas.data, kPad, s->canvas.size);
  return true;
}

// ---------------------------------------------------------------------------
// InitCanvasOnly: the CANVAS half of InitSession, WITHOUT an RKNN context.
//
// WHY: the decoupled pipeline runs the pre-process stage (RGA compose + the
// u8->fp16 widening) in its own thread, and that thread owns its own canvas.
// The board tolerates exactly TWO RKNN contexts, so the pre-process stage may
// NOT open a third one -- it only needs somewhere to compose into.
//
// It lays out the tiles, refuses an empty tile selection, allocates the RGA
// buffers and fills the canvas background.  It NEVER touches s->npu, so the
// session it returns has no context and must never be handed to PrepareFrame
// or RunFrame (those do touch s->npu).
// ---------------------------------------------------------------------------
inline bool InitCanvasOnly(Session* s, const std::string& model_path, CanvasMode mode,
                            const std::vector<int>& only_tiles, bool have_tiles) {
  if (!s) return false;
  s->opt.model = model_path;
  s->opt.canvas_mode = mode;
  s->opt.tiles = only_tiles;
  s->have_tiles = have_tiles;
  if (!LayoutTiles(s, mode, only_tiles, have_tiles)) return false;
  if (!AllocCanvasBuffers(s)) return false;
  return true;
}

// ---------------------------------------------------------------------------
// InitSession: load the model, open the RKNN context, allocate the RGA
// buffers, fill the canvas background and lay out the tiles.
//
// The body is the code that used to live inline in main(); it is moved here
// verbatim so the two binaries cannot drift.
//
// v10: the body is split, but InitSession KEEPS ITS ORIGINAL SEQUENCE AND
// ITS ORIGINAL OUTPUT.  The canvas work lives in LayoutTiles() and
// AllocCanvasBuffers() so that InitCanvasOnly() can share it, and the two
// helpers are called from exactly the places the inline code used to be:
// the tiles before the NPU context is opened, the RGA buffers after the
// input tensor is bound.  A reader of the log sees the same lines in the
// same order as before.
// ---------------------------------------------------------------------------
bool InitSession(Session* s, const std::string& model_path, CanvasMode mode,
                 const std::vector<int>& only_tiles, bool have_tiles, InputMode im) {
  s->opt.model = model_path;
  s->opt.canvas_mode = mode;
  s->opt.input_mode = im;
  s->opt.tiles = only_tiles;
  s->have_tiles = have_tiles;

  // ---- tiles --------------------------------------------------------------
  if (!LayoutTiles(s, mode, only_tiles, have_tiles)) return false;

  // ---- NPU ----------------------------------------------------------------
  std::vector<uint8_t> model;
  if (!read_file(model_path.c_str(), &model)) {
    fprintf(stderr, "[FATAL] cannot read model %s\n", model_path.c_str());
    return false;
  }
  if (rknn_init(&s->npu.ctx, model.data(), (uint32_t)model.size(), 0, nullptr) != 0) {
    fprintf(stderr, "[FATAL] rknn_init failed\n");
    return false;
  }
  {
    rknn_sdk_version ver;
    memset(&ver, 0, sizeof(ver));
    if (rknn_query(s->npu.ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver)) == 0)
      fprintf(stderr, "[i] rknn runtime api=%s driver=%s\n", ver.api_version, ver.drv_version);
  }
  rknn_input_output_num io;
  memset(&io, 0, sizeof(io));
  rknn_query(s->npu.ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
  if (io.n_input != 1 || io.n_output < 1) {
    fprintf(stderr, "[FATAL] expected 1 input / >=1 outputs, got %u / %u\n", io.n_input, io.n_output);
    return false;
  }
  rknn_tensor_attr in_attr;
  memset(&in_attr, 0, sizeof(in_attr));
  in_attr.index = 0;
  rknn_query(s->npu.ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
  // Keep the queried input attribute.  The decoupled pipeline binds a
  // DIFFERENT buffer to this context for every frame (a Buffer A slot) with
  // rknn_set_io_mem(), and that call needs the attribute.
  s->npu.in_attr = in_attr;
  if (in_attr.fmt == RKNN_TENSOR_NCHW) {
    s->npu.in_c = (int)in_attr.dims[1];
    s->npu.in_h = (int)in_attr.dims[2];
    s->npu.in_w = (int)in_attr.dims[3];
  } else {
    s->npu.in_h = (int)in_attr.dims[1];
    s->npu.in_w = (int)in_attr.dims[2];
    s->npu.in_c = (int)in_attr.dims[3];
  }
  s->npu.input_is_f16 = (in_attr.type == RKNN_TENSOR_FLOAT16);
  s->npu.input_is_f32 = (in_attr.type == RKNN_TENSOR_FLOAT32);
  s->npu.input_is_int8 = (in_attr.type == RKNN_TENSOR_INT8);
  s->npu.input_nchw = (in_attr.fmt == RKNN_TENSOR_NCHW);
  s->npu.input_zp = in_attr.zp;
  fprintf(stderr, "[i] input %dx%dx%d fmt=%s type=%s zp=%d scale=%.8f\n", s->npu.in_h, s->npu.in_w,
          s->npu.in_c, tensor_fmt_name(in_attr.fmt), tensor_type_name(in_attr.type), in_attr.zp,
          (double)in_attr.scale);
  if (s->npu.in_h != kCanvas || s->npu.in_w != kCanvas || s->npu.in_c != kSrcC) {
    fprintf(stderr, "[FATAL] model input is %dx%dx%d, expected %dx%dx%d\n", s->npu.in_h, s->npu.in_w,
            s->npu.in_c, kCanvas, kCanvas, kSrcC);
    return false;
  }
  s->npu.n_out = io.n_output;
  s->npu.out_attr.assign(s->npu.n_out, rknn_tensor_attr{});
  for (uint32_t i = 0; i < s->npu.n_out; i++) {
    memset(&s->npu.out_attr[i], 0, sizeof(rknn_tensor_attr));
    s->npu.out_attr[i].index = i;
    rknn_query(s->npu.ctx, RKNN_QUERY_OUTPUT_ATTR, &s->npu.out_attr[i], sizeof(s->npu.out_attr[i]));
    fprintf(stderr, "[i] out[%u] dims=[", i);
    for (uint32_t d = 0; d < s->npu.out_attr[i].n_dims; d++)
      fprintf(stderr, "%u%s", s->npu.out_attr[i].dims[d], d + 1 < s->npu.out_attr[i].n_dims ? "," : "");
    fprintf(stderr, "] fmt=%s type=%s\n", tensor_fmt_name(s->npu.out_attr[i].fmt),
            tensor_type_name(s->npu.out_attr[i].type));
  }
  {
    const rknn_tensor_attr& a = s->npu.out_attr[0];
    if (a.n_dims != 4 || a.dims[0] != 1 || a.dims[1] != 1) {
      fprintf(stderr, "[FATAL] out[0] is not [1,1,H,W]\n");
      return false;
    }
    s->npu.hm_h = (int)a.dims[2];
    s->npu.hm_w = (int)a.dims[3];
  }
  fprintf(stderr, "[i] heatmap grid %dx%d (canvas %dx%d, stride %d)\n", s->npu.hm_w, s->npu.hm_h,
          kCanvas, kCanvas, kCanvas / s->npu.hm_w);
  for (const Tile& t : s->tiles) {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    HeatRect(t, s->npu.hm_w, s->npu.hm_h, &x0, &y0, &x1, &y1);
    fprintf(stderr,
            "[i] tile tag=%d canvas(%d,%d,%d,%d) -> heatmap rows[%d,%d) cols[%d,%d)  %dx\n", t.tag,
            t.x, t.y, t.w, t.h, y0, y1, x0, x1, (x1 - x0) * kSrcW / (t.w ? t.w : 1));
  }

  // ---- zero-copy input buffer (optional) ----------------------------------
  // Only the two io-mem modes bind the input tensor to host memory.
  // kSetF16 is NOT an io-mem mode: it still goes through rknn_inputs_set,
  // we just hand it a FLOAT16 buffer instead of a uint8 one.
  if (im == InputMode::kIoMemNchw || im == InputMode::kIoMemNhwc) {
    const uint32_t mem_sz = in_attr.size_with_stride ? in_attr.size_with_stride : in_attr.size;
    s->npu.io_mem = rknn_create_mem(s->npu.ctx, mem_sz);
    if (!s->npu.io_mem) {
      fprintf(stderr, "[FATAL] rknn_create_mem(%u) failed\n", mem_sz);
      return false;
    }
    if (rknn_set_io_mem(s->npu.ctx, s->npu.io_mem, &in_attr) != 0) {
      fprintf(stderr, "[FATAL] rknn_set_io_mem failed\n");
      return false;
    }
    fprintf(stderr, "[i] io-mem input bound: %p, %u B (attr size %u)\n", s->npu.io_mem->virt_addr,
            mem_sz, in_attr.size);
  }

  // ---- RGA buffers --------------------------------------------------------
  if (!AllocCanvasBuffers(s)) return false;

  // InitPostWorkspace 会把 native_map 和稀疏路径的 scratch 一次性分配好
  // （与原来只分配 native_map 等价，多出来的只是稀疏路径的预分配 scratch）。
  InitPostWorkspace(&s->post);
  return true;
}

void ReleaseSession(Session* s) {
  if (!s) return;
  BufFree(&s->canvas);
  BufFree(&s->pat);
  BufFree(&s->stage);
  if (s->npu.io_mem) rknn_destroy_mem(s->npu.ctx, s->npu.io_mem);
  if (s->npu.ctx) rknn_destroy(s->npu.ctx);
  s->npu.ctx = 0;
  s->npu.io_mem = nullptr;
}

// Per-frame cost breakdown of ProcessFrame.
struct FrameOut {
  std::vector<Comp> comps;
  double rga_import = 0;
  double rga_improcess = 0;
  double upload = 0;
  double run = 0;
  double outputs = 0;
  double backproj = 0;
};

// ---------------------------------------------------------------------------
// ProcessFrame: the whole per-frame inference path.
//
//   in : `rgb` holds kSrcW*kSrcH*kSrcC bytes in the MODEL's channel order
//        [residual, diff, I_t] (see the pixel contract below).
//        This is byte-identical to what stb_image returns for the BMP feature
//        files, so the BMP-driven binary and this one see the same numbers.
//
//   out: the tagged connected components of every enabled tile.
//
// PIXEL CONTRACT (unchanged from the BMP path): the feature BMP stores BGR,
// stb returns RGB, and that RGB byte order IS the PC side's
// model_order_bmp() == cv2.imread(bmp)[..., ::-1], i.e. the model's channel
// order [residual, diff, I_t].  No flip here.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// The three pipeline STAGES.
//
// The frame path is deliberately split into three services so that each one can
// be scheduled -- and measured -- on its own:
//
//   PrepareFrame   RGA compose into the canvas + upload into the ctx input
//   RunFrame       rknn_run + outputs_get + dequant   (the NPU service)
//   BackProject    bilinear back-projection + 8-connected components
//
// ProcessFrame below is the three of them back to back, which is what the
// BMP-driven binary needs; the library drives the stages separately so the
// post-processing can run while the NPU is already busy with the next frame.
//
// The scratch the back-projection needs (the native-resolution map and the
// connected-component workspace) belongs to the STAGE, not to the Session:
// the stage may run on a different thread than the NPU that produced the
// heatmap, and two Sessions must never share it.
// ---------------------------------------------------------------------------

// ---- stage 1a: RGA compose (NO NPU ACCESS) ---------------------------------
//
// v10: this is the first half of the old PrepareFrame, split out so the
// decoupled pipeline can run the composition in the PRE-PROCESS thread.  That
// thread owns a canvas-only session (no RKNN context), so this function must
// NOT touch s->npu -- it does not, and it never did: the old code only
// reached s->npu in the upload section, which stayed in PrepareFrame.
//
// The body below is the old section 1, VERBATIM.
bool ComposeFrame(Session* s, const uint8_t* rgb, int w, int h, FrameOut* out) {
  if (w != kSrcW || h != kSrcH) {
    fprintf(stderr, "[FATAL] expected %dx%d, got %dx%d\n", kSrcW, kSrcH, w, h);
    return false;
  }

  // ---- 1. RGA: import + improcess ----------------------------------------
  const uint8_t* src_pixels = rgb;
  if (s->use_staging) {
    memcpy(s->stage.data, rgb, s->stage.size);
    src_pixels = s->stage.data;
  }
  bool rga_ok = true;
  for (int attempt = 0; attempt < 2 && rga_ok; attempt++) {
    const double i0 = now_ms();
    // The source buffer must be imported AFTER its content is final.
    const rga_buffer_handle_t src_handle =
        importbuffer_virtualaddr(const_cast<uint8_t*>(src_pixels), kSrcW, kSrcH, kRgaFmt);
    const bool canvas_ok = BufImport(&s->canvas);
    const double i1 = now_ms();
    out->rga_import = i1 - i0;
    if (src_handle <= 0 || !canvas_ok) {
      fprintf(stderr, "[FATAL] RGA import failed (src=%d dst=%d)\n", (int)(src_handle > 0),
              (int)canvas_ok);
      if (src_handle > 0) releasebuffer_handle(src_handle);
      rga_ok = false;
      break;
    }
    const rga_buffer_t srcb = wrapbuffer_handle(src_handle, kSrcW, kSrcH, kRgaFmt);

    const double m0 = now_ms();
    // NOTE: EVERY tile is composed into the canvas, exactly as the BMP-driven
    // binary did.  A disabled tile is merely not READ BACK by the
    // back-projection, and the two tiles occupy disjoint canvas regions, so
    // composing it or not cannot change a detection -- but keeping the call
    // identical keeps the timing buckets and the canvas bytes identical too.
    for (const Tile& t : s->tiles) {
      const IM_STATUS st = improcess(srcb, BufAs(s->canvas), BufAs(s->pat),
                                     MakeRect(0, 0, kSrcW, kSrcH), MakeRect(t.x, t.y, t.w, t.h),
                                     MakeRect(0, 0, 0, 0), 0);
      if (st != IM_STATUS_SUCCESS && st != IM_STATUS_NOERROR) {
        fprintf(stderr, "[FATAL] improcess tile %d -> %s (%d)\n", t.tag, rga_status_name(st),
                (int)st);
        rga_ok = false;
        break;
      }
    }
    out->rga_improcess = now_ms() - m0;
    releasebuffer_handle(src_handle);
    if (!rga_ok) break;

    if (!s->verified) {
      // One-shot self-check: the native tile is a 1:1 blit, so the canvas
      // region MUST equal the source byte for byte.  This catches the
      // "RGA returned SUCCESS but wrote nothing" failure mode.
      const Tile* nat = &s->tiles[0];
      if (nat->tag == 0 && nat->w == kCanvas) {
        const size_t off_bytes = (size_t)nat->y * kCanvas * kSrcC;
        const size_t len = (size_t)nat->h * kCanvas * kSrcC;
        const bool same = memcmp(s->canvas.data + off_bytes, src_pixels, len) == 0;
        if (!same && !s->use_staging) {
          fprintf(stderr,
                  "[WARN] the RGA did not write the native tile from the source buffer; "
                  "switching to an aligned staging buffer for the rest of the run.\n");
          s->use_staging = true;
          memcpy(s->stage.data, rgb, s->stage.size);
          src_pixels = s->stage.data;
          rga_ok = true;  // retry this frame with the staging buffer
          continue;
        }
        if (!same) {
          fprintf(stderr,
                  "[FATAL] the RGA did not write the native tile even from the staging buffer; "
                  "the mosaic canvas would be garbage.\n");
          return false;
        }
        fprintf(stderr, "[i] RGA native-tile self-check: OK\n");
      }
      s->verified = true;
    }
  }
  if (!rga_ok) return false;
  return true;
}

// ---- stage 1b: widen the canvas into the caller's fp16 buffer --------------
//
// v10.  Widen the u8 canvas into an fp16 NHWC buffer (the DMA slot of
// Buffer A).  Uses EXACTLY the same call as the io-mem path so the bytes are
// identical: u8_to_f16(s->canvas.data, dst, elems, 1.0f).
//
// WHY THE BYTES MUST MATCH BIT FOR BIT: the acceptance criterion of the
// whole pipeline is a per-frame md5, so this function may not "improve" the
// widening.  Every integer 0..255 is exactly representable in fp16, which is
// why u8_to_f16_int is the closed form and there is no rounding to argue
// about.
bool WidenCanvasToF16(const Session* s, uint16_t* dst, size_t elems) {
  if (!s || !dst) return false;
  if (s->npu.input_nchw) {
    // The model's own layout is NCHW, so the NHWC canvas has to be
    // transposed -- the same transpose the io-mem NCHW path does.
    s->f16_nchw.resize(elems);
    u8_to_f16(s->canvas.data, s->f16_nchw.data(), elems, 1.0f);
    nhwc_to_nchw_f16(s->f16_nchw.data(), dst, kCanvas, kSrcC);
  } else {
    u8_to_f16(s->canvas.data, dst, elems, 1.0f);
  }
  return true;
}

// ---- hand an fp16 NHWC buffer to the runtime with pass_through=1 ----------
//
// v10.  pass_through=1 means the runtime does NO conversion and NO layout
// fix-up: the bytes we hand over ARE the tensor.  That is only legal when
// the buffer already is the model's layout, so this helper REFUSES an NCHW
// model instead of silently feeding the runtime a transposed buffer.
//
// It is the single definition of the pass-through upload, shared by the
// kSetF16Pt input mode and by the host-memory arm of the decoupled pipeline,
// so the two can never drift apart.
inline bool SetF16PassThrough(NpuCtx* npu, const uint16_t* buf, size_t elems) {
  if (!npu || !npu->ctx || !buf) return false;
  if (npu->input_nchw) {
    fprintf(stderr,
            "[FATAL] the pass-through fp16 upload needs an NHWC model; this one is NCHW\n");
    return false;
  }
  rknn_input in{};
  in.index = 0;
  in.pass_through = 1;
  in.fmt = RKNN_TENSOR_NHWC;
  in.type = RKNN_TENSOR_FLOAT16;
  in.size = (uint32_t)(elems * sizeof(uint16_t));
  in.buf = (void*)buf;
  if (rknn_inputs_set(npu->ctx, 1, &in) != 0) {
    fprintf(stderr, "[FATAL] rknn_inputs_set (pass-through fp16) failed\n");
    return false;
  }
  return true;
}

// ---- stage 1c: upload ------------------------------------------------------
//
// v10: PrepareFrame is now ComposeFrame + the upload section.  The upload
// section is the old one, VERBATIM: the fused pipeline still calls
// PrepareFrame, so its numbers, its bytes and its timing buckets are
// unchanged.
bool PrepareFrame(Session* s, const uint8_t* rgb, int w, int h, FrameOut* out) {
  if (!ComposeFrame(s, rgb, w, h, out)) return false;

  NpuCtx& npu = s->npu;

  // ---- 2. upload ----------------------------------------------------------
  const size_t canvas_elems = s->canvas.size;
  const double u0 = now_ms();
  if (s->opt.input_mode == InputMode::kSetF16) {
    // WE widen uint8 -> fp16 ourselves (integer NEON, ~0.8 ms for 1.2 M
    // elements) and hand the runtime a FLOAT16 buffer.  The runtime then
    // only COPIES 1.92 MB, where the `set` path made it WIDEN 960 KB from
    // uint8 -- that scalar widening measured 12.78 ms/frame on the board,
    // the single largest HOST cost of the fp16 arm, and it sat on the NPU's
    // critical path.
    //
    // The numbers are identical: every integer 0..255 is exactly
    // representable in fp16, so u8_to_f16_int is the closed form -- there is
    // no rounding to disagree about.  The LAYOUT must be the model's own
    // (RKNN_QUERY_INPUT_ATTR); the NCHW case goes through the same transpose
    // the io-mem path uses.
    s->f16_tmp.resize(canvas_elems);
    if (npu.input_nchw) {
      s->f16_nchw.resize(canvas_elems);
      u8_to_f16(s->canvas.data, s->f16_nchw.data(), canvas_elems, 1.0f);
      nhwc_to_nchw_f16(s->f16_nchw.data(), s->f16_tmp.data(), kCanvas, kSrcC);
    } else {
      u8_to_f16(s->canvas.data, s->f16_tmp.data(), canvas_elems, 1.0f);
    }
    rknn_input in{};
    in.index = 0;
    in.pass_through = 0;
    in.fmt = npu.input_nchw ? RKNN_TENSOR_NCHW : RKNN_TENSOR_NHWC;
    in.type = RKNN_TENSOR_FLOAT16;
    in.size = (uint32_t)(canvas_elems * sizeof(uint16_t));
    in.buf = s->f16_tmp.data();
    if (rknn_inputs_set(npu.ctx, 1, &in) != 0) {
      fprintf(stderr, "[FATAL] rknn_inputs_set failed\n");
      return false;
    }
  } else if (s->opt.input_mode == InputMode::kSetF16Pt) {
    // v10.  The same widened buffer as kSetF16, but handed over with
    // pass_through=1: the runtime takes the bytes as they are.  This is the
    // host-memory arm of the decoupled pipeline -- the counterpart of
    // binding a Buffer A DMA slot with rknn_set_io_mem().
    if (npu.input_nchw) {
      fprintf(stderr, "[FATAL] the pass-through fp16 upload needs an NHWC model\n");
      return false;
    }
    s->f16_tmp.resize(canvas_elems);
    if (!WidenCanvasToF16(s, s->f16_tmp.data(), canvas_elems)) return false;
    if (!SetF16PassThrough(&npu, s->f16_tmp.data(), canvas_elems)) return false;
  } else if (s->opt.input_mode == InputMode::kSet) {
    // The default path: a plain uint8 NHWC upload, structurally identical
    // to the certified single-scale board arm (the runtime does the
    // uint8 -> fp16 widening itself).
    rknn_input in{};
    in.index = 0;
    in.pass_through = 0;
    in.fmt = RKNN_TENSOR_NHWC;
    in.type = RKNN_TENSOR_UINT8;
    in.size = (uint32_t)canvas_elems;
    in.buf = s->canvas.data;
    if (rknn_inputs_set(npu.ctx, 1, &in) != 0) {
      fprintf(stderr, "[FATAL] rknn_inputs_set failed\n");
      return false;
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
      if (s->opt.input_mode == InputMode::kIoMemNchw) {
        s->f16_tmp.resize(canvas_elems);
        u8_to_f16(s->canvas.data, s->f16_tmp.data(), canvas_elems, 1.0f);
        nhwc_to_nchw_f16(s->f16_tmp.data(), (uint16_t*)dst, kCanvas, kSrcC);
      } else {
        u8_to_f16(s->canvas.data, (uint16_t*)dst, canvas_elems, 1.0f);
      }
    } else if (npu.input_is_f32) {
      // A float32 graph consumes x/255 (its std_values are baked in).
      float* fdst = (float*)dst;
      for (size_t i = 0; i < canvas_elems; i++) fdst[i] = (float)s->canvas.data[i] / 255.0f;
    } else if (npu.input_is_int8) {
      // An INT8 input holds CODES: with scale=1/255 and zp=-128 the code of
      // a host byte x is x + zp.
      int8_t* d8 = (int8_t*)dst;
      for (size_t i = 0; i < canvas_elems; i++)
        d8[i] = (int8_t)((int)s->canvas.data[i] + npu.input_zp);
    } else {
      memcpy(dst, s->canvas.data, canvas_elems);
    }
  }
  out->upload = now_ms() - u0;
  // NOTE ON THE COST OF THIS BUCKET (for table C):
  //   in `set` mode this bucket is librknnrt's uint8 -> fp16 widening of all
  //   640*640*3 elements, which measured 45.4 ms/frame on an RK3588 A55
  //   (1.8 GHz) and ~11-15 ms on an A76 (2.35 GHz) -- it is the single
  //   largest HOST cost of the fp16 arm, not the NPU.
  //   in the io-mem modes the conversion is ours (~0.8 ms, integer NEON),
  //   but the runtime then stages the buffer into NPU memory inside
  //   rknn_run, so the saving can be a net LOSS.  Do not read the io-mem
  //   numbers as "the upload is free".

  return true;
}

// ---- stage 2: run + outputs -------------------------------------------------
bool RunFrame(Session* s, std::vector<float>* hm, FrameOut* out) {
  NpuCtx& npu = s->npu;
  s->outs.assign(npu.n_out, rknn_output{});
  for (uint32_t i = 0; i < npu.n_out; i++) {
    s->outs[i].index = i;
    s->outs[i].want_float = 0;
  }
  const double r0 = now_ms();
  if (rknn_run(npu.ctx, nullptr) != 0) {
    fprintf(stderr, "[FATAL] rknn_run failed\n");
    return false;
  }
  const double r1 = now_ms();
  if (rknn_outputs_get(npu.ctx, npu.n_out, s->outs.data(), nullptr) != 0) {
    fprintf(stderr, "[FATAL] rknn_outputs_get failed\n");
    return false;
  }
  // Only the heatmap is dequantised: the offset output is not used by the
  // connected-component detector (the PC side's detect_components does not
  // use it either).
  if (!dequant(s->outs[0], npu.out_attr[0], hm)) {
    fprintf(stderr, "[FATAL] dequant of the heatmap output failed\n");
    rknn_outputs_release(npu.ctx, npu.n_out, s->outs.data());
    return false;
  }
  // NOTE: this segment covers rknn_outputs_get + dequant ONLY.  It must not
  // start at r0, or the rknn_run time would be counted twice and
  // sum_of_segments would exceed total_wall.
  out->outputs = now_ms() - r1;
  rknn_outputs_release(npu.ctx, npu.n_out, s->outs.data());
  out->run = r1 - r0;
  return true;
}

// ---- stage 3: back-projection + connected components -----------------------
void BackProjectFrame(Session* s, const std::vector<float>& hm, PostWorkspace* ws,
                      std::vector<Comp>* out, double* backproj_ms) {
  const NpuCtx& npu = s->npu;
  const double b0 = now_ms();
  out->clear();
  // 0 = dense（默认，行为与原来完全一样）
  // 1 = sparse（只求值可能过阈值的像素）
  // 2 = verify（dense 与 sparse 同时跑并逐字段比对，输出 dense 的结果）
  // 3 = gpu（OpenCL 局部极大路径；钩子没注册 / GPU 不可用时回退 dense）
  int mode = ws->post_mode;
  if (mode == kPostModeGpu) {
    bool gpu_ok = false;
    if (g_gpu_post_hook) {
      GpuPostTiming gt;
      gpu_ok = g_gpu_post_hook(s, hm, out, &gt);
      if (gpu_ok) {
        ws->gpu_upload_ms += gt.upload_ms;
        ws->gpu_kernel_ms += gt.kernel_ms;
        ws->gpu_readback_ms += gt.readback_ms;
        ws->gpu_total_ms += gt.total_ms;
        ws->gpu_frames++;
        ws->gpu_peaks += gt.peaks;
        ws->gpu_peaks_dropped += gt.peaks_dropped;
        ws->gpu_upload_samples.push_back(gt.upload_ms);
        ws->gpu_kernel_samples.push_back(gt.kernel_ms);
        ws->gpu_readback_samples.push_back(gt.readback_ms);
        ws->gpu_total_samples.push_back(gt.total_ms);
        if (ws->gpu_frames % 60 == 0) {
          const double n = (double)ws->gpu_frames;
          fprintf(stderr,
                  "[i] gpu post: upload=%.3f kernel=%.3f readback=%.3f total=%.3f ms/帧 "
                  "(peaks=%.1f dropped=%lld)\n",
                  ws->gpu_upload_ms / n, ws->gpu_kernel_ms / n, ws->gpu_readback_ms / n,
                  ws->gpu_total_ms / n, (double)ws->gpu_peaks / n,
                  (long long)ws->gpu_peaks_dropped);
        }
      }
    }
    if (!gpu_ok) {
      if (!ws->gpu_fallback_warned) {
        ws->gpu_fallback_warned = 1;
        fprintf(stderr, "[WARN] GPU 后处理不可用（钩子未注册或设备初始化失败），本帧回退 dense\n");
      }
      out->clear();          // 钩子可能已经写了半截结果
      mode = kPostModeDense;  // 这一帧走 dense
    } else {
      if (backproj_ms) *backproj_ms = now_ms() - b0;
      return;                  // GPU 路径已经把所有 tile 做完并打好 tag
    }
  }
  for (const Tile& t : s->tiles) {
    if (!t.enabled) continue;
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    HeatRect(t, npu.hm_w, npu.hm_h, &x0, &y0, &x1, &y1);
    const int sw = x1 - x0, sh = y1 - y0;
    if (sw <= 0 || sh <= 0) continue;
    // The tile's heatmap sub-rectangle, enlarged back onto the native grid.
    // The source rows are still npu.hm_w floats apart -- the sub-rectangle
    // is a WINDOW of the heatmap, not a packed image.
    const float* src = hm.data() + (size_t)y0 * npu.hm_w + x0;
    if (mode == kPostModeVerify) {
      // ---- 自检模式：dense 先跑（留作基准），再跑 sparse，逐字段比对 ----
      ResizeBilinearF32(src, npu.hm_w, sw, sh, ws->native_map.data(), kSrcW, kSrcH);
      DetectComponents(&ws->cc, ws->native_map.data(), kSrcW, kSrcH, s->region_threshold,
                       s->min_area, s->max_detections, &ws->comps);
      ws->verify_dense.clear();
      for (const Comp& c : ws->comps) {
        Comp tagged = c;
        tagged.tag = t.tag;
        ws->verify_dense.push_back(tagged);
      }
      SparseBackProjectTile(src, npu.hm_w, sw, sh, s->region_threshold, s->min_area,
                            s->max_detections, ws, &ws->comps);
      ws->verify_sparse.clear();
      for (const Comp& c : ws->comps) {
        Comp tagged = c;
        tagged.tag = t.tag;
        ws->verify_sparse.push_back(tagged);
      }
      VerifyCompareComps(t.tag, ws->verify_dense, ws->verify_sparse, ws);
      // 自检模式输出 dense 的结果，保证板卡上的可见行为与 dense 一致
      for (const Comp& c : ws->verify_dense) out->push_back(c);
      continue;
    }
    if (mode == kPostModeSparse) {
      SparseBackProjectTile(src, npu.hm_w, sw, sh, s->region_threshold, s->min_area,
                            s->max_detections, ws, &ws->comps);
    } else {
      ResizeBilinearF32(src, npu.hm_w, sw, sh, ws->native_map.data(), kSrcW, kSrcH);
      DetectComponents(&ws->cc, ws->native_map.data(), kSrcW, kSrcH, s->region_threshold,
                       s->min_area, s->max_detections, &ws->comps);
    }
    for (const Comp& c : ws->comps) {
      Comp tagged = c;
      tagged.tag = t.tag;
      out->push_back(tagged);
    }
  }
  if (mode == kPostModeVerify) ws->verify_frames++;
  if (backproj_ms) *backproj_ms = now_ms() - b0;
}

// ---------------------------------------------------------------------------
// ProcessFrame: the three stages back to back.  This is the whole frame path
// in one call, which is what the BMP-driven binary wants; the library calls
// the stages separately so the back-projection can overlap the NPU.
// ---------------------------------------------------------------------------
bool ProcessFrame(Session* s, const uint8_t* rgb, int w, int h, FrameOut* out) {
  out->comps.clear();
  if (!PrepareFrame(s, rgb, w, h, out)) return false;
  if (!RunFrame(s, &s->hm, out)) return false;
  BackProjectFrame(s, s->hm, &s->post, &out->comps, &out->backproj);
  return true;
}

// ---------------------------------------------------------------------------
// FormatTsvLine: byte-identical to what the BMP-driven binary wrote.
// ---------------------------------------------------------------------------
std::string FormatTsvLine(const std::string& stem, const std::vector<Comp>& comps) {
  char buf[64];
  std::string line = stem;
  snprintf(buf, sizeof(buf), "\t%zu", comps.size());
  line += buf;
  for (const Comp& c : comps) {
    snprintf(buf, sizeof(buf), "\t%d %.4f %.4f %d %d %d %.6f %.4f", c.tag, (double)c.cx, (double)c.cy,
             c.w, c.h, c.area, (double)c.peak, (double)c.sum);
    line += buf;
  }
  line += "\n";
  return line;
}
}  // namespace mosaic
