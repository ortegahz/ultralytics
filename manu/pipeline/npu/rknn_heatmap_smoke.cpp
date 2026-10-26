// ===========================================================================
// rknn_heatmap_smoke.cpp -- single-frame RK3588 NPU inference for the
// Trial-0474 heatmap detector (manu/models/heatmap_model.py).
//
// Contract (all of it read out of the checkpoint or the ONNX graph, none of it
// assumed):
//   input   uint8 NHWC 640x640x3, RGB, letterboxed with 114 (mean 0 / std 255
//           on the RKNN side => the graph consumes uint8/255, which is exactly
//           what PyTorch consumes, so host and board see identical numbers)
//   output0 heatmap  (1, 1, 320, 320) sigmoid, clamped [1e-5, 1-1e-5]
//   output1 offset   (1, 2, 320, 320) sub-pixel, in stride-cell units
//   stride  2  -> a cell at (x, y) maps to ((x+dx)*2, (y+dy)*2)
//
// Design constraints, each one measured on this board rather than assumed:
//   * No OpenCV, no RGA -- the board has neither (see memory/opencv_arm64_parity.md).
//     JPEG decode uses the SDK's header-only stb_image; the letterbox is done by
//     hand so that it can be made bit-identical to the host's
//     manu/pipeline/npu/hm_common.py::letterbox (which mirrors ultralytics'
//     val-mode LetterBox).
//   * The peak decoder here is a 1:1 port of
//     manu/evaluation/heatmap_evaluate.py::extract_peaks. The host re-decodes the
//     dumped tensors with the numpy port and the two must agree -- that is the
//     check, not eyeballing.
//
// Honesty rules baked into the program (see rules.md 2c / 4.2d):
//   * an all-zero input is refused outright rather than "inferred on";
//   * a null control (inverted input) must move the output bytes, otherwise the
//     input path is dead and every downstream number is meaningless;
//   * determinism is checked but reported separately -- it is not evidence that
//     the input reached the NPU.
//
// Usage:
//   rknn_heatmap_smoke <model.rknn> <feature.jpg> [--th 0.25] [--stride 2]
//                      [--topk 100] [--loops N] [--warmup N] [--perf]
//                      [--dump <dir>] [--tag <name>]
//   rknn_heatmap_smoke <model.rknn> --batch <list.txt> [--dump <dir>]
// ===========================================================================
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <sched.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <chrono>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <string>
#include <vector>
#include <algorithm>

#include "rknn_api.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

constexpr int kDefaultSize = 640;
constexpr uint8_t kPadValue = 114;

double now_ms() {
  // std::chrono rather than clock_gettime: the raw POSIX call returned values
  // that made every reported duration absurd (5.4e11 ms for a letterbox), while
  // %.6f on an ordinary double printed fine -- i.e. it was the clock, not the
  // formatting. steady_clock is also immune to wall-clock steps.
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

uint32_t fnv1a(const void* data, size_t n) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 16777619u; }
  return h;
}

bool read_file(const char* path, std::vector<uint8_t>* out) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  out->resize((size_t)n);
  size_t got = n > 0 ? fread(out->data(), 1, (size_t)n, f) : 0;
  fclose(f);
  return got == (size_t)n;
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

// ---------------------------------------------------------------------------
// fp16 bit helpers + NEON uint8->fp16 conversion.
//
// Why this exists: an fp16 graph's input tensor is FLOAT16 (2 B/px), so
// rknn_inputs_set has to widen all 640*640*3 elements in userspace -- measured
// at 45.4 ms/frame on an A55 (37 ns/element, a scalar loop) and 14.8 ms on an
// A76. That is the single biggest non-NPU cost of the fp16 arm.
//
// With rknn_set_io_mem the host owns the input tensor's storage, so the
// conversion can be done HERE, once, with NEON, instead of inside librknnrt.
// fp16 represents every integer in [0, 255] exactly (11-bit mantissa, exact
// up to 2048), so u8 -> f16 is lossless and the only question is the SCALE
// convention: whether the tensor holds raw 0..255 or the normalised 0..1
// (the graph's std_values=[[255,255,255]]). --conv-div selects it and the
// detections are the judge.
// ---------------------------------------------------------------------------
#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define HM_HAVE_NEON 1
#endif

// float -> fp16 bits, round-to-nearest-even (portable fallback path).
static uint16_t f32_to_f16_bits(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
  uint32_t man = x & 0x7FFFFFu;
  if (((x >> 23) & 0xFF) == 0xFF) {  // inf / nan
    return (uint16_t)(sign | 0x7C00u | (man ? (0x200u | (man >> 13)) : 0u));
  }
  if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);  // overflow -> inf
  if (exp <= 0) {                                    // subnormal / zero
    if (exp < -10) return (uint16_t)sign;
    man |= 0x800000u;
    uint32_t shift = (uint32_t)(14 - exp);
    uint32_t half = man >> shift;
    if ((man >> (shift - 1)) & 1u) {
      if ((man & ((1u << (shift - 1)) - 1u)) || (half & 1u)) half++;
    }
    return (uint16_t)(sign | half);
  }
  uint32_t half = (uint32_t)(exp << 10) | (man >> 13);
  if ((man >> 12) & 1u) {  // round to nearest even
    if ((man & 0xFFFu) || (half & 1u)) half++;
  }
  return (uint16_t)(sign | half);
}

// value -> fp16 bits, for an integer input value (exact, no rounding).
#ifdef HM_HAVE_NEON
static inline uint32x4_t fp16_bits_from_u32(uint32x4_t v, uint32x4_t bias, uint32x4_t zero) {
  uint32x4_t f = vreinterpretq_u32_f32(vcvtq_f32_u32(v));
  uint32x4_t h = vsubq_u32(vshrq_n_u32(f, 13), bias);
  uint32x4_t isz = vceqq_u32(v, zero);
  return vbicq_u32(h, isz);
}
#endif

// FAST PATH for the integer case (div == 1, i.e. the tensor holds the raw byte
// values -- which is what the baseline rknn_inputs_set path feeds the model).
//
// Every integer 0..255 is exactly representable in fp16, and its bit pattern
// has a closed form:
//
//     fp16(n) = (f32_bits(n) >> 13) - 0x1C000     for n > 0
//     fp16(0) = 0
//
// because f32_bits(n) = (E<<23) | ((n - 2^e) << (23-e)) with E = 127+e, so
// (bits>>13) already holds (E<<10) | ((n - 2^e) << (10-e)) and only the
// exponent bias (127-15=112 -> 112<<10 = 0x1C000) has to come off. The low 13
// bits of the f32 mantissa are zero for integers with e <= 7, so no rounding
// is needed: the result is exact.
//
// That turns the conversion into ~7 integer NEON ops per 4 elements instead
// of the full float round-trip, and it needs no FP16 ISA (the A55 cores of
// the RK3588 do not implement FP16 -- only the A76 does).
static void u8_to_f16_int(const uint8_t* src, uint16_t* dst, size_t n) {
#ifdef HM_HAVE_NEON
  const uint32x4_t bias = vdupq_n_u32(0x1C000u);
  const uint32x4_t zero = vdupq_n_u32(0);
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    uint8x16_t b = vld1q_u8(src + i);
    uint16x8_t h = vmovl_u8(vget_low_u8(b));
    uint16x8_t h2 = vmovl_u8(vget_high_u8(b));
    uint32x4_t v0 = vmovl_u16(vget_low_u16(h));
    uint32x4_t v1 = vmovl_u16(vget_high_u16(h));
    uint32x4_t v2 = vmovl_u16(vget_low_u16(h2));
    uint32x4_t v3 = vmovl_u16(vget_high_u16(h2));
    uint32x4_t o0 = fp16_bits_from_u32(v0, bias, zero);
    uint32x4_t o1 = fp16_bits_from_u32(v1, bias, zero);
    uint32x4_t o2 = fp16_bits_from_u32(v2, bias, zero);
    uint32x4_t o3 = fp16_bits_from_u32(v3, bias, zero);
    vst1_u16(dst + i, vmovn_u32(o0));
    vst1_u16(dst + i + 4, vmovn_u32(o1));
    vst1_u16(dst + i + 8, vmovn_u32(o2));
    vst1_u16(dst + i + 12, vmovn_u32(o3));
  }
  for (; i < n; i++) {
    float f = (float)src[i];
    uint32_t b;
    memcpy(&b, &f, 4);
    uint32_t o = src[i] == 0 ? 0u : ((b >> 13) - 0x1C000u);
    dst[i] = (uint16_t)o;
  }
#else
  for (size_t i = 0; i < n; i++) {
    float f = (float)src[i];
    uint32_t b;
    memcpy(&b, &f, 4);
    dst[i] = (uint16_t)(src[i] == 0 ? 0u : ((b >> 13) - 0x1C000u));
  }
#endif
}

// fp16 bit patterns of the 256 byte values, computed once. The fused letterbox
// needs a value->bits lookup for the pixels it computes one at a time (the
// bilinear path), and a table turns that into a single L1 load.
struct Fp16Lut {
  uint16_t v[256];
  Fp16Lut() {
    for (int i = 0; i < 256; i++) {
      float f = (float)i;
      uint32_t b;
      memcpy(&b, &f, 4);
      // fp16(n) = (f32_bits(n) >> 13) - 0x1C000, exact for n <= 255.
      // n = 0 has to be special-cased: 0 - 0x1C000 underflows the unsigned
      // subtraction and would come out as 0x4000 (which is fp16 for 2.0).
      v[i] = (i == 0) ? (uint16_t)0 : (uint16_t)((b >> 13) - 0x1C000u);
    }
  }
};
static const Fp16Lut g_fp16_lut;

// uint8 NHWC -> NCHW (planar). Scalar reference; the point of the
// --upload-nchw arm is to find out whether the driver's own transpose is a
// measurable part of rknn_inputs_set.
static void nhwc_to_nchw_u8(const uint8_t* src, uint8_t* dst, int hw, int c) {
  for (int ch = 0; ch < c; ch++)
    for (int y = 0; y < hw; y++)
      for (int x = 0; x < hw; x++)
        dst[(size_t)ch * hw * hw + (size_t)y * hw + x] = src[((size_t)y * hw + x) * c + ch];
}

// Transpose an NHWC canvas (H*W*3, interleaved) into NCHW (3*H*W, planar).
// Scalar reference implementation -- used to TEST whether the bound input
// tensor wants NCHW (the ONNX input is NCHW) or NHWC (what
// RKNN_QUERY_INPUT_ATTR reports).
static void nhwc_to_nchw_f16(const uint16_t* src, uint16_t* dst, int hw, int c) {
  for (int ch = 0; ch < c; ch++)
    for (int y = 0; y < hw; y++)
      for (int x = 0; x < hw; x++)
        dst[(size_t)ch * hw * hw + (size_t)y * hw + x] = src[((size_t)y * hw + x) * c + ch];
}

// value = u8 / div, written as fp16. div == 1 -> raw 0..255.
//
// NOTE ON THE ISA: the obvious intrinsic (vcvtq_f16_f32) needs the optional
// FP16 extension, which the RK3588's Cortex-A55 does NOT implement (only the
// A76 does) -- a binary that used it would SIGILL on a small core. So the
// f32->f16 step here is done with INTEGER NEON (bit manipulation), which runs
// on every core. Only the normal fp16 range is handled (our values are
// x/255 in [0,1] or raw x in [0,255]); zero is special-cased.
static void u8_to_f16(const uint8_t* src, uint16_t* dst, size_t n, float div) {
  if (div == 1.0f) { u8_to_f16_int(src, dst, n); return; }
#ifdef HM_HAVE_NEON
  const float inv = 1.0f / div;
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    uint16x8_t u16 = vmovl_u8(vld1_u8(src + i));
    uint32x4_t u32lo = vmovl_u16(vget_low_u16(u16));
    uint32x4_t u32hi = vmovl_u16(vget_high_u16(u16));
    float32x4_t flo = vmulq_n_f32(vcvtq_f32_u32(u32lo), inv);
    float32x4_t fhi = vmulq_n_f32(vcvtq_f32_u32(u32hi), inv);
    // f32 -> f16 (round to nearest even), integer NEON
    uint32x4_t xlo = vreinterpretq_u32_f32(flo);
    uint32x4_t xhi = vreinterpretq_u32_f32(fhi);
    uint32x4_t signlo = vshlq_n_u32(vandq_u32(xlo, vdupq_n_u32(0x80000000u)), 16);
    uint32x4_t signhi = vshlq_n_u32(vandq_u32(xhi, vdupq_n_u32(0x80000000u)), 16);
    // The exponent has to come from x itself: mag (the mantissa field) has the
    // exponent bits masked OFF, so mag>>23 is always 0.
    uint32x4_t elo = vsubq_u32(vshrq_n_u32(xlo, 23), vdupq_n_u32(112));
    uint32x4_t ehi = vsubq_u32(vshrq_n_u32(xhi, 23), vdupq_n_u32(112));
    uint32x4_t mlo = vorrq_u32(vandq_u32(xlo, vdupq_n_u32(0x7FFFFFu)), vdupq_n_u32(0x800000u));
    uint32x4_t mhi = vorrq_u32(vandq_u32(xhi, vdupq_n_u32(0x7FFFFFu)), vdupq_n_u32(0x800000u));
    // m>>13 keeps the implicit leading 1; the fp16 mantissa field is the low
    // 10 bits of that, and the 13 dropped bits decide the rounding.
    uint32x4_t mslo = vandq_u32(vshrq_n_u32(mlo, 13), vdupq_n_u32(0x3FFu));
    uint32x4_t mshi = vandq_u32(vshrq_n_u32(mhi, 13), vdupq_n_u32(0x3FFu));
    uint32x4_t remlo = vandq_u32(mlo, vdupq_n_u32(0x1FFFu));
    uint32x4_t remhi = vandq_u32(mhi, vdupq_n_u32(0x1FFFu));
    // Round to nearest, ties to even:
    //   rem > 0x1000  -> always up
    //   rem == 0x1000 -> up only when the kept mantissa is odd
    //   rem <  0x1000 -> down
    uint32x4_t gtlo = vcgtq_u32(remlo, vdupq_n_u32(0x1000u));
    uint32x4_t gthi = vcgtq_u32(remhi, vdupq_n_u32(0x1000u));
    uint32x4_t eqlo = vceqq_u32(remlo, vdupq_n_u32(0x1000u));
    uint32x4_t eqhi = vceqq_u32(remhi, vdupq_n_u32(0x1000u));
    // NOTE: the comparisons return ALL-ONES masks, not 0/1 -- the final AND
    // with 1 is what turns them into an increment.
    uint32x4_t inclo = vandq_u32(vorrq_u32(gtlo, vandq_u32(eqlo, vandq_u32(mslo, vdupq_n_u32(1u)))),
                                  vdupq_n_u32(1u));
    uint32x4_t inchi = vandq_u32(vorrq_u32(gthi, vandq_u32(eqhi, vandq_u32(mshi, vdupq_n_u32(1u)))),
                                  vdupq_n_u32(1u));
    uint32x4_t hlo = vaddq_u32(vshlq_n_u32(elo, 10), vaddq_u32(mslo, inclo));
    uint32x4_t hhi = vaddq_u32(vshlq_n_u32(ehi, 10), vaddq_u32(mshi, inchi));
    uint32x4_t zlo = vceqq_u32(xlo, vdupq_n_u32(0));
    uint32x4_t zhi = vceqq_u32(xhi, vdupq_n_u32(0));
    uint32x4_t rlo = vorrq_u32(signlo, vbicq_u32(hlo, zlo));
    uint32x4_t rhi = vorrq_u32(signhi, vbicq_u32(hhi, zhi));
    // The 4 fp16 values live in the LOW 16 bits of the 4 u32 lanes, so the
    // lanes have to be NARROWED (movn) before storing: reinterpreting a
    // uint32x4_t as uint16x8_t would interleave the high halves as separate
    // elements.
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
  uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
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

// Dequantize one whole output tensor into float32, whatever the runtime hands
// back. want_float=0 keeps the native int8 so the SDK's own scale/zp apply;
// the fp16 arm comes back as FLOAT16.
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
// Letterbox -- 1:1 with hm_common.letterbox, which mirrors what the ultralytics
// val path actually does:
//
//   load_image(index, rect_mode=True) -> the LONG side is scaled to imgsz FIRST
//   -> LetterBox(scaleup=False) then runs on the already-resized image
//
// So the ratio is NOT capped at 1.0. A 640x512 source gives r = 1.0 and 64 px of
// padding; a SQUARE 512x512 source gives r = 1.25, is upscaled to 640x640, and
// gets no padding at all. 4682 of the 31613 val frames (14.8%) are square, so
// capping the ratio here would silently corrupt every one of them.
//
// The resize is full 2-D bilinear with half-pixel centres, matching
// cv2.INTER_LINEAR (which is what the host uses). An earlier version resized on
// the horizontal axis only; that was unreachable while every source was <= 640
// wide, and became a 200+ level error once square sources started upscaling.
// ---------------------------------------------------------------------------
void letterbox(const uint8_t* src, int sw, int sh, int size, std::vector<uint8_t>* dst) {
  double r = std::min((double)size / (double)sh, (double)size / (double)sw);
  int nw = (int)lround(sw * r), nh = (int)lround(sh * r);
  int dw = (size - nw) / 2, dh = (size - nh) / 2;
  int left = (int)lround(dw - 0.1), top = (int)lround(dh - 0.1);
  int right = (int)lround(dw + 0.1), bottom = (int)lround(dh + 0.1);
  dst->assign((size_t)size * size * 3, kPadValue);

  if (nw == sw && nh == sh) {
    for (int y = 0; y < sh; y++)
      memcpy(dst->data() + (size_t)(y + top) * size * 3 + left * 3, src + (size_t)y * sw * 3,
             (size_t)sw * 3);
  } else {
    // cv2.INTER_LINEAR: src = (dst + 0.5) * scale - 0.5, clamped to the edge.
    const double inv_x = 1.0 / r, inv_y = 1.0 / r;
    for (int y = 0; y < nh; y++) {
      double fy = (y + 0.5) * inv_y - 0.5;
      int y0 = (int)floor(fy), y1 = y0 + 1;
      float ay = (float)(fy - y0);
      y0 = std::max(0, std::min(sh - 1, y0));
      y1 = std::max(0, std::min(sh - 1, y1));
      const uint8_t* r0 = src + (size_t)y0 * sw * 3;
      const uint8_t* r1 = src + (size_t)y1 * sw * 3;
      uint8_t* drow = dst->data() + (size_t)(y + top) * size * 3 + left * 3;
      for (int x = 0; x < nw; x++) {
        double fx = (x + 0.5) * inv_x - 0.5;
        int x0 = (int)floor(fx), x1 = x0 + 1;
        float ax = (float)(fx - x0);
        x0 = std::max(0, std::min(sw - 1, x0));
        x1 = std::max(0, std::min(sw - 1, x1));
        for (int c = 0; c < 3; c++) {
          float topv = r0[(size_t)x0 * 3 + c] * (1.f - ax) + r0[(size_t)x1 * 3 + c] * ax;
          float botv = r1[(size_t)x0 * 3 + c] * (1.f - ax) + r1[(size_t)x1 * 3 + c] * ax;
          drow[(size_t)x * 3 + c] = (uint8_t)lroundf(topv * (1.f - ay) + botv * ay);
        }
      }
    }
  }
  printf("  letterbox %dx%d -> %dx%d (r=%.6f, pad l=%d t=%d r=%d b=%d)\n", sw, sh, size, size, r,
         left, top, right, bottom);
}

// ---------------------------------------------------------------------------
// Fused letterbox -> fp16.
//
// Same geometry and same rounding as letterbox() (the uint8 buffer) followed by
// the u8->f16 conversion, but the intermediate 1.2 MB uint8 buffer never
// exists: the padded canvas is written straight out as fp16.
//
// Bit-exactness is a REQUIREMENT, not a hope: the bilinear path rounds to
// uint8 exactly like letterbox() does (same lroundf on the same float), and
// then converts that integer with the same closed form, so the fp16 bytes are
// identical to "letterbox() then u8->f16". The no-resize path uses the same
// NEON converter. --selftest-lb checks that on the board.
// ---------------------------------------------------------------------------
void letterbox_to_f16(const uint8_t* src, int sw, int sh, int size, uint16_t* dst) {
  const uint16_t* lut = g_fp16_lut.v;
  const uint16_t pad = lut[kPadValue];
  double r = std::min((double)size / (double)sh, (double)size / (double)sw);
  int nw = (int)lround(sw * r), nh = (int)lround(sh * r);
  int dw = (size - nw) / 2, dh = (size - nh) / 2;
  int left = (int)lround(dw - 0.1), top = (int)lround(dh - 0.1);
  const size_t total = (size_t)size * size * 3;

  if (nw == sw && nh == sh) {
    // No resize: the content is a row-for-row copy, so the conversion is the
    // vectorised one and only the padding bands have to be materialised.
    // (Filling the whole canvas with the pad value scalar-wise was 1.4 ms of
    // the 1.66 ms this function used to cost.)
#ifdef HM_HAVE_NEON
    const uint16x8_t padv = vdupq_n_u16(pad);
    auto fill_pad = [&](size_t from, size_t n) {
      size_t i = from;
      const size_t stop = from + n;
      for (; i + 8 <= stop; i += 8) vst1q_u16(dst + i, padv);
      for (; i < stop; i++) dst[i] = pad;
    };
    // Only the padding BANDS, not the whole canvas: for a 640x512 source the
    // padding is 20% of the canvas, and filling all of it scalar-wise was
    // costing more than the conversion itself.
    fill_pad(0, (size_t)top * size * 3);
    fill_pad((size_t)(top + nh) * size * 3, (size_t)(size - top - nh) * size * 3);
    for (int y = 0; y < nh; y++) {
      size_t row = (size_t)(y + top) * size * 3;
      fill_pad(row, (size_t)left * 3);
      fill_pad(row + (size_t)(left + nw) * 3, (size_t)(size - left - nw) * 3);
    }
#else
    for (size_t i = 0; i < total; i++) dst[i] = pad;
#endif
    for (int y = 0; y < sh; y++) {
      uint16_t* drow = dst + (size_t)(y + top) * size * 3 + left * 3;
      u8_to_f16_int(src + (size_t)y * sw * 3, drow, (size_t)sw * 3);
    }
    return;
  }

  // Resize path: the bilinear loop is scalar and its rounding has to match
  // letterbox() exactly, so the canvas is built in uint8 first and then
  // converted with the vectorised routine. Same bytes, no scalar LUT lookup
  // per pixel.
  static thread_local std::vector<uint8_t> tmp;
  tmp.resize(total);
  letterbox(src, sw, sh, size, &tmp);
  u8_to_f16_int(tmp.data(), dst, total);
}


struct Det {
  float x, y, score;  // x, y = centre in 640x640 letterbox-canvas pixels
};

void extract_peaks(const float* hm, const float* off, int H, int W, int stride, float thr,
                   int top_k, std::vector<Det>* dets) {
  dets->clear();
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      float v = hm[(size_t)y * W + x];
      if (v < thr) continue;
      // 3x3 local max == F.max_pool2d(k=3, s=1, p=1) equality test
      bool is_max = true;
      for (int dy = -1; dy <= 1 && is_max; dy++)
        for (int dx = -1; dx <= 1 && is_max; dx++) {
          int ny = std::max(0, std::min(H - 1, y + dy));
          int nx = std::max(0, std::min(W - 1, x + dx));
          if (hm[(size_t)ny * W + nx] > v) is_max = false;
        }
      if (!is_max) continue;
      float dx = off[(size_t)y * W + x];
      float dy = off[(size_t)(H * W) + (size_t)y * W + x];
      dets->push_back({(x + dx) * (float)stride, (y + dy) * (float)stride, v});
    }
  }
  if ((int)dets->size() > top_k) {
    std::stable_sort(dets->begin(), dets->end(),
                     [](const Det& a, const Det& b) { return a.score > b.score; });
    dets->resize(top_k);
    std::stable_sort(dets->begin(), dets->end(), [](const Det& a, const Det& b) {
      if (a.y != b.y) return a.y < b.y;
      return a.x < b.x;
    });
  }
}

bool write_bin(const std::string& path, const void* data, size_t bytes) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) return false;
  size_t n = fwrite(data, 1, bytes, f);
  fclose(f);
  return n == bytes;
}

struct Ctx {
  rknn_context ctx = 0;
  uint32_t n_out = 0;
  std::vector<rknn_tensor_attr> out_attr;
  int in_h = 0, in_w = 0, in_c = 0;
  int hm_h = 0, hm_w = 0, off_h = 0, off_w = 0;
  // True when the graph's input tensor is FLOAT16 (an fp16 / --no-quant build).
  // An fp16 graph fed RKNN_TENSOR_UINT8 makes the runtime widen every one of the
  // 640*640*3 elements inside rknn_inputs_set, and for this graph that measured
  // 45.4 ms/frame -- 29% of a 162 ms fp16 frame. A caller that already holds the
  // widened buffer can hand it over and skip the work entirely.
  // Zero-copy input buffer (rknn_set_io_mem). When set, the input path never
  // calls rknn_inputs_set: the host fills this tensor itself.
  rknn_tensor_mem* io_mem = nullptr;
  // Upload layout for the rknn_inputs_set path. The SDK header says the NPU's
  // internal input format is NCHW by default and that "entering NCHW data can
  // avoid the format conversion in the driver" -- so the transpose can be
  // moved to the host (or skipped, if the driver was doing NHWC natively).
  bool upload_nchw = false;
  bool input_is_f16 = false;
  // True when the graph's input tensor is FLOAT32 (a float32 build). Kept
  // apart from input_is_f16 because the zero-copy fill path needs to know
  // which conversion to apply.
  bool input_is_f32 = false;
  // True when the graph's input tensor is INT8 (quantized codes). The
  // zero-point decides how a host byte maps to a code.
  bool input_is_int8 = false;
  int input_zp = 0;
  // NPU core mask requested for this context. 0 == leave the runtime's default
  // (RKNN_NPU_CORE_AUTO), which is what every run used until the core-0/core-1
  // split was made explicit -- recorded because "two workers" and "one worker per
  // NPU core" are different claims and only the second one is a measurement.
  int core_mask = 0;
};

// One full run; on success fills hm/off with float32. t_stage, when non-null,
// receives {inputs_set_ms, run_ms, outputs_get_ms, dequant_ms}.
//
// dequant is timed separately because it converts the raw INT8 tensors into the
// float32 buffers the peak decoder consumes: 3 x 320x320 elements per frame,
// and it sits OUTSIDE the a0..a3 window above. Reporting only a0..a3 made the
// published "TOTAL" understate the per-frame cost, which is how a 168 ms/frame
// wall clock came out looking like 57 ms/frame.
bool run_once(const Ctx& c, const std::vector<uint8_t>& input, std::vector<float>* hm,
              std::vector<float>* off, std::vector<rknn_output>* out, double* t_stage,
              const uint16_t* in_f16 = nullptr, size_t in_f16_elems = 0,
              const float* in_f32 = nullptr, size_t in_f32_elems = 0,
              bool in_f32_passthrough = true) {
  rknn_input in{};
  in.index = 0;
  if (in_f32 && in_f32_elems && in_f32_passthrough) {
    // THE pass_through path, and the only one that can work. Measured, not
    // assumed: the driver rejects anything smaller than the model's own input
    // and prints the requirement --
    //   E RKNN: rknn_inputs_set, param input size(2457600) < model input size(4915200)
    // 4915200 = 640*640*3*4, i.e. the runtime wants a float32 HOST buffer even
    // though RKNN_QUERY_INPUT_ATTR reports in_attr.type == FLOAT16 for this
    // graph. A 2-byte-per-element upload can therefore never be accepted, which
    // is why the uint16 branch below fails on every single frame.
    //
    // pass_through=1 means no conversion at all, so the 45.4 ms uint8 widening
    // + std_values normalisation leaves the NPU worker's critical path -- but
    // the values handed over must already be in MODEL space, which is what the
    // div255 switch on the letterbox side decides. That is a measurement to run,
    // not a fact to assume.
    //
    // fmt stays zero: the SDK header says pass_through=TRUE needs neither type
    // nor fmt, and a stale fmt next to pass_through=1 is exactly the kind of
    // self-contradiction worth not sending.
    in.pass_through = 1;
    in.size = (uint32_t)(in_f32_elems * sizeof(float));
    in.buf = const_cast<float*>(in_f32);
  } else if (in_f32 && in_f32_elems) {
    // Same float32 buffer, but handed over as an ordinary typed upload instead of
    // a pass-through. pass_through=1 is the mode that SEGVs inside librknnrt's
    // memcpy, so this variant exists to price the alternative: the runtime still
    // converts (float32 -> float16), but a correctly typed upload may take a
    // vectorised path where the uint8 -> float16 route measured 37 ns/element.
    in.pass_through = 0;
    in.fmt = RKNN_TENSOR_NHWC;
    in.type = RKNN_TENSOR_FLOAT32;
    in.size = (uint32_t)(in_f32_elems * sizeof(float));
    in.buf = const_cast<float*>(in_f32);
  } else if (in_f16 && in_f16_elems) {
    // rknn_api.h on pass_through:
    //   TRUE  -> "the buf data is passed directly to the input node of the rknn
    //             model without any conversion"
    //   FALSE -> "the buf data is converted into an input consistent with the
    //             model according to the following type and fmt"
    // So pass_through=0 is precisely what forces the 45.4 ms CPU conversion this
    // path exists to avoid -- and feeding FLOAT16 with it still gave ZERO
    // detections on all 537 frames, i.e. it converted the fp16 values again.
    //
    // With pass_through=1 the buffer must already be in the model's own input
    // format, which for this graph RKNN_QUERY_INPUT_ATTR reports as NHWC
    // FLOAT16. We therefore hand over exactly that and set neither type nor fmt
    // (the header says they need not be set).
    //
    // The values are the same 0..255 integers the UINT8 path carries -- the
    // graph's std_values=[[255,255,255]] normalisation is baked into the model,
    // not into the upload. That is an expectation, not a guarantee: the caller
    // must diff the detections TSV against the uint8 path to confirm it.
    in.pass_through = 1;
    in.size = (uint32_t)(in_f16_elems * sizeof(uint16_t));
    in.buf = const_cast<uint16_t*>(in_f16);
  } else {
    in.pass_through = 0;
    in.fmt = c.upload_nchw ? RKNN_TENSOR_NCHW : RKNN_TENSOR_NHWC;
    in.type = RKNN_TENSOR_UINT8;
    in.size = (uint32_t)input.size();
    in.buf = const_cast<uint8_t*>(input.data());
  }

  out->assign(c.n_out, rknn_output{});
  for (uint32_t i = 0; i < c.n_out; i++) {
    (*out)[i].index = i;
    (*out)[i].want_float = 0;
  }
  // Stage breakdown matters here: this graph returns 1.2 MB of tensors per
  // frame, and for a model that size the read-back can rival the NPU itself.
  double a0 = now_ms();
  if (rknn_inputs_set(c.ctx, 1, &in) != 0) { fprintf(stderr, "inputs_set failed\n"); return false; }
  double a1 = now_ms();
  if (rknn_run(c.ctx, NULL) != 0)           { fprintf(stderr, "rknn_run failed\n"); return false; }
  double a2 = now_ms();
  if (rknn_outputs_get(c.ctx, c.n_out, out->data(), NULL) != 0) {
    fprintf(stderr, "outputs_get failed\n");
    return false;
  }
  double a3 = now_ms();
  if (t_stage) { t_stage[0] = a1 - a0; t_stage[1] = a2 - a1; t_stage[2] = a3 - a2; t_stage[3] = 0; }
  double d0 = now_ms();
  if (hm) { if (!dequant((*out)[0], c.out_attr[0], hm)) return false; }
  if (off) { if (!dequant((*out)[1], c.out_attr[1], off)) return false; }
  if (t_stage) { t_stage[3] = now_ms() - d0; }
  return true;
}

// Upload a host fp16 buffer as RKNN_TENSOR_FLOAT16 (pass_through = 0).
//
// This is the combination the 2026-10-16 experiment never tried: it measured
// a FLOAT32 typed upload (47 ms, no gain) and a FLOAT16 pass_through upload
// (rejected / SIGSEGV), but never FLOAT16 *typed*. If the runtime's
// float16->float16 step is a copy, this is the way to hand the runtime the
// buffer our own NEON code produced (0.78 ms) and still get its efficient
// staging into NPU memory.
bool run_once_f16(const Ctx& c, const uint16_t* f16, size_t n, std::vector<float>* hm,
                  std::vector<float>* off, std::vector<rknn_output>* out, double* t_stage) {
  rknn_input in{};
  in.index = 0;
  in.pass_through = 0;
  in.fmt = RKNN_TENSOR_NHWC;
  in.type = RKNN_TENSOR_FLOAT16;
  in.size = (uint32_t)(n * sizeof(uint16_t));
  in.buf = const_cast<uint16_t*>(f16);

  out->assign(c.n_out, rknn_output{});
  for (uint32_t i = 0; i < c.n_out; i++) {
    (*out)[i].index = i;
    (*out)[i].want_float = 0;
  }
  double a0 = now_ms();
  if (rknn_inputs_set(c.ctx, 1, &in) != 0) { fprintf(stderr, "inputs_set failed\n"); return false; }
  double a1 = now_ms();
  if (rknn_run(c.ctx, NULL) != 0)           { fprintf(stderr, "rknn_run failed\n"); return false; }
  double a2 = now_ms();
  if (rknn_outputs_get(c.ctx, c.n_out, out->data(), NULL) != 0) {
    fprintf(stderr, "outputs_get failed\n");
    return false;
  }
  double a3 = now_ms();
  if (t_stage) { t_stage[0] = a1 - a0; t_stage[1] = a2 - a1; t_stage[2] = a3 - a2; t_stage[3] = 0; }
  double d0 = now_ms();
  if (hm) { if (!dequant((*out)[0], c.out_attr[0], hm)) return false; }
  if (off) { if (!dequant((*out)[1], c.out_attr[1], off)) return false; }
  if (t_stage) { t_stage[3] = now_ms() - d0; }
  return true;
}

void release(Ctx& c, std::vector<rknn_output>* out) {
  if (!out->empty() && (*out)[0].buf) rknn_outputs_release(c.ctx, c.n_out, out->data());
}

// ---------------------------------------------------------------------------
// Zero-copy input path -- the host owns the input tensor's storage.
//
// rknn_set_io_mem binds an rknn_tensor_mem to the model's input node, so
// rknn_inputs_set is NEVER called: no driver-side type conversion, no
// driver-side copy. The host fills the tensor itself, and the only cost left
// is the conversion we do on purpose.
//
// This is the only way to bypass librknnrt's per-element widening on an fp16
// graph. pass_through=1 does not work here: the driver checks the upload size
// against float32 (4 B/px = 4,915,200 B) while RKNN_QUERY_INPUT_ATTR reports
// FLOAT16 (2 B/px = 2,457,600 B) -- handing over 2 B/px is rejected
// ("param input size(2457600) < model input size(4915200)") and handing over
// 4 B/px segfaults inside librknnrt's memcpy. rknn_set_io_mem has no such
// check: the mem comes from the attr's own size.
//
// t_stage receives {fill_ms, run_ms, outputs_get_ms, dequant_ms}; there is no
// inputs_set stage because that call does not exist on this path.
// ---------------------------------------------------------------------------
// Fill the bound input tensor from a uint8 buffer, in the model's own dtype.
void io_mem_fill_u8(const Ctx& c, const uint8_t* src, size_t n, float conv_div, bool nchw = false) {
  uint8_t* dst = (uint8_t*)c.io_mem->virt_addr;
  if (c.input_is_f16 && nchw) {
    // Fill in the model's INTERNAL layout (NCHW, as the ONNX declares) instead
    // of the layout the runtime asks the host for (NHWC). If the runtime
    // transposes internally at run time, doing it here moves that cost off
    // rknn_run.
    static thread_local std::vector<uint16_t> tmp;
    tmp.resize(n);
    u8_to_f16(src, tmp.data(), n, conv_div);
    const int hw = (int)std::lround(std::sqrt((double)n / 3.0));   // canvas edge
    nhwc_to_nchw_f16(tmp.data(), (uint16_t*)dst, hw, 3);
  } else if (c.input_is_f16) {
    u8_to_f16(src, (uint16_t*)dst, n, conv_div);
  } else if (c.input_is_f32) {
    float* fdst = (float*)dst;
    for (size_t i = 0; i < n; i++) fdst[i] = (float)src[i] / conv_div;
  } else if (c.input_is_int8) {
    // An INT8 input tensor holds CODES, not values: with scale=1/255 and
    // zp=-128 the code of a host byte x is x + zp (= x - 128). The driver does
    // exactly this reinterpretation, which is why the int8 arm's
    // rknn_inputs_set is only 0.34 ms (a byte-level pass, no arithmetic).
    // Copying the bytes verbatim is WRONG: 0x80 would decode as -128
    // instead of 0, and the detections change (measured: TSV md5 differs).
    int8_t* d8 = (int8_t*)dst;
    const int zp = c.input_zp;
#ifdef HM_HAVE_NEON
    size_t i = 0;
    const int8x16_t zpv = vdupq_n_s8((int8_t)zp);
    for (; i + 16 <= n; i += 16) {
      int8x16_t v = vreinterpretq_s8_u8(vld1q_u8(src + i));
      vst1q_s8(d8 + i, vaddq_s8(v, zpv));
    }
    for (; i < n; i++) d8[i] = (int8_t)((int)src[i] + zp);
#else
    for (size_t i = 0; i < n; i++) d8[i] = (int8_t)((int)src[i] + zp);
#endif
  } else {
    // UINT8: same width and same codes, plain copy.
    memcpy(dst, src, n);
  }
}

// Run with the bound input tensor ALREADY filled (fused letterbox path).
// t_stage = {fill(0), run_ms, outputs_get_ms, dequant_ms}.
bool run_once_io_mem_filled(const Ctx& c, std::vector<float>* hm, std::vector<float>* off,
                            std::vector<rknn_output>* out, double* t_stage) {
  out->assign(c.n_out, rknn_output{});
  for (uint32_t i = 0; i < c.n_out; i++) {
    (*out)[i].index = i;
    (*out)[i].want_float = 0;
  }
  double a0 = now_ms();
  if (rknn_run(c.ctx, NULL) != 0) { fprintf(stderr, "rknn_run failed\n"); return false; }
  double a2 = now_ms();
  if (rknn_outputs_get(c.ctx, c.n_out, out->data(), NULL) != 0) {
    fprintf(stderr, "outputs_get failed\n");
    return false;
  }
  double a3 = now_ms();
  if (t_stage) { t_stage[0] = 0.0; t_stage[1] = a2 - a0; t_stage[2] = a3 - a2; t_stage[3] = 0; }
  double d0 = now_ms();
  if (hm) { if (!dequant((*out)[0], c.out_attr[0], hm)) return false; }
  if (off) { if (!dequant((*out)[1], c.out_attr[1], off)) return false; }
  if (t_stage) { t_stage[3] = now_ms() - d0; }
  return true;
}

// Fill the bound input tensor from a uint8 buffer, then run.
// t_stage = {fill_ms, run_ms, outputs_get_ms, dequant_ms}.
bool run_once_io_mem(const Ctx& c, const uint8_t* input, size_t n, float conv_div,
                     std::vector<float>* hm, std::vector<float>* off,
                     std::vector<rknn_output>* out, double* t_stage, bool nchw = false) {
  double a0 = now_ms();
  io_mem_fill_u8(c, input, n, conv_div, nchw);
  double a1 = now_ms();
  const double fill_ms = a1 - a0;
  if (!run_once_io_mem_filled(c, hm, off, out, t_stage)) return false;
  if (t_stage) t_stage[0] = fill_ms;   // the filled variant zeroes it; keep ours
  return true;
}


}  // namespace

// ---------------------------------------------------------------------------
#ifndef RKNN_HEATMAP_SMOKE_NO_MAIN
int main(int argc, char** argv) {
  if (argc < 3) {
    printf("Usage: %s <model.rknn> <feature.jpg> [--th F] [--stride N] [--topk N]\n"
           "          [--loops N] [--warmup N] [--perf] [--dump DIR] [--tag NAME]\n"
           "          [--dump-input DIR]   save the exact uint8 NHWC NPU input\n"
           "       %s <model.rknn> --batch <list.txt> [--th F] [--dump DIR]\n"
           "       [--io-mem]           zero-copy input (rknn_set_io_mem); no rknn_inputs_set\n"
           "       [--conv-div K]       value written into the input tensor = u8 / K (default 1)\n"
           "       [--cpu-affinity L]   pin this process to CPU list, e.g. 6 or 4-7\n",
           argv[0], argv[0]);
    return 2;
  }
  const char* model_path = argv[1];
  const char* image_path = nullptr;
  const char* batch_list = nullptr;
  const char* dump_dir = nullptr;
  const char* tag = "frame";
  int loops = 0, warmup = 0, top_k = 100, stride = 2;
  float thr = 0.25f;
  int use_perf = 0;
  const char* input_dump = nullptr;
  // Compact per-frame detection log. The human-readable block below only prints
  // when frames.size() <= 12, so a 600-frame batch produces no detections at all
  // unless the raw tensors are dumped -- and dumping 1.2 GB over NFS runs at
  // ~2 frames/minute. This flag writes one line per frame instead.
  const char* det_out = nullptr;
  // ---- zero-copy input experiment (opt-in; default path untouched) ---------
  int use_io_mem = 0;          // rknn_set_io_mem: never call rknn_inputs_set
  float conv_div = 1.0f;       // value written into the tensor = u8 / conv_div
  const char* cpu_affinity = nullptr;  // e.g. "6" or "4-7"
  int selftest_conv = 0;       // check the NEON u8->f16 against the portable one
  int bench_fill = 0;          // micro-benchmark: conversion vs device-memory write
  int fused_lb = 0;            // letterbox straight into the io-mem (no uint8 buffer)
  int io_mem_nchw = 0;         // write the bound tensor in NCHW order (transpose)
  int up_nchw = 0;             // upload the uint8 canvas as NCHW (skip the driver's transpose)
  int up_f16 = 0;              // upload our NEON fp16 buffer as FLOAT16 (typed, not pass-through)
  int selftest_lb = 0;         // check the fused letterbox against letterbox()+u8->f16

  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--batch") && i + 1 < argc)        batch_list = argv[++i];
    else if (!strcmp(argv[i], "--th") && i + 1 < argc)      thr = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "--stride") && i + 1 < argc)  stride = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--topk") && i + 1 < argc)    top_k = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--loops") && i + 1 < argc)   loops = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--warmup") && i + 1 < argc)  warmup = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--perf"))                    use_perf = 1;
    else if (!strcmp(argv[i], "--dump") && i + 1 < argc)    dump_dir = argv[++i];
    else if (!strcmp(argv[i], "--dump-input") && i + 1 < argc) input_dump = argv[++i];
    else if (!strcmp(argv[i], "--tag") && i + 1 < argc)     tag = argv[++i];
    else if (!strcmp(argv[i], "--detections-out") && i + 1 < argc) det_out = argv[++i];
    else if (!strcmp(argv[i], "--io-mem"))                  use_io_mem = 1;
    else if (!strcmp(argv[i], "--selftest-conv"))          selftest_conv = 1;
    else if (!strcmp(argv[i], "--bench-fill"))             bench_fill = 1;
    else if (!strcmp(argv[i], "--io-mem-fused"))           fused_lb = 1;
    else if (!strcmp(argv[i], "--io-mem-nchw"))           io_mem_nchw = 1;
    else if (!strcmp(argv[i], "--upload-nchw"))           up_nchw = 1;
    else if (!strcmp(argv[i], "--upload-f16"))            up_f16 = 1;
    else if (!strcmp(argv[i], "--selftest-lb"))            selftest_lb = 1;
    else if (!strcmp(argv[i], "--conv-div") && i + 1 < argc) conv_div = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "--cpu-affinity") && i + 1 < argc) cpu_affinity = argv[++i];
    else if (!image_path && argv[i][0] != '-')              image_path = argv[i];
    else { printf("unknown option: %s\n", argv[i]); return 2; }
  }

  // The NEON u8->f16 conversion is hand-written bit manipulation, so it gets
  // its own self-test: every one of the 256 possible byte values, at every
  // div we might use, against the portable reference implementation.
  if (selftest_conv) {
    int bad = 0;
    for (int trial = 0; trial < 2; trial++) {
      float dv = trial == 0 ? 1.0f : 255.0f;
      uint8_t src[256];
      uint16_t a[256], b[256];
      for (int i = 0; i < 256; i++) src[i] = (uint8_t)i;
      u8_to_f16(src, a, 256, dv);
      for (int i = 0; i < 256; i++) b[i] = f32_to_f16_bits((float)src[i] * (1.0f / dv));
      for (int i = 0; i < 256; i++) {
        if (a[i] != b[i]) {
          if (bad < 10)
            printf("[selftest] div=%g u8=%3d neon=%04x ref=%04x (%.8f vs %.8f)\n", dv, i, a[i], b[i],
                   fp16_to_f32(a[i]), fp16_to_f32(b[i]));
          bad++;
        }
      }
      // also check the exactness claim: raw bytes are exact in fp16
      if (dv == 1.0f) {
        for (int i = 0; i < 256; i++) {
          if (fp16_to_f32(a[i]) != (float)i) {
            printf("[selftest] div=1 u8=%d not exact (%.8f)\n", i, fp16_to_f32(a[i]));
            bad++;
          }
        }
      }
    }
    printf("[selftest] u8->f16: %s (%d mismatches)\n", bad == 0 ? "PASS" : "FAIL", bad);
    return bad == 0 ? 0 : 1;
  }

  // The fused letterbox must be BIT-EXACT with letterbox() + u8->f16, or the
  // zero-copy path is a different model, not a faster one.
  if (selftest_lb) {
    int bad = 0;
    for (int trial = 0; trial < 3; trial++) {
      int sw = (trial == 0) ? 512 : (trial == 1) ? 640 : 501;
      int sh = (trial == 0) ? 512 : (trial == 1) ? 512 : 377;
      std::vector<uint8_t> src((size_t)sw * sh * 3);
      for (size_t i = 0; i < src.size(); i++) src[i] = (uint8_t)(i * 31 + (i >> 7));
      std::vector<uint8_t> boxed;
      letterbox(src.data(), sw, sh, kDefaultSize, &boxed);
      std::vector<uint16_t> fused((size_t)kDefaultSize * kDefaultSize * 3);
      letterbox_to_f16(src.data(), sw, sh, kDefaultSize, fused.data());
      std::vector<uint16_t> sep(boxed.size());
      u8_to_f16_int(boxed.data(), sep.data(), boxed.size());
      int bad_trial = 0;
      for (size_t i = 0; i < fused.size(); i++) {
        if (fused[i] != sep[i]) {
          bad_trial++;
          if (bad < 6) {
            // decode the fp16 back to the byte it represents (exact for 0..255)
            float fv = fp16_to_f32(fused[i]);
            float sv = fp16_to_f32(sep[i]);
            printf("[selftest-lb] %dx%d idx %zu (px %zu ch %zu): fused=%04x(%.1f) sep=%04x(%.1f) "
                   "u8=%d\n",
                   sw, sh, i, i / 3, i % 3, fused[i], fv, sep[i], sv, (int)boxed[i]);
          }
          bad++;
        }
      }
      printf("[selftest-lb] %dx%d: %s\n", sw, sh, bad_trial ? "DIFF" : "identical");
    }
    printf("[selftest-lb] fused letterbox: %s (%d mismatches)\n", bad == 0 ? "PASS" : "FAIL", bad);
    return bad == 0 ? 0 : 1;
  }



  // CPU affinity: rknn_inputs_set's widening is a single-threaded scalar
  // loop, so WHICH core it runs on is worth 3x (45.4 ms on an A55 vs 14.8 ms
  // on an A76). Applied to the whole process, because the driver call happens
  // on the calling thread.
  if (cpu_affinity) {
    cpu_set_t set;
    CPU_ZERO(&set);
    const char* p = cpu_affinity;
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
    if (sched_setaffinity(0, sizeof(set), &set) != 0)
      fprintf(stderr, "[WARN] sched_setaffinity(%s) failed\n", cpu_affinity);
    else
      printf("[0] cpu affinity pinned to %s\n", cpu_affinity);
  }

  printf("=====================================================================\n");
  printf(" RK3588 NPU -- Trial-0474 heatmap detector\n");
  printf(" model : %s\n", model_path);
  printf(" th=%.4f stride=%d topk=%d\n", thr, stride, top_k);
  printf("=====================================================================\n");

  // --- load + init ----------------------------------------------------------
  std::vector<uint8_t> model;
  if (!read_file(model_path, &model)) { fprintf(stderr, "cannot read model\n"); return 1; }
  printf("\n[1] model: %zu bytes, fnv1a32=%08x\n", model.size(), fnv1a(model.data(), model.size()));
  {
    double c0 = now_ms();
    usleep(50000);
    double c1 = now_ms();
    printf("[1] clock sanity: now_ms=%.3f, after 50ms usleep=%.3f, delta=%.3f ms%s\n", c0, c1,
           c1 - c0, (c1 - c0 > 49.0 && c1 - c0 < 52.0) ? " [ok]" : " [BAD -- durations below are meaningless]");
  }

  Ctx c;
  uint32_t flag = use_perf ? RKNN_FLAG_COLLECT_PERF_MASK : 0u;
  int ret = rknn_init(&c.ctx, model.data(), (uint32_t)model.size(), flag, NULL);
  if (ret != 0) { fprintf(stderr, "rknn_init failed ret=%d\n", ret); return 1; }
  printf("[2] rknn_init ok\n");

  rknn_sdk_version ver;
  memset(&ver, 0, sizeof(ver));
  if (rknn_query(c.ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver)) == 0)
    printf("    runtime api=%s driver=%s\n", ver.api_version, ver.drv_version);

  rknn_input_output_num io;
  memset(&io, 0, sizeof(io));
  rknn_query(c.ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
  printf("    n_input=%u n_output=%u\n", io.n_input, io.n_output);
  if (io.n_input != 1 || io.n_output != 2) {
    fprintf(stderr, "[FATAL] expected 1 input / 2 outputs (heatmap, offset)\n");
    return 1;
  }

  rknn_tensor_attr in_attr;
  memset(&in_attr, 0, sizeof(in_attr));
  in_attr.index = 0;
  rknn_query(c.ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
  printf("    input dims=[");
  for (uint32_t d = 0; d < in_attr.n_dims; d++) printf("%u%s", in_attr.dims[d], d + 1 < in_attr.n_dims ? "," : "");
  printf("] fmt=%s type=%s\n", tensor_fmt_name(in_attr.fmt), tensor_type_name(in_attr.type));
  printf("    input attr extra: zp=%d scale=%.10f size=%u size_with_stride=%u w_stride=%u\n",
         in_attr.zp, (double)in_attr.scale, in_attr.size, in_attr.size_with_stride,
         in_attr.w_stride);
  c.upload_nchw = (up_nchw != 0);
  c.input_is_f16 = (in_attr.type == RKNN_TENSOR_FLOAT16);
  c.input_is_f32 = (in_attr.type == RKNN_TENSOR_FLOAT32);
  c.input_is_int8 = (in_attr.type == RKNN_TENSOR_INT8);
  c.input_zp = in_attr.zp;
  if (in_attr.fmt == RKNN_TENSOR_NCHW) {
    c.in_c = (int)in_attr.dims[1]; c.in_h = (int)in_attr.dims[2]; c.in_w = (int)in_attr.dims[3];
  } else {
    c.in_h = (int)in_attr.dims[1]; c.in_w = (int)in_attr.dims[2]; c.in_c = (int)in_attr.dims[3];
  }
  if (c.in_h != kDefaultSize || c.in_w != kDefaultSize || c.in_c != 3) {
    fprintf(stderr, "[FATAL] model input is %dx%dx%d, expected 640x640x3\n", c.in_h, c.in_w, c.in_c);
    return 1;
  }

  // ---- zero-copy input buffer ---------------------------------------------
  rknn_tensor_mem* in_mem = nullptr;
  if (use_io_mem) {
    uint32_t mem_sz = in_attr.size_with_stride ? in_attr.size_with_stride : in_attr.size;
    in_mem = rknn_create_mem(c.ctx, mem_sz);
    if (!in_mem) { fprintf(stderr, "[FATAL] rknn_create_mem(%u) failed\n", mem_sz); return 1; }
    int rc = rknn_set_io_mem(c.ctx, in_mem, &in_attr);
    if (rc != 0) {
      fprintf(stderr, "[FATAL] rknn_set_io_mem failed rc=%d\n", rc);
      return 1;
    }
    c.io_mem = in_mem;
    printf("    [io-mem] zero-copy input: mem=%p size=%u (attr size=%u) conv_div=%.3f\n",
           (void*)in_mem->virt_addr, mem_sz, in_attr.size, conv_div);
  }

  // Where does the zero-copy fill time actually go? The conversion itself is
  // pure compute, but the io-mem is DEVICE memory (uncached), so the stores
  // are memory-mapped writes, not cache writes. This separates the two.
  if (bench_fill) {
    const size_t N = 640 * 640 * 3;
    std::vector<uint8_t> src(N);
    for (size_t i = 0; i < N; i++) src[i] = (uint8_t)(i * 7 + (i >> 5));
    std::vector<uint16_t> heap(N);           // cached, ordinary memory
    const int iters = 20;
    double t0 = now_ms();
    for (int k = 0; k < iters; k++) u8_to_f16(src.data(), heap.data(), N, 1.0f);
    double t_heap = (now_ms() - t0) / iters;
    double t0b = now_ms();
    for (int k = 0; k < iters; k++) memcpy(heap.data(), src.data(), N);
    double t_memcpy = (now_ms() - t0b) / iters;
    printf("[bench-fill] u8->f16 into CACHED heap : %.3f ms/frame\n", t_heap);
    printf("[bench-fill] memcpy 1.2 MB into heap   : %.3f ms/frame\n", t_memcpy);
    if (use_io_mem) {
      double t1 = now_ms();
      for (int k = 0; k < iters; k++) u8_to_f16(src.data(), (uint16_t*)in_mem->virt_addr, N, 1.0f);
      double t_dev = (now_ms() - t1) / iters;
      printf("[bench-fill] u8->f16 into DEVICE mem  : %.3f ms/frame\n", t_dev);
      double t2 = now_ms();
      for (int k = 0; k < iters; k++) memcpy(in_mem->virt_addr, src.data(), N);
      double t_devcopy = (now_ms() - t2) / iters;
      printf("[bench-fill] memcpy 1.2 MB into device: %.3f ms/frame\n", t_devcopy);
    }
    return 0;
  }

  c.n_out = io.n_output;
  c.out_attr.assign(c.n_out, rknn_tensor_attr{});
  for (uint32_t i = 0; i < c.n_out; i++) {
    memset(&c.out_attr[i], 0, sizeof(rknn_tensor_attr));
    c.out_attr[i].index = i;
    rknn_query(c.ctx, RKNN_QUERY_OUTPUT_ATTR, &c.out_attr[i], sizeof(c.out_attr[i]));
    printf("    out[%u] dims=[", i);
    for (uint32_t d = 0; d < c.out_attr[i].n_dims; d++)
      printf("%u%s", c.out_attr[i].dims[d], d + 1 < c.out_attr[i].n_dims ? "," : "");
    printf("] fmt=%s type=%s zp=%d scale=%.8f\n", tensor_fmt_name(c.out_attr[i].fmt),
           tensor_type_name(c.out_attr[i].type), c.out_attr[i].zp, (double)c.out_attr[i].scale);
  }
  {
    const rknn_tensor_attr& a = c.out_attr[0];
    if (a.n_dims != 4 || a.dims[0] != 1 || a.dims[1] != 1) {
      fprintf(stderr, "[FATAL] out[0] is not [1,1,H,W]\n"); return 1;
    }
    c.hm_h = (int)a.dims[2]; c.hm_w = (int)a.dims[3];
    const rknn_tensor_attr& b = c.out_attr[1];
    if (b.n_dims != 4 || b.dims[0] != 1 || b.dims[1] != 2) {
      fprintf(stderr, "[FATAL] out[1] is not [1,2,H,W]\n"); return 1;
    }
    c.off_h = (int)b.dims[2]; c.off_w = (int)b.dims[3];
    if (c.off_h != c.hm_h || c.off_w != c.hm_w) {
      fprintf(stderr, "[FATAL] heatmap %dx%d and offset %dx%d grids disagree\n", c.hm_h, c.hm_w,
              c.off_h, c.off_w);
      return 1;
    }
    if (c.in_h / c.hm_h != stride || c.in_w / c.hm_w != stride) {
      fprintf(stderr, "[FATAL] input/grid ratio %d does not match --stride %d\n",
              c.in_h / c.hm_h, stride);
      return 1;
    }
    printf("    heatmap %dx%d, offset %dx%d, implied stride %d\n", c.hm_h, c.hm_w, c.off_h, c.off_w,
           c.in_h / c.hm_h);
  }

  if (!image_path && !batch_list) { printf("need an image or --batch list\n"); return 2; }

  // --- frames ---------------------------------------------------------------
  std::vector<std::string> frames;
  if (batch_list) {
    FILE* f = fopen(batch_list, "r");
    if (!f) { fprintf(stderr, "cannot read batch list\n"); return 1; }
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
      std::string s(line);
      while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
      if (!s.empty()) frames.push_back(s);
    }
    fclose(f);
    printf("[3] batch mode: %zu frames from %s\n", frames.size(), batch_list);
  } else {
    frames.push_back(image_path);
  }

  std::vector<uint8_t> input;
  std::vector<float> hm, off;
  std::vector<rknn_output> out;
  std::vector<Det> dets;

  // One dispatch point for both input paths so the control/determinism logic
  // and the timing bookkeeping stay identical.
  auto run_frame = [&](const std::vector<uint8_t>& in, std::vector<float>* h,
                       std::vector<float>* o, std::vector<rknn_output>* outp,
                       double* ts) -> bool {
    if (up_f16) {
      // Our NEON conversion (0.78 ms) + a typed FLOAT16 upload: the runtime
      // then stages the buffer into NPU memory the same way it does for the
      // uint8 path, and only the type conversion is skipped.
      static thread_local std::vector<uint16_t> f16buf;
      f16buf.resize(in.size());
      u8_to_f16(in.data(), f16buf.data(), in.size(), conv_div);
      return run_once_f16(c, f16buf.data(), f16buf.size(), h, o, outp, ts);
    }
    if (up_nchw) {
      // Feed the driver the canvas in the NPU's internal layout, so it does
      // not have to transpose it. The transpose moves to the host.
      static thread_local std::vector<uint8_t> tr;
      tr.resize(in.size());
      const int hw = (int)std::lround(std::sqrt((double)in.size() / 3.0));
      nhwc_to_nchw_u8(in.data(), tr.data(), hw, 3);
      return run_once(c, tr, h, o, outp, ts);
    }
    if (use_io_mem) {
      if (fused_lb) return run_once_io_mem_filled(c, h, o, outp, ts);
      return run_once_io_mem(c, in.data(), in.size(), conv_div, h, o, outp, ts, io_mem_nchw != 0);
    }
    return run_once(c, in, h, o, outp, ts);
  };

  double sum_total = 0;
  double sum_stage[4] = {0, 0, 0, 0};
  // Per-frame host-side stages that the [7] block used to ignore entirely.
  // These are wall-clock costs the board actually pays, so they have to be in
  // the same accounting as inputs_set/rknn_run/outputs_get/dequant.
  double sum_load = 0, sum_lb = 0, sum_scan = 0, sum_peaks = 0, sum_write = 0;
  int n_timed = 0;

  std::ofstream detf;
  if (det_out) {
    detf.open(det_out, std::ios::out | std::ios::trunc);
    if (!detf) { fprintf(stderr, "cannot open --detections-out %s: %s\n", det_out, strerror(errno)); return 1; }
    printf("[3b] detection log -> %s\n", det_out);
  }

  for (size_t fi = 0; fi < frames.size(); fi++) {
    const std::string& path = frames[fi];
    double t_write = 0;
    int sw = 0, sh = 0, sc = 0;
    double t_ld0 = now_ms();
    uint8_t* src = stbi_load(path.c_str(), &sw, &sh, &sc, 3);
    if (!src) { fprintf(stderr, "stbi_load failed: %s\n", path.c_str()); return 1; }
    double t_load = now_ms() - t_ld0;
    const bool fused = (fused_lb && use_io_mem);
    std::vector<uint8_t> src_copy;          // only used by the fused null control
    double t_lb0 = now_ms();
    if (fused) {
      // Fused: the padded canvas goes straight into the bound input tensor as
      // fp16. No 1.2 MB uint8 buffer is materialised, and the bytes are
      // bit-identical to letterbox() + u8->f16 (--selftest-lb proves it).
      letterbox_to_f16(src, sw, sh, kDefaultSize, (uint16_t*)c.io_mem->virt_addr);
      src_copy.assign(src, src + (size_t)sw * sh * 3);
    } else {
      letterbox(src, sw, sh, kDefaultSize, &input);
    }
    double t_lb = now_ms() - t_lb0;
    free(src);

    // blank-input refusal (rules 4.2d: non-zero output != usable model)
    double t_sc0 = now_ms();
    int mn = 255, mx = 0;
    if (fused) {
      const uint16_t* p16 = (const uint16_t*)c.io_mem->virt_addr;
      const size_t n16 = (size_t)kDefaultSize * kDefaultSize * 3;
      for (size_t i = 0; i < n16; i++)
        if (p16[i] != 0) { mx = 1; break; }
    } else {
      for (uint8_t v : input) { mn = std::min(mn, (int)v); mx = std::max(mx, (int)v); }
    }
    double t_scan = now_ms() - t_sc0;
    // Guard on the MAX, not the min: a real infrared frame is full of zero
    // background, so testing `mn == 0` rejects every legitimate image and tests
    // nothing. Only an entirely zero buffer means the input path produced
    // nothing at all (rules 4.2d: non-zero output != usable model).
    if (mx == 0) { fprintf(stderr, "[FATAL] letterboxed input is all zero: %s\n", path.c_str()); return 1; }

    // Optional: dump the exact uint8 NHWC buffer that goes into the NPU, so the
    // host can hash it and diff against its own preprocessing. The board
    // decodes JPEG with stb_image while the host uses libjpeg-turbo via OpenCV,
    // and resizes with float bilinear while the host uses cv2's fixed-point
    // INTER_LINEAR -- neither is bit-identical, so "the NPU saw exactly what
    // PyTorch saw" has to be measured rather than assumed.
    if (input_dump && fused) {
      static bool warned = false;
      if (!warned) {
        warned = true;
        fprintf(stderr, "[WARN] --dump-input skipped: the fused path has no uint8 buffer\n");
      }
    } else if (input_dump) {
      const char* slash = strrchr(path.c_str(), '/');
      std::string stem_name = slash ? slash + 1 : path;
      for (auto& ch : stem_name)
        if (ch == '.') ch = '_';
      char path_buf[1200];
      snprintf(path_buf, sizeof(path_buf), "%s%s%s.input.bin", input_dump,
               (input_dump[strlen(input_dump) - 1] == '/') ? "" : "/", stem_name.c_str());
      if (mkdir(input_dump, 0755) != 0 && errno != EEXIST) { /* may already exist */ }
      if (!write_bin(path_buf, input.data(), input.size())) {
        fprintf(stderr, "[FATAL] could not write input dump %s\n", path_buf);
        return 1;
      }
      printf("[4] dumped input buffer -> %s (%zu B)\n", path_buf, input.size());
    }

    if (fi == 0) {
      printf("[4] preprocess: %dx%d -> %dx%d uint8 NHWC in %.2f ms, fnv1a32=%08x "
             "(min=%d max=%d)\n",
             sw, sh, kDefaultSize, kDefaultSize, t_lb, fnv1a(input.data(), input.size()), mn, mx);

      // null control + determinism, once, on the first frame
      if (!run_frame(input, &hm, &off, &out, nullptr)) return 1;
      uint32_t h1 = fnv1a(out[0].buf, out[0].size);
      uint32_t h1b = fnv1a(out[1].buf, out[1].size);
      release(c, &out);

      std::vector<uint8_t> inv;
      if (fused) {
        // The fused path has no uint8 buffer to invert; invert the SOURCE
        // instead and letterbox that. It is still a different input, which is
        // all this control has to prove.
        inv = src_copy;
        for (auto& v : inv) v = (uint8_t)(255 - v);
        letterbox_to_f16(inv.data(), sw, sh, kDefaultSize, (uint16_t*)c.io_mem->virt_addr);
      } else {
        inv = input;
        for (auto& v : inv) v = (uint8_t)(255 - v);
      }
      if (!run_frame(inv, nullptr, nullptr, &out, nullptr)) return 1;
      uint32_t hc = fnv1a(out[0].buf, out[0].size);
      uint32_t hcb = fnv1a(out[1].buf, out[1].size);
      release(c, &out);
      if (fused) {
        // The control just left the INVERTED canvas in the bound tensor; put
        // the real frame back before anything is detected from it.
        letterbox_to_f16(src_copy.data(), sw, sh, kDefaultSize, (uint16_t*)c.io_mem->virt_addr);
      }
      printf("[5] null control (inverted input): heatmap %08x -> %08x %s, offset %08x -> %08x %s\n",
             h1, hc, hc != h1 ? "CHANGED" : "IDENTICAL(BAD)", h1b, hcb, hcb != h1b ? "CHANGED" : "IDENTICAL(BAD)");
      if (hc == h1 || hcb == h1b) {
        fprintf(stderr, "[FATAL] NPU output independent of input -- results are not trustworthy\n");
        return 1;
      }

      if (!run_frame(input, &hm, &off, &out, nullptr)) return 1;
      uint32_t h2 = fnv1a(out[0].buf, out[0].size);
      printf("[5] determinism: run#1=%08x run#2=%08x -> %s\n", h1, h2, h1 == h2 ? "IDENTICAL" : "DIFFER");
      release(c, &out);
    } else {
      // Batch mode times THIS inference -- the one that actually produces the
      // detections. It used to be run untimed and then repeated a second time
      // purely to fill the [7] block, so every frame paid 2x rknn_run while the
      // report quoted the cost of only one. The repeat is discarded (hm/off are
      // nullptr), so dropping it cannot change any detection.
      double ts[4] = {0, 0, 0, 0};
      if (!run_frame(input, &hm, &off, &out, ts)) return 1;
      for (int s = 0; s < 4; s++) sum_stage[s] += ts[s];
      sum_total += ts[0] + ts[1] + ts[2] + ts[3];
      n_timed++;
    }

    // peak decode on the board
    double t_pk0 = now_ms();
    extract_peaks(hm.data(), off.data(), c.hm_h, c.hm_w, stride, thr, top_k, &dets);
    double t_peaks = now_ms() - t_pk0;

    const char* base = strrchr(path.c_str(), '/');
    base = base ? base + 1 : path.c_str();
    float hm_max = 0;
    for (float v : hm) hm_max = std::max(hm_max, v);

    if (det_out) {
      // Format: <stem>\t<ndet>\t<x y score>...   -- coordinates are canvas-space
      // and already stride-compensated, matching hm_common.extract_peaks_np.
      double t_w0 = now_ms();
      detf << base << "\t" << dets.size();
      for (size_t k = 0; k < dets.size(); k++)
        detf << "\t" << dets[k].x << " " << dets[k].y << " " << dets[k].score;
      detf << "\n";
      detf.flush();
      t_write = now_ms() - t_w0;
    }

    if (frames.size() <= 12) {
      printf("\n--- %s  (src %dx%d, hm max=%.6f, th=%.4f) ---\n", base, sw, sh, hm_max, thr);
      printf("    detections: %zu\n", dets.size());
      for (size_t k = 0; k < dets.size() && k < 20; k++)
        printf("      [%2zu] centre_x=%8.3f centre_y=%8.3f  score=%.6f\n", k, dets[k].x, dets[k].y, dets[k].score);
    }

    if (dump_dir) {
      std::string stem = std::string(dump_dir) + "/" + tag + "_" + base;
      for (auto& ch : stem) if (ch == '.') ch = '_';
      // strip the trailing "_jpg" we just created
      if (stem.size() > 4) stem = stem.substr(0, stem.size() - 4);
      bool ok1 = write_bin(stem + "_hm.bin", hm.data(), hm.size() * 4);
      bool ok2 = write_bin(stem + "_off.bin", off.data(), off.size() * 4);
      if (fi == 0)
        printf("[6] dumped raw outputs -> %s_hm.bin (%zu B), %s_off.bin (%zu B) [%s]\n",
               stem.c_str(), hm.size() * 4, stem.c_str(), off.size() * 4,
               (ok1 && ok2) ? "ok" : "WRITE-FAILED");
      if (!ok1 || !ok2) return 1;
    }

    // Accumulate the host-side stages. In batch mode every frame pays all of
    // them exactly once; summing them here keeps [7] an honest per-frame total
    // instead of a partial one.
    sum_load += t_load;
    sum_lb += t_lb;
    sum_scan += t_scan;
    sum_peaks += t_peaks;
    sum_write += t_write;

    // timing
    // Single-image mode keeps its benchmark behaviour (--warmup/--loops repeats).
    // Batch mode must NOT: the per-frame inference above is already timed, so
    // repeating it here would re-introduce the duplicate rknn_run per frame.
    if (batch_list) continue;
    int w = warmup;
    for (int k = 0; k < w; k++) { run_frame(input, nullptr, nullptr, &out, nullptr); release(c, &out); }
    int n = std::max(1, loops);
    for (int k = 0; k < n; k++) {
      double ts[4] = {0, 0, 0, 0};
      if (!run_frame(input, nullptr, nullptr, &out, ts)) return 1;
      release(c, &out);
      sum_total += ts[0] + ts[1] + ts[2];
      sum_stage[0] += ts[0];
      sum_stage[1] += ts[1];
      sum_stage[2] += ts[2];
      n_timed++;
    }
  }

  if (n_timed) {
    double avg = sum_total / n_timed;
    double h_load = sum_load / n_timed, h_lb = sum_lb / n_timed, h_scan = sum_scan / n_timed;
    double h_peaks = sum_peaks / n_timed, h_write = sum_write / n_timed;
    double host = h_load + h_lb + h_scan + h_peaks + h_write;
    printf("\n[7] latency over %d timed runs (full per-frame cost)\n", n_timed);
    printf("    --- NPU stage ---\n");
    if (use_io_mem)
      printf("    input fill   avg %7.3f ms  (host fill into the io-mem; no inputs_set)\n",
             sum_stage[0] / n_timed);
    else
      printf("    inputs_set   avg %7.3f ms\n", sum_stage[0] / n_timed);
    printf("    rknn_run     avg %7.3f ms  (%.1f%%)\n", sum_stage[1] / n_timed,
           100.0 * sum_stage[1] / sum_total);
    printf("    outputs_get  avg %7.3f ms  (%.1f%%)   [1.2 MB of float32 tensors]\n",
           sum_stage[2] / n_timed, 100.0 * sum_stage[2] / sum_total);
    printf("    dequant      avg %7.3f ms  (%.1f%%)\n",
           sum_stage[3] / n_timed, 100.0 * sum_stage[3] / sum_total);
    printf("    --- host stage ---\n");
    printf("    stbi_load    avg %7.3f ms  (file read + BMP decode)\n", h_load);
    printf("    letterbox    avg %7.3f ms\n", h_lb);
    printf("    minmax scan  avg %7.3f ms  (640*640*3 bytes)\n", h_scan);
    printf("    extract_peaks avg %7.3f ms\n", h_peaks);
    printf("    det write    avg %7.3f ms  (incl. per-frame flush)\n", h_write);
    printf("    ---\n");
    printf("    NPU+dequant  avg %7.3f ms  -> %.1f fps\n", avg, 1000.0 / avg);
    printf("    + host       avg %7.3f ms  -> %.1f fps   <-- real per-frame cost\n",
           avg + host, 1000.0 / (avg + host));
    printf("    host share   %7.1f%% of the per-frame total\n", 100.0 * host / (avg + host));
    if (use_perf) printf("    (--perf ON: RKNN_FLAG_COLLECT_PERF_MASK inflates rknn_run)\n");
  }
  printf("=====================================================================\n");
  rknn_destroy(c.ctx);
  return 0;
}
#endif  // RKNN_HEATMAP_SMOKE_NO_MAIN
