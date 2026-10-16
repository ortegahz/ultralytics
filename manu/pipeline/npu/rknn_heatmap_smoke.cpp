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
// Peak extraction -- 1:1 with manu/evaluation/heatmap_evaluate.py::extract_peaks
// ---------------------------------------------------------------------------
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
  bool input_is_f16 = false;
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
    in.fmt = RKNN_TENSOR_NHWC;
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

void release(Ctx& c, std::vector<rknn_output>* out) {
  if (!out->empty() && (*out)[0].buf) rknn_outputs_release(c.ctx, c.n_out, out->data());
}

}  // namespace

// ---------------------------------------------------------------------------
// Reuse hook. Defining RKNN_HEATMAP_SMOKE_NO_MAIN lets e2e_core.cpp #include
// this file and call the very same letterbox / run_once / dequant /
// extract_peaks on the very same Ctx. The main() body below is NOT touched, so
// the e2e demo and this binary produce identical heatmaps by construction.
// ---------------------------------------------------------------------------
#ifndef RKNN_HEATMAP_SMOKE_NO_MAIN
int main(int argc, char** argv) {
  if (argc < 3) {
    printf("Usage: %s <model.rknn> <feature.jpg> [--th F] [--stride N] [--topk N]\n"
           "          [--loops N] [--warmup N] [--perf] [--dump DIR] [--tag NAME]\n"
           "          [--dump-input DIR]   save the exact uint8 NHWC NPU input\n"
           "       %s <model.rknn> --batch <list.txt> [--th F] [--dump DIR]\n",
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
    else if (!image_path && argv[i][0] != '-')              image_path = argv[i];
    else { printf("unknown option: %s\n", argv[i]); return 2; }
  }
  if (!image_path && !batch_list) { printf("need an image or --batch list\n"); return 2; }

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
  if (in_attr.fmt == RKNN_TENSOR_NCHW) {
    c.in_c = (int)in_attr.dims[1]; c.in_h = (int)in_attr.dims[2]; c.in_w = (int)in_attr.dims[3];
  } else {
    c.in_h = (int)in_attr.dims[1]; c.in_w = (int)in_attr.dims[2]; c.in_c = (int)in_attr.dims[3];
  }
  if (c.in_h != kDefaultSize || c.in_w != kDefaultSize || c.in_c != 3) {
    fprintf(stderr, "[FATAL] model input is %dx%dx%d, expected 640x640x3\n", c.in_h, c.in_w, c.in_c);
    return 1;
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
    double t_lb0 = now_ms();
    letterbox(src, sw, sh, kDefaultSize, &input);
    double t_lb = now_ms() - t_lb0;
    free(src);

    // blank-input refusal (rules 4.2d: non-zero output != usable model)
    double t_sc0 = now_ms();
    int mn = 255, mx = 0;
    for (uint8_t v : input) { mn = std::min(mn, (int)v); mx = std::max(mx, (int)v); }
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
    if (input_dump) {
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
      if (!run_once(c, input, &hm, &off, &out, nullptr)) return 1;
      uint32_t h1 = fnv1a(out[0].buf, out[0].size);
      uint32_t h1b = fnv1a(out[1].buf, out[1].size);
      release(c, &out);

      std::vector<uint8_t> inv = input;
      for (auto& v : inv) v = (uint8_t)(255 - v);
      if (!run_once(c, inv, nullptr, nullptr, &out, nullptr)) return 1;
      uint32_t hc = fnv1a(out[0].buf, out[0].size);
      uint32_t hcb = fnv1a(out[1].buf, out[1].size);
      release(c, &out);
      printf("[5] null control (inverted input): heatmap %08x -> %08x %s, offset %08x -> %08x %s\n",
             h1, hc, hc != h1 ? "CHANGED" : "IDENTICAL(BAD)", h1b, hcb, hcb != h1b ? "CHANGED" : "IDENTICAL(BAD)");
      if (hc == h1 || hcb == h1b) {
        fprintf(stderr, "[FATAL] NPU output independent of input -- results are not trustworthy\n");
        return 1;
      }

      if (!run_once(c, input, &hm, &off, &out, nullptr)) return 1;
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
      if (!run_once(c, input, &hm, &off, &out, ts)) return 1;
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
    for (int k = 0; k < w; k++) { run_once(c, input, nullptr, nullptr, &out, nullptr); release(c, &out); }
    int n = std::max(1, loops);
    for (int k = 0; k < n; k++) {
      double ts[4] = {0, 0, 0, 0};
      if (!run_once(c, input, nullptr, nullptr, &out, ts)) return 1;
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
