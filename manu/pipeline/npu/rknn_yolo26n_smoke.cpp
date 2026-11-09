// ===========================================================================
// rknn_yolo26n_smoke.cpp
//
// RK3588 NPU inference example for the STOCK Ultralytics YOLO26n detector.
//
// Why this file exists
// --------------------
// The project's own detector is YOLO26HeatmapDetector (YOLO26 + P2 Neck +
// 320x320 heatmap head, 21.48 G MAC, 1.79 M params). Its NPU time on this
// board is 55.5 ms (single-scale probe) / 108.2 ms (mosaic e2e). The open
// question is how a STOCK YOLO26n (P3-P5, 6.1 GFLOPs) behaves on the SAME
// NPU, so every structural change has a reference point instead of a guess.
//
// This example is deliberately modelled on Rockchip's own SDK example
// (external/rknpu2/examples/rknn_yolov5_demo): the classic
// rknn_inputs_set -> rknn_run -> rknn_outputs_get flow, host-side decode and
// NMS, and a warmup + timed loop. Two deliberate departures from that demo:
//
//   1. The SDK demo STRETCHES the image to the model size (it uses RGA and
//      changes the aspect ratio). A detector must LETTERBOX instead, so this
//      file carries its own bilinear letterbox with pad 114, identical to the
//      one in rknn_heatmap_smoke.cpp (and to cv2.INTER_LINEAR on the host).
//   2. The SDK demo decodes YOLOv5 anchors. YOLO26 is ANCHOR-FREE with
//      reg_max=1 (no DFL), so its head output is already xyxy in canvas
//      pixels -- no anchors, no stride arithmetic, just decode + NMS.
//
// Model contract (measured, not assumed -- the binary PRINTS the attrs it
// actually got from rknn_query and asserts against these):
//
//   input : int8, NCHW, (1, 3, 640, 640), quantized affine
//           scale = 1/255, zp = -128  =>  the code of a host byte x is x + zp.
//   output: int8, (1, 84, 8400)
//           channels [0:4]   = xyxy, NORMALIZED to [0,1]  (multiply by 640)
//           channels [4:84]  = 80 class scores (already sigmoid'd in-graph)
//           8400 = 6400 (P3/8) + 1600 (P4/16) + 400 (P5/32), P3 first.
//
//   The normalized box coords come from Ultralytics' own RKNN int8 path
//   (_NormalizeCoords, ultralytics/utils/export/engine.py:17): pixel-valued
//   coordinates have a range that per-tensor int8 flattens, so they are
//   divided by the input size to keep relative precision.
//
// Build (cross, single file, no CMake -- see build_rknn_yolo26n_arm64.sh):
//   aarch64-rockchip-linux-gnu-g++ -std=c++17 -O2 rknn_yolo26n_smoke.cpp
//       -o rknn_yolo26n_smoke -lrknnrt
//
// The board has NO OpenCV and NO RGA, so this file uses the SDK's header-only
// stb_image for decoding and writes OSD with stb_image_write.
//
// Usage on the board:
//   ./rknn_yolo26n_smoke model.rknn image.bmp --loops 30 --warmup 10
//       --th 0.25 --detections-out out.tsv --osd out.jpg
// ===========================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

#include "rknn_api.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

constexpr int kModelSize = 640;    // model input edge
constexpr int kNumClasses = 80;    // COCO classes (stock yolo26n)
constexpr int kNumChannels = 3;
constexpr uint8_t kPadValue = 114; // letterbox pad, same as the project uses
constexpr int kMaxDet = 300;       // Ultralytics Detect.max_det

// ---------------------------------------------------------------------------
// 5x7 bitmap font, digits + '.' only, so the OSD can show a class id and a
// confidence without pulling in a font library the board does not have.
// ---------------------------------------------------------------------------
struct Glyph {
  char c;
  uint8_t rows[7];  // bit 4 = leftmost pixel
};

const Glyph kFont[] = {
    {'0', {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}},
    {'1', {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}},
    {'2', {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}},
    {'3', {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E}},
    {'4', {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}},
    {'5', {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E}},
    {'6', {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}},
    {'7', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}},
    {'8', {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}},
    {'9', {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C}},
    {'.', {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C}},
};

const uint8_t* glyph_for(char c) {
  for (const auto& g : kFont)
    if (g.c == c) return g.rows;
  return nullptr;
}

void draw_char(std::vector<uint8_t>* rgb, int w, int h, int x, int y, char c, const uint8_t color[3]) {
  const uint8_t* rows = glyph_for(c);
  if (!rows) return;
  for (int r = 0; r < 7; r++) {
    for (int b = 0; b < 5; b++) {
      if (!(rows[r] & (1 << (4 - b)))) continue;
      int px = x + b, py = y + r;
      if (px < 0 || py < 0 || px >= w || py >= h) continue;
      size_t o = ((size_t)py * w + px) * 3;
      (*rgb)[o + 0] = color[0];
      (*rgb)[o + 1] = color[1];
      (*rgb)[o + 2] = color[2];
    }
  }
}

void draw_text(std::vector<uint8_t>* rgb, int w, int h, int x, int y, const std::string& s,
               const uint8_t color[3]) {
  for (size_t i = 0; i < s.size(); i++)
    draw_char(rgb, w, h, x + (int)i * 6, y, s[i], color);
}

// ---------------------------------------------------------------------------
// Timing helper. std::chrono::steady_clock, because a previous version of the
// project's smoke tool used clock_gettime wrongly and reported 5.4e11 ms.
// ---------------------------------------------------------------------------
double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

// FNV-1a over the raw output bytes: enough to prove "same input -> same
// output" and "different input -> different output" without pulling in a
// crypto library.
uint64_t fnv1a(const void* data, size_t n) {
  const uint8_t* p = (const uint8_t*)data;
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n; i++) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

// ---------------------------------------------------------------------------
// Letterbox: full 2-D bilinear with half-pixel centres (cv2.INTER_LINEAR
// equivalent), pad 114.
//
// The scale ratio is NOT capped at 1.0 on purpose: a 640x512 source gives
// r = 1.0 and 64 px of padding, while a SQUARE 512x512 source is upscaled to
// 640x640. Capping the ratio would silently corrupt every square source.
// ---------------------------------------------------------------------------
void letterbox(const uint8_t* src, int sw, int sh, int size, std::vector<uint8_t>* dst) {
  double r = std::min((double)size / (double)sh, (double)size / (double)sw);
  int nw = (int)lround(sw * r), nh = (int)lround(sh * r);
  int dw = (size - nw) / 2, dh = (size - nh) / 2;
  int left = (int)lround(dw - 0.1), top = (int)lround(dh - 0.1);
  dst->assign((size_t)size * size * 3, kPadValue);

  if (nw == sw && nh == sh) {
    for (int y = 0; y < sh; y++)
      memcpy(dst->data() + (size_t)(y + top) * size * 3 + left * 3, src + (size_t)y * sw * 3,
             (size_t)sw * 3);
  } else {
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
}

// ---------------------------------------------------------------------------
// Fill the model's input tensor.
//
// The model's input is INT8 NCHW with an affine quantization (scale ~1/255,
// zp ~-128). An int8 tensor holds CODES, not values: the driver reinterprets
// the byte, so the code of a host byte x is x + zp. Copying the bytes
// verbatim would decode 0x80 as -128 instead of 0 and silently change every
// detection.
//
// The general affine form is used (not a hardcoded x-128) so the code stays
// correct if the toolkit ever picks a different scale for the input:
//     real = (code - zp) * scale   must equal   x / 255
//     code = round(x / 255 / scale) + zp
// ---------------------------------------------------------------------------
void fill_input(const uint8_t* canvas_nhwc, int size, rknn_tensor_type type, int zp, float scale,
                std::vector<uint8_t>* buf) {
  const size_t n = (size_t)size * size * 3;
  buf->assign(n, 0);
  const float inv = 1.0f / (255.0f * scale);

  if (type == RKNN_TENSOR_INT8) {
    int8_t* dst = (int8_t*)buf->data();
    for (size_t i = 0; i < n; i++) {
      int q = (int)lroundf((float)canvas_nhwc[i] * inv) + zp;
      dst[i] = (int8_t)std::max(-128, std::min(127, q));
    }
  } else if (type == RKNN_TENSOR_UINT8) {
    memcpy(buf->data(), canvas_nhwc, n);
  } else if (type == RKNN_TENSOR_FLOAT32) {
    float* dst = (float*)buf->data();
    for (size_t i = 0; i < n; i++) dst[i] = (float)canvas_nhwc[i] / 255.0f;
  } else {
    fprintf(stderr, "[FATAL] unsupported input tensor type %d\n", (int)type);
    exit(2);
  }
}

// NHWC canvas -> NCHW planar, in place order (used when the model wants NCHW).
void nhwc_to_nchw_u8(const uint8_t* src, int size, uint8_t* dst) {
  const size_t plane = (size_t)size * size;
  for (int c = 0; c < 3; c++)
    for (size_t i = 0; i < plane; i++)
      dst[c * plane + i] = src[i * 3 + c];
}

struct Det {
  float x1, y1, x2, y2;
  int cls;
  float score;
};

float iou(const Det& a, const Det& b) {
  float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
  float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
  float w = ix2 - ix1, h = iy2 - iy1;
  if (w <= 0 || h <= 0) return 0.f;
  float inter = w * h;
  float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0.f;
}

// Per-class NMS, matching Ultralytics' agnostic_nms=False semantics.
std::vector<Det> nms_per_class(std::vector<Det>& cand, float iou_thr) {
  std::vector<Det> kept;
  for (int c = 0; c < kNumClasses; c++) {
    std::vector<Det> cls;
    for (const auto& d : cand)
      if (d.cls == c) cls.push_back(d);
    if (cls.empty()) continue;
    std::sort(cls.begin(), cls.end(), [](const Det& a, const Det& b) { return a.score > b.score; });
    std::vector<Det> keep;
    for (const auto& d : cls) {
      bool ok = true;
      for (const auto& k : keep)
        if (iou(d, k) > iou_thr) { ok = false; break; }
      if (ok) keep.push_back(d);
    }
    kept.insert(kept.end(), keep.begin(), keep.end());
  }
  std::sort(kept.begin(), kept.end(), [](const Det& a, const Det& b) { return a.score > b.score; });
  if ((int)kept.size() > kMaxDet) kept.resize(kMaxDet);
  return kept;
}

bool load_model(const char* path, std::vector<uint8_t>* out) {
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "[FATAL] cannot open model %s\n", path); return false; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz <= 0) { fprintf(stderr, "[FATAL] bad model size %ld\n", sz); fclose(f); return false; }
  out->resize((size_t)sz);
  size_t got = fread(out->data(), 1, (size_t)sz, f);
  fclose(f);
  if (got != (size_t)sz) { fprintf(stderr, "[FATAL] short read %zu/%ld\n", got, sz); return false; }
  return true;
}

void usage(const char* argv0) {
  fprintf(stderr,
          "usage: %s <model.rknn> <image> [options]\n"
          "  --loops N          timed iterations (default 20)\n"
          "  --warmup N         untimed warmup iterations (default 5)\n"
          "  --th F             score threshold (default 0.25)\n"
          "  --iou F            NMS IoU threshold (default 0.45)\n"
          "  --detections-out F write detections as TSV\n"
          "  --osd F            write an OSD image\n"
          "  --selftest         run determinism / inverted-input checks\n"
          "  --layout L         auto|nhwc|nchw  host input layout (default auto)\n"
          "  --pass-through 0|1 hand the input buffer to the NPU verbatim (default 1)\n"
          "  --core-mask M      0=auto 7=all three NPU cores\n",
          argv0);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) { usage(argv[0]); return 2; }

  const char* model_path = argv[1];
  const char* img_path = argv[2];
  int loops = 20, warmup = 5;
  float th = 0.25f, iou_thr = 0.45f;
  const char* det_out = nullptr;
  const char* osd_path = nullptr;
  bool selftest = false;
  // Input layout for the host buffer.
  //   auto -> follow what rknn_query reports for the input attr
  //   nhwc -> host buffer is HWC (what the runtime asked for)
  //   nchw -> host buffer is planar NCHW (what the MODEL's first conv wants)
  // The difference is not cosmetic: with NHWC the runtime has to transpose
  // 1.2 MB on every inputs_set, which measured 16 ms/frame.
  std::string layout = "auto";
  // pass_through=1 by default: measured byte-identical output (same FNV hash)
  // to pass_through=0, but inputs_set drops from 15.05 ms to 0.21 ms because
  // the runtime stops re-quantizing and transposing 1.2 MB every frame.
  int pass_through = 1;
  // NPU core mask: RKNN_NPU_CORE_AUTO (0, default) / 0_1_2 (7, all three
  // cores). RK3588 has three NPU cores; whether the runtime spreads one model
  // across them is a knob, not a given.
  int core_mask = 0;

  for (int i = 3; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) { fprintf(stderr, "[FATAL] %s needs a value\n", what); exit(2); }
      return argv[++i];
    };
    if (a == "--loops") loops = atoi(next("--loops"));
    else if (a == "--warmup") warmup = atoi(next("--warmup"));
    else if (a == "--th") th = (float)atof(next("--th"));
    else if (a == "--iou") iou_thr = (float)atof(next("--iou"));
    else if (a == "--detections-out") det_out = next("--detections-out");
    else if (a == "--osd") osd_path = next("--osd");
    else if (a == "--selftest") selftest = true;
    else if (a == "--layout") layout = next("--layout");
    else if (a == "--pass-through") pass_through = atoi(next("--pass-through"));
    else if (a == "--core-mask") core_mask = atoi(next("--core-mask"));
    else { fprintf(stderr, "[FATAL] unknown arg %s\n", argv[i]); usage(argv[0]); return 2; }
  }

  // --- model ---------------------------------------------------------------
  std::vector<uint8_t> model;
  if (!load_model(model_path, &model)) return 2;

  rknn_context ctx = 0;
  uint32_t flag = 0;  // NOT RKNN_FLAG_COLLECT_PERF_MASK: it inflates the time ~54%
  int ret = rknn_init(&ctx, model.data(), (uint32_t)model.size(), flag, NULL);
  if (ret != 0) { fprintf(stderr, "[FATAL] rknn_init failed ret=%d\n", ret); return 2; }

  if (core_mask != 0) {
    int rc = rknn_set_core_mask(ctx, (rknn_core_mask)core_mask);
    printf("[core  ] rknn_set_core_mask(%d) -> %d\n", core_mask, rc);
    if (rc != 0) fprintf(stderr, "[WARN] core mask not applied\n");
  } else {
    printf("[core  ] default (RKNN_NPU_CORE_AUTO)\n");
  }

  rknn_sdk_version ver;
  memset(&ver, 0, sizeof(ver));
  rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver));
  printf("[rknn  ] api_version=%s driver_version=%s\n", ver.api_version, ver.drv_version);

  rknn_input_output_num io;
  memset(&io, 0, sizeof(io));
  rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
  printf("[rknn  ] n_input=%u n_output=%u\n", io.n_input, io.n_output);
  if (io.n_input != 1 || io.n_output != 1) {
    fprintf(stderr, "[FATAL] expected 1 input / 1 output\n");
    rknn_destroy(ctx);
    return 2;
  }

  rknn_tensor_attr in_attr;
  memset(&in_attr, 0, sizeof(in_attr));
  in_attr.index = 0;
  rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
  printf("[in    ] n_dims=%u dims=[%u %u %u %u] fmt=%d type=%d zp=%d scale=%.8g size=%u\n",
         in_attr.n_dims, in_attr.dims[0], in_attr.dims[1], in_attr.dims[2], in_attr.dims[3],
         (int)in_attr.fmt, (int)in_attr.type, in_attr.zp, in_attr.scale, in_attr.size);

  rknn_tensor_attr out_attr;
  memset(&out_attr, 0, sizeof(out_attr));
  out_attr.index = 0;
  rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
  printf("[out   ] n_dims=%u dims=[%u %u %u %u] fmt=%d type=%d zp=%d scale=%.8g n_elems=%u size=%u\n",
         out_attr.n_dims, out_attr.dims[0], out_attr.dims[1], out_attr.dims[2], out_attr.dims[3],
         (int)out_attr.fmt, (int)out_attr.type, out_attr.zp, out_attr.scale, out_attr.n_elems,
         out_attr.size);

  // --- contract assertions (fail loudly instead of decoding garbage) -------
  const int kExpectedAnchors = 80 * 80 + 40 * 40 + 20 * 20;  // P3 + P4 + P5
  const int kExpectedChans = 4 + kNumClasses;
  if ((int)out_attr.n_elems != kExpectedChans * kExpectedAnchors) {
    fprintf(stderr, "[FATAL] output n_elems=%u, expected %d (%d chans x %d anchors)\n",
            out_attr.n_elems, kExpectedChans * kExpectedAnchors, kExpectedChans, kExpectedAnchors);
    rknn_destroy(ctx);
    return 2;
  }
  const bool model_wants_nchw = (in_attr.fmt == RKNN_TENSOR_NCHW);
  const bool use_nchw = (layout == "nchw") || (layout == "auto" && model_wants_nchw);
  if (layout != "auto" && layout != "nhwc" && layout != "nchw") {
    fprintf(stderr, "[FATAL] --layout must be auto|nhwc|nchw\n");
    rknn_destroy(ctx);
    return 2;
  }

  // --- image ---------------------------------------------------------------
  int sw = 0, sh = 0, sc = 0;
  uint8_t* img = stbi_load(img_path, &sw, &sh, &sc, 3);
  if (!img) { fprintf(stderr, "[FATAL] cannot load image %s\n", img_path); rknn_destroy(ctx); return 2; }
  printf("[image ] %s  %dx%d channels=%d\n", img_path, sw, sh, sc);

  std::vector<uint8_t> canvas;
  double t_lb0 = now_ms();
  letterbox(img, sw, sh, kModelSize, &canvas);
  double t_lb = now_ms() - t_lb0;

  // Letterbox geometry, needed to map canvas boxes back to source pixels.
  double r = std::min((double)kModelSize / (double)sh, (double)kModelSize / (double)sw);
  int nw = (int)lround(sw * r), nh = (int)lround(sh * r);
  int left = (int)lround(((kModelSize - nw) / 2) - 0.1), top = (int)lround(((kModelSize - nh) / 2) - 0.1);
  printf("[letter] %dx%d -> %dx%d  r=%.6f pad l=%d t=%d  (%.3f ms)\n", sw, sh, kModelSize, kModelSize, r,
         left, top, t_lb);

  // --- input buffer --------------------------------------------------------
  std::vector<uint8_t> in_buf;
  double t_fill = 0;
  {
    double t0 = now_ms();
    if (use_nchw) {
      std::vector<uint8_t> nchw;
      nhwc_to_nchw_u8(canvas.data(), kModelSize, nchw.data());
      fill_input(nchw.data(), kModelSize, in_attr.type, in_attr.zp, in_attr.scale, &in_buf);
    } else {
      fill_input(canvas.data(), kModelSize, in_attr.type, in_attr.zp, in_attr.scale, &in_buf);
    }
    t_fill = now_ms() - t0;
  }
  printf("[input ] layout=%s (runtime asked %s) type=%d bytes=%zu pass_through=%d  (fill %.3f ms)\n",
         use_nchw ? "NCHW" : "NHWC", model_wants_nchw ? "NCHW" : "NHWC", (int)in_attr.type,
         in_buf.size(), pass_through, t_fill);

  rknn_input in;
  memset(&in, 0, sizeof(in));
  in.index = 0;
  in.type = in_attr.type;
  in.fmt = use_nchw ? RKNN_TENSOR_NCHW : RKNN_TENSOR_NHWC;
  in.pass_through = (uint8_t)pass_through;
  in.size = (uint32_t)in_buf.size();
  in.buf = in_buf.data();

  rknn_output out;
  memset(&out, 0, sizeof(out));
  out.index = 0;
  out.want_float = 0;  // keep int8; dequantize on the host (want_float=1 would
                       // make the runtime materialize 2.8 MB of float per frame)

  // --- run once to see it work --------------------------------------------
  std::vector<uint8_t> out_bytes;
  auto run_one = [&](std::vector<uint8_t>* raw, double* t_in, double* t_run, double* t_get) -> bool {
    double a0 = now_ms();
    if (rknn_inputs_set(ctx, io.n_input, &in) != 0) { fprintf(stderr, "[FATAL] inputs_set\n"); return false; }
    double a1 = now_ms();
    if (rknn_run(ctx, NULL) != 0) { fprintf(stderr, "[FATAL] rknn_run\n"); return false; }
    double a2 = now_ms();
    if (rknn_outputs_get(ctx, io.n_output, &out, NULL) != 0) { fprintf(stderr, "[FATAL] outputs_get\n"); return false; }
    double a3 = now_ms();
    if (t_in) *t_in = a1 - a0;
    if (t_run) *t_run = a2 - a1;
    if (t_get) *t_get = a3 - a2;
    if (raw) {
      raw->assign((uint8_t*)out.buf, (uint8_t*)out.buf + out.size);
      rknn_outputs_release(ctx, io.n_output, &out);
    }
    return true;
  };

  std::vector<uint8_t> raw;
  if (!run_one(&raw, nullptr, nullptr, nullptr)) { rknn_destroy(ctx); return 2; }
  printf("[out   ] bytes=%u  hash=%016llx\n", out.size, (unsigned long long)fnv1a(raw.data(), raw.size()));

  // --- decode + NMS (host side) -------------------------------------------
  // The 4 box channels are (cx, cy, w, h) in canvas pixels, NOT xyxy:
  // Detect._get_decode_boxes -> decode_bboxes(xywh=True default) -> dist2bbox
  // returns centre+size, and _NormalizeCoords then divides by the input size.
  // (ultralytics/nn/modules/head.py:210,227-234)
  auto decode = [&](const std::vector<uint8_t>& buf, std::vector<Det>* dets) {
    const float scale = out_attr.scale;
    const int zp = out_attr.zp;
    auto q = [&](size_t i) { return ((float)((int8_t)buf[i]) - (float)zp) * scale; };
    dets->clear();
    for (int a = 0; a < kExpectedAnchors; a++) {
      float cx = q((size_t)0 * kExpectedAnchors + a) * kModelSize;
      float cy = q((size_t)1 * kExpectedAnchors + a) * kModelSize;
      float w = q((size_t)2 * kExpectedAnchors + a) * kModelSize;
      float h = q((size_t)3 * kExpectedAnchors + a) * kModelSize;
      int best = -1;
      float best_s = th;
      for (int c = 0; c < kNumClasses; c++) {
        float s = q((size_t)(4 + c) * kExpectedAnchors + a);
        if (s > best_s) { best_s = s; best = c; }
      }
      if (best < 0) continue;
      Det d;
      d.x1 = cx - w * 0.5f; d.y1 = cy - h * 0.5f;
      d.x2 = cx + w * 0.5f; d.y2 = cy + h * 0.5f;
      d.cls = best; d.score = best_s;
      dets->push_back(d);
    }
    *dets = nms_per_class(*dets, iou_thr);
  };

  std::vector<Det> dets;
  decode(raw, &dets);
  printf("[decode] %zu detections (th=%.2f)\n", dets.size(), th);

  // --- optional self-tests (determinism + inverted input) -----------------
  if (selftest) {
    std::vector<uint8_t> raw2;
    run_one(&raw2, nullptr, nullptr, nullptr);
    uint64_t h1 = fnv1a(raw.data(), raw.size()), h2 = fnv1a(raw2.data(), raw2.size());
    printf("[selftest] determinism: %s (%016llx vs %016llx)\n", h1 == h2 ? "PASS" : "FAIL",
           (unsigned long long)h1, (unsigned long long)h2);

    std::vector<uint8_t> inv = canvas;
    for (auto& v : inv) v = (uint8_t)(255 - v);
    std::vector<uint8_t> inv_buf;
    if (use_nchw) {
      std::vector<uint8_t> nchw;
      nhwc_to_nchw_u8(inv.data(), kModelSize, nchw.data());
      fill_input(nchw.data(), kModelSize, in_attr.type, in_attr.zp, in_attr.scale, &inv_buf);
    } else {
      fill_input(inv.data(), kModelSize, in_attr.type, in_attr.zp, in_attr.scale, &inv_buf);
    }
    std::vector<uint8_t> saved = in_buf;
    in_buf = inv_buf;
    std::vector<uint8_t> raw3;
    run_one(&raw3, nullptr, nullptr, nullptr);
    uint64_t h3 = fnv1a(raw3.data(), raw3.size());
    in_buf = saved;
    printf("[selftest] inverted input changes output: %s (%016llx)\n", h1 != h3 ? "PASS" : "FAIL",
           (unsigned long long)h3);
  }

  // --- timed loop ----------------------------------------------------------
  printf("\n--- timing: warmup %d, loops %d ---\n", warmup, warmup > 0 ? loops : loops);
  for (int i = 0; i < warmup; i++) {
    double ti, tr, tg;
    if (!run_one(nullptr, &ti, &tr, &tg)) { rknn_destroy(ctx); return 2; }
  }

  double sum_in = 0, sum_run = 0, sum_get = 0;
  double worst_run = 0;
  std::vector<double> runs;
  for (int i = 0; i < loops; i++) {
    double ti = 0, tr = 0, tg = 0;
    if (!run_one(nullptr, &ti, &tr, &tg)) { rknn_destroy(ctx); return 2; }
    sum_in += ti; sum_run += tr; sum_get += tg;
    worst_run = std::max(worst_run, tr);
    runs.push_back(tr);
  }
  std::sort(runs.begin(), runs.end());
  double med_run = runs[runs.size() / 2];
  double p95_run = runs[(size_t)((double)runs.size() * 0.95)];

  double avg_in = sum_in / loops, avg_run = sum_run / loops, avg_get = sum_get / loops;
  printf("    inputs_set   avg %8.3f ms\n", avg_in);
  printf("    rknn_run     avg %8.3f ms  (median %8.3f, p95 %8.3f, worst %8.3f)  <== NPU time\n", avg_run,
         med_run, p95_run, worst_run);
  printf("    outputs_get  avg %8.3f ms\n", avg_get);
  printf("    NPU total    avg %8.3f ms  -> %.2f fps\n", avg_in + avg_run + avg_get,
         1000.0 / (avg_in + avg_run + avg_get));
  printf("    rknn_run only        -> %.2f fps\n", 1000.0 / avg_run);
  printf("    host: letterbox %.3f ms, input fill %.3f ms, decode+nms %.3f ms\n", t_lb, t_fill, 0.0);

  // --- outputs -------------------------------------------------------------
  if (det_out) {
    FILE* f = fopen(det_out, "w");
    if (!f) fprintf(stderr, "[WARN] cannot write %s\n", det_out);
    else {
      fprintf(f, "# frame\tcls\tscore\tx1\ty1\tx2\ty2 (canvas px, %d)\n", kModelSize);
      for (const auto& d : dets)
        fprintf(f, "0\t%d\t%.4f\t%.2f\t%.2f\t%.2f\t%.2f\n", d.cls, d.score, d.x1, d.y1, d.x2, d.y2);
      fclose(f);
      printf("[out   ] detections -> %s\n", det_out);
    }
  }

  if (osd_path) {
    // Draw on a copy of the letterboxed canvas.
    std::vector<uint8_t> osd = canvas;
    for (const auto& d : dets) {
      int x1 = (int)d.x1, y1 = (int)d.y1, x2 = (int)d.x2, y2 = (int)d.y2;
      for (int x = std::max(0, x1); x < std::min(kModelSize, x2); x++) {
        for (int y : {y1, y2}) {
          if (y < 0 || y >= kModelSize) continue;
          size_t o = ((size_t)y * kModelSize + x) * 3;
          osd[o] = 0; osd[o + 1] = 255; osd[o + 2] = 0;
        }
      }
      for (int y = std::max(0, y1); y < std::min(kModelSize, y2); y++) {
        for (int x : {x1, x2}) {
          if (x < 0 || x >= kModelSize) continue;
          size_t o = ((size_t)y * kModelSize + x) * 3;
          osd[o] = 0; osd[o + 1] = 255; osd[o + 2] = 0;
        }
      }
      std::string label = std::to_string(d.cls) + " " + std::to_string((int)(d.score * 100));
      draw_text(&osd, kModelSize, kModelSize, std::max(0, x1), std::max(0, y1 - 8), label,
                (const uint8_t[]){0, 255, 0});
    }
    if (stbi_write_jpg(osd_path, kModelSize, kModelSize, 3, osd.data(), 90))
      printf("[out   ] osd -> %s\n", osd_path);
    else
      fprintf(stderr, "[WARN] cannot write %s\n", osd_path);
  }

  rknn_destroy(ctx);
  stbi_image_free(img);
  return 0;
}
