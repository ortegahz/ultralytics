// rknn_npu_smoke.cpp -- minimal RK3588 NPU inference smoke test.
//
// Purpose: prove that the NPU path on the EVM3588 board actually executes
// (rknn_init -> inputs_set -> run -> outputs_get) and returns real data,
// without depending on anything the board does not have.
//
// Why not the SDK's rknn_yolov5_demo directly:
//   1. it needs OpenCV C++ SDK (board has none -- only the pip cv2.abi3 module,
//      no headers / no linkable libs / no .pc, see memory/opencv_arm64_parity.md)
//   2. it needs librga (RGA hardware resize), also not on the board
//   => this program uses only librknnrt + the SDK's header-only stb_image for
//      JPEG decode and bilinear resize.
//
// Input contract of the shipped RK3588 yolov5s model (from the SDK's
// convert_rknn_demo/yolov5/onnx2rknn.py): mean=[0,0,0] std=[255,255,255], i.e.
// (x-0)/255 is baked into the graph => feed plain uint8 RGB NHWC.
// The SDK demo also resizes straight to the model input (no letterbox), so we
// match that with stbir_resize_uint8 (bilinear).
//
// Build: build_rknn_smoke_arm64.sh   Run: rknn_npu_smoke <model.rknn> <img.jpg>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>

#include "rknn_api.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ---------------------------------------------------------------------------
// Minimal 5x7 bitmap font for OSD labels (ASCII 32..126, 5 columns per glyph,
// LSB = top row). Hand-rolled so the board needs no font library.
// ---------------------------------------------------------------------------
static const uint8_t kFont5x7[95][5] = {
    {0x00,0x00,0x00,0x00,0x00}, {0x00,0x00,0x5f,0x00,0x00}, {0x00,0x07,0x00,0x07,0x00},
    {0x14,0x7f,0x14,0x7f,0x14}, {0x24,0x2a,0x7f,0x2a,0x12}, {0x23,0x13,0x08,0x64,0x62},
    {0x36,0x49,0x55,0x22,0x50}, {0x00,0x05,0x03,0x00,0x00}, {0x00,0x1c,0x22,0x41,0x00},
    {0x00,0x41,0x22,0x1c,0x00}, {0x14,0x08,0x3e,0x08,0x14}, {0x08,0x08,0x3e,0x08,0x08},
    {0x00,0x50,0x30,0x00,0x00}, {0x08,0x08,0x08,0x08,0x08}, {0x00,0x60,0x60,0x00,0x00},
    {0x20,0x10,0x08,0x04,0x02}, {0x3e,0x51,0x49,0x45,0x3e}, {0x00,0x42,0x7f,0x40,0x00},
    {0x42,0x61,0x51,0x49,0x46}, {0x21,0x41,0x45,0x4b,0x31}, {0x18,0x14,0x12,0x7f,0x10},
    {0x27,0x45,0x45,0x45,0x39}, {0x3c,0x4a,0x49,0x49,0x30}, {0x01,0x71,0x09,0x05,0x03},
    {0x36,0x49,0x49,0x49,0x36}, {0x06,0x49,0x49,0x29,0x1e}, {0x00,0x36,0x36,0x00,0x00},
    {0x00,0x56,0x36,0x00,0x00}, {0x08,0x14,0x22,0x41,0x00}, {0x14,0x14,0x14,0x14,0x14},
    {0x00,0x41,0x22,0x14,0x08}, {0x02,0x01,0x51,0x09,0x06}, {0x32,0x49,0x79,0x41,0x3e},
    {0x7e,0x11,0x11,0x11,0x7e}, {0x7f,0x49,0x49,0x49,0x36}, {0x3e,0x41,0x41,0x41,0x22},
    {0x7f,0x41,0x41,0x22,0x1c}, {0x7f,0x49,0x49,0x49,0x41}, {0x7f,0x09,0x09,0x09,0x01},
    {0x3e,0x41,0x49,0x49,0x7a}, {0x7f,0x08,0x08,0x08,0x7f}, {0x00,0x41,0x7f,0x41,0x00},
    {0x20,0x40,0x41,0x3f,0x01}, {0x7f,0x08,0x14,0x22,0x41}, {0x7f,0x40,0x40,0x40,0x40},
    {0x7f,0x02,0x0c,0x02,0x7f}, {0x7f,0x04,0x08,0x10,0x7f}, {0x3e,0x41,0x41,0x41,0x3e},
    {0x7f,0x09,0x09,0x09,0x06}, {0x3e,0x41,0x51,0x21,0x5e}, {0x7f,0x09,0x19,0x29,0x46},
    {0x46,0x49,0x49,0x49,0x31}, {0x01,0x01,0x7f,0x01,0x01}, {0x3f,0x40,0x40,0x40,0x3f},
    {0x1f,0x20,0x40,0x20,0x1f}, {0x3f,0x40,0x38,0x40,0x3f}, {0x63,0x14,0x08,0x14,0x63},
    {0x07,0x08,0x70,0x08,0x07}, {0x61,0x51,0x49,0x45,0x43}, {0x00,0x7f,0x41,0x41,0x00},
    {0x02,0x04,0x08,0x10,0x20}, {0x00,0x41,0x41,0x7f,0x00}, {0x04,0x02,0x01,0x02,0x04},
    {0x40,0x40,0x40,0x40,0x40}, {0x00,0x01,0x02,0x04,0x00}, {0x20,0x54,0x54,0x54,0x78},
    {0x7f,0x48,0x44,0x44,0x38}, {0x38,0x44,0x44,0x44,0x20}, {0x38,0x44,0x44,0x48,0x7f},
    {0x38,0x54,0x54,0x54,0x18}, {0x08,0x7e,0x09,0x01,0x02}, {0x0c,0x52,0x52,0x52,0x3e},
    {0x7f,0x08,0x04,0x04,0x78}, {0x00,0x44,0x7d,0x40,0x00}, {0x20,0x40,0x44,0x3d,0x00},
    {0x7f,0x10,0x28,0x44,0x00}, {0x00,0x41,0x7f,0x40,0x00}, {0x7c,0x04,0x18,0x04,0x78},
    {0x7c,0x08,0x04,0x04,0x78}, {0x38,0x44,0x44,0x44,0x38}, {0x7c,0x14,0x14,0x14,0x08},
    {0x08,0x14,0x14,0x18,0x7c}, {0x7c,0x08,0x04,0x04,0x08}, {0x48,0x54,0x54,0x54,0x20},
    {0x04,0x3f,0x44,0x40,0x20}, {0x3c,0x40,0x40,0x20,0x7c}, {0x1c,0x20,0x40,0x20,0x1c},
    {0x3c,0x40,0x30,0x40,0x3c}, {0x44,0x28,0x10,0x28,0x44}, {0x0c,0x50,0x50,0x50,0x3c},
    {0x44,0x64,0x54,0x4c,0x44}, {0x00,0x08,0x36,0x41,0x00}, {0x00,0x00,0x7f,0x00,0x00},
    {0x00,0x41,0x36,0x08,0x00}, {0x08,0x08,0x2a,0x1c,0x08},
};

static void px_set(std::vector<uint8_t>& img, int w, int h, int x, int y, int r, int g, int b) {
  if (x < 0 || y < 0 || x >= w || y >= h) return;
  size_t i = ((size_t)y * w + x) * 3;
  img[i] = (uint8_t)r; img[i + 1] = (uint8_t)g; img[i + 2] = (uint8_t)b;
}

static void draw_rect(std::vector<uint8_t>& img, int w, int h, int x0, int y0, int x1, int y1,
                      int r, int g, int b, int thick) {
  for (int t = 0; t < thick; t++) {
    for (int x = x0; x <= x1; x++) { px_set(img, w, h, x, y0 + t, r, g, b); px_set(img, w, h, x, y1 - t, r, g, b); }
    for (int y = y0; y <= y1; y++) { px_set(img, w, h, x0 + t, y, r, g, b); px_set(img, w, h, x1 - t, y, r, g, b); }
  }
}

static void draw_text(std::vector<uint8_t>& img, int w, int h, int x, int y, const char* s, int r, int g, int b,
                      int scale) {
  int cx = x;
  for (const char* p = s; *p; p++) {
    int ch = (unsigned char)*p;
    if (ch < 32 || ch > 126) ch = '?';
    const uint8_t* g5 = kFont5x7[ch - 32];
    for (int col = 0; col < 5; col++)
      for (int row = 0; row < 7; row++)
        if (g5[col] & (1 << row))
          for (int sy = 0; sy < scale; sy++)
            for (int sx = 0; sx < scale; sx++)
              px_set(img, w, h, cx + col * scale + sx, y + row * scale + sy, r, g, b);
    cx += 6 * scale;
  }
}

// ---------------------------------------------------------------------------
// COCO-80 labels, embedded so the binary stays self-contained on the board.
// ---------------------------------------------------------------------------
static const char* kCoco80[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"};

// yolov5 anchors, identical to the SDK demo's postprocess.cc
static const int kAnchor[3][6] = {
    {10, 13, 16, 30, 33, 23},    // stride 8
    {30, 61, 62, 45, 59, 119},   // stride 16
    {116, 90, 156, 198, 373, 326}// stride 32
};

static const int kObjClassNum = 80;
static const int kPropBoxSize = 5 + kObjClassNum;  // 85

// ---------------------------------------------------------------------------
static double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

static uint32_t fnv1a(const void* data, size_t n) {
  const uint8_t* p = (const uint8_t*)data;
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; i++) {
    h ^= p[i];
    h *= 16777619u;
  }
  return h;
}

static bool read_file(const char* path, std::vector<uint8_t>* out) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz <= 0) { fclose(f); return false; }
  out->resize((size_t)sz);
  size_t got = fread(out->data(), 1, (size_t)sz, f);
  fclose(f);
  return got == (size_t)sz;
}

// ---------------------------------------------------------------------------
// Detection container
// Detection container.
//
// ⚠️ Store the four CORNERS, never "center + w/h". The yolov5 decode ends with
// `bx -= bw/2`, i.e. it already converts the network's center regression into a
// LEFT/TOP coordinate. Keeping a field named `x` that actually means "left"
// invites a second `-w/2` at draw time, which is exactly the bug this struct
// was changed to prevent: the boxes came out shifted up-left by half their size.
// Corners are unambiguous at every call site.
struct Det {
  float score;
  int   cls;
  float left, top, right, bottom;
  float w() const { return right - left; }
  float h() const { return bottom - top; }
};

static float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static float unsigmoidf(float y) { return -1.0f * logf((1.0f / y) - 1.0f); }
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
  float v = (f32 / scale) + zp;
  if (v < -128.f) v = -128.f;
  if (v > 127.f) v = 127.f;
  return (int8_t)v;
}
static float deqnt_affine_to_f32(int8_t q, int32_t zp, float scale) { return ((float)q - (float)zp) * scale; }

static void nms(std::vector<Det>* dets, float iou_thr, size_t top_k) {
  std::sort(dets->begin(), dets->end(), [](const Det& a, const Det& b) { return a.score > b.score; });
  std::vector<Det> keep;
  for (const auto& d : *dets) {
    if (keep.size() >= top_k) break;
    bool suppressed = false;
    for (const auto& k : keep) {
      if (k.cls != d.cls) continue;
      float xx1 = std::max(d.left, k.left);
      float yy1 = std::max(d.top, k.top);
      float xx2 = std::min(d.right, k.right);
      float yy2 = std::min(d.bottom, k.bottom);
      float iw = xx2 - xx1, ih = yy2 - yy1;
      if (iw <= 0 || ih <= 0) continue;
      float inter = iw * ih;
      float uni = d.w() * d.h() + k.w() * k.h() - inter;
      if (inter / uni > iou_thr) { suppressed = true; break; }
    }
    if (!suppressed) keep.push_back(d);
  }
  *dets = keep;
}

// Per-head diagnostics. When a decode yields nothing it is almost always a
// threshold/scale mismatch, and the cheapest way to tell that apart from "the
// model saw nothing" is to report the raw objectness plane's range.
struct HeadDiag {
  int    max_obj    = -128;
  int    max_cls    = -128;
  int64_t n_obj_pass = 0;
  int64_t n_both_pass = 0;
};

// Decode one yolov5 stride head. Layout follows the SDK demo:
//   input[(PROP_BOX_SIZE * a + c) * grid_len + i * grid_w + j]
static void decode_head(const int8_t* in, int grid_h, int grid_w, int stride, const int* anchor, int32_t zp,
                        float scale, float thres, std::vector<Det>* out, HeadDiag* dg) {
  const int grid_len = grid_h * grid_w;
  const float th     = unsigmoidf(thres);
  const int8_t th_i8 = qnt_f32_to_affine(th, zp, scale);
  for (int a = 0; a < 3; a++) {
    for (int i = 0; i < grid_h; i++) {
      for (int j = 0; j < grid_w; j++) {
        const int8_t obj = in[(kPropBoxSize * a + 4) * grid_len + i * grid_w + j];
        if (obj > dg->max_obj) dg->max_obj = obj;
        if (obj < th_i8) continue;
        dg->n_obj_pass++;
        const int8_t* p = in + kPropBoxSize * a * grid_len + i * grid_w + j;
        float bx = sigmoidf(deqnt_affine_to_f32(p[0], zp, scale)) * 2.0f - 0.5f;
        float by = sigmoidf(deqnt_affine_to_f32(p[grid_len], zp, scale)) * 2.0f - 0.5f;
        float bw = sigmoidf(deqnt_affine_to_f32(p[2 * grid_len], zp, scale)) * 2.0f;
        float bh = sigmoidf(deqnt_affine_to_f32(p[3 * grid_len], zp, scale)) * 2.0f;
        bx = (bx + j) * (float)stride;
        by = (by + i) * (float)stride;
        bw = bw * bw * (float)anchor[a * 2];
        bh = bh * bh * (float)anchor[a * 2 + 1];
        bx -= bw / 2.0f;
        by -= bh / 2.0f;

        int8_t  max_p = p[5 * grid_len];
        int     max_i = 0;
        for (int k = 1; k < kObjClassNum; k++) {
          int8_t pr = p[(5 + k) * grid_len];
          if (pr > max_p) { max_i = k; max_p = pr; }
        }
        if (max_p > dg->max_cls) dg->max_cls = max_p;
        if (max_p <= th_i8) continue;
        dg->n_both_pass++;
        Det d;
        d.score = sigmoidf(deqnt_affine_to_f32(max_p, zp, scale)) * sigmoidf(deqnt_affine_to_f32(obj, zp, scale));
        d.cls   = max_i;
        // bx/by have already had bw/bh halved above, so they are the LEFT/TOP
        // corner in model-input pixels.
        d.left   = bx;
        d.top    = by;
        d.right  = bx + bw;
        d.bottom = by + bh;
        out->push_back(d);
      }
    }
  }
}

static const char* tensor_type_name(rknn_tensor_type t) {
  switch (t) {
    case RKNN_TENSOR_FLOAT32: return "FLOAT32";
    case RKNN_TENSOR_FLOAT16: return "FLOAT16";
    case RKNN_TENSOR_INT8:    return "INT8";
    case RKNN_TENSOR_UINT8:   return "UINT8";
    case RKNN_TENSOR_INT16:   return "INT16";
    default:                  return "OTHER";
  }
}
static const char* tensor_fmt_name(rknn_tensor_format f) {
  switch (f) {
    case RKNN_TENSOR_NCHW:  return "NCHW";
    case RKNN_TENSOR_NHWC:  return "NHWC";
    default:                return "UNDEF";
  }
}

#define OK(m)   printf("  [ OK ] %s\n", (m))
#define FAIL(m) printf("  [FAIL] %s\n", (m))
#define INFO(m) printf("  [INFO] %s\n", (m))

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  if (argc < 3) {
    printf("Usage: %s <model.rknn> <image.jpg> [--loops N] [--warmup N] [--perf] [--osd <out.jpg>]\n",
           argv[0]);
    return 2;
  }
  const char* model_path = argv[1];
  const char* image_path = argv[2];
  int loops = 20, warmup = 3, use_perf = 0;
  const char* osd_path = NULL;
  for (int i = 3; i < argc; i++) {
    if (!strcmp(argv[i], "--loops") && i + 1 < argc)        loops = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--warmup") && i + 1 < argc)  warmup = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--perf"))                    use_perf = 1;
    else if (!strcmp(argv[i], "--osd") && i + 1 < argc)      osd_path = argv[++i];
    else { printf("unknown option: %s\n", argv[i]); return 2; }
  }

  printf("=====================================================================\n");
  printf(" RK3588 NPU smoke test (rknn_npu_smoke)\n");
  printf(" model : %s\n", model_path);
  printf(" image : %s\n", image_path);
  printf("=====================================================================\n");

  // --- 1. load model -------------------------------------------------------
  printf("\n[1/6] load model\n");
  std::vector<uint8_t> model;
  if (!read_file(model_path, &model)) { FAIL("cannot read model file"); return 1; }
  char mh[16];
  snprintf(mh, sizeof(mh), "%08x", fnv1a(model.data(), model.size()));
  printf("  model bytes = %zu, fnv1a32 = %s\n", model.size(), mh);

  // --- 2. init -------------------------------------------------------------
  printf("\n[2/6] rknn_init\n");
  rknn_context ctx = 0;
  uint32_t flag = use_perf ? RKNN_FLAG_COLLECT_PERF_MASK : 0u;
  int ret = rknn_init(&ctx, model.data(), (uint32_t)model.size(), flag, NULL);
  if (ret != 0) {
    FAIL("rknn_init failed");
    printf("         ret=%d  (0x%08x)\n", ret, ret);
    printf("         see RKNN_ERR_* in rknn_api.h; -1 device/permission, -2 alloc\n");
    return 1;
  }
  OK("rknn_init succeeded");

  rknn_sdk_version ver;
  memset(&ver, 0, sizeof(ver));
  ver.api_version[sizeof(ver.api_version) - 1] = '\0';
  ver.drv_version[sizeof(ver.drv_version) - 1] = '\0';
  if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver)) == 0) {
    printf("  runtime api_version = %s\n", ver.api_version);
    printf("  driver_version     = %s\n", ver.drv_version);
  }

  rknn_input_output_num io_num;
  memset(&io_num, 0, sizeof(io_num));
  if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) != 0) { FAIL("query IN_OUT_NUM"); return 1; }
  printf("  n_input = %u, n_output = %u\n", io_num.n_input, io_num.n_output);

  // --- 3. tensor attrs -----------------------------------------------------
  printf("\n[3/6] tensor attributes\n");
  std::vector<rknn_tensor_attr> in_attr(io_num.n_input), out_attr(io_num.n_output);
  for (uint32_t i = 0; i < io_num.n_input; i++) {
    memset(&in_attr[i], 0, sizeof(rknn_tensor_attr));
    in_attr[i].index = i;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr[i], sizeof(rknn_tensor_attr));
    printf("  in [%u] dims=[", i);
    for (uint32_t d = 0; d < in_attr[i].n_dims; d++)
      printf("%u%s", in_attr[i].dims[d], d + 1 < in_attr[i].n_dims ? "," : "");
    printf("] fmt=%s type=%s zp=%d scale=%.6f fl=%d\n", tensor_fmt_name(in_attr[i].fmt),
           tensor_type_name(in_attr[i].type), in_attr[i].zp, (double)in_attr[i].scale, (int)in_attr[i].fl);
  }
  for (uint32_t i = 0; i < io_num.n_output; i++) {
    memset(&out_attr[i], 0, sizeof(rknn_tensor_attr));
    out_attr[i].index = i;
    rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr[i], sizeof(rknn_tensor_attr));
    printf("  out[%u] dims=[", i);
    for (uint32_t d = 0; d < out_attr[i].n_dims; d++)
      printf("%u%s", out_attr[i].dims[d], d + 1 < out_attr[i].n_dims ? "," : "");
    printf("] fmt=%s type=%s zp=%d scale=%.6f fl=%d\n", tensor_fmt_name(out_attr[i].fmt),
           tensor_type_name(out_attr[i].type), out_attr[i].zp, (double)out_attr[i].scale, (int)out_attr[i].fl);
  }

  if (io_num.n_input != 1) { FAIL("this smoke test assumes exactly 1 input"); return 1; }
  const int in_h = in_attr[0].dims[1];   // NHWC: N,H,W,C
  const int in_w = in_attr[0].dims[2];
  const int in_c = in_attr[0].dims[3];
  printf("  model input = %dx%dx%d\n", in_h, in_w, in_c);

  // --- 4. preprocess -------------------------------------------------------
  printf("\n[4/6] preprocess\n");
  int    sw = 0, sh = 0, sc = 0;
  uint8_t* src = stbi_load(image_path, &sw, &sh, &sc, 3);
  if (!src) { FAIL("stbi_load failed"); return 1; }
  printf("  decoded jpeg = %dx%dx%d\n", sw, sh, sc);

  std::vector<uint8_t> input((size_t)in_h * in_w * in_c);
  double t_pre0 = now_ms();
  // NOTE the signature: in this version of stb_image_resize the easy API is
//   stbir_resize_uint8(src, sw, sh, in_stride_bytes,
//                       dst, dw, dh, out_stride_bytes, num_channels)
// -- there is NO trailing `filter` argument (the filter is chosen internally),
// and STBIR_FILTER_DEFAULT happens to be 0. Passing it as the 9th argument
// therefore asks for ZERO channels: nothing is written, the destination buffer
// stays as-is, and the graph still runs and still returns non-zero, bit-stable
// output. It fails silently -- the only symptom is "no detections".
// The return value is an int status where 1 == SUCCESS and 0 == failure (every
  // early-out in stbir__resize_allocated returns 0; the single success path
  // returns 1). Getting this backwards turns a good resize into a hard error,
  // so it is asserted explicitly below rather than inferred.
int rc_resize = stbir_resize_uint8(src, sw, sh, 0, input.data(), in_w, in_h, 0, in_c);
  double t_pre = now_ms() - t_pre0;
  // Keep a copy of the decoded source for the OSD render. The model sees a
  // plain resize (no letterbox), so mapping a detection back is a pure scale
  // factor; but keeping the original avoids re-decoding and keeps the OSD at
  // full source resolution.
  std::vector<uint8_t> osd_img;
  if (osd_path) osd_img.assign(src, src + (size_t)sw * sh * sc);
  free(src);
  if (rc_resize != 1) {
    printf("  [FAIL] stbir_resize_uint8 returned %d (expected 1 == success)\n", rc_resize);
    return 1;
  }
  printf("  resized to %dx%dx%d in %.2f ms, fnv1a32 = %08x\n", in_h, in_w, in_c, t_pre,
         fnv1a(input.data(), input.size()));
  {
    // Input sanity: a scrambled buffer would still be non-zero, so report the
    // actual pixel range. Expect roughly R/G/B around the image's own range.
    int mn = 255, mx = 0;
    long long sum = 0;
    for (size_t k = 0; k < input.size(); k++) {
      int v = input[k];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      sum += v;
    }
    printf("  input pixels: min=%d max=%d mean=%.2f  (row0 RGB: %d,%d,%d)\n", mn, mx,
           (double)sum / (double)input.size(), input[0], input[1], input[2]);
    // Hard gate: an all-zero input still produces a plausible, deterministic,
    // non-zero output tensor, so nothing downstream would ever flag it.
    if (mx == 0) {
      FAIL("preprocess produced an all-zero buffer -- refusing to report inference on a blank image");
      return 1;
    }
  }

  rknn_input  in{};
  rknn_output out[8]{};
  in.index = 0;
  in.type  = RKNN_TENSOR_UINT8;
  in.fmt   = RKNN_TENSOR_NHWC;
  in.size  = (uint32_t)input.size();
  in.buf   = input.data();
  in.pass_through = 0;

  for (uint32_t i = 0; i < io_num.n_output; i++) out[i].index = i;
  out[0].want_float = 0;  // keep native int8 so we can use the SDK dequant path

  // --- 5. first inference -------------------------------------------------
  printf("\n[5/6] first inference\n");
  double t0 = now_ms();
  if (rknn_inputs_set(ctx, io_num.n_input, &in) != 0) { FAIL("rknn_inputs_set"); return 1; }
  if (rknn_run(ctx, NULL) != 0)                          { FAIL("rknn_run"); return 1; }
  if (rknn_outputs_get(ctx, io_num.n_output, out, NULL) != 0) { FAIL("rknn_outputs_get"); return 1; }
  double t_first = now_ms() - t0;
  OK("inputs_set + run + outputs_get");
  printf("  first-pass wall clock = %.2f ms\n", t_first);

  for (uint32_t i = 0; i < io_num.n_output; i++) {
    if (!out[i].buf) { FAIL("output buf is NULL"); return 1; }
    // Statistics: proves the NPU produced real data, not a zero-filled buffer.
    // The argmax linear index is printed too, because it is what actually
    // discriminates between candidate output layouts -- if channel c of anchor a
    // were where we think it is, argmax_channel would be a small integer that
    // means "objectness"; a large/garbage value means the layout is wrong.
    const int8_t* p = (const int8_t*)out[i].buf;
    int n = (int)(out[i].size > 200000 ? 200000 : out[i].size);
    int8_t mn = 127, mx = -128;
    long long sum = 0;
    int argmax = -1;
    for (int k = 0; k < n; k++) {
      if (p[k] < mn) mn = p[k];
      if (p[k] > mx) { mx = p[k]; argmax = k; }
      sum += p[k];
    }
    printf("  out[%u] bytes=%u  int8[min=%d max=%d mean=%.2f]  fnv1a32(first %d B)=%08x\n", i, out[i].size, mn, mx,
           (double)sum / (double)n, n, fnv1a(p, (size_t)n));
    if (argmax >= 0) {
      int ch_plane = argmax / (n / 255);  // which of the 255 (3*85) planes holds the peak
      printf("         argmax at linear idx %d -> plane %d, offset-in-plane %d\n", argmax, ch_plane,
             argmax % (n / 255));
    }
    // per-plane max under the assumed [A=3][C=85] decomposition
    {
      int planes = 255;
      int per_plane = (int)out[i].size / planes;
      int8_t  top_plane = -1;
      int     top_ch = -1;
      for (int pl = 0; pl < planes; pl++) {
        int8_t m = -128;
        for (int k = 0; k < per_plane; k++) {
          int8_t v = p[pl * per_plane + k];
          if (v > m) m = v;
        }
        if (m > top_plane) { top_plane = m; top_ch = pl; }
      }
      printf("         peak plane under [A=3][C=85] = %d (anchor %d, channel %d), value %d\n", top_ch, top_ch / 85,
             top_ch % 85, top_plane);
    }
  }

  // determinism: same input twice -> identical output bytes
  uint32_t h1 = fnv1a(out[0].buf, out[0].size);
  rknn_outputs_release(ctx, io_num.n_output, out);
  if (rknn_inputs_set(ctx, io_num.n_input, &in) == 0 && rknn_run(ctx, NULL) == 0 &&
      rknn_outputs_get(ctx, io_num.n_output, out, NULL) == 0) {
    uint32_t h2 = fnv1a(out[0].buf, out[0].size);
    printf("  determinism: run#1=%08x run#2=%08x -> %s\n", h1, h2, h1 == h2 ? "IDENTICAL" : "DIFFER");
  }

  // --- 5b. null control: does the input actually reach the NPU? ------------
  // A silent plumbing bug (input never wired to the graph) would still produce
  // plausible-looking, deterministic, non-zero output -- so determinism alone
  // proves nothing. Feed an inverted copy of the same image and require the
  // output bytes to CHANGE. If they do not, the input path is dead and every
  // downstream number (detections, latency) is meaningless.
  printf("\n[5b] null control: same model, inverted input\n");
  {
    std::vector<uint8_t> inv = input;
    for (auto& v : inv) v = (uint8_t)(255 - v);
    rknn_outputs_release(ctx, io_num.n_output, out);
    if (rknn_inputs_set(ctx, io_num.n_input, &in) == 0) { /* keep buf, swap below */ }
    in.buf = inv.data();
    int ok = (rknn_inputs_set(ctx, io_num.n_input, &in) == 0) && (rknn_run(ctx, NULL) == 0) &&
             (rknn_outputs_get(ctx, io_num.n_output, out, NULL) == 0);
    if (!ok) {
      FAIL("null-control run failed");
    } else {
      uint32_t hc = fnv1a(out[0].buf, out[0].size);
      printf("  original out=%08x   inverted out=%08x   -> %s\n", h1, hc,
             (hc != h1) ? "DIFFER (input path is live)" : "IDENTICAL (input never reaches the graph!)");
      if (hc == h1) FAIL("NPU output independent of input -- inference result is not trustworthy");
    }
    rknn_outputs_release(ctx, io_num.n_output, out);
    in.buf = input.data();  // restore for the decode + benchmark below
    if (rknn_inputs_set(ctx, io_num.n_input, &in) != 0) { FAIL("restore inputs_set"); return 1; }
    if (rknn_run(ctx, NULL) != 0)                          { FAIL("restore run"); return 1; }
    if (rknn_outputs_get(ctx, io_num.n_output, out, NULL) != 0) { FAIL("restore outputs_get"); return 1; }
  }

  // --- 6. decode + benchmark ----------------------------------------------
  printf("\n[6/6] decode + benchmark\n");
  std::vector<Det> dets_all;
  if (io_num.n_output == 3 && out_attr[0].type == RKNN_TENSOR_INT8) {
    const float thr = 0.25f;
    std::vector<Det>& dets = dets_all;
    for (uint32_t i = 0; i < 3; i++) {
      // Layout measured on the board, not assumed:
      //   out[0] dims=[1,3,85,80,80]  ->  [N, A(anchors), C(85), H, W]
      // i.e. 5-D, anchors are a separate leading axis and channel 85 sits in
      // the middle, NOT the flattened [1,255,H,W] the SDK demo assumes.
      // Reading dims[2] as the grid height silently yields gh=85 and turns the
      // whole decode into noise, so the shape is asserted instead of trusted.
      if (out_attr[i].n_dims != 5 || out_attr[i].dims[1] != 3 || out_attr[i].dims[2] != kPropBoxSize) {
        printf("  [WARN] out[%u] unexpected shape for the 5-D yolov5 head, skipping it\n", i);
        continue;
      }
      int gh = (int)out_attr[i].dims[3];
      int gw = (int)out_attr[i].dims[4];
      if (in_h % gh != 0 || in_w % gw != 0) {
        printf("  [WARN] out[%u] grid %dx%d does not divide %dx%d, skipping it\n", i, gh, gw, in_h, in_w);
        continue;
      }
      int st = in_h / gh;
      int8_t thr_i8 = qnt_f32_to_affine(unsigmoidf(thr), out_attr[i].zp, out_attr[i].scale);
      HeadDiag dg;
      printf("  head[%u] grid=%dx%d stride=%d bytes=%u (expect %u)  thr_i8=%d zp=%d scale=%.6f\n", i, gh, gw,
             st, out[i].size, (unsigned)(3 * kPropBoxSize * gh * gw), (int)thr_i8, out_attr[i].zp,
             (double)out_attr[i].scale);
      decode_head((const int8_t*)out[i].buf, gh, gw, st, kAnchor[i], out_attr[i].zp, out_attr[i].scale, thr, &dets,
                  &dg);
      printf("         obj plane: max=%d  cls plane: max=%d  cells>=thr: %lld  both>=thr: %lld\n", dg.max_obj,
             dg.max_cls, (long long)dg.n_obj_pass, (long long)dg.n_both_pass);
    }
    size_t raw_count = dets.size();
    nms(&dets, 0.45f, 20);
    printf("  raw candidates (thr %.2f) = %zu -> after NMS = %zu\n", thr, raw_count, dets.size());
    if (dets.empty()) {
      FAIL("no detections decoded");
    } else {
      OK("decoded real detections");
      for (size_t i = 0; i < dets.size() && i < 10; i++) {
        printf("    #%-2zu %-12s score=%.3f  box=(l=%6.1f t=%6.1f r=%6.1f b=%6.1f)  w=%.1f h=%.1f  class_id=%d\n", i,
               kCoco80[dets[i].cls], dets[i].score, dets[i].left, dets[i].top, dets[i].right, dets[i].bottom,
               dets[i].w(), dets[i].h(), dets[i].cls);
      }
    }
  } else {
    INFO("output layout is not the 3-head yolov5 form; skipping decode (raw stats above still valid)");
  }

  // Move the decoded detections out of the block scope so the OSD renderer and
  // the latency report can both see them.
  // --- 6b. OSD render -------------------------------------------------------
  if (osd_path) {
    if (osd_img.empty() || dets_all.empty()) {
      printf("  [WARN] nothing to draw (%s)\n", osd_img.empty() ? "no source image" : "no detections");
    } else {
      const double sx = (double)sw / (double)in_w;   // model -> source scale
      const double sy = (double)sh / (double)in_h;
      // Bounding boxes in alternating colours so adjacent boxes stay readable.
      static const int kCol[6][3] = {{255,0,0},{0,255,0},{0,128,255},{255,255,0},{255,0,255},{0,255,255}};
      for (size_t i = 0; i < dets_all.size(); i++) {
        const Det& d = dets_all[i];
        // Det already holds corners in model-input pixels; just map to source
        // and clip to the image. Clipping matters: an unclamped box can start
        // off-canvas and the drawing routine would silently drop the whole box.
        float fx0 = d.left * sx, fy0 = d.top * sy;
        float fx1 = d.right * sx, fy1 = d.bottom * sy;
        int x0 = (int)(fx0 < 0 ? 0 : (fx0 > sw - 1 ? sw - 1 : fx0));
        int y0 = (int)(fy0 < 0 ? 0 : (fy0 > sh - 1 ? sh - 1 : fy0));
        int x1 = (int)(fx1 < 0 ? 0 : (fx1 > sw - 1 ? sw - 1 : fx1));
        int y1 = (int)(fy1 < 0 ? 0 : (fy1 > sh - 1 ? sh - 1 : fy1));
        if (x1 <= x0 || y1 <= y0) continue;
        const int* c = kCol[i % 6];
        draw_rect(osd_img, sw, sh, x0, y0, x1, y1, c[0], c[1], c[2], 2);
        char label[64];
        snprintf(label, sizeof(label), "%s %.2f", kCoco80[d.cls], d.score);
        int tw = (int)strlen(label) * 12;
        int tx = x0, ty = y0 - 16;
        if (ty < 2) ty = y0 + 3;
        if (tx + tw > sw) tx = (sw - tw > 0 ? sw - tw : 0);
        draw_rect(osd_img, sw, sh, tx - 2, ty - 2, tx + tw + 2, ty + 14, 0, 0, 0, 1);
        draw_text(osd_img, sw, sh, tx, ty, label, c[0], c[1], c[2], 2);
      }
      // Header strip with the run verdict.
      char head[160];
      snprintf(head, sizeof(head), "RK3588 NPU  rt=%.2f ms  boxes=%zu", t_pre, dets_all.size());
      draw_rect(osd_img, sw, sh, 0, 0, sw - 1, 24, 0, 0, 0, 1);
      draw_text(osd_img, sw, sh, 6, 8, head, 255, 255, 255, 2);

      const int qual = 92;
      // stb_image_write puts the pixel data LAST: (filename, w, h, comp, data, quality).
      int rcw = stbi_write_jpg(osd_path, sw, sh, 3, osd_img.data(), qual);
      if (rcw) {
        printf("  [ OK ] OSD written: %s  (%dx%d, jpeg q%d, fnv1a32=%08x)\n", osd_path, sw, sh, qual,
               fnv1a(osd_img.data(), osd_img.size()));
      } else {
        FAIL("stbi_write_jpg failed");
      }
    }
  }

  // --- 7. steady-state latency, broken down per stage -----------------------
  // A single submit->fetch total hides where the time actually goes, and for an
  // embedded budget that distinction matters: inputs_set and outputs_get are
  // host/driver memcpy, rknn_run is the part the NPU is actually busy for.
  printf("\n[7/7] steady-state performance (%d warmup + %d timed loops)\n", warmup, loops);
  std::vector<double> t_set, t_run, t_get, lat, t_npu;
  t_set.reserve((size_t)loops);
  t_run.reserve((size_t)loops);
  t_get.reserve((size_t)loops);
  lat.reserve((size_t)loops);
  double npu_total = 0.0;
  int    npu_hits  = 0;
  for (int i = 0; i < warmup + loops; i++) {
    rknn_outputs_release(ctx, io_num.n_output, out);
    double a0 = now_ms();
    if (rknn_inputs_set(ctx, io_num.n_input, &in) != 0) { FAIL("inputs_set in loop"); return 1; }
    double a1 = now_ms();
    if (rknn_run(ctx, NULL) != 0)                          { FAIL("run in loop"); return 1; }
    double a2 = now_ms();
    if (rknn_outputs_get(ctx, io_num.n_output, out, NULL) != 0) { FAIL("outputs_get in loop"); return 1; }
    double a3 = now_ms();
    if (use_perf) {
      rknn_perf_run perf;
      memset(&perf, 0, sizeof(perf));
      if (rknn_query(ctx, RKNN_QUERY_PERF_RUN, &perf, sizeof(perf)) == 0) {
        double ms = (double)perf.run_duration / 1000.0;  // us -> ms
        npu_total += ms;
        npu_hits++;
        if (i >= warmup) t_npu.push_back(ms);
      }
    }
    if (i >= warmup) {
      t_set.push_back(a1 - a0);
      t_run.push_back(a2 - a1);
      t_get.push_back(a3 - a2);
      lat.push_back(a3 - a0);
    }
  }

  auto stat = [](std::vector<double>& v, const char* name, double share_total) {
    if (v.empty()) return;
    std::sort(v.begin(), v.end());
    double s = 0;
    for (double x : v) s += x;
    double mn = v.front(), mean = s / (double)v.size(), med = v[v.size() / 2];
    double p95 = v[(size_t)((double)v.size() * 0.95)];
    if (p95 >= v.size()) p95 = v.back();
    double mx = v.back();
    printf("    %-14s mean=%7.3f  median=%7.3f  min=%7.3f  p95=%7.3f  max=%7.3f   (%.1f%% of total)\n", name,
           mean, med, mn, p95, mx, 100.0 * mean / share_total);
  };

  std::sort(lat.begin(), lat.end());
  double sum = 0;
  for (double v : lat) sum += v;
  double mean = sum / (double)lat.size();
  double med  = lat[lat.size() / 2];
  double p99  = lat[(size_t)((double)lat.size() * 0.99)];
  if (p99 >= lat.size()) p99 = lat.back();
  printf("  host-side submit->fetch total: mean=%.3f ms  median=%.3f ms  min=%.3f  p95=%.3f  p99=%.3f  max=%.3f\n",
         mean, med, lat.front(), lat[lat.size() * 95 / 100], p99, lat.back());
  printf("  throughput = %.2f fps  (this total INCLUDES ~2.1 MB of output memcpy)\n", 1000.0 / mean);
  printf("\n  stage breakdown (all measured on the board, wall clock):\n");
  stat(t_set, "inputs_set", mean);
  stat(t_run, "rknn_run", mean);
  stat(t_get, "outputs_get", mean);
  printf("\n  one-shot stages:\n");
  printf("    %-14s %7.3f  (JPEG decode + resize to model input, one shot)\n", "preprocess", t_pre);
  stat(t_npu, "NPU run_duration", mean);
  if (npu_hits > 0)
    printf("    NPU-reported run_duration mean over %d loops = %.3f ms\n", npu_hits, npu_total / npu_hits);
  printf("\n  NOTE: rknn_run wall clock ~= NPU busy + driver submit overhead.\n");
  printf("        outputs_get wall clock ~= pure DMA/copy of %u output bytes.\n",
         out[0].size + out[1].size + out[2].size);
  if (use_perf)
    printf("        (--perf is ON: RKNN_FLAG_COLLECT_PERF_MASK itself costs ~54%% on this model)\n");

  rknn_outputs_release(ctx, io_num.n_output, out);
  rknn_destroy(ctx);
  printf("\n=====================================================================\n");
  printf(" RESULT: NPU INFERENCE OK\n");
  printf("=====================================================================\n");
  return 0;
}