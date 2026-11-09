// ===========================================================================
// rknn_bench_generic.cpp -- model-agnostic RK3588 NPU frame-rate bench.
//
// WHY THIS EXISTS
//   The SDK's rknn_yolov5_demo prints "loop count = 10 , average run X ms",
//   but that number INCLUDES post_process (PERF_WITH_POST=1) and RGA resize.
//   The SDK's rknn_mobilenet_demo has NO timing loop at all -- it runs the
//   model exactly once and prints Top5. So neither SDK demo gives a clean,
//   like-for-like "pure rknn_run" number for the lightweight model.
//
//   This tool closes that gap with the SAME API sequence both SDK demos use
//   (rknn_init -> rknn_inputs_set -> rknn_run -> rknn_outputs_get ->
//   rknn_outputs_release), with the timing discipline of
//   rknn_yolo26n_smoke.cpp: std::chrono::steady_clock, untimed warmup, and
//   inputs_set / rknn_run / outputs_get timed SEPARATELY so the NPU time can
//   be read out directly instead of being hidden inside a fused number.
//
//   It is deliberately model-agnostic: no YOLO-specific decode, no NMS, no
//   letterbox. It only needs the model's input shape, which it asks the
//   runtime for via rknn_query(RKNN_QUERY_INPUT_ATTR). That makes it usable
//   as the control harness for the SDK's mobilenet_v1 (0.57 GFLOPs @ 224) and
//   for any detector we want to compare against it.
//
// Usage:
//   rknn_bench_generic <model.rknn> [--warmup N] [--loops N]
//                      [--cores auto|0|1|2|0_1|0_1_2] [--label NAME]
//
// Build: ./build_rknn_bench_generic_arm64.sh
// ===========================================================================
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rknn_api.h"

// ---------------------------------------------------------------------------
// Timing helper. std::chrono::steady_clock, because a previous version of the
// project's smoke tool used clock_gettime wrongly and reported 5.4e11 ms.
// ---------------------------------------------------------------------------
static double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static const char* fmt_str(rknn_tensor_format f) {
  switch (f) {
    case RKNN_TENSOR_NCHW: return "NCHW";
    case RKNN_TENSOR_NHWC: return "NHWC";
    default: return "UNDEF";
  }
}
static const char* type_str(rknn_tensor_type t) {
  switch (t) {
    case RKNN_TENSOR_FLOAT32: return "FLOAT32";
    case RKNN_TENSOR_FLOAT16: return "FLOAT16";
    case RKNN_TENSOR_INT8: return "INT8";
    case RKNN_TENSOR_UINT8: return "UINT8";
    case RKNN_TENSOR_INT16: return "INT16";
    default: return "UNDEF";
  }
}

static bool read_file(const char* path, std::vector<uint8_t>* out) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "[FATAL] cannot open %s\n", path);
    return false;
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (n <= 0) {
    fprintf(stderr, "[FATAL] empty model %s\n", path);
    fclose(f);
    return false;
  }
  out->resize((size_t)n);
  size_t got = fread(out->data(), 1, (size_t)n, f);
  fclose(f);
  if (got != (size_t)n) {
    fprintf(stderr, "[FATAL] short read on %s\n", path);
    return false;
  }
  return true;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("Usage: %s <model.rknn> [--warmup N] [--loops N] [--cores auto|0|1|2|0_1|0_1_2] [--label NAME]\n",
           argv[0]);
    return 1;
  }
  const char* model_path = argv[1];
  const char* label      = "unnamed";
  int warmup = 10, loops = 50;
  int core_mask = 0;  // 0 == AUTO (SDK default; do not call rknn_set_core_mask)

  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--warmup") && i + 1 < argc)      warmup = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--loops") && i + 1 < argc)  loops = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--label") && i + 1 < argc)  label = argv[++i];
    else if (!strcmp(argv[i], "--cores") && i + 1 < argc) {
      const char* c = argv[++i];
      if (!strcmp(c, "auto"))          core_mask = 0;
      else if (!strcmp(c, "0"))        core_mask = RKNN_NPU_CORE_0;
      else if (!strcmp(c, "1"))        core_mask = RKNN_NPU_CORE_1;
      else if (!strcmp(c, "2"))        core_mask = RKNN_NPU_CORE_2;
      else if (!strcmp(c, "0_1"))      core_mask = RKNN_NPU_CORE_0_1;
      else if (!strcmp(c, "0_1_2"))    core_mask = RKNN_NPU_CORE_0_1_2;
      else { fprintf(stderr, "[FATAL] bad --cores %s\n", c); return 1; }
    }
  }

  std::vector<uint8_t> model;
  if (!read_file(model_path, &model)) return 2;

  rknn_context ctx = 0;
  int ret = rknn_init(&ctx, model.data(), (uint32_t)model.size(), 0, NULL);
  if (ret != 0) { fprintf(stderr, "[FATAL] rknn_init failed ret=%d\n", ret); return 2; }

  rknn_sdk_version ver;
  memset(&ver, 0, sizeof(ver));
  rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver));
  printf("=== bench: %s ===\n", label);
  printf("model      : %s (%zu bytes)\n", model_path, model.size());
  printf("sdk version: api=%s drv=%s\n", ver.api_version, ver.drv_version);

  if (core_mask != 0) {
    ret = rknn_set_core_mask(ctx, (rknn_core_mask)core_mask);
    printf("core mask  : explicitly set to 0x%x (rknn_set_core_mask ret=%d)\n", core_mask, ret);
  } else {
    printf("core mask  : AUTO (SDK demo default -- rknn_set_core_mask NOT called)\n");
  }

  rknn_input_output_num io;
  memset(&io, 0, sizeof(io));
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
  if (ret != 0) { fprintf(stderr, "[FATAL] rknn_query IN_OUT_NUM ret=%d\n", ret); rknn_destroy(ctx); return 2; }
  printf("io num     : in=%d out=%d\n", io.n_input, io.n_output);

  rknn_tensor_attr in_attr;
  memset(&in_attr, 0, sizeof(in_attr));
  in_attr.index = 0;
  ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
  if (ret != 0) { fprintf(stderr, "[FATAL] rknn_query INPUT_ATTR ret=%d\n", ret); rknn_destroy(ctx); return 2; }
  printf("input attr : name=%s dims=[%d %d %d %d] fmt=%s type=%s n_elems=%u size=%u\n",
         in_attr.name, in_attr.dims[0], in_attr.dims[1], in_attr.dims[2], in_attr.dims[3],
         fmt_str(in_attr.fmt), type_str(in_attr.type), in_attr.n_elems, in_attr.size);

  rknn_tensor_attr out_attr;
  memset(&out_attr, 0, sizeof(out_attr));
  out_attr.index = 0;
  rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
  printf("output attr: name=%s dims=[%d %d %d %d] fmt=%s type=%s\n", out_attr.name,
         out_attr.dims[0], out_attr.dims[1], out_attr.dims[2], out_attr.dims[3],
         fmt_str(out_attr.fmt), type_str(out_attr.type));

  // --- derive the NHWC byte layout of the image input ----------------------
  int w = 0, h = 0, c = 3;
  if (in_attr.fmt == RKNN_TENSOR_NCHW) {
    c = (int)in_attr.dims[1]; h = (int)in_attr.dims[2]; w = (int)in_attr.dims[3];
  } else if (in_attr.fmt == RKNN_TENSOR_NHWC) {
    h = (int)in_attr.dims[1]; w = (int)in_attr.dims[2]; c = (int)in_attr.dims[3];
  } else {
    c = 3; h = (int)in_attr.dims[2]; w = (int)in_attr.dims[3];
  }
  if (w <= 0 || h <= 0 || c <= 0) {
    fprintf(stderr, "[FATAL] cannot derive input shape from attr\n");
    rknn_destroy(ctx);
    return 2;
  }
  printf("input shape: %dx%dx%d (NHWC uint8, %.2f MB)\n", w, h, c,
         (double)w * h * c / (1024.0 * 1024.0));

  // Synthetic image content. Deterministic and non-constant, so no runtime
  // shortcut (e.g. constant folding in the driver) can make the graph cheap.
  std::vector<uint8_t> img((size_t)w * h * c);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      uint8_t* px = &img[((size_t)y * w + x) * c];
      px[0] = (uint8_t)((x * 7 + y * 3) & 0xff);
      px[1] = (uint8_t)((x * 3 + y * 11) & 0xff);
      px[2] = (uint8_t)((x + y) & 0xff);
    }

  rknn_input inputs[1];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].type  = RKNN_TENSOR_UINT8;   // same as both SDK demos
  inputs[0].size  = (uint32_t)img.size();
  inputs[0].fmt   = RKNN_TENSOR_NHWC;    // same as both SDK demos
  inputs[0].buf   = img.data();

  rknn_output outputs[1];
  memset(outputs, 0, sizeof(outputs));
  outputs[0].want_float = 0;

  // --- warmup (untimed) ----------------------------------------------------
  for (int i = 0; i < warmup; i++) {
    if (rknn_inputs_set(ctx, io.n_input, inputs) != 0) { fprintf(stderr, "[FATAL] inputs_set\n"); rknn_destroy(ctx); return 2; }
    if (rknn_run(ctx, NULL) != 0) { fprintf(stderr, "[FATAL] rknn_run\n"); rknn_destroy(ctx); return 2; }
    if (rknn_outputs_get(ctx, 1, outputs, NULL) != 0) { fprintf(stderr, "[FATAL] outputs_get\n"); rknn_destroy(ctx); return 2; }
    rknn_outputs_release(ctx, 1, outputs);
  }

  // --- timed loop ----------------------------------------------------------
  std::vector<double> t_in, t_run, t_get;
  t_in.reserve(loops); t_run.reserve(loops); t_get.reserve(loops);
  for (int i = 0; i < loops; i++) {
    double a = now_ms();
    rknn_inputs_set(ctx, io.n_input, inputs);
    double b = now_ms();
    rknn_run(ctx, NULL);
    double c2 = now_ms();
    rknn_outputs_get(ctx, 1, outputs, NULL);
    double d = now_ms();
    rknn_outputs_release(ctx, 1, outputs);
    t_in.push_back(b - a);
    t_run.push_back(c2 - b);
    t_get.push_back(d - c2);
  }

  auto stat = [](std::vector<double> v, double* avg, double* med, double* p95, double* worst) {
    std::sort(v.begin(), v.end());
    double s = 0;
    for (double x : v) s += x;
    *avg = s / (double)v.size();
    *med = v[v.size() / 2];
    *p95 = v[(size_t)((double)v.size() * 0.95)];
    *worst = v.back();
  };
  double ai, am, ap, aw, ri, rm, rp, rw, gi, gm, gp, gw;
  stat(t_in, &ai, &am, &ap, &aw);
  stat(t_run, &ri, &rm, &rp, &rw);
  stat(t_get, &gi, &gm, &gp, &gw);

  printf("\n--- timing: warmup %d, loops %d ---\n", warmup, loops);
  printf("    inputs_set   avg %8.3f ms  (median %8.3f, p95 %8.3f, worst %8.3f)\n", ai, am, ap, aw);
  printf("    rknn_run     avg %8.3f ms  (median %8.3f, p95 %8.3f, worst %8.3f)  <== NPU time\n", ri, rm, rp, rw);
  printf("    outputs_get  avg %8.3f ms  (median %8.3f, p95 %8.3f, worst %8.3f)\n", gi, gm, gp, gw);
  double total = ai + ri + gi;
  printf("    NPU total    avg %8.3f ms  -> %.2f fps\n", total, 1000.0 / total);
  printf("    rknn_run only        -> %.2f fps\n", 1000.0 / ri);

  // --- runtime's own view of the timing ------------------------------------
  rknn_perf_run perf_run;
  memset(&perf_run, 0, sizeof(perf_run));
  if (rknn_query(ctx, RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run)) == 0) {
    printf("    RKNN_QUERY_PERF_RUN   : %lld us\n", (long long)perf_run.run_duration);
  }
  rknn_perf_detail perf_detail;
  memset(&perf_detail, 0, sizeof(perf_detail));
  if (rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail)) == 0 &&
      perf_detail.perf_data && perf_detail.data_len > 0) {
    printf("\n--- RKNN_QUERY_PERF_DETAIL (runtime text, %llu bytes) ---\n",
           (unsigned long long)perf_detail.data_len);
    std::string s(perf_detail.perf_data, (size_t)perf_detail.data_len);
    size_t pos = 0;
    while (pos < s.size()) {
      size_t nl = s.find('\n', pos);
      if (nl == std::string::npos) nl = s.size();
      printf("%s\n", s.substr(pos, nl - pos).c_str());
      pos = nl + 1;
    }
  }

  rknn_destroy(ctx);
  return 0;
}
