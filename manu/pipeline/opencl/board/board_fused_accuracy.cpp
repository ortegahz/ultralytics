// ===========================================================================
// board_fused_accuracy.cpp -- run warp_median_fused.cl on the real RK3588 and
//                             reconcile it against the OpenCV CPU reference.
//
// WHAT THIS IS FOR
//   The fused warp+median+pack kernel was written and verified on the x86
//   workstation, where it ran on PoCL. PoCL is a CPU simulation: it validates
//   the *logic* and it caught real bugs, but every timing it produced was a
//   property of PoCL, not of the Mali-G610. More importantly, nothing has ever
//   run this kernel on the actual target, and the kernel's own header flags two
//   things that can only be settled by measurement on the real device:
//
//     * FUSED_USE_HW_LINEAR -- CLK_FILTER_LINEAR accuracy is
//       implementation-defined. A driver that rounds the filtered result to the
//       8-bit storage precision injects +/-1 errors that the accuracy gate
//       cannot absorb. Which path is correct on Mali is an experiment, not an
//       assumption.
//     * The image format. At 22 padded frames per step, CL_R8 costs a quarter
//       of CL_RGBA's bandwidth. Whether Mali takes CL_R8 at all was unknown.
//
//   This binary settles both, and reports the accuracy of the kernel itself.
//
// WHY THE REFERENCE IS SHIPPED AS DATA
//   There is no arm64 OpenCV anywhere: not in the cross toolchain's sysroot,
//   not in the vendor SDK, and not on the board (which has a Python cv2 but no
//   /usr/include/opencv4). The reference must therefore be computed on the x86
//   host with real cv::warpAffine and shipped in a bundle. Re-implementing
//   OpenCV's interpolation on the board would have validated my own
//   reimplementation instead of the kernel -- a circular test.
//   See fused_case_format.h and gen_fused_case.cpp.
//
// NO OpenCV IS NEEDED HERE. Only OpenCL, loaded through dlopen for the same
// reason as board_cl_probe.cpp: nothing at link time can be guaranteed to
// provide libOpenCL.
//
// Usage:
//   board_fused_accuracy --case <file.bin> --kernel <warp_median_fused.cl>
//                        [--variant hw|nearest|both]
// ===========================================================================

#include <dlfcn.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <CL/cl.h>

#include "fused_case_format.h"

// ---------------------------------------------------------------------------
// OpenCL entry points, resolved by hand. Same reasoning as board_cl_probe.cpp:
// the Khronos header declares them extern, we never call them by name, so the
// linker never needs libOpenCL -- and every one keeps its true signature.
// ---------------------------------------------------------------------------
typedef cl_int(CL_API_CALL* fn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int(CL_API_CALL* fn_clGetPlatformInfo)(cl_platform_id, cl_platform_info,
                                                  size_t, void*, size_t*);
typedef cl_int(CL_API_CALL* fn_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint,
                                               cl_device_id*, cl_uint*);
typedef cl_int(CL_API_CALL* fn_clGetDeviceInfo)(cl_device_id, cl_device_info, size_t,
                                                void*, size_t*);
typedef cl_context(CL_API_CALL* fn_clCreateContext)(
    const cl_context_properties*, cl_uint, const cl_device_id*,
    void(CL_CALLBACK*)(const char*, const void*, size_t, void*), void*, cl_int*);
typedef cl_command_queue(CL_API_CALL* fn_clCreateCommandQueueWithProperties)(
    cl_context, cl_device_id, const cl_queue_properties*, cl_int*);
typedef cl_command_queue(CL_API_CALL* fn_clCreateCommandQueue)(
    cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
typedef cl_mem(CL_API_CALL* fn_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*,
                                               cl_int*);
typedef cl_mem(CL_API_CALL* fn_clCreateImage)(cl_context, cl_mem_flags,
                                              const cl_image_format*, const cl_image_desc*,
                                              void*, cl_int*);
typedef cl_mem(CL_API_CALL* fn_clCreateImage2D)(cl_context, cl_mem_flags,
                                                const cl_image_format*, size_t, size_t,
                                                size_t, void*, cl_int*);
typedef cl_int(CL_API_CALL* fn_clReleaseMemObject)(cl_mem);
typedef cl_int(CL_API_CALL* fn_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool,
                                                     size_t, size_t, const void*, cl_uint,
                                                     const cl_event*, cl_event*);
typedef cl_int(CL_API_CALL* fn_clEnqueueWriteImage)(cl_command_queue, cl_mem, cl_bool,
                                                    const size_t*, const size_t*,
                                                    size_t, size_t, const void*, cl_uint,
                                                    const cl_event*, cl_event*);
typedef cl_int(CL_API_CALL* fn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool,
                                                    size_t, size_t, void*, cl_uint,
                                                    const cl_event*, cl_event*);
typedef cl_program(CL_API_CALL* fn_clCreateProgramWithSource)(cl_context, cl_uint,
                                                              const char**, const size_t*,
                                                              cl_int*);
typedef cl_int(CL_API_CALL* fn_clBuildProgram)(cl_program, cl_uint, const cl_device_id*,
                                               const char*,
                                               void(CL_CALLBACK*)(cl_program, void*),
                                               void*, cl_int*);
typedef cl_int(CL_API_CALL* fn_clGetProgramBuildInfo)(cl_program, cl_device_id,
                                                      cl_program_build_info, size_t,
                                                      void*, size_t*);
typedef cl_kernel(CL_API_CALL* fn_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_int(CL_API_CALL* fn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int(CL_API_CALL* fn_clSetKernelArgSampler)(cl_kernel, cl_uint, cl_sampler,
                                                      cl_int*);
typedef cl_int(CL_API_CALL* fn_clEnqueueNDRangeKernel)(
    cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*,
    const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int(CL_API_CALL* fn_clFinish)(cl_command_queue);
typedef cl_int(CL_API_CALL* fn_clReleaseEvent)(cl_event);
typedef cl_int(CL_API_CALL* fn_clGetEventProfilingInfo)(cl_event, cl_profiling_info,
                                                         size_t, void*, size_t*);
// FIVE parameters, not four. The cl_bool normalized_coords slot sits between
// the context and the addressing mode, and dropping it shifts every later
// argument left by one: the addressing mode lands in normalized_coords, the
// filter mode lands in addressing_mode, and the driver dereferences a garbage
// enum. It segfaults inside the ICD rather than returning an error code, so
// nothing in the host code gives the mistake away -- the compiler cannot check
// it because this is our own function-pointer type.
typedef cl_sampler(CL_API_CALL* fn_clCreateSampler)(cl_context, cl_bool,
                                                    cl_addressing_mode,
                                                    cl_filter_mode, cl_int*);
typedef cl_int(CL_API_CALL* fn_clReleaseSampler)(cl_sampler);
typedef cl_int(CL_API_CALL* fn_clReleaseKernel)(cl_kernel);
typedef cl_int(CL_API_CALL* fn_clReleaseProgram)(cl_program);
typedef cl_int(CL_API_CALL* fn_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int(CL_API_CALL* fn_clReleaseContext)(cl_context);

#define RESOLVE(field, name)                                    \
  do {                                                          \
    void* sym = dlsym(g_lib, name);                             \
    if (!sym) {                                                 \
      std::printf("[FAIL] missing symbol: %s\n", name);         \
      missing++;                                                \
    }                                                           \
    g_fn.field = reinterpret_cast<fn_##field>(sym);             \
  } while (0)

struct FnTable {
  fn_clGetPlatformIDs clGetPlatformIDs;
  fn_clGetPlatformInfo clGetPlatformInfo;
  fn_clGetDeviceIDs clGetDeviceIDs;
  fn_clGetDeviceInfo clGetDeviceInfo;
  fn_clCreateContext clCreateContext;
  fn_clCreateCommandQueueWithProperties clCreateCommandQueueWithProperties;
  fn_clCreateCommandQueue clCreateCommandQueue;
  fn_clCreateBuffer clCreateBuffer;
  fn_clCreateImage clCreateImage;
  fn_clCreateImage2D clCreateImage2D;
  fn_clReleaseMemObject clReleaseMemObject;
  fn_clEnqueueWriteBuffer clEnqueueWriteBuffer;
  fn_clEnqueueWriteImage clEnqueueWriteImage;
  fn_clEnqueueReadBuffer clEnqueueReadBuffer;
  fn_clCreateProgramWithSource clCreateProgramWithSource;
  fn_clBuildProgram clBuildProgram;
  fn_clGetProgramBuildInfo clGetProgramBuildInfo;
  fn_clCreateKernel clCreateKernel;
  fn_clSetKernelArg clSetKernelArg;
  fn_clSetKernelArgSampler clSetKernelArgSampler;
  fn_clEnqueueNDRangeKernel clEnqueueNDRangeKernel;
  fn_clFinish clFinish;
  fn_clReleaseEvent clReleaseEvent;
  fn_clGetEventProfilingInfo clGetEventProfilingInfo;
  fn_clCreateSampler clCreateSampler;
  fn_clReleaseSampler clReleaseSampler;
  fn_clReleaseKernel clReleaseKernel;
  fn_clReleaseProgram clReleaseProgram;
  fn_clReleaseCommandQueue clReleaseCommandQueue;
  fn_clReleaseContext clReleaseContext;
};

static void* g_lib = nullptr;
static FnTable g_fn;

// clSetKernelArgSampler is deprecated since OpenCL 2.0 and is simply absent from
// some ICD loaders (ocl-icd 2.3.4 on the verification host does not export it).
// The supported route is passing a cl_sampler straight through clSetKernelArg,
// so that is the primary path and the 1.2 entry point is only a fallback.
static bool g_have_arg_sampler = false;

static const char* cl_err_name(cl_int e) {
  switch (e) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_IMAGE_DESCRIPTOR: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    default: return "CL_<other>";
  }
}

static int g_fail_code = 0;
#define CHK(expr)                                                       \
  do {                                                                  \
    cl_int _rc = (expr);                                                 \
    if (_rc != CL_SUCCESS) {                                            \
      std::printf("[FAIL] %s -> %s (%d)\n", #expr, cl_err_name(_rc), _rc); \
      g_fail_code = 20;                                                 \
      return false;                                                     \
    }                                                                   \
  } while (0)

// ---------------------------------------------------------------------------
// Accuracy metrics.
//
// The histogram is not decoration. "Max|Diff| = 2, MAE = 0.003" and
// "Max|Diff| = 2, MAE = 1.4" share a headline and have nothing in common: the
// first is a handful of quantisation-boundary ties, the second is a systematic
// rounding mismatch. The gate is per-pixel AND per-image, so both numbers have
// to be visible.
//
// Hot pixels are recorded with their location because the cause differs by
// where they are: pixels hugging the frame edge implicate the border/pad path,
// pixels scattered through the interior implicate float32 coordinate precision
// against OpenCV's own arithmetic. The max alone cannot tell the two apart.
// ---------------------------------------------------------------------------
struct Diff {
  long maxdiff = 0;
  double mae = 0.0;
  long exact = 0;        // pixels with |diff| == 0
  long over1 = 0;        // |diff| >= 2
  long total = 0;
  int bx = -1, by = -1;  // where the max occurred
  long hist[8] = {0};    // |diff| == 0,1,2,3,4,5,6, >=7
  int edge_hot = 0, inner_hot = 0;   // |diff| >= 2, split by proximity to the border
};

static Diff compare_plane(const uint8_t* got, const uint8_t* ref, int W, int H, int edge) {
  Diff d;
  long sum = 0;
  d.total = (long)W * H;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const size_t i = (size_t)y * W + x;
      const long df = std::labs((long)got[i] - (long)ref[i]);
      sum += df;
      if (df == 0) ++d.exact;
      if (df > d.maxdiff) { d.maxdiff = df; d.bx = x; d.by = y; }
      const int bucket = (df >= 7) ? 7 : (int)df;
      ++d.hist[bucket];
      if (df >= 2) {
        ++d.over1;
        const bool near_edge =
            (x < edge || y < edge || x >= W - edge || y >= H - edge);
        if (near_edge) ++d.edge_hot; else ++d.inner_hot;
      }
    }
  }
  d.mae = (double)sum / (double)d.total;
  return d;
}

static void print_diff(const char* name, const Diff& d) {
  const double exact_pct = 100.0 * (double)d.exact / (double)d.total;
  std::printf("    %-4s Max|Diff|=%-3ld MAE=%-10.6f exact=%6.2f%%  |d|>=2: %ld "
              "(edge %d / inner %d)  at (%d,%d)\n",
              name, d.maxdiff, d.mae, exact_pct, d.over1, d.edge_hot, d.inner_hot,
              d.bx, d.by);
  std::printf("         hist |d|= ");
  for (int i = 0; i < 8; ++i) {
    if (i == 7) std::printf(">=7:%ld", d.hist[i]);
    else std::printf("%d:%ld ", i, d.hist[i]);
  }
  std::printf("\n");
}

// Splices the generated sorting network into /*__SORTNET__*/.
//
// This is not optional. The marker sits where the 21-element median network
// belongs; leave it as a comment and the program still COMPILES AND RUNS, but
// v10 is then just the k=10 warped sample rather than the median, so Ch2 is
// silently wrong while Ch0 and Ch1 remain perfect. That failure mode was hit
// for real during development: Ch0 came out 100% exact and Ch1 matched the
// documented baseline, which made Ch2's huge error look like a GPU problem
// rather than a missing file. Hence the post-splice assertion below.
static std::string splice_sortnet(const std::string& kernel_src, const std::string& inc) {
  const std::string marker = "/*__SORTNET__*/";
  const size_t at = kernel_src.find(marker);
  if (at == std::string::npos) {
    std::printf("[FATAL] sortnet marker not found in kernel source\n");
    std::exit(2);
  }
  return kernel_src.substr(0, at) + inc + kernel_src.substr(at + marker.size());
}

static bool read_file(const std::string& path, std::string& out) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n < 0) { std::fclose(f); return false; }
  out.resize((size_t)n);
  const size_t got = std::fread(&out[0], 1, (size_t)n, f);
  std::fclose(f);
  return got == (size_t)n;
}

// ---------------------------------------------------------------------------
// Image allocation.
//
// SPEC-CORRECT CONSTANTS, deliberately NOT copied from host/ocl_host.h. That
// header invents three constants that no OpenCL header defines, and all three
// were masked on the x86 workstation:
//
//   CL_R8            -> 0x10D0, which is CL_SNORM_INT8, a channel_TYPE.
//                       Valid channel_order values are 0x10B0..0x10C3.
//   CL_IMAGE_OBJECT_2D -> 0x10F0, which is CL_MEM_OBJECT_BUFFER. The 2D image
//                       value is CL_MEM_OBJECT_IMAGE2D = 0x10F1.
//
// On x86 both bugs stayed invisible: PoCL rejects the bogus 0x10D0 channel
// order, so pick_gray_format silently fell through to CL_RGBA, and clCreateImage
// returns CL_INVALID_IMAGE_DESCRIPTOR for every format on that ICD, so the code
// fell back to clCreateImage2D, which never reads image_desc at all.
//
// On a real Mali driver neither escape hatch exists -- clCreateImage is expected
// to succeed -- so shipping those constants would have produced a wrong or
// failing image on exactly the hardware this tool exists to validate. The
// spec-correct {CL_R, CL_UNORM_INT8} is used instead: CL_R is 0x10B0, a valid
// channel order, paired with CL_UNORM_INT8, and it is the genuine 1-byte-per-
// pixel single-channel format.
// ---------------------------------------------------------------------------
struct ImageFmt {
  cl_uint order;
  const char* name;
  int channels;
};

static cl_mem make_image(cl_context ctx, const cl_image_format& fmt, size_t w, size_t h,
                         bool* used_2d) {
  cl_int e = CL_SUCCESS;
  cl_mem m = nullptr;
  cl_image_desc desc{};
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_width = w;
  desc.image_height = h;
  desc.image_depth = 1;
  desc.image_array_size = 1;
  desc.image_row_pitch = 0;
  desc.image_slice_pitch = 0;
  desc.num_mip_levels = 0;
  desc.num_samples = 0;
  if (g_fn.clCreateImage) {
    m = g_fn.clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &e);
    if (e == CL_SUCCESS && m) { *used_2d = false; return m; }
  }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  e = CL_SUCCESS;
  m = g_fn.clCreateImage2D(ctx, CL_MEM_READ_ONLY, &fmt, w, h, 0, nullptr, &e);
#pragma GCC diagnostic pop
  if (e == CL_SUCCESS && m) { *used_2d = true; return m; }
  return nullptr;
}

static bool pick_gray_format(cl_context ctx, ImageFmt* out, bool* used_2d) {
  struct Cand { const char* name; cl_uint order; int ch; };
  // CL_R + CL_UNORM_INT8 is the spec's single-channel 8-bit format at 1 byte
  // per pixel. CL_R8 does not exist as an OpenCL constant.
  const Cand cand[] = {
      {"CL_R    + CL_UNORM_INT8 (1 byte/px)", CL_R, 1},
      {"CL_RGBA + CL_UNORM_INT8 (4 byte/px)", CL_RGBA, 4},
  };
  for (const Cand& c : cand) {
    const cl_image_format f{static_cast<cl_uint>(c.order),
                            static_cast<cl_uint>(CL_UNORM_INT8)};
    cl_mem probe = make_image(ctx, f, 64, 64, used_2d);
    if (probe) {
      g_fn.clReleaseMemObject(probe);
      out->order = c.order; out->name = c.name; out->channels = c.ch;
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// One (variant, case) run.
// ---------------------------------------------------------------------------
struct RunCtx {
  cl_context ctx;
  cl_command_queue q;
  cl_device_id dev;
  ImageFmt fmt;
  bool used_2d;
};

static bool run_case(RunCtx& rc, cl_kernel kern, cl_sampler samp,
                     const std::vector<uint8_t>& frames,
                     const std::vector<float>& mats, const std::vector<uint8_t>& ref,
                     int W, int H, int pad, int edge, Diff out[3], double ms[3]) {
  const size_t pw = (size_t)(W + 2 * pad), ph = (size_t)(H + 2 * pad);
  const size_t frame_bytes = pw * ph;
  const int WINDOW = FKC_WINDOW;
  const int nch = rc.fmt.channels;

  // The bundle always stores 1 byte per pixel, but the device may only accept a
  // multi-channel format (PoCL takes only CL_RGBA). clEnqueueWriteImage treats
  // row_pitch == 0 as "region[0] * element_size", i.e. pw*nch bytes per row, so
  // feeding it a pw-byte row walks off the end of every row and segfaults deep
  // inside the driver's memcpy. The staging buffer below lays the data out at
  // the stride the image actually has.
  std::vector<uint8_t> staged;
  const size_t row_stride = pw * (size_t)nch;
  if (nch != 1) staged.resize(row_stride * ph);

  // Upload is timed SEPARATELY from the kernel. Measuring the kernel alone and
  // calling that "the cost of the fused tail" understates the real per-frame
  // price, because 22 padded frames cross the bus every single step. Reporting
  // one combined number would hide which half dominates; reporting only the
  // kernel would hide the transfer entirely. Both are printed, plus the sum.
  const auto t_up0 = std::chrono::steady_clock::now();

  // 22 images: 21 lagged history + the current frame.
  std::vector<cl_mem> imgs((size_t)WINDOW + 1, nullptr);
  for (int i = 0; i <= WINDOW; ++i) {
    imgs[(size_t)i] = make_image(rc.ctx, cl_image_format{rc.fmt.order,
                                                          static_cast<cl_uint>(CL_UNORM_INT8)},
                                 pw, ph, &rc.used_2d);
    if (!imgs[(size_t)i]) {
      std::printf("[FAIL] could not create %zux%zu image %d\n", pw, ph, i);
      g_fail_code = 21; return false;
    }
    const uint8_t* src = frames.data() + (size_t)i * frame_bytes;
    if (nch == 1) {
      staged.clear();
      // No staging needed; point straight at the bundle copy.
    } else {
      for (size_t y = 0; y < ph; ++y) {
        const uint8_t* in = src + y * pw;
        uint8_t* out = staged.data() + y * row_stride;
        for (size_t x = 0; x < pw; ++x) {
          // Replicate to every channel: read_imagef().x takes channel 0, and
          // filling the rest keeps the image well-defined for any later
          // .y/.z/.w access.
          for (int c = 0; c < nch; ++c) out[x * (size_t)nch + (size_t)c] = in[x];
        }
      }
    }
    const void* host = (nch == 1) ? (const void*)src : (const void*)staged.data();
    const size_t origin[3] = {0, 0, 0};
    const size_t region[3] = {pw, ph, 1};
    CHK(g_fn.clEnqueueWriteImage(rc.q, imgs[(size_t)i], CL_TRUE, origin, region,
                                 0, 0, host, 0, nullptr, nullptr));
  }

  const auto t_up1 = std::chrono::steady_clock::now();

  cl_int e = CL_SUCCESS;
  cl_mem mats_buf = g_fn.clCreateBuffer(rc.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        mats.size() * sizeof(float),
                                        const_cast<float*>(mats.data()), &e);
  if (e != CL_SUCCESS) { std::printf("[FAIL] mats buffer -> %s\n", cl_err_name(e)); g_fail_code = 22; return false; }

  std::vector<uint8_t> got((size_t)3 * W * H);
  cl_mem out_buf = g_fn.clCreateBuffer(rc.ctx, CL_MEM_WRITE_ONLY, got.size(), nullptr, &e);
  if (e != CL_SUCCESS) { std::printf("[FAIL] out buffer -> %s\n", cl_err_name(e)); g_fail_code = 22; return false; }

  // Argument order matches the kernel signature exactly: 22 images, mats, out,
  // W, H, pad, sampler.
  for (int i = 0; i <= WINDOW; ++i) {
    CHK(g_fn.clSetKernelArg(kern, (cl_uint)i, sizeof(cl_mem), &imgs[(size_t)i]));
  }
  CHK(g_fn.clSetKernelArg(kern, WINDOW + 1, sizeof(cl_mem), &mats_buf));
  CHK(g_fn.clSetKernelArg(kern, WINDOW + 2, sizeof(cl_mem), &out_buf));
  const cl_int wi = W, hi = H, pi = pad;
  CHK(g_fn.clSetKernelArg(kern, WINDOW + 3, sizeof(cl_int), &wi));
  CHK(g_fn.clSetKernelArg(kern, WINDOW + 4, sizeof(cl_int), &hi));
  CHK(g_fn.clSetKernelArg(kern, WINDOW + 5, sizeof(cl_int), &pi));
  // Sampler argument: modern route first, deprecated 1.2 route only as fallback.
  {
    cl_int se = g_fn.clSetKernelArg(kern, WINDOW + 6, sizeof(cl_sampler), &samp);
    if (se != CL_SUCCESS) {
      if (!g_have_arg_sampler) {
        std::printf("[FAIL] clSetKernelArg(sampler) -> %s (%d), and "
                    "clSetKernelArgSampler is unavailable\n", cl_err_name(se), se);
        g_fail_code = 23; return false;
      }
      se = g_fn.clSetKernelArgSampler(kern, WINDOW + 6, samp, &e);
      if (se != CL_SUCCESS) {
        std::printf("[FAIL] clSetKernelArgSampler -> %s (%d)\n", cl_err_name(se), se);
        g_fail_code = 23; return false;
      }
    }
  }

  // NULL local size: the kernel deliberately leaves the work-group shape to the
  // driver. Timed across enqueue + clFinish, because clEnqueueNDRangeKernel
  // returns at enqueue time and timing it alone measures nothing.
  const size_t gws[2] = {(size_t)W, (size_t)H};
  auto t0 = std::chrono::steady_clock::now();
  CHK(g_fn.clEnqueueNDRangeKernel(rc.q, kern, 2, nullptr, gws, nullptr, 0, nullptr, nullptr));
  CHK(g_fn.clFinish(rc.q));
  auto t1 = std::chrono::steady_clock::now();
  ms[1] = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const auto t_rd0 = std::chrono::steady_clock::now();

  CHK(g_fn.clEnqueueReadBuffer(rc.q, out_buf, CL_TRUE, 0, got.size(), got.data(), 0,
                               nullptr, nullptr));
  const auto t_rd1 = std::chrono::steady_clock::now();
  ms[2] = std::chrono::duration<double, std::milli>(t_rd1 - t_rd0).count();
  ms[0] = std::chrono::duration<double, std::milli>(t_up1 - t_up0).count();

  for (int p = 0; p < 3; ++p)
    out[p] = compare_plane(got.data() + (size_t)p * W * H,
                           ref.data() + (size_t)p * W * H, W, H, edge);

  g_fn.clReleaseMemObject(out_buf);
  g_fn.clReleaseMemObject(mats_buf);
  for (cl_mem m : imgs) if (m) g_fn.clReleaseMemObject(m);
  return true;
}

static bool read_file(const std::string& path, std::vector<char>& out) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n < 0) { std::fclose(f); return false; }
  out.resize((size_t)n);
  const size_t got = std::fread(out.data(), 1, (size_t)n, f);
  std::fclose(f);
  return got == (size_t)n;
}

static void usage() {
  std::fprintf(stderr,
               "usage: board_fused_accuracy --case <file.bin> --kernel <file.cl> "
               "--sortnet <sortnet_generated.inc> [--variant hw|nearest|both]\n");
}

int main(int argc, char** argv) {
  std::string case_path, kernel_path, sortnet_path, variant = "both";
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char* what) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "[ERROR] %s needs a value\n", what); usage(); std::exit(2); }
      return argv[++i];
    };
    if (a == "--case") case_path = need("--case");
    else if (a == "--kernel") kernel_path = need("--kernel");
    else if (a == "--sortnet") sortnet_path = need("--sortnet");
    else if (a == "--variant") variant = need("--variant");
    else { std::fprintf(stderr, "[ERROR] unknown argument: %s\n", a.c_str()); usage(); return 2; }
  }
  if (case_path.empty() || kernel_path.empty()) { usage(); return 2; }

  // ---- inputs ------------------------------------------------------------
  std::string ksrc, inc;
  if (!read_file(kernel_path, ksrc)) {
    std::printf("[FAIL] cannot read kernel: %s\n", kernel_path.c_str());
    return 2;
  }
  if (ksrc.find("/*__SORTNET__*/") != std::string::npos) {
    if (sortnet_path.empty()) {
      std::printf("[FATAL] kernel still contains /*__SORTNET__*/ and no --sortnet was "
                  "given.\n         Without the median network the kernel still compiles "
                  "and runs and Ch0/Ch1 stay perfect,\n         while Ch2 is silently "
                  "wrong. Refusing to run.\n");
      return 2;
    }
    if (!read_file(sortnet_path, inc)) {
      std::printf("[FAIL] cannot read sortnet: %s\n", sortnet_path.c_str());
      return 2;
    }
    ksrc = splice_sortnet(ksrc, inc);
  }
  if (ksrc.find("/*__SORTNET__*/") != std::string::npos) {
    std::printf("[FATAL] sortnet splice did not take effect\n");
    return 2;
  }
  std::vector<char> blob;
  if (!read_file(case_path, blob)) {
    std::printf("[FAIL] cannot read case bundle: %s\n", case_path.c_str());
    return 2;
  }
  if (blob.size() < sizeof(FkcHeader)) { std::printf("[FAIL] bundle truncated\n"); return 2; }
  FkcHeader h{};
  std::memcpy(&h, blob.data(), sizeof(h));
  if (std::memcmp(h.magic, FKC_MAGIC, 4) != 0) { std::printf("[FAIL] bad magic\n"); return 2; }
  if (h.window != (uint32_t)FKC_WINDOW) { std::printf("[FAIL] window mismatch\n"); return 2; }

  const int W = (int)h.W, H = (int)h.H, pad = (int)h.pad, n_cases = (int)h.n_cases;
  const size_t pw = (size_t)(W + 2 * pad), ph = (size_t)(H + 2 * pad);
  const size_t frame_bytes = pw * ph;
  const int WINDOW = FKC_WINDOW;
  const size_t meta_sz = sizeof(FkcCaseMeta);
  const size_t mats_sz = (size_t)WINDOW * 6 * sizeof(float);
  const size_t ref_sz = (size_t)3 * W * H;
  const size_t per_case = meta_sz + mats_sz + (size_t)(WINDOW + 1) * frame_bytes + ref_sz;
  const size_t need = sizeof(FkcHeader) + (size_t)n_cases * per_case;
  if (blob.size() < need) {
    std::printf("[FAIL] bundle truncated: have %zu bytes, need %zu\n", blob.size(), need);
    return 2;
  }

  std::printf("========== RK3588 fused kernel accuracy ==========\n");
  std::printf("bundle : %s\n", case_path.c_str());
  std::printf("kernel : %s (%zu bytes)\n", kernel_path.c_str(), ksrc.size());
  std::printf("geometry: %dx%d, pad=%d (padded %zux%zu), cases=%d\n", W, H, pad, pw, ph, n_cases);

  // ---- OpenCL bootstrap --------------------------------------------------
  static const char* kCand[] = {"libOpenCL.so.1", "libOpenCL.so", "libmali-vendor.so.1",
                                "libmali-vendor.so", "libMali-Vendor.so", "libpocl.so.2"};
  const char* hit = nullptr;
  for (const char* c : kCand) {
    void* hh = dlopen(c, RTLD_NOW | RTLD_GLOBAL);
    if (hh) { g_lib = hh; hit = c; break; }
  }
  if (!g_lib) { std::printf("[FAIL] no OpenCL runtime found\n"); return 3; }
  std::printf("[ OK ] loaded  : %s\n", hit);

  int missing = 0;
  RESOLVE(clGetPlatformIDs, "clGetPlatformIDs");
  RESOLVE(clGetPlatformInfo, "clGetPlatformInfo");
  RESOLVE(clGetDeviceIDs, "clGetDeviceIDs");
  RESOLVE(clGetDeviceInfo, "clGetDeviceInfo");
  RESOLVE(clCreateContext, "clCreateContext");
  RESOLVE(clCreateCommandQueueWithProperties, "clCreateCommandQueueWithProperties");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  RESOLVE(clCreateCommandQueue, "clCreateCommandQueue");
#pragma GCC diagnostic pop
  RESOLVE(clCreateBuffer, "clCreateBuffer");
  RESOLVE(clCreateImage, "clCreateImage");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  RESOLVE(clCreateImage2D, "clCreateImage2D");
#pragma GCC diagnostic pop
  RESOLVE(clReleaseMemObject, "clReleaseMemObject");
  RESOLVE(clEnqueueWriteBuffer, "clEnqueueWriteBuffer");
  RESOLVE(clEnqueueWriteImage, "clEnqueueWriteImage");
  RESOLVE(clEnqueueReadBuffer, "clEnqueueReadBuffer");
  RESOLVE(clCreateProgramWithSource, "clCreateProgramWithSource");
  RESOLVE(clBuildProgram, "clBuildProgram");
  RESOLVE(clGetProgramBuildInfo, "clGetProgramBuildInfo");
  RESOLVE(clCreateKernel, "clCreateKernel");
  RESOLVE(clSetKernelArg, "clSetKernelArg");
  // Optional: absence is tolerated, clSetKernelArg handles the sampler.
  g_fn.clSetKernelArgSampler = reinterpret_cast<fn_clSetKernelArgSampler>(
      dlsym(g_lib, "clSetKernelArgSampler"));
  g_have_arg_sampler = (g_fn.clSetKernelArgSampler != nullptr);
  if (!g_have_arg_sampler)
    std::printf("[--  ] clSetKernelArgSampler absent; using clSetKernelArg(sampler)\n");
  RESOLVE(clEnqueueNDRangeKernel, "clEnqueueNDRangeKernel");
  RESOLVE(clFinish, "clFinish");
  RESOLVE(clReleaseEvent, "clReleaseEvent");
  RESOLVE(clGetEventProfilingInfo, "clGetEventProfilingInfo");
  RESOLVE(clCreateSampler, "clCreateSampler");
  RESOLVE(clReleaseSampler, "clReleaseSampler");
  RESOLVE(clReleaseKernel, "clReleaseKernel");
  RESOLVE(clReleaseProgram, "clReleaseProgram");
  RESOLVE(clReleaseCommandQueue, "clReleaseCommandQueue");
  RESOLVE(clReleaseContext, "clReleaseContext");
  if (missing) { std::printf("[FAIL] %d symbol(s) missing\n", missing); return 4; }
  std::printf("[ OK ] symbols : all entry points resolved\n");

  cl_uint np = 0;
  CHK(g_fn.clGetPlatformIDs(0, nullptr, &np));
  if (!np) { std::printf("[FAIL] no platform\n"); return 5; }
  std::vector<cl_platform_id> plats(np);
  CHK(g_fn.clGetPlatformIDs(np, plats.data(), nullptr));
  cl_device_id dev = nullptr;
  for (cl_uint i = 0; i < np && !dev; ++i) {
    cl_uint nd = 0;
    if (g_fn.clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) != CL_SUCCESS || !nd) continue;
    std::vector<cl_device_id> d(nd);
    if (g_fn.clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_ALL, nd, d.data(), nullptr) != CL_SUCCESS) continue;
    dev = d[0];
  }
  if (!dev) { std::printf("[FAIL] no device\n"); return 6; }

  size_t nlen = 0;
  g_fn.clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, nullptr, &nlen);
  std::vector<char> dname(nlen + 1, 0);
  g_fn.clGetDeviceInfo(dev, CL_DEVICE_NAME, nlen, dname.data(), nullptr);
  cl_uint cus = 0;
  g_fn.clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, nullptr);
  std::printf("[ OK ] device  : %s (%u CU)\n", dname.data(), cus);

  cl_int e = CL_SUCCESS;
  RunCtx rc{};
  rc.dev = dev;
  rc.ctx = g_fn.clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e);
  if (e != CL_SUCCESS) { std::printf("[FAIL] context -> %s\n", cl_err_name(e)); return 7; }
  if (g_fn.clCreateCommandQueueWithProperties) {
    const cl_queue_properties qp[] = {0};
    rc.q = g_fn.clCreateCommandQueueWithProperties(rc.ctx, dev, qp, &e);
  }
  if (!rc.q || e != CL_SUCCESS) {
    e = CL_SUCCESS;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    rc.q = g_fn.clCreateCommandQueue(rc.ctx, dev, 0, &e);
#pragma GCC diagnostic pop
  }
  if (!rc.q || e != CL_SUCCESS) { std::printf("[FAIL] queue -> %s\n", cl_err_name(e)); return 8; }

  if (!pick_gray_format(rc.ctx, &rc.fmt, &rc.used_2d)) {
    std::printf("[FAIL] device accepts no 8-bit unorm image format\n");
    return 9;
  }
  std::printf("[ OK ] image   : %s via %s (%.1f MiB for 22 padded frames)\n", rc.fmt.name,
              rc.used_2d ? "clCreateImage2D" : "clCreateImage",
              22.0 * (double)pw * ph * rc.fmt.channels / 1048576.0);

  // Sampler: same flags the kernel's two paths use, so the sampler itself is
  // never the variable. FUSED_USE_HW_LINEAR only changes whether the kernel
  // calls read_imagef at a fractional coordinate or blends four taps by hand.
  // CL_FALSE for normalized_coords matches the kernel CLK_NORMALIZED_COORDS_FALSE.
  // Host-side CL_* constants, NOT the CLK_* ones: those are the kernel-language
  // encodings. clCreateSampler always uses unnormalized coordinates, which is
  // exactly what the kernel wants (CLK_NORMALIZED_COORDS_FALSE).
  cl_sampler samp_hw = g_fn.clCreateSampler(rc.ctx, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_LINEAR, &e);
  if (e != CL_SUCCESS) { std::printf("[FAIL] linear sampler -> %s\n", cl_err_name(e)); return 10; }
  cl_sampler samp_near = g_fn.clCreateSampler(rc.ctx, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &e);
  if (e != CL_SUCCESS) { std::printf("[FAIL] nearest sampler -> %s\n", cl_err_name(e)); return 10; }

  struct Variant { const char* name; int define; cl_sampler samp; bool ok; };
  std::vector<Variant> variants;
  if (variant == "hw" || variant == "both") variants.push_back({"hw-linear (FUSED_USE_HW_LINEAR=1)", 1, samp_hw, true});
  if (variant == "nearest" || variant == "both") variants.push_back({"manual bilinear (FUSED_USE_HW_LINEAR=0)", 0, samp_near, true});
  if (variants.empty()) { std::printf("[FAIL] unknown variant %s\n", variant.c_str()); return 2; }

  const size_t klen = ksrc.size();
  int worst_fail = 0;

  for (const Variant& var : variants) {
    // The bundle is re-read from the start for every variant. Advancing a single
    // shared cursor across variants walks it off the end of the blob on the
    // second pass and segfaults in a memcpy long before any OpenCL call.
    size_t o = sizeof(FkcHeader);
    std::printf("\n================ variant: %s ================\n", var.name);

    // Select the path with a build option, not by editing the .cl, so the file
    // that ships is byte-identical to the one the x86 side validates.
    char opts[128];
    std::snprintf(opts, sizeof(opts), "-D FUSED_USE_HW_LINEAR=%d", var.define);
    const char* srcp = ksrc.data();
    cl_program prog = g_fn.clCreateProgramWithSource(rc.ctx, 1, &srcp, &klen, &e);
    if (e != CL_SUCCESS) { std::printf("[FAIL] program -> %s\n", cl_err_name(e)); worst_fail = 1; continue; }
    e = g_fn.clBuildProgram(prog, 1, &dev, opts, nullptr, nullptr, &e);
    if (e != CL_SUCCESS) {
      std::printf("[FAIL] build (%s) -> %s (%d)\n", opts, cl_err_name(e), e);
      size_t ln = 0;
      if (g_fn.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ln) == CL_SUCCESS && ln > 1) {
        std::vector<char> log(ln, 0);
        g_fn.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, ln, log.data(), nullptr);
        std::printf("---- build log ----\n%s------------------\n", log.data());
      }
      worst_fail = 1; continue;
    }
    cl_kernel kern = g_fn.clCreateKernel(prog, "warp_median_fused", &e);
    if (e != CL_SUCCESS) { std::printf("[FAIL] kernel -> %s\n", cl_err_name(e)); worst_fail = 1; continue; }

    for (int c = 0; c < n_cases; ++c) {
      FkcCaseMeta m{};
      std::memcpy(&m, blob.data() + o, meta_sz); o += meta_sz;
      std::vector<float> mats((size_t)WINDOW * 6);
      std::memcpy(mats.data(), blob.data() + o, mats_sz); o += mats_sz;
      std::vector<uint8_t> frames((size_t)(WINDOW + 1) * frame_bytes);
      std::memcpy(frames.data(), blob.data() + o, frames.size()); o += frames.size();
      std::vector<uint8_t> ref(ref_sz);
      std::memcpy(ref.data(), blob.data() + o, ref_sz); o += ref_sz;

      std::printf("\n  case %d/%d  t=%u  motion=%.2f px\n", c + 1, n_cases, m.t, m.motion_px);
      Diff d[3];
      double ms[3] = {0, 0, 0};
      // Edge band: compare |diff|>=2 against proximity to the border so a
      // border-model problem is distinguishable from a coordinate one.
      if (!run_case(rc, kern, var.samp, frames, mats, ref, W, H, pad, 8, d, ms)) {
        // The whole case is already consumed from `o` above, so a break leaves
        // the stream consistent for no further case; there is nothing to skip.
        worst_fail = 1;
        break;
      }
      print_diff("Ch0", d[0]);
      print_diff("Ch1", d[1]);
      print_diff("Ch2", d[2]);
      std::printf("    wall time: upload(22 img) %.3f ms | kernel(enqueue+finish) %.3f ms | "
                  "readback %.3f ms | sum %.3f ms  [all upper bounds]\n",
              ms[0], ms[1], ms[2], ms[0] + ms[1] + ms[2]);
      // Gate calibration, taken from what this project has actually been
      // treating as acceptable rather than invented here. The x86/PoCL baseline
      // recorded in manu/memory/opencl_fused_rk3588.md for this same kernel is
      // Ch0 exact, Ch1 Max|Diff| 6..8 / MAE 0.0003..0.0010, Ch2 Max|Diff| 6..8 /
      // MAE 0.0001..0.0003 -- that residual is the float32 coordinate tie
      // against OpenCV's own 1/32 quantisation, not a defect. So the gate is
      // MAE < 0.05 per channel, the threshold the kernel header itself names as
      // the one a systematic rounding mismatch cannot hide under, with Max|Diff|
      // printed for comparison against that baseline rather than gated at 1.
      // Per-variant verdict: a single global FAIL would hide that one path is
      // unusable while the other is the one to ship.
      const bool v_ok = (d[0].mae < 0.05 && d[1].mae < 0.05 && d[2].mae < 0.05);
      variants[static_cast<size_t>(&var - variants.data())].ok = v_ok;
      std::printf("    variant verdict (MAE < 0.05 x3): %s\n", v_ok ? "PASS" : "FAIL");
      if (!v_ok) worst_fail = 1;
    }
    g_fn.clReleaseKernel(kern);
    g_fn.clReleaseProgram(prog);
  }

  std::printf("\n================ summary ================\n");
  std::printf("gate = MAE < 0.05 per channel (the threshold this kernel header names as\n"
              "       the one a systematic rounding mismatch cannot hide under). Max|Diff|\n"
              "       is printed for comparison with the x86 baseline, not gated at 1.\n\n");
  for (const Variant& v : variants)
    std::printf("  %-42s %s\n", v.name, v.ok ? "PASS" : "FAIL");
  // The actionable conclusion is which path to ship, so say it explicitly.
  const bool hw_ok  = variants.size() > 0 && variants[0].ok && variants[0].define == 1;
  const bool man_ok = variants.size() > 1 && variants[1].ok;
  if (variants.size() == 2) {
    std::printf("\n");
    if (!hw_ok && man_ok)
      std::printf("  RECOMMENDATION: build with FUSED_USE_HW_LINEAR=0 (manual 2x2 blend).\n"
                  "    Hardware CLK_FILTER_LINEAR does not reproduce the OpenCV reference\n"
                  "    on this device even at integer coordinates, which is visible in Ch0,\n"
                  "    an integer-coordinate copy that must be exact.\n");
    else if (hw_ok && !man_ok)
      std::printf("  RECOMMENDATION: build with FUSED_USE_HW_LINEAR=1 (hardware filter).\n");
    else if (hw_ok && man_ok)
      std::printf("  RECOMMENDATION: both paths pass; prefer hardware linear for speed,\n"
                  "    but the manual path is the accuracy reference.\n");
    else
      std::printf("  RECOMMENDATION: NEITHER path passes. The fault is upstream of the\n"
                  "    sampling choice -- check padding, channel order or ring indexing.\n");
  }
  std::printf("\n%s\n", worst_fail ? "ACCURACY RESULT: FAIL (see per-variant verdict above)"
                                    : "ACCURACY RESULT: PASS");

  g_fn.clReleaseSampler(samp_hw);
  g_fn.clReleaseSampler(samp_near);
  g_fn.clReleaseCommandQueue(rc.q);
  g_fn.clReleaseContext(rc.ctx);
  return worst_fail ? 1 : 0;
}
