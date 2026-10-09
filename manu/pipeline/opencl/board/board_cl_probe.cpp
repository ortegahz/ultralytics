// ===========================================================================
// board_cl_probe.cpp -- RK3588 Mali GPU OpenCL environment probe (read-only)
//
// Purpose: answer "does this board actually have a usable OpenCL stack, and
// which one?" before any of the fused GMC/median kernel work is ported over.
// It is a probe, not a benchmark: it enumerates, it runs one trivial kernel,
// it verifies the result, and it prints a PASS/FAIL checklist.
//
// Why dlopen instead of linking -lOpenCL (see memory/opencl_fused_rk3588.md):
//   The x86 cross toolchain's sysroot ships no libOpenCL, and the vendor SDK
//   carries only a buildroot recipe, so a link-time dependency cannot be
//   satisfied at build time. dlopen defers that decision to the board, where
//   the answer is actually observable, and it turns "no OpenCL" into a clean
//   report instead of a link error. It also lets the probe name the library
//   that answered, which is the single most useful line for telling an ICD
//   loader apart from a directly-linked Mali driver.
//
// What it does NOT do: never writes outside /tmp, never touches device nodes,
// never runs anything resembling the real pipeline.
//
// Build: build_board_probe.sh, next to this file.
// ===========================================================================

#include <dlfcn.h>
#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <CL/cl.h>

// ---------------------------------------------------------------------------
// OpenCL entry points, resolved by hand.
//
// The Khronos header declares these as extern functions; we never call them by
// name, so the linker never needs libOpenCL. Each one keeps its true signature
// so the calls below stay type-correct -- funnelling them through a narrower
// function-pointer type would be undefined behaviour on the ABI level.
// ---------------------------------------------------------------------------

typedef cl_int(CL_API_CALL* fn_clGetPlatformIDs)(cl_uint, cl_platform_id*,
                                                  cl_uint*);
typedef cl_int(CL_API_CALL* fn_clGetPlatformInfo)(cl_platform_id,
                                                  cl_platform_info, size_t,
                                                  void*, size_t*);
typedef cl_int(CL_API_CALL* fn_clGetDeviceIDs)(cl_platform_id, cl_device_type,
                                               cl_uint, cl_device_id*, cl_uint*);
typedef cl_int(CL_API_CALL* fn_clGetDeviceInfo)(cl_device_id, cl_device_info,
                                                size_t, void*, size_t*);
typedef cl_context(CL_API_CALL* fn_clCreateContext)(
    const cl_context_properties*, cl_uint, const cl_device_id*,
    void(CL_CALLBACK*)(const char*, const void*, size_t, void*), void*, cl_int*);
typedef cl_command_queue(CL_API_CALL* fn_clCreateCommandQueue)(
    cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
typedef cl_command_queue(CL_API_CALL* fn_clCreateCommandQueueWithProperties)(
    cl_context, cl_device_id, const cl_queue_properties*, cl_int*);
typedef cl_mem(CL_API_CALL* fn_clCreateBuffer)(cl_context, cl_mem_flags, size_t,
                                               void*, cl_int*);
typedef cl_program(CL_API_CALL* fn_clCreateProgramWithSource)(cl_context, cl_uint,
                                                              const char**,
                                                              const size_t*,
                                                              cl_int*);
typedef cl_int(CL_API_CALL* fn_clBuildProgram)(
    cl_program, cl_uint, const cl_device_id*, const char*,
    void(CL_CALLBACK*)(cl_program, void*), void*, cl_int*);
typedef cl_int(CL_API_CALL* fn_clGetProgramBuildInfo)(cl_program,
                                                      cl_device_id,
                                                      cl_program_build_info,
                                                      size_t, void*, size_t*);
typedef cl_kernel(CL_API_CALL* fn_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_int(CL_API_CALL* fn_clSetKernelArg)(cl_kernel, cl_uint, size_t,
                                               const void*);
typedef cl_int(CL_API_CALL* fn_clEnqueueNDRangeKernel)(
    cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*,
    const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int(CL_API_CALL* fn_clEnqueueWriteBuffer)(cl_command_queue, cl_mem,
                                                     cl_bool, size_t, size_t,
                                                     const void*, cl_uint,
                                                     const cl_event*, cl_event*);
typedef cl_int(CL_API_CALL* fn_clEnqueueReadBuffer)(cl_command_queue, cl_mem,
                                                    cl_bool, size_t, size_t,
                                                    void*, cl_uint,
                                                    const cl_event*, cl_event*);
typedef cl_int(CL_API_CALL* fn_clFinish)(cl_command_queue);
typedef cl_int(CL_API_CALL* fn_clReleaseEvent)(cl_event);
typedef cl_int(CL_API_CALL* fn_clGetEventProfilingInfo)(cl_event,
                                                         cl_profiling_info,
                                                         size_t, void*, size_t*);

#define RESOLVE(field, name)                                       \
  do {                                                             \
    void* sym = dlsym(g_lib, name);                                \
    if (!sym) {                                                    \
      std::printf("[FAIL] missing symbol: %s\n", name);            \
      missing++;                                                   \
    }                                                              \
    g_fn.field = reinterpret_cast<fn_##field>(sym);                \
  } while (0)

struct FnTable {
  fn_clGetPlatformIDs clGetPlatformIDs;
  fn_clGetPlatformInfo clGetPlatformInfo;
  fn_clGetDeviceIDs clGetDeviceIDs;
  fn_clGetDeviceInfo clGetDeviceInfo;
  fn_clCreateContext clCreateContext;
  fn_clCreateCommandQueue clCreateCommandQueue;
  fn_clCreateCommandQueueWithProperties clCreateCommandQueueWithProperties;
  fn_clCreateBuffer clCreateBuffer;
  fn_clCreateProgramWithSource clCreateProgramWithSource;
  fn_clBuildProgram clBuildProgram;
  fn_clGetProgramBuildInfo clGetProgramBuildInfo;
  fn_clCreateKernel clCreateKernel;
  fn_clSetKernelArg clSetKernelArg;
  fn_clEnqueueNDRangeKernel clEnqueueNDRangeKernel;
  fn_clEnqueueWriteBuffer clEnqueueWriteBuffer;
  fn_clEnqueueReadBuffer clEnqueueReadBuffer;
  fn_clFinish clFinish;
  fn_clReleaseEvent clReleaseEvent;
  fn_clGetEventProfilingInfo clGetEventProfilingInfo;
};

static void* g_lib = nullptr;
static FnTable g_fn;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static const char* cl_err_name(cl_int e) {
  switch (e) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
    default: return "CL_<other>";
  }
}

#define OCL(expr)                                                            \
  do {                                                                      \
    cl_int _rc = (expr);                                                     \
    if (_rc != CL_SUCCESS) {                                                 \
      std::printf("[FAIL] %s -> %s (%d)\n", #expr, cl_err_name(_rc), _rc);   \
      return 1;                                                             \
    }                                                                       \
  } while (0)

// Every clSetKernelArg result is checked. An unchecked one turns into a launch
// failure much later, with an error code that points at the wrong thing.
#define SET_ARG(index, value)                                                 \
  do {                                                                       \
    cl_int _rc = g_fn.clSetKernelArg(kern, (index), sizeof(value), &(value));  \
    if (_rc != CL_SUCCESS) {                                                 \
      std::printf("[FAIL] clSetKernelArg(%d, %s) -> %s (%d)\n", (index),      \
                  #value, cl_err_name(_rc), _rc);                            \
      return 13;                                                             \
    }                                                                        \
  } while (0)

static std::string platform_str(cl_platform_id p, cl_platform_info q) {
  size_t n = 0;
  // CL_SUCCESS is 0, so this must be an explicit != CL_SUCCESS. Testing the
  // result for truth here would treat *success* as failure -- CL_SUCCESS==0
  // makes !rc true. (The ulongs below got this right and strings did not,
  // which is exactly the kind of split that survives a code review.)
  if (g_fn.clGetPlatformInfo(p, q, 0, nullptr, &n) != CL_SUCCESS || n == 0)
    return "<n/a>";
  std::vector<char> buf(n + 1, '\0');
  if (g_fn.clGetPlatformInfo(p, q, n, buf.data(), nullptr) != CL_SUCCESS)
    return "<n/a>";
  return std::string(buf.data());
}

static std::string device_str(cl_device_id d, cl_device_info q) {
  size_t n = 0;
  if (g_fn.clGetDeviceInfo(d, q, 0, nullptr, &n) != CL_SUCCESS || n == 0)
    return "<n/a>";
  std::vector<char> buf(n + 1, '\0');
  if (g_fn.clGetDeviceInfo(d, q, n, buf.data(), nullptr) != CL_SUCCESS)
    return "<n/a>";
  return std::string(buf.data());
}

static cl_ulong device_val(cl_device_id d, cl_device_info q) {
  cl_ulong v = 0;
  if (g_fn.clGetDeviceInfo(d, q, sizeof(v), &v, nullptr) != CL_SUCCESS) return 0;
  return v;
}

static void section(const char* title) {
  std::printf("\n=== %s ===\n", title);
}

static void print_build_log(cl_program prog, cl_device_id dev) {
  size_t n = 0;
  if (!g_fn.clGetProgramBuildInfo) return;
  if (g_fn.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                                &n) != CL_SUCCESS ||
      n <= 1)
    return;
  std::vector<char> buf(n, '\0');
  if (g_fn.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, n, buf.data(),
                                nullptr) != CL_SUCCESS)
    return;
  std::printf("---- build log ----\n%s------------------\n", buf.data());
}

int main(int argc, char** argv) {
  static const char* const kCandidates[] = {
      "libOpenCL.so.1",      // ICD loader (ocl-icd) -- the usual Linux answer
      "libOpenCL.so",        // vendor drop-in, or a symlink to the loader
      "libmali-vendor.so.1", // Mali user-space driver, when linked directly
      "libmali-vendor.so",
      "libMali-Vendor.so",
      "libpocl.so.2",        // last resort: PoCL running on the board itself
  };
  const int kCount = (argc > 1)
                         ? argc - 1
                         : (int)(sizeof(kCandidates) / sizeof(kCandidates[0]));
  const char* const* cands =
      (argc > 1) ? reinterpret_cast<const char* const*>(argv + 1) : kCandidates;

  std::printf("========== RK3588 OpenCL environment probe ==========\n");
  std::printf("sonames to try (%d):", kCount);
  for (int i = 0; i < kCount; ++i) std::printf(" %s", cands[i]);
  std::printf("\n");

  const char* hit = nullptr;
  for (int i = 0; i < kCount; ++i) {
    void* h = dlopen(cands[i], RTLD_NOW | RTLD_GLOBAL);
    if (h) {
      g_lib = h;
      hit = cands[i];
      break;
    }
    std::printf("       dlopen %-20s -> %s\n", cands[i], dlerror());
  }
  if (!g_lib) {
    std::printf("[FAIL] could not dlopen any OpenCL soname.\n");
    std::printf("       -> no OpenCL runtime is visible to this process.\n");
    return 2;
  }
  std::printf("[ OK ] loaded: %s\n", hit);

  // An ICD loader only knows a vendor exists if an .icd entry names one.
  //
  // Two layouts exist and conflating them produces a misleading report: the
  // Khronos layout is a single FILE at /etc/OpenCL/vendors, while ARM's Mali
  // stack uses a DIRECTORY of per-vendor .icd files. fopen() on a directory
  // succeeds on glibc and the first read then fails with EISDIR, so a naive
  // implementation reports "empty" for a perfectly valid configuration. That
  // is exactly what happened on the RK3588 before this was fixed.
  section("ICD vendor configuration");
  struct stat icd_stat;
  const char* kIcdPath = "/etc/OpenCL/vendors";
  if (stat(kIcdPath, &icd_stat) != 0) {
    std::printf("[--  ] %s absent (expected for a directly-linked driver)\n",
                kIcdPath);
  } else if (S_ISDIR(icd_stat.st_mode)) {
    std::printf("[ OK ] %s is a directory (ARM/Mali-style ICD)\n", kIcdPath);
    std::vector<std::string> entries;
    if (DIR* d = opendir(kIcdPath)) {
      while (struct dirent* e = readdir(d)) {
        const std::string name = e->d_name;
        if (name == "." || name == "..") continue;
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".icd") == 0)
          entries.push_back(name);
      }
      closedir(d);
    }
    if (entries.empty()) {
      std::printf("[FAIL] directory contains no .icd entry\n");
    } else {
      for (const std::string& name : entries) {
        const std::string full = std::string(kIcdPath) + "/" + name;
        std::printf("[ OK ] icd entry: %s = ", name.c_str());
        if (std::FILE* f = std::fopen(full.c_str(), "rb")) {
          char line[512] = {0};
          if (std::fgets(line, sizeof(line), f))
            std::printf("%s", line);
          else
            std::printf("<unreadable>\n");
          std::fclose(f);
        } else {
          std::printf("<cannot open>\n");
        }
      }
    }
  } else {
    std::printf("[ OK ] %s is a file (Khronos-style ICD)\n", kIcdPath);
    std::FILE* f = std::fopen(kIcdPath, "rb");
    if (!f) {
      std::printf("[FAIL] cannot open for reading\n");
    } else {
      char line[512];
      int n = 0;
      while (std::fgets(line, sizeof(line), f)) {
        size_t len = std::strlen(line);
        if (len == 0 || line[0] == '\n' || line[0] == '#') continue;
        std::printf("[ OK ] icd entry: %s", line);
        if (line[len - 1] != '\n') std::printf("\n");
        ++n;
      }
      std::fclose(f);
      if (n == 0) std::printf("[FAIL] %s contains no vendor entry\n", kIcdPath);
    }
  }

  section("symbol resolution");
  int missing = 0;
  RESOLVE(clGetPlatformIDs, "clGetPlatformIDs");
  RESOLVE(clGetPlatformInfo, "clGetPlatformInfo");
  RESOLVE(clGetDeviceIDs, "clGetDeviceIDs");
  RESOLVE(clGetDeviceInfo, "clGetDeviceInfo");
  RESOLVE(clCreateContext, "clCreateContext");
  // Deprecated in OpenCL 2.0, but kept deliberately: some vendor stacks still
  // only export it, and the probe's job is to find out what the board has.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  RESOLVE(clCreateCommandQueue, "clCreateCommandQueue");
  RESOLVE(clCreateCommandQueueWithProperties, "clCreateCommandQueueWithProperties");
#pragma GCC diagnostic pop
  RESOLVE(clCreateBuffer, "clCreateBuffer");
  RESOLVE(clCreateProgramWithSource, "clCreateProgramWithSource");
  RESOLVE(clBuildProgram, "clBuildProgram");
  RESOLVE(clGetProgramBuildInfo, "clGetProgramBuildInfo");
  RESOLVE(clCreateKernel, "clCreateKernel");
  RESOLVE(clSetKernelArg, "clSetKernelArg");
  RESOLVE(clEnqueueNDRangeKernel, "clEnqueueNDRangeKernel");
  RESOLVE(clEnqueueWriteBuffer, "clEnqueueWriteBuffer");
  RESOLVE(clEnqueueReadBuffer, "clEnqueueReadBuffer");
  RESOLVE(clFinish, "clFinish");
  RESOLVE(clReleaseEvent, "clReleaseEvent");
  RESOLVE(clGetEventProfilingInfo, "clGetEventProfilingInfo");
  if (missing) {
    std::printf("[FAIL] %d required symbol(s) missing\n", missing);
    return 3;
  }
  std::printf("[ OK ] all entry points resolved\n");
  if (!g_fn.clCreateCommandQueueWithProperties)
    std::printf("[--  ] clCreateCommandQueueWithProperties absent, "
                "using the deprecated clCreateCommandQueue\n");

  section("platforms");
  cl_uint np = 0;
  OCL(g_fn.clGetPlatformIDs(0, nullptr, &np));
  if (np == 0) {
    std::printf("[FAIL] clGetPlatformIDs returned 0 platforms "
                "(loader present but no driver registered)\n");
    return 4;
  }
  std::printf("[ OK ] %u platform(s)\n", np);
  std::vector<cl_platform_id> plats(np);
  OCL(g_fn.clGetPlatformIDs(np, plats.data(), nullptr));
  for (cl_uint i = 0; i < np; ++i) {
    std::printf("  [%u] name    : %s\n", i,
                platform_str(plats[i], CL_PLATFORM_NAME).c_str());
    std::printf("      vendor  : %s\n",
                platform_str(plats[i], CL_PLATFORM_VENDOR).c_str());
    std::printf("      version : %s\n",
                platform_str(plats[i], CL_PLATFORM_VERSION).c_str());
    std::printf("      profile : %s\n",
                platform_str(plats[i], CL_PLATFORM_PROFILE).c_str());
  }

  section("devices");
  cl_device_id chosen_d = nullptr;
  int total_devices = 0;
  for (cl_uint i = 0; i < np && !chosen_d; ++i) {
    cl_uint nd = 0;
    if (g_fn.clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) !=
            CL_SUCCESS ||
        nd == 0) {
      std::printf("  platform %u: no device\n", i);
      continue;
    }
    std::vector<cl_device_id> devs(nd);
    if (g_fn.clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_ALL, nd, devs.data(),
                            nullptr) != CL_SUCCESS)
      continue;
    total_devices += (int)nd;
    for (cl_uint j = 0; j < nd; ++j) {
      const bool is_gpu =
          (device_val(devs[j], CL_DEVICE_TYPE) & CL_DEVICE_TYPE_GPU) != 0;
      std::printf("  platform %u device %u%s\n", i, j, is_gpu ? "   <-- GPU" : "");
      std::printf("      name           : %s\n",
                  device_str(devs[j], CL_DEVICE_NAME).c_str());
      std::printf("      vendor         : %s\n",
                  device_str(devs[j], CL_DEVICE_VENDOR).c_str());
      std::printf("      driver version : %s\n",
                  device_str(devs[j], CL_DEVICE_VERSION).c_str());
      std::printf("      C version      : %s\n",
                  device_str(devs[j], CL_DEVICE_OPENCL_C_VERSION).c_str());
      std::printf("      compute units  : %llu\n",
                  (unsigned long long)device_val(devs[j],
                                                 CL_DEVICE_MAX_COMPUTE_UNITS));
      std::printf("      max work group : %llu\n",
                  (unsigned long long)device_val(devs[j],
                                                 CL_DEVICE_MAX_WORK_GROUP_SIZE));
      std::printf("      global mem     : %.1f MiB\n",
                  (double)device_val(devs[j], CL_DEVICE_GLOBAL_MEM_SIZE) /
                      (1024.0 * 1024.0));
      std::printf("      local mem      : %llu KiB\n",
                  (unsigned long long)(device_val(devs[j],
                                                  CL_DEVICE_LOCAL_MEM_SIZE) /
                                       1024));
      std::printf("      extensions     : %s\n",
                  device_str(devs[j], CL_DEVICE_EXTENSIONS).c_str());
      if (!chosen_d) chosen_d = devs[j];
    }
  }
  if (!chosen_d) {
    std::printf("[FAIL] no OpenCL device on any platform\n");
    return 5;
  }
  std::printf("\n[ OK ] %d device(s); using the first for the smoke test\n",
              total_devices);

  // ------------------------------------------------------------------
  // Trivial kernel: out[i] = in[i]*k + 1. Short enough to audit by eye,
  // and exact enough that a wrong result cannot be blamed on rounding.
  // ------------------------------------------------------------------
  section("kernel smoke test (out[i] = in[i]*3 + 1)");
  cl_int err = CL_SUCCESS;
  cl_context ctx = g_fn.clCreateContext(nullptr, 1, &chosen_d, nullptr, nullptr,
                                        &err);
  if (!ctx || err != CL_SUCCESS) {
    std::printf("[FAIL] clCreateContext -> %s (%d)\n", cl_err_name(err), err);
    return 6;
  }
  std::printf("[ OK ] context created\n");

  // Prefer the non-deprecated entry point; older drivers only have the old one.
  cl_command_queue q = nullptr;
  if (g_fn.clCreateCommandQueueWithProperties) {
    cl_queue_properties qp[] = {0};
    q = g_fn.clCreateCommandQueueWithProperties(ctx, chosen_d, qp, &err);
  }
  if (!q || err != CL_SUCCESS) {
    err = CL_SUCCESS;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    q = g_fn.clCreateCommandQueue(ctx, chosen_d, 0, &err);
#pragma GCC diagnostic pop
  }
  if (!q || err != CL_SUCCESS) {
    std::printf("[FAIL] clCreateCommandQueue -> %s (%d)\n", cl_err_name(err), err);
    return 7;
  }
  std::printf("[ OK ] command queue created\n");

  const size_t N = 4096;
  const cl_int kMul = 3;
  std::vector<cl_int> in(N), out(N, -1);
  for (size_t i = 0; i < N; ++i) in[i] = (cl_int)i;

  cl_mem b_in = g_fn.clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    N * sizeof(cl_int), in.data(), &err);
  if (!b_in || err != CL_SUCCESS) {
    std::printf("[FAIL] clCreateBuffer(in) -> %s (%d)\n", cl_err_name(err), err);
    return 8;
  }
  cl_mem b_out =
      g_fn.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, N * sizeof(cl_int), nullptr, &err);
  if (!b_out || err != CL_SUCCESS) {
    std::printf("[FAIL] clCreateBuffer(out) -> %s (%d)\n", cl_err_name(err), err);
    return 8;
  }

  const char* src =
      "__kernel void axpy(__global int* out, __global const int* in, "
      "const int n, const int k) {\n"
      "  int i = get_global_id(0);\n"
      "  if (i < n) out[i] = in[i] * k + 1;\n"
      "}\n";
  const size_t src_len = std::strlen(src);
  cl_program prog =
      g_fn.clCreateProgramWithSource(ctx, 1, &src, &src_len, &err);
  if (!prog || err != CL_SUCCESS) {
    std::printf("[FAIL] clCreateProgramWithSource -> %s (%d)\n",
                cl_err_name(err), err);
    return 9;
  }
  err = g_fn.clBuildProgram(prog, 1, &chosen_d, /*options=*/nullptr,
                            /*pfn_notify=*/nullptr, /*user_data=*/nullptr, &err);
  if (err != CL_SUCCESS) {
    std::printf("[FAIL] clBuildProgram -> %s (%d)\n", cl_err_name(err), err);
    print_build_log(prog, chosen_d);
    return 10;
  }
  std::printf("[ OK ] program compiled\n");

  cl_kernel kern = g_fn.clCreateKernel(prog, "axpy", &err);
  if (!kern || err != CL_SUCCESS) {
    std::printf("[FAIL] clCreateKernel -> %s (%d)\n", cl_err_name(err), err);
    return 11;
  }
  // The kernel declares `const int n`, so the argument must be set as a 4-byte
  // cl_int. Passing sizeof(size_t) here fails with CL_INVALID_ARG_SIZE, the
  // argument stays unset, and the launch then fails -- and PoCL reports that as
  // CL_INVALID_GLOBAL_WORK_SIZE (-52) rather than the accurate -51, which is why
  // it is so easy to chase this in the wrong place. Hence SET_ARG checks.
  const cl_int n_i = (cl_int)N;
  SET_ARG(0, b_out);
  SET_ARG(1, b_in);
  SET_ARG(2, n_i);
  SET_ARG(3, kMul);
  std::printf("[ OK ] kernel instantiated and arguments set\n");

  OCL(g_fn.clEnqueueWriteBuffer(q, b_in, CL_TRUE, 0, N * sizeof(cl_int),
                                in.data(), 0, nullptr, nullptr));

  const size_t lws = 64;
  const size_t gws = 256;
  const size_t global = ((N + gws - 1) / gws) * gws;
  // Timing discipline (memory/opencl_fused_rk3588.md sec.7): clEnqueueNDRangeKernel
  // returns at *enqueue* time. Timing around it alone reads ~0.04 ms for 116 ms
  // of work because the compute hides inside the blocking read-back -- that
  // mistake produced a wrong conclusion once already. So: one wall-clock span
  // across enqueue + clFinish (an upper bound, launch overhead included), plus
  // a device-side number from event profiling when the driver offers it.
  auto t0 = std::chrono::steady_clock::now();
  cl_event ev = nullptr;
  // clEnqueueNDRangeKernel's real ABI order is (offset, size, local). All three
  // are the same pointer type, so a mix-up cannot be caught by the compiler --
  // and passing offset=&global with size=&lws trips
  // CL_INVALID_GLOBAL_WORK_SIZE, which reads like a launch-size problem.
  OCL(g_fn.clEnqueueNDRangeKernel(q, kern, /*work_dim=*/1,
                                  /*global_work_offset=*/nullptr, &global, &lws,
                                  /*num_events=*/0, nullptr, &ev));
  OCL(g_fn.clFinish(q));
  auto t1 = std::chrono::steady_clock::now();
  const double wall_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::printf("[ OK ] kernel executed\n");
  std::printf("       wall (enqueue+finish, upper bound) : %.3f ms\n", wall_ms);

  if (ev && g_fn.clGetEventProfilingInfo) {
    cl_ulong start = 0, end = 0;
    if (g_fn.clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(start),
                                     &start, nullptr) == CL_SUCCESS &&
        g_fn.clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(end),
                                     &end, nullptr) == CL_SUCCESS &&
        end >= start) {
      std::printf("       device-side (profiling event)     : %.3f ms\n",
                  (double)(end - start) / 1.0e6);
    } else {
      std::printf("       device-side (profiling event)     : "
                  "<unavailable on this driver>\n");
    }
    if (g_fn.clReleaseEvent) g_fn.clReleaseEvent(ev);
  }

  OCL(g_fn.clEnqueueReadBuffer(q, b_out, CL_TRUE, 0, N * sizeof(cl_int),
                               out.data(), 0, nullptr, nullptr));

  size_t bad = 0, first_bad = 0;
  for (size_t i = 0; i < N; ++i) {
    if (out[i] != (cl_int)i * kMul + 1) {
      if (!bad) first_bad = i;
      ++bad;
    }
  }
  section("result verification");
  if (bad == 0) {
    std::printf("[ OK ] all %zu elements correct\n", N);
  } else {
    std::printf("[FAIL] %zu of %zu wrong; first at i=%zu (got %d, want %d)\n",
                bad, N, first_bad, (int)out[first_bad],
                (int)((cl_int)first_bad * kMul + 1));
  }

  section("summary");
  std::printf("soname        : %s\n", hit);
  std::printf("device count  : %d\n", total_devices);
  std::printf("wall time     : %.3f ms (N=%zu, launch overhead included)\n",
              wall_ms, N);
  std::printf("kernel verify : %s\n", bad == 0 ? "PASS" : "FAIL");
  std::printf("\n%s\n", bad == 0 ? "PROBE RESULT: PASS" : "PROBE RESULT: FAIL");
  return bad == 0 ? 0 : 12;
}
