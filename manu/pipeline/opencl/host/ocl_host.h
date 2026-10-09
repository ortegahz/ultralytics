// ============================================================================
// ocl_host.h -- minimal, RAII, no-deprecated-API OpenCL host boilerplate.
//
// Split out from the accuracy checker so the eventual RK3588 pipeline can reuse
// it unchanged. Deliberately narrow: context + program + sampler + images +
// buffer + one blocking enqueue. No queues-per-device, no event pools, no
// thread pool, nothing that would need an OpenMP or std::thread runtime the
// RK3588 BSP image does not ship.
//
// Deployment notes:
//   * clCreateSamplerWithProperties (OpenCL 2.0) rather than clCreateSampler,
//     which is deprecated in 2.0 and emits warnings on a -Wall build.
//   * OpenCL is loaded at link time against the ICD loader (libOpenCL.so.1).
//     On RK3588 that resolves to the Mali libmali driver; here it resolves to
//     the PoCL ICD. Nothing else changes.
// ============================================================================
#ifndef MANU_OCL_HOST_H
#define MANU_OCL_HOST_H

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

/* Debian's opencl-headers omits the 8-bit image channel-order constants entirely
 * (they are absent from cl.h even though cl_channel_type is present). They are
 * in the OpenCL spec and Mali's vendor headers do define them, so these are
 * #ifndef-guarded: on RK3588 the vendor values win, here the spec values fill in.
 * Without this the kernel cannot be given a single-channel image at all. */
#ifndef CL_R8
#define CL_R8    0x10D0
#endif
#ifndef CL_RG8
#define CL_RG8   0x10D1
#endif
#ifndef CL_RGBA8
#define CL_RGBA8 0x10D2
#endif

/* Same story, larger scope: the whole cl_mem_object_type / cl_sampler /
 * cl_samples enumeration block is absent. dpkg -L opencl-headers lists no
 * headers at all, so /usr/include/CL here comes from opencl-c-headers
 * 3.0~2025.07.22, which declares the OpenCL 2.0 FUNCTIONS (clCreateImage,
 * clCreateSamplerWithProperties) but not the OpenCL 2.0 ENUMERATORS. These are
 * fixed by the OpenCL 2.0/3.0 specification and are #ifndef-guarded so Mali's
 * vendor headers keep their own values on RK3588. A wrong value here would fail
 * loudly at clCreateImage, not silently. */
#ifndef CL_MEM_OBJECT_NONE
#define CL_MEM_OBJECT_NONE          0x0000
#endif
#ifndef CL_MEM_OBJECT_OPENCL_2D
#define CL_MEM_OBJECT_OPENCL_2D     0x1000
#endif
#ifndef CL_MEM_OBJECT_OPENCL_2D_ARRAY
#define CL_MEM_OBJECT_OPENCL_2D_ARRAY 0x1001
#endif
#ifndef CL_MEM_OBJECT_OPENCL_3D
#define CL_MEM_OBJECT_OPENCL_3D     0x1002
#endif
#ifndef CL_MEM_OBJECT_OPENCL_BUFFER
#define CL_MEM_OBJECT_OPENCL_BUFFER 0x1010
#endif
#ifndef CL_IMAGE_OBJECT_2D
#define CL_IMAGE_OBJECT_2D          0x10F0
#endif
#ifndef CL_IMAGE_OBJECT_2D_ARRAY
#define CL_IMAGE_OBJECT_2D_ARRAY    0x10F1
#endif
#ifndef CL_IMAGE_OBJECT_3D
#define CL_IMAGE_OBJECT_3D          0x10F2
#endif
#ifndef CL_IMAGE_OBJECT_BUFFER
#define CL_IMAGE_OBJECT_BUFFER      0x10F3
#endif
#ifndef CL_IMAGE_OBJECT_2D_DEPTH
#define CL_IMAGE_OBJECT_2D_DEPTH    0x10F4
#endif
#ifndef CL_SAMPLES_UINT
#define CL_SAMPLES_UINT             0x1099
#endif
#ifndef CL_SAMPLES_INT
#define CL_SAMPLES_INT              0x109A
#endif
#ifndef CL_SAMPLES_UINT_DEPTH
#define CL_SAMPLES_UINT_DEPTH       0x10B0
#endif
#ifndef CL_SAMPLES_INT_DEPTH
#define CL_SAMPLES_INT_DEPTH        0x10B1
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ocl {

// Self-contained: clGetErrorString is OpenCL 2.0 and is absent from some
// distro header revisions, and the pipeline must build against whatever
// opencl-headers the RK3588 BSP happens to ship.
inline const char* err_str(cl_int e) {
    switch (e) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        default: return "see OpenCL spec, error code not enumerated here";
    }
}

inline void check(cl_int e, const char* what, const char* file, int line) {
    if (e != CL_SUCCESS) {
        std::fprintf(stderr, "[OCL FATAL] %s failed: %s (%d) at %s:%d\n", what,
                     err_str(e), (int)e, file, line);
        std::exit(2);
    }
}
#define OCL_CHECK(e, what) ::ocl::check((e), (what), __FILE__, __LINE__)

// RAII: cl_mem / cl_kernel / cl_program / cl_sampler / cl_event all released in
// reverse acquisition order. Leaking device memory here would not show up as a
// crash -- it would show up as the pipeline getting slower every hour -- so the
// destructors are not optional.
template <class T, cl_int (*R)(T)>
class Handle {
public:
    Handle() = default;
    explicit Handle(T h) : h_(h) {}
    ~Handle() { reset(); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    Handle(Handle&& o) noexcept : h_(o.h_) { o.h_ = nullptr; }
    Handle& operator=(Handle&& o) noexcept {
        if (this != &o) { reset(); h_ = o.h_; o.h_ = nullptr; }
        return *this;
    }
    void reset(T h = nullptr) {
        if (h_) R(h_);
        h_ = h;
    }
    T get() const { return h_; }
    explicit operator bool() const { return h_ != nullptr; }

private:
    T h_ = nullptr;
};

inline cl_int r_mem(cl_mem h) { return clReleaseMemObject(h); }
inline cl_int r_kernel(cl_kernel h) { return clReleaseKernel(h); }
inline cl_int r_program(cl_program h) { return clReleaseProgram(h); }
inline cl_int r_sampler(cl_sampler h) { return clReleaseSampler(h); }
inline cl_int r_event(cl_event h) { return clReleaseEvent(h); }

using Mem = Handle<cl_mem, r_mem>;
using Kernel = Handle<cl_kernel, r_kernel>;
using Program = Handle<cl_program, r_program>;
using Sampler = Handle<cl_sampler, r_sampler>;
using Event = Handle<cl_event, r_event>;

// ---------------------------------------------------------------------------
inline std::string device_info(cl_device_id d, cl_device_info what) {
    size_t n = 0;
    if (clGetDeviceInfo(d, what, 0, nullptr, &n) != CL_SUCCESS || n == 0) return "?";
    std::string s(n, '\0');
    clGetDeviceInfo(d, what, n, &s[0], nullptr);
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

inline std::string platform_info(cl_platform_id p, cl_platform_info what) {
    size_t n = 0;
    if (clGetPlatformInfo(p, what, 0, nullptr, &n) != CL_SUCCESS || n == 0) return "?";
    std::string s(n, '\0');
    clGetPlatformInfo(p, what, n, &s[0], nullptr);
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

// Picks the first platform/device that exists. On RK3588 that is the Mali
// driver; on the verification workstation it is PoCL's CPU device.
inline cl_device_id pick_device(cl_platform_id* out_platform = nullptr) {
    cl_uint np = 0;
    OCL_CHECK(clGetPlatformIDs(0, nullptr, &np), "clGetPlatformIDs(count)");
    if (np == 0) {
        std::fprintf(stderr, "[OCL FATAL] no OpenCL platform found\n");
        std::exit(2);
    }
    std::vector<cl_platform_id> ps(np);
    OCL_CHECK(clGetPlatformIDs(np, ps.data(), nullptr), "clGetPlatformIDs(list)");
    for (cl_platform_id p : ps) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) != CL_SUCCESS || nd == 0)
            continue;
        std::vector<cl_device_id> ds(nd);
        OCL_CHECK(clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, nd, ds.data(), nullptr),
                  "clGetDeviceIDs");
        if (out_platform) *out_platform = p;
        return ds[0];
    }
    std::fprintf(stderr, "[OCL FATAL] no OpenCL device found\n");
    std::exit(2);
}

inline void describe(cl_device_id d, cl_platform_id p) {
    // CL_DEVICE_MAX_WORK_GROUP_SIZE is a size_t and CL_DEVICE_GLOBAL_MEM_SIZE a
    // cl_ulong: querying either into a cl_uint silently returns CL_INVALID_VALUE
    // and a wall of zeroes, which reads like "the device has no memory".
    size_t wg = 0;
    cl_ulong mem = 0;
    cl_bool img = CL_FALSE;
    cl_uint cus = 0;
    clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wg), &wg, nullptr);
    clGetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
    clGetDeviceInfo(d, CL_DEVICE_IMAGE_SUPPORT, sizeof(img), &img, nullptr);
    clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, nullptr);
    std::printf("[OCL] platform : %s (%s)\n", platform_info(p, CL_PLATFORM_NAME).c_str(),
                platform_info(p, CL_PLATFORM_VERSION).c_str());
    std::printf("[OCL] device   : %s\n", device_info(d, CL_DEVICE_NAME).c_str());
    std::printf("[OCL] compute  : %u CU, max work-group %zu, %.0f MiB global, images %s\n", cus,
                wg, (double)mem / 1048576.0, img ? "yes" : "NO");
}

// cl_context is released by clReleaseContext, which has a different signature,
// so it gets its own tiny holder rather than bending Handle.
class Context {
public:
    explicit Context(cl_device_id d) {
        cl_int e;
        ctx_ = clCreateContext(nullptr, 1, &d, nullptr, nullptr, &e);
        OCL_CHECK(e, "clCreateContext");
        queue_ = clCreateCommandQueueWithProperties(ctx_, d, nullptr, &e);
        OCL_CHECK(e, "clCreateCommandQueueWithProperties");
    }
    ~Context() {
        if (queue_) clReleaseCommandQueue(queue_);
        if (ctx_) clReleaseContext(ctx_);
    }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    cl_context get() const { return ctx_; }
    cl_command_queue queue() const { return queue_; }

private:
    cl_context ctx_ = nullptr;
    cl_command_queue queue_ = nullptr;
};

// Splices the generated sorting network into the kernel source at the
// /*__SORTNET__*/ marker. Done host-side rather than with OpenCL's #include
// because #include in OpenCL C is optional (CL2.0 only) and not portable to the
// Mali driver; string splicing always works.
inline std::string splice_sortnet(const std::string& kernel_src, const std::string& inc_src,
                                  const std::string& marker = "/*__SORTNET__*/") {
    const size_t at = kernel_src.find(marker);
    if (at == std::string::npos) {
        std::fprintf(stderr, "[OCL FATAL] sortnet marker %s not found in kernel source\n",
                     marker.c_str());
        std::exit(2);
    }
    return kernel_src.substr(0, at) + inc_src + kernel_src.substr(at + marker.size());
}

inline Program build_program(cl_context ctx, cl_device_id d, const std::string& src,
                             const std::string& label, bool hw_linear, bool fp64_coord = false) {

    const char* s = src.c_str();
    const size_t n = src.size();
    cl_int e;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &s, &n, &e);
    OCL_CHECK(e, "clCreateProgramWithSource");

    // -D selects the sampling path. Both are compiled and measured; the flag is
    // a measurement knob, not a guess (see the FUSED_USE_HW_LINEAR note in the
    // kernel). -cl-mad-enable / -cl-fast-relaxed-math are deliberately NOT used:
    // the kernel's numerics are checked against a CPU golden reference, so
    // letting the compiler reassociate the coordinate arithmetic would make the
    // verification meaningless.
    char opts[256];
    std::snprintf(opts, sizeof(opts), "-cl-std=CL1.2 -D FUSED_USE_HW_LINEAR=%d -D FUSED_FP64_COORD=%d",
                  hw_linear ? 1 : 0, fp64_coord ? 1 : 0);
    e = clBuildProgram(prog, 1, &d, opts, nullptr, nullptr);
    if (e != CL_SUCCESS) {
        size_t len = 0;
        clGetProgramBuildInfo(prog, d, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        std::string log(len, '\0');
        clGetProgramBuildInfo(prog, d, CL_PROGRAM_BUILD_LOG, len, &log[0], nullptr);
        std::fprintf(stderr, "[OCL FATAL] build failed (%s)\n%s\n", err_str(e),
                     log.c_str());
        std::exit(2);
    }
    (void)label;
    return Program(prog);
}

/* Sampler construction is probed rather than assumed.
 *
 * clCreateSamplerWithProperties is the non-deprecated 2.0 entry point, but its
 * accepted property sets differ between runtimes -- PoCL 6.0 rejects a list that
 * the Mali driver accepts, and vice versa. Rather than hard-code whichever one
 * this verification box happens to like, try the legal sets in decreasing order
 * of specificity and keep the first the device accepts. The reported set is
 * printed so a run on RK3588 says which path it took.
 *
 * clCreateSampler (1.x) is kept only as a last resort and is wrapped in a scoped
 * diagnostic pragma so the build stays warning-free, per the acceptance bar.
 */
inline Sampler make_sampler(cl_context ctx, bool hw_linear) {
    const cl_filter_mode filt = hw_linear ? CL_FILTER_LINEAR : CL_FILTER_NEAREST;
    cl_int e = CL_INVALID_VALUE;

    /* CL_SAMPLER_MIP_FILTER_MODE is deliberately absent. Its desired value here is
     * CL_NONE, but CL_NONE == 0 and 0 is also the property-list TERMINATOR -- so
     * "CL_SAMPLER_MIP_FILTER_MODE, CL_NONE" is read as the property immediately
     * followed by the end of the list, and every runtime rejects it with
     * CL_INVALID_VALUE. CL_NONE is the default, so omitting it is both correct
     * and the only way to spell it. Verified against PoCL 6.0. */
    cl_sampler_properties props[] = {CL_SAMPLER_NORMALIZED_COORDS,
                                     static_cast<cl_sampler_properties>(CL_FALSE),
                                     CL_SAMPLER_ADDRESSING_MODE,
                                     static_cast<cl_sampler_properties>(CL_ADDRESS_CLAMP_TO_EDGE),
                                     CL_SAMPLER_FILTER_MODE,
                                     static_cast<cl_sampler_properties>(filt), 0};
    cl_sampler s = clCreateSamplerWithProperties(ctx, props, &e);
    if (e == CL_SUCCESS) return Sampler(s);

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    /* Last resort for a driver that rejects the 2.0 property list outright.
     * Wrapped in a scoped pragma so the build still carries no deprecation
     * warning, which is part of the acceptance bar. */
    // NOTE the 5-argument form: this header declares clCreateSampler with the
    // OpenCL 1.0 signature (no mip_filter_mode argument) while declaring
    // clCreateSamplerWithProperties and clCreateImage from OpenCL 2.0. Match
    // what the header actually provides rather than the 1.1 spec signature.
    s = clCreateSampler(ctx, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, filt, &e);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    OCL_CHECK(e, "clCreateSampler");
    std::fprintf(stderr, "[OCL] note: clCreateSamplerWithProperties was rejected; using "
                         "clCreateSampler\n");
    return Sampler(s);
}

/* 2D image allocation with a fallback, because two real obstacles were measured:
 *
 *  1. clCreateImage (OpenCL 2.0) returns CL_INVALID_IMAGE_DESCRIPTOR for EVERY
 *     format on this ICD -- including ones clCreateImage2D accepts. The distro
 *     cl.h declares a cl_image_desc whose layout does not match what the ICD
 *     loader dispatches on. So the 2.0 entry point is tried first (it is the
 *     non-deprecated one and will win on a correct stack, e.g. Mali) and
 *     clCreateImage2D is the fallback, wrapped in a scoped pragma so the build
 *     stays free of deprecation warnings.
 *  2. Channel order is probed separately: PoCL 6.0 rejects CL_R/CL_R8 for 8-bit
 *     images and accepts only CL_RGBA. Mali accepts CL_R8, which is a quarter of
 *     the bandwidth -- and at 22 padded frames per step that matters on RK3588.
 *     So the fastest format the device will take is used, and the choice is
 *     reported rather than assumed.
 */
inline bool try_image2d(cl_context ctx, const cl_image_format& fmt, size_t w, size_t h,
                        cl_mem* out) {
    cl_int e = CL_SUCCESS;
    cl_image_desc desc{};
    desc.image_type = CL_IMAGE_OBJECT_2D;
    desc.image_width = w;
    desc.image_height = h;
    desc.image_depth = 1;
    desc.image_array_size = 1;
    desc.num_samples = CL_SAMPLES_UINT;
    cl_mem m = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &e);
    if (e == CL_SUCCESS) { *out = m; return true; }

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    m = clCreateImage2D(ctx, CL_MEM_READ_ONLY, &fmt, w, h, 0, nullptr, &e);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    if (e != CL_SUCCESS) return false;
    *out = m;
    return true;
}

inline Mem make_image2d(cl_context ctx, const cl_image_format& fmt, size_t w, size_t h) {
    cl_mem m = nullptr;
    if (!try_image2d(ctx, fmt, w, h, &m)) {
        std::fprintf(stderr, "[OCL FATAL] clCreateImage/clCreateImage2D both rejected "
                             "%zux%zu (order 0x%04X type 0x%04X)\n",
                     w, h, (unsigned)fmt.image_channel_order,
                     (unsigned)fmt.image_channel_data_type);
        std::exit(2);
    }
    return Mem(m);
}

// Returns the cheapest 8-bit single-channel format the device actually accepts,
// and reports the choice. Unorm is forced: a signed 8-bit image is not the same
// texel interpretation even though it is the same byte count.
inline void pick_gray_format(cl_context ctx, cl_image_format* out, int* channels) {
    struct { const char* name; cl_uint order; int ch; } cand[] = {
        {"CL_R8   (1 byte/px, preferred on Mali/RK3588)", CL_R8, 1},
        {"CL_R    (1 byte/px)", CL_R, 1},
        {"CL_RGBA (4 byte/px, PoCL fallback)", CL_RGBA, 4},
    };
    for (const auto& c : cand) {
        const cl_image_format f{static_cast<cl_uint>(c.order),
                                static_cast<cl_uint>(CL_UNORM_INT8)};
        cl_mem probe = nullptr;
        if (try_image2d(ctx, f, 64, 64, &probe)) {
            clReleaseMemObject(probe);
            *out = f;
            *channels = c.ch;
            std::printf("[OCL] image fmt : %s\n", c.name);
            return;
        }
    }
    std::fprintf(stderr, "[OCL FATAL] device accepts no 8-bit unorm image format\n");
    std::exit(2);
}

inline std::string read_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "[OCL FATAL] cannot open %s\n", path.c_str());
        std::exit(2);
    }
    std::string out;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    std::fclose(f);
    return out;
}

}  // namespace ocl

#endif  // MANU_OCL_HOST_H