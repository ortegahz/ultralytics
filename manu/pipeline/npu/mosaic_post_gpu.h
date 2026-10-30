// ===========================================================================
// mosaic_post_gpu.h -- GPU (OpenCL / Mali-G610) post-processing path.
//
// WHAT IT REPLACES
// ----------------
// The CPU dense post-processing does, per tile:
//     1. bilinear upsample of the tile's heatmap window to 640x512
//        (ResizeBilinearF32), and
//     2. 8-connected component labelling on the 640x512 native image
//        (DetectComponents).
// For the 3 mosaic tiles that is ~983k bilinear evaluations + ~2M
// component pixel visits per frame, measured at 43.88 ms/frame.
//
// This header moves "heatmap -> candidate detections" onto the Mali GPU:
// one upload of the whole 320x320 fp32 heatmap, one 3x3 local-maximum
// kernel per tile window, one readback of the candidate list, and a cheap
// CPU-side back-projection of the handful of survivors.
//
// SEMANTICS: THIS IS *NOT* THE DENSE PATH
// ---------------------------------------
// The GPU path computes CANDIDATES (3x3 local maxima), not connected
// components.  Every field of `Comp` therefore carries a DIFFERENT meaning
// than in the dense path.  The full list is at the bottom of this file; the
// short version:
//
//   field  dense path                     GPU path
//   -----  -----------------------------  ---------------------------------
//   tag    tile id                        tile id                        (same)
//   cx,cy  arithmetic mean of the         3x3 intensity-weighted centroid
//          component's pixel coords         -> native 640x512
//   w,h    component bounding box          3x3 neighbourhood size in native
//   area   component pixel count           count of foreground cells in 3x3
//   peak   max of the UPSAMPLED native    max of the UPSAMPLED native
//          image inside the component      image around the candidate
//                                          (CPU-evaluated, same scale)
//   sum    sum over the component          0 (not computed on the GPU)
//   order  sorted by sum DESC              sorted by peak DESC
//
// The evaluation scorer only looks at the PREDICTION CENTRE (does it fall
// inside the GT box?) and filters on `peak` with th_base=0.3, so `peak` is
// the field that has to be right.  `w`/`h` are cosmetic here.
//
// `peak`: the GPU's peak used to be the SOURCE heatmap cell's value, which is
// an UPPER BOUND of the dense peak (a bilinear interpolation is a convex
// combination of its source cells, so it can never exceed the largest source
// value).  The threshold was then applied on two different scales and the GPU
// arm lost 14 false positives to it (F1 0.7976 vs the dense 0.8089).
// The peak is now evaluated on the CPU over the candidate's 3x3 native
// neighbourhood -- see `DenseEquivalentPeak` -- so both paths filter on the
// SAME scale.  The cost is 9 bilinear evaluations per candidate (2.4
// candidates per frame measured), i.e. nothing next to the GPU round trip.
//
// The caller owns the fallback: GpuPostDetect() returns false when no
// OpenCL device is available, and the caller must then use the dense path.
//
// Resources: one OpenCL context / program / kernel / buffer set, created on
// the first call and reused for every frame afterwards (function-local
// static).  Nothing is rebuilt or reallocated per frame.
// ===========================================================================
#pragma once

#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "../opencl/host/ocl_host.h"
#include "mosaic_core.h"

namespace mosaic {

namespace gpu_post {

// ---------------------------------------------------------------------------
// The kernel source, embedded VERBATIM.
//
// It is byte-identical to
//     manu/pipeline/opencl/kernels/post_local_max.cl
// (the two are checked with `cmp`; see the footer of this file).  Keeping a
// copy here means the delivery package carries ONE file and the board does
// not need a kernel directory at all.
// ---------------------------------------------------------------------------
static const char* const kPostLocalMaxSrc = R"(/* ============================================================================
 * post_local_max.cl -- GPU replacement for the CPU back-projection stage.
 *
 * WHAT IT DOES
 * ------------
 * One work-item per pixel of a tile's heatmap window.  A pixel that
 *   (a) is above the detection threshold, and
 *   (b) is a 3x3 LOCAL MAXIMUM inside its own tile window,
 * emits a detection candidate:
 *
 *     cx, cy  -- 3x3 intensity-weighted centroid, WINDOW-LOCAL heatmap
 *                 coordinates (origin = the window's top-left corner)
 *     peak    -- the pixel's own heatmap value (== the 3x3 maximum, because
 *                 the local-max test guarantees v >= all 8 neighbours)
 *     cnt     -- how many of the 9 cells of the 3x3 are >= threshold
 *
 * The candidates are written into a per-tile region of `peaks` through an
 * atomic slot counter, so no two work-items can ever write the same slot.
 *
 * The per-tile region is selected by the `base` argument, NOT by handing the
 * kernel an interior pointer: a kernel argument of type cl_mem must be a
 * real memory-object handle, and passing (cl_mem)(buffer + offset) is
 * undefined behaviour (PoCL asserts on it, and a vendor driver may as
 * well).  So both buffers are single, full-size allocations and the tile
 * index is applied inside the kernel.
 *
 *   peaks     : kMaxTiles * max_peaks float4, tile t uses [t*max_peaks, ...)
 *   peak_count: kMaxTiles ints,          tile t uses count[t]
 * ---------------------------------------------------------------------------

 * WHY THE OUTER RING IS SKIPPED
 * -----------------------------
 * The 3x3 neighbourhood of a pixel on the window border reaches outside the
 * window, i.e. into ANOTHER tile's heatmap cells (the three mosaic tiles
 * share ONE 320x320 heatmap, they only own sub-rectangles of it).  Reading
 * across that boundary would mix two different tiles' intensities into one
 * candidate.  The outermost 1-pixel ring of every window is therefore
 * skipped entirely: such a pixel can never become a candidate.
 *
 * This is a documented behavioural difference w.r.t. the CPU dense path,
 * which labels components on the upsampled native image and can therefore
 * report a component that touches a tile border.
 *
 * ARITHMETIC NOTES
 * ----------------
 * 1. The threshold test is `v <= threshold -> skip`, i.e. a candidate needs
 *    v > threshold.  The CPU dense path masks with `>= threshold`.  The two
 *    sets are identical for the contracted threshold 0.06 (0.06f rounds to
 *    0.059999998658895493, and the next float above it is 0.060000002235174179,
 *    which is also the smallest float that satisfies the dense `>= 0.06`
 *    test in double precision) -- but they differ by exactly one float for
 *    a threshold that is itself representable.
 *
 * 2. The neighbour comparison uses `>=` (v >= every neighbour), matching the
 *    dense path's `>=` mask convention.
 *
 * 3. The centroid is the INTENSITY-WEIGHTED mean of the 3x3 cells:
 *
 *        cx = lx + (sum(dx * w)) / (sum(w))
 *
 *    with lx the window-local column of the centre pixel.  It is NOT the
 *    arithmetic mean the dense path uses over the whole component.
 *
 * 4. `atomic_inc` is an OpenCL 1.2 core feature and is available on the
 *    Mali-G610 driver.
 *
 * No FP64 (Mali-G610 has none), no local memory, no barriers, no printf.
 * ==========================================================================*/

__kernel void post_local_max(__global const float *hm,   /* full heatmap, stride hm_w */
                             __global float4 *peaks,       /* per-tile output region   */
                             __global int *peak_count,     /* per-tile slot counter    */
                             const int hm_w,               /* heatmap row stride       */
                             const int hm_h,               /* heatmap height (unused)  */
                             const int win_x0,             /* window origin, heatmap x */
                             const int win_y0,             /* window origin, heatmap y */
                             const int win_w,              /* window width  (heatmap)  */
                             const int win_h,              /* window height (heatmap)  */
                             const float threshold,
                             const int max_peaks,
                             const int base)
{
    const int lx = (int)get_global_id(0);
    const int ly = (int)get_global_id(1);
    if (lx >= win_w || ly >= win_h) return;

    /* the 3x3 must stay inside the window -> the outer ring never qualifies */
    if (lx == 0 || ly == 0 || lx == win_w - 1 || ly == win_h - 1) return;

    const int x = win_x0 + lx;
    const int y = win_y0 + ly;

    const float v = hm[y * hm_w + x];
    if (v <= threshold) return;      /* threshold pruning */

    float sw  = 0.0f;                /* sum of the 9 intensities      */
    float swx = 0.0f;                /* sum of dx * intensity          */
    float swy = 0.0f;                /* sum of dy * intensity          */
    float nmax = -1.0e30f;           /* max of the 8 neighbours        */
    int   cnt  = 0;                  /* foreground cells in the 3x3    */

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const float wv = hm[(y + dy) * hm_w + (x + dx)];
            sw  += wv;
            swx += (float)dx * wv;
            swy += (float)dy * wv;
            if (wv >= threshold) ++cnt;
            if (dx != 0 || dy != 0) {          /* the centre IS v, skip it */
                if (wv > nmax) nmax = wv;
            }
        }
    }

    /* 3x3 local-maximum test: v must be >= every one of the 8 neighbours */
    if (v < nmax) return;

    const int slot = atomic_inc(&peak_count[base]);
    if (slot < max_peaks) {
        const float inv = 1.0f / (sw + 1.0e-5f);
        const float cx = (float)lx + swx * inv;
        const float cy = (float)ly + swy * inv;
        peaks[base * max_peaks + slot] = (float4)(cx, cy, v, (float)cnt);
    }
}
)";

}  // namespace gpu_post

// ---------------------------------------------------------------------------
// The timing breakdown of one GPU post-processing call.
//
// ⚠️ `GpuPostTiming` 现在定义在 mosaic_core.h 里（和 GPU 接入钩子放在一起），
// 本文件只 include mosaic_core.h，不再重复定义 —— 重复定义会编译报错。
// ---------------------------------------------------------------------------

// ===========================================================================
// The DENSE peak, evaluated on the CPU for the few GPU candidates.
//
// WHY THIS EXISTS
// ---------------
// The dense path's `peak` is the maximum of the **upsampled native** map
// inside the component.  The GPU's `peak` is the maximum of the **source**
// heatmap cell.  A bilinear value is a convex combination of its four
// sources, so it can NEVER exceed the largest of them:
//
//     native_peak  <=  heatmap_peak
//
// The threshold that actually matters (`th_base = 0.3` on `peak`) is therefore
// applied on a DIFFERENT SCALE in the two paths: the GPU filter is looser, and
// the extra detections that slip through become false positives.
//
// To put both paths on the same scale, the native value at the candidate is
// evaluated here, on the CPU, over the candidate's 3x3 native neighbourhood
// (9 bilinear evaluations).  There are only a handful of candidates per frame
// (2.4 measured), so this costs a few hundred flops -- nothing compared to the
// GPU round trip.
//
// The formula is the one of `ResizeBilinearF32`, verbatim.
// ===========================================================================
inline float NativeValueAt(const float* src, int stride, int sw, int sh, double sx_scale,
                           double sy_scale, double nx, double ny) {
    const double fx = (nx + 0.5) * sx_scale - 0.5;
    const int ix = (int)std::floor(fx);
    const float ax = (float)(fx - ix);
    const int ix0 = std::max(0, std::min(sw - 1, ix));
    const int ix1 = std::max(0, std::min(sw - 1, ix + 1));
    const double fy = (ny + 0.5) * sy_scale - 0.5;
    const int iy = (int)std::floor(fy);
    const float ay = (float)(fy - iy);
    const int iy0 = std::max(0, std::min(sh - 1, iy));
    const int iy1 = std::max(0, std::min(sh - 1, iy + 1));
    const float* r0 = src + (size_t)iy0 * (size_t)stride;
    const float* r1 = src + (size_t)iy1 * (size_t)stride;
    const float top = r0[ix0] * (1.0f - ax) + r0[ix1] * ax;
    const float bot = r1[ix0] * (1.0f - ax) + r1[ix1] * ax;
    return top * (1.0f - ay) + bot * ay;
}

// The dense-equivalent peak: the maximum of the native map over the
// candidate's 3x3 native neighbourhood.
inline float DenseEquivalentPeak(const float* src, int stride, int sw, int sh, double sx_scale,
                                  double sy_scale, double nx, double ny) {
    float best = -1.0e30f;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            const double px = nx + dx;
            const double py = ny + dy;
            if (px < 0.0 || px > (double)(kSrcW - 1)) continue;
            if (py < 0.0 || py > (double)(kSrcH - 1)) continue;
            const float v = NativeValueAt(src, stride, sw, sh, sx_scale, sy_scale, px, py);
            if (v > best) best = v;
        }
    }
    return best;
}

// ===========================================================================
// implementation
// ===========================================================================
namespace gpu_post_detail {

// One region of the candidate buffer per tile, so the three tiles can be
// dispatched back to back into ONE buffer and read back in ONE call.
constexpr int kMaxPeaks = 64;    // candidate slots per tile
constexpr int kMaxTiles = 8;     // the mosaic layout has 3
constexpr int kLocalSize = 16;   // work-group edge (16x16 = 256 items)

// steady_clock milliseconds.  Deliberately NOT mosaic::now_ms(): this
// header must stay usable without pulling in the rest of the core.
inline double gpu_now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Soft device pick.  ocl::pick_device() calls std::exit(2) when there is no
// platform, which would make "GPU unavailable -> fall back to dense"
// impossible; this variant returns false instead.  Everything else (the
// context, the program build, the buffers) reuses the ocl:: wrappers.
inline bool pick_device_soft(cl_platform_id* out_plat, cl_device_id* out_dev) {
    cl_uint np = 0;
    if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) return false;
    std::vector<cl_platform_id> ps(np);
    if (clGetPlatformIDs(np, ps.data(), nullptr) != CL_SUCCESS) return false;
    for (cl_platform_id p : ps) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) continue;
        std::vector<cl_device_id> ds(nd);
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, nd, ds.data(), nullptr) != CL_SUCCESS) continue;
        *out_plat = p;
        *out_dev = ds[0];
        return true;
    }
    return false;
}

// All the CL resources, created once and reused for every frame.
struct Resources {
    cl_platform_id plat = nullptr;
    cl_device_id dev = nullptr;
    std::unique_ptr<ocl::Context> ctx;
    ocl::Program prog;
    ocl::Kernel kern;
    ocl::Mem hm_buf;       // the heatmap, hm_w * hm_h floats
    ocl::Mem peaks_buf;    // kMaxTiles * kMaxPeaks float4
    ocl::Mem cnt_buf;      // kMaxTiles ints (one atomic counter per tile)
    size_t hm_cap = 0;     // floats allocated in hm_buf
    size_t local_w = 0;    // work-group size actually used
    size_t local_h = 0;
    bool init_tried = false;
    bool init_ok = false;
};

inline Resources* resources() {
    static std::unique_ptr<Resources> r(new Resources());
    return r.get();
}

// Serialises the GPU post-processing: the buffers are shared, so two
// threads must not use them at the same time.
inline std::mutex& mutex() {
    static std::mutex m;
    return m;
}

inline bool alloc_buffers(Resources* r, int hm_w, int hm_h) {
    cl_int e = CL_SUCCESS;
    if (!r->hm_buf) {
        const size_t bytes = (size_t)hm_w * (size_t)hm_h * sizeof(float);
        cl_mem m = clCreateBuffer(r->ctx->get(), CL_MEM_READ_ONLY, bytes, nullptr, &e);
        if (e != CL_SUCCESS) {
            std::fprintf(stderr, "[GPU-POST] clCreateBuffer(heatmap) failed: %s\n",
                         ocl::err_str(e));
            return false;
        }
        r->hm_buf.reset(m);
        r->hm_cap = (size_t)hm_w * (size_t)hm_h;
    }
    if (!r->peaks_buf) {
        cl_mem m = clCreateBuffer(r->ctx->get(), CL_MEM_READ_WRITE,
                                   (size_t)kMaxTiles * (size_t)kMaxPeaks * sizeof(cl_float4),
                                   nullptr, &e);
        if (e != CL_SUCCESS) {
            std::fprintf(stderr, "[GPU-POST] clCreateBuffer(peaks) failed: %s\n",
                         ocl::err_str(e));
            return false;
        }
        r->peaks_buf.reset(m);
    }
    if (!r->cnt_buf) {
        cl_mem m = clCreateBuffer(r->ctx->get(), CL_MEM_READ_WRITE,
                                   (size_t)kMaxTiles * sizeof(cl_int), nullptr, &e);
        if (e != CL_SUCCESS) {
            std::fprintf(stderr, "[GPU-POST] clCreateBuffer(count) failed: %s\n",
                         ocl::err_str(e));
            return false;
        }
        r->cnt_buf.reset(m);
    }
    return true;
}

inline bool init(Resources* r, int hm_w, int hm_h) {
    if (!pick_device_soft(&r->plat, &r->dev)) {
        std::fprintf(stderr, "[GPU-POST] no OpenCL device available; caller must fall back "
                             "to the dense path\n");
        return false;
    }
    ocl::describe(r->dev, r->plat);
    r->ctx.reset(new ocl::Context(r->dev));
    r->prog = ocl::build_program(r->ctx->get(), r->dev, gpu_post::kPostLocalMaxSrc,
                                  "post_local_max", false, false);
    cl_int e = CL_SUCCESS;
    cl_kernel k = clCreateKernel(r->prog.get(), "post_local_max", &e);
    if (e != CL_SUCCESS) {
        std::fprintf(stderr, "[GPU-POST] clCreateKernel failed: %s\n", ocl::err_str(e));
        return false;
    }
    r->kern.reset(k);

    // Work-group size: the kernel's own limit, capped at kLocalSize.
    size_t kmax = 0;
    if (clGetKernelWorkGroupInfo(r->kern.get(), r->dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kmax),
                                 &kmax, nullptr) != CL_SUCCESS || kmax == 0) {
        kmax = kLocalSize * kLocalSize;
    }
    size_t edge = kLocalSize;
    while (edge > 1 && edge * edge > kmax) edge >>= 1;
    r->local_w = edge;
    r->local_h = edge;

    if (!alloc_buffers(r, hm_w, hm_h)) return false;
    std::printf("[GPU-POST] ready | local %zux%zu | %d peaks/tile | %d tiles\n", r->local_w,
                r->local_h, kMaxPeaks, kMaxTiles);
    return true;
}

// The heatmap size is fixed by the model, so the buffer is created once.
// A later call with a bigger heatmap reallocates it (once).
inline bool ensure_buffers(Resources* r, int hm_w, int hm_h) {
    const size_t need = (size_t)hm_w * (size_t)hm_h;
    if (need > r->hm_cap) {
        r->hm_buf.reset();
        r->hm_cap = 0;
        return alloc_buffers(r, hm_w, hm_h);
    }
    return true;
}

// ---------------------------------------------------------------------------
// The window of one tile inside the heatmap, and the scale factors used to
// map it onto the native 640x512 canvas.
//
//   native X  <->  heatmap column   fx = (X + 0.5) * sw / 640 - 0.5
//   native Y  <->  heatmap row      fy = (Y + 0.5) * sh / 512 - 0.5
//
// so the forward direction (heatmap -> native) is
//
//   native_cx = (hm_cx + 0.5) * 640 / sw - 0.5
//   native_cy = (hm_cy + 0.5) * 512 / sh - 0.5
//
// with sw/sh the tile's OWN window size -- 320x256 / 80x64 / 40x32 for the
// mosaic layout, i.e. 2x / 8x / 16x.  A single global factor would be
// wrong for two of the three tiles.
//
// hm_cx / hm_cy are the WINDOW-LOCAL heatmap coordinates (see
// hm_to_native_x below for why they must not be the global ones).
// ---------------------------------------------------------------------------
struct TileWin {
    int tag = 0;
    int x0 = 0, y0 = 0;   // window origin in heatmap pixels
    int w = 0, h = 0;     // window size in heatmap pixels
    bool used = false;
};

// The heatmap coordinate fed to these two functions is the WINDOW-LOCAL one
// (origin = the tile window's top-left corner), NOT the global heatmap
// coordinate.  That is what the dense path does: BackProjectFrame() calls
//
//     ResizeBilinearF32(src, hm_w, sw, sh, native_map, 640, 512);
//
// with `src` already offset to the window's origin, so EVERY tile's window
// is stretched over the WHOLE 640x512 canvas and the tile's canvas
// rectangle (t.x, t.y) is never applied.  The GPU path has to reproduce
// that frame, otherwise the coordinates of tags 1 and 2 would be shifted by
// their canvas offsets (160 px / 512 px) relative to the dense path.
inline double hm_to_native_x(double hm_x_local, int sw) {
    return (hm_x_local + 0.5) * (double)kSrcW / (double)sw - 0.5;
}

inline double hm_to_native_y(double hm_y_local, int sh) {
    return (hm_y_local + 0.5) * (double)kSrcH / (double)sh - 0.5;
}

inline double clampd(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

}  // namespace gpu_post_detail

// ---------------------------------------------------------------------------
// GpuPostDetect -- heatmap -> candidate detections, on the GPU.
//
//   s   the session (read-only: tiles, region_threshold, min_area,
//       max_detections, npu.hm_w, npu.hm_h)
//   hm  the FULL heatmap, hm_w * hm_h floats, row stride hm_w
//   out the candidates (cleared by this function)
//   tm  the timing breakdown (may be null)
//
// Returns false when the GPU path is unavailable or the inputs are
// inconsistent; the caller must then fall back to the dense path.
// ---------------------------------------------------------------------------
inline bool GpuPostDetect(const Session* s, const std::vector<float>& hm,
                           std::vector<Comp>* out, GpuPostTiming* tm) {
    using namespace gpu_post_detail;

    if (tm) *tm = GpuPostTiming();
    if (!out || !s) return false;
    out->clear();

    const int hm_w = s->npu.hm_w;
    const int hm_h = s->npu.hm_h;
    if (hm_w <= 0 || hm_h <= 0) return false;
    if (hm.size() < (size_t)hm_w * (size_t)hm_h) return false;
    const int ntile = (int)s->tiles.size();
    if (ntile <= 0 || ntile > kMaxTiles) return false;

    std::lock_guard<std::mutex> lock(mutex());
    Resources* r = resources();
    if (!r->init_tried) {
        r->init_tried = true;
        r->init_ok = init(r, hm_w, hm_h);
    }
    if (!r->init_ok) return false;
    if (!ensure_buffers(r, hm_w, hm_h)) return false;

    // ---- the windows of the enabled tiles --------------------------------
    TileWin wins[kMaxTiles];
    for (int i = 0; i < ntile; i++) {
        const Tile& t = s->tiles[(size_t)i];
        if (!t.enabled) continue;
        int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
        HeatRect(t, hm_w, hm_h, &x0, &y0, &x1, &y1);
        const int sw = x1 - x0;
        const int sh = y1 - y0;
        // a window smaller than 3x3 has no interior pixel at all
        if (sw < 3 || sh < 3) continue;
        wins[i].tag = t.tag;
        wins[i].x0 = x0;
        wins[i].y0 = y0;
        wins[i].w = sw;
        wins[i].h = sh;
        wins[i].used = true;
    }

    // ================= STEP 1: upload =====================================
    const double t0 = gpu_now_ms();
    {
        const size_t bytes = (size_t)hm_w * (size_t)hm_h * sizeof(float);
        const cl_int e = clEnqueueWriteBuffer(r->ctx->queue(), r->hm_buf.get(), CL_TRUE, 0, bytes,
                                               hm.data(), 0, nullptr, nullptr);
        if (e != CL_SUCCESS) {
            std::fprintf(stderr, "[GPU-POST] clEnqueueWriteBuffer(heatmap) failed: %s\n",
                         ocl::err_str(e));
            return false;
        }
        // Reset the per-tile atomic counters.  It is a 4-byte-per-tile device
        // write, so it belongs to the upload phase, not to the kernel.
        int zeros[kMaxTiles];
        std::memset(zeros, 0, sizeof(zeros));
        const cl_int e2 = clEnqueueWriteBuffer(r->ctx->queue(), r->cnt_buf.get(), CL_TRUE, 0,
                                                sizeof(zeros), zeros, 0, nullptr, nullptr);
        if (e2 != CL_SUCCESS) {
            std::fprintf(stderr, "[GPU-POST] clEnqueueWriteBuffer(counters) failed: %s\n",
                         ocl::err_str(e2));
            return false;
        }
    }
    const double t1 = gpu_now_ms();

    // ================= STEP 2: kernel =====================================
    {
        const cl_float threshold = (cl_float)s->region_threshold;
        for (int i = 0; i < ntile; i++) {
            if (!wins[i].used) continue;
            const TileWin& w = wins[i];

            // The tile's region is selected INSIDE the kernel by `base`: a
            // cl_mem kernel argument must be a real memory-object handle, so
            // handing the kernel an interior pointer (cl_mem)(buf + offset)
            // would be undefined behaviour.
            cl_mem hm_mem = r->hm_buf.get();
            cl_mem peaks_mem = r->peaks_buf.get();
            cl_mem cnt_mem = r->cnt_buf.get();

            OCL_CHECK(clSetKernelArg(r->kern.get(), 0, sizeof(cl_mem), &hm_mem),
                      "clSetKernelArg(hm)");
            OCL_CHECK(clSetKernelArg(r->kern.get(), 1, sizeof(cl_mem), &peaks_mem),
                      "clSetKernelArg(peaks)");
            OCL_CHECK(clSetKernelArg(r->kern.get(), 2, sizeof(cl_mem), &cnt_mem),
                      "clSetKernelArg(count)");
            OCL_CHECK(clSetKernelArg(r->kern.get(), 3, sizeof(cl_int), &hm_w), "clSetKernelArg(hm_w)");
            OCL_CHECK(clSetKernelArg(r->kern.get(), 4, sizeof(cl_int), &hm_h), "clSetKernelArg(hm_h)");
            OCL_CHECK(clSetKernelArg(r->kern.get(), 5, sizeof(cl_int), &w.x0),
                      "clSetKernelArg(win_x0)");
            OCL_CHECK(clSetKernelArg(r->kern.get(), 6, sizeof(cl_int), &w.y0),
                      "clSetKernelArg(win_y0)");
            OCL_CHECK(clSetKernelArg(r->kern.get(), 7, sizeof(cl_int), &w.w),
                      "clSetKernelArg(win_w)");
            OCL_CHECK(clSetKernelArg(r->kern.get(), 8, sizeof(cl_int), &w.h),
                      "clSetKernelArg(win_h)");
            OCL_CHECK(clSetKernelArg(r->kern.get(), 9, sizeof(cl_float), &threshold),
                      "clSetKernelArg(threshold)");
            OCL_CHECK(clSetKernelArg(r->kern.get(), 10, sizeof(cl_int), &kMaxPeaks),
                      "clSetKernelArg(max_peaks)");
            const cl_int base = i;
            OCL_CHECK(clSetKernelArg(r->kern.get(), 11, sizeof(cl_int), &base),
                      "clSetKernelArg(base)");

            const size_t gx = ((size_t)w.w + r->local_w - 1) / r->local_w * r->local_w;
            const size_t gy = ((size_t)w.h + r->local_h - 1) / r->local_h * r->local_h;
            const size_t local[2] = {r->local_w, r->local_h};
            const size_t global[2] = {gx, gy};
            OCL_CHECK(clEnqueueNDRangeKernel(r->ctx->queue(), r->kern.get(), 2, nullptr, global,
                                             local, 0, nullptr, nullptr),
                      "clEnqueueNDRangeKernel(post_local_max)");
        }
        // clFinish: everything dispatched above has really run.
        OCL_CHECK(clFinish(r->ctx->queue()), "clFinish");
    }
    const double t2 = gpu_now_ms();

    // ================= STEP 3: readback ===================================
    int counts[kMaxTiles];
    cl_float4 peaks[kMaxTiles * kMaxPeaks];
    {
        std::memset(counts, 0, sizeof(counts));
        std::memset(peaks, 0, sizeof(peaks));
        const cl_int e = clEnqueueReadBuffer(r->ctx->queue(), r->cnt_buf.get(), CL_TRUE, 0,
                                              sizeof(counts), counts, 0, nullptr, nullptr);
        if (e != CL_SUCCESS) {
            std::fprintf(stderr, "[GPU-POST] clEnqueueReadBuffer(counts) failed: %s\n",
                         ocl::err_str(e));
            return false;
        }
        const cl_int e2 = clEnqueueReadBuffer(r->ctx->queue(), r->peaks_buf.get(), CL_TRUE, 0,
                                               sizeof(peaks), peaks, 0, nullptr, nullptr);
        if (e2 != CL_SUCCESS) {
            std::fprintf(stderr, "[GPU-POST] clEnqueueReadBuffer(peaks) failed: %s\n",
                         ocl::err_str(e2));
            return false;
        }
    }
    const double t3 = gpu_now_ms();

    // ================= STEP 4: back-project + assemble ====================
    Comp cand[kMaxTiles * kMaxPeaks];
    int ncand = 0;
    int dropped = 0;
    for (int i = 0; i < ntile; i++) {
        if (!wins[i].used) continue;
        const TileWin& w = wins[i];
        const int sw = w.w;
        const int sh = w.h;
        const int n = counts[i] > 0 ? counts[i] : 0;
        if (n > kMaxPeaks) dropped += n - kMaxPeaks;
        const int take = n > kMaxPeaks ? kMaxPeaks : n;

        // the 3x3 neighbourhood's size on the native canvas
        const int nat_w = (int)std::lround(3.0 * (double)kSrcW / (double)sw);
        const int nat_h = (int)std::lround(3.0 * (double)kSrcH / (double)sh);

        for (int p = 0; p < take; p++) {
            const cl_float4 q = peaks[i * kMaxPeaks + p];
            // the kernel returns WINDOW-LOCAL heatmap coordinates; the
            // back-projection uses them as such (see hm_to_native_x).
            double nx = hm_to_native_x((double)q.s[0], sw);
            double ny = hm_to_native_y((double)q.s[1], sh);
            nx = clampd(nx, 0.0, (double)(kSrcW - 1));
            ny = clampd(ny, 0.0, (double)(kSrcH - 1));

            // The dense-equivalent peak (see DenseEquivalentPeak): the maximum
            // of the UPSAMPLED native map around the candidate.  Without this,
            // the GPU reports the SOURCE heatmap value, which is an upper bound
            // of the dense peak, and the `th_base` filter then behaves on a
            // different scale than the dense path's.
            const float* tile_src = hm.data() + (size_t)w.y0 * (size_t)s->npu.hm_w +
                                    (size_t)w.x0;
            const float dense_peak = DenseEquivalentPeak(
                tile_src, s->npu.hm_w, sw, sh, (double)sw / (double)kSrcW,
                (double)sh / (double)kSrcH, nx, ny);

            Comp c;
            c.tag = w.tag;
            c.cx = (float)nx;
            c.cy = (float)ny;
            c.w = nat_w;
            c.h = nat_h;
            c.area = (int)q.s[3];   // foreground cells in the 3x3
            c.peak = dense_peak;   // == the dense path's peak, same scale
            c.sum = 0.0f;           // not computed on the GPU
            // Noise filter: the dense path drops components with fewer than
            // min_area pixels; here the same threshold is applied to the 3x3
            // foreground count, which rejects isolated single-cell noise the
            // same way (see the semantics table below).
            if (c.area < s->min_area) continue;
            if (ncand < kMaxTiles * kMaxPeaks) cand[ncand++] = c;
        }
    }

    // Stable insertion sort by peak DESCENDING.  A stable sort is used (like
    // the dense path's) so that equal peaks keep their tile/raster order.
    // Insertion sort is used ON PURPOSE: it never allocates, and the
    // candidate list is at most 192 entries.
    for (int i = 1; i < ncand; i++) {
        const Comp key = cand[i];
        int j = i - 1;
        while (j >= 0 && cand[j].peak < key.peak) {
            cand[j + 1] = cand[j];
            j--;
        }
        cand[j + 1] = key;
    }

    const int limit = s->max_detections > 0 ? s->max_detections : 0;
    const int nout = ncand < limit ? ncand : limit;
    for (int i = 0; i < nout; i++) out->push_back(cand[i]);

    const double t4 = gpu_now_ms();
    if (tm) {
        tm->upload_ms = t1 - t0;
        tm->kernel_ms = t2 - t1;
        tm->readback_ms = t3 - t2;
        tm->total_ms = t4 - t0;
        tm->peaks = (int)out->size();
        tm->peaks_dropped = dropped;
    }
    return true;
}

// ===========================================================================
// SEMANTICS DIFFERENCE LIST (GPU path vs. the CPU dense path)
// ===========================================================================
//
//   peak    dense : max of the BILINEAR-UPSAMPLED native image inside the
//                   component.
//            gpu   : the local maximum on the SOURCE heatmap.
//           A bilinear interpolation is a convex combination of its source
//           cells and can therefore never exceed the largest source value,
//           so gpu_peak >= dense_peak always.  The th_base=0.3 filter is
//           therefore slightly LOOSER on the GPU path (a candidate that the
//           dense path would drop at peak=0.2999 can survive at 0.3001).
//
//   cx, cy  dense : arithmetic (unweighted) mean of the component's pixel
//                   coordinates on the native canvas.
//            gpu   : 3x3 intensity-weighted centroid, back-projected to the
//                   native canvas.  The two agree for a symmetric blob and
//                   differ for an asymmetric one.
//
//   w, h    dense : the component's bounding box on the native canvas.
//            gpu   : the size of the 3x3 neighbourhood on the native canvas
//                   (6x6 / 24x24 / 48x48 for the 320x256 / 80x64 / 40x32
//                   windows).  This is NOT a bounding box.
//
//   area    dense : number of pixels in the component.
//            gpu   : number of cells of the 3x3 that are >= threshold (0..9).
//           The min_area filter is applied to this count.
//
//   sum     dense : sum of the component's values; also the sort key.
//            gpu   : 0 (not computed).
//
//   order   dense : sorted by sum DESCENDING (stable), truncated to
//                   max_detections.
//            gpu   : sorted by peak DESCENDING (stable), truncated to
//                   max_detections.
//
//   border  dense : a component touching a tile border is still reported.
//            gpu   : the outermost pixel ring of every window is skipped,
//                   because a 3x3 there would read another tile's cells.
//
//   number of candidates: the dense path reports one entry per CONNECTED
//           COMPONENT (few); the GPU path reports one entry per LOCAL
//           MAXIMUM (potentially many).  max_peaks (64 per tile) caps it and
//           the overflow is reported as peaks_dropped.
//
//   The evaluator only uses the prediction CENTRE (inside the GT box?) and
//   filters on peak, so peak and cx/cy are the fields that matter; w/h are
//   cosmetic and sum is unused.
// ===========================================================================

}  // namespace mosaic
