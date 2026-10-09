/* ============================================================================
 * warp_median_fused.cl -- single-pass fused GMC warp + median + 3-channel pack
 *
 *   Ch0 = I_t
 *   Ch1 = |I_t - W(I_{t-2})|
 *   Ch2 = max(0, I_t - B_t),  B_t = median{W(I_{t-2k})}, k = 1..21
 *
 * ONE work-item owns ONE output pixel and does all of it. The 21 warped samples
 * live in private scalars for the whole kernel; no intermediate warped image is
 * ever written to global memory. That is the entire point of the fusion -- the
 * CPU path materialises 21 full-frame Mats per output frame, and those
 * intermediate write-backs plus 22 separate remaps are what dominate it.
 *
 * ----------------------------------------------------------------------------
 * RK3588 / Mali-G610 CONSTRAINTS THIS FILE IS SHAPED AROUND
 * ----------------------------------------------------------------------------
 * 1. The 21 samples are 21 DISTINCT NAMED SCALARS (v0..v20), never `v[k]`.
 *    Under a loop index the compiler must assume the index is dynamic and spills
 *    the array to private memory, which Mali emulates in *global* memory. Named
 *    scalars make every index a compile-time constant so the values stay in
 *    registers -- which is also why the sorting network is spliced in as
 *    `CAS(v3, v11)` rather than as a table walk.
 * 2. float only. No double in the kernel: Mali-G610 has no usable FP64. The
 *    affine inversion happens on the host in double, matching OpenCV.
 * 3. No local memory, no barriers, no atomics -- one work-item, one pixel.
 * 4. Global size (width, height), NULL local size. The work-group shape is never
 *    hard-coded; the driver owns it.
 * 5. No printf, no 64-bit integer arithmetic, no dynamic allocation.
 * 6. History frames are 22 separate image2d_t args rather than one
 *    image2d_array_t, so the ring rotates by pointer and only the one new frame
 *    crosses the bus each step. An array layout would force all 21 slices to be
 *    rewritten per output frame (5.5 MB/frame instead of 256 KB at 512x512),
 *    which on RK3588 costs more than the arithmetic this kernel saves.
 * ==========================================================================*/

#define WINDOW   21
#define MEDIAN_I 10              /* odd window -> pure selection, exact anywhere */

/* FUSED_USE_HW_LINEAR
 *   1 = quantise the coordinate here, then let a CLK_FILTER_LINEAR sampler do the
 *       2x2 blend. This is the default and matches the task spec's sampler.
 *   0 = sample with CLK_FILTER_NEAREST at integer coordinates and blend by hand.
 *       Slower, but it removes all dependence on device filter precision: OpenCL
 *       leaves CLK_FILTER_LINEAR accuracy implementation-defined, and a driver
 *       that rounds the filtered result to the 8-bit storage precision injects
 *       +/-1 errors that a MAE < 0.05 gate cannot absorb. The host flag
 *       --hw-linear 0/1 selects between them so the choice is made by measurement
 *       on the real device rather than by assumption. Both paths use identical
 *       arithmetic on identical, already-quantised weights.
 */
#ifndef FUSED_USE_HW_LINEAR
#define FUSED_USE_HW_LINEAR 1
#endif

/* Diagnostic switch, see the FP64 branch in sample_at. Defaults off. */
#ifndef FUSED_FP64_COORD
#define FUSED_FP64_COORD 0
#endif

#if FUSED_FP64_COORD
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#if FUSED_USE_HW_LINEAR
#define SAMPLER_FLAGS (CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR)
#else
#define SAMPLER_FLAGS (CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST)
#endif

/* Branchless compare-and-swap on two uchars. Ascending, no predicates, no
 * divergence: every pixel executes the same min/max sequence. */
#define CAS(a, b) { uchar mn_ = min((a), (b)); uchar mx_ = max((a), (b)); (a) = mn_; (b) = mx_; }

/* cv::warpAffine's border model, settled by measurement rather than by reading
 * OpenCV's source (tools/probe_border.cpp, 96000 out-of-frame tap samples on
 * noise):
 *
 *   A  reflect the base, then take base and base+1   -> 29.0% of taps WRONG
 *   B  reflect each of the two taps independently   ->  0.0% of taps wrong
 *
 * So B is correct, and B is precisely what a host-side
 * cv::copyMakeBorder(..., BORDER_REFLECT) pad provides for free: sampling the
 * padded image at (sx + pad) yields src[reflect(floor(sx))] and
 * src[reflect(floor(sx)+1)]. An earlier version of this kernel instead reflected
 * the base coordinate in the kernel (model A) and measured Ch1 Max|Diff| = 239
 * on 640-wide sequences. Do not "optimise" this back into the kernel without
 * re-running probe_border.
 *
 * Consequence for the host: the pad must cover the largest tap excursion, which
 * for a rotating warp reaches ~50 px. See required_pad() in the host.
 */

/* One warped sample from the padded frame `img` with the inverse affine row pair
 * m[0..2], m[3..5]. Two implementations, chosen by FUSED_FP64_COORD; see the note
 * there. Everything else about them is identical. */
#if FUSED_FP64_COORD

/* DIAGNOSTIC ONLY -- never enable on RK3588.
 *
 * Mali-G610 has no usable FP64, so this exists purely to answer one question by
 * measurement: is the residual Max|Diff| = 2 (on ~6 pixels in 5.5M) caused by our
 * float32 coordinate losing the tie against OpenCV's own 1/32 quantisation, or by
 * a genuinely wrong model? If the max drops to <= 1 the cause is coordinate
 * precision and the only fix on the target is double-float arithmetic at real ALU
 * cost. If it does not drop, the model is wrong and no precision will save it. */
static inline float sample_at(__constant const float *m, int x, int y, image2d_t img, sampler_t s,
                               int pad, int sp_w, int sp_h) {
    const double sx = (double)m[0] * (double)x + (double)m[1] * (double)y + (double)m[2];
    const double sy = (double)m[3] * (double)x + (double)m[4] * (double)y + (double)m[5];
    const double fx = floor(sx), fy = floor(sy);
    const double qx = floor((sx - fx) * 32.0 + 0.5) * (1.0 / 32.0);
    const double qy = floor((sy - fy) * 32.0 + 0.5) * (1.0 / 32.0);
    const float2 sp = (float2)((float)(fx + qx) + (float)pad,
                               (float)(fy + qy) + (float)pad);
    const int x0 = (int)floor(sp.x);
    const int y0 = (int)floor(sp.y);
    const float ax = sp.x - (float)x0;
    const float ay = sp.y - (float)y0;
    const float p00 = read_imagef(img, s, (float2)((float)x0,        (float)y0)).x;
    const float p10 = read_imagef(img, s, (float2)((float)x0 + 1.0f, (float)y0)).x;
    const float p01 = read_imagef(img, s, (float2)((float)x0,        (float)y0 + 1.0f)).x;
    const float p11 = read_imagef(img, s, (float2)((float)x0 + 1.0f, (float)y0 + 1.0f)).x;
    return (1.0f - ax) * (1.0f - ay) * p00 + ax * (1.0f - ay) * p10 +
           (1.0f - ax) * ay * p01 + ax * ay * p11;
}

#else

static inline float sample_at(__constant const float *m, int x, int y, image2d_t img, sampler_t s,
                              int pad, int sp_w, int sp_h) {
    /* src = M^-1 * [x, y, 1]. No half-pixel offset -- measured against OpenCV,
     * not assumed: adding one costs MAE 39 instead of ~0.01. */
    float sx = m[0] * (float)x + m[1] * (float)y + m[2];
    float sy = m[3] * (float)x + m[4] * (float)y + m[5];

    /* OpenCV's warpPerspective quantises the interpolation POSITION to
     * INTER_BITS = 5 and reads back a 33-level staircase, rounded half-up.
     * Reproduced because one 1/32 cell is ~8 grey levels on a noisy frame --
     * far past the Max|Diff| <= 1 gate -- and because the rounding is
     * systematically downward, so a median of 21 would carry the bias straight
     * into Ch2 as an MAE the gate cannot absorb. */
    const float fx = floor(sx);
    const float fy = floor(sy);
    const float qx = floor((sx - fx) * 32.0f + 0.5f) * (1.0f / 32.0f);
    const float qy = floor((sy - fy) * 32.0f + 0.5f) * (1.0f / 32.0f);
    /* Shift into the host-padded frame. The border transform is baked into the
     * pad by cv::copyMakeBorder(BORDER_REFLECT) -- see the note above. */
    const float2 sp = (float2)(fx + qx + (float)pad, fy + qy + (float)pad);

#if FUSED_USE_HW_LINEAR
    return read_imagef(img, s, sp).x;
#else
    const int x0 = (int)floor(sp.x);
    const int y0 = (int)floor(sp.y);
    const float ax = sp.x - (float)x0;
    const float ay = sp.y - (float)y0;
    const float p00 = read_imagef(img, s, (float2)((float)x0,        (float)y0)).x;
    const float p10 = read_imagef(img, s, (float2)((float)x0 + 1.0f, (float)y0)).x;
    const float p01 = read_imagef(img, s, (float2)((float)x0,        (float)y0 + 1.0f)).x;
    const float p11 = read_imagef(img, s, (float2)((float)x0 + 1.0f, (float)y0 + 1.0f)).x;
    return (1.0f - ax) * (1.0f - ay) * p00 + ax * (1.0f - ay) * p10 +
           (1.0f - ax) * ay * p01 + ax * ay * p11;
#endif
}

#endif

__kernel void warp_median_fused(
    image2d_t hist0, image2d_t hist1,  image2d_t hist2,
    image2d_t hist3, image2d_t hist4,  image2d_t hist5,
    image2d_t hist6, image2d_t hist7,  image2d_t hist8,
    image2d_t hist9, image2d_t hist10, image2d_t hist11,
    image2d_t hist12,image2d_t hist13,image2d_t hist14,
    image2d_t hist15,image2d_t hist16,image2d_t hist17,
    image2d_t hist18,image2d_t hist19,image2d_t hist20,
    image2d_t cur,               /* padded I_t                                   */
    __constant const float *mats,          /* WINDOW*6 inverse affine coefficients, float32  */
    __global uchar *out,                   /* 3 * H * W, plane-major [Ch0, Ch1, Ch2]         */
    const int W, const int H, const int pad, sampler_t samp)
{
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= W || y >= H) return;

    /* 21 named scalars, not an array. See the RK3588 note at the top. */
    uchar v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14,
          v15, v16, v17, v18, v19, v20;

#define SAMPLE(K, IMG, DST) \
    DST = (uchar)clamp((int)(sample_at(mats + 6 * (K), x, y, (IMG), samp, pad, W, H) \
                                   * 255.0f + 0.5f), 0, 255);

    SAMPLE( 0, hist0,  v0);   SAMPLE( 1, hist1,  v1);   SAMPLE( 2, hist2,  v2);
    SAMPLE( 3, hist3,  v3);   SAMPLE( 4, hist4,  v4);   SAMPLE( 5, hist5,  v5);
    SAMPLE( 6, hist6,  v6);   SAMPLE( 7, hist7,  v7);   SAMPLE( 8, hist8,  v8);
    SAMPLE( 9, hist9,  v9);   SAMPLE(10, hist10, v10);  SAMPLE(11, hist11, v11);
    SAMPLE(12, hist12, v12);  SAMPLE(13, hist13, v13);  SAMPLE(14, hist14, v14);
    SAMPLE(15, hist15, v15);  SAMPLE(16, hist16, v16);  SAMPLE(17, hist17, v17);
    SAMPLE(18, hist18, v18);  SAMPLE(19, hist19, v19);  SAMPLE(20, hist20, v20);

    /* Ch1 reuses the k=1 sample. The CPU baseline computes
     *     ch1      = absdiff(frame, warp(frame_at_lag(2), mats[2]))
     *     history0 =           warp(frame_at_lag(2), mats[2])
     * -- the SAME warp, so latching v0 before the network permutes the registers
     * is exact and saves an entire remap. */
    /* Ch0 is an integer-coordinate read of the interior, so no border transform
     * applies. It is the pipeline's self-check: if padding, channel order, ring
     * indexing or the coordinate convention were wrong, this would not be
     * bit-exact even though Ch1 and Ch2 might still look plausible. */
    const uchar cur8 = (uchar)clamp(
        (int)(read_imagef(cur, samp, (float2)((float)(x + pad), (float)(y + pad))).x * 255.0f + 0.5f),
        0, 255);
    const uchar ch1 = (uchar)(cur8 > v0 ? cur8 - v0 : v0 - cur8);

    /* ---- proof-preserving 21-element sorting network, spliced at build time ----
     * PROVEN correct on all 2^21 = 2,097,152 zero-one inputs by exhaustive
     * enumeration -- see tools/sortnet_verify.c. Fully unrolled and branchless:
     * every pixel executes the identical CAS sequence. */
    /*__SORTNET__*/

    const uchar bg  = v10;                        /* the median */
    const int   d   = (int)cur8 - (int)bg;
    const uchar ch2 = (uchar)(d > 0 ? d : 0);

    const int plane = W * H;
    const int idx   = y * W + x;
    out[0 * plane + idx] = cur8;
    out[1 * plane + idx] = ch1;
    out[2 * plane + idx] = ch2;
}