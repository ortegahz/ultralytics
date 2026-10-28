/* ============================================================================
 * gmc_feature_gpu.cl -- GPU port of the GMC "feature preparation" stage.
 *
 *   K1  gftt_min_eigen   Shi-Tomasi min-eigenvalue score map, 1 work-item per
 *                        pixel.  Replicates cv::cornerMinEigenVal(img, eig,
 *                        blockSize=3, ksize=3, BORDER_DEFAULT) -- i.e. exactly
 *                        what cv::goodFeaturesToTrack uses when
 *                        useHarrisDetector=false.
 *
 *   K2  lk_pyr_track     Pyramidal Lucas-Kanade, 1 work-item per corner.
 *                        Replicates the arithmetic of cv::calcOpticalFlowPyrLK
 *                        (lkpyramid.cpp, LKTrackerInvoker::operator()) for
 *                        winSize=(15,15), maxLevel=2 and the DEFAULT TermCriteria
 *                        (COUNT|EPS, 30, 0.01) -- which is what the shipped GMC
 *                        path actually uses (fit_iters == 20 selects the
 *                        no-criteria overload in gmc_stream_ocl.cpp).
 *
 * ---------------------------------------------------------------------------
 * WHY THE PYRAMID IS BUILT ON THE CPU
 * ---------------------------------------------------------------------------
 * The probe uploads the pyramid levels (image + Scharr derivative planes) that
 * cv::buildOpticalFlowPyramid already produced on the host.  Measured on the
 * board (manu/memory/gmc_fit_cache_rk3588.md 1.2) the pyramid build is 3.2% of
 * a fit, so it is not worth re-implementing pyrDown + Scharr on the GPU: the
 * GPU only does the ITERATION, which is the part that parallelises over
 * corners.  The padded planes are uploaded verbatim, so the border model is
 * OpenCV's, not ours:
 *
 *   * image levels  : padded with BORDER_REFLECT_101 by buildOpticalFlowPyramid
 *   * deriv  levels : padded with BORDER_CONSTANT (derivBorder default)
 *   * pad width     : winSize.width == winSize.height == 15 (NOT 7)
 *
 * The kernel therefore samples the PADDED planes at (roi_x + 15, roi_y + 15)
 * and reproduces OpenCV's out-of-ROI reads exactly: OpenCV's level planes are
 * ROI views into a padded buffer, so a patch that hangs over the ROI edge
 * reads the border ring, not garbage.
 *
 * ---------------------------------------------------------------------------
 * ARITHMETIC NOTES (the part that is easy to get wrong)
 * ---------------------------------------------------------------------------
 * 1. OpenCV's LK keeps the previous patch and its derivatives in a private
 *    buffer and re-uses them for all iterations of a level.  We do the same,
 *    but in a GLOBAL scratch buffer laid out TRANSPOSED:
 *
 *        patch[c][p][gid]      c in {0:prev value, 1:Ix, 2:Iy}, p in [0,win*win)
 *
 *    Transposed so that the lanes of a work-group touch consecutive
 *    addresses (gid is the fastest index) => coalesced.  The alternative
 *    (per-lane contiguous patch) makes every lane pull its own cache line.
 *
 * 2. OpenCV's LK uses 14-bit fixed point for the bilinear weights and keeps
 *    the previous patch in 32x units (ival = DESCALE(sum, 9)).  The scale
 *    factors CANCEL in the delta: A ~ k^2, b ~ k^2, D ~ k^4.  We therefore use
 *    exact float weights and keep the OpenCV scale convention explicitly:
 *
 *        diff = 32 * (curr - prev)      <- matches OpenCV's 32x ival
 *        A11  = sum Ix^2                <- raw Scharr planes (32x true deriv)
 *        b1   = sum diff * Ix
 *        delta = (A12*b2 - A22*b1) / (A11*A22 - A12^2)
 *
 *    With diff scaled by 32 the result is the true displacement; without it
 *    every delta would be 32x too small.
 *
 * 3. The min-eigenvalue rejection test MUST use OpenCV's scale
 *    (A multiplied by FLT_SCALE = 2^-20) because minEigThreshold is 1e-4 in
 *    that scale.  We compute both: the scaled values only for the two
 *    rejection tests, the unscaled ones for the solve.
 *
 * 4. Point scaling between levels: prevPt = p0 / 2^level (the ORIGINAL point,
 *    re-scaled every level, not the running estimate).  The running estimate
 *    is carried as nextPts and DOUBLED when going one level finer.
 *
 * 5. The Sobel normalisation: OpenCV's cornerEigenValsVecs calls
 *    Sobel(src, dx, CV_32F, 1, 0, 3, scale=1/3060, 0, BORDER_DEFAULT) with
 *    scale = 1/((1<<(ksize-1)) * blockSize * 255) for a CV_8U source, and
 *    OpenCV's 3x3 Sobel kernel is [-1,0,1] x [1,2,1] with the scale applied
 *    to the smoothing kernel.  The structure tensor is then box-filtered over
 *    3x3 with BORDER_REFLECT_101 and the MIN eigenvalue is taken.
 *
 * No FP64 (Mali-G610 has none), no local memory, no barriers, no atomics,
 * no dynamic indexing into private arrays, no printf.
 * ==========================================================================*/

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

/* OpenCV's BORDER_REFLECT_101 index map:  ... 2 1 | 0 1 2 3 | 2 1  ...
 * i.e. -1 -> 0, -2 -> 1, n -> n-2, n+1 -> n-3. */
static inline int reflect101(int i, const int n)
{
    if (n <= 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i - 1;
        else       i = 2 * n - 2 - i;
    }
    return i;
}

/* 3x3 Sobel, OpenCV normalisation, BORDER_DEFAULT = REFLECT_101 on the source.
 *
 * OpenCV's cornerEigenValsVecs (modules/imgproc/src/corner.cpp:237) computes
 *     scale = 1 / ((1 << (ksize-1)) * blockSize * 255)   for a CV_8U source
 *            = 1 / (4 * 3 * 255) = 1/3060
 * and calls Sobel(src, dx, CV_32F, 1, 0, 3, scale, 0, BORDER_DEFAULT).
 * OpenCV's 3x3 Sobel kernel is [-1,0,1] x [1,2,1] (no 1/8 normalisation) and
 * the scale is applied to the SMOOTHING kernel.  Getting this constant wrong
 * shifts the whole score map by a constant factor -- the corner SELECTION is
 * invariant to it, but the score map would no longer be comparable with
 * cv::cornerMinEigenVal, so it is pinned here and checked by the probe.
 */
static inline void sobel3(__global const uchar *img, const int W, const int H,
                           const int x, const int y, float *out_dx, float *out_dy)
{
    const float s = 1.0f / 3060.0f;   /* 1 / (4 * 3 * 255) */

    const float m11 = (float)img[reflect101(y - 1, H) * W + reflect101(x - 1, W)];
    const float m12 = (float)img[reflect101(y - 1, H) * W + reflect101(x,     W)];
    const float m13 = (float)img[reflect101(y - 1, H) * W + reflect101(x + 1, W)];
    const float m21 = (float)img[reflect101(y,     H) * W + reflect101(x - 1, W)];
    const float m22 = (float)img[reflect101(y,     H) * W + reflect101(x,     W)];
    const float m23 = (float)img[reflect101(y,     H) * W + reflect101(x + 1, W)];
    const float m31 = (float)img[reflect101(y + 1, H) * W + reflect101(x - 1, W)];
    const float m32 = (float)img[reflect101(y + 1, H) * W + reflect101(x,     W)];
    const float m33 = (float)img[reflect101(y + 1, H) * W + reflect101(x + 1, W)];

    /* d/dx : smooth [1,2,1] along y, differentiate [-1,0,1] along x */
    *out_dx = s * ((m13 + 2.0f * m23 + m33) - (m11 + 2.0f * m21 + m31));
    /* d/dy : smooth [1,2,1] along x, differentiate [-1,0,1] along y */
    *out_dy = s * ((m31 + 2.0f * m32 + m33) - (m11 + 2.0f * m12 + m13));
}

/* ------------------------------------------------------------------ */
/* K1: Shi-Tomasi min-eigenvalue score map                             */
/* ------------------------------------------------------------------ */
/*
 * One work-item per pixel of the (already downscaled) grey image.
 *
 * The 3x3 structure tensor is accumulated over the box window directly, with
 * the gradient evaluated at the REFLECT_101-extended positions.  That is
 * equivalent to OpenCV's three boxFilter passes on (dx*dx, dx*dy, dy*dy)
 * with BORDER_REFLECT_101, because Sobel is linear and the extension is
 * applied to the source image in both cases.
 */
__kernel void gftt_min_eigen(__global const uchar *img,
                             __global float *score,
                             const int W, const int H)
{
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= W || y >= H) return;

    float A = 0.0f;   /* sum dx*dx  */
    float B = 0.0f;   /* sum dx*dy  */
    float C = 0.0f;   /* sum dy*dy  */

    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            float gx, gy;
            sobel3(img, W, H, x + i, y + j, &gx, &gy);
            A += gx * gx;
            B += gx * gy;
            C += gy * gy;
        }
    }

    /* min eigenvalue of [[A, B], [B, C]] */
    const float u = 0.5f * (A + C);
    const float v = sqrt(0.25f * (A - C) * (A - C) + B * B);
    score[y * W + x] = u - v;
}

/* ------------------------------------------------------------------ */
/* K2: pyramidal Lucas-Kanade                                         */
/* ------------------------------------------------------------------ */

/* bilinear sample of a uchar plane; (x, y) are coordinates in the PADDED
 * plane (the caller adds the pad offset), (fx, fy) the fractional part. */
static inline float bilin_u8(__global const uchar *buf, const int off, const int stride,
                             const int x, const int y, const float fx, const float fy)
{
    const float v00 = (float)buf[off + y * stride + x];
    const float v10 = (float)buf[off + y * stride + x + 1];
    const float v01 = (float)buf[off + (y + 1) * stride + x];
    const float v11 = (float)buf[off + (y + 1) * stride + x + 1];
    return (1.0f - fx) * (1.0f - fy) * v00 + fx * (1.0f - fy) * v10 +
           (1.0f - fx) * fy * v01 + fx * fy * v11;
}

/* bilinear sample of the 2-channel int16 Scharr derivative plane. */
static inline float2 bilin_der(__global const short *buf, const int off, const int stride,
                               const int x, const int y, const float fx, const float fy)
{
    const int b00 = off + 2 * (y * stride + x);
    const int b10 = b00 + 2;
    const int b01 = b00 + 2 * stride;
    const int b11 = b01 + 2;

    const float ix =
        (1.0f - fx) * (1.0f - fy) * (float)buf[b00] + fx * (1.0f - fy) * (float)buf[b10] +
        (1.0f - fx) * fy * (float)buf[b01] + fx * fy * (float)buf[b11];
    const float iy =
        (1.0f - fx) * (1.0f - fy) * (float)buf[b00 + 1] + fx * (1.0f - fy) * (float)buf[b10 + 1] +
        (1.0f - fx) * fy * (float)buf[b01 + 1] + fx * fy * (float)buf[b11 + 1];
    return (float2)(ix, iy);
}

/*
 * desc layout, 8 ints per level:
 *   [0] padded width  (stride of the uploaded planes)
 *   [1] padded height
 *   [2] roi width     (the level size, i.e. the size OpenCV checks against)
 *   [3] roi height
 *   [4] byte offset into prev_img
 *   [5] byte offset into prev_der
 *   [6] byte offset into curr_img
 *   [7] unused (keeps the stride a power of two)
 */
__kernel void lk_pyr_track(
    __global const uchar *prev_img,
    __global const short *prev_der,
    __global const uchar *curr_img,
    __global const int   *desc,
    __global float       *patch,      /* scratch, see header */
    __global const float *pts_in,
    __global float       *pts_out,
    __global int         *status_out,
    const int npts,
    const int nlevels,
    const int win,          /* window edge, also the pyramid pad width */
    const int maxit,        /* criteria.maxCount */
    const float eps_sq,     /* criteria.epsilon AFTER OpenCV's epsilon*=epsilon */
    const float min_eig_thr)
{
    const int gid = (int)get_global_id(0);
    if (gid >= npts) return;

    const int npix = win * win;
    const float2 half_win = (float2)((float)(win - 1) * 0.5f, (float)(win - 1) * 0.5f);
    const float2 p0 = (float2)(pts_in[2 * gid], pts_in[2 * gid + 1]);

    float2 next_pts = (float2)(0.0f, 0.0f);
    int st = 1;

    for (int L = nlevels - 1; L >= 0; --L) {
        const int padw   = desc[L * 8 + 0];
        const int roiw   = desc[L * 8 + 2];
        const int roih   = desc[L * 8 + 3];
        const int poff   = desc[L * 8 + 4];
        const int doff   = desc[L * 8 + 5];
        const int coff   = desc[L * 8 + 6];

        /* the ORIGINAL previous point, re-scaled for this level */
        float2 prev_pt = p0 * (1.0f / (float)(1 << L));
        /* the running estimate: the level above is twice as coarse */
        float2 next_pt = (L == nlevels - 1) ? prev_pt : next_pts * 2.0f;
        next_pts = next_pt;                 /* OpenCV: nextPts[ptidx] = nextPt */

        prev_pt -= half_win;                /* -> patch origin */
        const int ipx = (int)floor(prev_pt.x);
        const int ipy = (int)floor(prev_pt.y);

        if (ipx < -win || ipx >= roiw || ipy < -win || ipy >= roih) {
            if (L == 0) st = 0;
            continue;
        }
        const float fx = prev_pt.x - (float)ipx;
        const float fy = prev_pt.y - (float)ipy;

        /* ---- previous patch + derivatives, and the 2x2 normal matrix ---- */
        float A11 = 0.0f, A12 = 0.0f, A22 = 0.0f;
        for (int j = 0; j < win; ++j) {
            const int sy = ipy + j + win;          /* + pad */
            for (int i = 0; i < win; ++i) {
                const int sx = ipx + i + win;       /* + pad */
                const float v  = bilin_u8(prev_img, poff, padw, sx, sy, fx, fy);
                const float2 d = bilin_der(prev_der, doff, padw, sx, sy, fx, fy);
                const int p = j * win + i;
                patch[0 * npix * npts + p * npts + gid] = v;
                patch[1 * npix * npts + p * npts + gid] = d.x;
                patch[2 * npix * npts + p * npts + gid] = d.y;
                A11 += d.x * d.x;
                A12 += d.x * d.y;
                A22 += d.y * d.y;
            }
        }

        /* rejection tests in OpenCV's scale (A carries FLT_SCALE = 2^-20) */
        const float FLT_SCALE = 1.0f / 1048576.0f;
        const float A11s = A11 * FLT_SCALE;
        const float A12s = A12 * FLT_SCALE;
        const float A22s = A22 * FLT_SCALE;
        const float Ds = A11s * A22s - A12s * A12s;
        const float min_eig =
            (A22s + A11s - sqrt((A11s - A22s) * (A11s - A22s) + 4.0f * A12s * A12s)) /
            (2.0f * (float)(win * win));

        if (min_eig < min_eig_thr || Ds < 1.1920929e-07f) {
            if (L == 0) st = 0;
            continue;
        }

        const float D = 1.0f / (A11 * A22 - A12 * A12);

        next_pt -= half_win;               /* OpenCV: nextPt -= halfWin */
        float2 prev_delta = (float2)(0.0f, 0.0f);

        for (int j = 0; j < maxit; ++j) {
            const int inx = (int)floor(next_pt.x);
            const int iny = (int)floor(next_pt.y);

            if (inx < -win || inx >= roiw || iny < -win || iny >= roih) {
                if (L == 0) st = 0;
                break;
            }
            const float gx = next_pt.x - (float)inx;
            const float gy = next_pt.y - (float)iny;

            float ib1 = 0.0f, ib2 = 0.0f;
            for (int q = 0; q < win; ++q) {
                const int sy = iny + q + win;
                for (int p = 0; p < win; ++p) {
                    const int sx = inx + p + win;
                    const int idx = q * win + p;
                    const float cur = bilin_u8(curr_img, coff, padw, sx, sy, gx, gy);
                    /* OpenCV keeps the previous patch at 32x scale: 32*(cur-prev) */
                    const float diff = 32.0f * (cur - patch[0 * npix * npts + idx * npts + gid]);
                    ib1 += diff * patch[1 * npix * npts + idx * npts + gid];
                    ib2 += diff * patch[2 * npix * npts + idx * npts + gid];
                }
            }

            const float2 delta = (float2)((A12 * ib2 - A22 * ib1) * D,
                                          (A12 * ib1 - A11 * ib2) * D);
            next_pt += delta;
            next_pts = next_pt + half_win;

            if (delta.x * delta.x + delta.y * delta.y <= eps_sq) break;
            if (j > 0 && fabs(delta.x + prev_delta.x) < 0.01f &&
                       fabs(delta.y + prev_delta.y) < 0.01f) {
                next_pts -= delta * 0.5f;
                break;
            }
            prev_delta = delta;
        }
    }

    pts_out[2 * gid]     = next_pts.x;
    pts_out[2 * gid + 1] = next_pts.y;
    status_out[gid] = st;
}

/* ------------------------------------------------------------------ */
/* K3: pyramidal Lucas-Kanade, MULTI-FIT (batched over the anchors)     */
/* ------------------------------------------------------------------ */
/*
 * WHY A SECOND LK KERNEL
 * ----------------------
 * A production FRAME is not one fit: it is the 5 anchor fits
 * (lags 2/12/22/32/42, all of them ending on the CURRENT frame) plus the
 * stride-2 step fit (t-4 -> t-2).  Those fits are INDEPENDENT of each
 * other -- they only share the current frame's pyramid -- so on the GPU
 * they can all live in ONE dispatch:
 *
 *     work-items = sum over the fits of that fit's corner count
 *
 * instead of one dispatch per fit.  Two things fall out of that:
 *
 *   * the GPU gets nfits x more work-items, which is what the 4 CUs need
 *     to be busy (a single anchor's ~600 work-items are only a few waves);
 *   * the current frame's pyramid is uploaded ONCE and is read by every
 *     fit, and the fits that share a current frame share its cached
 *     pyramid slot.
 *
 * The arithmetic of every fit is the SAME as lk_pyr_track's: this kernel is
 * a re-indexing of that one, not a re-derivation.  lk_pyr_track is left
 * untouched so the single-anchor measurements stay reproducible.
 *
 * desc layout, 8 ints per (fit, level):
 *   desc[(fit*nlevels + L)*8 + 0] padded width  (stride of the uploaded planes)
 *   ... +1 padded height
 *   ... +2 roi width
 *   ... +3 roi height
 *   ... +4 byte offset into pyr_img of the PREV image level of this fit
 *   ... +5 ELEMENT offset into pyr_der of the PREV derivative level
 *   ... +6 byte offset into pyr_img of the CURR image level of this fit
 *   ... +7 unused
 *
 * The PREV and the CURR planes live in the SAME buffer (pyr_img): a fit's
 * prev is a PAST frame's pyramid and a fit's curr is the current frame's
 * (or, for the step fit, frame t-2's).  Keeping them in one arena is what
 * makes "one upload per frame" possible -- the anchor fits of frame t all
 * read frame t's pyramid, and frames t-2/t-4/... are already there from
 * earlier frames.
 *
 * Work decomposition:
 *   get_global_id(0) = point index inside the fit
 *   get_global_id(1) = fit index
 * A work-item whose point index is >= that fit's ACTUAL point count
 * returns immediately: the fits have DIFFERENT corner counts (they are
 * different frames), so the tail of each fit's range is idle.
 *
 * The patch scratch is indexed by the GLOBAL slot
 *     gid = fit * npts_max + point
 * and its stride is the TOTAL work-item count (nfits * npts_max), NOT the
 * per-fit point count: the scratch is shared by all the fits of the
 * dispatch, so its stride must cover all of them.
 */
__kernel void lk_pyr_track_multi(
    __global const uchar *pyr_img,      /* arena: every resident pyramid's image planes */
    __global const short *pyr_der,      /* arena: every resident pyramid's deriv planes */
    __global const int   *desc,         /* nfits * nlevels * 8 */
    __global const int   *fit_count,    /* per fit: how many points it actually has */
    __global float       *patch,        /* scratch, stride = nfits * npts_max */
    __global const float *pts_in,       /* (fit*npts_max + point) * 2 */
    __global float       *pts_out,      /* same layout */
    __global int         *status_out,   /* same layout */
    const int nfits,
    const int nlevels,
    const int win,
    const int maxit,
    const float eps_sq,
    const float min_eig_thr,
    const int npts_max)
{
    const int p   = (int)get_global_id(0);
    const int fit = (int)get_global_id(1);
    if (fit >= nfits) return;
    const int n = fit_count[fit];
    if (p >= n) return;

    const int gid    = fit * npts_max + p;   /* global scratch slot */
    const int stride = nfits * npts_max;     /* scratch stride */
    const int dbase  = fit * nlevels * 8;     /* this fit's descriptor block */

    const int npix = win * win;
    const float2 half_win = (float2)((float)(win - 1) * 0.5f, (float)(win - 1) * 0.5f);
    const float2 p0 = (float2)(pts_in[2 * gid], pts_in[2 * gid + 1]);

    float2 next_pts = (float2)(0.0f, 0.0f);
    int st = 1;

    for (int L = nlevels - 1; L >= 0; --L) {
        const int padw   = desc[dbase + L * 8 + 0];
        const int roiw   = desc[dbase + L * 8 + 2];
        const int roih   = desc[dbase + L * 8 + 3];
        const int poff   = desc[dbase + L * 8 + 4];
        const int doff   = desc[dbase + L * 8 + 5];
        const int coff   = desc[dbase + L * 8 + 6];

        /* the ORIGINAL previous point, re-scaled for this level */
        float2 prev_pt = p0 * (1.0f / (float)(1 << L));
        /* the running estimate: the level above is twice as coarse */
        float2 next_pt = (L == nlevels - 1) ? prev_pt : next_pts * 2.0f;
        next_pts = next_pt;

        prev_pt -= half_win;                /* -> patch origin */
        const int ipx = (int)floor(prev_pt.x);
        const int ipy = (int)floor(prev_pt.y);

        if (ipx < -win || ipx >= roiw || ipy < -win || ipy >= roih) {
            if (L == 0) st = 0;
            continue;
        }
        const float fx = prev_pt.x - (float)ipx;
        const float fy = prev_pt.y - (float)ipy;

        /* ---- previous patch + derivatives, and the 2x2 normal matrix ---- */
        float A11 = 0.0f, A12 = 0.0f, A22 = 0.0f;
        for (int j = 0; j < win; ++j) {
            const int sy = ipy + j + win;          /* + pad */
            for (int i = 0; i < win; ++i) {
                const int sx = ipx + i + win;       /* + pad */
                const float v  = bilin_u8(pyr_img, poff, padw, sx, sy, fx, fy);
                const float2 d = bilin_der(pyr_der, doff, padw, sx, sy, fx, fy);
                const int q = j * win + i;
                patch[0 * npix * stride + q * stride + gid] = v;
                patch[1 * npix * stride + q * stride + gid] = d.x;
                patch[2 * npix * stride + q * stride + gid] = d.y;
                A11 += d.x * d.x;
                A12 += d.x * d.y;
                A22 += d.y * d.y;
            }
        }

        /* rejection tests in OpenCV's scale (A carries FLT_SCALE = 2^-20) */
        const float FLT_SCALE = 1.0f / 1048576.0f;
        const float A11s = A11 * FLT_SCALE;
        const float A12s = A12 * FLT_SCALE;
        const float A22s = A22 * FLT_SCALE;
        const float Ds = A11s * A22s - A12s * A12s;
        const float min_eig =
            (A22s + A11s - sqrt((A11s - A22s) * (A11s - A22s) + 4.0f * A12s * A12s)) /
            (2.0f * (float)(win * win));

        if (min_eig < min_eig_thr || Ds < 1.1920929e-07f) {
            if (L == 0) st = 0;
            continue;
        }

        const float D = 1.0f / (A11 * A22 - A12 * A12);

        next_pt -= half_win;               /* OpenCV: nextPt -= halfWin */
        float2 prev_delta = (float2)(0.0f, 0.0f);

        for (int j = 0; j < maxit; ++j) {
            const int inx = (int)floor(next_pt.x);
            const int iny = (int)floor(next_pt.y);

            if (inx < -win || inx >= roiw || iny < -win || iny >= roih) {
                if (L == 0) st = 0;
                break;
            }
            const float gx = next_pt.x - (float)inx;
            const float gy = next_pt.y - (float)iny;

            float ib1 = 0.0f, ib2 = 0.0f;
            for (int q = 0; q < win; ++q) {
                const int sy = iny + q + win;
                for (int p = 0; p < win; ++p) {
                    const int sx = inx + p + win;
                    const int idx = q * win + p;
                    const float cur = bilin_u8(pyr_img, coff, padw, sx, sy, gx, gy);
                    /* OpenCV keeps the previous patch at 32x scale: 32*(cur-prev) */
                    const float diff = 32.0f * (cur - patch[0 * npix * stride + idx * stride + gid]);
                    ib1 += diff * patch[1 * npix * stride + idx * stride + gid];
                    ib2 += diff * patch[2 * npix * stride + idx * stride + gid];
                }
            }

            const float2 delta = (float2)((A12 * ib2 - A22 * ib1) * D,
                                          (A12 * ib1 - A11 * ib2) * D);
            next_pt += delta;
            next_pts = next_pt + half_win;

            if (delta.x * delta.x + delta.y * delta.y <= eps_sq) break;
            if (j > 0 && fabs(delta.x + prev_delta.x) < 0.01f &&
                       fabs(delta.y + prev_delta.y) < 0.01f) {
                next_pts -= delta * 0.5f;
                break;
            }
            prev_delta = delta;
        }
    }

    pts_out[2 * gid]     = next_pts.x;
    pts_out[2 * gid + 1] = next_pts.y;
    status_out[gid] = st;
}
