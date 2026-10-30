/* ============================================================================
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
