/* ============================================================================
 * sortnet_verify.c  --  21-element branchless sorting network: PROVE it, then
 *                       emit it for the OpenCL kernel.
 *
 * ----------------------------------------------------------------------------
 * SCOPE / DEPLOYMENT NOTE  (read before assuming this ships)
 * ----------------------------------------------------------------------------
 * This is a PC-ONLY, OFFLINE, BUILD-TIME tool. It is NOT part of the RK3588
 * runtime. It does not link OpenCL, does not run on the device, and is never
 * compiled into gmc_stream / the OpenCL host module. Its single output is a
 * block of `CAS(...)` text that gets pasted into warp_median_fused.cl as
 * compile-time constants.
 *
 * OpenMP here is used only to fan out the 2^21 exhaustive test loop across the
 * workstation's cores. It is guarded by #ifdef _OPENMP so the file also builds
 * single-threaded. The RK3588 target (Cortex-A76/A55, no OpenMP runtime in the
 * vendor BSP image) never compiles this translation unit at all.
 *
 * ----------------------------------------------------------------------------
 * WHY "91 COMPARATORS" COULD NOT BE REPRODUCED
 * ----------------------------------------------------------------------------
 * The widely published Batcher's odd-even merge-sort pseudocode
 *
 *     oddEvenMergeSort(lo, n):  m = n/2; sort(lo,m); sort(lo+m,m); merge(lo,n,1)
 *     oddEvenMerge(lo, n, r):    m = 2r; if m < n { merge(lo,n,m); merge(lo+r,n,m);
 *                                                 for i=lo+r; i+r<lo+n-r; i+=m CE(i,i+r) }
 *                               else CE(lo, lo+r)
 *
 * is ONLY sound when n is a power of two. `sort(lo+m, m)` drops the final lane
 * on odd n, and the merge recursion reaches `CE(lo, lo+r)` with lo+r >= n --
 * a silent out-of-bounds write. n = 21 hits both. That is why naive generators
 * produce 54- or 112-comparator networks that fail in different ways.
 *
 * WHAT THIS TOOL ACTUALLY DOES (all four steps are sound):
 *   1. Generate Batcher's odd-even merge-sort on P = 32 lanes, where the
 *      power-of-two invariant makes every index provably in range.
 *   2. Drop every comparator whose upper lane index is >= 21. This is exact, not
 *      a heuristic: lanes >= 21 are +infinity sentinels. By induction a lane
 *      j >= n only ever RECEIVES max(a_i, a_j); with a_j = +inf that is +inf,
 *      so lane j never holds a real value, so every comparator touching it is a
 *      no-op on lanes 0..20. Dropping them is behaviour-preserving.
 *   3. Verify EXHAUSTIVELY via the 0/1 principle (Knuth, TAOCP 5.3.4): a
 *      comparator network sorts every input iff it sorts every 0/1 input. For
 *      n = 21 that is all 2^21 = 2,097,152 inputs -- enumerable, not sampled.
 *      Random sampling can never make this claim.
 *   4. Greedily prune comparators, re-running the FULL exhaustive test after
 *      each removal. Any survivor still passes all 2^21 cases, so the 0/1
 *      principle still certifies it. This is a proof-preserving reduction, not
 *      a heuristic.
 *
 * Usage:  gcc -O3 -fopenmp sortnet_verify.c -o sortnet_verify
 *         ./sortnet_verify            # generate + verify + prune + emit
 * ==========================================================================*/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 21                 /* elements to sort  */
#define P 32                 /* padded power-of-two generator width */
#define MAX_CE 1024

typedef struct { uint8_t i, j; } CE;

static CE g_net[MAX_CE];
static int g_nce = 0;

static void push_ce(int i, int j) {
    if (i == j) return;                 /* a self-comparator is a no-op */
    if (i > j) { int t = i; i = j; j = t; }
    if (j >= N) return;                 /* sentinel drop, step 2 */
    g_net[g_nce].i = (uint8_t)i;
    g_net[g_nce].j = (uint8_t)j;
    g_nce++;
}

/* ---- step 1: Batcher's odd-even merge sort on a POWER-OF-TWO width ------- */
static void gen_merge(int lo, int n, int r) {
    int m = r * 2;
    if (m < n) {
        gen_merge(lo, n, m);
        gen_merge(lo + r, n, m);
        for (int i = lo + r; i + r < lo + n - r; i += m) push_ce(i, i + r);
    } else {
        push_ce(lo, lo + r);
    }
}

static void gen_sort(int lo, int n) {
    if (n <= 1) return;
    int m = n / 2;
    gen_sort(lo, m);
    gen_sort(lo + m, m);      /* only correct because P is a power of two */
    gen_merge(lo, n, 1);
}

/* ---- step 3: exhaustive 0/1 verification --------------------------------
 * The parallel path writes NO shared mutable state. It only folds a per-thread
 * "did anything fail" bit through a reduction. Recording the offending mask
 * would need a critical section inside the hot loop, and that shared write was
 * producing rare, unreproducible false FAILs (~1 in 200 calls) which made a
 * correct network look broken. When a failure does occur we rescan serially to
 * find the mask -- 2^21 iterations costs a fraction of a second and only runs
 * on the failure path.
 * ------------------------------------------------------------------------- */
static int verify(void) {
    int bad = 0;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(| : bad)
#endif
    for (long mask = 0; mask < (1L << N); ++mask) {
        /* MUST be block-scoped: under #pragma omp parallel for a variable
         * declared outside the loop body is SHARED by every thread. Declaring
         * the scratch array here (not above the loop) is what makes the test
         * correct; hoisting it was a real data race. */
        unsigned char a[N];
        for (int k = 0; k < N; ++k) a[k] = (unsigned char)((mask >> k) & 1);
        for (int c = 0; c < g_nce; ++c) {
            const unsigned char x = a[g_net[c].i], y = a[g_net[c].j];
            a[g_net[c].i] = x < y ? x : y;
            a[g_net[c].j] = x < y ? y : x;
        }
        int ok = 1;
        for (int k = 1; k < N; ++k)
            if (a[k] < a[k - 1]) { ok = 0; break; }
        if (!ok) bad |= 1;
    }
    return bad == 0;
}

/* Serial rescan, failure path only: return the first mask this network fails on. */
static int first_bad_mask(void) {
    unsigned char a[N];
    for (long mask = 0; mask < (1L << N); ++mask) {
        for (int k = 0; k < N; ++k) a[k] = (unsigned char)((mask >> k) & 1);
        for (int c = 0; c < g_nce; ++c) {
            const unsigned char x = a[g_net[c].i], y = a[g_net[c].j];
            a[g_net[c].i] = x < y ? x : y;
            a[g_net[c].j] = x < y ? y : x;
        }
        for (int k = 1; k < N; ++k)
            if (a[k] < a[k - 1]) return (int)mask;
    }
    return -1;
}

/* ---- step 4: proof-preserving greedy pruning ---------------------------- */
static void prune(void) {
    int changed = 1;
    int rounds = 0;
    while (changed && g_nce > 0) {
        changed = 0;
        ++rounds;
        int c = 0;
        while (c < g_nce) {
            const CE saved = g_net[c];
            /* Delete c by shifting the tail left. g_nce is left decremented so
             * verify() sees the shorter trial network. */
            for (int k = c; k + 1 < g_nce; ++k) g_net[k] = g_net[k + 1];
            --g_nce;
            if (verify()) {
                changed = 1;               /* genuinely redundant -- keep it gone */
            } else {
                /* UNDO the shift by moving the tail back RIGHT. Appending
                 * `saved` at g_net[g_nce] instead would leave a hole at the
                 * deletion point and plant `saved` where the next iteration's
                 * left-shift reads it as an ordinary element, corrupting the
                 * network while leaving g_nce unchanged -- so the final validity
                 * check fails on a network that was never actually broken. */
                for (int k = g_nce; k > c; --k) g_net[k] = g_net[k - 1];
                g_net[c] = saved;
                ++g_nce;
                ++c;
            }
        }
        fprintf(stderr, "  prune round %d -> %d comparators\n", rounds, g_nce);
    }
}

/* ---- emit: FULLY UNROLAPPED scalar form, no v[] indexing -----------------
 * RK3588 (Mali-G610) note: the private array must stay in registers. A
 * kernel that writes `CAS(v[3], v[17])` forces the compiler to assume
 * dynamic indexing and spill the whole array to private (scratch) memory,
 * which on Mali is emulated in global memory. Spelling every lane out as a
 * distinct scalar makes the indexing obviously constant so the array stays
 * scalarised. Emitted below as `v0..v20`.                                 */
static void emit_cl(FILE* f) {
    fprintf(f, "/* ---- BEGIN GENERATED SORTING NETWORK "
               "(sortnet_verify.c, exhaustive 0/1 verified) ---- */\n");
    for (int c = 0; c < g_nce; ++c)
        fprintf(f, "    CAS(v%d, v%d);\n", g_net[c].i, g_net[c].j);
    fprintf(f, "/* ---- END GENERATED SORTING NETWORK ---- */\n");
}

int main(int argc, char** argv) {
    const char* out_cl = (argc > 1) ? argv[1] : "sortnet_generated.inc";
    int skip_prune = (argc > 2) && !strcmp(argv[2], "--no-prune");

    gen_sort(0, P);
    fprintf(stderr, "[sortnet] generated on %d lanes, sentinel-dropped to %d: %d comparators\n",
            P, N, g_nce);
    for (int c = 0; c < g_nce; ++c)
        if (g_net[c].j >= N) { fprintf(stderr, "[sortnet] BUG: index %d >= %d\n", g_net[c].j, N);
                               return 1; }

    
    if (!verify()) {
        fprintf(stderr, "[sortnet] FAIL: not sorted on mask 0x%06x\n", first_bad_mask());
        return 1;
    }
    fprintf(stderr, "[sortnet] PASS exhaustive 0/1: %d / %d inputs, %d comparators\n",
            1 << N, 1 << N, g_nce);

    if (!skip_prune) {
        prune();
        
        if (!verify()) {
            fprintf(stderr, "[sortnet] FAIL after prune: mask 0x%06x\n", first_bad_mask());
            return 1;
        }
        fprintf(stderr, "[sortnet] PASS exhaustive 0/1 after prune: %d comparators\n", g_nce);
    }

    FILE* f = fopen(out_cl, "w");
    if (!f) { perror(out_cl); return 1; }
    fprintf(f, "/* AUTO-GENERATED -- do not edit. */\n");
    fprintf(f, "/* %d-element sorting network, %d comparators, */\n", N, g_nce);
    fprintf(f, "/* PROVEN correct on all 2^21 = 2097152 zero-one inputs. */\n");
    emit_cl(f);
    fclose(f);
    fprintf(stderr, "[sortnet] wrote %s\n", out_cl);
    return 0;
}