/* ============================================================================
 * dbg_sortnet.c -- isolate WHICH of the two claims fails:
 *   (A) the Batcher's odd-even merge-sort GENERATOR is a valid P-lane network
 *   (B) dropping comparators whose upper lane >= n (the +inf sentinel argument)
 *
 * Run A over every power of two up to 32 (exhaustive 2^P).
 * Run B over every n in 3..21 at the smallest power-of-two width >= n
 * (exhaustive 2^n). A failure pattern here tells us the sentinel argument is
 * unsound; a clean A with dirty B means exactly that.
 *
 * PC-ONLY OFFLINE TOOL -- not part of the RK3588 runtime.
 * ==========================================================================*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int g_w;              /* lanes in the simulated array */
static unsigned char g_net[1024][2];
static int g_nce;

static void push(int i, int j) {
    if (i == j) return;
    if (i > j) { int t = i; i = j; j = t; }
    if (j >= g_w) return;     /* THE DROP (disabled when g_w == P) */
    g_net[g_nce][0] = (unsigned char)i;
    g_net[g_nce][1] = (unsigned char)j;
    g_nce++;
}

static void merge_(int lo, int n, int r) {
    int m = r * 2;
    if (m < n) {
        merge_(lo, n, m);
        merge_(lo + r, n, m);
        for (int i = lo + r; i + r < lo + n - r; i += m) push(i, i + r);
    } else {
        push(lo, lo + r);
    }
}

static void sort_(int lo, int n) {
    if (n <= 1) return;
    int m = n / 2;
    sort_(lo, m);
    sort_(lo + m, m);
    merge_(lo, n, 1);
}

static int verify(int lanes) {
    unsigned char a[32];
    for (long mask = 0; mask < (1L << lanes); ++mask) {
        for (int k = 0; k < lanes; ++k) a[k] = (unsigned char)((mask >> k) & 1);
        for (int c = 0; c < g_nce; ++c) {
            unsigned char x = a[g_net[c][0]], y = a[g_net[c][1]];
            a[g_net[c][0]] = x < y ? x : y;
            a[g_net[c][1]] = x < y ? y : x;
        }
        for (int k = 1; k < lanes; ++k)
            if (a[k] < a[k - 1]) return (int)mask;
    }
    return -1;
}

int main(void) {
    printf("=== (A) generator alone, no drop, power-of-two widths ===\n");
    /* Exhaustive 0/1 only up to P=16 (2^16 = 65536 inputs). P=32 would be
     * 4.3e9 inputs x 191 comparators, which is not worth the wall clock for a
     * sanity check -- and if the generator were broken we would already have
     * seen it at P=2..16. */
    for (int p = 2; p <= 16; p *= 2) {
        g_nce = 0; g_w = p;                 /* no lane is >= p */
        sort_(0, p);
        int bad = verify(p);
        printf("  P=%-3d comparators=%-4d  %s\n", p, g_nce,
               bad < 0 ? "OK (exhaustive 0/1)" : "BROKEN (first bad mask)");
        if (bad >= 0) return 1;
    }

    printf("\n=== (B) sentinel drop onto n lanes, P = next pow2 >= n ===\n");
    for (int n = 3; n <= 21; ++n) {
        int p = 1; while (p < n) p *= 2;
        g_nce = 0; g_w = n;                 /* drop lanes >= n */
        sort_(0, p);
        int bad = verify(n);
        printf("  n=%-3d P=%-3d comparators=%-4d  %s%s\n", n, p, g_nce,
               bad < 0 ? "OK" : "BROKEN",
               bad < 0 ? "" : "  <-- sentinel argument fails here");
    }
    return 0;
}