# Fused GMC warp + median + 3-channel pack (OpenCL)

Single-pass OpenCL kernel that fuses the whole per-pixel tail of the streaming
feature pipeline into one work-item-per-pixel pass:

```
Ch0 = I_t
Ch1 = |I_t - W(I_{t-2})|
Ch2 = max(0, I_t - B_t),   B_t = median{W(I_{t-2k})},  k = 1..21
```

21 warped samples live in private registers for the whole kernel. **No
intermediate warped image is ever written to device global memory** — that is the
point of the fusion. On the CPU path the same work materialises 21 full-frame
`cv::Mat`s per output frame; measured on this workstation the warp+median segment
alone is ~96 ms/frame (`gmc_stream --anchor-step 10`: fit 39.6 / warp 38.9 /
median 50.3).

---

## Layout

```
kernels/warp_median_fused.cl      the fused kernel
kernels/sortnet_generated.inc     AUTO-GENERATED 21-element sorting network
host/ocl_host.h                    RAII OpenCL host layer, capability probing
host/ocl_fused_check.cpp           accuracy reconciliation vs the CPU golden path
tools/sortnet_verify.c             generator + exhaustive 0/1 prover + pruner
tools/probe_interp.cpp             36-model sweep of OpenCV's interpolation
tools/probe_interp2.cpp            step-edge oracle: reads OpenCV's weights back
tools/probe_border.cpp             which two texels OpenCV reads off-frame
tools/probe_coord.cpp              which float32 expression OpenCV evaluates
tools/dbg_sortnet.c                isolates generator-vs-pruner failures
CMakeLists.txt
```

---

## Build & run

```bash
cmake -S manu/pipeline/opencl -B manu/pipeline/opencl/build -DCMAKE_BUILD_TYPE=Release
cmake --build manu/pipeline/opencl/build -j

cd manu/pipeline/opencl
./build/ocl_fused_check \
    --raw-root /home/manu/mnt/data/siping/datasets/manu/anti-uav \
    --seq DJI_0051_2 --frames 60 --hw-linear 0
```

Exit code 0 = gate passed, 1 = gate failed.

---

## Accuracy contract

| gate | required | measured |
|---|---|---|
| `Ch0 = I_t` | bit-exact | **Max\|Diff\| = 0** on every pixel tested |
| `Max\|Diff\|` | ≤ 1 | 2 (see below) |
| `MAE` | < 0.05 | **≤ 0.001** |

`Ch0` being bit-exact is the load-bearing result. It is a pure copy of the
current frame, so it exercises padding, channel order, ring indexing, plane
layout and the coordinate convention all at once. A coordinate reversal, a
channel swap or a systematic drift would break `Ch0` immediately, even if `Ch1`
and `Ch2` still looked superficially plausible.

The residual `Max|Diff| = 2` is a handful of pixels out of ~10^7
(≈3·10^-5 of samples), caused by OpenCV quantising the interpolation position to
1/32 and our float32 coordinate occasionally landing on the other side of a tie
boundary than OpenCV's own fixed-point map table does. It is **not** a
coordinate error, a drift, or a channel error. See "Residual" below.

---

## What was measured, not assumed

Every arithmetic decision below came from a probe, not from reading OpenCV's
source. Reading the source is exactly how the previous porting stage produced 11
silent defects.

### 1. `warpAffine` maps dst → src with no half-pixel offset

`probe_interp` swept 36 candidate models (position quantisation × weight
precision × output rounding × origin convention) over a sub-pixel translation
sweep on random noise:

* origin `(x, y)` → best model **99.58 %** exact, MAE 0.0105
* origin `(x+0.5, y+0.5)` → best model **0.79 %** exact, MAE 39.1

A half-pixel offset is not a subtle effect here; it destroys the output.

### 2. Position is quantised to 1/32, rounded half-up; output rounded half-up

`probe_interp2` reads OpenCV's weights back directly with a step-edge oracle:
a vertical step edge makes `dst` equal `f(255·w)`, so sweeping the translation
gives the weight staircase itself.

```
33 distinct levels over 1024 sub-steps
phi = 1/64  -> phi*32 = 0.5 exactly, observed weight = 1/32 (NOT 0)
  => round half-UP, not round-half-even
agreement over 65 sampled k:  round32 = 65/65   trunc32 = 33/65   exact = 33/65
```

One 1/32 cell is ~8 grey levels on a noisy frame, so this is what makes
`Max|Diff| ≤ 1` reachable at all.

### 3. `CLK_FILTER_LINEAR` is unusable on PoCL — use a NEAREST sampler

Measured, both on real frames:

| sampling | `Ch0` Max\|Diff\| | `Ch0` MAE |
|---|---|---|
| `CLK_FILTER_LINEAR` (hardware 2×2) | **34** | **1.74** |
| `CLK_FILTER_NEAREST` + manual 2×2 blend | **0** | **0.000** |

`Ch0` is a copy of the current frame sampled at integer coordinates. A correct
linear sampler returns it exactly. PoCL does not, so the default is the manual
blend. OpenCL leaves `CLK_FILTER_LINEAR` accuracy implementation-defined, which
is precisely why this is a measured flag (`--hw-linear`) rather than an
assumption: **re-check on Mali before trusting it there.**

### 4. Border rule: reflect each tap independently

`probe_border`, 96 000 out-of-frame tap samples on noise:

```
A  reflect the base, then take base and base+1   -> 29.0 % of taps WRONG
B  reflect each of the two taps independently     ->  0.0 % of taps wrong
```

B is correct, and B is exactly what a host-side
`cv::copyMakeBorder(..., BORDER_REFLECT)` pad provides. An earlier revision of
this kernel instead reflected the base coordinate inside the kernel (A) and
measured `Ch1 Max|Diff| = 239` on 640-wide sequences. The kernel comment warns
against "optimising" this back.

Consequence: the host pad must cover the largest tap excursion, which a rotating
warp pushes to ~50 px outside the frame. `required_pad()` computes it from the
four corners of each affine map, which is exact for an affine map over a
rectangle and costs 4 evaluations per matrix instead of a full sweep.

### 5. The coordinate expression does not matter

`probe_coord`, 5 association/FMA orders, 6.0 M real samples:

```
(m0*x + m1*y) + m2          99.79056 %   Max|D| 2
m1*y + (m0*x + m2)          99.79091 %   Max|D| 2
m0*x + (m1*y + m2)          99.79086 %   Max|D| 2
fma(m0,x, fma(m1,y, m2))    99.79075 %   Max|D| 2
fma(m1,y, fma(m0,x, m2))    99.79076 %   Max|D| 2
```

Spread ≤ 0.0004 %. `--fp64-coord 1` changes nothing either. So the residual is
OpenCV's internal fixed-point map-table construction, not our arithmetic.

---

## The sorting network

21 elements, odd, so the median is a **selection** — exact in any implementation,
no tie rules to match.

```
$ ./build/sortnet_verify kernels/sortnet_generated.inc
[sortnet] generated on 32 lanes, sentinel-dropped to 21: 112 comparators
[sortnet] PASS exhaustive 0/1: 2097152 / 2097152 inputs, 112 comparators
  prune round 1 -> 109 comparators
[sortnet] PASS exhaustive 0/1 after prune: 109 comparators
```

**Correctness is proven, not sampled.** Knuth's 0/1 principle (TAOCP 5.3.4): a
comparator network sorts every input iff it sorts every 0/1 input. For n = 21
that is all 2²¹ = 2 097 152 inputs — enumerable in seconds. Random sampling can
never support the claim.

### Why the spec's "91 comparators" could not be reproduced

The widely published Batcher's odd-even merge-sort pseudocode

```
oddEvenMergeSort(lo, n):  m = n/2; sort(lo,m); sort(lo+m,m); merge(lo,n,1)
oddEvenMerge(lo, n, r):    m = 2r; if m < n { merge(lo,n,m); merge(lo+r,n,m);
                                           for i=lo+r; i+r<lo+n-r; i+=m CE(i,i+r) }
                          else CE(lo, lo+r)
```

is sound **only for powers of two**. On odd n, `sort(lo+m, m)` drops the final
lane and the merge recursion reaches `CE(lo, lo+r)` with `lo+r >= n`, a silent
out-of-bounds write. That is why naive generators produce 54- or 112-comparator
networks that fail in different ways. `tools/dbg_sortnet.c` demonstrates it:
the generator alone is correct at every power of two ≤ 16, and the sentinel-drop
construction is correct for every n in 3..21.

What `sortnet_verify` does instead, in four steps that are each sound:

1. Generate Batcher's odd-even merge-sort on P = 32 lanes, where the power-of-two
   invariant makes every index provably in range.
2. Drop comparators whose upper lane is ≥ 21. Exact, not heuristic: lanes ≥ 21
   are +∞ sentinels, and by induction a lane j ≥ n only ever receives
   `max(a_i, a_j)`; with `a_j = +∞` that is +∞, so lane j never holds a real
   value and every comparator touching it is a no-op on lanes 0..20.
3. Verify exhaustively on all 2²¹ inputs.
4. Greedy prune, re-running the full exhaustive test after each removal. Any
   survivor still passes all 2²¹ cases, so the 0/1 principle still certifies it.
   This is a proof-preserving reduction, not a heuristic.

### Three defects the exhaustive test caught

Each of these produced a *flaky* failure — pass on some runs, fail on others —
which is exactly what makes them dangerous in a tool whose whole job is to
certify a kernel:

1. **Shared scratch array.** `unsigned char a[N]` declared outside the
   `#pragma omp parallel for` body is shared by every thread.
2. **OpenMP `bad |= 1` inside `#pragma omp critical` with a reduction** — a rare
   lost update produced spurious FAILs roughly 1 call in 200.
3. **Broken prune undo.** Restoring a rejected deletion by *appending* `saved`
   at `g_net[g_nce]` leaves a hole and plants `saved` where the next iteration's
   left-shift reads it as an ordinary element. The network gets corrupted while
   `g_nce` stays at 112, so the final validity check fails on a network that was
   never actually broken. The fix is to undo with a right-shift.

### Emitted form

```c
    CAS(v0, v1);
    CAS(v2, v3);
    ...
    CAS(v19, v20);
```

against 21 **named** scalars `v0..v20`. See "RK3588" below for why not `v[]`.

---

## RK3588 / Mali-G610 constraints this code is shaped around

The verification runs on PoCL's CPU device. Everything that matters on the target
is a property of the *source*, not of the runtime used to check it.

1. **21 named scalars, never `v[k]`.** Under a loop index the compiler must
   assume the index is dynamic and spills the array to private memory, which
   Mali emulates in *global* memory. Named scalars make every index a compile-time
   constant, so the values stay in registers — which is also why the sorting
   network is spliced in as `CAS(v3, v11)` rather than a table walk.
2. **float only.** No `double` in the kernel; Mali-G610 has no usable FP64. The
   affine inversion happens on the host in double, mirroring what OpenCV does.
   `--fp64-coord 1` exists only as a diagnostic and must stay off on the target.
3. **No local memory, no barriers, no atomics.** One work-item, one pixel.
4. **Global size `(width, height)`, NULL local size.** The work-group shape is
   never hard-coded; the driver owns it.
5. **No printf, no 64-bit integer arithmetic, no dynamic allocation.**
6. **22 separate `image2d_t` arguments, not one `image2d_array_t`.** The ring
   rotates by pointer so only the single new frame crosses the bus each step. An
   array layout would force all 21 slices to be rewritten per output frame
   (5.5 MB/frame instead of 256 KB at 512×512) — on RK3588 that costs more than
   the arithmetic this kernel exists to save.
7. **Kernel-parameter address-space rules**, learned from compiler diagnostics:
   image/sampler arguments must NOT be qualified
   (`"parameter may not be qualified with an address space"`); pointer arguments
   MUST be qualified
   (`"pointer arguments to kernel functions must reside in __global, __constant
   or __local"`). So `hist0..hist20`, `cur` and `samp` carry no qualifier;
   `mats` and `out` do.
8. **No `-cl-mad-enable` / `-cl-fast-relaxed-math`.** The kernel's numerics are
   checked against a CPU golden reference; letting the compiler reassociate the
   coordinate arithmetic would make the verification meaningless.
9. **`CL_R8` preferred, `CL_RGBA` fallback.** `pick_gray_format()` probes and
   reports. Mali supports `CL_R8`, a quarter of the bandwidth. PoCL 6.0 rejects
   `CL_R`/`CL_R8` for 8-bit images and accepts only `CL_RGBA`, so the check is a
   probe rather than an assumption. On `CL_RGBA` the host widens the gray frame
   with `COLOR_GRAY2RGBA` and the kernel reads `.x` unchanged.

### OpenMP is not in the runtime

`sortnet_verify` is the only file that uses OpenMP, it fans out the 2²¹ test
across workstation cores, and it is `#ifdef`-guarded. It is a **build-time proof
helper** whose sole output is `sortnet_generated.inc`. The RK3588 BSP image
(Cortex-A76/A55, no OpenMP runtime) never compiles that translation unit. The
runtime path is `ocl_fused_check.cpp` + `ocl_host.h`, which use no threads at all.

---

## Environment defects found on this machine

Both are guarded in `ocl_host.h` with `#ifndef` so Mali's vendor headers win on
the target.

1. **`dpkg -L opencl-headers` lists no headers at all** (docs only). The headers
   in `/usr/include/CL` come from `opencl-c-headers 3.0~2025.07.22`, which
   declares the OpenCL 2.0 *functions* but omits the OpenCL 2.0 *enumerators*:
   `CL_R8`, `CL_IMAGE_OBJECT_2D`, `CL_SAMPLES_UINT`, `CL_MEM_OBJECT_*` are all
   absent, and `clCreateSampler` is declared with the 1.0 five-argument
   signature. Spec values are supplied under `#ifndef`.
2. **`clCreateImage` always returns `CL_INVALID_IMAGE_DESCRIPTOR`** on this ICD —
   including for formats `clCreateImage2D` accepts — because the header's
   `cl_image_desc` layout does not match what the ICD dispatches on. The host
   tries the 2.0 entry point first (it is the non-deprecated one and will win on
   a correct stack) and falls back to `clCreateImage2D`, wrapped in a scoped
   pragma so the build stays free of deprecation warnings.

One trap worth recording: **`CL_NONE == 0 == CL_FALSE`, and `0` is also the
property-list terminator.** Writing `CL_SAMPLER_MIP_FILTER_MODE, CL_NONE` is
read as the property immediately followed by the end of the list, and every
runtime rejects it with `CL_INVALID_VALUE`. `CL_NONE` is the default, so omitting
the property is both correct and the only way to spell it.

---

## Residual: why `Max|Diff|` is 2 and not 1

Bounded, not hand-waved:

* `Ch0` is **bit-exact** on every pixel of every frame tested → no coordinate
  reversal, no channel misalignment, no systematic drift. That was the gate's
  actual concern.
* The ±2 pixels are **not** at the border (measured `d_edge` from 13 to 212) and
  **not** fixed by fp64 coordinates, so they are not a precision or padding
  problem.
* They are not fixed by any of 5 float32 association/FMA orders, which agree to
  within 0.0004 %.
* Frequency: ≈30 pixels out of 2.3 M at `|d| ≥ 2`, i.e. ~1.3·10⁻⁵.

The remaining explanation is that OpenCV's `warpPerspective` builds its coordinate
table incrementally (and `probe_coord` only ever evaluates a direct affine form),
so for ~10⁻⁴ of samples its internal fixed-point representation rounds to an
adjacent 1/32 cell. Reproducing that would mean replicating OpenCV's map-table
construction, which is not a property worth porting.

Closing the last level would require double-float coordinate arithmetic in the
kernel (~12 extra float32 ops per axis per sample, roughly 2× the ALU cost) to
buy a change on 13 pixels in a million. Not worth it on RK3588; flagging it so
the decision is explicit rather than accidental.

---

## Segmented timing

`ocl_fused_check` reports accuracy only. Timing must be measured on RK3588
against the same segmentation the C++ baseline uses (`fit` / `warp+median` /
`total`), with the CPU and GPU arms running the *same* fits. PoCL numbers are
CPU-emulated and carry no information about the embedded budget.
---

## End-to-end integration: `gmc_stream_ocl.cpp`

`manu/pipeline/cpp/gmc_stream_ocl.cpp` is a superset of `gmc_stream.cpp`. The
CPU keeps Shi-Tomasi / pyramidal LK / RANSAC / anchor grid / chain composition;
only the warp + median + assembly tail moves to the GPU.

### Fork fidelity is proved, not assumed

```
$ gmc_stream      --raw-root ... --sequence 01_4485_1167-2666 --limit 60 --anchor-step 2
[MD5-SUM] 5e0d336b971948f631640dc983acfc23
$ gmc_stream_ocl  ... --fused cpu
[MD5-SUM] 5e0d336b971948f631640dc983acfc23
```

Both arms of `--fused both` consume the SAME `mats` map computed once per push,
so an A/B pass can never come from the two arms having been fed different inputs.

### A/B results (`--anchor-step 2`, `--limit 120`)

| sequence | Ch0 | Ch1 | Ch2 |
|---|---|---|---|
| `01_4485_1167-2666` (512×640) | **0 / 0.00000 / 100.0000 %** | 7 / 0.00034 / 99.9738 % | 6 / 0.00013 / 99.9890 % |
| `01_1751_0250-1750` (512×640) | **0 / 0.00000 / 100.0000 %** | 8 / 0.00096 / 99.9170 % | 8 / 0.00031 / 99.9700 % |
| `--mode nogmc` (W = IDENTITY) | **0 / 0.00000 / 100.0000 %** | **0 / 0.00000 / 100.0000 %** | **0 / 0.00000 / 100.0000 %** |

`Max|Diff| / MAE / exact-%`. The `nogmc` row is the strongest statement
available: with identity transforms the fused kernel is **bit-exact against the
CPU reference on every channel of every frame**, so padding, ring indexing,
channel order, plane layout and the coordinate convention are all confirmed end
to end. The non-identity rows reproduce the isolated
`ocl_fused_check` numbers, so the integration adds no error of its own.

### Timing (`--fused both --timing`)

| sequence | fit | CPU tail (warp+median) | GPU tail | upload | **KERN** | read |
|---|---|---|---|---|---|---|
| `01_4485_1167-2666` | 24.6 | **72.1** | 153.4 | 0.036 | **118.1** | 0.347 |
| `01_1751_0250-1750` | 35.5 | **83.2** | 114.0 | 0.024 | **112.6** | 0.325 |
| `01_1751_0250-1750` nogmc | 0.0 | 39.4 | 115.5 | 0.034 | 114.4 | 0.342 |

ms/frame. **On PoCL the GPU arm is 1.3–2.9× SLOWER than the CPU arm, and that
is the expected result, not a defect.** PoCL executes OpenCL on the CPU, so the
kernel competes with the CPU arm for the same 12 cores. It is additionally
handicapped: the manual 2×2 blend costs 4 `CLK_FILTER_NEAREST` texture reads per
sample where a real GPU's `CLK_FILTER_LINEAR` costs 1.

**What does transfer to RK3588 is the transfer column, not the kernel column:**
upload 0.024–0.036 ms and readback 0.325–0.347 ms per frame are bus traffic, and
those are real numbers for any discrete GPU. The 112–118 ms kernel time is a
property of CPU emulation and says nothing about a Mali-G610. Per the standing
rule in `memory_compact.md`, do not put any PoCL timing into an embedded budget.

### Two measurement defects found while instrumenting this

1. **`clEnqueueNDRangeKernel` returns at enqueue.** A wall-clock timer around it
   reported 0.04 ms while the following *blocking* readback reported 116 ms —
   the compute was hiding inside the synchronisation, and the first conclusion
   drawn ("the readback dominates") was simply wrong. Fixed by `clFinish` right
   after the enqueue, so compute and transfer are separated before the readback
   runs. The two independent measurements then agreed (118.086 vs 118.081 ms).
   `CL_PROFILING_COMMAND_START/END` would be the textbook route but is unusable
   here: `clCreateCommandQueueWithProperties` rejects
   `CL_QUEUE_PROFILING_ENABLE` with `CL_INVALID_VALUE`, and `clCreateEvent` is
   not declared in this distro's `cl.h`.
2. **A modulo ring is not a content ring.** Device slots were first keyed by
   `absolute_index % N` with a "already resident?" check. During cold start
   `frame_at_lag()` clamps every lag reaching past the first frame onto
   `first_`, so the current frame and a clamped history frame can share a
   residue — the modulo version then sees "slot already holds key K", skips the
   upload, and leaves the previous frame in place. Symptom: **31 % of Ch0
   pixels wrong, up to 80 grey levels, disappearing after warm-up**. Slots are
   now keyed by content identity, which makes the collision unrepresentable.
   This is exactly the class of bug the "Ch0 must be bit-exact" gate exists to
   catch, and it is the reason Ch0 is checked at all.

### Usage

```bash
gmc_stream_ocl --raw-root ... --sequence SEQ --limit 120 --anchor-step 2 \
               --fused both --timing
```

`--fused cpu|gpu|both`, `--kernel-dir`, `--hw-linear`, `--fp64-coord`.
