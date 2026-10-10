# RK3588 board tooling — OpenCL probe, fused-kernel accuracy harness, and OpenCV parity

Two groups of read-only programs plus the machinery to build, ship and run them.

**Group 1 — OpenCL / fused kernel.** Both `dlopen` OpenCL at run time, so
**neither has any link-time OpenCL dependency** and both cross-compile with
nothing but the Rockchip toolchain and `libdl`.

**Group 2 — OpenCV parity.** Cross-compiles OpenCV 4.10.0 for the board and
measures whether it agrees with the x86 reference operator by operator.

| tool | runs on | what it answers |
| :--- | :--- | :--- |
| `board_cl_probe` | board | Does this board have a usable OpenCL stack, and which one? |
| `gen_fused_case` | **x86 + OpenCV** | Builds accuracy test cases: real frames, real GMC, and the CPU reference |
| `board_fused_accuracy` | board (no OpenCL dep) | Runs the real `warp_median_fused.cl` and reconciles it against that reference |
| `board_opencv_probe` | board | Does this board have an OpenCV C++ SDK? |
| `build_opencv_arm64.sh` | x86 | Cross-compiles **OpenCV 4.10.0** for arm64 (6 modules, static) |
| `gen_opencv_parity_case` | **x86 + OpenCV** | Decodes real frames to raw grayscale, once |
| `opencv_parity` | **both** | Runs the feature-channel CPU chain and dumps every operator's output |
| `compare_opencv_parity` | x86 | Element-wise per-operator comparison of the two dumps |

Supporting files: `fused_case_format.h` (the on-disk case contract),
`build_board_probe.sh` / `build_fused_accuracy.sh` (cross-compile),
`deploy_and_run.py` (pexpect driver), `.gitignore`.

## Why dlopen instead of linking `-lOpenCL`

Nothing at link time can be assumed to provide `libOpenCL` here:

* the x86 cross toolchain's sysroot ships **no** `libOpenCL`, and
* the Rockchip SDK carries only a **buildroot recipe**
  (`buildroot/package/opengl/libopencl/libopencl.mk`), not a prebuilt library.

`dlopen` defers that decision to the board, where the answer is actually
observable. A missing OpenCL stack becomes a clean report instead of a link
error, and the tool can name the library that answered — the single most useful
line for telling an ICD loader apart from a directly-linked Mali driver.

Candidates are tried in order: `libOpenCL.so.1`, `libOpenCL.so`,
`libmali-vendor.so.1`, `libmali-vendor.so`, `libMali-Vendor.so`, `libpocl.so.2`.
Override from the command line if needed.

## Why the CPU reference is shipped as data

`gen_fused_case` runs on the x86 host and needs real OpenCV. That is not an
accident of convenience — **there is no arm64 OpenCV anywhere in this project**:
not in the cross toolchain's sysroot, not in the vendor SDK, and not on the
board (which has a Python `cv2` but no `/usr/include/opencv4`).

The bit-exactness argument for this kernel is "the GPU reproduces what OpenCV
would have produced", and OpenCV's `warpAffine` quantises interpolation
positions to 1/32 and has a specific `BORDER_REFLECT` model. Re-implementing
those on the board would mean validating *my own reimplementation of OpenCV*
rather than the kernel — a circular test. So the reference is computed with
real `cv::warpAffine` on the host and travels as data.

What this does **not** rely on: the GMC matrices are computed by the same
primitives `gmc_stream.cpp` uses (Shi-Tomasi → LK → `estimateAffinePartial2D`
with RANSAC), on real Anti-UAV frames, not synthetic displacements.

## Build

```bash
cd manu/pipeline/opencl/board
bash ./build_board_probe.sh          # -> out/board_cl_probe
bash ./build_fused_accuracy.sh       # -> out/board_fused_accuracy
```

The Khronos headers are staged into `.build-include/` rather than pointing `-I`
at `/usr/include`, because that would drag the **x86 glibc headers** into the
cross build. The CL headers are plain C declarations with no
architecture-specific content, so the host copies are correct for aarch64.

Outputs are ~30–40 KB each, dynamically linked, interpreter
`/lib/ld-linux-aarch64.so.1`, depending only on `libdl`, `libstdc++`, `libm`,
`libgcc_s` and `libc`.

## Deploy and run

**Executables go over SSH; only data files go over NFS.** Writing a file into
the board's `/mnt/manu` from x86 and then executing it there fails with
`Text file busy` — and it fails even for a brand-new filename, so it is the NFS
server's write still being open, not a stale inode. `deploy_and_run.py`
therefore ships binaries through the SSH session and leaves the (large) case
bundle on the NFS mount.

```bash
cd manu/pipeline/opencl/board

# 1. build the cases on x86 (needs the dataset mount)
SEQ=/home/manu/mnt/data/siping/datasets/manu/anti-uav/train/01_4485_1167-2666
./gen_fused_case --seq "$SEQ" --out /tmp/case_gmc.bin --cases 3 --transform gmc

# 2. run on the board
RK3588_PASSWORD='<board password>' python3 deploy_and_run.py \
  --exec 'cd /mnt/manu/fused_exec && ./board_fused_accuracy \
          --case /mnt/manu/fused/case_gmc.bin \
          --kernel /mnt/manu/fused/warp_median_fused.cl \
          --sortnet /mnt/manu/fused/sortnet_generated.inc \
          --variant both'
```

`deploy_and_run.py --exec` accepts any shell command, so it doubles as the
general board query channel. The password is read from `RK3588_PASSWORD` or
prompted for — it is never written to a file.

## Results measured on the board (`evm3588`, 2026-10-09)

Probe: `PROBE RESULT: PASS`, exit 0.

| item | measured |
| :--- | :--- |
| board | `evm3588`, Ubuntu 20.04.6 LTS, kernel `5.10.160 #1 ... aarch64` |
| CPU | 8 cores = 4× Cortex-A76 (`0xd05`) + 4× Cortex-A55 (`0xd0b`) |
| RAM | 7.7 GiB total, 7.3 GiB available |
| disk | `/dev/root` 16 G, 1.7 G free; `/mnt/manu` is the same filesystem |
| GPU | `/dev/mali0` present, driver **built into the kernel** |
| OpenCL | ARM ICD loader + `libmali.so.1.9.0`; ICD layout is a **directory** |
| device | **Mali-G610 r0p0**, OpenCL 3.0 `v1.g13p0-01eac0`, `FULL_PROFILE` |
| capacity | 4 compute units, max work group 1024, 7902.1 MiB global, 32 KiB local |

Accuracy, real GMC on `01_4485_1167-2666` (640×512, pad 67, 3 cases):

| variant | Ch0 | Ch1 | Ch2 | verdict |
| :--- | :--- | :--- | :--- | :--- |
| `hw-linear` | Max\|Diff\|=**130**, MAE 0.285, 88.5% exact | MAE 0.307~0.421 | MAE 0.315~0.330 | **FAIL** |
| `manual bilinear` | Max\|Diff\|=**0**, **100.00% exact** | Max\|Diff\|=5~6, MAE 0.0011 | Max\|Diff\|=1~4, MAE 0.00075 | **PASS** |

Three things this settled, all previously unknown:

1. **`FUSED_USE_HW_LINEAR` must be 0.** The decisive evidence is Ch0: it is an
   integer-coordinate copy of the current frame that must reproduce bit for
   bit, yet the hardware-filter arm is off by 130. Mali's `CLK_FILTER_LINEAR`
   does not reproduce the source texel even at integer coordinates.
2. **Mali accepts 1 byte per pixel.** `{CL_R, CL_UNORM_INT8}` works, so 22
   padded frames cost **10.5 MiB** instead of 42 MiB for `CL_RGBA`.
   `clCreateImage` (OpenCL 2.0) also works on Mali, so the 1.x fallback is not
   needed.
3. **First trustworthy board timing, measured in three parts.** The first pass
   put only `clEnqueueNDRangeKernel + clFinish` inside the timing window, so the
   "3.7~4.8 ms" figure was the **kernel alone** — uploads happened before `t0`
   and the read-back after `t1`, neither counted. The tool now times all three
   and prints the sum:

   | stage | ms/frame (manual arm, 3 cases) |
   | :--- | :--- |
   | upload (22 padded images) | 3.326 / 2.870 / 2.719 |
   | kernel (enqueue + finish) | 5.706 / 4.054 / 3.867 |
   | readback | 0.122 / 0.110 / 0.111 |
   | **total tail cost** | **9.154 / 7.035 / 6.697** |

   **Upload is ~40% of it** and is the same order as the kernel itself, so quote
   the total (6.7~9.2 ms), never the kernel figure on its own.

   ⚠️ **Even the total excludes the GMC fit.** `board_fused_accuracy` does not
   recompute fits — the matrices arrive from the x86 host, because there is no
   arm64 OpenCV to run Shi-Tomasi / LK / RANSAC with. The fit has therefore
   never been measured on this board, and on x86 it has historically cost far
   more than the tail (tree anchors: 6.19 fits/frame at a folded 4.8~6.6 ms per
   fit, i.e. roughly 30~41 ms/frame — and that folded figure is itself disputed
   by a 5x discrepancy against `stage timing` that is still unresolved).
   **"The fused kernel is on the GPU" does not mean "the whole pipeline is a few
   milliseconds."**

   Remaining qualifiers: all figures are wall-clock **upper bounds** (event
   profiling is unavailable on this driver), the first launch of a given kernel
   includes JIT compilation, and the geometry is 640×512 with pad 67
   (774×646 padded).

For scale, the same three cases total 131~135 ms on PoCL, so real hardware
is roughly **15~20x** faster than the CPU simulation here.

Scale of the accuracy cost: moving from the CPU simulation to real hardware
raises MAE by only **2~3×** (still ~1e-3, two orders below the 0.05 gate),
whereas choosing the wrong sampling path costs **300~400×**. The hardware is
not the risk; the path choice is.

> **The ICD file lies.** `mali.icd` names `libMaliOpenCL.so.1`, and that file
> does not exist. The driver actually in use is `libmali.so.1.9.0`, from the
> Debian package `libmali-valhall-g610-g13p0-x11-gbm`. Use
> `find / -name 'libmali*'` to locate it.

> **OpenCL runs on the GPU and says nothing about the NPU.** `/dev/rknpu` does
> not exist and `/proc/devices` has no `rknpu` entry, even though
> `/usr/lib/librknnrt.so` is installed. The current kernel does not enable the
> rknpu driver, so the RKNN path cannot run as things stand.

## Accuracy gates

The gate is **MAE < 0.05 per channel**, the threshold the kernel header itself
names as the one a systematic rounding mismatch cannot hide under. `Max|Diff|`
is printed per channel for comparison against the x86 baseline (Ch0 exact;
Ch1 6..8; Ch2 6..8) rather than gated at 1, because that residual is the
float32 coordinate tie against OpenCV's own 1/32 quantisation, not a defect.

The verdict is reported **per variant** plus an explicit recommendation, because
a single global FAIL would hide that one path is unusable while the other is
the one to ship.

## Host self-test

Both programs are portable, so the same source builds and runs natively on the
verification host, which catches host-side bugs before the board ever sees the
binary:

```bash
mkdir -p .hostcheck && cp -r .build-include/CL .hostcheck/
g++ -std=c++17 -O2 -Wall -Wextra -DCL_TARGET_OPENCL_VERSION=300 \
    -I.hostcheck board_cl_probe.cpp -o .hostcheck/probe -ldl
./.hostcheck/probe

# the accuracy harness needs a case bundle first
./gen_fused_case --seq <frames dir> --out /tmp/case.bin --cases 1
g++ -std=c++17 -O2 -DCL_TARGET_OPENCL_VERSION=300 -I.hostcheck \
    board_fused_accuracy.cpp -o .hostcheck/acc -ldl
./.hostcheck/acc --case /tmp/case.bin --kernel ../kernels/warp_median_fused.cl \
    --sortnet ../kernels/sortnet_generated.inc --variant both
```

> PoCL is a **CPU simulation**. Its timings say nothing about the Mali GPU and
> must never enter an embedded budget.

## Traps this tooling hit, and guards against

Recorded in full in `manu/memory/falsified_archive.md` §34 and §35. The ones
that would bite you first:

- **`/*__SORTNET__*/` is a build-time placeholder.** Feed the raw `.cl` and the
  kernel compiles, runs, and reports a perfect Ch0 and Ch1 — while Ch2 is
  silently wrong, because `v10` is not the median. `--sortnet` is required and
  the tool **refuses to run** if the marker survives the splice.
- **Upload must be expanded to the device's channel count.** The bundle is
  always 1 byte/pixel; if the device only accepts `CL_RGBA`, `row_pitch == 0`
  means `pw * nch` bytes per row and a 1-byte row walks off the end of every
  row, segfaulting inside the driver's `memcpy`.
- **`clCreateSampler` takes five arguments**, the middle one being
  `cl_bool normalized_coords`. Omitting it from a hand-written function-pointer
  typedef shifts every later argument left and segfaults inside the driver.
- **Use `clSetKernelArg` with a `cl_sampler`, not `clSetKernelArgSampler`.**
  The latter is deprecated since 2.0 and is simply absent from ocl-icd 2.3.4.

---

# Part 2 — OpenCV 4.10.0 for arm64, and what it actually agrees with

## Why cross-compile instead of `apt install libopencv-dev`

The board has no OpenCV C++ SDK (four independent probes agree, and
`g++ -lopencv_core` fails outright). apt offers **4.2.0+dfsg-5** — eight minor
versions behind the x86 reference **4.10.0**. Installing it would silently
change the numerics that the fused-kernel work was validated against, so the
same tag was built from source instead.

The build uses a **git worktree** at `/media/manu/1TB-Volume/workspace/opencv-4.10.0-arm64`;
the user's existing 5.x checkout is untouched.

Only the six modules the pipeline actually calls are built
(`core imgproc features2d video calib3d flann`) with every third-party
dependency off. This is not just a time optimisation: the cross sysroot has
**no zlib/libpng/libjpeg/libtiff headers at all**, so `imgcodecs` could not be
built without first building those libraries. The parity test therefore ships
**raw pixels** instead of images, which sidesteps `imgcodecs` entirely and
guarantees both machines read byte-identical input rather than each decoding
independently and hoping they agree.

## The floating-point contract, and what it does and does not fix

`-ffp-contract=off -fno-fast-math` is applied project-wide, and it is the right
call for aarch64: `fmadd` is in the baseline, so GCC's default
`-ffp-contract=fast` would fuse `a*b+c` into a single rounding where the x86
build cannot.

It does **not** make the two architectures bit-identical, and an ablation
proves it. Running the x86 build with
`OPENCV_CPU_DISABLE=AVX2,AVX512_SKX,FP16,SSE4_1,SSE4_2` — same machine, same
data, only the FMA3-carrying dispatch layers removed — leaves **all 13 stages
bit-exact over 59 frames**. So the LK difference has nothing to do with FMA.
The x86 package dispatches to AVX2/AVX512 code compiled *with* FMA3 even though
its baseline is plain SSE2; the residual difference is the architecture-specific
SIMD kernels themselves (x86 SSE/AVX vs NEON/carotene) plus a six-year compiler
gap (gcc 15.2.0 vs gcc 9.3.0). Neither is removable.

## Measured result (59 frames, 19,333,120 elements per stage)

| operator | verdict |
| :--- | :--- |
| `cv::resize` INTER_LINEAR | bit-exact |
| `cv::goodFeaturesToTrack` | bit-exact |
| `calcOpticalFlowPyrLK` status, input points | bit-exact |
| `estimateAffinePartial2D` **inlier mask** | bit-exact |
| LK sub-pixel positions | max 1.06e-4 px, mean 2.65e-7 |
| RANSAC matrix | max 1.64e-5 |
| `cv::warpAffine` INTER_LINEAR REFLECT | **56/59 frames exact**; 11 of 19.3M pixels differ (0.000057%), max 2/255 |
| 21-element uint8 median | **38/39 frames exact**; 1 of 12.8M pixels differs, max 1/255 |

The RANSAC inlier mask matching bit-exactly is the load-bearing result: the
decide-what-counts-as-an-inlier step agrees across architectures, and the
divergence stays confined to the last bits of float32 sub-pixel coordinates.

**A caveat this tooling once got wrong**: a first pass over only 23 frames
reported `warp_dst` and `median_out` as fully identical, because the 4 frames
that actually differ were not sampled. Sparse disagreements must be reported
as an absolute element count over the whole run — "zero observations" is not
"does not happen".

## Running it

```bash
./build_opencv_arm64.sh 4                       # OpenCV for arm64 (~4 min)
./build_opencv_parity.sh                        # both parity binaries

N=/home/manu/mnt/nfs/ocvparity                  # data on NFS: the board disk has ~1.7 GiB
./out/gen_opencv_parity_case --seq-dir <seq> --out-dir $N/in --frames 60
./out/opencv_parity_x86 --in-dir $N/in --out-dir $N/out_x86b --tag x86

RK3588_PASSWORD=... python3 push_and_run.py push out/opencv_parity_arm64 \
    /mnt/manu/ocvparity_exec/parity              # scp: executables must not go over NFS
RK3588_PASSWORD=... python3 push_and_run.py run -- /mnt/manu/ocvparity_exec/parity \
    --in-dir /mnt/manu/ocvparity/in --out-dir /mnt/manu/ocvparity/out_arm64b

/home/manu/anaconda3/bin/python compare_opencv_parity.py \
    --ref $N/out_x86b --test $N/out_arm64b       # system python3 has no numpy
```

Executables travel by SSH because the board reports `Text file busy` for
anything executed off the NFS mount. Everything else — inputs, dumps, logs —
goes to NFS, so x86 reads the board's results in place instead of scp'ing tens
of megabytes back.

Note that NFS here does **not** mean "free". `/mnt/manu` is the same filesystem
as `/`, so the dumps still occupy the board's disk — this run took it from
1.7 GiB free to 1.5 GiB. The saving is the *transfer*, not the *storage*: the
board's results are readable in place instead of being copied back, and reruns
write incrementally instead of re-uploading every dump. Size the dumps anyway.

## Link-time traps in the cross build

* **`carotene_o4t::split*` undefined** — looks like a missing third-party
  dependency, but OpenCV names that target **`tegra_hal`** (a Tegra-era
  leftover) while namespacing its symbols `carotene_o4t`. It installs to
  **`lib/opencv4/3rdparty/`**, not `lib/`. Link with
  `-L$INSTALL/lib/opencv4/3rdparty -ltegra_hal`.
* **`gzopen`/`gzread` undefined** — `core/src/persistence.cpp` references zlib
  unconditionally for gzipped XML even with `WITH_JPEG=OFF`. The file is
  `libzlib.a`, hence **`-lzlib`**, not `-lz`.
* **System CMake 4.2.3 refuses OpenCV 4.10** outright, because 4.10 declares
  `cmake_minimum_required(VERSION 3.1)` and CMake ≥ 4.0 hard-errors below 3.5.
  A local CMake 3.31.6 is used rather than patching the tree, so the source
  stays byte-identical to the tag the reference was built from.

---

# Part 3 — the full feature channel on the board (`gmc_stream_ocl.cpp`)

Parts 1 and 2 measured pieces. This is the first time the **whole** feature
channel ran on the RK3588: read frames → GMC fit (CPU) → fused OpenCL kernel
(Mali-G610) → `[Ch0, Ch1, Ch2]`.

**No source change was needed.** `gmc_stream_ocl.cpp` already accepts `.bmp`
(`gmc_stream.cpp:156` has the same extension whitelist) and already takes
`--kernel-dir` as a runtime argument, so the same file that runs on x86 was
cross-compiled for aarch64 and shipped to the board as-is.

## Accuracy against the CPU-authoritative baseline

Comparing board output against x86 `gmc_stream.cpp` — the pure-CPU reference,
verified to contain zero OpenCL references — over the same 60 BMP frames with
identical parameters:

| channel | Max\|Diff\| | MAE | exact |
| :--- | :--- | :--- | :--- |
| Ch0 | **0** | 0.000000 | 100.0000% |
| Ch1 | **0** | 0.000000 | 100.0000% |
| Ch2 | 1 | 0.000000 | 100.0000% |

**59 of 60 frames are bit-identical across all three channels.** The single
exception is one Ch2 pixel on frame 41, and frames 40 and 42 are identical —
so this is a single-pixel rounding-boundary tie, not drift. If the upstream fit
were diverging systematically the difference would accumulate frame by frame.
The transform matrices `(21,6)` match on **126/126 entries**.

On x86, `gmc_stream.cpp` and `gmc_stream_ocl.cpp` produce the *same* MD5, so the
two implementations corroborate each other: board ≈ x86 CPU ≈ x86 OCL.

## Performance (median of 3 runs each, ms/frame)

| configuration | fit | warp | median | GPU tail | total | fps |
| :--- | --- | --- | --- | --- | --- | --- |
| **board, GPU arm** | 17.34 | 67.37 | 67.23 | **5.38** | **156.50** | **6.39** |
| board, CPU arm | 21.51 | 67.69 | 64.73 | — | 150.36 | 6.65 |
| x86 OCL (PoCL) | 7.49 | 27.04 | 36.34 | 106.63 | 176.52 | 5.67 |
| **x86 `gmc_stream.cpp`, pure CPU** | 10.15 | 27.20 | 33.03 | — | 68.69 | 14.56 |

Three readings:

1. **The fused GPU tail is 24.6× faster** than the CPU tail (132.4 → 5.38 ms),
   with the kernel itself at 3.84 ms/frame.
2. **The board's CPU path is 2.19× slower than x86's** (150.4 vs 68.7 ms) for the
   same source, same input, single-threaded — x86 has AVX2/AVX512 runtime
   dispatch, the board has only NEON. This is the practical reason to prefer the
   GPU tail on the board.
3. **On x86 the GPU is *slower* than the CPU (0.59×)** because that is PoCL
   simulating on the CPU. The same binary spans 106.6 ms vs 5.4 ms across the
   two machines — a 19.8× spread that says everything about why PoCL timings must
   never enter an embedded budget.

**The bottleneck is CPU `warp` + `median` (86.1%), not the GPU (3.4%) and not
`fit` (11.1%).** Using only the GPU tail would reach 44 fps, but as long as the
21-sample median stays on the CPU that number is unreachable — median at 67 ms is
the next thing worth moving.

Note that `total` is the wall clock of the whole `push()` call, so it includes
`imread`, state maintenance and CSV writing; on x86 the gap is 105.6 ms because
of PoCL initialisation. Always say which number you are quoting.

## Running it

```bash
./build_opencv_arm64.sh 4      # OpenCV 4.10.0 for arm64, incl. imgcodecs for BMP
./build_gmc_ocl_arm64.sh       # cross-compile gmc_stream_ocl.cpp

N=/home/manu/mnt/nfs/ocvparity
./convert_seq_to_bmp.py --seq-dir <jpeg seq> --out-dir $N/bmp --frames 60

RK3588_PASSWORD=... python3 push_and_run.py push out/gmc_stream_ocl_arm64 \
    /mnt/manu/ocvparity_exec/gmc_ocl
RK3588_PASSWORD=... python3 push_and_run.py run -- /mnt/manu/ocvparity_exec/gmc_ocl \
    --raw-root /mnt/manu/ocvparity --sequence bmp --limit 60 --mode tree \
    --anchor-step 10 --window 21 --stride-step 2 --downscale 2 \
    --fused both --kernel-dir /mnt/manu/ocvparity_exec/kernels --timing \
    --dump-dir /mnt/manu/ocvparity/dump_arm64

/home/manu/anaconda3/bin/python compare_board_vs_x86.py \
    --x86 $N/dump_cpuref --board $N/dump_arm64
```

Linking `gmc_stream_ocl` against OpenCL needs the board's ICD loader staged on
the **x86 host** (`rk3588/boardlibs/`) — the cross sysroot has no OpenCL, and the
board disk is too small to be a staging area. At run time the binary resolves
`libOpenCL.so.1` from the board's own ldconfig path.
