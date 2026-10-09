# RK3588 board tooling — OpenCL probe and fused-kernel accuracy harness

Two read-only programs plus the machinery to build, ship and run them. Both
dlopen OpenCL at run time, so **neither has any link-time OpenCL dependency**
and both cross-compile with nothing but the Rockchip toolchain and `libdl`.

| tool | runs on | what it answers |
| :--- | :--- | :--- |
| `board_cl_probe` | board | Does this board have a usable OpenCL stack, and which one? |
| `gen_fused_case` | **x86 + OpenCV** | Builds accuracy test cases: real frames, real GMC, and the CPU reference |
| `board_fused_accuracy` | board (no OpenCV) | Runs the real `warp_median_fused.cl` and reconciles it against that reference |

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
3. **First trustworthy board timing: 3.7~4.8 ms/frame.** Three qualifiers must
   travel with it: it is an `clEnqueue` + `clFinish` wall-clock **upper bound**
   (event profiling is unavailable on this driver), it **includes** uploading 22
   images and reading the result back, and Mali **compiles a kernel on first
   launch**.

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
