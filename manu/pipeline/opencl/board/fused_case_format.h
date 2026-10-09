// ===========================================================================
// fused_case_format.h -- on-disk contract between the x86 case generator and
// the RK3588 accuracy harness.
//
// The split exists because of one hard fact measured on 2026-10-09: there is
// no arm64 OpenCV anywhere. The cross toolchain's sysroot has no OpenCV, the
// vendor SDK ships only a buildroot recipe, and the board itself has a Python
// `cv2` but no /usr/include/opencv4. So the CPU reference -- which MUST come
// from real OpenCV, because that is exactly what the bit-exactness argument
// rests on -- has to be computed on the x86 host and shipped to the board as
// data. The board then only needs OpenCL, which it demonstrably has.
//
// Both sides are little-endian (x86-64 and aarch64), so the struct is written
// and read raw. Every count is an explicit uint32 rather than relying on the
// compiler's struct layout rules.
// ===========================================================================
#ifndef FUSED_CASE_FORMAT_H
#define FUSED_CASE_FORMAT_H

#include <stdint.h>

#define FKC_MAGIC "FKC1"
#define FKC_WINDOW 21

struct FkcHeader {
  char     magic[4];      // "FKC1"
  uint32_t W;
  uint32_t H;
  uint32_t pad;           // border baked into every uploaded frame
  uint32_t n_cases;
  uint32_t window;        // 21, asserted equal on both sides
  uint32_t reserved;      // 0
};

struct FkcCaseMeta {
  char     seq[64];       // source sequence name, for the report
  uint32_t t;             // index of the current frame in the sequence
  uint32_t gmc_failed;    // lags that fell back to identity (should be 0)
  float    motion_px;     // mean |translation| over the 21 lags, for context
};

#endif  // FUSED_CASE_FORMAT_H
