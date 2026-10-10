# ===========================================================================
# rk3588-aarch64-toolchain.cmake -- cross-compile OpenCV 4.10.0 for RK3588.
#
# CMake 3.31 is required, NOT the system CMake 4.x: OpenCV 4.10.0 declares
# `cmake_minimum_required(VERSION 3.1)` and CMake >= 4.0 hard-errors on any
# compatibility level below 3.5. This file does not work around that by
# patching OpenCV's sources, because the worktree must stay byte-identical to
# the 4.10.0 tag that the x86 reference was built from.
#
# Only the Rockchip buildroot sysroot is visible:
#   CMAKE_FIND_ROOT_PATH_MODE_INCLUDE/LIBRARY/PACKAGE = ONLY
# so a stray -I/usr/include can never drag x86 glibc headers into the build.
# Program lookup stays NEVER so host tools (git, python) remain reachable.
# ===========================================================================

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(_rk_tc "/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu")
set(_rk_prefix "${_rk_tc}/bin/aarch64-rockchip-linux-gnu-")
set(_rk_sysroot "${_rk_tc}/aarch64-rockchip-linux-gnu/sysroot")

set(CMAKE_C_COMPILER   "${_rk_prefix}gcc")
set(CMAKE_CXX_COMPILER "${_rk_prefix}g++")
set(CMAKE_AR           "${_rk_prefix}ar" CACHE FILEPATH "")
set(CMAKE_RANLIB       "${_rk_prefix}ranlib" CACHE FILEPATH "")
set(CMAKE_STRIP        "${_rk_prefix}strip" CACHE FILEPATH "")
set(CMAKE_OBJCOPY      "${_rk_prefix}objcopy" CACHE FILEPATH "")
set(CMAKE_OBJDUMP      "${_rk_prefix}objdump" CACHE FILEPATH "")

# find_package host programs that the toolchain still needs to locate
set(CMAKE_FIND_ROOT_PATH "${_rk_sysroot}")

set(CMAKE_CROSSCOMPILING TRUE)

# Prefix choice, measured rather than assumed. The toolchain ships several; two
# of them link successfully and only one carries a sysroot:
#
#   aarch64-linux-gcc              generic, has no sysroot at all
#   aarch64-rockchip-linux-gnu-    sysroot glibc 2.29  <-- used here
#   aarch64-rockchip930-linux-gnu- no sysroot directory
#
# "Links successfully" is a weak test on its own -- any aarch64 prefix happily
# produces an ELF. What actually matters is the glibc symbol version the result
# requires. The board runs glibc 2.31 (Ubuntu 20.04.6), the sysroot provides
# 2.29, and a binary built against 2.29 runs on 2.31 because glibc is backward
# compatible in that direction. Building against the 930 prefix instead would
# pull in whatever that vendor tree pins and reintroduce the mismatch.
#
# Verified on the board: `ldd` resolves libpthread/libm/libc with no "not found",
# and the binary executes.

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# ---------------------------------------------------------------------------
# Floating-point contract.
#
# This is not stylistic. The x86 reference is Ubuntu's libopencv-dev
# 4.10.0+dfsg-7ubuntu5, built for the plain `x86-64` baseline where `gcc -Q`
# reports `-mfma [disabled]`. With `-ffp-contract=fast` (the GCC default on both
# toolchains) x86 therefore *cannot* emit FMA and every multiply-add rounds
# twice -- which is exactly `-ffp-contract=off` semantics.
#
# aarch64 has `fmadd` in the baseline, so the same default would fuse a*b+c into
# one rounding. That is a systematic, unavoidable divergence from the reference
# for every float32 expression in OpenCV. Turning contraction off on the
# arm64 side is what puts both architectures back on the same semantics.
#
# `-fno-fast-math` matches the project-wide rule and is also required for
# OpenCV's own fast-math-gated code paths to stay off.
# ---------------------------------------------------------------------------
set(_rk_fp_flags "-ffp-contract=off -fno-fast-math")
set(CMAKE_C_FLAGS_INIT   "${_rk_fp_flags}")
set(CMAKE_CXX_FLAGS_INIT "${_rk_fp_flags}")

# The buildroot gcc already knows its own sysroot; passing the same path again
# is harmless and keeps the intent explicit.
set(CMAKE_SYSROOT "${_rk_sysroot}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
