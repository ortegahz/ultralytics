# rk3588-aarch64-npu-toolchain.cmake -- minimal toolchain for building the
# SDK's rknn_yolov5_demo against the cross toolchain.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(_tc "/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu")

set(CMAKE_C_COMPILER   "${_tc}/bin/aarch64-rockchip-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${_tc}/bin/aarch64-rockchip-linux-gnu-g++")

# The demo's CMakeLists keys off "does the compiler name contain aarch64" to
# decide armhf vs aarch64, so the toolchain prefix must keep that substring.

set(CMAKE_FIND_ROOT_PATH "${_tc}/aarch64-rockchip-linux-gnu/sysroot")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Same floating-point contract as the rest of this project.
set(CMAKE_C_FLAGS   "-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations")
set(CMAKE_CXX_FLAGS "-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations")