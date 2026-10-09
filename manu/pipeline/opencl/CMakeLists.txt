# ============================================================================
# manu/pipeline/opencl -- fused GMC warp + median + 3-channel pack (OpenCL)
#
# Targets
#   ocl_fused_check   accuracy reconciliation against the CPU golden reference
#   sortnet_verify    offline generator/prover for the 21-element sorting net
#
# The OpenCL host layer (host/ocl_host.h) links against the ICD loader
# (libOpenCL.so.1): PoCL on the verification workstation, libmali on RK3588.
# Nothing else differs between the two.
# ============================================================================
cmake_minimum_required(VERSION 3.16)
project(manu_ocl_fused LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(OpenCL REQUIRED)

message(STATUS "OpenCV     : ${OpenCV_VERSION}  (${OpenCV_INCLUDE_DIRS})")
message(STATUS "OpenCL     : ${OpenCL_VERSION_STRING} (${OpenCL_LIBRARIES})")

# ---------------------------------------------------------------------------
# ocl_fused_check
#
# -Wall -Wextra is on because the acceptance bar is a build with no errors and no
# deprecated-API warnings; OpenCL's 1.x entry points are deprecated in 2.0 and
# the host layer deliberately routes around them (see ocl_host.h).
# ---------------------------------------------------------------------------
add_executable(ocl_fused_check host/ocl_fused_check.cpp)
target_include_directories(ocl_fused_check PRIVATE
  ${OpenCV_INCLUDE_DIRS} host ${OpenCL_INCLUDE_DIRS})
target_link_directories(ocl_fused_check PRIVATE ${OpenCL_LIBRARY_DIRS})
target_link_libraries(ocl_fused_check PRIVATE ${OpenCV_LIBS} ${OpenCL_LIBRARIES})
target_compile_options(ocl_fused_check PRIVATE -Wall -Wextra)

# ---------------------------------------------------------------------------
# sortnet_verify -- offline tool, PC only.
#
# This is emphatically NOT part of the RK3588 runtime: it is a build-time proof
# helper whose only output is kernels/sortnet_generated.inc. OpenMP is used
# solely to fan out the 2^21 exhaustive test across the workstation's cores and
# is #ifdef-guarded, so it also builds without it. The RK3588 BSP image
# (Cortex-A76/A55, no OpenMP runtime) never compiles this translation unit.
# ---------------------------------------------------------------------------
add_executable(sortnet_verify tools/sortnet_verify.c)
target_include_directories(sortnet_verify PRIVATE ${OpenCL_INCLUDE_DIRS})
target_compile_options(sortnet_verify PRIVATE -Wall -Wextra)
find_package(OpenMP QUIET)
if(OpenMP_C_FOUND)
  target_link_libraries(sortnet_verify PRIVATE OpenMP::OpenMP_C)
endif()

# ---------------------------------------------------------------------------
# Offline probes. These are how the interpolation model was pinned down by
# measurement rather than by reading OpenCV's source, and they are kept because
# the numbers they print are the justification for the kernel's arithmetic.
# ---------------------------------------------------------------------------
foreach(probe probe_interp probe_interp2 probe_coord)
  add_executable(${probe} tools/${probe}.cpp)
  target_include_directories(${probe} PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${probe} PRIVATE ${OpenCV_LIBS})
  target_compile_options(${probe} PRIVATE -Wall -Wextra)
endforeach()

# Regenerate kernels/sortnet_generated.inc. Run from this directory.
add_custom_target(regen_sortnet
  COMMAND sortnet_verify ${CMAKE_CURRENT_SOURCE_DIR}/kernels/sortnet_generated.inc
  DEPENDS sortnet_verify
  COMMENT "Regenerating + exhaustively re-verifying the 21-element sorting network")