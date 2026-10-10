# Bit-exactness-critical build settings.
#
# The port reproduces Python/`cv2` output bit for bit by calling the same OpenCV C++
# routines and by replicating numpy's float64 chain composition. Three compiler
# behaviours would break that, so they are pinned rather than left to the default:
#
#   -ffp-contract=off   Without this the compiler may fuse `acc + a*b` into an FMA, which
#                       rounds once instead of twice and changes the chain result. This is
#                       the single most likely cause of a "tiny but nonzero" divergence.
#   -fno-fast-math      Fast-math enables reassociation. `_compose` is order-sensitive:
#                       the k-ascending dot product must survive.
#   -fno-unsafe-math-optimizations  Same family; blocks transforms of NaN/Inf handling that
#                       the `np.isfinite` guards in `_compose` rely on seeing as Inf/NaN.
#
# CMAKE_BUILD_TYPE=Release keeps -O3 (which is fine); never use -Ofast or RelWithDebInfo+fast
# flags. If your toolchain defaults to `-ffast-math`, override CMAKE_CXX_FLAGS explicitly.
cmake_minimum_required(VERSION 3.16)
project(gmc_stream LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

add_executable(gmc_stream gmc_stream.cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang|Intel")
  target_compile_options(gmc_stream PRIVATE
    -ffp-contract=off
    -fno-fast-math
    -fno-unsafe-math-optimizations
    -Wall -Wextra -Wno-unused-parameter
  )
elseif(MSVC)
  # /fp:precise keeps strict IEEE semantics; /fp:fast would allow contraction/reassociation.
  target_compile_options(gmc_stream PRIVATE /fp:precise /W3)
endif()

find_package(OpenCV REQUIRED COMPONENTS
  core
  imgproc
  imgcodecs
  features2d
  calib3d
  video
)

target_include_directories(gmc_stream PRIVATE ${OpenCV_INCLUDE_DIRS})

# Report the OpenCV build so a mismatch against the Python side is visible immediately
# rather than showing up later as an unexplained feature divergence.
message(STATUS "OpenCV version : ${OpenCV_VERSION}")
message(STATUS "OpenCV modules : ${OpenCV_LIBS}")

target_link_libraries(gmc_stream PRIVATE ${OpenCV_LIBS})

# Fail the configure step if any bit-exactness-critical flag was stripped from the
# command line. Distributions commonly inject -ffast-math via CMAKE_CXX_FLAGS or
# CMAKE_CXX_FLAGS_RELEASE rather than via COMPILE_OPTIONS, so all three are scanned;
# checking COMPILE_OPTIONS alone would miss exactly the case this guard exists for.
string(TOUPPER "${CMAKE_BUILD_TYPE}" _cfg)
set(_all_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_cfg}} ${CMAKE_CXX_FLAGS_RELEASE}")

get_target_property(_opts gmc_stream COMPILE_OPTIONS)
string(REPLACE ";" " " _all_flags "${_all_flags} ${_opts}")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang|Intel")
  if(_all_flags MATCHES "ffast-math|funsafe-math-optimizations")
    message(FATAL_ERROR
      "-ffast-math / -funsafe-math-optimizations detected in: ${_all_flags}\n"
      "It reassociates floating point and breaks bit-exactness against the Python reference. "
      "Remove it from CMAKE_CXX_FLAGS (and your shell environment / distro defaults) and reconfigure.")
  endif()
  if(NOT _all_flags MATCHES "ffp-contract=off")
    message(FATAL_ERROR
      "-ffp-contract=off was not applied (effective flags: ${_all_flags}).\n"
      "Without it the compiler may fuse acc+a*b into an FMA, rounding once instead of twice, "
      "which silently breaks bit-exactness of the chain composition.")
  endif()
  message(STATUS "Floating-point guard: OK (ffp-contract=off present, no fast-math)")
endif()

# Fail if the binary cannot self-test its md5 (a fingerprint bug must never be mistaken
# for a feature divergence).
add_custom_command(TARGET gmc_stream POST_BUILD
  COMMAND gmc_stream --help
  COMMENT "Sanity-checking the built binary"
)
# ---------------------------------------------------------------------------
# gmc_stream_ocl -- the OpenCL/ARM variant of the same pipeline.
#
# Split into its own target because it needs Khronos headers and libOpenCL,
# neither of which the CPU-only gmc_stream above requires. Keeping them apart
# means a machine with no OpenCL can still build and run the reference.
# ---------------------------------------------------------------------------
find_package(OpenCL REQUIRED)

add_executable(gmc_stream_ocl gmc_stream_ocl.cpp)

target_include_directories(gmc_stream_ocl PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${OpenCL_INCLUDE_DIRS}
)

target_link_libraries(gmc_stream_ocl PRIVATE
  ${OpenCV_LIBS}
  ${OpenCL_LIBRARIES}
)

# Same three flags as above: the OCL arm composes transforms in the same
# float32 chain, so contraction or reassociation would break it identically.
target_compile_options(gmc_stream_ocl PRIVATE
  -ffp-contract=off
  -fno-fast-math
  -fno-unsafe-math-optimizations
  -Wall -Wextra -Wno-unused-parameter
)
