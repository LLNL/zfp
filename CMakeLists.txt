# CMake build system for zfp
# ==========================
#
# Available options:
#
#   BUILD_STATIC: default: ON
#       Build the static version of the zfp library
#   BUILD_SHARED: default: ON
#       Build the shared library version of the zfp library
#   BIT_STREAM_WORD_TYPE: default: uint64
#       Bit stream word type (uint8, uint16, uint32 or uint64),
#       use smaller type for finer rate granularity
#   BIT_STREAM_STRIDED: default: OFF
#       Enable strided access for progressive zfp streams
#   TESTZFP_MEDIUM_ARRAYS: default: OFF
#       Run regression test with medium sized arrays
#   TESTZFP_LARGE_ARRAYS: default: OFF
#       Run regression test with large arrays
#   WITHOUT_COMPRESSION: default: OFF
#       Run diffusion without compression
#   ALIGNED_ALLOC: default: OFF
#       Use aligned memory allocation
#   CACHE_TWOWAY: default: OFF
#       Use two-way skew-associative cache
#   CACHE_FAST_HASH: default: OFF
#       Use faster but more collision prone hash function
#   CACHE_PROFILE: default: OFF
#       Count cache misses
#

cmake_minimum_required(VERSION 3.1)

project(zfp)

enable_testing()

include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)
include(GNUInstallDirs)

# Parse version number from zfp.h

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/inc/zfp.h _zfp_h_contents)
string(REGEX REPLACE ".*#define[ \t]+ZFP_VERSION_MAJOR[ \t]+([0-9]+).*"
     "\\1" ZFP_VERSION_MAJOR ${_zfp_h_contents})
string(REGEX REPLACE ".*#define[ \t]+ZFP_VERSION_MINOR[ \t]+([0-9]+).*"
    "\\1" ZFP_VERSION_MINOR ${_zfp_h_contents})
string(REGEX REPLACE ".*#define[ \t]+ZFP_VERSION_RELEASE[ \t]+([0-9]+).*"
    "\\1" ZFP_VERSION_PATCH ${_zfp_h_contents})
set(ZFP_VERSION_STRING "${ZFP_VERSION_MAJOR}.${ZFP_VERSION_MINOR}.${ZFP_VERSION_PATCH}")

# Options

set(BIT_STREAM_WORD_TYPE "uint64" CACHE STRING
    "Bit stream word type (use smaller for finer rate granularity)")
option(BUILD_STATIC
    "Build the static version of the zfp library" ON)
option(BUILD_SHARED
    "Build the shared library version of the zfp library" ON)
set(BIT_STREAM_WORD_TYPE "uint64" CACHE STRING
    "Bit stream word type (use smaller for finer rate granularity)")
option(BIT_STREAM_STRIDED
    "Enable strided access for progressive zfp streams")
option(TESTZFP_MEDIUM_ARRAYS
    "Run regression test with medium sized arrays")
option(TESTZFP_LARGE_ARRAYS
    "Run regression test with large arrays")
option(WITHOUT_COMPRESSION
    "Run diffusion without compression")
option(ALIGNED_ALLOC
    "Use aligned memory allocation")
option(CACHE_TWOWAY
    "Use two-way skew-associative cache")
option(CACHE_FAST_HASH
    "Use faster but more collision prone hash function")
option(CACHE_PROFILE
    "Count cache misses")

if(NOT BUILD_STATIC AND NOT BUILD_SHARED)
    message(FATAL_ERROR "At least one of BUILD_STATIC and BUILD_SHARED must be set")
endif()

# If the build type is not set, default to Release

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

# Configure compiler flags

set(CMAKE_C_STANDARD 90)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 98)
set(CMAKE_CXX_EXTENSIONS OFF)

check_c_compiler_flag(-Wall C_COMPILER_SUPPORTS_WALL)
if(C_COMPILER_SUPPORTS_WALL)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
endif()

check_c_compiler_flag(-Wno-unused-function C_COMPILER_SUPPORTS_WNO_UNUSED_FUNCTION)
if(C_COMPILER_SUPPORTS_WNO_UNUSED_FUNCTION)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
endif()

check_cxx_compiler_flag(-Wall CXX_COMPILER_SUPPORTS_WALL)
if(CXX_COMPILER_SUPPORTS_WALL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

add_definitions(-DBIT_STREAM_WORD_TYPE=${BIT_STREAM_WORD_TYPE})

if(BIT_STREAM_STRIDED)
    add_definitions(-DBIT_STREAM_STRIDED)
endif()

if(TESTZFP_MEDIUM_ARRAYS)
    add_definitions(-DTESTZFP_MEDIUM_ARRAYS)
endif()

if(TESTZFP_LARGE_ARRAYS)
    add_definitions(-DTESTZFP_LARGE_ARRAYS)
endif()

if(WITHOUT_COMPRESSION)
    add_definitions(-DWITHOUT_COMPRESSION)
endif()

if(ALIGNED_ALLOC)
    add_definitions(-DALIGNED_ALLOC)
endif()

if(CACHE_TWOWAY)
    add_definitions(-DCACHE_TWOWAY)
endif()

if(CACHE_FAST_HASH)
    add_definitions(-DCACHE_FAST_HASH)
endif()

if(CACHE_PROFILE)
    add_definitions(-DCACHE_PROFILE)
endif()

add_subdirectory(src)
add_subdirectory(examples)
