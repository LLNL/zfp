cmake_minimum_required(VERSION 3.1)

# Fail immediately if not using an out-of-source build
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not supported.  Please create a build directory "
    "separate from the source directory")
endif()

project(ZFP VERSION 0.5.0)

#------------------------------------------------------------------------------#
# Some boilerplate to setup nice output directories
#------------------------------------------------------------------------------#

list(INSERT CMAKE_MODULE_PATH 0 "${ZFP_SOURCE_DIR}/cmake")
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${ZFP_BINARY_DIR}/lib)
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${ZFP_BINARY_DIR}/lib)
endif()
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${ZFP_BINARY_DIR}/bin)
endif()

#------------------------------------------------------------------------------#
# Top level options
#------------------------------------------------------------------------------#

# Force C99
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

include(CMakeDependentOption)

# Setup shared library / -fPIC stuff
get_property(SHARED_LIBS_SUPPORTED GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS)
cmake_dependent_option(BUILD_SHARED_LIBS
  "Whether or not to build shared libraries" ON
  "SHARED_LIBS_SUPPORTED" OFF)
if(SHARED_LIBS_SUPPORTED)
  cmake_dependent_option(ZFP_ENABLE_PIC
    "Build with Position Independent Code" ON
    "NOT BUILD_SHARED_LIBS" ON)
endif()
if(ZFP_ENABLE_PIC)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()


set(ZFP_BIT_STREAM_WORD_SIZE 64 CACHE STRING
  "Use smaller bit stream word type for finer rate granularity")
set_property(CACHE ZFP_BIT_STREAM_WORD_SIZE PROPERTY STRINGS "8;16;32;64")

option(ZFP_WITH_BIT_STREAM_STRIDED
  "Enable strided access for progressive zfp streams" OFF)

option(ZFP_WITH_COMPRESSION "Enable or disable compression" ON)

option(ZFP_WITH_ALIGNED_ALLOC "Enabled aligned memory allocation" OFF)

option(ZFP_WITH_CACHE_TWOWAY "Use two-way skew-associative cache" OFF)

option(ZFP_WITH_CACHE_FASH_HASH
  "Use a faster but more collision prone hash function" OFF)

option(ZFP_WITH_CACHE_PROFILING "Count cache misses" OFF)

#------------------------------------------------------------------------------#
# Add source code
#------------------------------------------------------------------------------#
include(CTest)

add_subdirectory(src)

option(ZFP_BUILD_EXAMPLES "Build Examples" OFF)
add_subdirectory(examples)

#------------------------------------------------------------------------------#
# Header install
#------------------------------------------------------------------------------#

install(DIRECTORY inc/   DESTINATION include)
install(DIRECTORY array/ DESTINATION include)
