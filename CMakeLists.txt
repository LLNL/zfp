cmake_minimum_required(VERSION 3.1)

# Fail immediately if not using an out-of-source build
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not supported.  Please create a build directory "
    "separate from the source directory")
endif()

#------------------------------------------------------------------------------#
# Parse version number from zfp.h
#------------------------------------------------------------------------------#
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/inc/zfp.h _zfp_h_contents)
string(REGEX REPLACE ".*#define[ \t]+ZFP_VERSION_MAJOR[ \t]+([0-9]+).*"
     "\\1" ZFP_VERSION_MAJOR ${_zfp_h_contents})
string(REGEX REPLACE ".*#define[ \t]+ZFP_VERSION_MINOR[ \t]+([0-9]+).*"
    "\\1" ZFP_VERSION_MINOR ${_zfp_h_contents})
string(REGEX REPLACE ".*#define[ \t]+ZFP_VERSION_RELEASE[ \t]+([0-9]+).*"
    "\\1" ZFP_VERSION_PATCH ${_zfp_h_contents})
set(ZFP_VERSION
  "${ZFP_VERSION_MAJOR}.${ZFP_VERSION_MINOR}.${ZFP_VERSION_PATCH}")

project(ZFP VERSION ${ZFP_VERSION})

#------------------------------------------------------------------------------#
# Some boilerplate to setup nice output directories
#------------------------------------------------------------------------------#
include(GNUInstallDirs)

list(INSERT CMAKE_MODULE_PATH 0 "${ZFP_SOURCE_DIR}/cmake")
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${ZFP_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${ZFP_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
endif()
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${ZFP_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
endif()

#------------------------------------------------------------------------------#
# Top level options
#------------------------------------------------------------------------------#

# Suggest C99
# Falls back to C89 if 99 is not supported
set(CMAKE_C_STANDARD 99)

include(CMakeDependentOption)

# Typically you'd always be able to enable shared libraries but default
# configurations with the Cray toolchain will explicitly disable shared lib
# supprot and only allow static libs.  Making this a cmake_dependent_option
# will ensure that shared library support will be disabled if the system doesnt
# support it.

# Setup shared library / -fPIC stuff
get_property(SHARED_LIBS_SUPPORTED GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS)
cmake_dependent_option(BUILD_SHARED_LIBS
  "Whether or not to build shared libraries" ON
  "SHARED_LIBS_SUPPORTED" OFF)

# PIC is always on for shared libs.  This allows it to be selectable for
# static libs
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
add_subdirectory(src)

option(BUILD_UTILITIES "Build command line utilities for zfp" ON)
if(BUILD_UTILITIES)
  add_subdirectory(utils)
endif()

option(BUILD_EXAMPLES "Build Examples" OFF)
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

include(CTest)
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

#------------------------------------------------------------------------------#
# Header install
#------------------------------------------------------------------------------#

install(DIRECTORY inc/   DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY array/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
