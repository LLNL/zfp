cmake_minimum_required(VERSION 3.12)

project(zfpy)

list(APPEND CMAKE_MODULE_PATH "${zfpy_SOURCE_DIR}/cmake")

find_package(Python3 COMPONENTS Interpreter Development NumPy)
find_package(PythonExtensions)
find_package(Cython REQUIRED)

set(CMAKE_SHARED_MODULE_PREFIX "")

include(${zfpy_SOURCE_DIR}/cmake/zfpy.cmake)

zfpy_add_module(version version.pyx)
zfpy_add_module(zfpy zfpy.pyx)

## Build to the current binary dir to avoid conflicts with other libraries named zfp
#set(PYLIB_BUILD_DIR "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Directory where zfp python library will be built")
#set_target_properties(zfpy PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYLIB_BUILD_DIR})
#
## Install to the typical python module directory
#set(python_install_lib_dir "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/")
#install(TARGETS zfpy LIBRARY DESTINATION ${python_install_lib_dir})
