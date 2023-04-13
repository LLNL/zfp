cmake_minimum_required(VERSION 3.12)

project(zfpy)

list(APPEND CMAKE_MODULE_PATH "${zfpy_SOURCE_DIR}/cmake")

find_package(Python3 COMPONENTS Interpreter Development NumPy)
find_package(PythonExtensions)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)
include_directories(${NumPy_INCLUDE_DIR})

set(CMAKE_SHARED_MODULE_PREFIX "")
include(${zfpy_SOURCE_DIR}/cmake/zfpy.cmake)

zfpy_add_module(zfpy_version)
zfpy_add_module(zfpy)
