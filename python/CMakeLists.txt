cmake_minimum_required(VERSION 3.12)

project(zfpy)

set(CMAKE_SHARED_MODULE_PREFIX "")
include(zfpy)

zfpy_add_module(zfpy_version)
zfpy_add_module(zfpy)
