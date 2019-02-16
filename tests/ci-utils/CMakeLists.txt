# This empty project is used to determine if OpenMP is available on CI machines
# without compiling any ZFP code.

if(WIN32)
  cmake_minimum_required(VERSION 3.4)
else()
  cmake_minimum_required(VERSION 3.1)
endif()

find_package(OpenMP COMPONENTS C REQUIRED)
