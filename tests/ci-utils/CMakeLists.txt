# This empty project is used to determine if OpenMP is available on CI machines
# without compiling any ZFP code.

cmake_minimum_required(VERSION 3.9)

find_package(OpenMP COMPONENTS C REQUIRED)
