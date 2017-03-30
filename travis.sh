#!/usr/bin/env sh
set -e

mkdir build
cd build
cmake ..
cmake --build .
ctest -V -C "Debug"

