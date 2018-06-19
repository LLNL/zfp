#!/usr/bin/env sh
set -e

mkdir build
cd build

# build/test without OpenMP, with CFP
cmake .. -DCMAKE_C_STANDARD=${C_STANDARD:-99} -DCMAKE_CXX_STANDARD=${CXX_STANDARD:-98} -DZFP_WITH_OPENMP=OFF -DBUILD_CFP=ON
cmake --build .
ctest -V -C "Debug"

rm -rf ./*

# build/test with OpenMP
cmake .. -DCMAKE_C_STANDARD=${C_STANDARD:-99} -DCMAKE_CXX_STANDARD=${CXX_STANDARD:-98}
cmake --build .
ctest -V -C "Debug"
