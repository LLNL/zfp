#!/usr/bin/env sh
set -e

mkdir build
cd build


if [ -n "${COVERAGE}" ]; then
  # build
  ctest -V -C "Debug" -DC_STANDARD=${C_STANDARD:-99} -DCXX_STANDARD=${CXX_STANDARD:-98} -DBUILD_CFP=ON -DBUILD_OPENMP=ON -DBUILD_CUDA=OFF -DWITH_COVERAGE=ON -S $TRAVIS_BUILD_DIR/cmake/travis.cmake
else
  # build/test without OpenMP, with CFP
  ctest -V -C "Debug" -DC_STANDARD=${C_STANDARD:-99}  -DCXX_STANDARD=${CXX_STANDARD:-98} -DBUILD_CFP=ON -DBUILD_OPENMP=OFF -DBUILD_CUDA=OFF -S $TRAVIS_BUILD_DIR/cmake/travis.cmake

  # build/test with CFP custom namespace
  ctest -V -C "Debug" -DC_STANDARD=${C_STANDARD:-99} -DCXX_STANDARD=${CXX_STANDARD:-98} -DBUILD_CFP=ON -DCFP_NAMESPACE=cfp2 -S $TRAVIS_BUILD_DIR/cmake/travis.cmake
fi
