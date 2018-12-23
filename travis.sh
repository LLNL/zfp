#!/usr/bin/env sh
set -e

mkdir build
cd build

CTEST_FLAGS="-V -C \"Debug\" -DC_STANDARD=${C_STANDARD:-99} -DCXX_STANDARD=${CXX_STANDARD:-98} -DBUILD_PYTHON=${BUILD_PYTHON:-off}"

if [ -n "${COVERAGE}" ]; then
  # build (linux)
  ctest ${CTEST_FLAGS} -DBUILD_CFP=ON -DBUILD_ZFORP=ON -DBUILD_OPENMP=ON -DBUILD_CUDA=OFF -DWITH_COVERAGE=ON -S $TRAVIS_BUILD_DIR/cmake/travis.cmake
else
  # build/test without OpenMP, with CFP (and custom namespace), with Fortran (linux only)
  if [[ "$OSTYPE" == "darwin"* ]]; then
    BUILD_ZFORP=OFF
  else
    BUILD_ZFORP=ON
  fi

  ctest ${CTEST_FLAGS} -DBUILD_CFP=ON -DCFP_NAMESPACE=cfp2 -DBUILD_ZFORP=${BUILD_ZFORP} -DZFP_WITH_ALIGNED_ALLOC=1 -DBUILD_OPENMP=OFF -DBUILD_CUDA=OFF -S $TRAVIS_BUILD_DIR/cmake/travis.cmake

  rm -rf ./* ;

  # if OpenMP available, start a 2nd build with it
  if cmake ../tests/ci-utils/ ; then
    rm -rf ./* ;

    # build/test with OpenMP
    ctest ${CTEST_FLAGS} -DBUILD_OPENMP=ON -S $TRAVIS_BUILD_DIR/cmake/travis.cmake
  fi
fi
