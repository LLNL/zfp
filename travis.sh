#!/usr/bin/env sh
set -e

# pass additional args in $1 (starting with whitespace character)
run_all () {
  run_all_cmd="ctest -V -C Debug -DC_STANDARD=${C_STANDARD:-99} -DCXX_STANDARD=${CXX_STANDARD:-98} -S \"$TRAVIS_BUILD_DIR/cmake/travis.cmake\""
  eval "${run_all_cmd}$1"
}

mkdir build
cd build

# technically, flags are passed on to cmake/* and actually set there
BUILD_FLAGS=""

if [ -n "${COVERAGE}" ]; then
  # build (linux)

  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_UTILITIES=ON"
  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_EXAMPLES=ON"
  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_CFP=ON"
  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_ZFPY=ON"
  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_ZFORP=ON"
  BUILD_FLAGS="$BUILD_FLAGS -DZFP_WITH_ALIGNED_ALLOC=ON"
  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_OPENMP=ON"
  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_CUDA=OFF"
  BUILD_FLAGS="$BUILD_FLAGS -DWITH_COVERAGE=ON"

  run_all "$BUILD_FLAGS"
else
  # build/test without OpenMP, with CFP (and custom namespace), with zfPy, with Fortran (linux only)
  if [[ "$OSTYPE" == "darwin"* ]]; then
    BUILD_ZFORP=OFF
  else
    BUILD_ZFORP=ON
  fi

  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_UTILITIES=ON"
  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_EXAMPLES=ON"
  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_CFP=ON"
  BUILD_FLAGS="$BUILD_FLAGS -DCFP_NAMESPACE=cfp2"
  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_ZFPY=ON"
  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_ZFORP=$BUILD_ZFORP"
  BUILD_FLAGS="$BUILD_FLAGS -DZFP_WITH_ALIGNED_ALLOC=ON"
  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_OPENMP=OFF"
  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_CUDA=OFF"
  run_all "$BUILD_FLAGS"

  rm -rf ./* ;

  # if OpenMP available, start a 2nd build with it
  if cmake ../tests/ci-utils/ ; then
    rm -rf ./* ;

    # build/test with OpenMP
    BUILD_FLAGS=""
    BUILD_FLAGS="$BUILD_FLAGS -DBUILD_OPENMP=ON"
    run_all "$BUILD_FLAGS"
  fi
fi
