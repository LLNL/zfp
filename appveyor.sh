#!/usr/bin/env sh
set -e

# pass additional args in $1 (starting with whitespace character)
run_all () {
  run_all_cmd="ctest -V -C $BUILD_TYPE -DGENERATOR=\"$GENERATOR\" -S \"$APPVEYOR_BUILD_FOLDER/cmake/appveyor.cmake\""
  eval "${run_all_cmd}$1"
}

# create build dir for out-of-source build
mkdir build
cd build

# technically, flags are passed on to cmake/* and actually set there
# config without OpenMP, with CFP (and custom namespace), with aligned allocations (compressed arrays)
BUILD_FLAGS=""
BUILD_FLAGS="$BUILD_FLAGS -DBUILD_UTILITIES=ON"
BUILD_FLAGS="$BUILD_FLAGS -DBUILD_EXAMPLES=ON"
BUILD_FLAGS="$BUILD_FLAGS -DBUILD_CFP=ON"
BUILD_FLAGS="$BUILD_FLAGS -DCFP_NAMESPACE=cfp2"

# zfpy only built for MSVC, Release builds
if [ $COMPILER == "msvc" ] && [ $BUILD_TYPE == "Release" ]; then
  # verify active python version matches what was specified in appveyor.yml

  # fetch python version X.Y (single digits only)
  ACTIVE_PY_VERSION=$(python -c 'import platform; print(platform.python_version())' | cut -c1-3)
  # $PYTHON_VERSION comes from appveyor.yml and has form XY (no dot separating major and minor versions)
  ACTIVE_PY_VERSION=${ACTIVE_PY_VERSION:0:1}${ACTIVE_PY_VERSION:2:1}

  if [ $ACTIVE_PY_VERSION != $PYTHON_VERSION ]; then
    exit 1
  fi

  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_ZFPY=ON"
fi

BUILD_FLAGS="$BUILD_FLAGS -DBUILD_OPENMP=OFF"
BUILD_FLAGS="$BUILD_FLAGS -DBUILD_CUDA=OFF"

run_all "$BUILD_FLAGS"

# build empty project requiring OpenMP, in a temp directory that ZFP is oblivious to
mkdir tmpBuild
cd tmpBuild
# (CMAKE_SH satisfies mingw builds)
set +e
if [ $COMPILER != "msvc" ]; then
  cmake -G "$GENERATOR" -A "$PLATFORM" "$APPVEYOR_BUILD_FOLDER/tests/ci-utils" -DCMAKE_SH=CMAKE_SH-NOTFOUND
else
  cmake -G "$GENERATOR" -A "$PLATFORM" "$APPVEYOR_BUILD_FOLDER/tests/ci-utils"
fi

if [ $? -eq 0 ]; then
  echo "OpenMP found, starting 2nd zfp build"
  set -e
  cd ..
  # keep compiled testing frameworks, to speedup Appveyor
  rm CMakeCache.txt

  # only run tests not run in previous build, due to appveyor time limit (1 hour)
  # but continue to build utilities & examples because some support OpenMP
  BUILD_FLAGS=""
  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_UTILITIES=ON"
  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_EXAMPLES=ON"
  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_OPENMP=ON"
  BUILD_FLAGS="$BUILD_FLAGS -DOMP_TESTS_ONLY=ON"

  run_all "$BUILD_FLAGS"
else
  echo "OpenMP not found, build completed."
fi
